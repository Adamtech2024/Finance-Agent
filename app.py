"""Finance Agent — a FastAPI financial management workspace.

Backend, storage, reports, assistant integration, and page markup live in this
file. Frontend styles and scripts are inlined as STYLE_CSS and APP_JS, so
everything lives here.

Run with:  python app.py
Default:   http://127.0.0.1:2611

Reporting basis: accrual. Revenue is recognised on the invoice issue date for
invoices that have been issued (sent, overdue or paid). Draft and cancelled
invoices are excluded. Expenses are recognised on the date incurred. The
balance sheet is a partial management position, not a complete statutory
statement; its assumptions are returned by the API and shown in the UI.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import types
import urllib.error
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field as dataclass_field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Literal, Optional
from uuid import uuid4
from xml.etree import ElementTree as ET

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pymupdf
except ImportError:
    pymupdf = None


# ===========================================================================
# Section 1: Core Financial Primitives
# ===========================================================================

from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

ZERO = Decimal("0")
CENT = Decimal("0.01")
Q4 = Decimal("0.0001")
MONEY_MAX = Decimal("1000000000000")

INVOICE_STATUS = ("draft", "sent", "paid", "overdue", "cancelled")
ISSUED_STATUS = ("sent", "overdue", "paid")
EXPENSE_CATEGORIES = (
    "office", "travel", "utilities", "software", "hardware",
    "food", "transport", "rent", "services", "other",
)


def dec(value) -> Decimal:
    """Parse a stored or submitted value without binary float arithmetic."""
    if value is None or value == "":
        return ZERO
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return ZERO


def q2(value) -> Decimal:
    return dec(value).quantize(CENT, rounding=ROUND_HALF_UP)


def q4(value) -> Decimal:
    return dec(value).quantize(Q4, rounding=ROUND_HALF_UP)


def s2(value) -> str:
    return format(q2(value), "f")


def s4(value) -> str:
    return format(q4(value), "f")


def json_default(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_iso_date(value) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def invoice_total(invoice: dict) -> Decimal:
    return q2(dec((invoice.get("totals") or {}).get("total")))


def invoice_subtotal(invoice: dict) -> Decimal:
    return q2(dec((invoice.get("totals") or {}).get("subtotal")))


def invoice_tax(invoice: dict) -> Decimal:
    return q2(dec((invoice.get("totals") or {}).get("tax_amount")))


def expense_amount(expense: dict) -> Decimal:
    return q2(dec(expense.get("amount")))


def issued_invoices(invoices: list[dict]) -> list[dict]:
    return [invoice for invoice in invoices if invoice.get("status") in ISSUED_STATUS]


def in_period(value, start: date, end: date) -> bool:
    parsed = parse_iso_date(value)
    if parsed is None:
        return False
    return start <= parsed <= end

# ===========================================================================
# Section 2: AI Provider Abstraction
# ===========================================================================


import json
import os
import re
import time
import urllib.error
import urllib.request

from typing import Any, Callable, Optional

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
MAX_IMAGE_PARTS = 8


class ProviderError(RuntimeError):
    """A safe, user-facing AI provider failure.

    `provider` names the backend ("gemini" or "nvidia") and `kind` classifies
    the failure so the web layer can decide whether to fall back quietly.
    """

    def __init__(self, message: str, provider: str = "", kind: str = "error") -> None:
        super().__init__(message)
        self.provider = provider
        self.kind = kind


@dataclass
class AIConfig:
    """Raw provider configuration. Keys live only in memory here."""

    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    nvidia_api_key: str = ""
    nvidia_model: str = DEFAULT_NVIDIA_MODEL
    nvidia_base_url: str = DEFAULT_NVIDIA_BASE_URL
    nvidia_enable_thinking: bool = False
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    # Injected transports make the module testable without network access.
    urlopen: Callable[..., Any] = dataclass_field(default=urllib.request.urlopen, repr=False)
    sleep: Callable[[float], None] = dataclass_field(default=time.sleep, repr=False)

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "AIConfig":
        source = os.environ if env is None else env

        def value(name: str, default: str = "") -> str:
            return str(source.get(name, default) or "").strip()

        def flag(name: str, default: bool = False) -> bool:
            raw = value(name).lower()
            if raw in ("1", "true", "yes", "on"):
                return True
            if raw in ("0", "false", "no", "off"):
                return False
            return default

        def number(name: str, default: int) -> int:
            try:
                parsed = int(float(value(name)))
            except (TypeError, ValueError):
                return default
            return parsed if parsed > 0 else default

        return cls(
            gemini_api_key=value("GEMINI_API_KEY"),
            gemini_model=value("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL,
            nvidia_api_key=value("NVIDIA_API_KEY"),
            nvidia_model=value("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL) or DEFAULT_NVIDIA_MODEL,
            nvidia_base_url=(value("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL) or DEFAULT_NVIDIA_BASE_URL).rstrip("/"),
            nvidia_enable_thinking=flag("NVIDIA_ENABLE_THINKING", False),
            timeout_seconds=number("AI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_retries=number("AI_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        )

    def secrets(self) -> tuple[str, ...]:
        return tuple(secret for secret in (self.gemini_api_key, self.nvidia_api_key) if secret)


def gemini_available(cfg: AIConfig) -> bool:
    return bool(cfg.gemini_api_key)


def nvidia_available(cfg: AIConfig) -> bool:
    return bool(cfg.nvidia_api_key)


def redact(text: str, cfg: AIConfig) -> str:
    """Blank out anything that looks like an API key before returning text."""
    cleaned = str(text or "")
    for secret in cfg.secrets():
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = re.sub(r"nvapi-[0-9A-Za-z_\-]{10,}", "[redacted]", cleaned)
    cleaned = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[redacted]", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def _post_json(cfg: AIConfig, url: str, payload: dict, headers: dict) -> dict:
    """POST JSON and decode the reply, retrying transient failures."""
    body = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST"
        )
        try:
            with cfg.urlopen(request, timeout=cfg.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < cfg.max_retries:
                attempt += 1
                cfg.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < cfg.max_retries:
                attempt += 1
                cfg.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise
        except ValueError:
            # A body that is not valid JSON is not worth retrying.
            raise


def _provider_message(provider: str, error: urllib.error.HTTPError, label: str) -> str:
    name = "Gemini" if provider == "gemini" else "NVIDIA NIM"
    if error.code in (401, 403):
        return f"{name} rejected the API key. Check {label} in your .env file."
    if error.code == 429:
        return f"{name} is rate limited right now. Please try again shortly."
    if error.code == 404:
        return f"{name} model was not found. Check the configured model name."
    if error.code >= 500:
        return f"{name} had a server error (HTTP {error.code}). Try again later."
    return f"{name} request failed (HTTP {error.code}). Try again later."


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _gemini_text(cfg: AIConfig, system_prompt: str, parts: list[dict],
                 max_output_tokens: int, temperature: float) -> str:
    if not gemini_available(cfg):
        raise ProviderError("Gemini is not configured.", "gemini", "unconfigured")
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }
    try:
        payload = _post_json(
            cfg,
            GEMINI_ENDPOINT.format(model=cfg.gemini_model),
            body,
            {"x-goog-api-key": cfg.gemini_api_key},
        )
    except urllib.error.HTTPError as error:
        raise ProviderError(
            _provider_message("gemini", error, "GEMINI_API_KEY"), "gemini", "http"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ProviderError(
            "Gemini could not be reached. Check your connection and try again.", "gemini", "network"
        ) from error
    except ValueError as error:
        raise ProviderError("Gemini returned an unreadable response.", "gemini", "malformed") from error

    try:
        candidates = payload.get("candidates") or []
        reply_parts = candidates[0]["content"]["parts"]
        text = "\n".join(part.get("text", "") for part in reply_parts).strip()
    except (IndexError, KeyError, TypeError) as error:
        raise ProviderError("Gemini returned an empty response. Try again.", "gemini", "empty") from error
    if not text:
        raise ProviderError("Gemini returned an empty response. Try again.", "gemini", "empty")
    return redact(text, cfg)


def _parse_json_object(reply: str, provider: str) -> dict:
    cleaned = str(reply or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ProviderError("The AI returned a response that is not valid JSON. Try again.", provider, "invalid_json")
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as error:
        raise ProviderError("The AI returned a response that is not valid JSON. Try again.", provider, "invalid_json") from error
    if not isinstance(parsed, dict):
        raise ProviderError("The AI returned an unexpected JSON shape. Try again.", provider, "invalid_json")
    return parsed


def gemini_text(cfg: AIConfig, system_prompt: str, parts: list[dict],
                max_output_tokens: int = 700, temperature: float = 0.25) -> str:
    return _gemini_text(cfg, system_prompt, parts, max_output_tokens, temperature)


def gemini_json(cfg: AIConfig, system_prompt: str, parts: list[dict],
                max_output_tokens: int = 1400, temperature: float = 0.1) -> dict:
    reply = _gemini_text(cfg, system_prompt, parts, max_output_tokens, temperature)
    return _parse_json_object(reply, "gemini")


# ---------------------------------------------------------------------------
# NVIDIA NIM (OpenAI-compatible)
# ---------------------------------------------------------------------------

def to_openai_parts(parts: list[dict]) -> list[dict]:
    """Translate Gemini-style content parts into OpenAI-compatible parts."""
    converted: list[dict] = []
    images = 0
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            converted.append({"type": "text", "text": str(part.get("text") or "")})
            continue
        inline = part.get("inline_data")
        if isinstance(inline, dict):
            mime = str(inline.get("mime_type") or "image/png")
            data = str(inline.get("data") or "")
            if not data or not mime.startswith("image/"):
                continue
            if images >= MAX_IMAGE_PARTS:
                continue
            images += 1
            converted.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
    return converted


def _nvidia_chat(cfg: AIConfig, messages: list[dict], max_tokens: int,
                 temperature: float, top_p: float = 1.0) -> str:
    if not nvidia_available(cfg):
        raise ProviderError("NVIDIA NIM is not configured.", "nvidia", "unconfigured")
    body: dict[str, Any] = {
        "model": cfg.nvidia_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(cfg.nvidia_enable_thinking)},
    }
    url = f"{cfg.nvidia_base_url}/v1/chat/completions"
    try:
        payload = _post_json(cfg, url, body, {"Authorization": f"Bearer {cfg.nvidia_api_key}"})
    except urllib.error.HTTPError as error:
        raise ProviderError(
            _provider_message("nvidia", error, "NVIDIA_API_KEY"), "nvidia", "http"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ProviderError(
            "NVIDIA NIM could not be reached. Check your connection and try again.", "nvidia", "network"
        ) from error
    except ValueError as error:
        raise ProviderError("NVIDIA NIM returned an unreadable response.", "nvidia", "malformed") from error

    try:
        choices = payload.get("choices") or []
        message = choices[0]["message"]
        text = message.get("content") or ""
        if not text and message.get("reasoning_content"):
            text = message["reasoning_content"]
        text = str(text).strip()
    except (IndexError, KeyError, TypeError) as error:
        raise ProviderError("NVIDIA NIM returned an empty response. Try again.", "nvidia", "empty") from error
    if not text:
        raise ProviderError("NVIDIA NIM returned an empty response. Try again.", "nvidia", "empty")
    return redact(text, cfg)


def nvidia_text(cfg: AIConfig, system_prompt: str, parts: list[dict],
                max_tokens: int = 900, temperature: float = 0.2) -> str:
    messages = [
        {"role": "system", "content": str(system_prompt or "")},
        {"role": "user", "content": to_openai_parts(parts)},
    ]
    return _nvidia_chat(cfg, messages, max_tokens, temperature)


def nvidia_json(cfg: AIConfig, system_prompt: str, parts: list[dict],
                max_tokens: int = 1600, temperature: float = 0.1) -> dict:
    reply = nvidia_text(cfg, system_prompt, parts, max_tokens=max_tokens, temperature=temperature)
    return _parse_json_object(reply, "nvidia")


def document_json(cfg: AIConfig, system_prompt: str, parts: list[dict],
                  max_output_tokens: int = 2400) -> dict:
    """Route document extraction to NIM, falling back to Gemini.

    NIM (Nemotron Omni) handles image understanding, including rendered PDF
    pages. Gemini remains the fallback so a deployment with only a Gemini key
    keeps working.
    """
    if nvidia_available(cfg):
        return nvidia_json(cfg, system_prompt, parts, max_tokens=max_output_tokens, temperature=0.1)
    if gemini_available(cfg):
        return gemini_json(cfg, system_prompt, parts, max_output_tokens=max_output_tokens, temperature=0.1)
    raise ProviderError(
        "The AI is not configured. Add NVIDIA_API_KEY (or GEMINI_API_KEY) to your "
        ".env file and restart the app to read documents.",
        "",
        "unconfigured",
    )


def provider_status(cfg: AIConfig) -> dict:
    """Non-secret status for the UI."""
    return {
        "gemini_configured": gemini_available(cfg),
        "nvidia_configured": nvidia_available(cfg),
        "gemini_model": cfg.gemini_model,
        "nvidia_model": cfg.nvidia_model,
        "summary_provider": "gemini" if gemini_available(cfg) else "local",
        "other_provider": "nvidia" if nvidia_available(cfg) else ("gemini" if gemini_available(cfg) else "local"),
    }

# ===========================================================================
# Section 3: SpendWise Analytics Engine
# ===========================================================================

from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Optional


# ---------------------------------------------------------------------------
# Reporting periods
# ---------------------------------------------------------------------------

PERIODS = {
    "7d": "Last 7 days",
    "30d": "Last 30 days",
    "month": "Current month",
    "last_month": "Previous month",
    "3m": "Last 3 months",
    "6m": "Last 6 months",
    "year": "Current year",
    "custom": "Custom range",
}

MONTH_LABELS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

# A spending change is only called out beyond this percentage.
NOTABLE_CHANGE_PCT = Decimal("15")
# A budget is "nearly exhausted" at this utilisation.
BUDGET_NEAR_PCT = Decimal("80")
# Unusual spending needs a comparable history and a clear multiple.
UNUSUAL_RATIO = Decimal("3")
UNUSUAL_MIN_HISTORY = 5
UNUSUAL_LOOKBACK_MONTHS = 6
# Recurring suggestions need this many charges before they are mentioned.
RECURRING_MIN_CHARGES = 3
RECURRING_MIN_MONTHS = 2
RECURRING_MIN_INTERVAL_DAYS = 25
RECURRING_MAX_INTERVAL_DAYS = 35


class SpendWiseError(ValueError):
    """A user-facing configuration or period error."""


def month_start(value: date) -> date:
    return value.replace(day=1)


def month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def month_label(value: date) -> str:
    return f"{MONTH_LABELS[value.month - 1]} {value.year}"


def add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    day = min(value.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def resolve_period(key: str, from_date: Optional[str], to_date: Optional[str],
                   today: Optional[date] = None) -> dict:
    """Return the reporting window and the matching previous window."""
    today = today or date.today()
    key = (key or "month").strip()
    if key not in PERIODS:
        raise SpendWiseError(f"Unknown period '{key}'. Choose one of: {', '.join(PERIODS)}.")

    if key == "7d":
        start, end = today - timedelta(days=6), today
    elif key == "30d":
        start, end = today - timedelta(days=29), today
    elif key == "month":
        start, end = month_start(today), today
    elif key == "last_month":
        first = add_months(month_start(today), -1)
        start, end = first, first.replace(day=_days_in_month(first.year, first.month))
    elif key == "3m":
        start, end = add_months(month_start(today), -2), today
    elif key == "6m":
        start, end = add_months(month_start(today), -5), today
    elif key == "year":
        start, end = date(today.year, 1, 1), today
    else:  # custom
        start = parse_iso_date(from_date)
        end = parse_iso_date(to_date)
        if start is None or end is None:
            raise SpendWiseError("A custom period needs both a start date and an end date.")
        if end < start:
            raise SpendWiseError("The end date cannot be before the start date.")

    days = (end - start).days + 1
    previous_end = start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=days - 1)
    return {
        "key": key,
        "label": PERIODS[key],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": days,
        "previous": {
            "start": previous_start.isoformat(),
            "end": previous_end.isoformat(),
            "days": days,
        },
    }


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_expenses(expenses: list[dict], start: date, end: date) -> list[dict]:
    return [expense for expense in expenses if in_period(expense.get("date_incurred"), start, end)]


def _sum_amounts(expenses: list[dict]) -> Decimal:
    return sum((expense_amount(expense) for expense in expenses), ZERO)


def _percent(part: Decimal, whole: Decimal) -> Decimal:
    if whole <= 0:
        return ZERO
    return q2(part * Decimal("100") / whole)


def _change(current: Decimal, previous: Decimal) -> dict:
    difference = current - previous
    if previous <= 0:
        percent = None if current <= 0 else Decimal("100")
    else:
        percent = q2(difference * Decimal("100") / previous)
    return {"amount": s2(difference), "percent": (s2(percent) if percent is not None else None)}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spending_stats(expenses: list[dict], period: dict) -> dict:
    """Headline spending figures for the selected period."""
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    previous_start = parse_iso_date(period["previous"]["start"])
    previous_end = parse_iso_date(period["previous"]["end"])

    current = filter_expenses(expenses, start, end)
    previous = filter_expenses(expenses, previous_start, previous_end)
    total = _sum_amounts(current)
    previous_total = _sum_amounts(previous)

    month_total = _sum_amounts([
        expense for expense in current
        if (parse_iso_date(expense.get("date_incurred")) or start) >= month_start(start)
    ])

    months_spanned = _month_span(start, end)
    average_monthly = q2(total / months_spanned) if months_spanned else ZERO

    category_totals = _group_totals(current, lambda expense: expense.get("category") or "other")
    vendor_totals = _group_totals(current, vendor_key)

    top_category = _top(category_totals)
    top_vendor = _top(vendor_totals)

    return {
        "total": s2(total),
        "count": len(current),
        "average_transaction": s2(q2(total / len(current)) if current else ZERO),
        "month_total": s2(month_total),
        "average_monthly": s2(average_monthly),
        "months_spanned": months_spanned,
        "previous_total": s2(previous_total),
        "change": _change(total, previous_total),
        "top_category": top_category,
        "top_vendor": top_vendor,
        "daily_average": s2(q2(total / period["days"]) if period["days"] else ZERO),
    }


def _month_span(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    return max(months, 1)


def vendor_key(expense: dict) -> str:
    vendor = str(expense.get("vendor") or "").strip()
    return vendor if vendor else "(no vendor)"


def _group_totals(expenses: list[dict], key) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {}
    for expense in expenses:
        bucket = key(expense)
        totals[bucket] = totals.get(bucket, ZERO) + expense_amount(expense)
    return totals


def _top(totals: dict[str, Decimal]) -> Optional[dict]:
    if not totals:
        return None
    name, amount = max(totals.items(), key=lambda item: (item[1], item[0]))
    return {"name": name, "amount": s2(amount)}


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

def monthly_trend(expenses: list[dict], start: date, end: date) -> list[dict]:
    buckets: dict[str, dict] = {}
    cursor = month_start(start)
    while cursor <= end:
        key = month_key(cursor)
        buckets[key] = {"key": key, "label": month_label(cursor), "total": ZERO, "count": 0}
        cursor = add_months(cursor, 1)

    for expense in expenses:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed is None:
            continue
        key = month_key(parsed)
        if key in buckets:
            buckets[key]["total"] += expense_amount(expense)
            buckets[key]["count"] += 1

    return [
        {"key": bucket["key"], "label": bucket["label"], "total": s2(bucket["total"]),
         "count": bucket["count"]}
        for bucket in buckets.values()
    ]


def weekly_trend(expenses: list[dict], start: date, end: date) -> list[dict]:
    buckets: dict[str, dict] = {}
    cursor = start - timedelta(days=start.weekday())
    while cursor <= end:
        key = cursor.isoformat()
        buckets[key] = {"key": key, "label": cursor.strftime("%d %b"), "total": ZERO, "count": 0}
        cursor += timedelta(days=7)

    for expense in expenses:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed is None:
            continue
        week_start = parsed - timedelta(days=parsed.weekday())
        key = week_start.isoformat()
        if key in buckets:
            buckets[key]["total"] += expense_amount(expense)
            buckets[key]["count"] += 1

    return [
        {"key": bucket["key"], "label": bucket["label"], "total": s2(bucket["total"]),
         "count": bucket["count"]}
        for bucket in buckets.values()
    ]


def category_breakdown(expenses: list[dict], period: dict) -> list[dict]:
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    current = filter_expenses(expenses, start, end)
    previous = filter_expenses(
        expenses,
        parse_iso_date(period["previous"]["start"]),
        parse_iso_date(period["previous"]["end"]),
    )
    total = _sum_amounts(current)
    previous_totals = _group_totals(previous, lambda expense: expense.get("category") or "other")

    rows = []
    for category, amount in _group_totals(current, lambda expense: expense.get("category") or "other").items():
        count = sum(
            1 for expense in current
            if (expense.get("category") or "other") == category
        )
        previous_total = previous_totals.get(category, ZERO)
        rows.append({
            "category": category,
            "total": s2(amount),
            "share": s2(_percent(amount, total)),
            "count": count,
            "average": s2(q2(amount / count) if count else ZERO),
            "previous_total": s2(previous_total),
            "change": _change(amount, previous_total),
        })
    rows.sort(key=lambda row: (dec(row["total"]), row["category"]), reverse=True)
    return rows


def vendor_breakdown(expenses: list[dict], period: dict, limit: int = 12) -> list[dict]:
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    current = filter_expenses(expenses, start, end)
    previous = filter_expenses(
        expenses,
        parse_iso_date(period["previous"]["start"]),
        parse_iso_date(period["previous"]["end"]),
    )
    total = _sum_amounts(current)
    previous_totals = _group_totals(previous, vendor_key)

    rows = []
    for vendor, amount in _group_totals(current, vendor_key).items():
        count = sum(1 for expense in current if vendor_key(expense) == vendor)
        previous_total = previous_totals.get(vendor, ZERO)
        rows.append({
            "vendor": vendor,
            "total": s2(amount),
            "share": s2(_percent(amount, total)),
            "count": count,
            "average": s2(q2(amount / count) if count else ZERO),
            "previous_total": s2(previous_total),
            "change": _change(amount, previous_total),
        })
    rows.sort(key=lambda row: (dec(row["total"]), row["vendor"]), reverse=True)
    return rows[:limit]


def spending_matrix(expenses: list[dict], start: date, end: date, group_by) -> dict:
    """Spending per month for each group (category or vendor).

    Returns the month columns and one row per group with a value per month, so
    the same data can drive a category-over-time and a vendor-over-time view.
    """
    months: list[dict] = []
    cursor = month_start(start)
    while cursor <= end:
        months.append({"key": month_key(cursor), "label": month_label(cursor)})
        cursor = add_months(cursor, 1)
    month_index = {month["key"]: position for position, month in enumerate(months)}

    rows: dict[str, list[Decimal]] = {}
    totals: dict[str, Decimal] = {}
    for expense in expenses:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed is None or not (start <= parsed <= end):
            continue
        position = month_index.get(month_key(parsed))
        if position is None:
            continue
        name = group_by(expense)
        if name not in rows:
            rows[name] = [ZERO for _ in months]
            totals[name] = ZERO
        rows[name][position] += expense_amount(expense)
        totals[name] += expense_amount(expense)

    result = []
    for name, values in rows.items():
        result.append({
            "name": name,
            "total": s2(totals[name]),
            "values": [s2(value) for value in values],
        })
    result.sort(key=lambda row: (dec(row["total"]), row["name"]), reverse=True)
    return {"months": months, "rows": result}


def category_transactions(expenses: list[dict], category: str) -> list[dict]:
    rows = []
    for expense in expenses:
        if (expense.get("category") or "other") != category:
            continue
        rows.append({
            "id": expense.get("id"),
            "date": expense.get("date_incurred"),
            "description": expense.get("description"),
            "vendor": expense.get("vendor"),
            "category": expense.get("category"),
            "amount": s2(expense_amount(expense)),
            "reimbursable": bool(expense.get("reimbursable")),
            "reimbursed": bool(expense.get("reimbursed")),
        })
    rows.sort(key=lambda row: (str(row["date"]), row["id"] or 0), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def _budget_scope_expenses(expenses: list[dict], budget: dict, anchor: date) -> list[dict]:
    month = str(budget.get("month") or "").strip() or month_key(anchor)
    category = str(budget.get("category") or "").strip()
    start = date.fromisoformat(f"{month}-01")
    end = start.replace(day=_days_in_month(start.year, start.month))
    scoped = [
        expense for expense in expenses
        if in_period(expense.get("date_incurred"), start, end)
    ]
    if category:
        scoped = [
            expense for expense in scoped
            if (expense.get("category") or "other") == category
        ]
    return scoped


def budget_status(budgets: list[dict], expenses: list[dict], anchor: date) -> list[dict]:
    rows = []
    for budget in budgets:
        if not budget.get("enabled", True):
            continue
        amount = dec(budget.get("amount", 0))
        scoped = _budget_scope_expenses(expenses, budget, anchor)
        spent = _sum_amounts(scoped)
        remaining = amount - spent
        percent = _percent(spent, amount)
        if amount <= 0:
            state = "unset"
        elif spent > amount:
            state = "over"
        elif percent >= BUDGET_NEAR_PCT:
            state = "near"
        else:
            state = "on_track"
        rows.append({
            "id": budget.get("id"),
            "name": budget.get("name") or budget.get("category") or "Monthly budget",
            "category": budget.get("category") or "",
            "month": budget.get("month") or month_key(anchor),
            "amount": s2(amount),
            "spent": s2(spent),
            "remaining": s2(remaining),
            "percent": s2(percent),
            "state": state,
            "exceeded": state == "over",
            "transactions": len(scoped),
        })
    rows.sort(key=lambda row: (
        {"over": 0, "near": 1, "on_track": 2, "unset": 3}.get(row["state"], 4),
        -dec(row["amount"]),
    ))
    return rows


def budget_totals(budgets: list[dict], expenses: list[dict], anchor: date) -> dict:
    """Budget summary.

    Budgets are reported per scope. When more than one budget applies, the
    figures are not added together, because two budgets can cover the same
    spending. The most used budget is reported as the headline.
    """
    rows = budget_status(budgets, expenses, anchor)
    if not rows:
        return {
            "count": 0,
            "budget": "0.00",
            "spent": "0.00",
            "remaining": "0.00",
            "percent": "0.00",
            "primary": None,
            "over_budget": [],
            "combined": False,
        }
    primary = max(rows, key=lambda row: (dec(row["percent"]), dec(row["amount"])))
    budget_amount = sum((dec(row["amount"]) for row in rows), ZERO)
    spent = sum((dec(row["spent"]) for row in rows), ZERO)
    return {
        "count": len(rows),
        "budget": s2(budget_amount),
        "spent": s2(spent),
        "remaining": s2(budget_amount - spent),
        "percent": s2(_percent(spent, budget_amount)),
        "primary": {
            "id": primary["id"],
            "name": primary["name"],
            "amount": primary["amount"],
            "spent": primary["spent"],
            "percent": primary["percent"],
            "state": primary["state"],
        },
        "over_budget": [row["name"] for row in rows if row["exceeded"]],
        "combined": len(rows) > 1,
    }


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

def recurring_suggestions(expenses: list[dict]) -> list[dict]:
    """Suggest possible recurring charges. These are suggestions, not confirmed subscriptions."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for expense in expenses:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed is None:
            continue
        vendor = str(expense.get("vendor") or "").strip()
        description = str(expense.get("description") or "").strip().lower()
        if not vendor and not description:
            continue
        key = (vendor.lower(), description)
        groups.setdefault(key, []).append(expense)

    suggestions = []
    for (vendor, description), items in groups.items():
        if len(items) < RECURRING_MIN_CHARGES:
            continue
        dates = sorted(
            parse_iso_date(item.get("date_incurred")) for item in items
            if parse_iso_date(item.get("date_incurred"))
        )
        if len({month_key(value) for value in dates}) < RECURRING_MIN_MONTHS:
            continue
        intervals = [
            (later - earlier).days for earlier, later in zip(dates, dates[1:])
        ]
        if not intervals:
            continue
        average_interval = sum(intervals) / len(intervals)
        if not RECURRING_MIN_INTERVAL_DAYS <= average_interval <= RECURRING_MAX_INTERVAL_DAYS:
            continue
        amounts = [expense_amount(item) for item in items]
        average_amount = q2(sum(amounts, ZERO) / len(amounts))
        suggestions.append({
            "description": items[-1].get("description"),
            "vendor": items[-1].get("vendor") or "(no vendor)",
            "charges": len(items),
            "months": len({month_key(value) for value in dates}),
            "average_amount": s2(average_amount),
            "average_interval_days": int(round(average_interval)),
            "cadence": "monthly",
            "first": dates[0].isoformat(),
            "last": dates[-1].isoformat(),
        })
    suggestions.sort(key=lambda row: (dec(row["average_amount"]), row["charges"]), reverse=True)
    return suggestions


def unusual_expenses(expenses: list[dict]) -> list[dict]:
    """Flag transactions that are large against their own history.

    A transaction needs at least five comparable earlier charges in the same
    category and must be three times the median of that history. The reason is
    reported with every flag so the user can judge it.
    """
    dated = []
    for expense in expenses:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed is not None:
            dated.append((parsed, expense))
    dated.sort(key=lambda item: item[0])

    flags = []
    for index, (when, expense) in enumerate(dated):
        category = expense.get("category") or "other"
        history = [
            expense_amount(other)
            for other_when, other in dated[:index]
            if other_when >= when - timedelta(days=30 * UNUSUAL_LOOKBACK_MONTHS)
            and (other.get("category") or "other") == category
        ]
        if len(history) < UNUSUAL_MIN_HISTORY:
            continue
        middle = Decimal(median(history))
        if middle <= 0:
            continue
        amount = expense_amount(expense)
        if amount < middle * UNUSUAL_RATIO:
            continue
        if amount <= max(history):
            continue
        flags.append({
            "id": expense.get("id"),
            "date": expense.get("date_incurred"),
            "description": expense.get("description"),
            "vendor": expense.get("vendor") or "(no vendor)",
            "category": category,
            "amount": s2(amount),
            "median": s2(middle),
            "ratio": s2(q2(amount / middle)),
            "history_count": len(history),
            "reason": (
                f"{s2(amount)} is {s2(q2(amount / middle))}× the median {category} "
                f"expense of {s2(middle)} across {len(history)} comparable charges "
                f"in the previous {UNUSUAL_LOOKBACK_MONTHS} months."
            ),
        })
    flags.sort(key=lambda row: dec(row["amount"]), reverse=True)
    return flags


DUPLICATE_WINDOW_DAYS = 4


def duplicate_charges(expenses: list[dict]) -> list[dict]:
    """Flag charges that look like accidental duplicates.

    Two expenses with the same vendor and the same amount within a few days are
    worth a human check. Nothing is deleted or merged; the reason is reported so
    the user can judge it.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for expense in expenses:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed is None:
            continue
        vendor = str(expense.get("vendor") or "").strip().lower()
        amount = s2(expense_amount(expense))
        if not vendor or dec(amount) <= 0:
            continue
        groups.setdefault((vendor, amount), []).append(expense)

    flags = []
    for (vendor, amount), items in groups.items():
        if len(items) < 2:
            continue
        dated = sorted(items, key=lambda row: parse_iso_date(row.get("date_incurred")))
        duplicate_days = []
        for earlier, later in zip(dated, dated[1:]):
            gap = (parse_iso_date(later.get("date_incurred"))
                   - parse_iso_date(earlier.get("date_incurred"))).days
            if 0 <= gap <= DUPLICATE_WINDOW_DAYS:
                duplicate_days.append(gap)
        if not duplicate_days:
            continue
        flags.append({
            "vendor": dated[-1].get("vendor") or "(no vendor)",
            "amount": amount,
            "count": len(items),
            "ids": [row.get("id") for row in items],
            "dates": [parse_iso_date(row.get("date_incurred")).isoformat() for row in dated],
            "reason": (
                f"{len(items)} charges of {amount} at {dated[-1].get('vendor') or '(no vendor)'} "
                f"within {max(duplicate_days)} day(s) of each other. Check whether one is a "
                "duplicate before acting."
            ),
        })
    flags.sort(key=lambda row: dec(row["amount"]), reverse=True)
    return flags


def build_insights(expenses: list[dict], invoices: list[dict], period: dict,
                   budgets: list[dict], rules: list[dict]) -> list[dict]:
    """Deterministic observations about the selected period."""
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    current = filter_expenses(expenses, start, end)
    previous = filter_expenses(
        expenses,
        parse_iso_date(period["previous"]["start"]),
        parse_iso_date(period["previous"]["end"]),
    )
    total = _sum_amounts(current)
    previous_total = _sum_amounts(previous)
    insights: list[dict] = []

    if previous_total > 0 and current:
        change = _change(total, previous_total)
        percent = dec(change["percent"] or 0)
        if abs(percent) >= NOTABLE_CHANGE_PCT:
            direction = "up" if percent > 0 else "down"
            insights.append({
                "id": "period-change",
                "severity": "notice" if direction == "up" else "positive",
                "title": f"Spending {direction} {s2(abs(percent))}% against the previous period",
                "detail": (
                    f"{s2(total)} recorded between {period['start']} and {period['end']}, "
                    f"against {s2(previous_total)} in the previous {period['days']} days."
                ),
            })

    if total > 0:
        categories = category_breakdown(expenses, period)
        if categories:
            top = categories[0]
            share = dec(top["share"])
            if share >= Decimal("35"):
                insights.append({
                    "id": "category-concentration",
                    "severity": "info",
                    "title": f"{top['category'].capitalize()} is {s2(share)}% of recorded spending",
                    "detail": (
                        f"{s2(top['total'])} across {top['count']} transactions, "
                        f"averaging {top['average']} each."
                    ),
                })
        vendors = vendor_breakdown(expenses, period, limit=1)
        if vendors:
            top_vendor = vendors[0]
            share = dec(top_vendor["share"])
            if share >= Decimal("30"):
                insights.append({
                    "id": "vendor-concentration",
                    "severity": "info",
                    "title": f"{top_vendor['vendor']} accounts for {s2(share)}% of spending",
                    "detail": (
                        f"{s2(top_vendor['total'])} across {top_vendor['count']} transactions "
                        f"in this period."
                    ),
                })

    for row in budget_status(budgets, expenses, end):
        if row["state"] == "over":
            insights.append({
                "id": f"budget-over-{row['id']}",
                "severity": "notice",
                "title": f"{row['name']} budget exceeded",
                "detail": (
                    f"{row['spent']} spent against a {row['amount']} budget "
                    f"({row['percent']}% used)."
                ),
            })
        elif row["state"] == "near":
            insights.append({
                "id": f"budget-near-{row['id']}",
                "severity": "notice",
                "title": f"{row['name']} budget is nearly exhausted",
                "detail": (
                    f"{row['spent']} of {row['amount']} used ({row['percent']}%), "
                    f"{row['remaining']} remaining."
                ),
            })

    for flag in unusual_expenses(current)[:3]:
        insights.append({
            "id": f"unusual-{flag['id']}",
            "severity": "notice",
            "title": f"Unusually large {flag['category']} expense: {flag['description']}",
            "detail": flag["reason"],
        })

    for suggestion in recurring_suggestions(expenses)[:3]:
        insights.append({
            "id": f"recurring-{suggestion['vendor']}-{suggestion['description']}",
            "severity": "info",
            "title": f"Possible recurring {suggestion['cadence']} charge: {suggestion['description']}",
            "detail": (
                f"{suggestion['charges']} charges over {suggestion['months']} months at "
                f"{suggestion['vendor']}, averaging {suggestion['average_amount']}. "
                "Suggested pattern, not a confirmed subscription."
            ),
        })

    for flag in duplicate_charges(current)[:3]:
        insights.append({
            "id": f"duplicate-{flag['vendor']}-{flag['amount']}",
            "severity": "notice",
            "title": f"Possible duplicate charge at {flag['vendor']}",
            "detail": flag["reason"],
        })

    return insights


# ---------------------------------------------------------------------------
# Cash movement (not accrual profit)
# ---------------------------------------------------------------------------

CASH_FLOW_NOTE = (
    "Cash movement view. Paid invoices are dated by their issue date because "
    "payment dates are not recorded, and recorded expenses are treated as settled "
    "outflows. These figures are not an accrual profit and loss result."
)


def cash_flow(invoices: list[dict], expenses: list[dict], period: dict) -> dict:
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    paid = [
        invoice for invoice in invoices
        if invoice.get("status") == "paid" and in_period(invoice.get("issue_date"), start, end)
    ]
    outgoing = filter_expenses(expenses, start, end)
    incoming_total = sum((invoice_total(invoice) for invoice in paid), ZERO)
    outgoing_total = _sum_amounts(outgoing)

    months: dict[str, dict] = {}
    cursor = month_start(start)
    while cursor <= end:
        key = month_key(cursor)
        months[key] = {"key": key, "label": month_label(cursor), "incoming": ZERO, "outgoing": ZERO}
        cursor = add_months(cursor, 1)

    for invoice in paid:
        parsed = parse_iso_date(invoice.get("issue_date"))
        if parsed and month_key(parsed) in months:
            months[month_key(parsed)]["incoming"] += invoice_total(invoice)
    for expense in outgoing:
        parsed = parse_iso_date(expense.get("date_incurred"))
        if parsed and month_key(parsed) in months:
            months[month_key(parsed)]["outgoing"] += expense_amount(expense)

    return {
        "basis": "cash",
        "note": CASH_FLOW_NOTE,
        "incoming": s2(incoming_total),
        "outgoing": s2(outgoing_total),
        "net": s2(incoming_total - outgoing_total),
        "incoming_count": len(paid),
        "outgoing_count": len(outgoing),
        "months": [
            {
                "key": row["key"],
                "label": row["label"],
                "incoming": s2(row["incoming"]),
                "outgoing": s2(row["outgoing"]),
                "net": s2(row["incoming"] - row["outgoing"]),
            }
            for row in months.values()
        ],
    }


# ---------------------------------------------------------------------------
# Categorisation rules
# ---------------------------------------------------------------------------

def _rule_specificity(rule: dict) -> int:
    return (1 if rule.get("vendor") else 0) + (1 if rule.get("keyword") else 0)


def ordered_rules(rules: list[dict]) -> list[dict]:
    """Deterministic rule order: priority, then specificity, then creation order."""
    return sorted(
        rules,
        key=lambda rule: (
            0 if rule.get("enabled", True) else 1,
            int(rule.get("priority", 100) or 100),
            -_rule_specificity(rule),
            int(rule.get("id", 0) or 0),
        ),
    )


def suggest_category(rules: list[dict], expense: dict) -> Optional[dict]:
    """Return the category a rule suggests for one expense, or None."""
    vendor = str(expense.get("vendor") or "").strip().lower()
    description = str(expense.get("description") or "").strip().lower()
    for rule in ordered_rules(rules):
        if not rule.get("enabled", True):
            continue
        rule_vendor = str(rule.get("vendor") or "").strip().lower()
        rule_keyword = str(rule.get("keyword") or "").strip().lower()
        if not rule_vendor and not rule_keyword:
            continue
        if rule_vendor and rule_vendor not in vendor:
            continue
        if rule_keyword and rule_keyword not in description:
            continue
        return {
            "rule_id": rule.get("id"),
            "rule_name": rule.get("name") or "Rule",
            "category": rule.get("category"),
        }
    return None


def rule_suggestions(rules: list[dict], expenses: list[dict]) -> list[dict]:
    """Suggested categories for stored expenses.

    Manually assigned categories are reported but never changed here; the apply
    step only touches expenses still sitting in the default category.
    """
    rows = []
    for expense in expenses:
        suggestion = suggest_category(rules, expense)
        if not suggestion:
            continue
        current = expense.get("category") or "other"
        if current == suggestion["category"]:
            continue
        rows.append({
            "id": expense.get("id"),
            "date": expense.get("date_incurred"),
            "description": expense.get("description"),
            "vendor": expense.get("vendor") or "",
            "current_category": current,
            "suggested_category": suggestion["category"],
            "rule_id": suggestion["rule_id"],
            "rule_name": suggestion["rule_name"],
            "applyable": current == "other",
        })
    rows.sort(key=lambda row: (str(row["date"]), row["id"] or 0), reverse=True)
    return rows

# ===========================================================================
# Section 4: Spending Analysis
# ===========================================================================


from decimal import Decimal
from typing import Any



MAX_SUGGESTIONS = 6


def analysis_build_facts(expenses: list[dict], invoices: list[dict], period: dict,
                budgets: list[dict], rules: list[dict]) -> dict:
    """Assemble the deterministic spending facts for a period."""
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    stats = spending_stats(expenses, period)
    return {
        "period": {
            "label": period.get("label"),
            "start": period.get("start"),
            "end": period.get("end"),
            "days": period.get("days"),
        },
        "stats": stats,
        "categories": category_breakdown(expenses, period),
        "vendors": vendor_breakdown(expenses, period, limit=8),
        "recurring": recurring_suggestions(expenses),
        "unusual": unusual_expenses(
            filter_expenses(expenses, start, end)
        ),
        "duplicates": duplicate_charges(
            filter_expenses(expenses, start, end)
        ),
        "budgets": budget_status(budgets, expenses, end),
        "insights": build_insights(expenses, invoices, period, budgets, rules),
    }


def savings_potential(facts: dict) -> list[dict]:
    """Quantified, deterministic savings opportunities from the facts."""
    opportunities: list[dict] = []

    recurring = facts.get("recurring") or []
    if recurring:
        monthly = sum((dec(row.get("average_amount")) for row in recurring), ZERO)
        opportunities.append({
            "id": "recurring",
            "label": "Review recurring charges",
            "detail": (
                f"{len(recurring)} recurring charges total about {s2(monthly)} per month. "
                "Confirm which are still needed; cancelling an unused one removes its full cost."
            ),
            "monthly": s2(q2(monthly)),
            "annual": s2(q2(monthly) * 12),
        })

    duplicates = facts.get("duplicates") or []
    if duplicates:
        monthly = sum((dec(row.get("amount")) for row in duplicates), ZERO)
        opportunities.append({
            "id": "duplicates",
            "label": "Check possible duplicates",
            "detail": (
                f"{len(duplicates)} charge group(s) look duplicated. If confirmed, "
                f"about {s2(monthly)} could be recovered."
            ),
            "monthly": s2(q2(monthly)),
            "annual": s2(q2(monthly)),
        })

    over = [row for row in (facts.get("budgets") or []) if row.get("state") == "over"]
    if over:
        monthly = sum((-dec(row.get("remaining")) for row in over), ZERO)
        if monthly < ZERO:
            monthly = abs(monthly)
        opportunities.append({
            "id": "over-budget",
            "label": "Budgets over their limit",
            "detail": (
                f"{len(over)} budget(s) are over their limit by {s2(monthly)}. "
                "Bringing them back to plan recovers that amount."
            ),
            "monthly": s2(q2(monthly)),
            "annual": s2(q2(monthly) * 12),
        })

    return opportunities


def analysis_build_facts_text(facts: dict, currency: str) -> str:
    """A compact prompt block: totals, categories, and obvious flags only."""
    stats = facts.get("stats") or {}
    lines = [f"Currency: {currency}"]
    period = facts.get("period") or {}
    if period:
        lines.append(f"Period: {period.get('start')} to {period.get('end')}")
    lines.append(f"Total spending: {stats.get('total', '0')}")
    lines.append(f"Previous period: {stats.get('previous_total', '0')}")
    change = stats.get("change") or {}
    if change:
        lines.append(f"Change: {change.get('amount', '0')} ({change.get('percent', '0')}%)")

    categories = facts.get("categories") or []
    for row in categories[:5]:
        lines.append(
            f"Category {row['category']}: {row['total']} "
            f"({row['share']}% of spending, {row['count']} charges)"
        )

    budgets = facts.get("budgets") or []
    for row in budgets[:6]:
        lines.append(
            f"Budget {row['name']}: {row['spent']} of {row['amount']} used "
            f"({row['percent']}%, state {row['state']})"
        )

    for row in (facts.get("duplicates") or [])[:3]:
        lines.append(f"Possible duplicate: {row['vendor']} {row['amount']} x{row['count']}")

    for row in (facts.get("recurring") or [])[:5]:
        lines.append(
            f"Recurring {row['vendor']}: {row['average_amount']} monthly over {row['months']} months"
        )
    return "\n".join(lines)


def parse_suggestions(payload: Any) -> list[dict]:
    """Validate the model's suggestions. Figures never come from the model."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("suggestions")
    if not isinstance(raw, list):
        return []
    suggestions: list[dict] = []
    for entry in raw[:MAX_SUGGESTIONS]:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                suggestions.append({"title": text[:160], "detail": "", "refers_to": ""})
            continue
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or entry.get("action") or "").strip()
        if not title:
            continue
        suggestions.append({
            "title": title[:160],
            "detail": str(entry.get("detail") or entry.get("reason") or "").strip()[:400],
            "refers_to": str(entry.get("refers_to") or entry.get("fact") or "").strip()[:60],
        })
    return suggestions

# ===========================================================================
# Section 5: Subscription Detection & Optimisation
# ===========================================================================


from decimal import Decimal
from typing import Any, Optional



MAX_ALTERNATIVES = 6


def subscription_detect(expenses: list[dict], min_monthly: Decimal = Decimal("1")) -> list[dict]:
    """Find recurring charges that behave like subscriptions.

    Wraps the SpendWise recurring-charge detector and normalises each candidate
    to a monthly cost plus the yearly equivalent.
    """
    rows: list[dict] = []
    for suggestion in recurring_suggestions(expenses):
        monthly = q2(dec(suggestion.get("average_amount")))
        if monthly < min_monthly:
            continue
        rows.append({
            "key": f"{suggestion.get('vendor', '').lower()}|{str(suggestion.get('description', '')).lower()}",
            "vendor": suggestion.get("vendor") or "(no vendor)",
            "description": suggestion.get("description") or "",
            "charges": int(suggestion.get("charges") or 0),
            "months": int(suggestion.get("months") or 0),
            "cadence": suggestion.get("cadence") or "monthly",
            "average_amount": s2(monthly),
            "monthly_cost": s2(monthly),
            "annual_cost": s2(monthly * 12),
            "first": suggestion.get("first"),
            "last": suggestion.get("last"),
        })
    return rows


def subscription_totals(subscriptions: list[dict]) -> dict:
    monthly = sum((dec(row.get("monthly_cost")) for row in subscriptions), ZERO)
    return {
        "count": len(subscriptions),
        "monthly": s2(monthly),
        "annual": s2(q2(monthly) * 12),
    }


def _safe_price(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    amount = dec(value)
    if amount != amount:  # NaN guard
        return None
    if amount < 0:
        return None
    return q2(amount)


def subscription_compare(current_monthly: Decimal, alternative_monthly: Optional[Decimal]) -> dict:
    """Savings from switching, given a current and an estimated cost."""
    current = q2(dec(current_monthly))
    if alternative_monthly is None:
        return {
            "switching_worthwhile": None,
            "savings_monthly": None,
            "savings_annual": None,
        }
    alternative = q2(alternative_monthly)
    savings = q2(current - alternative)
    return {
        "switching_worthwhile": savings > 0,
        "savings_monthly": s2(savings),
        "savings_annual": s2(savings * 12),
    }


def subscription_parse_alternatives(payload: Any, current_monthly: Decimal) -> list[dict]:
    """Validate the model's alternative suggestions against one subscription.

    Prices without a traceable source are kept but flagged, so the UI can show
    a clear "unverified price" warning instead of presenting an invented figure
    as fact.
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("alternatives")
    if not isinstance(raw, list):
        return []

    alternatives: list[dict] = []
    for entry in raw[:MAX_ALTERNATIVES]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("product") or entry.get("name") or "").strip()
        if not name:
            continue
        monthly = _safe_price(entry.get("estimated_monthly_cost"))
        yearly = _safe_price(entry.get("estimated_yearly_cost"))
        if monthly is None and yearly is not None:
            monthly = q2(yearly / 12)
        if yearly is None and monthly is not None:
            yearly = q2(monthly * 12)

        source = str(entry.get("pricing_source") or "").strip()
        verified_date = str(entry.get("verified_date") or "").strip()
        flags: list[str] = []
        if monthly is None:
            flags.append("No price could be verified for this option.")
        elif monthly == ZERO and not source:
            flags.append("A free or zero cost was claimed without a source; treat it as unverified.")
        if not source:
            flags.append("No pricing source was provided; verify before switching.")

        comparison = subscription_compare(current_monthly, monthly)
        alternatives.append({
            "product": name[:120],
            "estimated_monthly_cost": s2(monthly) if monthly is not None else None,
            "estimated_yearly_cost": s2(yearly) if yearly is not None else None,
            "capabilities": _as_list(entry.get("capabilities")),
            "limitations": _as_list(entry.get("limitations")),
            "trade_offs": _as_list(entry.get("trade_offs") or entry.get("compromises")),
            "notes": str(entry.get("notes") or "").strip()[:400],
            "pricing_source": source[:300],
            "verified_date": verified_date[:40],
            "verified": bool(source),
            "flags": flags,
            **comparison,
        })
    def sort_key(row: dict) -> tuple:
        value = row.get("savings_monthly")
        savings = ZERO if value is None else dec(value)
        # Verified prices first, then the biggest saving, then unpriced options.
        return (0 if row.get("verified") else 1, value is None, -savings)

    alternatives.sort(key=sort_key)
    return alternatives


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip()[:200] for item in value if str(item).strip()][:12]
    return []


def subscription_build_facts(candidate: dict, usage: str, currency: str) -> str:
    """A compact prompt block. Only the single subscription is shared."""
    lines = [
        f"Currency: {currency}",
        f"Subscription: {candidate.get('vendor')} — {candidate.get('description') or 'recurring charge'}",
        f"Detected monthly cost: {candidate.get('monthly_cost')}",
        f"Detected yearly cost: {candidate.get('annual_cost')}",
        f"Observed charges: {candidate.get('charges')} over {candidate.get('months')} months",
        f"Stated usage: {usage.strip() if usage and usage.strip() else 'not provided'}",
    ]
    return "\n".join(lines)

# ===========================================================================
# Section 6: Document Intelligence
# ===========================================================================


import base64
import io
import re
import zipfile
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4
from xml.etree import ElementTree as ET


try:  # pypdf is the only new runtime dependency of this module.
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - only when pypdf is not installed
    PdfReader = None  # type: ignore[assignment]

try:  # PyMuPDF renders scanned pages and reads tables. Optional.
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover - only when PyMuPDF is not installed
    pymupdf = None  # type: ignore[assignment]

__all__ = [
    "DocumentError",
    "ExtractionError",
    "SUPPORTED_EXTENSIONS",
    "BLOCKED_EXTENSIONS",
    "MIME_TYPES",
    "MAX_FILE_BYTES",
    "MAX_BATCH_FILES",
    "DOCUMENT_FIELDS",
    "EXTRACTION_SYSTEM_PROMPT",
    "validate_upload",
    "sanitize_filename",
    "is_safe_stored_name",
    "store_file",
    "read_file",
    "delete_file",
    "extract_content",
    "build_extraction_prompt",
    "build_user_parts",
    "parse_extraction",
    "validate_financials",
    "find_duplicates",
    "to_import_payload",
    "run_extraction",
    "validate_batch",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DocumentError(RuntimeError):
    """User-facing problem: bad file, too large, unsupported type, unsafe name."""


class ExtractionError(RuntimeError):
    """The AI response could not be understood."""


# ---------------------------------------------------------------------------
# Types and limits
# ---------------------------------------------------------------------------

class _ExtensionMap(dict):
    """A plain ``dict`` of extension -> kind that tolerates a missing dot.

    ``SUPPORTED_EXTENSIONS[".pdf"]`` and ``SUPPORTED_EXTENSIONS["pdf"]`` both
    work, so a caller that uses ``Path(name).suffix`` and a caller that uses
    ``name.rsplit(".", 1)[-1]`` agree with each other.
    """

    @staticmethod
    def _normalise(key: Any) -> str:
        text = str(key).strip().lower()
        if text and not text.startswith("."):
            text = "." + text
        return text

    def __getitem__(self, key: Any) -> str:
        return dict.__getitem__(self, self._normalise(key))

    def __contains__(self, key: Any) -> bool:
        return dict.__contains__(self, self._normalise(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return dict.get(self, self._normalise(key), default)


#: Extension -> kind. The kinds are "pdf", "csv", "xlsx", "xls", "docx",
#: "doc", "txt", and "image".
SUPPORTED_EXTENSIONS: dict[str, str] = _ExtensionMap({
    ".pdf": "pdf",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".docx": "docx",
    ".doc": "doc",
    ".txt": "txt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
})

#: Executable content is refused by extension, whatever it claims to be.
BLOCKED_EXTENSIONS = frozenset({
    ".exe", ".sh", ".bat", ".cmd", ".com", ".msi", ".dll", ".scr", ".ps1", ".jar", ".py",
})

#: Extension -> MIME type for the formats Gemini reads as inline images.
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

MAX_FILE_BYTES = 0  # 0 = no per-file size limit
MAX_BATCH_FILES = 0  # 0 = no batch size limit

#: Longest text sent to the model in one prompt.
MAX_TEXT_CHARS = 40_000

#: Scanned PDF pages are rendered for the vision model. A hard cap keeps a very
#: long scan from turning into an enormous request.
MAX_RENDERED_PAGES = 8
RENDER_DPI = 150

#: The fields asked of Gemini for each document type.
DOCUMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "invoice": (
        "invoice_number", "supplier", "customer", "issue_date", "due_date",
        "currency", "subtotal", "tax", "total", "payment_status",
    ),
    "receipt": (
        "merchant", "transaction_date", "currency", "subtotal", "tax", "total",
        "payment_method",
    ),
    "bank_statement": (
        "account_holder", "account_identifier_masked", "period_start",
        "period_end", "currency", "opening_balance", "closing_balance",
    ),
    "expense": (
        "vendor", "description", "date", "category", "amount", "tax", "currency",
    ),
    "general": (
        "title", "organisation", "dates", "key_figures", "payment_terms",
        "obligations",
    ),
}

MONEY_FIELD_NAMES = frozenset({
    "subtotal", "tax", "total", "amount", "opening_balance", "closing_balance",
})
RATE_FIELD_NAMES = frozenset({"tax_rate"})
DATE_FIELD_NAMES = frozenset({
    "issue_date", "due_date", "transaction_date", "date", "period_start",
    "period_end",
})
CURRENCY_FIELD_NAMES = frozenset({"currency"})

STORED_NAME_RE = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{1,10}$")
LEGACY_OFFICE_ERROR = (
    "Legacy .xls/.doc files cannot be read. Save the file as .xlsx, .docx, or "
    "PDF and upload it again."
)

XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _extension_of(name: str) -> str:
    """Return the lower-case extension of a file name, including the dot."""
    base = str(name or "").strip().lower().replace("\\", "/").split("/")[-1]
    index = base.rfind(".")
    if index <= 0:
        return ""
    return base[index:]


def _strict_decimal(value: Any) -> Optional[Decimal]:
    """Parse a number without the lenient fallback of `core.dec`."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    text = str(value).strip().replace(",", "").replace(" ", "").replace("\u00a0", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _as_money(value: Any) -> Optional[Decimal]:
    """Read a money value as a two-decimal Decimal, or None."""
    number = _strict_decimal(value)
    if number is None:
        return None
    return q2(number)


def _as_rate(value: Any) -> Optional[Decimal]:
    """Read a rate: 0.2 is 20%, and 20 is also read as 20%."""
    number = _strict_decimal(value)
    if number is None:
        return None
    if number > 1:
        number = number / Decimal("100")
    return q4(number)


def _as_text(value: Any) -> Optional[str]:
    """Read a free-text value."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, Decimal):
        text = format(value, "f")
    elif isinstance(value, float):
        text = str(value)
    else:
        text = str(value).strip()
    return text or None


def _as_date(value: Any) -> Optional[str]:
    """Read a date as an ISO `YYYY-MM-DD` string, or None."""
    text = _as_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _as_currency(value: Any) -> Optional[str]:
    """Read a currency as a three-letter code, or None."""
    text = _as_text(value)
    if not text:
        return None
    code = text.strip().upper()
    if len(code) == 3 and code.isalpha():
        return code
    return None


def _clamp_confidence(value: Any) -> float:
    """Read a confidence as a float between 0 and 1."""
    number = _strict_decimal(value)
    if number is None:
        return 0.0
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return float(number)


def _normalise_key(value: str) -> str:
    return re.sub(r"[\s\-]+", "", str(value or "").strip().lower())


def _normalise_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """Reduce an uploaded name to a harmless base name.

    Directory components, traversal sequences, control characters, and unusual
    punctuation are removed. The result never contains a path separator and
    never points at a parent directory. An empty result becomes "document".
    """
    name = str(filename or "").strip()
    name = name.replace("\\", "/")
    name = name.split("/")[-1]
    name = name.replace("\x00", "")
    name = re.sub(r"\.{2,}", ".", name)
    name = re.sub(r"[^\w.\- ()&+,']", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip(". ")
    if len(name) > 120:
        name = name[:120].strip()
    return name or "document"


def is_safe_stored_name(name: str) -> bool:
    """True only for a generated name of the shape `<32 hex chars>.<ext>`.

    Path separators, traversal sequences, and anything else that is not a
    generated name are refused. This is the path traversal defence for
    `read_file` and `delete_file`.
    """
    if not isinstance(name, str) or not name:
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    if not STORED_NAME_RE.match(name):
        return False
    return "." + name.rsplit(".", 1)[1] in SUPPORTED_EXTENSIONS


def validate_upload(filename: str, size_bytes: int) -> dict:
    """Check an upload and describe it.

    Returns {"extension", "kind", "safe_name"}. Raises `DocumentError` when
    the name is unusable, the type is unsupported or executable, the file is
    empty, or the file is larger than `MAX_FILE_BYTES`.
    """
    raw = str(filename or "").strip()
    if not raw:
        raise DocumentError("The file has no name.")

    safe_name = sanitize_filename(raw)
    extension = _extension_of(safe_name)

    if extension in BLOCKED_EXTENSIONS:
        raise DocumentError(
            f"{extension} files are not accepted. Upload a PDF, CSV, XLSX, DOCX, "
            "TXT, PNG, JPG, or WEBP document."
        )

    kind = SUPPORTED_EXTENSIONS.get(extension)
    if kind is None:
        shown = extension or safe_name or raw
        raise DocumentError(
            f"Unsupported file type '{shown}'. Upload a PDF, CSV, XLSX, DOCX, TXT, "
            "PNG, JPG, or WEBP document."
        )

    try:
        size = int(size_bytes)
    except (TypeError, ValueError) as error:
        raise DocumentError("The size of the file could not be read.") from error

    if size <= 0:
        raise DocumentError("The file is empty.")
    if MAX_FILE_BYTES and size > MAX_FILE_BYTES:
        raise DocumentError(
            f"The file is {size} bytes. The limit is {MAX_FILE_BYTES} bytes (10 MB)."
        )

    return {"extension": extension, "kind": kind, "safe_name": safe_name}


def validate_batch(uploads: Iterable[tuple[str, int]]) -> list[dict]:
    """Validate a batch of uploads and refuse more than `MAX_BATCH_FILES`."""
    items = list(uploads or [])
    if MAX_BATCH_FILES and len(items) > MAX_BATCH_FILES:
        raise DocumentError(
            f"{len(items)} files were uploaded. At most {MAX_BATCH_FILES} files can be "
            "processed at a time."
        )
    return [validate_upload(name, size) for name, size in items]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _document_folder(data_dir: Path) -> Path:
    return Path(data_dir) / "documents"


def store_file(data_dir: Path, filename: str, content: bytes) -> dict:
    """Write the uploaded bytes under `<data_dir>/documents`.

    The stored name is `uuid4().hex` plus the validated extension, so the
    uploaded name never reaches the file system path. Returns
    {"stored_name", "size", "extension"}.
    """
    if content is None:
        content = b""
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise DocumentError("The content of the file could not be read.")
    content = bytes(content)

    info = validate_upload(filename, len(content))
    folder = _document_folder(data_dir)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DocumentError(f"The document folder could not be created: {error}") from error

    stored_name = f"{uuid4().hex}{info['extension']}"
    try:
        (folder / stored_name).write_bytes(content)
    except OSError as error:
        raise DocumentError(f"The document could not be saved: {error}") from error

    return {
        "stored_name": stored_name,
        "size": len(content),
        "extension": info["extension"],
    }


def read_file(data_dir: Path, stored_name: str) -> bytes:
    """Read a stored document. Raises `DocumentError` for an unsafe name."""
    if not is_safe_stored_name(stored_name):
        raise DocumentError("That is not a valid stored document name.")
    path = _document_folder(data_dir) / stored_name
    try:
        return path.read_bytes()
    except OSError as error:
        raise DocumentError(f"The document could not be read: {error}") from error


def delete_file(data_dir: Path, stored_name: str) -> None:
    """Delete a stored document. Raises `DocumentError` for an unsafe name.

    A document that is already gone is not an error.
    """
    if not is_safe_stored_name(stored_name):
        raise DocumentError("That is not a valid stored document name.")
    path = _document_folder(data_dir) / stored_name
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DocumentError(f"The document could not be deleted: {error}") from error


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "kind": "unsupported",
        "text": None,
        "mime": None,
        "images": [],
        "error": None,
        "notes": [],
    }


def _render_pdf_pages(content: bytes, page_numbers: list[int]) -> tuple[list[dict], list[str]]:
    """Render the requested 0-based PDF pages to PNG image parts.

    Returns (images, notes). An empty list means rendering was not possible, and
    the caller falls back to asking the user for an image instead.
    """
    notes: list[str] = []
    _effective_pymupdf = getattr(sys.modules.get("documents"), "pymupdf", pymupdf)
    if _effective_pymupdf is None:
        notes.append(
            "Scanned pages could not be rendered because PyMuPDF is not installed."
        )
        return [], notes
    images: list[dict] = []
    try:
        with _effective_pymupdf.open(stream=content, filetype="pdf") as document:
            for number in page_numbers[:MAX_RENDERED_PAGES]:
                try:
                    page = document.load_page(number)
                    pixmap = page.get_pixmap(dpi=RENDER_DPI)
                    data = pixmap.tobytes("png")
                except Exception:  # a page that will not render must not lose the rest
                    notes.append(f"Page {number + 1} could not be rendered.")
                    continue
                images.append({
                    "mime_type": "image/png",
                    "data": base64.b64encode(data).decode("ascii"),
                })
    except Exception as error:
        notes.append(f"The PDF pages could not be rendered: {error}")
        return [], notes
    return images, notes


def _pdf_tables_text(content: bytes) -> tuple[str, list[str]]:
    """Read tables from a PDF as plain rows, when PyMuPDF is available."""
    _effective_pymupdf = getattr(sys.modules.get("documents"), "pymupdf", pymupdf)
    if _effective_pymupdf is None:
        return "", []
    notes: list[str] = []
    lines: list[str] = []
    try:
        with _effective_pymupdf.open(stream=content, filetype="pdf") as document:
            for number, page in enumerate(document, start=1):
                try:
                    tables = page.find_tables()
                except Exception:
                    continue
                for table in tables:
                    try:
                        rows = table.extract()
                    except Exception:
                        continue
                    if not rows:
                        continue
                    lines.append(f"Table on page {number}:")
                    for row in rows:
                        cells = [str(cell) if cell is not None else "" for cell in row]
                        lines.append(" | ".join(cells))
    except Exception as error:
        notes.append(f"Tables could not be read from the PDF: {error}")
        return "", notes
    return "\n".join(lines).strip(), notes


def _extract_pdf(content: bytes, notes: list[str]) -> dict:
    """Read a PDF: selectable text plus rendered images for scanned pages.

    Text is read with pypdf. Pages that carry no extractable text are treated as
    scans and rendered to PNG so the vision model can read them.
    """
    result = _empty_result()
    result["notes"] = notes
    if PdfReader is None:
        result["error"] = "PDF reading is unavailable because pypdf is not installed."
        return result
    try:
        reader = PdfReader(io.BytesIO(content))
        pages: list[str] = []
        scanned_pages: list[int] = []
        for number, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:  # a broken page must not lose the other pages
                text = ""
                notes.append(f"Page {number + 1} could not be read.")
            if not text.strip():
                scanned_pages.append(number)
            pages.append(text)
    except Exception as error:
        result["error"] = (
            f"The PDF could not be read: {error}. It may be damaged or password protected."
        )
        return result

    text = "\n".join(pages).strip()

    tables_text, table_notes = _pdf_tables_text(content)
    notes.extend(table_notes)
    combined = text
    if tables_text:
        combined = (text + "\n\n" + tables_text).strip() if text else tables_text

    images: list[dict] = []
    if scanned_pages:
        notes.append(
            f"{len(scanned_pages)} page(s) have no selectable text and were treated as scans."
        )
        images, render_notes = _render_pdf_pages(content, scanned_pages)
        notes.extend(render_notes)

    if not combined and not images:
        result["error"] = (
            "This PDF has no readable text and its pages could not be rendered. "
            "Upload a PNG or JPG of the page and it will be read as an image."
        )
        return result

    result.update({
        "kind": "pdf",
        "text": combined or None,
        "images": images,
        "scanned_pages": len(scanned_pages),
        "pages": len(pages),
    })
    return result


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    strings: list[str] = []
    if "xl/sharedStrings.xml" not in archive.namelist():
        return strings
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except (ET.ParseError, KeyError, zipfile.BadZipFile):
        return strings
    for item in root.findall(f"{{{XLSX_NS}}}si"):
        strings.append("".join(node.text or "" for node in item.iter(f"{{{XLSX_NS}}}t")))
    return strings


def _xlsx_cell_text(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.get("t") or ""
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{XLSX_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{XLSX_NS}}}t"))
    value = cell.find(f"{{{XLSX_NS}}}v")
    text = (value.text or "") if value is not None else ""
    if cell_type == "s":
        try:
            index = int(text)
        except (TypeError, ValueError):
            return ""
        if 0 <= index < len(shared):
            return shared[index]
        return ""
    return text


def _extract_xlsx(content: bytes, notes: list[str]) -> dict:
    result = _empty_result()
    result["notes"] = notes
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError):
        result["error"] = "The XLSX file could not be opened; it is not a readable Excel file."
        return result

    rows: list[str] = []
    with archive:
        shared = _xlsx_shared_strings(archive)
        sheets = sorted(
            name for name in archive.namelist()
            if name.startswith("xl/worksheets/") and name.endswith(".xml")
        )
        if not sheets:
            result["error"] = "The XLSX file does not contain a worksheet."
            return result
        try:
            root = ET.fromstring(archive.read(sheets[0]))
        except (ET.ParseError, KeyError, zipfile.BadZipFile) as error:
            result["error"] = f"The XLSX worksheet could not be read: {error}."
            return result
        for row in root.iter(f"{{{XLSX_NS}}}row"):
            cells = [_xlsx_cell_text(cell, shared) for cell in row.findall(f"{{{XLSX_NS}}}c")]
            line = " | ".join(cells)
            if line.strip(" |"):
                rows.append(line)

    text = "\n".join(rows)
    if not text.strip():
        notes.append("No cell values were found in the first worksheet.")
        result["error"] = "The XLSX file has no readable rows."
        return result
    result.update({"kind": "text", "text": text})
    return result


def _extract_docx(content: bytes, notes: list[str]) -> dict:
    result = _empty_result()
    result["notes"] = notes
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError):
        result["error"] = "The DOCX file could not be opened; it is not a readable Word file."
        return result

    with archive:
        if "word/document.xml" not in archive.namelist():
            result["error"] = "The DOCX file does not contain a document body."
            return result
        try:
            root = ET.fromstring(archive.read("word/document.xml"))
        except (ET.ParseError, KeyError, zipfile.BadZipFile) as error:
            result["error"] = f"The DOCX document could not be read: {error}."
            return result

    parts = [(node.text or "").strip() for node in root.iter(f"{{{DOCX_NS}}}t")]
    text = " ".join(part for part in parts if part)
    if not text.strip():
        notes.append("The document body contains no text.")
        result["error"] = "The DOCX file has no readable text."
        return result
    result.update({"kind": "text", "text": text})
    return result


def extract_content(filename: str, content: bytes) -> dict:
    """Read the document content.

    Returns {"kind", "text", "mime", "error", "notes"} where kind is "text",
    "image", or "unsupported". Text formats are decoded to text, images are
    passed to Gemini as inline image data, and anything unsupported explains
    what to upload instead.
    """
    result = _empty_result()
    notes = result["notes"]
    if content is None:
        content = b""
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise DocumentError("The content of the file could not be read.")
    content = bytes(content)

    extension = _extension_of(sanitize_filename(filename))

    if extension in (".txt", ".csv"):
        text = content.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]
            notes.append("The file started with a byte order mark; it was removed.")
        result.update({"kind": "text", "text": text})
        return result

    if extension == ".pdf":
        return _extract_pdf(content, notes)

    if extension == ".xlsx":
        return _extract_xlsx(content, notes)

    if extension == ".docx":
        return _extract_docx(content, notes)

    if extension in (".xls", ".doc"):
        notes.append("Legacy binary Office formats are not supported.")
        result["error"] = LEGACY_OFFICE_ERROR
        return result

    if extension in MIME_TYPES:
        result.update({"kind": "image", "mime": MIME_TYPES[extension]})
        return result

    shown = extension or filename or "file"
    result["error"] = (
        f"'{shown}' cannot be read. Upload a PDF, CSV, XLSX, DOCX, TXT, PNG, JPG, or "
        "WEBP document."
    )
    return result


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You read financial documents for a small business finance app.
Return one JSON object and nothing else. No prose, no markdown fences, no commentary.

Rules you must follow:
- Report only what the document actually states. Never guess, infer, or invent a value.
- When a value is missing, ambiguous, or unreadable, use null and set "uncertain": true
  with a low confidence for that field. A null is always better than a guess.
- Dates are ISO YYYY-MM-DD strings or null.
- Currency is a three-letter code such as EUR, USD, or GBP, or null.
- Amounts are plain numbers in the currency of the document, without symbols or thousands
  separators.
- Every field listed for the document type must appear in "fields", even when its value
  is null.
- "line_items" holds one entry per billed line, with description, quantity, unit_price,
  amount, and confidence. Use an empty list when the document has no line items.
- "summary" is one short paragraph describing the document.
- "warnings" lists anything that is unusual, unreadable, or worth a human check."""

_FIELD_LINES = "\n".join(
    f"- {name}: {', '.join(fields)}" for name, fields in DOCUMENT_FIELDS.items()
)

_SCHEMA_BLOCK = """Return exactly this shape:
{
  "document_type": "invoice | receipt | bank_statement | expense | general",
  "confidence": 0.0,
  "summary": "one paragraph describing the document",
  "fields": {
    "<field name>": {"value": "string or null", "confidence": 0.0, "uncertain": false}
  },
  "line_items": [
    {"description": "string", "quantity": 1, "unit_price": 12.5, "amount": 12.5, "confidence": 0.9}
  ],
  "warnings": ["short strings"]
}"""


def build_extraction_prompt(text: str, mime: Optional[str], kind: str, currency_hint: str,
                            has_images: bool = False) -> str:
    """Build the user prompt that asks the model for the extraction JSON."""
    hint = str(currency_hint or "").strip() or "the currency shown on the document"
    body = str(text or "")
    truncated = False
    if len(body) > MAX_TEXT_CHARS:
        body = body[:MAX_TEXT_CHARS]
        truncated = True

    lines = [
        "Extract the structured contents of the document below.",
        "",
        _SCHEMA_BLOCK,
        "",
        "Fields by document type:",
        _FIELD_LINES,
        "",
        "Notes:",
        f"- Amounts are in {hint}. If the document states a different currency, report that.",
        "- quantity defaults to 1 when the document shows a single unit.",
        "- amount is quantity multiplied by unit_price.",
        "- Do not include a field that is not listed for the document type.",
        "",
    ]

    if kind == "image":
        lines.extend([
            "--- DOCUMENT IMAGE ---",
            f"The document is attached as an image ({mime or 'image'}). Read it directly.",
            "--- END DOCUMENT IMAGE ---",
        ])
    else:
        if body:
            if truncated:
                lines.append("(The text below was cut short because the document is very long.)")
            lines.extend([
                "--- DOCUMENT TEXT ---",
                body,
                "--- END DOCUMENT TEXT ---",
            ])
        if kind == "pdf" and has_images:
            lines.extend([
                "--- DOCUMENT PAGE IMAGES ---",
                "Pages with no selectable text are attached as images. Read the values from "
                "the images as well as the text.",
                "--- END DOCUMENT PAGE IMAGES ---",
            ])
        elif not body:
            lines.append("The document has no readable text. Use the attached pages if present.")
    return "\n".join(lines)


def build_user_parts(prompt: str, content: Optional[bytes] = None,
                     mime: Optional[str] = None,
                     images: Optional[list[dict]] = None) -> list[dict]:
    """Build the content parts for a document.

    The text part carries the prompt; an image document, or a scanned PDF page,
    adds an inline_data part with the base64 encoded bytes. The provider layer
    translates these to the OpenAI-compatible shape for NVIDIA NIM.
    """
    parts: list[dict] = [{"text": str(prompt or "")}]
    if mime and content and str(mime).startswith("image/"):
        parts.append({
            "inline_data": {
                "mime_type": str(mime),
                "data": base64.b64encode(bytes(content)).decode("ascii"),
            },
        })
    for image in images or []:
        data = str(image.get("data") or "")
        if not data:
            continue
        parts.append({
            "inline_data": {
                "mime_type": str(image.get("mime_type") or "image/png"),
                "data": data,
            },
        })
    return parts


# ---------------------------------------------------------------------------
# Parsing the AI response
# ---------------------------------------------------------------------------

def _normalise_field_value(key: str, value: Any) -> tuple[Any, Optional[str]]:
    """Return the parsed value of a field and an optional problem message."""
    if value is None:
        return None, None
    if isinstance(value, str) and not value.strip():
        return None, None

    if key in MONEY_FIELD_NAMES:
        money = _as_money(value)
        if money is None:
            return None, f"The value for '{key}' was not a number and was left empty."
        return money, None

    if key in RATE_FIELD_NAMES:
        rate = _as_rate(value)
        if rate is None:
            return None, f"The value for '{key}' was not a number and was left empty."
        return rate, None

    if key in DATE_FIELD_NAMES:
        iso = _as_date(value)
        if iso is None:
            return None, f"The value for '{key}' was not an ISO date and was left empty."
        return iso, None

    if key in CURRENCY_FIELD_NAMES:
        code = _as_currency(value)
        if code is None:
            return None, f"The value for '{key}' was not a currency code and was left empty."
        return code, None

    return _as_text(value), None


def parse_extraction(payload: dict) -> dict:
    """Validate the AI JSON against the extraction schema.

    Returns the normalised document:
    {"document_type", "confidence", "summary", "fields", "line_items", "warnings"}.
    Raises `ExtractionError` when the structure cannot be used. A value the
    model did not state stays None with "uncertain" set to True.
    """
    if not isinstance(payload, dict):
        raise ExtractionError("The AI response was not a JSON object.")

    raw_type = payload.get("document_type")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ExtractionError("The AI response did not name a document type.")
    document_type = raw_type.strip().lower().replace(" ", "_").replace("-", "_")
    if document_type not in DOCUMENT_FIELDS:
        raise ExtractionError(
            f"The AI returned an unsupported document type: {raw_type.strip()!r}."
        )

    problems: list[str] = []

    raw_fields = payload.get("fields")
    if raw_fields is None:
        raw_fields = {}
    if not isinstance(raw_fields, dict):
        raise ExtractionError("The AI response contained a malformed 'fields' object.")

    fields: dict[str, dict] = {}
    for name, entry in raw_fields.items():
        key = str(name).strip()
        if not key:
            continue
        if isinstance(entry, dict):
            raw_value = entry.get("value")
            confidence = _clamp_confidence(entry.get("confidence"))
            uncertain = bool(entry.get("uncertain"))
        else:
            # Models sometimes return a bare value per field instead of the
            # nested {value, confidence, uncertain} object. Accept it rather
            # than rejecting the whole document.
            raw_value = entry
            confidence = 0.0
            uncertain = False
        value, problem = _normalise_field_value(key, raw_value)
        if problem:
            problems.append(problem)
            uncertain = True
            confidence = 0.0
        if value is None:
            uncertain = True
        fields[key] = {"value": value, "confidence": confidence, "uncertain": uncertain}

    # A field the document type expects is always present, even when empty.
    for name in DOCUMENT_FIELDS[document_type]:
        if name not in fields:
            fields[name] = {"value": None, "confidence": 0.0, "uncertain": True}

    ordered: dict[str, dict] = {name: fields[name] for name in DOCUMENT_FIELDS[document_type]}
    for name, entry in fields.items():
        if name not in ordered:
            ordered[name] = entry

    raw_lines = payload.get("line_items")
    if raw_lines is None:
        raw_lines = []
    if not isinstance(raw_lines, list):
        raise ExtractionError("The AI response contained a malformed 'line_items' list.")

    line_items: list[dict] = []
    for index, item in enumerate(raw_lines, start=1):
        if not isinstance(item, dict):
            problems.append(f"Line item {index} was not an object and was ignored.")
            continue

        description = _as_text(item.get("description")) or ""

        quantity_raw = item.get("quantity")
        quantity = _strict_decimal(quantity_raw) if quantity_raw is not None else None
        if quantity_raw is not None and quantity is None:
            problems.append(f"Line item {index} had a quantity that is not a number.")
        if quantity is None:
            quantity = Decimal("1")

        unit_raw = item.get("unit_price")
        unit_price = _strict_decimal(unit_raw) if unit_raw not in (None, "") else None
        if unit_raw not in (None, "") and unit_price is None:
            problems.append(f"Line item {index} had a unit price that is not a number.")

        amount_raw = item.get("amount")
        amount = _strict_decimal(amount_raw) if amount_raw not in (None, "") else None
        if amount_raw not in (None, "") and amount is None:
            problems.append(f"Line item {index} had an amount that is not a number.")

        line_items.append({
            "description": description,
            "quantity": q4(quantity),
            "unit_price": q2(unit_price) if unit_price is not None else None,
            "amount": q2(amount) if amount is not None else None,
            "confidence": _clamp_confidence(item.get("confidence")),
        })

    raw_warnings = payload.get("warnings")
    warnings: list[str] = []
    if isinstance(raw_warnings, list):
        for entry in raw_warnings:
            text = _as_text(entry)
            if text:
                warnings.append(text)
    elif raw_warnings is not None:
        warnings.append("The AI response contained warnings in an unexpected format.")

    summary_raw = payload.get("summary")
    summary = _as_text(summary_raw) or ""

    return {
        "document_type": document_type,
        "confidence": _clamp_confidence(payload.get("confidence")),
        "summary": summary,
        "fields": ordered,
        "line_items": line_items,
        "warnings": warnings + problems,
    }


# ---------------------------------------------------------------------------
# Deterministic financial checks
# ---------------------------------------------------------------------------

def _field_value(document: dict, name: str) -> Any:
    fields = document.get("fields") or {}
    entry = fields.get(name)
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def validate_financials(document: dict, default_tax_rate: Decimal) -> dict:
    """Recompute the totals from the line items and report the differences.

    Returns {"flags": [{"code", "severity", "message"}], "calculated": {...}}.
    The document is never changed: the flags describe the difference between
    the extracted figures and the calculated ones.
    """
    if not isinstance(document, dict):
        raise ExtractionError("The extraction is not usable.")

    flags: list[dict] = []
    lines = document.get("line_items") or []

    extracted_subtotal = _as_money(_field_value(document, "subtotal"))
    extracted_tax = _as_money(_field_value(document, "tax"))
    extracted_total = _as_money(_field_value(document, "total"))
    tax_rate = _as_rate(_field_value(document, "tax_rate"))

    calculated: dict[str, Any] = {
        "subtotal": None,
        "tax": None,
        "total": None,
        "line_count": len(lines),
        "lines_skipped": 0,
    }

    line_amounts: list[Decimal] = []
    for index, line in enumerate(lines, start=1):
        if not isinstance(line, dict):
            continue
        quantity = _strict_decimal(line.get("quantity"))
        unit_price = _strict_decimal(line.get("unit_price"))
        if quantity is None or unit_price is None:
            calculated["lines_skipped"] += 1
            continue
        computed = q2(quantity * unit_price)
        line_amounts.append(computed)

        extracted_amount = _as_money(line.get("amount"))
        if extracted_amount is not None and extracted_amount != computed:
            description = _as_text(line.get("description")) or f"#{index}"
            flags.append({
                "code": "line_mismatch",
                "severity": "warning",
                "message": (
                    f"Line {index} '{description}' amount {s2(extracted_amount)} does not "
                    f"match quantity {s4(quantity)} x unit price {s2(unit_price)} = "
                    f"{s2(computed)}."
                ),
            })

    calculated_subtotal = sum(line_amounts, ZERO) if line_amounts else None
    if calculated_subtotal is not None:
        calculated["subtotal"] = s2(calculated_subtotal)
        if extracted_subtotal is not None and extracted_subtotal != calculated_subtotal:
            flags.append({
                "code": "subtotal_mismatch",
                "severity": "warning",
                "message": (
                    f"Extracted subtotal {s2(extracted_subtotal)} does not match the "
                    f"calculated subtotal {s2(calculated_subtotal)} from the line items."
                ),
            })

    # The tax used to rebuild the total: the rate when the document states one,
    # otherwise the tax the document states.
    if calculated_subtotal is not None and tax_rate is not None:
        tax_for_total = q2(calculated_subtotal * tax_rate)
    else:
        tax_for_total = extracted_tax

    # The tax implied by the document, used only to report a difference.
    if tax_rate is not None:
        rate_base = calculated_subtotal if calculated_subtotal is not None else extracted_subtotal
        if rate_base is not None:
            expected_tax = q2(rate_base * tax_rate)
            tax_source = "the extracted tax rate"
            compare_tax = True
        else:
            expected_tax = None
            tax_source = ""
            compare_tax = False
    elif calculated_subtotal is not None and extracted_total is not None and extracted_subtotal is not None:
        expected_tax = q2(extracted_total - extracted_subtotal)
        tax_source = "the extracted total and subtotal"
        compare_tax = True
    elif calculated_subtotal is not None:
        fallback_rate = _as_rate(default_tax_rate)
        expected_tax = q2(calculated_subtotal * (fallback_rate if fallback_rate is not None else ZERO))
        tax_source = "the default tax rate"
        compare_tax = True
    else:
        expected_tax = None
        tax_source = ""
        compare_tax = False

    if expected_tax is not None:
        calculated["tax"] = s2(tax_for_total if tax_for_total is not None else expected_tax)
        if compare_tax and extracted_tax is not None and extracted_tax != expected_tax:
            flags.append({
                "code": "tax_mismatch",
                "severity": "warning",
                "message": (
                    f"Extracted tax {s2(extracted_tax)} does not match the calculated tax "
                    f"{s2(expected_tax)} from {tax_source}."
                ),
            })

    if calculated_subtotal is not None and tax_for_total is not None:
        calculated_total = calculated_subtotal + tax_for_total
        total_source = "from the line items"
    elif extracted_subtotal is not None and extracted_tax is not None:
        calculated_total = extracted_subtotal + extracted_tax
        total_source = "from the extracted subtotal and tax"
    else:
        calculated_total = None
        total_source = ""

    if calculated_total is not None:
        calculated["total"] = s2(calculated_total)
        if extracted_total is not None and extracted_total != calculated_total:
            flags.append({
                "code": "total_mismatch",
                "severity": "warning",
                "message": (
                    f"Extracted total {s2(extracted_total)} does not match the calculated "
                    f"total {s2(calculated_total)} {total_source}."
                ),
            })

    return {"flags": flags, "calculated": calculated}


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _invoice_number_of(record: dict) -> str:
    for key in ("invoice_number", "invoice_no", "number"):
        value = _as_text(record.get(key))
        if value:
            return value
    return ""


def _invoice_party_names(record: dict) -> set[str]:
    names = set()
    for key in ("supplier", "vendor", "client_name"):
        value = _normalise_name(record.get(key))
        if value:
            names.add(value)
    return names


def _invoice_total_of(record: dict) -> Optional[Decimal]:
    totals = record.get("totals")
    if isinstance(totals, dict):
        money = _as_money(totals.get("total"))
        if money is not None:
            return money
    return _as_money(record.get("total"))


def _invoice_reference(record: dict) -> str:
    identifier = record.get("id")
    try:
        return f"INV-{int(identifier):04d}"
    except (TypeError, ValueError):
        return _invoice_number_of(record) or str(identifier or "")


def find_duplicates(document: dict, invoices: list[dict], expenses: list[dict]) -> list[dict]:
    """Warn about records that already look like this document.

    Returns a list of {"kind", "id", "reference", "reason"}. Nothing is
    changed and nothing is rejected: this only tells the user what to check.
    """
    if not isinstance(document, dict):
        return []

    values: dict[str, Any] = {}
    for name, entry in (document.get("fields") or {}).items():
        values[name] = entry.get("value") if isinstance(entry, dict) else entry

    duplicates: list[dict] = []

    # Check both ledgers, not only the one this document was read as. The same
    # bill can be read as a receipt on one upload and an invoice on the next, so
    # a per-type check would file it twice.
    check_invoices = True
    check_expenses = True

    if check_invoices:
        wanted_number = _normalise_key(values.get("invoice_number"))
        wanted_names = {
            _normalise_name(values.get(key))
            for key in ("supplier", "customer")
        }
        wanted_names.discard("")
        wanted_date = _as_date(values.get("issue_date"))
        wanted_total = _as_money(values.get("total"))

        for invoice in invoices or []:
            if not isinstance(invoice, dict):
                continue
            identifier = invoice.get("id")
            if wanted_number:
                existing_number = _normalise_key(_invoice_number_of(invoice))
                if existing_number and existing_number == wanted_number:
                    duplicates.append({
                        "kind": "invoice",
                        "id": identifier,
                        "reference": _invoice_reference(invoice),
                        "reason": (
                            f"Invoice number {values.get('invoice_number')} is already used by "
                            f"invoice {_invoice_reference(invoice)}."
                        ),
                    })
                    continue
            if wanted_names and wanted_date and wanted_total is not None:
                existing_names = _invoice_party_names(invoice)
                existing_date = _as_date(invoice.get("issue_date"))
                existing_total = _invoice_total_of(invoice)
                if (
                    existing_names & wanted_names
                    and existing_date == wanted_date
                    and existing_total is not None
                    and abs(existing_total - wanted_total) <= Decimal("0.01")
                ):
                    duplicates.append({
                        "kind": "invoice",
                        "id": identifier,
                        "reference": _invoice_reference(invoice),
                        "reason": (
                            "The same party, issue date, and total already appear on invoice "
                            f"{_invoice_reference(invoice)}."
                        ),
                    })

    if check_expenses:
        wanted_description = _normalise_name(
            values.get("description") or values.get("merchant") or values.get("vendor")
        )
        wanted_date = _as_date(values.get("date") or values.get("transaction_date"))
        wanted_amount = _as_money(values.get("amount"))
        if wanted_amount is None:
            wanted_amount = _as_money(values.get("total"))

        for expense in expenses or []:
            if not isinstance(expense, dict):
                continue
            existing_description = _normalise_name(expense.get("description"))
            existing_date = _as_date(expense.get("date_incurred"))
            existing_amount = _as_money(expense.get("amount"))
            if (
                wanted_description
                and existing_description == wanted_description
                and wanted_date
                and existing_date == wanted_date
                and wanted_amount is not None
                and existing_amount is not None
                and abs(existing_amount - wanted_amount) <= Decimal("0.01")
            ):
                duplicates.append({
                    "kind": "expense",
                    "id": expense.get("id"),
                    "reference": _as_text(expense.get("description")) or str(expense.get("id") or ""),
                    "reason": (
                        "The same description, date, and amount already appear on expense "
                        f"#{expense.get('id')}."
                    ),
                })

    return duplicates


# ---------------------------------------------------------------------------
# Import payload
# ---------------------------------------------------------------------------

def _correction_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _document_values(document: dict) -> dict:
    values: dict[str, Any] = {}
    for name, entry in (document.get("fields") or {}).items():
        values[str(name)] = entry.get("value") if isinstance(entry, dict) else entry
    return values


def _invoice_payload(values: dict, document: dict, corrected: frozenset[str]) -> dict:
    customer = _as_text(values.get("customer"))
    supplier = _as_text(values.get("supplier"))
    client_name = customer or supplier or ""

    line_items: list[dict] = []
    for line in document.get("line_items") or []:
        if not isinstance(line, dict):
            continue
        description = _as_text(line.get("description")) or ""
        quantity = _strict_decimal(line.get("quantity"))
        if quantity is None or quantity == 0:
            quantity = Decimal("1")
        unit_price = _strict_decimal(line.get("unit_price"))
        if unit_price is None:
            unit_price = ZERO
        line_items.append({
            "description": description,
            "quantity": s4(quantity),
            "unit_price": s2(unit_price),
        })

    tax_rate = _as_rate(values.get("tax_rate"))
    extracted_subtotal = _as_money(values.get("subtotal"))
    extracted_tax = _as_money(values.get("tax"))
    extracted_total = _as_money(values.get("total"))
    if tax_rate is None and extracted_subtotal and extracted_tax is not None:
        tax_rate = q4(extracted_tax / extracted_subtotal)
    if tax_rate is None:
        tax_rate = ZERO

    if line_items:
        calculated_subtotal = ZERO
        for line in line_items:
            calculated_subtotal += q2(dec(line["quantity"]) * dec(line["unit_price"]))
        calculated_tax = q2(calculated_subtotal * tax_rate)
    else:
        calculated_subtotal = extracted_subtotal if extracted_subtotal is not None else ZERO
        calculated_tax = extracted_tax if extracted_tax is not None else ZERO

    # A figure the user corrected wins; the rest stay as calculated.
    subtotal = extracted_subtotal if "subtotal" in corrected else calculated_subtotal
    tax_amount = extracted_tax if "tax" in corrected else calculated_tax
    if "total" in corrected:
        total = extracted_total if extracted_total is not None else subtotal + tax_amount
    elif not line_items and extracted_total is not None:
        total = extracted_total
    else:
        total = subtotal + tax_amount

    payment_status = (_as_text(values.get("payment_status")) or "").lower()
    status = {
        "paid": "paid",
        "settled": "paid",
        "sent": "sent",
        "issued": "sent",
        "overdue": "overdue",
    }.get(payment_status, "draft")
    if status not in INVOICE_STATUS:
        status = "draft"

    summary = _as_text(document.get("summary")) or ""
    invoice_number = _as_text(values.get("invoice_number"))

    payload = {
        "client_name": client_name,
        "client_email": "",
        "issue_date": _as_date(values.get("issue_date")),
        "due_date": _as_date(values.get("due_date")),
        "line_items": line_items,
        "tax_rate": s4(tax_rate),
        "totals": {
            "subtotal": s2(subtotal),
            "discount_pct": "0.00",
            "tax_rate": s4(tax_rate),
            "tax_amount": s2(tax_amount),
            "total": s2(total),
        },
        "notes": summary[:5000],
        "status": status,
    }
    if invoice_number:
        # The ledger invoice record has no number field of its own; keeping it
        # here lets a later upload find this invoice as a duplicate.
        payload["invoice_number"] = invoice_number
    return payload


def _expense_payload(values: dict, document: dict) -> dict:
    description = (
        _as_text(values.get("description"))
        or _as_text(values.get("merchant"))
        or _as_text(values.get("vendor"))
        or ""
    )
    amount = _as_money(values.get("amount"))
    if amount is None:
        amount = _as_money(values.get("total"))
    if amount is None:
        amount = ZERO

    category = (_as_text(values.get("category")) or "").lower()
    if category not in EXPENSE_CATEGORIES:
        category = "other"

    vendor = _as_text(values.get("vendor")) or _as_text(values.get("merchant")) or ""
    summary = _as_text(document.get("summary")) or ""

    return {
        "description": description,
        "amount": s2(amount),
        "category": category,
        "date_incurred": _as_date(values.get("date")) or _as_date(values.get("transaction_date")),
        "vendor": vendor,
        "notes": summary[:5000],
        "reimbursable": False,
        "reimbursed": False,
    }


def to_import_payload(document: dict, corrections: Optional[dict] = None) -> dict:
    """Build the ledger payload for a confirmed import.

    Returns {"kind": "invoice" | "expense", "payload": {...}} where the payload
    matches the record shape used by the ledger (an invoice record or an
    expense record). Corrected values replace the extracted ones. The stored
    extraction is never modified.
    """
    if not isinstance(document, dict):
        raise ExtractionError("The extraction is not usable.")

    values = _document_values(document)
    corrected: set[str] = set()
    for name, value in (corrections or {}).items():
        key = str(name)
        values[key] = _correction_value(value)
        corrected.add(key)

    document_type = str(document.get("document_type") or "").strip().lower()
    if document_type == "invoice":
        return {"kind": "invoice", "payload": _invoice_payload(values, document, frozenset(corrected))}
    if document_type in ("expense", "receipt"):
        return {"kind": "expense", "payload": _expense_payload(values, document)}
    raise ExtractionError(
        f"A {document_type or 'general'} document cannot be imported as an invoice or an "
        "expense."
    )


# ---------------------------------------------------------------------------
# The whole extraction, with an injected Gemini callable
# ---------------------------------------------------------------------------

def run_extraction(filename: str, content: bytes, gemini_json,
                   currency_hint: str = "EUR",
                   default_tax_rate: Decimal = Decimal("0.20")) -> dict:
    """Read a document, ask the AI, and check the result.

    `gemini_json(system_prompt, user_parts) -> dict` is the injected AI call.
    The web layer passes a callable that routes to NVIDIA NIM (Nemotron vision)
    with a Gemini fallback. Returns {"content", "document", "validation"}.
    """
    if not callable(gemini_json):
        raise ExtractionError("The AI service is not available.")

    extracted = extract_content(filename, content)
    if extracted["kind"] == "unsupported":
        raise ExtractionError(extracted["error"] or "This document could not be read.")

    prompt = build_extraction_prompt(
        extracted.get("text") or "",
        extracted.get("mime"),
        extracted.get("kind") or "text",
        currency_hint,
        has_images=bool(extracted.get("images")),
    )
    parts = build_user_parts(prompt, content, extracted.get("mime"), extracted.get("images"))
    payload = gemini_json(EXTRACTION_SYSTEM_PROMPT, parts)
    document = parse_extraction(payload)
    validation = validate_financials(document, default_tax_rate)
    return {"content": extracted, "document": document, "validation": validation}


def local_extraction(filename: str, content: bytes) -> dict:
    """Deterministic, model-free summary used when the AI reader is unavailable.

    It describes what the file actually contains and returns a `general`
    document. A general document cannot be imported as an invoice or an expense,
    so a failed or unavailable read never files anything into the ledger.
    """
    extracted = extract_content(filename, content)
    if extracted["kind"] == "unsupported":
        raise ExtractionError(extracted["error"] or "This document could not be read.")

    text = str(extracted.get("text") or "").strip()
    extension = _extension_of(sanitize_filename(filename))
    warnings = [str(note) for note in (extracted.get("notes") or []) if note]
    title = sanitize_filename(filename) or "Document"
    key_figures: list[str] = []
    summary = ""

    if extension == ".csv" and text:
        try:
            headers, rows, errors = _read_csv(text)
        except ValueError as error:
            warnings.append(str(error))
            headers, rows, errors = [], [], []
        warnings.extend(errors)
        if headers:
            key_figures.append(f"{len(rows)} data row(s)")
            key_figures.append(f"{len(headers)} column(s)")
            summary = (
                f"Table with {len(rows)} data row(s) and {len(headers)} column(s): "
                + ", ".join(headers[:12])
                + "."
            )
        elif text:
            summary = " ".join(text.split())[:280]
    elif text:
        summary = " ".join(text.split())[:280]
        key_figures.append(f"{len(text.split())} word(s)")
    else:
        summary = "No readable text was found in this file."
        if extracted.get("mime"):
            warnings.append(
                "The file is an image; connect a vision model to read its contents."
            )

    fields = {
        name: {"value": None, "confidence": 0.0, "uncertain": True}
        for name in DOCUMENT_FIELDS["general"]
    }
    fields["title"] = {"value": title, "confidence": 1.0, "uncertain": False}
    if key_figures:
        fields["key_figures"] = {
            "value": "; ".join(key_figures),
            "confidence": 1.0,
            "uncertain": False,
        }
    return {
        "document_type": "general",
        "confidence": 0.0,
        "summary": summary or "The file was stored but its contents could not be summarised.",
        "fields": fields,
        "line_items": [],
        "warnings": warnings,
    }

# ===========================================================================
# Section 7: AI-written Financial Summaries
# ===========================================================================


import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional



SUMMARY_SCOPES = ("spendwise", "overview")
SUMMARY_GROUP_KEYS = ("highlights", "observations", "cautions", "opportunities")
MAX_FACTS_TEXT = 2500


class SummaryError(RuntimeError):
    """The AI reply could not be used."""


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _period_facts(period: dict) -> dict:
    return {
        "key": period.get("key"),
        "label": period.get("label"),
        "start": period.get("start"),
        "end": period.get("end"),
        "days": period.get("days"),
        "previous_start": (period.get("previous") or {}).get("start"),
        "previous_end": (period.get("previous") or {}).get("end"),
    }


def _share(part, whole) -> str:
    if dec(whole) <= 0:
        return "0.00"
    return s2(q2(dec(part) * Decimal("100") / dec(whole)))


def build_spendwise_facts(expenses: list[dict], invoices: list[dict],
                          period: dict, budgets: list[dict]) -> dict:
    """Deterministic spending facts for the SpendWise scope."""
    expenses = expenses or []
    invoices = invoices or []
    budgets = budgets or []

    stats = spending_stats(expenses, period)
    categories = category_breakdown(expenses, period)
    vendors = vendor_breakdown(expenses, period, limit=12)
    anchor = parse_iso_date(period.get("end")) or date.today()
    budget_rows = budget_status(budgets, expenses, anchor)
    budget_summary = budget_totals(budgets, expenses, anchor)
    start = parse_iso_date(period["start"])
    end = parse_iso_date(period["end"])
    current = filter_expenses(expenses, start, end)
    recurring = recurring_suggestions(expenses)
    unusual = unusual_expenses(current)
    cash = cash_flow(invoices, expenses, period)
    insights = build_insights(expenses, invoices, period, budgets, [])

    top_category = None
    if categories:
        row = categories[0]
        top_category = {"name": row["category"], "total": row["total"], "share": row["share"]}
    top_vendor = None
    if vendors:
        row = vendors[0]
        top_vendor = {"name": row["vendor"], "total": row["total"], "share": row["share"]}

    return {
        "scope": "spendwise",
        "period": _period_facts(period),
        "currency": None,
        "total": stats["total"],
        "previous_total": stats["previous_total"],
        "change_amount": stats["change"]["amount"],
        "change_percent": stats["change"]["percent"],
        "transaction_count": stats["count"],
        "average_transaction": stats["average_transaction"],
        "average_monthly": stats["average_monthly"],
        "daily_average": stats["daily_average"],
        "top_category": top_category,
        "top_vendor": top_vendor,
        "categories": [
            {
                "name": row["category"],
                "total": row["total"],
                "share": row["share"],
                "count": row["count"],
                "previous_total": row["previous_total"],
                "change_percent": row["change"]["percent"],
            }
            for row in categories
        ],
        "vendors": [
            {
                "name": row["vendor"],
                "total": row["total"],
                "share": row["share"],
                "count": row["count"],
            }
            for row in vendors
        ],
        "budgets": [
            {
                "name": row["name"],
                "category": row["category"],
                "amount": row["amount"],
                "spent": row["spent"],
                "remaining": row["remaining"],
                "percent": row["percent"],
                "state": row["state"],
            }
            for row in budget_rows
        ],
        "budget_totals": {
            "count": budget_summary["count"],
            "budget": budget_summary["budget"],
            "spent": budget_summary["spent"],
            "remaining": budget_summary["remaining"],
            "percent": budget_summary["percent"],
            "combined": budget_summary["combined"],
            "over_budget": budget_summary["over_budget"],
        },
        "recurring": [
            {
                "description": row["description"],
                "vendor": row["vendor"],
                "charges": row["charges"],
                "average_amount": row["average_amount"],
                "cadence": row["cadence"],
            }
            for row in recurring[:6]
        ],
        "unusual": [
            {
                "description": row["description"],
                "category": row["category"],
                "amount": row["amount"],
                "median": row["median"],
                "ratio": row["ratio"],
            }
            for row in unusual[:6]
        ],
        "cash": {
            "incoming": cash["incoming"],
            "outgoing": cash["outgoing"],
            "net": cash["net"],
        },
        "insights": [
            {"title": row["title"], "detail": row["detail"]} for row in insights
        ],
    }


def build_overview_facts(invoices: list[dict], expenses: list[dict],
                         spendwise_facts: dict, settings: dict) -> dict:
    """Deterministic facts for the Overview scope: accounting plus spending."""
    invoices = invoices or []
    expenses = expenses or []
    settings = settings or {}

    issued = [
        invoice for invoice in invoices
        if invoice.get("status") in ("sent", "overdue", "paid")
    ]
    invoiced_gross = sum((_invoice_total(invoice) for invoice in issued), ZERO)
    net_revenue = sum((_invoice_subtotal(invoice) for invoice in issued), ZERO)
    tax_invoiced = sum((_invoice_tax(invoice) for invoice in issued), ZERO)
    collected = sum(
        (_invoice_total(invoice) for invoice in invoices if invoice.get("status") == "paid"),
        ZERO,
    )
    outstanding = sum(
        (_invoice_total(invoice) for invoice in invoices
         if invoice.get("status") in ("sent", "overdue")),
        ZERO,
    )
    expenses_total = sum((_expense_amount(expense) for expense in expenses), ZERO)
    net_profit = net_revenue - expenses_total
    overdue = [invoice for invoice in invoices if invoice.get("status") == "overdue"]

    recent = sorted(invoices, key=lambda item: item.get("id", 0), reverse=True)[:12]
    return {
        "scope": "overview",
        "currency": settings.get("currency_code") or "",
        "invoiced_gross": s2(invoiced_gross),
        "net_revenue": s2(net_revenue),
        "tax_invoiced": s2(tax_invoiced),
        "collected": s2(collected),
        "outstanding": s2(outstanding),
        "overdue_count": len(overdue),
        "expenses_total": s2(expenses_total),
        "net_profit": s2(net_profit),
        "reporting_basis": "accrual",
        "spending": spendwise_facts,
        "recent_invoices": [
            {
                "reference": _invoice_ref(invoice.get("id")),
                "client": invoice.get("client_name") or "",
                "total": s2(_invoice_total(invoice)),
                "status": invoice.get("status") or "",
            }
            for invoice in recent
        ],
    }


def _invoice_total(invoice: dict) -> Decimal:
    return q2(dec((invoice.get("totals") or {}).get("total")))


def _invoice_subtotal(invoice: dict) -> Decimal:
    return q2(dec((invoice.get("totals") or {}).get("subtotal")))


def _invoice_tax(invoice: dict) -> Decimal:
    return q2(dec((invoice.get("totals") or {}).get("tax_amount")))


def _expense_amount(expense: dict) -> Decimal:
    return q2(dec(expense.get("amount")))


def _invoice_ref(item_id) -> str:
    try:
        return f"INV-{int(item_id):04d}"
    except (TypeError, ValueError):
        return "INV-0000"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _collect_numbers(value, out: set[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, out)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        _collect_numbers(str(value), out)
    elif isinstance(value, str):
        out.add(value)


def _number_variants(text: str) -> set[str]:
    """Every spelling of a numeric string that a writer might produce."""
    variants: set[str] = set()
    raw = str(text).strip()
    if not raw:
        return variants
    variants.add(raw)
    try:
        value = Decimal(raw)
    except Exception:
        return variants
    plain = format(value.normalize(), "f")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    variants.add(plain)
    variants.add(format(value, "f"))
    if value > 0:
        variants.add("-" + plain)
    # thousands separators, integer part only
    integer = plain.split(".")[0]
    if len(integer.lstrip("-")) >= 4:
        sign = "-" if integer.startswith("-") else ""
        digits = integer.lstrip("-")
        grouped = ""
        while len(digits) > 3:
            grouped = "," + digits[-3:] + grouped
            digits = digits[:-3]
        grouped = digits + grouped
        variants.add(sign + grouped + ("." + plain.split(".")[1] if "." in plain else ""))
    return variants


def _allowed_figures(facts: dict) -> set[str]:
    collected: set[str] = set()
    _collect_numbers(facts, collected)
    allowed: set[str] = set()
    for item in collected:
        allowed |= _number_variants(item)
    return allowed


def facts_to_text(facts: dict, currency: str) -> str:
    """Compact, readable text block built only from calculated facts."""
    lines: list[str] = []
    scope = facts.get("scope")
    unit = currency or facts.get("currency") or ""

    if scope == "overview":
        lines.append(f"Currency: {unit}")
        lines.append("Accounting (accrual basis):")
        lines.append(f"- Gross invoiced value on issued invoices: {facts.get('invoiced_gross')}")
        lines.append(f"- Net revenue (excludes tax): {facts.get('net_revenue')}")
        lines.append(f"- Tax invoiced: {facts.get('tax_invoiced')}")
        lines.append(f"- Collected from paid invoices: {facts.get('collected')}")
        lines.append(f"- Outstanding (sent and overdue): {facts.get('outstanding')}")
        lines.append(f"- Overdue invoice count: {facts.get('overdue_count')}")
        lines.append(f"- Recorded expenses: {facts.get('expenses_total')}")
        lines.append(f"- Net revenue less expenses: {facts.get('net_profit')}")
        spending = facts.get("spending") or {}
        lines.append("")
        lines.extend(_spending_lines(spending, unit))
    else:
        lines.append(f"Currency: {unit}")
        lines.extend(_spending_lines(facts, unit))
    text = "\n".join(line for line in lines if line is not None)
    if len(text) > MAX_FACTS_TEXT:
        text = text[:MAX_FACTS_TEXT]
    return text


def _spending_lines(facts: dict, unit: str) -> list[str]:
    period = facts.get("period") or {}
    lines = [
        f"Spending period: {period.get('label')} "
        f"({period.get('start')} to {period.get('end')})",
        f"- Total spending: {facts.get('total')} ({facts.get('transaction_count')} transactions)",
        f"- Previous period total: {facts.get('previous_total')}",
        f"- Change: {facts.get('change_amount')} "
        f"({facts.get('change_percent')} percent)",
        f"- Average transaction: {facts.get('average_transaction')}",
    ]
    top_category = facts.get("top_category")
    if top_category:
        lines.append(
            f"- Largest category: {top_category['name']} "
            f"{top_category['total']} ({top_category['share']} percent)"
        )
    top_vendor = facts.get("top_vendor")
    if top_vendor:
        lines.append(
            f"- Largest vendor: {top_vendor['name']} "
            f"{top_vendor['total']} ({top_vendor['share']} percent)"
        )
    lines.append("- Categories:")
    for row in (facts.get("categories") or [])[:8]:
        lines.append(
            f"    {row['name']}: {row['total']} ({row['share']} percent, "
            f"{row['count']} transactions, previous {row['previous_total']})"
        )
    budgets = facts.get("budgets") or []
    if budgets:
        lines.append("- Budgets:")
        for row in budgets[:8]:
            lines.append(
                f"    {row['name']}: {row['spent']} of {row['amount']} "
                f"({row['percent']} percent, {row['state']})"
            )
    if facts.get("recurring"):
        lines.append("- Possible recurring charges:")
        for row in facts["recurring"][:4]:
            lines.append(
                f"    {row['description']} at {row['vendor']}: "
                f"{row['charges']} charges averaging {row['average_amount']}"
            )
    if facts.get("unusual"):
        lines.append("- Unusually large expenses:")
        for row in facts["unusual"][:4]:
            lines.append(
                f"    {row['description']} ({row['category']}): {row['amount']}, "
                f"{row['ratio']} times the median {row['median']}"
            )
    cash = facts.get("cash") or {}
    if cash:
        lines.append(
            f"- Cash movement: in {cash.get('incoming')}, out {cash.get('outgoing')}, "
            f"net {cash.get('net')} (not accrual profit)"
        )
    return lines


def build_summary_prompt(facts: dict, currency: str, scope: str) -> str:
    """System prompt asking for a strict JSON explanation of the facts."""
    scope_line = (
        "the selected spending period in SpendWise"
        if scope == "spendwise"
        else "the whole business across accounting and spending"
    )
    return (
        "You are the Finance Agent summary writer. Explain the calculated financial "
        f"facts for {scope_line} in plain business English.\n\n"
        "Rules you must follow:\n"
        "- The FACTS below were calculated by the application. Treat them as the only truth.\n"
        "- Never invent a number, a date, a customer, a vendor or a category. Every figure "
        "you write must appear in the FACTS.\n"
        "- Do not add, subtract, or estimate new totals. Do not claim a saving amount unless "
        "the FACTS state it.\n"
        "- Recurring charges are patterns, not confirmed subscriptions. Say so.\n"
        "- The cash figure is not accrual profit. Keep the two apart.\n"
        "- If the FACTS are thin, say what is missing rather than filling the gap.\n"
        "- Amounts are in " + (currency or "the ledger currency") + ".\n\n"
        "Return only a JSON object with this shape:\n"
        "{\n"
        '  "headline": "one sentence",\n'
        '  "highlights": ["short factual points"],\n'
        '  "observations": ["patterns worth noticing"],\n'
        '  "cautions": ["risks or data problems"],\n'
        '  "opportunities": ["only where the facts support it, otherwise empty"]\n'
        "}\n"
        "Each list holds at most four short strings. Use an empty list when there is "
        "nothing factual to say."
    )


# ---------------------------------------------------------------------------
# Parsing and verification
# ---------------------------------------------------------------------------

def _clean_list(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SummaryError("The AI summary had a list field that was not a list.")
    items = []
    for entry in value:
        text = str(entry or "").strip()
        if text:
            items.append(text[:600])
    return items


def parse_summary(payload: dict) -> dict:
    """Validate the AI JSON into the summary groups."""
    if not isinstance(payload, dict):
        raise SummaryError("The AI summary was not a JSON object.")
    headline = str(payload.get("headline") or "").strip()
    if not headline:
        raise SummaryError("The AI summary had no headline.")
    return {
        "headline": headline[:800],
        "highlights": _clean_list(payload.get("highlights")),
        "observations": _clean_list(payload.get("observations")),
        "cautions": _clean_list(payload.get("cautions")),
        "opportunities": _clean_list(payload.get("opportunities")),
    }


_MONTH_NAME = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
)

_DATE_PATTERNS = (
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
    # Month-name dates: "November 1", "November 15, 2026", "1st November 2026".
    rf"\b{_MONTH_NAME}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\d)(?:,?\s+\d{{4}})?",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAME}(?:\s+\d{{4}})?\b",
    rf"\b{_MONTH_NAME}\s+\d{{4}}\b",
    r"\d{1,2}/\d{1,2}/\d{2,4}",
    r"\b\d{4}\b",
)


def verify_figures(text: str, facts: dict) -> dict:
    """Check that every number in the text appears in the calculated facts."""
    allowed = _allowed_figures(facts)
    cleaned = str(text or "")
    for pattern in _DATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    # References such as INV-0001 are identifiers, not amounts. Only strip a
    # number that is glued to letters, never a number after a space.
    cleaned = re.sub(r"[A-Za-z]+[-_]?\d+", " ", cleaned)

    unsupported: set[str] = set()
    for match in re.findall(r"\d[\d,]*(?:\.\d+)?", cleaned):
        token = match.strip().rstrip(".")
        if not token:
            continue
        variants = _number_variants(token.replace(",", ""))
        if not variants & allowed:
            unsupported.add(token)
    return {"ok": not unsupported, "unsupported": sorted(unsupported)}


# ---------------------------------------------------------------------------
# Deterministic summary
# ---------------------------------------------------------------------------

def _summary_money(value, currency: str) -> str:
    """Grouped amount with the ledger currency code, e.g. `8,600.00 EUR`."""
    try:
        text = f"{dec(value):,.2f}"
    except (ValueError, TypeError):
        text = str(value)
    return f"{text} {currency}".strip()


def _count(value, singular: str, plural: Optional[str] = None) -> str:
    number = int(value or 0)
    word = singular if number == 1 else (plural or singular + "s")
    return f"{number} {word}"


def _fmt_change(amount: str, percent: Optional[str], currency: str) -> str:
    direction = "up" if dec(amount) >= 0 else "down"
    if percent is None:
        return f"spending is {direction} from the previous period"
    return f"spending is {direction} {s2(abs(dec(percent)))} percent from the previous period"


def compute_summary(facts: dict, currency: str, scope: str) -> dict:
    """A factual summary written from the calculated facts alone. No AI."""
    if scope == "overview":
        return _compute_overview(facts, currency)
    total = dec(facts.get("total"))
    period = facts.get("period") or {}
    count = facts.get("transaction_count") or 0
    unit = currency or ""

    if total <= 0:
        headline = (
            f"No spending is recorded for {period.get('label') or 'this period'}."
        )
    else:
        headline = (
            f"{_summary_money(total, unit)} was recorded across {_count(count, 'transaction')} for "
            f"{period.get('label') or 'this period'}; "
            f"{_fmt_change(facts.get('change_amount'), facts.get('change_percent'), unit)}."
        )

    highlights: list[str] = []
    top_category = facts.get("top_category")
    if top_category:
        highlights.append(
            f"Largest category: {top_category['name']} at {_summary_money(top_category['total'], unit)} "
            f"({top_category['share']} percent of spending)."
        )
    top_vendor = facts.get("top_vendor")
    if top_vendor:
        highlights.append(
            f"Largest vendor: {top_vendor['name']} at {_summary_money(top_vendor['total'], unit)} "
            f"({top_vendor['share']} percent of spending)."
        )
    budget_totals = facts.get("budget_totals") or {}
    if budget_totals.get("count"):
        highlights.append(
            f"{_count(budget_totals['count'], 'budget')} set; combined utilisation is "
            f"{budget_totals.get('percent')} percent."
        )

    observations = [
        f"{row['name']} accounts for {row['share']} percent of spending "
        f"({_summary_money(row['total'], unit)} across {_count(row['count'], 'transaction')})."
        for row in (facts.get("categories") or [])
        if dec(row.get("share")) >= 25
    ]

    cautions: list[str] = []
    for row in facts.get("budgets") or []:
        if row.get("state") == "over":
            cautions.append(
                f"{row['name']} is over budget: {_summary_money(row['spent'], unit)} of "
                f"{_summary_money(row['amount'], unit)}."
            )
    percent = facts.get("change_percent")
    if percent is not None and dec(percent) > 25:
        cautions.append(
            f"Spending is up {s2(dec(percent))} percent against the previous period."
        )

    return {
        "headline": headline,
        "highlights": highlights,
        "observations": observations,
        "cautions": cautions,
        "opportunities": [],
    }


def _compute_overview(facts: dict, currency: str) -> dict:
    unit = currency or facts.get("currency") or ""
    net = facts.get("net_profit")
    headline = (
        f"Net revenue is {_summary_money(facts.get('net_revenue'), unit)} against "
        f"{_summary_money(facts.get('expenses_total'), unit)} of recorded expenses, "
        f"a net position of {_summary_money(net, unit)}."
    )
    spending = facts.get("spending") or {}
    highlights = [
        f"Outstanding on sent and overdue invoices: {_summary_money(facts.get('outstanding'), unit)}.",
        f"Collected from paid invoices: {_summary_money(facts.get('collected'), unit)}.",
    ]
    if spending.get("total"):
        highlights.append(
            f"Spending for {spending['period']['label']}: {_summary_money(spending['total'], unit)} "
            f"across {_count(spending['transaction_count'], 'transaction')}."
        )
    if spending.get("top_category"):
        top = spending["top_category"]
        highlights.append(
            f"Largest spending category: {top['name']} at {_summary_money(top['total'], unit)}."
        )
    observations = []
    if spending.get("change_percent") is not None:
        observations.append(
            f"Spending is {_summary_money(spending['change_amount'], unit)} "
            f"({spending['change_percent']} percent) versus the previous period."
        )
    cautions = []
    if facts.get("overdue_count"):
        cautions.append(
            f"{_count(facts['overdue_count'], 'invoice')} overdue, "
            f"{_summary_money(facts.get('outstanding'), unit)} outstanding in total."
        )
    for row in spending.get("budgets") or []:
        if row.get("state") == "over":
            cautions.append(f"{row['name']} is over budget.")
    return {
        "headline": headline,
        "highlights": highlights,
        "observations": observations,
        "cautions": cautions,
        "opportunities": [],
    }


# ---------------------------------------------------------------------------
# Fingerprint and cache
# ---------------------------------------------------------------------------

def summary_fingerprint(facts: dict, scope: str) -> str:
    """Stable hash of scope plus facts. Changes when a figure changes."""
    try:
        body = json.dumps(facts, sort_keys=True, default=str)
    except (TypeError, ValueError):
        body = repr(facts)
    digest = hashlib.sha256(f"{scope}:{body}".encode("utf-8")).hexdigest()
    return digest[:32]


def cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "summaries.json"


def read_cache(data_dir: Path) -> dict:
    path = cache_path(data_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_cache(data_dir: Path, cache: dict) -> None:
    path = cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def get_cached(data_dir: Path, scope: str, facts_fingerprint: str) -> Optional[dict]:
    entry = read_cache(data_dir).get(scope)
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != facts_fingerprint:
        return None
    if not isinstance(entry.get("summary"), dict):
        return None
    return entry


def store_cached(data_dir: Path, scope: str, facts_fingerprint: str,
                 payload: dict) -> dict:
    stored = {
        "scope": scope,
        "fingerprint": facts_fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": payload.get("source", "computed"),
        "summary": payload.get("summary"),
        "computed": payload.get("computed"),
        "verified": bool(payload.get("verified", False)),
        "unverified_figures": list(payload.get("unverified_figures") or []),
        "error": payload.get("error") or "",
    }
    cache = read_cache(data_dir)
    cache[scope] = stored
    write_cache(data_dir, cache)
    return stored


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _summary_text(summary: dict) -> str:
    parts = [summary.get("headline") or ""]
    for key in SUMMARY_GROUP_KEYS:
        parts.extend(summary.get(key) or [])
    return "\n".join(str(part) for part in parts)


def build_summary(facts: dict, currency: str, scope: str,
                  gemini_json: Optional[Callable] = None) -> dict:
    """Produce a summary payload, never raising for an AI failure.

    On success the payload has source "gemini"; otherwise it carries the
    deterministic computed summary with source "computed" and an error note.
    """
    computed = compute_summary(facts, currency, scope)
    base = {
        "scope": scope,
        "currency": currency,
        "summary": computed,
        "source": "computed",
        "computed": computed,
        "verified": True,
        "unverified_figures": [],
        "error": "",
    }
    if not callable(gemini_json):
        base["error"] = "The AI service is not configured, so this summary was computed by the app."
        return base

    try:
        payload = gemini_json(
            build_summary_prompt(facts, currency, scope),
            [{"text": facts_to_text(facts, currency)}],
        )
        summary = parse_summary(payload)
    except SummaryError as error:
        base["error"] = str(error)
        return base
    except Exception as error:  # Gemini failures, timeouts, unreadable replies
        base["error"] = f"The AI summary could not be generated: {error}"
        return base

    check = verify_figures(_summary_text(summary), facts)
    return {
        "scope": scope,
        "currency": currency,
        "summary": summary,
        "source": "gemini",
        "computed": computed,
        "verified": check["ok"],
        "unverified_figures": check["unsupported"],
        "error": "",
    }

# ===========================================================================
# Section 8: What-if Scenario Engine
# ===========================================================================


from decimal import Decimal, InvalidOperation
from typing import Any, Optional


# A single monthly adjustment may not exceed this; it stops a typo from
# producing absurd output while still allowing large, legitimate scenarios.
MAX_MONTHLY_DELTA = Decimal("100000")
MAX_ONE_TIME = Decimal("10000000")
PERCENT_MIN = Decimal("-100")
PERCENT_MAX = Decimal("500")


class ScenarioError(ValueError):
    """A scenario that could not be interpreted safely."""


BASELINE_FIELDS = ("income", "expenses", "subscriptions", "savings")

# Order matters only for presentation.
ADJUSTMENT_KEYS = (
    "income_delta",
    "income_change_percent",
    "expenses_delta",
    "subscriptions_delta",
    "subscriptions_target",
    "bills_delta",
    "savings_delta",
    "one_time_purchase",
)


def empty_scenario() -> dict:
    return {
        "income_delta": ZERO,
        "income_change_percent": None,
        "expenses_delta": ZERO,
        "subscriptions_delta": ZERO,
        "subscriptions_target": None,
        "bills_delta": ZERO,
        "savings_delta": ZERO,
        "one_time_purchase": ZERO,
        "category_budgets": [],
        "label": "",
    }


def _number(value: Any, field: str) -> Decimal:
    """Strict numeric parse: a non-numeric string is an error, not zero."""
    if isinstance(value, bool):
        raise ScenarioError(f"'{field}' was not a number.")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ScenarioError(f"'{field}' was not a number.") from error
    if amount != amount:  # NaN guard (Decimal NaN is never equal to itself)
        raise ScenarioError(f"'{field}' was not a number.")
    return amount


def _scenario_money(value: Any, field: str, limit: Decimal,
           minimum: Optional[Decimal] = None) -> Decimal:
    if value is None or value == "":
        return ZERO
    amount = _number(value, field)
    if abs(amount) > limit:
        raise ScenarioError(f"'{field}' is outside the allowed range.")
    if minimum is not None and amount < minimum:
        raise ScenarioError(f"'{field}' cannot be negative.")
    return q2(amount)


def _optional_scenario_money(value: Any, field: str, limit: Decimal,
                    minimum: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    return _scenario_money(value, field, limit, minimum)


def _scenario_percent(value: Any, field: str) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    amount = _number(value, field)
    if amount < PERCENT_MIN or amount > PERCENT_MAX:
        raise ScenarioError(f"'{field}' must be between -100 and 500 percent.")
    return q2(amount)


def scenario_validate(raw: Any) -> dict:
    """Normalise and check a scenario. Raises `ScenarioError` when unsafe."""
    if raw is None:
        return empty_scenario()
    if not isinstance(raw, dict):
        raise ScenarioError("The scenario must be a JSON object.")

    scenario = empty_scenario()
    scenario["income_delta"] = _scenario_money(raw.get("income_delta"), "income_delta", MAX_MONTHLY_DELTA)
    scenario["income_change_percent"] = _scenario_percent(raw.get("income_change_percent"), "income_change_percent")
    scenario["expenses_delta"] = _scenario_money(raw.get("expenses_delta"), "expenses_delta", MAX_MONTHLY_DELTA)
    scenario["subscriptions_delta"] = _scenario_money(raw.get("subscriptions_delta"), "subscriptions_delta", MAX_MONTHLY_DELTA)
    scenario["subscriptions_target"] = _optional_scenario_money(
        raw.get("subscriptions_target"), "subscriptions_target", MAX_MONTHLY_DELTA, ZERO
    )
    scenario["bills_delta"] = _scenario_money(raw.get("bills_delta"), "bills_delta", MAX_MONTHLY_DELTA)
    scenario["savings_delta"] = _scenario_money(raw.get("savings_delta"), "savings_delta", MAX_MONTHLY_DELTA)
    scenario["one_time_purchase"] = _scenario_money(raw.get("one_time_purchase"), "one_time_purchase", MAX_ONE_TIME)
    scenario["label"] = str(raw.get("label") or "").strip()[:200]

    budgets = raw.get("category_budgets")
    if budgets:
        if not isinstance(budgets, list):
            raise ScenarioError("'category_budgets' must be a list.")
        cleaned: list[dict] = []
        for entry in budgets[:40]:
            if not isinstance(entry, dict):
                continue
            category = str(entry.get("category") or "").strip()
            if not category:
                continue
            monthly = _scenario_money(entry.get("monthly"), "category_budgets", MAX_MONTHLY_DELTA)
            if monthly < 0:
                raise ScenarioError("A category budget cannot be negative.")
            cleaned.append({"category": category[:80], "monthly": monthly})
        scenario["category_budgets"] = cleaned
    return scenario


def _baseline_value(baseline: dict, field_name: str) -> Decimal:
    return q2(dec(baseline.get(field_name)))


def baseline_category_monthly(baseline: dict) -> dict[str, Decimal]:
    categories = baseline.get("categories") or {}
    if not isinstance(categories, dict):
        return {}
    return {str(key): q2(dec(value)) for key, value in categories.items()}


def scenario_describe(baseline: dict, scenario: dict) -> list[str]:
    """Plain-English list of the adjustments, for prompts and the UI."""
    notes: list[str] = []
    if scenario.get("income_change_percent") not in (None, ZERO):
        direction = "increase" if dec(scenario["income_change_percent"]) > 0 else "decrease"
        notes.append(f"Income {direction} of {abs(dec(scenario['income_change_percent']))}% per month.")
    if scenario.get("income_delta"):
        notes.append(f"Income changes by {s2(scenario['income_delta'])} per month.")
    if scenario.get("subscriptions_target") is not None:
        notes.append(f"Subscription spending set to {s2(scenario['subscriptions_target'])} per month.")
    elif scenario.get("subscriptions_delta"):
        notes.append(f"Subscription spending changes by {s2(scenario['subscriptions_delta'])} per month.")
    if scenario.get("expenses_delta"):
        notes.append(f"Other monthly expenses change by {s2(scenario['expenses_delta'])}.")
    if scenario.get("bills_delta"):
        notes.append(f"Recurring bills change by {s2(scenario['bills_delta'])} per month.")
    if scenario.get("savings_delta"):
        notes.append(f"Monthly savings change by {s2(scenario['savings_delta'])}.")
    if scenario.get("one_time_purchase"):
        notes.append(f"One-time purchase of {s2(scenario['one_time_purchase'])} this month.")
    for entry in scenario.get("category_budgets") or []:
        baseline_value = baseline_category_monthly(baseline).get(entry["category"], ZERO)
        notes.append(
            f"Budget for {entry['category']} moves from {s2(baseline_value)} "
            f"to {s2(entry['monthly'])} per month."
        )
    return notes


def scenario_simulate(baseline: dict, scenario: dict, months: int = 12) -> dict:
    """Apply a validated scenario to a baseline and return the comparison.

    `baseline` carries monthly figures: income, expenses, subscriptions,
    savings, and an optional category→monthly-average map.
    """
    months = max(1, int(months or 12))

    income = _baseline_value(baseline, "income")
    expenses = _baseline_value(baseline, "expenses")
    subscriptions = _baseline_value(baseline, "subscriptions")
    savings = _baseline_value(baseline, "savings")

    income_delta = dec(scenario.get("income_delta"))
    percent = scenario.get("income_change_percent")
    if percent not in (None, ""):
        income_delta += q2(income * dec(percent) / Decimal("100"))

    expense_delta = dec(scenario.get("expenses_delta")) + dec(scenario.get("bills_delta"))

    if scenario.get("subscriptions_target") is not None:
        subscription_delta = dec(scenario["subscriptions_target"]) - subscriptions
    else:
        subscription_delta = dec(scenario.get("subscriptions_delta"))
    expense_delta += subscription_delta

    category_rows = []
    baseline_categories = baseline_category_monthly(baseline)
    category_delta = ZERO
    for entry in scenario.get("category_budgets") or []:
        current = baseline_categories.get(entry["category"], ZERO)
        target = q2(entry["monthly"])
        change = q2(target - current)
        category_delta += change
        category_rows.append({
            "category": entry["category"],
            "current": s2(current),
            "simulated": s2(target),
            "monthly_difference": s2(change),
            "annual_difference": s2(change * months),
        })
    expense_delta += category_delta

    savings_delta = dec(scenario.get("savings_delta"))
    one_time = q2(dec(scenario.get("one_time_purchase")))

    new_income = q2(income + income_delta)
    new_expenses = q2(expenses + expense_delta)
    new_subscriptions = q2(subscriptions + subscription_delta)
    new_savings = q2(savings + savings_delta)

    # Free cash is what is left after expenses and planned savings.
    current_free = q2(income - expenses - savings)
    simulated_free = q2(new_income - new_expenses - new_savings)

    monthly_difference = q2(simulated_free - current_free)
    annual_difference = q2(monthly_difference * months - one_time)
    first_month_difference = q2(monthly_difference - one_time)

    return {
        "months": months,
        "current": {
            "income": s2(income),
            "expenses": s2(expenses),
            "subscriptions": s2(subscriptions),
            "savings": s2(savings),
            "free_cash": s2(current_free),
        },
        "simulated": {
            "income": s2(new_income),
            "expenses": s2(new_expenses),
            "subscriptions": s2(new_subscriptions),
            "savings": s2(new_savings),
            "free_cash": s2(simulated_free),
        },
        "difference": {
            "income": s2(income_delta),
            "expenses": s2(expense_delta),
            "subscriptions": s2(subscription_delta),
            "savings": s2(savings_delta),
            "monthly": s2(monthly_difference),
            "annual": s2(annual_difference),
            "first_month": s2(first_month_difference),
            "one_time_purchase": s2(one_time),
        },
        "categories": category_rows,
        "notes": scenario_describe(baseline, scenario),
    }


# ---------------------------------------------------------------------------
# Module shims and namespaces
# ---------------------------------------------------------------------------

class _CoreNamespace:
    ZERO = ZERO
    CENT = CENT
    Q4 = Q4
    MONEY_MAX = MONEY_MAX
    INVOICE_STATUS = INVOICE_STATUS
    ISSUED_STATUS = ISSUED_STATUS
    EXPENSE_CATEGORIES = EXPENSE_CATEGORIES
    dec = staticmethod(dec)
    q2 = staticmethod(q2)
    q4 = staticmethod(q4)
    s2 = staticmethod(s2)
    s4 = staticmethod(s4)
    json_default = staticmethod(json_default)
    parse_iso_date = staticmethod(parse_iso_date)
    invoice_total = staticmethod(invoice_total)
    invoice_subtotal = staticmethod(invoice_subtotal)
    invoice_tax = staticmethod(invoice_tax)
    expense_amount = staticmethod(expense_amount)
    issued_invoices = staticmethod(issued_invoices)
    in_period = staticmethod(in_period)

core = _CoreNamespace()
_parse_iso_date = parse_iso_date
_json_default = json_default

class _AINamespace:
    DEFAULT_GEMINI_MODEL = DEFAULT_GEMINI_MODEL
    DEFAULT_NVIDIA_MODEL = DEFAULT_NVIDIA_MODEL
    DEFAULT_NVIDIA_BASE_URL = DEFAULT_NVIDIA_BASE_URL
    GEMINI_ENDPOINT = GEMINI_ENDPOINT
    DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
    DEFAULT_MAX_RETRIES = DEFAULT_MAX_RETRIES
    RETRY_BACKOFF_SECONDS = RETRY_BACKOFF_SECONDS
    MAX_IMAGE_PARTS = MAX_IMAGE_PARTS
    ProviderError = ProviderError
    AIConfig = AIConfig
    gemini_available = staticmethod(gemini_available)
    nvidia_available = staticmethod(nvidia_available)
    redact = staticmethod(redact)
    gemini_text = staticmethod(gemini_text)
    gemini_json = staticmethod(gemini_json)
    to_openai_parts = staticmethod(to_openai_parts)
    nvidia_text = staticmethod(nvidia_text)
    nvidia_json = staticmethod(nvidia_json)
    document_json = staticmethod(document_json)
    provider_status = staticmethod(provider_status)

ai = _AINamespace()

class _SpendwiseNamespace:
    PERIODS = PERIODS
    MONTH_LABELS = MONTH_LABELS
    NOTABLE_CHANGE_PCT = NOTABLE_CHANGE_PCT
    BUDGET_NEAR_PCT = BUDGET_NEAR_PCT
    UNUSUAL_RATIO = UNUSUAL_RATIO
    UNUSUAL_MIN_HISTORY = UNUSUAL_MIN_HISTORY
    UNUSUAL_LOOKBACK_MONTHS = UNUSUAL_LOOKBACK_MONTHS
    RECURRING_MIN_CHARGES = RECURRING_MIN_CHARGES
    RECURRING_MIN_MONTHS = RECURRING_MIN_MONTHS
    RECURRING_MIN_INTERVAL_DAYS = RECURRING_MIN_INTERVAL_DAYS
    RECURRING_MAX_INTERVAL_DAYS = RECURRING_MAX_INTERVAL_DAYS
    DUPLICATE_WINDOW_DAYS = DUPLICATE_WINDOW_DAYS
    CASH_FLOW_NOTE = CASH_FLOW_NOTE
    SpendWiseError = SpendWiseError
    month_start = staticmethod(month_start)
    month_key = staticmethod(month_key)
    month_label = staticmethod(month_label)
    add_months = staticmethod(add_months)
    resolve_period = staticmethod(resolve_period)
    filter_expenses = staticmethod(filter_expenses)
    spending_stats = staticmethod(spending_stats)
    vendor_key = staticmethod(vendor_key)
    monthly_trend = staticmethod(monthly_trend)
    weekly_trend = staticmethod(weekly_trend)
    category_breakdown = staticmethod(category_breakdown)
    vendor_breakdown = staticmethod(vendor_breakdown)
    spending_matrix = staticmethod(spending_matrix)
    category_transactions = staticmethod(category_transactions)
    budget_status = staticmethod(budget_status)
    budget_totals = staticmethod(budget_totals)
    recurring_suggestions = staticmethod(recurring_suggestions)
    unusual_expenses = staticmethod(unusual_expenses)
    duplicate_charges = staticmethod(duplicate_charges)
    build_insights = staticmethod(build_insights)
    cash_flow = staticmethod(cash_flow)
    ordered_rules = staticmethod(ordered_rules)
    suggest_category = staticmethod(suggest_category)
    rule_suggestions = staticmethod(rule_suggestions)
    parse_iso_date = staticmethod(parse_iso_date)

spendwise = _SpendwiseNamespace()

class _AnalysisNamespace:
    MAX_SUGGESTIONS = MAX_SUGGESTIONS
    build_facts = staticmethod(analysis_build_facts)
    savings_potential = staticmethod(savings_potential)
    build_facts_text = staticmethod(analysis_build_facts_text)
    parse_suggestions = staticmethod(parse_suggestions)

analysis = _AnalysisNamespace()

class _SubscriptionsNamespace:
    MAX_ALTERNATIVES = MAX_ALTERNATIVES
    detect = staticmethod(subscription_detect)
    totals = staticmethod(subscription_totals)
    compare = staticmethod(subscription_compare)
    parse_alternatives = staticmethod(subscription_parse_alternatives)
    build_facts = staticmethod(subscription_build_facts)

subscriptions = _SubscriptionsNamespace()

class _DocumentsNamespace:
    DocumentError = DocumentError
    ExtractionError = ExtractionError
    pymupdf = pymupdf
    date = date
    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS
    BLOCKED_EXTENSIONS = BLOCKED_EXTENSIONS
    MIME_TYPES = MIME_TYPES
    MAX_FILE_BYTES = MAX_FILE_BYTES
    MAX_BATCH_FILES = MAX_BATCH_FILES
    DOCUMENT_FIELDS = DOCUMENT_FIELDS
    EXTRACTION_SYSTEM_PROMPT = EXTRACTION_SYSTEM_PROMPT
    sanitize_filename = staticmethod(sanitize_filename)
    is_safe_stored_name = staticmethod(is_safe_stored_name)
    validate_upload = staticmethod(validate_upload)
    validate_batch = staticmethod(validate_batch)
    store_file = staticmethod(store_file)
    read_file = staticmethod(read_file)
    delete_file = staticmethod(delete_file)
    extract_content = staticmethod(extract_content)
    build_extraction_prompt = staticmethod(build_extraction_prompt)
    build_user_parts = staticmethod(build_user_parts)
    parse_extraction = staticmethod(parse_extraction)
    validate_financials = staticmethod(validate_financials)
    find_duplicates = staticmethod(find_duplicates)
    to_import_payload = staticmethod(to_import_payload)
    run_extraction = staticmethod(run_extraction)
    local_extraction = staticmethod(local_extraction)

documents = _DocumentsNamespace()

class _SummariesNamespace:
    SUMMARY_SCOPES = SUMMARY_SCOPES
    SUMMARY_GROUP_KEYS = SUMMARY_GROUP_KEYS
    MAX_FACTS_TEXT = MAX_FACTS_TEXT
    SummaryError = SummaryError
    build_spendwise_facts = staticmethod(build_spendwise_facts)
    build_overview_facts = staticmethod(build_overview_facts)
    facts_to_text = staticmethod(facts_to_text)
    build_summary_prompt = staticmethod(build_summary_prompt)
    parse_summary = staticmethod(parse_summary)
    verify_figures = staticmethod(verify_figures)
    compute_summary = staticmethod(compute_summary)
    fingerprint = staticmethod(summary_fingerprint)
    cache_path = staticmethod(cache_path)
    read_cache = staticmethod(read_cache)
    write_cache = staticmethod(write_cache)
    get_cached = staticmethod(get_cached)
    store_cached = staticmethod(store_cached)
    build_summary = staticmethod(build_summary)

summaries = _SummariesNamespace()

class _ScenarioNamespace:
    MAX_MONTHLY_DELTA = MAX_MONTHLY_DELTA
    MAX_ONE_TIME = MAX_ONE_TIME
    PERCENT_MIN = PERCENT_MIN
    PERCENT_MAX = PERCENT_MAX
    BASELINE_FIELDS = BASELINE_FIELDS
    ADJUSTMENT_KEYS = ADJUSTMENT_KEYS
    ScenarioError = ScenarioError
    empty_scenario = staticmethod(empty_scenario)
    validate = staticmethod(scenario_validate)
    simulate = staticmethod(scenario_simulate)
    describe = staticmethod(scenario_describe)
    baseline_category_monthly = staticmethod(baseline_category_monthly)

scenario = _ScenarioNamespace()
scenario_module = scenario

# Register modules in sys.modules so `import spendwise`, `from documents import ...` work everywhere
for _mod_name, _obj in [
    ("core", core),
    ("ai", ai),
    ("spendwise", spendwise),
    ("analysis", analysis),
    ("subscriptions", subscriptions),
    ("documents", documents),
    ("summaries", summaries),
    ("scenario", scenario),
]:
    _mod = types.ModuleType(_mod_name)
    for _k in dir(_obj):
        if not _k.startswith("__"):
            setattr(_mod, _k, getattr(_obj, _k))
    sys.modules[_mod_name] = _mod



# ===========================================================================
# Section 9: Application Web Layer & Storage
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
# FINANCE_DATA_DIR is the current name; ACCOUNTING_DATA_DIR stays supported so
# existing installations keep pointing at the same ledger files.
DATA_DIR = Path(
    os.getenv("FINANCE_DATA_DIR") or os.getenv("ACCOUNTING_DATA_DIR") or str(BASE_DIR / "data")
).expanduser().resolve()
ENV_FILE = BASE_DIR / ".env"
BACKUP_DIR = DATA_DIR / "backups"
SETTINGS_FILE = DATA_DIR / "settings.json"

def _sync_data_dir() -> Path:
    global DATA_DIR, BACKUP_DIR, SETTINGS_FILE
    env_dir = os.getenv("FINANCE_DATA_DIR") or os.getenv("ACCOUNTING_DATA_DIR")
    if env_dir:
        target = Path(env_dir).expanduser().resolve()
        if DATA_DIR != target:
            DATA_DIR = target
            BACKUP_DIR = DATA_DIR / "backups"
            SETTINGS_FILE = DATA_DIR / "settings.json"
    return DATA_DIR
DEFAULT_PORT = 2611
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
APP_VERSION = "2.0.0"

REPORTING_BASIS = (
    "Accrual basis. Revenue is recognised when an invoice is issued "
    "(sent, overdue or paid). Draft and cancelled invoices are excluded. "
    "Revenue excludes sales tax, which is reported separately. "
    "Expenses are recognised on the date incurred."
)
BALANCE_SHEET_NOTE = (
    "Partial management position, not a complete statutory balance sheet. "
    "Accounts receivable and cash received include invoiced tax. Output tax on issued "
    "invoices is treated as payable because tax remittances are not tracked. Recorded "
    "expenses are treated as settled cash outflows, and recoverable input tax is not "
    "modelled. Only reimbursable expenses that have not been reimbursed are presented "
    "as reimbursement liabilities."
)

DEFAULT_SETTINGS = {
    "currency_code": "EUR",
    "currency_symbol": "€",
    "locale": "en-IE",
    "default_tax_rate": "0.20",
    "date_format": "medium",
    "business_name": "",
    "business_address": "",
    "business_email": "",
    "payment_details": "",
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
LOCALE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_env_file(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE pairs without overriding real environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", ai.DEFAULT_NVIDIA_MODEL).strip()
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", ai.DEFAULT_NVIDIA_BASE_URL).strip()
ASSISTANT_MAX_MESSAGE = 1000
ASSISTANT_WINDOW_SECONDS = 60
ASSISTANT_MAX_REQUESTS = 12
ASSISTANT_MAX_ANSWER = 6000
GEMINI_TIMEOUT_SECONDS = 60


def ai_config() -> "ai.AIConfig":
    """Build the provider configuration from the environment on every call.

    Reading it each time means a deployment (or a test) can change the keys or
    model names without restarting, and the keys are never cached where the
    response layer could leak them.
    """
    return ai.AIConfig.from_env()


def nvidia_ready() -> bool:
    return ai.nvidia_available(ai_config())


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _validate_settings_dict(raw: dict) -> dict:
    settings = {**DEFAULT_SETTINGS, **{key: raw[key] for key in DEFAULT_SETTINGS if key in raw}}
    code = str(settings["currency_code"] or "EUR").strip().upper()
    if not CURRENCY_RE.match(code):
        raise ValueError("Currency code must be a three-letter ISO code such as EUR or USD.")
    symbol = str(settings["currency_symbol"] or "€").strip()
    if not 1 <= len(symbol) <= 4:
        raise ValueError("Currency symbol must be between 1 and 4 characters.")
    locale = str(settings["locale"] or "en-IE").strip()
    if not LOCALE_RE.match(locale):
        raise ValueError("Locale must look like en-US, en-GB or en-IE.")
    rate = dec(settings["default_tax_rate"])
    if rate < 0 or rate > 1:
        raise ValueError("Default tax rate must be between 0 and 100 percent.")
    date_format = str(settings["date_format"] or "medium").strip().lower()
    if date_format not in ("short", "medium", "long", "iso"):
        raise ValueError("Date format must be short, medium, long or iso.")
    return {
        "currency_code": code,
        "currency_symbol": symbol,
        "locale": locale,
        "default_tax_rate": s4(rate),
        "date_format": date_format,
        "business_name": str(settings["business_name"] or "").strip()[:200],
        "business_address": str(settings["business_address"] or "").strip()[:500],
        "business_email": str(settings["business_email"] or "").strip()[:254],
        "payment_details": str(settings["payment_details"] or "").strip()[:500],
    }


def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if not isinstance(stored, dict):
            raise ValueError("Settings must be a JSON object.")
        return _validate_settings_dict(stored)
    except (OSError, ValueError, json.JSONDecodeError):
        # Settings are not ledger records. A damaged settings file must not make
        # the ledger unusable; defaults are used and the file is left untouched.
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> dict:
    clean = _validate_settings_dict(settings)
    _atomic_write_json(SETTINGS_FILE, clean)
    return clean


# ---------------------------------------------------------------------------
# JSON storage
# ---------------------------------------------------------------------------

class StorageError(RuntimeError):
    """A safe, user-facing storage failure."""


_DATA_LOCK = threading.RLock()


def _ensure_dir() -> None:
    _sync_data_dir()
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _filepath(name: str) -> Path:
    _sync_data_dir()
    return DATA_DIR / f"{name}.json"


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically so an interrupted save cannot truncate records."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
    except OSError as error:
        raise StorageError(f"Could not prepare a save location for {path.name}: {error}") from error

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise StorageError(f"Could not save {path.name} safely: {error}") from error


def _read(name: str) -> list[dict]:
    _ensure_dir()
    path = _filepath(name)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        raise StorageError(
            f"{path.name} is not valid JSON (line {error.lineno}). "
            "The file has not been overwritten. Restore a backup or repair the file."
        ) from error
    except OSError as error:
        raise StorageError(f"{path.name} could not be read: {error}") from error
    if not isinstance(data, list):
        raise StorageError(f"{path.name} must contain a JSON list of records.")
    return data


def _write(name: str, data: list[dict]) -> None:
    _ensure_dir()
    _atomic_write_json(_filepath(name), data)


def _next_id(items: list[dict]) -> int:
    if not items:
        return 1
    try:
        return max(int(item.get("id", 0)) for item in items) + 1
    except (TypeError, ValueError):
        return len(items) + 1


def all_items(name: str) -> list[dict]:
    return _read(name)


def get_item(name: str, item_id: int) -> Optional[dict]:
    for item in _read(name):
        if item.get("id") == item_id:
            return item
    return None


def create_item(name: str, item: dict) -> dict:
    with _DATA_LOCK:
        items = _read(name)
        item["id"] = _next_id(items)
        item["created_at"] = datetime.now().isoformat()
        item["updated_at"] = item["created_at"]
        items.append(item)
        _write(name, items)
        return item


def update_item(name: str, item_id: int, updates: dict) -> Optional[dict]:
    with _DATA_LOCK:
        items = _read(name)
        for index, item in enumerate(items):
            if item.get("id") == item_id:
                updates.pop("id", None)
                updates.pop("created_at", None)
                updates["updated_at"] = datetime.now().isoformat()
                items[index].update(updates)
                _write(name, items)
                return items[index]
        return None


def delete_item(name: str, item_id: int) -> bool:
    with _DATA_LOCK:
        items = _read(name)
        remaining = [item for item in items if item.get("id") != item_id]
        if len(remaining) == len(items):
            return False
        _write(name, remaining)
        return True


def delete_items(name: str, item_ids: list[int]) -> int:
    with _DATA_LOCK:
        ids = set(item_ids)
        items = _read(name)
        remaining = [item for item in items if item.get("id") not in ids]
        removed = len(items) - len(remaining)
        if removed:
            _write(name, remaining)
        return removed


def update_items(name: str, item_ids: list[int], updates: dict) -> int:
    with _DATA_LOCK:
        ids = set(item_ids)
        items = _read(name)
        updated = 0
        timestamp = datetime.now().isoformat()
        for item in items:
            if item.get("id") in ids:
                item.update(updates)
                item["updated_at"] = timestamp
                updated += 1
        if updated:
            _write(name, items)
        return updated


# ---------------------------------------------------------------------------
# Activity history
# ---------------------------------------------------------------------------

def log_activity(action: str, record_type: str, record_id: Optional[int] = None,
                 reference: str = "", details: str = "") -> None:
    entry = {
        "id": 0,
        "action": action[:80],
        "record_type": record_type[:40],
        "record_id": record_id,
        "reference": reference[:120],
        "details": details[:500],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with _DATA_LOCK:
            items = _read("activity")
            entry["id"] = _next_id(items)
            items.append(entry)
            if len(items) > 500:
                items = items[-500:]
            _write("activity", items)
    except StorageError:
        # Activity is secondary. Never overwrite a damaged file or block a
        # financial operation because history cannot be read.
        return


def read_activity(limit: int = 100) -> list[dict]:
    try:
        items = _read("activity")
    except StorageError:
        raise
    return sorted(items, key=lambda item: item.get("id", 0), reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Backups and validation
# ---------------------------------------------------------------------------

def _backup_settings() -> dict:
    # Only public settings. API keys are never part of a backup.
    return load_settings()


def create_backup(reason: str = "manual") -> tuple[dict, Path]:
    backup = {
        "format": "finance-agent-backup",
        "version": 1,
        "app": "Finance Agent",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "reason": reason[:120],
        "settings": _backup_settings(),
        "invoices": all_items("invoices"),
        "expenses": all_items("expenses"),
        "activity": read_activity(500),
        "budgets": all_items("budgets"),
        "rules": all_items("rules"),
    }
    slug = re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-") or "backup"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = BACKUP_DIR / f"{stamp}-{slug}.json"
    _atomic_write_json(path, backup)
    return backup, path


def backup_before(reason: str) -> Path:
    """Create a protective backup. A failure cancels the destructive action."""
    _, path = create_backup(reason)
    return path


def _is_decimal(value) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        Decimal(str(value))
        return True
    except (InvalidOperation, TypeError, ValueError):
        return False


def _validate_backup_invoice(invoice: dict, index: int) -> None:
    prefix = f"Invoice #{index + 1}"
    if not isinstance(invoice, dict):
        raise ValueError(f"{prefix} is not an object.")
    if not isinstance(invoice.get("id"), int) or invoice["id"] < 1:
        raise ValueError(f"{prefix} has an invalid id.")
    if not str(invoice.get("client_name") or "").strip():
        raise ValueError(f"{prefix} has no client name.")
    if invoice.get("status") not in INVOICE_STATUS:
        raise ValueError(f"{prefix} has an invalid status.")
    if not isinstance(invoice.get("line_items"), list):
        raise ValueError(f"{prefix} has invalid line items.")
    for item_index, item in enumerate(invoice["line_items"]):
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} line item #{item_index + 1} is not an object.")
        if not str(item.get("description") or "").strip():
            raise ValueError(f"{prefix} line item #{item_index + 1} has no description.")
        if not _is_decimal(item.get("quantity")) or dec(item.get("quantity")) < 0:
            raise ValueError(f"{prefix} line item #{item_index + 1} has an invalid quantity.")
        if not _is_decimal(item.get("unit_price")) or dec(item.get("unit_price")) < 0:
            raise ValueError(f"{prefix} line item #{item_index + 1} has an invalid unit price.")
    if not isinstance(invoice.get("totals"), dict):
        raise ValueError(f"{prefix} has invalid totals.")
    for key in ("subtotal", "tax_amount", "total"):
        if not _is_decimal(invoice["totals"].get(key)):
            raise ValueError(f"{prefix} has an invalid {key}.")
        if dec(invoice["totals"].get(key)) < 0:
            raise ValueError(f"{prefix} has a negative {key}.")
    for key in ("issue_date", "due_date"):
        if invoice.get(key) and _parse_iso_date(invoice.get(key)) is None:
            raise ValueError(f"{prefix} has an invalid {key}.")


def _validate_backup_expense(expense: dict, index: int) -> None:
    prefix = f"Expense #{index + 1}"
    if not isinstance(expense, dict):
        raise ValueError(f"{prefix} is not an object.")
    if not isinstance(expense.get("id"), int) or expense["id"] < 1:
        raise ValueError(f"{prefix} has an invalid id.")
    if not str(expense.get("description") or "").strip():
        raise ValueError(f"{prefix} has no description.")
    if dec(expense.get("amount")) <= 0:
        raise ValueError(f"{prefix} has an invalid amount.")
    if expense.get("category") not in EXPENSE_CATEGORIES:
        raise ValueError(f"{prefix} has an invalid category.")
    if expense.get("date_incurred") and _parse_iso_date(expense.get("date_incurred")) is None:
        raise ValueError(f"{prefix} has an invalid date_incurred.")


def _validate_backup_budget(budget: dict, index: int) -> None:
    prefix = f"Budget #{index + 1}"
    if not isinstance(budget, dict):
        raise ValueError(f"{prefix} is not an object.")
    if not isinstance(budget.get("id"), int) or budget["id"] < 1:
        raise ValueError(f"{prefix} has an invalid id.")
    if not str(budget.get("name") or "").strip():
        raise ValueError(f"{prefix} has no name.")
    if not _is_decimal(budget.get("amount")) or dec(budget.get("amount")) <= 0:
        raise ValueError(f"{prefix} has an invalid amount.")
    category = str(budget.get("category") or "")
    if category and category not in EXPENSE_CATEGORIES:
        raise ValueError(f"{prefix} has an unknown category.")
    month = str(budget.get("month") or "")
    if month and not MONTH_RE.match(month):
        raise ValueError(f"{prefix} has an invalid month.")


def _validate_backup_rule(rule: dict, index: int) -> None:
    prefix = f"Rule #{index + 1}"
    if not isinstance(rule, dict):
        raise ValueError(f"{prefix} is not an object.")
    if not isinstance(rule.get("id"), int) or rule["id"] < 1:
        raise ValueError(f"{prefix} has an invalid id.")
    if not str(rule.get("name") or "").strip():
        raise ValueError(f"{prefix} has no name.")
    if not str(rule.get("vendor") or "").strip() and not str(rule.get("keyword") or "").strip():
        raise ValueError(f"{prefix} needs a vendor or a keyword to match.")
    if rule.get("category") not in EXPENSE_CATEGORIES:
        raise ValueError(f"{prefix} has an unknown category.")


def validate_backup(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Backup must be a JSON object.")
    # Backups written by the previous Accounting Agent release are still accepted.
    if payload.get("format") not in ("finance-agent-backup", "accounting-agent-backup"):
        raise ValueError("This file is not a Finance Agent backup.")
    if payload.get("version") != 1:
        raise ValueError("Unsupported backup version.")
    invoices = payload.get("invoices")
    expenses = payload.get("expenses")
    if not isinstance(invoices, list) or not isinstance(expenses, list):
        raise ValueError("Backup must contain invoice and expense lists.")
    for index, invoice in enumerate(invoices):
        _validate_backup_invoice(invoice, index)
    for index, expense in enumerate(expenses):
        _validate_backup_expense(expense, index)
    invoice_ids = [invoice.get("id") for invoice in invoices]
    if len(set(invoice_ids)) != len(invoice_ids):
        raise ValueError("Backup contains duplicate invoice ids.")
    expense_ids = [expense.get("id") for expense in expenses]
    if len(set(expense_ids)) != len(expense_ids):
        raise ValueError("Backup contains duplicate expense ids.")
    if "settings" in payload:
        if not isinstance(payload["settings"], dict):
            raise ValueError("Backup settings must be an object.")
        try:
            _validate_settings_dict(payload["settings"])
        except ValueError as error:
            raise ValueError(f"Backup settings are invalid: {error}") from error
    if "activity" in payload and not isinstance(payload["activity"], list):
        raise ValueError("Backup activity must be a list.")
    for key, validator in (("budgets", _validate_backup_budget), ("rules", _validate_backup_rule)):
        if key in payload:
            records = payload[key]
            if not isinstance(records, list):
                raise ValueError(f"Backup {key} must be a list.")
            for index, record in enumerate(records):
                validator(record, index)
    return payload


def restore_backup(payload) -> dict:
    backup = validate_backup(payload)
    backup_before("before-restore")
    _write("invoices", backup["invoices"])
    _write("expenses", backup["expenses"])
    if isinstance(backup.get("settings"), dict):
        try:
            save_settings(backup["settings"])
        except ValueError:
            pass
    if isinstance(backup.get("activity"), list):
        _write("activity", backup["activity"][-500:])
    if isinstance(backup.get("budgets"), list):
        _write("budgets", backup["budgets"])
    if isinstance(backup.get("rules"), list):
        _write("rules", backup["rules"])
    log_activity(
        "restored", "backup", None, str(backup.get("created_at") or ""),
        f"Restored {len(backup['invoices'])} invoices and {len(backup['expenses'])} expenses.",
    )
    return {
        "ok": True,
        "invoices": len(backup["invoices"]),
        "expenses": len(backup["expenses"]),
        "budgets": len(backup.get("budgets") or []),
        "rules": len(backup.get("rules") or []),
        "created_at": backup.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Domain calculations (Decimal)
# ---------------------------------------------------------------------------

def normalize_line_items(line_items: list[dict]) -> list[dict]:
    normalized = []
    for item in line_items:
        normalized.append({
            "description": str(item.get("description") or "").strip(),
            "quantity": s4(item.get("quantity", 1)),
            "unit_price": s2(item.get("unit_price", 0)),
        })
    return normalized


def calc_totals(line_items: list[dict], tax_rate=Decimal("0.20")) -> dict:
    """Line amounts and tax are rounded to cents, then summed deterministically."""
    subtotal = ZERO
    for item in line_items:
        amount = q2(dec(item.get("quantity", 1)) * dec(item.get("unit_price", 0)))
        subtotal += amount
    rate = dec(tax_rate)
    tax_amount = q2(subtotal * rate)
    total = subtotal + tax_amount
    return {
        "subtotal": s2(subtotal),
        "discount_pct": "0.00",
        "tax_rate": s4(rate),
        "tax_amount": s2(tax_amount),
        "total": s2(total),
    }


def invoice_ref(invoice_id: int) -> str:
    return f"INV-{int(invoice_id):04d}"


def _paginate(items: list, page: int, page_size: int) -> dict:
    total = len(items)
    pages = (total + page_size - 1) // page_size if total else 0
    start = (page - 1) * page_size
    return {
        "items": items[start:start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


def _filter_invoices(items: list[dict], status: Optional[str], search: Optional[str]) -> list[dict]:
    if status:
        items = [item for item in items if item.get("status") == status]
    if search:
        query = search.strip().lower()
        if query:
            def matches(invoice: dict) -> bool:
                haystack = " ".join([
                    invoice_ref(invoice.get("id", 0)),
                    str(invoice.get("id", "")),
                    str(invoice.get("client_name") or ""),
                    str(invoice.get("client_email") or ""),
                    str(invoice.get("status") or ""),
                ]).lower()
                return query in haystack
            items = [invoice for invoice in items if matches(invoice)]
    return sorted(items, key=lambda item: item.get("id", 0), reverse=True)


def _filter_expenses(items: list[dict], category: Optional[str], search: Optional[str]) -> list[dict]:
    if category:
        items = [item for item in items if item.get("category") == category]
    if search:
        query = search.strip().lower()
        if query:
            def matches(expense: dict) -> bool:
                haystack = " ".join([
                    str(expense.get("description") or ""),
                    str(expense.get("vendor") or ""),
                    str(expense.get("category") or ""),
                    s2(expense.get("amount")),
                    str(expense.get("date_incurred") or ""),
                    str(expense.get("notes") or ""),
                ]).lower()
                return query in haystack
            items = [expense for expense in items if matches(expense)]
    return sorted(items, key=lambda item: item.get("id", 0), reverse=True)


# ---------------------------------------------------------------------------
# Gemini assistant
# ---------------------------------------------------------------------------

class GeminiError(RuntimeError):
    """A safe, user-facing Gemini failure."""


def is_configured() -> bool:
    return bool(GEMINI_API_KEY)


def build_ledger_snapshot() -> str:
    """Return a compact, read-only slice of the ledger for the model."""
    settings = load_settings()
    invoices = all_items("invoices")
    expenses = all_items("expenses")
    issued = issued_invoices(invoices)
    window = spendwise.resolve_period("month", None, None)

    def by_status(status: str) -> list[dict]:
        return [invoice for invoice in invoices if invoice.get("status") == status]

    status_counts = {status: len(by_status(status)) for status in INVOICE_STATUS}
    status_values = {status: s2(sum((invoice_total(i) for i in by_status(status)), ZERO))
                     for status in status_counts}
    categories: dict[str, Decimal] = {}
    for expense in expenses:
        key = expense.get("category") or "other"
        categories[key] = categories.get(key, ZERO) + expense_amount(expense)

    recent_invoices = sorted(invoices, key=lambda item: item.get("id", 0), reverse=True)[:12]
    recent_expenses = sorted(
        expenses,
        key=lambda item: (item.get("date_incurred") or "", item.get("id", 0)),
        reverse=True,
    )[:20]
    overdue = [invoice for invoice in invoices if invoice.get("status") == "overdue"][:20]
    unreimbursed = [
        expense for expense in expenses
        if expense.get("reimbursable") and not expense.get("reimbursed")
    ][:20]

    def invoice_row(invoice: dict) -> dict:
        return {
            "reference": invoice_ref(invoice.get("id", 0)),
            "client": invoice.get("client_name"),
            "issue_date": invoice.get("issue_date"),
            "due_date": invoice.get("due_date"),
            "status": invoice.get("status"),
            "total": s2(invoice_total(invoice)),
        }

    def expense_row(expense: dict) -> dict:
        return {
            "id": expense.get("id"),
            "date": expense.get("date_incurred"),
            "description": expense.get("description"),
            "vendor": expense.get("vendor"),
            "category": expense.get("category"),
            "amount": s2(expense_amount(expense)),
            "reimbursable": bool(expense.get("reimbursable")),
            "reimbursed": bool(expense.get("reimbursed")),
        }

    snapshot = {
        "today": date.today().isoformat(),
        "currency": settings["currency_code"],
        "reporting_basis": REPORTING_BASIS,
        "invoice_totals": {            "count": len(invoices),
            "issued_count": len(issued),
            "issued_value_gross": s2(sum((invoice_total(i) for i in issued), ZERO)),
            "net_revenue": s2(sum((invoice_subtotal(i) for i in issued), ZERO)),
            "tax_invoiced": s2(sum((invoice_tax(i) for i in issued), ZERO)),
            "collected_value": s2(sum((invoice_total(i) for i in invoices if i.get("status") == "paid"), ZERO)),
            "status_counts": status_counts,
            "value_by_status": status_values,
        },
        "expense_totals": {
            "count": len(expenses),
            "recorded": s2(sum((expense_amount(e) for e in expenses), ZERO)),
            "by_category": {key: s2(value) for key, value in categories.items()},
            "awaiting_reimbursement": s2(sum((expense_amount(e) for e in unreimbursed), ZERO)),
        },
        "spendwise": {
            "period": window,
            "stats": spendwise.spending_stats(expenses, window),
            "categories": spendwise.category_breakdown(expenses, window),
            "vendors": spendwise.vendor_breakdown(expenses, window, limit=8),
            "budgets": spendwise.budget_status(all_items("budgets"), expenses, date.today()),
            "insights": spendwise.build_insights(
                expenses, invoices, window, all_items("budgets"), all_items("rules")
            ),
            "cash_flow": spendwise.cash_flow(invoices, expenses, window),
        },
        "recent_invoices": [invoice_row(invoice) for invoice in recent_invoices],
        "overdue_invoices": [invoice_row(invoice) for invoice in overdue],
        "recent_expenses": [expense_row(expense) for expense in recent_expenses],
        "unreimbursed_expenses": [expense_row(expense) for expense in unreimbursed],
        "documents": document_rows(all_items(DOCUMENT_LIBRARY)),
    }
    return json.dumps(snapshot, indent=2, default=_json_default)


SYSTEM_PROMPT = """You are the Finance Agent assistant, embedded in a small business finance application.

Use only the ledger snapshot provided below. It is read-only data and is the source of truth.
- Answer in plain, concise business English. Amounts are in {currency} and follow an accrual basis.
- Revenue, net revenue and profit exclude sales tax. Gross issued value includes tax. Do not add tax to revenue.
- The SpendWise section holds calculated spending analytics: period totals, category and vendor
  breakdowns, budget status, insights, and a cash-movement view. Quote those figures directly.
- The cash-movement figures are not accrual profit. Keep the two apart when comparing them.
- The application calculates all financial figures deterministically. Do not recalculate, override,
  or replace those figures with your own estimates. Explain the figures that are present.
- Refer to invoices by their INV-#### reference and expenses by description/date.
- Summarise, explain, or compare the data when asked.
- You may point out unusual or notable transactions, but explain why they stand out.
- Recurring-charge suggestions are patterns, not confirmed subscriptions. Say so.
- The documents section lists uploaded files with their detected type and any linked ledger record.
  Extracted figures were read by the AI from the document and confirmed by the user before import;
  treat the linked ledger record, not the raw document, as the accounting source of truth.
- Never invent records, totals, clients, dates or tax rules. If the snapshot does not contain enough
  information, say so and suggest which page of the app to check. If a figure is missing, say it is missing.
- Do not give legal, tax, or investment advice. You can explain what the ledger shows.
- Do not claim tax compliance for any jurisdiction.
- Keep responses under 200 words unless the user asks for a detailed breakdown.

LEDGER SNAPSHOT
{snapshot}
"""


def _sanitize_answer(text: str) -> str:
    text = str(text or "").replace(GEMINI_API_KEY, "[redacted]") if GEMINI_API_KEY else str(text or "")
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[redacted]", text)
    return text.strip()[:ASSISTANT_MAX_ANSWER]


def _request_gemini(question: str) -> str:
    settings = load_settings()
    return _request_gemini_text(
        SYSTEM_PROMPT.format(snapshot=build_ledger_snapshot(), currency=settings["currency_code"]),
        [{"text": question}],
    )


def _request_gemini_text(system_prompt: str, user_parts: list[dict],
                          max_output_tokens: int = 700,
                          temperature: float = 0.25) -> str:
    """Call Gemini with text and optional inline media parts.

    `user_parts` entries are Gemini content parts, for example
    `{"text": "..."}` or `{"inline_data": {"mime_type": "image/png", "data": "<base64>"}}`.
    """
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": user_parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    request = urllib.request.Request(
        GEMINI_ENDPOINT.format(model=GEMINI_MODEL),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=GEMINI_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise GeminiError(
                "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
            ) from error
        if error.code == 429:
            raise GeminiError("Gemini is rate limited right now. Please try again shortly.") from error
        if error.code == 404:
            raise GeminiError(
                f"Gemini model \u201c{GEMINI_MODEL}\u201d was not found. Check GEMINI_MODEL in your .env file."
            ) from error
        raise GeminiError(f"Gemini request failed (HTTP {error.code}). Try again later.") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise GeminiError("Gemini could not be reached. Check your connection and try again.") from error
    except (ValueError, KeyError) as error:
        raise GeminiError("Gemini returned an unreadable response. Try again later.") from error

    try:
        candidates = payload.get("candidates") or []
        parts = candidates[0]["content"]["parts"]
        text = "\n".join(part.get("text", "") for part in parts).strip()
    except (IndexError, KeyError, TypeError) as error:
        raise GeminiError("Gemini returned an empty response. Try again.") from error
    if not text:
        raise GeminiError("Gemini returned an empty response. Try again.")
    return _sanitize_answer(text)


def _request_gemini_json(system_prompt: str, user_parts: list[dict],
                         max_output_tokens: int = 1400,
                         temperature: float = 0.1) -> dict:
    """Call Gemini and parse a JSON object out of the reply."""
    reply = _request_gemini_text(
        system_prompt,
        user_parts,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    cleaned = reply.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise GeminiError("Gemini returned a response that is not valid JSON. Try again.")
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as error:
        raise GeminiError("Gemini returned a response that is not valid JSON. Try again.") from error
    if not isinstance(parsed, dict):
        raise GeminiError("Gemini returned an unexpected JSON shape. Try again.")
    return parsed


async def ask_gemini(question: str) -> str:
    """Ask Gemini without blocking the FastAPI event loop."""
    if not is_configured():
        raise GeminiError("Gemini is not configured.")
    return await asyncio.to_thread(_request_gemini, question)


_ASSISTANT_CALLS: dict[str, deque] = {}
_ASSISTANT_LOCK = threading.Lock()


def check_rate_limit(client_key: str) -> None:
    now = time.monotonic()
    with _ASSISTANT_LOCK:
        calls = _ASSISTANT_CALLS.setdefault(client_key, deque())
        while calls and now - calls[0] > ASSISTANT_WINDOW_SECONDS:
            calls.popleft()
        if len(calls) >= ASSISTANT_MAX_REQUESTS:
            raise HTTPException(
                429,
                "Too many assistant requests. Wait a minute before trying again.",
            )
        calls.append(now)


# ---------------------------------------------------------------------------
# Request models and validation
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1, max_length=300)
    quantity: Decimal = Field(default=Decimal("1"), gt=0, le=Decimal("1000000"))
    unit_price: Decimal = Field(default=Decimal("0"), ge=0, le=MONEY_MAX)

    @field_validator("description")
    @classmethod
    def description_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Line item description is required.")
        return value

    @field_validator("quantity")
    @classmethod
    def quantity_precision(cls, value: Decimal) -> Decimal:
        if value != q4(value):
            raise ValueError("Quantity may have at most four decimal places.")
        return q4(value)

    @field_validator("unit_price")
    @classmethod
    def price_precision(cls, value: Decimal) -> Decimal:
        if value != q2(value):
            raise ValueError("Unit price may have at most two decimal places.")
        return q2(value)


class InvoiceCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    client_name: str = Field(min_length=1, max_length=200)
    client_email: str = Field(default="", max_length=254)
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    line_items: list[LineItem] = Field(min_length=1, max_length=200)
    tax_rate: Optional[Decimal] = Field(default=None, ge=0, le=1)
    notes: str = Field(default="", max_length=5000)
    status: Literal["draft", "sent", "paid", "overdue", "cancelled"] = "draft"

    @field_validator("client_name")
    @classmethod
    def client_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Client name is required.")
        return value

    @field_validator("client_email")
    @classmethod
    def email_valid(cls, value: str) -> str:
        value = value.strip()
        if value and not EMAIL_RE.match(value):
            raise ValueError("Client email is not a valid email address.")
        return value

    @field_validator("tax_rate")
    @classmethod
    def tax_precision(cls, value: Decimal) -> Decimal:
        if value is None:
            return value
        if value != q4(value):
            raise ValueError("Tax rate may have at most four decimal places.")
        return q4(value)

    @model_validator(mode="after")
    def dates_in_order(self):
        if self.issue_date and self.due_date and self.due_date < self.issue_date:
            raise ValueError("Due date cannot be before the issue date.")
        return self


class InvoiceUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    client_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    client_email: Optional[str] = Field(default=None, max_length=254)
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    line_items: Optional[list[LineItem]] = Field(default=None, min_length=1, max_length=200)
    tax_rate: Optional[Decimal] = Field(default=None, ge=0, le=1)
    notes: Optional[str] = Field(default=None, max_length=5000)
    status: Optional[Literal["draft", "sent", "paid", "overdue", "cancelled"]] = None

    @field_validator("client_name")
    @classmethod
    def client_required(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Client name cannot be empty.")
        return value

    @field_validator("client_email")
    @classmethod
    def email_valid(cls, value):
        if value is None:
            return value
        value = value.strip()
        if value and not EMAIL_RE.match(value):
            raise ValueError("Client email is not a valid email address.")
        return value

    @field_validator("tax_rate")
    @classmethod
    def tax_precision(cls, value):
        if value is None:
            return value
        if value != q4(value):
            raise ValueError("Tax rate may have at most four decimal places.")
        return q4(value)


class InvoiceBulkAction(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=1000)
    action: Literal["delete", "status"]
    status: Optional[Literal["draft", "sent", "paid", "overdue", "cancelled"]] = None

    @model_validator(mode="after")
    def status_required(self):
        if self.action == "status" and self.status is None:
            raise ValueError("A status is required for a status update.")
        return self


class ExpenseCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1, max_length=300)
    amount: Decimal = Field(gt=0, le=MONEY_MAX)
    category: Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ] = "other"
    date_incurred: Optional[date] = None
    vendor: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=5000)
    reimbursable: bool = False
    reimbursed: bool = False

    @field_validator("description")
    @classmethod
    def description_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Expense description is required.")
        return value

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, value: Decimal) -> Decimal:
        if value != q2(value):
            raise ValueError("Amount may have at most two decimal places.")
        return q2(value)

    @model_validator(mode="after")
    def reimbursement_consistent(self):
        if self.reimbursed and not self.reimbursable:
            raise ValueError("An expense must be reimbursable before it can be marked reimbursed.")
        return self


class ExpenseUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    description: Optional[str] = Field(default=None, min_length=1, max_length=300)
    amount: Optional[Decimal] = Field(default=None, gt=0, le=MONEY_MAX)
    category: Optional[Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ]] = None
    date_incurred: Optional[date] = None
    vendor: Optional[str] = Field(default=None, max_length=200)
    notes: Optional[str] = Field(default=None, max_length=5000)
    reimbursable: Optional[bool] = None
    reimbursed: Optional[bool] = None

    @field_validator("description")
    @classmethod
    def description_required(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Expense description cannot be empty.")
        return value

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, value):
        if value is None:
            return value
        if value != q2(value):
            raise ValueError("Amount may have at most two decimal places.")
        return q2(value)


class ExpenseBulkAction(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=1000)
    action: Literal["delete", "reimbursement"]
    reimbursed: Optional[bool] = None

    @model_validator(mode="after")
    def reimbursement_required(self):
        if self.action == "reimbursement" and self.reimbursed is None:
            raise ValueError("A reimbursement value is required.")
        return self


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=ASSISTANT_MAX_MESSAGE)

    @field_validator("message")
    @classmethod
    def message_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a question for the assistant.")
        return value


class SettingsUpdate(BaseModel):
    currency_code: Optional[str] = Field(default=None, min_length=3, max_length=3)
    currency_symbol: Optional[str] = Field(default=None, min_length=1, max_length=4)
    locale: Optional[str] = Field(default=None, max_length=10)
    default_tax_rate: Optional[Decimal] = Field(default=None, ge=0, le=1)
    date_format: Optional[Literal["short", "medium", "long", "iso"]] = None
    business_name: Optional[str] = Field(default=None, max_length=200)
    business_address: Optional[str] = Field(default=None, max_length=500)
    business_email: Optional[str] = Field(default=None, max_length=254)
    payment_details: Optional[str] = Field(default=None, max_length=500)

    @field_validator("currency_code")
    @classmethod
    def currency_valid(cls, value):
        if value is None:
            return value
        value = value.strip().upper()
        if not CURRENCY_RE.match(value):
            raise ValueError("Currency code must be three letters, such as USD or EUR.")
        return value

    @field_validator("locale")
    @classmethod
    def locale_valid(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not LOCALE_RE.match(value):
            raise ValueError("Locale must look like en-US, en-GB or en-IE.")
        return value

    @field_validator("business_email")
    @classmethod
    def business_email_valid(cls, value):
        if value is None:
            return value
        value = value.strip()
        if value and not EMAIL_RE.match(value):
            raise ValueError("Business email is not a valid email address.")
        return value


class CsvImportRequest(BaseModel):
    csv_text: str = Field(min_length=1)
    allow_duplicates: bool = False


# ---------------------------------------------------------------------------
# SpendWise models
# ---------------------------------------------------------------------------

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class BudgetCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    amount: Decimal = Field(gt=0, le=MONEY_MAX)
    category: Optional[Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ]] = None
    month: str = Field(default="", max_length=7)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def name_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Budget name is required.")
        return value

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, value: Decimal) -> Decimal:
        if value != q2(value):
            raise ValueError("Budget amount may have at most two decimal places.")
        return q2(value)

    @field_validator("month")
    @classmethod
    def month_valid(cls, value: str) -> str:
        value = (value or "").strip()
        if value and not MONTH_RE.match(value):
            raise ValueError("Month must look like 2026-11.")
        return value


class BudgetUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    amount: Optional[Decimal] = Field(default=None, gt=0, le=MONEY_MAX)
    category: Optional[Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ]] = None
    month: Optional[str] = Field(default=None, max_length=7)
    enabled: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def name_required(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Budget name cannot be empty.")
        return value

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, value):
        if value is None:
            return value
        if value != q2(value):
            raise ValueError("Budget amount may have at most two decimal places.")
        return q2(value)

    @field_validator("month")
    @classmethod
    def month_valid(cls, value):
        if value is None:
            return value
        value = (value or "").strip()
        if value and not MONTH_RE.match(value):
            raise ValueError("Month must look like 2026-11.")
        return value


class RuleApplyRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ids: list[int] = Field(default_factory=list, max_length=1000)


class RuleCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    vendor: str = Field(default="", max_length=200)
    keyword: str = Field(default="", max_length=200)
    category: Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ] = "other"
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def name_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Rule name is required.")
        return value

    @model_validator(mode="after")
    def match_required(self):
        if not (self.vendor or "").strip() and not (self.keyword or "").strip():
            raise ValueError("A rule needs a vendor or a keyword to match.")
        return self


class RuleUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    vendor: Optional[str] = Field(default=None, max_length=200)
    keyword: Optional[str] = Field(default=None, max_length=200)
    category: Optional[Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ]] = None
    priority: Optional[int] = Field(default=None, ge=0, le=1000)
    enabled: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def name_required(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Rule name cannot be empty.")
        return value

    @model_validator(mode="after")
    def match_required(self):
        vendor = self.vendor if self.vendor is not None else ""
        keyword = self.keyword if self.keyword is not None else ""
        if self.vendor is not None or self.keyword is not None:
            if not vendor.strip() and not keyword.strip():
                raise ValueError("A rule needs a vendor or a keyword to match.")
        return self


# ---------------------------------------------------------------------------
# CSV import helpers
# ---------------------------------------------------------------------------

CSV_MAX_ROWS = 200000
CSV_ALIASES = {
    "client_name": {"clientname", "client", "customer", "customername", "name"},
    "client_email": {"clientemail", "email", "customeremail"},
    "invoice_number": {"invoicenumber", "invoice", "reference", "ref", "number"},
    "issue_date": {"issuedate", "invoicedate", "date"},
    "due_date": {"duedate", "paymentdue", "duedate"},
    "status": {"status", "invoicestatus"},
    "description": {"description", "item", "itemdescription", "lineitem", "service"},
    "quantity": {"quantity", "qty", "hours"},
    "unit_price": {"unitprice", "rate", "price", "amount"},
    "tax_rate": {"taxrate", "tax", "taxpercent", "taxpct"},
    "notes": {"notes", "note", "memo"},
    "vendor": {"vendor", "supplier", "merchant", "payee"},
    "category": {"category", "type", "expensecategory"},
    "amount": {"amount", "total", "cost", "value"},
    "date_incurred": {"dateincurred", "date", "expensedate"},
    "reimbursable": {"reimbursable", "reimbursement", "rebillable"},
    "reimbursed": {"reimbursed", "paidback", "settled"},
}


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _csv_mapping(headers: list[str], fields: tuple[str, ...]) -> dict[str, str]:
    mapping = {}
    for field in fields:
        aliases = CSV_ALIASES.get(field, set())
        for header in headers:
            normalized = _normalize_header(header)
            if normalized == _normalize_header(field) or normalized in aliases:
                mapping[field] = header
                break
    return mapping


def _read_csv(text: str) -> tuple[list[str], list[dict], list[str]]:
    text = text.lstrip("\ufeff")
    try:
        sample = text[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = [header.strip() for header in (reader.fieldnames or []) if header and header.strip()]
    if not headers:
        raise ValueError("CSV file needs a header row.")
    rows = []
    errors = []
    for index, row in enumerate(reader, start=2):
        if index - 1 > CSV_MAX_ROWS:
            errors.append(f"CSV contains more than {CSV_MAX_ROWS} data rows.")
            break
        if not any(str(value or "").strip() for value in row.values()):
            continue
        rows.append({"row": index, "values": row})
    return headers, rows, errors


def _parse_csv_date(value: str, field: str) -> Optional[date]:
    raw = str(value or "").strip()
    if not raw:
        return None
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw[:10], pattern).date()
        except ValueError:
            continue
    raise ValueError(f"{field} must be a date such as 2026-11-14 or 14/11/2026.")


def _parse_csv_decimal(value: str, field: str) -> Decimal:
    raw = str(value or "").strip().replace(",", "").replace(" ", "")
    raw = raw.replace("$", "").replace("€", "").replace("£", "")
    if not raw:
        return ZERO
    try:
        return Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a number.") from error


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_tax_rate(value: str) -> Decimal:
    raw = _parse_csv_decimal(value, "tax rate")
    if raw < 0:
        raise ValueError("tax rate cannot be negative.")
    if raw > 1:
        raw = raw / Decimal("100")
    if raw > 1:
        raise ValueError("tax rate cannot exceed 100 percent.")
    return q4(raw)


def _existing_invoice_keys(invoices: list[dict]) -> set[tuple]:
    return {
        (
            str(invoice.get("client_name") or "").strip().lower(),
            str(invoice.get("issue_date") or ""),
            s2(invoice_total(invoice)),
        )
        for invoice in invoices
    }


def _existing_expense_keys(expenses: list[dict]) -> set[tuple]:
    return {
        (
            str(expense.get("description") or "").strip().lower(),
            str(expense.get("date_incurred") or ""),
            s2(expense_amount(expense)),
        )
        for expense in expenses
    }


def prepare_invoice_import(csv_text: str) -> dict:
    headers, rows, file_errors = _read_csv(csv_text)
    fields = ("client_name", "client_email", "invoice_number", "issue_date", "due_date",
              "status", "description", "quantity", "unit_price", "tax_rate", "notes")
    mapping = _csv_mapping(headers, fields)
    errors = list(file_errors)
    for required in ("client_name", "description", "unit_price"):
        if required not in mapping:
            errors.append(f"Missing required column: {required}. Detected headers: {', '.join(headers)}.")
    groups: dict[str, dict] = {}
    preview_rows = []
    existing = _existing_invoice_keys(all_items("invoices"))

    if "client_name" not in mapping or "description" not in mapping or "unit_price" not in mapping:
        return {
            "kind": "invoices", "headers": headers, "mapping": mapping,
            "rows": preview_rows, "errors": errors, "duplicates": [],
            "summary": {"rows": len(rows), "invoices": 0, "errors": len(errors), "duplicates": 0},
        }

    for row in rows:
        values = row["values"]
        row_errors = []
        try:
            client = str(values.get(mapping["client_name"]) or "").strip()
            if not client:
                raise ValueError("client_name is required.")
            email = str(values.get(mapping.get("client_email", ""), "") or "").strip()
            if email and not EMAIL_RE.match(email):
                raise ValueError("client_email is not valid.")
            issue = _parse_csv_date(values.get(mapping.get("issue_date", ""), ""), "issue_date") or date.today()
            due = _parse_csv_date(values.get(mapping.get("due_date", ""), ""), "due_date")
            if due and due < issue:
                raise ValueError("due_date cannot be before issue_date.")
            status = str(values.get(mapping.get("status", ""), "") or "draft").strip().lower() or "draft"
            if status not in INVOICE_STATUS:
                raise ValueError(f"status must be one of {', '.join(INVOICE_STATUS)}.")
            description = str(values.get(mapping["description"]) or "").strip()
            if not description:
                raise ValueError("description is required.")
            quantity = _parse_csv_decimal(values.get(mapping.get("quantity", ""), ""), "quantity")
            if quantity == 0:
                quantity = Decimal("1")
            if quantity < 0:
                raise ValueError("quantity cannot be negative.")
            unit_price = _parse_csv_decimal(values.get(mapping["unit_price"]), "unit_price")
            if unit_price < 0:
                raise ValueError("unit_price cannot be negative.")
            tax_rate = _parse_tax_rate(values.get(mapping.get("tax_rate", ""), ""))
            number = str(values.get(mapping.get("invoice_number", ""), "") or "").strip()
            key = number.lower() if number else f"row-{row['row']}"
            if key not in groups:
                groups[key] = {
                    "row": row["row"], "client_name": client, "client_email": email,
                    "issue_date": issue.isoformat(), "due_date": due.isoformat() if due else None,
                    "status": status, "notes": str(values.get(mapping.get("notes", ""), "") or "").strip(),
                    "tax_rate": tax_rate, "line_items": [],
                }
            elif groups[key]["client_name"].lower() != client.lower() or groups[key]["issue_date"] != issue.isoformat():
                raise ValueError("rows sharing invoice_number must have the same client and issue date.")
            groups[key]["line_items"].append({
                "description": description, "quantity": s4(quantity), "unit_price": s2(unit_price),
            })
            preview_rows.append({
                "row": row["row"], "client_name": client, "issue_date": issue.isoformat(),
                "description": description, "amount": s2(q2(quantity) * q2(unit_price)),
                "status": status, "errors": row_errors,
            })
        except ValueError as error:
            row_errors.append(str(error))
            preview_rows.append({"row": row["row"], "errors": row_errors})
        if row_errors:
            errors.extend(f"Row {row['row']}: {message}" for message in row_errors)

    prepared = []
    duplicates = []
    for group in groups.values():
        totals = calc_totals(group["line_items"], group["tax_rate"])
        key = (group["client_name"].lower(), group["issue_date"], totals["total"])
        duplicate = key in existing
        if duplicate:
            duplicates.append({
                "client_name": group["client_name"], "issue_date": group["issue_date"],
                "total": totals["total"],
            })
        prepared.append({**group, "totals": totals, "duplicate": duplicate})
    return {
        "kind": "invoices", "headers": headers, "mapping": mapping,
        "rows": preview_rows[:100], "prepared": prepared,
        "errors": errors, "duplicates": duplicates,
        "summary": {
            "rows": len(rows), "invoices": len(prepared),
            "errors": len(errors), "duplicates": len(duplicates),
        },
    }


def prepare_expense_import(csv_text: str) -> dict:
    headers, rows, file_errors = _read_csv(csv_text)
    fields = ("description", "amount", "date_incurred", "vendor", "category",
              "reimbursable", "reimbursed", "notes")
    mapping = _csv_mapping(headers, fields)
    errors = list(file_errors)
    for required in ("description", "amount"):
        if required not in mapping:
            errors.append(f"Missing required column: {required}. Detected headers: {', '.join(headers)}.")
    preview_rows = []
    prepared = []
    existing = _existing_expense_keys(all_items("expenses"))
    rules = all_items("rules")

    if "description" not in mapping or "amount" not in mapping:
        return {
            "kind": "expenses", "headers": headers, "mapping": mapping,
            "rows": preview_rows, "errors": errors, "duplicates": [],
            "summary": {"rows": len(rows), "expenses": 0, "errors": len(errors), "duplicates": 0},
        }

    for row in rows:
        values = row["values"]
        try:
            description = str(values.get(mapping["description"]) or "").strip()
            if not description:
                raise ValueError("description is required.")
            amount = _parse_csv_decimal(values.get(mapping["amount"]), "amount")
            if amount <= 0:
                raise ValueError("amount must be greater than zero.")
            incurred = _parse_csv_date(values.get(mapping.get("date_incurred", ""), ""), "date") or date.today()
            vendor = str(values.get(mapping.get("vendor", ""), "") or "").strip()
            suggestion = spendwise.suggest_category(rules, {
                "vendor": vendor, "description": description,
            })
            raw_category = str(values.get(mapping.get("category", ""), "") or "").strip().lower()
            if raw_category:
                if raw_category not in EXPENSE_CATEGORIES:
                    hint = ""
                    if suggestion:
                        hint = (
                            f" Categorisation rule '{suggestion['rule_name']}' would suggest "
                            f"'{suggestion['category']}'."
                        )
                    raise ValueError(
                        f"category must be one of {', '.join(EXPENSE_CATEGORIES)}.{hint}"
                    )
                category = raw_category
                suggested = ""
                suggested_by = ""
            elif suggestion:
                category = suggestion["category"]
                suggested = suggestion["category"]
                suggested_by = suggestion["rule_name"]
            else:
                category = "other"
                suggested = ""
                suggested_by = ""
            reimbursable = _parse_bool(values.get(mapping.get("reimbursable", ""), ""))
            reimbursed = _parse_bool(values.get(mapping.get("reimbursed", ""), "")) and reimbursable
            item = {
                "description": description, "amount": s2(amount),
                "date_incurred": incurred.isoformat(), "category": category,
                "vendor": vendor,
                "notes": str(values.get(mapping.get("notes", ""), "") or "").strip(),
                "reimbursable": reimbursable, "reimbursed": reimbursed,
                "suggested_category": suggested,
                "suggested_by": suggested_by,
            }
            duplicate = (description.lower(), incurred.isoformat(), s2(amount)) in existing
            prepared.append({**item, "duplicate": duplicate, "row": row["row"]})
            preview_rows.append({**item, "row": row["row"], "duplicate": duplicate, "errors": []})
        except ValueError as error:
            preview_rows.append({"row": row["row"], "errors": [str(error)]})
            errors.append(f"Row {row['row']}: {error}")

    duplicates = [row for row in prepared if row["duplicate"]]
    suggested_count = sum(1 for row in prepared if row.get("suggested_category"))
    return {
        "kind": "expenses", "headers": headers, "mapping": mapping,
        "rows": preview_rows[:100], "prepared": prepared,
        "errors": errors, "duplicates": duplicates,
        "summary": {
            "rows": len(rows), "expenses": len(prepared),
            "errors": len(errors), "duplicates": len(duplicates),
            "suggested": suggested_count,
        },
    }


# ---------------------------------------------------------------------------
# PDF invoice generation (standard library only)
# ---------------------------------------------------------------------------

def _pdf_escape(text: str) -> str:
    safe = str(text or "").replace("\r", " ").replace("\n", " ")
    safe = safe.encode("cp1252", "replace").decode("cp1252")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, size: float, text: str,
              font: str = "F1", color: tuple = (0.1, 0.1, 0.1)) -> str:
    return (
        f"BT /{font} {size:.2f} Tf {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg "
        f"1 0 0 1 {x:.2f} {y:.2f} Tm ({_pdf_escape(text)}) Tj ET"
    )


def _pdf_right(right: float, y: float, size: float, text: str,
               font: str = "F1", color: tuple = (0.1, 0.1, 0.1)) -> str:
    width = len(str(text)) * size * 0.52
    return _pdf_text(right - width, y, size, text, font, color)


def _pdf_rect(x: float, y: float, width: float, height: float,
              color: tuple = (0.1, 0.1, 0.1)) -> str:
    return f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f"


def _pdf_line(x1: float, y1: float, x2: float, y2: float,
              color: tuple = (0.82, 0.82, 0.82), width: float = 0.6) -> str:
    return (
        f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG {width:.2f} w "
        f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
    )


def _pdf_money(settings: dict, value) -> str:
    symbol = str(settings.get("currency_symbol") or "$")
    try:
        symbol.encode("cp1252")
    except UnicodeEncodeError:
        # Standard-library PDF fonts use WinAnsi encoding; fall back to the ISO code.
        symbol = f"{settings.get('currency_code') or ''} ".lstrip()
    return f"{symbol}{s2(value)}"


def _pdf_wrap(text: str, limit: int = 70) -> list[str]:
    words = str(text or "").split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > limit:
            if current:
                lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def build_invoice_pdf(invoice: dict, settings: dict) -> bytes:
    reference = invoice_ref(invoice.get("id", 0))
    client = str(invoice.get("client_name") or "Client")
    email = str(invoice.get("client_email") or "")
    issue = str(invoice.get("issue_date") or "—")
    due = str(invoice.get("due_date") or "—")
    status = str(invoice.get("status") or "draft").title()
    items = invoice.get("line_items") or []
    totals = invoice.get("totals") or {}
    business = settings.get("business_name") or "Finance Agent"
    address = [line for line in str(settings.get("business_address") or "").splitlines() if line.strip()]
    payment = str(settings.get("payment_details") or "").strip()
    notes = str(invoice.get("notes") or "").strip()
    pages: list[list[str]] = []
    rows_per_page = 18
    item_pages = [items[index:index + rows_per_page] for index in range(0, len(items), rows_per_page)] or [[]]

    for page_index, page_items in enumerate(item_pages):
        content: list[str] = [
            _pdf_rect(0, 752, 595, 90, (0.035, 0.035, 0.045)),
            _pdf_rect(0, 752, 6, 90, (0.78, 0.64, 0.30)),
            _pdf_text(50, 808, 9, "ACCOUNTING AGENT", "F2", (0.78, 0.64, 0.30)),
            _pdf_text(50, 779, 26, "INVOICE" if page_index == 0 else "INVOICE · CONTINUED", "F4", (1, 1, 1)),
            _pdf_right(545, 808, 10, reference, "F2", (1, 1, 1)),
            _pdf_right(545, 789, 9, f"Issued {issue}  ·  Due {due}", "F1", (0.85, 0.85, 0.85)),
            _pdf_text(50, 717, 9, "FROM", "F2", (0.45, 0.45, 0.45)),
            _pdf_text(50, 701, 11, business, "F2"),
        ]
        y = 685
        for line in address[:3]:
            content.append(_pdf_text(50, y, 9, line, "F1", (0.28, 0.28, 0.28)))
            y -= 13
        content.extend([
            _pdf_text(320, 717, 9, "BILLED TO", "F2", (0.45, 0.45, 0.45)),
            _pdf_text(320, 701, 11, client, "F2"),
            _pdf_text(320, 686, 9, email or "No email on file", "F1", (0.28, 0.28, 0.28)),
            _pdf_text(320, 667, 9, f"Status: {status}", "F1", (0.28, 0.28, 0.28)),
            _pdf_line(50, 650, 545, 650),
            _pdf_text(50, 630, 8, "DESCRIPTION", "F2", (0.45, 0.45, 0.45)),
            _pdf_right(360, 630, 8, "QTY", "F2", (0.45, 0.45, 0.45)),
            _pdf_right(450, 630, 8, "RATE", "F2", (0.45, 0.45, 0.45)),
            _pdf_right(545, 630, 8, "AMOUNT", "F2", (0.45, 0.45, 0.45)),
            _pdf_line(50, 621, 545, 621),
        ])
        y = 601
        for item in page_items:
            description_lines = _pdf_wrap(item.get("description") or "Item", 52)
            content.append(_pdf_text(50, y, 9, description_lines[0], "F1"))
            for extra in description_lines[1:3]:
                y -= 11
                content.append(_pdf_text(50, y, 9, extra, "F1", (0.35, 0.35, 0.35)))
            quantity = dec(item.get("quantity", 1))
            price = q2(item.get("unit_price", 0))
            content.append(_pdf_text(300, y, 9, format(quantity.normalize(), "f"), "F1", (0.25, 0.25, 0.25)))
            content.append(_pdf_right(450, y, 9, _pdf_money(settings, price), "F1", (0.25, 0.25, 0.25)))
            content.append(_pdf_right(545, y, 9, _pdf_money(settings, q2(quantity * price)), "F1"))
            y -= 21
            content.append(_pdf_line(50, y + 8, 545, y + 8, (0.91, 0.91, 0.91), 0.4))

        if page_index == len(item_pages) - 1:
            y -= 6
            content.extend([
                _pdf_text(330, y, 9, "Subtotal", "F1", (0.35, 0.35, 0.35)),
                _pdf_right(545, y, 9, _pdf_money(settings, totals.get("subtotal")), "F1"),
                _pdf_text(330, y - 17, 9, "Tax", "F1", (0.35, 0.35, 0.35)),
                _pdf_right(545, y - 17, 9, _pdf_money(settings, totals.get("tax_amount")), "F1"),
                _pdf_line(330, y - 27, 545, y - 27),
                _pdf_text(330, y - 44, 11, "TOTAL", "F2"),
                _pdf_right(545, y - 44, 11, _pdf_money(settings, totals.get("total")), "F2"),
            ])
            footer_y = y - 78
            if notes:
                content.append(_pdf_text(50, footer_y, 8, "NOTES", "F2", (0.45, 0.45, 0.45)))
                content.append(_pdf_text(50, footer_y - 14, 9, notes[:160], "F1", (0.3, 0.3, 0.3)))
            if payment:
                content.append(_pdf_text(50, footer_y - 40, 8, "PAYMENT DETAILS", "F2", (0.45, 0.45, 0.45)))
                content.append(_pdf_text(50, footer_y - 54, 9, payment[:160], "F1", (0.3, 0.3, 0.3)))
            content.append(_pdf_text(50, 38, 8, "Generated by Finance Agent · Figures calculated by the application", "F1", (0.5, 0.5, 0.5)))
            content.append(_pdf_right(545, 38, 8, f"Page {page_index + 1} of {len(item_pages)}", "F1", (0.5, 0.5, 0.5)))
        else:
            content.append(_pdf_text(50, 38, 8, "Continued on next page", "F1", (0.5, 0.5, 0.5)))
        pages.append(content)

    objects: list[Optional[str]] = [None]

    def add_object(value: str) -> int:
        objects.append(value)
        return len(objects) - 1

    font_regular = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_bold = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    font_serif = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /WinAnsiEncoding >>")
    font_serif_bold = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold /Encoding /WinAnsiEncoding >>")
    pages_id = add_object("")
    page_ids = []
    for content_lines in pages:
        stream = "\n".join(content_lines)
        stream_bytes = stream.encode("cp1252", "replace")
        content_id = add_object(f"<< /Length {len(stream_bytes)} >>\nstream\n{stream}\nendstream")
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R "
            f"/F3 {font_serif} 0 R /F4 {font_serif_bold} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)
    objects[pages_id] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
        f"/Count {len(page_ids)} >>"
    )
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, value in enumerate(objects[1:], start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("cp1252"))
        output.extend(str(value or "").encode("cp1252", "replace"))
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)}\n".encode("cp1252"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("cp1252"))
    output.extend(
        f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode("cp1252")
    )
    return bytes(output)


# ---------------------------------------------------------------------------
# Application and exception handlers
# ---------------------------------------------------------------------------

app = FastAPI(title="Finance Agent", version=APP_VERSION)


@app.exception_handler(StorageError)
async def storage_error_handler(request: Request, exc: StorageError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = error.get("msg", "Invalid value")
        details.append(f"{location}: {message}" if location else message)
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/api/health")
def health():
    return {"ok": True, "version": APP_VERSION, "reporting_basis": "accrual"}


# ---------------------------------------------------------------------------
# Invoice API
# ---------------------------------------------------------------------------

@app.get("/api/invoices")
def list_invoices(
    status: Optional[Literal["draft", "sent", "paid", "overdue", "cancelled"]] = None,
    search: Optional[str] = Query(None, max_length=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    items = _filter_invoices(all_items("invoices"), status, search)
    return _paginate(items, page, page_size)


@app.get("/api/invoices/summary")
def invoice_summary():
    items = all_items("invoices")
    amounts = {status: sum((invoice_total(i) for i in items if i.get("status") == status), ZERO)
               for status in INVOICE_STATUS}
    return {
        "total": len(items),
        "draft": sum(1 for item in items if item.get("status") == "draft"),
        "sent": sum(1 for item in items if item.get("status") == "sent"),
        "paid": sum(1 for item in items if item.get("status") == "paid"),
        "overdue": sum(1 for item in items if item.get("status") == "overdue"),
        "cancelled": sum(1 for item in items if item.get("status") == "cancelled"),
        "total_sent_amount": s2(amounts["sent"]),
        "total_paid_amount": s2(amounts["paid"]),
        "total_overdue_amount": s2(amounts["overdue"]),
        "total_draft_amount": s2(amounts["draft"]),
        "total_cancelled_amount": s2(amounts["cancelled"]),
        "reporting_basis": "accrual",
    }


@app.get("/api/invoices/ids")
def invoice_ids(
    status: Optional[Literal["draft", "sent", "paid", "overdue", "cancelled"]] = None,
    search: Optional[str] = Query(None, max_length=100),
):
    items = _filter_invoices(all_items("invoices"), status, search)
    return {"ids": [item["id"] for item in items], "total": len(items)}


@app.post("/api/invoices/bulk")
def bulk_invoices(data: InvoiceBulkAction):
    ids = sorted(set(data.ids))
    existing = {item.get("id") for item in all_items("invoices")}
    missing = [item_id for item_id in ids if item_id not in existing]
    if missing:
        raise HTTPException(404, f"Invoice not found: {', '.join(str(i) for i in missing)}")
    backup_before(f"bulk-invoice-{data.action}")

    if data.action == "delete":
        deleted = delete_items("invoices", ids)
        log_activity("bulk-deleted", "invoice", None, f"{deleted} invoices", f"Bulk delete of {deleted} invoices.")
        return {"ok": True, "deleted": deleted}

    updated = update_items("invoices", ids, {"status": data.status})
    log_activity("bulk-status", "invoice", None, f"{updated} invoices", f"Status set to {data.status}.")
    return {"ok": True, "updated": updated, "status": data.status}


@app.get("/api/invoices/{invoice_id}")
def get_invoice(invoice_id: int):
    item = get_item("invoices", invoice_id)
    if not item:
        raise HTTPException(404, "Invoice not found")
    return item


@app.post("/api/invoices", status_code=201)
def create_invoice(data: InvoiceCreate):
    items_list = normalize_line_items([item.model_dump() for item in data.line_items])
    rate = data.tax_rate if data.tax_rate is not None else dec(load_settings()["default_tax_rate"])
    payload = {
        "client_name": data.client_name,
        "client_email": data.client_email,
        "issue_date": (data.issue_date or date.today()).isoformat(),
        "due_date": data.due_date.isoformat() if data.due_date else None,
        "line_items": items_list,
        "tax_rate": s4(rate),
        "totals": calc_totals(items_list, rate),
        "notes": data.notes,
        "status": data.status,
    }
    created = create_item("invoices", payload)
    log_activity("created", "invoice", created["id"], invoice_ref(created["id"]),
                 f"{created['client_name']} · {s2(invoice_total(created))} · {created['status']}")
    return created


@app.put("/api/invoices/{invoice_id}")
def update_invoice(invoice_id: int, data: InvoiceUpdate):
    existing = get_item("invoices", invoice_id)
    if not existing:
        raise HTTPException(404, "Invoice not found")
    updates = data.model_dump(exclude_unset=True)
    if "line_items" in updates and data.line_items is None:
        raise HTTPException(422, "Line items cannot be empty.")
    payload = {}
    for field in ("client_name", "client_email", "notes", "status"):
        if field in updates:
            payload[field] = updates[field]
    if "issue_date" in updates:
        payload["issue_date"] = data.issue_date.isoformat() if data.issue_date else None
    if "due_date" in updates:
        payload["due_date"] = data.due_date.isoformat() if data.due_date else None
    if "tax_rate" in updates:
        payload["tax_rate"] = s4(data.tax_rate)

    merged_issue = data.issue_date.isoformat() if data.issue_date else existing.get("issue_date")
    merged_due = (data.due_date.isoformat() if data.due_date else None) if "due_date" in updates else existing.get("due_date")
    if merged_issue and merged_due:
        issue_date, due_date = _parse_iso_date(merged_issue), _parse_iso_date(merged_due)
        if issue_date and due_date and due_date < issue_date:
            raise HTTPException(422, "Due date cannot be before the issue date.")
    if not payload.get("issue_date", existing.get("issue_date")):
        raise HTTPException(422, "Issue date is required.")

    recalc = "line_items" in updates or "tax_rate" in updates
    if "line_items" in updates:
        payload["line_items"] = normalize_line_items([item.model_dump() for item in data.line_items])
    if recalc:
        items_list = payload.get("line_items", existing.get("line_items", []))
        rate = dec(payload.get("tax_rate", existing.get("tax_rate", "0.20")))
        payload["totals"] = calc_totals(items_list, rate)
    result = update_item("invoices", invoice_id, payload)
    if not result:
        raise HTTPException(404, "Invoice not found")
    log_activity("updated", "invoice", invoice_id, invoice_ref(invoice_id),
                 f"Updated {', '.join(sorted(payload.keys()))}.")
    return result


@app.delete("/api/invoices/{invoice_id}")
def delete_invoice(invoice_id: int):
    item = get_item("invoices", invoice_id)
    if not item:
        raise HTTPException(404, "Invoice not found")
    backup_before("before-invoice-delete")
    if not delete_item("invoices", invoice_id):
        raise HTTPException(404, "Invoice not found")
    log_activity("deleted", "invoice", invoice_id, invoice_ref(invoice_id),
                 f"{item.get('client_name', '')} · {s2(invoice_total(item))}")
    return {"ok": True}


@app.post("/api/invoices/{invoice_id}/mark-paid")
def mark_paid(invoice_id: int):
    result = update_item("invoices", invoice_id, {"status": "paid"})
    if not result:
        raise HTTPException(404, "Invoice not found")
    log_activity("marked-paid", "invoice", invoice_id, invoice_ref(invoice_id),
                 f"{s2(invoice_total(result))} collected.")
    return result


@app.get("/api/invoices/{invoice_id}/pdf")
def invoice_pdf(invoice_id: int):
    invoice = get_item("invoices", invoice_id)
    if not invoice:
        raise HTTPException(404, "Invoice not found")
    settings = load_settings()
    pdf = build_invoice_pdf(invoice, settings)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{invoice_ref(invoice_id)}.pdf"'},
    )


# ---------------------------------------------------------------------------
# Expense API
# ---------------------------------------------------------------------------

@app.get("/api/expenses")
def list_expenses(
    category: Optional[Literal[
        "office", "travel", "utilities", "software", "hardware",
        "food", "transport", "rent", "services", "other",
    ]] = None,
    search: Optional[str] = Query(None, max_length=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    items = _filter_expenses(all_items("expenses"), category, search)
    return _paginate(items, page, page_size)


@app.get("/api/expenses/summary")
def expense_summary():
    items = all_items("expenses")
    total = sum((expense_amount(item) for item in items), ZERO)
    by_category: dict[str, Decimal] = {}
    for item in items:
        category = item.get("category", "other")
        by_category[category] = by_category.get(category, ZERO) + expense_amount(item)
    return {
        "total": s2(total),
        "count": len(items),
        "by_category": {key: s2(value) for key, value in by_category.items()},
        "reimbursable": sum(1 for item in items if item.get("reimbursable")),
        "unreimbursed": sum(
            1 for item in items if item.get("reimbursable") and not item.get("reimbursed")
        ),
        "unreimbursed_total": s2(sum(
            (expense_amount(item) for item in items
             if item.get("reimbursable") and not item.get("reimbursed")), ZERO
        )),
    }


@app.get("/api/expenses/categories")
def get_categories():
    return EXPENSE_CATEGORIES


@app.get("/api/expenses/ids")
def expense_ids(
    category: Optional[str] = Query(None, max_length=40),
    search: Optional[str] = Query(None, max_length=100),
):
    items = _filter_expenses(all_items("expenses"), category, search)
    return {"ids": [item["id"] for item in items], "total": len(items)}


@app.post("/api/expenses/bulk")
def bulk_expenses(data: ExpenseBulkAction):
    ids = sorted(set(data.ids))
    items = all_items("expenses")
    existing = {item.get("id") for item in items}
    missing = [item_id for item_id in ids if item_id not in existing]
    if missing:
        raise HTTPException(404, f"Expense not found: {', '.join(str(i) for i in missing)}")
    backup_before(f"bulk-expense-{data.action}")

    if data.action == "delete":
        deleted = delete_items("expenses", ids)
        log_activity("bulk-deleted", "expense", None, f"{deleted} expenses", f"Bulk delete of {deleted} expenses.")
        return {"ok": True, "deleted": deleted}

    reimbursable_ids = [
        item_id for item_id in ids
        if next(item for item in items if item.get("id") == item_id).get("reimbursable")
    ]
    updated = update_items("expenses", reimbursable_ids, {"reimbursed": data.reimbursed})
    log_activity("bulk-reimbursement", "expense", None, f"{updated} expenses",
                 f"Reimbursement set to {'yes' if data.reimbursed else 'no'}.")
    return {
        "ok": True, "updated": updated,
        "skipped": len(ids) - len(reimbursable_ids),
        "reimbursed": data.reimbursed,
    }


@app.get("/api/expenses/{expense_id}")
def get_expense(expense_id: int):
    item = get_item("expenses", expense_id)
    if not item:
        raise HTTPException(404, "Expense not found")
    return item


@app.post("/api/expenses", status_code=201)
def create_expense(data: ExpenseCreate):
    payload = {
        "description": data.description,
        "amount": s2(data.amount),
        "category": data.category,
        "date_incurred": (data.date_incurred or date.today()).isoformat(),
        "vendor": data.vendor,
        "notes": data.notes,
        "reimbursable": data.reimbursable,
        "reimbursed": data.reimbursed,
    }
    created = create_item("expenses", payload)
    log_activity("created", "expense", created["id"], created["description"],
                 f"{s2(expense_amount(created))} · {created['category']}")
    return created


@app.put("/api/expenses/{expense_id}")
def update_expense(expense_id: int, data: ExpenseUpdate):
    existing = get_item("expenses", expense_id)
    if not existing:
        raise HTTPException(404, "Expense not found")
    updates = data.model_dump(exclude_unset=True)
    payload = {}
    for field in ("description", "category", "vendor", "notes"):
        if field in updates:
            payload[field] = updates[field]
    if "amount" in updates:
        payload["amount"] = s2(data.amount)
    if "date_incurred" in updates:
        payload["date_incurred"] = data.date_incurred.isoformat() if data.date_incurred else None
    if "reimbursable" in updates:
        payload["reimbursable"] = data.reimbursable
    if "reimbursed" in updates:
        payload["reimbursed"] = data.reimbursed
    merged = {**existing, **payload}
    if merged.get("reimbursed") and not merged.get("reimbursable"):
        raise HTTPException(422, "An expense must be reimbursable before it can be marked reimbursed.")
    result = update_item("expenses", expense_id, payload)
    if not result:
        raise HTTPException(404, "Expense not found")
    log_activity("updated", "expense", expense_id, result.get("description", ""),
                 f"Updated {', '.join(sorted(payload.keys()))}.")
    return result


@app.delete("/api/expenses/{expense_id}")
def delete_expense(expense_id: int):
    item = get_item("expenses", expense_id)
    if not item:
        raise HTTPException(404, "Expense not found")
    backup_before("before-expense-delete")
    if not delete_item("expenses", expense_id):
        raise HTTPException(404, "Expense not found")
    log_activity("deleted", "expense", expense_id, item.get("description", ""),
                 f"{s2(expense_amount(item))} · {item.get('category', '')}")
    return {"ok": True}


# ---------------------------------------------------------------------------
# SpendWise API
# ---------------------------------------------------------------------------

def _spendwise_period(period: Optional[str], from_date: Optional[str],
                       to_date: Optional[str]) -> dict:
    try:
        return spendwise.resolve_period(period or "month", from_date, to_date)
    except spendwise.SpendWiseError as error:
        raise HTTPException(422, str(error)) from error


def _budgets() -> list[dict]:
    return all_items("budgets")


def _rules() -> list[dict]:
    return all_items("rules")


def _spendwise_context(period: dict) -> dict:
    """Read the ledger once and return everything SpendWise needs."""
    return {
        "expenses": all_items("expenses"),
        "invoices": all_items("invoices"),
        "budgets": _budgets(),
        "rules": _rules(),
    }


@app.get("/api/spendwise/overview")
def spendwise_overview(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    context = _spendwise_context(window)
    expenses = context["expenses"]
    start = _parse_iso_date(window["start"])
    end = _parse_iso_date(window["end"])
    anchor = end or date.today()
    categories = spendwise.category_breakdown(expenses, window)
    vendors = spendwise.vendor_breakdown(expenses, window)
    return {
        "period": window,
        "stats": spendwise.spending_stats(expenses, window),
        "monthly_trend": spendwise.monthly_trend(expenses, start, end),
        "weekly_trend": spendwise.weekly_trend(expenses, start, end),
        "categories": categories,
        "vendors": vendors,
        "category_matrix": spendwise.spending_matrix(
            expenses, start, end, lambda expense: expense.get("category") or "other"
        ),
        "vendor_matrix": spendwise.spending_matrix(expenses, start, end, spendwise.vendor_key),
        "budgets": spendwise.budget_status(context["budgets"], expenses, anchor),
        "budget_totals": spendwise.budget_totals(context["budgets"], expenses, anchor),
        "cash_flow": spendwise.cash_flow(context["invoices"], expenses, window),
        "insights": spendwise.build_insights(
            expenses, context["invoices"], window, context["budgets"], context["rules"]
        ),
        "recurring": spendwise.recurring_suggestions(expenses),
        "unusual": spendwise.unusual_expenses(
            spendwise.filter_expenses(expenses, start, end)
        ),
        "period_options": [{"key": key, "label": label} for key, label in spendwise.PERIODS.items()],
    }


@app.get("/api/spendwise/categories")
def spendwise_categories(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    return {
        "period": window,
        "categories": spendwise.category_breakdown(all_items("expenses"), window),
    }


@app.get("/api/spendwise/categories/{category}")
def spendwise_category_detail(
    category: str,
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    if category not in EXPENSE_CATEGORIES:
        raise HTTPException(404, "Unknown expense category.")
    expenses = all_items("expenses")
    rows = spendwise.category_breakdown(expenses, window)
    detail = next((row for row in rows if row["category"] == category), None)
    start = _parse_iso_date(window["start"])
    end = _parse_iso_date(window["end"])
    return {
        "period": window,
        "category": category,
        "summary": detail,
        "transactions": spendwise.category_transactions(
            spendwise.filter_expenses(expenses, start, end), category
        ),
    }


@app.get("/api/spendwise/vendors")
def spendwise_vendors(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    return {
        "period": window,
        "vendors": spendwise.vendor_breakdown(all_items("expenses"), window),
    }


@app.get("/api/spendwise/insights")
def spendwise_insights(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    context = _spendwise_context(window)
    return {
        "period": window,
        "insights": spendwise.build_insights(
            context["expenses"], context["invoices"], window,
            context["budgets"], context["rules"],
        ),
        "recurring": spendwise.recurring_suggestions(context["expenses"]),
    }


@app.get("/api/spendwise/cash-flow")
def spendwise_cash_flow(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    return spendwise.cash_flow(all_items("invoices"), all_items("expenses"), window)


# --- Budgets ---------------------------------------------------------------

@app.get("/api/spendwise/budgets")
def list_budgets(month: Optional[str] = Query(None)):
    budgets = _budgets()
    anchor = date.today()
    if month:
        parsed = _parse_iso_date(f"{month}-01")
        if parsed is None:
            raise HTTPException(422, "Month must look like 2026-11.")
        anchor = parsed
    return {
        "budgets": budgets,
        "status": spendwise.budget_status(budgets, all_items("expenses"), anchor),
        "totals": spendwise.budget_totals(budgets, all_items("expenses"), anchor),
        "month": anchor.strftime("%Y-%m"),
    }


@app.post("/api/spendwise/budgets", status_code=201)
def create_budget(data: BudgetCreate):
    record = {
        "name": data.name,
        "amount": s2(data.amount),
        "category": data.category or "",
        "month": data.month,
        "enabled": data.enabled,
    }
    created = create_item("budgets", record)
    log_activity("created", "budget", created.get("id"), created["name"],
                 f"{created['amount']} · {created['category'] or 'all spending'}")
    return created


@app.put("/api/spendwise/budgets/{budget_id}")
def update_budget(budget_id: int, data: BudgetUpdate):
    updates = data.model_dump(exclude_none=True)
    if "amount" in updates:
        updates["amount"] = s2(updates["amount"])
    if "category" in updates:
        updates["category"] = updates["category"] or ""
    updated = update_item("budgets", budget_id, updates)
    if not updated:
        raise HTTPException(404, "Budget not found")
    log_activity("updated", "budget", budget_id, updated.get("name", ""),
                 f"{updated.get('amount')} · {updated.get('category') or 'all spending'}")
    return updated


@app.delete("/api/spendwise/budgets/{budget_id}")
def delete_budget(budget_id: int):
    budget = get_item("budgets", budget_id)
    if not budget:
        raise HTTPException(404, "Budget not found")
    backup_before("before-budget-delete")
    if not delete_item("budgets", budget_id):
        raise HTTPException(404, "Budget not found")
    log_activity("deleted", "budget", budget_id, budget.get("name", ""), budget.get("amount", ""))
    return {"ok": True}


# --- Categorisation rules --------------------------------------------------

@app.get("/api/spendwise/rules")
def list_rules():
    return {"rules": spendwise.ordered_rules(_rules())}


@app.post("/api/spendwise/rules", status_code=201)
def create_rule(data: RuleCreate):
    record = {
        "name": data.name,
        "vendor": data.vendor,
        "keyword": data.keyword,
        "category": data.category,
        "priority": data.priority,
        "enabled": data.enabled,
    }
    created = create_item("rules", record)
    log_activity("created", "rule", created.get("id"), created["name"],
                 f"→ {created['category']}")
    return created


@app.put("/api/spendwise/rules/{rule_id}")
def update_rule(rule_id: int, data: RuleUpdate):
    updates = data.model_dump(exclude_none=True)
    updated = update_item("rules", rule_id, updates)
    if not updated:
        raise HTTPException(404, "Rule not found")
    log_activity("updated", "rule", rule_id, updated.get("name", ""),
                 f"→ {updated.get('category')}")
    return updated


@app.delete("/api/spendwise/rules/{rule_id}")
def delete_rule(rule_id: int):
    rule = get_item("rules", rule_id)
    if not rule:
        raise HTTPException(404, "Rule not found")
    backup_before("before-rule-delete")
    if not delete_item("rules", rule_id):
        raise HTTPException(404, "Rule not found")
    log_activity("deleted", "rule", rule_id, rule.get("name", ""), rule.get("category", ""))
    return {"ok": True}


@app.get("/api/spendwise/rules/suggestions")
def rule_suggestions_endpoint(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    start = _parse_iso_date(window["start"])
    end = _parse_iso_date(window["end"])
    expenses = spendwise.filter_expenses(all_items("expenses"), start, end)
    suggestions = spendwise.rule_suggestions(_rules(), expenses)
    return {"period": window, "suggestions": suggestions}


@app.post("/api/spendwise/rules/apply")
def apply_rule_suggestions(data: RuleApplyRequest):
    rules = _rules()
    expenses = all_items("expenses")
    suggestions = spendwise.rule_suggestions(rules, expenses)
    wanted = set(data.ids or [])
    updated = []
    skipped = []
    for row in suggestions:
        if wanted and row["id"] not in wanted:
            continue
        if not row["applyable"]:
            skipped.append(row)
            continue
        if update_item("expenses", row["id"], {"category": row["suggested_category"]}):
            updated.append(row)
    if updated:
        log_activity("updated", "expense", None, "Categorisation rules",
                     f"{len(updated)} expenses recategorised from rules")
    return {
        "updated": len(updated),
        "skipped": len(skipped),
        "updated_expenses": updated,
        "skipped_expenses": skipped,
    }


# ---------------------------------------------------------------------------
# SpendWise AI: what-if, subscription optimiser, spending analysis
# ---------------------------------------------------------------------------

WHATIF_SYSTEM_PROMPT = """You translate a plain-English financial scenario into parameters for a simulator.
Return one JSON object and nothing else. No prose, no markdown fences, no commentary.

You never calculate totals, savings, or any result. A deterministic backend does all arithmetic.
Return only the change the user asked for, using this exact shape:
{
  "label": "short name for the scenario",
  "income_delta": 0,
  "income_change_percent": null,
  "expenses_delta": 0,
  "subscriptions_delta": 0,
  "subscriptions_target": null,
  "bills_delta": 0,
  "savings_delta": 0,
  "one_time_purchase": 0,
  "category_budgets": [{"category": "software", "monthly": 0}]
}

Rules:
- Use income_change_percent only when the user phrases the change as a percentage (for example
  "income decreases 15%" -> income_change_percent: -15). Otherwise use income_delta with the
  monthly amount.
- subscriptions_target sets the new monthly subscription spend. subscriptions_delta changes it by
  an amount. Use at most one of the two.
- one_time_purchase is a single payment, not a monthly amount.
- Provide only the keys the user actually asked about; leave the rest at 0 or null.
- Never invent figures the user did not give or imply."""

ANALYSIS_SYSTEM_PROMPT = """You are a cautious spending analyst for a small business finance app.
You are given verified, calculated facts about one period. Suggest at most five practical actions.

Return one JSON object and nothing else:
{"suggestions": [{"title": "short action", "detail": "one sentence", "refers_to": "the fact it uses"}]}

Rules:
- Quote only figures that appear in the facts you were given. Never invent or recompute a number.
- "refers_to" names the fact (for example a category, budget, vendor, or duplicate) the action uses.
- Separate a suggestion from a fact: do not restate a fact as if it were a recommendation.
- Do not give legal, tax, or investment advice, and do not promise a saving.
- If the facts are too thin to suggest anything useful, return an empty list."""

SUBSCRIPTION_SYSTEM_PROMPT = """You suggest alternatives to a paid subscription for a small business.
You are given one subscription, its detected monthly cost, and what the user says they use it for.

Return one JSON object and nothing else:
{"alternatives": [{
  "product": "name",
  "estimated_monthly_cost": 0,
  "estimated_yearly_cost": 0,
  "capabilities": ["..."],
  "limitations": ["..."],
  "trade_offs": ["..."],
  "pricing_source": "the page you saw the price on",
  "verified_date": "YYYY-MM-DD",
  "notes": "one short sentence"
}]}

Rules:
- Never invent a price. If you cannot attribute a price to a source, omit the cost fields and say so in notes.
- Never claim a service is unlimited or free unless a source confirms it.
- Remember a coding agent or client is not a model provider: include the underlying model or API cost
  when you suggest such a tool.
- Give at most four alternatives, ordered by how likely they are to save money.
- Costs are monthly figures in the stated currency. Do not total anything; the backend does that."""

EXPLAIN_SYSTEM_PROMPT = """You explain figures that a finance application already calculated.
Use only the numbers you are given. Never recalculate, round differently, or invent a figure.
Write two or three short sentences in plain business English. Stay neutral and factual.
Do not give legal, tax, or investment advice, and do not tell the user what they must do."""


class WhatIfRequest(BaseModel):
    period: str = "6m"
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    savings_baseline: Decimal = Field(default=Decimal("0"), ge=0, le=MONEY_MAX)
    scenario: dict = Field(default_factory=dict)
    months: int = Field(default=12, ge=1, le=120)


class WhatIfInterpretRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    period: str = "6m"
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    savings_baseline: Decimal = Field(default=Decimal("0"), ge=0, le=MONEY_MAX)
    months: int = Field(default=12, ge=1, le=120)


class SubscriptionQueryRequest(BaseModel):
    key: str = Field(min_length=1, max_length=400)
    usage: str = Field(default="", max_length=600)


def _months_spanned(start, end) -> int:
    return max((end.year - start.year) * 12 + (end.month - start.month) + 1, 1)


def _monthly_income(invoices: list[dict], window: dict) -> Decimal:
    start = _parse_iso_date(window["start"])
    end = _parse_iso_date(window["end"])
    issued = [
        invoice for invoice in issued_invoices(invoices)
        if in_period(invoice.get("issue_date"), start, end)
    ]
    total = sum((invoice_subtotal(invoice) for invoice in issued), ZERO)
    return q2(total / _months_spanned(start, end))


def _whatif_baseline(window: dict, savings_baseline: Decimal) -> dict:
    """Monthly averages that seed the simulator. Computed, never stored."""
    invoices = all_items("invoices")
    expenses = all_items("expenses")
    stats = spendwise.spending_stats(expenses, window)
    categories = {
        row["category"]: row["total"]
        for row in spendwise.category_breakdown(expenses, window)
    }
    detected = subscriptions.detect(expenses)
    return {
        "income": s2(_monthly_income(invoices, window)),
        "expenses": s2(stats.get("average_monthly") or "0"),
        "subscriptions": s2(subscriptions.totals(detected)["monthly"]),
        "savings": s2(savings_baseline),
        "categories": categories,
        "period": window,
    }


def _whatif_response(window: dict, baseline: dict, raw_scenario: dict, months: int) -> dict:
    scenario = scenario_module.validate(raw_scenario)
    simulation = scenario_module.simulate(baseline, scenario, months=months)
    return {
        "period": window,
        "months": months,
        "baseline": baseline,
        "scenario": {key: _plain(value) for key, value in scenario.items()},
        "simulation": simulation,
    }


# `scenario` is both a request field name and the engine module; alias the module.
# scenario_module = scenario


@app.get("/api/spendwise/whatif/baseline")
def whatif_baseline(
    period: Optional[str] = Query("6m"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    savings: Decimal = Query(Decimal("0"), ge=0, le=MONEY_MAX),
):
    window = _spendwise_period(period, from_date, to_date)
    baseline = _whatif_baseline(window, savings)
    return {
        "period": window,
        "baseline": baseline,
        "currency": load_settings()["currency_code"],
        "ai_available": nvidia_ready(),
    }


@app.post("/api/spendwise/whatif/simulate")
def whatif_simulate(data: WhatIfRequest):
    window = _spendwise_period(data.period, data.from_date, data.to_date)
    baseline = _whatif_baseline(window, data.savings_baseline)
    try:
        return _whatif_response(window, baseline, data.scenario, data.months)
    except scenario.ScenarioError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/spendwise/whatif/interpret")
async def whatif_interpret(data: WhatIfInterpretRequest):
    window = _spendwise_period(data.period, data.from_date, data.to_date)
    baseline = _whatif_baseline(window, data.savings_baseline)
    cfg = ai_config()
    if not ai.nvidia_available(cfg):
        return {
            "ai_available": False,
            "message": (
                "Natural-language scenarios need NVIDIA NIM. Add NVIDIA_API_KEY to your .env file "
                "and restart the app, or set the sliders manually."
            ),
            "baseline": baseline,
        }

    facts = [
        f"Monthly income: {baseline['income']}",
        f"Monthly expenses: {baseline['expenses']}",
        f"Monthly subscription spending: {baseline['subscriptions']}",
        f"Monthly savings: {baseline['savings']}",
        f"Categories (monthly): {json.dumps(baseline['categories'])}",
        "",
        f"User scenario: {data.text.strip()}",
    ]
    try:
        payload = await asyncio.to_thread(
            ai.nvidia_json, cfg, WHATIF_SYSTEM_PROMPT,
            [{"text": "\n".join(facts)}], 900, 0.1,
        )
    except ai.ProviderError as error:
        raise HTTPException(502, str(error)) from error

    try:
        result = _whatif_response(window, baseline, payload, data.months)
    except scenario.ScenarioError as error:
        raise HTTPException(422, f"The scenario could not be applied: {error}") from error

    explanation = ""
    explanation_error = ""
    sim = result["simulation"]
    explain_text = (
        f"Scenario: {data.text.strip()}\n"
        f"Current monthly free cash: {sim['current']['free_cash']}\n"
        f"Simulated monthly free cash: {sim['simulated']['free_cash']}\n"
        f"Monthly difference: {sim['difference']['monthly']}\n"
        f"Annual difference: {sim['difference']['annual']}\n"
        f"One-time purchase: {sim['difference']['one_time_purchase']}\n"
        f"Changes: {'; '.join(sim['notes']) or 'none'}"
    )
    try:
        explanation = await asyncio.to_thread(
            ai.nvidia_text, cfg, EXPLAIN_SYSTEM_PROMPT, [{"text": explain_text}], 400, 0.2,
        )
    except ai.ProviderError as error:
        explanation_error = str(error)

    result["explanation"] = explanation
    result["explanation_error"] = explanation_error
    result["source"] = "nvidia"
    result["model"] = cfg.nvidia_model
    return result


@app.get("/api/spendwise/analysis")
def spendwise_analysis(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    ai_suggestions: int = Query(1, ge=0, le=1),
):
    window = _spendwise_period(period, from_date, to_date)
    currency = load_settings()["currency_code"]
    facts = analysis.build_facts(
        all_items("expenses"), all_items("invoices"), window, _budgets(), _rules()
    )
    savings = analysis.savings_potential(facts)
    suggestions: list[dict] = []
    ai_error = ""
    cfg = ai_config()
    if ai_suggestions:
        if ai.nvidia_available(cfg):
            try:
                payload = ai.nvidia_json(
                    cfg, ANALYSIS_SYSTEM_PROMPT,
                    [{"text": analysis.build_facts_text(facts, currency)}], 900, 0.1,
                )
                suggestions = analysis.parse_suggestions(payload)
            except ai.ProviderError as error:
                ai_error = str(error)
        else:
            ai_error = (
                "AI suggestions need NVIDIA NIM. Add NVIDIA_API_KEY to your .env file to enable them."
            )
    return {
        "period": window,
        "currency": currency,
        "facts": facts,
        "savings": savings,
        "suggestions": suggestions,
        "ai_error": ai_error,
        "ai_available": ai.nvidia_available(cfg),
    }


@app.get("/api/spendwise/subscriptions")
def list_subscriptions():
    expenses = all_items("expenses")
    detected = subscriptions.detect(expenses)
    return {
        "subscriptions": detected,
        "totals": subscriptions.totals(detected),
        "currency": load_settings()["currency_code"],
        "ai_available": nvidia_ready(),
    }


@app.post("/api/spendwise/subscriptions/alternatives")
async def subscription_alternatives(data: SubscriptionQueryRequest):
    expenses = all_items("expenses")
    detected = subscriptions.detect(expenses)
    candidate = next((row for row in detected if row["key"] == data.key), None)
    if candidate is None:
        raise HTTPException(404, "That subscription was not found in the current ledger.")
    currency = load_settings()["currency_code"]
    cfg = ai_config()
    if not ai.nvidia_available(cfg):
        return {
            "subscription": candidate,
            "alternatives": [],
            "ai_available": False,
            "message": (
                "Alternative suggestions need NVIDIA NIM. Add NVIDIA_API_KEY to your .env file "
                "to compare options; the detected cost above is still calculated locally."
            ),
        }
    facts = subscriptions.build_facts(candidate, data.usage, currency)
    try:
        payload = await asyncio.to_thread(
            ai.nvidia_json, cfg, SUBSCRIPTION_SYSTEM_PROMPT, [{"text": facts}], 1400, 0.2,
        )
    except ai.ProviderError as error:
        raise HTTPException(502, str(error)) from error
    alternatives = subscriptions.parse_alternatives(payload, candidate["monthly_cost"])
    return {
        "subscription": candidate,
        "alternatives": alternatives,
        "ai_available": True,
        "model": cfg.nvidia_model,
        "disclaimer": (
            "Prices are AI-summarised and can be out of date. Verify every figure against the "
            "linked source before switching. This app never cancels or purchases anything."
        ),
    }


@app.get("/api/ai/status")
def ai_status():
    cfg = ai_config()
    status = ai.provider_status(cfg)
    status["summary_model"] = GEMINI_MODEL
    status["currency"] = load_settings()["currency_code"]
    return status


# ---------------------------------------------------------------------------
# Intelligent document intake
# ---------------------------------------------------------------------------

DOCUMENT_LIBRARY = "documents"


class DocumentUploadItem(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    # No max_length: the per-file and per-batch upload caps are both 0 (none).
    data_base64: str = Field(default="")


class DocumentUploadRequest(BaseModel):
    files: list[DocumentUploadItem] = Field(default_factory=list)


class DocumentImportRequest(BaseModel):
    destination: Literal["expense", "invoice", "details"]
    fields: dict = Field(default_factory=dict)
    line_items: Optional[list[dict]] = None


def _plain(value):
    """Convert Decimal and date values to JSON-safe forms (money as strings)."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _decode_base64(data: str) -> bytes:
    text = str(data or "").strip()
    if not text:
        raise ValueError("The file was empty.")
    if text.startswith("data:"):
        marker = text.find(",")
        if marker == -1:
            raise ValueError("The file data could not be read.")
        text = text[marker + 1:]
    try:
        return base64.b64decode(text, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("The file data was not valid base64.") from error


def _documents() -> list[dict]:
    return _read(DOCUMENT_LIBRARY)


def _extraction_ai_json(system_prompt: str, user_parts: list[dict]) -> dict:
    """Extract a document with NVIDIA NIM (Nemotron vision), Gemini as fallback."""
    cfg = ai_config()
    if not (ai.nvidia_available(cfg) or ai.gemini_available(cfg)):
        raise documents.ExtractionError(
            "The AI is not configured. Add NVIDIA_API_KEY (or GEMINI_API_KEY) to your "
            ".env file and restart the app to read documents."
        )
    try:
        return ai.document_json(cfg, system_prompt, user_parts, max_output_tokens=2400)
    except ai.ProviderError as error:
        raise documents.ExtractionError(str(error)) from error


def _process_document(record: dict, content: bytes) -> dict:
    """Read one stored document and update its index record in place."""
    settings = load_settings()
    record["error"] = ""
    record["document_type"] = None
    record["extraction"] = None
    record["flags"] = []
    record["calculated"] = {}
    record["duplicates"] = []
    default_tax_rate = dec(settings.get("default_tax_rate", "0.20"))
    try:
        result = documents.run_extraction(
            record["filename"],
            content,
            _extraction_ai_json,
            currency_hint=settings.get("currency_code") or "EUR",
            default_tax_rate=default_tax_rate,
        )
    except (documents.ExtractionError, documents.DocumentError) as error:
        # A provider outage or an unusable model reply must not lose the file.
        # Summarise it locally instead; the result is a general document that
        # cannot be imported as an invoice or an expense.
        try:
            document = documents.local_extraction(record["filename"], content)
            validation = documents.validate_financials(document, default_tax_rate)
        except (documents.ExtractionError, documents.DocumentError):
            record["status"] = "failed"
            record["error"] = str(error)
            return record
        validation.setdefault("flags", []).append({
            "code": "local_fallback",
            "severity": "info",
            "message": (
                "The AI reader was unavailable, so this file was summarised "
                "locally and cannot be imported as an invoice or an expense."
            ),
        })
        result = {"document": document, "validation": validation}

    document = _plain(result["document"])
    record["status"] = "processed"
    record["document_type"] = document.get("document_type")
    record["extraction"] = document
    record["flags"] = result["validation"].get("flags", [])
    record["calculated"] = result["validation"].get("calculated", {})
    record["duplicates"] = documents.find_duplicates(
        document, all_items("invoices"), all_items("expenses")
    )
    _auto_file_document(record)
    return record


def _unstated_fields(extraction: dict, detected: str) -> list[str]:
    """Field names the reader left blank for this document type."""
    fields = extraction.get("fields") or {}
    unstated: list[str] = []
    for name in DOCUMENT_FIELDS.get(detected, ()):
        entry = fields.get(name)
        value = entry.get("value") if isinstance(entry, dict) else entry
        if name in MONEY_FIELD_NAMES:
            stated = _as_money(value) is not None
        elif name in DATE_FIELD_NAMES:
            stated = _as_date(value) is not None
        else:
            stated = bool(_as_text(value))
        if not stated:
            unstated.append(name)
    return unstated


def _auto_file_document(record: dict) -> None:
    """File a read invoice, receipt, or expense right away.

    The reader's type is trusted: an invoice becomes an invoice, and a receipt or
    expense becomes an expense. A field the reader left blank never blocks the
    filing. The value simply stays empty and the document is flagged
    `Unstated <field> on file <name>` so it can be filled in later. Statements,
    general documents, and likely duplicates stay in the library for review.
    A wrong record can be corrected or deleted from its ledger page.
    """
    if record.get("record") or record.get("status") != "processed":
        return
    if record.get("duplicates"):
        return
    extraction = record.get("extraction")
    if not isinstance(extraction, dict):
        return

    detected = str(extraction.get("document_type") or "").strip().lower()
    destination = {"invoice": "invoice", "receipt": "expense", "expense": "expense"}.get(detected)
    if not destination:
        return

    document = copy.deepcopy(extraction)
    document["document_type"] = destination
    try:
        built = documents.to_import_payload(document, {})
        kind = built["kind"]
        payload = built["payload"]
        if kind == "invoice":
            payload.setdefault("client_name", "")
            payload.setdefault("client_email", "")
            payload.setdefault("line_items", [])
            payload.setdefault("totals", {"subtotal": "0.00", "tax_amount": "0.00", "total": "0.00"})
            payload.setdefault("status", "draft")
            created = create_item("invoices", payload)
            reference = invoice_ref(created["id"])
        else:
            payload.setdefault("reimbursable", False)
            payload.setdefault("reimbursed", False)
            created = create_item("expenses", payload)
            reference = payload.get("description") or "Expense"
    except Exception:  # a failed auto-file must never break the upload
        return

    linked = {"kind": kind, "id": created["id"], "reference": reference}
    record["record"] = linked

    filename = record.get("filename", "")
    flags = list(record.get("flags") or [])
    flags.append({
        "code": "auto_filed",
        "severity": "info",
        "message": (
            f"Filed automatically as {kind} {reference}. Correct or delete it if the "
            "extraction was wrong."
        ),
    })
    for name in _unstated_fields(extraction, detected):
        flags.append({
            "code": "unstated",
            "severity": "warning",
            "message": f"Unstated {name} on file {filename}.",
        })
    record["flags"] = flags
    log_activity("imported", kind, created["id"], reference,
                 f"auto-filed from document {filename}")


def _document_result(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "filename": record.get("filename"),
        "extension": record.get("extension"),
        "size": record.get("size"),
        "status": record.get("status"),
        "document_type": record.get("document_type"),
        "uploaded_at": record.get("uploaded_at"),
        "extraction": record.get("extraction"),
        "flags": record.get("flags") or [],
        "calculated": record.get("calculated") or {},
        "duplicates": record.get("duplicates") or [],
        "record": record.get("record"),
        "error": record.get("error") or "",
    }


def _library_row(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "filename": record.get("filename"),
        "extension": record.get("extension"),
        "size": record.get("size"),
        "status": record.get("status"),
        "document_type": record.get("document_type"),
        "uploaded_at": record.get("uploaded_at"),
        "record": record.get("record"),
        "error": record.get("error") or "",
    }


def document_rows(records: list[dict], limit: int = 12) -> list[dict]:
    """Compact, read-only document rows for the assistant snapshot."""
    rows = sorted(records, key=lambda row: row.get("id", 0), reverse=True)[:limit]
    return [
        {
            "id": row.get("id"),
            "filename": row.get("filename"),
            "type": row.get("document_type"),
            "status": row.get("status"),
            "linked": (row.get("record") or {}).get("reference") if row.get("record") else None,
        }
        for row in rows
    ]


def _normalise_line_items(lines: list[dict]) -> list[dict]:
    normalised = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        normalised.append({
            "description": str(line.get("description") or "").strip()[:500],
            "quantity": s4(dec(line.get("quantity", 1)) or Decimal("1")),
            "unit_price": s2(dec(line.get("unit_price", 0))),
            "amount": s2(dec(line.get("amount", 0))) if line.get("amount") not in (None, "") else None,
        })
    return normalised


@app.post("/api/documents/upload")
def upload_documents(data: DocumentUploadRequest):
    if not data.files:
        raise HTTPException(422, "No files were uploaded.")
    if documents.MAX_BATCH_FILES and len(data.files) > documents.MAX_BATCH_FILES:
        raise HTTPException(
            422, f"Upload at most {documents.MAX_BATCH_FILES} files at a time."
        )

    results: list[dict] = []
    rejected: list[dict] = []
    for item in data.files:
        name = item.name
        try:
            content = _decode_base64(item.data_base64)
        except ValueError as error:
            rejected.append({"name": name, "error": str(error)})
            continue
        try:
            info = documents.validate_upload(name, len(content))
            stored = documents.store_file(DATA_DIR, name, content)
        except documents.DocumentError as error:
            rejected.append({"name": name, "error": str(error)})
            continue

        record = {
            "filename": info["safe_name"],
            "stored_name": stored["stored_name"],
            "extension": info["extension"],
            "size": stored["size"],
            "status": "pending",
            "document_type": None,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "extraction": None,
            "flags": [],
            "calculated": {},
            "duplicates": [],
            "record": None,
            "error": "",
        }
        created = create_item(DOCUMENT_LIBRARY, record)
        _process_document(created, content)
        saved = update_item(DOCUMENT_LIBRARY, created["id"], _plain(created))
        log_activity("uploaded", "document", created["id"], created["filename"],
                     f"{created['status']}")
        results.append(_document_result(saved or created))

    return {"results": results, "rejected": rejected}


@app.get("/api/documents")
def list_documents():
    items = sorted(_documents(), key=lambda row: row.get("id", 0), reverse=True)
    return {"documents": [_library_row(row) for row in items]}


@app.get("/api/documents/{document_id}")
def get_document(document_id: int):
    record = get_item(DOCUMENT_LIBRARY, document_id)
    if not record:
        raise HTTPException(404, "Document not found")
    return _document_result(record)


@app.post("/api/documents/{document_id}/retry")
def retry_document(document_id: int):
    record = get_item(DOCUMENT_LIBRARY, document_id)
    if not record:
        raise HTTPException(404, "Document not found")
    try:
        content = documents.read_file(DATA_DIR, record.get("stored_name"))
    except documents.DocumentError as error:
        raise HTTPException(422, str(error)) from error
    _process_document(record, content)
    saved = update_item(DOCUMENT_LIBRARY, document_id, _plain(record))
    log_activity("processed", "document", document_id, record.get("filename", ""),
                 record.get("status", ""))
    return _document_result(saved or record)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int):
    record = get_item(DOCUMENT_LIBRARY, document_id)
    if not record:
        raise HTTPException(404, "Document not found")
    try:
        documents.delete_file(DATA_DIR, record.get("stored_name"))
    except documents.DocumentError as error:
        raise HTTPException(422, str(error)) from error
    if not delete_item(DOCUMENT_LIBRARY, document_id):
        raise HTTPException(404, "Document not found")
    log_activity("deleted", "document", document_id, record.get("filename", ""), "")
    return {"ok": True}


@app.post("/api/documents/{document_id}/import")
def import_document(document_id: int, data: DocumentImportRequest):
    record = get_item(DOCUMENT_LIBRARY, document_id)
    if not record:
        raise HTTPException(404, "Document not found")
    extraction = record.get("extraction")

    if data.destination in ("invoice", "expense") and isinstance(extraction, dict):
        detected = str(extraction.get("document_type") or "").strip().lower()
        if detected not in ("invoice", "receipt", "expense"):
            raise HTTPException(
                422,
                f"This document was read as a {detected or 'general'} document, so it "
                "cannot be filed as an invoice or an expense. Save the details only, "
                "or add the entry manually.",
            )

    if data.destination == "details":
        update_item(DOCUMENT_LIBRARY, document_id, {"record": None})
        log_activity("saved", "document", document_id, record.get("filename", ""),
                     "details saved without a ledger entry")
        return {"ok": True, "created": None, "duplicates": record.get("duplicates") or []}

    if not isinstance(extraction, dict):
        if not data.fields:
            raise HTTPException(422, "This document has no extraction to import.")
        extraction = {
            "document_type": data.destination,
            "confidence": 0.0,
            "summary": "",
            "fields": {},
            "line_items": [],
            "warnings": [],
        }

    document = copy.deepcopy(extraction)
    document["document_type"] = "invoice" if data.destination == "invoice" else "expense"
    if data.line_items is not None:
        document["line_items"] = _normalise_line_items(data.line_items)

    try:
        built = documents.to_import_payload(document, data.fields or {})
    except documents.ExtractionError as error:
        raise HTTPException(422, str(error)) from error

    kind = built["kind"]
    payload = built["payload"]
    if kind == "invoice":
        payload.setdefault("client_name", "")
        payload.setdefault("client_email", "")
        payload.setdefault("line_items", [])
        payload.setdefault("totals", {"subtotal": "0.00", "tax_amount": "0.00", "total": "0.00"})
        payload.setdefault("status", "draft")
        created = create_item("invoices", payload)
        reference = invoice_ref(created["id"])
    else:
        payload.setdefault("reimbursable", False)
        payload.setdefault("reimbursed", False)
        created = create_item("expenses", payload)
        reference = payload.get("description") or "Expense"

    linked = {"kind": kind, "id": created["id"], "reference": reference}
    update_item(DOCUMENT_LIBRARY, document_id, {"record": linked})
    log_activity("imported", kind, created["id"], reference,
                 f"from document {record.get('filename', '')}")
    return {"ok": True, "created": linked, "duplicates": record.get("duplicates") or []}


# ---------------------------------------------------------------------------
# AI summaries
# ---------------------------------------------------------------------------

def _summary_callable():
    return _request_gemini_json if is_configured() else None


def _spendwise_facts(period: dict) -> dict:
    return summaries.build_spendwise_facts(
        all_items("expenses"), all_items("invoices"), period, _budgets()
    )


def _overview_facts() -> tuple[dict, dict, dict]:
    period = spendwise.resolve_period("month", None, None)
    spendwise_facts = _spendwise_facts(period)
    facts = summaries.build_overview_facts(
        all_items("invoices"), all_items("expenses"), spendwise_facts, load_settings()
    )
    return facts, spendwise_facts, period


def _spending_figures(spendwise_facts: dict) -> dict:
    totals = spendwise_facts.get("budget_totals") or {}
    return {
        "total": spendwise_facts.get("total"),
        "previous_total": spendwise_facts.get("previous_total"),
        "change_amount": spendwise_facts.get("change_amount"),
        "change_percent": spendwise_facts.get("change_percent"),
        "top_category": spendwise_facts.get("top_category"),
        "budget_percent": totals.get("percent"),
        "budget_note": (
            f"{totals['count']} budgets · combined utilisation {totals['percent']}%"
            if totals.get("count")
            else "No budgets set"
        ),
        "period": spendwise_facts.get("period"),
    }


def _summary_response(scope: str, facts: dict, currency: str,
                      stored: Optional[dict], period: Optional[dict] = None,
                      spending: Optional[dict] = None) -> dict:
    computed = summaries.compute_summary(facts, currency, scope)
    response = {
        "scope": scope,
        "period": period,
        "currency": currency,
        "cached": stored is not None,
        "generated_at": (stored or {}).get("generated_at"),
        "source": (stored or {}).get("source"),
        "verified": (stored or {}).get("verified"),
        "unverified_figures": (stored or {}).get("unverified_figures") or [],
        "error": (stored or {}).get("error") or "",
        "summary": (stored or {}).get("summary"),
        "computed": computed,
    }
    if spending is not None:
        response["spending"] = spending
    return response


@app.get("/api/summaries/spendwise")
def get_spendwise_summary(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    facts = _spendwise_facts(window)
    currency = load_settings()["currency_code"]
    fingerprint = summaries.fingerprint(facts, "spendwise")
    stored = summaries.get_cached(DATA_DIR, "spendwise", fingerprint)
    return _summary_response("spendwise", facts, currency, stored, window)


@app.post("/api/summaries/spendwise/refresh")
def refresh_spendwise_summary(
    period: Optional[str] = Query("month"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    window = _spendwise_period(period, from_date, to_date)
    facts = _spendwise_facts(window)
    currency = load_settings()["currency_code"]
    payload = summaries.build_summary(facts, currency, "spendwise", _summary_callable())
    stored = summaries.store_cached(
        DATA_DIR, "spendwise", summaries.fingerprint(facts, "spendwise"), payload
    )
    return _summary_response("spendwise", facts, currency, stored, window)


@app.get("/api/summaries/overview")
def get_overview_summary():
    facts, spendwise_facts, _period = _overview_facts()
    currency = load_settings()["currency_code"]
    fingerprint = summaries.fingerprint(facts, "overview")
    stored = summaries.get_cached(DATA_DIR, "overview", fingerprint)
    return _summary_response("overview", facts, currency, stored,
                             spending=_spending_figures(spendwise_facts))


@app.post("/api/summaries/overview/refresh")
def refresh_overview_summary():
    facts, spendwise_facts, _period = _overview_facts()
    currency = load_settings()["currency_code"]
    payload = summaries.build_summary(facts, currency, "overview", _summary_callable())
    stored = summaries.store_cached(
        DATA_DIR, "overview", summaries.fingerprint(facts, "overview"), payload
    )
    return _summary_response("overview", facts, currency, stored,
                             spending=_spending_figures(spendwise_facts))


# ---------------------------------------------------------------------------
# Dashboard and reports API
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/summary")
def dashboard_summary():
    settings = load_settings()
    invoices = all_items("invoices")
    expenses = all_items("expenses")
    issued = issued_invoices(invoices)
    total_invoiced = sum((invoice_total(item) for item in issued), ZERO)
    net_revenue = sum((invoice_subtotal(item) for item in issued), ZERO)
    total_paid = sum((invoice_total(item) for item in invoices if item.get("status") == "paid"), ZERO)
    total_outstanding = sum(
        (invoice_total(item) for item in invoices if item.get("status") in ("sent", "overdue")), ZERO
    )
    total_expense_amount = sum((expense_amount(item) for item in expenses), ZERO)
    net_profit = net_revenue - total_expense_amount
    recent = sorted(invoices, key=lambda item: item.get("created_at", ""), reverse=True)[:5]
    unreimbursed = sum(
        (expense_amount(item) for item in expenses
         if item.get("reimbursable") and not item.get("reimbursed")), ZERO
    )
    return {
        "total_invoiced": s2(total_invoiced),
        "net_revenue": s2(net_revenue),
        "total_paid": s2(total_paid),
        "total_outstanding": s2(total_outstanding),
        "total_expenses": s2(total_expense_amount),
        "net_profit": s2(net_profit),
        "overdue_count": sum(1 for item in invoices if item.get("status") == "overdue"),
        "draft_count": sum(1 for item in invoices if item.get("status") == "draft"),
        "unreimbursed_expenses": s2(unreimbursed),
        "reporting_basis": "accrual",
        "basis_note": REPORTING_BASIS,
        "currency": settings["currency_code"],
        "recent_invoices": [
            {
                "id": item["id"],
                "client_name": item.get("client_name", ""),
                "total": s2(invoice_total(item)),
                "status": item.get("status", ""),
                "date": item.get("issue_date", item.get("created_at", "")),
            }
            for item in recent
        ],
    }


def _parse_date_range(from_date: Optional[str], to_date: Optional[str]) -> tuple[date, date]:
    today = date.today()
    start = date(today.year, today.month, 1)
    end = today
    if from_date:
        try:
            start = date.fromisoformat(from_date)
        except ValueError as error:
            raise HTTPException(422, "from_date must be a valid ISO date (YYYY-MM-DD).") from error
    if to_date:
        try:
            end = date.fromisoformat(to_date)
        except ValueError as error:
            raise HTTPException(422, "to_date must be a valid ISO date (YYYY-MM-DD).") from error
    if end < start:
        raise HTTPException(422, "to_date cannot be before from_date.")
    return start, end


@app.get("/api/reports/profit-loss")
def profit_loss(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    start, end = _parse_date_range(from_date, to_date)
    invoices = all_items("invoices")
    expenses = all_items("expenses")

    income_items = []
    for invoice in invoices:
        if invoice.get("status") not in ISSUED_STATUS:
            continue
        issued = invoice.get("issue_date", "")
        if not in_period(issued, start, end):
            continue
        income_items.append({
            "date": issued,
            "description": f"{invoice_ref(invoice['id'])} — {invoice.get('client_name', '')}",
            "amount": s2(invoice_subtotal(invoice)),
            "gross": s2(invoice_total(invoice)),
            "tax": s2(invoice_tax(invoice)),
            "status": invoice.get("status"),
        })

    expense_items = []
    for expense in expenses:
        incurred = expense.get("date_incurred", "")
        if not in_period(incurred, start, end):
            continue
        expense_items.append({
            "date": incurred,
            "description": expense.get("description", ""),
            "amount": s2(expense_amount(expense)),
            "category": expense.get("category", "other"),
        })

    total_income = sum((dec(item["amount"]) for item in income_items), ZERO)
    total_expenses = sum((dec(item["amount"]) for item in expense_items), ZERO)
    return {
        "period": {"from": start.isoformat(), "to": end.isoformat()},
        "reporting_basis": "accrual",
        "basis_note": REPORTING_BASIS,
        "total_income": s2(total_income),
        "total_expenses": s2(total_expenses),
        "net_profit": s2(total_income - total_expenses),
        "income": sorted(income_items, key=lambda item: item["date"]),
        "expenses": sorted(expense_items, key=lambda item: item["date"]),
    }


@app.get("/api/reports/balance-sheet")
def balance_sheet():
    invoices = all_items("invoices")
    expenses = all_items("expenses")
    total_receivable = sum(
        (invoice_total(item) for item in invoices if item.get("status") in ("sent", "overdue")), ZERO
    )
    total_received = sum(
        (invoice_total(item) for item in invoices if item.get("status") == "paid"), ZERO
    )
    recorded_expenses = sum((expense_amount(item) for item in expenses), ZERO)
    reimbursement_payable = sum(
        (expense_amount(item) for item in expenses
         if item.get("reimbursable") and not item.get("reimbursed")), ZERO
    )
    output_tax_payable = sum(
        (invoice_tax(item) for item in invoices if item.get("status") in ISSUED_STATUS), ZERO
    )
    net_cash = total_received - recorded_expenses
    total_assets = total_receivable + net_cash
    total_liabilities = reimbursement_payable + output_tax_payable
    equity = total_assets - total_liabilities
    return {
        "as_of": date.today().isoformat(),
        "reporting_basis": "accrual",
        "is_complete": False,
        "basis_note": BALANCE_SHEET_NOTE,
        "assets": {
            "accounts_receivable": s2(total_receivable),
            "cash_received": s2(total_received),
            "recorded_expenses": s2(recorded_expenses),
            "net_cash": s2(net_cash),
            "total_assets": s2(total_assets),
        },
        "liabilities": {
            "reimbursement_payable": s2(reimbursement_payable),
            "output_tax_payable": s2(output_tax_payable),
            "total_liabilities": s2(total_liabilities),
        },
        "equity": {
            "retained_earnings": s2(equity),
            "total_equity": s2(equity),
        },
        "assumptions": [
            "Recorded expenses are treated as settled cash outflows.",
            "Only reimbursable, unreimbursed expenses are treated as reimbursement liabilities.",
            "Output tax on issued invoices is treated as payable because tax remittances are not tracked.",
            "Recoverable input tax on expenses and opening equity are not modelled.",
            "Vendor payables and bank balances are not tracked.",
        ],
    }


@app.get("/api/reports/tax-summary")
def tax_summary(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
):
    start, end = _parse_date_range(from_date, to_date)
    invoices = all_items("invoices")
    issued = [invoice for invoice in invoices
              if invoice.get("status") in ISSUED_STATUS and in_period(invoice.get("issue_date"), start, end)]
    paid = [invoice for invoice in issued if invoice.get("status") == "paid"]
    drafts = [invoice for invoice in invoices
              if invoice.get("status") == "draft" and in_period(invoice.get("issue_date"), start, end)]
    cancelled = [invoice for invoice in invoices
                 if invoice.get("status") == "cancelled" and in_period(invoice.get("issue_date"), start, end)]
    net_sales = sum((invoice_subtotal(item) for item in issued), ZERO)
    tax_invoiced = sum((invoice_tax(item) for item in issued), ZERO)
    tax_collected = sum((invoice_tax(item) for item in paid), ZERO)
    return {
        "period": {"from": start.isoformat(), "to": end.isoformat()},
        "reporting_basis": "accrual",
        "basis_note": (
            "Tax invoiced covers issued invoices (sent, overdue, paid). "
            "Tax collected covers invoices marked paid. Draft and cancelled invoices are shown separately."
        ),
        "total_sales": s2(net_sales + tax_invoiced),
        "net_sales": s2(net_sales),
        "total_tax_invoiced": s2(tax_invoiced),
        "total_tax_collected": s2(tax_collected),
        "tax_outstanding": s2(tax_invoiced - tax_collected),
        "draft_count": len(drafts),
        "draft_value": s2(sum((invoice_total(item) for item in drafts), ZERO)),
        "draft_tax": s2(sum((invoice_tax(item) for item in drafts), ZERO)),
        "cancelled_count": len(cancelled),
        "cancelled_value": s2(sum((invoice_total(item) for item in cancelled), ZERO)),
        "cancelled_tax": s2(sum((invoice_tax(item) for item in cancelled), ZERO)),
        "taxable_transactions": sum(1 for item in issued if invoice_tax(item) > 0),
    }


# ---------------------------------------------------------------------------
# Settings, backups, activity, and CSV import API
# ---------------------------------------------------------------------------

@app.get("/api/settings")
def get_settings():
    return {
        "settings": load_settings(),
        "options": {
            "date_formats": ["short", "medium", "long", "iso"],
            "default_tax_rate": "0.20",
            "reporting_basis": "accrual",
        },
    }


@app.put("/api/settings")
def update_settings(data: SettingsUpdate):
    current = load_settings()
    updated = {**current, **data.model_dump(exclude_unset=True)}
    try:
        saved = save_settings(updated)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    log_activity("updated", "settings", None, "application settings", "Currency, tax or formatting settings changed.")
    return saved


@app.get("/api/backup")
def download_backup():
    backup, _ = create_backup("manual-download")
    payload = json.dumps(backup, indent=2, ensure_ascii=False, default=_json_default)
    filename = f"finance-agent-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/backup/restore")
def restore_backup_endpoint(payload: dict = Body(...)):
    try:
        result = restore_backup(payload)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return result


@app.get("/api/activity")
def activity(limit: int = Query(100, ge=1, le=500)):
    return {"items": read_activity(limit)}


@app.post("/api/import/invoices/preview")
def preview_invoice_import_endpoint(data: CsvImportRequest):
    try:
        return prepare_invoice_import(data.csv_text)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/import/invoices")
def import_invoices_endpoint(data: CsvImportRequest):
    try:
        prepared = prepare_invoice_import(data.csv_text)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if prepared["errors"]:
        raise HTTPException(422, prepared["errors"])
    if prepared["duplicates"] and not data.allow_duplicates:
        raise HTTPException(409, "Duplicate invoices detected. Confirm that duplicates should be imported.")
    backup_before("before-invoice-import")
    created = 0
    for group in prepared["prepared"]:
        create_item("invoices", {
            "client_name": group["client_name"],
            "client_email": group["client_email"],
            "issue_date": group["issue_date"],
            "due_date": group["due_date"],
            "line_items": group["line_items"],
            "tax_rate": s4(group["tax_rate"]),
            "totals": group["totals"],
            "notes": group["notes"],
            "status": group["status"],
        })
        created += 1
    log_activity("imported", "invoice", None, f"{created} invoices", "CSV invoice import.")
    return {"ok": True, "created": created, "duplicates": len(prepared["duplicates"])}


@app.post("/api/import/expenses/preview")
def preview_expense_import_endpoint(data: CsvImportRequest):
    try:
        return prepare_expense_import(data.csv_text)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/import/expenses")
def import_expenses_endpoint(data: CsvImportRequest):
    try:
        prepared = prepare_expense_import(data.csv_text)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if prepared["errors"]:
        raise HTTPException(422, prepared["errors"])
    if prepared["duplicates"] and not data.allow_duplicates:
        raise HTTPException(409, "Duplicate expenses detected. Confirm that duplicates should be imported.")
    backup_before("before-expense-import")
    created = 0
    for row in prepared["prepared"]:
        create_item("expenses", {
            "description": row["description"], "amount": row["amount"],
            "date_incurred": row["date_incurred"], "category": row["category"],
            "vendor": row["vendor"], "notes": row["notes"],
            "reimbursable": row["reimbursable"], "reimbursed": row["reimbursed"],
        })
        created += 1
    log_activity("imported", "expense", None, f"{created} expenses", "CSV expense import.")
    return {"ok": True, "created": created, "duplicates": len(prepared["duplicates"])}


# ---------------------------------------------------------------------------
# Assistant API
# ---------------------------------------------------------------------------

@app.get("/api/assistant/status")
def assistant_status():
    settings = load_settings()
    cfg = ai_config()
    return {
        "configured": is_configured(),
        "provider": "Google Gemini",
        "model": GEMINI_MODEL,
        "currency": settings["currency_code"],
        "max_message_length": ASSISTANT_MAX_MESSAGE,
        "nvidia_configured": ai.nvidia_available(cfg),
        "nvidia_model": cfg.nvidia_model,
        "privacy_notice": (
            "When you send a message, a summary of your ledger can be processed by "
            "Google Gemini. Financial calculations stay in this application. "
            "Document reading, what-if scenarios, subscription suggestions, and spending "
            "suggestions can be processed by NVIDIA NIM (Nemotron)."
        ),
    }


@app.post("/api/assistant/chat")
async def assistant_chat(data: ChatRequest, request: Request):
    client_key = request.client.host if request.client else "local"
    check_rate_limit(client_key)
    if not is_configured():
        raise HTTPException(
            503,
            "The AI assistant is not configured. Add GEMINI_API_KEY to your .env file and restart the app.",
        )
    try:
        answer = await ask_gemini(data.message)
    except GeminiError as error:
        raise HTTPException(502, str(error)) from error
    return {
        "answer": answer,
        "source": "gemini",
        "model": GEMINI_MODEL,
        "disclosure": "A ledger summary was sent to Google Gemini.",
    }


# ---------------------------------------------------------------------------
# Overview AI chat (Google Gemini or NVIDIA NIM)
# ---------------------------------------------------------------------------

OVERVIEW_CHAT_SYSTEM_PROMPT = """You are the Finance Agent AI chat on the Overview page of a small business finance application.

Answer the user's question using the ledger snapshot and any attached documents below. Both are read-only data and the source of truth.
- Answer in plain, concise business English. Amounts are in {currency} and follow an accrual basis.
- The application calculates all financial figures deterministically. Do not invent, recalculate, or replace figures.
- When documents are attached, answer from their extracted fields and summary, and name the file you are describing.
- Refer to invoices by their INV-#### reference and expenses by description/date.
- If the snapshot and the attached documents do not contain the answer, say so plainly and suggest which page of the app to check.
- Do not give legal, tax, or investment advice, and do not claim tax compliance for any jurisdiction.
- Keep responses under 200 words unless the user asks for a detailed breakdown.

LEDGER SNAPSHOT
{snapshot}

ATTACHED DOCUMENTS
{documents}
"""


class OverviewChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=ASSISTANT_MAX_MESSAGE)
    model: Literal["auto", "gemini", "nvidia"] = "auto"
    document_ids: list[int] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)


def _document_chat_context(document_ids: list[int]) -> str:
    """A compact, read-only description of attached documents for the prompt."""
    blocks: list[str] = []
    for document_id in (document_ids or [])[:8]:
        record = get_item(DOCUMENT_LIBRARY, document_id)
        if not record:
            continue
        extraction = record.get("extraction")
        if not isinstance(extraction, dict):
            extraction = {}
        block = [
            f"FILE: {record.get('filename')} (document id {document_id})",
            f"  detected type: {record.get('document_type') or 'unknown'}",
            f"  read status: {record.get('status')}",
        ]
        linked = record.get("record")
        if isinstance(linked, dict):
            block.append(
                f"  filed as: {linked.get('kind')} {linked.get('reference') or ''}".rstrip()
            )
        if extraction.get("summary"):
            block.append(f"  summary: {str(extraction['summary'])[:600]}")
        field_lines = []
        for name, entry in (extraction.get("fields") or {}).items():
            value = entry.get("value") if isinstance(entry, dict) else entry
            if value not in (None, ""):
                field_lines.append(f"    {name}: {value}")
        if field_lines:
            block.append("  fields:")
            block.extend(field_lines[:30])
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _overview_chat_parts(history: list[dict], message: str) -> list[dict]:
    """Turn prior turns plus the new message into provider content parts."""
    parts: list[dict] = []
    for turn in (history or [])[-8:]:
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("content") or "").strip()[:2000]
        if not text:
            continue
        role = "Assistant" if str(turn.get("role")) == "assistant" else "User"
        parts.append({"text": f"{role}: {text}"})
    parts.append({"text": f"User: {message.strip()}"})
    return parts


@app.post("/api/overview/chat")
async def overview_chat(data: OverviewChatRequest, request: Request):
    client_key = request.client.host if request.client else "local"
    check_rate_limit(client_key)

    cfg = ai_config()
    nvidia_ready = ai.nvidia_available(cfg)
    gemini_ready = is_configured()

    choice = data.model if data.model in ("auto", "gemini", "nvidia") else "auto"
    if choice == "auto":
        choice = "nvidia" if nvidia_ready else "gemini"
    if choice == "nvidia" and not nvidia_ready:
        raise HTTPException(
            503,
            "NVIDIA NIM is not configured. Add NVIDIA_API_KEY to your .env file and "
            "restart the app.",
        )
    if choice == "gemini" and not gemini_ready:
        raise HTTPException(
            503,
            "Gemini is not configured. Add GEMINI_API_KEY to your .env file and restart "
            "the app.",
        )

    settings = load_settings()
    system_prompt = OVERVIEW_CHAT_SYSTEM_PROMPT.format(
        currency=settings["currency_code"],
        snapshot=build_ledger_snapshot(),
        documents=_document_chat_context(data.document_ids)
        or "No documents are attached to this conversation.",
    )
    parts = _overview_chat_parts(data.history, data.message)

    try:
        if choice == "nvidia":
            answer = await asyncio.to_thread(
                ai.nvidia_text, cfg, system_prompt, parts, 900, 0.25
            )
            model = cfg.nvidia_model
        else:
            answer = await asyncio.to_thread(
                _request_gemini_text, system_prompt, parts, 900, 0.25
            )
            model = GEMINI_MODEL
    except ai.ProviderError as error:
        raise HTTPException(502, str(error)) from error
    except GeminiError as error:
        raise HTTPException(502, str(error)) from error

    return {
        "answer": answer,
        "source": choice,
        "model": model,
        "document_ids": list(data.document_ids or []),
    }


# ===========================================================================
# Frontend assets — inlined so the app is a single file.
# Originally static/style.css and static/app.js.
# ===========================================================================

STYLE_CSS = r"""
/* ==========================================================================
   Finance Agent — design system
   Editorial accounting UI: warm charcoal / paper, gold accents, flat surfaces.
   ========================================================================== */

:root {
  --font-display: "Playfair Display", Georgia, "Times New Roman", serif;
  --font-sans: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "DM Mono", "SFMono-Regular", Consolas, monospace;

  --sidebar-width: 244px;
  --content-max: 1440px;
  --radius: 8px;
  --radius-sm: 5px;
  --radius-lg: 10px;
  --ease: 160ms cubic-bezier(0.2, 0.6, 0.2, 1);

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;
}

/* Dark (default) */
:root,
[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0a0a0c;
  --bg-raised: #0d0d10;
  --surface: #121216;
  --surface-2: #17171c;
  --surface-3: #1d1d23;
  --border: #26262e;
  --border-strong: #383840;
  --text: #f1eee6;
  --text-secondary: #a8a398;
  --text-muted: #8f8a7f;
  --gold: #c8a84b;
  --gold-strong: #e0c36b;
  --gold-soft: rgba(200, 168, 75, 0.11);
  --gold-line: rgba(200, 168, 75, 0.28);
  --on-gold: #151105;
  --positive: #7dbd93;
  --positive-soft: rgba(125, 189, 147, 0.1);
  --negative: #db8484;
  --negative-soft: rgba(219, 132, 132, 0.1);
  --warning: #d4b061;
  --info: #94aecf;
  --overlay: rgba(3, 3, 4, 0.72);
  --shadow: 0 18px 50px rgba(0, 0, 0, 0.55);
  --chart-expense: #4b4b55;
  --track: #232329;
}

/* Light */
[data-theme="light"] {
  color-scheme: light;
  --bg: #f4f2ec;
  --bg-raised: #fbfaf6;
  --surface: #ffffff;
  --surface-2: #f7f5ef;
  --surface-3: #efede5;
  --border: #e1ddd3;
  --border-strong: #c9c3b4;
  --text: #1c1a15;
  --text-secondary: #625e52;
  --text-muted: #6f6a5e;
  --gold: #9a7420;
  --gold-strong: #7d5c12;
  --gold-soft: rgba(154, 116, 32, 0.09);
  --gold-line: rgba(154, 116, 32, 0.25);
  --on-gold: #ffffff;
  --positive: #2f7a50;
  --positive-soft: rgba(47, 122, 80, 0.08);
  --negative: #b04a4a;
  --negative-soft: rgba(176, 74, 74, 0.08);
  --warning: #96701a;
  --info: #4f6f96;
  --overlay: rgba(30, 27, 20, 0.46);
  --shadow: 0 18px 46px rgba(45, 38, 20, 0.14);
  --chart-expense: #b9b3a5;
  --track: #e9e5da;
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

html {
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.55;
  color: var(--text);
  background: var(--bg);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1,
h2,
h3,
h4,
p,
ul,
ol,
figure {
  margin: 0;
}

ul,
ol {
  padding: 0;
  list-style: none;
}

button,
input,
select,
textarea {
  font: inherit;
  color: inherit;
}

button {
  cursor: pointer;
}

a {
  color: inherit;
}

svg {
  display: block;
}

::selection {
  background: var(--gold);
  color: var(--on-gold);
}

:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: 2px;
}

/* Scrollbars */
* {
  scrollbar-width: thin;
  scrollbar-color: var(--border-strong) transparent;
}
::-webkit-scrollbar {
  width: 9px;
  height: 9px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border: 3px solid var(--bg);
  border-radius: 20px;
}

/* ==========================================================================
   Shell
   ========================================================================== */

.app-shell {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  position: fixed;
  inset: 0 auto 0 0;
  z-index: 100;
  display: flex;
  flex-direction: column;
  width: var(--sidebar-width);
  background: var(--bg-raised);
  border-right: 1px solid var(--border);
}

.sidebar-brand {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 72px;
  padding: 0 var(--space-5);
  border-bottom: 1px solid var(--border);
}

.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  text-decoration: none;
  min-width: 0;
}

.brand-mark {
  display: block;
  width: 31px;
  height: 31px;
  flex: 0 0 31px;
  border-radius: 8px;
  object-fit: contain;
}

.brand-text {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.015em;
  white-space: nowrap;
}

.brand-text em {
  color: var(--gold);
  font-style: italic;
  margin-left: 2px;
}

.sidebar .sidebar-close {
  display: none;
}

.workspace {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin: var(--space-5) var(--space-4) 0;
  padding: var(--space-3) var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.workspace-label {
  font-family: var(--font-mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.workspace-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.workspace-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}

.workspace-status .dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--positive);
  box-shadow: 0 0 0 3px var(--positive-soft);
}

.sidebar-nav {
  flex: 1;
  padding: var(--space-6) var(--space-3) var(--space-4);
  overflow-y: auto;
}

.nav-section {
  display: block;
  padding: 0 var(--space-3) var(--space-2);
  font-family: var(--font-mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 9px var(--space-3);
  margin-bottom: 2px;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 13.5px;
  font-weight: 500;
  text-decoration: none;
  transition: background var(--ease), color var(--ease);
}

.nav-item:hover {
  background: var(--surface-2);
  color: var(--text);
}

.nav-item.active {
  background: var(--gold-soft);
  color: var(--gold);
}

.nav-item.active::before {
  content: "";
  position: absolute;
  left: -12px;
  top: 50%;
  width: 2px;
  height: 18px;
  transform: translateY(-50%);
  background: var(--gold);
}

.nav-icon {
  display: flex;
  flex: 0 0 auto;
  color: currentColor;
  opacity: 0.9;
}

.nav-label {
  flex: 1;
}

.nav-count {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
  min-width: 18px;
  text-align: right;
}

.nav-item.active .nav-count {
  color: var(--gold);
}

.sidebar-footer {
  padding: var(--space-3) var(--space-3) var(--space-4);
  border-top: 1px solid var(--border);
}

.assistant-cta {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px var(--space-3);
  margin-bottom: var(--space-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12.5px;
  font-weight: 500;
  text-align: left;
  transition: border-color var(--ease), color var(--ease), background var(--ease);
}

.assistant-cta:hover {
  border-color: var(--gold-line);
  color: var(--gold);
  background: var(--gold-soft);
}

.assistant-cta .spark {
  margin-left: auto;
  color: var(--gold);
}

.theme-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12.5px;
  font-weight: 500;
  text-align: left;
  transition: border-color var(--ease), color var(--ease), background var(--ease);
}

.theme-toggle:hover {
  border-color: var(--gold-line);
  color: var(--gold);
  background: var(--gold-soft);
}

.theme-toggle .theme-icon {
  display: inline-flex;
  color: currentColor;
}

.icon-moon {
  display: none;
}

[data-theme="light"] .icon-sun {
  display: none;
}

[data-theme="light"] .icon-moon {
  display: block;
}

.app-main {
  flex: 1;
  min-width: 0;
  margin-left: var(--sidebar-width);
}

.content {
  width: 100%;
  max-width: var(--content-max);
  margin: 0 auto;
  padding: var(--space-10) var(--space-10) var(--space-12);
  /* Keep wide, internally scrollable tables from expanding the mobile
     layout viewport (which would misplace fixed dialogs and the assistant). */
  contain: layout;
}

.mobile-bar {
  display: none;
  align-items: center;
  gap: var(--space-3);
  height: 56px;
  padding: 0 var(--space-4);
  border-bottom: 1px solid var(--border);
  background: var(--bg-raised);
  position: sticky;
  top: 0;
  z-index: 90;
}

.mobile-brand {
  flex: 1;
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 600;
}

.sidebar-scrim {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 99;
  background: var(--overlay);
  opacity: 0;
  transition: opacity var(--ease);
}

/* ==========================================================================
   Page headers
   ========================================================================== */

.page-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-6);
  margin-bottom: var(--space-8);
  padding-bottom: var(--space-6);
  border-bottom: 1px solid var(--border);
}

.eyebrow {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: var(--space-2);
}

.page-title {
  font-family: var(--font-display);
  font-size: clamp(30px, 3.2vw, 40px);
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.05;
}

.page-sub {
  margin-top: var(--space-2);
  color: var(--text-secondary);
  font-size: 13.5px;
  max-width: 560px;
}

.page-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex: 0 0 auto;
  padding-bottom: 2px;
}

/* ==========================================================================
   Buttons
   ========================================================================== */

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  min-height: 36px;
  padding: 0 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 600;
  line-height: 1;
  white-space: nowrap;
  text-decoration: none;
  transition: background var(--ease), border-color var(--ease), color var(--ease);
}

.btn:hover {
  border-color: var(--border-strong);
  color: var(--text);
  background: var(--surface-2);
}

.btn svg {
  flex: 0 0 auto;
}

.btn-primary {
  border-color: var(--gold);
  background: var(--gold);
  color: var(--on-gold);
}

.btn-primary:hover {
  border-color: var(--gold-strong);
  background: var(--gold-strong);
  color: var(--on-gold);
}

.btn-ghost {
  border-color: transparent;
  background: transparent;
}

.btn-ghost:hover {
  border-color: transparent;
  background: var(--surface-2);
}

.btn-danger {
  border-color: var(--negative);
  background: transparent;
  color: var(--negative);
}

.btn-danger:hover {
  border-color: var(--negative);
  background: var(--negative-soft);
  color: var(--negative);
}

.btn-sm {
  min-height: 30px;
  padding: 0 10px;
  font-size: 12px;
}

.btn-icon {
  width: 32px;
  min-height: 32px;
  padding: 0;
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  padding: 0;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
  transition: background var(--ease), color var(--ease), border-color var(--ease);
}

.icon-btn:hover {
  background: var(--surface-2);
  border-color: var(--border);
  color: var(--text);
}

.icon-btn.danger:hover {
  background: var(--negative-soft);
  border-color: transparent;
  color: var(--negative);
}

.btn:disabled,
.icon-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.text-link {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 0;
  background: none;
  color: var(--gold);
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
}

.text-link:hover {
  color: var(--gold-strong);
  text-decoration: underline;
  text-underline-offset: 3px;
}

/* ==========================================================================
   Panels
   ========================================================================== */

.panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  min-width: 0;
}

.dash-grid > * {
  min-width: 0;
}

.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-5) var(--space-5) var(--space-4);
  border-bottom: 1px solid var(--border);
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  letter-spacing: -0.01em;
}

.panel-sub {
  margin-top: 2px;
  font-size: 11.5px;
  color: var(--text-muted);
}

.panel-body {
  padding: var(--space-5);
}

.panel-body.flush {
  padding: 0;
}

/* SpendWise vertical rhythm: every top-level block is separated by one step
   and the page keeps clear space at the bottom for the floating assistant. */
[data-page="spendwise"] .content > * {
  margin-bottom: var(--space-4);
}

[data-page="spendwise"] .content > :last-child {
  margin-bottom: 0;
}

[data-page="spendwise"] .content {
  padding-bottom: calc(var(--space-12) + var(--space-6));
}

[data-page="spendwise"] .metric-band {
  margin-bottom: var(--space-5);
}

/* Overview and Documents share the same vertical rhythm as SpendWise so cards
   never touch and every page keeps clear space above the bottom edge. */
[data-page="overview"] .content,
[data-page="documents"] .content {
  padding-bottom: calc(var(--space-12) + var(--space-6));
}

[data-page="overview"] .content > *:not(.page-head),
[data-page="documents"] .content > *:not(.page-head) {
  margin-bottom: var(--space-4);
}

[data-page="overview"] .content > :last-child,
[data-page="documents"] .content > :last-child {
  margin-bottom: 0;
}

/* ==========================================================================
   Metric band
   ========================================================================== */

.metric-band {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  margin-bottom: var(--space-6);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  overflow: hidden;
}

.metric {
  position: relative;
  padding: var(--space-5);
  min-width: 0;
}

.metric + .metric {
  border-left: 1px solid var(--border);
}

.metric.featured {
  background: var(--gold-soft);
}

.metric-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.metric-value {
  display: block;
  margin-top: var(--space-3);
  font-family: var(--font-mono);
  font-size: clamp(20px, 2.1vw, 27px);
  font-weight: 500;
  letter-spacing: -0.03em;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}

.metric.featured .metric-value {
  color: var(--gold);
}

.metric-value.negative {
  color: var(--negative);
}

.metric-note {
  display: block;
  margin-top: 6px;
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.metric-note .up {
  color: var(--positive);
}

.metric-note .down {
  color: var(--negative);
}

/* ==========================================================================
   Dashboard
   ========================================================================== */

.dash-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.65fr) minmax(300px, 1fr);
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.dash-grid.lower {
  grid-template-columns: minmax(0, 1.65fr) minmax(300px, 1fr);
}

/* Wide financial tables need the full row to stay readable. */
.dash-grid.stack {
  grid-template-columns: minmax(0, 1fr);
}

/* Chart */
.chart-legend {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  flex-wrap: wrap;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-secondary);
}

.legend-swatch {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: var(--gold);
}

.legend-swatch.expense {
  background: var(--chart-expense);
}

.chart-wrap {
  padding: var(--space-5) var(--space-5) var(--space-4);
}

.chart-shell {
  display: flex;
  gap: var(--space-3);
  height: 220px;
}

.chart-y-axis {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  width: 46px;
  padding-bottom: 22px;
  font-family: var(--font-mono);
  font-size: 9.5px;
  color: var(--text-muted);
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.chart-main {
  position: relative;
  flex: 1;
  min-width: 0;
}

.chart-gridlines {
  position: absolute;
  inset: 0 0 22px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  pointer-events: none;
}

.chart-gridlines span {
  display: block;
  border-top: 1px solid var(--border);
}

.chart-gridlines span:last-child {
  border-top-color: var(--border-strong);
}

.chart-groups {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: stretch;
  gap: 6px;
}

.chart-group {
  position: relative;
  display: flex;
  flex: 1;
  min-width: 0;
  align-items: flex-end;
  justify-content: center;
  gap: 4px;
  padding: 0 4px 22px;
}

.chart-bar {
  position: relative;
  width: min(16px, 38%);
  min-height: 2px;
  border-radius: 2px 2px 0 0;
  background: var(--gold);
  transition: opacity var(--ease), background var(--ease);
}

.chart-bar.expense {
  background: var(--chart-expense);
}

.chart-bar:hover {
  opacity: 0.78;
}

.chart-bar::after {
  content: attr(data-tip);
  display: none;
  position: absolute;
  left: 50%;
  bottom: calc(100% + 7px);
  transform: translateX(-50%) translateY(3px);
  padding: 5px 8px;
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  background: var(--surface-3);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 10px;
  white-space: nowrap;
  pointer-events: none;
  z-index: 3;
}

.chart-bar:hover::after {
  display: block;
  animation: chart-tip-in 140ms var(--ease) forwards;
}

@keyframes chart-tip-in {
  to { opacity: 1; transform: translateX(-50%) translateY(0); }
}

.chart-month {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.chart-empty {
  display: grid;
  place-items: center;
  height: 220px;
  color: var(--text-muted);
  font-size: 12px;
  text-align: center;
}

/* Status meter */
.status-meter {
  display: flex;
  height: 12px;
  margin: var(--space-2) 0 var(--space-5);
  border-radius: 3px;
  overflow: hidden;
  background: var(--track);
}

.status-seg {
  height: 100%;
  min-width: 0;
  transition: width 400ms var(--ease);
}

.status-seg.paid {
  background: var(--positive);
}
.status-seg.sent {
  background: var(--gold);
}
.status-seg.draft {
  background: var(--text-muted);
}
.status-seg.overdue {
  background: var(--negative);
}
.status-seg.empty {
  background: var(--track);
}

.status-legend {
  display: flex;
  flex-direction: column;
}

.status-row {
  display: grid;
  grid-template-columns: 12px 1fr auto auto;
  align-items: center;
  gap: var(--space-3);
  padding: 9px 0;
  border-top: 1px solid var(--border);
  font-size: 12.5px;
}

.status-row:first-child {
  border-top: 0;
}

.status-row .dot {
  width: 7px;
  height: 7px;
  border-radius: 2px;
}

.status-row .dot.paid {
  background: var(--positive);
}
.status-row .dot.sent {
  background: var(--gold);
}
.status-row .dot.draft {
  background: var(--text-muted);
}
.status-row .dot.overdue {
  background: var(--negative);
}

.status-row .label {
  color: var(--text-secondary);
}

.status-row .count {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
  min-width: 22px;
  text-align: right;
}

.status-row .amount {
  min-width: 82px;
  text-align: right;
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}

.status-total {
  display: flex;
  justify-content: space-between;
  margin-top: var(--space-3);
  padding-top: var(--space-3);
  border-top: 1px solid var(--border-strong);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.status-total strong {
  color: var(--text);
  font-family: var(--font-mono);
  font-weight: 500;
}

/* Attention list */
.attention-list {
  display: flex;
  flex-direction: column;
}

.attention-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-5);
  border-bottom: 1px solid var(--border);
  text-decoration: none;
  transition: background var(--ease);
}

.attention-item:last-child {
  border-bottom: 0;
}

.attention-item:hover {
  background: var(--surface-2);
}

.attention-mark {
  display: grid;
  place-items: center;
  width: 27px;
  height: 27px;
  flex: 0 0 27px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text-muted);
  margin-top: 1px;
}

.attention-mark.overdue {
  border-color: var(--negative-soft);
  background: var(--negative-soft);
  color: var(--negative);
}

.attention-mark.reimburse {
  border-color: var(--gold-line);
  background: var(--gold-soft);
  color: var(--gold);
}

.attention-mark.draft {
  color: var(--text-muted);
}

.attention-info {
  flex: 1;
  min-width: 0;
}

.attention-title {
  display: block;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.attention-meta {
  display: block;
  margin-top: 1px;
  font-size: 10.5px;
  color: var(--text-muted);
}

.attention-amount {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

/* ==========================================================================
   Tables
   ========================================================================== */

.table-wrap {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}

.data-table th {
  padding: 10px var(--space-5);
  border-bottom: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  text-align: left;
  white-space: nowrap;
}

.data-table td {
  padding: 12px var(--space-5);
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
  vertical-align: middle;
}

.data-table tbody tr:last-child td {
  border-bottom: 0;
}

.data-table tbody tr {
  transition: background var(--ease);
}

.data-table tbody tr:hover {
  background: var(--surface-2);
}

.data-table .num {
  text-align: right;
  font-family: var(--font-mono);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--text);
  white-space: nowrap;
}

.data-table .num strong {
  font-weight: 500;
}

.data-table .cell-main {
  color: var(--text);
  font-weight: 500;
}

.data-table .cell-sub {
  display: block;
  margin-top: 1px;
  font-size: 10.5px;
  color: var(--text-muted);
  font-weight: 400;
}

.data-table .actions-col {
  width: 1%;
  white-space: nowrap;
  text-align: right;
}

.data-table .select-col {
  width: 1%;
  padding-right: 0;
}

.check {
  width: 15px;
  height: 15px;
  margin: 0;
  accent-color: var(--gold);
  cursor: pointer;
  vertical-align: middle;
}

.check:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: 2px;
}

.select-all-mobile {
  display: none;
  align-items: center;
  gap: 6px;
  font-size: 11.5px;
  color: var(--text-secondary);
  white-space: nowrap;
  cursor: pointer;
}

.data-table tbody tr.selected {
  background: var(--gold-soft);
}

.data-table tbody tr.selected:hover {
  background: var(--gold-soft);
}

/* Bulk action bar */
.bulk-bar {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
  padding: 10px var(--space-5);
  border-bottom: 1px solid var(--gold-line);
  background: var(--gold-soft);
}

.bulk-bar[hidden] {
  display: none;
}

.bulk-info {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.bulk-count {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text);
  white-space: nowrap;
}

.text-button {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--gold);
  font: inherit;
  font-size: 11.5px;
  text-decoration: underline;
  text-underline-offset: 2px;
  cursor: pointer;
}

.text-button:hover {
  color: var(--gold-strong);
}

.bulk-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
  margin-left: auto;
}

.bulk-field {
  display: inline-flex;
}

.select-sm {
  width: auto;
  min-height: 30px;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 9px;
  font-size: 12px;
}

.row-actions {
  display: flex;
  justify-content: flex-end;
  gap: 2px;
  opacity: 0.75;
  transition: opacity var(--ease);
}

.data-table tr:hover .row-actions {
  opacity: 1;
}

.invoice-id {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
}

/* Status text (not pills) */
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  white-space: nowrap;
}

.status::before {
  content: "";
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: currentColor;
  flex: 0 0 auto;
}

.status.draft {
  color: var(--text-muted);
}
.status.sent {
  color: var(--gold);
}
.status.paid {
  color: var(--positive);
}
.status.overdue {
  color: var(--negative);
}
.status.cancelled {
  color: var(--text-muted);
  text-decoration: line-through;
}

.status.reimbursed {
  color: var(--positive);
}

.status.pending {
  color: var(--warning);
}

.status.plain::before {
  display: none;
}

/* Category tag */
.tag {
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border: 1px solid var(--border);
  border-radius: 3px;
  background: var(--surface-2);
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  white-space: nowrap;
}

/* Empty states */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-10) var(--space-5);
  text-align: center;
  color: var(--text-muted);
}

.empty-state svg {
  color: var(--text-muted);
  opacity: 0.55;
  margin-bottom: 2px;
}

.empty-state strong {
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 600;
}

.empty-state p {
  max-width: 320px;
  font-size: 11.5px;
  line-height: 1.6;
}

.empty-state .btn {
  margin-top: var(--space-2);
}

.empty-state.compact {
  padding: var(--space-6) var(--space-4);
}

/* ==========================================================================
   Filters / forms / controls
   ========================================================================== */

.toolbar {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-4) var(--space-5);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.toolbar .spacer {
  flex: 1;
}

.search-field {
  position: relative;
  flex: 1 1 230px;
  max-width: 330px;
  min-width: 180px;
}

.search-field svg {
  position: absolute;
  top: 50%;
  left: 11px;
  transform: translateY(-50%);
  color: var(--text-muted);
  pointer-events: none;
}

.search-field .input {
  width: 100%;
  padding-left: 33px;
}

.input,
.select {
  width: 100%;
  min-height: 36px;
  padding: 8px 11px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
  color: var(--text);
  font-size: 12.5px;
  transition: border-color var(--ease), background var(--ease);
}

.input:hover,
.select:hover {
  border-color: var(--border-strong);
}

.input:focus,
.select:focus {
  outline: none;
  border-color: var(--gold);
  box-shadow: 0 0 0 3px var(--gold-soft);
}

.input::placeholder {
  color: var(--text-muted);
}

.input[readonly] {
  color: var(--text-muted);
  background: var(--surface-2);
}

.select {
  appearance: none;
  padding-right: 30px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%238b8679' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 11px center;
  cursor: pointer;
}

textarea.input {
  min-height: 78px;
  resize: vertical;
  line-height: 1.5;
}

input[type="date"].input {
  color-scheme: inherit;
}

.pills {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 2px;
  max-width: 100%;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
  overflow-x: auto;
  scrollbar-width: none;
}

.pills::-webkit-scrollbar {
  display: none;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  border: 0;
  border-radius: 3px;
  background: transparent;
  color: var(--text-muted);
  font-size: 11.5px;
  font-weight: 500;
  white-space: nowrap;
  transition: background var(--ease), color var(--ease);
}

.pill:hover {
  color: var(--text);
}

.pill.active {
  background: var(--surface-3);
  color: var(--text);
}

.pill .pill-count {
  font-family: var(--font-mono);
  font-size: 9.5px;
  color: var(--text-muted);
}

.pill.active .pill-count {
  color: var(--gold);
}

.ledger-note {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
  margin-bottom: var(--space-4);
  font-size: 12px;
  color: var(--text-secondary);
}

.ledger-note .sep {
  width: 1px;
  height: 12px;
  background: var(--border-strong);
  margin: 0 4px;
}

.ledger-note strong {
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 11.5px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

/* Switch */
.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
}

.switch-copy {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.switch-copy strong {
  font-size: 12.5px;
  font-weight: 500;
}

.switch-copy span {
  font-size: 10.5px;
  color: var(--text-muted);
}

.switch {
  position: relative;
  width: 34px;
  height: 19px;
  flex: 0 0 34px;
  cursor: pointer;
}

.switch input {
  position: absolute;
  inset: 0;
  z-index: 2;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: pointer;
}

.switch-track {
  position: absolute;
  inset: 0;
  border: 1px solid var(--border-strong);
  border-radius: 20px;
  background: var(--surface-3);
  transition: background var(--ease), border-color var(--ease);
}

.switch-track::before {
  content: "";
  position: absolute;
  top: 3px;
  left: 3px;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--text-muted);
  transition: transform var(--ease), background var(--ease);
}

.switch input:checked + .switch-track {
  border-color: var(--gold);
  background: var(--gold-soft);
}

.switch input:checked + .switch-track::before {
  transform: translateX(15px);
  background: var(--gold);
}

.switch input:focus-visible + .switch-track {
  outline: 2px solid var(--gold);
  outline-offset: 2px;
}

/* ==========================================================================
   Modal
   ========================================================================== */

.modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: none;
  align-items: flex-start;
  justify-content: center;
  padding: 6vh var(--space-4) var(--space-8);
  overflow-y: auto;
  background: var(--overlay);
}

.modal-overlay.active {
  display: flex;
}

.modal-content {
  width: 100%;
  min-width: 0;
  max-width: 660px;
  margin: auto;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow: var(--shadow);
  animation: modal-in 180ms ease both;
}

.modal-content.wide {
  max-width: 820px;
}

@keyframes modal-in {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-5) var(--space-6) var(--space-4);
  border-bottom: 1px solid var(--border);
}

.modal-head .eyebrow {
  margin-bottom: 4px;
}

.modal-title {
  font-family: var(--font-display);
  font-size: 23px;
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.15;
}

.modal-close {
  flex: 0 0 auto;
}

.modal-body {
  padding: var(--space-5) var(--space-6);
}

.modal-foot {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-2);
  padding: var(--space-4) var(--space-6);
  border-top: 1px solid var(--border);
  background: var(--bg-raised);
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}

.modal-foot .danger-link {
  margin-right: auto;
}

.form-section {
  margin-bottom: var(--space-6);
}

.form-section:last-child {
  margin-bottom: 0;
}

.form-section-title {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 9.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
}

.form-grid .span-2 {
  grid-column: 1 / -1;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-width: 0;
}

.field-label {
  font-size: 11.5px;
  font-weight: 500;
  color: var(--text-secondary);
}

.field-label .req {
  color: var(--gold);
}

.field-hint {
  font-size: 10.5px;
  color: var(--text-muted);
}

.form-error {
  display: none;
  padding: 9px 11px;
  margin-bottom: var(--space-3);
  border: 1px solid var(--negative);
  border-radius: var(--radius-sm);
  background: var(--negative-soft);
  color: var(--negative);
  font-size: 11.5px;
}

.form-error.visible {
  display: block;
}

/* Line items */
.line-items {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.line-head,
.line-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 64px 100px 94px 30px;
  gap: var(--space-2);
  align-items: center;
  padding: 8px var(--space-3);
}

.line-head {
  border-bottom: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.line-row {
  border-bottom: 1px solid var(--border);
}

.line-row:last-child {
  border-bottom: 0;
}

.line-row .input {
  min-height: 32px;
  min-width: 0;
  width: 100%;
  padding: 6px 8px;
  font-size: 12px;
}

.line-amount {
  text-align: right;
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  padding-right: 2px;
}

.remove-item {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
}

.remove-item:hover {
  background: var(--negative-soft);
  color: var(--negative);
}

.line-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-3);
  border-top: 1px solid var(--border);
  background: var(--bg-raised);
}

/* Totals */
.form-totals {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 240px;
  margin-left: auto;
}

.total-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-5);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.total-row .value {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}

.total-row.grand {
  margin-top: 4px;
  padding-top: 8px;
  border-top: 1px solid var(--border-strong);
  color: var(--text);
  font-size: 13px;
  font-weight: 600;
}

.total-row.grand .value {
  color: var(--gold);
  font-size: 17px;
}

.totals-block {
  margin-top: var(--space-4);
  margin-left: auto;
  max-width: 280px;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
}

/* Invoice document */
.invoice-doc {
  padding: var(--space-2) 0;
}

.doc-head {
  display: flex;
  justify-content: space-between;
  gap: var(--space-6);
  padding-bottom: var(--space-5);
  margin-bottom: var(--space-5);
  border-bottom: 1px solid var(--border);
}

.doc-brand {
  display: flex;
  align-items: center;
  gap: 9px;
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 600;
}

.doc-brand img {
  width: 26px;
  height: 26px;
  border-radius: 6px;
}

.doc-number {
  margin-top: 4px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}

.doc-status {
  text-align: right;
}

.doc-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-5);
  margin-bottom: var(--space-5);
}

.doc-meta-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 3px;
}

.doc-meta-value {
  font-size: 12.5px;
  color: var(--text);
}

.doc-meta-value span {
  display: block;
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 1px;
}

.doc-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-bottom: var(--space-4);
}

.doc-table th {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border-strong);
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  text-align: left;
}

.doc-table td {
  padding: 9px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
}

.doc-table .num {
  text-align: right;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  color: var(--text);
}

.doc-notes {
  padding: var(--space-3) var(--space-4);
  margin-top: var(--space-4);
  border-left: 2px solid var(--gold);
  background: var(--gold-soft);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.doc-notes strong {
  display: block;
  margin-bottom: 2px;
  color: var(--text);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* Confirm dialog */
.confirm-overlay {
  position: fixed;
  inset: 0;
  z-index: 1100;
  display: none;
  align-items: center;
  justify-content: center;
  padding: var(--space-4);
  background: var(--overlay);
}

.confirm-overlay.active {
  display: flex;
}

.confirm-box {
  width: 100%;
  max-width: 380px;
  padding: var(--space-6);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow: var(--shadow);
  animation: modal-in 160ms ease both;
}

.confirm-kicker {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--negative);
  margin-bottom: var(--space-2);
}

.confirm-title {
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin-bottom: var(--space-2);
}

.confirm-message {
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
  margin-top: var(--space-6);
}

/* ==========================================================================
   Reports
   ========================================================================== */

.report-tabs {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  margin-bottom: var(--space-5);
  border-bottom: 1px solid var(--border);
}

.report-tab {
  position: relative;
  padding: 10px 14px;
  border: 0;
  background: transparent;
  color: var(--text-muted);
  font-size: 12.5px;
  font-weight: 500;
  white-space: nowrap;
  transition: color var(--ease);
}

.report-tab:hover {
  color: var(--text);
}

.report-tab.active {
  color: var(--text);
}

.report-tab.active::after {
  content: "";
  position: absolute;
  left: 10px;
  right: 10px;
  bottom: -1px;
  height: 2px;
  background: var(--gold);
}

.report-toolbar {
  display: flex;
  align-items: flex-end;
  gap: var(--space-3);
  flex-wrap: wrap;
  padding: var(--space-4) var(--space-5);
  border: 1px solid var(--border);
  border-radius: var(--radius) var(--radius) 0 0;
  background: var(--surface);
  border-bottom: 0;
}

.report-toolbar .field {
  gap: 4px;
}

.report-toolbar .field-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.report-toolbar .input {
  width: 150px;
  min-height: 34px;
}

.report-toolbar .spacer {
  flex: 1;
}

.range-presets {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
}

.range-preset {
  padding: 5px 9px;
  border: 0;
  border-radius: 3px;
  background: transparent;
  color: var(--text-muted);
  font-size: 10.5px;
  font-weight: 500;
  white-space: nowrap;
}

.range-preset:hover {
  color: var(--text);
  background: var(--surface-2);
}

.range-preset.active {
  background: var(--surface-3);
  color: var(--gold);
}

.report-sheet {
  border: 1px solid var(--border);
  border-radius: 0 0 var(--radius) var(--radius);
  background: var(--surface);
  padding: var(--space-8);
}

.report-masthead {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-6);
  padding-bottom: var(--space-5);
  margin-bottom: var(--space-6);
  border-bottom: 1px solid var(--border-strong);
}

.report-kicker {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: var(--space-2);
}

.report-title {
  font-family: var(--font-display);
  font-size: 29px;
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.1;
}

.report-period {
  margin-top: var(--space-2);
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--text-muted);
}

.report-headline {
  text-align: right;
  flex: 0 0 auto;
}

.report-headline-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.report-headline-value {
  margin-top: 4px;
  font-family: var(--font-mono);
  font-size: 31px;
  font-weight: 500;
  letter-spacing: -0.04em;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}

.report-headline-value.positive {
  color: var(--positive);
}

.report-headline-value.negative {
  color: var(--negative);
}

.report-headline-value.gold {
  color: var(--gold);
}

.report-summary {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-4);
  margin-bottom: var(--space-8);
}

.report-stat {
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
}

.report-stat-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.report-stat-value {
  margin-top: 6px;
  font-family: var(--font-mono);
  font-size: 20px;
  font-weight: 500;
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
}

.report-stat-value.positive {
  color: var(--positive);
}

.report-stat-value.negative {
  color: var(--negative);
}

.report-stat-value.gold {
  color: var(--gold);
}

.statement {
  margin-bottom: var(--space-8);
}

.statement:last-child {
  margin-bottom: 0;
}

.statement-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-4);
  padding-bottom: var(--space-2);
  margin-bottom: var(--space-3);
  border-bottom: 1px solid var(--border-strong);
}

.statement-title {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
}

.statement-note {
  font-family: var(--font-mono);
  font-size: 9.5px;
  color: var(--text-muted);
}

.statement-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}

.statement-table td {
  padding: 8px 4px;
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
  vertical-align: top;
}

.statement-table td:last-child {
  text-align: right;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.statement-table tr:last-child td {
  border-bottom: 0;
}

.statement-table .entry-date {
  width: 92px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--text-muted);
  white-space: nowrap;
}

.statement-table .entry-category {
  width: 110px;
}

.statement-table tr.subtotal td {
  padding-top: var(--space-3);
  border-top: 1px solid var(--border-strong);
  border-bottom: 0;
  color: var(--text);
  font-weight: 600;
}

.statement-table tr.subtotal td:last-child {
  color: var(--text);
  font-weight: 500;
}

.statement-table tr.net td {
  padding: 12px 4px;
  border-top: 1px solid var(--gold-line);
  border-bottom: 1px solid var(--gold-line);
  background: var(--gold-soft);
  color: var(--text);
  font-weight: 600;
}

.statement-table tr.net td:last-child {
  color: var(--gold);
  font-size: 15px;
  font-weight: 500;
}

.statement-table .muted {
  color: var(--text-muted);
  font-style: italic;
}

.balance-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-8);
}

.balance-grid .statement {
  margin-bottom: 0;
}

.balance-stack {
  display: flex;
  flex-direction: column;
  gap: var(--space-8);
}

.balance-equation {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  flex-wrap: wrap;
  margin-top: var(--space-6);
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--text-muted);
}

.balance-equation strong {
  color: var(--text);
  font-weight: 500;
}

.tax-note {
  margin-top: var(--space-5);
  padding: var(--space-3) var(--space-4);
  border-left: 2px solid var(--gold);
  background: var(--gold-soft);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.report-empty {
  padding: var(--space-10);
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
}

/* ==========================================================================
   Chat assistant
   ========================================================================== */

.chat-panel {
  position: fixed;
  right: var(--space-5);
  bottom: var(--space-5);
  z-index: 900;
  display: flex;
  flex-direction: column;
  width: 348px;
  max-height: min(540px, calc(100vh - 40px));
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow: var(--shadow);
  opacity: 0;
  visibility: hidden;
  transform: translateY(10px);
  transition: opacity var(--ease), transform var(--ease), visibility var(--ease);
}

.chat-panel.open {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-4);
  border-bottom: 1px solid var(--border);
}

.chat-heading {
  display: flex;
  align-items: center;
  gap: 9px;
}

.chat-mark {
  display: block;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border-radius: 7px;
  object-fit: contain;
}

.chat-title {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 600;
}

.chat-status {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  color: var(--text-muted);
}

.chat-status .dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--positive);
}

.chat-status.local .dot {
  background: var(--warning);
}

.typing-dots {
  display: inline-flex;
  gap: 4px;
  padding: 2px 0;
}

.typing-dots span {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--text-muted);
  animation: typing-bounce 1s infinite ease-in-out;
}

.typing-dots span:nth-child(2) {
  animation-delay: 0.15s;
}

.typing-dots span:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes typing-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.45; }
  30% { transform: translateY(-3px); opacity: 1; }
}

.chat-messages {
  flex: 1;
  min-height: 180px;
  overflow-y: auto;
  padding: var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.message {
  max-width: 88%;
  font-size: 12px;
  line-height: 1.6;
}

.message-content {
  padding: 9px 11px;
  border-radius: var(--radius-sm);
  white-space: pre-line;
}

.message.assistant {
  align-self: flex-start;
}

.message.assistant .message-content {
  border: 1px solid var(--border);
  border-left: 2px solid var(--gold-line);
  background: var(--bg-raised);
  color: var(--text-secondary);
}

.message.user {
  align-self: flex-end;
}

.message.user .message-content {
  background: var(--gold-soft);
  border: 1px solid var(--gold-line);
  color: var(--text);
}

.message b {
  color: var(--text);
  font-weight: 600;
}

.chat-input-area {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-3);
  border-top: 1px solid var(--border);
}

.chat-input-area input {
  flex: 1;
  min-width: 0;
  min-height: 34px;
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-raised);
  font-size: 12px;
  outline: none;
}

.chat-input-area input:focus {
  border-color: var(--gold);
  box-shadow: 0 0 0 3px var(--gold-soft);
}

.chat-input-area button {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border: 1px solid var(--gold);
  border-radius: var(--radius-sm);
  background: var(--gold);
  color: var(--on-gold);
}

.chat-input-area button:hover {
  background: var(--gold-strong);
  border-color: var(--gold-strong);
}

/* --- Chat page --- */

/* The chat owns the whole tab: no card inset, composer pinned to the bottom. */
[data-page="chat"] .app-main {
  display: flex;
  flex-direction: column;
  height: 100dvh;
  min-height: 0;
}

[data-page="chat"] .content {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  max-width: none;
  padding: 0;
  contain: none;
}

.chat-page {
  display: grid;
  grid-template-columns: 264px minmax(0, 1fr);
  flex: 1 1 auto;
  min-height: 0;
  background: var(--surface);
  overflow: hidden;
}

.chat-rail {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border-right: 1px solid var(--border);
  background: var(--bg);
}

.chat-rail__head {
  padding: var(--space-3);
}

.chat-rail__new {
  width: 100%;
  justify-content: center;
}

.chat-rail__list {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin: 0;
  padding: 0 var(--space-2) var(--space-2);
  list-style: none;
}

.chat-rail__button {
  display: block;
  width: 100%;
  padding: 9px var(--space-3);
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: 12.5px;
  text-align: left;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: background var(--ease), color var(--ease);
}

.chat-rail__button:hover {
  background: var(--surface-2);
  color: var(--text);
}

.chat-rail__button.is-active {
  background: var(--gold-soft);
  color: var(--gold);
}

.chat-rail__foot {
  padding: var(--space-3);
  border-top: 1px solid var(--border);
  font-size: 10.5px;
  color: var(--text-muted);
}

.chat-main {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}

.chat-main.is-dragging {
  outline: 2px dashed var(--gold-line);
  outline-offset: -6px;
}

.chat-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--border);
}

.chat-bar__title-wrap {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  min-width: 0;
}

.chat-bar__title {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chat-bar__actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

/* Custom model dropdown */
.model-picker {
  position: relative;
}

.model-picker__button {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font-size: 12.5px;
  font-weight: 500;
  transition: border-color var(--ease), background var(--ease);
}

.model-picker__button:hover {
  border-color: var(--gold-line);
}

.model-picker__dot {
  width: 7px;
  height: 7px;
  flex: 0 0 7px;
  border-radius: 50%;
  background: var(--positive);
}

.model-picker__dot[data-provider="gemini"] {
  background: #4c8bf5;
}

.model-picker__dot[data-provider="auto"] {
  background: var(--gold);
}

.model-picker__caret {
  color: var(--text-muted);
  transition: transform var(--ease);
}

.model-picker.is-open .model-picker__caret {
  transform: rotate(180deg);
}

.model-picker__menu {
  position: absolute;
  right: 0;
  top: calc(100% + 6px);
  z-index: 20;
  width: 224px;
  margin: 0;
  padding: 4px;
  list-style: none;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow);
}

.model-picker__option {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  cursor: pointer;
}

.model-picker__option:hover {
  background: var(--surface-2);
}

.model-picker__option[aria-selected="true"] {
  background: var(--gold-soft);
}

.model-picker__option-title {
  font-size: 12.5px;
  color: var(--text);
}

.model-picker__option-hint {
  margin-left: auto;
  font-size: 10px;
  color: var(--text-muted);
}

/* Thread */
.chat-thread {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  padding: var(--space-6) var(--space-5);
}

.chat-empty {
  margin: auto;
  max-width: 560px;
  padding: var(--space-6) 0;
  text-align: center;
}

.chat-empty__mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  height: 52px;
  margin-bottom: var(--space-4);
  border: 1px solid var(--gold-line);
  border-radius: 14px;
  background: var(--gold-soft);
  color: var(--gold);
  font-family: var(--font-display);
  font-size: 26px;
  font-weight: 700;
}

.chat-empty__title {
  margin-bottom: var(--space-2);
  font-family: var(--font-display);
  font-size: 22px;
}

.chat-empty__sub {
  margin-bottom: var(--space-5);
  color: var(--text-secondary);
  font-size: 13px;
}

.chat-suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  justify-content: center;
}

.chat-suggestions button {
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-secondary);
  font-size: 12px;
  transition: border-color var(--ease), color var(--ease);
}

.chat-suggestions button:hover {
  border-color: var(--gold-line);
  color: var(--text);
}

.chat-msg {
  display: flex;
  gap: var(--space-3);
  max-width: 100%;
}

.chat-msg--user {
  justify-content: flex-end;
}

.chat-msg__avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border: 1px solid var(--gold-line);
  border-radius: 50%;
  background: var(--gold-soft);
  color: var(--gold);
  font-family: var(--font-display);
  font-size: 14px;
  font-weight: 700;
}

.chat-msg__body {
  min-width: 0;
}

.chat-user-bubble {
  max-width: min(640px, 100%);
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 14px 14px 3px 14px;
  background: var(--surface-2);
  color: var(--text);
  font-size: 13.5px;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.chat-ai-text {
  color: var(--text);
  font-size: 13.5px;
  line-height: 1.7;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.chat-note {
  align-self: center;
  max-width: 640px;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--warning);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text-muted);
  font-size: 12px;
  text-align: center;
}

/* Composer */
.chat-composer {
  flex: 0 0 auto;
  padding: var(--space-3) var(--space-4) var(--space-4);
  border-top: 1px solid var(--border);
}

.chat-composer__form {
  display: flex;
  align-items: flex-end;
  gap: var(--space-2);
  padding: 6px 6px 6px 8px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--bg);
  transition: border-color var(--ease);
}

.chat-composer__form:focus-within {
  border-color: var(--gold-line);
}

.chat-composer__attach,
.chat-composer__send {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  transition: background var(--ease), color var(--ease);
}

.chat-composer__attach:hover {
  background: var(--surface-2);
  color: var(--text);
}

.chat-composer__send {
  border-color: var(--gold);
  background: var(--gold);
  color: var(--on-gold);
}

.chat-composer__send:hover {
  border-color: var(--gold-strong);
  background: var(--gold-strong);
}

.chat-composer__send:disabled {
  opacity: 0.5;
}

.chat-composer__input {
  flex: 1 1 auto;
  min-width: 0;
  max-height: 180px;
  padding: 8px 4px;
  border: 0;
  background: transparent;
  color: var(--text);
  font: inherit;
  font-size: 13.5px;
  line-height: 1.6;
  resize: none;
  outline: none;
}

.chat-composer__note {
  margin: var(--space-2) 2px 0;
}

.ov-chat__files {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin: 0 0 var(--space-2);
  padding: 0;
  list-style: none;
}

.ov-chat__files[hidden] {
  display: none;
}

.ov-chat__file {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 6px 4px 11px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  font-size: 11.5px;
}

.ov-chat__file-name {
  color: var(--text);
}

.ov-chat__file-meta {
  color: var(--text-muted);
}

.ov-chat__file-remove {
  display: inline-flex;
  padding: 2px;
  border: 0;
  background: transparent;
  color: var(--text-muted);
}

.ov-chat__file-remove:hover {
  color: var(--text);
}

@media (max-width: 820px) {
  .chat-page {
    grid-template-columns: 1fr;
    grid-template-rows: auto minmax(0, 1fr);
  }
  .chat-rail {
    flex-direction: row;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2);
    border-right: 0;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
  }
  .chat-rail__head {
    flex: 0 0 auto;
    padding: 0;
  }
  .chat-rail__new {
    width: auto;
    white-space: nowrap;
    padding: 8px var(--space-3);
  }
  .chat-rail__list {
    flex-direction: row;
    flex-wrap: nowrap;
    min-width: 0;
    max-height: none;
    overflow-x: auto;
    overflow-y: hidden;
    gap: var(--space-2);
    padding: 0;
  }
  .chat-rail__button {
    width: auto;
    max-width: 62vw;
    white-space: nowrap;
    border: 1px solid var(--border);
  }
  .chat-rail__foot {
    display: none;
  }
  .chat-bar {
    flex-wrap: wrap;
  }
}

/* ==========================================================================
   Toasts
   ========================================================================== */

.toasts {
  position: fixed;
  right: var(--space-5);
  bottom: var(--space-5);
  z-index: 1200;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  width: min(340px, calc(100vw - 32px));
  pointer-events: none;
}

.toast {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  box-shadow: var(--shadow);
  pointer-events: auto;
  animation: toast-in 180ms ease both;
}

.toast.leaving {
  animation: toast-out 160ms ease both;
}

.toast-mark {
  display: grid;
  place-items: center;
  width: 20px;
  height: 20px;
  flex: 0 0 20px;
  border-radius: 50%;
  margin-top: 1px;
}

.toast.success .toast-mark {
  background: var(--positive-soft);
  color: var(--positive);
}

.toast.error .toast-mark {
  background: var(--negative-soft);
  color: var(--negative);
}

.toast.info .toast-mark {
  background: var(--gold-soft);
  color: var(--gold);
}

.toast-message {
  flex: 1;
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.toast-message strong {
  display: block;
  color: var(--text);
  font-size: 12px;
  font-weight: 600;
}

@keyframes toast-in {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes toast-out {
  to {
    opacity: 0;
    transform: translateY(6px);
  }
}

/* ==========================================================================
   Utilities
   ========================================================================== */

.mono {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

.muted {
  color: var(--text-muted);
}

.nowrap {
  white-space: nowrap;
}

.hidden {
  display: none !important;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* ==========================================================================
   Responsive
   ========================================================================== */

@media (max-width: 1240px) {
  .content {
    padding: var(--space-8) var(--space-8) var(--space-10);
  }
  .metric-band {
    grid-template-columns: repeat(3, 1fr);
  }
  .metric:nth-child(4) {
    border-left: 0;
    border-top: 1px solid var(--border);
  }
  .metric:nth-child(5) {
    border-top: 1px solid var(--border);
  }
}

@media (max-width: 1080px) {
  .sidebar {
    transform: translateX(-100%);
    transition: transform 220ms ease;
  }
  .sidebar.open {
    transform: translateX(0);
  }
  .sidebar .sidebar-close {
    display: inline-flex;
  }
  .app-main {
    margin-left: 0;
  }
  .mobile-bar {
    display: flex;
  }
  .sidebar-scrim.open {
    display: block;
    opacity: 1;
  }
  .dash-grid,
  .dash-grid.lower {
    grid-template-columns: 1fr;
  }
  .balance-grid {
    grid-template-columns: 1fr;
  }
  .report-sheet {
    padding: var(--space-6);
  }
}

@media (max-width: 760px) {
  .content {
    padding: var(--space-5) var(--space-4) var(--space-8);
  }
  .page-head {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-4);
    padding-bottom: var(--space-5);
  }
  .page-actions {
    width: 100%;
    flex-wrap: wrap;
  }
  .page-actions .btn {
    flex: 1;
  }
  .metric-band {
    grid-template-columns: 1fr 1fr;
  }
  .metric {
    padding: var(--space-4);
  }
  .metric + .metric {
    border-left: 0;
  }
  .metric:nth-child(even) {
    border-left: 1px solid var(--border);
  }
  .metric:nth-child(n + 3) {
    border-top: 1px solid var(--border);
  }
  .metric:nth-child(5) {
    grid-column: 1 / -1;
  }
  .toolbar {
    padding: var(--space-3) var(--space-4);
  }
  .select-all-mobile {
    display: inline-flex;
  }
  .bulk-bar {
    padding: 10px var(--space-4);
  }
  .bulk-actions {
    width: 100%;
    margin-left: 0;
  }
  .bulk-field {
    flex: 1 1 140px;
  }
  .bulk-field .select {
    width: 100%;
  }
  .bulk-actions .btn {
    flex: 1 1 auto;
  }
  .search-field {
    max-width: none;
    flex-basis: 100%;
  }
  .data-table th,
  .data-table td {
    padding-left: var(--space-4);
    padding-right: var(--space-4);
  }
  .data-table th:first-child,
  .data-table td:first-child {
    padding-left: var(--space-4);
  }
  /* Stack ledger rows into readable cards on phones so amounts,
     statuses and row actions never require horizontal scrolling. */
  .invoices-table,
  .expenses-table {
    display: block;
    width: 100%;
  }
  .invoices-table thead,
  .expenses-table thead {
    display: none;
  }
  .invoices-table tbody,
  .expenses-table tbody {
    display: block;
  }
  .invoices-table tbody tr,
  .expenses-table tbody tr {
    display: grid;
    border-bottom: 1px solid var(--border);
    padding: 14px var(--space-4);
  }
  .invoices-table tbody tr:last-child,
  .expenses-table tbody tr:last-child {
    border-bottom: 0;
  }
  .invoices-table tbody td,
  .expenses-table tbody td {
    border-bottom: 0;
    padding: 0;
  }
  .invoices-table tbody tr {
    grid-template-columns: auto minmax(0, 1fr) auto;
    grid-template-areas:
      "select id status"
      "client client client"
      "issued issued due"
      "amount amount actions";
    gap: 4px 12px;
    align-items: center;
  }
  .invoices-table tbody td:nth-child(1) { grid-area: select; align-self: start; }
  .invoices-table tbody td:nth-child(2) { grid-area: id; }
  .invoices-table tbody td:nth-child(3) { grid-area: client; margin-bottom: 2px; }
  .invoices-table tbody td:nth-child(4) { grid-area: issued; }
  .invoices-table tbody td:nth-child(5) { grid-area: due; text-align: right; }
  .invoices-table tbody td:nth-child(6) { grid-area: status; justify-self: end; }
  .invoices-table tbody td:nth-child(7) { grid-area: amount; text-align: left; margin-top: 4px; }
  .invoices-table tbody td:nth-child(8) { grid-area: actions; justify-self: end; margin-top: 4px; }
  .invoices-table tbody td:nth-child(4)::before,
  .invoices-table tbody td:nth-child(5)::before {
    font-family: var(--font-mono);
    font-size: 8.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
  }
  .invoices-table tbody td:nth-child(4)::before { content: "Issued "; }
  .invoices-table tbody td:nth-child(5)::before { content: "Due "; }
  .expenses-table tbody tr {
    grid-template-columns: auto minmax(0, 1fr) auto;
    grid-template-areas:
      "select date amount"
      "desc desc desc"
      "cat cat reimb"
      "actions actions actions";
    gap: 5px 12px;
    align-items: center;
  }
  .expenses-table tbody td:nth-child(1) { grid-area: select; align-self: start; }
  .expenses-table tbody td:nth-child(2) { grid-area: date; font-size: 11px; color: var(--text-muted); align-self: center; }
  .expenses-table tbody td:nth-child(3) { grid-area: desc; margin: 2px 0; }
  .expenses-table tbody td:nth-child(4) { grid-area: cat; justify-self: start; }
  .expenses-table tbody td:nth-child(5) { grid-area: reimb; justify-self: end; }
  .expenses-table tbody td:nth-child(6) { grid-area: amount; text-align: right; }
  .expenses-table tbody td:nth-child(7) { grid-area: actions; justify-self: end; margin-top: 4px; }
  .invoices-table tbody tr:has(td[colspan]),
  .expenses-table tbody tr:has(td[colspan]) {
    display: block;
    padding: 0;
  }
  .invoices-table tbody td[colspan],
  .expenses-table tbody td[colspan] {
    display: block;
  }
  .report-toolbar .input {
    width: 100%;
  }
  .report-toolbar .field {
    flex: 1 1 140px;
  }
  .range-presets {
    width: 100%;
    overflow-x: auto;
  }
  .report-sheet {
    padding: var(--space-5);
  }
  .report-masthead {
    flex-direction: column;
  }
  .report-headline {
    text-align: left;
  }
  .report-summary {
    grid-template-columns: 1fr;
    gap: var(--space-2);
  }
  .form-grid {
    grid-template-columns: 1fr;
  }
  .modal-overlay {
    padding: 0;
    align-items: flex-end;
  }
  .modal-content {
    max-width: none;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    border-bottom: 0;
    max-height: 92vh;
    overflow-y: auto;
  }
  .modal-head,
  .modal-body {
    padding-left: var(--space-4);
    padding-right: var(--space-4);
  }
  .modal-foot {
    padding: var(--space-3) var(--space-4);
    border-radius: 0;
    position: sticky;
    bottom: 0;
  }
  .line-head,
  .line-row {
    grid-template-columns: minmax(0, 1fr) 48px 80px 76px 28px;
    gap: 4px;
    padding: 6px 8px;
  }
  .line-head {
    font-size: 8px;
  }
  .chat-panel {
    right: var(--space-3);
    bottom: var(--space-3);
    left: var(--space-3);
    width: auto;
    max-height: calc(100vh - 24px);
  }
  .toasts {
    right: var(--space-3);
    bottom: var(--space-3);
  }
  .doc-meta {
    grid-template-columns: 1fr;
    gap: var(--space-3);
  }
  .doc-head {
    flex-direction: column;
    gap: var(--space-3);
  }
  .doc-status {
    text-align: left;
  }
  .data-table .actions-col {
    width: auto;
  }
}

@media (max-width: 430px) {
  .metric-band {
    grid-template-columns: 1fr;
  }
  .metric:nth-child(even) {
    border-left: 0;
  }
  .metric:nth-child(n + 2) {
    border-top: 1px solid var(--border);
  }
  .metric:nth-child(5) {
    grid-column: auto;
  }
  .report-tabs {
    overflow-x: auto;
  }
  .report-tab {
    padding: 10px;
    font-size: 11.5px;
  }
  .chart-shell {
    height: 190px;
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* ==========================================================================
   Pagination, settings, imports, activity
   ========================================================================== */

.pagination {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  flex-wrap: wrap;
  padding: var(--space-3) var(--space-5);
  border-top: 1px solid var(--border);
  font-size: 11.5px;
  color: var(--text-muted);
}

.pagination-size {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
}

.pagination-count {
  font-family: var(--font-mono);
  font-size: 10.5px;
}

.pagination-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-left: auto;
}

.pagination-page {
  min-width: 72px;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--text-secondary);
}

.pagination .btn[disabled] {
  opacity: 0.4;
  cursor: not-allowed;
}

.text-button[hidden] {
  display: none;
}

/* Settings */
.settings-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: var(--space-5);
  margin-bottom: var(--space-5);
}

.settings-form .form-grid {
  margin-bottom: var(--space-5);
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
  padding-top: var(--space-4);
  border-top: 1px solid var(--border);
}

.settings-stack {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.settings-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  padding-bottom: var(--space-4);
  border-bottom: 1px solid var(--border);
}

.settings-row:last-of-type {
  padding-bottom: 0;
  border-bottom: 0;
}

.settings-row strong {
  display: block;
  margin-bottom: 2px;
  font-size: 13px;
  font-weight: 600;
}

.settings-row p {
  max-width: 46ch;
  font-size: 11.5px;
}

.settings-file {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
  justify-content: flex-end;
}

.settings-file .input[type="file"] {
  max-width: 240px;
  padding: 6px 8px;
  font-size: 11px;
}

.settings-copy {
  max-width: 88ch;
  font-size: 12.5px;
  color: var(--text-secondary);
}

.settings-copy + .settings-copy {
  margin-top: var(--space-3);
}

.settings-copy code {
  padding: 1px 4px;
  border: 1px solid var(--border);
  border-radius: 3px;
  background: var(--surface-2);
  font-family: var(--font-mono);
  font-size: 11px;
}

.settings-form .form-error,
.settings-stack .form-error {
  margin-bottom: var(--space-3);
}

.settings-form .form-error.visible,
.settings-stack .form-error.visible {
  display: block;
}

/* CSV import */
.import-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-6);
}

.import-block {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.import-block .field-hint {
  margin: var(--space-2) 0 var(--space-3);
  line-height: 1.6;
  /* Reserve three lines so both columns start their dropzones at the same
     height, even when one hint wraps to fewer lines than the other. */
  min-height: calc(3 * 1.6em);
}

.import-block .upload-zone {
  flex: 1 1 auto;
}

.import-block .settings-file {
  margin-top: var(--space-3);
}

.import-preview {
  margin-top: var(--space-4);
}

.import-summary {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2) var(--space-4);
  padding: var(--space-3);
  margin-bottom: var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.import-summary .negative {
  color: var(--negative);
}

.import-errors {
  padding: var(--space-3);
  margin-bottom: var(--space-3);
  border: 1px solid var(--negative);
  border-radius: var(--radius-sm);
  background: var(--negative-soft);
  color: var(--negative);
  font-size: 11.5px;
}

.import-errors ul {
  margin: var(--space-2) 0 0;
  padding-left: 18px;
  line-height: 1.6;
}

.import-table-wrap {
  max-height: 320px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.import-table {
  font-size: 11.5px;
}

.import-table th {
  position: sticky;
  top: 0;
  background: var(--surface-2);
  z-index: 1;
}

.import-table .row-error td {
  background: var(--negative-soft);
}

.import-table .row-error-detail td {
  padding-top: 0;
  color: var(--negative);
  font-size: 10.5px;
}

.import-mapping {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
  font-size: 11px;
}

.import-mapping .muted {
  margin-right: var(--space-1);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 9.5px;
}

.import-actions {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
  margin-top: var(--space-3);
}

.import-message {
  padding: 9px 11px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.import-message.error {
  border-color: var(--negative);
  background: var(--negative-soft);
  color: var(--negative);
}

.import-message.success {
  border-color: var(--positive);
  background: var(--positive-soft);
  color: var(--positive);
}

/* Assistant additions */
.chat-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}

.chat-privacy {
  padding: var(--space-2) var(--space-4);
  border-top: 1px solid var(--border);
  font-size: 10px;
  line-height: 1.5;
  color: var(--text-muted);
}

/* Report assumptions */
.assumption-list {
  margin: var(--space-2) 0 0;
  padding-left: 18px;
  line-height: 1.6;
}

.assumption-list li + li {
  margin-top: 2px;
}

@media (max-width: 1080px) {
  .settings-grid,
  .import-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .settings-row {
    flex-direction: column;
  }
  .settings-file {
    justify-content: flex-start;
    width: 100%;
  }
  .pagination {
    gap: var(--space-2);
  }
  .pagination-actions {
    margin-left: 0;
    width: 100%;
    justify-content: space-between;
  }
}

/* ==========================================================================
   SpendWise
   ========================================================================== */

.sw-range {
  display: flex;
  align-items: flex-end;
  gap: var(--space-4);
  flex-wrap: wrap;
  padding: var(--space-4) var(--space-5);
  margin-bottom: var(--space-5);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.sw-range .field {
  min-width: 150px;
}

.sw-range-note {
  padding-bottom: 6px;
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--text-muted);
}

.sw-toggle {
  display: inline-flex;
  gap: var(--space-1);
}

.sw-toggle .pill {
  cursor: pointer;
}

.chart-wrap-compact {
  padding: 0;
}

.sw-cash-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.sw-cash-cell {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

.sw-cash-label {
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.sw-cash-value {
  font-family: var(--font-mono);
  font-size: 15px;
  color: var(--text);
}

.sw-cash-value.negative {
  color: var(--negative);
}

.sw-share {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  justify-content: flex-end;
}

.sw-share-bar {
  position: relative;
  display: block;
  width: 54px;
  height: 5px;
  border-radius: 2px;
  background: var(--surface-3);
  overflow: hidden;
}

.sw-share-bar span {
  position: absolute;
  inset: 0 auto 0 0;
  background: var(--gold);
}

.sw-row-click {
  cursor: pointer;
}

.sw-row-click:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: -2px;
}

.sw-budget-layout {
  display: grid;
  grid-template-columns: minmax(0, 320px) minmax(0, 1fr);
  gap: var(--space-6);
  align-items: start;
}

.sw-budget-form .form-grid {
  grid-template-columns: 1fr;
}

.sw-budget-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.sw-budget-column {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-width: 0;
}

.sw-budget {
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.sw-budget-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-3);
}

.sw-budget-amount {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.sw-budget-amount strong {
  color: var(--text);
}

.sw-meter {
  position: relative;
  height: 6px;
  border-radius: 3px;
  background: var(--surface-3);
  overflow: hidden;
}

.sw-meter span {
  position: absolute;
  inset: 0 auto 0 0;
  background: var(--gold);
}

.sw-meter.near span {
  background: var(--gold-strong);
}

.sw-meter.over span {
  background: var(--negative);
}

.sw-budget-foot {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  flex-wrap: wrap;
  margin-top: var(--space-3);
  font-size: 11.5px;
}

.sw-state {
  font-family: var(--font-mono);
  font-size: 10.5px;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}

.sw-state.ok {
  color: var(--positive);
}

.sw-state.near {
  color: var(--gold-strong);
}

.sw-state.over {
  color: var(--negative);
}

.sw-budget-actions {
  display: flex;
  gap: var(--space-3);
  margin-left: auto;
}

.text-button.danger {
  color: var(--negative);
}

.sw-rule-layout {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
}

.sw-rule-form .form-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.sw-suggestions {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.sw-suggestion-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-4);
  flex-wrap: wrap;
}

.sw-suggestion-head .muted {
  font-size: 11.5px;
}

.sw-insight-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
  gap: var(--space-6);
}

.sw-insights {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  margin: 0;
  padding: 0;
  list-style: none;
}

.sw-insight {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: var(--space-3) var(--space-4);
  border-left: 2px solid var(--border-strong);
  background: var(--surface-2);
}

.sw-insight.notice {
  border-left-color: var(--gold);
}

.sw-insight.positive {
  border-left-color: var(--positive);
}

.sw-insight-title {
  font-size: 12.5px;
  font-weight: 600;
}

.sw-insight-detail {
  font-size: 11.5px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.sw-insight-side {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.sw-insight-side .statement-title {
  margin-top: var(--space-2);
}

.sw-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: 11.5px;
}

.sw-list li {
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--border);
}

.sw-list li:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.sw-list .cell-main {
  display: block;
  color: var(--text);
}

.sw-list .cell-sub {
  display: block;
  margin-top: 2px;
  color: var(--text-muted);
  line-height: 1.55;
}

.sw-table th,
.sw-table td {
  padding: 9px var(--space-3);
}

.sw-table td:first-child,
.sw-table th:first-child {
  padding-left: var(--space-4);
}

.sw-table td:last-child,
.sw-table th:last-child {
  padding-right: var(--space-4);
}

.sw-table .num {
  white-space: nowrap;
}

.sw-table .cell-main {
  overflow-wrap: anywhere;
}

.sw-vendor-filter {
  min-width: 170px;
}

.sw-import-layout {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.sw-modal-summary {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2) var(--space-5);
  margin-bottom: var(--space-4);
  padding: var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
  font-size: 11.5px;
  color: var(--text-secondary);
}

.sw-modal-summary strong {
  color: var(--text);
}

@media (max-width: 1080px) {
  .sw-budget-layout,
  .sw-insight-layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .sw-cash-row {
    grid-template-columns: 1fr;
  }
  .sw-rule-form .form-grid {
    grid-template-columns: 1fr;
  }
  .sw-budget-head {
    flex-direction: column;
    gap: var(--space-2);
  }
}

/* ==========================================================================
   Print
   ========================================================================== */

@media print {
  :root,
  [data-theme="dark"],
  [data-theme="light"] {
    --bg: #ffffff;
    --surface: #ffffff;
    --surface-2: #ffffff;
    --bg-raised: #ffffff;
    --text: #111111;
    --text-secondary: #333333;
    --text-muted: #666666;
    --border: #cccccc;
    --border-strong: #888888;
    --gold: #7d5c12;
  }
  body {
    background: #ffffff;
  }
  .sidebar,
  .mobile-bar,
  .page-head,
  .report-tabs,
  .report-toolbar,
  .chat-panel,
  .toasts,
  .modal-overlay,
  .confirm-overlay,
  .sidebar-scrim {
    display: none !important;
  }
  .app-main {
    margin: 0;
  }
  .content {
    padding: 0;
    max-width: none;
  }
  .report-sheet {
    border: 0;
    padding: 0;
  }
  .statement-table tr {
    break-inside: avoid;
  }
  .report-headline-value,
  .report-stat-value,
  .statement-table td:last-child {
    color: #111111 !important;
  }
}

/* ==========================================================================
   Finance Agent — shared components
   Custom upload control and AI summary panel. Flat surfaces, gold accents,
   matching the existing design tokens in style.css.
   ========================================================================== */

/* --- Custom upload control ------------------------------------------------ */

.upload-zone {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-8) var(--space-6);
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface-2);
  color: var(--text-secondary);
  text-align: center;
  cursor: pointer;
  transition: border-color var(--ease), background var(--ease), color var(--ease);
}

.upload-zone--compact {
  padding: var(--space-4) var(--space-4);
  gap: var(--space-1);
}

.upload-zone:hover {
  border-color: var(--gold-line);
  color: var(--text);
}

.upload-zone:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: 2px;
  border-color: var(--gold-line);
}

.upload-zone.is-over {
  border-color: var(--gold);
  background: var(--gold-soft);
  color: var(--text);
}

.upload-zone.is-busy {
  cursor: progress;
  opacity: 0.75;
}

.upload-zone__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  border-radius: var(--radius-sm);
  background: var(--gold-soft);
  color: var(--gold);
}

.upload-zone--compact .upload-zone__icon {
  width: 30px;
  height: 30px;
}

.upload-title {
  font-weight: 600;
  color: var(--text);
}

.upload-zone .field-hint {
  margin: 0;
}

.upload-input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.upload-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-top: var(--space-3);
}

.upload-list.is-empty {
  margin-top: 0;
}

.upload-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
}

.upload-item__name {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
}

.upload-item__meta {
  flex: 0 0 auto;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
}

.upload-item__remove {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
  color: var(--text-secondary);
  cursor: pointer;
  transition: color var(--ease), border-color var(--ease), background var(--ease);
}

.upload-item__remove:hover:not(:disabled) {
  color: var(--negative);
  border-color: var(--negative);
}

.upload-item__remove:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: 1px;
}

.upload-item__remove:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.upload-actions {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
  margin-top: var(--space-4);
}

.upload-progress {
  display: inline-flex;
  align-items: center;
  gap: var(--space-3);
  font-size: 13px;
  color: var(--text-secondary);
}

.upload-progress__track {
  position: relative;
  width: 120px;
  height: 4px;
  border-radius: 999px;
  background: var(--track);
  overflow: hidden;
}

.upload-progress__bar {
  position: absolute;
  inset: 0 auto 0 0;
  width: 0;
  background: var(--gold);
  transition: width var(--ease);
}

.upload-note {
  margin-top: var(--space-2);
  font-size: 13px;
  color: var(--text-muted);
}

.upload-note.is-error {
  color: var(--negative);
}

/* --- AI summary panel ----------------------------------------------------- */

.ai-panel {
  border-left: 2px solid var(--gold-line);
}

.panel-actions {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
}

.ai-state {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: 3px var(--space-3);
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  background: var(--surface-2);
}

.ai-state::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}

.ai-state[data-state="loading"] { color: var(--warning); }
.ai-state[data-state="ready"] { color: var(--positive); }
.ai-state[data-state="stale"] { color: var(--warning); }
.ai-state[data-state="error"] { color: var(--negative); }

.ai-summary {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.ai-summary__headline {
  font-family: var(--font-display);
  font-size: 19px;
  line-height: 1.4;
  color: var(--text);
}

.ai-summary__label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.ai-summary__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.ai-summary__item {
  position: relative;
  padding-left: var(--space-5);
  color: var(--text-secondary);
}

.ai-summary__item::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0.62em;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--gold);
}

.ai-summary__caution::before {
  background: var(--warning);
}

.ai-summary__computed {
  padding: var(--space-3);
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.ov-spend-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.ov-spend-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.ov-spend-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.ov-spend-value {
  font-family: var(--font-display);
  font-size: 20px;
  color: var(--text);
}

@media (max-width: 640px) {
  .upload-zone {
    padding: var(--space-5) var(--space-4);
  }
  .upload-actions {
    align-items: stretch;
  }
  .upload-actions .btn {
    flex: 1 1 auto;
  }
}

/* ==========================================================================
   SpendWise AI — What-if Simulator, Subscription Optimiser, analysis, docs
   ========================================================================== */

/* Insights layout gains a second column of content. */
.sw-insight-main {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  min-width: 0;
}

.sw-ai-block {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding-top: var(--space-4);
  border-top: 1px dashed var(--border);
}

/* `display: flex` above would otherwise beat the [hidden] attribute, so the
   suggestions block would show before the user asked for them. */
.sw-ai-block[hidden] {
  display: none;
}

.ai-tag {
  margin-left: 6px;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.ai-ref {
  display: inline-block;
  margin-top: 2px;
  font-size: 11px;
  color: var(--text-muted);
}

.ai-summary-text {
  color: var(--text-secondary);
  line-height: 1.6;
}

/* What-if simulator */
.sw-sim-body {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: var(--space-6);
  align-items: start;
}

.sw-sim-controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  min-width: 0;
}

.sw-sim-baseline {
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.sw-sim-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: var(--space-3);
}

.sw-sim-base-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-3);
}

.sw-sim-base-grid span {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.sw-sim-base-grid em {
  font-style: normal;
  font-size: 11px;
  color: var(--text-muted);
}

.sw-sim-base-grid strong {
  font-family: var(--font-display);
  font-size: 18px;
}

.range-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 110px;
  gap: var(--space-3);
  align-items: center;
}

.range {
  width: 100%;
  accent-color: var(--gold);
  height: 22px;
}

.input-num {
  text-align: right;
}

.ai-input-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--space-2);
  align-items: center;
}

.sw-sim-results {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-width: 0;
}

.sw-sim-cards {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
}

.sw-sim-card {
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.sw-sim-card-title {
  margin-bottom: var(--space-3);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.sw-sim-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-3);
  padding: 3px 0;
  font-size: 13px;
}

.sw-sim-row > span {
  color: var(--text-secondary);
}

.sw-sim-row.sw-sim-diff {
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.is-positive { color: var(--positive); }
.is-negative { color: var(--negative); }

/* Subscription optimiser */
.sw-subs-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.sw-subs-summary span {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.sw-subs-summary em {
  font-style: normal;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.sw-subs-summary strong {
  font-family: var(--font-display);
  font-size: 18px;
}

.sw-subs-list,
.sw-docs-list,
.sw-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.sw-subs-item,
.sw-docs-item {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.sw-subs-main,
.sw-docs-main {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  min-width: 0;
}

.sw-subs-cost {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
  font-variant-numeric: tabular-nums;
}

.sw-subs-cost em,
.sw-list-value em {
  font-style: normal;
  font-size: 11px;
  color: var(--text-muted);
}

.sw-subs-detail {
  margin-top: var(--space-4);
  padding: var(--space-4);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface-2);
}

.sw-subs-detail-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.sw-subs-alts {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  margin-top: var(--space-4);
}

.sw-alt {
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
}

.sw-alt header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-3);
}

.sw-alt h4 {
  font-size: 14px;
  font-weight: 600;
}

.sw-alt-saving {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
}

.sw-alt-block {
  margin-bottom: var(--space-2);
}

.sw-alt-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 2px;
}

.sw-alt-block ul,
.sw-alt-flags {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 13px;
  color: var(--text-secondary);
}

.sw-alt-source {
  margin-top: var(--space-3);
  font-size: 12px;
  color: var(--text-muted);
  overflow-wrap: anywhere;
}

.sw-alt-flags {
  margin-top: var(--space-2);
}

.sw-alt-flag {
  color: var(--warning);
  font-size: 12px;
}

.sw-list-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.sw-list-main {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.sw-list-value {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  text-align: right;
}

.sw-docs-status {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.sw-docs-status[data-state="processed"] { color: var(--positive); }
.sw-docs-status[data-state="failed"] { color: var(--negative); }
.sw-docs-status[data-state="imported"] { color: var(--gold); }

.btn-small {
  padding: 4px var(--space-3);
  font-size: 12px;
}

/* Local-storage note inside Settings > AI privacy */
.privacy-local {
  display: flex;
  gap: var(--space-3);
  padding: var(--space-4);
  margin-bottom: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-2);
}

.privacy-local strong {
  font-size: 13px;
}

.privacy-local p {
  margin-top: 2px;
}

.privacy-local-dot {
  flex: none;
  width: 8px;
  height: 8px;
  margin-top: 6px;
  border-radius: 50%;
  background: var(--positive);
}

@media (max-width: 900px) {
  .sw-sim-body {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 640px) {
  .sw-sim-cards {
    grid-template-columns: minmax(0, 1fr);
  }
  .sw-sim-base-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .range-row {
    grid-template-columns: minmax(0, 1fr) 90px;
  }
}



/* ==========================================================================
   Finance Agent — Documents page
   Extraction review cards, validation flags, and the document library.
   ========================================================================== */

.doc-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-2);
  padding: var(--space-5);
  margin-bottom: var(--space-4);
}

.doc-card:last-child {
  margin-bottom: 0;
}

.doc-card__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  flex-wrap: wrap;
}

.doc-card__title {
  font-weight: 600;
  overflow-wrap: anywhere;
}

.doc-card__meta {
  font-size: 12px;
  color: var(--text-muted);
}

.doc-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: 3px var(--space-3);
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  white-space: nowrap;
}

.doc-badge--processed { color: var(--positive); border-color: var(--positive); }
.doc-badge--failed { color: var(--negative); border-color: var(--negative); }
.doc-badge--imported { color: var(--gold); border-color: var(--gold-line); }

.doc-type {
  display: inline-block;
  margin-top: var(--space-2);
  font-family: var(--font-mono);
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--gold);
}

.doc-summary {
  margin-top: var(--space-3);
  color: var(--text-secondary);
}

.doc-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--space-3);
  margin-top: var(--space-4);
}

.doc-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.doc-field__label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.doc-field__value {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font: inherit;
}

.doc-field__value:focus-visible {
  outline: 2px solid var(--gold);
  outline-offset: 1px;
  border-color: var(--gold-line);
}

.doc-field--uncertain .doc-field__value {
  border-color: var(--warning);
}

.doc-field--uncertain .doc-field__label::after {
  content: " · check";
  color: var(--warning);
  font-weight: 500;
}

.doc-lines {
  margin-top: var(--space-4);
}

.doc-lines table {
  width: 100%;
  border-collapse: collapse;
}

.doc-lines th,
.doc-lines td {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--border);
  text-align: left;
}

.doc-lines input {
  width: 100%;
  padding: 4px var(--space-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font: inherit;
}

.doc-flag {
  display: flex;
  gap: var(--space-2);
  margin-top: var(--space-3);
  padding: var(--space-3);
  border: 1px solid var(--warning);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 13px;
}

.doc-duplicate {
  display: flex;
  gap: var(--space-2);
  margin-top: var(--space-2);
  padding: var(--space-3);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 13px;
}

.doc-error {
  margin-top: var(--space-3);
  color: var(--negative);
  font-size: 13px;
}

.doc-import-bar {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
  margin-top: var(--space-4);
  padding-top: var(--space-4);
  border-top: 1px solid var(--border);
}

.doc-import-bar select {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  font: inherit;
}

.doc-manual {
  margin-top: var(--space-3);
  padding-top: var(--space-3);
  border-top: 1px dashed var(--border);
}

.doc-manual .doc-import-bar {
  margin-top: var(--space-2);
  padding-top: 0;
  border-top: 0;
}

.doc-manual .doc-fields:empty {
  display: none;
}

.doc-row-actions {
  display: flex;
  gap: var(--space-2);
  justify-content: flex-end;
}

@media (max-width: 640px) {
  .doc-import-bar {
    align-items: stretch;
  }
  .doc-import-bar .btn,
  .doc-import-bar select {
    flex: 1 1 auto;
  }
}
"""


APP_JS = r"""
// === upload.js ===
/* Custom file-upload widget for Finance Agent.
   The native <input type="file"> stays in the DOM, hidden and focusable, and is
   kept in sync so existing code that reads input.files keeps working. The
   visible control is the dropzone. */

(function () {
  const ICON_REMOVE =
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>';

  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return value + ' B';
    if (value < 1024 * 1024) return (value / 1024).toFixed(1) + ' KB';
    return (value / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function escapeText(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[character]));
  }

  function createUploadWidget(options) {
    const config = options || {};
    const zone = document.getElementById(config.dropzoneId);
    const input = document.getElementById(config.inputId);
    const list = config.listId ? document.getElementById(config.listId) : null;
    const note = config.noteId ? document.getElementById(config.noteId) : null;
    if (!zone || !input) return null;

    const multiple = config.multiple !== false;
    let files = [];
    let busy = false;

    if (config.accept) input.setAttribute('accept', config.accept);
    input.multiple = multiple;
    zone.setAttribute('role', 'button');
    zone.setAttribute('tabindex', zone.getAttribute('tabindex') || '0');
    if (config.label) zone.setAttribute('aria-label', config.label);

    function setNote(text, kind) {
      const target = note || zone.querySelector('.upload-note');
      if (!target) return;
      target.textContent = text || '';
      target.classList.toggle('is-error', kind === 'error');
    }

    function syncInput() {
      try {
        const transfer = new DataTransfer();
        files.forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
      } catch (error) {
        /* DataTransfer is unavailable (rare); the widget still tracks files. */
      }
    }

    function render() {
      if (!list) return;
      list.innerHTML = files
        .map((file, index) => `
          <div class="upload-item">
            <span class="upload-item__name" title="${escapeText(file.name)}">${escapeText(file.name)}</span>
            <span class="upload-item__meta">${formatBytes(file.size)}</span>
            <button class="upload-item__remove" type="button" data-remove="${index}"
              aria-label="Remove ${escapeText(file.name)}"${busy ? ' disabled' : ''}>${ICON_REMOVE}</button>
          </div>`)
        .join('');
      list.classList.toggle('is-empty', files.length === 0);
    }

    function notify() {
      if (typeof config.onChange === 'function') config.onChange(files.slice());
    }

    function addFiles(incoming) {
      if (busy) return;
      const incomingList = Array.from(incoming || []);
      if (!incomingList.length) return;
      const rejected = [];
      const accepted = [];
      incomingList.forEach((file) => {
        if (config.maxBytes && file.size > config.maxBytes) {
          rejected.push(`"${file.name}" is ${formatBytes(file.size)}; the limit is ${formatBytes(config.maxBytes)}.`);
          return;
        }
        accepted.push(file);
      });
      if (!multiple) files = accepted.slice(-1);
      else files = files.concat(accepted);
      if (rejected.length) setNote(rejected.join(' '), 'error');
      else setNote('');
      syncInput();
      render();
      notify();
    }

    function clear() {
      files = [];
      try { input.value = ''; } catch (error) { /* ignore */ }
      syncInput();
      render();
      notify();
    }

    function openDialog() {
      if (busy) return;
      input.click();
    }

    zone.addEventListener('click', (event) => {
      if (event.target.closest('.upload-item__remove')) return;
      if (event.target === input) return;
      openDialog();
    });

    zone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openDialog();
      }
    });

    zone.addEventListener('dragover', (event) => {
      event.preventDefault();
      if (!busy) zone.classList.add('is-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('is-over'));
    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      zone.classList.remove('is-over');
      if (event.dataTransfer && event.dataTransfer.files) addFiles(event.dataTransfer.files);
    });

    input.addEventListener('change', () => {
      addFiles(input.files);
    });

    if (list) {
      list.addEventListener('click', (event) => {
        const button = event.target.closest('[data-remove]');
        if (!button || busy) return;
        files.splice(Number(button.dataset.remove), 1);
        syncInput();
        render();
        notify();
      });
    }

    function setBusy(value) {
      busy = Boolean(value);
      zone.classList.toggle('is-busy', busy);
      zone.setAttribute('aria-busy', busy ? 'true' : 'false');
      input.disabled = busy;
      list?.querySelectorAll('button').forEach((button) => { button.disabled = busy; });
    }

    render();

    return {
      getFiles: () => files.slice(),
      clear,
      setBusy,
      setNote,
      formatBytes,
    };
  }

  window.createUploadWidget = createUploadWidget;
  window.formatFileBytes = formatBytes;
})();


// === app.js ===
/* Finance Agent — consolidated frontend */

/* --- Assistant history storage --- */
class ChatStorage {
  static KEY = 'accounting_agent_chats';

  static getChats() {
    try {
      return JSON.parse(localStorage.getItem(this.KEY)) || [];
    } catch {
      return [];
    }
  }

  static saveChats(chats) {
    localStorage.setItem(this.KEY, JSON.stringify(chats));
  }

  static addMessage(role, content) {
    const chats = this.getChats();
    chats.push({ role, content, timestamp: new Date().toISOString() });
    if (chats.length > 200) chats.splice(0, chats.length - 200);
    this.saveChats(chats);
    return chats;
  }

  static clearChats() {
    localStorage.removeItem(this.KEY);
  }
}


/* --- Shared UI, API, assistant --- */
/* ==========================================================================
   Finance Agent — shared UI behaviour
   ========================================================================== */

const THEME_KEY = 'aa-theme';

/* --- Application settings (currency, formatting, business details) --- */
const DEFAULT_SETTINGS = {
  currency_code: 'USD',
  currency_symbol: '$',
  locale: 'en-US',
  default_tax_rate: '0.20',
  date_format: 'medium',
  business_name: '',
  business_address: '',
  business_email: '',
  payment_details: '',
};

let appSettings = { ...DEFAULT_SETTINGS };

function setAppSettings(settings) {
  appSettings = { ...DEFAULT_SETTINGS, ...(settings || {}) };
}

function currencySymbol() {
  return appSettings.currency_symbol || appSettings.currency_code || '$';
}

async function loadAppSettings() {
  try {
    const payload = await api('/api/settings');
    setAppSettings(payload?.settings);
  } catch (error) {
    console.error('Load settings error:', error);
    setAppSettings(DEFAULT_SETTINGS);
  }
  return appSettings;
}

const settingsReady = loadAppSettings();

/* --------------------------------------------------------------------------
   Theme
   -------------------------------------------------------------------------- */

function currentTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

function syncThemeControls() {
  const theme = currentTheme();
  const next = theme === 'dark' ? 'light' : 'dark';
  document.querySelectorAll('[data-theme-label]').forEach((el) => {
    el.textContent = next === 'light' ? 'Light mode' : 'Dark mode';
  });
  document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
    btn.setAttribute('aria-label', `Switch to ${next} theme`);
    btn.setAttribute('title', `Switch to ${next} theme`);
  });
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', theme === 'dark' ? '#0a0a0c' : '#f4f2ec');
}

function setTheme(theme) {
  const next = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch (e) {
    /* Storage can be unavailable in private contexts. */
  }
  syncThemeControls();
}

document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
  btn.addEventListener('click', () => {
    setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  });
});
syncThemeControls();

/* --------------------------------------------------------------------------
   Mobile navigation
   -------------------------------------------------------------------------- */

const sidebar = document.getElementById('sidebar');
const sidebarScrim = document.getElementById('sidebarScrim');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarClose = document.getElementById('sidebarClose');

function openSidebar() {
  if (!sidebar) return;
  sidebar.classList.add('open');
  sidebarScrim?.classList.add('open');
  sidebarToggle?.setAttribute('aria-expanded', 'true');
  document.body.style.overflow = 'hidden';
}

function closeSidebar() {
  if (!sidebar) return;
  sidebar.classList.remove('open');
  sidebarScrim?.classList.remove('open');
  sidebarToggle?.setAttribute('aria-expanded', 'false');
  document.body.style.overflow = '';
}

sidebarToggle?.addEventListener('click', openSidebar);
sidebarClose?.addEventListener('click', closeSidebar);
sidebarScrim?.addEventListener('click', closeSidebar);
sidebar?.querySelectorAll('.nav-item').forEach((link) => {
  link.addEventListener('click', () => {
    if (window.matchMedia('(max-width: 1080px)').matches) closeSidebar();
  });
});

/* --------------------------------------------------------------------------
   Navigation
   -------------------------------------------------------------------------- */

function navigateTo(path) {
  window.location.href = path;
}

/* --------------------------------------------------------------------------
   API
   -------------------------------------------------------------------------- */

async function api(url, options = {}) {
  const config = {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  };
  if (config.body && typeof config.body === 'object') {
    config.body = JSON.stringify(config.body);
  }
  const res = await fetch(url, config);
  if (!res.ok) {
    let detail = res.statusText || 'Request failed';
    try {
      const payload = await res.json();
      if (payload && payload.detail) {
        detail = Array.isArray(payload.detail)
          ? payload.detail.map((item) => item.msg || String(item)).join(', ')
          : payload.detail;
      }
    } catch (e) {
      /* Response had no JSON body. */
    }
    const error = new Error(detail);
    error.status = res.status;
    throw error;
  }
  if (res.status === 204) return null;
  return res.json();
}

async function fetchAllRecords(path, params = {}) {
  const collected = [];
  let page = 1;
  let pages = 1;
  do {
    const query = new URLSearchParams({ ...params, page: String(page), page_size: '200' });
    const payload = await api(`${path}?${query.toString()}`);
    collected.push(...(payload.items || []));
    pages = Number(payload.pages) || 0;
    page += 1;
  } while (page <= pages && page <= 100);
  return collected;
}

/* --------------------------------------------------------------------------
   Formatting helpers
   -------------------------------------------------------------------------- */

function esc(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

function numberValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function fmt(value, digits = 2) {
  return numberValue(value).toLocaleString(appSettings.locale || 'en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function money(value, digits = 2) {
  const n = numberValue(value);
  const sign = n < 0 ? '−' : '';
  return `${sign}${currencySymbol()}${fmt(Math.abs(n), digits)}`;
}

function fmtCompact(value) {
  const n = numberValue(value);
  const abs = Math.abs(n);
  const sign = n < 0 ? '−' : '';
  const symbol = currencySymbol();
  if (abs >= 1000000) return `${sign}${symbol}${(abs / 1000000).toFixed(1)}m`;
  if (abs >= 1000) return `${sign}${symbol}${(abs / 1000).toFixed(abs >= 10000 ? 0 : 1)}k`;
  return `${sign}${symbol}${Math.round(abs)}`;
}

function parseISO(value) {
  if (!value) return null;
  const match = String(value).slice(0, 10).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

function todayISO() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

function csvCell(value) {
  return `"${String(value == null ? '' : value).replace(/"/g, '""')}"`;
}

function downloadCsv(filename, rows) {
  const csv = `\ufeff${rows.map((row) => row.map(csvCell).join(',')).join('\r\n')}`;
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
}

function fmtDate(value, options = {}) {
  const date = parseISO(value);
  if (!date) return '—';
  const mode = options.long ? 'long' : appSettings.date_format || 'medium';
  if (mode === 'iso') {
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  }
  if (mode === 'short' && !options.long) {
    return date.toLocaleDateString(appSettings.locale || 'en-US', {
      day: '2-digit',
      month: '2-digit',
      year: '2-digit',
    });
  }
  const month = mode === 'long' ? 'long' : 'short';
  return date.toLocaleDateString(appSettings.locale || 'en-US', {
    day: mode === 'long' ? 'numeric' : '2-digit',
    month,
    year: options.year === false ? undefined : 'numeric',
  });
}

function fmtMonth(value) {
  const date = parseISO(value);
  if (!date) return '—';
  return date.toLocaleDateString(appSettings.locale || 'en-US', { month: 'short' });
}

function invoiceRef(id) {
  return `INV-${String(id).padStart(4, '0')}`;
}

function statusLabel(status) {
  return String(status || 'draft').replace(/_/g, ' ');
}

function statusHtml(status) {
  const safe = ['draft', 'sent', 'paid', 'overdue', 'cancelled'].includes(status) ? status : 'draft';
  return `<span class="status ${safe}">${esc(statusLabel(safe))}</span>`;
}

function debounce(fn, delay = 220) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

/* --------------------------------------------------------------------------
   Toasts
   -------------------------------------------------------------------------- */

const toastHost = document.getElementById('toasts');

function toast(message, type = 'success', title = '') {
  if (!toastHost) return;
  const item = document.createElement('div');
  item.className = `toast ${type}`;
  const icons = {
    success:
      '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
    error:
      '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 8v5M12 17h.01"/></svg>',
    info:
      '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 16v-5M12 8h.01"/></svg>',
  };
  item.innerHTML = `
    <span class="toast-mark">${icons[type] || icons.info}</span>
    <span class="toast-message">${title ? `<strong>${esc(title)}</strong>` : ''}${esc(message)}</span>
  `;
  toastHost.appendChild(item);
  const remove = () => {
    item.classList.add('leaving');
    window.setTimeout(() => item.remove(), 170);
  };
  window.setTimeout(remove, 4200);
  item.addEventListener('click', remove);
}

/* --------------------------------------------------------------------------
   Modal
   -------------------------------------------------------------------------- */

const modalOverlay = document.getElementById('modalOverlay');
const modalContent = document.getElementById('modalContent');
let modalReturnFocus = null;

function modalHeader(eyebrow, title) {
  return `
    <div class="modal-head">
      <div>
        <p class="eyebrow">${esc(eyebrow)}</p>
        <h2 class="modal-title">${esc(title)}</h2>
      </div>
      <button class="icon-btn modal-close" type="button" data-modal-close aria-label="Close dialog">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    </div>
  `;
}

function openModal(html, options = {}) {
  if (!modalOverlay || !modalContent) return;
  modalReturnFocus = document.activeElement;
  modalContent.innerHTML = html;
  modalContent.classList.toggle('wide', Boolean(options.wide));
  modalOverlay.classList.add('active');
  modalOverlay.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
  const target = modalContent.querySelector(
    'input:not([type="hidden"]):not([readonly]), select, textarea, button:not([data-modal-close])'
  );
  if (target) {
    target.focus({ preventScroll: true });
  }
}

function closeModal() {
  if (!modalOverlay) return;
  modalOverlay.classList.remove('active');
  modalOverlay.setAttribute('aria-hidden', 'true');
  modalContent.innerHTML = '';
  if (!document.getElementById('confirmOverlay')?.classList.contains('active')) {
    document.body.style.overflow = '';
  }
  if (modalReturnFocus && typeof modalReturnFocus.focus === 'function') {
    modalReturnFocus.focus({ preventScroll: true });
  }
  modalReturnFocus = null;
}

modalOverlay?.addEventListener('click', (event) => {
  if (event.target === modalOverlay || event.target.closest('[data-modal-close]')) {
    closeModal();
  }
});

/* --------------------------------------------------------------------------
   Confirmation dialog
   -------------------------------------------------------------------------- */

const confirmOverlay = document.getElementById('confirmOverlay');
const confirmTitle = document.getElementById('confirmTitle');
const confirmKicker = document.getElementById('confirmKicker');
const confirmMessage = document.getElementById('confirmMessage');
const confirmAccept = document.getElementById('confirmAccept');
const confirmCancel = document.getElementById('confirmCancel');
let confirmReturnFocus = null;
let confirmResolve = null;

function confirmDialog({
  title = 'Confirm action',
  message = '',
  confirmLabel = 'Confirm',
  kicker = 'Please confirm',
  danger = true,
} = {}) {
  return new Promise((resolve) => {
    if (!confirmOverlay) {
      resolve(window.confirm(message || title));
      return;
    }
    confirmReturnFocus = document.activeElement;
    confirmTitle.textContent = title;
    confirmKicker.textContent = kicker;
    confirmMessage.textContent = message;
    confirmAccept.textContent = confirmLabel;
    confirmAccept.className = danger ? 'btn btn-danger' : 'btn btn-primary';
    confirmOverlay.classList.add('active');
    confirmOverlay.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    confirmResolve = resolve;
    confirmCancel.focus({ preventScroll: true });
  });
}

function settleConfirm(result) {
  if (!confirmOverlay) return;
  confirmOverlay.classList.remove('active');
  confirmOverlay.setAttribute('aria-hidden', 'true');
  if (!modalOverlay?.classList.contains('active')) {
    document.body.style.overflow = '';
  }
  if (confirmReturnFocus && typeof confirmReturnFocus.focus === 'function') {
    confirmReturnFocus.focus({ preventScroll: true });
  }
  confirmReturnFocus = null;
  const resolve = confirmResolve;
  confirmResolve = null;
  if (resolve) resolve(result);
}

confirmAccept?.addEventListener('click', () => settleConfirm(true));
confirmCancel?.addEventListener('click', () => settleConfirm(false));
confirmOverlay?.addEventListener('click', (event) => {
  if (event.target === confirmOverlay) settleConfirm(false);
});

/* --------------------------------------------------------------------------
   Keyboard handling
   -------------------------------------------------------------------------- */

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (confirmOverlay?.classList.contains('active')) {
    settleConfirm(false);
    return;
  }
  if (modalOverlay?.classList.contains('active')) {
    closeModal();
    return;
  }
  if (document.getElementById('chatPanel')?.classList.contains('open')) {
    closeChat();
    return;
  }
  if (sidebar?.classList.contains('open')) closeSidebar();
});

/* --------------------------------------------------------------------------
   Assistant
   -------------------------------------------------------------------------- */

const chatPanel = document.getElementById('chatPanel');
const chatToggle = document.getElementById('chatToggle');
const chatClose = document.getElementById('chatClose');
const chatClear = document.getElementById('chatClear');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');
const chatMessages = document.getElementById('chatMessages');
const chatPrivacy = document.querySelector('.chat-privacy');

function chatMessageHtml(role, content) {
  return `<div class="message ${role === 'user' ? 'user' : 'assistant'}"><div class="message-content">${content}</div></div>`;
}

function appendChatMessage(role, content, persist = true) {
  if (!chatMessages) return;
  chatMessages.insertAdjacentHTML('beforeend', chatMessageHtml(role, content));
  chatMessages.scrollTop = chatMessages.scrollHeight;
  if (persist && window.ChatStorage) ChatStorage.addMessage(role, content);
}

function renderChatHistory() {
  if (!chatMessages) return;
  chatMessages.innerHTML = '';
  let history = [];
  try {
    history = window.ChatStorage ? ChatStorage.getChats() : [];
  } catch (e) {
    history = [];
  }
  if (!history.length) {
    appendChatMessage(
      'assistant',
      'Hello. I can help you navigate the ledger — ask about invoices, expenses, reports, or your current position.',
      false
    );
  } else {
    history.slice(-40).forEach((message) => {
      appendChatMessage(message.role, message.content, false);
    });
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function openChat() {
  renderChatHistory();
  chatPanel?.classList.add('open');
  chatPanel?.setAttribute('aria-hidden', 'false');
  chatToggle?.setAttribute('aria-expanded', 'true');
  window.setTimeout(() => chatInput?.focus(), 120);
}

function closeChat() {
  chatPanel?.classList.remove('open');
  chatPanel?.setAttribute('aria-hidden', 'true');
  chatToggle?.setAttribute('aria-expanded', 'false');
}

chatToggle?.addEventListener('click', () => {
  if (chatPanel?.classList.contains('open')) closeChat();
  else openChat();
});
chatClose?.addEventListener('click', closeChat);

function clearChatConversation() {
  if (window.ChatStorage) ChatStorage.clearChats();
  if (chatMessages) chatMessages.innerHTML = '';
  appendChatMessage(
    'assistant',
    'Conversation cleared. Ask me anything about the ledger. Remember that using the assistant can send a ledger summary to Google Gemini.',
    false
  );
}

chatClear?.addEventListener('click', clearChatConversation);

function handleChatSend() {
  const text = chatInput?.value.trim();
  if (!text) return;
  appendChatMessage('user', esc(text));
  chatInput.value = '';
  processChatMessage(text);
}

chatSend?.addEventListener('click', handleChatSend);
chatInput?.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    handleChatSend();
  }
});

let assistantStatus = { configured: false, model: '' };

async function loadAssistantStatus() {
  const statusText = document.getElementById('chatStatusText');
  const statusBox = document.querySelector('.chat-status');
  try {
    assistantStatus = await api('/api/assistant/status');
  } catch (error) {
    assistantStatus = { configured: false, model: '' };
  }
  if (statusText) {
    statusText.textContent = assistantStatus.configured
      ? `Gemini connected · ${assistantStatus.model || 'model set'}`
      : 'Gemini not configured · local guidance only';
  }
  statusBox?.classList.toggle('local', !assistantStatus.configured);
  if (chatPrivacy && assistantStatus.privacy_notice) {
    chatPrivacy.textContent = assistantStatus.privacy_notice;
  }
}

function appendThinking() {
  if (!chatMessages) return null;
  const item = document.createElement('div');
  item.className = 'message assistant thinking';
  item.innerHTML = '<div class="message-content"><span class="typing-dots" aria-label="Assistant is thinking"><span></span><span></span><span></span></span></div>';
  chatMessages.appendChild(item);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return item;
}

function assistantAnswerHtml(text) {
  return esc(text).replace(/\r?\n/g, '<br>');
}

async function processChatMessage(text) {
  const lower = text.toLowerCase();

  if (/^(clear|reset)( chat| conversation)?$/.test(lower.trim())) {
    clearChatConversation();
    return;
  }

  if (assistantStatus.configured) {
    const thinking = appendThinking();
    try {
      const result = await api('/api/assistant/chat', {
        method: 'POST',
        body: { message: text },
      });
      thinking?.remove();
      appendChatMessage('assistant', assistantAnswerHtml(result.answer));
      return;
    } catch (error) {
      thinking?.remove();
      console.error('Assistant error:', error);
      const note = assistantStatus.configured
        ? 'Gemini is unavailable right now, so this answer comes from the app’s built-in guidance. '
        : 'Gemini is not configured, so this answer comes from the app’s built-in guidance. ';
      appendChatMessage('assistant', assistantAnswerHtml(note + error.message));
      toast(error.message, 'error', 'Assistant unavailable');
    }
  } else {
    appendChatMessage(
      'assistant',
      'Gemini is not configured, so I can only explain the app and read the totals stored here. Add GEMINI_API_KEY to .env and restart to enable full answers.',
      false
    );
  }

  await localChatAnswer(lower);
}

async function localChatAnswer(lower) {
  if (/(spen|spend|budget|categor|vendor|expense)/.test(lower)) {
    try {
      const data = await api('/api/spendwise/overview?period=month');
      const stats = data.stats || {};
      const categories = data.categories || [];
      const vendors = data.vendors || [];
      const budgets = data.budgets || [];
      const parts = [];
      if (stats.total) {
        parts.push(
          `Spending this month is <b>${money(stats.total)}</b> across ${stats.count} transaction${
            stats.count === 1 ? '' : 's'
          }.`
        );
      } else {
        parts.push('No spending has been recorded this month.');
      }
      if (categories.length) {
        parts.push(
          `The largest category is <b>${categoryLabel(categories[0].category)}</b> at ${money(
            categories[0].total
          )} (${categories[0].share}%).`
        );
      }
      if (vendors.length) {
        parts.push(`The largest vendor is <b>${vendors[0].vendor}</b> at ${money(vendors[0].total)}.`);
      }
      if (budgets.length) {
        const over = budgets.filter((row) => row.exceeded);
        parts.push(
          over.length
            ? `${over.length} budget${over.length === 1 ? ' is' : 's are'} over: ${over
                .map((row) => `${row.name} (${row.percent}%)`)
                .join(', ')}.`
            : `All ${budgets.length} budget${budgets.length === 1 ? '' : 's'} are within their limits.`
        );
      }
      const insight = (data.insights || [])[0];
      if (insight) parts.push(insight.title);
      parts.push('Open <b>SpendWise</b> for the full breakdown.');
      appendChatMessage('assistant', parts.join('<br>'));
      return;
    } catch (err) {
      appendChatMessage(
        'assistant',
        'I could not read the spending analytics just now. Please open the SpendWise page and try again.'
      );
      return;
    }
  }

  if (/(summary|position|outstanding|overdue|revenue|profit|how much|balance)/.test(lower)) {
    try {
      const summary = await api('/api/dashboard/summary');
      appendChatMessage(
        'assistant',
        `At the moment you have invoiced <b>${money(summary.total_invoiced)}</b> including tax, collected <b>${money(
          summary.total_paid
        )}</b>, and <b>${money(summary.total_outstanding)}</b> is outstanding.<br>` +
          `Expenses total <b>${money(summary.total_expenses)}</b>. Accrual net revenue less expenses gives a net profit of <b>${money(
            summary.net_profit
          )}</b>.<br>` +
          `${summary.overdue_count} invoice${summary.overdue_count === 1 ? '' : 's'} overdue and <b>${money(
            summary.unreimbursed_expenses
          )}</b> awaiting reimbursement.`
      );
      return;
    } catch (err) {
      appendChatMessage('assistant', 'I could not read the ledger summary just now. Please open the Overview page and try again.');
      return;
    }
  }

  if (/(document|upload|receipt|extract|scan|attachment)/.test(lower)) {
    try {
      const payload = await api('/api/documents');
      const documents = payload.documents || [];
      if (!documents.length) {
        appendChatMessage(
          'assistant',
          'No documents have been uploaded yet. Open <b>Documents</b> to upload a PDF, spreadsheet, image, or text file; the app reads the details and files invoices, receipts, and expenses automatically. Everything else stays in the library for review.'
        );
        return;
      }
      const processed = documents.filter((doc) => doc.status === 'processed').length;
      const failed = documents.filter((doc) => doc.status === 'failed').length;
      const linked = documents.filter((doc) => doc.record && doc.record.kind);
      const recent = documents.slice(0, 3).map((doc) => {
        const kind = doc.record && doc.record.kind ? ` → ${doc.record.kind} ${doc.record.reference || ''}` : '';
        return `• ${esc(doc.filename)} (${esc(doc.status || 'unknown')}${kind})`;
      });
      appendChatMessage(
        'assistant',
        `You have <b>${documents.length}</b> uploaded document${documents.length === 1 ? '' : 's'}: ${processed} processed, ${failed} failed, <b>${linked.length}</b> linked to a ledger record.<br>` +
          `Most recent:<br>${recent.join('<br>')}<br>` +
          'Open <b>Documents</b> to review, retry, or delete them.'
      );
      return;
    } catch (err) {
      appendChatMessage('assistant', 'I could not read the document library just now. Please open the Documents page and try again.');
      return;
    }
  }

  if (lower.includes('invoice') || lower.includes('bill')) {
    appendChatMessage(
      'assistant',
      'Open <b>Invoices</b> to create or edit an invoice. Use <b>New invoice</b> to add client details and line items, then set the status to Sent when it has been issued.'
    );
    return;
  }
  if (lower.includes('expense') || lower.includes('spend') || lower.includes('vendor')) {
    appendChatMessage(
      'assistant',
      'Open <b>Expenses</b> to record a cost. Add the vendor, category, and amount, and mark it reimbursable if the business owes it back to you.'
    );
    return;
  }
  if (lower.includes('report') || lower.includes('tax') || lower.includes('pnl')) {
    appendChatMessage(
      'assistant',
      'The <b>Reports</b> page holds three statements: Profit &amp; loss, Balance sheet, and Tax summary. Choose a date range, then generate the statement.'
    );
    return;
  }
  if (lower.includes('help') || lower.includes('what can you')) {
    appendChatMessage(
      'assistant',
      'I can help with:\n• Your invoiced, collected, and outstanding position\n• Recording invoices and expenses\n• Profit &amp; loss, balance sheet, and tax statements\n• Spending, budgets, and uploaded documents\n• Navigating the ledger\n\nTry “what is outstanding?” or “how are expenses looking?”'
    );
    return;
  }
  if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey')) {
    appendChatMessage('assistant', 'Hello. How can I help you with the ledger today?');
    return;
  }

  appendChatMessage(
    'assistant',
    'I am not sure about that one. Ask me about <b>invoices</b>, <b>expenses</b>, <b>reports</b>, or your current <b>position</b>.'
  );
}

loadAssistantStatus();
renderChatHistory();

/* --- Overview --- */
/* Dashboard — real ledger data, no placeholder metrics. */

const STATUS_META = [
  { key: 'paid', label: 'Paid', tone: 'paid' },
  { key: 'sent', label: 'Sent', tone: 'sent' },
  { key: 'draft', label: 'Draft', tone: 'draft' },
  { key: 'overdue', label: 'Overdue', tone: 'overdue' },
];

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function invoiceAmount(invoice) {
  return numberValue(invoice?.totals?.total);
}

function sumBy(items, predicate, selector) {
  return items.reduce((total, item) => (predicate(item) ? total + numberValue(selector(item)) : total), 0);
}

async function loadDashboard() {
  try {
    const [summary, invoiceSummary, invoices, expenses] = await Promise.all([
      api('/api/dashboard/summary'),
      api('/api/invoices/summary'),
      fetchAllRecords('/api/invoices'),
      fetchAllRecords('/api/expenses'),
    ]);

    renderMetrics(summary, invoices, expenses);
    renderStatus(invoiceSummary, invoices);
    renderCashFlow(invoices, expenses);
    renderRecentInvoices(summary.recent_invoices || []);
    renderAttention(invoices, expenses);
  } catch (error) {
    console.error('Dashboard load error:', error);
    toast('The dashboard could not read the ledger. Check that the server is running.', 'error', 'Unable to load');
    document.querySelectorAll('.metric-value').forEach((el) => {
      if (el.textContent === '—') el.textContent = 'Unavailable';
    });
    const body = document.getElementById('recentInvoicesBody');
    if (body) {
      body.innerHTML = `<tr><td colspan="5"><div class="empty-state compact"><p>Ledger unavailable. Reload the page to try again.</p></div></td></tr>`;
    }
  }
}

function renderMetrics(summary, invoices, expenses) {
  setText('totalInvoiced', money(summary.total_invoiced));
  setText('totalPaid', money(summary.total_paid));
  setText('totalOutstanding', money(summary.total_outstanding));
  setText('totalExpenses', money(summary.total_expenses));
  setText('netProfit', money(summary.net_profit));

  const net = document.getElementById('netProfit');
  if (net) net.classList.toggle('negative', numberValue(summary.net_profit) < 0);

  const openCount = invoices.filter((invoice) => ['sent', 'overdue'].includes(invoice.status)).length;
  const paidCount = invoices.filter((invoice) => invoice.status === 'paid').length;
  const collectionRate = numberValue(summary.total_invoiced) > 0
    ? (numberValue(summary.total_paid) / numberValue(summary.total_invoiced)) * 100
    : 0;

  setText(
    'invoicedNote',
    invoices.length ? `${invoices.length} invoice${invoices.length === 1 ? '' : 's'} on file · incl. tax` : 'No invoices yet'
  );
  setText(
    'paidNote',
    paidCount
      ? `${paidCount} paid · ${collectionRate.toFixed(0)}% of invoiced value`
      : 'No payments recorded yet'
  );
  setText(
    'outstandingNote',
    openCount
      ? `${openCount} open · ${summary.overdue_count} overdue`
      : 'Nothing outstanding'
  );

  let topCategory = '';
  const totals = new Map();
  expenses.forEach((expense) => {
    const category = expense.category || 'other';
    totals.set(category, (totals.get(category) || 0) + numberValue(expense.amount));
  });
  const ranked = [...totals.entries()].sort((a, b) => b[1] - a[1]);
  if (ranked.length) topCategory = ranked[0][0];

  setText(
    'expensesNote',
    expenses.length
      ? `${expenses.length} recorded${topCategory ? ` · largest: ${topCategory}` : ''}`
      : 'No expenses recorded'
  );

  const revenue = numberValue(summary.net_revenue);
  const margin = revenue > 0 ? (numberValue(summary.net_profit) / revenue) * 100 : null;
  setText(
    'netProfitNote',
    margin === null
      ? 'Accrual · net revenue less expenses'
      : `${margin >= 0 ? 'Margin' : 'Deficit'} of ${Math.abs(margin).toFixed(1)}% on net revenue`
  );
}

function renderStatus(invoiceSummary, invoices) {
  const amounts = {};
  STATUS_META.forEach((status) => {
    amounts[status.key] = sumBy(
      invoices,
      (invoice) => invoice.status === status.key,
      (invoice) => invoiceAmount(invoice)
    );
  });

  const totalAmount = STATUS_META.reduce((total, status) => total + amounts[status.key], 0);
  const meter = document.getElementById('statusMeter');
  if (meter) {
    if (totalAmount <= 0) {
      meter.innerHTML = '<span class="status-seg empty" style="width:100%"></span>';
    } else {
      meter.innerHTML = STATUS_META.map((status) => {
        const width = (amounts[status.key] / totalAmount) * 100;
        return width > 0
          ? `<span class="status-seg ${status.tone}" style="width:${width.toFixed(2)}%" title="${esc(
              status.label
            )}: ${money(amounts[status.key])}"></span>`
          : '';
      }).join('');
    }
  }

  const legend = document.getElementById('statusLegend');
  if (legend) {
    legend.innerHTML = STATUS_META.map((status) => `
      <li class="status-row">
        <span class="dot ${status.tone}" aria-hidden="true"></span>
        <span class="label">${status.label}</span>
        <span class="count">${numberValue(invoiceSummary[status.key])}</span>
        <span class="amount">${money(amounts[status.key])}</span>
      </li>
    `).join('');
  }

  const total = document.getElementById('statusTotal');
  if (total) {
    total.innerHTML = `<span>Total invoice value</span><strong>${money(totalAmount)}</strong>`;
  }
}

function monthKey(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
}

function niceCeiling(max) {
  if (max <= 0) return 100;
  const power = Math.pow(10, Math.floor(Math.log10(max)));
  const normalized = max / power;
  let step;
  if (normalized <= 1) step = 1;
  else if (normalized <= 2) step = 2;
  else if (normalized <= 5) step = 5;
  else step = 10;
  return step * power;
}

function renderCashFlow(invoices, expenses) {
  const host = document.getElementById('cashflowChart');
  if (!host) return;

  const now = new Date();
  const buckets = [];
  for (let offset = 5; offset >= 0; offset -= 1) {
    const start = new Date(now.getFullYear(), now.getMonth() - offset, 1);
    buckets.push({
      key: monthKey(start),
      label: start.toLocaleDateString('en-GB', { month: 'short' }),
      income: 0,
      expense: 0,
    });
  }
  const bucketMap = new Map(buckets.map((bucket) => [bucket.key, bucket]));

  invoices.forEach((invoice) => {
    if (!['sent', 'overdue', 'paid'].includes(invoice.status)) return;
    const date = parseISO(invoice.issue_date);
    if (!date) return;
    const bucket = bucketMap.get(monthKey(date));
    if (bucket) bucket.income += numberValue(invoice?.totals?.subtotal);
  });

  expenses.forEach((expense) => {
    const date = parseISO(expense.date_incurred);
    if (!date) return;
    const bucket = bucketMap.get(monthKey(date));
    if (bucket) bucket.expense += numberValue(expense.amount);
  });

  const highest = Math.max(...buckets.map((bucket) => Math.max(bucket.income, bucket.expense)), 0);
  if (highest <= 0) {
    host.innerHTML = `
      <div class="chart-empty">
        <div>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" style="margin:0 auto 8px;opacity:.6">
            <path d="M3 3v18h18"/><path d="M7 14h2M12 10h2M17 6h2"/>
          </svg>
          <p>No issued revenue or expenses in the last six months.</p>
        </div>
      </div>`;
    return;
  }

  const scale = niceCeiling(highest);
  const ticks = [1, 0.75, 0.5, 0.25, 0].map((fraction) => fmtCompact(scale * fraction));

  const groups = buckets.map((bucket) => {
    const incomeHeight = (bucket.income / scale) * 100;
    const expenseHeight = (bucket.expense / scale) * 100;
    return `
      <div class="chart-group">
        <div class="chart-bar${bucket.income > 0 ? '' : ' is-empty'}" style="height:${bucket.income > 0 ? Math.max(incomeHeight, 1.2).toFixed(2) : 0}%;${bucket.income > 0 ? '' : 'min-height:0;background:transparent'}" data-tip="${bucket.label} · Net revenue ${money(bucket.income)}"></div>
        <div class="chart-bar expense${bucket.expense > 0 ? '' : ' is-empty'}" style="height:${bucket.expense > 0 ? Math.max(expenseHeight, 1.2).toFixed(2) : 0}%;${bucket.expense > 0 ? '' : 'min-height:0;background:transparent'}" data-tip="${bucket.label} · Expenses ${money(bucket.expense)}"></div>
        <span class="chart-month">${bucket.label}</span>
      </div>
    `;
  }).join('');

  host.innerHTML = `
    <div class="chart-shell">
      <div class="chart-y-axis" aria-hidden="true">${ticks.map((tick) => `<span>${tick}</span>`).join('')}</div>
      <div class="chart-main">
        <div class="chart-gridlines" aria-hidden="true">${ticks.map(() => '<span></span>').join('')}</div>
        <div class="chart-groups">${groups}</div>
      </div>
    </div>
  `;
}

function renderRecentInvoices(recent) {
  const body = document.getElementById('recentInvoicesBody');
  if (!body) return;
  if (!recent.length) {
    body.innerHTML = `
      <tr>
        <td colspan="5">
          <div class="empty-state compact">
            <strong>No invoices yet</strong>
            <p>Your first invoice will appear here once it is created.</p>
            <a class="btn btn-sm" href="/invoices?new=1">Create an invoice</a>
          </div>
        </td>
      </tr>`;
    return;
  }

  body.innerHTML = recent.map((invoice) => `
    <tr>
      <td><span class="invoice-id">${invoiceRef(invoice.id)}</span></td>
      <td><span class="cell-main">${esc(invoice.client_name || 'Unnamed client')}</span></td>
      <td>${fmtDate(invoice.date)}</td>
      <td>${statusHtml(invoice.status)}</td>
      <td class="num"><strong>${money(invoice.total)}</strong></td>
    </tr>
  `).join('');
}

function renderAttention(invoices, expenses) {
  const list = document.getElementById('attentionList');
  const count = document.getElementById('attentionCount');
  if (!list) return;

  const overdue = invoices.filter((invoice) => invoice.status === 'overdue');
  const pending = expenses.filter((expense) => expense.reimbursable && !expense.reimbursed);
  const drafts = invoices.filter((invoice) => invoice.status === 'draft');
  const totalOpen = overdue.length + pending.length + drafts.length;

  if (count) count.textContent = totalOpen ? `${totalOpen} open` : '';

  if (!totalOpen) {
    list.innerHTML = `
      <li class="empty-state compact">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><path d="M22 4L12 14l-3-3"/></svg>
        <strong>All clear</strong>
        <p>No overdue invoices, pending reimbursements, or drafts awaiting issue.</p>
      </li>`;
    return;
  }

  const items = [];

  overdue.slice(0, 3).forEach((invoice) => {
    items.push(`
      <li>
        <a class="attention-item" href="/invoices">
          <span class="attention-mark overdue" aria-hidden="true">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
          </span>
          <span class="attention-info">
            <span class="attention-title">${invoiceRef(invoice.id)} · ${esc(invoice.client_name || 'Unnamed client')}</span>
            <span class="attention-meta">Overdue${invoice.due_date ? ` since ${fmtDate(invoice.due_date)}` : ''}</span>
          </span>
          <span class="attention-amount">${money(invoiceAmount(invoice))}</span>
        </a>
      </li>`);
  });

  pending.slice(0, 2).forEach((expense) => {
    items.push(`
      <li>
        <a class="attention-item" href="/expenses">
          <span class="attention-mark reimburse" aria-hidden="true">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 6H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          </span>
          <span class="attention-info">
            <span class="attention-title">${esc(expense.description || 'Expense')}</span>
            <span class="attention-meta">Reimbursement pending${expense.vendor ? ` · ${esc(expense.vendor)}` : ''}</span>
          </span>
          <span class="attention-amount">${money(expense.amount)}</span>
        </a>
      </li>`);
  });

  if (drafts.length) {
    const draftValue = sumBy(drafts, () => true, (invoice) => invoiceAmount(invoice));
    items.push(`
      <li>
        <a class="attention-item" href="/invoices">
          <span class="attention-mark draft" aria-hidden="true">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>
          </span>
          <span class="attention-info">
            <span class="attention-title">${drafts.length} draft invoice${drafts.length === 1 ? '' : 's'}</span>
            <span class="attention-meta">Not yet issued to clients</span>
          </span>
          <span class="attention-amount">${money(draftValue)}</span>
        </a>
      </li>`);
  }

  list.innerHTML = items.join('');
}

settingsReady.then(() => {
  if (document.body.dataset.page === 'overview') {
    loadDashboard();
    initOverviewAi();
  }
});

/* --- Overview — unified financial summary --- */

let overviewAiHadSummary = false;

function ovAiTimeLabel(iso) {
  if (!iso) return '';
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return '';
  try {
    return when.toLocaleString(undefined, {
      day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  } catch (error) {
    return when.toISOString().slice(0, 16).replace('T', ' ');
  }
}

function ovAiSetState(value, label) {
  const chip = document.getElementById('ovAiState');
  if (!chip) return;
  chip.dataset.state = value;
  chip.textContent = label || value;
}

function renderOverviewSpending(spending) {
  const host = document.getElementById('ovSpendFigures');
  if (!host || !spending) return;
  const change = spending.change_percent == null
    ? '—'
    : `${numberValue(spending.change_percent) >= 0 ? '+' : '−'}${fmt(Math.abs(numberValue(spending.change_percent)))}%`;
  const cells = [
    ['Spending this month', money(spending.total)],
    ['Change vs previous', change],
    ['Largest category', spending.top_category ? spending.top_category.name : '—'],
    ['Budget use', spending.budget_note || '—'],
  ];
  host.innerHTML = cells
    .map(([label, value]) => `
      <div class="ov-spend-cell">
        <span class="ov-spend-label">${esc(label)}</span>
        <span class="ov-spend-value">${esc(value)}</span>
      </div>`)
    .join('');
}

function renderOverviewSummary(payload) {
  const body = document.getElementById('ovAiBody');
  if (!body) return;
  const summary = payload.summary || payload.computed || {};
  const computedOnly = payload.source === 'computed';
  const groups = [
    ['Highlights', summary.highlights || [], false],
    ['Observations', summary.observations || [], false],
    ['Cautions', summary.cautions || [], true],
    ['Opportunities', summary.opportunities || [], false],
  ];
  const lists = groups
    .filter(([, items]) => items.length)
    .map(([label, items, caution]) => `
      <div>
        <p class="ai-summary__label">${esc(label)}</p>
        <ul class="ai-summary__list">${items
          .map((item) => `<li class="ai-summary__item${caution ? ' ai-summary__caution' : ''}">${esc(item)}</li>`)
          .join('')}</ul>
      </div>`)
    .join('');
  body.className = computedOnly ? 'ai-summary ai-summary__computed' : 'ai-summary';
  body.innerHTML = `
    <p class="ai-summary__headline">${esc(summary.headline || '')}</p>
    ${lists}`;
  const meta = document.getElementById('ovAiMeta');
  if (meta) {
    const when = ovAiTimeLabel(payload.generated_at);
    meta.textContent = when ? `Generated ${when}` : 'Computed by the app';
  }
  const note = document.getElementById('ovAiNote');
  if (note) {
    const messages = [];
    if (computedOnly) {
      messages.push(payload.error || 'The AI was unavailable, so these figures are computed by the app.');
    } else if (payload.error) {
      messages.push(payload.error);
    }
    if (payload.verified === false) {
      messages.push(`Some figures are not in the calculated data: ${(payload.unverified_figures || []).join(', ')}.`);
    }
    note.textContent = messages.join(' ');
  }
  overviewAiHadSummary = true;
}

async function initOverviewAi() {
  const panel = document.getElementById('ovAiPanel');
  if (!panel) return;
  document.getElementById('ovAiGenerate')?.addEventListener('click', generateOverviewSummary);
  document.getElementById('ovAiRefresh')?.addEventListener('click', generateOverviewSummary);
  try {
    const payload = await api('/api/summaries/overview');
    renderOverviewSpending(payload.spending);
    const generate = document.getElementById('ovAiGenerate');
    const refresh = document.getElementById('ovAiRefresh');
    if (payload.cached) {
      renderOverviewSummary(payload);
      ovAiSetState('ready', 'Ready');
      generate?.classList.add('hidden');
      refresh?.classList.remove('hidden');
    } else if (overviewAiHadSummary) {
      ovAiSetState('stale', 'Stale');
      generate?.classList.remove('hidden');
      refresh?.classList.remove('hidden');
      const note = document.getElementById('ovAiNote');
      if (note) note.textContent = 'The figures changed since this summary was generated. Refresh to update it.';
    } else {
      ovAiSetState('idle', 'Idle');
      generate?.classList.remove('hidden');
      refresh?.classList.add('hidden');
    }
  } catch (error) {
    ovAiSetState('error', 'Error');
    const note = document.getElementById('ovAiNote');
    if (note) note.textContent = error.message;
  }
}

async function generateOverviewSummary() {
  const body = document.getElementById('ovAiBody');
  const generate = document.getElementById('ovAiGenerate');
  const refresh = document.getElementById('ovAiRefresh');
  ovAiSetState('loading', 'Working');
  if (body) {
    body.className = 'ai-summary';
    body.innerHTML = '<p class="muted">Bringing the accounts and spending together…</p>';
  }
  if (generate) generate.disabled = true;
  if (refresh) refresh.disabled = true;
  try {
    const payload = await api('/api/summaries/overview/refresh', { method: 'POST' });
    renderOverviewSpending(payload.spending);
    renderOverviewSummary(payload);
    ovAiSetState('ready', 'Ready');
    generate?.classList.add('hidden');
    refresh?.classList.remove('hidden');
  } catch (error) {
    ovAiSetState('error', 'Error');
    if (body) body.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    const note = document.getElementById('ovAiNote');
    if (note) note.textContent = error.message;
  } finally {
    if (generate) generate.disabled = false;
    if (refresh) refresh.disabled = false;
  }
}

/* --- Chat page --- */
/* A ChatGPT-style assistant: a rail of saved conversations, a scrolling
   thread, and a composer. Attaching a file reads it and files it into the
   ledger automatically, exactly like the Documents page. */

const CHAT_STORE_KEY = 'aa-chat-store-v1';
const CHAT_MODEL_KEY = 'aa-chat-model';
const CHAT_DEFAULT_MODEL = 'nvidia';

const CHAT_MODELS = [
  { value: 'nvidia', label: 'Nemotron', hint: 'NVIDIA NIM' },
  { value: 'gemini', label: 'Gemini', hint: 'Google' },
  { value: 'auto', label: 'Auto', hint: 'Best available' },
];

let chatStore = { conversations: [], activeId: null };
let chatBusy = false;

function chatEl(id) {
  return document.getElementById(id);
}

function chatUid() {
  return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function chatNow() {
  return Date.now();
}

function chatModelInfo(value) {
  return CHAT_MODELS.find((model) => model.value === value) || CHAT_MODELS[0];
}

function chatDefaultModel() {
  try {
    const saved = localStorage.getItem(CHAT_MODEL_KEY);
    if (saved && CHAT_MODELS.some((model) => model.value === saved)) return saved;
  } catch (error) { /* storage unavailable */ }
  return CHAT_DEFAULT_MODEL;
}

function chatLoadStore() {
  try {
    const raw = localStorage.getItem(CHAT_STORE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.conversations)) {
        chatStore = { conversations: parsed.conversations, activeId: parsed.activeId || null };
      }
    }
  } catch (error) { /* storage unavailable */ }
  chatStore.conversations.forEach((conversation) => {
    if (!Array.isArray(conversation.messages)) conversation.messages = [];
    if (!Array.isArray(conversation.attachments)) conversation.attachments = [];
    if (!conversation.model) conversation.model = chatDefaultModel();
  });
  if (!chatStore.conversations.some((conversation) => conversation.id === chatStore.activeId)) {
    chatStore.activeId = chatStore.conversations[0] ? chatStore.conversations[0].id : null;
  }
}

function chatSaveStore() {
  try {
    const conversations = chatStore.conversations.slice(0, 50).map((conversation) => ({
      ...conversation,
      messages: conversation.messages.slice(-200),
    }));
    localStorage.setItem(CHAT_STORE_KEY, JSON.stringify({
      conversations,
      activeId: chatStore.activeId,
    }));
  } catch (error) { /* quota or storage unavailable */ }
}

function chatActive() {
  return chatStore.conversations.find((conversation) => conversation.id === chatStore.activeId) || null;
}

function chatEnsureActive() {
  let conversation = chatActive();
  if (!conversation) {
    conversation = {
      id: chatUid(),
      title: 'New chat',
      model: chatDefaultModel(),
      createdAt: chatNow(),
      updatedAt: chatNow(),
      attachments: [],
      messages: [],
    };
    chatStore.conversations.unshift(conversation);
    chatStore.activeId = conversation.id;
  }
  return conversation;
}

function chatNewConversation() {
  const conversation = {
    id: chatUid(),
    title: 'New chat',
    model: chatDefaultModel(),
    createdAt: chatNow(),
    updatedAt: chatNow(),
    attachments: [],
    messages: [],
  };
  chatStore.conversations.unshift(conversation);
  chatStore.activeId = conversation.id;
  chatSaveStore();
  chatRenderAll();
  chatEl('chatPageInput')?.focus();
}

/* --- rendering --- */

function chatRenderAll() {
  chatRenderRail();
  chatRenderThread();
  chatRenderFiles();
  chatRenderModel();
}

function chatRenderRail() {
  const list = chatEl('chatList');
  if (!list) return;
  list.innerHTML = chatStore.conversations.map((conversation) => `
    <li>
      <button class="chat-rail__button ${conversation.id === chatStore.activeId ? 'is-active' : ''}"
        type="button" data-chat-id="${conversation.id}"
        title="${esc(conversation.title || 'New chat')}">
        ${esc(conversation.title || 'New chat')}
      </button>
    </li>`).join('');
}

function chatAnswerHtml(text) {
  return esc(text)
    .replace(/\r?\n/g, '<br>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
}

function chatMessageHtml(message) {
  if (message.role === 'note') {
    return `<div class="chat-note">${chatAnswerHtml(message.content)}</div>`;
  }
  if (message.role === 'user') {
    return `<div class="chat-msg chat-msg--user">
      <div class="chat-user-bubble">${assistantAnswerHtml(message.content)}</div>
    </div>`;
  }
  return `<div class="chat-msg chat-msg--assistant">
      <div class="chat-msg__avatar" aria-hidden="true">A</div>
      <div class="chat-msg__body"><div class="chat-ai-text">${chatAnswerHtml(message.content)}</div></div>
    </div>`;
}

function chatEmptyHtml() {
  const prompts = [
    'What is my net profit this month?',
    'Which invoices are still unpaid?',
    'Where did I spend the most?',
    'Summarise my recent expenses.',
  ];
  return `
    <div class="chat-empty">
      <div class="chat-empty__mark" aria-hidden="true">A</div>
      <h2 class="chat-empty__title">How can I help?</h2>
      <p class="chat-empty__sub">Ask about your ledger, or attach a file to read and file it automatically.</p>
      <div class="chat-suggestions">
        ${prompts.map((prompt) => `<button type="button" data-chat-prompt="${esc(prompt)}">${esc(prompt)}</button>`).join('')}
      </div>
    </div>`;
}

function chatScrollBottom() {
  const host = chatEl('chatThread');
  if (host) host.scrollTop = host.scrollHeight;
}

function chatRenderThread() {
  const host = chatEl('chatThread');
  if (!host) return;
  const conversation = chatActive();
  const messages = conversation ? conversation.messages : [];
  host.innerHTML = messages.length ? messages.map(chatMessageHtml).join('') : chatEmptyHtml();
  chatScrollBottom();
  const title = chatEl('chatTitle');
  if (title) title.textContent = conversation ? (conversation.title || 'New chat') : 'New chat';
}

function chatRenderFiles() {
  const host = chatEl('chatFiles');
  if (!host) return;
  const conversation = chatActive();
  const items = conversation ? conversation.attachments : [];
  host.hidden = items.length === 0;
  host.innerHTML = items.map((item) => `
      <li class="ov-chat__file">
        <span class="ov-chat__file-name">${esc(item.filename)}</span>
        <span class="ov-chat__file-meta">${esc(item.type || '')}</span>
        <button class="ov-chat__file-remove" type="button" data-chat-file="${item.id}"
          aria-label="Remove ${esc(item.filename)}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </li>`).join('');
}

function chatRenderModel() {
  const conversation = chatActive();
  const value = conversation ? conversation.model : chatDefaultModel();
  const info = chatModelInfo(value);
  const valueEl = chatEl('chatModelValue');
  if (valueEl) valueEl.textContent = info.label;
  const dot = document.querySelector('#chatModelButton .model-picker__dot');
  if (dot) dot.dataset.provider = value;
  const menu = chatEl('chatModelMenu');
  if (menu) {
    menu.innerHTML = CHAT_MODELS.map((model) => `
      <li role="option" class="model-picker__option" data-model="${model.value}"
        aria-selected="${model.value === value}">
        <span class="model-picker__dot" data-provider="${model.value}" aria-hidden="true"></span>
        <span class="model-picker__option-title">${esc(model.label)}</span>
        <span class="model-picker__option-hint">${esc(model.hint)}</span>
      </li>`).join('');
  }
}

function chatSetState(value, label) {
  const chip = chatEl('chatState');
  if (!chip) return;
  chip.dataset.state = value;
  chip.textContent = label || value;
}

/* --- messages --- */

function chatAppendMessage(role, content) {
  const conversation = chatActive();
  if (!conversation) return null;
  const message = { role, content, ts: chatNow() };
  conversation.messages.push(message);
  conversation.updatedAt = message.ts;
  chatSaveStore();
  const host = chatEl('chatThread');
  if (host) {
    if (host.querySelector('.chat-empty')) host.innerHTML = '';
    host.insertAdjacentHTML('beforeend', chatMessageHtml(message));
    chatScrollBottom();
  }
  return message;
}

function chatThinking() {
  const host = chatEl('chatThread');
  if (!host) return null;
  if (host.querySelector('.chat-empty')) host.innerHTML = '';
  const node = document.createElement('div');
  node.className = 'chat-msg chat-msg--assistant chat-thinking';
  node.innerHTML = '<div class="chat-msg__avatar" aria-hidden="true">A</div>' +
    '<div class="chat-msg__body"><span class="typing-dots" aria-label="The AI is thinking">' +
    '<span></span><span></span><span></span></span></div>';
  host.appendChild(node);
  chatScrollBottom();
  return node;
}

function chatHistory() {
  const conversation = chatActive();
  if (!conversation) return [];
  return conversation.messages
    .filter((message) => message.role === 'user' || message.role === 'assistant')
    .slice(-8)
    .map((message) => ({ role: message.role, content: message.content }));
}

/* --- model picker --- */

function chatOpenModelMenu() {
  const picker = chatEl('chatModelPicker');
  const menu = chatEl('chatModelMenu');
  if (!picker || !menu) return;
  const open = menu.hidden;
  menu.hidden = !open;
  picker.classList.toggle('is-open', open);
  chatEl('chatModelButton')?.setAttribute('aria-expanded', String(open));
}

function chatCloseModelMenu() {
  const picker = chatEl('chatModelPicker');
  const menu = chatEl('chatModelMenu');
  if (menu) menu.hidden = true;
  if (picker) picker.classList.remove('is-open');
  chatEl('chatModelButton')?.setAttribute('aria-expanded', 'false');
}

function chatSetModel(value) {
  const conversation = chatEnsureActive();
  conversation.model = value;
  chatSaveStore();
  try { localStorage.setItem(CHAT_MODEL_KEY, value); } catch (error) { /* ignore */ }
  chatRenderModel();
  chatCloseModelMenu();
}

/* --- attaching and filing documents --- */

function chatFileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const comma = result.indexOf(',');
      resolve(comma === -1 ? '' : result.slice(comma + 1));
    };
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

function chatFileSummary(result) {
  if (result.status === 'failed') {
    return `${result.filename} could not be read${result.error ? ': ' + result.error : ''}.`;
  }
  const type = result.document_type || 'document';
  if (result.record && result.record.kind) {
    const where = result.record.kind === 'invoice' ? 'Invoices' : 'Expenses';
    return `${result.filename} read as ${type} and filed as ${result.record.kind} ` +
      `${result.record.reference || ''} (see ${where}).`.replace('  ', ' ');
  }
  if ((result.duplicates || []).length) {
    return `${result.filename} read as ${type} — looks like a duplicate, so it was not filed.`;
  }
  return `${result.filename} read as ${type}. It stays in the Documents library; you can file it there.`;
}

async function chatHandleFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const conversation = chatEnsureActive();
  chatBusy = true;
  chatSetState('working', 'Reading');
  try {
    for (const file of files) {
      let data;
      try {
        data = await chatFileToBase64(file);
      } catch (error) {
        chatAppendMessage('note', `${file.name} could not be read: ${error.message}`);
        continue;
      }
      try {
        const payload = await api('/api/documents/upload', {
          method: 'POST',
          body: { files: [{ name: file.name, data_base64: data }] },
        });
        const rejected = payload.rejected || [];
        if (rejected.length) {
          chatAppendMessage('note', `${rejected[0].name || file.name} was not accepted: ${rejected[0].error || ''}`);
          continue;
        }
        const result = (payload.results || [])[0];
        if (!result) {
          chatAppendMessage('note', `${file.name} could not be processed.`);
          continue;
        }
        if (!conversation.attachments.some((item) => item.id === result.id)) {
          conversation.attachments.push({
            id: result.id,
            filename: result.filename || file.name,
            type: result.document_type || 'document',
          });
        }
        chatAppendMessage('note', chatFileSummary(result));
        (result.flags || [])
          .filter((flag) => flag.code === 'unstated')
          .forEach((flag) => chatAppendMessage('note', flag.message));
      } catch (error) {
        chatAppendMessage('note', `${file.name} could not be uploaded: ${error.message}`);
      }
    }
  } finally {
    chatSaveStore();
    chatRenderFiles();
    chatSetState('ready', 'Ready');
    chatBusy = false;
  }
}

/* --- sending --- */

async function chatSendMessage(explicit) {
  const input = chatEl('chatPageInput');
  const text = (explicit != null ? explicit : (input ? input.value : '')).trim();
  if (!text || chatBusy) return;
  const conversation = chatEnsureActive();
  if (input) {
    input.value = '';
    input.style.height = 'auto';
  }
  chatAppendMessage('user', text);
  if (!conversation.title || conversation.title === 'New chat') {
    conversation.title = text.length > 48 ? `${text.slice(0, 45).trimEnd()}…` : text;
    chatSaveStore();
    chatRenderRail();
    const title = chatEl('chatTitle');
    if (title) title.textContent = conversation.title;
  }
  const history = chatHistory().slice(0, -1);
  const thinking = chatThinking();
  chatSetState('thinking', 'Thinking');
  const send = chatEl('chatPageSend');
  chatBusy = true;
  if (send) send.disabled = true;
  try {
    const result = await api('/api/overview/chat', {
      method: 'POST',
      body: {
        message: text,
        model: conversation.model || 'auto',
        document_ids: conversation.attachments.map((item) => item.id),
        history,
      },
    });
    thinking?.remove();
    chatAppendMessage('assistant', result.answer || '(No answer was returned.)');
    chatSetState('ready', result.source === 'nvidia' ? 'Nemotron' : 'Gemini');
  } catch (error) {
    thinking?.remove();
    chatAppendMessage('assistant', error.message);
    chatSetState('error', 'Error');
  } finally {
    chatBusy = false;
    if (send) send.disabled = false;
    chatEl('chatPageInput')?.focus();
  }
}

/* --- conversation actions --- */

function chatRenameActive() {
  const conversation = chatActive();
  if (!conversation) return;
  const next = window.prompt('Rename conversation', conversation.title || 'New chat');
  if (next == null) return;
  conversation.title = next.trim() || 'New chat';
  chatSaveStore();
  chatRenderRail();
  const title = chatEl('chatTitle');
  if (title) title.textContent = conversation.title;
}

async function chatDeleteActive() {
  const conversation = chatActive();
  if (!conversation) return;
  const confirmed = await confirmDialog({
    kicker: 'Chat',
    title: 'Delete this conversation?',
    message: `“${conversation.title || 'New chat'}” and all of its messages will be removed from this browser.`,
    confirmLabel: 'Delete',
    danger: true,
  });
  if (!confirmed) return;
  chatStore.conversations = chatStore.conversations.filter((item) => item.id !== conversation.id);
  chatStore.activeId = chatStore.conversations[0] ? chatStore.conversations[0].id : null;
  chatEnsureActive();
  chatSaveStore();
  chatRenderAll();
}

/* --- init --- */

function initChatPage() {
  if (document.body.dataset.page !== 'chat') return;

  chatLoadStore();
  chatEnsureActive();
  chatRenderAll();
  chatSetState('idle', 'Idle');

  chatEl('chatNew')?.addEventListener('click', () => {
    const conversation = chatActive();
    if (conversation && conversation.messages.length === 0) {
      chatEl('chatPageInput')?.focus();
      return;
    }
    chatNewConversation();
  });

  chatEl('chatList')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-chat-id]');
    if (!button) return;
    chatStore.activeId = button.dataset.chatId;
    chatSaveStore();
    chatRenderAll();
  });

  chatEl('chatRename')?.addEventListener('click', chatRenameActive);
  chatEl('chatDelete')?.addEventListener('click', chatDeleteActive);

  chatEl('chatModelButton')?.addEventListener('click', (event) => {
    event.stopPropagation();
    chatOpenModelMenu();
  });
  chatEl('chatModelMenu')?.addEventListener('click', (event) => {
    const option = event.target.closest('[data-model]');
    if (!option) return;
    chatSetModel(option.dataset.model);
  });
  document.addEventListener('click', (event) => {
    const picker = chatEl('chatModelPicker');
    if (picker && !picker.contains(event.target)) chatCloseModelMenu();
  });

  chatEl('chatForm')?.addEventListener('submit', (event) => {
    event.preventDefault();
    chatSendMessage();
  });

  const input = chatEl('chatPageInput');
  input?.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  });
  input?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      chatSendMessage();
    }
  });

  const fileInput = chatEl('chatFileInput');
  chatEl('chatAttach')?.addEventListener('click', () => fileInput?.click());
  fileInput?.addEventListener('change', async () => {
    await chatHandleFiles(fileInput.files);
    fileInput.value = '';
  });

  chatEl('chatThread')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-chat-prompt]');
    if (!button) return;
    chatSendMessage(button.dataset.chatPrompt);
  });

  const main = document.querySelector('.chat-main');
  if (main) {
    main.addEventListener('dragover', (event) => {
      if (event.dataTransfer) {
        event.preventDefault();
        main.classList.add('is-dragging');
      }
    });
    main.addEventListener('dragleave', () => main.classList.remove('is-dragging'));
    main.addEventListener('drop', async (event) => {
      if (!event.dataTransfer) return;
      event.preventDefault();
      main.classList.remove('is-dragging');
      await chatHandleFiles(event.dataTransfer.files);
    });
  }

  chatEl('chatFiles')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-chat-file]');
    if (!button) return;
    const conversation = chatActive();
    if (!conversation) return;
    const id = Number(button.dataset.chatFile);
    conversation.attachments = conversation.attachments.filter((item) => item.id !== id);
    chatSaveStore();
    chatRenderFiles();
  });
}

initChatPage();

/* --- Invoices --- */
/* Invoices — create, edit, inspect, and settle client invoices. */

let invoiceStatusFilter = '';
let invoiceSearchTerm = '';
let lineItemCount = 0;
let selectedInvoiceIds = new Set();
let visibleInvoiceIds = [];
let invoicePage = 1;
let invoicePageSizeValue = 25;
let invoiceMeta = { total: 0, page: 1, page_size: 25, pages: 0 };
let invoiceMatchingAll = false;

const invoiceBody = document.getElementById('invoicesBody');
const invoiceSearchInput = document.getElementById('invoiceSearch');

/* --------------------------------------------------------------------------
   List
   -------------------------------------------------------------------------- */

async function loadInvoices() {
  const params = new URLSearchParams();
  if (invoiceStatusFilter) params.set('status', invoiceStatusFilter);
  if (invoiceSearchTerm) params.set('search', invoiceSearchTerm);
  params.set('page', String(invoicePage));
  params.set('page_size', String(invoicePageSizeValue));

  try {
    const [payload, summary] = await Promise.all([
      api(`/api/invoices?${params.toString()}`),
      api('/api/invoices/summary'),
    ]);
    renderInvoiceSummary(summary);
    invoiceMeta = {
      total: Number(payload.total) || 0,
      page: Number(payload.page) || 1,
      page_size: Number(payload.page_size) || invoicePageSizeValue,
      pages: Number(payload.pages) || 0,
    };
    if (invoiceMeta.pages && invoicePage > invoiceMeta.pages) {
      invoicePage = invoiceMeta.pages;
      return loadInvoices();
    }
    invoiceMatchingAll = false;
    renderInvoiceTable(payload.items || []);
    updateInvoicePagination();
  } catch (error) {
    console.error('Load invoices error:', error);
    if (invoiceBody) {
      invoiceBody.innerHTML = `
        <tr><td colspan="8"><div class="empty-state"><strong>Invoices unavailable</strong><p>${esc(
          error.message
        )}</p><button class="btn btn-sm" type="button" onclick="loadInvoices()">Retry</button></div></td></tr>`;
    }
    toast(error.message, 'error', 'Could not load invoices');
  }
}

function updateInvoicePagination() {
  const count = document.getElementById('invoicePageCount');
  const label = document.getElementById('invoicePageLabel');
  const prev = document.getElementById('invoicePrev');
  const next = document.getElementById('invoiceNext');
  const selectAllMatching = document.getElementById('invoiceSelectAllMatching');
  const shown = visibleInvoiceIds.length;
  const total = invoiceMeta.total;
  const start = total ? (invoiceMeta.page - 1) * invoiceMeta.page_size + 1 : 0;
  const end = total ? start + shown - 1 : 0;
  if (count) {
    count.textContent = total
      ? `${start}–${end} of ${total} record${total === 1 ? '' : 's'}`
      : '0 records';
  }
  if (label) label.textContent = `Page ${invoiceMeta.page || 1}${invoiceMeta.pages > 1 ? ` of ${invoiceMeta.pages}` : ''}`;
  if (prev) prev.disabled = invoiceMeta.page <= 1;
  if (next) next.disabled = !invoiceMeta.pages || invoiceMeta.page >= invoiceMeta.pages;
  if (selectAllMatching) {
    selectAllMatching.hidden = total <= shown;
    selectAllMatching.textContent = invoiceMatchingAll ? `All ${total} matching selected` : `Select all ${total} matching`;
    selectAllMatching.disabled = invoiceMatchingAll;
  }
}

function renderInvoiceSummary(summary) {
  const note = document.getElementById('invoiceLedgerNote');
  if (note) {
    note.innerHTML = `
      <span><strong>${summary.total}</strong> invoice${summary.total === 1 ? '' : 's'} in the ledger</span>
      <span class="sep" aria-hidden="true"></span>
      <span><strong>${money(summary.total_sent_amount)}</strong> sent</span>
      <span><strong>${money(summary.total_paid_amount)}</strong> paid</span>
      <span><strong>${money(summary.total_overdue_amount)}</strong> overdue</span>
    `;
  }

  const counts = {
    all: summary.total,
    draft: summary.draft,
    sent: summary.sent,
    paid: summary.paid,
    overdue: summary.overdue,
  };
  Object.entries(counts).forEach(([key, value]) => {
    const el = document.querySelector(`#invoiceStatusPills [data-count="${key}"]`);
    if (el) el.textContent = value;
  });
}

function renderInvoiceTable(invoices) {
  if (!invoiceBody) return;

  const count = document.getElementById('invoiceRowCount');
  if (count) count.textContent = `${invoices.length} on this page`;

  visibleInvoiceIds = invoices.map((invoice) => invoice.id);

  if (!invoices.length) {
    const filtered = Boolean(invoiceStatusFilter || invoiceSearchTerm);
    invoiceBody.innerHTML = `
      <tr>
        <td colspan="8">
          <div class="empty-state">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h5"/></svg>
            <strong>${filtered ? 'No invoices match these filters' : 'No invoices yet'}</strong>
            <p>${filtered ? 'Try a different client name or payment status.' : 'Create the first invoice to begin tracking receivables.'}</p>
            ${
              filtered
                ? '<button class="btn btn-sm" type="button" onclick="clearInvoiceFilters()">Clear filters</button>'
                : '<button class="btn btn-sm btn-primary" type="button" onclick="openInvoiceForm()">New invoice</button>'
            }
          </div>
        </td>
      </tr>`;
    syncInvoiceSelectionUI();
    return;
  }

  invoiceBody.innerHTML = invoices.map((invoice) => {
    const total = invoice?.totals?.total || 0;
    const email = invoice.client_email
      ? `<span class="cell-sub">${esc(invoice.client_email)}</span>`
      : '';
    return `
      <tr${selectedInvoiceIds.has(invoice.id) ? ' class="selected"' : ''}>
        <td class="select-col">
          <input class="check" type="checkbox" data-invoice-id="${invoice.id}" ${
            selectedInvoiceIds.has(invoice.id) ? 'checked' : ''
          } aria-label="Select ${invoiceRef(invoice.id)}">
        </td>
        <td><span class="invoice-id">${invoiceRef(invoice.id)}</span></td>
        <td><span class="cell-main">${esc(invoice.client_name || 'Unnamed client')}</span>${email}</td>
        <td>${fmtDate(invoice.issue_date)}</td>
        <td>${fmtDate(invoice.due_date)}</td>
        <td>${statusHtml(invoice.status)}</td>
        <td class="num"><strong>${money(total)}</strong></td>
        <td class="actions-col">
          <div class="row-actions">
            <button class="icon-btn" type="button" onclick="viewInvoice(${invoice.id})" title="View invoice" aria-label="View ${invoiceRef(
              invoice.id
            )}">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
            </button>
            <button class="icon-btn" type="button" onclick="editInvoice(${invoice.id})" title="Edit invoice" aria-label="Edit ${invoiceRef(
              invoice.id
            )}">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>
            </button>
            ${
              invoice.status !== 'paid' && invoice.status !== 'cancelled'
                ? `<button class="icon-btn" type="button" onclick="markPaid(${invoice.id})" title="Mark as paid" aria-label="Mark ${invoiceRef(
                    invoice.id
                  )} as paid">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
                  </button>`
                : ''
            }
            <button class="icon-btn danger" type="button" onclick="deleteInvoice(${invoice.id})" title="Delete invoice" aria-label="Delete ${invoiceRef(
              invoice.id
            )}">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/></svg>
            </button>
          </div>
        </td>
      </tr>`;
  }).join('');
  syncInvoiceSelectionUI();
}

/* --------------------------------------------------------------------------
   Selection and bulk actions
   -------------------------------------------------------------------------- */

function syncInvoiceSelectionUI() {
  const selected = selectedInvoiceIds.size;
  const bar = document.getElementById('invoiceBulkBar');
  if (bar) bar.hidden = selected === 0;

  const count = document.getElementById('invoiceSelectedCount');
  if (count) {
    const hidden = Math.max(selected - visibleInvoiceIds.filter((id) => selectedInvoiceIds.has(id)).length, 0);
    count.textContent = invoiceMatchingAll
      ? `${selected} selected (all matching filters)`
      : `${selected} selected${hidden ? ` (${hidden} on other pages)` : ''}`;
  }

  const selectedVisible = visibleInvoiceIds.filter((id) => selectedInvoiceIds.has(id)).length;
  ['invoiceSelectAll', 'invoiceSelectAllMobile'].forEach((id) => {
    const selectAll = document.getElementById(id);
    if (!selectAll) return;
    selectAll.checked = visibleInvoiceIds.length > 0 && selectedVisible === visibleInvoiceIds.length;
    selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleInvoiceIds.length;
    selectAll.disabled = visibleInvoiceIds.length === 0;
  });

  invoiceBody?.querySelectorAll('[data-invoice-id]').forEach((checkbox) => {
    const active = selectedInvoiceIds.has(Number(checkbox.dataset.invoiceId));
    checkbox.checked = active;
    checkbox.closest('tr')?.classList.toggle('selected', active);
  });
}

function clearInvoiceSelection() {
  selectedInvoiceIds.clear();
  invoiceMatchingAll = false;
  syncInvoiceSelectionUI();
  updateInvoicePagination();
}

invoiceBody?.addEventListener('change', (event) => {
  const checkbox = event.target.closest('[data-invoice-id]');
  if (!checkbox) return;
  const id = Number(checkbox.dataset.invoiceId);
  if (checkbox.checked) selectedInvoiceIds.add(id);
  else selectedInvoiceIds.delete(id);
  syncInvoiceSelectionUI();
});

['invoiceSelectAll', 'invoiceSelectAllMobile'].forEach((id) => {
  document.getElementById(id)?.addEventListener('change', (event) => {
    const select = event.target.checked;
    visibleInvoiceIds.forEach((recordId) => {
      if (select) selectedInvoiceIds.add(recordId);
      else selectedInvoiceIds.delete(recordId);
    });
    invoiceMatchingAll = false;
    syncInvoiceSelectionUI();
    updateInvoicePagination();
  });
});

document.getElementById('invoiceSelectAllMatching')?.addEventListener('click', async () => {
  const params = new URLSearchParams();
  if (invoiceStatusFilter) params.set('status', invoiceStatusFilter);
  if (invoiceSearchTerm) params.set('search', invoiceSearchTerm);
  try {
    const result = await api(`/api/invoices/ids?${params.toString()}`);
    (result.ids || []).forEach((id) => selectedInvoiceIds.add(id));
    invoiceMatchingAll = true;
    syncInvoiceSelectionUI();
    updateInvoicePagination();
    toast(
      `All ${result.total} invoice${result.total === 1 ? '' : 's'} matching the current filters are selected.`,
      'success',
      'Matching records selected'
    );
  } catch (error) {
    console.error('Select all invoices error:', error);
    toast(error.message, 'error', 'Could not select matching invoices');
  }
});

document.getElementById('invoicePrev')?.addEventListener('click', () => {
  if (invoicePage > 1) {
    invoicePage -= 1;
    loadInvoices();
  }
});

document.getElementById('invoiceNext')?.addEventListener('click', () => {
  if (invoiceMeta.pages && invoicePage < invoiceMeta.pages) {
    invoicePage += 1;
    loadInvoices();
  }
});

document.getElementById('invoicePageSize')?.addEventListener('change', (event) => {
  invoicePageSizeValue = Math.max(1, Number(event.target.value) || 25);
  invoicePage = 1;
  loadInvoices();
});

async function exportSelectedInvoices() {
  if (!selectedInvoiceIds.size) {
    toast('Select at least one invoice to export.', 'error', 'Nothing selected');
    return;
  }

  try {
    const invoices = await fetchAllRecords('/api/invoices');
    const rows = invoices.filter((invoice) => selectedInvoiceIds.has(invoice.id));
    if (!rows.length) {
      toast('The selected invoices are no longer in the ledger.', 'error', 'Export failed');
      return;
    }

    const header = ['Reference', 'Client', 'Email', 'Issued', 'Due', 'Status', 'Subtotal', 'Tax', 'Total'];
    const lines = [
      header,
      ...rows.map((invoice) => [
        invoiceRef(invoice.id),
        invoice.client_name || '',
        invoice.client_email || '',
        invoice.issue_date || '',
        invoice.due_date || '',
        invoice.status || '',
        numberValue(invoice?.totals?.subtotal).toFixed(2),
        numberValue(invoice?.totals?.tax_amount).toFixed(2),
        numberValue(invoice?.totals?.total).toFixed(2),
      ]),
    ];
    downloadCsv(`invoices-${todayISO()}.csv`, lines);
    toast(`${rows.length} invoice${rows.length === 1 ? '' : 's'} exported.`, 'success', 'CSV ready');
  } catch (error) {
    console.error('Export invoices error:', error);
    toast(error.message, 'error', 'Export failed');
  }
}

async function applyInvoiceBulkStatus() {
  const select = document.getElementById('invoiceBulkStatus');
  const status = select?.value || '';
  if (!selectedInvoiceIds.size) {
    toast('Select at least one invoice first.', 'error', 'Nothing selected');
    return;
  }
  if (!status) {
    toast('Choose the status to apply.', 'error', 'No status selected');
    select?.focus();
    return;
  }

  const ids = [...selectedInvoiceIds];
  try {
    const result = await api('/api/invoices/bulk', {
      method: 'POST',
      body: { ids, action: 'status', status },
    });
    if (select) select.value = '';
    await loadInvoices();
    toast(
      `${result.updated} invoice${result.updated === 1 ? '' : 's'} set to ${status}.`,
      'success',
      'Status updated'
    );
  } catch (error) {
    console.error('Bulk invoice status error:', error);
    toast(error.message, 'error', 'Status not updated');
  }
}

async function deleteSelectedInvoices() {
  if (!selectedInvoiceIds.size) {
    toast('Select at least one invoice first.', 'error', 'Nothing selected');
    return;
  }

  const ids = [...selectedInvoiceIds];
  const confirmed = await confirmDialog({
    kicker: 'Bulk delete',
    title: `Delete ${ids.length} invoice${ids.length === 1 ? '' : 's'}?`,
    message: `The selected invoice${ids.length === 1 ? '' : 's'} and their line items will be removed from the ledger. This cannot be undone.`,
    confirmLabel: `Delete ${ids.length} invoice${ids.length === 1 ? '' : 's'}`,
  });
  if (!confirmed) return;

  try {
    const result = await api('/api/invoices/bulk', {
      method: 'POST',
      body: { ids, action: 'delete' },
    });
    ids.forEach((id) => selectedInvoiceIds.delete(id));
    await loadInvoices();
    toast(`${result.deleted} invoice${result.deleted === 1 ? '' : 's'} deleted.`, 'success', 'Ledger updated');
  } catch (error) {
    console.error('Bulk delete invoices error:', error);
    toast(error.message, 'error', 'Invoices not deleted');
  }
}

function setInvoiceStatus(status) {
  invoiceStatusFilter = status;
  invoicePage = 1;
  document.querySelectorAll('#invoiceStatusPills .pill').forEach((pill) => {
    const active = pill.dataset.status === status;
    pill.classList.toggle('active', active);
    pill.setAttribute('aria-pressed', String(active));
  });
  loadInvoices();
}

function clearInvoiceFilters() {
  invoiceSearchTerm = '';
  if (invoiceSearchInput) invoiceSearchInput.value = '';
  setInvoiceStatus('');
}

document.querySelectorAll('#invoiceStatusPills .pill').forEach((pill) => {
  pill.setAttribute('aria-pressed', String(pill.classList.contains('active')));
  pill.addEventListener('click', () => setInvoiceStatus(pill.dataset.status || ''));
});

invoiceSearchInput?.addEventListener('input', debounce(() => {
  invoiceSearchTerm = invoiceSearchInput.value.trim();
  invoicePage = 1;
  loadInvoices();
}, 240));

/* --------------------------------------------------------------------------
   Create / edit form
   -------------------------------------------------------------------------- */

function todayISOValue() {
  return todayISO();
}

function openInvoiceForm(invoice) {
  const isEdit = Boolean(invoice);
  const taxPercent = isEdit
    ? Math.round(numberValue(invoice.tax_rate ?? appSettings.default_tax_rate ?? 0.2) * 10000) / 100
    : Math.round(numberValue(appSettings.default_tax_rate ?? 0.2) * 10000) / 100;
  const status = isEdit ? invoice.status || 'draft' : 'draft';
  const items = isEdit && invoice.line_items?.length
    ? invoice.line_items
    : [{ description: '', quantity: 1, unit_price: 0 }];
  const rows = items.map((item) => lineItemRow(item)).join('');

  const statusOptions = [
    ['draft', 'Draft'],
    ['sent', 'Sent'],
    ['paid', 'Paid'],
    ['overdue', 'Overdue'],
    ['cancelled', 'Cancelled'],
  ];

  const html = `
    ${modalHeader(isEdit ? 'Edit invoice' : 'New invoice', isEdit ? invoiceRef(invoice.id) : 'Create an invoice')}
    <form id="invoiceForm" novalidate>
      <div class="modal-body">
        <div class="form-error" id="invoiceFormError" role="alert"></div>

        <div class="form-section">
          <div class="form-section-title">Client</div>
          <div class="form-grid">
            <div class="field span-2">
              <label class="field-label" for="invoiceClient">Client name <span class="req">*</span></label>
              <input class="input" id="invoiceClient" name="client_name" value="${esc(invoice?.client_name || '')}" placeholder="e.g. Meridian Studio" required>
            </div>
            <div class="field span-2">
              <label class="field-label" for="invoiceEmail">Client email</label>
              <input class="input" id="invoiceEmail" name="client_email" type="email" value="${esc(invoice?.client_email || '')}" placeholder="billing@client.com">
            </div>
          </div>
        </div>

        <div class="form-section">
          <div class="form-section-title">Terms</div>
          <div class="form-grid">
            <div class="field">
              <label class="field-label" for="invoiceIssueDate">Issue date</label>
              <input class="input" id="invoiceIssueDate" name="issue_date" type="date" value="${esc(invoice?.issue_date || todayISOValue())}">
            </div>
            <div class="field">
              <label class="field-label" for="invoiceDueDate">Due date</label>
              <input class="input" id="invoiceDueDate" name="due_date" type="date" value="${esc(invoice?.due_date || '')}">
            </div>
            <div class="field">
              <label class="field-label" for="invoiceTaxRate">Tax rate</label>
              <input class="input" id="invoiceTaxRate" name="tax_rate_pct" type="number" min="0" max="100" step="0.1" value="${esc(taxPercent)}">
              <span class="field-hint">Percentage applied to the subtotal.</span>
            </div>
            <div class="field">
              <label class="field-label" for="invoiceStatus">Status</label>
              <select class="select" id="invoiceStatus" name="status">
                ${statusOptions.map(([value, label]) => `<option value="${value}" ${value === status ? 'selected' : ''}>${label}</option>`).join('')}
              </select>
            </div>
          </div>
        </div>

        <div class="form-section">
          <div class="form-section-title">Line items</div>
          <div class="line-items">
            <div class="line-head" aria-hidden="true">
              <span>Description</span>
              <span>Qty</span>
              <span>Rate</span>
              <span style="text-align:right">Amount</span>
              <span></span>
            </div>
            <div id="lineItems">${rows}</div>
            <div class="line-foot">
              <button class="btn btn-sm" type="button" onclick="addLineItem()">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
                Add line
              </button>
              <div class="form-totals">
                <div class="total-row"><span>Subtotal</span><span class="value" id="invoiceSubtotal">$0.00</span></div>
                <div class="total-row"><span id="invoiceTaxLabel">Tax (20%)</span><span class="value" id="invoiceTax">$0.00</span></div>
                <div class="total-row grand"><span>Total</span><span class="value" id="invoiceTotal">$0.00</span></div>
              </div>
            </div>
          </div>
        </div>

        <div class="form-section">
          <div class="form-section-title">Notes</div>
          <div class="field">
            <label class="field-label sr-only" for="invoiceNotes">Notes</label>
            <textarea class="input" id="invoiceNotes" name="notes" rows="3" placeholder="Payment terms, reference number, or a short message to the client.">${esc(invoice?.notes || '')}</textarea>
          </div>
        </div>
      </div>
      <div class="modal-foot">
        <button class="btn" type="button" data-modal-close>Cancel</button>
        <button class="btn btn-primary" type="submit" id="invoiceSubmit">${isEdit ? 'Save changes' : 'Create invoice'}</button>
      </div>
    </form>
  `;

  openModal(html);
  const form = document.getElementById('invoiceForm');
  form?.addEventListener('input', updateInvoiceTotals);
  form?.addEventListener('submit', (event) => saveInvoice(event, isEdit ? invoice.id : null));
  updateInvoiceTotals();
}

function lineItemRow(item = {}) {
  lineItemCount += 1;
  const id = `line-${lineItemCount}`;
  const quantity = numberValue(item.quantity ?? 1);
  const price = numberValue(item.unit_price ?? 0);
  return `
    <div class="line-row" id="${id}">
      <input class="input" name="item_desc" placeholder="Description" value="${esc(item.description || '')}" aria-label="Line item description">
      <input class="input" name="item_qty" type="number" min="0" step="0.01" value="${quantity}" aria-label="Quantity">
      <input class="input" name="item_price" type="number" min="0" step="0.01" value="${price}" aria-label="Unit price">
      <span class="line-amount">${money(quantity * price)}</span>
      <button class="remove-item" type="button" onclick="removeLineItem('${id}')" aria-label="Remove line item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    </div>
  `;
}

function addLineItem() {
  const host = document.getElementById('lineItems');
  if (!host) return;
  host.insertAdjacentHTML('beforeend', lineItemRow());
  updateInvoiceTotals();
  const rows = host.querySelectorAll('.line-row');
  const last = rows[rows.length - 1];
  last?.querySelector('[name="item_desc"]')?.focus();
}

function removeLineItem(id) {
  const row = document.getElementById(id);
  const host = document.getElementById('lineItems');
  if (!row || !host) return;
  row.remove();
  if (!host.querySelector('.line-row')) {
    addLineItem();
  } else {
    updateInvoiceTotals();
  }
}

function updateInvoiceTotals() {
  const form = document.getElementById('invoiceForm');
  if (!form) return;

  let subtotal = 0;
  form.querySelectorAll('.line-row').forEach((row) => {
    const quantity = numberValue(row.querySelector('[name="item_qty"]')?.value);
    const price = numberValue(row.querySelector('[name="item_price"]')?.value);
    const amount = quantity * price;
    subtotal += amount;
    const amountEl = row.querySelector('.line-amount');
    if (amountEl) amountEl.textContent = money(amount);
  });

  const rawPercent = parseFloat(form.querySelector('[name="tax_rate_pct"]')?.value);
  const percent = Number.isFinite(rawPercent) ? Math.min(Math.max(rawPercent, 0), 100) : 20;
  const tax = subtotal * (percent / 100);

  const subtotalEl = document.getElementById('invoiceSubtotal');
  const taxEl = document.getElementById('invoiceTax');
  const taxLabel = document.getElementById('invoiceTaxLabel');
  const totalEl = document.getElementById('invoiceTotal');
  if (subtotalEl) subtotalEl.textContent = money(subtotal);
  if (taxEl) taxEl.textContent = money(tax);
  if (taxLabel) taxLabel.textContent = `Tax (${percent.toFixed(percent % 1 === 0 ? 0 : 1)}%)`;
  if (totalEl) totalEl.textContent = money(subtotal + tax);
}

function showInvoiceError(message) {
  const box = document.getElementById('invoiceFormError');
  if (!box) {
    toast(message, 'error', 'Invoice not saved');
    return;
  }
  box.textContent = message;
  box.classList.add('visible');
  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function saveInvoice(event, invoiceId) {
  event.preventDefault();
  const form = document.getElementById('invoiceForm');
  if (!form) return;

  const errorBox = document.getElementById('invoiceFormError');
  if (errorBox) {
    errorBox.classList.remove('visible');
    errorBox.textContent = '';
  }

  const formData = new FormData(form);
  const clientName = String(formData.get('client_name') || '').trim();
  if (!clientName) {
    showInvoiceError('Enter a client name before saving this invoice.');
    form.querySelector('[name="client_name"]')?.focus();
    return;
  }

  const descriptions = form.querySelectorAll('[name="item_desc"]');
  const quantities = form.querySelectorAll('[name="item_qty"]');
  const prices = form.querySelectorAll('[name="item_price"]');
  const lineItems = [];
  for (let index = 0; index < descriptions.length; index += 1) {
    const description = descriptions[index].value.trim();
    const quantity = numberValue(quantities[index].value) || 1;
    const unitPrice = numberValue(prices[index].value);
    if (description || unitPrice > 0) {
      lineItems.push({ description: description || `Item ${index + 1}`, quantity, unit_price: unitPrice });
    }
  }

  if (!lineItems.length) {
    showInvoiceError('Add at least one line item with a description or a rate.');
    return;
  }

  const taxRaw = parseFloat(formData.get('tax_rate_pct'));
  const taxRate = Number.isFinite(taxRaw) ? Math.min(Math.max(taxRaw, 0), 100) / 100 : 0.2;

  const payload = {
    client_name: clientName,
    client_email: String(formData.get('client_email') || '').trim(),
    issue_date: formData.get('issue_date') || todayISO(),
    due_date: formData.get('due_date') || null,
    line_items: lineItems,
    tax_rate: taxRate,
    notes: String(formData.get('notes') || ''),
    status: formData.get('status') || 'draft',
  };

  const submit = document.getElementById('invoiceSubmit');
  if (submit) {
    submit.disabled = true;
    submit.textContent = invoiceId ? 'Saving…' : 'Creating…';
  }

  try {
    if (invoiceId) {
      await api(`/api/invoices/${invoiceId}`, { method: 'PUT', body: payload });
      closeModal();
      toast(`${invoiceRef(invoiceId)} updated.`, 'success', 'Invoice saved');
    } else {
      const created = await api('/api/invoices', { method: 'POST', body: payload });
      closeModal();
      toast(`${invoiceRef(created.id)} created for ${clientName}.`, 'success', 'Invoice created');
    }
    await loadInvoices();
  } catch (error) {
    console.error('Save invoice error:', error);
    showInvoiceError(error.message || 'The invoice could not be saved.');
    toast(error.message, 'error', 'Invoice not saved');
  } finally {
    if (submit) {
      submit.disabled = false;
      submit.textContent = invoiceId ? 'Save changes' : 'Create invoice';
    }
  }
}

/* --------------------------------------------------------------------------
   View / actions
   -------------------------------------------------------------------------- */

async function viewInvoice(id) {
  try {
    const invoice = await api(`/api/invoices/${id}`);
    const rows = (invoice.line_items || []).map((item) => `
      <tr>
        <td>${esc(item.description || 'Item')}</td>
        <td class="num">${numberValue(item.quantity)}</td>
        <td class="num">${money(item.unit_price)}</td>
        <td class="num">${money(numberValue(item.quantity) * numberValue(item.unit_price))}</td>
      </tr>
    `).join('');

    const taxPercent = Math.round(numberValue(invoice?.totals?.tax_rate ?? 0.2) * 10000) / 100;
    const taxLabel = taxPercent.toFixed(taxPercent % 1 === 0 ? 0 : 1);

    const html = `
      ${modalHeader('Invoice document', invoiceRef(invoice.id))}
      <div class="modal-body">
        <div class="invoice-doc">
          <div class="doc-head">
            <div>
              <div class="doc-brand"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2IiByb2xlPSJpbWciIGFyaWEtbGFiZWw9IkZpbmFuY2UgQWdlbnQiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJ0aWxlIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzIwMjAxYiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxNDE0MTAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImxldHRlciIgeDE9IjAuMTUiIHkxPSIxIiB4Mj0iMC44NSIgeTI9IjAiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiM4ZTZkMjAiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIwLjMyIiBzdG9wLWNvbG9yPSIjYzlhODQ3Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMC41OCIgc3RvcC1jb2xvcj0iI2YwZDQ4OSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNzgiIHN0b3AtY29sb3I9IiNkM2FlNTIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjOWQ3OTI3Ii8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJlZGdlIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI2I2OTM0MCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNSIgc3RvcC1jb2xvcj0iIzZmNTcyMCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNhOTg1MzIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHg9IjgiIHk9IjgiIHdpZHRoPSIyNDAiIGhlaWdodD0iMjQwIiByeD0iNDgiIGZpbGw9InVybCgjdGlsZSkiIHN0cm9rZT0idXJsKCNlZGdlKSIgc3Ryb2tlLXdpZHRoPSI1Ii8+CiAgPHJlY3QgeD0iMTciIHk9IjE3IiB3aWR0aD0iMjIyIiBoZWlnaHQ9IjIyMiIgcng9IjQwIiBmaWxsPSJub25lIiBzdHJva2U9IiNjOGE4NGIiIHN0cm9rZS1vcGFjaXR5PSIwLjE0IiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY5LjY2IDE4NS45OCkgc2NhbGUoMC4xOCAtMC4xOCkiIGZpbGw9InVybCgjbGV0dGVyKSIgZD0iTTM1MyA3MTEgNTk2IDg0UTYxMSA0OCA2MjguNSAzNC41UTY0NiAyMSA2NjEgMjBWMFE2MzUgMiA1OTguNSAyLjVRNTYyIDMgNTI0IDNRNDgwIDMgNDQyLjAgMi41UTQwNCAyIDM4MCAwVjIwUTQzMSAyMiA0NDUuMCAzNy41UTQ1OSA1MyA0MzkgMTA0TDI2MyA1NzlMMjgxIDYwMUwxMjEgMTg2UTEwMiAxMzYgOTguMCAxMDQuMFE5NCA3MiAxMDMuMCA1NC4wUTExMiAzNiAxMzMuMCAyOC41UTE1NCAyMSAxODUgMjBWMFExNTMgMiAxMjMuMCAyLjVROTMgMyA2MyAzUTQxIDMgMjEuNSAyLjVRMiAyIC0xMyAwVjIwUTkgMjUgMzAuNSA0Ni4wUTUyIDY3IDcyIDExOUwzMDEgNzExUTMxMyA3MTAgMzI3LjAgNzEwLjBRMzQxIDcxMCAzNTMgNzExWk00NDAgMjg4VjI2OEgxNDJMMTUyIDI4OFoiLz4KPC9zdmc+" width="26" height="26" alt="">Finance Agent</div>
              <div class="doc-number">${invoiceRef(invoice.id)} · Issued ${fmtDate(invoice.issue_date)}</div>
            </div>
            <div class="doc-status">
              ${statusHtml(invoice.status)}
              <div class="doc-number">Due ${fmtDate(invoice.due_date)}</div>
            </div>
          </div>

          <div class="doc-meta">
            <div>
              <div class="doc-meta-label">Billed to</div>
              <div class="doc-meta-value">
                ${esc(invoice.client_name || 'Unnamed client')}
                <span>${esc(invoice.client_email || 'No email on file')}</span>
              </div>
            </div>
            <div>
              <div class="doc-meta-label">Terms</div>
              <div class="doc-meta-value">
                Issued ${fmtDate(invoice.issue_date)}
                <span>Due ${fmtDate(invoice.due_date)}</span>
              </div>
            </div>
          </div>

          <table class="doc-table">
            <thead>
              <tr>
                <th scope="col">Description</th>
                <th scope="col" class="num">Qty</th>
                <th scope="col" class="num">Rate</th>
                <th scope="col" class="num">Amount</th>
              </tr>
            </thead>
            <tbody>
              ${rows || '<tr><td colspan="4" class="muted">No line items recorded.</td></tr>'}
            </tbody>
          </table>

          <div class="totals-block">
            <div class="total-row"><span>Subtotal</span><span class="value">${money(invoice?.totals?.subtotal)}</span></div>
            <div class="total-row"><span>Tax (${taxLabel}%)</span><span class="value">${money(invoice?.totals?.tax_amount)}</span></div>
            <div class="total-row grand"><span>Total</span><span class="value">${money(invoice?.totals?.total)}</span></div>
          </div>

          ${invoice.notes ? `<div class="doc-notes"><strong>Notes</strong>${esc(invoice.notes)}</div>` : ''}
        </div>
      </div>
      <div class="modal-foot">
        <a class="btn" href="/api/invoices/${invoice.id}/pdf" download>${invoiceRef(invoice.id)}.pdf</a>
        <button class="btn" type="button" data-modal-close>Close</button>
        <button class="btn" type="button" onclick="editInvoice(${invoice.id})">Edit invoice</button>
        ${
          invoice.status !== 'paid' && invoice.status !== 'cancelled'
            ? `<button class="btn btn-primary" type="button" onclick="markPaid(${invoice.id})">Mark as paid</button>`
            : ''
        }
      </div>
    `;
    openModal(html, { wide: true });
  } catch (error) {
    console.error('View invoice error:', error);
    toast(error.message, 'error', 'Could not open invoice');
  }
}

async function editInvoice(id) {
  try {
    const invoice = await api(`/api/invoices/${id}`);
    openInvoiceForm(invoice);
  } catch (error) {
    console.error('Edit invoice error:', error);
    toast(error.message, 'error', 'Could not open invoice');
  }
}

async function markPaid(id) {
  let invoice = null;
  try {
    invoice = await api(`/api/invoices/${id}`);
  } catch (error) {
    toast(error.message, 'error', 'Could not load invoice');
    return;
  }

  const confirmed = await confirmDialog({
    kicker: 'Payment',
    title: 'Mark this invoice as paid?',
    message: `${invoiceRef(id)} for ${invoice.client_name || 'this client'} (${money(
      invoice?.totals?.total
    )}) will be recorded as collected.`,
    confirmLabel: 'Mark as paid',
    danger: false,
  });
  if (!confirmed) return;

  try {
    await api(`/api/invoices/${id}/mark-paid`, { method: 'POST' });
    closeModal();
    await loadInvoices();
    toast(`${invoiceRef(id)} marked as paid.`, 'success', 'Payment recorded');
  } catch (error) {
    console.error('Mark paid error:', error);
    toast(error.message, 'error', 'Payment not recorded');
  }
}

async function deleteInvoice(id) {
  const confirmed = await confirmDialog({
    kicker: 'Delete invoice',
    title: 'Remove this invoice permanently?',
    message: `${invoiceRef(id)} and its line items will be deleted from the ledger. This cannot be undone.`,
    confirmLabel: 'Delete invoice',
  });
  if (!confirmed) return;

  try {
    await api(`/api/invoices/${id}`, { method: 'DELETE' });
    await loadInvoices();
    toast(`${invoiceRef(id)} deleted.`, 'success', 'Ledger updated');
  } catch (error) {
    console.error('Delete invoice error:', error);
    toast(error.message, 'error', 'Invoice not deleted');
  }
}

/* --------------------------------------------------------------------------
   Boot
   -------------------------------------------------------------------------- */

settingsReady.then(() => {
  if (document.body.dataset.page === 'invoices') {
    loadInvoices();
    if (new URLSearchParams(window.location.search).get('new') === '1') {
      openInvoiceForm();
    }
  }
});

/* --- Expenses --- */
/* Expenses — record, categorise, and track reimbursements. */

let expenseCategories = [];
let activeExpenseCategory = '';
let expenseVendorSearch = '';
let selectedExpenseIds = new Set();
let visibleExpenseIds = [];
let expensePage = 1;
let expensePageSizeValue = 25;
let expenseMeta = { total: 0, page: 1, page_size: 25, pages: 0 };
let expenseMatchingAll = false;

const expenseBody = document.getElementById('expensesBody');
const expenseSearchInput = document.getElementById('expenseSearch');

const FALLBACK_CATEGORIES = [
  'office',
  'travel',
  'utilities',
  'software',
  'hardware',
  'food',
  'transport',
  'rent',
  'services',
  'other',
];

/* --------------------------------------------------------------------------
   List
   -------------------------------------------------------------------------- */

async function loadCategories() {
  try {
    expenseCategories = await api('/api/expenses/categories');
  } catch (error) {
    console.error('Load categories error:', error);
    expenseCategories = [...FALLBACK_CATEGORIES];
  }
}

function categoryLabel(category) {
  return String(category || 'other')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function buildCategoryPills() {
  const host = document.getElementById('expenseCategoryPills');
  if (!host) return;
  const categories = expenseCategories.length ? expenseCategories : FALLBACK_CATEGORIES;
  host.innerHTML = `
    <button class="pill active" type="button" data-category="" aria-pressed="true">All</button>
    ${categories
      .map(
        (category) =>
          `<button class="pill" type="button" data-category="${esc(category)}" aria-pressed="false">${esc(
            categoryLabel(category)
          )}</button>`
      )
      .join('')}
  `;
  host.querySelectorAll('.pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      activeExpenseCategory = pill.dataset.category || '';
      host.querySelectorAll('.pill').forEach((item) => {
        const active = item === pill;
        item.classList.toggle('active', active);
        item.setAttribute('aria-pressed', String(active));
      });
      expensePage = 1;
      loadExpenses();
    });
  });
}

async function loadExpenses() {
  const params = new URLSearchParams();
  if (activeExpenseCategory) params.set('category', activeExpenseCategory);
  if (expenseVendorSearch) params.set('search', expenseVendorSearch);
  params.set('page', String(expensePage));
  params.set('page_size', String(expensePageSizeValue));

  try {
    const [payload, summary] = await Promise.all([
      api(`/api/expenses?${params.toString()}`),
      api('/api/expenses/summary'),
    ]);
    renderExpenseSummary(summary);
    expenseMeta = {
      total: Number(payload.total) || 0,
      page: Number(payload.page) || 1,
      page_size: Number(payload.page_size) || expensePageSizeValue,
      pages: Number(payload.pages) || 0,
    };
    if (expenseMeta.pages && expensePage > expenseMeta.pages) {
      expensePage = expenseMeta.pages;
      return loadExpenses();
    }
    expenseMatchingAll = false;
    renderExpenseTable(payload.items || []);
    updateExpensePagination();
  } catch (error) {
    console.error('Load expenses error:', error);
    if (expenseBody) {
      expenseBody.innerHTML = `
        <tr><td colspan="7"><div class="empty-state"><strong>Expenses unavailable</strong><p>${esc(
          error.message
        )}</p><button class="btn btn-sm" type="button" onclick="loadExpenses()">Retry</button></div></td></tr>`;
    }
    toast(error.message, 'error', 'Could not load expenses');
  }
}

function updateExpensePagination() {
  const count = document.getElementById('expensePageCount');
  const label = document.getElementById('expensePageLabel');
  const prev = document.getElementById('expensePrev');
  const next = document.getElementById('expenseNext');
  const selectAllMatching = document.getElementById('expenseSelectAllMatching');
  const shown = visibleExpenseIds.length;
  const total = expenseMeta.total;
  const start = total ? (expenseMeta.page - 1) * expenseMeta.page_size + 1 : 0;
  const end = total ? start + shown - 1 : 0;
  if (count) {
    count.textContent = total
      ? `${start}–${end} of ${total} record${total === 1 ? '' : 's'}`
      : '0 records';
  }
  if (label) label.textContent = `Page ${expenseMeta.page || 1}${expenseMeta.pages > 1 ? ` of ${expenseMeta.pages}` : ''}`;
  if (prev) prev.disabled = expenseMeta.page <= 1;
  if (next) next.disabled = !expenseMeta.pages || expenseMeta.page >= expenseMeta.pages;
  if (selectAllMatching) {
    selectAllMatching.hidden = total <= shown;
    selectAllMatching.textContent = expenseMatchingAll ? `All ${total} matching selected` : `Select all ${total} matching`;
    selectAllMatching.disabled = expenseMatchingAll;
  }
}

function renderExpenseSummary(summary) {
  const note = document.getElementById('expenseLedgerNote');
  if (!note) return;

  const pendingAmount = numberValue(summary.unreimbursed_total);

  let topCategory = '';
  const ranked = Object.entries(summary.by_category || {}).sort((a, b) => numberValue(b[1]) - numberValue(a[1]));
  if (ranked.length) topCategory = ranked[0][0];

  note.innerHTML = `
    <span><strong>${money(summary.total)}</strong> recorded across <strong>${summary.count}</strong> expense${
      summary.count === 1 ? '' : 's'
    }</span>
    <span class="sep" aria-hidden="true"></span>
    <span><strong>${money(pendingAmount)}</strong> awaiting reimbursement</span>
    ${topCategory ? `<span class="sep" aria-hidden="true"></span><span>Largest category <strong>${esc(categoryLabel(topCategory))}</strong></span>` : ''}
  `;
}

function renderExpenseTable(expenses) {
  if (!expenseBody) return;

  const count = document.getElementById('expenseRowCount');
  if (count) count.textContent = `${expenses.length} on this page`;

  visibleExpenseIds = expenses.map((expense) => expense.id);

  if (!expenses.length) {
    const filtered = Boolean(activeExpenseCategory || expenseVendorSearch);
    expenseBody.innerHTML = `
      <tr>
        <td colspan="7">
          <div class="empty-state">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3v18l3-2 2 2 2-2 2 2 2-2 3 2V3z"/><path d="M9 8h6M9 12h6M9 16h3"/></svg>
            <strong>${filtered ? 'No expenses match these filters' : 'No expenses recorded'}</strong>
            <p>${filtered ? 'Try a different vendor or category.' : 'Record the first business cost to start tracking spend.'}</p>
            ${
              filtered
                ? '<button class="btn btn-sm" type="button" onclick="clearExpenseFilters()">Clear filters</button>'
                : '<button class="btn btn-sm btn-primary" type="button" onclick="openExpenseForm()">Add expense</button>'
            }
          </div>
        </td>
      </tr>`;
    syncExpenseSelectionUI();
    return;
  }

  expenseBody.innerHTML = expenses.map((expense) => {
    let reimbursement = '<span class="muted">—</span>';
    if (expense.reimbursable) {
      reimbursement = expense.reimbursed
        ? '<span class="status reimbursed">Reimbursed</span>'
        : '<span class="status pending">Pending</span>';
    }
    return `
      <tr${selectedExpenseIds.has(expense.id) ? ' class="selected"' : ''}>
        <td class="select-col">
          <input class="check" type="checkbox" data-expense-id="${expense.id}" ${
            selectedExpenseIds.has(expense.id) ? 'checked' : ''
          } aria-label="Select ${esc(expense.description || 'expense')}">
        </td>
        <td class="nowrap">${fmtDate(expense.date_incurred)}</td>
        <td>
          <span class="cell-main">${esc(expense.description || 'Expense')}</span>
          ${expense.vendor ? `<span class="cell-sub">${esc(expense.vendor)}</span>` : ''}
        </td>
        <td><span class="tag">${esc(categoryLabel(expense.category))}</span></td>
        <td>${reimbursement}</td>
        <td class="num"><strong>${money(expense.amount)}</strong></td>
        <td class="actions-col">
          <div class="row-actions">
            <button class="icon-btn" type="button" onclick="editExpense(${expense.id})" title="Edit expense" aria-label="Edit ${esc(
              expense.description || 'expense'
            )}">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>
            </button>
            <button class="icon-btn danger" type="button" onclick="deleteExpense(${expense.id})" title="Delete expense" aria-label="Delete ${esc(
              expense.description || 'expense'
            )}">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/></svg>
            </button>
          </div>
        </td>
      </tr>`;
  }).join('');
  syncExpenseSelectionUI();
}

/* --------------------------------------------------------------------------
   Selection and bulk actions
   -------------------------------------------------------------------------- */

function syncExpenseSelectionUI() {
  const selected = selectedExpenseIds.size;
  const bar = document.getElementById('expenseBulkBar');
  if (bar) bar.hidden = selected === 0;

  const count = document.getElementById('expenseSelectedCount');
  if (count) {
    const hidden = Math.max(selected - visibleExpenseIds.filter((id) => selectedExpenseIds.has(id)).length, 0);
    count.textContent = expenseMatchingAll
      ? `${selected} selected (all matching filters)`
      : `${selected} selected${hidden ? ` (${hidden} on other pages)` : ''}`;
  }

  const selectedVisible = visibleExpenseIds.filter((id) => selectedExpenseIds.has(id)).length;
  ['expenseSelectAll', 'expenseSelectAllMobile'].forEach((id) => {
    const selectAll = document.getElementById(id);
    if (!selectAll) return;
    selectAll.checked = visibleExpenseIds.length > 0 && selectedVisible === visibleExpenseIds.length;
    selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleExpenseIds.length;
    selectAll.disabled = visibleExpenseIds.length === 0;
  });

  expenseBody?.querySelectorAll('[data-expense-id]').forEach((checkbox) => {
    const active = selectedExpenseIds.has(Number(checkbox.dataset.expenseId));
    checkbox.checked = active;
    checkbox.closest('tr')?.classList.toggle('selected', active);
  });
}

function clearExpenseSelection() {
  selectedExpenseIds.clear();
  expenseMatchingAll = false;
  syncExpenseSelectionUI();
  updateExpensePagination();
}

expenseBody?.addEventListener('change', (event) => {
  const checkbox = event.target.closest('[data-expense-id]');
  if (!checkbox) return;
  const id = Number(checkbox.dataset.expenseId);
  if (checkbox.checked) selectedExpenseIds.add(id);
  else selectedExpenseIds.delete(id);
  syncExpenseSelectionUI();
});

['expenseSelectAll', 'expenseSelectAllMobile'].forEach((id) => {
  document.getElementById(id)?.addEventListener('change', (event) => {
    const select = event.target.checked;
    visibleExpenseIds.forEach((recordId) => {
      if (select) selectedExpenseIds.add(recordId);
      else selectedExpenseIds.delete(recordId);
    });
    expenseMatchingAll = false;
    syncExpenseSelectionUI();
    updateExpensePagination();
  });
});

document.getElementById('expenseSelectAllMatching')?.addEventListener('click', async () => {
  const params = new URLSearchParams();
  if (activeExpenseCategory) params.set('category', activeExpenseCategory);
  if (expenseVendorSearch) params.set('search', expenseVendorSearch);
  try {
    const result = await api(`/api/expenses/ids?${params.toString()}`);
    (result.ids || []).forEach((id) => selectedExpenseIds.add(id));
    expenseMatchingAll = true;
    syncExpenseSelectionUI();
    updateExpensePagination();
    toast(
      `All ${result.total} expense${result.total === 1 ? '' : 's'} matching the current filters are selected.`,
      'success',
      'Matching records selected'
    );
  } catch (error) {
    console.error('Select all expenses error:', error);
    toast(error.message, 'error', 'Could not select matching expenses');
  }
});

document.getElementById('expensePrev')?.addEventListener('click', () => {
  if (expensePage > 1) {
    expensePage -= 1;
    loadExpenses();
  }
});

document.getElementById('expenseNext')?.addEventListener('click', () => {
  if (expenseMeta.pages && expensePage < expenseMeta.pages) {
    expensePage += 1;
    loadExpenses();
  }
});

document.getElementById('expensePageSize')?.addEventListener('change', (event) => {
  expensePageSizeValue = Math.max(1, Number(event.target.value) || 25);
  expensePage = 1;
  loadExpenses();
});

async function exportSelectedExpenses() {
  if (!selectedExpenseIds.size) {
    toast('Select at least one expense to export.', 'error', 'Nothing selected');
    return;
  }

  try {
    const expenses = await fetchAllRecords('/api/expenses');
    const rows = expenses.filter((expense) => selectedExpenseIds.has(expense.id));
    if (!rows.length) {
      toast('The selected expenses are no longer in the ledger.', 'error', 'Export failed');
      return;
    }

    const header = ['Date', 'Description', 'Vendor', 'Category', 'Amount', 'Reimbursable', 'Reimbursed', 'Notes'];
    const lines = [
      header,
      ...rows.map((expense) => [
        expense.date_incurred || '',
        expense.description || '',
        expense.vendor || '',
        expense.category || '',
        numberValue(expense.amount).toFixed(2),
        expense.reimbursable ? 'yes' : 'no',
        expense.reimbursed ? 'yes' : 'no',
        expense.notes || '',
      ]),
    ];
    downloadCsv(`expenses-${todayISO()}.csv`, lines);
    toast(`${rows.length} expense${rows.length === 1 ? '' : 's'} exported.`, 'success', 'CSV ready');
  } catch (error) {
    console.error('Export expenses error:', error);
    toast(error.message, 'error', 'Export failed');
  }
}

async function applyExpenseBulkReimbursement() {
  const select = document.getElementById('expenseBulkReimbursement');
  const raw = select?.value || '';
  if (!selectedExpenseIds.size) {
    toast('Select at least one expense first.', 'error', 'Nothing selected');
    return;
  }
  if (raw === '') {
    toast('Choose a reimbursement status to apply.', 'error', 'No status selected');
    select?.focus();
    return;
  }

  const reimbursed = raw === 'true';
  const ids = [...selectedExpenseIds];
  try {
    const result = await api('/api/expenses/bulk', {
      method: 'POST',
      body: { ids, action: 'reimbursement', reimbursed },
    });
    if (select) select.value = '';
    await loadExpenses();
    const skipped = result.skipped
      ? ` ${result.skipped} non-reimbursable expense${result.skipped === 1 ? ' was' : 's were'} skipped.`
      : '';
    toast(
      `${result.updated} expense${result.updated === 1 ? '' : 's'} marked ${
        reimbursed ? 'reimbursed' : 'pending'
      }.${skipped}`,
      'success',
      'Reimbursement updated'
    );
  } catch (error) {
    console.error('Bulk reimbursement error:', error);
    toast(error.message, 'error', 'Reimbursement not updated');
  }
}

async function deleteSelectedExpenses() {
  if (!selectedExpenseIds.size) {
    toast('Select at least one expense first.', 'error', 'Nothing selected');
    return;
  }

  const ids = [...selectedExpenseIds];
  const confirmed = await confirmDialog({
    kicker: 'Bulk delete',
    title: `Delete ${ids.length} expense${ids.length === 1 ? '' : 's'}?`,
    message: `The selected expense${ids.length === 1 ? '' : 's'} will be removed from the ledger. This cannot be undone.`,
    confirmLabel: `Delete ${ids.length} expense${ids.length === 1 ? '' : 's'}`,
  });
  if (!confirmed) return;

  try {
    const result = await api('/api/expenses/bulk', {
      method: 'POST',
      body: { ids, action: 'delete' },
    });
    ids.forEach((id) => selectedExpenseIds.delete(id));
    await loadExpenses();
    toast(`${result.deleted} expense${result.deleted === 1 ? '' : 's'} deleted.`, 'success', 'Ledger updated');
  } catch (error) {
    console.error('Bulk delete expenses error:', error);
    toast(error.message, 'error', 'Expenses not deleted');
  }
}

function clearExpenseFilters() {
  activeExpenseCategory = '';
  expenseVendorSearch = '';
  expensePage = 1;
  if (expenseSearchInput) expenseSearchInput.value = '';
  document.querySelectorAll('#expenseCategoryPills .pill').forEach((pill) => {
    const active = !pill.dataset.category;
    pill.classList.toggle('active', active);
    pill.setAttribute('aria-pressed', String(active));
  });
  loadExpenses();
}

expenseSearchInput?.addEventListener('input', debounce(() => {
  expenseVendorSearch = expenseSearchInput.value.trim();
  expensePage = 1;
  loadExpenses();
}, 240));

/* --------------------------------------------------------------------------
   Create / edit form
   -------------------------------------------------------------------------- */

function openExpenseForm(expense) {
  const isEdit = Boolean(expense);
  const categories = expenseCategories.length ? expenseCategories : FALLBACK_CATEGORIES;
  const selectedCategory = expense?.category || 'other';
  const categoryOptions = categories
    .map(
      (category) =>
        `<option value="${esc(category)}" ${category === selectedCategory ? 'selected' : ''}>${esc(
          categoryLabel(category)
        )}</option>`
    )
    .join('');

  const html = `
    ${modalHeader(isEdit ? 'Edit expense' : 'New expense', isEdit ? 'Update a recorded cost' : 'Record a business cost')}
    <form id="expenseForm" novalidate>
      <div class="modal-body">
        <div class="form-error" id="expenseFormError" role="alert"></div>

        <div class="form-section">
          <div class="form-section-title">Expense</div>
          <div class="form-grid">
            <div class="field span-2">
              <label class="field-label" for="expenseDescription">Description <span class="req">*</span></label>
              <input class="input" id="expenseDescription" name="description" value="${esc(
                expense?.description || ''
              )}" placeholder="e.g. Studio rent — November" required>
            </div>
            <div class="field">
              <label class="field-label" for="expenseAmount">Amount</label>
              <input class="input" id="expenseAmount" name="amount" type="number" min="0.01" step="0.01" value="${esc(
                expense?.amount ?? ''
              )}" placeholder="0.00" required>
            </div>
            <div class="field">
              <label class="field-label" for="expenseCategory">Category</label>
              <select class="select" id="expenseCategory" name="category">${categoryOptions}</select>
            </div>
            <div class="field">
              <label class="field-label" for="expenseDate">Date incurred</label>
              <input class="input" id="expenseDate" name="date_incurred" type="date" value="${esc(
                expense?.date_incurred || todayISO()
              )}">
            </div>
            <div class="field">
              <label class="field-label" for="expenseVendor">Vendor</label>
              <input class="input" id="expenseVendor" name="vendor" value="${esc(expense?.vendor || '')}" placeholder="e.g. Eastline Properties">
            </div>
          </div>
        </div>

        <div class="form-section">
          <div class="form-section-title">Reimbursement</div>
          <div class="switch-row">
            <div class="switch-copy">
              <strong>Reimbursable expense</strong>
              <span>Money the business should pay back.</span>
            </div>
            <label class="switch">
              <input type="checkbox" id="expenseReimbursable" name="reimbursable" ${expense?.reimbursable ? 'checked' : ''}>
              <span class="switch-track" aria-hidden="true"></span>
              <span class="sr-only">Reimbursable expense</span>
            </label>
          </div>
          <div class="switch-row" id="expenseReimbursedRow" style="margin-top:8px;${expense?.reimbursable ? '' : 'display:none'}">
            <div class="switch-copy">
              <strong>Already reimbursed</strong>
              <span>Mark this once the payment has been settled.</span>
            </div>
            <label class="switch">
              <input type="checkbox" id="expenseReimbursed" name="reimbursed" ${expense?.reimbursed ? 'checked' : ''}>
              <span class="switch-track" aria-hidden="true"></span>
              <span class="sr-only">Already reimbursed</span>
            </label>
          </div>
        </div>

        <div class="form-section">
          <div class="form-section-title">Notes</div>
          <div class="field">
            <label class="field-label sr-only" for="expenseNotes">Notes</label>
            <textarea class="input" id="expenseNotes" name="notes" rows="3" placeholder="Receipt reference, project, or any detail worth keeping.">${esc(
              expense?.notes || ''
            )}</textarea>
          </div>
        </div>
      </div>
      <div class="modal-foot">
        <button class="btn" type="button" data-modal-close>Cancel</button>
        <button class="btn btn-primary" type="submit" id="expenseSubmit">${isEdit ? 'Save changes' : 'Add expense'}</button>
      </div>
    </form>
  `;

  openModal(html);
  const form = document.getElementById('expenseForm');
  form?.addEventListener('submit', (event) => saveExpense(event, isEdit ? expense.id : null));

  const reimbursable = document.getElementById('expenseReimbursable');
  const reimbursedRow = document.getElementById('expenseReimbursedRow');
  const reimbursed = document.getElementById('expenseReimbursed');
  reimbursable?.addEventListener('change', () => {
    const enabled = reimbursable.checked;
    if (reimbursedRow) reimbursedRow.style.display = enabled ? '' : 'none';
    if (!enabled && reimbursed) reimbursed.checked = false;
  });
}

function showExpenseError(message) {
  const box = document.getElementById('expenseFormError');
  if (!box) {
    toast(message, 'error', 'Expense not saved');
    return;
  }
  box.textContent = message;
  box.classList.add('visible');
  box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function saveExpense(event, expenseId) {
  event.preventDefault();
  const form = document.getElementById('expenseForm');
  if (!form) return;

  const errorBox = document.getElementById('expenseFormError');
  if (errorBox) {
    errorBox.classList.remove('visible');
    errorBox.textContent = '';
  }

  const formData = new FormData(form);
  const description = String(formData.get('description') || '').trim();
  const amount = Number(formData.get('amount'));

  if (!description) {
    showExpenseError('Enter a description for this expense.');
    form.querySelector('[name="description"]')?.focus();
    return;
  }
  if (!Number.isFinite(amount) || amount <= 0) {
    showExpenseError('Enter an amount greater than zero.');
    form.querySelector('[name="amount"]')?.focus();
    return;
  }

  const reimbursable = formData.get('reimbursable') === 'on';
  const payload = {
    description,
    amount,
    category: formData.get('category') || 'other',
    date_incurred: formData.get('date_incurred') || todayISO(),
    vendor: String(formData.get('vendor') || '').trim(),
    notes: String(formData.get('notes') || ''),
    reimbursable,
    reimbursed: reimbursable && formData.get('reimbursed') === 'on',
  };

  const submit = document.getElementById('expenseSubmit');
  if (submit) {
    submit.disabled = true;
    submit.textContent = expenseId ? 'Saving…' : 'Adding…';
  }

  try {
    if (expenseId) {
      await api(`/api/expenses/${expenseId}`, { method: 'PUT', body: payload });
      toast(`${description} updated.`, 'success', 'Expense saved');
    } else {
      await api('/api/expenses', { method: 'POST', body: payload });
      toast(`${description} added to the ledger.`, 'success', 'Expense recorded');
    }
    closeModal();
    await loadExpenses();
  } catch (error) {
    console.error('Save expense error:', error);
    showExpenseError(error.message || 'The expense could not be saved.');
    toast(error.message, 'error', 'Expense not saved');
  } finally {
    if (submit) {
      submit.disabled = false;
      submit.textContent = expenseId ? 'Save changes' : 'Add expense';
    }
  }
}

async function editExpense(id) {
  try {
    const expense = await api(`/api/expenses/${id}`);
    openExpenseForm(expense);
  } catch (error) {
    console.error('Edit expense error:', error);
    toast(error.message, 'error', 'Could not open expense');
  }
}

async function deleteExpense(id) {
  const confirmed = await confirmDialog({
    kicker: 'Delete expense',
    title: 'Remove this expense permanently?',
    message: 'The entry will be deleted from the ledger. This cannot be undone.',
    confirmLabel: 'Delete expense',
  });
  if (!confirmed) return;

  try {
    await api(`/api/expenses/${id}`, { method: 'DELETE' });
    await loadExpenses();
    toast('Expense deleted.', 'success', 'Ledger updated');
  } catch (error) {
    console.error('Delete expense error:', error);
    toast(error.message, 'error', 'Expense not deleted');
  }
}

/* --------------------------------------------------------------------------
   Boot
   -------------------------------------------------------------------------- */

(async function initExpenses() {
  await settingsReady;
  if (document.body.dataset.page !== 'expenses') return;
  await loadCategories();
  buildCategoryPills();
  await loadExpenses();
  if (new URLSearchParams(window.location.search).get('new') === '1') {
    openExpenseForm();
  }
})();

/* --- Reports --- */
/* Reports — statement-style profit & loss, balance sheet, and tax summary. */

let currentReport = 'pnl';

/* --------------------------------------------------------------------------
   Controls
   -------------------------------------------------------------------------- */

function isoDate(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(
    date.getDate()
  ).padStart(2, '0')}`;
}

function setReportDates(from, to) {
  const fromInput = document.getElementById('reportFrom');
  const toInput = document.getElementById('reportTo');
  if (fromInput) fromInput.value = from;
  if (toInput) toInput.value = to;
}

function applyRange(range) {
  const now = new Date();
  let from;
  let to = new Date(now.getFullYear(), now.getMonth(), now.getDate());

  if (range === 'month') {
    from = new Date(now.getFullYear(), now.getMonth(), 1);
  } else if (range === 'last-month') {
    from = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    to = new Date(now.getFullYear(), now.getMonth(), 0);
  } else if (range === 'quarter') {
    const quarterStart = Math.floor(now.getMonth() / 3) * 3;
    from = new Date(now.getFullYear(), quarterStart, 1);
  } else {
    from = new Date(now.getFullYear(), 0, 1);
  }

  setReportDates(isoDate(from), isoDate(to));
  document.querySelectorAll('.range-preset').forEach((button) => {
    button.classList.toggle('active', button.dataset.range === range);
  });
  loadReport();
}

document.querySelectorAll('.range-preset').forEach((button) => {
  button.addEventListener('click', () => applyRange(button.dataset.range));
});

['reportFrom', 'reportTo'].forEach((id) => {
  document.getElementById(id)?.addEventListener('change', () => {
    document.querySelectorAll('.range-preset').forEach((button) => button.classList.remove('active'));
  });
});

function updateReportControls() {
  const isBalance = currentReport === 'balance';
  document.querySelectorAll('#reportToolbar .field').forEach((field) => {
    field.classList.toggle('hidden', isBalance);
  });
  document.querySelector('.range-presets')?.classList.toggle('hidden', isBalance);
  document.getElementById('reportAsOf')?.classList.toggle('hidden', !isBalance);
  const asOf = document.getElementById('reportAsOf');
  if (asOf) asOf.textContent = isBalance ? `Position as of ${fmtDate(todayISO(), { long: true })}` : '';
  const generateLabel = document.getElementById('reportGenerateLabel');
  if (generateLabel) generateLabel.textContent = isBalance ? 'Refresh' : 'Generate';
}

function switchReport(type) {
  currentReport = type;
  document.querySelectorAll('.report-tab').forEach((tab) => {
    const active = tab.dataset.report === type;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  updateReportControls();
  loadReport();
}

/* --------------------------------------------------------------------------
   Loading
   -------------------------------------------------------------------------- */

async function loadReport() {
  const results = document.getElementById('reportResults');
  if (!results) return;
  results.setAttribute('aria-busy', 'true');
  results.innerHTML = '<div class="report-empty">Reading the ledger…</div>';

  const from = document.getElementById('reportFrom')?.value || '';
  const to = document.getElementById('reportTo')?.value || '';
  const params = new URLSearchParams();
  if (from) params.set('from_date', from);
  if (to) params.set('to_date', to);
  const query = params.toString();

  try {
    if (currentReport === 'pnl') {
      const data = await api(`/api/reports/profit-loss${query ? `?${query}` : ''}`);
      results.innerHTML = renderProfitLoss(data);
    } else if (currentReport === 'balance') {
      const data = await api('/api/reports/balance-sheet');
      results.innerHTML = renderBalanceSheet(data);
    } else {
      const data = await api(`/api/reports/tax-summary${query ? `?${query}` : ''}`);
      results.innerHTML = renderTaxSummary(data);
    }
  } catch (error) {
    console.error('Report load error:', error);
    results.innerHTML = `
      <div class="empty-state">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 16v-4M12 16V8M17 16v-6"/></svg>
        <strong>Statement unavailable</strong>
        <p>${esc(error.message)}</p>
        <button class="btn btn-sm" type="button" onclick="loadReport()">Retry</button>
      </div>`;
    toast(error.message, 'error', 'Could not generate report');
  } finally {
    results.setAttribute('aria-busy', 'false');
  }
}

/* --------------------------------------------------------------------------
   Rendering helpers
   -------------------------------------------------------------------------- */

function periodLabel(period) {
  if (!period?.from || !period?.to) return 'Current period';
  return `${fmtDate(period.from, { long: true })} — ${fmtDate(period.to, { long: true })}`;
}

function masthead(kicker, title, period, headlineLabel, headlineValue, tone = '') {
  return `
    <div class="report-masthead">
      <div>
        <div class="report-kicker">${esc(kicker)}</div>
        <h2 class="report-title">${esc(title)}</h2>
        <div class="report-period">${esc(period)}</div>
      </div>
      <div class="report-headline">
        <div class="report-headline-label">${esc(headlineLabel)}</div>
        <div class="report-headline-value ${tone}">${headlineValue}</div>
      </div>
    </div>
  `;
}

function summaryStat(label, value, tone = '') {
  return `
    <div class="report-stat">
      <div class="report-stat-label">${esc(label)}</div>
      <div class="report-stat-value ${tone}">${value}</div>
    </div>
  `;
}

/* --------------------------------------------------------------------------
   Profit & loss
   -------------------------------------------------------------------------- */

function renderProfitLoss(data) {
  const income = data.income || [];
  const expenses = data.expenses || [];
  const netTone = numberValue(data.net_profit) >= 0 ? 'positive' : 'negative';
  const margin = numberValue(data.total_income) > 0
    ? (numberValue(data.net_profit) / numberValue(data.total_income)) * 100
    : 0;

  const incomeRows = income.length
    ? income.map((entry) => `
        <tr>
          <td class="entry-date">${fmtDate(entry.date)}</td>
          <td>${esc(entry.description)}</td>
          <td class="entry-category"></td>
          <td>${money(entry.amount)}</td>
        </tr>`).join('')
    : '<tr><td colspan="4" class="muted">No income recorded in this period.</td></tr>';

  const expenseRows = expenses.length
    ? expenses.map((entry) => `
        <tr>
          <td class="entry-date">${fmtDate(entry.date)}</td>
          <td>${esc(entry.description)}</td>
          <td class="entry-category"><span class="tag">${esc(entry.category || 'other')}</span></td>
          <td>${money(entry.amount)}</td>
        </tr>`).join('')
    : '<tr><td colspan="4" class="muted">No expenses recorded in this period.</td></tr>';

  return `
    ${masthead(
      'Statement · Profit & loss',
      'Profit & loss',
      periodLabel(data.period),
      'Net profit',
      money(data.net_profit),
      netTone
    )}
    <div class="report-summary">
      ${summaryStat('Net revenue', money(data.total_income), 'positive')}
      ${summaryStat('Total expenses', money(data.total_expenses), 'negative')}
      ${summaryStat('Net margin', `${margin.toFixed(1)}%`, netTone)}
    </div>

    <section class="statement">
      <div class="statement-head">
        <h3 class="statement-title">Income</h3>
        <span class="statement-note">Accrual · issued invoices (sent, overdue or paid), excluding sales tax</span>
      </div>
      <table class="statement-table">
        <thead class="sr-only"><tr><th scope="col">Date</th><th scope="col">Description</th><th scope="col">Category</th><th scope="col">Amount</th></tr></thead>
        <tbody>
          ${incomeRows}
          <tr class="subtotal"><td></td><td>Net revenue</td><td></td><td>${money(data.total_income)}</td></tr>
        </tbody>
      </table>
    </section>

    <section class="statement">
      <div class="statement-head">
        <h3 class="statement-title">Expenses</h3>
        <span class="statement-note">Recorded costs within the period</span>
      </div>
      <table class="statement-table">
        <thead class="sr-only"><tr><th scope="col">Date</th><th scope="col">Description</th><th scope="col">Category</th><th scope="col">Amount</th></tr></thead>
        <tbody>
          ${expenseRows}
          <tr class="subtotal"><td></td><td>Total expenses</td><td></td><td>${money(data.total_expenses)}</td></tr>
          <tr class="net"><td></td><td>Net profit</td><td></td><td>${money(data.net_profit)}</td></tr>
        </tbody>
      </table>
    </section>
    ${data.basis_note ? `<div class="tax-note">${esc(data.basis_note)}</div>` : ''}
  `;
}

/* --------------------------------------------------------------------------
   Balance sheet
   -------------------------------------------------------------------------- */

function renderBalanceSheet(data) {
  const assets = data.assets || {};
  const liabilities = data.liabilities || {};
  const equity = data.equity || {};
  const equityTone = numberValue(equity.total_equity) >= 0 ? 'positive' : 'negative';
  const assumptions = (data.assumptions || []).map((line) => `<li>${esc(line)}</li>`).join('');

  return `
    ${masthead(
      'Partial position · Management view',
      'Balance sheet',
      `Position as of ${fmtDate(data.as_of || todayISO(), { long: true })}`,
      'Net position',
      money(equity.total_equity),
      equityTone
    )}
    <div class="report-summary">
      ${summaryStat('Receivable', money(assets.accounts_receivable), 'positive')}
      ${summaryStat('Total liabilities', money(liabilities.total_liabilities), 'negative')}
      ${summaryStat('Retained position', money(equity.total_equity), equityTone)}
    </div>

    <div class="balance-grid">
      <section class="statement">
        <div class="statement-head">
          <h3 class="statement-title">Assets</h3>
          <span class="statement-note">Amounts owed to the business and cash recorded</span>
        </div>
        <table class="statement-table">
          <tbody>
            <tr><td>Accounts receivable</td><td>${money(assets.accounts_receivable)}</td></tr>
            <tr><td>Cash received</td><td>${money(assets.cash_received)}</td></tr>
            <tr><td>Recorded expenses (settled)</td><td>−${money(assets.recorded_expenses)}</td></tr>
            <tr class="subtotal"><td>Net cash</td><td>${money(assets.net_cash)}</td></tr>
            <tr class="subtotal"><td>Total assets</td><td>${money(assets.total_assets)}</td></tr>
          </tbody>
        </table>
      </section>

      <div class="balance-stack">
        <section class="statement">
          <div class="statement-head">
            <h3 class="statement-title">Liabilities</h3>
            <span class="statement-note">Only obligations supported by recorded data</span>
          </div>
          <table class="statement-table">
            <tbody>
              <tr><td>Reimbursement payable</td><td>${money(liabilities.reimbursement_payable)}</td></tr>
              <tr><td>Output tax payable</td><td>${money(liabilities.output_tax_payable)}</td></tr>
              <tr class="subtotal"><td>Total liabilities</td><td>${money(liabilities.total_liabilities)}</td></tr>
            </tbody>
          </table>
        </section>

        <section class="statement">
          <div class="statement-head">
            <h3 class="statement-title">Equity</h3>
            <span class="statement-note">Residual management position</span>
          </div>
          <table class="statement-table">
            <tbody>
              <tr><td>Retained position</td><td>${money(equity.retained_earnings)}</td></tr>
              <tr class="subtotal"><td>Total equity</td><td>${money(equity.total_equity)}</td></tr>
            </tbody>
          </table>
        </section>
      </div>
    </div>

    <div class="balance-equation">
      <span>Assets <strong>${money(assets.total_assets)}</strong></span>
      <span>Liabilities <strong>${money(liabilities.total_liabilities)}</strong></span>
      <span>Equity <strong>${money(equity.total_equity)}</strong></span>
    </div>
    <div class="tax-note">
      <strong>Not a complete statutory balance sheet.</strong>
      ${esc(data.basis_note || '')}
      ${assumptions ? `<ul class="assumption-list">${assumptions}</ul>` : ''}
    </div>
  `;
}

/* --------------------------------------------------------------------------
   Tax summary
   -------------------------------------------------------------------------- */

function renderTaxSummary(data) {
  const invoiced = numberValue(data.total_tax_invoiced);
  const collected = numberValue(data.total_tax_collected);
  const outstanding = numberValue(data.tax_outstanding);

  return `
    ${masthead(
      'Statement · Tax',
      'Tax summary',
      periodLabel(data.period),
      'Tax invoiced',
      money(invoiced),
      'gold'
    )}
    <div class="report-summary">
      ${summaryStat('Tax invoiced', money(invoiced), 'gold')}
      ${summaryStat('Tax collected', money(collected), 'positive')}
      ${summaryStat('Tax outstanding', money(outstanding), outstanding > 0 ? 'negative' : '')}
    </div>

    <section class="statement">
      <div class="statement-head">
        <h3 class="statement-title">Issued invoices</h3>
        <span class="statement-note">Sent, overdue and paid invoices only</span>
      </div>
      <table class="statement-table">
        <tbody>
          <tr>
            <td>Reporting period</td>
            <td>${esc(data.period?.from || '—')} to ${esc(data.period?.to || '—')}</td>
          </tr>
          <tr>
            <td>Net sales (excluding tax)</td>
            <td>${money(data.net_sales)}</td>
          </tr>
          <tr>
            <td>Total sales (including tax)</td>
            <td>${money(data.total_sales)}</td>
          </tr>
          <tr>
            <td>Tax invoiced</td>
            <td>${money(invoiced)}</td>
          </tr>
          <tr>
            <td>Tax collected (paid invoices)</td>
            <td>${money(collected)}</td>
          </tr>
          <tr class="subtotal">
            <td>Tax outstanding</td>
            <td>${money(outstanding)}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="statement">
      <div class="statement-head">
        <h3 class="statement-title">Excluded documents</h3>
        <span class="statement-note">Not part of issued revenue or collected tax</span>
      </div>
      <table class="statement-table">
        <tbody>
          <tr>
            <td>Draft invoices (${numberValue(data.draft_count)})</td>
            <td>Tax ${money(data.draft_tax)} · Value ${money(data.draft_value)}</td>
          </tr>
          <tr>
            <td>Cancelled invoices (${numberValue(data.cancelled_count)})</td>
            <td>Tax ${money(data.cancelled_tax)} · Value ${money(data.cancelled_value)}</td>
          </tr>
          <tr class="subtotal">
            <td>Taxable issued transactions</td>
            <td>${numberValue(data.taxable_transactions)}</td>
          </tr>
        </tbody>
      </table>
    </section>
    <div class="tax-note">${esc(data.basis_note || '')}</div>
  `;
}

/* --------------------------------------------------------------------------
   Boot
   -------------------------------------------------------------------------- */

(async function initReports() {
  await settingsReady;
  if (document.body.dataset.page !== 'reports') return;
  const now = new Date();
  const first = new Date(now.getFullYear(), now.getMonth(), 1);
  setReportDates(isoDate(first), isoDate(now));
  document.querySelectorAll('.range-preset').forEach((button) => {
    button.classList.toggle('active', button.dataset.range === 'month');
  });
  updateReportControls();
  loadReport();
})();

/* --------------------------------------------------------------------------
   Settings, backups, CSV import, and activity
   -------------------------------------------------------------------------- */

async function readFileText(input) {
  const file = input?.files?.[0];
  if (!file) throw new Error('Choose a file first.');
  return file.text();
}

function showImportMessage(host, message, type = 'error') {
  if (!host) return;
  if (!message) {
    host.innerHTML = '';
    host.hidden = true;
    return;
  }
  host.hidden = false;
  host.innerHTML = `<div class="import-message ${type}">${esc(message)}</div>`;
}

function renderImportPreview(host, preview, kind) {
  if (!host) return;
  host.hidden = false;
  const summary = preview.summary || {};
  const errors = preview.errors || [];
  const duplicates = preview.duplicates || [];
  const rows = (preview.rows || []).slice(0, 12);
  const columns = kind === 'invoices'
    ? ['row', 'client_name', 'issue_date', 'description', 'amount', 'status']
    : ['row', 'date_incurred', 'description', 'vendor', 'category', 'amount'];

  const head = columns.map((column) => `<th scope="col">${esc(column.replace(/_/g, ' '))}</th>`).join('');
  const body = rows.map((row) => {
    const hasError = (row.errors || []).length;
    const cells = columns.map((column) => {
      if (column === 'amount') return `<td class="num">${row.amount ? money(row.amount) : '—'}</td>`;
      if (column === 'row') return `<td>${row.row ?? '—'}</td>`;
      return `<td>${esc(row[column] ?? '—')}</td>`;
    }).join('');
    return `<tr class="${hasError ? 'row-error' : ''}">${cells}</tr>${
      hasError ? `<tr class="row-error-detail"><td colspan="${columns.length}">${esc((row.errors || []).join('; '))}</td></tr>` : ''
    }`;
  }).join('');

  const matches = kind === 'invoices' ? summary.invoices : summary.expenses;
  const suggested = summary.suggested || 0;
  const mapping = preview.mapping || {};
  const mappingEntries = Object.entries(mapping).filter(([, source]) => source);
  const mappingRow = mappingEntries.length
    ? `<div class="import-mapping"><span class="muted">Columns read</span>${mappingEntries
        .map(([field, source]) => `<span class="tag">${esc(field.replace(/_/g, ' '))} ← ${esc(source)}</span>`)
        .join('')}</div>`
    : '';
  host.innerHTML = `
    <div class="import-summary">
      <span><strong>${summary.rows || 0}</strong> rows read</span>
      <span><strong>${matches || 0}</strong> ${kind === 'invoices' ? 'invoices' : 'expenses'} ready</span>
      <span class="${errors.length ? 'negative' : ''}"><strong>${errors.length}</strong> errors</span>
      <span class="${duplicates.length ? 'negative' : ''}"><strong>${duplicates.length}</strong> possible duplicates</span>
      ${kind === 'expenses' && suggested ? `<span><strong>${suggested}</strong> categorised by rules</span>` : ''}
    </div>
    ${mappingRow}
    ${errors.length ? `<div class="import-errors"><strong>Fix these before importing</strong><ul>${errors.slice(0, 12).map((error) => `<li>${esc(error)}</li>`).join('')}</ul>${errors.length > 12 ? `<p class="muted">…and ${errors.length - 12} more.</p>` : ''}</div>` : ''}
    <div class="table-wrap import-table-wrap">
      <table class="data-table import-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body || `<tr><td colspan="${columns.length}" class="muted">No preview rows.</td></tr>`}</tbody>
      </table>
    </div>
    <div class="import-actions">
      <button class="btn btn-primary" type="button" data-import-confirm="${kind}" ${errors.length ? 'disabled' : ''}>${duplicates.length ? 'Import including duplicates' : 'Import records'}</button>
      <span class="field-hint">${duplicates.length ? 'Duplicates are flagged by matching client, date and total (or description, date and amount).' : 'All rows are validated before anything is saved.'}</span>
    </div>
  `;
}

/* CSV import is shared by the Settings page and SpendWise. One preview
   implementation, one confirmation flow, one set of endpoints. */
function importTargets(kind, options = {}) {
  const suffix = kind === 'invoices' ? 'invoice' : 'expense';
  return {
    fileInputId: options.fileInputId || `${suffix}CsvFile`,
    hostId: options.hostId || `${suffix}CsvPreviewHost`,
  };
}

async function previewCsvImport(kind, options = {}) {
  const targets = importTargets(kind, options);
  const fileInput = document.getElementById(targets.fileInputId);
  const host = document.getElementById(targets.hostId);
  try {
    const csvText = await readFileText(fileInput);
    const preview = await api(`/api/import/${kind}/preview`, {
      method: 'POST',
      body: { csv_text: csvText },
    });
    renderImportPreview(host, preview, kind);
    const confirmButton = host?.querySelector('[data-import-confirm]');
    confirmButton?.addEventListener('click', () => importCsvRecords(kind, csvText, preview, options));
  } catch (error) {
    console.error('CSV preview error:', error);
    showImportMessage(host, error.message);
  }
}

async function importCsvRecords(kind, csvText, preview, options = {}) {
  const targets = importTargets(kind, options);
  const duplicates = (preview.duplicates || []).length;
  const confirmed = await confirmDialog({
    kicker: 'CSV import',
    title: duplicates ? 'Import matching records anyway?' : `Import ${kind}?`,
    message: duplicates
      ? `${duplicates} possible duplicate${duplicates === 1 ? '' : 's'} were detected. These records will still be saved if you continue. A backup is created first.`
      : `Validated records will be added to the ledger. A backup is created first.`,
    confirmLabel: 'Import records',
    danger: Boolean(duplicates),
  });
  if (!confirmed) return;

  const host = document.getElementById(targets.hostId);
  try {
    const result = await api(`/api/import/${kind}`, {
      method: 'POST',
      body: { csv_text: csvText, allow_duplicates: Boolean(duplicates) },
    });
    toast(`${result.created} ${kind} imported.`, 'success', 'Import complete');
    if (host) {
      host.innerHTML = `<div class="import-message success">${result.created} ${kind} imported successfully.</div>`;
    }
    const fileInput = document.getElementById(targets.fileInputId);
    if (fileInput) fileInput.value = '';
    if (typeof options.onImported === 'function') options.onImported(result);
  } catch (error) {
    console.error('CSV import error:', error);
    showImportMessage(host, error.message);
    toast(error.message, 'error', 'Import failed');
  }
}

function initImportBlock({ kind, buttonId, fileInputId, hostId, onImported }) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  button.addEventListener('click', () => previewCsvImport(kind, { fileInputId, hostId, onImported }));
}

function fillSettingsForm(settings) {
  const mapping = {
    settingCurrencyCode: 'currency_code',
    settingCurrencySymbol: 'currency_symbol',
    settingLocale: 'locale',
    settingTaxRate: 'default_tax_rate',
    settingDateFormat: 'date_format',
    settingBusinessName: 'business_name',
    settingBusinessAddress: 'business_address',
    settingBusinessEmail: 'business_email',
    settingPaymentDetails: 'payment_details',
  };
  Object.entries(mapping).forEach(([id, key]) => {
    const input = document.getElementById(id);
    if (!input) return;
    let value = settings[key] ?? '';
    if (key === 'default_tax_rate') {
      value = (numberValue(value) * 100).toFixed(2);
    }
    input.value = value;
  });
}

async function saveSettingsForm(event) {
  event.preventDefault();
  const errorBox = document.getElementById('settingsError');
  const submit = document.getElementById('settingsSave');
  if (errorBox) {
    errorBox.textContent = '';
    errorBox.classList.remove('visible');
  }
  const payload = {
    currency_code: document.getElementById('settingCurrencyCode')?.value.trim(),
    currency_symbol: document.getElementById('settingCurrencySymbol')?.value,
    locale: document.getElementById('settingLocale')?.value.trim(),
    default_tax_rate: String(numberValue(document.getElementById('settingTaxRate')?.value) / 100),
    date_format: document.getElementById('settingDateFormat')?.value,
    business_name: document.getElementById('settingBusinessName')?.value.trim(),
    business_address: document.getElementById('settingBusinessAddress')?.value.trim(),
    business_email: document.getElementById('settingBusinessEmail')?.value.trim(),
    payment_details: document.getElementById('settingPaymentDetails')?.value.trim(),
  };
  if (submit) submit.disabled = true;
  try {
    const saved = await api('/api/settings', { method: 'PUT', body: payload });
    setAppSettings(saved);
    fillSettingsForm(saved);
    toast('Currency, tax and formatting settings saved.', 'success', 'Settings updated');
  } catch (error) {
    console.error('Save settings error:', error);
    if (errorBox) {
      errorBox.textContent = error.message;
      errorBox.classList.add('visible');
    }
    toast(error.message, 'error', 'Settings not saved');
  } finally {
    if (submit) submit.disabled = false;
  }
}

async function restoreSelectedBackup() {
  const fileInput = document.getElementById('restoreFile');
  const errorBox = document.getElementById('restoreError');
  if (errorBox) {
    errorBox.textContent = '';
    errorBox.classList.remove('visible');
  }
  let payload;
  try {
    const text = await readFileText(fileInput);
    payload = JSON.parse(text);
  } catch (error) {
    const message = error instanceof SyntaxError ? 'That file is not valid JSON.' : error.message;
    if (errorBox) {
      errorBox.textContent = message;
      errorBox.classList.add('visible');
    }
    toast(message, 'error', 'Restore failed');
    return;
  }

  const confirmed = await confirmDialog({
    kicker: 'Restore backup',
    title: 'Replace current records with this backup?',
    message: 'Invoices, expenses, settings and activity will be replaced. The current records are backed up first. Malformed or incompatible files are rejected.',
    confirmLabel: 'Restore backup',
    danger: true,
  });
  if (!confirmed) return;

  try {
    const result = await api('/api/backup/restore', { method: 'POST', body: payload });
    toast(
      `Restored ${result.invoices} invoice${result.invoices === 1 ? '' : 's'} and ${result.expenses} expense${result.expenses === 1 ? '' : 's'}.`,
      'success',
      'Backup restored'
    );
    await loadAppSettings();
    fillSettingsForm(appSettings);
    if (fileInput) fileInput.value = '';
  } catch (error) {
    console.error('Restore error:', error);
    if (errorBox) {
      errorBox.textContent = error.message;
      errorBox.classList.add('visible');
    }
    toast(error.message, 'error', 'Restore failed');
  }
}

function renderActivity(items) {
  const body = document.getElementById('activityBody');
  if (!body) return;
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="4"><div class="empty-state compact"><strong>No activity yet</strong><p>Ledger changes will appear here.</p></div></td></tr>';
    return;
  }
  body.innerHTML = items.map((item) => {
    const record = item.record_type === 'invoice' && item.record_id
      ? invoiceRef(item.record_id)
      : item.reference || item.record_type || '—';
    return `
      <tr>
        <td class="nowrap">${fmtDate(item.timestamp)}</td>
        <td><span class="tag">${esc(String(item.action || '').replace(/-/g, ' '))}</span></td>
        <td><span class="cell-main">${esc(record)}</span><span class="cell-sub">${esc(item.record_type || '')}</span></td>
        <td>${esc(item.details || '')}</td>
      </tr>`;
  }).join('');
}

async function initSettingsPage() {
  await settingsReady;
  if (document.body.dataset.page !== 'settings') return;
  fillSettingsForm(appSettings);
  document.getElementById('settingsForm')?.addEventListener('submit', saveSettingsForm);
  document.getElementById('restoreButton')?.addEventListener('click', restoreSelectedBackup);
  initImportBlock({ kind: 'invoices', buttonId: 'invoiceCsvPreview' });
  initImportBlock({ kind: 'expenses', buttonId: 'expenseCsvPreview' });

  if (typeof createUploadWidget === 'function') {
    createUploadWidget({
      dropzoneId: 'restoreDropzone',
      inputId: 'restoreFile',
      listId: 'restoreFileList',
      multiple: false,
      accept: '.json,application/json',
      label: 'Choose a backup file',
    });
    createUploadWidget({
      dropzoneId: 'invoiceCsvDropzone',
      inputId: 'invoiceCsvFile',
      listId: 'invoiceCsvFileList',
      multiple: false,
      accept: '.csv,text/csv',
      label: 'Choose a CSV file of invoices',
      onChange: () => {
        const host = document.getElementById('invoiceCsvPreviewHost');
        if (host) {
          host.hidden = true;
          host.innerHTML = '';
        }
      },
    });
    createUploadWidget({
      dropzoneId: 'expenseCsvDropzone',
      inputId: 'expenseCsvFile',
      listId: 'expenseCsvFileList',
      multiple: false,
      accept: '.csv,text/csv',
      label: 'Choose a CSV file of expenses',
      onChange: () => {
        const host = document.getElementById('expenseCsvPreviewHost');
        if (host) {
          host.hidden = true;
          host.innerHTML = '';
        }
      },
    });
  }
}

async function initActivityPage() {
  await settingsReady;
  if (document.body.dataset.page !== 'activity') return;
  try {
    const payload = await api('/api/activity?limit=100');
    renderActivity(payload.items || []);
  } catch (error) {
    console.error('Load activity error:', error);
    const body = document.getElementById('activityBody');
    if (body) {
      body.innerHTML = `<tr><td colspan="4"><div class="empty-state compact"><strong>Activity unavailable</strong><p>${esc(error.message)}</p><button class="btn btn-sm" type="button" onclick="initActivityPage()">Retry</button></div></td></tr>`;
    }
  }
}

initSettingsPage();
initActivityPage();

// === spendwise.js ===
/* SpendWise — spending analytics, budgets, and rules for Finance Agent.
   Every figure shown here is calculated by the backend from the same expense
   records used by the Expenses page. This file renders, it never recalculates. */

(function initSpendWise() {
  if (document.body.dataset.page !== 'spendwise') return;

  const state = {
    period: 'month',
    from: '',
    to: '',
    trend: 'monthly',
    matrix: 'category',
    vendorFilter: '',
    categories: [],
    vendors: [],
    editingBudget: null,
    editingRule: null,
  };

  const el = (id) => document.getElementById(id);

  /* --------------------------------------------------------------------
     Data
     -------------------------------------------------------------------- */

  function periodQuery() {
    const params = new URLSearchParams({ period: state.period });
    if (state.period === 'custom') {
      if (state.from) params.set('from_date', state.from);
      if (state.to) params.set('to_date', state.to);
    }
    return params.toString();
  }

  async function loadSpendWise() {
    const trendHost = el('spendwiseTrend');
    try {
      const [overview, rulesPayload, suggestionPayload] = await Promise.all([
        api(`/api/spendwise/overview?${periodQuery()}`),
        api('/api/spendwise/rules'),
        api(`/api/spendwise/rules/suggestions?${periodQuery()}`),
      ]);

      state.categories = overview.categories || [];
      state.vendors = overview.vendors || [];
      state.lastOverview = overview;

      renderMetrics(overview);
      renderTrend(overview);
      renderCash(overview);
      renderCategories(state.categories);
      renderVendors(state.vendors);
      renderMatrix(overview);
      renderBudgets(overview.budgets || [], overview.budget_totals || {});
      renderRules(rulesPayload.rules || []);
      renderSuggestions(suggestionPayload.suggestions || []);
      renderInsights(overview);
      renderPeriodNote(overview.period);
      syncAiPanel();
    } catch (error) {
      console.error('SpendWise load error:', error);
      if (trendHost) {
        trendHost.innerHTML = `<div class="chart-empty"><div><strong>Spending data unavailable</strong><p>${esc(
          error.message
        )}</p></div></div>`;
      }
      toast(error.message, 'error', 'Could not load SpendWise');
    }
  }

  function renderPeriodNote(period) {
    const note = el('spendwisePeriodNote');
    if (!note || !period) return;
    note.textContent = `${fmtDate(period.start, { long: true })} — ${fmtDate(period.end, {
      long: true,
    })}`;
  }

  /* --------------------------------------------------------------------
     Metrics
     -------------------------------------------------------------------- */

  function renderMetrics(data) {
    const stats = data.stats || {};
    const totals = data.budget_totals || {};

    setText('swTotal', money(stats.total));
    setText(
      'swTotalNote',
      stats.count
        ? `${stats.count} transaction${stats.count === 1 ? '' : 's'} · ${money(stats.average_transaction)} average`
        : 'No spending recorded'
    );

    setText('swMonth', money(stats.month_total));
    setText(
      'swMonthNote',
      stats.daily_average ? `${money(stats.daily_average)} per day on average` : 'No daily average yet'
    );

    setText('swAverage', money(stats.average_monthly));
    setText(
      'swAverageNote',
      `Across ${stats.months_spanned || 0} month${stats.months_spanned === 1 ? '' : 's'} in this period`
    );

    const top = stats.top_category;
    setText('swTopCategory', top ? money(top.amount) : '—');
    setText(
      'swTopCategoryNote',
      top
        ? `${categoryLabel(top.name)} · ${money(stats.top_vendor?.amount || 0)} at ${stats.top_vendor?.name || 'no vendor'}`
        : 'No categories yet'
    );

    if (totals.count) {
      const primary = totals.primary;
      setText('swBudget', primary ? `${primary.percent}%` : `${totals.percent}%`);
      setText(
        'swBudgetNote',
        primary
          ? `${primary.name}: ${money(primary.spent)} of ${money(primary.amount)} used`
          : `${money(totals.spent)} of ${money(totals.budget)} used`
      );
    } else {
      setText('swBudget', '—');
      setText('swBudgetNote', 'No budgets configured yet');
    }

    const change = stats.change || {};
    if (change.percent !== null && change.percent !== undefined && Number(stats.previous_total) > 0) {
      const percent = numberValue(change.percent);
      const note = el('swTotalNote');
      if (note) {
        const direction = percent > 0 ? 'up' : 'down';
        note.innerHTML = `${stats.count} transaction${
          stats.count === 1 ? '' : 's'
        } · ${esc(direction)} ${esc(Math.abs(percent).toFixed(1))}% vs previous period`;
      }
    }
  }

  /* --------------------------------------------------------------------
     Trend chart
     -------------------------------------------------------------------- */

  function renderTrend(data) {
    const host = el('spendwiseTrend');
    if (!host) return;
    const buckets = state.trend === 'weekly' ? data.weekly_trend || [] : data.monthly_trend || [];
    const note = el('swTrendNote');
    if (note) {
      note.textContent =
        state.trend === 'weekly'
          ? 'Recorded spending by week'
          : 'Recorded spending by month';
    }

    const highest = buckets.reduce((max, bucket) => Math.max(max, numberValue(bucket.total)), 0);
    if (!buckets.length || highest <= 0) {
      host.innerHTML = '<div class="chart-empty"><p>No spending recorded in this period.</p></div>';
      return;
    }

    const scale = niceCeiling(highest);
    const ticks = [1, 0.75, 0.5, 0.25, 0].map((fraction) => fmtCompact(scale * fraction));
    const groups = buckets
      .map((bucket) => {
        const total = numberValue(bucket.total);
        const height = total > 0 ? Math.max((total / scale) * 100, 1.2) : 0;
        return `
          <div class="chart-group">
            <div class="chart-bar${total > 0 ? '' : ' is-empty'}" style="height:${height.toFixed(
          2
        )}%;${total > 0 ? '' : 'min-height:0;background:transparent'}" data-tip="${esc(
          bucket.label
        )} · ${esc(money(total))}"></div>
            <span class="chart-month">${esc(bucket.label)}</span>
          </div>`;
      })
      .join('');

    host.innerHTML = `
      <div class="chart-shell">
        <div class="chart-y-axis" aria-hidden="true">${ticks.map((tick) => `<span>${esc(tick)}</span>`).join('')}</div>
        <div class="chart-main">
          <div class="chart-gridlines" aria-hidden="true">${ticks.map(() => '<span></span>').join('')}</div>
          <div class="chart-groups">${groups}</div>
        </div>
      </div>`;
  }

  /* --------------------------------------------------------------------
     Cash movement
     -------------------------------------------------------------------- */

  function renderCash(data) {
    const cash = data.cash_flow || {};
    setText('swCashIn', money(cash.incoming));
    setText('swCashOut', money(cash.outgoing));
    const net = numberValue(cash.net);
    setText('swCashNet', money(cash.net));
    const netHost = el('swCashNet');
    if (netHost) netHost.classList.toggle('negative', net < 0);

    const note = el('swCashNote');
    if (note) note.textContent = cash.note || '';

    const host = el('spendwiseCashChart');
    if (!host) return;
    const months = cash.months || [];
    const highest = months.reduce(
      (max, month) => Math.max(max, numberValue(month.incoming), numberValue(month.outgoing)),
      0
    );
    if (!months.length || highest <= 0) {
      host.innerHTML = '<div class="chart-empty"><p>No cash movement in this period.</p></div>';
      return;
    }
    const scale = niceCeiling(highest);
    const ticks = [1, 0.75, 0.5, 0.25, 0].map((fraction) => fmtCompact(scale * fraction));
    const groups = months
      .map((month) => {
        const incoming = numberValue(month.incoming);
        const outgoing = numberValue(month.outgoing);
        const inHeight = incoming > 0 ? Math.max((incoming / scale) * 100, 1.2) : 0;
        const outHeight = outgoing > 0 ? Math.max((outgoing / scale) * 100, 1.2) : 0;
        return `
          <div class="chart-group">
            <div class="chart-bar${incoming > 0 ? '' : ' is-empty'}" style="height:${inHeight.toFixed(
          2
        )}%;${incoming > 0 ? '' : 'min-height:0;background:transparent'}" data-tip="${esc(
          month.label
        )} · In ${esc(money(incoming))}"></div>
            <div class="chart-bar expense${outgoing > 0 ? '' : ' is-empty'}" style="height:${outHeight.toFixed(
          2
        )}%;${outgoing > 0 ? '' : 'min-height:0;background:transparent'}" data-tip="${esc(
          month.label
        )} · Out ${esc(money(outgoing))}"></div>
            <span class="chart-month">${esc(month.label)}</span>
          </div>`;
      })
      .join('');

    host.innerHTML = `
      <div class="chart-shell">
        <div class="chart-y-axis" aria-hidden="true">${ticks.map((tick) => `<span>${esc(tick)}</span>`).join('')}</div>
        <div class="chart-main">
          <div class="chart-gridlines" aria-hidden="true">${ticks.map(() => '<span></span>').join('')}</div>
          <div class="chart-groups">${groups}</div>
        </div>
      </div>`;
  }

  /* --------------------------------------------------------------------
     Categories and vendors
     -------------------------------------------------------------------- */

  function changeCell(row) {
    const change = row?.change;
    if (!change) return '<td class="num muted">—</td>';
    const previousTotal = numberValue(row.previous_total);
    if (previousTotal <= 0) {
      return `<td class="num muted">${numberValue(change.amount) > 0 ? 'new' : '—'}</td>`;
    }
    if (change.percent === null || change.percent === undefined) {
      return '<td class="num muted">—</td>';
    }
    const percent = numberValue(change.percent);
    const tone = percent > 0 ? 'negative' : percent < 0 ? 'positive' : 'muted';
    const arrow = percent > 0 ? '▲' : percent < 0 ? '▼' : '–';
    return `<td class="num ${tone}">${arrow} ${esc(Math.abs(percent).toFixed(1))}%</td>`;
  }

  function renderCategories(rows) {
    const body = el('spendwiseCategories');
    if (!body) return;
    if (!rows.length) {
      body.innerHTML =
        '<tr><td colspan="6"><div class="empty-state compact"><strong>No spending in this period</strong><p>Record an expense or widen the period.</p></div></td></tr>';
      return;
    }
    body.innerHTML = rows
      .map(
        (row) => `
        <tr class="sw-row-click" data-category="${esc(row.category)}" tabindex="0" role="button"
            aria-label="Show transactions for ${esc(row.category)}">
          <td><span class="cell-main">${esc(categoryLabel(row.category))}</span></td>
          <td class="num">${esc(money(row.total))}</td>
          <td class="num">
            <div class="sw-share">
              <span class="sw-share-bar"><span style="width:${Math.min(numberValue(row.share), 100)}%"></span></span>
              <span class="mono">${esc(row.share)}%</span>
            </div>
          </td>
          <td class="num">${row.count}</td>
          ${changeCell(row)}
        </tr>`
      )
      .join('');
  }

  function renderVendors(rows) {
    const body = el('spendwiseVendors');
    const filter = el('spendwiseVendorFilter');
    if (filter) {
      const current = filter.value;
      const options = ['<option value="">All vendors</option>']
        .concat(
          rows.map(
            (row) =>
              `<option value="${esc(row.vendor)}">${esc(row.vendor)} (${esc(row.count)})</option>`
          )
        )
        .join('');
      if (filter.innerHTML !== options) filter.innerHTML = options;
      filter.value = current;
    }
    if (!body) return;

    const visible = state.vendorFilter
      ? rows.filter((row) => row.vendor === state.vendorFilter)
      : rows;
    if (!visible.length) {
      body.innerHTML =
        '<tr><td colspan="6"><div class="empty-state compact"><strong>No vendor spending in this period</strong><p>Expenses without a vendor are grouped as (no vendor).</p></div></td></tr>';
      return;
    }
    body.innerHTML = visible
      .map(
        (row) => `
        <tr>
          <td><span class="cell-main">${esc(row.vendor)}</span></td>
          <td class="num">${esc(money(row.total))}</td>
          <td class="num">
            <div class="sw-share">
              <span class="sw-share-bar"><span style="width:${Math.min(numberValue(row.share), 100)}%"></span></span>
              <span class="mono">${esc(row.share)}%</span>
            </div>
          </td>
          <td class="num">${row.count}</td>
          ${changeCell(row)}
        </tr>`
      )
      .join('');
  }

  async function showCategoryTransactions(category) {
    try {
      const payload = await api(`/api/spendwise/categories/${encodeURIComponent(category)}?${periodQuery()}`);
      const rows = payload.transactions || [];
      const summary = payload.summary;
      const body = rows.length
        ? rows
            .map(
              (row) => `
        <tr>
          <td class="nowrap">${esc(fmtDate(row.date))}</td>
          <td><span class="cell-main">${esc(row.description || '')}</span><span class="cell-sub">${esc(
                row.vendor || ''
              )}</span></td>
          <td><span class="tag">${esc(categoryLabel(row.category || category))}</span></td>
          <td class="num">${esc(money(row.amount))}</td>
        </tr>`
            )
            .join('')
        : `<tr><td colspan="4" class="muted">No transactions in this period.</td></tr>`;

      openModal(`
        ${modalHeader('SpendWise', `${categoryLabel(category)} spending`)}
        <div class="modal-body">
          ${
            summary
              ? `<div class="sw-modal-summary">
                   <span>Total <strong>${esc(money(summary.total))}</strong></span>
                   <span>${summary.count} transaction${summary.count === 1 ? '' : 's'}</span>
                   <span>Average <strong>${esc(money(summary.average))}</strong></span>
                   <span>${esc(summary.share)}% of spending</span>
                 </div>`
              : ''
          }
          <div class="table-wrap">
            <table class="data-table">
              <thead><tr><th scope="col">Date</th><th scope="col">Description</th><th scope="col">Category</th><th scope="col">Amount</th></tr></thead>
              <tbody>${body}</tbody>
            </table>
          </div>
        </div>
        <div class="modal-foot">
          <button class="btn" type="button" data-modal-close>Close</button>
          <a class="btn btn-primary" href="/expenses">Open Expenses</a>
        </div>
      `);
    } catch (error) {
      console.error('Category detail error:', error);
      toast(error.message, 'error', 'Could not load transactions');
    }
  }

  /* --------------------------------------------------------------------
     Budgets
     -------------------------------------------------------------------- */

  function budgetStateLabel(state) {
    if (state === 'over') return 'Over budget';
    if (state === 'near') return 'Nearly spent';
    return 'On track';
  }

  function renderBudgets(rows, totals) {
    const host = el('spendwiseBudgets');
    if (!host) return;
    if (!rows.length) {
      host.innerHTML =
        '<div class="empty-state compact"><strong>No budgets yet</strong><p>Add a monthly limit to track spending against it.</p></div>';
      return;
    }
    host.innerHTML = rows
      .map((row) => {
        const percent = Math.min(numberValue(row.percent), 100);
        const tone = row.state === 'over' ? 'over' : row.state === 'near' ? 'near' : 'ok';
        return `
        <div class="sw-budget">
          <div class="sw-budget-head">
            <div>
              <span class="cell-main">${esc(row.name)}</span>
              <span class="cell-sub">${esc(
          row.category ? `Category: ${categoryLabel(row.category)}` : 'All spending'
        )} · ${esc(row.month)}</span>
            </div>
            <div class="sw-budget-amount">
              <strong>${esc(money(row.spent))}</strong> of ${esc(money(row.amount))}
            </div>
          </div>
          <div class="sw-meter ${tone}"><span style="width:${percent}%"></span></div>
          <div class="sw-budget-foot">
            <span class="sw-state ${tone}">${esc(budgetStateLabel(row.state))} · ${esc(row.percent)}% used</span>
            <span class="muted">${esc(
              numberValue(row.remaining) >= 0 ? `${money(row.remaining)} left` : `${money(
                Math.abs(numberValue(row.remaining))
              )} over`
            )}</span>
            <span class="sw-budget-actions">
              <button class="text-button" type="button" data-budget-edit="${row.id}">Edit</button>
              <button class="text-button danger" type="button" data-budget-delete="${row.id}">Remove</button>
            </span>
          </div>
        </div>`;
      })
      .join('');

    const note = el('spendwiseBudgetNote');
    if (note && totals && totals.count) {
      note.textContent = totals.combined
        ? `Each budget is tracked separately. ${money(totals.spent)} of ${money(
            totals.budget
          )} is spent across ${totals.count} budgets, but the scopes can overlap so the totals are not added together.`
        : `${money(totals.spent)} of ${money(totals.budget)} used.`;
    }
  }

  function fillBudgetForm(budget) {
    state.editingBudget = budget || null;
    el('spendwiseBudgetId').value = budget ? String(budget.id) : '';
    el('spendwiseBudgetName').value = budget?.name || '';
    el('spendwiseBudgetAmount').value = budget?.amount || '';
    el('spendwiseBudgetCategory').value = budget?.category || '';
    el('spendwiseBudgetMonth').value = budget?.month || '';
    el('spendwiseBudgetSubmit').textContent = budget ? 'Update budget' : 'Add budget';
    el('spendwiseBudgetCancel').classList.toggle('hidden', !budget);
  }

  async function submitBudget(event) {
    event.preventDefault();
    const errorBox = el('spendwiseBudgetError');
    const submit = el('spendwiseBudgetSubmit');
    if (errorBox) {
      errorBox.textContent = '';
      errorBox.classList.remove('visible');
    }
    const payload = {
      name: el('spendwiseBudgetName').value.trim(),
      amount: el('spendwiseBudgetAmount').value.trim(),
      category: el('spendwiseBudgetCategory').value || null,
      month: el('spendwiseBudgetMonth').value || '',
    };
    if (submit) submit.disabled = true;
    try {
      if (state.editingBudget) {
        await api(`/api/spendwise/budgets/${state.editingBudget.id}`, { method: 'PUT', body: payload });
        toast('Budget updated.', 'success', 'Budget saved');
      } else {
        await api('/api/spendwise/budgets', { method: 'POST', body: payload });
        toast('Budget added.', 'success', 'Budget saved');
      }
      fillBudgetForm(null);
      await loadSpendWise();
    } catch (error) {
      console.error('Save budget error:', error);
      if (errorBox) {
        errorBox.textContent = error.message;
        errorBox.classList.add('visible');
      }
      toast(error.message, 'error', 'Budget not saved');
    } finally {
      if (submit) submit.disabled = false;
    }
  }

  async function removeBudget(id) {
    const confirmed = await confirmDialog({
      kicker: 'SpendWise',
      title: 'Remove this budget?',
      message: 'The budget is deleted. Recorded expenses are not affected.',
      confirmLabel: 'Remove budget',
      danger: true,
    });
    if (!confirmed) return;
    try {
      await api(`/api/spendwise/budgets/${id}`, { method: 'DELETE' });
      toast('Budget removed.', 'success', 'Budget deleted');
      await loadSpendWise();
    } catch (error) {
      console.error('Delete budget error:', error);
      toast(error.message, 'error', 'Budget not removed');
    }
  }

  /* --------------------------------------------------------------------
     Categorisation rules
     -------------------------------------------------------------------- */

  function renderRules(rules) {
    const body = el('spendwiseRules');
    if (!body) return;
    if (!rules.length) {
      body.innerHTML =
        '<tr><td colspan="6"><div class="empty-state compact"><strong>No rules yet</strong><p>Rules suggest a category from a vendor or a keyword.</p></div></td></tr>';
      return;
    }
    body.innerHTML = rules
      .map((rule) => {
        const matches = [
          rule.vendor ? `vendor contains “${rule.vendor}”` : '',
          rule.keyword ? `description contains “${rule.keyword}”` : '',
        ]
          .filter(Boolean)
          .join(' · ');
        return `
        <tr>
          <td><span class="cell-main">${esc(rule.name)}</span></td>
          <td>${esc(matches)}</td>
          <td><span class="tag">${esc(categoryLabel(rule.category))}</span></td>
          <td class="num mono">${rule.priority}</td>
          <td><span class="sw-state ${rule.enabled ? 'ok' : 'muted'}">${rule.enabled ? 'Active' : 'Disabled'}</span></td>
          <td class="actions-col">
            <div class="row-actions">
              <button class="btn-icon" type="button" data-rule-edit="${rule.id}" title="Edit rule" aria-label="Edit rule ${esc(
          rule.name
        )}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
              </button>
              <button class="btn-icon" type="button" data-rule-toggle="${rule.id}" title="${
          rule.enabled ? 'Disable rule' : 'Enable rule'
        }" aria-label="${rule.enabled ? 'Disable' : 'Enable'} rule ${esc(rule.name)}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${
          rule.enabled
            ? '<path d="M12 5v14M5 12h14"/>'
            : '<circle cx="12" cy="12" r="9"/><path d="M9 12h6"/>'
        }</svg>
              </button>
              <button class="btn-icon" type="button" data-rule-delete="${rule.id}" title="Remove rule" aria-label="Remove rule ${esc(
          rule.name
        )}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
              </button>
            </div>
          </td>
        </tr>`;
      })
      .join('');
  }

  function fillRuleForm(rule) {
    state.editingRule = rule || null;
    el('spendwiseRuleId').value = rule ? String(rule.id) : '';
    el('spendwiseRuleName').value = rule?.name || '';
    el('spendwiseRuleVendor').value = rule?.vendor || '';
    el('spendwiseRuleKeyword').value = rule?.keyword || '';
    el('spendwiseRuleCategory').value = rule?.category || 'other';
    el('spendwiseRulePriority').value = rule?.priority ?? 100;
    el('spendwiseRuleSubmit').textContent = rule ? 'Update rule' : 'Add rule';
    el('spendwiseRuleCancel').classList.toggle('hidden', !rule);
  }

  async function submitRule(event) {
    event.preventDefault();
    const errorBox = el('spendwiseRuleError');
    const submit = el('spendwiseRuleSubmit');
    if (errorBox) {
      errorBox.textContent = '';
      errorBox.classList.remove('visible');
    }
    const payload = {
      name: el('spendwiseRuleName').value.trim(),
      vendor: el('spendwiseRuleVendor').value.trim(),
      keyword: el('spendwiseRuleKeyword').value.trim(),
      category: el('spendwiseRuleCategory').value,
      priority: Number(el('spendwiseRulePriority').value) || 100,
    };
    if (submit) submit.disabled = true;
    try {
      if (state.editingRule) {
        await api(`/api/spendwise/rules/${state.editingRule.id}`, { method: 'PUT', body: payload });
        toast('Rule updated.', 'success', 'Rule saved');
      } else {
        await api('/api/spendwise/rules', { method: 'POST', body: payload });
        toast('Rule added.', 'success', 'Rule saved');
      }
      fillRuleForm(null);
      await loadSpendWise();
    } catch (error) {
      console.error('Save rule error:', error);
      if (errorBox) {
        errorBox.textContent = error.message;
        errorBox.classList.add('visible');
      }
      toast(error.message, 'error', 'Rule not saved');
    } finally {
      if (submit) submit.disabled = false;
    }
  }

  async function toggleRule(id) {
    const rules = await api('/api/spendwise/rules');
    const rule = (rules.rules || []).find((item) => String(item.id) === String(id));
    if (!rule) return;
    try {
      await api(`/api/spendwise/rules/${id}`, { method: 'PUT', body: { enabled: !rule.enabled } });
      await loadSpendWise();
    } catch (error) {
      console.error('Toggle rule error:', error);
      toast(error.message, 'error', 'Rule not updated');
    }
  }

  async function removeRule(id) {
    const confirmed = await confirmDialog({
      kicker: 'SpendWise',
      title: 'Remove this rule?',
      message: 'The rule is deleted. Categories already applied to expenses stay as they are.',
      confirmLabel: 'Remove rule',
      danger: true,
    });
    if (!confirmed) return;
    try {
      await api(`/api/spendwise/rules/${id}`, { method: 'DELETE' });
      toast('Rule removed.', 'success', 'Rule deleted');
      await loadSpendWise();
    } catch (error) {
      console.error('Delete rule error:', error);
      toast(error.message, 'error', 'Rule not removed');
    }
  }

  function renderSuggestions(suggestions) {
    const host = el('spendwiseSuggestions');
    if (!host) return;
    if (!suggestions.length) {
      host.innerHTML = '';
      return;
    }
    const applyable = suggestions.filter((row) => row.applyable).length;
    host.innerHTML = `
      <div class="sw-suggestion-head">
        <h3 class="statement-title">Suggested categories</h3>
        <span class="muted">${applyable} of ${suggestions.length} can be applied without touching a manually set category.</span>
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col">Expense</th>
              <th scope="col">Current</th>
              <th scope="col">Suggested</th>
              <th scope="col">Rule</th>
            </tr>
          </thead>
          <tbody>
            ${suggestions
              .map(
                (row) => `
              <tr>
                <td class="nowrap">${esc(fmtDate(row.date))}</td>
                <td><span class="cell-main">${esc(row.description || '')}</span><span class="cell-sub">${esc(
                  row.vendor || ''
                )}</span></td>
                <td><span class="tag">${esc(categoryLabel(row.current_category))}</span></td>
                <td><span class="tag">${esc(categoryLabel(row.suggested_category))}</span></td>
                <td class="muted">${esc(row.rule_name)}</td>
              </tr>`
              )
              .join('')}
          </tbody>
        </table>
      </div>
      <div class="import-actions">
        <button class="btn btn-primary" type="button" id="spendwiseApplySuggestions" ${
          applyable ? '' : 'disabled'
        }>Apply ${applyable} suggestion${applyable === 1 ? '' : 's'}</button>
        <span class="field-hint">Expenses with a category you set yourself are left untouched.</span>
      </div>`;

    el('spendwiseApplySuggestions')?.addEventListener('click', async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const result = await api('/api/spendwise/rules/apply', { method: 'POST', body: { ids: [] } });
        toast(
          `${result.updated} expense${result.updated === 1 ? '' : 's'} recategorised${
            result.skipped ? `, ${result.skipped} left as set` : ''
          }.`,
          'success',
          'Rules applied'
        );
        await loadSpendWise();
      } catch (error) {
        console.error('Apply rules error:', error);
        toast(error.message, 'error', 'Rules not applied');
        button.disabled = false;
      }
    });
  }

  /* --------------------------------------------------------------------
     Spending over time (category or vendor by month)
     -------------------------------------------------------------------- */

  function renderMatrix(data) {
    const body = el('spendwiseMatrix');
    const table = el('spendwiseMatrixTable');
    const head = el('spendwiseMatrixHead');
    const note = el('swMatrixNote');
    if (!body || !table) return;

    const isVendor = state.matrix === 'vendor';
    const matrix = isVendor ? data.vendor_matrix : data.category_matrix;
    const months = matrix?.months || [];
    const rows = matrix?.rows || [];

    if (note) {
      note.textContent = isVendor
        ? 'Recorded spending per month, by vendor'
        : 'Recorded spending per month, by category';
    }
    if (head) head.textContent = isVendor ? 'Vendor' : 'Category';

    const headerRow = table.querySelector('thead tr');
    if (headerRow) {
      headerRow.innerHTML = `
        <th scope="col">${esc(isVendor ? 'Vendor' : 'Category')}</th>
        ${months.map((month) => `<th scope="col" class="num">${esc(month.label)}</th>`).join('')}
        <th scope="col" class="num">Total</th>`;
    }

    if (!rows.length || !months.length) {
      body.innerHTML = `<tr><td colspan="${months.length + 2}"><div class="empty-state compact"><strong>No spending over time</strong><p>Record expenses in this period to build the table.</p></div></td></tr>`;
      return;
    }

    body.innerHTML = rows
      .map(
        (row) => `
        <tr>
          <td><span class="cell-main">${esc(isVendor ? row.name : categoryLabel(row.name))}</span></td>
          ${row.values.map((value) => `<td class="num">${numberValue(value) > 0 ? esc(money(value)) : '—'}</td>`).join('')}
          <td class="num">${esc(money(row.total))}</td>
        </tr>`
      )
      .join('');
  }

  /* --------------------------------------------------------------------
     Insights
     -------------------------------------------------------------------- */

  function renderInsights(data) {
    const host = el('spendwiseInsights');
    if (host) {
      const insights = data.insights || [];
      host.innerHTML = insights.length
        ? insights
            .map(
              (insight) => `
          <li class="sw-insight ${esc(insight.severity || 'info')}">
            <span class="sw-insight-title">${esc(insight.title)}</span>
            <span class="sw-insight-detail">${esc(insight.detail)}</span>
          </li>`
            )
            .join('')
        : '<li class="muted">No insights for this period. More recorded history gives clearer comparisons.</li>';
    }

    const recurring = el('spendwiseRecurring');
    if (recurring) {
      const rows = data.recurring || [];
      recurring.innerHTML = rows.length
        ? rows
            .map(
              (row) => `
          <li>
            <span class="cell-main">${esc(row.description || '')}</span>
            <span class="cell-sub">${esc(row.vendor)} · ${row.charges} charges · ${esc(
                row.average_amount
              )} average</span>
          </li>`
            )
            .join('')
        : '<li class="muted">No recurring pattern detected yet.</li>';
    }

    const unusual = el('spendwiseUnusual');
    if (unusual) {
      const rows = data.unusual || [];
      unusual.innerHTML = rows.length
        ? rows
            .map(
              (row) => `
          <li>
            <span class="cell-main">${esc(row.description || '')}</span>
            <span class="cell-sub">${esc(row.reason)}</span>
          </li>`
            )
            .join('')
        : '<li class="muted">No unusually large expenses in this period.</li>';
    }
  }

  /* --------------------------------------------------------------------
     Wiring
     -------------------------------------------------------------------- */

  function fillCategorySelects(categories) {
    const budgetCategory = el('spendwiseBudgetCategory');
    if (budgetCategory) {
      const current = budgetCategory.value;
      budgetCategory.innerHTML = ['<option value="">All spending</option>']
        .concat(
          categories.map(
            (category) => `<option value="${esc(category)}">${esc(categoryLabel(category))}</option>`
          )
        )
        .join('');
      budgetCategory.value = current;
    }
    const ruleCategory = el('spendwiseRuleCategory');
    if (ruleCategory) {
      const current = ruleCategory.value;
      ruleCategory.innerHTML = categories
        .map(
          (category) => `<option value="${esc(category)}">${esc(categoryLabel(category))}</option>`
        )
        .join('');
      ruleCategory.value = current || 'other';
    }
  }

  /* --------------------------------------------------------------------
     AI summary panel
     -------------------------------------------------------------------- */

  let aiHadSummary = false;

  function aiTimeLabel(iso) {
    if (!iso) return '';
    const when = new Date(iso);
    if (Number.isNaN(when.getTime())) return '';
    try {
      return when.toLocaleString(undefined, {
        day: 'numeric', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch (error) {
      return when.toISOString().slice(0, 16).replace('T', ' ');
    }
  }

  function setAiState(value, label) {
    const chip = el('swAiState');
    if (!chip) return;
    chip.dataset.state = value;
    chip.textContent = label || value;
  }

  function renderAiSummary(payload) {
    const body = el('swAiBody');
    if (!body) return;
    const summary = payload.summary || payload.computed || {};
    const computedOnly = payload.source === 'computed';
    const groups = [
      ['Highlights', summary.highlights || [], false],
      ['Observations', summary.observations || [], false],
      ['Cautions', summary.cautions || [], true],
      ['Opportunities', summary.opportunities || [], false],
    ];
    const lists = groups
      .filter(([, items]) => items.length)
      .map(([label, items, caution]) => `
        <div>
          <p class="ai-summary__label">${esc(label)}</p>
          <ul class="ai-summary__list">${items
            .map((item) => `<li class="ai-summary__item${caution ? ' ai-summary__caution' : ''}">${esc(item)}</li>`)
            .join('')}</ul>
        </div>`)
      .join('');
    body.className = computedOnly ? 'ai-summary ai-summary__computed' : 'ai-summary';
    body.innerHTML = `
      <p class="ai-summary__headline">${esc(summary.headline || '')}</p>
      ${lists}`;
    const meta = el('swAiMeta');
    if (meta) {
      const when = aiTimeLabel(payload.generated_at);
      const period = state.lastOverview?.period?.label || '';
      meta.textContent = `${when ? `Generated ${when}` : 'Computed by the app'} · ${period}`;
    }
    const note = el('swAiNote');
    if (note) {
      const messages = [];
      if (computedOnly) {
        messages.push(payload.error || 'The AI was unavailable, so these figures are computed by the app.');
      } else if (payload.error) {
        messages.push(payload.error);
      }
      if (payload.verified === false) {
        messages.push(`Some figures are not in the calculated data: ${(payload.unverified_figures || []).join(', ')}.`);
      }
      note.textContent = messages.join(' ');
    }
    aiHadSummary = true;
  }

  async function syncAiPanel() {
    if (!el('swAiPanel')) return;
    try {
      const payload = await api(`/api/summaries/spendwise?${periodQuery()}`);
      const generate = el('swAiGenerate');
      const refresh = el('swAiRefresh');
      if (payload.cached) {
        renderAiSummary(payload);
        setAiState('ready', 'Ready');
        generate?.classList.add('hidden');
        refresh?.classList.remove('hidden');
      } else if (aiHadSummary) {
        setAiState('stale', 'Stale');
        generate?.classList.remove('hidden');
        refresh?.classList.remove('hidden');
        const note = el('swAiNote');
        if (note) note.textContent = 'The figures changed since this summary was generated. Refresh to update it.';
      } else {
        setAiState('idle', 'Idle');
        generate?.classList.remove('hidden');
        refresh?.classList.add('hidden');
      }
    } catch (error) {
      setAiState('error', 'Error');
      const note = el('swAiNote');
      if (note) note.textContent = error.message;
    }
  }

  async function generateAiSummary() {
    if (!el('swAiPanel')) return;
    const body = el('swAiBody');
    const generate = el('swAiGenerate');
    const refresh = el('swAiRefresh');
    setAiState('loading', 'Working');
    if (body) {
      body.className = 'ai-summary';
      body.innerHTML = '<p class="muted">Reading the ledger and writing a summary…</p>';
    }
    if (generate) generate.disabled = true;
    if (refresh) refresh.disabled = true;
    try {
      const payload = await api(`/api/summaries/spendwise/refresh?${periodQuery()}`, { method: 'POST' });
      renderAiSummary(payload);
      setAiState('ready', 'Ready');
      generate?.classList.add('hidden');
      refresh?.classList.remove('hidden');
    } catch (error) {
      setAiState('error', 'Error');
      if (body) body.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
      const note = el('swAiNote');
      if (note) note.textContent = error.message;
    } finally {
      if (generate) generate.disabled = false;
      if (refresh) refresh.disabled = false;
    }
  }

  async function init() {
    await settingsReady;
    const categories = await api('/api/expenses/categories');
    fillCategorySelects(categories);

    el('spendwisePeriod')?.addEventListener('change', (event) => {
      state.period = event.target.value;
      el('spendwiseCustomRange')?.classList.toggle('hidden', state.period !== 'custom');
      if (state.period === 'custom' && !state.from) {
        const today = new Date();
        const first = new Date(today.getFullYear(), today.getMonth(), 1);
        el('spendwiseFrom').value = first.toISOString().slice(0, 10);
        el('spendwiseTo').value = today.toISOString().slice(0, 10);
        state.from = el('spendwiseFrom').value;
        state.to = el('spendwiseTo').value;
      }
      loadSpendWise();
    });

    el('spendwiseApplyRange')?.addEventListener('click', () => {
      state.from = el('spendwiseFrom').value;
      state.to = el('spendwiseTo').value;
      loadSpendWise();
    });

    el('spendwiseRefresh')?.addEventListener('click', loadSpendWise);

    document.querySelectorAll('[data-trend]').forEach((button) => {
      button.addEventListener('click', () => {
        state.trend = button.dataset.trend;
        document.querySelectorAll('[data-trend]').forEach((item) => {
          item.classList.toggle('active', item === button);
        });
        if (state.lastOverview) renderTrend(state.lastOverview);
      });
    });

    document.querySelectorAll('[data-matrix]').forEach((button) => {
      button.addEventListener('click', () => {
        state.matrix = button.dataset.matrix;
        document.querySelectorAll('[data-matrix]').forEach((item) => {
          item.classList.toggle('active', item === button);
        });
        if (state.lastOverview) renderMatrix(state.lastOverview);
      });
    });

    el('spendwiseVendorFilter')?.addEventListener('change', (event) => {
      state.vendorFilter = event.target.value;
      renderVendors(state.vendors);
    });

    el('spendwiseCategories')?.addEventListener('click', (event) => {
      const row = event.target.closest('[data-category]');
      if (row) showCategoryTransactions(row.dataset.category);
    });

    el('spendwiseCategories')?.addEventListener('keydown', (event) => {
      const row = event.target.closest('[data-category]');
      if (row && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        showCategoryTransactions(row.dataset.category);
      }
    });

    el('spendwiseBudgetForm')?.addEventListener('submit', submitBudget);
    el('spendwiseBudgetCancel')?.addEventListener('click', () => fillBudgetForm(null));
    el('spendwiseBudgets')?.addEventListener('click', (event) => {
      const edit = event.target.closest('[data-budget-edit]');
      const remove = event.target.closest('[data-budget-delete]');
      if (edit) {
        const budget = (state.lastOverview?.budgets || []).find(
          (row) => String(row.id) === String(edit.dataset.budgetEdit)
        );
        if (budget) {
          fillBudgetForm({
            id: budget.id,
            name: budget.name,
            amount: budget.amount,
            category: budget.category,
            month: budget.month,
          });
        }
      }
      if (remove) removeBudget(remove.dataset.budgetDelete);
    });

    el('spendwiseRuleForm')?.addEventListener('submit', submitRule);
    el('spendwiseRuleCancel')?.addEventListener('click', () => fillRuleForm(null));
    el('spendwiseRules')?.addEventListener('click', (event) => {
      const edit = event.target.closest('[data-rule-edit]');
      const toggle = event.target.closest('[data-rule-toggle]');
      const remove = event.target.closest('[data-rule-delete]');
      if (edit) {
        api('/api/spendwise/rules').then((payload) => {
          const rule = (payload.rules || []).find(
            (item) => String(item.id) === String(edit.dataset.ruleEdit)
          );
          if (rule) fillRuleForm(rule);
        });
      }
      if (toggle) toggleRule(toggle.dataset.ruleToggle);
      if (remove) removeRule(remove.dataset.ruleDelete);
    });

    if (typeof initImportBlock === 'function') {
      initImportBlock({
        kind: 'expenses',
        buttonId: 'spendwiseCsvPreview',
        fileInputId: 'spendwiseCsvFile',
        hostId: 'spendwiseCsvPreviewHost',
        onImported: () => {
          loadSpendWise();
        },
      });
    }

    if (typeof createUploadWidget === 'function') {
      createUploadWidget({
        dropzoneId: 'spendwiseCsvDropzone',
        inputId: 'spendwiseCsvFile',
        listId: 'spendwiseCsvFileList',
        accept: '.csv,text/csv',
        multiple: false,
        label: 'Choose a CSV file of expenses',
        onChange: () => {
          const host = el('spendwiseCsvPreviewHost');
          if (host) {
            host.hidden = true;
            host.innerHTML = '';
          }
        },
      });
    }

    el('swAiGenerate')?.addEventListener('click', generateAiSummary);
    el('swAiRefresh')?.addEventListener('click', generateAiSummary);

    await loadSpendWise();
  }

  init();
})();


// === spendwise_ai.js ===
/* SpendWise AI — What-if Simulator, Subscription Optimiser, spending analysis
   suggestions, and a document overview, inside the SpendWise page.

   Everything rendered here comes from the backend. The simulator sends slider
   values, never totals; the backend recalculates. No API key is ever read or
   sent from this file. */

(function initSpendWiseAI() {
  if (document.body.dataset.page !== 'spendwise') return;

  const el = (id) => document.getElementById(id);
  const SIM_PERIOD = '6m';

  const state = {
    baseline: null,
    simulation: null,
    subscriptions: [],
    activeKey: '',
    currency: 'EUR',
    aiAvailable: false,
    simTimer: null,
    loading: false,
  };

  /* --------------------------------------------------------------------
     Small helpers
     -------------------------------------------------------------------- */

  function setState(id, value, label) {
    const node = el(id);
    if (!node) return;
    node.dataset.state = value;
    node.textContent = label;
  }

  function ul(items, render, emptyText) {
    if (!items || !items.length) {
      return `<li class="muted">${esc(emptyText)}</li>`;
    }
    return items.map(render).join('');
  }

  function aiHint(node, message, tone) {
    if (!node) return;
    node.textContent = message || '';
    node.dataset.tone = tone || '';
  }

  /* --------------------------------------------------------------------
     What-if Simulator
     -------------------------------------------------------------------- */

  const SLIDERS = [
    { range: 'swSimSubs', num: 'swSimSubsNum' },
    { range: 'swSimIncome', num: 'swSimIncomeNum' },
    { range: 'swSimExpenses', num: 'swSimExpensesNum' },
    { range: 'swSimSavings', num: 'swSimSavingsNum' },
  ];

  function sliderValue(baseId) {
    const num = el(baseId);
    return num ? numberValue(num.value) : 0;
  }

  function linkSlider(rangeId, numId) {
    const range = el(rangeId);
    const num = el(numId);
    if (!range || !num) return;
    range.addEventListener('input', () => {
      num.value = range.value;
      scheduleSimulate();
    });
    num.addEventListener('input', () => {
      range.value = num.value;
      scheduleSimulate();
    });
  }

  function scheduleSimulate() {
    if (state.simTimer) window.clearTimeout(state.simTimer);
    state.simTimer = window.setTimeout(simulate, 180);
  }

  async function loadBaseline() {
    try {
      const payload = await api(`/api/spendwise/whatif/baseline?period=${SIM_PERIOD}`);
      state.baseline = payload.baseline;
      state.currency = payload.currency || state.currency;
      state.aiAvailable = Boolean(payload.ai_available);
      renderBaseline(payload.baseline);

      const subsSlider = el('swSimSubs');
      const subsNum = el('swSimSubsNum');
      const start = Math.round(numberValue(payload.baseline.subscriptions));
      if (subsSlider && subsNum) {
        const max = Math.max(2000, start * 2);
        subsSlider.max = String(max);
        subsNum.max = String(max);
        subsSlider.value = String(start);
        subsNum.value = String(start);
      }
      const subsHint = el('swSimSubsHint');
      if (subsHint) {
        subsHint.textContent = `Detected from recurring charges: ${money(payload.baseline.subscriptions)} per month.`;
      }
      simulate();
    } catch (error) {
      const host = el('swSimBaseline');
      if (host) host.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    }
  }

  function renderBaseline(baseline) {
    const host = el('swSimBaseline');
    if (!host || !baseline) return;
    const categories = Object.entries(baseline.categories || {})
      .sort((a, b) => numberValue(b[1]) - numberValue(a[1]))
      .slice(0, 5);
    host.innerHTML = `
      <p class="sw-sim-label">Baseline · last six months, averaged per month</p>
      <div class="sw-sim-base-grid">
        <span><em>Income</em><strong>${money(baseline.income)}</strong></span>
        <span><em>Expenses</em><strong>${money(baseline.expenses)}</strong></span>
        <span><em>Subscriptions</em><strong>${money(baseline.subscriptions)}</strong></span>
      </div>
      ${
        categories.length
          ? `<p class="field-hint">Largest categories: ${categories
              .map(([name, total]) => `${esc(name)} ${money(total)}`)
              .join(' · ')}</p>`
          : ''
      }`;
  }

  function currentScenario() {
    const scenario = {};
    const income = sliderValue('swSimIncomeNum');
    const expenses = sliderValue('swSimExpensesNum');
    const savings = sliderValue('swSimSavingsNum');
    const oneTime = numberValue(el('swSimOneTime')?.value);
    if (income) scenario.income_delta = income;
    if (expenses) scenario.expenses_delta = expenses;
    if (savings) scenario.savings_delta = savings;
    if (oneTime) scenario.one_time_purchase = oneTime;
    if (state.baseline) {
      const target = sliderValue('swSimSubsNum');
      const original = numberValue(state.baseline.subscriptions);
      if (Math.abs(target - original) > 0.001) scenario.subscriptions_target = target;
    }
    return scenario;
  }

  async function simulate() {
    const cards = el('swSimCards');
    if (!cards) return;
    const scenario = currentScenario();
    try {
      const payload = await api('/api/spendwise/whatif/simulate', {
        method: 'POST',
        body: { period: SIM_PERIOD, scenario },
      });
      state.simulation = payload.simulation;
      renderSimulation(payload.simulation);
    } catch (error) {
      cards.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    }
  }

  function simCard(title, situation, difference) {
    const rows = [
      ['Income', situation.income],
      ['Expenses', situation.expenses],
      ['Savings', situation.savings],
      ['Free cash', situation.free_cash],
    ]
      .map(
        ([label, value]) => `
        <div class="sw-sim-row"><span>${label}</span><strong>${money(value)}</strong></div>`
      )
      .join('');
    let foot = '';
    if (difference) {
      const monthly = numberValue(difference.monthly);
      const annual = numberValue(difference.annual);
      const oneTime = numberValue(difference.one_time_purchase);
      foot = `
        ${
          oneTime
            ? `<div class="sw-sim-row sw-sim-diff"><span>One-time purchase</span><strong class="is-negative">${signedMoney(-oneTime)}</strong></div>`
            : ''
        }
        <div class="sw-sim-row sw-sim-diff"><span>Per month</span><strong class="${classFor(monthly)}">${signedMoney(monthly)}</strong></div>
        <div class="sw-sim-row sw-sim-diff"><span>Per year</span><strong class="${classFor(annual)}">${signedMoney(annual)}</strong></div>`;
    }
    return `
      <article class="sw-sim-card">
        <p class="sw-sim-card-title">${title}</p>
        ${rows}
        ${foot}
      </article>`;
  }

  function classFor(value) {
    if (value > 0.0001) return 'is-positive';
    if (value < -0.0001) return 'is-negative';
    return '';
  }

  function signedMoney(value) {
    const prefix = value > 0.0001 ? '+' : '';
    return `${prefix}${money(value)}`;
  }

  function renderSimulation(sim) {
    const cards = el('swSimCards');
    if (!cards || !sim) return;
    cards.innerHTML =
      simCard('Current situation', sim.current, null) +
      simCard('Simulated situation', sim.simulated, sim.difference);
  }

  function resetSimulation() {
    const original = state.baseline ? Math.round(numberValue(state.baseline.subscriptions)) : 0;
    const setPair = (rangeId, numId, value) => {
      const range = el(rangeId);
      const num = el(numId);
      if (range) range.value = String(value);
      if (num) num.value = String(value);
    };
    setPair('swSimSubs', 'swSimSubsNum', original);
    setPair('swSimIncome', 'swSimIncomeNum', 0);
    setPair('swSimExpenses', 'swSimExpensesNum', 0);
    setPair('swSimSavings', 'swSimSavingsNum', 0);
    if (el('swSimOneTime')) el('swSimOneTime').value = '0';
    if (el('swSimText')) el('swSimText').value = '';
    aiHint(el('swSimAiNote'), '');
    const explain = el('swSimExplain');
    if (explain) explain.innerHTML = '';
    simulate();
  }

  async function interpretScenario() {
    const input = el('swSimText');
    const note = el('swSimAiNote');
    const button = el('swSimInterpret');
    const text = (input?.value || '').trim();
    if (!text) {
      aiHint(note, 'Describe a change first, such as “reduce subscriptions by 50”.', 'notice');
      return;
    }
    if (button) button.disabled = true;
    aiHint(note, 'Interpreting…');
    try {
      const payload = await api('/api/spendwise/whatif/interpret', {
        method: 'POST',
        body: { text, period: SIM_PERIOD },
      });
      if (payload.ai_available === false) {
        aiHint(note, payload.message || 'Natural-language scenarios are unavailable.', 'notice');
        return;
      }
      applyScenarioToControls(payload.scenario);
      renderSimulation(payload.simulation);
      state.simulation = payload.simulation;
      const notes = (payload.simulation.notes || []).join(' ');
      aiHint(note, notes ? `Interpreted: ${notes}` : 'Scenario applied.', 'ok');
      const explain = el('swSimExplain');
      if (explain) {
        if (payload.explanation) {
          explain.innerHTML = `<p class="ai-summary-text">${esc(payload.explanation)}</p>`;
        } else if (payload.explanation_error) {
          explain.innerHTML = `<p class="field-hint">${esc(payload.explanation_error)}</p>`;
        }
      }
    } catch (error) {
      aiHint(note, error.message, 'error');
    } finally {
      if (button) button.disabled = false;
    }
  }

  function applyScenarioToControls(scenario) {
    const sc = scenario || {};
    if (sc.subscriptions_target !== null && sc.subscriptions_target !== undefined && sc.subscriptions_target !== '') {
      setPair('swSimSubs', 'swSimSubsNum', Math.round(numberValue(sc.subscriptions_target)));
    } else if (numberValue(sc.subscriptions_delta)) {
      const base = numberValue(state.baseline?.subscriptions);
      setPair('swSimSubs', 'swSimSubsNum', Math.max(0, Math.round(base + numberValue(sc.subscriptions_delta))));
    }
    if (sc.income_change_percent !== null && sc.income_change_percent !== undefined && sc.income_change_percent !== '') {
      const base = numberValue(state.baseline?.income);
      setPair('swSimIncome', 'swSimIncomeNum', Math.round((base * numberValue(sc.income_change_percent)) / 100));
    } else if (numberValue(sc.income_delta)) {
      setPair('swSimIncome', 'swSimIncomeNum', Math.round(numberValue(sc.income_delta)));
    }
    if (numberValue(sc.expenses_delta)) {
      setPair('swSimExpenses', 'swSimExpensesNum', Math.round(numberValue(sc.expenses_delta)));
    }
    if (numberValue(sc.bills_delta)) {
      const current = numberValue(el('swSimExpensesNum')?.value);
      setPair('swSimExpenses', 'swSimExpensesNum', Math.round(current + numberValue(sc.bills_delta)));
    }
    if (numberValue(sc.savings_delta)) {
      setPair('swSimSavings', 'swSimSavingsNum', Math.round(numberValue(sc.savings_delta)));
    }
    if (numberValue(sc.one_time_purchase) && el('swSimOneTime')) {
      el('swSimOneTime').value = String(numberValue(sc.one_time_purchase));
    }
  }

  function setPair(rangeId, numId, value) {
    const range = el(rangeId);
    const num = el(numId);
    if (range) range.value = String(value);
    if (num) num.value = String(value);
  }

  /* --------------------------------------------------------------------
     Spending analysis: potential savings + AI suggestions
     -------------------------------------------------------------------- */

  async function loadSavings() {
    const host = el('spendwiseSavings');
    if (!host) return;
    try {
      const payload = await api('/api/spendwise/analysis?period=month&ai_suggestions=0');
      host.innerHTML = ul(
        payload.savings,
        (row) => `
        <li class="sw-list-item">
          <div class="sw-list-main">
            <strong>${esc(row.label)}</strong>
            <span class="muted">${esc(row.detail)}</span>
          </div>
          <span class="sw-list-value">${money(row.monthly)}<em>/mo</em></span>
        </li>`,
        'No obvious savings opportunities in this period.'
      );
    } catch (error) {
      host.innerHTML = `<li class="muted">${esc(error.message)}</li>`;
    }
  }

  async function analyseSpending() {
    const block = el('spendwiseAiBlock');
    const list = el('spendwiseAiSuggestions');
    const note = el('spendwiseAiNote');
    const button = el('spendwiseAnalyse');
    if (button) button.disabled = true;
    setState('swAnalysisState', 'loading', 'Analysing…');
    try {
      const payload = await api('/api/spendwise/analysis?period=month&ai_suggestions=1');
      if (block) block.hidden = false;
      if (list) {
        list.innerHTML = ul(
          payload.suggestions,
          (row) => `
          <li class="sw-list-item">
            <div class="sw-list-main">
              <strong>${esc(row.title)}</strong>
              ${row.detail ? `<span class="muted">${esc(row.detail)}</span>` : ''}
              ${row.refers_to ? `<span class="ai-ref">Based on: ${esc(row.refers_to)}</span>` : ''}
            </div>
          </li>`,
          'No suggestions were returned for this period.'
        );
      }
      if (payload.suggestions.length) {
        setState('swAnalysisState', 'ready', 'Suggestions ready');
        aiHint(note, 'AI suggestions are ideas, not verified figures. The amounts above are calculated by the app.');
      } else if (payload.ai_error) {
        setState('swAnalysisState', 'unavailable', 'AI unavailable');
        aiHint(note, payload.ai_error, 'notice');
      } else {
        setState('swAnalysisState', 'ready', 'No suggestions');
        aiHint(note, 'The AI did not return any suggestions for this period.');
      }
      if (payload.savings) {
        const host = el('spendwiseSavings');
        if (host) {
          host.innerHTML = ul(
            payload.savings,
            (row) => `
            <li class="sw-list-item">
              <div class="sw-list-main">
                <strong>${esc(row.label)}</strong>
                <span class="muted">${esc(row.detail)}</span>
              </div>
              <span class="sw-list-value">${money(row.monthly)}<em>/mo</em></span>
            </li>`,
            'No obvious savings opportunities in this period.'
          );
        }
      }
    } catch (error) {
      setState('swAnalysisState', 'error', 'Failed');
      if (block) block.hidden = false;
      if (list) list.innerHTML = `<li class="muted">${esc(error.message)}</li>`;
    } finally {
      if (button) button.disabled = false;
    }
  }

  /* --------------------------------------------------------------------
     Subscription optimiser
     -------------------------------------------------------------------- */

  async function loadSubscriptions() {
    const list = el('swSubsList');
    const summary = el('swSubsSummary');
    if (!list) return;
    list.innerHTML = '<li class="muted">Reading recurring charges…</li>';
    try {
      const payload = await api('/api/spendwise/subscriptions');
      state.subscriptions = payload.subscriptions || [];
      state.currency = payload.currency || state.currency;
      state.aiAvailable = Boolean(payload.ai_available);
      if (summary) {
        const totals = payload.totals || {};
        summary.innerHTML = state.subscriptions.length
          ? `<span><em>Detected subscriptions</em><strong>${totals.count}</strong></span>
             <span><em>Total monthly</em><strong>${money(totals.monthly)}</strong></span>
             <span><em>Total yearly</em><strong>${money(totals.annual)}</strong></span>`
          : '';
      }
      list.innerHTML = ul(
        state.subscriptions,
        (row) => `
        <li class="sw-subs-item">
          <div class="sw-subs-main">
            <strong>${esc(row.vendor)}</strong>
            <span class="muted">${esc(row.description || 'recurring charge')} · ${row.charges} charges over ${row.months} months</span>
          </div>
          <div class="sw-subs-cost">
            <span>${money(row.monthly_cost)}<em>/mo</em></span>
            <span class="muted">${money(row.annual_cost)}<em>/yr</em></span>
          </div>
          <button class="btn btn-small" type="button" data-subs-key="${esc(row.key)}">Compare</button>
        </li>`,
        'No recurring subscriptions were detected in the ledger yet.'
      );
      list.querySelectorAll('[data-subs-key]').forEach((button) => {
        button.addEventListener('click', () => openSubscription(button.dataset.subsKey));
      });
    } catch (error) {
      list.innerHTML = `<li class="muted">${esc(error.message)}</li>`;
    }
  }

  function openSubscription(key) {
    const row = state.subscriptions.find((item) => item.key === key);
    const detail = el('swSubsDetail');
    if (!row || !detail) return;
    state.activeKey = key;
    detail.hidden = false;
    detail.innerHTML = `
      <div class="sw-subs-detail-head">
        <div>
          <h3 class="statement-title">${esc(row.vendor)} — ${money(row.monthly_cost)}/month</h3>
          <p class="muted">Costs are detected from the ledger. Alternatives below are AI suggestions.</p>
        </div>
        <button class="btn btn-small" type="button" id="swSubsClose">Close</button>
      </div>
      <div class="field">
        <label class="field-label" for="swSubsUsage">What do you use ${esc(row.vendor)} for?</label>
        <textarea class="input" id="swSubsUsage" rows="2" maxlength="600" placeholder="e.g. video editing, document export, code assistance"></textarea>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" type="button" id="swSubsFind">Find alternatives</button>
      </div>
      <p class="field-hint" id="swSubsNote">This app never cancels a subscription or buys anything for you.</p>
      <div class="sw-subs-alts" id="swSubsAlts"></div>`;
    el('swSubsClose')?.addEventListener('click', () => {
      detail.hidden = true;
      detail.innerHTML = '';
    });
    el('swSubsFind')?.addEventListener('click', () => findAlternatives(row));
  }

  async function findAlternatives(row) {
    const alts = el('swSubsAlts');
    const note = el('swSubsNote');
    const button = el('swSubsFind');
    const usage = el('swSubsUsage')?.value || '';
    if (button) button.disabled = true;
    if (alts) alts.innerHTML = '<p class="muted">Asking for alternatives…</p>';
    try {
      const payload = await api('/api/spendwise/subscriptions/alternatives', {
        method: 'POST',
        body: { key: row.key, usage },
      });
      if (alts) {
        if (!payload.alternatives.length) {
          alts.innerHTML = `<div class="empty-state compact"><strong>No alternatives to show</strong><p>${esc(
            payload.message || 'The AI did not return any alternatives.'
          )}</p></div>`;
        } else {
          alts.innerHTML = payload.alternatives.map(renderAlternative).join('');
        }
      }
      if (note) note.textContent = payload.disclaimer || 'Verify every price before switching.';
    } catch (error) {
      if (alts) alts.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    } finally {
      if (button) button.disabled = false;
    }
  }

  function renderAlternative(alt) {
    const flags = (alt.flags || [])
      .map((flag) => `<li class="sw-alt-flag">${esc(flag)}</li>`)
      .join('');
    const listBlock = (title, items) =>
      items && items.length
        ? `<div class="sw-alt-block"><p class="sw-alt-label">${title}</p><ul>${items
            .map((item) => `<li>${esc(item)}</li>`)
            .join('')}</ul></div>`
        : '';
    const saving = alt.savings_monthly !== null && alt.savings_monthly !== undefined;
    return `
      <article class="sw-alt">
        <header>
          <div>
            <h4>${esc(alt.product)}</h4>
            <p class="muted">${money(alt.estimated_monthly_cost)}<em>/mo</em> · ${money(alt.estimated_yearly_cost)}<em>/yr</em></p>
          </div>
          ${
            saving
              ? `<span class="sw-alt-saving ${classFor(numberValue(alt.savings_monthly))}">${
                  numberValue(alt.savings_monthly) > 0 ? 'saves ' : ''
                }${signedMoney(numberValue(alt.savings_monthly))}/mo</span>`
              : ''
          }
        </header>
        ${listBlock('Capabilities', alt.capabilities)}
        ${listBlock('Limitations', alt.limitations)}
        ${listBlock('Trade-offs', alt.trade_offs)}
        ${alt.notes ? `<p class="field-hint">${esc(alt.notes)}</p>` : ''}
        <p class="sw-alt-source">
          ${
            alt.verified
              ? `Price source: ${esc(alt.pricing_source)}${alt.verified_date ? ` · verified ${esc(alt.verified_date)}` : ''}`
              : 'Price source not provided — treat this cost as unverified.'
          }
        </p>
        ${flags ? `<ul class="sw-alt-flags">${flags}</ul>` : ''}
      </article>`;
  }

  /* --------------------------------------------------------------------
     Document analysis overview
     -------------------------------------------------------------------- */

  async function loadDocuments() {
    const host = el('swDocsList');
    if (!host) return;
    try {
      const payload = await api('/api/documents');
      const rows = (payload.documents || []).slice(0, 6);
      host.innerHTML = ul(
        rows,
        (row) => `
        <li class="sw-docs-item">
          <div class="sw-docs-main">
            <strong>${esc(row.filename)}</strong>
            <span class="muted">${esc(row.document_type || row.status || 'pending')}${
          row.record ? ` · linked to ${esc(row.record.kind)} ${esc(row.record.reference || '')}` : ''
        }</span>
          </div>
          <span class="sw-docs-status" data-state="${esc(row.status || '')}">${esc(row.status || '')}</span>
        </li>`,
        'No documents uploaded yet. Open the Documents page to add one.'
      );
    } catch (error) {
      host.innerHTML = `<li class="muted">${esc(error.message)}</li>`;
    }
  }

  /* --------------------------------------------------------------------
     Init
     -------------------------------------------------------------------- */

  SLIDERS.forEach((pair) => linkSlider(pair.range, pair.num));
  el('swSimOneTime')?.addEventListener('input', scheduleSimulate);
  el('swSimReset')?.addEventListener('click', resetSimulation);
  el('swSimInterpret')?.addEventListener('click', interpretScenario);
  el('swSimText')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      interpretScenario();
    }
  });
  el('spendwiseAnalyse')?.addEventListener('click', analyseSpending);
  el('swSubsRefresh')?.addEventListener('click', loadSubscriptions);
  el('spendwisePeriod')?.addEventListener('change', loadSavings);

  Promise.all([loadBaseline(), loadSavings(), loadSubscriptions(), loadDocuments()]).catch(() => {});
})();


// === documents.js ===
/* Documents — intelligent document intake for Finance Agent.
   Uploads files, shows the extraction for review, and imports a record only
   after the user confirms. All figures come from the backend. */

(function initDocuments() {
  if (document.body.dataset.page !== 'documents') return;

  const el = (id) => document.getElementById(id);
  const bytes = window.formatFileBytes || ((value) => `${value} bytes`);
  const results = new Map();
  let widget = null;

  /* --------------------------------------------------------------------
     Helpers
     -------------------------------------------------------------------- */

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || '');
        const comma = result.indexOf(',');
        resolve(comma === -1 ? '' : result.slice(comma + 1));
      };
      reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
      reader.readAsDataURL(file);
    });
  }

  function statusBadge(status, imported) {
    if (imported) return '<span class="doc-badge doc-badge--imported">Imported</span>';
    const value = status === 'processed' ? 'processed' : status === 'failed' ? 'failed' : status;
    return `<span class="doc-badge doc-badge--${value}">${esc(value)}</span>`;
  }

  function fieldEntry(name, entry) {
    const value = entry && typeof entry === 'object' ? entry.value : entry;
    const uncertain = entry && typeof entry === 'object' && entry.uncertain;
    const label = String(name).replace(/_/g, ' ');
    const text = value === null || value === undefined ? '' : String(value);
    return `
      <div class="doc-field${uncertain ? ' doc-field--uncertain' : ''}">
        <label class="doc-field__label" for="doc-${esc(name)}">${esc(label)}</label>
        <input class="doc-field__value" id="doc-${esc(name)}" data-field="${esc(name)}"
          value="${esc(text)}" placeholder="Unstated">
      </div>`;
  }

  function lineRow(line) {
    return `
      <tr data-line>
        <td><input data-line-description value="${esc(line.description || '')}" placeholder="Description"></td>
        <td><input data-line-quantity value="${esc(line.quantity ?? '')}" placeholder="1"></td>
        <td><input data-line-unit-price value="${esc(line.unit_price ?? '')}" placeholder="0.00"></td>
        <td><input data-line-amount value="${esc(line.amount ?? '')}" placeholder="0.00"></td>
      </tr>`;
  }

  const MANUAL_FIELDS = {
    expense: [
      ['description', 'Description'],
      ['vendor', 'Vendor'],
      ['amount', 'Amount'],
      ['date', 'Date'],
      ['category', 'Category'],
    ],
    invoice: [
      ['customer', 'Customer'],
      ['issue_date', 'Issue date'],
      ['subtotal', 'Subtotal'],
      ['tax', 'Tax'],
      ['total', 'Total'],
    ],
  };

  function manualFields(destination) {
    const fields = MANUAL_FIELDS[destination] || [];
    return fields
      .map(([name, label]) => `
        <div class="doc-field">
          <label class="doc-field__label" for="manual-${esc(name)}">${esc(label)}</label>
          <input class="doc-field__value" id="manual-${esc(name)}" data-field="${esc(name)}"
            value="" placeholder="${esc(label)}">
        </div>`)
      .join('');
  }

  function resultCard(result) {
    results.set(String(result.id), result);
    const extraction = result.extraction || {};
    const fields = extraction.fields || {};
    const lines = extraction.line_items || [];
    const flags = result.flags || [];
    const duplicates = result.duplicates || [];
    const imported = result.record && result.record.kind;

    if (result.status === 'failed') {
      return `
        <div class="doc-card" data-id="${result.id}">
          <div class="doc-card__head">
            <div>
              <p class="doc-card__title">${esc(result.filename)}</p>
              <p class="doc-card__meta">${esc((result.extension || '').replace('.', '').toUpperCase())} · ${bytes(result.size || 0)}</p>
            </div>
            ${statusBadge(result.status, imported)}
          </div>
          <p class="doc-error">${esc(result.error || 'This document could not be processed.')}</p>
          <div class="doc-manual">
            <p class="field-hint">The AI could not read this file. Enter the details yourself to save a record. Nothing is saved until you choose a destination and confirm the import.</p>
            <div class="doc-import-bar">
              <button class="btn" type="button" data-retry="${result.id}">Retry processing</button>
              <select data-manual-destination aria-label="Where to save this document">
                <option value="">Save manually as...</option>
                <option value="expense">Expense</option>
                <option value="invoice">Invoice</option>
                <option value="details">Save details only</option>
              </select>
              <button class="btn btn-primary" type="button" data-import="${result.id}" disabled>Import to ledger</button>
            </div>
            <div class="doc-fields" data-manual-fields></div>
            <p class="doc-error" data-import-error="${result.id}" hidden></p>
          </div>
        </div>`;
    }

    const fieldsHtml = Object.keys(fields).length
      ? `<div class="doc-fields">${Object.entries(fields).map(([name, entry]) => fieldEntry(name, entry)).join('')}</div>`
      : '';
    const linesHtml = lines.length
      ? `<div class="doc-lines">
           <table>
             <thead><tr><th scope="col">Description</th><th scope="col">Qty</th><th scope="col">Unit price</th><th scope="col">Amount</th></tr></thead>
             <tbody>${lines.map(lineRow).join('')}</tbody>
           </table>
         </div>`
      : '';
    const flagsHtml = flags
      .map((flag) => `<div class="doc-flag">${esc(flag.message || flag.code || String(flag))}</div>`)
      .join('');
    const duplicatesHtml = duplicates
      .map((dup) => `<div class="doc-duplicate">Possible duplicate: ${esc(dup.reason || '')} (${esc(dup.reference || '')})</div>`)
      .join('');

    const docType = String(extraction.document_type || result.document_type || '').toLowerCase();
    const importable = docType === 'invoice' || docType === 'expense' || docType === 'receipt';
    const importOptions = importable
      ? `<option value="">Save as...</option>
         <option value="expense">Expense</option>
         <option value="invoice">Invoice</option>
         <option value="details">Save details only</option>`
      : `<option value="">Not an invoice or expense</option>
         <option value="details">Save details only</option>`;
    const importLabel = importable ? 'Import to ledger' : 'Save details';
    const importBar = imported
      ? `<p class="field-hint">Filed automatically as ${esc(result.record.kind)} ${esc(result.record.reference || '')}. Edit or delete it on the ${result.record.kind === 'invoice' ? 'Invoices' : 'Expenses'} page.</p>
         <div class="doc-import-bar">
           <button class="btn" type="button" data-open="${result.id}">Reload details</button>
         </div>`
      : `<div class="doc-import-bar">
           <select data-import-destination aria-label="Where to save this document">
             ${importOptions}
           </select>
           <button class="btn btn-primary" type="button" data-import="${result.id}" disabled>${importLabel}</button>
           <button class="btn" type="button" data-open="${result.id}">Reload details</button>
         </div>`;

    return `
      <div class="doc-card" data-id="${result.id}">
        <div class="doc-card__head">
          <div>
            <p class="doc-card__title">${esc(result.filename)}</p>
            <p class="doc-card__meta">${esc((result.extension || '').replace('.', '').toUpperCase())} · ${bytes(result.size || 0)} · uploaded ${esc(result.uploaded_at || '')}</p>
            <span class="doc-type">${esc(extraction.document_type || result.document_type || 'document')}</span>
          </div>
          ${statusBadge(result.status, imported)}
        </div>
        ${extraction.summary ? `<p class="doc-summary">${esc(extraction.summary)}</p>` : ''}
        ${flagsHtml}
        ${duplicatesHtml}
        ${fieldsHtml}
        ${linesHtml}
        ${importBar}
        <p class="doc-error" data-import-error="${result.id}" hidden></p>
      </div>`;
  }

  function renderResults(list) {
    const host = el('docResults');
    if (!host) return;
    if (!list.length) {
      host.innerHTML = '<div class="empty-state compact"><strong>No documents processed yet</strong><p>Upload a file and its extracted details appear here.</p></div>';
      return;
    }
    host.innerHTML = list.map(resultCard).join('');
  }

  function libraryRow(doc) {
    const record = doc.record;
    const recordText = record && record.kind
      ? `${esc(record.reference || '')} <span class="muted">(${esc(record.kind)})</span>`
      : '<span class="muted">—</span>';
    const status = doc.status || 'unknown';
    return `
      <tr data-doc="${doc.id}">
        <td>${esc(doc.filename)}</td>
        <td>${esc((doc.extension || '').replace('.', '').toUpperCase())}</td>
        <td>${esc((doc.uploaded_at || '').replace('T', ' ').slice(0, 16))}</td>
        <td>${statusBadge(status, record && record.kind)}</td>
        <td>${recordText}</td>
        <td>
          <div class="doc-row-actions">
            <button class="btn btn-small" type="button" data-open="${doc.id}">Open</button>
            <button class="btn btn-small" type="button" data-retry="${doc.id}">Retry</button>
            <button class="btn btn-small btn-danger" type="button" data-delete="${doc.id}">Delete</button>
          </div>
        </td>
      </tr>`;
  }

  async function loadLibrary() {
    const host = el('docLibrary');
    if (!host) return;
    try {
      const payload = await api('/api/documents');
      const documents = payload.documents || [];
      if (!documents.length) {
        host.innerHTML = '<tr><td colspan="6"><div class="empty-state compact"><p>No documents yet. Upload a file to start.</p></div></td></tr>';
        return;
      }
      host.innerHTML = documents.map(libraryRow).join('');
    } catch (error) {
      host.innerHTML = `<tr><td colspan="6"><div class="empty-state compact"><p>${esc(error.message)}</p></div></td></tr>`;
    }
  }

  /* --------------------------------------------------------------------
     Upload
     -------------------------------------------------------------------- */

  async function uploadFiles() {
    if (!widget) return;
    const files = widget.getFiles();
    if (!files.length) return;
    const button = el('docUploadButton');
    const progress = el('docUploadProgress');
    const note = el('docUploadNote');
    if (button) button.disabled = true;
    widget.setBusy(true);
    if (note) {
      note.textContent = '';
      note.classList.remove('is-error');
    }
    const collected = [];
    const rejected = [];
    for (let index = 0; index < files.length; index += 1) {
      if (progress) {
        progress.hidden = false;
        progress.textContent = `Processing ${files[index].name}… (${index + 1} of ${files.length})`;
      }
      try {
        const data = await fileToBase64(files[index]);
        const response = await api('/api/documents/upload', {
          method: 'POST',
          body: { files: [{ name: files[index].name, data_base64: data }] },
        });
        collected.push(...(response.results || []));
        rejected.push(...(response.rejected || []));
      } catch (error) {
        rejected.push({ name: files[index].name, error: error.message });
      }
    }
    widget.setBusy(false);
    widget.clear();
    if (button) button.disabled = true;
    if (progress) {
      progress.hidden = true;
      progress.textContent = '';
    }
    if (note && rejected.length) {
      note.textContent = rejected.map((item) => `${item.name}: ${item.error}`).join(' ');
      note.classList.add('is-error');
    }
    renderResults(collected);
    if (collected.length) toast(`${collected.length} document(s) processed.`, 'success', 'Upload complete');
    await loadLibrary();
  }

  async function openDocument(id) {
    try {
      const result = await api(`/api/documents/${id}`);
      renderResults([result]);
      el('docResults')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
      toast(error.message, 'error', 'Could not open document');
    }
  }

  async function retryDocument(id) {
    try {
      const result = await api(`/api/documents/${id}/retry`, { method: 'POST' });
      renderResults([result]);
      await loadLibrary();
    } catch (error) {
      toast(error.message, 'error', 'Retry failed');
    }
  }

  async function deleteDocument(id) {
    const confirmed = await confirmDialog({
      title: 'Delete document',
      message: 'Delete the stored file and its extraction? The ledger record it created stays.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!confirmed) return;
    try {
      await api(`/api/documents/${id}`, { method: 'DELETE' });
      await loadLibrary();
      toast('Document deleted.', 'success');
    } catch (error) {
      toast(error.message, 'error', 'Could not delete');
    }
  }

  async function importDocument(id, destination, card) {
    const errorHost = card.querySelector(`[data-import-error="${id}"]`);
    if (errorHost) {
      errorHost.hidden = true;
      errorHost.textContent = '';
    }
    const fields = {};
    card.querySelectorAll('[data-field]').forEach((input) => {
      fields[input.dataset.field] = input.value;
    });
    const lineItems = [];
    card.querySelectorAll('[data-line]').forEach((row) => {
      lineItems.push({
        description: row.querySelector('[data-line-description]')?.value || '',
        quantity: row.querySelector('[data-line-quantity]')?.value || '1',
        unit_price: row.querySelector('[data-line-unit-price]')?.value || '0',
        amount: row.querySelector('[data-line-amount]')?.value || '',
      });
    });
    const button = card.querySelector(`[data-import="${id}"]`);
    if (button) button.disabled = true;
    try {
      const response = await api(`/api/documents/${id}/import`, {
        method: 'POST',
        body: { destination, fields, line_items: lineItems },
      });
      if (response.created) {
        toast(`Saved as ${response.created.kind} ${response.created.reference}.`, 'success', 'Imported');
      } else {
        toast('Document details saved without a ledger entry.', 'success', 'Saved');
      }
      await loadLibrary();
      await openDocument(id);
    } catch (error) {
      if (errorHost) {
        errorHost.hidden = false;
        errorHost.textContent = error.message;
      }
      toast(error.message, 'error', 'Import failed');
      if (button) button.disabled = false;
    }
  }

  /* --------------------------------------------------------------------
     Wiring
     -------------------------------------------------------------------- */

  document.addEventListener('change', (event) => {
    const select = event.target.closest('[data-import-destination]');
    if (select) {
      const card = select.closest('.doc-card');
      const button = card?.querySelector('[data-import]');
      if (button) button.disabled = !select.value;
      return;
    }
    const manual = event.target.closest('[data-manual-destination]');
    if (manual) {
      const card = manual.closest('.doc-card');
      const host = card?.querySelector('[data-manual-fields]');
      const button = card?.querySelector('[data-import]');
      if (host) host.innerHTML = manual.value && manual.value !== 'details'
        ? manualFields(manual.value)
        : '';
      if (button) button.disabled = !manual.value;
    }
  });

  document.addEventListener('click', (event) => {
    const upload = event.target.closest('#docUploadButton');
    if (upload) {
      uploadFiles();
      return;
    }
    const importButton = event.target.closest('[data-import]');
    if (importButton) {
      const id = importButton.dataset.import;
      const card = importButton.closest('.doc-card');
      const destination = card?.querySelector('[data-import-destination], [data-manual-destination]')?.value;
      if (destination) importDocument(id, destination, card);
      return;
    }
    const open = event.target.closest('[data-open]');
    if (open) {
      openDocument(open.dataset.open);
      return;
    }
    const retry = event.target.closest('[data-retry]');
    if (retry) {
      retryDocument(retry.dataset.retry);
      return;
    }
    const remove = event.target.closest('[data-delete]');
    if (remove) {
      deleteDocument(remove.dataset.delete);
    }
  });

  async function init() {
    await (window.settingsReady || Promise.resolve());
    if (typeof createUploadWidget === 'function') {
      widget = createUploadWidget({
        dropzoneId: 'docDropzone',
        inputId: 'docFileInput',
        listId: 'docFileList',
        noteId: 'docUploadNote',
        multiple: true,
        accept: '.pdf,.csv,.xlsx,.xls,.docx,.doc,.txt,.png,.jpg,.jpeg,.webp',
        maxBytes: 0,
        label: 'Choose files or drop them here',
        onChange: (files) => {
          const button = el('docUploadButton');
          const clear = el('docClearFiles');
          if (button) button.disabled = files.length === 0;
          if (clear) clear.classList.toggle('hidden', files.length === 0);
        },
      });
    }
    el('docClearFiles')?.addEventListener('click', () => {
      widget?.clear();
      const button = el('docUploadButton');
      if (button) button.disabled = true;
      el('docClearFiles')?.classList.add('hidden');
    });
    await loadLibrary();
  }

  init();
})();
"""


SHELL = r"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#0a0a0c">
  <title>{{TITLE}} · Finance Agent</title>
  <link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2IiByb2xlPSJpbWciIGFyaWEtbGFiZWw9IkZpbmFuY2UgQWdlbnQiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJ0aWxlIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzIwMjAxYiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxNDE0MTAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImxldHRlciIgeDE9IjAuMTUiIHkxPSIxIiB4Mj0iMC44NSIgeTI9IjAiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiM4ZTZkMjAiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIwLjMyIiBzdG9wLWNvbG9yPSIjYzlhODQ3Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMC41OCIgc3RvcC1jb2xvcj0iI2YwZDQ4OSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNzgiIHN0b3AtY29sb3I9IiNkM2FlNTIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjOWQ3OTI3Ii8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJlZGdlIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI2I2OTM0MCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNSIgc3RvcC1jb2xvcj0iIzZmNTcyMCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNhOTg1MzIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHg9IjgiIHk9IjgiIHdpZHRoPSIyNDAiIGhlaWdodD0iMjQwIiByeD0iNDgiIGZpbGw9InVybCgjdGlsZSkiIHN0cm9rZT0idXJsKCNlZGdlKSIgc3Ryb2tlLXdpZHRoPSI1Ii8+CiAgPHJlY3QgeD0iMTciIHk9IjE3IiB3aWR0aD0iMjIyIiBoZWlnaHQ9IjIyMiIgcng9IjQwIiBmaWxsPSJub25lIiBzdHJva2U9IiNjOGE4NGIiIHN0cm9rZS1vcGFjaXR5PSIwLjE0IiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY5LjY2IDE4NS45OCkgc2NhbGUoMC4xOCAtMC4xOCkiIGZpbGw9InVybCgjbGV0dGVyKSIgZD0iTTM1MyA3MTEgNTk2IDg0UTYxMSA0OCA2MjguNSAzNC41UTY0NiAyMSA2NjEgMjBWMFE2MzUgMiA1OTguNSAyLjVRNTYyIDMgNTI0IDNRNDgwIDMgNDQyLjAgMi41UTQwNCAyIDM4MCAwVjIwUTQzMSAyMiA0NDUuMCAzNy41UTQ1OSA1MyA0MzkgMTA0TDI2MyA1NzlMMjgxIDYwMUwxMjEgMTg2UTEwMiAxMzYgOTguMCAxMDQuMFE5NCA3MiAxMDMuMCA1NC4wUTExMiAzNiAxMzMuMCAyOC41UTE1NCAyMSAxODUgMjBWMFExNTMgMiAxMjMuMCAyLjVROTMgMyA2MyAzUTQxIDMgMjEuNSAyLjVRMiAyIC0xMyAwVjIwUTkgMjUgMzAuNSA0Ni4wUTUyIDY3IDcyIDExOUwzMDEgNzExUTMxMyA3MTAgMzI3LjAgNzEwLjBRMzQxIDcxMCAzNTMgNzExWk00NDAgMjg4VjI2OEgxNDJMMTUyIDI4OFoiLz4KPC9zdmc+" type="image/svg+xml">
  <link rel="alternate icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAFNUlEQVR4nLxXXWxURRT+7ty7P91tu9uFLhSwLVA0RTEU5U8qIfxHDCE+aeKDJiY+8uRPjCY+EBVNNMZE33zyTWKMxgQ1kjRQHkQqSMDYgtBC+e/aLe3+9d4Z58y9d7v37t3+kMJks+ecOffMmfnmnDMzBqqbduTT7uWiiFZL4wnLtFQnEeIsy2Ys04SpZNvIcpiSozdNlGUGLVvg1tBHXw9cll3C46xS+O6zrZ39g7kPb40UVw2PlJaM58ykkJ8LYdsoHo5cwTtqm6/SA5GQNtoU167Ho9pAOoG3Pv7m8j9VE3jntbW7CiX+1Y+9Iyu50MqD0QiaZsv3Q7mkOmNK1hmwupVdChn89SO/Dv9WngCtvOd05ugPvZlW8soFHkhjmu1xTbs+tHypsftziQSjrv7LucM/nbSdW1yU4ZxvSkML+Xdh0GrNZMQnalIUcMMjhU6Laz7nmm+Q+ZG5WiQwUeCdb7/S3s4o2m/eLaUrA8078wcjZydE2oDRzijV7uXNRgo5O3C8Rq5sU69+9/oGvLgrVVNfy56Qzk/yRugiySjPhZsvFbDZSVQb1nAI2Nedwo71TViYDFXpp7N3YhEGdFAQgnNeYQTfIMHyhtUNqI+HUVcXws6NqTnbqxnoMgipwrnweGGbXt6+PoloNIRIJIS9W1o8g8/GnhCX/sEshcDcUmpRKoRjf2RRnGSIRONINyew6cn0rO3dIgU1AVmryyEwrdGUftvTKfSeHcXxs/cQDsckCvXYv2PVrO3d7dB1igHLLg7VRlrgYLqugTFDHUK/nLyNsEQgHIlh26aVWJCsm9G+Uq9L54xONYHZ5/GWtQvRc3pEyUM3xjF4oygRiMt4iOOFPWtmtC8DQF4JATo1pxDAjHTFIwlcvZkryz+fuIaQRMAIRXBgZ4evDtSmFAMUhAZBKXyw2QHilYm2NMeQaIhh3eq06tOYjjuZvOKL+XEsbGTo6mxG34U7gfYuddIAugG3DlSnip+SfsfmVpzouyV5JuNAVzSTzeHkqQFMFidQLIzjuS1LatoHZYHh3mRcBLwVbEomrrE+hlPn/vb0E03EdXS2dsGcLOCZNQ2IR3V52JjTjkf7YEgI7DoguGfGQPUKNq9bht//uhmo7zl1FWPZrNqGUrGIrV1NNZF07R0A7Dqgrk9iqiIKn0ztwK7H0Xf+eqC+NMlxtHcIuXwRExN5dEsU/PZB4+tUilUd8OWnd4UaOtpS2NTVLk8xHqgn+XjfHeSk81y+gESMo31xuOZ4NgJOJaTbq5jm1GtbmsThN/eiPsJx6OCzeKIjBX+RaWuJoU06zOUKEoEC8pI+v7EOzUkdwUXJyQKdgpBYzssweffM5j/48pikXH2nSrevqHB5xTn/7zjOXRxzILYV7pU9qCg5h6GdBUIFYnDeXrn2HwaHRwPz2aVDtwrT6v1UEKUJUCWkR0PE0MaYNlPtnj9Kq2+oM8YkGWX0YmlqMG4/LOcuXZwybkcNXGH0XGqq1/6kO7sG4bvb1b5c3K+eHic6E1i2KNL30vtnrlApFumk/t7ajuglOJPwB0xlCwqo2eppkXT53d/d1N/WHHnX3Q7V9mxYuk/ekL+/MFg0LD71rNLgrd2Yo+wmHcmGDPvtTyVKiai++4tvL/Z4JkDt4MvLH7t71zw8XuCPZrJmS8EUSRqMMTtAmaYAK5+WzIlmm4oK6v2uMRYabVkQvrE4Fe5f0RJ549VDZwZcn54JuI1eLPRooHs7XZ2dn2q6y0uGKB2p1AzVaSh9wHejFHC0535f/wMAAP//nQNEcgAAAAZJREFUAwAhRAVaL7bKhgAAAABJRU5ErkJggg==" type="image/png">
  <link rel="apple-touch-icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAAAQAElEQVR4nOx9CZQdxZXli/x77VWqklTaV9AOkhDahYSQMBgsYwY8GBvGc8Z093FPu8fDjIf2eCx7zrjtbow93ePxmO7jPraP7Wk4xm0LjFklViFAYhESaF9AElVaal/+ltGxZGRERuZfSqpS/V8VT6f0/suXGRmZeePmyxeRkRYYMTKCxAIjRkaQGEAbGVEShhKUn//t5sqo3TnFyqKxWRRCABn/SpnApYGSyQZsnG9JjoIzRS7IWa9Mgc19zkxRx5gp8kRkitwg69tA8SHAGGdb+5OJkw/+4t0eKDFBMEyybevSilRlxQYbZ1cgZE21s/ZEQGgiOVsTibsKY1I5Ujub/EDAziKrLsY21yBszMqjGpENbBuz7ZzFynaYHS22ecHYKZeX76yH1PKU7d31tfrQ5bZTLtbrAwH7U7YPqL+nnlxp+/OW5zkf4vjd49GOH/z74efLZtpff/34guqDusj5PkXMUwgD0fhE1kavxaLdO7Y+fLoXhkEuG6AfeeSOUPjEqWvJ6biBnAr6t5Kc4wj1eU++BAO9Bpj/x5eDsgz8F415sNwndjbA6vbKfmTRAWAEuW+3SPEb44D6+Ld3a4rBX26e+rsW1rb37M/f6MC3vrf++nlWz1HO41dsdd9B9Xf3hyFFzJ1EP4NsePZw/6E3H30UfPfJoZAhB/Tvfrh8XCYb+V+ESD6LGPPmYzbBHDoz6wx6eZnZ3Y/YrhhmzrUf21+ul5nV7fIxs3b86n7UO0jAfnj5xTBzrjtT8HXTlyvH0U3uJL8OZexvfOfXx1pgCGXIAP0UiYO7Ue9/Iy32fmLG6TKsMhtI8KQzNvT02dDdn2W6h+he+pe0vQypbQcBbJFvvYDNtOU4yB1YB52tlM39ZWsOvR4Y/Avy10FjTn1Vzw5kuKEWirWyucI59ucpDqLkySsWQRAlfzFyj42FEbMtS1bAd2dhi3E/2f476b7kgz949KM+GAIZdEBv3QrWVVVr/gOB4rdIyxzPmECLObPk59FTffD+iT5471gPtHVlQd41OeMIEbbxl76/pgLBhAYLJjaGobHGYgDPcUc6DTZ8/W9/deRnENBuLkUGFdC/fWjdZlLhH5K/ueLBRDIUgt0Hu+C9o71w8MM+SKa97Iry1MT4y88fCQGMrw8RcFswpSnkrse1C/K9GKOvPPSrI9thkGRQAE0b6WMPrfkRYeI/02Muqncf6IEndp6Hrl5badFGjxZdGbdg4dQQTBkbVsEscYLQd7//y6MPwCDIJQP6Jz9ZGmnqrniUUPEW0J7qj57ph9/sOA9nzic9BylEP3i6YThksdYeshDTloUC1gve3viH3k8hQx/As+SPahpR8AfywtvXkpBk6ewINFQhJWMD4lb+6IfZY3ddajbkkgD9u++trs6G0TZSzHW8xfGn57PtGdj2ynnYf7wvb8uNhC3yhyAeCzHNTk5WnCQ78GQZPbxakA6VMOnzQhYlHwuSqSykSBhJddaGvOVMHBOCBYSxqxOWzM5wSD3b053c8vC2i89hXzSgH/m7tU3hNN5OajJffajefaAb/v/zZwkQ3QcA5hAHEyIngd6C4jF+++lP2ewkpDMiX+xd39jlYNsQj4Z41oNcV/IkyMDd3Zsh4A7ent6Br5kdhkljLG/WC+O3s3Z4898/evgsXIRcFKAf/+s19ekIeh0jmIWdFkZb2rZX2mDHW+05W3ZNVZikfCzoI+m4/mThlmx0eepIxCLXmRBXIkyucwa6+7KM4ILWnz3BImwddsHONEKHUlZ2+Y9/dbINBiihAa7PYuZ4JvoM+XmVuFEkCcv+4+MtjJ1BvYEA/1mZCEFddQTSaRvauzKk9fJwgou2vrHL3qZhIr3j9vZnCGuHoLYqzNg3lc4SrFqe9c93AbR12yzdp7DrmDC21kye1/6z/fsHltYb8Gi7pq7ET8ieV4ruXgrQHz5ymqXiBBNzzWOsxvoIhAkrn2tPkZZq860cv1hf3Cj07Y2/vP0UXl09GXLt04S1ETQ1xEio4d/+4zYMO/amoT8NcjnCqyeFpv9fGKAMiKEf+/6arxL1NXEIScK0/+c3p6GlLe1bl/Yc1ddEoKM7A719vOPEyOgUeu3pXTxDYsz6mihkSM8wDTdVSRIItbTbMHVsiHXIIN5Ilq5aUNe28732XcXuq2hAP/bg6k2k6fycRTjEpmmbf3y8FU62JAG0ULymMgwVJMy40Jkhlcc+v2zBujb+keynIKYJgLraKFhknZTbucb9qQxAew92OmIcJkdw46pFtS/t3NtxHIqQogD92A/WzUUYPUd+xtgCUo9tr15wYmb1QDBrgbQiFzpShpWN+IRiorcvAxXxMCRItiuZ9Kade/p52nZcHYcm741At62c3/DYzn1t5wuVX1QMjWz8zyT6rQTnKXTPwW6SzehwKyikrjrGBhq1d6VBtDovqJEP5MY/Gv2IYSRLKLu2OurzHzyVhVPnsjwUYEwNNRDCv4AipCCgSajxGZKSW8grh+HEx0n49bMyRSjuLnXVYbbz7l5vi0OeuxHWbOMfzf7OHg7a2uqwz//G4Qy0ddluAYSpr73/rum3QgHJC2g6cg4w+g57FsW8u/NnT7YyjZ2mR1VFPEQ6TCzW6kSLVP3B2viNn6TsOkkGhGCHYkj1Z0mP8a6DGeBpaSebYqG/gQKSF9ALq9feQzpPrmSFkn8793VCRw9/0UwE/rEogqqKEKlYytk5ePzS1rXxGz+XCwQ7FEO0p1H103j6RGsWlM65Of/18zPuhjySE9C0A4U84/1P0ZJSJDZ+5o12reXx3j/aymiL4sv1lolzaOM3fm7TXkSaRKCdb/wdSel/70QGsLe7/dt33JE7mZET0E2dCToUdJJ4EHz53U7o6s06hfKWVlVBuzYx6/lTk+U6U8sWZvzGH+ynb+bTt5SqK6MeP81PHz6dkYyO0IxpsRn3QQ7JCWhS5p/yHwSwJJf8zBttjskDd5qio3FPd0/KrZTq99rGb/yF/T29GUjEENDBfKr//Y+yBPBYZrWRg80ACQT07763egJRc0Whuw90sTdM5M5p50mEjaai4wfVSskWmG8Ai/Ebv385zT/39tvszq8up72MH50TwyYYJBc98MVZTUUDOo2sLZjvle1cH9dMxy7TAJ7eIoIrDXkOxviNP7e/uzcN8ZjFWFpdfuo8D3cpqtk7qunslqIBTXpSbhKxc4Y0mw9O9rqFU6E9PH1JZb4LpVLGNval2YiN/aEjNFX/x222O1UFRTW2rJsgQHyAfmTrPBKVw0ZR2AECZlsZt0wlEQuxca7CVluSsY19qTYd9BaNWh4/jaFbO2zXJhvcGJTt8DN0ov46AuUKQDx1sv94r6el0HCDmvwNE7ncaKMHS7OsGfkXduAqltPucErRPJZGlTMrZ60FTXyAJtmLxWoh+4/R17tkS4lGaLiRdW1nK2Mbe1Dt/lSGdNqFPP5T51mo4GY7LAyLQRM/Q2PUzLGMSF97BjrZ2AzZgui7Y8mk7drORsY29qDa/QRjNPGg+ulMWmlnqBDFvo1xM2jiAzRJjowXhdN3wZzNXX84DCwv7RVkbGMPqk3HSofDyOfvS4kHQzq0AxUGNFm3WbSULnfkHLfpmwR8ULYOaGMbe/Bt+nKIhbz+pOflqCIYmmDZYWhgDI2VfYUspNwW3PWNbewhsWl2zbK87E1nDHCXFMPQwBgaGFXzsRvKyshfAYSMbeyhsemL2JaGUMrQDJ5sxQIMvX3r+jABbA1P82EGaAFgatNJYrJZaXNtbGMPjU07UiwH4cLfnxL5aqbr77tvaQQU8QD6dDg1EZDcuKdPMjQvBLFWQ908GQ4ev7CN3/gHw0/Jk5Ko6k86z3DsrXDys7mvrTknoEmQHOJg5iur4GYM7U6cCB4t/Pry0e6fOj4KY2pDpGeVD+CqqQzBuPowNDdGzPkp0i8iEGGLLyhwhiYAj4Q8H74K/AqWl/5lt6RsWdI2/mD/1PEx+ModjcxmF4Zp/pveNh/48SnWnWvOX36/z3bIVnTCxDXsBnSsOAqLHhmsFApK4SIfaPxB/uXzEuwJnS6zLD4ugduIDbxZNCthzt/F+EFoAF/6GnKMtuOY5oUYPXBNhz4unVPJflvOY7pIPwlQr7mqxpyvi9Qg/scABQHNNsK8JfAmgh16N7pYveTKCohHLQ3Ekqnp37zpCXfqB3PeBqYdTIMIO/ICGpxMhoNucEHuthBjF7JXzK90gSsYmYMZnOXApn1YS1janK+B2TKGZh7Qxc/QIDZmbYItc1uGaBrgtY1f2g01YZg9OeEwsgQ1FRlL89O+ZnEtK8Gcv4H5BTMXydDgtgzRAnRb18Yv7TVXVSnhBfgY2lK+HdNUF4G5MyrM+RuI32FmCXKvBI3lUApBrs0F+bTxK15yklctrM6R3VBBLpevIyx9ueo3IvzYYWYRfmiiAbqf/c8L4S3Ay+r+FmT80rtgRoJ1ngQzshXI3Mvn10JF3BqU/Y8KPxIaBfBzjiwHbwCC7iFAoxzLR7d/9VXVAYycC9x8PTrmd83Vdeb8DcCPAsmWS3CWw90a3I28GudYPnr91RWks2RmhfMAGJx/Fsv1WPr6ZfXm/A7Aj8EJOwIkOA8dYHtja2Pr9qpF1SwVx0ErwUpFj6VF6o5+QIf+ntqcgOkTEiPqfAyVTX9IcBeKoZPgLczZCHuY2thB9volNS6IJWiRL2bWlwtwb7y2wZzPIm2m6X8a+VLJPbed2zL0mNrYuj17chwaSQouKIshbHay8+Sl1y2td6aIMOezoA0oZ9iRNw/NCxMtxOhceu3VNQHZDJ2h8+el6cSXKxfVm/NZhAYnxmadgLECgFbzz7wQ2UKM9mv6xdTlC2pAZ2SvXUxe2oJNKxrNeS2klUwFA3cyD6CTAQwNIEAtUinGVm0KZhoqBDMvytNTKMEMjn/B7BoYPyZuzm8e203ZOZ0sRTA0KwOwmzoRhRk7yL5uSY1nFB0KDC+QZjsPhA4zWyhEDaY3rWwy57cIG4rPQ4PSMoTGRgdo+irVrMkV7MQeONHH5iyRD4BqHhp5NQU+BTM5/UiEIYgz9aaV4835LaB51IGCoJs7Dy0053XRIpCxFXv9knp+Egkon9l1Ad54v8ux1ewG0rIbHLwUzIyp6T8r5IQdIWiojcGyBWPM+c1jCywXx9AI5QS1saVNO1GuW1rLQEkn5NlzoAde2tOu5Ju9MbMbXqhDSBEHM3LALJbfuHo8jLbzWbytoBiDT4LHcoAMN7ioLcXYVC+ZUwmViTAD6yvvdLCvor5/vBfaujLB2Q1AbjaDM7KlgDjkNgK6fNXV49jXxcz5DrZdZg6g6KLGQxvt1xuW1rugfe51+UGlF3a3g3wQFGC1FEZWB/lzP9UIOes5625eNQFK+fiHTSNgZCseELUkR+7x0OAG4qrX2NSm7wIuml3FgHfkoz44cy7p+p9/44InmwEK81pIMLFkZnCZW6zPAf3JdZOhaKZi9gAAEABJREFUXM/PUNqAQYmhCzC0lcEO7vlWMnUnxNjU3nBNPYuh6e8dhJFV/9m2FHxAQg9LhBVIHUIqwSxBLUEsNA1PJjdXw/xZdeZ857DlA6NXAublEMyMpWm0hyk2LGtgJ5Z+LPKlt9p96+148wIjBA5mJ2YG/wMg0sGsrX/zuinmvAdoBmbnWU+XAlkOTu+6dlYblf4FM6tgbH2UgfDVdzvY90D09V566wIBO9JiZiXMENkNJXUn8tAIyfU3LJ8IiXjInH+fX1mxEKA9YHZsb6E62EeXf8Oyene03HYSLwdtTz+nsGtvuwSxCDNEvlkNQ8Cf9RCNoCIRheuXTzLnX/OL7u8Ags41lkPVyNNCRrNNP2e3YmEdA9v5jgzsP9qTc/3tr58NYGTBxJarkeWNsS2HuQWD37Jhhjn/qo3Es13RDC20CMCxp6WMZvs6kqqjXwGjDP3UznN519/zQTt09tjsglgCzErnivsAqDIzkgOVeA+iBQtnN8KUCdXm/HtslIugc81tJ2kfABnt6E3LxzAw05vec7su5F2ffk7h2ddaPD2AOiMDSC3z0CHPelRvuX6mOf+CqR0Y5yDoPOOheRkA4G0po1XPmBiH6ZMqGMj2vN9F2DddcLtndn7shBteRnbBq+WhkTKUVGXuT66fCeEQMtdBaOBSMA8txDuWAykaRq29cXkjCCZ9dte5orY/eaYXjnzYAyKbQU+3yGKo2QxwUnW5xnrU1SRg7TWTR/X5V232vwNuXQLmtuObiEKUJe4a3qT2yPdHwiFYf80YBjL63Zk39rUXvf0zr57mzAsBMbOHkZW8NRLZEBl7f2rj7GE7/lL0q+BWJe94aG9hUgeDfeT6Vy6qhaqKKGPS518/zz5mU+z2z772MWRtyDHKznJTd748tNo9TpatXjoFGusTZXn+BtvvB72U3PNyOFovfDTam1aOc8H41KutA9q+qycFr+89B3KUnZKHRnKQvzcP7Wdw+rflhjmj/noA1kHtlSJnTlILGV32uDExWHRFLQPZ+0e74fTZvgGX99TLHzlvqCDP61ZBeWjkAbWMvam+bfNc4C/CjN7rIQf5+8EMkIuh3RhFEHX5P0hcrL1pxVgIOWB85rWzF1Xea++0Qk9vxpuyQ/r4aGXAkvrgqDxQThpfC0sXTBzV14P9jzhTB2E66FvfDqhpOyjDlM4gaoQwf3uEACuZAXhp99mLKieTzcLTr37kgFftXEHK4H/ZTQ7uw6HlawS33zi/ZM7P8GkBavBJ0fNDy5YyeuylcxvYO34UTC++eRb6U/ZFl/fkiyeUPLTytjd4ewZV5tYZnIL+hjWzoaoyNiqvh7AFMQ8wDy1jGFaI0kJGi7151Ti3J+/pnS2XVN7B421w4nQ3uHloK18eWnvDBWT3eSIWhVs2zBmV10NlaBFBFAS0/MYKdsANbqGjya6ujMDKq5sYuOgbKfuPdFxy+U+8cAzcPDSob3urD4b8z33DRQGzYPDbb1o46q6H1+aoLszQbLhdMJhHm960Yhzpbubg+sNLZwal3Ce2H6Uc4zCvMthf7Um0vPN0eEbrIX655sxohCtnNI7e68PPIl9Q3Nx2aiFYK3R02Devm8jARE/dsztPD0r57V398Pq7Zzx5aABtrIcYOqqAmS/nZWE7S37Y8JnN80bl9QFmI8nQxc1tp2rkaSGjwZ4zvQYmj69iC1975zwBYnrQyn9ix1EQnSigjt3QYmsV1HQ7ykoUzPQC27YNt1x/BUQjodF3fZwfAtSxgcxt5yzxtJTRYN+0ZiJnRhLDPkk7RQax/B2vH2fZEgD/PB0oYJ4OTNiYMrKdzbLfgqEr42G4fuX0UXl92FgO9QExH6Alqnkh4Elu63rk+SnrbVjezEB1oZN2W58f1PJTBMxPvXgkcHy0eACk6zIw27bLyIxiqA3cpuDesvGKkjt/l8PP/5cpPVUCY2gWcjOaZ0GL8LgtRNojz7/+2nGQiEfZCXv61dMMOIO9/23bDwJC6guyyjuHwMc92zS8AMnIzFY0/Vu2sBkmjK0c9PqVvN9h5uIYmpXBC5EtQrQUrNkjz3/z2ikuuB7f8eGQ7P/t/WfgdGu3ZGQku73BYWC6MXaY2LWd8EMwN9WcpUfP9UEu2UKgFOgpBLewYD2y/JPIg+DCKxoYc7574AJ8fK53yPb/L8+87+kB5Oecg1VlZB5mOLbNwa0y960bZipMNrKvD9cSzAXz0KEQRpKg9RZDf460k+P137x2kpt9oA+DQ7n/3z79PvvNsxgYVOa1XVDbEtTU9jA09zfWxWDFVc2DXr/S9XOR4UkcVPExtC3CDacwCWrQ7JHlp6395uumst+pFIbtu04N6f5bz3fB7vdOKYxse2LkIFD7bOD2pzbMGPLzM1x+lZmZHxwwY3bVQHzOW4ivp5B/Mku2CH+gPjLtVYvHQV11nIUbdGRcKp0d8v3/9qn3JHhBghZroFZtHfRUr17cTOoeG/L6Doeta9fthllehg6DJqyFIOTG0NxUUyhSi+UjQd+8bqo7EGnx3LHw9/99PYiuZrqcHrXl2Mj53ARSbSw/Q4HcHkDEtSgHWe61oGCMx8LAH/SwE054U3RU63bQ8jCJkjavngKP/PHwiLouOnPz/LODR8AQ9GToATTtKQwhUMAsWwxCI9eurY7A6iUTAJzeummTamA60ufNEIPukbueOwgfyUli+KB8DmZ3QJGyPU+5YYd5sRM+sOyyAlpnOZbZDaxpffmnr58G//zkoRF3ffgCrz/fzEkBDA0KmHnqhBfGWwZ/QdTL3Kq/HO1brpvG39sjoDvyYQd092ZAxNWBTKvZSGNuS2Nuqtn+BLPS/ds8v03ZddGV43xMjQMYGwcydJax1qSxFTB/ZgPsO3Kh7K+Haqu4RBzNznKJz7yAFmCW4QUooFZ3BiPGv2XjTAZo0rsMX/72dujqUcdugANadTsYND/VT/30bmhqqHBBrII3FyMH2Z9cNwn2H70wsq4PQj4/O5/Awa2LlRfMqPDFKHf/1XMaYcK4anYqXt5z2gPmy7F/qrc9f9B5IHS6tUF0b6tMzQcmyfXEQCWpN147HiJha0Rdn1x+sbwgoEHJcgiGUbXKPCPBf+v1M0CMpfj980eHpX6PPf0BiOwFKBor3d3ghhe2B8SA5Si8aATBxuXNI+r6BPn5kyEESu7x0O42WGspWGsh5euviIdh48qprHPjfHuSjVUejvqdbu2Cdz9ocZlXZWqs2lRnHabGWYWpbXe9T6waP2KuTyF/EEXnnDkJgWgZSGspI8fetHoqSZ1F2LJt248Af04bnvo8vuOwxry2Mv7ZsbHC1FhlaNu158+ohYlNFZe9/pfNRkiSrXAoEvhOIV3ZdleWqZSRZm/ZOAvAyVb8/vnDQ76/fPbTrxyDdCarMLKImTmI9ZjZHYWHJUMLffOa8XC56395bc7MBcdy8G5EJ3UCohDktpSRZE9proX5s5tYuLF7Xwucae0Z1vrQh9Edu056mDfowQ+ccANjbVQeSKa+4domUAcsjaTrx7MbWDk+r2iAjoPMcgAIcHPByk5A2Vl5+m+/cbY7dJOz8/DX7w8vHvM+CGJ/jOyxAfvGSVNdUxmGFQvqL3v9L5dfzUPr4mNoHjvLQiTdo0Bdjn46efin6Dhi4uzty8LzO4+XRP12vnMG2jr7Pak5PUYGrDwgesZ8yOX0M82bV4y77PW/nP5cPYW53ykMZGr1NlC+/vXLp7LZh2iP3pMvHnE+zTb89aO4fPKl46Cn8NR3Cf3LJYh5GZjpxVfWEKaOjMDrByDHevgl9/zQ7GkSa6kTf8spRz+dPFzMIbft+UMlVb/HXzipjYv2vrkil3vHTwtQU0BTm169TcsbR+D140owtHesXd55OfQWInSu5eXhH1OXgFVLprAG23qhD/YdOltS9Tt2qhMOnWgHnmf2MjF4mFpnZj7vnm1zUGfJ3ybycDjSrh+oqnAMrRYKjB28yezytz9z4xx3AvE/vnC4JOv75MsfKiC1te5wP5gZMzvrC01H7I2pDZO8dNWIun4Cwx7Gzgdo+aaKi2mQo5qQZpeXn+otN1zpzofx9MtHSrL+z+46A5lMBuQQUjkvhwCtl5lpD6IDZnU5AfX6JQ0j5vpJP/CwA4piaDXsgLwtRvf7dWn5ly2awCYNp6ei5Xwv7D98tiTr39aZhF17zwbGzlgBbVZlZM9y7MbSy+fXsjEe5XB9ivUzaAuGjhcAtGwZWGE20FqMznyQQ5eWn84Hxz/vEIIn2NwYpVv/P75yGsSgf5wrZrYdZrY5uOVyyeQhhGHtVXVlcX2K9jvMzGLo/jyATiYlmAHJWFoUpoJetBgd5KXqr66KwqbVM915l1/dc7Kk6//a3nPQ1Z1ymdYTM+vM7IDbz9g8hUcBXerXZyB+AAef9GdhhhbMzJoCE7VwaUuQC7/XLi3/rRuuhEg0AoD4tLV7D7SUdP3TGQzPv9HiZi8kA2PJzFhnZrFcrM/Ze9r4GIyrj5T09RmYn6/gmB4JmDmJg5murLYQUDZXW5LqD15eGv7bb1oA4iOWB4+dg/5kpuTr/9TOFsnMdo6YucjlG5bUl/T1KdbPsAwOqAPEO9FMGCMPzXNYO16/5mWWvn31vGa4csZYdyb8dw+0lEX9D33YDSfO9CgMDAoz4wBm1hgby/VXLqhiXf7leP08tsLMQZjO3VMIgqHFSXCLcW3J3KVt33XrVfJLrfR7g4dby6b+z75+VgOpZGC7UCytpPLiUQuunl1Z8sdb0HZQnIOgAwDtMrTKyDpTl489pj4BN6+fw86AmDejty9dMvUrZL+45ywHaQAD2zoz2wF5aRGDE3sVYelyu35Bthtjg1/y9hRiT2Hlqf/9v1kG4XDInQmf/pdMpcum/mfb03DwZNeAmVlldNEIrpgUY0NLy/l6ChiLWFqX3F+S5Zv5wo5yshsbquDuTy8Gz0z4dpY9EJbT8bx1oNMDXk8qLwczi+U81Sfz2NddVVXyx5vPduNhagXEHbnz0PyXEpgjx0aaXbr+//KldeydQX1Wz/GNlWVRf+E/eLI7cKxGtkhmzrrMjmHdoiqojKOyOn6/H5T1vBIwlsPRWLQQALWl+HVp+pctmgRbNs0HfT5lOjPR0gUTyur4Dn/U64+JbQlSrIA9q2Q9sG99m/UcblpaXVbH77EVMBc52g48hYgW4k1ul7ZdWx2DB//qFm3csJzN85oFzWV1PF29acnASje3h5m1MCTf8uVzEySWtsrz+oJjs/ADCgNahBu8MLWFQNnYf/fNLdBEshty3LDtmU+Z+m7bNKdsjqe6IuKPmbNOntlW8tIaM9s51rcIS392fU15Xl9mqeFIAUBLepctRNW5lpeCv6YqBj///mdh2cKJ4J1PWddZ+NqXVsLsqfVlcXxrFtUOCjOrD5QzxkfgznVVZXV9hZ9RcxCaIddYDg+oZUsRZfhHQQ2/f96scbDtH+4l4cQED3iDZwlk6y4AAA2ySURBVL7HEAkj+Ol3boZ7Pr0AYtFwSR4f7QzZsrYJPrtxbGBsnCtmzrncljE1XXbVzBjcuqKSzYdX6tdX93NQg0+CZx91iF00CBXcpWavWDwFPnfr1bBpzSy+XJ1/2Z2dU7f5u3rxmAX/8e7FcOcnroDtr52E3fs+ht37W6GzO3XZ6q/bDTURmD4hAQtnVMKqhbUQiyCXmRlAsZg/2vmN5YMhdsBqY9u7XPutygoST8+dEoOn3uyB/SdTkErj0rreyGs7UHZjaP2dwsAZ/OVTJDhabyEAsgXhgBZ1efzf+stNcOcnFzFDn0/ZVqamZbYdPCM+DT+a6uNwx42zyN9Mnqfuz7Dvcnf2pOAnj+yDne+0DOnxff3fzYJxDRHW6RG2OKGwebgxZ1p64Xi2BiTDCjtru+sLJgYX1HJ9ftzBUkseEO+8rpr93vFOLzyzp7ckrq/EH/jwBw5z92sNNIChRbjhDcjdFuPTw+evrY6DGkYUMxO+B/w55lmmb3iMbYgToEdhTG1syI9vWnMcKuKWZF4PeCW4BTsLZpaM7V+fgd4ZPqqzcj6pqwqVzPXlMPT7QQF7QYYGN9yQW7vjox2RATrS/Oiy+g8cbWVP7D29SRImEEbtTrILiJAAN93cdk6G7RyXzUqxHZv66eeQaUwdizo6YhENECVd5ufa+7T6DP7xvX+8mx0Hvd3TjxWlMrR7PsvGRFPbZgyLneNwtHN8/JnHOR7s7AcJRnaOV9s/PcYQuRNEQkSHuE3vDGHyX0tbpmSur8Sffv7ldoCLZmjVBrcwflJVe/j8P/7lLrh8MnTH99CvT0DpyfBf30C/w8ycjPx3ngLzcoiNjTa6dLTMcshGICTHN1bo/8gpwtlYaTHGNvZw2jJx4Zcco+1AY2pkbGOXhA1Yz8J5JXdPIdI3NtroUtEwQIYGvVMFGW10yWg3hAaUH9BsOLQSbjDTsVWaN7axh89GLi4ZqvMBmq3iRhtY2khtIcY29jDawJkZS1TnB7QAsUrvRhtdMpojVDJ03pmTcNoWzAyY0ztvIRzkwiVs3W9sYw+mTb+pnrWxx+8CELhN+jrtnIDOhBIXOO7p1sBGegnGpmXQd9NClrpT7PEb29iDadNZJ8S4FOFPxCyGUOWNlXZQxAPo7/54bxtZKcULFRvLFmM7rUS2GPD4jW3swbRDjKG9/uqEBcobK+kvbn07N6B5adDKGwqCeFQyNBXaWkIh5NpGGz2UWrCyury6IuT42IPhadDEn+WwcQvfGLMhjaCkRuhQRUvZgrcorNnGb/yD46dz8dEwV/VXEUArTP4xaOIf4I/Qx7xJIPYKECjJ62yGMLSCaMHkXhuM3/gHxU8nCFLwzPw1DkMzP0YtoEnQ3HYt4KTsElEeqzgOForTcbb6LJZGGz3YOhwCluGQwn9XV1iSyS0/Q/sATQLuFmdoPHsojEdDrodKX9Jm7+LJlmW00YOv47EQ9CeznuWUnSvjIeUBEhcGdBbsPWxV4EnsyU0iKuEgp/PCcZBz281bG9vYg2gnHECr/nnTEuyXG0NjtAc08QE6Fun9A7Zp6o6HHVPGRp2d8VLSGWCv7dCkNy8cefzGNval2qEQZviir6Cp/jlT4hLMAKlEV+Jp0MQH6K0Pn+4lG23HzoPhhMYIyL503oL6U9h5YBQtyus3trEvxY5Hw9DXn/H4KX9eOUUwNKKLnr7zBzv7QJPAsRw24N+LwuiLlM0NYRfUVPf0pkn6JMy/Ke20INVvbGNfrE2lknSe9PRnPf7Zk+PsxV7eQ4hpuPH7IOwGA9oK/Yu4C9BCJzVF3cKpzpBYPZm2oboy4mlhslLGNvbF2ZWJEND56DMZr38uCzcQ8GEZRKejjxUN6L/56YHTGMHbToOBmROiznRRYqAIhq6eDNs5mzIAqy0OtMqC8Rt/UX4aG1RXhqGjO+Xx0zFFS6+s4syMGEe/ce9fv34eigU0FYLTH7GdOQUunB5zdsJjmGyWxtKcpdXbh/DLSoLxG39R/tqqKPT00VDD679hKZ8SjduEQC2OzQEB+kjf4X8ihR5loCaVmD8tDnHlm9FUOrvTLL0SDfMdqy1Q2Lo2fuMPWh6NWBAjiQb6fKYur60MweqFVRz0wMB/4Bh++xeQQ3IC+tFHIUsayl+JwulT5tWzE04LE2shcntIQ31thAfsbgt0vG4LlLbxG7/up72C9TURQpAZsDHy+G+8tpaFu/zBEehYowe2boWcE/WFII+8+k7bvlWL6m8jZY2n4K2vDsHxljQkU9hdJ5Plv2vI7aLXSbUYMVKsUJCOqYuRUCND8JP1+JrqInDnhjFsHQf8b977rXe+mq88CwoIiWceAEb3mLH02oUVrEWpQuOedMYmrSzqVlKvtLGNHWQ31EYJKdoMQ6qfzrf3hc2NEsyY9V0/AAUkVGiFnXvbDq9cWH89+TmVBjEVMQuqSHbjZGvas15/0oaKBJ/sMJmywYiRQlJbxfs32jrSPt+9NzbBzIlxNywhP16+d+vb3yhUZkGGppLFoS+SFtJBf9Mx0dPGRchDYoz5RIxN5UJHkgGaxkPg66sX4rWNfzT6McNImDx4XWjv9/lvuKYW5k9PcCbnD5JtELLvgSKkKED/8FeHjmKEbieVy/LECcDimQmYMCYsWxDbN4Lz7Wnmp5OI01dokH6/Aa9t/KPLT8fTNzXEGUbOd9AhQ14Izp9WAZ9YVqs8UKIs+XnrPd949xgUIUUBmspDvzz6HElOf1U0OBrRrF1YCdUJvmPZEjG0d6ZJTJSGxvqoG2+rftXWtfGPXD+dSL6xPgK95AGwrSPl848jvs/TuJk+rGG3afzp5//H269AkYJggPKf7pr+MNn5l3glAHpJ7Lzj7V5o78n61qV5xbrqCPQls6xn0Xf3MTIqhJIt/dwGHePc3pUOfMZqHhOBL90ylsXVbBtgmP7RPVvf/nMYgBTN0EJ6qhu+THb3sgAnfUi8cVkVCz+oqC2OVrz1fD87IPp5B/o+GHZn1MdGj3iN2UCjsQ0xdkc/eyHpDNr3rjd3agL+4vbxHMw0ZmYeeCU+b/ZXYIAyYIam8uXbJo4JRyN0+vyZ7kMo+ffW4T7YdyLJKumNrTF7bYt2k0cJa9MeRpoVUf36+sYub7syEYYqwsrptA0dXSnWYRK0/sYltXDTCiVmBsbMR0Kx7LK7H9jbBgOUiwI0lT/73JT6uG09Tuq2Clmy7/3YmRTsfL+XjfXw9uFzTWPq2uoo61lMO98SoZ0ztOValuVb3+jS11RofKx+q4b2S9AwM5W2A7ejYfLnNzfBopkJN8/sgPpVK5a95WLAfEmAprJ1/fpwR/Pxh0mT+iIoLayzNwu7D/XDh62pnCeBzu9Bx4DQkxB2TkKGfQEVMU1FaAOa0tGUdCgYhQ6FOTjZB47oh46ynKTUAUa6XkBScresqofG2oiAjQDiP02Cuvs2bN1x0V3OlwRoIV/9t9P+gtT/IVLZkDNclcnZjiy8uq8HOnrsIk8WPzmhkMUqRkFPh6caMJWOpndeCtas8/1DTjrFbT+xKQa3ra2H6c10hkWHkanfskhgjf/zF7759v+GS5RBATSVv/zcjBssjH9DSqxhhSInGqIB0ekU7CGM3ZtUuzfl7crYI9uuqQzBLSvrYckVlRzE4IYXFICdJCi5/d6tbz0LgyCDBmgqX7lr+rgQwt+1ELqHEq5aeTrHQsuFDJxoScHJs3wgykhmstGu6fwZC6ZXsvBi1qQ4D1OoXzKzTfQvQpHM1z739fdaYJBkUAEt5P4vzFxAHmv/H6n1avZ+gbIXB+MkHKHgTsNxAvDOgBy2kfKTsXURWDCjgoF48tgYu9bupReMzNULAPaf37P13fdgkGVIAC3k/rtn3kbQ+z1yELPZzpD6NCtbNJ0gh779kkzziWySaf42DP2iKpY9RvI2hqTtroAdPy0XvOWzs4jB3S9o5SGk2c5asnxveeLbHur2dH0rx21XP17k1gc8Ix889fEcj378ym2b7Q/lPR61fNDOnzwejWlznF/6o4J0kNB5D2m/QkUMscFqdAIYmulwqxdw/OT8HLJt/LV7v/3Ob2GIZEgBTeWOOyA0LTbjPnJQ3yIH1aS2WHXnSKuJCwJAOfzuL6APjqCtp7jFWr5ykOp3we6L8XzbyfUDygnwB9fL+bCCdpz+cpT6I395cv/F1FeCt7jzH1Cedj61annrK4+vFSHrm/F5s/7hzjsfHdLb8ZADWsjWO+ZFe+KplQhnN5Kn2o3kWK8lx8q6F3O1aJXRMYCHuVRG8cZmktm0mE1eJCSIN2g/XgbNWy9n8veg/bj718rx70erl7aej9m18yD3o6/P3zLylGflOs6g/fjvJIH1CjpOQOQei18nu3vORtZzyWZr55/8ye40XAa5bIDWhQC8KpnoX0/OwnJyLqaQkzGJVGYiYDQREK4qxMxeRh0YM+cqJ5hp1QL9zF00MxdRjpfp8jA38jK9d/+g7T9POVq9kNKKizqfGHcRNJ8iN8hTZMkpcqc8Qfa0y+q1d9zz4Ls9MAwybIDOJ/d/YVFlHPqmoDAei7FSx7BHFZZwmE7wQJL/dIuMs2WmYAlh9Uee1XO6czjk4qD6ZJz6guc4C9dWKZ8fLtNFre+rWI56OWvS/zHK4jCOtEJ/5uRwgTaflCSgjRi5WBnwaDsjRkpZDKCNjCj5VwAAAP//mdPFfAAAAAZJREFUAwAvr0XgsROXYwAAAABJRU5ErkJggg==">
  <script>
    (function () {
      try {
        var saved = localStorage.getItem('aa-theme');
        document.documentElement.setAttribute('data-theme', saved === 'light' ? 'light' : 'dark');
      } catch (e) {
        document.documentElement.setAttribute('data-theme', 'dark');
      }
    })();
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap" rel="stylesheet">
  <style>
{{CSS}}
  </style>
</head>
<body data-page="{{PAGE}}">
  <div class="app-shell">
    <aside class="sidebar" id="sidebar" aria-label="Sidebar">
      <div class="sidebar-brand">
        <a href="/" class="brand" aria-label="Finance Agent home">
          <img class="brand-mark" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2IiByb2xlPSJpbWciIGFyaWEtbGFiZWw9IkZpbmFuY2UgQWdlbnQiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJ0aWxlIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzIwMjAxYiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxNDE0MTAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImxldHRlciIgeDE9IjAuMTUiIHkxPSIxIiB4Mj0iMC44NSIgeTI9IjAiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiM4ZTZkMjAiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIwLjMyIiBzdG9wLWNvbG9yPSIjYzlhODQ3Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMC41OCIgc3RvcC1jb2xvcj0iI2YwZDQ4OSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNzgiIHN0b3AtY29sb3I9IiNkM2FlNTIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjOWQ3OTI3Ii8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJlZGdlIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI2I2OTM0MCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNSIgc3RvcC1jb2xvcj0iIzZmNTcyMCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNhOTg1MzIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHg9IjgiIHk9IjgiIHdpZHRoPSIyNDAiIGhlaWdodD0iMjQwIiByeD0iNDgiIGZpbGw9InVybCgjdGlsZSkiIHN0cm9rZT0idXJsKCNlZGdlKSIgc3Ryb2tlLXdpZHRoPSI1Ii8+CiAgPHJlY3QgeD0iMTciIHk9IjE3IiB3aWR0aD0iMjIyIiBoZWlnaHQ9IjIyMiIgcng9IjQwIiBmaWxsPSJub25lIiBzdHJva2U9IiNjOGE4NGIiIHN0cm9rZS1vcGFjaXR5PSIwLjE0IiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY5LjY2IDE4NS45OCkgc2NhbGUoMC4xOCAtMC4xOCkiIGZpbGw9InVybCgjbGV0dGVyKSIgZD0iTTM1MyA3MTEgNTk2IDg0UTYxMSA0OCA2MjguNSAzNC41UTY0NiAyMSA2NjEgMjBWMFE2MzUgMiA1OTguNSAyLjVRNTYyIDMgNTI0IDNRNDgwIDMgNDQyLjAgMi41UTQwNCAyIDM4MCAwVjIwUTQzMSAyMiA0NDUuMCAzNy41UTQ1OSA1MyA0MzkgMTA0TDI2MyA1NzlMMjgxIDYwMUwxMjEgMTg2UTEwMiAxMzYgOTguMCAxMDQuMFE5NCA3MiAxMDMuMCA1NC4wUTExMiAzNiAxMzMuMCAyOC41UTE1NCAyMSAxODUgMjBWMFExNTMgMiAxMjMuMCAyLjVROTMgMyA2MyAzUTQxIDMgMjEuNSAyLjVRMiAyIC0xMyAwVjIwUTkgMjUgMzAuNSA0Ni4wUTUyIDY3IDcyIDExOUwzMDEgNzExUTMxMyA3MTAgMzI3LjAgNzEwLjBRMzQxIDcxMCAzNTMgNzExWk00NDAgMjg4VjI2OEgxNDJMMTUyIDI4OFoiLz4KPC9zdmc+" width="30" height="30" alt="" aria-hidden="true">
          <span class="brand-text">Finance <em>Agent</em></span>
        </a>
        <button class="icon-btn sidebar-close" id="sidebarClose" type="button" aria-label="Close navigation">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>

      <nav class="sidebar-nav" aria-label="Primary navigation">
        {{NAV}}
      </nav>

      <div class="sidebar-footer">
        <button class="assistant-cta" id="chatToggle" type="button" aria-haspopup="dialog" aria-expanded="false">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6z"/><path d="M18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z"/></svg>
          <span>Assistant</span>
          <span class="spark">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M9 18l6-6-6-6"/></svg>
          </span>
        </button>
        <button class="theme-toggle" type="button" data-theme-toggle>
          <span class="theme-icon" aria-hidden="true">
            <svg class="icon-sun" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
            <svg class="icon-moon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
          </span>
          <span data-theme-label>Light mode</span>
        </button>
      </div>
    </aside>
    <div class="sidebar-scrim" id="sidebarScrim"></div>

    <div class="app-main">
      <header class="mobile-bar">
        <button class="icon-btn" id="sidebarToggle" type="button" aria-label="Open navigation" aria-controls="sidebar" aria-expanded="false">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>
        </button>
        <span class="mobile-brand">Finance Agent</span>
        <button class="icon-btn" type="button" data-theme-toggle aria-label="Switch theme">
          <svg class="icon-sun" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
          <svg class="icon-moon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
        </button>
      </header>

      <main class="content" id="mainContent">
        {{CONTENT}}
      </main>
    </div>
  </div>

  <section class="chat-panel" id="chatPanel" role="dialog" aria-label="Finance assistant" aria-hidden="true">
    <div class="chat-header">
      <div class="chat-heading">
        <img class="chat-mark" src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2IiByb2xlPSJpbWciIGFyaWEtbGFiZWw9IkZpbmFuY2UgQWdlbnQiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJ0aWxlIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzIwMjAxYiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxNDE0MTAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImxldHRlciIgeDE9IjAuMTUiIHkxPSIxIiB4Mj0iMC44NSIgeTI9IjAiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiM4ZTZkMjAiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIwLjMyIiBzdG9wLWNvbG9yPSIjYzlhODQ3Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMC41OCIgc3RvcC1jb2xvcj0iI2YwZDQ4OSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNzgiIHN0b3AtY29sb3I9IiNkM2FlNTIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjOWQ3OTI3Ii8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJlZGdlIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iI2I2OTM0MCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjAuNSIgc3RvcC1jb2xvcj0iIzZmNTcyMCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNhOTg1MzIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHg9IjgiIHk9IjgiIHdpZHRoPSIyNDAiIGhlaWdodD0iMjQwIiByeD0iNDgiIGZpbGw9InVybCgjdGlsZSkiIHN0cm9rZT0idXJsKCNlZGdlKSIgc3Ryb2tlLXdpZHRoPSI1Ii8+CiAgPHJlY3QgeD0iMTciIHk9IjE3IiB3aWR0aD0iMjIyIiBoZWlnaHQ9IjIyMiIgcng9IjQwIiBmaWxsPSJub25lIiBzdHJva2U9IiNjOGE4NGIiIHN0cm9rZS1vcGFjaXR5PSIwLjE0IiBzdHJva2Utd2lkdGg9IjEuNSIvPgogIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY5LjY2IDE4NS45OCkgc2NhbGUoMC4xOCAtMC4xOCkiIGZpbGw9InVybCgjbGV0dGVyKSIgZD0iTTM1MyA3MTEgNTk2IDg0UTYxMSA0OCA2MjguNSAzNC41UTY0NiAyMSA2NjEgMjBWMFE2MzUgMiA1OTguNSAyLjVRNTYyIDMgNTI0IDNRNDgwIDMgNDQyLjAgMi41UTQwNCAyIDM4MCAwVjIwUTQzMSAyMiA0NDUuMCAzNy41UTQ1OSA1MyA0MzkgMTA0TDI2MyA1NzlMMjgxIDYwMUwxMjEgMTg2UTEwMiAxMzYgOTguMCAxMDQuMFE5NCA3MiAxMDMuMCA1NC4wUTExMiAzNiAxMzMuMCAyOC41UTE1NCAyMSAxODUgMjBWMFExNTMgMiAxMjMuMCAyLjVROTMgMyA2MyAzUTQxIDMgMjEuNSAyLjVRMiAyIC0xMyAwVjIwUTkgMjUgMzAuNSA0Ni4wUTUyIDY3IDcyIDExOUwzMDEgNzExUTMxMyA3MTAgMzI3LjAgNzEwLjBRMzQxIDcxMCAzNTMgNzExWk00NDAgMjg4VjI2OEgxNDJMMTUyIDI4OFoiLz4KPC9zdmc+" width="28" height="28" alt="" aria-hidden="true">
        <div>
          <div class="chat-title">Ledger assistant</div>
          <div class="chat-status"><span class="dot" aria-hidden="true"></span><span id="chatStatusText">Checking AI status...</span></div>
        </div>
      </div>
      <div class="chat-actions">
        <button class="icon-btn" id="chatClear" type="button" aria-label="Clear conversation" title="Clear conversation">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/></svg>
        </button>
        <button class="icon-btn" id="chatClose" type="button" aria-label="Close assistant">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
    </div>
    <div class="chat-messages" id="chatMessages" aria-live="polite"></div>
    <p class="chat-privacy">When you send a message, a summary of your ledger can be processed by Google Gemini. Calculations stay in this app.</p>
    <div class="chat-input-area">
      <input type="text" id="chatInput" placeholder="Ask about invoices, expenses, reports..." autocomplete="off" aria-label="Message the assistant">
      <button id="chatSend" type="button" aria-label="Send message">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4z"/></svg>
      </button>
    </div>
  </section>

  <div class="modal-overlay" id="modalOverlay" aria-hidden="true">
    <div class="modal-content" id="modalContent" role="dialog" aria-modal="true"></div>
  </div>

  <div class="confirm-overlay" id="confirmOverlay" aria-hidden="true">
    <div class="confirm-box" role="alertdialog" aria-modal="true" aria-labelledby="confirmTitle" aria-describedby="confirmMessage">
      <p class="confirm-kicker" id="confirmKicker">Please confirm</p>
      <h2 class="confirm-title" id="confirmTitle">Confirm action</h2>
      <p class="confirm-message" id="confirmMessage"></p>
      <div class="confirm-actions">
        <button class="btn" type="button" id="confirmCancel">Cancel</button>
        <button class="btn btn-danger" type="button" id="confirmAccept">Confirm</button>
      </div>
    </div>
  </div>

  <div class="toasts" id="toasts" aria-live="polite" aria-atomic="false"></div>

  <script>
{{JS}}
  </script>
</body>
</html>
"""

SHELL = SHELL.replace("{{CSS}}", STYLE_CSS).replace("{{JS}}", APP_JS)


PAGE_CONTENT = {
    'overview': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Financial overview</p>
    <h1 class="page-title">Overview</h1>
    <p class="page-sub">A live view of receivables, operating expenses, and the position of the ledger.</p>
  </div>
  <div class="page-actions">
    <a class="btn" href="/expenses?new=1">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
      Add expense
    </a>
    <a class="btn btn-primary" href="/invoices?new=1">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
      New invoice
    </a>
  </div>
</div>

<section class="metric-band" aria-label="Key financial metrics">
  <article class="metric">
    <span class="metric-label">Invoiced</span>
    <span class="metric-value" id="totalInvoiced">—</span>
    <span class="metric-note" id="invoicedNote">Reading ledger…</span>
  </article>
  <article class="metric">
    <span class="metric-label">Collected</span>
    <span class="metric-value" id="totalPaid">—</span>
    <span class="metric-note" id="paidNote">Reading ledger…</span>
  </article>
  <article class="metric">
    <span class="metric-label">Outstanding</span>
    <span class="metric-value" id="totalOutstanding">—</span>
    <span class="metric-note" id="outstandingNote">Reading ledger…</span>
  </article>
  <article class="metric">
    <span class="metric-label">Expenses</span>
    <span class="metric-value" id="totalExpenses">—</span>
    <span class="metric-note" id="expensesNote">Reading ledger…</span>
  </article>
  <article class="metric featured">
    <span class="metric-label">Net profit</span>
    <span class="metric-value" id="netProfit">—</span>
    <span class="metric-note" id="netProfitNote">Accrual · net revenue less expenses</span>
  </article>
</section>

<section class="panel ai-panel" id="ovAiPanel" aria-labelledby="ovAiTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="ovAiTitle">Financial summary</h2>
      <p class="panel-sub" id="ovAiMeta">Not generated yet</p>
    </div>
    <div class="panel-actions">
      <span class="ai-state" id="ovAiState" data-state="idle">Idle</span>
      <button class="btn" type="button" id="ovAiGenerate">Generate summary</button>
      <button class="btn hidden" type="button" id="ovAiRefresh">Refresh</button>
    </div>
  </div>
  <div class="panel-body">
    <div class="ov-spend-row" id="ovSpendFigures"></div>
    <div class="ai-summary" id="ovAiBody" aria-live="polite">
      <p class="muted">Generate a summary to see invoices, expenses and spending explained together.</p>
    </div>
    <p class="field-hint" id="ovAiNote"></p>
  </div>
</section>

<div class="dash-grid">
  <section class="panel" aria-labelledby="cashflowTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="cashflowTitle">Revenue &amp; expenses</h2>
        <p class="panel-sub">Accrual net revenue against recorded expenses · last six months</p>
      </div>
      <div class="chart-legend" aria-hidden="true">
        <span class="legend-item"><span class="legend-swatch"></span> Net revenue</span>
        <span class="legend-item"><span class="legend-swatch expense"></span> Expenses</span>
      </div>
    </div>
    <div class="chart-wrap" id="cashflowChart" aria-label="Six month revenue and expenses chart">
      <div class="chart-empty">Reading ledger…</div>
    </div>
  </section>

  <section class="panel" aria-labelledby="statusTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="statusTitle">Invoice status</h2>
        <p class="panel-sub">Receivables by payment stage</p>
      </div>
      <a class="text-link" href="/invoices">
        All invoices
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
      </a>
    </div>
    <div class="panel-body">
      <div class="status-meter" id="statusMeter" aria-hidden="true"></div>
      <ul class="status-legend" id="statusLegend"></ul>
      <div class="status-total" id="statusTotal"></div>
    </div>
  </section>
</div>

<div class="dash-grid lower">
  <section class="panel" aria-labelledby="recentTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="recentTitle">Recent invoices</h2>
        <p class="panel-sub">Newest entries in the ledger</p>
      </div>
      <a class="text-link" href="/invoices">
        View all
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
      </a>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th scope="col">Invoice</th>
            <th scope="col">Client</th>
            <th scope="col">Issued</th>
            <th scope="col">Status</th>
            <th scope="col" class="num">Amount</th>
          </tr>
        </thead>
        <tbody id="recentInvoicesBody">
          <tr><td colspan="5"><div class="empty-state compact"><p>Reading ledger…</p></div></td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="panel" aria-labelledby="attentionTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="attentionTitle">Needs attention</h2>
        <p class="panel-sub" id="attentionSub">Overdue invoices and open reimbursements</p>
      </div>
      <span class="mono muted" id="attentionCount" aria-label="Open items"></span>
    </div>
    <ul class="attention-list" id="attentionList">
      <li class="empty-state compact"><p>Reviewing open items…</p></li>
    </ul>
  </section>
</div>
""",
    'chat': r"""
<div class="chat-page">
  <aside class="chat-rail" aria-label="Conversations">
    <div class="chat-rail__head">
      <button class="btn btn-primary btn-sm chat-rail__new" type="button" id="chatNew">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
        New chat
      </button>
    </div>
    <ul class="chat-rail__list" id="chatList" role="list"></ul>
    <p class="chat-rail__foot">Conversations are saved in this browser only.</p>
  </aside>

  <section class="chat-main" aria-label="Conversation">
    <header class="chat-bar">
      <div class="chat-bar__title-wrap">
        <span class="chat-bar__title" id="chatTitle">New chat</span>
        <span class="ai-state" id="chatState" data-state="idle">Idle</span>
      </div>
      <div class="chat-bar__actions">
        <div class="model-picker" id="chatModelPicker">
          <button class="model-picker__button" type="button" id="chatModelButton" aria-haspopup="listbox" aria-expanded="false">
            <span class="model-picker__dot" data-provider="nvidia" aria-hidden="true"></span>
            <span class="model-picker__value" id="chatModelValue">Nemotron</span>
            <svg class="model-picker__caret" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
          </button>
          <ul class="model-picker__menu" id="chatModelMenu" role="listbox" aria-label="AI model" hidden></ul>
        </div>
        <button class="icon-btn" type="button" id="chatRename" aria-label="Rename conversation" title="Rename conversation">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>
        </button>
        <button class="icon-btn" type="button" id="chatDelete" aria-label="Delete conversation" title="Delete conversation">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/></svg>
        </button>
      </div>
    </header>

    <div class="chat-thread" id="chatThread" aria-live="polite"></div>

    <div class="chat-composer">
      <ul class="ov-chat__files" id="chatFiles" hidden></ul>
      <form class="chat-composer__form" id="chatForm">
        <input type="file" id="chatFileInput" class="sr-only" multiple
          accept=".pdf,.csv,.xlsx,.xls,.docx,.doc,.txt,.png,.jpg,.jpeg,.webp">
        <button class="chat-composer__attach" type="button" id="chatAttach" aria-label="Attach a file" title="Attach a file">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21.4 11.6l-8.5 8.5a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.7 1.7 0 0 1-2.4-2.4l7.8-7.8"/></svg>
        </button>
        <textarea class="chat-composer__input" id="chatPageInput" rows="1" placeholder="Message the AI..." aria-label="Message the AI"></textarea>
        <button class="chat-composer__send" type="submit" id="chatPageSend" aria-label="Send" title="Send">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>
        </button>
      </form>
      <p class="field-hint chat-composer__note" id="chatNote">Messages and attached files can be sent to the selected AI provider. Attaching a file reads it and files it into your ledger automatically.</p>
    </div>
  </section>
</div>
""",
    'invoices': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Receivables</p>
    <h1 class="page-title">Invoices</h1>
    <p class="page-sub">Create client invoices, follow their payment status, and reconcile the money owed to you.</p>
  </div>
  <div class="page-actions">
    <button class="btn btn-primary" type="button" onclick="openInvoiceForm()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
      New invoice
    </button>
  </div>
</div>

<div class="ledger-note" id="invoiceLedgerNote">
  <span>Reading invoice ledger…</span>
</div>

<section class="panel" aria-labelledby="invoiceListTitle">
  <h2 class="sr-only" id="invoiceListTitle">Invoice ledger</h2>
  <div class="toolbar">
    <label class="search-field">
      <span class="sr-only">Search invoices by number, client, email or status</span>
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
      <input class="input" type="search" id="invoiceSearch" placeholder="Search number, client, email, status…" autocomplete="off">
    </label>
    <div class="pills" id="invoiceStatusPills" role="group" aria-label="Filter invoices by status">
      <button class="pill active" type="button" data-status="">All <span class="pill-count" data-count="all">0</span></button>
      <button class="pill" type="button" data-status="draft">Draft <span class="pill-count" data-count="draft">0</span></button>
      <button class="pill" type="button" data-status="sent">Sent <span class="pill-count" data-count="sent">0</span></button>
      <button class="pill" type="button" data-status="paid">Paid <span class="pill-count" data-count="paid">0</span></button>
      <button class="pill" type="button" data-status="overdue">Overdue <span class="pill-count" data-count="overdue">0</span></button>
    </div>
    <span class="spacer"></span>
    <button class="text-button" type="button" id="invoiceSelectAllMatching" hidden>Select all matching</button>
    <span class="mono muted" id="invoiceRowCount" aria-live="polite"></span>
    <label class="select-all-mobile">
      <input class="check" type="checkbox" id="invoiceSelectAllMobile">
      <span>Select all</span>
    </label>
  </div>
  <div class="bulk-bar" id="invoiceBulkBar" hidden>
    <div class="bulk-info">
      <span class="bulk-count" id="invoiceSelectedCount">0 selected</span>
      <button class="text-button" type="button" onclick="clearInvoiceSelection()">Clear selection</button>
    </div>
    <div class="bulk-actions">
      <button class="btn btn-sm" type="button" onclick="exportSelectedInvoices()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M4 21h16"/></svg>
        Export CSV
      </button>
      <label class="bulk-field">
        <span class="sr-only">Set status for selected invoices</span>
        <select class="select select-sm" id="invoiceBulkStatus">
          <option value="">Set status...</option>
          <option value="draft">Draft</option>
          <option value="sent">Sent</option>
          <option value="paid">Paid</option>
          <option value="overdue">Overdue</option>
          <option value="cancelled">Cancelled</option>
        </select>
      </label>
      <button class="btn btn-sm" type="button" onclick="applyInvoiceBulkStatus()">Apply</button>
      <button class="btn btn-sm btn-danger" type="button" onclick="deleteSelectedInvoices()">Delete selected</button>
    </div>
  </div>
  <div class="table-wrap">
    <table class="data-table invoices-table">
      <thead>
        <tr>
          <th scope="col" class="select-col">
            <input class="check" type="checkbox" id="invoiceSelectAll" aria-label="Select all invoices in view">
          </th>
          <th scope="col">Invoice</th>
          <th scope="col">Client</th>
          <th scope="col">Issued</th>
          <th scope="col">Due</th>
          <th scope="col">Status</th>
          <th scope="col" class="num">Amount</th>
          <th scope="col" class="actions-col"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody id="invoicesBody">
        <tr><td colspan="8"><div class="empty-state"><p>Reading invoice ledger…</p></div></td></tr>
      </tbody>
    </table>
  </div>
  <div class="pagination" id="invoicePagination" aria-label="Invoice pages">
    <label class="pagination-size">Rows
      <select class="select select-sm" id="invoicePageSize">
        <option value="10">10</option>
        <option value="25" selected>25</option>
        <option value="50">50</option>
        <option value="100">100</option>
      </select>
    </label>
    <span class="pagination-count" id="invoicePageCount">0 records</span>
    <div class="pagination-actions">
      <button class="btn btn-sm" type="button" id="invoicePrev" disabled>Previous</button>
      <span class="pagination-page" id="invoicePageLabel">Page 1</span>
      <button class="btn btn-sm" type="button" id="invoiceNext" disabled>Next</button>
    </div>
  </div>
</section>
""",
    'expenses': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Operating costs</p>
    <h1 class="page-title">Expenses</h1>
    <p class="page-sub">Record what the business spends, keep it categorised, and track reimbursements owed back to you.</p>
  </div>
  <div class="page-actions">
    <button class="btn btn-primary" type="button" onclick="openExpenseForm()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
      Add expense
    </button>
  </div>
</div>

<div class="ledger-note" id="expenseLedgerNote">
  <span>Reading expense ledger…</span>
</div>

<section class="panel" aria-labelledby="expenseListTitle">
  <h2 class="sr-only" id="expenseListTitle">Expense ledger</h2>
  <div class="toolbar">
    <label class="search-field">
      <span class="sr-only">Search expenses by vendor, description, category or amount</span>
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
      <input class="input" type="search" id="expenseSearch" placeholder="Search vendor, description, category, amount…" autocomplete="off">
    </label>
    <div class="pills" id="expenseCategoryPills" role="group" aria-label="Filter expenses by category">
      <button class="pill active" type="button" data-category="">All</button>
    </div>
    <span class="spacer"></span>
    <button class="text-button" type="button" id="expenseSelectAllMatching" hidden>Select all matching</button>
    <span class="mono muted" id="expenseRowCount" aria-live="polite"></span>
    <label class="select-all-mobile">
      <input class="check" type="checkbox" id="expenseSelectAllMobile">
      <span>Select all</span>
    </label>
  </div>
  <div class="bulk-bar" id="expenseBulkBar" hidden>
    <div class="bulk-info">
      <span class="bulk-count" id="expenseSelectedCount">0 selected</span>
      <button class="text-button" type="button" onclick="clearExpenseSelection()">Clear selection</button>
    </div>
    <div class="bulk-actions">
      <button class="btn btn-sm" type="button" onclick="exportSelectedExpenses()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M4 21h16"/></svg>
        Export CSV
      </button>
      <label class="bulk-field">
        <span class="sr-only">Set reimbursement status for selected expenses</span>
        <select class="select select-sm" id="expenseBulkReimbursement">
          <option value="">Reimbursement...</option>
          <option value="true">Mark reimbursed</option>
          <option value="false">Mark pending</option>
        </select>
      </label>
      <button class="btn btn-sm" type="button" onclick="applyExpenseBulkReimbursement()">Apply</button>
      <button class="btn btn-sm btn-danger" type="button" onclick="deleteSelectedExpenses()">Delete selected</button>
    </div>
  </div>
  <div class="table-wrap">
    <table class="data-table expenses-table">
      <thead>
        <tr>
          <th scope="col" class="select-col">
            <input class="check" type="checkbox" id="expenseSelectAll" aria-label="Select all expenses in view">
          </th>
          <th scope="col">Date</th>
          <th scope="col">Expense</th>
          <th scope="col">Category</th>
          <th scope="col">Reimbursement</th>
          <th scope="col" class="num">Amount</th>
          <th scope="col" class="actions-col"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody id="expensesBody">
        <tr><td colspan="7"><div class="empty-state"><p>Reading expense ledger…</p></div></td></tr>
      </tbody>
    </table>
  </div>
  <div class="pagination" id="expensePagination" aria-label="Expense pages">
    <label class="pagination-size">Rows
      <select class="select select-sm" id="expensePageSize">
        <option value="10">10</option>
        <option value="25" selected>25</option>
        <option value="50">50</option>
        <option value="100">100</option>
      </select>
    </label>
    <span class="pagination-count" id="expensePageCount">0 records</span>
    <div class="pagination-actions">
      <button class="btn btn-sm" type="button" id="expensePrev" disabled>Previous</button>
      <span class="pagination-page" id="expensePageLabel">Page 1</span>
      <button class="btn btn-sm" type="button" id="expenseNext" disabled>Next</button>
    </div>
  </div>
</section>
""",
    'reports': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Financial statements</p>
    <h1 class="page-title">Reports</h1>
    <p class="page-sub">Accrual-basis profitability, a partial financial position, and tax invoiced versus tax collected.</p>
  </div>
  <div class="page-actions">
    <button class="btn" type="button" onclick="window.print()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9V2h12v7"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
      Print statement
    </button>
  </div>
</div>

<div class="report-tabs" role="tablist" aria-label="Report type">
  <button class="report-tab active" type="button" role="tab" aria-selected="true" data-report="pnl" onclick="switchReport('pnl')">Profit &amp; loss</button>
  <button class="report-tab" type="button" role="tab" aria-selected="false" data-report="balance" onclick="switchReport('balance')">Balance sheet</button>
  <button class="report-tab" type="button" role="tab" aria-selected="false" data-report="tax" onclick="switchReport('tax')">Tax summary</button>
</div>

<div class="report-toolbar" id="reportToolbar">
  <div class="field">
    <label class="field-label" for="reportFrom">From</label>
    <input class="input" type="date" id="reportFrom">
  </div>
  <div class="field">
    <label class="field-label" for="reportTo">To</label>
    <input class="input" type="date" id="reportTo">
  </div>
  <div class="range-presets" role="group" aria-label="Quick date ranges">
    <button class="range-preset" type="button" data-range="month">This month</button>
    <button class="range-preset" type="button" data-range="last-month">Last month</button>
    <button class="range-preset" type="button" data-range="quarter">This quarter</button>
    <button class="range-preset" type="button" data-range="ytd">Year to date</button>
  </div>
  <span class="spacer"></span>
  <span class="mono muted hidden" id="reportAsOf"></span>
  <button class="btn btn-primary" type="button" onclick="loadReport()">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/></svg>
    <span id="reportGenerateLabel">Generate</span>
  </button>
</div>

<section class="report-sheet" id="reportResults" aria-live="polite">
  <div class="report-empty">Reading the ledger…</div>
</section>
""",
    'activity': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Audit trail</p>
    <h1 class="page-title">Activity</h1>
    <p class="page-sub">A lightweight history of accounting changes. API keys and secrets are never recorded.</p>
  </div>
</div>

<section class="panel" aria-labelledby="activityTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="activityTitle">Recent activity</h2>
      <p class="panel-sub">Newest first · last 100 events</p>
    </div>
  </div>
  <div class="table-wrap">
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">When</th>
          <th scope="col">Action</th>
          <th scope="col">Record</th>
          <th scope="col">Details</th>
        </tr>
      </thead>
      <tbody id="activityBody">
        <tr><td colspan="4"><div class="empty-state compact"><p>Reading activity...</p></div></td></tr>
      </tbody>
    </table>
  </div>
</section>
""",
    'spendwise': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Spending analytics</p>
    <h1 class="page-title">SpendWise</h1>
    <p class="page-sub">Spending analysis, budgets, and cash movement — calculated from the same expense records as the ledger.</p>
  </div>
  <div class="page-actions">
    <div class="field">
      <label class="field-label" for="spendwisePeriod">Period</label>
      <select class="select" id="spendwisePeriod">
        <option value="7d">Last 7 days</option>
        <option value="30d">Last 30 days</option>
        <option value="month" selected>Current month</option>
        <option value="last_month">Previous month</option>
        <option value="3m">Last 3 months</option>
        <option value="6m">Last 6 months</option>
        <option value="year">Current year</option>
        <option value="custom">Custom range</option>
      </select>
    </div>
    <button class="btn" type="button" id="spendwiseRefresh">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/></svg>
      Refresh
    </button>
  </div>
</div>

<div class="sw-range hidden" id="spendwiseCustomRange">
  <div class="field">
    <label class="field-label" for="spendwiseFrom">From</label>
    <input class="input" type="date" id="spendwiseFrom">
  </div>
  <div class="field">
    <label class="field-label" for="spendwiseTo">To</label>
    <input class="input" type="date" id="spendwiseTo">
  </div>
  <button class="btn" type="button" id="spendwiseApplyRange">Apply range</button>
  <span class="sw-range-note" id="spendwisePeriodNote"></span>
</div>

<section class="metric-band" aria-label="Spending metrics">
  <article class="metric">
    <span class="metric-label">Total spending</span>
    <span class="metric-value" id="swTotal">—</span>
    <span class="metric-note" id="swTotalNote">Reading ledger…</span>
  </article>
  <article class="metric">
    <span class="metric-label">This month</span>
    <span class="metric-value" id="swMonth">—</span>
    <span class="metric-note" id="swMonthNote"></span>
  </article>
  <article class="metric">
    <span class="metric-label">Average month</span>
    <span class="metric-value" id="swAverage">—</span>
    <span class="metric-note" id="swAverageNote"></span>
  </article>
  <article class="metric">
    <span class="metric-label">Largest category</span>
    <span class="metric-value" id="swTopCategory">—</span>
    <span class="metric-note" id="swTopCategoryNote"></span>
  </article>
  <article class="metric featured">
    <span class="metric-label">Budget used</span>
    <span class="metric-value" id="swBudget">—</span>
    <span class="metric-note" id="swBudgetNote"></span>
  </article>
</section>

<section class="panel ai-panel" id="swAiPanel" aria-labelledby="swAiTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swAiTitle">AI summary</h2>
      <p class="panel-sub" id="swAiMeta">Not generated for this period yet</p>
    </div>
    <div class="panel-actions">
      <span class="ai-state" id="swAiState" data-state="idle">Idle</span>
      <button class="btn" type="button" id="swAiGenerate">Generate summary</button>
      <button class="btn hidden" type="button" id="swAiRefresh">Refresh</button>
    </div>
  </div>
  <div class="panel-body">
    <div class="ai-summary" id="swAiBody" aria-live="polite">
      <p class="muted">Generate a summary to see this period's spending explained in words.</p>
    </div>
    <p class="field-hint" id="swAiNote"></p>
  </div>
</section>

<div class="dash-grid">
  <section class="panel" aria-labelledby="swTrendTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="swTrendTitle">Spending trend</h2>
        <p class="panel-sub" id="swTrendNote">Recorded spending over the selected period</p>
      </div>
      <div class="sw-toggle" role="group" aria-label="Trend grouping">
        <button class="pill active" type="button" data-trend="monthly">Monthly</button>
        <button class="pill" type="button" data-trend="weekly">Weekly</button>
      </div>
    </div>
    <div class="chart-wrap" id="spendwiseTrend" aria-label="Spending trend chart">
      <div class="chart-empty">Reading ledger…</div>
    </div>
  </section>

  <section class="panel" aria-labelledby="swCashTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="swCashTitle">Cash movement</h2>
        <p class="panel-sub">Paid invoices in, recorded expenses out</p>
      </div>
    </div>
    <div class="panel-body">
      <div class="sw-cash-row">
        <div class="sw-cash-cell">
          <span class="sw-cash-label">Incoming</span>
          <span class="sw-cash-value" id="swCashIn">—</span>
        </div>
        <div class="sw-cash-cell">
          <span class="sw-cash-label">Outgoing</span>
          <span class="sw-cash-value" id="swCashOut">—</span>
        </div>
        <div class="sw-cash-cell">
          <span class="sw-cash-label">Net movement</span>
          <span class="sw-cash-value" id="swCashNet">—</span>
        </div>
      </div>
      <div class="chart-wrap chart-wrap-compact" id="spendwiseCashChart" aria-label="Cash movement chart"></div>
      <p class="field-hint" id="swCashNote"></p>
    </div>
  </section>
</div>

<div class="dash-grid stack">
  <section class="panel" aria-labelledby="swCategoryTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="swCategoryTitle">Categories</h2>
        <p class="panel-sub">Select a category to inspect its transactions</p>
      </div>
    </div>
    <div class="table-wrap">
      <table class="data-table sw-table">
        <thead>
          <tr>
            <th scope="col">Category</th>
            <th scope="col">Spend</th>
            <th scope="col">Share</th>
            <th scope="col">Count</th>
            <th scope="col">vs previous</th>
          </tr>
        </thead>
        <tbody id="spendwiseCategories">
          <tr><td colspan="5"><div class="empty-state compact"><p>Reading ledger…</p></div></td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="panel" aria-labelledby="swVendorTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="swVendorTitle">Vendors</h2>
        <p class="panel-sub">Where the money goes</p>
      </div>
      <div class="field sw-vendor-filter">
        <label class="field-label" for="spendwiseVendorFilter">Filter</label>
        <select class="select select-sm" id="spendwiseVendorFilter">
          <option value="">All vendors</option>
        </select>
      </div>
    </div>
    <div class="table-wrap">
      <table class="data-table sw-table">
        <thead>
          <tr>
            <th scope="col">Vendor</th>
            <th scope="col">Spend</th>
            <th scope="col">Share</th>
            <th scope="col">Count</th>
            <th scope="col">vs previous</th>
          </tr>
        </thead>
        <tbody id="spendwiseVendors">
          <tr><td colspan="5"><div class="empty-state compact"><p>Reading ledger…</p></div></td></tr>
        </tbody>
      </table>
    </div>
  </section>
</div>

<section class="panel" aria-labelledby="swMatrixTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swMatrixTitle">Spending over time</h2>
      <p class="panel-sub" id="swMatrixNote">Recorded spending per month, by category</p>
    </div>
    <div class="sw-toggle" role="group" aria-label="Group spending over time">
      <button class="pill active" type="button" data-matrix="category">Category</button>
      <button class="pill" type="button" data-matrix="vendor">Vendor</button>
    </div>
  </div>
  <div class="table-wrap">
    <table class="data-table sw-table" id="spendwiseMatrixTable">
      <thead>
        <tr>
          <th scope="col" id="swMatrixHead">Category</th>
          <th scope="col" class="num">Total</th>
        </tr>
      </thead>
      <tbody id="spendwiseMatrix">
        <tr><td colspan="2"><div class="empty-state compact"><p>Reading ledger…</p></div></td></tr>
      </tbody>
    </table>
  </div>
</section>

<section class="panel" aria-labelledby="swBudgetTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swBudgetTitle">Budgets</h2>
      <p class="panel-sub">Monthly limits, per category or across all spending</p>
    </div>
  </div>
  <div class="panel-body sw-budget-layout">
    <form class="sw-budget-form" id="spendwiseBudgetForm">
      <div class="form-error" id="spendwiseBudgetError" role="alert"></div>
      <input type="hidden" id="spendwiseBudgetId" value="">
      <div class="form-grid">
        <div class="field">
          <label class="field-label" for="spendwiseBudgetName">Budget name</label>
          <input class="input" id="spendwiseBudgetName" maxlength="120" placeholder="e.g. Software" required>
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseBudgetAmount">Amount</label>
          <input class="input" id="spendwiseBudgetAmount" type="number" min="0.01" step="0.01" placeholder="0.00" required>
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseBudgetCategory">Category</label>
          <select class="select" id="spendwiseBudgetCategory">
            <option value="">All spending</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseBudgetMonth">Month</label>
          <input class="input" id="spendwiseBudgetMonth" type="month" placeholder="Every month">
          <span class="field-hint">Leave empty to apply to every month.</span>
        </div>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" type="submit" id="spendwiseBudgetSubmit">Add budget</button>
        <button class="btn hidden" type="button" id="spendwiseBudgetCancel">Cancel edit</button>
      </div>
    </form>
    <div class="sw-budget-column">
      <div class="sw-budget-list" id="spendwiseBudgets">
        <div class="empty-state compact"><p>Reading budgets…</p></div>
      </div>
      <p class="field-hint" id="spendwiseBudgetNote"></p>
    </div>
  </div>
</section>

<section class="panel" aria-labelledby="swRuleTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swRuleTitle">Categorisation rules</h2>
      <p class="panel-sub">Suggest a category from a vendor or a keyword. Manually set categories are never overwritten.</p>
    </div>
  </div>
  <div class="panel-body sw-rule-layout">
    <form class="sw-rule-form" id="spendwiseRuleForm">
      <div class="form-error" id="spendwiseRuleError" role="alert"></div>
      <input type="hidden" id="spendwiseRuleId" value="">
      <div class="form-grid">
        <div class="field">
          <label class="field-label" for="spendwiseRuleName">Rule name</label>
          <input class="input" id="spendwiseRuleName" maxlength="120" placeholder="e.g. Cloud hosting" required>
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseRuleCategory">Category</label>
          <select class="select" id="spendwiseRuleCategory"></select>
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseRuleVendor">Vendor contains</label>
          <input class="input" id="spendwiseRuleVendor" maxlength="200" placeholder="e.g. Hetzner">
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseRuleKeyword">Description contains</label>
          <input class="input" id="spendwiseRuleKeyword" maxlength="200" placeholder="e.g. hosting">
          <span class="field-hint">A transaction must match every field you fill in.</span>
        </div>
        <div class="field">
          <label class="field-label" for="spendwiseRulePriority">Priority</label>
          <input class="input" id="spendwiseRulePriority" type="number" min="0" max="1000" step="1" value="100">
          <span class="field-hint">Lower runs first when several rules match.</span>
        </div>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" type="submit" id="spendwiseRuleSubmit">Add rule</button>
        <button class="btn hidden" type="button" id="spendwiseRuleCancel">Cancel edit</button>
      </div>
    </form>
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th scope="col">Rule</th>
            <th scope="col">Matches</th>
            <th scope="col">Category</th>
            <th scope="col">Priority</th>
            <th scope="col">State</th>
            <th scope="col" class="actions-col"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody id="spendwiseRules">
          <tr><td colspan="6"><div class="empty-state compact"><p>Reading rules…</p></div></td></tr>
        </tbody>
      </table>
    </div>
    <div class="sw-suggestions" id="spendwiseSuggestions"></div>
  </div>
</section>

<section class="panel" aria-labelledby="swInsightTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swInsightTitle">Insights</h2>
      <p class="panel-sub">Calculated observations from the records in the selected period</p>
    </div>
    <div class="panel-actions">
      <span class="ai-state" id="swAnalysisState" data-state="idle">Idle</span>
      <button class="btn" type="button" id="spendwiseAnalyse">Analyse spending</button>
    </div>
  </div>
  <div class="panel-body sw-insight-layout">
    <div class="sw-insight-main">
      <ul class="sw-insights" id="spendwiseInsights">
        <li class="muted">Reading ledger&hellip;</li>
      </ul>
      <h3 class="statement-title">Potential savings</h3>
      <ul class="sw-list" id="spendwiseSavings"></ul>
      <div class="sw-ai-block" id="spendwiseAiBlock" hidden>
        <h3 class="statement-title">AI suggestions <span class="ai-tag">separate from the figures</span></h3>
        <ul class="sw-list" id="spendwiseAiSuggestions"></ul>
        <p class="field-hint" id="spendwiseAiNote"></p>
      </div>
    </div>
    <div class="sw-insight-side">
      <h3 class="statement-title">Possible recurring charges</h3>
      <ul class="sw-list" id="spendwiseRecurring"></ul>
      <h3 class="statement-title">Unusually large expenses</h3>
      <ul class="sw-list" id="spendwiseUnusual"></ul>
    </div>
  </div>
</section>

<section class="panel" aria-labelledby="swWhatIfTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swWhatIfTitle">What-if Simulator</h2>
      <p class="panel-sub">Change a few monthly figures and compare the outcome. A simulation never touches your ledger.</p>
    </div>
    <div class="panel-actions">
      <button class="btn" type="button" id="swSimReset">Reset</button>
    </div>
  </div>
  <div class="panel-body sw-sim-body">
    <div class="sw-sim-controls">
      <div class="sw-sim-baseline" id="swSimBaseline">
        <p class="muted">Reading the last six months...</p>
      </div>

      <div class="field">
        <label class="field-label" for="swSimSubsNum">Monthly subscriptions</label>
        <div class="range-row">
          <input class="range" type="range" id="swSimSubs" min="0" max="2000" step="5" value="0" aria-label="Monthly subscriptions slider">
          <input class="input input-num" type="number" id="swSimSubsNum" min="0" step="5" value="0">
        </div>
        <span class="field-hint" id="swSimSubsHint"></span>
      </div>

      <div class="field">
        <label class="field-label" for="swSimIncomeNum">Monthly income change</label>
        <div class="range-row">
          <input class="range" type="range" id="swSimIncome" min="-3000" max="3000" step="50" value="0" aria-label="Monthly income change slider">
          <input class="input input-num" type="number" id="swSimIncomeNum" step="50" value="0">
        </div>
        <span class="field-hint">A positive number is extra income each month.</span>
      </div>

      <div class="field">
        <label class="field-label" for="swSimExpensesNum">Other monthly expenses change</label>
        <div class="range-row">
          <input class="range" type="range" id="swSimExpenses" min="-2000" max="2000" step="25" value="0" aria-label="Monthly expenses change slider">
          <input class="input input-num" type="number" id="swSimExpensesNum" step="25" value="0">
        </div>
      </div>

      <div class="field">
        <label class="field-label" for="swSimSavingsNum">Monthly savings change</label>
        <div class="range-row">
          <input class="range" type="range" id="swSimSavings" min="-2000" max="2000" step="25" value="0" aria-label="Monthly savings change slider">
          <input class="input input-num" type="number" id="swSimSavingsNum" step="25" value="0">
        </div>
        <span class="field-hint">Savings you set aside are removed from free cash.</span>
      </div>

      <div class="field">
        <label class="field-label" for="swSimOneTime">One-time purchase</label>
        <input class="input" type="number" id="swSimOneTime" min="0" step="10" value="0" placeholder="0">
        <span class="field-hint">A single payment this month, such as a laptop.</span>
      </div>

      <div class="field">
        <label class="field-label" for="swSimText">Or describe it in words</label>
        <div class="ai-input-row">
          <input class="input" id="swSimText" placeholder="What if my income decreases by 15%?" maxlength="1000">
          <button class="btn" type="button" id="swSimInterpret">Interpret</button>
        </div>
        <span class="field-hint" id="swSimAiNote"></span>
      </div>
    </div>

    <div class="sw-sim-results">
      <div class="sw-sim-cards" id="swSimCards"></div>
      <div class="sw-sim-explain" id="swSimExplain"></div>
      <p class="field-hint">Simulated figures are estimates for planning. They are not financial advice and are never saved.</p>
    </div>
  </div>
</section>

<section class="panel" aria-labelledby="swSubsTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swSubsTitle">Subscription Optimiser</h2>
      <p class="panel-sub">Recurring charges detected from the ledger. Confirm what you use each one for before comparing alternatives.</p>
    </div>
    <div class="panel-actions">
      <button class="btn" type="button" id="swSubsRefresh">Refresh</button>
    </div>
  </div>
  <div class="panel-body">
    <div class="sw-subs-summary" id="swSubsSummary"></div>
    <ul class="sw-subs-list" id="swSubsList">
      <li class="muted">Reading recurring charges...</li>
    </ul>
    <div class="sw-subs-detail" id="swSubsDetail" hidden></div>
  </div>
</section>

<section class="panel" aria-labelledby="swDocsTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swDocsTitle">Document Analysis</h2>
      <p class="panel-sub">Recently uploaded documents and what the AI read from them, before you confirm an import.</p>
    </div>
    <a class="text-link" href="/documents">
      Open Documents
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
    </a>
  </div>
  <div class="panel-body">
    <ul class="sw-docs-list" id="swDocsList">
      <li class="muted">Reading documents...</li>
    </ul>
  </div>
</section>

<section class="panel" aria-labelledby="swImportTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="swImportTitle">Import spending records</h2>
      <p class="panel-sub">CSV import writes ordinary expenses — they appear on the Expenses page and feed these analytics.</p>
    </div>
  </div>
  <div class="panel-body sw-import-layout">
    <div class="import-block">
      <p class="field-hint">Columns: description, amount, date_incurred, vendor, category, reimbursable, reimbursed, notes. A missing category is suggested from your rules.</p>
      <div class="upload-zone upload-zone--compact" id="spendwiseCsvDropzone" tabindex="0" role="button" aria-label="Choose a CSV file of expenses">
        <span class="upload-zone__icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v13"/></svg>
        </span>
        <span class="upload-title">Choose a CSV file</span>
        <span class="field-hint">Expenses · any size</span>
        <input class="upload-input" type="file" id="spendwiseCsvFile" accept=".csv,text/csv">
      </div>
      <div class="upload-list" id="spendwiseCsvFileList"></div>
      <div class="settings-file">
        <button class="btn" type="button" id="spendwiseCsvPreview">Preview</button>
      </div>
      <div class="import-preview" id="spendwiseCsvPreviewHost" hidden></div>
    </div>
  </div>
</section>
""",
    'documents': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Document intake</p>
    <h1 class="page-title">Documents</h1>
    <p class="page-sub">Upload invoices, receipts, bank statements, and other business documents. Finance Agent extracts the details so you can review them before anything is saved.</p>
  </div>
  <div class="page-actions">
    <a class="btn" href="/spendwise">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 4 4-7 5 5"/><path d="M3 21h18"/></svg>
      SpendWise
    </a>
  </div>
</div>

<section class="panel" aria-labelledby="docUploadTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="docUploadTitle">Upload documents</h2>
      <p class="panel-sub">Uploaded content is read by NVIDIA NIM (Nemotron), or Google Gemini when NIM is not configured. Invoices, receipts, and expenses are filed automatically; a blank field is listed as &ldquo;Unstated&rdquo;, and other documents stay in the library for review.</p>
    </div>
  </div>
  <div class="panel-body">
    <div class="upload-zone" id="docDropzone" tabindex="0" role="button" aria-label="Choose files or drop them here">
      <span class="upload-zone__icon" aria-hidden="true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v13"/></svg>
      </span>
      <span class="upload-title">Drop files here or choose files</span>
      <span class="field-hint">PDF, CSV, XLSX, DOCX, TXT, PNG, JPG, or WEBP &middot; no size limit</span>
      <input class="upload-input" type="file" id="docFileInput" multiple accept=".pdf,.csv,.xlsx,.xls,.docx,.doc,.txt,.png,.jpg,.jpeg,.webp">
    </div>
    <div class="upload-list" id="docFileList"></div>
    <div class="upload-actions">
      <button class="btn btn-primary" type="button" id="docUploadButton" disabled>Upload and extract</button>
      <button class="btn hidden" type="button" id="docClearFiles">Clear files</button>
      <span class="upload-progress" id="docUploadProgress" hidden></span>
    </div>
    <p class="field-hint" id="docUploadNote"></p>
  </div>
</section>

<section class="panel" aria-labelledby="docResultsTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="docResultsTitle">Extraction results</h2>
      <p class="panel-sub">Filed documents link straight to their ledger record; correct or delete them from there.</p>
    </div>
  </div>
  <div class="panel-body">
    <div id="docResults">
      <div class="empty-state compact"><strong>No documents processed yet</strong><p>Upload a file and its extracted details appear here.</p></div>
    </div>
  </div>
</section>

<section class="panel" aria-labelledby="docLibraryTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="docLibraryTitle">Document library</h2>
      <p class="panel-sub">Previously processed files, their status, and the record they produced</p>
    </div>
  </div>
  <div class="table-wrap">
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">File</th>
          <th scope="col">Type</th>
          <th scope="col">Uploaded</th>
          <th scope="col">Status</th>
          <th scope="col">Record</th>
          <th scope="col" class="actions-col"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody id="docLibrary">
        <tr><td colspan="6"><div class="empty-state compact"><p>Reading documents&hellip;</p></div></td></tr>
      </tbody>
    </table>
  </div>
</section>
""",
    'settings': r"""
<div class="page-head">
  <div>
    <p class="eyebrow">Workspace controls</p>
    <h1 class="page-title">Settings &amp; data</h1>
    <p class="page-sub">Currency, tax defaults, backups, and CSV imports. Financial records stay on this device; AI requests are described separately.</p>
  </div>
</div>

<div class="settings-grid">
  <section class="panel" aria-labelledby="localeTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="localeTitle">Currency &amp; formatting</h2>
        <p class="panel-sub">Used consistently in tables, reports, PDFs, exports, and the assistant snapshot.</p>
      </div>
    </div>
    <form class="panel-body settings-form" id="settingsForm">
      <div class="form-error" id="settingsError" role="alert"></div>
      <div class="form-grid">
        <div class="field">
          <label class="field-label" for="settingCurrencyCode">Currency code</label>
          <input class="input" id="settingCurrencyCode" name="currency_code" maxlength="3" placeholder="USD" required>
          <span class="field-hint">Three-letter ISO code, e.g. USD, EUR, GBP.</span>
        </div>
        <div class="field">
          <label class="field-label" for="settingCurrencySymbol">Currency symbol</label>
          <input class="input" id="settingCurrencySymbol" name="currency_symbol" maxlength="4" placeholder="$" required>
        </div>
        <div class="field">
          <label class="field-label" for="settingLocale">Locale</label>
          <input class="input" id="settingLocale" name="locale" placeholder="en-US" required>
          <span class="field-hint">Controls number and date formatting.</span>
        </div>
        <div class="field">
          <label class="field-label" for="settingTaxRate">Default tax rate</label>
          <input class="input" id="settingTaxRate" name="default_tax_rate" type="number" min="0" max="100" step="0.01" required>
          <span class="field-hint">Percent applied to new invoices. Existing invoices keep their stored rate.</span>
        </div>
        <div class="field">
          <label class="field-label" for="settingDateFormat">Date format</label>
          <select class="select" id="settingDateFormat" name="date_format">
            <option value="short">Short (14/11/26)</option>
            <option value="medium">Medium (14 Nov 2026)</option>
            <option value="long">Long (14 November 2026)</option>
            <option value="iso">ISO (2026-11-14)</option>
          </select>
        </div>
        <div class="field">
          <label class="field-label" for="settingBusinessName">Business name</label>
          <input class="input" id="settingBusinessName" name="business_name" maxlength="200" placeholder="Shown on PDF invoices">
        </div>
        <div class="field span-2">
          <label class="field-label" for="settingBusinessAddress">Business address</label>
          <textarea class="input" id="settingBusinessAddress" name="business_address" rows="3" maxlength="500" placeholder="Street, city, postal code, country"></textarea>
        </div>
        <div class="field">
          <label class="field-label" for="settingBusinessEmail">Business email</label>
          <input class="input" id="settingBusinessEmail" name="business_email" type="email" maxlength="254" placeholder="billing@yourbusiness.com">
        </div>
        <div class="field span-2">
          <label class="field-label" for="settingPaymentDetails">Payment details</label>
          <textarea class="input" id="settingPaymentDetails" name="payment_details" rows="3" maxlength="500" placeholder="Bank name, account reference, or payment instructions"></textarea>
          <span class="field-hint">Printed on PDF invoices. Never put API keys or passwords here.</span>
        </div>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" type="submit" id="settingsSave">Save settings</button>
      </div>
    </form>
  </section>

  <section class="panel" aria-labelledby="backupTitle">
    <div class="panel-head">
      <div>
        <h2 class="panel-title" id="backupTitle">Backups &amp; recovery</h2>
        <p class="panel-sub">Backups include records and public settings, never API keys.</p>
      </div>
    </div>
    <div class="panel-body settings-stack">
      <div class="settings-row">
        <div>
          <strong>Download a full backup</strong>
          <p class="muted">A timestamped JSON file. Automatic backups are also created before destructive operations.</p>
        </div>
        <a class="btn" href="/api/backup" download>Download backup</a>
      </div>
      <div class="settings-row">
        <div>
          <strong>Restore from a backup</strong>
          <p class="muted">The current records are backed up first. Invalid or incompatible files are rejected.</p>
        </div>
        <div class="settings-file">
          <div class="upload-zone upload-zone--compact" id="restoreDropzone" tabindex="0" role="button" aria-label="Choose a backup file">
            <span class="upload-zone__icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v13"/></svg>
            </span>
            <span class="upload-title">Choose a backup file</span>
            <span class="field-hint">JSON</span>
            <input class="upload-input" type="file" id="restoreFile" accept=".json,application/json">
          </div>
          <div class="upload-list" id="restoreFileList"></div>
          <button class="btn btn-danger" type="button" id="restoreButton">Restore</button>
        </div>
      </div>
      <div class="form-error" id="restoreError" role="alert"></div>
    </div>
  </section>
</div>

<section class="panel" aria-labelledby="importTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="importTitle">CSV import</h2>
      <p class="panel-sub">Preview every row before saving. Invalid rows block the import instead of being discarded silently.</p>
    </div>
  </div>
  <div class="panel-body import-grid">
    <div class="import-block">
      <h3 class="statement-title">Invoices</h3>
      <p class="field-hint">Columns: client_name, client_email, invoice_number, issue_date, due_date, status, description, quantity, unit_price, tax_rate, notes. Tax rate is a percentage (20 = 20%). Rows with the same invoice_number are grouped.</p>
      <div class="upload-zone upload-zone--compact" id="invoiceCsvDropzone" tabindex="0" role="button" aria-label="Choose a CSV file of invoices">
        <span class="upload-zone__icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v13"/></svg>
        </span>
        <span class="upload-title">Choose a CSV file</span>
        <span class="field-hint">Invoices · any size</span>
        <input class="upload-input" type="file" id="invoiceCsvFile" accept=".csv,text/csv">
      </div>
      <div class="upload-list" id="invoiceCsvFileList"></div>
      <div class="settings-file">
        <button class="btn" type="button" id="invoiceCsvPreview">Preview</button>
      </div>
      <div class="import-preview" id="invoiceCsvPreviewHost" hidden></div>
    </div>
    <div class="import-block">
      <h3 class="statement-title">Expenses</h3>
      <p class="field-hint">Columns: description, amount, date_incurred, vendor, category, reimbursable, reimbursed, notes. Categories must match the configured list.</p>
      <div class="upload-zone upload-zone--compact" id="expenseCsvDropzone" tabindex="0" role="button" aria-label="Choose a CSV file of expenses">
        <span class="upload-zone__icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v13"/></svg>
        </span>
        <span class="upload-title">Choose a CSV file</span>
        <span class="field-hint">Expenses · any size</span>
        <input class="upload-input" type="file" id="expenseCsvFile" accept=".csv,text/csv">
      </div>
      <div class="upload-list" id="expenseCsvFileList"></div>
      <div class="settings-file">
        <button class="btn" type="button" id="expenseCsvPreview">Preview</button>
      </div>
      <div class="import-preview" id="expenseCsvPreviewHost" hidden></div>
    </div>
  </div>
</section>

<section class="panel" aria-labelledby="privacyTitle">
  <div class="panel-head">
    <div>
      <h2 class="panel-title" id="privacyTitle">AI privacy</h2>
      <p class="panel-sub">The assistant is optional and read-only.</p>
    </div>
  </div>
  <div class="panel-body">
    <div class="privacy-local">
      <span class="privacy-local-dot" aria-hidden="true"></span>
      <div>
        <strong>Your ledger is stored locally on this device</strong>
        <p class="settings-copy">Records live in plain JSON files in this app's <code>data</code> folder. They are not uploaded to a cloud account. Backups are files you download yourself, and they never include API keys.</p>
      </div>
    </div>
    <p class="settings-copy">When you use the assistant, a compact summary of invoices and expenses is sent to Google Gemini for processing. Financial calculations are performed by this application, not by the model. The assistant cannot edit, delete, or recalculate records. If Gemini is unavailable, the panel says so and offers local navigation guidance only.</p>
    <p class="settings-copy">Document reading, the What-if Simulator's natural-language input, subscription suggestions, and spending suggestions can be processed by NVIDIA NIM (Nemotron). Only the small section of data each task needs is sent, and every figure you see is calculated by this application, not by the model.</p>
    <p class="settings-copy">The API keys are read from <code>.env</code> on the server and are never sent to the browser or included in backups.</p>
  </div>
</section>
""",
}

NAV_ITEMS = [
    ('/', 'Overview', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>"""),
    ('/chat', 'Chat', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 8.9 8.9 0 0 1-3.2-.6L3 21l1.7-5.1A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z"/></svg>"""),
    ('/invoices', 'Invoices', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h5"/></svg>"""),
    ('/expenses', 'Expenses', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3v18l3-2 2 2 2-2 2 2 2-2 3 2V3z"/><path d="M9 8h6M9 12h6M9 16h3"/></svg>"""),
    ('/spendwise', 'SpendWise', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 4 4-7 5 5"/><path d="M3 21h18"/></svg>"""),
    ('/documents', 'Documents', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h4"/></svg>"""),
    ('/reports', 'Reports', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 16v-4M12 16V8M17 16v-6"/></svg>"""),
    ('/activity', 'Activity', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-6 3 12 2.5-6H21"/></svg>"""),
    ('/settings', 'Settings', r"""<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>"""),
]


def _nav_item(href: str, label: str, icon: str, active: bool) -> str:
    classes = "nav-item active" if active else "nav-item"
    current = ' aria-current="page"' if active else ""
    return (
        f'<a href="{href}" class="{classes}"{current}>\n'
        f'          <span class="nav-icon">{icon}</span>\n'
        f'          <span class="nav-label">{label}</span>\n'
        f'        </a>'
    )



# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

def render_page(page: str, path: str, title: str) -> HTMLResponse:
    nav = "\n        ".join(
        _nav_item(href, label, icon, path == href) for href, label, icon in NAV_ITEMS
    )
    html = (
        SHELL
        .replace("{{TITLE}}", title)
        .replace("{{PAGE}}", page)
        .replace("{{NAV}}", nav)
        .replace("{{CONTENT}}", PAGE_CONTENT[page])
    )
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
def overview_page(request: Request):
    return render_page("overview", request.url.path, "Overview")


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return render_page("chat", request.url.path, "Chat")


@app.get("/invoices", response_class=HTMLResponse)
def invoices_page(request: Request):
    return render_page("invoices", request.url.path, "Invoices")


@app.get("/expenses", response_class=HTMLResponse)
def expenses_page(request: Request):
    return render_page("expenses", request.url.path, "Expenses")


@app.get("/spendwise", response_class=HTMLResponse)
def spendwise_page(request: Request):
    return render_page("spendwise", request.url.path, "SpendWise")


@app.get("/documents", response_class=HTMLResponse)
def documents_page(request: Request):
    return render_page("documents", request.url.path, "Documents")


@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request):
    return render_page("reports", request.url.path, "Reports")


@app.get("/activity", response_class=HTMLResponse)
def activity_page(request: Request):
    return render_page("activity", request.url.path, "Activity")


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return render_page("settings", request.url.path, "Settings & data")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=int(os.getenv("PORT", str(DEFAULT_PORT))),
    )
