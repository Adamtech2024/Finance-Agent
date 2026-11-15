# Finance Agent

A local-first finance workspace for freelancers and small businesses: invoicing,
expenses, reports, document intake, and **SpendWise** spending analytics over the
same records.

Everything runs on one machine. There is no database server, no cloud account,
and no telemetry. The optional assistant and document reader call AI providers
from the server only.

## Structure

```text
app.py            The whole application — backend, API, page markup, CSS and JS
.env, .env.example  Optional AI provider keys and runtime settings
requirements.txt  Python dependencies
data/             Local ledger, settings, and backups (never committed)
```

`app.py` is self-contained. The design system and all frontend behaviour are
inlined as `STYLE_CSS` and `APP_JS`, so there is no `static/` directory and no
asset requests beyond the page itself. Branding is inlined as data URIs.

## Features

- **Overview** — invoiced value, cash collected, outstanding receivables,
  expenses, accrual net profit, and a six-month chart.
- **Chat** — a dedicated page under Overview with saved conversations, a
  ChatGPT-style thread, and a model picker. Attach a file to read and file it,
  then ask about it or the ledger. Answers come from Google Gemini or NVIDIA NIM
  (Nemotron).
- **Invoices** — create, edit, view, mark paid, and delete invoices with line
  items, tax rates, due dates, and status tracking.
- **Expenses** — record costs by category and vendor, with reimbursement
  tracking.
- **Search and pagination** — backend search, configurable page size, and clear
  *select visible* vs *select all matching* behaviour.
- **Bulk actions** — delete, change status, update reimbursements, or export the
  selection to CSV. Destructive actions confirm and back up first.
- **Reports** — profit and loss, balance sheet, and tax summary for a chosen
  period on a documented accrual basis.
- **PDF invoices** — paginated PDFs generated from stored records.
- **CSV import** — validate and preview invoice or expense rows, flag
  duplicates, show row-level errors, and confirm before saving.
- **Document intake** — uploads are read and invoices, receipts, and expenses are
  filed automatically. Statements and other files stay in the library for review,
  and likely duplicates are skipped. There is no file-size or batch limit.
- **SpendWise** — period statistics, category and vendor breakdowns, trends,
  budgets, categorisation rules, insights, and a cash-movement view.
- **What-if Simulator** — adjust income, expenses, subscriptions, savings, and a
  one-time purchase; compare current and simulated outcomes. Never writes to the
  ledger.
- **Subscription Optimiser** — detect recurring charges and compare AI-suggested
  alternatives with calculated savings. Prices without a source are flagged.
- **Backups** — download or restore a full JSON backup, with automatic backups
  before destructive operations.
- **Settings** — currency, locale, default tax, date format, business details,
  and payment instructions.
- **Activity history** — a lightweight audit trail. Secrets are never recorded.
- **Themes** — dark and light modes, saved in the browser.

## Reporting basis

The app uses an **accrual basis**:

- Revenue is recognised on the invoice issue date for `sent`, `overdue`, and
  `paid` invoices. Draft and cancelled invoices are excluded.
- Revenue and net profit exclude sales tax. Gross invoiced value, including tax,
  is reported separately for receivables.
- Expenses are recognised on `date_incurred`.
- Dashboard net profit is net issued revenue minus all recorded expenses, not
  cash collected.
- The balance sheet is a **partial management position**, not a statutory
  balance sheet. Vendor payables, bank balances, opening equity, and statutory
  reporting are not modelled.
- The SpendWise cash-movement view is a cash presentation and must not be read as
  accrual profit.

All money is calculated with Python `Decimal`, rounded to cents with
`ROUND_HALF_UP`, and returned as fixed-precision decimal strings.

This is not tax advice and claims no compliance with any jurisdiction.

## Requirements

- Python 3.10+
- FastAPI, Pydantic 2, Uvicorn
- `pypdf` and PyMuPDF for PDF text, tables, and page rendering
- `playwright` for browser-based checks
- Optional: a Google Gemini key for the assistant and summaries, and an NVIDIA
  NIM key for document reading and suggestions

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example file and edit it if you want the AI features:

```bash
cp .env.example .env
```

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | No | Server-side Gemini key for summaries and the assistant. |
| `GEMINI_MODEL` | No | Gemini model; defaults to `gemini-3.1-flash-lite`. |
| `NVIDIA_API_KEY` | No | Server-side NIM key for documents, what-if, and suggestions. |
| `NVIDIA_MODEL` | No | NIM model; defaults to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. |
| `NVIDIA_BASE_URL` | No | OpenAI-compatible NIM endpoint. |
| `NVIDIA_ENABLE_THINKING` | No | Set `true` to include thinking tokens. |
| `AI_TIMEOUT_SECONDS` | No | Per-request AI timeout; defaults to `60`. |
| `AI_MAX_RETRIES` | No | Retries for transient AI failures; defaults to `2`. |
| `PORT` | No | Port for `python app.py`; defaults to `2611`. |
| `HOST` | No | Bind address for `python app.py`; defaults to `0.0.0.0` (every interface). Set `127.0.0.1` to keep it local-only. |
| `FINANCE_DATA_DIR` | No | Alternate ledger directory; defaults to `./data`. |
| `ACCOUNTING_DATA_DIR` | No | Legacy name, still supported. |

Real environment variables take precedence over `.env`. The `.env` file is
git-ignored; never commit or share it. Keys are read only by the backend and are
excluded from backups and activity logs.

Currency, locale, tax defaults, and business details live on the **Settings**
page and are stored in `data/settings.json`. A new install starts with **EUR**,
**€**, and **en-IE**.

## Running

```bash
python app.py
```

Open <http://127.0.0.1:2611>. `python app.py` listens on every network
interface, so other devices on the same network can open
`http://<your-computer-ip>:2611`. There is no login, so only do this on a
network you trust; set `HOST=127.0.0.1` to keep it local-only. `python app.py`
does not auto-reload. For development:

```bash
uvicorn app:app --reload --port 2611 --host 0.0.0.0
```

The `data/` directory is created automatically.

## Data, backups, and recovery

- Ledger: `data/invoices.json`, `data/expenses.json`, `data/activity.json`
- SpendWise: `data/budgets.json`, `data/rules.json`
- Settings: `data/settings.json`
- Backups: `data/backups/*.json`

Writes are atomic: JSON is written to a temporary file, flushed, and atomically
replaced, so an interrupted save cannot truncate a ledger. Corrupt JSON is
reported, never silently overwritten.

A backup contains invoices, expenses, public settings, activity, budgets, and
rules — never API keys. Restore validates format, version, records, statuses,
categories, dates, amounts, duplicate IDs, budgets, and rules before replacing
anything, and backs up the current data first.

## AI setup and privacy

Two providers are configured independently and never share a key.

**Google Gemini** — financial summaries and the read-only assistant. Create a
key in Google AI Studio, set `GEMINI_API_KEY`, optionally set `GEMINI_MODEL`.

**NVIDIA NIM (Nemotron)** — document reading (including scans and images),
what-if interpretation, subscription alternatives, and spending suggestions.
Create a key at <https://build.nvidia.com>, set `NVIDIA_API_KEY`, optionally set
`NVIDIA_MODEL` and `NVIDIA_BASE_URL`.

Restart after changing `.env`.

When the assistant is used, a compact ledger summary can be sent to Gemini. When
a document is read, extracted text (and page images for scans) is sent to NIM,
falling back to Gemini. The what-if box sends baseline figures and your
sentence. Suggestion features send only the single item and its calculated
facts.

Financial calculations stay in deterministic Python. The model returns
structured parameters or prose; Python produces every number shown. The
assistant is read-only. Every AI feature degrades gracefully — with no key, the
calculated features keep working and the panel says the AI part is unavailable.

Live answer quality has not been verified against a live provider response in
this repository; treat it as prepared, not proven, until you test with your own
key.

## CSV import format

Invoices accept (common aliases recognised):

```text
client_name, client_email, invoice_number, issue_date, due_date, status,
description, quantity, unit_price, tax_rate, notes
```

Rows sharing an `invoice_number` are grouped into one invoice. `tax_rate` may be
a percentage (`20`) or a fraction (`0.2`). Required: `client_name`,
`description`, `unit_price`.

Expenses accept:

```text
description, amount, date_incurred, vendor, category, reimbursable, reimbursed, notes
```

Required: `description`, `amount`. Categories must match the configured list
(`office`, `travel`, `utilities`, `software`, `hardware`, `food`, `transport`,
`rent`, `services`, `other`). Validation preview is mandatory in the UI;
invalid rows block the import instead of being silently dropped.

## API overview

List endpoints return `{items, total, page, page_size, pages}`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health and version |
| `GET` / `POST` | `/api/invoices` | List or create invoices |
| `GET` / `PUT` / `DELETE` | `/api/invoices/{id}` | Read, update, or delete an invoice |
| `POST` | `/api/invoices/{id}/mark-paid` | Mark an invoice as paid |
| `GET` | `/api/invoices/{id}/pdf` | Download a PDF invoice |
| `POST` | `/api/invoices/bulk` | Delete or update invoice statuses |
| `GET` / `POST` | `/api/expenses` | List or create expenses |
| `GET` / `PUT` / `DELETE` | `/api/expenses/{id}` | Read, update, or delete an expense |
| `POST` | `/api/expenses/bulk` | Delete or update reimbursements |
| `GET` | `/api/dashboard/summary` | Accrual dashboard totals |
| `GET` | `/api/reports/profit-loss`, `/balance-sheet`, `/tax-summary` | Reports |
| `GET` | `/api/spendwise/overview` | Statistics, trends, budgets, cash, insights |
| `GET` | `/api/spendwise/categories`, `/vendors`, `/insights`, `/cash-flow` | Breakdowns |
| `GET` / `POST` / `PUT` / `DELETE` | `/api/spendwise/budgets`, `/rules` | Manage budgets and rules |
| `POST` | `/api/spendwise/whatif/simulate`, `/interpret` | What-if simulator |
| `GET` / `POST` | `/api/spendwise/analysis`, `/subscriptions` | Analysis and optimiser |
| `GET` / `PUT` | `/api/settings` | Read or update public settings |
| `GET` / `POST` | `/api/backup`, `/api/backup/restore` | Download or restore a backup |
| `GET` | `/api/activity` | Recent activity |
| `POST` | `/api/import/invoices/preview`, `/invoices` | Validate or import invoice CSV |
| `POST` | `/api/import/expenses/preview`, `/expenses` | Validate or import expense CSV |
| `GET` / `POST` | `/api/assistant/status`, `/api/assistant/chat` | Assistant |
| `POST` | `/api/overview/chat` | Chat page (Gemini or Nemotron) |
| `GET` | `/api/documents`, `/api/documents/{id}` | Document library |
| `POST` | `/api/documents/upload` | Upload and extract a document |
| `POST` | `/api/documents/{id}/import` | Confirm an import into the ledger |

Invalid input returns `422` with readable field-level messages. Missing records
return `404`. Storage failures return `500` without overwriting the affected
file.

## Known limitations

- There is **no authentication** or access control. The app is for local or
  trusted-network use only and is **not hardened for public internet
  deployment**.
- The balance sheet is partial and states its assumptions.
- Tax reporting does not implement any jurisdiction's filing rules.
- Cash movement dates paid invoices by their issue date because payment dates
  are not recorded.
- AI-suggested prices are model-produced; verify against the linked source
  before acting. The app never cancels or purchases anything.
- AI processing sends a limited slice of data to Gemini or NIM. There is no
  offline-only mode beyond running with no keys.
- Auto-filing trusts the extracted type, so a document read as an invoice or an
  expense is filed even when details are missing. A blank field is reported as
  "Unstated &lt;field&gt; on file &lt;name&gt;". A wrong record can be corrected or deleted
  from its ledger page.
- Uploads have no file-size or batch limit, so a very large file is read into
  memory before it is stored.
- Scanned-PDF support renders at most eight pages per document.
- Backup restore validates structure but replaces the dataset; it does not
  merge.
- JSON storage suits a single local instance, not concurrent multi-user writes.

---

> **Note:** This is the 2nd Gen, the original code had way too many files and so I have compiled everything into one app.py