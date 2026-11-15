# How to use Finance Agent

A practical, task-by-task guide. If you want the reference material (full API,
environment variables, architecture), see [README.md](README.md).

Finance Agent is a private finance workspace that runs entirely on your own
machine. There is no account, no cloud sync, and no telemetry. Your ledger lives
in plain JSON files in a `data/` folder next to `app.py`.

---

## Contents

1. [Quick start (5 minutes)](#1-quick-start-5-minutes)
2. [First-time setup](#2-first-time-setup)
3. [Finding your way around](#3-finding-your-way-around)
4. [Overview](#4-overview)
5. [Invoices](#5-invoices)
6. [Expenses](#6-expenses)
7. [Documents (AI intake)](#7-documents-ai-intake)
8. [Chat](#8-chat)
9. [SpendWise](#9-spendwise)
10. [Reports](#10-reports)
11. [Activity](#11-activity)
12. [Backups and restore](#12-backups-and-restore)
13. [CSV import](#13-csv-import)
14. [AI setup and privacy](#14-ai-setup-and-privacy)
15. [Using it on your phone or another device](#15-using-it-on-your-phone-or-another-device)
16. [Where your data lives](#16-where-your-data-lives)
17. [Troubleshooting](#17-troubleshooting)
18. [Honest limitations](#18-honest-limitations)

---

## 1. Quick start (5 minutes)

### Install

```bash
cd Finance-Agent          # your extracted or cloned folder
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: add AI keys

Everything works without keys except the AI features (document reading, chat,
summaries). To turn those on:

```bash
cp .env.example .env
# open .env and paste your GEMINI_API_KEY and/or NVIDIA_API_KEY
```

Get a Gemini key at <https://aistudio.google.com> and an NVIDIA NIM key at
<https://build.nvidia.com>.

### Run

```bash
python app.py
```

Open **<http://127.0.0.1:2611>** in your browser. The `data/` folder is created
automatically on first run.

> `python app.py` does not auto-reload after code edits. For development use:
> ```bash
> uvicorn app:app --reload --host 0.0.0.0 --port 2611
> ```

### Your first 5 minutes

1. Go to **Settings** and fill in your business name, currency, and default tax
   rate. Save.
2. Go to **Invoices → New invoice**, create one invoice for a real client.
3. Go to **Expenses → Add expense**, record one cost.
4. Go back to **Overview** — the metrics and the six-month chart should now be
   populated.
5. Go to **Documents**, drop in a PDF invoice or receipt, and watch it get filed.

That is the whole loop: record money in, record money out, review the position.

---

## 2. First-time setup

Open **Settings** (the gear at the bottom of the sidebar).

### Currency & formatting

| Field | What it does |
| --- | --- |
| Currency code | Three-letter ISO code, e.g. `USD`, `EUR`, `GBP` |
| Currency symbol | What appears before amounts, e.g. `$`, `€` |
| Locale | Controls number and date formatting, e.g. `en-US`, `en-IE` |
| Default tax rate | Percent pre-filled on new invoices (existing invoices keep their own rate) |
| Date format | Short, Medium, Long, or ISO |

A brand-new install starts with **EUR**, **€**, and **en-IE**. Change it here
before you enter data so everything is formatted consistently.

### Business details

- **Business name** and the first three lines of **Business address** print in
  the "From" block of a PDF invoice.
- **Payment details** is a free-text block (bank name, IBAN, reference) printed
  in the invoice footer. Never put passwords or API keys here.
- **Business email** is stored for your records; it is not printed on the PDF.

Click **Save settings**.

---

## 3. Finding your way around

The left sidebar is the whole app:

| Page | What it is for |
| --- | --- |
| **Overview** | Your financial position at a glance |
| **Chat** | Ask questions about your ledger, or attach a file to file it |
| **Invoices** | Money you are owed |
| **Expenses** | Money you have spent |
| **SpendWise** | Spending analytics, budgets, what-if planning |
| **Documents** | Upload invoices/receipts and let AI read them |
| **Reports** | Profit & loss, balance sheet, tax summary |
| **Activity** | A lightweight audit trail of what changed |
| **Settings** | Currency, backups, CSV import |

Two more controls sit at the bottom of the sidebar:

- **Assistant** opens a small read-only chat panel from any page. It can answer
  questions but cannot edit anything.
- **Light mode / Dark mode** toggles the theme (saved in your browser).

On a phone or narrow window the sidebar collapses into a **hamburger menu** in
the top bar.

---

## 4. Overview

The home page answers "where do I stand right now?".

### The five metrics

| Metric | Meaning |
| --- | --- |
| **Invoiced** | Total value of issued invoices, including tax |
| **Collected** | Total of invoices marked *paid* |
| **Outstanding** | Total of invoices that are *sent* or *overdue* |
| **Expenses** | Every recorded expense |
| **Net profit** | Net revenue (excluding tax) minus expenses, on an accrual basis |

Click through the panels below:

- **Financial summary** — click **Generate summary** for a written explanation
  of your invoices, expenses, and spending. Needs a Gemini key.
- **Revenue & expenses** — a six-month chart of net revenue against expenses.
- **Invoice status** — how much is paid, sent, draft, and overdue.
- **Recent invoices** — the newest entries; **View all** goes to Invoices.
- **Needs attention** — overdue invoices and expenses awaiting reimbursement.

> **Accrual, not cash.** Revenue is counted when an invoice is *issued*
> (sent, overdue, or paid). Draft and cancelled invoices do not count. So "Net
> profit" is not the same as money in your bank account.

---

## 5. Invoices

### Create an invoice

1. Click **New invoice**.
2. Fill in **Client name** (required) and **Client email** (optional).
3. Set **Issue date** and **Due date**.
4. Set **Tax rate** (a percentage, e.g. `20`).
5. Add **Line items**: description, quantity, unit price. Use **Add line** for
   more rows.
6. Choose a **Status**: draft, sent, paid, overdue, or cancelled.
7. Add **Notes** if you want payment terms on the document.
8. Save.

The invoice gets a reference like `INV-0001`.

### Work with an invoice

Click the eye icon to **View invoice**. From there you can:

- Download the PDF (the button is labelled with the invoice reference).
- **Edit invoice**.
- **Mark as paid** (hidden once it is paid or cancelled).
- **Delete invoice** (asks for confirmation and backs up first).

### Find things

- **Search** by number, client, email, or status.
- Filter with the **All / Draft / Sent / Paid / Overdue** pills.
- Choose **Rows** per page (10/25/50/100) and page with **Previous / Next**.

### Bulk actions

Tick rows (or use the header checkbox), then use the bar that appears:

- **Export CSV** of the selection.
- **Set status… → Apply** to change many invoices at once.
- **Delete selected** (confirms and backs up first).

If your selection spans pages, **Select all matching** extends it to every
result of the current search/filter.

---

## 6. Expenses

### Add an expense

1. Click **Add expense**.
2. Enter **Description** and **Amount** (both required).
3. Pick a **Category**:
   `office, travel, utilities, software, hardware, food, transport, rent,
   services, other`.
4. Set **Date incurred** and **Vendor**.
5. Tick **Reimbursable** if you paid personally and want it paid back. A second
   **Reimbursed** switch appears once it is reimbursable.
6. Add **Notes** (receipt reference, project, and so on).
7. Save.

### Find, filter, and bulk-edit

- **Search** by vendor, description, category, or amount.
- Filter with the **category pills**.
- Bulk bar supports **Export CSV**, **Reimbursement… → Apply** (mark reimbursed
  or pending), and **Delete selected**.

---

## 7. Documents (AI intake)

This is the "drop a file, get a ledger record" feature. It needs an NVIDIA NIM
key (with Gemini as fallback).

### Upload

1. Open **Documents**.
2. Drop files onto the zone, or click **Drop files here or choose files**.
3. Accepted: **PDF, CSV, XLSX, DOCX, TXT, PNG, JPG, WEBP**. There is **no size
   limit** and no batch limit.
4. Click **Upload and extract**.

The app reads each file and:

- **Invoices, receipts, and expenses** are filed automatically into the ledger.
- **Bank statements and other documents** stay in the library for review.
- A field the model could not read is recorded as **"Unstated"** rather than
  guessed at.
- A likely duplicate is skipped.

### Review results

- **Extraction results** shows what was read and links to the record it created.
- **Document library** lists every file with its type, status, and resulting
  record.
- If reading failed, use **Retry processing**.
- If it is a document type that is not auto-filed, use **Import to ledger** after
  reviewing the extracted fields.
- Blank or wrong records can be corrected or deleted from the ledger page they
  were filed to.

> Auto-filing trusts the extracted type. If a receipt is misread as an invoice,
> fix or delete it in **Invoices**. Nothing is hidden from you.

---

## 8. Chat

A dedicated page for talking to your data. Apart from attaching a file to file it,
it is read-only: it never edits or deletes a record.

### Ask a question

1. Open **Chat**.
2. Click one of the suggested prompts, or type in **Message the AI…**.
3. Press **Enter** to send, **Shift+Enter** for a new line.

### Pick a model

Use the dropdown in the top-right (it shows **Nemotron** by default):

| Option | Backed by |
| --- | --- |
| **Nemotron** | NVIDIA NIM |
| **Gemini** | Google |
| **Auto** | Best available |

Your choice is remembered.

### Attach a file

Click the **paperclip → Attach a file**. Attaching a document sends it to be
read and filed, just like the Documents page. This is handy when you want to
ask about the thing you just uploaded.

### Manage conversations

- **New chat** starts a fresh conversation.
- Older conversations are listed in the left rail; click one to reopen it.
- Use the **rename** and **delete** buttons on each conversation.

> Conversations are saved **in this browser only** (`localStorage`). They do not
> sync between devices, and clearing site data erases them.

The separate **Assistant** button at the bottom of the sidebar is a lighter,
always-available panel. It answers questions and offers navigation help.

---

## 9. SpendWise

Spending analytics built from the same expense records, so you never enter data
twice.

### Choose a period

Use the period selector to switch between **Last 7 days**, **Last 30 days**,
**Current month**, **Previous month**, **Last 3 months**, **Last 6 months**,
**Current year**, and **Custom range**. Click **Refresh** to reload.

### What you can read

- **Top metrics** — total spending, this month, average month, largest category,
  budget used.
- **AI summary** — **Generate summary** for a written explanation of the period.
- **Spending trend** — monthly or weekly bars.
- **Cash movement** — paid invoices in, expenses out, net movement. This is a
  *cash* view and is not the same as accrual profit.
- **Categories** and **Vendors** — click any row to drill into its transactions.

### Budgets

1. Give the budget a **Budget name**, an **Amount**, a **Category**, and a
   **Month** (leave the month as "Every month" for a recurring monthly budget).
2. Click **Add budget**.
3. The budget-use metric and insights update to show what is over or close to
   the amount.

### Categorisation rules

Rules auto-assign a category when a vendor or description matches.

1. Give the rule a **name**.
2. Set the match: **Vendor contains** or **Description contains**.
3. Pick the **Category** and **Priority**.
4. Click **Add rule**.

Suggested categories appear as you type, based on your existing expenses.

### Insights and savings

- **Insights** highlights patterns in the period.
- **Potential savings** lists model suggestions separately from calculated
  figures. Treat suggested prices as a starting point, not verified facts.
- **Possible recurring charges** flags subscriptions detected from repeated
  charges.
- **Unusually large expenses** flags outliers.

### What-if Simulator

Answer "what happens if I earn less / spend more / buy a laptop?" without
touching your ledger.

1. Drag the sliders (or type numbers) for **Monthly subscriptions**, **Monthly
   income change**, **Other monthly expenses change**, **Monthly savings
   change**, and **One-time purchase**.
2. Or type a sentence such as *"What if my income decreases by 15%?"* into **Or
   describe it in words** and click **Interpret**.
3. Compare the baseline with the simulated outcome.
4. **Reset** clears everything.

Nothing here is saved. It is planning, not accounting.

### Subscription Optimiser

Lists recurring charges and, if you confirm what each one is used for, compares
AI-suggested alternatives and the calculated savings. The app never cancels or
buys anything.

### Document Analysis and Import

- **Document Analysis** shows recently uploaded documents and what the AI read.
- **Import spending records** imports an expenses CSV straight into SpendWise.
  A missing category is suggested from your rules first.

---

## 10. Reports

Three statements, all on a documented **accrual basis**.

### Choose the report

Use the tabs:

- **Profit & loss** — revenue minus expenses for a period.
- **Balance sheet** — a *partial* management position (see the note on the page).
- **Tax summary** — tax invoiced versus tax collected.

### Choose the period

Set **From** and **To**, or use a preset: **This month**, **Last month**,
**This quarter**, **Year to date**. Then click **Generate**.

**Print statement** opens your browser's print dialog for the current report.

> The balance sheet is not a statutory balance sheet. Vendor payables, bank
> balances, and opening equity are not modelled. This app is not tax advice.

---

## 11. Activity

A lightweight audit trail: what was created, edited, deleted, imported, or filed,
with timestamps. It is deliberately short and never records secrets. Use it when
you want to know "did I already enter this?".

---

## 12. Backups and restore

Both live in **Settings → Backups & recovery**.

### Download a backup

Click **Download backup**. You get a timestamped JSON file containing invoices,
expenses, budgets, rules, activity, and public settings. **API keys are never
included.**

Keep these somewhere safe; they are your only portable copy of the ledger.

### Restore a backup

1. Drop a `.json` backup onto **Choose a backup file**.
2. Click **Restore**.
3. Confirm.

Restore validates the file before replacing anything, and it backs up your
current data first. It **replaces** the dataset; it does not merge.

### Automatic backups

The app writes a backup into `data/backups/` before destructive operations
(deleting, bulk changes, restore). If something goes wrong, that folder is your
undo history.

---

## 13. CSV import

Two places do CSV import: **Settings → CSV import** (both invoices and expenses)
and **SpendWise → Import spending records** (expenses only).

Every import is **validated and previewed first**. Invalid rows block the import
rather than being silently dropped.

### Invoice CSV columns

```text
client_name, client_email, invoice_number, issue_date, due_date, status,
description, quantity, unit_price, tax_rate, notes
```

- Required: `client_name`, `description`, `unit_price`.
- Rows sharing an `invoice_number` are grouped into one invoice.
- `tax_rate` may be a percentage (`20`) or a fraction (`0.2`).
- Common aliases are recognised.

### Expense CSV columns

```text
description, amount, date_incurred, vendor, category, reimbursable, reimbursed, notes
```

- Required: `description`, `amount`.
- `category` must match one of the configured categories.

### The workflow

1. Choose the CSV file.
2. Click **Preview**.
3. Fix any row-level errors it reports.
4. Confirm to import.

There is no file-size limit, so large exports are fine. A single import is
capped at 200,000 rows.

---

## 14. AI setup and privacy

AI is **optional**. With no keys, all the calculations, invoices, expenses,
reports, and CSV import still work; the AI panels simply say they are unavailable.

### Two independent providers

| Provider | Used for |
| --- | --- |
| **Google Gemini** (`GEMINI_API_KEY`) | Financial summaries and the read-only assistant |
| **NVIDIA NIM / Nemotron** (`NVIDIA_API_KEY`) | Document reading, what-if interpretation, subscription alternatives, spending suggestions |

Set them in `.env` and **restart the app**.

### What leaves your machine

Only what a feature needs:

- **Assistant / chat** sends a compact summary of invoices and expenses.
- **Document reading** sends the extracted text, and page images for scans.
- **What-if text** sends baseline figures plus your sentence.
- **Suggestion features** send the single item and its calculated facts.

Financial arithmetic always happens in deterministic Python. The model returns
structure or prose; it never produces the numbers you see. Keys stay on the
server and are never sent to the browser or written into backups.

---

## 15. Using it on your phone or another device

By default `python app.py` listens on **every network interface**, so any device
on the same Wi-Fi can open the app.

1. Find your computer's local IP:
   - Linux: `hostname -I`
   - macOS: `ipconfig getifaddr en0`
   - Windows: `ipconfig` (look for IPv4 Address)
2. On your phone, open `http://<that-ip>:2611`.

If it does not load, check that your firewall allows port `2611` and that both
devices are on the same network.

> ⚠️ There is **no login**. Anyone who can reach the address can read and change
> your ledger. Use this only on a network you trust. To keep it local-only, start
> with `HOST=127.0.0.1 python app.py`.

The layout is mobile-optimised: the sidebar becomes a hamburger menu, tables
scroll horizontally, and the Chat page keeps its composer pinned to the bottom.

---

## 16. Where your data lives

All inside the `data/` folder next to `app.py`:

| File | Contents |
| --- | --- |
| `data/invoices.json` | All invoices |
| `data/expenses.json` | All expenses |
| `data/activity.json` | Activity trail |
| `data/settings.json` | Currency, locale, tax, business details |
| `data/budgets.json` | SpendWise budgets |
| `data/rules.json` | Categorisation rules |
| `data/backups/*.json` | Automatic and downloaded backups |

Writes are atomic, so an interrupted save cannot truncate a ledger. Corrupt JSON
is reported, never silently overwritten.

To use a different folder, set `FINANCE_DATA_DIR=/path/to/folder`.

---

## 17. Troubleshooting

**The page will not load.**
Check the terminal is still running `python app.py`, and that you are opening the
right port. If `2611` is taken, start with `PORT=2620 python app.py`.

**I can't open it from my phone.**
Confirm both are on the same Wi-Fi, use the computer's LAN IP (not
`127.0.0.1`), and allow port `2611` through the firewall.

**AI features say unavailable.**
Add the matching key to `.env` and restart the app. Gemini powers summaries and
the assistant; NVIDIA NIM powers document reading, suggestions, and the what-if
interpreter.

**A document was not read correctly.**
Scanned PDFs are rendered at most eight pages. Use **Retry processing**, or enter
the details yourself; nothing is saved until you confirm.

**The import is blocked.**
The preview found row-level errors. Fix the flagged rows (usually a missing
required column or a category not in the list) and preview again.

**My numbers look different from my bank balance.**
The app is accrual-based: revenue counts when an invoice is *issued*, not when
you are paid. Draft and cancelled invoices are excluded. The SpendWise
cash-movement view is the cash-style picture.

**Restore was rejected.**
The file is invalid, an incompatible version, or has bad records. Restore never
partially applies; nothing was changed.

**Something looks corrupted.**
Do not hand-edit the JSON. Restore the most recent file from `data/backups/`, or
from a backup you downloaded.

**I need to reset everything.**
Stop the app, move `data/` somewhere safe, and start it again. A fresh, empty
dataset is created.

---

## 18. Honest limitations

- **No authentication.** For local or trusted-network use only, not the public
  internet.
- **Single instance.** JSON storage suits one user; it is not built for
  concurrent multi-user writes.
- **Partial balance sheet.** It states its assumptions and is not statutory.
- **No tax filing rules.** Tax summary is informational only.
- **Cash movement** dates paid invoices by issue date because payment dates are
  not recorded.
- **AI suggestions are suggestions.** Verify any price against its source. The
  app never buys, cancels, or pays anything.
- **Scan reading** is capped at eight pages per document.
- **Restore replaces**, it does not merge.

---

## More

- Full reference, API, and environment variables: [README.md](README.md)
- Licence: [LICENSE.txt](LICENSE.txt)
- Dataset credits: [CREDITS.md](CREDITS.md)