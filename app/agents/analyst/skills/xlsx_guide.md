---
name: xlsx-guide
description: Expert guidelines and patterns for generating professional, formula-driven Excel spreadsheets using openpyxl and pandas.
---

# Professional Excel Spreadsheet Generation (`openpyxl`)

This guide provides technical rules, formula compatibility constraints, and aesthetic conventions when generating spreadsheets via the `excel_writer` tool.

## Library Selection

| Task | Library | Notes |
|---|---|---|
| **Formulas & Styled Workbooks** | `openpyxl` | Primary tool for dynamic formulas, styling, conditional formats |
| **Bulk Data Ingestion & Export** | `pandas` (`read_excel`, `to_excel`) | Use for pure tabular input/output |
| **Reading Values & Formulas** | Two-pass `load_workbook` | Pass 1: `data_only=False` (formulas), Pass 2: `data_only=True` (values) |

## Mandatory Requirements for Every Output

1. **Professional Font**: Use `Arial` or `Segoe UI` (size 10-11 for data, 11-12 bold for headers).
2. **Dynamic Formulas, Never Hardcode**:
   - Write `ws['B10'] = '=SUM(B2:B9)'`, NOT the static evaluated Python number.
   - The sheet must recalculate automatically when raw data values change.
3. **Follow Specifications Literally**: Use exact tab names, column headers, and calculation logic.
4. **Document Assumptions & Constants**: Add comments or an adjacent note column for constants (`Source: Scraped Danawa Data, 2026-08`).
5. **Format Numbers Explicitly**:
   - Currency: `#,##0` or `₩#,##0` (or include units in the column header: `Price (KRW)`)
   - Percentages: `0.0%` stored as decimal fraction (`0.15` displays `15.0%`)
   - Zero values: render cleanly (`#,##0;(#,##0);-`)

## Formula Compatibility & Safety Rules

`openpyxl` writes raw formula strings into Excel XML without pre-evaluating them. To ensure clean opening across Microsoft Excel, LibreOffice, and Google Sheets:

### ✅ Safe Classic Formulas (Excel 2007 Compatible)
`SUM`, `AVERAGE`, `COUNT`, `SUMIFS`, `COUNTIFS`, `AVERAGEIFS`, `INDEX`, `MATCH`, `IFERROR`, `VLOOKUP`, `IF`, `MAX`, `MIN`

### ⚠️ Post-2007 Functions (Prefix Required)
Post-2007 functions require the `_xlfn.` prefix when written via `openpyxl` (Excel hides this prefix in the UI):
```python
_xlfn.TEXTJOIN, _xlfn.CONCAT, _xlfn.IFS, _xlfn.SWITCH, _xlfn.MAXIFS, _xlfn.MINIFS
```
*Writing them without `_xlfn.` yields `#NAME?` in Excel.*

### ❌ Strictly Prohibited Array/Spill Functions
Do NOT write dynamic array/spill formulas (`XLOOKUP`, `XMATCH`, `SORT`, `FILTER`, `UNIQUE`, `SEQUENCE`).
- **Alternative**: Use standard `INDEX`/`MATCH` pairs for lookups. Perform sorting, filtering, and deduplication in Python prior to cell population.

## Critical `openpyxl` Gotchas

- **Destructive `data_only=True`**: Saving a workbook loaded with `data_only=True` permanently wipes all formulas into static literals. Never write to a `data_only=True` workbook.
- **Merged Cells**: Write ONLY to the top-left anchor cell. Secondary cells in a merged range are read-only `MergedCell` instances.
- **Cross-Sheet References**: Sheet names containing spaces must be single-quoted: `='Sales Data'!$B$5`.

## Financial Modeling & Color Conventions

| Category | Text / Fill Color | RGB Code | Purpose |
|---|---|---|---|
| **Input / Constants** | Blue text | `RGB(0, 0, 255)` / `0000FF` | Raw scraped data, user parameters |
| **Formulas / Computed** | Black text | Default / `000000` | Calculated totals, ratios, projections |
| **Cross-Sheet Links** | Green text | `RGB(0, 128, 0)` / `008000` | References linking other tabs |
| **Table Header** | White text on Slate/Blue fill | Fill `4472C4` / `2C3E50` | Professional header styling |

## Standard Python Implementation Template

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()
ws = wb.active
ws.title = "Market Analysis"

# Styling definitions
font_header = Font(name="Arial", size=11, bold=True, color="FFFFFF")
fill_header = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
align_center = Alignment(horizontal="center", vertical="center")
align_right = Alignment(horizontal="right", vertical="center")

headers = ["Product Name", "Price", "Switch", "Connection", "Reviews", "Price / Review"]
for col_idx, h in enumerate(headers, 1):
    c = ws.cell(row=1, column=col_idx, value=h)
    c.font = font_header
    c.fill = fill_header
    c.alignment = align_center

# Populate records from DataFrame
for r_idx, row in enumerate(df.to_dict('records'), start=2):
    ws.cell(row=r_idx, column=1, value=row.get('product_name', ''))
    ws.cell(row=r_idx, column=2, value=row.get('price', 0))
    ws.cell(row=r_idx, column=3, value=row.get('switch_type', ''))
    ws.cell(row=r_idx, column=4, value=row.get('connection_type', ''))
    ws.cell(row=r_idx, column=5, value=row.get('review_count', 0))
    # Dynamic formula
    ws.cell(row=r_idx, column=6, value=f'=IFERROR(B{r_idx}/E{r_idx}, "-")')

# Summary / Total row
last_r = len(df) + 1
ws.cell(row=last_r + 1, column=1, value="Average / Total").font = Font(name="Arial", size=11, bold=True)
ws.cell(row=last_r + 1, column=2, value=f'=AVERAGE(B2:B{last_r})')
ws.cell(row=last_r + 1, column=5, value=f'=SUM(E2:E{last_r})')

# Format columns
for r in ws.iter_rows(min_row=2, max_row=last_r + 1, min_col=2, max_col=2):
    for cell in r:
        cell.number_format = '#,##0'

# Auto-adjust column widths
for col in ws.columns:
    max_len = max(len(str(cell.value or '')) for cell in col)
    col_letter = get_column_letter(col[0].column)
    ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 50)
```
