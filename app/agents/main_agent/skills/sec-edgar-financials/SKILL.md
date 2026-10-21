---
name: sec-edgar-financials
description: US SEC EDGAR filing search and financial statement extraction skill (10-K, 10-Q, 8-K, Form 4 insider trading) using edgartools and sec-api. Provides exact numeric extraction for balance sheets, income statements, and cash flows.
category: finance
tags: [sec, edgar, 10-k, 10-q, balance-sheet, income-statement, us-equity, disclosure]
dependencies: [edgartools, pandas, pydantic]
---

# sec-edgar-financials Skill

Automated toolset for querying and extracting regulatory disclosures and financial statements directly from the US Securities and Exchange Commission (SEC) EDGAR system.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Corporate Annual & Quarterly Reporting**: Parse 10-K (Annual) and 10-Q (Quarterly) filings for S&P 500 / NASDAQ firms.
- **Accurate Financial Statements**: Extract structured Balance Sheets, Income Statements, and Statements of Cash Flows directly from XBRL data.
- **Material Event Analysis (8-K)**: Detect unscheduled material corporate events, M&A filings, and executive changes.
- **Insider Trading Activity (Form 4)**: Track buying/selling transactions by C-suite executives and board directors.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
sec_edgar_tools.py - SEC EDGAR integration tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
import os
from edgar import set_identity, Company
from langchain_core.tools import tool

# Set SEC user-agent identity (mandatory by SEC policy: Name email@domain.com)
set_identity("FinancialAgentAdmin admin@financial-agent.com")

@tool
def get_company_filings(ticker: str, form: str = "10-K", count: int = 3) -> List[Dict[str, Any]]:
    """
    Search recent SEC filings for a US listed company.
    :param ticker: Company ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA')
    :param form: Filing form type ('10-K', '10-Q', '8-K', '4')
    :param count: Maximum number of recent filings to return
    """
    company = Company(ticker)
    filings = company.get_filings(form=form).latest(count)
    
    results = []
    for f in filings:
        results.append({
            "form": f.form,
            "filing_date": str(f.filing_date),
            "accession_number": f.accession_number,
            "period_of_report": str(f.period_of_report) if hasattr(f, "period_of_report") else None,
            "url": f.url
        })
    return results

@tool
def get_latest_financial_statement(ticker: str, statement_type: str = "income") -> Dict[str, Any]:
    """
    Extract the latest structured financial statement from a 10-K or 10-Q filing.
    :param ticker: Stock ticker (e.g. 'NVDA')
    :param statement_type: 'income' (Income Statement), 'balance' (Balance Sheet), 'cash' (Cash Flow)
    """
    company = Company(ticker)
    latest_10k = company.get_filings(form="10-K").latest(1)
    if not latest_10k:
        return {"error": f"No 10-K found for {ticker}"}
    
    filing_obj = latest_10k[0].obj()
    financials = filing_obj.financials
    
    if statement_type == "income" and financials.income_statement:
        df = financials.income_statement.to_dataframe()
    elif statement_type == "balance" and financials.balance_sheet:
        df = financials.balance_sheet.to_dataframe()
    elif statement_type == "cash" and financials.cash_flow_statement:
        df = financials.cash_flow_statement.to_dataframe()
    else:
        return {"error": f"Statement type '{statement_type}' not found or parsed."}

    # Format into lightweight JSON preview
    preview = {}
    for idx, row in df.head(15).iterrows():
        preview[str(row.iloc[0])] = row.iloc[1] if len(row) > 1 else None

    return {
        "ticker": ticker.upper(),
        "statement_type": statement_type,
        "filing_date": str(latest_10k[0].filing_date),
        "data_preview": preview
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "엔비디아(NVDA)의 최신 10-K 연례보고서에서 손익계산서 요약과 최근 8-K 주요 공시 내역을 알려줘."
  agent_steps:
    1. Call `get_latest_financial_statement(ticker="NVDA", statement_type="income")`.
    2. Call `get_company_filings(ticker="NVDA", form="8-K", count=3)`.
    3. Generate a structured summary of revenue growth and key corporate disclosures.
```
