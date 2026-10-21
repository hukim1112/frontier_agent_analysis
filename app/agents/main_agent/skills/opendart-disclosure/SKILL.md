---
name: opendart-disclosure
description: Korean Financial Supervisory Service (FSS) Open DART API skill for retrieving corporate filings, quarterly/annual business reports, major shareholding status, dividend announcements, and key corporate events for KOSPI and KOSDAQ listed companies.
category: finance
tags: [korea, dart, opendart, fss, disclosure, business-report]
dependencies: [opendartreader, pandas, requests]
---

# opendart-disclosure Skill

Specialized skill for integrating with the Financial Supervisory Service (FSS) Open DART API (전자공시시스템). Essential for domestic fundamental analysis, corporate governance inspection, and compliance checking.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Corporate Disclosure Search**: Query real-time disclosures, major contracts, capital increases, and treasury stock acquisitions.
- **Financial Statements Comparison**: Compare historical Balance Sheets and Income Statements across multiple fiscal years.
- **Executive Compensation & Shareholding**: Query major shareholder changes (5% rule) and executive compensation data.
- **Dividend Payouts & Schedules**: Analyze historical cash and stock dividends.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
opendart_tools.py - Open DART integration tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
import os
import OpenDartReader
from langchain_core.tools import tool

# API Key can be loaded from environment variables
DART_API_KEY = os.getenv("OPENDART_API_KEY", "")

def get_dart_client():
    if not DART_API_KEY:
        raise ValueError("OPENDART_API_KEY is not set in environment variables.")
    return OpenDartReader(DART_API_KEY)

@tool
def search_recent_disclosures(corp_name: str, days_back: int = 30) -> List[Dict[str, Any]]:
    """
    Search recent filings from DART for a specific Korean company (e.g., '삼성전자', '카카오', '현대차').
    """
    dart = get_dart_client()
    from datetime import datetime, timedelta
    bgn_de = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    
    df = dart.list(corp_name, start=bgn_de)
    if df is None or df.empty:
        return []
    
    results = []
    for _, row in df.head(10).iterrows():
        results.append({
            "corp_name": row["corp_name"],
            "report_nm": row["report_nm"],
            "rcept_dt": row["rcept_dt"],
            "flr_nm": row["flr_nm"],
            "rcept_no": row["rcept_no"]
        })
    return results

@tool
def get_dart_financial_indicators(corp_name: str, bsns_year: int = 2023, reprt_code: str = "11011") -> Dict[str, Any]:
    """
    Fetch key balance sheet and income statement items from DART business report.
    :param corp_name: Company name (e.g. '현대자동차', '005380')
    :param bsns_year: Fiscal Year (e.g. 2023, 2024)
    :param reprt_code: Report code - '11013' (Q1), '11012' (Half), '11014' (Q3), '11011' (Annual)
    """
    dart = get_dart_client()
    df = dart.finstate(corp_name, bsns_year, reprt_code=reprt_code)
    
    if df is None or df.empty:
        return {"error": f"No financial statement found for {corp_name} in {bsns_year}"}
    
    # Filter major account items
    key_accounts = ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계"]
    summary = {}
    
    for _, row in df.iterrows():
        account_name = row["account_nm"]
        if any(k in account_name for k in key_accounts):
            summary[account_name] = {
                "current_amount": row.get("thstrm_amount", "0"),
                "previous_amount": row.get("pvrm_amount", "0")
            }
            
    return {
        "corp_name": corp_name,
        "year": bsns_year,
        "report_type": reprt_code,
        "accounts": summary
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "카카오의 최근 3개월간 주요 공시 목록과 2023년 사업보고서 기준 영업이익을 알려줘."
  agent_steps:
    1. Call `search_recent_disclosures(corp_name="카카오", days_back=90)`.
    2. Call `get_dart_financial_indicators(corp_name="카카오", bsns_year=2023, reprt_code="11011")`.
    3. Generate a structured Korean disclosure briefing.
```
