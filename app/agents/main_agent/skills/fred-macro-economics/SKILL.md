---
name: fred-macro-economics
description: Federal Reserve Economic Data (FRED) integration skill. Retrieves macroeconomic indicators including Fed Funds Rate, US Treasury Yield Curve (10Y-2Y spread), CPI Inflation, Unemployment, GDP Growth, and M2 Money Supply.
category: finance
tags: [macroeconomics, fred, interest-rates, inflation, yield-curve, gdp, central-bank]
dependencies: [fredapi, pandas, requests]
---

# fred-macro-economics Skill

Macroeconomic data skill powered by the Federal Reserve Bank of St. Louis (FRED) API. Essential for macro market research, interest rate cycle analysis, and recession risk monitoring.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Central Bank Monetary Policy**: Track Federal Funds Effective Rate (FEDFUNDS) and policy rate shifts.
- **Yield Curve Inversion**: Monitor US 10-Year vs 2-Year Treasury spread (T10Y2Y) as a leading recession indicator.
- **Inflation & Price Indexes**: Query Consumer Price Index (CPIAUCSL) and Core PCE inflation trends.
- **Labor Market Dynamics**: Track Nonfarm Payrolls (PAYEMS) and Civilian Unemployment Rate (UNRATE).

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
fred_macro_tools.py - FRED Macroeconomic tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import requests
from langchain_core.tools import tool

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Key Series ID Registry
POPULAR_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "treasury_10y_2y_spread": "T10Y2Y",
    "treasury_10y_yield": "DGS10",
    "cpi_inflation": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "real_gdp": "GDPC1",
    "m2_money_supply": "M2SL"
}

@tool
def get_macro_indicator_series(series_name_or_id: str, observations: int = 12) -> Dict[str, Any]:
    """
    Fetch recent observations for a macroeconomic indicator from FRED.
    :param series_name_or_id: Indicator alias ('fed_funds_rate', 'treasury_10y_2y_spread', 'cpi_inflation', 'unemployment_rate') or direct FRED series ID.
    :param observations: Number of recent data points to return.
    """
    series_id = POPULAR_SERIES.get(series_name_or_id.lower(), series_name_or_id.upper())
    
    if not FRED_API_KEY:
        return {"error": "FRED_API_KEY environment variable is required."}
        
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": observations
    }
    
    resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
    data = resp.json().get("observations", [])
    
    if not data:
        return {"error": f"Failed to retrieve data for series {series_id}."}
        
    history = []
    for obs in data:
        if obs["value"] != ".":
            history.append({
                "date": obs["date"],
                "value": float(obs["value"])
            })
            
    return {
        "series_id": series_id,
        "latest_date": history[0]["date"] if history else None,
        "latest_value": history[0]["value"] if history else None,
        "history": list(reversed(history))
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "미국 연준 기준금리 추이와 10년-2년 국채 장단기 금리차(T10Y2Y) 최근 동향을 분석해줘."
  agent_steps:
    1. Call `get_macro_indicator_series(series_name_or_id="fed_funds_rate", observations=6)`.
    2. Call `get_macro_indicator_series(series_name_or_id="treasury_10y_2y_spread", observations=10)`.
    3. Generate a comprehensive macroeconomic policy analysis.
```
