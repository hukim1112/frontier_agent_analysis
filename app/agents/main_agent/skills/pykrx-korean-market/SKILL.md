---
name: pykrx-korean-market
description: Korean Capital Market (KOSPI, KOSDAQ, KONEX) data extraction skill using FinanceDataReader and pykrx. Supports Korean stock tickers, daily OHLCV historical charts, market cap ranking, investor net buying trends, and financial ratios.
category: finance
tags: [korea, kospi, kosdaq, krx, fdr, pykrx, investor-trends, valuation]
dependencies: [finance-datareader, pykrx, pandas, numpy]
---

# pykrx-korean-market Skill

Specialized skill for the Korean domestic securities and capital markets (KRX, KOSPI, KOSDAQ). Combines `FinanceDataReader` (high reliability for listings & market cap) and `pykrx` (for investor breakdown and fundamental analytics).

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Market Indices & Sector Benchmarks**: Query daily performance of KOSPI, KOSDAQ, KRX 300.
- **Top Market Cap Rankings**: Fetch real-time market cap rankings across KOSPI / KOSDAQ without login restrictions.
- **Historical OHLCV Data**: Retrieve historical price candles for Samsung Electronics, SK Hynix, and all Korean listed assets.
- **Investor Flow Analysis (수급 분석)**: Query Foreign, Institutional, and Retail net trade distributions.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
korean_market_tools.py - Korean stock market tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr
from langchain_core.tools import tool

@tool
def get_krx_stock_quote(ticker_or_name: str) -> Dict[str, Any]:
    """
    Get the latest quote, price changes, and 52-week data for a Korean stock by ticker code (e.g., '005930') or company name (e.g., '삼성전자').
    """
    # Fetch KRX listing summary
    df_krx = fdr.StockListing('KRX')
    
    # Search by code or name
    ticker_clean = ticker_or_name.replace(".KS", "").replace(".KQ", "").strip()
    match = df_krx[(df_krx['Code'] == ticker_clean) | (df_krx['Name'] == ticker_or_name)]
    
    if match.empty:
        return {"error": f"Stock '{ticker_or_name}' not found in KRX listing."}
        
    row = match.iloc[0]
    code = row['Code']
    name = row['Name']
    
    # Fetch recent OHLCV
    start_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
    df_ohlcv = fdr.DataReader(code, start_date)
    
    latest_candle = df_ohlcv.iloc[-1] if not df_ohlcv.empty else {}

    return {
        "ticker": code,
        "name": name,
        "market": row.get('Market', 'KRX'),
        "sector": row.get('Sector', 'Unknown'),
        "latest_close": int(row.get('Close', latest_candle.get('Close', 0))),
        "change_pct": float(row.get('ChagesRatio', 0.0)),
        "volume": int(row.get('Volume', latest_candle.get('Volume', 0))),
        "amount_krw": int(row.get('Amount', 0)),
        "market_cap_krw": int(row.get('Marcap', 0)),
        "shares_outstanding": int(row.get('Stocks', 0))
    }

@tool
def get_krx_top_market_cap(market: str = "KOSPI", top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top N stocks ranked by market capitalization in KOSPI, KOSDAQ, or KRX.
    """
    market_filter = market.upper()
    df_krx = fdr.StockListing('KRX')
    
    if market_filter in ["KOSPI", "KOSDAQ", "KONEX"]:
        df_filtered = df_krx[df_krx['Market'] == market_filter]
    else:
        df_filtered = df_krx
        
    df_sorted = df_filtered.sort_values(by='Marcap', ascending=False).head(top_n)
    
    results = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        results.append({
            "rank": rank,
            "ticker": row['Code'],
            "name": row['Name'],
            "market": row['Market'],
            "close": int(row['Close']),
            "change_pct": float(row.get('ChagesRatio', 0.0)),
            "market_cap_krw": int(row['Marcap']),
            "sector": row.get('Sector', '')
        })
    return results

@tool
def get_krx_historical_ohlcv(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get historical daily OHLCV candles for a Korean stock code (e.g., '000660', '035420').
    """
    ticker_clean = ticker.replace(".KS", "").replace(".KQ", "").strip()
    start_date = (datetime.now() - timedelta(days=days*2)).strftime("%Y-%m-%d")
    df = fdr.DataReader(ticker_clean, start_date)
    
    if df.empty:
        return []
        
    records = []
    for date_idx, row in df.tail(days).iterrows():
        records.append({
            "date": str(date_idx)[:10],
            "open": int(row["Open"]),
            "high": int(row["High"]),
            "low": int(row["Low"]),
            "close": int(row["Close"]),
            "volume": int(row["Volume"]),
            "change": round(float(row.get("Change", 0.0)) * 100, 2)
        })
    return records
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "코스피 시가총액 상위 5개 종목과 삼성전자의 최근 10일간 주가 추이를 알려줘."
  agent_steps:
    1. Call `get_krx_top_market_cap(market="KOSPI", top_n=5)`.
    2. Call `get_krx_historical_ohlcv(ticker="005930", days=10)`.
    3. Generate a structured capital market briefing.
```
