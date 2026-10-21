---
name: yfinance-market-data
description: Comprehensive Yahoo Finance toolset for global stock quotes, historical OHLCV data, fundamentals, company financial statements, dividends, and options chain analysis.
category: finance
tags: [stocks, market-data, yfinance, fundamentals, global-market]
dependencies: [yfinance, pandas, numpy]
---

# yfinance-market-data Skill

A comprehensive financial market data skill powered by `yfinance` to fetch real-time and historical stock market data, fundamentals, earnings, valuation metrics, and balance sheet data.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Real-time Price & Multi-Ticker Comparison**: Compare global equities (AAPL, NVDA, TSLA) and Korean ADRs/tickers (005930.KS).
- **Fundamental & Valuation Analysis**: Extract P/E, P/B, EV/EBITDA, EPS, Debt-to-Equity, and Operating Margins.
- **Historical Backtesting Feed**: Retrieve daily/weekly/monthly OHLCV candles for quantitative backtesting and charting.
- **Options & Derivatives Exploration**: Query call/put chains, strike prices, and open interest.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
yfinance_tools.py - Yahoo Finance market data tools for LangChain and FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
import yfinance as yf
import pandas as pd
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Fetch the latest market price, day high/low, volume, and 52-week range for a given ticker.
    Examples: 'AAPL', 'NVDA', '005930.KS' (Samsung Electronics), '^GSPC' (S&P 500).
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    fast_info = stock.fast_info
    
    return {
        "ticker": ticker.upper(),
        "current_price": fast_info.last_price,
        "previous_close": fast_info.previous_close,
        "day_high": fast_info.day_high,
        "day_low": fast_info.day_low,
        "currency": info.get("currency", "USD"),
        "market_cap": info.get("marketCap"),
        "52_week_high": fast_info.year_high,
        "52_week_low": fast_info.year_low,
        "exchange": info.get("exchange", "Unknown")
    }

@tool
def get_historical_ohlcv(ticker: str, period: str = "1mo", interval: str = "1d") -> List[Dict[str, Any]]:
    """
    Fetch historical Open, High, Low, Close, Volume (OHLCV) candles.
    :param ticker: Stock symbol (e.g. 'MSFT', '000660.KS')
    :param period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max')
    :param interval: Candle interval ('1m', '5m', '15m', '1h', '1d', '1wk')
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        return []
    
    df = df.reset_index()
    records = []
    for _, row in df.tail(30).iterrows():
        records.append({
            "date": str(row["Date"])[:10],
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"])
        })
    return records

@tool
def get_financial_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch valuation and financial health indicators (P/E ratio, P/B ratio, ROE, Margins, Dividends).
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        "ticker": ticker.upper(),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "trailing_eps": info.get("trailingEps"),
        "roe": info.get("returnOnEquity"),
        "profit_margins": info.get("profitMargins"),
        "dividend_yield": info.get("dividendYield"),
        "total_revenue": info.get("totalRevenue"),
        "free_cashflow": info.get("freeCashflow"),
        "analyst_target_mean": info.get("targetMeanPrice")
    }
```

---

## 🤖 ReAct Agent Prompt Integration

```yaml
system_prompt_instruction: |
  You have access to real-time market data through `get_stock_price`, `get_historical_ohlcv`, and `get_financial_fundamentals`.
  When a user asks about stock performance, valuation, or comparisons:
  1. Always query the precise ticker using the appropriate exchange suffix (e.g., `.KS` for KOSPI, `.KQ` for KOSDAQ).
  2. Synthesize valuation metrics (P/E, P/B) alongside recent price movements.
  3. Format numbers with proper commas and currency units.
```
