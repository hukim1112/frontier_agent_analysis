---
name: alpha-vantage-finance
description: Alpha Vantage financial API integration skill. Supports real-time intraday equity quotes, forex exchange rates (FX), global market news sentiment, and economic indicators.
category: finance
tags: [alpha-vantage, fx, forex, news-sentiment, economic-indicators, stocks]
dependencies: [requests, pandas, pydantic]
---

# alpha-vantage-finance Skill

A multi-asset financial data skill integrating Alpha Vantage for real-time global stock quotes, foreign exchange (FX) rates, news sentiment scores, and economic data series.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Currency & Forex Conversion**: Real-time FX rates (USD/KRW, EUR/USD, JPY/KRW) and intraday exchange rate movements.
- **News & Sentiment Scoring**: Automated categorization of market news articles with bull/bear sentiment scores per ticker.
- **Intraday Market Micro-Structure**: Fetch 1-min / 5-min intervals with volume-weighted metrics.
- **Commodities & Macro Indicators**: Treasury yields, CPI, and Brent/WTI crude oil prices.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
alpha_vantage_tools.py - Alpha Vantage API tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
import os
import requests
from langchain_core.tools import tool

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
BASE_URL = "https://www.alphavantage.co/query"

@tool
def get_forex_rate(from_currency: str, to_currency: str = "KRW") -> Dict[str, Any]:
    """
    Get real-time foreign exchange rate between two currencies.
    Examples: from_currency='USD', to_currency='KRW' or from_currency='EUR', to_currency='USD'.
    """
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    data = resp.json().get("Realtime Currency Exchange Rate", {})
    
    if not data:
        return {"error": "Failed to fetch exchange rate."}
        
    return {
        "from": data.get("1. From_Currency Code"),
        "to": data.get("3. To_Currency Code"),
        "exchange_rate": float(data.get("5. Exchange Rate", 0)),
        "bid_price": float(data.get("8. Bid Price", 0)),
        "ask_price": float(data.get("9. Ask Price", 0)),
        "last_refreshed": data.get("6. Last Refreshed")
    }

@tool
def get_market_news_sentiment(tickers: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch latest market news articles and AI sentiment classification for specific tickers.
    :param tickers: Comma-separated ticker symbols (e.g. 'AAPL,NVDA' or 'CRYPTO:BTC')
    :param limit: Number of top news items to return
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers.upper(),
        "limit": limit,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    feed = resp.json().get("feed", [])
    
    results = []
    for item in feed[:limit]:
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "source": item.get("source"),
            "time_published": item.get("time_published"),
            "overall_sentiment_score": item.get("overall_sentiment_score"),
            "overall_sentiment_label": item.get("overall_sentiment_label"),
            "summary": item.get("summary")
        })
    return results
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "원/달러 환율과 애플(AAPL) 관련 최신 뉴스 감성 점수를 확인해줘."
  agent_steps:
    1. Call `get_forex_rate(from_currency="USD", to_currency="KRW")`.
    2. Call `get_market_news_sentiment(tickers="AAPL", limit=3)`.
    3. Generate a combined foreign exchange and sentiment overview.
```
