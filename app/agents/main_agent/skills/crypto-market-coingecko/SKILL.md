---
name: crypto-market-coingecko
description: Cryptocurrency and digital asset market intelligence skill using CoinGecko API. Supports real-time token prices, 24h volume, market cap dominance, gas metrics, and DeFi TVL trends.
category: finance
tags: [crypto, bitcoin, ethereum, digital-assets, coingecko, defi, altcoins]
dependencies: [requests, pandas]
---

# crypto-market-coingecko Skill

Digital asset and cryptocurrency data skill powered by the CoinGecko API. Provides real-time pricing, market capitalization rankings, price change intervals, and global crypto market metrics.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Top Digital Assets Overview**: Fetch live prices, 24h changes, and market cap for BTC, ETH, SOL, and stablecoins.
- **Global Market Dominance & Volume**: Query Bitcoin Dominance (BTC.D), Total Market Cap, and 24h trading volume.
- **Crypto-Equity Correlation Analysis**: Contrast Bitcoin and Ethereum price cycles with major stock indices (KOSPI, S&P 500, NASDAQ).

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
coingecko_tools.py - CoinGecko crypto tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List, Optional
import requests
from langchain_core.tools import tool

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

@tool
def get_crypto_prices(coin_ids: str = "bitcoin,ethereum,solana,ripple", vs_currency: str = "usd") -> Dict[str, Any]:
    """
    Get live cryptocurrency prices, 24h market cap, 24h volume, and 24h price change.
    :param coin_ids: Comma-separated CoinGecko coin IDs (e.g. 'bitcoin,ethereum,solana')
    :param vs_currency: Target currency code ('usd', 'krw', 'eur', 'jpy')
    """
    url = f"{COINGECKO_BASE}/simple/price"
    params = {
        "ids": coin_ids,
        "vs_currencies": vs_currency.lower(),
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true"
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        return {"error": f"CoinGecko API error: {resp.status_code}"}
        
    data = resp.json()
    currency = vs_currency.lower()
    
    results = {}
    for coin_id, stats in data.items():
        results[coin_id] = {
            "price": stats.get(currency),
            "market_cap": stats.get(f"{currency}_24h_vol"),
            "change_24h_pct": round(stats.get(f"{currency}_24h_change", 0), 2)
        }
    return results

@tool
def get_crypto_global_market_overview() -> Dict[str, Any]:
    """
    Get global cryptocurrency market metrics (Total Market Cap, 24h Volume, BTC & ETH Dominance).
    """
    url = f"{COINGECKO_BASE}/global"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return {"error": "Failed to fetch global crypto data"}
        
    data = resp.json().get("data", {})
    
    return {
        "active_cryptocurrencies": data.get("active_cryptocurrencies"),
        "total_market_cap_usd": round(data.get("total_market_cap", {}).get("usd", 0)),
        "total_volume_24h_usd": round(data.get("total_volume", {}).get("usd", 0)),
        "btc_market_cap_percentage": round(data.get("market_cap_percentage", {}).get("btc", 0), 2),
        "eth_market_cap_percentage": round(data.get("market_cap_percentage", {}).get("eth", 0), 2)
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "비트코인과 이더리움의 현재 USD/KRW 시세 및 암호화폐 전체 시장 점유율(Dominance)을 알려줘."
  agent_steps:
    1. Call `get_crypto_prices(coin_ids="bitcoin,ethereum", vs_currency="usd")`.
    2. Call `get_crypto_global_market_overview()`.
    3. Synthesize a comprehensive digital asset summary.
```
