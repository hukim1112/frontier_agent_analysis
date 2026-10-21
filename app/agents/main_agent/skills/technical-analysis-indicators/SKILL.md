---
name: technical-analysis-indicators
description: Technical analysis indicator computation and signal generation skill for financial market charts. Computes Moving Averages (SMA/EMA), RSI, MACD, Bollinger Bands, Stochastic Oscillator, ATR, and breakout signals.
category: finance
tags: [technical-analysis, rsi, macd, bollinger-bands, indicators, trading-signals]
dependencies: [pandas, numpy, yfinance, ta]
---

# technical-analysis-indicators Skill

Technical analysis toolkit that computes classic momentum, trend, volatility, and volume indicators to empower AI agents to generate structured charting signals and trend interpretations.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Momentum & Overbought/Oversold Detection**: RSI (Relative Strength Index) & Stochastic Oscillator thresholds.
- **Trend Following & Crossover Strategy**: Fast/Slow EMA crossovers and MACD signal line crossovers.
- **Volatility & Support/Resistance Levels**: Bollinger Bands width and ATR (Average True Range) risk sizing.
- **Automated Chart Technical Summary**: Condense 10+ technical indicators into a coherent bullish/bearish/neutral score.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
ta_indicators_tools.py - Technical analysis tools for LangChain & FastMCP Agents.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.tools import tool

@tool
def calculate_technical_indicators(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Compute essential technical indicators (RSI 14, MACD, Bollinger Bands, 20/50/200 SMA) for a given ticker.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval="1d")
    
    if df.empty or len(df) < 50:
        return {"error": f"Insufficient historical data for {ticker}"}
    
    close = df["Close"]
    
    # 1. Moving Averages
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
    sma_200 = close.rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
    
    # 2. RSI (14 periods)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1])
    
    # 3. MACD (12, 26, 9)
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # 4. Bollinger Bands (20, 2)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    bb_lower = (bb_mid - 2 * bb_std).iloc[-1]
    
    current_price = float(close.iloc[-1])
    
    # Signal Interpretation
    signals = []
    if current_rsi >= 70:
        signals.append("RSI Overbought (>70)")
    elif current_rsi <= 30:
        signals.append("RSI Oversold (<30)")
    else:
        signals.append("RSI Neutral")
        
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        signals.append("MACD Bullish Crossover")
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        signals.append("MACD Bearish Crossover")

    return {
        "ticker": ticker.upper(),
        "latest_close": round(current_price, 2),
        "sma": {
            "sma_20": round(float(sma_20), 2),
            "sma_50": round(float(sma_50), 2) if sma_50 else None,
            "sma_200": round(float(sma_200), 2) if sma_200 else None
        },
        "rsi_14": round(current_rsi, 2),
        "macd": {
            "macd": round(float(macd_line.iloc[-1]), 2),
            "signal": round(float(signal_line.iloc[-1]), 2),
            "histogram": round(float(macd_hist.iloc[-1]), 2)
        },
        "bollinger_bands": {
            "upper": round(float(bb_upper), 2),
            "middle": round(float(bb_mid.iloc[-1]), 2),
            "lower": round(float(bb_lower), 2)
        },
        "active_signals": signals
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "테슬라(TSLA)의 일봉 기준 RSI와 MACD 지표를 확인하고 매수/매도 관점에서 기술적 분석을 해줘."
  agent_steps:
    1. Call `calculate_technical_indicators(ticker="TSLA", period="6mo")`.
    2. Interpret the relationship between Current Price, Bollinger Bands, and RSI.
    3. Generate a structured technical chart report.
```
