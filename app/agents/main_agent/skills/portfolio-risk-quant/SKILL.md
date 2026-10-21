---
name: portfolio-risk-quant
description: Quantitative finance and portfolio risk analytics skill. Implements Modern Portfolio Theory (MPT), Sharpe ratio optimization, Value-at-Risk (VaR / CVaR), Maximum Drawdown (MDD), beta, covariance matrix, and asset allocation rebalancing.
category: finance
tags: [quant, portfolio, risk, sharpe-ratio, var, mdd, asset-allocation, optimization]
dependencies: [numpy, pandas, scipy, yfinance]
---

# portfolio-risk-quant Skill

Quantitative toolkit for evaluating portfolio performance, computing statistical risk measures, and executing Modern Portfolio Theory (MPT) mean-variance optimization.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Portfolio Health & Risk Metrics**: Calculate Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, and Beta against benchmark.
- **Tail Risk Evaluation**: Compute Historical and Parametric Value-at-Risk (VaR 95%, 99%) and Conditional VaR (Expected Shortfall).
- **Drawdown Analysis**: Calculate historical Maximum Drawdown (MDD) and recovery durations.
- **Optimal Weights Allocation**: Find maximum Sharpe ratio or minimum volatility weights given a multi-asset universe.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
portfolio_risk_tools.py - Quantitative portfolio risk analysis tools.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from langchain_core.tools import tool

@tool
def calculate_portfolio_metrics(tickers: List[str], weights: List[float], period: str = "1y", benchmark: str = "SPY") -> Dict[str, Any]:
    """
    Calculate annualized return, annualized volatility, Sharpe ratio, and Maximum Drawdown (MDD) for a given portfolio.
    :param tickers: List of asset symbols (e.g. ['AAPL', 'MSFT', 'NVDA', 'TLT'])
    :param weights: List of portfolio weights summing to 1.0 (e.g. [0.3, 0.3, 0.2, 0.2])
    :param period: Historical period (e.g. '1y', '3y', '5y')
    :param benchmark: Benchmark ticker symbol for Beta calculation
    """
    if len(tickers) != len(weights):
        return {"error": "Tickers and weights must have the same length."}
    
    # Download data
    all_tickers = tickers + [benchmark]
    df = yf.download(all_tickers, period=period, progress=False)["Close"]
    
    if df.empty:
        return {"error": "Failed to fetch price data."}
    
    returns = df.pct_change().dropna()
    port_returns = returns[tickers].dot(weights)
    bench_returns = returns[benchmark]
    
    # Annualized metrics (252 trading days)
    ann_return = float(port_returns.mean() * 252)
    ann_volatility = float(port_returns.std() * np.sqrt(252))
    risk_free_rate = 0.04  # Assumed 4%
    sharpe_ratio = float((ann_return - risk_free_rate) / ann_volatility) if ann_volatility > 0 else 0
    
    # Beta
    covariance = np.cov(port_returns, bench_returns)[0][1]
    bench_variance = np.var(bench_returns)
    beta = float(covariance / bench_variance) if bench_variance > 0 else 1.0
    
    # Maximum Drawdown (MDD)
    cumulative = (1 + port_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())
    
    # VaR 95% (Daily Historical)
    var_95 = float(np.percentile(port_returns, 5))

    return {
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_volatility_pct": round(ann_volatility * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "beta_to_benchmark": round(beta, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "daily_var_95_pct": round(var_95 * 100, 2),
        "weights_applied": dict(zip(tickers, weights))
    }

@tool
def optimize_portfolio_weights(tickers: List[str], period: str = "2y") -> Dict[str, Any]:
    """
    Find optimal portfolio weights that maximize the Sharpe ratio using Mean-Variance Optimization.
    """
    df = yf.download(tickers, period=period, progress=False)["Close"]
    returns = df.pct_change().dropna()
    num_assets = len(tickers)
    
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    rf = 0.04
    
    def neg_sharpe(weights):
        p_ret = np.sum(mean_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - rf) / p_vol
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not opt.success:
        return {"error": "Optimization failed to converge."}
        
    opt_weights = [round(w, 4) for w in opt.x]
    opt_return = float(np.sum(mean_returns * opt.x))
    opt_vol = float(np.sqrt(np.dot(opt.x.T, np.dot(cov_matrix, opt.x))))
    
    return {
        "tickers": tickers,
        "optimal_weights": dict(zip(tickers, opt_weights)),
        "expected_annual_return_pct": round(opt_return * 100, 2),
        "expected_volatility_pct": round(opt_vol * 100, 2),
        "maximized_sharpe_ratio": round((opt_return - rf) / opt_vol, 2)
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "AAPL, MSFT, GOOGL, TLT로 구성된 포트폴리오의 샤프지수를 극대화하는 최적 자산배분 비중을 계산해줘."
  agent_steps:
    1. Call `optimize_portfolio_weights(tickers=["AAPL", "MSFT", "GOOGL", "TLT"], period="2y")`.
    2. Format optimal weights with expected risk-adjusted return insights.
```
