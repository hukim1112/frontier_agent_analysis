---
name: financial-news-sentiment
description: Financial news crawling, RSS aggregation, and LLM-powered market sentiment analysis skill. Extracts headlines, evaluates positive/negative catalysts, analyzes market impact on target stocks, and generates structured morning briefings.
category: finance
tags: [news, sentiment-analysis, market-briefing, rss, catalysts]
dependencies: [feedparser, beautifulsoup4, requests, pandas]
---

# financial-news-sentiment Skill

Financial news intelligence skill that aggregates market news feeds, parses corporate announcements, and scores positive/negative catalysts for target companies and market sectors.

## 🎯 Educational & Practical Use Cases (Practical Focus)
- **Morning Market Briefing Generator**: Summarize overnight global financial developments, Wall Street closes, and KOSPI market opening outlook.
- **Stock-Specific News Sentiment**: Score news headlines (+1.0 to -1.0) and identify bullish/bearish catalysts.
- **Breaking News Event Extraction**: Detect M&A, earnings releases, regulatory actions, and credit rating adjustments.

---

## 🛠️ LangChain & FastMCP Implementation

```python
"""
news_sentiment_tools.py - Financial news scraping and sentiment tools.
"""
from typing import Dict, Any, List
import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# Major Financial RSS Feeds
FINANCIAL_RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex"
}

@tool
def fetch_top_financial_headlines(source_key: str = "yahoo_finance", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch top breaking headlines from major financial news RSS feeds.
    :param source_key: 'yahoo_finance', 'marketwatch_top', or 'reuters_business'
    :param limit: Number of stories to retrieve
    """
    feed_url = FINANCIAL_RSS_FEEDS.get(source_key, FINANCIAL_RSS_FEEDS["yahoo_finance"])
    feed = feedparser.parse(feed_url)
    
    articles = []
    for entry in feed.entries[:limit]:
        summary_clean = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()
        articles.append({
            "title": entry.get("title"),
            "link": entry.get("link"),
            "published": entry.get("published", entry.get("updated", "")),
            "summary": summary_clean[:200]
        })
    return articles

@tool
def analyze_headline_sentiment(headlines: List[str]) -> Dict[str, Any]:
    """
    Rule-based and keyword sentiment scorer for quick headline triage.
    """
    bullish_keywords = ["surge", "jump", "record high", "beat", "rally", "profit", "growth", "upgrade", "bullish", "상승", "호실적", "급등"]
    bearish_keywords = ["plunge", "drop", "miss", "fall", "slump", "loss", "recession", "downgrade", "bearish", "하락", "적자", "급락"]
    
    scored_items = []
    total_score = 0
    
    for h in headlines:
        h_lower = h.lower()
        pos = sum(1 for w in bullish_keywords if w in h_lower)
        neg = sum(1 for w in bearish_keywords if w in h_lower)
        score = pos - neg
        total_score += score
        
        label = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
        scored_items.append({"headline": h, "score": score, "sentiment": label})
        
    avg_score = total_score / len(headlines) if headlines else 0
    overall = "Positive" if avg_score > 0.3 else "Negative" if avg_score < -0.3 else "Neutral"

    return {
        "overall_market_sentiment": overall,
        "average_sentiment_score": round(avg_score, 2),
        "headline_breakdown": scored_items
    }
```

---

## 🤖 ReAct Agent Scenario

```yaml
agent_interaction:
  user_query: "오늘 글로벌 금융시장의 주요 헤드라인을 수집하고 전체적인 시장 센티먼트를 분석해줘."
  agent_steps:
    1. Call `fetch_top_financial_headlines(source_key="yahoo_finance", limit=6)`.
    2. Extract headlines and call `analyze_headline_sentiment(headlines=...)`.
    3. Generate a structured financial market intelligence briefing.
```
