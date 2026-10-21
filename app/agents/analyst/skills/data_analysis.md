---
name: data-analysis
description: Efficient data ingestion, automated statistical profiling, and Pandas analytical query patterns for AI data agents.
---

# Pandas Data Analysis & Statistical Profiling Patterns

This guide provides query execution patterns, automated format loading algorithms, and aggregation snippets used by the `data_profiler` and `data_query` tools.

## Universal Data Ingestion

The ingestion engine dynamically detects file extensions and structured formats:

```python
import json
from pathlib import Path
import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """Auto-detect format and load as a Pandas DataFrame."""
    ext = Path(file_path).suffix.lower()
    
    if ext in ('.csv', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        return pd.read_csv(file_path, sep=sep)
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return pd.DataFrame(v)
            return pd.DataFrame([data])
        raise ValueError("Unsupported JSON layout")
    elif ext == '.jsonl':
        return pd.read_json(file_path, lines=True)
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
```

## Analytical Query Patterns (DuckDB SQL to Pandas Translation)

| Analytical Operation | SQL Equivalent | Pandas Pattern |
|---|---|---|
| **Multi-Metric Grouping** | `SELECT cat, count(), avg(p) GROUP BY 1` | `df.groupby('cat')['price'].agg(['count', 'mean', 'min', 'max'])` |
| **Filtered Aggregation** | `SELECT count() FILTER (WHERE price > 100k)` | `(df['price'] > 100000).sum()` |
| **Cross-Tabulation** | `PIVOT data ON switch USING count()` | `pd.crosstab(df['connection_type'], df['switch_type'], margins=True)` |
| **Top-N per Category** | `ROW_NUMBER() OVER (PARTITION BY cat)` | `df.groupby('cat').apply(lambda g: g.nlargest(3, 'price')).reset_index(drop=True)` |
| **Column Exclusion** | `SELECT * EXCLUDE (id, created_at)` | `df.drop(columns=['id', 'created_at'])` |
| **Summary Statistics** | `SUMMARIZE dataset` | `df.describe(include='all')` |

## Production Query Examples

### 1. Market Segmentation by Price Ranges

```python
bins = [0, 50000, 100000, 200000, float('inf')]
labels = ['< 50K', '50K-100K', '100K-200K', '200K+']
df['price_tier'] = pd.cut(df['price'], bins=bins, labels=labels, right=False)

result = df.groupby('price_tier', observed=False).agg(
    product_count=('product_name', 'count'),
    avg_price=('price', 'mean'),
    total_reviews=('review_count', 'sum')
).round(0)
```

### 2. Switch Type & Connection Cross-Analysis

```python
result = df.pivot_table(
    index='switch_type',
    columns='connection_type',
    values='price',
    aggfunc=['count', 'mean']
).fillna(0).round(0)
```
