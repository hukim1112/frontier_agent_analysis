---
name: chart-patterns
description: Publication-quality data visualization patterns and templates using Matplotlib, Seaborn, and Plotly with cross-platform CJK font support.
---

# Data Visualization & Chart Patterns

This guide outlines chart selection criteria, curated color palettes, cross-platform font configuration, and production-grade code templates for the `chart_generator` tool.

## Chart Selection Matrix

| Objective / Data Relationship | Recommended Chart | Matplotlib Implementation | Plotly (Interactive HTML) |
|---|---|---|---|
| **Category Comparison** | Horizontal / Vertical Bar | `ax.barh()` / `ax.bar()` | `px.bar()` |
| **Time-Series Trend** | Line with optional area fill | `ax.plot()` + `ax.fill_between()` | `px.line()` |
| **Correlation / Scatter** | Scatter with trendline | `ax.scatter()` | `px.scatter(trendline='ols')` |
| **Composition / Market Share** | Donut / Pie chart | `ax.pie(wedgeprops={'width': 0.6})` | `px.pie(hole=0.5)` |
| **Distribution** | Histogram / Box Plot | `ax.hist()` / `ax.boxplot()` | `px.histogram()` / `px.box()` |
| **Matrix / Cross-Tabulation** | Heatmap | `sns.heatmap()` / `ax.imshow()` | `px.imshow()` |

## Curated Color Palettes

Avoid default generic saturated primaries (plain red/blue/green). Use cohesive, publication-grade palettes:

```python
# Semantic & Brand Colors
PALETTE = {
    'primary': '#4A90E2',    # Calming Slate Blue
    'secondary': '#7B68EE',  # Medium Slate Blue
    'accent': '#FF6B6B',     # Coral Red
    'success': '#27AE60',    # Forest Emerald
    'warning': '#F39C12',    # Warm Amber
    'neutral_dark': '#2C3E50',
    'neutral_light': '#F8F9FA',
    'grid': '#E2E8F0',
}

# Categorical Palette (Up to 10 distinct categories)
CATEGORY_COLORS = [
    '#4A90E2', '#27AE60', '#E74C3C', '#F39C12', '#9B59B6',
    '#1ABC9C', '#34495E', '#E67E22', '#3498DB', '#95A5A6'
]
```

## Cross-Platform Korean & CJK Font Configuration

To prevent glyph missing warnings (`Glyph \N{...} missing from font`):

```python
import platform
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    # Linux / WSL Environment
    available = {f.name for f in fontManager.ttflist}
    if 'NanumGothic' in available:
        plt.rcParams['font.family'] = 'NanumGothic'
    elif 'NanumBarunGothic' in available:
        plt.rcParams['font.family'] = 'NanumBarunGothic'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
```

## Production Code Templates

### 1. Dual-Panel Statistical Overview (Histogram + Bar)

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Price Distribution Histogram
axes[0].hist(df['price'].dropna(), bins=15, color='#4A90E2', edgecolor='white', alpha=0.9)
axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel('Price (KRW)', fontsize=11)
axes[0].set_ylabel('Number of Products', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.4)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: Average Price by Category
grp = df.groupby('connection_type')['price'].mean().sort_values()
bars = axes[1].barh(grp.index, grp.values, color=['#4A90E2', '#E74C3C', '#27AE60'])
axes[1].set_title('Average Price by Connection Type', fontsize=14, fontweight='bold', pad=12)
axes[1].set_xlabel('Average Price (KRW)', fontsize=11)
axes[1].grid(True, axis='x', linestyle='--', alpha=0.4)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

for bar, val in zip(bars, grp.values):
    axes[1].text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
                 f'{val:,.0f} KRW', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
```

### 2. Plotly Interactive Dashboard Chart (HTML Output)

```python
import plotly.express as px

fig = px.bar(
    df,
    x='connection_type',
    y='price',
    color='switch_type',
    title='Keyboard Pricing by Connection & Switch Type',
    template='plotly_white',
    barmode='group'
)

fig.update_layout(
    font_family='Arial, Noto Sans KR, sans-serif',
    title_font_size=16,
    legend_title_text='Switch Type',
    margin=dict(l=40, r=40, t=60, b=40)
)
```
