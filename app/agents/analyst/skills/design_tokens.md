---
name: design-tokens
description: Modern UI design tokens, responsive layout principles, and interactive HTML report templates using Chart.js, Mermaid.js, and CSS Grid.
---

# Interactive HTML Dashboard Design Tokens & Patterns

This guide provides UI design tokens, layout structures, and complete HTML templates for the `html_report` tool.

## Design Philosophy

1. **Theme-Driven Cohesion**: Modern slate/indigo gradient hero banners, clean off-white card backgrounds, and crisp typographic hierarchy.
2. **Distinctive Typography**: Modern sans-serif stacks (`Noto Sans KR`, `Inter`, system font fallbacks).
3. **Data-Dense Yet Breathable**: KPI summary cards followed by dynamic charts and structured tables.
4. **Self-Contained & Portable**: Embed Chart.js and Mermaid via public CDN to produce standalone `.html` files that render seamlessly in any browser or Chainlit chat embed.

## CSS Design Tokens

```css
:root {
  /* Primary & Accent Palettes */
  --color-primary: #4A90E2;
  --color-primary-hover: #357ABD;
  --color-secondary: #7B68EE;
  --color-accent: #FF6B6B;

  /* Semantic State Colors */
  --color-success: #27AE60;
  --color-warning: #F39C12;
  --color-danger: #E74C3C;
  --color-info: #17A2B8;

  /* Backgrounds & Text */
  --color-bg-page: #F8F9FA;
  --color-bg-card: #FFFFFF;
  --color-text-main: #2C3E50;
  --color-text-muted: #6C757D;
  --color-border: #DEE2E6;

  /* Elevation Shadows */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
```

## Responsive Dashboard Template (Chart.js + Mermaid)

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{TITLE}}</title>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
      background: #F8F9FA;
      color: #2C3E50;
      line-height: 1.6;
    }
    .dashboard { max-width: 1200px; margin: 0 auto; padding: 24px; }
    .header {
      background: linear-gradient(135deg, #4A90E2, #7B68EE);
      color: white; padding: 32px; border-radius: 12px;
      margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .header h1 { font-size: 26px; font-weight: 700; }
    .header p { opacity: 0.9; margin-top: 6px; font-size: 14px; }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px; margin-bottom: 24px;
    }
    .stat-card {
      background: white; border-radius: 8px; padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border-left: 4px solid #4A90E2;
    }
    .stat-card .label { font-size: 13px; color: #6C757D; }
    .stat-card .value { font-size: 26px; font-weight: 700; margin-top: 4px; color: #2C3E50; }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
    .card {
      background: white; border-radius: 8px; padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 24px;
    }
    .card h2 {
      font-size: 18px; font-weight: 600; margin-bottom: 16px;
      padding-bottom: 8px; border-bottom: 2px solid #4A90E2;
    }
    .chart-container { position: relative; height: 300px; }

    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th { background: #4A90E2; color: white; padding: 12px 16px; text-align: left; font-weight: 500; }
    td { padding: 10px 16px; border-bottom: 1px solid #DEE2E6; }
    tr:hover { background: #F1F5F9; }
    tr:nth-child(even) { background: #FAFBFC; }

    @media (max-width: 768px) {
      .grid-2 { grid-template-columns: 1fr; }
      .stats-grid { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
<div class="dashboard">
  <!-- Content injected via Python report_code -->
</div>
</body>
</html>
```
