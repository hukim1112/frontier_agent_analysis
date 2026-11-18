"""ACI compliance, safe-import, and column-hint tests for AAWS/instructor/app/tools/analyst.py."""

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import ToolMessage

from app.tools import analyst


class AnalystToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="analyst-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv_path = self.root / "sales.csv"
        self.csv_path.write_text(
            "category,value\nalpha,10\nbeta,25\n",
            encoding="utf-8",
        )

    def test_all_analyst_tools_follow_langchain_aci_spec(self):
        tools = [
            analyst.data_profiler,
            analyst.data_query,
            analyst.chart_generator,
            analyst.file_converter,
            analyst.excel_writer,
            analyst.html_report,
        ]
        for tool_obj in tools:
            self.assertTrue(tool_obj.handle_tool_error, f"{tool_obj.name} handle_tool_error")
            self.assertTrue(
                callable(tool_obj.handle_validation_error),
                f"{tool_obj.name} handle_validation_error",
            )
            desc = tool_obj.description or ""
            self.assertNotIn("Args:", desc, f"{tool_obj.name} duplicated Args:")
            self.assertNotIn("Returns:", desc, f"{tool_obj.name} duplicated Returns:")
            self.assertFalse(
                desc.startswith((" ", "\t", "\n")),
                f"{tool_obj.name} description has leading whitespace",
            )

    def test_chart_generator_allows_documented_whitelisted_imports(self):
        plotly_out = self.root / "chart.html"
        res_plotly = analyst.chart_generator.invoke(
            {
                "file_path": str(self.csv_path),
                "chart_code": "import plotly.express as px\nfig = px.bar(df, x='category', y='value', title='비교')",
                "output_path": str(plotly_out),
            }
        )
        self.assertIn("차트 생성 완료", res_plotly)
        self.assertTrue(plotly_out.is_file())

        mpl_out = self.root / "chart.png"
        res_mpl = analyst.chart_generator.invoke(
            {
                "file_path": str(self.csv_path),
                "chart_code": (
                    "from matplotlib.font_manager import fontManager\n"
                    "fig, ax = plt.subplots(figsize=(6, 4))\n"
                    "df.groupby('category')['value'].mean().plot(kind='bar', ax=ax)\n"
                ),
                "output_path": str(mpl_out),
            }
        )
        self.assertIn("차트 생성 완료", res_mpl)
        self.assertTrue(mpl_out.is_file())

    def test_excel_writer_allows_openpyxl_imports(self):
        xlsx_out = self.root / "report.xlsx"
        res = analyst.excel_writer.invoke(
            {
                "data_source": str(self.csv_path),
                "output_path": str(xlsx_out),
                "excel_code": (
                    "from openpyxl.styles import Font\n"
                    "ws['A1'] = 'Category'\n"
                    "ws['A1'].font = Font(bold=True)\n"
                    "ws['B1'] = 'Value'\n"
                    "ws['A2'] = str(df.loc[0, 'category'])\n"
                    "ws['B2'] = int(df.loc[0, 'value'])\n"
                ),
            }
        )
        self.assertIn("Excel 보고서 생성 완료", res)
        self.assertTrue(xlsx_out.is_file())

    def test_data_query_allows_safe_imports_blocks_unsafe_and_hints_columns(self):
        ok = analyst.data_query.invoke(
            {
                "file_path": str(self.csv_path),
                "query_code": "import pandas as pd\nresult = pd.DataFrame({'total': [int(df['value'].sum())]})",
                "output_format": "json",
            }
        )
        parsed = json.loads(ok)
        self.assertEqual(parsed, [{"total": 35}])

        blocked_msg = analyst.data_query.invoke(
            {
                "type": "tool_call",
                "id": "call_unsafe_import",
                "name": analyst.data_query.name,
                "args": {
                    "file_path": str(self.csv_path),
                    "query_code": "import subprocess\nresult = 1",
                },
            }
        )
        self.assertIsInstance(blocked_msg, ToolMessage)
        self.assertEqual(blocked_msg.status, "error")
        self.assertIn("ImportError", str(blocked_msg.content))

        col_err_msg = analyst.data_query.invoke(
            {
                "type": "tool_call",
                "id": "call_missing_col",
                "name": analyst.data_query.name,
                "args": {
                    "file_path": str(self.csv_path),
                    "query_code": "result = df['nonexistent_col']",
                },
            }
        )
        self.assertIsInstance(col_err_msg, ToolMessage)
        self.assertEqual(col_err_msg.status, "error")
        self.assertIn("사용 가능한 df 컬럼", str(col_err_msg.content))
        self.assertIn("category", str(col_err_msg.content))
        self.assertIn("value", str(col_err_msg.content))


if __name__ == "__main__":
    unittest.main()
