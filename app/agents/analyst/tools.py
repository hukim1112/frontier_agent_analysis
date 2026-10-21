"""
===============================================================================
[Analyst Agent Tools] 분석·출력 및 공통 도구 통합 바인딩
===============================================================================
분석(3종) + 출력(3종) + 공통(6종) = 12종
===============================================================================
"""

from app.tools.analyst import (
    data_profiler, data_query, chart_generator,
    file_converter, excel_writer, html_report,
)
from app.tools.common import (
    file_read, file_writer, file_edit,
    grep_search, glob_search, bash_command,
)

tools_analyst = [
    # 분석 도구
    data_profiler, data_query, chart_generator,
    # 출력 도구
    file_converter, excel_writer, html_report,
    # Common File / Search (common.py 재사용)
    file_read, file_writer, file_edit,
    grep_search, glob_search, bash_command,
]

