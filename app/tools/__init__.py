from .common import (
    file_read, file_edit, file_writer, notebook_edit,
    bash_command, grep_search, glob_search, tool_search,
    web_fetch, web_search
)
from .navigator import (
    extract_dom_skeleton, get_page_section,
    verify_selectors, interact_page, take_screenshot,
    browse_web
)
from .plan import (
    enter_plan, exit_plan, task_create, task_list, task_update,
    tools_planning, PLANNING_AND_TASK_TOOLS
)
from .supervisor_tools import invoke_sub_agent, list_sub_agents, get_sub_agent_job_status
from .analyst import (
    data_profiler, data_query, chart_generator,
    file_converter, excel_writer, html_report
)

# 🏭 챗봇용 범용 도구 대통합 바인딩 (Tool Factory Groups)
tools_chatbot = [
    file_read, file_edit, file_writer, notebook_edit,
    bash_command, grep_search, glob_search, tool_search,
    web_fetch, web_search
]

# 🕷️ Scraper용 도구 바인딩: 네비게이팅(5종) + L3(1종) + 코딩/파일탐색(8종)
tools_scraper = [
    # L1 + L2 네비게이팅 도구
    extract_dom_skeleton, get_page_section,
    verify_selectors, interact_page, take_screenshot,
    # L3 browser-use 에이전트
    browse_web,
    # 코딩 및 파일 탐색 도구 (common.py 재사용)
    file_writer, file_read, file_edit,
    grep_search, glob_search,
    bash_command, web_search, web_fetch,
]

# 👑 Supervisor용 도구 바인딩: 계획(5종) + 오케스트레이션(3종) + 범용(6종)
tools_supervisor = [
    # Planning & Task Board
    enter_plan, exit_plan, task_create, task_list, task_update,
    # Sub-Agent Orchestration
    list_sub_agents, invoke_sub_agent, get_sub_agent_job_status,
    # Common File / Search / Shell
    file_read, file_writer, file_edit, grep_search, glob_search, web_search, bash_command,
]

# 🔬 Analyst용 도구 바인딩: 분석(3종) + 출력(3종) + 공통(6종) = 12종
tools_analyst = [
    # 분석 도구
    data_profiler, data_query, chart_generator,
    # 출력 도구
    file_converter, excel_writer, html_report,
    # Common File / Search (common.py 재사용)
    file_read, file_writer, file_edit,
    grep_search, glob_search, bash_command,
]

__all__ = [
    "tools_chatbot", "tools_scraper", "tools_supervisor", "tools_analyst",
    "tools_planning", "PLANNING_AND_TASK_TOOLS",
    "file_read", "file_edit", "file_writer", "notebook_edit",
    "bash_command", "grep_search", "glob_search", "tool_search",
    "web_fetch", "web_search",
    "extract_dom_skeleton", "get_page_section",
    "verify_selectors", "interact_page", "take_screenshot",
    "browse_web",
    "enter_plan", "exit_plan", "task_create", "task_list", "task_update",
    "list_sub_agents", "invoke_sub_agent", "get_sub_agent_job_status",
    "data_profiler", "data_query", "chart_generator",
    "file_converter", "excel_writer", "html_report",
]

