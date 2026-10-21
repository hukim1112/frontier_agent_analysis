"""
===============================================================================
[Main Agent Tools] 오케스트레이션 및 계획 도구 통합 바인딩
===============================================================================
Planning(계획 및 태스크 5종) + Orchestration(서브에이전트 위임 3종) + Common(공통 7종) = 15종
===============================================================================
"""

from app.tools.plan import (
    enter_plan, exit_plan, task_create, task_list, task_update,
)
from app.tools.supervisor_tools import (
    list_sub_agents, invoke_sub_agent, get_sub_agent_job_status,
)
from app.tools.common import (
    file_read, file_writer, file_edit, grep_search, glob_search, web_search, bash_command,
)

tools_supervisor = [
    # Planning & Task Board
    enter_plan, exit_plan, task_create, task_list, task_update,
    # Sub-Agent Orchestration
    list_sub_agents, invoke_sub_agent, get_sub_agent_job_status,
    # Common File / Search / Shell
    file_read, file_writer, file_edit, grep_search, glob_search, web_search, bash_command,
]

