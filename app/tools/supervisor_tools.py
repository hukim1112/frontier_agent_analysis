"""
===============================================================================
[AAWS Tool Module] Supervisor Orchestration Tools (supervisor_tools.py)
===============================================================================
Supervisor 에이전트의 멀티에이전트 오케스트레이션 핵심 도구:

  list_sub_agents: 서버에 등록된 호출 가능한 sub-agent 목록 조회.
    - GET /agents 로 동적 레지스트리 조회
    - invoke_sub_agent 호출 전 확인용도로 활용

  invoke_sub_agent: 전문 sub-agent에게 작업을 위임하는 도구.
    - ToolRuntime으로 supervisor thread_id 자동 추출
    - sub_thread_id = supervisor_tid + "_" + role 로 세션 연속성 보장
    - AsyncAgentClient.get_agents()로 Agent Registry 동적 조회
    - run_in_background=True 지원: 비동기 백그라운드 위임 & Reactive Wakeup
    - 첫 호출: Full Protocol Header 주입 (역할 수립)
    - 이후 호출: 간결한 Protocol Reminder만 주입 (기존 계약 참조)
    - [BLOCKER] 반환으로 Supervisor의 Backtracking 트리거

  get_sub_agent_job_status: 백그라운드 작업의 진행 상태 및 결과 조회 도구.
    - GET /jobs/{job_id} 로 작업 상태(RUNNING, SUCCESS, FAILED) 및 보고서 확인
===============================================================================
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from app.client import AsyncAgentClient

# =============================================================================
# Sub-Agent Protocol 오버레이 메시지 빌더
# =============================================================================

_FULL_PROTOCOL_HEADER = """\
[SUB-AGENT MODE - STRICT PROTOCOL]
You are operating as a sub-agent under a Supervisor. You MUST follow these rules:
1. NO greetings or conversational responses. Execute the task immediately.
2. On ANY blocker (corrupted file / site blocked / selector failure / access denied):
   STOP immediately and return EXACTLY:
   [BLOCKER: <concise reason>]
3. On success: write all results to disk first, then return EXACTLY:
   [TASK REPORT]
   - Status: SUCCESS
   - Target Files: <comma-separated file paths>
   - Artifacts Created: <created file paths>
   - Summary: <1-2 sentence core finding>
   - Issues: None
══════════════════════════════════════════════════════════"""

_PROTOCOL_REMINDER = """\
[CONTINUE SUB-AGENT PROTOCOL]
This is a follow-up task in the same session.
Apply the same strict sub-agent protocol established earlier:
no conversational responses, return [BLOCKER] on failure, [TASK REPORT] on success.
══════════════════════════════════════════════════════════"""


def _build_subagent_message(
    task_instruction: str,
    target_file_list: List[str],
    is_first_call: bool,
) -> str:
    """첫 호출 여부에 따라 적절한 프로토콜 헤더를 포함한 메시지를 구성합니다."""
    header = _FULL_PROTOCOL_HEADER if is_first_call else _PROTOCOL_REMINDER
    file_list_str = ", ".join(target_file_list) if target_file_list else "없음 (필요 시 직접 생성)"

    return (
        f"{header}\n"
        f"Target File List: {file_list_str}\n"
        f"══════════════════════════════════════════════════════════\n"
        f"Instruction: {task_instruction}"
    )


# =============================================================================
# list_sub_agents Tool
# =============================================================================

@tool
async def list_sub_agents() -> str:
    """Returns the list of specialist sub-agents currently available on the server.

    Use this tool to discover which sub-agents can be delegated to before calling
    invoke_sub_agent. The returned list includes each agent's name and description.

    Returns:
        A formatted string listing available sub-agent names and descriptions.
        Example:
            Available sub-agents:
            - scraper: 웹 데이터 수집 전문 에이전트
            - analyst: 데이터 분석 및 리포트 생성 전문 에이전트
    """
    client = AsyncAgentClient(base_url="http://localhost:8000")
    try:
        agents = await client.get_agents()
        if not agents:
            return "No sub-agents are currently registered on the server."

        lines = ["Available sub-agents:"]
        for agent in agents:
            if isinstance(agent, dict):
                name = agent.get("name", "unknown")
                desc = agent.get("description", "No description provided.")
                lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR: Failed to query agent registry — {str(e)}]"


# =============================================================================
# invoke_sub_agent Tool
# =============================================================================

class InvokeSubAgentInput(BaseModel):
    task_instruction: str = Field(
        description=(
            "A clear, actionable step-by-step instruction detailing what the sub-agent "
            "should do: what site to scrape, what data to collect, what files to generate, etc."
        )
    )
    target_file_list: List[str] = Field(
        default_factory=list,
        description=(
            "List of relative file paths the sub-agent should inspect or create "
            "(e.g. ['artifacts/data/quotes.json']). "
            "These define where the sub-agent stores its output artifacts."
        )
    )
    subagent_role: str = Field(
        default="scraper",
        description=(
            "The name of the specialist sub-agent to invoke. "
            "Must match an agent registered on the server (e.g. 'scraper', 'analyst'). "
            "Use get_agents API (internally) to discover available agents."
        )
    )
    run_in_background: bool = Field(
        default=False,
        description=(
            "Set to True for all production tasks (data analysis, chart generation, HTML dashboard, "
            "excel report, web scraping) to run asynchronously in the background. "
            "Even if raw data already exists locally, analysis and dashboard creation execute multiple tools "
            "and MUST use run_in_background=True. When True, returns immediately with Job ID, "
            "and the server will automatically wake up Supervisor when completed."
        )
    )


@tool(args_schema=InvokeSubAgentInput)
async def invoke_sub_agent(
    task_instruction: str,
    target_file_list: List[str] = [],
    subagent_role: str = "scraper",
    run_in_background: bool = False,
    runtime: ToolRuntime = None,
) -> str:
    """Delegates a focused task to a specialized sub-agent running on the server.

    WHEN TO USE:
      - Web scraping / data collection tasks          → subagent_role="scraper"
      - Data analysis / chart / report generation     → subagent_role="analyst"
      Use ONLY for tasks requiring specialized tools not available to Supervisor directly.
      Always check available sub-agents using list_sub_agents() before delegation if unsure.

    PROTOCOL — BEFORE CALLING:
      1. Register a task with task_create() to track this delegation.
      2. Set target_file_list to exact output artifact path(s)
         (e.g. ['artifacts/data/quotes.json']).

    PROTOCOL — BACKGROUND EXECUTION (run_in_background=True) [RECOMMENDED DEFAULT FOR SUB-AGENTS]:
      - MUST be used for all data analysis, chart/dashboard creation, Excel writing, and scraping.
      - Even if data already exists, generating charts & dashboards takes multiple tool steps (>1 min).
      - Returns immediately with [JOB SUBMITTED].
      - Inform the user that the background job has started (with Job ID).
      - When finished, the server will automatically notify and wake you up with full results to continue.

    PROTOCOL — AFTER RECEIVING RESPONSE (SYNCHRONOUS / run_in_background=False):
      - Use ONLY for lightweight text inquiries (e.g. asking agent capabilities).
      - [TASK REPORT] + Status: SUCCESS
          → Call task_update(task_id, 'COMPLETED').
      - [BLOCKER: <reason>]  (sub-agent encountered an unrecoverable issue)
          → Call task_update(task_id, 'BLOCKED').
          → Create a fallback task with task_create() and retry with
            corrected parameters or an alternative target.

    Returns:
        [TASK REPORT] formatted string on synchronous success.
        [JOB SUBMITTED] formatted string on background launch.
        [BLOCKER: reason] string on failure.
    """
    # 1. ToolRuntime으로 Supervisor의 thread_id 추출
    supervisor_tid = "supervisor_default"
    if runtime and runtime.execution_info and runtime.execution_info.thread_id:
        supervisor_tid = runtime.execution_info.thread_id

    # 2. Sub-agent 전용 세션 ID 구성 (연속성 보장)
    sub_thread_id = f"{supervisor_tid}_{subagent_role}"

    client = AsyncAgentClient(base_url="http://localhost:8000", timeout=600.0)

    # 3. Agent Registry 동적 조회 — 유효하지 않은 role 즉시 차단
    try:
        available_agents = await client.get_agents()
        agent_names = [a.get("name") for a in available_agents if isinstance(a, dict)]
        if subagent_role not in agent_names:
            return (
                f"[BLOCKER: Unknown subagent_role '{subagent_role}'. "
                f"Available agents on server: {agent_names}]"
            )
    except Exception as e:
        return f"[BLOCKER: Failed to query agent registry — {str(e)}]"

    # 4. 첫 호출 여부 판단 — 세션 이력 조회
    try:
        prior_messages = await client.get_messages(sub_thread_id)
        is_first_call = len(prior_messages) == 0
    except Exception:
        # 조회 실패 시 안전하게 첫 호출로 가정
        is_first_call = True

    # 5. 3-Layer 메시지 구성 (첫 호출: Full Protocol / 이후: Reminder)
    message = _build_subagent_message(task_instruction, target_file_list, is_first_call)

    # 6. 백그라운드 비동기 실행 모드
    if run_in_background:
        try:
            job_res = await client.submit_job(
                agent_name=subagent_role,
                message=message,
                thread_id=sub_thread_id,
                callback_agent="main_agent",
                callback_thread_id=supervisor_tid,
            )
            if job_res.get("status") == "error":
                return f"[BLOCKER: Failed to submit background job — {job_res.get('message')}]"

            job_id = job_res.get("job_id")
            return (
                f"[JOB SUBMITTED]\n"
                f"- Job ID: {job_id}\n"
                f"- Sub-Agent: {subagent_role}\n"
                f"- Status: RUNNING (Background)\n"
                f"- Tracking: The server will automatically notify and wake you up with full results when this job completes.\n"
                f"Please inform the user that the task has started in the background."
            )
        except Exception as e:
            return f"[BLOCKER: Background job submission exception — {str(e)}]"

    # 7. 일반 동기 실행 모드
    try:
        response = await client.async_invoke(
            agent_name=subagent_role,
            message=message,
            thread_id=sub_thread_id,
        )
        if response.get("type") == "error":
            return f"[BLOCKER: Sub-agent call failed — {response.get('content', 'unknown error')}]"
        return response.get("content", "[BLOCKER: Empty response from sub-agent]")
    except Exception as e:
        return f"[BLOCKER: Sub-agent execution exception — {str(e)}]"


# =============================================================================
# get_sub_agent_job_status Tool
# =============================================================================

class GetJobStatusInput(BaseModel):
    job_id: str = Field(
        description="The Job ID returned by invoke_sub_agent when run_in_background=True (e.g. 'job_a1b2c3d4')."
    )


@tool(args_schema=GetJobStatusInput)
async def get_sub_agent_job_status(job_id: str) -> str:
    """Checks the progress and completion result of a background sub-agent job.

    WHEN TO USE:
      - When the user asks for the status of a background task.
      - To inspect intermediate status or final reports of background jobs.

    Returns:
        Formatted status string containing status (RUNNING, SUCCESS, FAILED) and result/error.
    """
    client = AsyncAgentClient(base_url="http://localhost:8000", timeout=10.0)
    try:
        job = await client.get_job(job_id)
        if not job or job.get("status") == "error":
            return f"[ERROR: Job '{job_id}' not found or query failed — {job.get('message', '')}]"

        status = job.get("status")
        agent_name = job.get("agent_name")
        created_at = job.get("created_at")

        if status == "SUCCESS":
            return (
                f"[JOB STATUS: SUCCESS]\n"
                f"- Job ID: {job_id}\n"
                f"- Sub-Agent: {agent_name}\n"
                f"- Completed At: {job.get('completed_at')}\n"
                f"- Report Result:\n{job.get('result')}"
            )
        elif status == "FAILED":
            return (
                f"[JOB STATUS: FAILED]\n"
                f"- Job ID: {job_id}\n"
                f"- Sub-Agent: {agent_name}\n"
                f"- Error: {job.get('error')}"
            )
        else:
            return (
                f"[JOB STATUS: {status}]\n"
                f"- Job ID: {job_id}\n"
                f"- Sub-Agent: {agent_name}\n"
                f"- Created At: {created_at}\n"
                f"- Status: Currently in progress. Please wait for completion notification."
            )
    except Exception as e:
        return f"[ERROR: Failed to query job status — {str(e)}]"
