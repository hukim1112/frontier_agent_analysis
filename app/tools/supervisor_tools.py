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
    - 기본 호출(sub_thread_id=None): 호출마다 고유 sub_thread_id 자동 발급으로 동시 호출 시 단기 메모리 완벽 격리
    - 연속 호출(sub_thread_id 지정): 이전 반환값의 sub_thread_id를 전달하여 해당 자식의 Checkpointer 세션 연속성 보장
    - AsyncAgentClient.get_agents()로 Agent Registry 동적 조회
    - run_in_background=True 지원: 비동기 백그라운드 위임 & Reactive Wakeup
    - 첫 호출: Full Protocol Header 주입 (역할 수립)
    - 이후 호출: 간결한 Protocol Reminder만 주입 (기존 계약 참조)
    - [BLOCKER] 반환으로 Supervisor의 Backtracking 트리거

  get_sub_agent_job_status: 백그라운드 작업의 진행 상태(실시간 LLM/도구 텔레메트리 포함) 및 결과 조회 도구.
    - GET /jobs/{job_id} 또는 GET /sessions/{thread_id}/jobs 로 진행 상태(SUBMITTED, RUNNING, SUCCESS, FAILED, CANCELLED) 확인
    - job_id 생략 시 현재 세션에 연관된 모든 백그라운드 작업 요약 목록 반환

  cancel_sub_agent_job: 실행 중이거나 대기 중인 백그라운드 서브에이전트 작업 중단 도구.
    - POST /jobs/{job_id}/cancel 로 실행 중인 태스크 취소, 브라우저 탭 정리 및 Reactive Wakeup 억제
===============================================================================
"""

import os
import inspect
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import ToolRuntime


def _format_validation_error(error: Exception) -> str:
    """Formats Pydantic tool validation errors into an actionable error message."""
    details: list[str] = []
    errors_fn = getattr(error, "errors", None)
    if callable(errors_fn):
        try:
            for item in errors_fn():
                loc = ".".join(str(p) for p in item.get("loc", ())) or "(root)"
                details.append(f"{loc}: {item.get('msg', 'invalid value')}")
        except Exception:
            details = []
    if not details:
        details = [str(error)]
    return "[ERROR: Invalid tool arguments — " + "; ".join(details) + "]"

from app.client import AsyncAgentClient
from app.middleware.observability.agent_log_tracer import AgentLogTracer, _render_snapshot_report
from app.tools.plan import check_plan_mode_tool_allowed

# FastAPI 백엔드 기본 URL 및 공유 비동기 클라이언트 (커넥션 풀 재사용)
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
_DefaultAsyncAgentClient = AsyncAgentClient
_shared_client = _DefaultAsyncAgentClient(base_url=FASTAPI_BASE_URL, timeout=600.0)


def get_subagent_client() -> AsyncAgentClient:
    """서브에이전트 통신에 사용되는 공유 AsyncAgentClient 인스턴스를 반환합니다."""
    if AsyncAgentClient is not _DefaultAsyncAgentClient:
        return AsyncAgentClient(base_url=FASTAPI_BASE_URL, timeout=600.0)
    return _shared_client


# 이 도구 모듈을 사용하는 에이전트의 이름 (콜백 대상 자동 지정에 활용)
# agents/<agent_name>.py 의 AGENT_METADATA["name"] 값과 일치해야 합니다.
_CALLER_AGENT_NAME = "main_agent"

# =============================================================================
# Sub-Agent Protocol 오버레이 메시지 빌더
# =============================================================================

_FULL_PROTOCOL_HEADER = """\
[SUB-AGENT MODE - STRICT PROTOCOL]
You are operating as a sub-agent under a Main Agent. You MUST follow these rules:
1. NO greetings or conversational responses. Execute the task immediately.
2. Context & Plan: If a plan is shared, refer to it to understand the broader context and prior outputs.
   Do NOT modify the plan yourself. If adjustments are needed, propose them in your report.
3. Recoverable failures (selector mismatch, delayed rendering, temporary network errors):
   Read a relevant local skill and attempt evidence-based recovery within the task's time/retry budget.
   If no budget was given, attempt at most 2 distinct recovery strategies for the same problem.
   Never repeat the same failed action without new evidence. Recovery must stay within the authorized task.
   STOP when authorization/credentials are missing, human intervention is required, or the budget is exhausted.
   Then return EXACTLY:
   [BLOCKER: <reason; recovery attempts; preserved state/artifacts; required next step>]
4. On success: write all results to disk first, then return EXACTLY:
   [TASK REPORT]
   - Status: SUCCESS
   - Target Files: <comma-separated file paths>
   - Artifacts Created: <created file paths>
   - Summary: <1-2 sentence core finding>
   - Issues: None (or proposed plan adjustments)
══════════════════════════════════════════════════════════"""

_PROTOCOL_REMINDER = """\
[CONTINUE SUB-AGENT PROTOCOL]
This is a follow-up task in the same session.
Apply the same strict sub-agent protocol established earlier:
no conversational responses; try relevant skills for recoverable failures within the task budget
(at most 2 distinct strategies for the same problem if unspecified).
Return [BLOCKER] for missing authorization/credentials, required human intervention, or exhausted recovery;
include previous attempts and preserved state. Do not reset the recovery budget on follow-up calls.
Return [TASK REPORT] on success.
══════════════════════════════════════════════════════════"""


def _build_subagent_message(
    task_instruction: str,
    target_file_list: List[str],
    is_first_call: bool,
    workspace_info: Optional[str] = None,
) -> str:
    """첫 호출 여부에 따라 적절한 프로토콜 헤더 및 전용 워크스페이스 경로를 포함한 메시지를 구성합니다."""
    header = _FULL_PROTOCOL_HEADER if is_first_call else _PROTOCOL_REMINDER
    file_list_str = ", ".join(target_file_list) if target_file_list else "없음 (Sub-Agent Workspace 하위에 직접 생성)"
    ws_section = f"{workspace_info}\n" if workspace_info else ""

    return (
        f"{header}\n"
        f"{ws_section}"
        f"Target File List: {file_list_str}\n"
        f"══════════════════════════════════════════════════════════\n"
        f"Instruction: {task_instruction}"
    )


# =============================================================================
# list_sub_agents Tool
# =============================================================================

SUB_AGENT_ROLE = "sub"
_NON_SUB_AGENT_NAMES = {"supervisor", "main_agent", "chatbot"}


async def _get_sub_agents(client: AsyncAgentClient) -> List[dict]:
    """Agent Registry에서 AGENT_METADATA["roles"]에 'sub'가 포함된 에이전트만 반환합니다."""
    agents = await client.get_agents()
    result = []
    for a in (agents or []):
        if not isinstance(a, dict):
            continue
        roles = a.get("roles")
        if roles is None:
            roles = ["main"] if a.get("name") in _NON_SUB_AGENT_NAMES else ["main", "sub"]
        if SUB_AGENT_ROLE in roles:
            result.append(a)
    return result


@tool
async def list_sub_agents() -> str:
    """Returns the formatted list of specialist sub-agents currently available on the server.

    Use this tool to discover which sub-agents can be delegated to before calling
    invoke_sub_agent. The returned list includes each agent's name and description.
    """
    client = get_subagent_client()
    try:
        agents = await _get_sub_agents(client)
        if not agents:
            return "No sub-agents are currently registered on the server."

        lines = ["Available sub-agents:"]
        for agent in agents:
            name = agent.get("name", "unknown")
            desc = agent.get("description", "No description provided.")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(
            f"[ERROR: Failed to query agent registry at {FASTAPI_BASE_URL} — {str(e)}. Ensure the FastAPI server is running.]"
        ) from e


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
            "(including the Shared Plan File path returned by exit_plan, input data paths, and output artifact paths)."
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
    sub_thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session ID (Sub-Thread ID) of a previously invoked sub-agent to continue its work "
            "with preserved short-term memory (e.g. follow-up instructions or blocker recovery on the same task). "
            "Leave None (default) when starting a new task or invoking multiple sub-agents in parallel "
            "so a fresh, isolated sub_thread_id is automatically generated."
        )
    )


@tool(args_schema=InvokeSubAgentInput)
async def invoke_sub_agent(
    task_instruction: str,
    target_file_list: Optional[List[str]] = None,
    subagent_role: str = "scraper",
    run_in_background: bool = False,
    sub_thread_id: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> str:
    """Delegates a focused task to a specialized sub-agent and returns [TASK REPORT], [JOB SUBMITTED], or [BLOCKER].

    WHEN TO USE:
      - Web scraping / data collection tasks          → subagent_role="scraper"
      - Data analysis / chart / report generation     → subagent_role="analyst"
      Use ONLY for tasks requiring specialized tools not available to Supervisor directly.
      Always check available sub-agents using list_sub_agents() before delegation if unsure.

    SESSION ISOLATION & CONTINUATION (sub_thread_id):
      - NEW TASK / PARALLEL CALLS (sub_thread_id=None, default):
        Omit sub_thread_id to automatically generate a fresh, isolated session ID.
        This guarantees that concurrent or unrelated sub-agent calls never mix short-term memory.
      - FOLLOW-UP / CONTINUATION (sub_thread_id="<Sub-Thread ID from prior result>"):
        Pass the Sub-Thread ID returned by a previous call when you want the exact same sub-agent
        instance to continue its prior work or recover from a [BLOCKER] with its memory preserved.

    PROTOCOL — BEFORE CALLING:
      1. For multi-step tasks, enter_plan(intent='new' or 'revise') -> perform any needed read-only discovery -> exit_plan(plan_name, steps)
         to publish your plan on the Shared Blackboard before delegating.
      2. Register a task with task_create() to track this delegation.
      3. Include the Shared Plan File path (returned by exit_plan) along with target input/output artifact paths
         in target_file_list (e.g. ['artifacts/users/.../plan/plan_state.json', 'artifacts/data/quotes.json'])
         so the sub-agent shares the full context.

    PROTOCOL — EXECUTION MODE (run_in_background):
      - Background mode (run_in_background=True) [RECOMMENDED DEFAULT FOR SUB-AGENTS]:
          * MUST be used for all data analysis, chart/dashboard creation, Excel writing, and scraping.
          * Even if data already exists, generating charts & dashboards takes multiple tool steps (>1 min).
          * Returns immediately with [JOB SUBMITTED] including Job ID and Sub-Thread ID.
          * Inform the user that the background job has started (with Job ID).
          * When finished, the server will automatically notify and wake you up with full results to continue.
      - Synchronous mode (run_in_background=False):
          * Use ONLY for lightweight text inquiries (e.g. asking agent capabilities).

    PROTOCOL — AFTER RECEIVING SUB-AGENT RESULT (BACKGROUND WAKEUP OR SYNCHRONOUS):
      - [TASK REPORT] + Status: SUCCESS
          → Call task_update(task_id, 'COMPLETED').
      - [BLOCKER: <reason>]  (sub-agent encountered an unrecoverable issue)
          → Call task_update(task_id, 'BLOCKED').
          → Inspect the cause, previous recovery attempts, preserved artifacts,
            and remaining budget.
          → When new evidence and budget support an alternative strategy or fallback path,
            backtrack first: call enter_plan(intent='revise', goal=...) -> perform any needed
            read-only discovery -> call exit_plan(plan_name, steps) -> register the fallback
            task with task_create(), then re-delegate (pass the same sub_thread_id if continuing
            the same attempt history; do not reset retry counts on follow-up calls).
          → Report missing authorization/credentials, required human intervention,
            or exhausted recovery to the user with the required next step.
    """
    if not (task_instruction or "").strip():
        return "[BLOCKER: 'task_instruction' cannot be empty. Provide a clear, actionable instruction for the sub-agent.]"

    resolved_files = list(target_file_list or [])

    # 1. ToolRuntime으로 호출 에이전트 정보 추출
    supervisor_tid = "supervisor_default"
    if runtime and runtime.execution_info and runtime.execution_info.thread_id:
        supervisor_tid = runtime.execution_info.thread_id

    caller_agent = _CALLER_AGENT_NAME
    caller_user_id = "default"
    if runtime and hasattr(runtime, "context") and runtime.context:
        caller_agent = getattr(runtime.context, "agent_name", None) or _CALLER_AGENT_NAME
        caller_user_id = getattr(runtime.context, "user_id", None) or "default"

    guard_sid = (
        runtime.execution_info.thread_id
        if (runtime and runtime.execution_info and runtime.execution_info.thread_id)
        else None
    )
    guard_uid = (
        getattr(runtime.context, "user_id", None)
        if (runtime and hasattr(runtime, "context") and runtime.context)
        else None
    )
    block_msg = check_plan_mode_tool_allowed(
        "invoke_sub_agent", session_id=guard_sid, user_id=guard_uid
    )
    if block_msg:
        return block_msg

    client = get_subagent_client()

    # 2. Agent Registry 동적 조회 — sub 역할이 없는 에이전트(supervisor, chatbot 등)는 즉시 차단
    try:
        available_agents = await _get_sub_agents(client)
        agent_names = [a.get("name") for a in available_agents]
        if subagent_role not in agent_names:
            return (
                f"[BLOCKER: Unknown or non-delegatable subagent_role '{subagent_role}'. "
                f"Available sub-agents on server: {agent_names}]"
            )
    except Exception as e:
        return f"[BLOCKER: Failed to query agent registry — {str(e)}]"

    # 3. Sub-agent 전용 세션 ID 구성 및 첫 호출 여부 판단 (취소된 세션 재사용 차단 포함)
    if sub_thread_id and sub_thread_id.strip():
        resolved_sub_tid = sub_thread_id.strip()
        try:
            session_jobs = await client.get_session_jobs(supervisor_tid)
        except Exception:
            session_jobs = []
        if any(
            isinstance(j, dict)
            and j.get("sub_thread_id") == resolved_sub_tid
            and j.get("status") == "CANCELLED"
            for j in (session_jobs or [])
        ):
            return (
                f"[BLOCKER: Cannot reuse cancelled sub_thread_id '{resolved_sub_tid}'. "
                f"A cancelled sub-agent thread may contain incomplete tool-call checkpoints. "
                f"Omit sub_thread_id (leave it None) to start a fresh, isolated sub-agent session.]"
            )
        try:
            prior_messages = await client.get_messages(resolved_sub_tid)
            is_first_call = len(prior_messages) == 0
        except Exception:
            is_first_call = False
    else:
        resolved_sub_tid = f"{supervisor_tid}_{subagent_role}_{uuid.uuid4().hex[:8]}"
        is_first_call = True

    # 4. 서브에이전트 전용 워크스페이스 경로 해석 및 3-Layer 메시지 구성
    workspace_info = None
    try:
        from app.utils.workspace import resolve_session_workspace

        sub_ws = resolve_session_workspace(
            user_id=caller_user_id,
            session_id=resolved_sub_tid,
            parent_session_id=supervisor_tid,
            subagent_role=subagent_role,
            ensure_exists=True,
        )
        workspace_info = (
            f"Sub-Agent Dedicated Workspace: {sub_ws.rel_root}\n"
            f"  - Plan Dir: {sub_ws.rel_plan_dir}\n"
            f"  - Scripts Dir: {sub_ws.rel_scripts_dir}\n"
            f"  - Outputs Dir: {sub_ws.rel_outputs_dir}"
        )
    except Exception as ws_err:
        import logging
        logging.getLogger(__name__).warning(
            f"[invoke_sub_agent] Failed to resolve sub-agent workspace for '{resolved_sub_tid}': {ws_err}"
        )
        workspace_info = None

    message = _build_subagent_message(
        task_instruction,
        resolved_files,
        is_first_call,
        workspace_info=workspace_info,
    )

    # 5. 백그라운드 비동기 실행 모드
    if run_in_background:
        try:
            job_res = await client.submit_job(
                agent_name=subagent_role,
                message=message,
                thread_id=resolved_sub_tid,
                callback_agent=caller_agent,
                callback_thread_id=supervisor_tid,
                user_id=caller_user_id,
                parent_thread_id=supervisor_tid,
            )
            if job_res.get("status") == "error":
                return f"[BLOCKER: Failed to submit background job — {job_res.get('message')}]"

            job_id = job_res.get("job_id")
            return (
                f"[JOB SUBMITTED]\n"
                f"- Job ID: {job_id}\n"
                f"- Sub-Agent: {subagent_role}\n"
                f"- Sub-Thread ID: {resolved_sub_tid} (use sub_thread_id='{resolved_sub_tid}' to continue this agent)\n"
                f"- Status: RUNNING (Background)\n"
                f"- Tracking: The server will automatically notify and wake you up with full results when this job completes.\n"
                f"Please inform the user that the task has started in the background."
            )
        except Exception as e:
            return f"[BLOCKER: Background job submission exception — {str(e)}]"

    # 6. 일반 동기 실행 모드
    try:
        response = await client.async_invoke(
            agent_name=subagent_role,
            message=message,
            thread_id=resolved_sub_tid,
            user_id=caller_user_id,
            parent_thread_id=supervisor_tid,
        )
        if response.get("type") == "error":
            return f"[BLOCKER: Sub-agent call failed — {response.get('content', 'unknown error')}]"
        content = response.get("content", "[BLOCKER: Empty response from sub-agent]")
        return (
            f"{content}\n"
            f"- Sub-Thread ID: {resolved_sub_tid} (use sub_thread_id='{resolved_sub_tid}' to continue this agent)"
        )
    except Exception as e:
        return f"[BLOCKER: Sub-agent execution exception — {str(e)}]"


# =============================================================================
# get_sub_agent_job_status & cancel_sub_agent_job Tools
# =============================================================================

def _extract_caller_thread_id(runtime: ToolRuntime = None) -> Optional[str]:
    if runtime and getattr(runtime, "execution_info", None) and runtime.execution_info.thread_id:
        tid = str(runtime.execution_info.thread_id).strip()
        return tid or None
    return None


class GetJobStatusInput(BaseModel):
    job_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional Job ID returned by invoke_sub_agent when run_in_background=True "
            "(e.g. 'job_a1b2c3d4'). Omit to list all background jobs in the current session."
        ),
    )


@tool(args_schema=GetJobStatusInput)
async def get_sub_agent_job_status(
    job_id: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> str:
    """Checks the progress and completion result (SUBMITTED, RUNNING, SUCCESS, FAILED, CANCELLED) of background sub-agent jobs.

    WHEN TO USE:
      - Pass job_id to inspect live execution phase/tools/telemetry and final reports for a specific job.
      - Omit job_id to list all background jobs and their live summaries in the current session.
    """
    client = get_subagent_client()
    clean_job_id = (job_id or "").strip()
    caller_tid = _extract_caller_thread_id(runtime)

    try:
        if not clean_job_id:
            supervisor_tid = caller_tid
            if not supervisor_tid:
                raise ToolException(
                    "[ERROR: 'job_id' cannot be empty when no active session context is available. "
                    "Pass the Job ID returned by invoke_sub_agent.]"
                )
            jobs = await client.get_session_jobs(supervisor_tid)
            if not jobs:
                return (
                    f"[SESSION JOBS: {supervisor_tid}]\n"
                    "No background sub-agent jobs found for this session."
                )

            lines = [f"[SESSION JOBS: {supervisor_tid}] ({len(jobs)} job(s))"]
            for item in jobs:
                jid = item.get("job_id", "unknown")
                role = item.get("agent_name", "unknown")
                st = item.get("status", "UNKNOWN")
                sub_tid = item.get("sub_thread_id") or ""
                wu = item.get("wakeup_status") or ""
                summary = f"- Job ID: {jid} | Sub-Agent: {role} | Status: {st}"
                if wu:
                    summary += f" | Wakeup: {wu}"
                if sub_tid:
                    summary += f" | Sub-Thread ID: {sub_tid}"
                lines.append(summary)

                prog = item.get("progress")
                if not prog and sub_tid and st in ("SUBMITTED", "RUNNING"):
                    parent_sid = item.get("parent_thread_id") or item.get("callback_thread_id") or supervisor_tid
                    snap = AgentLogTracer.get_session_snapshot(
                        session_id=sub_tid,
                        user_id=item.get("user_id"),
                        parent_session_id=parent_sid,
                        workspace_dir=item.get("workspace_dir"),
                        recent_limit=3,
                        allow_disk_fallback=False,
                    )
                    snap_started = snap.get("started_at")
                    job_created_ts = item.get("created_at_ts")
                    stale_by_time = (
                        snap_started is not None
                        and job_created_ts is not None
                        and float(snap_started) < float(job_created_ts) - 0.05
                    )
                    if (
                        snap.get("found")
                        and not stale_by_time
                        and not (st == "SUBMITTED" and snap.get("status") in ("SUCCESS", "FAILED", "CANCELLED"))
                    ):
                        prog = snap
                if isinstance(prog, dict) and prog.get("found"):
                    elapsed_s = prog.get("elapsed_ms", 0) / 1000.0
                    detail = (
                        f"  Progress: phase={prog.get('phase')}, "
                        f"LLM steps={prog.get('llm_step_count', 0)}, "
                        f"Tool calls={prog.get('total_tool_calls', 0)} (failed={prog.get('failed_tool_calls', 0)}), "
                        f"elapsed={elapsed_s:.1f}s"
                    )
                    active_tools = prog.get("active_tools") or []
                    if active_tools:
                        active_names = ", ".join(
                            str(t.get("tool_name", "")) for t in active_tools if t.get("tool_name")
                        )
                        if active_names:
                            detail += f", active_tools=[{active_names}]"
                    lines.append(detail)
                elif st == "CANCELLED" and item.get("cancel_reason"):
                    lines.append(f"  Cancel Reason: {item['cancel_reason']}")
                elif st == "FAILED" and item.get("error"):
                    lines.append(f"  Error: {str(item['error'])[:160]}")
            return "\n".join(lines)

        job = await client.get_job(clean_job_id, caller_thread_id=caller_tid)
        if not job or job.get("status") == "error":
            err_msg = job.get("message", "") if isinstance(job, dict) else ""
            raise ToolException(
                f"[ERROR: Job '{clean_job_id}' not found or query failed — {err_msg}. Verify the Job ID from invoke_sub_agent.]"
            )

        if caller_tid:
            owners = {tid for tid in (job.get("callback_thread_id"), job.get("parent_thread_id")) if tid}
            if owners and caller_tid not in owners:
                raise ToolException(
                    f"[ERROR: Access denied — Job '{clean_job_id}' does not belong to session '{caller_tid}'.]"
                )

        status = job.get("status", "UNKNOWN")
        agent_name = job.get("agent_name", "unknown")
        sub_tid = job.get("sub_thread_id") or ""
        created_at = job.get("created_at", "")
        completed_at = job.get("completed_at") or ("Cancelled" if status == "CANCELLED" else "In Progress")
        wakeup_status = job.get("wakeup_status")

        lines = [
            "[JOB STATUS REPORT]",
            f"- Job ID: {clean_job_id}",
            f"- Sub-Agent: {agent_name}",
        ]
        if sub_tid:
            lines.append(f"- Sub-Thread ID: {sub_tid}")
        lines.extend([
            f"- Status: {status}",
            f"- Started: {created_at}",
            f"- Completed: {completed_at}",
        ])
        if wakeup_status:
            lines.append(f"- Wakeup Status: {wakeup_status}")

        progress_report = (job.get("progress_report") or "").strip()
        if not progress_report and isinstance(job.get("progress"), dict):
            progress_report = _render_snapshot_report(job["progress"]).strip()
        is_active = status in ("SUBMITTED", "RUNNING")
        if not progress_report and sub_tid:
            parent_sid = job.get("parent_thread_id") or job.get("callback_thread_id")
            snap = AgentLogTracer.get_session_snapshot(
                session_id=sub_tid,
                user_id=job.get("user_id"),
                parent_session_id=parent_sid,
                workspace_dir=job.get("workspace_dir"),
                recent_limit=5,
            )
            snap_started = snap.get("started_at")
            job_created_ts = job.get("created_at_ts")
            stale_by_time = (
                is_active
                and snap_started is not None
                and job_created_ts is not None
                and float(snap_started) < float(job_created_ts) - 0.05
            )
            stale_terminal = not is_active and snap.get("status") in ("RUNNING", "INTERRUPTED")
            if (
                snap.get("found")
                and not stale_by_time
                and not stale_terminal
                and not (status == "SUBMITTED" and snap.get("status") in ("SUCCESS", "FAILED", "CANCELLED"))
            ):
                progress_report = _render_snapshot_report(snap).strip()

        if progress_report:
            lines.append(f"\n[Live Execution Progress]\n{progress_report}")

        if status == "SUCCESS":
            lines.append(f"\n[Sub-Agent Final Report]\n{job.get('result', '')}")
            if job.get("supervisor_response"):
                lines.append(f"\n[Supervisor Follow-up Action]\n{job.get('supervisor_response')}")
        elif status == "FAILED":
            lines.append(f"\n[Error Detail]\n{job.get('error', 'Unknown error')}")
            if job.get("supervisor_response"):
                lines.append(f"\n[Supervisor Follow-up Action]\n{job.get('supervisor_response')}")
            if job.get("wakeup_error"):
                lines.append(f"\n[Wakeup Error]\n{job.get('wakeup_error')}")
        elif status == "CANCELLED":
            if job.get("cancel_reason"):
                lines.append(f"- Cancel Reason: {job['cancel_reason']}")
            lines.append(
                "- Note: This job was cancelled (reactive wakeup suppressed). "
                "Do not reuse this Sub-Thread ID; omit sub_thread_id to start a fresh session."
            )

        return "\n".join(lines)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"[ERROR: Exception while checking job status — {str(e)}]") from e


class CancelJobInput(BaseModel):
    job_id: str = Field(
        description="The Job ID of the background sub-agent job to cancel (e.g. 'job_a1b2c3d4')."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Optional concise reason for cancellation.",
    )


@tool(args_schema=CancelJobInput)
async def cancel_sub_agent_job(
    job_id: str,
    reason: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> str:
    """Cancels a queued or running background sub-agent job and cleans up its runtime resources.

    WHEN TO USE:
      - When user goals change, a background job is no longer needed, or live status shows a stall.
      - Stops task execution, closes isolated browser tabs, and suppresses reactive wakeup.
      - Do NOT reuse a cancelled job's sub_thread_id afterward.
    """
    clean_job_id = (job_id or "").strip()
    if not clean_job_id:
        raise ToolException("[ERROR: 'job_id' cannot be empty. Pass the Job ID to cancel.]")

    caller_tid = _extract_caller_thread_id(runtime)
    client = get_subagent_client()
    try:
        res = await client.cancel_job(clean_job_id, reason=reason, caller_thread_id=caller_tid)
        if not isinstance(res, dict) or res.get("status") == "error":
            err_msg = res.get("message", "unknown error") if isinstance(res, dict) else "invalid response"
            raise ToolException(f"[ERROR: Failed to cancel job '{clean_job_id}' — {err_msg}]")

        if res.get("already_finished") and not res.get("cancelled"):
            return (
                f"[JOB ALREADY FINISHED]\n"
                f"- Job ID: {clean_job_id}\n"
                f"- Status: {res.get('status', 'UNKNOWN')}\n"
                f"- Detail: {res.get('message', 'Job already finished before cancellation.')}"
            )

        lines = [
            "[JOB CANCELLED]",
            f"- Job ID: {clean_job_id}",
            f"- Sub-Agent: {res.get('agent_name', 'unknown')}",
        ]
        if res.get("sub_thread_id"):
            lines.append(f"- Sub-Thread ID: {res['sub_thread_id']}")
        lines.append(f"- Status: {res.get('status', 'CANCELLED')}")
        lines.append(f"- Reason: {res.get('cancel_reason') or reason or 'Cancelled by request'}")
        lines.append(
            "- Note: Reactive wakeup is suppressed for this job. "
            "Do not reuse this Sub-Thread ID; omit sub_thread_id to start a fresh session."
        )
        return "\n".join(lines)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"[ERROR: Exception while cancelling job — {str(e)}]") from e


def _finalize_supervisor_tool(t):
    """Supervisor 오케스트레이션 도구 공통 메타데이터(docstring 정제, 에러 핸들링)를 일관 적용합니다."""
    if getattr(t, "description", None):
        t.description = inspect.cleandoc(t.description)
    t.handle_tool_error = True
    t.handle_validation_error = _format_validation_error
    return t


for _sup_tool in (list_sub_agents, invoke_sub_agent, get_sub_agent_job_status, cancel_sub_agent_job):
    _finalize_supervisor_tool(_sup_tool)
