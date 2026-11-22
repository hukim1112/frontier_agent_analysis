import asyncio
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from langchain_core.messages import ToolMessage

import app.utils.workspace as workspace_mod
from app.context import AgentContext
from app.tools.supervisor_tools import (
    _FULL_PROTOCOL_HEADER,
    _PROTOCOL_REMINDER,
    cancel_sub_agent_job,
    get_sub_agent_job_status,
    invoke_sub_agent,
    list_sub_agents,
)


def _make_runtime(thread_id: str = "sup_session_100", agent_name: str = "main_agent"):
    return SimpleNamespace(
        execution_info=SimpleNamespace(thread_id=thread_id),
        context=SimpleNamespace(agent_name=agent_name, user_id="test_u"),
    )


class FakeAsyncAgentClient:
    """In-memory fake server client to verify thread_id isolation, protocol headers, and continuation."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 600.0):
        self.base_url = base_url
        self.timeout = timeout

    sessions: dict[str, list[dict]] = {}
    jobs: dict[str, dict] = {}
    invocations: list[dict] = []
    job_counter: int = 0

    @classmethod
    def reset(cls):
        cls.sessions = {}
        cls.jobs = {}
        cls.invocations = []
        cls.job_counter = 0

    async def get_agents(self):
        return [
            {"name": "scraper", "description": "Scraper agent", "roles": ["main", "sub"]},
            {"name": "analyst", "description": "Analyst agent", "roles": ["main", "sub"]},
            {"name": "main_agent", "description": "Main agent", "roles": ["main"]},
        ]

    async def get_messages(self, session_id: str):
        return list(self.sessions.get(session_id, []))

    async def async_invoke(self, agent_name: str, message: str, thread_id: str = None, **kwargs):
        self.invocations.append(
            {
                "mode": "sync",
                "agent_name": agent_name,
                "message": message,
                "thread_id": thread_id,
            }
        )
        msgs = self.sessions.setdefault(thread_id, [])
        msgs.append({"role": "user", "content": message})
        reply = f"[TASK REPORT]\n- Status: SUCCESS\n- Turn Count: {len(msgs)}"
        msgs.append({"role": "assistant", "content": reply})
        return {"type": "ai", "content": reply}

    async def submit_job(
        self,
        agent_name: str,
        message: str,
        thread_id: str = None,
        callback_agent: str = "main_agent",
        callback_thread_id: str = None,
        **kwargs,
    ):
        FakeAsyncAgentClient.job_counter += 1
        job_id = f"job_test_{FakeAsyncAgentClient.job_counter:04d}"
        self.invocations.append(
            {
                "mode": "background",
                "job_id": job_id,
                "agent_name": agent_name,
                "message": message,
                "thread_id": thread_id,
                "callback_agent": callback_agent,
                "callback_thread_id": callback_thread_id,
            }
        )
        msgs = self.sessions.setdefault(thread_id, [])
        msgs.append({"role": "user", "content": message})
        reply = f"[TASK REPORT]\n- Status: SUCCESS\n- Job: {job_id}"
        msgs.append({"role": "assistant", "content": reply})
        self.jobs[job_id] = {
            "job_id": job_id,
            "agent_name": agent_name,
            "sub_thread_id": thread_id,
            "callback_thread_id": callback_thread_id,
            "parent_thread_id": callback_thread_id,
            "status": "SUCCESS",
            "wakeup_status": "COMPLETED",
            "created_at": "2026-10-14T00:00:00Z",
            "completed_at": "2026-10-14T00:01:00Z",
            "result": reply,
        }
        return {"job_id": job_id, "status": "SUBMITTED", "agent_name": agent_name}

    async def get_job(self, job_id: str, caller_thread_id: Optional[str] = None):
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "error", "message": "not found"}
        caller_tid = (caller_thread_id or "").strip()
        if caller_tid:
            owners = {tid for tid in (job.get("callback_thread_id"), job.get("parent_thread_id")) if tid}
            if owners and caller_tid not in owners:
                return {
                    "status": "error",
                    "status_code": 403,
                    "message": f"403 Forbidden: Job '{job_id}' does not belong to session '{caller_tid}'",
                }
        return job

    async def get_session_jobs(self, thread_id: str):
        return [
            dict(j)
            for j in self.jobs.values()
            if j.get("callback_thread_id") == thread_id or j.get("parent_thread_id") == thread_id
        ]

    async def cancel_job(
        self,
        job_id: str,
        reason: str = None,
        caller_thread_id: Optional[str] = None,
    ):
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "error", "message": f"Job '{job_id}' not found"}
        caller_tid = (caller_thread_id or "").strip()
        if caller_tid:
            owners = {tid for tid in (job.get("callback_thread_id"), job.get("parent_thread_id")) if tid}
            if owners and caller_tid not in owners:
                return {
                    "status": "error",
                    "status_code": 403,
                    "message": f"403 Forbidden: Job '{job_id}' does not belong to session '{caller_tid}'",
                }
        if job.get("status") == "CANCELLED":
            return {
                "job_id": job_id,
                "agent_name": job.get("agent_name"),
                "sub_thread_id": job.get("sub_thread_id"),
                "status": "CANCELLED",
                "cancelled": True,
                "already_finished": True,
                "cancel_reason": job.get("cancel_reason"),
                "message": f"Job '{job_id}' was already cancelled.",
            }
        if job.get("status") not in ("SUBMITTED", "RUNNING"):
            return {
                "job_id": job_id,
                "agent_name": job.get("agent_name"),
                "sub_thread_id": job.get("sub_thread_id"),
                "status": job.get("status"),
                "cancelled": False,
                "already_finished": True,
                "message": f"Job '{job_id}' already finished with status '{job.get('status')}'.",
            }
        job["status"] = "CANCELLED"
        job["cancel_reason"] = reason or "Cancelled by request"
        job["wakeup_status"] = "SKIPPED"
        return {
            "job_id": job_id,
            "agent_name": job.get("agent_name"),
            "sub_thread_id": job.get("sub_thread_id"),
            "status": "CANCELLED",
            "cancelled": True,
            "already_finished": False,
            "cancel_reason": job["cancel_reason"],
            "wakeup_status": "SKIPPED",
        }


class SubAgentSessionIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeAsyncAgentClient.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.users_patch = patch.object(workspace_mod, "USERS_ROOT_DIR", Path(self.tmp.name) / "users")
        self.users_patch.start()
        self.addCleanup(self.users_patch.stop)

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_parallel_calls_generate_unique_sub_thread_ids_and_isolate_memory(self):
        """동일 부모 세션에서 동시에 여러 scraper를 호출할 때 각각 고유한 sub_thread_id가 발급되고 메모리가 격리되는지 검증."""
        runtime = _make_runtime(thread_id="sup_parallel_01")

        res1, res2 = await asyncio.gather(
            invoke_sub_agent.coroutine(
                task_instruction="Scrape site A",
                target_file_list=["artifacts/data/site_a.json"],
                subagent_role="scraper",
                run_in_background=True,
                runtime=runtime,
            ),
            invoke_sub_agent.coroutine(
                task_instruction="Scrape site B",
                target_file_list=["artifacts/data/site_b.json"],
                subagent_role="scraper",
                run_in_background=True,
                runtime=runtime,
            ),
        )

        self.assertIn("[JOB SUBMITTED]", res1)
        self.assertIn("[JOB SUBMITTED]", res2)

        m1 = re.search(r"- Sub-Thread ID: ([^\s]+)", res1)
        m2 = re.search(r"- Sub-Thread ID: ([^\s]+)", res2)
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)

        tid1 = m1.group(1)
        tid2 = m2.group(1)

        self.assertNotEqual(tid1, tid2)
        self.assertTrue(tid1.startswith("sup_parallel_01_scraper_"))
        self.assertTrue(tid2.startswith("sup_parallel_01_scraper_"))

        self.assertEqual(len(FakeAsyncAgentClient.invocations), 2)
        self.assertIn(_FULL_PROTOCOL_HEADER, FakeAsyncAgentClient.invocations[0]["message"])
        self.assertIn(_FULL_PROTOCOL_HEADER, FakeAsyncAgentClient.invocations[1]["message"])

        self.assertEqual(len(FakeAsyncAgentClient.sessions[tid1]), 2)
        self.assertEqual(len(FakeAsyncAgentClient.sessions[tid2]), 2)
        self.assertIn("Scrape site A", FakeAsyncAgentClient.sessions[tid1][0]["content"])
        self.assertNotIn("Scrape site B", FakeAsyncAgentClient.sessions[tid1][0]["content"])

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_continuation_with_sub_thread_id_reuses_session_and_injects_reminder(self):
        """반환된 sub_thread_id를 인자로 전달하여 재호출하면 동일 세션을 이어받고 _PROTOCOL_REMINDER가 주입되는지 검증."""
        runtime = _make_runtime(thread_id="sup_cont_01")

        first_res = await invoke_sub_agent.coroutine(
            task_instruction="Initial scrape of page 1",
            target_file_list=["artifacts/data/quotes_p1.json"],
            subagent_role="scraper",
            run_in_background=False,
            runtime=runtime,
        )
        m = re.search(r"- Sub-Thread ID: ([^\s]+)", first_res)
        self.assertIsNotNone(m)
        sub_tid = m.group(1)

        self.assertIn(_FULL_PROTOCOL_HEADER, FakeAsyncAgentClient.invocations[0]["message"])

        second_res = await invoke_sub_agent.coroutine(
            task_instruction="Continue scraping page 2 using the same selectors",
            target_file_list=["artifacts/data/quotes_p2.json"],
            subagent_role="scraper",
            run_in_background=False,
            sub_thread_id=sub_tid,
            runtime=runtime,
        )

        self.assertIn(f"- Sub-Thread ID: {sub_tid}", second_res)
        self.assertEqual(FakeAsyncAgentClient.invocations[1]["thread_id"], sub_tid)

        self.assertIn(_PROTOCOL_REMINDER, FakeAsyncAgentClient.invocations[1]["message"])
        self.assertNotIn(_FULL_PROTOCOL_HEADER, FakeAsyncAgentClient.invocations[1]["message"])

        self.assertEqual(len(FakeAsyncAgentClient.sessions[sub_tid]), 4)

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_job_status_includes_sub_thread_id(self):
        """get_sub_agent_job_status 조회 시에도 Sub-Thread ID가 포함되어 반환되는지 검증."""
        runtime = _make_runtime(thread_id="sup_job_01")
        submit_res = await invoke_sub_agent.coroutine(
            task_instruction="Analyze dataset",
            target_file_list=["artifacts/data/items.json"],
            subagent_role="analyst",
            run_in_background=True,
            runtime=runtime,
        )
        job_id = re.search(r"- Job ID: ([^\s]+)", submit_res).group(1)
        sub_tid = re.search(r"- Sub-Thread ID: ([^\s]+)", submit_res).group(1)

        status_res = await get_sub_agent_job_status.coroutine(job_id=job_id)
        self.assertIn(f"- Job ID: {job_id}", status_res)
        self.assertIn(f"- Sub-Thread ID: {sub_tid}", status_res)

    async def test_server_background_jobs_checkpointer_isolation_and_wakeup_serialization(self):
        """실제 AsyncSqliteSaver 체크포인터와 server._run_agent_job을 사용하여
        동시 백그라운드 실행 시 하위 에이전트 메모리 격리, 연속 호출 시 메모리 유지,
        그리고 부모 Reactive Wakeup 직렬화가 정상 동작하는지 검증."""
        import aiosqlite
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import app.server as server_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_checkpoints.db"
            conn = await aiosqlite.connect(db_path, check_same_thread=False)
            try:
                checkpointer = AsyncSqliteSaver(conn)
                await checkpointer.setup()

                class MemoryInspectModel(GenericFakeChatModel):
                    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                        human_texts = [
                            m.content for m in messages if getattr(m, "type", "") == "human"
                        ]
                        summary = " | ".join(
                            t.splitlines()[-1] for t in human_texts if t
                        )
                        return ChatResult(
                            generations=[
                                ChatGeneration(
                                    message=AIMessage(
                                        content=f"[TASK REPORT]\n- Seen: {summary}"
                                    )
                                )
                            ]
                        )

                sub_executor = create_agent(
                    model=MemoryInspectModel(messages=iter([])),
                    tools=[],
                    system_prompt="You are a test sub-agent.",
                    checkpointer=checkpointer,
                    context_schema=AgentContext,
                )
                sup_executor = create_agent(
                    model=MemoryInspectModel(messages=iter([])),
                    tools=[],
                    system_prompt="You are a test main agent.",
                    checkpointer=checkpointer,
                    context_schema=AgentContext,
                )

                fake_app = SimpleNamespace(
                    state=SimpleNamespace(
                        agents={"scraper": sub_executor, "main_agent": sup_executor},
                        load_lock=asyncio.Lock(),
                    )
                )

                sup_tid = "sup_real_cp_01"
                sub_tid_a = f"{sup_tid}_scraper_aaaa1111"
                sub_tid_b = f"{sup_tid}_scraper_bbbb2222"

                server_mod.job_store["job_a"] = {
                    "job_id": "job_a",
                    "agent_name": "scraper",
                    "sub_thread_id": sub_tid_a,
                    "status": "SUBMITTED",
                }
                server_mod.job_store["job_b"] = {
                    "job_id": "job_b",
                    "agent_name": "scraper",
                    "sub_thread_id": sub_tid_b,
                    "status": "SUBMITTED",
                }

                input_a = server_mod.JobSubmitInput(
                    message="Header\nInstruction: SECRET_SITE_ALPHA",
                    thread_id=sub_tid_a,
                    callback_agent="main_agent",
                    callback_thread_id=sup_tid,
                    user_id="test_u",
                )
                input_b = server_mod.JobSubmitInput(
                    message="Header\nInstruction: SECRET_SITE_BETA",
                    thread_id=sub_tid_b,
                    callback_agent="main_agent",
                    callback_thread_id=sup_tid,
                    user_id="test_u",
                )

                ctx_a = AgentContext(
                    agent_name="scraper",
                    session_id=sub_tid_a,
                    parent_session_id=sup_tid,
                    user_id="test_u",
                )
                ctx_b = AgentContext(
                    agent_name="scraper",
                    session_id=sub_tid_b,
                    parent_session_id=sup_tid,
                    user_id="test_u",
                )
                cfg_a = {"configurable": {"thread_id": sub_tid_a, "user_id": "test_u"}}
                cfg_b = {"configurable": {"thread_id": sub_tid_b, "user_id": "test_u"}}

                await asyncio.gather(
                    server_mod._run_agent_job("job_a", "scraper", input_a, ctx_a, cfg_a, fake_app),
                    server_mod._run_agent_job("job_b", "scraper", input_b, ctx_b, cfg_b, fake_app),
                )

                self.assertEqual(server_mod.job_store["job_a"]["status"], "SUCCESS")
                self.assertEqual(server_mod.job_store["job_b"]["status"], "SUCCESS")

                state_a = await sub_executor.aget_state({"configurable": {"thread_id": sub_tid_a}})
                state_b = await sub_executor.aget_state({"configurable": {"thread_id": sub_tid_b}})

                msgs_a = [m.content for m in state_a.values["messages"]]
                msgs_b = [m.content for m in state_b.values["messages"]]

                self.assertEqual(len(msgs_a), 2)
                self.assertEqual(len(msgs_b), 2)
                self.assertTrue(any("SECRET_SITE_ALPHA" in c for c in msgs_a))
                self.assertFalse(any("SECRET_SITE_BETA" in c for c in msgs_a))
                self.assertTrue(any("SECRET_SITE_BETA" in c for c in msgs_b))
                self.assertFalse(any("SECRET_SITE_ALPHA" in c for c in msgs_b))

                server_mod.job_store["job_a_followup"] = {
                    "job_id": "job_a_followup",
                    "agent_name": "scraper",
                    "sub_thread_id": sub_tid_a,
                    "status": "SUBMITTED",
                }
                input_a_followup = server_mod.JobSubmitInput(
                    message="Reminder\nInstruction: FOLLOWUP_ALPHA_PAGE_2",
                    thread_id=sub_tid_a,
                    callback_agent="main_agent",
                    callback_thread_id=sup_tid,
                    user_id="test_u",
                )
                ctx_a_followup = AgentContext(
                    agent_name="scraper",
                    session_id=sub_tid_a,
                    parent_session_id=sup_tid,
                    user_id="test_u",
                )
                cfg_a_followup = {"configurable": {"thread_id": sub_tid_a, "user_id": "test_u"}}
                await server_mod._run_agent_job(
                    "job_a_followup", "scraper", input_a_followup, ctx_a_followup, cfg_a_followup, fake_app
                )

                state_a_after = await sub_executor.aget_state({"configurable": {"thread_id": sub_tid_a}})
                msgs_a_after = [m.content for m in state_a_after.values["messages"]]
                self.assertEqual(len(msgs_a_after), 4)
                self.assertIn("SECRET_SITE_ALPHA", msgs_a_after[-1])
                self.assertIn("FOLLOWUP_ALPHA_PAGE_2", msgs_a_after[-1])

                sup_state = await sup_executor.aget_state({"configurable": {"thread_id": sup_tid}})
                sup_msgs = [m.content for m in sup_state.values["messages"]]
                self.assertEqual(len(sup_msgs), 6)
                self.assertTrue(any(f"- Sub-Thread ID: {sub_tid_a}" in c for c in sup_msgs))
                self.assertTrue(any(f"- Sub-Thread ID: {sub_tid_b}" in c for c in sup_msgs))
            finally:
                await conn.close()
                for k in ("job_a", "job_b", "job_a_followup"):
                    server_mod.job_store.pop(k, None)

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_supervisor_tools_aci_spec_and_error_handling(self):
        for tool_obj in (list_sub_agents, invoke_sub_agent, get_sub_agent_job_status, cancel_sub_agent_job):
            self.assertTrue(tool_obj.handle_tool_error, f"{tool_obj.name} handle_tool_error")
            self.assertTrue(
                callable(tool_obj.handle_validation_error),
                f"{tool_obj.name} handle_validation_error",
            )
            desc = tool_obj.description or ""
            self.assertNotIn("Returns:", desc, f"{tool_obj.name} duplicated Returns:")
            self.assertFalse(
                desc.startswith((" ", "\t", "\n")),
                f"{tool_obj.name} description has leading whitespace",
            )

        unknown_job_msg = await get_sub_agent_job_status.ainvoke(
            {
                "type": "tool_call",
                "id": "call_unknown_job",
                "name": get_sub_agent_job_status.name,
                "args": {"job_id": "job_does_not_exist"},
            }
        )
        self.assertIsInstance(unknown_job_msg, ToolMessage)
        self.assertEqual(unknown_job_msg.status, "error")
        self.assertIn("[ERROR:", str(unknown_job_msg.content))

        no_rt_list_msg = await get_sub_agent_job_status.ainvoke(
            {
                "type": "tool_call",
                "id": "call_no_runtime",
                "name": get_sub_agent_job_status.name,
                "args": {},
            }
        )
        self.assertIsInstance(no_rt_list_msg, ToolMessage)
        self.assertEqual(no_rt_list_msg.status, "error")
        self.assertIn("[ERROR:", str(no_rt_list_msg.content))

        unknown_cancel_msg = await cancel_sub_agent_job.ainvoke(
            {
                "type": "tool_call",
                "id": "call_unknown_cancel",
                "name": cancel_sub_agent_job.name,
                "args": {"job_id": "job_does_not_exist"},
            }
        )
        self.assertIsInstance(unknown_cancel_msg, ToolMessage)
        self.assertEqual(unknown_cancel_msg.status, "error")
        self.assertIn("[ERROR:", str(unknown_cancel_msg.content))

        empty_instr = await invoke_sub_agent.coroutine(task_instruction="   ")
        self.assertTrue(empty_instr.startswith("[BLOCKER:"))

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_job_status_lists_session_jobs_when_job_id_omitted_and_renders_progress(self):
        """job_id 생략 시 현재 세션의 백그라운드 작업 목록과 AgentLogTracer 텔레메트리를 반환하는지 검증."""
        from app.middleware.observability import AgentLogTracer

        runtime = _make_runtime(thread_id="sup_list_01")
        res1 = await invoke_sub_agent.coroutine(
            task_instruction="Scrape site",
            subagent_role="scraper",
            run_in_background=True,
            runtime=runtime,
        )
        res2 = await invoke_sub_agent.coroutine(
            task_instruction="Analyze site",
            subagent_role="analyst",
            run_in_background=True,
            runtime=runtime,
        )
        job_id1 = re.search(r"- Job ID: ([^\s]+)", res1).group(1)
        job_id2 = re.search(r"- Job ID: ([^\s]+)", res2).group(1)
        sub_tid1 = re.search(r"- Sub-Thread ID: ([^\s]+)", res1).group(1)

        FakeAsyncAgentClient.jobs[job_id1]["status"] = "RUNNING"
        FakeAsyncAgentClient.jobs[job_id1]["wakeup_status"] = "PENDING"

        tracer = AgentLogTracer(disk_enabled=False)
        tracer.start_turn(sub_tid1, "Scrape site")
        tracer.record_tool_call(
            "extract_dom_skeleton",
            {"url": "https://example.com"},
            duration_ms=120,
            success=True,
            session_id=sub_tid1,
        )

        list_out = await get_sub_agent_job_status.coroutine(runtime=runtime)
        self.assertIn("[SESSION JOBS: sup_list_01]", list_out)
        self.assertIn(job_id1, list_out)
        self.assertIn(job_id2, list_out)
        self.assertIn("phase=starting", list_out)
        self.assertIn("Tool calls=1", list_out)

        tracer.end_turn(final_response="done", session_id=sub_tid1)
        FakeAsyncAgentClient.jobs[job_id1]["status"] = "SUCCESS"
        FakeAsyncAgentClient.jobs[job_id1]["wakeup_status"] = "COMPLETED"

        detail_out = await get_sub_agent_job_status.coroutine(job_id=job_id1, runtime=runtime)
        self.assertIn("[Live Execution Progress]", detail_out)
        self.assertIn("extract_dom_skeleton", detail_out)
        self.assertIn("[Sub-Agent Final Report]", detail_out)

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_cancel_sub_agent_job_and_cancelled_sub_thread_id_reuse_blocked(self):
        """cancel_sub_agent_job 호출 및 취소된 sub_thread_id 재사용 차단을 검증."""
        runtime = _make_runtime(thread_id="sup_cancel_01")
        sub_res = await invoke_sub_agent.coroutine(
            task_instruction="Long crawl",
            subagent_role="scraper",
            run_in_background=True,
            runtime=runtime,
        )
        job_id = re.search(r"- Job ID: ([^\s]+)", sub_res).group(1)
        sub_tid = re.search(r"- Sub-Thread ID: ([^\s]+)", sub_res).group(1)

        FakeAsyncAgentClient.jobs[job_id]["status"] = "RUNNING"
        FakeAsyncAgentClient.jobs[job_id]["wakeup_status"] = "PENDING"

        cancel_out = await cancel_sub_agent_job.coroutine(
            job_id=job_id,
            reason="User changed scope",
            runtime=runtime,
        )
        self.assertIn("[JOB CANCELLED]", cancel_out)
        self.assertIn("User changed scope", cancel_out)
        self.assertEqual(FakeAsyncAgentClient.jobs[job_id]["status"], "CANCELLED")
        self.assertEqual(FakeAsyncAgentClient.jobs[job_id]["wakeup_status"], "SKIPPED")

        reuse_out = await invoke_sub_agent.coroutine(
            task_instruction="Resume cancelled session",
            subagent_role="scraper",
            run_in_background=True,
            sub_thread_id=sub_tid,
            runtime=runtime,
        )
        self.assertIn("[BLOCKER:", reuse_out)
        self.assertIn("Cannot reuse cancelled sub_thread_id", reuse_out)

        idempotent_out = await cancel_sub_agent_job.coroutine(job_id=job_id, runtime=runtime)
        self.assertIn("[JOB CANCELLED]", idempotent_out)

        done_res = await invoke_sub_agent.coroutine(
            task_instruction="Quick task",
            subagent_role="analyst",
            run_in_background=True,
            runtime=runtime,
        )
        done_job_id = re.search(r"- Job ID: ([^\s]+)", done_res).group(1)
        finished_out = await cancel_sub_agent_job.coroutine(job_id=done_job_id, runtime=runtime)
        self.assertIn("[JOB ALREADY FINISHED]", finished_out)

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_cancel_sub_agent_job_allowed_in_planning_mode(self):
        """Planning Mode 중에도 get_sub_agent_job_status와 cancel_sub_agent_job은 허용되는지 검증."""
        from app.tools.plan import enter_plan, plan_session_scope

        runtime = _make_runtime(thread_id="sup_plan_cancel")
        FakeAsyncAgentClient.jobs["job_plan_1"] = {
            "job_id": "job_plan_1",
            "agent_name": "scraper",
            "sub_thread_id": "sup_plan_cancel_scraper_1111",
            "callback_thread_id": "sup_plan_cancel",
            "status": "RUNNING",
            "wakeup_status": "PENDING",
        }

        with plan_session_scope("sup_plan_cancel", user_id="test_u"):
            enter_plan.invoke({"intent": "new", "goal": "Cancel old job while revising plan"})
            status_out = await get_sub_agent_job_status.coroutine(runtime=runtime)
            self.assertNotIn("[PLANNING MODE GUARDRAIL]", status_out)
            self.assertIn("job_plan_1", status_out)

            cancel_out = await cancel_sub_agent_job.coroutine(
                job_id="job_plan_1",
                reason="Switching strategy",
                runtime=runtime,
            )
            self.assertIn("[JOB CANCELLED]", cancel_out)
            self.assertNotIn("[PLANNING MODE GUARDRAIL]", cancel_out)

            blocked_out = await invoke_sub_agent.coroutine(
                task_instruction="Should be blocked during planning",
                subagent_role="scraper",
                run_in_background=True,
                runtime=runtime,
            )
            self.assertIn("[PLANNING MODE GUARDRAIL]", blocked_out)

    async def test_server_job_cancellation_suppresses_wakeup_cleans_tabs_and_records_tracer(self):
        """server.submit_agent_job -> cancel_job 시 태스크 취소, wakeup 억제, 브라우저 탭 정리, Tracer 기록 및 409 재사용 차단을 검증."""
        from fastapi import HTTPException
        from langchain.agents import create_agent
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool
        import app.server as server_mod
        import app.tools.navigator as nav_mod
        from app.middleware.observability import AgentLogTracer

        entered = asyncio.Event()
        sup_calls = 0
        closed_tabs: list[str] = []

        @tool
        async def slow_crawl(url: str) -> str:
            """취소 테스트용 느린 도구."""
            entered.set()
            await asyncio.sleep(10.0)
            return url

        class CancelTestModel(BaseChatModel):
            is_sup: bool = False

            @property
            def _llm_type(self) -> str:
                return "cancel-test"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                nonlocal sup_calls
                if self.is_sup:
                    sup_calls += 1
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="wakeup"))])
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="Crawling target",
                                tool_calls=[
                                    {
                                        "name": "slow_crawl",
                                        "args": {"url": "https://example.com"},
                                        "id": "call_slow_1",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                return self._generate(messages, stop, run_manager, **kwargs)

        tracer = AgentLogTracer(disk_enabled=False)
        sub_executor = create_agent(
            model=CancelTestModel(is_sup=False),
            tools=[slow_crawl],
            middleware=[tracer],
            context_schema=AgentContext,
        )
        sup_executor = create_agent(
            model=CancelTestModel(is_sup=True),
            tools=[],
            context_schema=AgentContext,
        )
        fake_req = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agents={"scraper": sub_executor, "main_agent": sup_executor},
                    load_lock=asyncio.Lock(),
                )
            )
        )

        async def _fake_close_tabs(session_id: str) -> int:
            closed_tabs.append(session_id)
            return 1

        orig_pm_instance = nav_mod.PlaywrightManager._instance
        created_job_ids: list[str] = []
        try:
            nav_mod.PlaywrightManager._instance = SimpleNamespace(close_session_tabs=_fake_close_tabs)
            with patch.object(
                server_mod,
                "api_list_agents",
                return_value=[
                    {"name": "scraper", "roles": ["main", "sub"]},
                    {"name": "main_agent", "roles": ["main"]},
                ],
            ):
                submitted = await server_mod.submit_agent_job(
                    "scraper",
                    server_mod.JobSubmitInput(
                        message="crawl",
                        parent_thread_id="sup_srv_can",
                        callback_agent="main_agent",
                        callback_thread_id="sup_srv_can",
                        user_id="u_can",
                    ),
                    fake_req,
                )
                job_id = submitted["job_id"]
                sub_tid = submitted["sub_thread_id"]
                created_job_ids.append(job_id)

                self.assertIn(job_id, server_mod.job_tasks)
                await asyncio.wait_for(entered.wait(), timeout=3.0)

                status_before = await server_mod.get_job_status(job_id)
                self.assertEqual(status_before["status"], "RUNNING")
                self.assertEqual(status_before["progress"]["phase"], "running_tool")
                self.assertIn("slow_crawl", status_before["progress_report"])

                cancel_res = await server_mod.cancel_job(
                    job_id,
                    server_mod.JobCancelInput(reason="Stop crawling"),
                )
                self.assertTrue(cancel_res["cancelled"])
                self.assertEqual(cancel_res["status"], "CANCELLED")
                self.assertEqual(cancel_res["wakeup_status"], "SKIPPED")
                self.assertEqual(closed_tabs, [sub_tid])
                self.assertEqual(sup_calls, 0)

                snap = tracer.get_progress_snapshot(sub_tid)
                self.assertEqual(snap["status"], "CANCELLED")
                self.assertEqual(snap["active_tools"], [])

                with self.assertRaises(HTTPException) as ctx:
                    await server_mod.submit_agent_job(
                        "scraper",
                        server_mod.JobSubmitInput(
                            message="retry same cancelled thread",
                            thread_id=sub_tid,
                            parent_thread_id="sup_srv_can",
                            callback_agent="main_agent",
                            callback_thread_id="sup_srv_can",
                            user_id="u_can",
                        ),
                        fake_req,
                    )
                self.assertEqual(ctx.exception.status_code, 409)
        finally:
            nav_mod.PlaywrightManager._instance = orig_pm_instance
            for jid in created_job_ids:
                task = server_mod.job_tasks.pop(jid, None)
                if task and not task.done():
                    task.cancel()
                server_mod.job_store.pop(jid, None)

    async def test_server_failed_job_triggers_reactive_wakeup_and_consecutive_failure_guard(self):
        """실패(FAILED) 시 Reactive Wakeup(텔레메트리 포함) 발화 및 연속 3회 초과 실패 억제/성공 후 리셋 검증."""
        from langchain.agents import create_agent
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        import app.server as server_mod
        from app.middleware.observability import AgentLogTracer

        should_fail = True
        sup_prompts: list[str] = []

        class FailingSubModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "failing-sub"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                if should_fail:
                    raise RuntimeError("boom network 503")
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="recovered ok"))])

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                return self._generate(messages, stop, run_manager, **kwargs)

        class SupSpyModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "sup-spy"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                sup_prompts.append(str(messages[-1].content))
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="handled failure"))])

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                return self._generate(messages, stop, run_manager, **kwargs)

        tracer = AgentLogTracer(disk_enabled=False)
        sub_executor = create_agent(
            model=FailingSubModel(),
            tools=[],
            middleware=[tracer],
            context_schema=AgentContext,
        )
        sup_executor = create_agent(
            model=SupSpyModel(),
            tools=[],
            context_schema=AgentContext,
        )
        fake_app = SimpleNamespace(
            state=SimpleNamespace(
                agents={"scraper": sub_executor, "main_agent": sup_executor},
                load_lock=asyncio.Lock(),
            )
        )

        sup_tid = "sup_fail_guard_01"
        job_ids = [f"job_fail_{i}" for i in range(1, 7)]
        try:
            async def _run_one(jid: str, sub_tid: str):
                server_mod.job_store[jid] = {
                    "job_id": jid,
                    "agent_name": "scraper",
                    "sub_thread_id": sub_tid,
                    "callback_thread_id": sup_tid,
                    "parent_thread_id": sup_tid,
                    "user_id": "test_u",
                    "status": "SUBMITTED",
                }
                inp = server_mod.JobSubmitInput(
                    message=f"task {jid}",
                    thread_id=sub_tid,
                    parent_thread_id=sup_tid,
                    callback_agent="main_agent",
                    callback_thread_id=sup_tid,
                    user_id="test_u",
                )
                ctx = AgentContext(
                    agent_name="scraper",
                    session_id=sub_tid,
                    parent_session_id=sup_tid,
                    user_id="test_u",
                )
                cfg = {"configurable": {"thread_id": sub_tid, "parent_thread_id": sup_tid, "user_id": "test_u"}}
                await server_mod._run_agent_job(jid, "scraper", inp, ctx, cfg, fake_app)

            # 실패 1, 2, 3 -> FAILED Wakeup 정상 발화
            for idx in range(3):
                await _run_one(job_ids[idx], f"{sup_tid}_scraper_{idx}")
                self.assertEqual(server_mod.job_store[job_ids[idx]]["status"], "FAILED")
                self.assertEqual(server_mod.job_store[job_ids[idx]]["wakeup_status"], "COMPLETED")

            self.assertEqual(len(sup_prompts), 3)
            self.assertIn("[SYSTEM NOTIFICATION: BACKGROUND TASK FAILED]", sup_prompts[0])
            self.assertIn("Execution Telemetry:", sup_prompts[0])
            self.assertIn("boom network 503", sup_prompts[0])
            self.assertIn("1/3", sup_prompts[0])
            self.assertIn("3/3", sup_prompts[2])

            # 실패 4 -> 연속 3회 초과로 Wakeup 억제
            await _run_one(job_ids[3], f"{sup_tid}_scraper_3")
            self.assertEqual(server_mod.job_store[job_ids[3]]["status"], "FAILED")
            self.assertEqual(server_mod.job_store[job_ids[3]]["wakeup_status"], "SKIPPED")
            self.assertIn("Suppressed reactive wakeup", server_mod.job_store[job_ids[3]]["wakeup_error"])
            self.assertEqual(len(sup_prompts), 3)

            # 성공 5 -> 연속 실패 카운터 초기화
            should_fail = False
            await _run_one(job_ids[4], f"{sup_tid}_scraper_4")
            self.assertEqual(server_mod.job_store[job_ids[4]]["status"], "SUCCESS")
            self.assertEqual(server_mod.job_store[job_ids[4]]["wakeup_status"], "COMPLETED")
            self.assertEqual(len(sup_prompts), 4)

            # 실패 6 -> 다시 FAILED Wakeup 1/3 로 발화
            should_fail = True
            await _run_one(job_ids[5], f"{sup_tid}_scraper_5")
            self.assertEqual(server_mod.job_store[job_ids[5]]["status"], "FAILED")
            self.assertEqual(server_mod.job_store[job_ids[5]]["wakeup_status"], "COMPLETED")
            self.assertEqual(len(sup_prompts), 5)
            self.assertIn("1/3", sup_prompts[4])
        finally:
            for jid in job_ids:
                server_mod.job_store.pop(jid, None)

    async def test_playwright_manager_close_session_tabs_closes_only_target_session(self):
        """PlaywrightManager.close_session_tabs가 대상 세션 탭만 정리하고 마지막 탭은 about:blank로 유지하는지 검증."""
        from app.tools.navigator import PlaywrightManager

        class FakeContext:
            def __init__(self):
                self.pages = []

        class FakePage:
            def __init__(self, context):
                self.context = context
                self.closed = False
                self.goto_calls = []
                context.pages.append(self)

            def is_closed(self):
                return self.closed

            async def close(self):
                self.closed = True

            async def goto(self, url, **kwargs):
                self.goto_calls.append(url)

        mgr = PlaywrightManager()
        ctx = FakeContext()
        page_a1 = FakePage(ctx)
        page_a2 = FakePage(ctx)
        page_b1 = FakePage(ctx)

        mgr._pages_by_session = {
            "sess_a::default": page_a1,
            "sess_a::detail": page_a2,
            "sess_b::default": page_b1,
        }
        mgr._target_ids_by_session = {
            "sess_a::default": "tid_a1",
            "sess_a::detail": "tid_a2",
            "sess_b::default": "tid_b1",
        }
        mgr._session_locks = {
            "sess_a::default": asyncio.Lock(),
            "sess_a::detail": asyncio.Lock(),
            "sess_b::default": asyncio.Lock(),
        }
        mgr._watched_pages = {page_a1, page_a2, page_b1}

        closed_a = await mgr.close_session_tabs("sess_a")
        self.assertEqual(closed_a, 2)
        self.assertTrue(page_a1.closed)
        self.assertTrue(page_a2.closed)
        self.assertFalse(page_b1.closed)
        self.assertEqual(list(mgr._pages_by_session.keys()), ["sess_b::default"])
        self.assertEqual(list(mgr._target_ids_by_session.keys()), ["sess_b::default"])
        self.assertEqual(list(mgr._session_locks.keys()), ["sess_b::default"])
        self.assertEqual(mgr._watched_pages, {page_b1})

        # 마지막 남은 탭은 닫지 않고 about:blank로 초기화
        closed_b = await mgr.close_session_tabs("sess_b")
        self.assertEqual(closed_b, 1)
        self.assertFalse(page_b1.closed)
        self.assertEqual(page_b1.goto_calls, ["about:blank"])
        self.assertEqual(mgr._pages_by_session, {})
        self.assertEqual(mgr._target_ids_by_session, {})
        self.assertEqual(mgr._session_locks, {})

    async def test_server_cancel_ignores_teardown_exception_and_suppresses_failed_wakeup(self):
        """취소 중 도구가 일반 예외를 던져도 CANCELLED를 유지하고 FAILED Wakeup을 억제하는지 검증."""
        from langchain.agents import create_agent
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.tools import tool
        import app.server as server_mod
        import app.tools.navigator as nav_mod

        entered = asyncio.Event()
        sup_calls = 0

        @tool
        async def teardown_error_tool(url: str) -> str:
            """취소 정리 중 일반 예외를 발생시키는 도구."""
            entered.set()
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                raise RuntimeError("TargetClosedError: page closed during teardown")
            return url

        class TeardownCancelModel(BaseChatModel):
            is_sup: bool = False

            @property
            def _llm_type(self) -> str:
                return "teardown-cancel"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                nonlocal sup_calls
                if self.is_sup:
                    sup_calls += 1
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="wakeup"))])
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="Running tool",
                                tool_calls=[
                                    {
                                        "name": "teardown_error_tool",
                                        "args": {"url": "https://example.com"},
                                        "id": "call_td_1",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                return self._generate(messages, stop, run_manager, **kwargs)

        sub_executor = create_agent(
            model=TeardownCancelModel(is_sup=False),
            tools=[teardown_error_tool],
            context_schema=AgentContext,
        )
        sup_executor = create_agent(
            model=TeardownCancelModel(is_sup=True),
            tools=[],
            context_schema=AgentContext,
        )
        fake_req = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    agents={"scraper": sub_executor, "main_agent": sup_executor},
                    load_lock=asyncio.Lock(),
                )
            )
        )

        async def _fake_close_tabs(_session_id: str) -> int:
            return 1

        orig_pm_instance = nav_mod.PlaywrightManager._instance
        created_job_ids: list[str] = []
        try:
            nav_mod.PlaywrightManager._instance = SimpleNamespace(close_session_tabs=_fake_close_tabs)
            with patch.object(
                server_mod,
                "api_list_agents",
                return_value=[
                    {"name": "scraper", "roles": ["main", "sub"]},
                    {"name": "main_agent", "roles": ["main"]},
                ],
            ):
                submitted = await server_mod.submit_agent_job(
                    "scraper",
                    server_mod.JobSubmitInput(
                        message="crawl",
                        parent_thread_id="sup_srv_td",
                        callback_agent="main_agent",
                        callback_thread_id="sup_srv_td",
                        user_id="u_td",
                    ),
                    fake_req,
                )
                jid = submitted["job_id"]
                created_job_ids.append(jid)
                await asyncio.wait_for(entered.wait(), timeout=3.0)

                cancel_res = await server_mod.cancel_job(
                    jid,
                    server_mod.JobCancelInput(reason="User cancelled during teardown"),
                )
                self.assertTrue(cancel_res["cancelled"])
                self.assertEqual(cancel_res["status"], "CANCELLED")
                self.assertEqual(server_mod.job_store[jid]["status"], "CANCELLED")
                self.assertEqual(server_mod.job_store[jid]["wakeup_status"], "SKIPPED")
                self.assertIsNone(server_mod.job_store[jid].get("error"))
                self.assertEqual(sup_calls, 0)
        finally:
            nav_mod.PlaywrightManager._instance = orig_pm_instance
            for jid in created_job_ids:
                task = server_mod.job_tasks.pop(jid, None)
                if task and not task.done():
                    task.cancel()
                server_mod.job_store.pop(jid, None)

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_cross_session_job_status_and_cancel_blocked(self):
        """다른 세션의 job_id 조회/취소가 도구 및 서버(403) 양쪽에서 차단되는지 검증."""
        from fastapi import HTTPException
        from langchain_core.tools import ToolException
        import app.server as server_mod

        runtime_a = _make_runtime("session_owner_a")
        runtime_b = _make_runtime("session_other_b")

        FakeAsyncAgentClient.jobs["job_own_1"] = {
            "job_id": "job_own_1",
            "agent_name": "scraper",
            "sub_thread_id": "session_owner_a_scraper_1111",
            "callback_thread_id": "session_owner_a",
            "parent_thread_id": "session_owner_a",
            "status": "RUNNING",
            "wakeup_status": "PENDING",
        }

        with self.assertRaises(ToolException):
            await get_sub_agent_job_status.coroutine(job_id="job_own_1", runtime=runtime_b)
        with self.assertRaises(ToolException):
            await cancel_sub_agent_job.coroutine(job_id="job_own_1", reason="steal", runtime=runtime_b)

        compat_msg = await get_sub_agent_job_status.ainvoke(
            {
                "type": "tool_call",
                "id": "call_cross_status",
                "name": get_sub_agent_job_status.name,
                "args": {"job_id": "job_own_1"},
            }
        )
        # runtime 미전달(ainvoke 직접) 시에는 하위 호환 조회 가능
        self.assertNotEqual(compat_msg.status, "error")

        # 서버가 검증을 건너뛰더라도 도구 자체의 2차 소유권 가드가 차단하는지 확인
        with patch.object(
            FakeAsyncAgentClient,
            "get_job",
            return_value=dict(FakeAsyncAgentClient.jobs["job_own_1"]),
        ):
            with self.assertRaises(ToolException) as ctx_tool:
                await get_sub_agent_job_status.coroutine(job_id="job_own_1", runtime=runtime_b)
            self.assertIn("Access denied", str(ctx_tool.exception))

        ok_status = await get_sub_agent_job_status.coroutine(job_id="job_own_1", runtime=runtime_a)
        self.assertIn("job_own_1", ok_status)
        ok_cancel = await cancel_sub_agent_job.coroutine(job_id="job_own_1", reason="owner cancel", runtime=runtime_a)
        self.assertIn("[JOB CANCELLED]", ok_cancel)

        # 서버 엔드포인트 직접 검증 (403 Forbidden 및 소유자 미지정 CLI Job 일관성)
        srv_jid = "job_srv_own_1"
        cli_jid = "job_srv_cli_no_owner"
        server_mod.job_store[srv_jid] = {
            "job_id": srv_jid,
            "agent_name": "scraper",
            "sub_thread_id": "session_owner_a_scraper_2222",
            "callback_thread_id": "session_owner_a",
            "parent_thread_id": "session_owner_a",
            "user_id": "test_u",
            "status": "RUNNING",
        }
        server_mod.job_store[cli_jid] = {
            "job_id": cli_jid,
            "agent_name": "scraper",
            "sub_thread_id": "cli_scraper_3333",
            "callback_thread_id": None,
            "parent_thread_id": None,
            "user_id": "test_u",
            "status": "RUNNING",
        }
        try:
            cli_srv = await server_mod.get_job_status(cli_jid, caller_thread_id="session_other_b")
            self.assertEqual(cli_srv["job_id"], cli_jid)

            with self.assertRaises(HTTPException) as ctx_get:
                await server_mod.get_job_status(srv_jid, caller_thread_id="session_other_b")
            self.assertEqual(ctx_get.exception.status_code, 403)

            with self.assertRaises(HTTPException) as ctx_can:
                await server_mod.cancel_job(
                    srv_jid,
                    server_mod.JobCancelInput(reason="steal", caller_thread_id="session_other_b"),
                )
            self.assertEqual(ctx_can.exception.status_code, 403)
            self.assertEqual(server_mod.job_store[srv_jid]["status"], "RUNNING")

            ok_srv = await server_mod.get_job_status(srv_jid, caller_thread_id="session_owner_a")
            self.assertEqual(ok_srv["job_id"], srv_jid)
            ok_srv_can = await server_mod.cancel_job(
                srv_jid,
                server_mod.JobCancelInput(reason="owner", caller_thread_id="session_owner_a"),
            )
            self.assertTrue(ok_srv_can["cancelled"])
        finally:
            server_mod.job_store.pop(srv_jid, None)
            server_mod.job_store.pop(cli_jid, None)

    async def test_same_sub_thread_id_progress_isolated_across_sequential_jobs(self):
        """동일 sub_thread_id 연속/대기 실행 시 각 Job이 자신의 실행 스냅샷만 소유하는지 검증."""
        import app.server as server_mod
        from app.middleware.observability import AgentLogTracer

        tracer = AgentLogTracer(disk_enabled=False)
        sup_tid = "sup_seq_iso_01"
        sub_tid = f"{sup_tid}_scraper_1111"

        tracer.start_turn(sub_tid, "turn 1")
        t1_ts = tracer._active_runs[sub_tid]["start_time"]

        j1_id = "job_seq_1"
        j2_id = "job_seq_2"
        server_mod.job_store[j1_id] = {
            "job_id": j1_id,
            "agent_name": "scraper",
            "sub_thread_id": sub_tid,
            "callback_thread_id": sup_tid,
            "parent_thread_id": sup_tid,
            "user_id": "test_u",
            "status": "RUNNING",
            "created_at_ts": t1_ts - 0.1,
            "execution_started_at_ts": t1_ts - 0.01,
        }
        server_mod.job_store[j2_id] = {
            "job_id": j2_id,
            "agent_name": "scraper",
            "sub_thread_id": sub_tid,
            "callback_thread_id": sup_tid,
            "parent_thread_id": sup_tid,
            "user_id": "test_u",
            "status": "RUNNING",
            "created_at_ts": t1_ts + 1.0,
            "execution_started_at_ts": None,
        }
        try:
            # 1) job_1 RUNNING + job_2 락 대기 중: job_1만 턴 1 진행 상황 유지
            e1_running = server_mod._enrich_job_with_progress(server_mod.job_store[j1_id])
            e2_waiting = server_mod._enrich_job_with_progress(server_mod.job_store[j2_id])
            self.assertIn("progress", e1_running)
            self.assertEqual(e1_running["progress"]["status"], "RUNNING")
            self.assertEqual(e1_running["progress"]["turn_id"], 1)
            self.assertNotIn("progress", e2_waiting)

            # 2) 레이지 로딩 경합: job_2가 job_1 락 획득보다 먼저 생성됐더라도 대기 중이면 역전되지 않음
            server_mod.job_store[j2_id]["created_at_ts"] = t1_ts - 0.05
            e1_race = server_mod._enrich_job_with_progress(server_mod.job_store[j1_id])
            e2_race = server_mod._enrich_job_with_progress(server_mod.job_store[j2_id])
            self.assertIn("progress", e1_race)
            self.assertNotIn("progress", e2_race)

            # 3) job_1 SUCCESS + job_2가 락 획득 직후 아직 턴 2 시작 전: job_1은 턴 1 유지, job_2는 비어 있음
            tracer.end_turn(final_response="t1 done", session_id=sub_tid)
            server_mod.job_store[j1_id]["status"] = "SUCCESS"
            server_mod.job_store[j2_id]["created_at_ts"] = t1_ts + 1.0
            server_mod.job_store[j2_id]["execution_started_at_ts"] = t1_ts + 1.0

            e1_done = server_mod._enrich_job_with_progress(server_mod.job_store[j1_id])
            e2_before = server_mod._enrich_job_with_progress(server_mod.job_store[j2_id])
            self.assertIn("progress", e1_done)
            self.assertEqual(e1_done["progress"]["status"], "SUCCESS")
            self.assertEqual(e1_done["progress"]["turn_id"], 1)
            self.assertNotIn("progress", e2_before)

            # 4) job_2가 턴 2 시작: 소유권이 job_2로 전환됨
            tracer.start_turn(sub_tid, "turn 2")
            tracer._active_runs[sub_tid]["start_time"] = t1_ts + 2.0

            e1_during_t2 = server_mod._enrich_job_with_progress(server_mod.job_store[j1_id])
            e2_during_t2 = server_mod._enrich_job_with_progress(server_mod.job_store[j2_id])
            self.assertNotIn("progress", e1_during_t2)
            self.assertIn("progress", e2_during_t2)
            self.assertEqual(e2_during_t2["progress"]["status"], "RUNNING")
            self.assertEqual(e2_during_t2["progress"]["turn_id"], 2)
        finally:
            server_mod.job_store.pop(j1_id, None)
            server_mod.job_store.pop(j2_id, None)

    async def test_session_job_list_skips_disk_reads_for_terminal_jobs_while_detail_uses_disk(self):
        """목록 조회(/jobs, /sessions/{tid}/jobs)는 완료 Job에 디스크 폴백을 타지 않고 상세 조회(/jobs/{id})만 읽는지 검증."""
        import gc
        import app.server as server_mod
        from app.middleware.observability import AgentLogTracer

        sup_tid = "sup_disk_skip_01"
        sub_tid = f"{sup_tid}_analyst_2222"
        ws = workspace_mod.resolve_session_workspace(
            user_id="u_disk_skip",
            session_id=sub_tid,
            parent_session_id=sup_tid,
            subagent_role="analyst",
            ensure_exists=True,
        )

        writer = AgentLogTracer(disk_enabled=True)
        writer.start_turn(sub_tid, "build chart")
        writer._active_runs[sub_tid]["user_id"] = "u_disk_skip"
        writer._active_runs[sub_tid]["parent_session_id"] = sup_tid
        writer._active_runs[sub_tid]["workspace_dir"] = ws.rel_root
        writer.end_turn(final_response="done", session_id=sub_tid)
        writer.flush()
        del writer
        gc.collect()

        jid = "job_disk_term_1"
        server_mod.job_store[jid] = {
            "job_id": jid,
            "agent_name": "analyst",
            "sub_thread_id": sub_tid,
            "callback_thread_id": sup_tid,
            "parent_thread_id": sup_tid,
            "user_id": "u_disk_skip",
            "workspace_dir": ws.rel_root,
            "status": "SUCCESS",
        }
        try:
            with patch.object(
                AgentLogTracer,
                "_build_snapshot_from_disk",
                wraps=AgentLogTracer._build_snapshot_from_disk,
            ) as disk_spy:
                sess_jobs = await server_mod.list_session_jobs(sup_tid)
                all_jobs = await server_mod.list_jobs()
                self.assertEqual(disk_spy.call_count, 0)
                self.assertEqual(len(sess_jobs), 1)
                self.assertNotIn("progress", sess_jobs[0])
                matching_all = [j for j in all_jobs if j.get("job_id") == jid]
                self.assertNotIn("progress", matching_all[0])

                detail = await server_mod.get_job_status(jid)
                self.assertGreaterEqual(disk_spy.call_count, 1)
                self.assertIn("progress", detail)
                self.assertTrue(detail["progress"]["found"])
                self.assertEqual(detail["progress"]["source"], "disk")
        finally:
            server_mod.job_store.pop(jid, None)


if __name__ == "__main__":
    unittest.main()
