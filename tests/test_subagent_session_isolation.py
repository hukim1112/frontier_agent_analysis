import asyncio
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import ToolMessage

import app.utils.workspace as workspace_mod
from app.context import AgentContext
from app.tools.supervisor_tools import (
    _FULL_PROTOCOL_HEADER,
    _PROTOCOL_REMINDER,
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
            "status": "SUCCESS",
            "created_at": "2026-10-14T00:00:00Z",
            "completed_at": "2026-10-14T00:01:00Z",
            "result": reply,
        }
        return {"job_id": job_id, "status": "SUBMITTED", "agent_name": agent_name}

    async def get_job(self, job_id: str):
        return self.jobs.get(job_id, {"status": "error", "message": "not found"})


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

    @patch("app.tools.supervisor_tools.AsyncAgentClient", FakeAsyncAgentClient)
    async def test_supervisor_tools_aci_spec_and_error_handling(self):
        for tool_obj in (list_sub_agents, invoke_sub_agent, get_sub_agent_job_status):
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

        empty_instr = await invoke_sub_agent.coroutine(task_instruction="   ")
        self.assertTrue(empty_instr.startswith("[BLOCKER:"))


if __name__ == "__main__":
    unittest.main()
