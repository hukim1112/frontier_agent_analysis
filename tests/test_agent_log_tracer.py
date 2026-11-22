import asyncio
import gc
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import aiosqlite
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware, hook_config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt

import app.utils.workspace as workspace_mod
from app.context import AgentContext
from app.middleware.error_control.self_recovery import ToolErrorHandlerMiddleware
from app.middleware.observability import AgentLogTracer
from app.middleware.observability.agent_log_tracer import _extract_turn_anchor_id
from app.utils.message_markers import ORIGIN_COMPACTION_SUMMARY, ORIGIN_STOP_HOOK, origin_kwargs


def _last_human_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


class ScriptedToolChatModel(BaseChatModel):
    """Async-friendly scripted chat model supporting bind_tools and per-session sequencing."""

    responder: Any  # Callable[[List[BaseMessage]], AIMessage | BaseException]
    model_name: str = "scripted-test-model"

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        out = self.responder(messages)
        if isinstance(out, BaseException):
            raise out
        return ChatResult(generations=[ChatGeneration(message=out)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        out = self.responder(messages)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, BaseException):
            raise out
        return ChatResult(generations=[ChatGeneration(message=out)])


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class AgentLogTracerScenarioTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.users_root = Path(self.tmp.name) / "users"
        self.users_patch = patch.object(workspace_mod, "USERS_ROOT_DIR", self.users_root)
        self.users_patch.start()
        self.addCleanup(self.users_patch.stop)

    async def test_concurrent_multi_subagent_isolation(self):
        """동시에 실행되는 다수 서브에이전트가 하나의 AgentLogTracer를 공유해도 경로/버퍼가 절대 섞이지 않는지 검증."""
        tracer = AgentLogTracer(stream_events=True, disk_enabled=True)

        @tool
        async def async_save_tool(file_path: str, payload: str) -> str:
            """테스트용 비동기 저장 도구."""
            await asyncio.sleep(0.03)
            return f"saved {payload} -> {file_path}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(
                    content=f"Done: {last.content}",
                    usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
                )
            text = _last_human_text(messages)
            return AIMessage(
                content=f"Planning write for {text}",
                tool_calls=[
                    {
                        "name": "async_save_tool",
                        "args": {"file_path": f"outputs/{text}.json", "payload": text},
                        "id": f"tc_{text}",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[async_save_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        parent_sid = "sup_conc_01"
        specs = [
            ("scraper", f"{parent_sid}_scraper_1111aaaa", "alpha"),
            ("scraper", f"{parent_sid}_scraper_2222bbbb", "beta"),
            ("analyst", f"{parent_sid}_analyst_3333cccc", "gamma"),
        ]

        async def run_one(role: str, sub_tid: str, tag: str):
            ws = workspace_mod.resolve_session_workspace(
                user_id="u_conc",
                session_id=sub_tid,
                parent_session_id=parent_sid,
                subagent_role=role,
                ensure_exists=True,
            )
            ctx = AgentContext(
                agent_name=role,
                session_id=sub_tid,
                parent_session_id=parent_sid,
                user_id="u_conc",
                workspace_dir=str(ws.root_dir),
            )
            return await agent.ainvoke(
                {"messages": [HumanMessage(content=tag)]},
                config={"configurable": {"thread_id": sub_tid, "user_id": "u_conc"}},
                context=ctx,
            )

        await asyncio.gather(*(run_one(r, tid, tag) for r, tid, tag in specs))
        tracer.flush()

        # 잘못된 메인 세션 레벨(sessions/{sub_tid})에 누수 폴더가 생기면 안 됨
        sessions_dir = self.users_root / "u_conc" / "sessions"
        self.assertEqual([p.name for p in sessions_dir.iterdir()], [parent_sid])

        for role, sub_tid, tag in specs:
            sub_dir_name = sub_tid[len(f"{parent_sid}_") :]
            audit_path = sessions_dir / parent_sid / "subagents" / sub_dir_name / "logs" / "audit.jsonl"
            self.assertTrue(audit_path.exists(), f"Missing audit log for {sub_tid}")

            records = _read_jsonl(audit_path)
            self.assertTrue(any(r.get("event") == "turn_summary" for r in records))
            self.assertTrue(all(r.get("session_id") == sub_tid for r in records))
            serialized = json.dumps(records, ensure_ascii=False)
            self.assertIn(tag, serialized)
            for _, _, other_tag in specs:
                if other_tag != tag:
                    self.assertNotIn(other_tag, serialized)

            snap = tracer.get_progress_snapshot(sub_tid)
            self.assertTrue(snap["found"])
            self.assertEqual(snap["status"], "SUCCESS")
            self.assertEqual(snap["total_tool_calls"], 1)
            self.assertEqual(snap["touched_files"], [f"outputs/{tag}.json"])
            self.assertEqual(snap["token_usage"]["total_tokens"], 35)

    async def test_main_vs_subagent_hierarchy_and_anonymous_safety(self):
        """메인/서브 계층 경로(workspace_dir 직접 주입 및 parent_session_id 폴백)와 익명 실행 격리 검증."""
        tracer = AgentLogTracer(stream_events=True, disk_enabled=True)

        anon_started = asyncio.Event()
        anon_release = asyncio.Event()

        @tool
        async def probe_tool(path: str) -> str:
            """테스트용 프로브 도구."""
            if path == "anon_block":
                anon_started.set()
                await anon_release.wait()
            return f"ok:{path}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(content=f"Finished {last.content}")
            text = _last_human_text(messages)
            return AIMessage(
                content=f"Calling probe for {text}",
                tool_calls=[
                    {
                        "name": "probe_tool",
                        "args": {"path": text},
                        "id": f"call_{text}",
                        "type": "tool_call",
                    }
                ],
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[probe_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        parent_sid = "sup_main_01"
        await agent.ainvoke(
            {"messages": [HumanMessage(content="main_work")]},
            config={"configurable": {"thread_id": parent_sid, "user_id": "u_hier"}},
            context=AgentContext(agent_name="main_agent", session_id=parent_sid, user_id="u_hier"),
        )

        # 서브에이전트 1: 명시적 workspace_dir
        sub1_tid = f"{parent_sid}_scraper_aa11"
        sub1_ws = workspace_mod.resolve_session_workspace(
            user_id="u_hier",
            session_id=sub1_tid,
            parent_session_id=parent_sid,
            subagent_role="scraper",
            ensure_exists=True,
        )
        await agent.ainvoke(
            {"messages": [HumanMessage(content="sub_explicit")]},
            config={"configurable": {"thread_id": sub1_tid, "user_id": "u_hier"}},
            context=AgentContext(
                agent_name="scraper",
                session_id=sub1_tid,
                parent_session_id=parent_sid,
                user_id="u_hier",
                workspace_dir=str(sub1_ws.root_dir),
            ),
        )

        # 서브에이전트 2: workspace_dir 비우고 parent_session_id 폴백 해석
        sub2_tid = f"{parent_sid}_analyst_bb22"
        await agent.ainvoke(
            {"messages": [HumanMessage(content="sub_fallback")]},
            config={
                "configurable": {
                    "thread_id": sub2_tid,
                    "parent_thread_id": parent_sid,
                    "user_id": "u_hier",
                }
            },
            context=AgentContext(
                agent_name="analyst",
                session_id=sub2_tid,
                parent_session_id=parent_sid,
                user_id="u_hier",
            ),
        )

        # 익명 실행(thread_id/session_id 없음)이 실행 중일 때 명명된 세션과 서로 간섭하지 않는지 검증
        anon_task = asyncio.create_task(
            agent.ainvoke({"messages": [HumanMessage(content="anon_block")]})
        )
        await asyncio.wait_for(anon_started.wait(), timeout=2.0)

        named_during_anon = "sup_named_during_anon"
        await agent.ainvoke(
            {"messages": [HumanMessage(content="named_work")]},
            config={"configurable": {"thread_id": named_during_anon, "user_id": "u_hier"}},
            context=AgentContext(agent_name="main_agent", session_id=named_during_anon, user_id="u_hier"),
        )
        anon_release.set()
        await anon_task

        # 완료 후 추가 익명 호출도 기존 명명 세션을 덮어쓰지 않아야 함
        await agent.ainvoke({"messages": [HumanMessage(content="anon_second")]})
        tracer.flush()

        main_audit = self.users_root / "u_hier" / "sessions" / parent_sid / "logs" / "audit.jsonl"
        sub1_audit = (
            self.users_root / "u_hier" / "sessions" / parent_sid / "subagents" / "scraper_aa11" / "logs" / "audit.jsonl"
        )
        sub2_audit = (
            self.users_root / "u_hier" / "sessions" / parent_sid / "subagents" / "analyst_bb22" / "logs" / "audit.jsonl"
        )
        self.assertTrue(main_audit.exists())
        self.assertTrue(sub1_audit.exists())
        self.assertTrue(sub2_audit.exists())

        self.assertEqual(tracer.get_progress_snapshot(parent_sid)["status"], "SUCCESS")
        self.assertEqual(tracer.get_progress_snapshot(sub1_tid)["status"], "SUCCESS")
        self.assertEqual(tracer.get_progress_snapshot(sub2_tid)["status"], "SUCCESS")
        self.assertEqual(tracer.get_progress_snapshot(named_during_anon)["status"], "SUCCESS")
        self.assertIn("main_work", tracer.get_logs(parent_sid)[1]["arguments"]["path"])

    async def test_live_inflight_progress_snapshot_and_report(self):
        """실행 중(RUNNING) LLM thinking 단계와 도구 실행(active_tools) 단계의 실시간 스냅샷/리포트 검증."""
        tracer = AgentLogTracer(stream_events=True, disk_enabled=True)

        llm_entered = asyncio.Event()
        llm_release = asyncio.Event()
        tool_entered = asyncio.Event()
        tool_release = asyncio.Event()

        @tool
        async def slow_file_tool(file_path: str, query: str) -> str:
            """실시간 진행 상태 확인용 느린 도구."""
            tool_entered.set()
            await tool_release.wait()
            return f"written {file_path} for {query}"

        async def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(content="All artifacts written.")
            llm_entered.set()
            await llm_release.wait()
            return AIMessage(
                content="Step 1: writing parsed market dataset to outputs/data.json",
                tool_calls=[
                    {
                        "name": "slow_file_tool",
                        "args": {"file_path": "outputs/data.json", "query": "check"},
                        "id": "call_live_1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={"input_tokens": 20, "output_tokens": 11, "total_tokens": 31},
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[slow_file_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        sub_tid = "sup_live_scraper_abcd1234"
        ctx = AgentContext(
            agent_name="scraper",
            session_id=sub_tid,
            parent_session_id="sup_live",
            user_id="u_live",
        )
        run_task = asyncio.create_task(
            agent.ainvoke(
                {"messages": [HumanMessage(content="Collect prices")]},
                config={"configurable": {"thread_id": sub_tid, "user_id": "u_live"}},
                context=ctx,
            )
        )

        # 1) LLM 추론(thinking) 중 스냅샷 검증 (인스턴스 + WeakSet 전역 클래스 조회 모두 확인)
        await asyncio.wait_for(llm_entered.wait(), timeout=2.0)
        snap_llm = AgentLogTracer.get_session_snapshot(sub_tid)
        report_llm = AgentLogTracer.format_session_report(sub_tid)
        self.assertTrue(snap_llm["found"])
        self.assertEqual(snap_llm["source"], "memory")
        self.assertEqual(snap_llm["status"], "RUNNING")
        self.assertEqual(snap_llm["phase"], "thinking")
        self.assertIsNotNone(snap_llm["active_model_call"])
        self.assertEqual(snap_llm["active_model_call"]["step"], 1)
        self.assertIn("Active LLM Call: Step #1", report_llm)

        llm_release.set()

        # 2) 도구 실행(running_tool) 중 스냅샷 검증
        await asyncio.wait_for(tool_entered.wait(), timeout=2.0)
        snap_tool = AgentLogTracer.get_session_snapshot(sub_tid)
        report_tool = AgentLogTracer.format_session_report(sub_tid)
        self.assertEqual(snap_tool["status"], "RUNNING")
        self.assertEqual(snap_tool["phase"], "running_tool")
        self.assertIsNone(snap_tool["active_model_call"])
        self.assertEqual(len(snap_tool["active_tools"]), 1)
        self.assertEqual(snap_tool["active_tools"][0]["tool_name"], "slow_file_tool")
        self.assertIn("outputs/data.json", snap_tool["active_tools"][0]["args_summary"])
        self.assertGreaterEqual(snap_tool["active_tools"][0]["elapsed_ms"], 0)
        self.assertIn("Step 1: writing parsed market dataset", snap_tool["latest_thought"])
        self.assertIn("Active Tool: slow_file_tool", report_tool)

        tool_release.set()
        await asyncio.wait_for(run_task, timeout=2.0)
        tracer.flush()

        # 3) 완료 스냅샷 검증
        snap_done = tracer.get_progress_snapshot(sub_tid)
        report_done = tracer.format_progress_report(sub_tid)
        self.assertEqual(snap_done["status"], "SUCCESS")
        self.assertEqual(snap_done["phase"], "completed")
        self.assertEqual(snap_done["active_tools"], [])
        self.assertIsNone(snap_done["active_model_call"])
        self.assertEqual(snap_done["llm_step_count"], 2)
        self.assertEqual(snap_done["total_tool_calls"], 1)
        self.assertEqual(snap_done["touched_files"], ["outputs/data.json"])
        self.assertIn("[OK] slow_file_tool", report_done)

    async def test_tool_error_soft_crash_and_interrupt_handling(self):
        """도구 소프트 에러(ToolErrorHandlerMiddleware), 미처리 크래시(FAILED), HITL 인터럽트(GraphBubbleUp) 구분 검증."""
        tracer = AgentLogTracer(stream_events=True, disk_enabled=True)

        @tool
        async def flaky_tool(mode: str) -> str:
            """테스트용 예외/에러 도구."""
            if mode == "soft_raise":
                raise RuntimeError("network timeout")
            if mode == "error_prefix":
                return "[ERROR: invalid selector on page]"
            if mode == "crash":
                raise ValueError("fatal db corruption")
            if mode == "interrupt":
                raise GraphInterrupt(())
            return "ok"

        def make_responder(mode: str):
            def responder(messages: List[BaseMessage]) -> AIMessage:
                last = messages[-1]
                if isinstance(last, ToolMessage):
                    return AIMessage(content="Recovered after tool observation.")
                return AIMessage(
                    content=f"Running mode {mode}",
                    tool_calls=[
                        {
                            "name": "flaky_tool",
                            "args": {"mode": mode},
                            "id": f"call_{mode}",
                            "type": "tool_call",
                        }
                    ],
                )

            return responder

        # A. ToolErrorHandlerMiddleware가 예외를 ToolMessage(status="error")로 변환하는 경우
        soft_agent = create_agent(
            model=ScriptedToolChatModel(responder=make_responder("soft_raise")),
            tools=[flaky_tool],
            middleware=[tracer, ToolErrorHandlerMiddleware(max_retries=0)],
            context_schema=AgentContext,
        )
        soft_tid = "sup_err_soft"
        await soft_agent.ainvoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": soft_tid, "user_id": "u_err"}},
            context=AgentContext(agent_name="scraper", session_id=soft_tid, user_id="u_err"),
        )
        snap_soft = tracer.get_progress_snapshot(soft_tid)
        report_soft = tracer.format_progress_report(soft_tid)
        self.assertEqual(snap_soft["status"], "SUCCESS")
        self.assertEqual(snap_soft["failed_tool_calls"], 1)
        self.assertIn("network timeout", snap_soft["last_error"])
        self.assertIn("[ERR] flaky_tool", report_soft)

        # A-2. 문자열 "[ERROR: ...]" 반환도 FAILED로 분류되는지 검증
        prefix_agent = create_agent(
            model=ScriptedToolChatModel(responder=make_responder("error_prefix")),
            tools=[flaky_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )
        prefix_tid = "sup_err_prefix"
        await prefix_agent.ainvoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": prefix_tid, "user_id": "u_err"}},
            context=AgentContext(agent_name="scraper", session_id=prefix_tid, user_id="u_err"),
        )
        snap_prefix = tracer.get_progress_snapshot(prefix_tid)
        self.assertEqual(snap_prefix["failed_tool_calls"], 1)
        self.assertIn("invalid selector", snap_prefix["last_error"])

        # B. ToolErrorHandlerMiddleware 없이 도구 예외가 전파되는 미처리 크래시 -> FAILED 영속화
        crash_agent = create_agent(
            model=ScriptedToolChatModel(responder=make_responder("crash")),
            tools=[flaky_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )
        crash_tid = "sup_err_scraper_crash"
        with self.assertRaises(ValueError):
            await crash_agent.ainvoke(
                {"messages": [HumanMessage(content="go")]},
                config={"configurable": {"thread_id": crash_tid, "user_id": "u_err"}},
                context=AgentContext(
                    agent_name="scraper",
                    session_id=crash_tid,
                    parent_session_id="sup_err",
                    user_id="u_err",
                ),
            )
        tracer.flush()

        snap_crash = tracer.get_progress_snapshot(crash_tid)
        self.assertEqual(snap_crash["status"], "FAILED")
        self.assertEqual(snap_crash["phase"], "failed")
        self.assertEqual(snap_crash["active_tools"], [])
        self.assertIn("fatal db corruption", snap_crash["last_error"])

        crash_audit = (
            self.users_root / "u_err" / "sessions" / "sup_err" / "subagents" / "scraper_crash" / "logs" / "audit.jsonl"
        )
        crash_records = _read_jsonl(crash_audit)
        self.assertTrue(
            any(r.get("event") == "turn_summary" and r.get("status") == "FAILED" for r in crash_records)
        )

        # C. LLM 단계 미처리 예외 -> FAILED 영속화
        def llm_crash_responder(_messages: List[BaseMessage]):
            raise RuntimeError("upstream 503")

        llm_crash_agent = create_agent(
            model=ScriptedToolChatModel(responder=llm_crash_responder),
            tools=[flaky_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )
        llm_crash_tid = "sup_err_llm_crash"
        with self.assertRaises(RuntimeError):
            await llm_crash_agent.ainvoke(
                {"messages": [HumanMessage(content="go")]},
                config={"configurable": {"thread_id": llm_crash_tid, "user_id": "u_err"}},
                context=AgentContext(agent_name="analyst", session_id=llm_crash_tid, user_id="u_err"),
            )
        tracer.flush()
        snap_llm_crash = tracer.get_progress_snapshot(llm_crash_tid)
        self.assertEqual(snap_llm_crash["status"], "FAILED")
        self.assertIn("upstream 503", snap_llm_crash["last_error"])

        # D. GraphBubbleUp(GraphInterrupt)는 FAILED로 오분류되면 안 됨
        from langgraph.errors import GraphBubbleUp

        intr_conn = await aiosqlite.connect(str(Path(self.tmp.name) / "intr_cp.db"), check_same_thread=False)
        try:
            intr_cp = AsyncSqliteSaver(intr_conn)
            await intr_cp.setup()
            intr_agent = create_agent(
                model=ScriptedToolChatModel(responder=make_responder("interrupt")),
                tools=[flaky_tool],
                middleware=[tracer],
                checkpointer=intr_cp,
                context_schema=AgentContext,
            )
            intr_tid = "sup_intr_01"
            intr_result = await intr_agent.ainvoke(
                {"messages": [HumanMessage(content="go")]},
                config={"configurable": {"thread_id": intr_tid, "user_id": "u_err"}},
                context=AgentContext(agent_name="main_agent", session_id=intr_tid, user_id="u_err"),
            )
            # LangGraph는 GraphInterrupt 발생 시 __interrupt__ 또는 미완료 AIMessage 상태로 중단
            self.assertEqual(len(intr_result["messages"]), 2)
            snap_intr = tracer.get_progress_snapshot(intr_tid)
            self.assertEqual(snap_intr["status"], "INTERRUPTED")
            self.assertEqual(snap_intr["phase"], "interrupted")
            self.assertEqual(snap_intr["active_tools"], [])
        finally:
            await intr_conn.close()

        # awrap_tool_call 직접 호출 시에도 GraphBubbleUp이 그대로 전파되며 FAILED가 아님을 검증
        from types import SimpleNamespace

        direct_tid = "sup_intr_direct"
        tracer.start_turn(direct_tid, "hitl")
        fake_req = SimpleNamespace(
            tool_call={"name": "flaky_tool", "args": {"mode": "interrupt"}, "id": "tc_intr"},
            runtime=SimpleNamespace(config={"configurable": {"thread_id": direct_tid}}),
        )

        async def _raise_intr(_req):
            raise GraphInterrupt(())

        with self.assertRaises(GraphBubbleUp):
            await tracer.awrap_tool_call(fake_req, _raise_intr)
        snap_direct = tracer.get_progress_snapshot(direct_tid)
        self.assertEqual(snap_direct["status"], "INTERRUPTED")
        self.assertEqual(snap_direct["phase"], "interrupted")
        self.assertEqual(snap_direct["active_tools"], [])

    async def test_mid_turn_task_cancellation(self):
        """실행 중 asyncio.Task 취소(asyncio.CancelledError) 시 active_tools 정리 및 CANCELLED 영속화 검증."""
        tracer = AgentLogTracer(stream_events=True, disk_enabled=True)
        tool_started = asyncio.Event()

        @tool
        async def blocking_tool(url: str) -> str:
            """취소 테스트용 블로킹 도구."""
            tool_started.set()
            await asyncio.sleep(10)
            return url

        def responder(messages: List[BaseMessage]) -> AIMessage:
            return AIMessage(
                content="Fetching target URL",
                tool_calls=[
                    {
                        "name": "blocking_tool",
                        "args": {"url": "https://example.com"},
                        "id": "call_cancel_1",
                        "type": "tool_call",
                    }
                ],
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[blocking_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        sub_tid = "sup_can_scraper_9988"
        task = asyncio.create_task(
            agent.ainvoke(
                {"messages": [HumanMessage(content="Crawl")]},
                config={"configurable": {"thread_id": sub_tid, "user_id": "u_can"}},
                context=AgentContext(
                    agent_name="scraper",
                    session_id=sub_tid,
                    parent_session_id="sup_can",
                    user_id="u_can",
                ),
            )
        )
        await asyncio.wait_for(tool_started.wait(), timeout=2.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        tracer.flush()

        snap = tracer.get_progress_snapshot(sub_tid)
        report = tracer.format_progress_report(sub_tid)
        self.assertEqual(snap["status"], "CANCELLED")
        self.assertEqual(snap["phase"], "cancelled")
        self.assertEqual(snap["active_tools"], [])
        self.assertIn("CANCELLED", report)

        audit_path = (
            self.users_root / "u_can" / "sessions" / "sup_can" / "subagents" / "scraper_9988" / "logs" / "audit.jsonl"
        )
        records = _read_jsonl(audit_path)
        self.assertTrue(
            any(r.get("event") == "turn_summary" and r.get("status") == "CANCELLED" for r in records)
        )

    async def test_disk_fallback_and_nonexistent_cleanliness(self):
        """메모리가 비어있는 새 Tracer 인스턴스에서 디스크 audit.jsonl 폴백 복원 및 미존재 세션 무생성 검증."""
        tracer_writer = AgentLogTracer(stream_events=True, disk_enabled=True)

        @tool
        async def export_tool(excel_path: str) -> str:
            """엑셀 출력 시뮬레이션 도구."""
            return f"created {excel_path}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(
                    content="Report exported.",
                    usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
                )
            return AIMessage(
                content="Generating Excel summary at outputs/summary.xlsx",
                tool_calls=[
                    {
                        "name": "export_tool",
                        "args": {"excel_path": "outputs/summary.xlsx"},
                        "id": "call_disk_1",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[export_tool],
            middleware=[tracer_writer],
            context_schema=AgentContext,
        )

        sup_tid = "sup_disk_01"
        sub_tid = f"{sup_tid}_analyst_dd44"
        sub_ws = workspace_mod.resolve_session_workspace(
            user_id="u_disk",
            session_id=sub_tid,
            parent_session_id=sup_tid,
            subagent_role="analyst",
            ensure_exists=True,
        )
        await agent.ainvoke(
            {"messages": [HumanMessage(content="Export")]},
            config={"configurable": {"thread_id": sub_tid, "user_id": "u_disk"}},
            context=AgentContext(
                agent_name="analyst",
                session_id=sub_tid,
                parent_session_id=sup_tid,
                user_id="u_disk",
                workspace_dir=str(sub_ws.root_dir),
            ),
        )
        tracer_writer.flush()
        del agent
        del tracer_writer
        gc.collect()

        # 살아있는 Tracer 메모리 없이도 클래스 메서드 get_session_snapshot/format_session_report가 디스크 SSOT에서 복원
        snap = AgentLogTracer.get_session_snapshot(
            session_id=sub_tid,
            user_id="u_disk",
            parent_session_id=sup_tid,
            workspace_dir=str(sub_ws.root_dir),
        )
        report = AgentLogTracer.format_session_report(
            session_id=sub_tid,
            user_id="u_disk",
            parent_session_id=sup_tid,
            workspace_dir=str(sub_ws.root_dir),
        )
        self.assertTrue(snap["found"])
        self.assertEqual(snap["source"], "disk")
        self.assertEqual(snap["status"], "SUCCESS")
        self.assertEqual(snap["phase"], "completed")
        self.assertEqual(snap["llm_step_count"], 2)
        self.assertEqual(snap["total_tool_calls"], 1)
        self.assertEqual(snap["touched_files"], ["outputs/summary.xlsx"])
        self.assertEqual(snap["token_usage"]["total_tokens"], 34)
        self.assertIn("outputs/summary.xlsx", report)

        # 존재하지 않는 세션 조회 시 빈 디렉터리/파일을 생성하지 않아야 함
        ghost_snap = AgentLogTracer.get_session_snapshot(
            session_id="does_not_exist_999",
            user_id="u_ghost",
            parent_session_id="sup_ghost",
        )
        ghost_report = AgentLogTracer.format_session_report(
            session_id="does_not_exist_999",
            user_id="u_ghost",
            parent_session_id="sup_ghost",
        )
        self.assertFalse(ghost_snap["found"])
        self.assertEqual(ghost_snap["source"], "none")
        self.assertEqual(ghost_report, "")
        self.assertFalse((self.users_root / "u_ghost").exists())

    async def test_disk_disabled_mode_and_subagent_factory_attachment(self):
        """disk_enabled=False 인메모리 모드, GC 안전성(WeakSet), scraper/analyst 자동 등록 검증."""
        from app.middleware.observability.agent_log_tracer import _TRACER_INSTANCES

        no_disk_dir = Path(self.tmp.name) / "no_disk_logs"

        @tool
        async def echo_file(file_path: str) -> str:
            """인메모리 테스트용 도구."""
            return f"ok:{file_path}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(
                    content="In-memory done.",
                    usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                )
            return AIMessage(
                content="Running echo_file",
                tool_calls=[
                    {
                        "name": "echo_file",
                        "args": {"file_path": "outputs/mem.txt"},
                        "id": "call_mem_1",
                        "type": "tool_call",
                    }
                ],
            )

        # 별도 변수/속성 바인딩 없이 인라인 생성된 AgentLogTracer도 컴파일된 그래프가 살아있는 동안 WeakSet에서 유지되는지 검증
        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[echo_file],
            middleware=[AgentLogTracer(log_dir=str(no_disk_dir), disk_enabled=False)],
            context_schema=AgentContext,
        )
        gc.collect()

        mem_tid = "mem_only_01"
        await agent.ainvoke(
            {"messages": [HumanMessage(content="Go")]},
            config={"configurable": {"thread_id": mem_tid}},
        )

        snap = AgentLogTracer.get_session_snapshot(mem_tid)
        self.assertTrue(snap["found"])
        self.assertEqual(snap["source"], "memory")
        self.assertEqual(snap["status"], "SUCCESS")
        self.assertEqual(snap["touched_files"], ["outputs/mem.txt"])
        self.assertEqual(snap["token_usage"]["total_tokens"], 8)
        self.assertFalse(no_disk_dir.exists())

        # scraper / analyst 팩토리가 별도 속성(.log_tracer) 노출 없이도 WeakSet에 Tracer를 자동 등록하는지 검증
        from app.agents.analyst import agent as analyst_mod
        from app.agents.scraper import agent as scraper_mod

        conn = await aiosqlite.connect(str(Path(self.tmp.name) / "cp.db"), check_same_thread=False)
        try:
            cp = AsyncSqliteSaver(conn)
            await cp.setup()

            orig_load_scraper = scraper_mod._load_config

            def _fake_load(path: str, default: dict) -> dict:
                if "logging.config" in path:
                    return {"logging_enabled": False}
                return orig_load_scraper(path, default)

            before_ids = {id(t) for t in list(_TRACER_INSTANCES)}
            with patch.object(scraper_mod, "_load_config", side_effect=_fake_load), patch.object(
                analyst_mod, "_load_config", side_effect=_fake_load
            ):
                scraper_exec = await scraper_mod.create_agent_executor(checkpointer=cp)
                analyst_exec = await analyst_mod.create_agent_executor(checkpointer=cp)

            gc.collect()
            self.assertFalse(hasattr(scraper_exec, "log_tracer"))
            self.assertFalse(hasattr(analyst_exec, "log_tracer"))
            new_tracers = [t for t in list(_TRACER_INSTANCES) if id(t) not in before_ids]
            self.assertEqual(len(new_tracers), 2)
            self.assertTrue(all(not t.disk_enabled for t in new_tracers))
        finally:
            await conn.close()

    async def test_anonymous_and_named_concurrent_zero_event_leakage(self):
        """익명 실행 중 명명 세션 및 다중 익명 실행이 동시 교차해도 이벤트가 전혀 섞이지 않는지 검증."""
        tracer = AgentLogTracer(disk_enabled=False)
        a_tool_entered = asyncio.Event()
        b_running = asyncio.Event()
        release_b = asyncio.Event()
        a_second_llm_done = asyncio.Event()

        @tool
        async def step(tag: str) -> str:
            """테스트용 인터리빙 도구."""
            if tag == "A1":
                a_tool_entered.set()
                await b_running.wait()
            elif tag == "B1":
                b_running.set()
                await release_b.wait()
            elif tag.startswith("anon_"):
                await asyncio.sleep(0.02)
            return f"done {tag}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            first = next(m for m in messages if getattr(m, "type", "") == "human").content
            if isinstance(last, ToolMessage):
                if first == "A":
                    a_second_llm_done.set()
                return AIMessage(content=f"final {first}")
            arg_tag = first if first.startswith("anon_") else f"{first}1"
            return AIMessage(
                content=f"plan {first}",
                tool_calls=[
                    {
                        "name": "step",
                        "args": {"tag": arg_tag},
                        "id": f"c_{first}",
                        "type": "tool_call",
                    }
                ],
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[step],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        # 1) 익명 실행 A 진행 중 명명 세션 B 동시 실행
        task_a = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage(content="A")]}))
        await asyncio.wait_for(a_tool_entered.wait(), timeout=2.0)

        task_b = asyncio.create_task(
            agent.ainvoke(
                {"messages": [HumanMessage(content="B")]},
                config={"configurable": {"thread_id": "named_B"}},
                context=AgentContext(session_id="named_B"),
            )
        )
        await asyncio.wait_for(a_second_llm_done.wait(), timeout=2.0)
        release_b.set()
        await asyncio.gather(task_a, task_b)

        b_events = tracer.get_logs("named_B")
        b_model_snips = [e.get("response_snippet") for e in b_events if e.get("type") == "model_call"]
        b_tool_args = [e.get("arguments", {}).get("tag") for e in b_events if e.get("type") == "tool_call"]
        self.assertEqual(b_model_snips, ["plan B", "final B"])
        self.assertEqual(b_tool_args, ["B1"])

        anon_a_keys = [k for k in tracer._active_runs.keys() if k != "named_B"]
        self.assertEqual(len(anon_a_keys), 1)
        a_events = tracer.get_logs(anon_a_keys[0])
        self.assertEqual(
            [e.get("response_snippet") for e in a_events if e.get("type") == "model_call"],
            ["plan A", "final A"],
        )
        self.assertEqual(
            [e.get("arguments", {}).get("tag") for e in a_events if e.get("type") == "tool_call"],
            ["A1"],
        )
        self.assertEqual(tracer.get_progress_snapshot(anon_a_keys[0])["status"], "SUCCESS")

        # 2) 다중 동시 익명 호출 격리 검증
        await asyncio.gather(
            agent.ainvoke({"messages": [HumanMessage(content="anon_X")]}),
            agent.ainvoke({"messages": [HumanMessage(content="anon_Y")]}),
        )
        concurrent_anon = [k for k in tracer._active_runs.keys() if k not in ("named_B", anon_a_keys[0])]
        self.assertEqual(len(concurrent_anon), 2)
        for sid in concurrent_anon:
            events = tracer.get_logs(sid)
            snips = [e.get("response_snippet") for e in events if e.get("type") == "model_call"]
            if "plan anon_X" in snips:
                self.assertEqual(snips, ["plan anon_X", "final anon_X"])
            else:
                self.assertEqual(snips, ["plan anon_Y", "final anon_Y"])

    async def test_bounded_completed_runs_eviction_and_disk_fallback(self):
        """완료된 세션만 max_completed_runs 상한 내로 퇴거하고 RUNNING 유지 및 디스크 폴백 조회 검증."""
        tracer = AgentLogTracer(max_completed_runs=5, stream_events=True, disk_enabled=True)
        running_entered = asyncio.Event()
        release_running = asyncio.Event()

        @tool
        async def work_tool(tag: str) -> str:
            """테스트용 작업 도구."""
            if tag == "long_running":
                running_entered.set()
                await release_running.wait()
            return f"ok:{tag}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(content=f"Completed {last.content}")
            tag = _last_human_text(messages)
            return AIMessage(
                content=f"Start {tag}",
                tool_calls=[
                    {
                        "name": "work_tool",
                        "args": {"tag": tag},
                        "id": f"tc_{tag}",
                        "type": "tool_call",
                    }
                ],
            )

        agent = create_agent(
            model=ScriptedToolChatModel(responder=responder),
            tools=[work_tool],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        # 1) 장기 RUNNING 세션 1건 시작 및 도구 내부 대기
        long_tid = "long_running_01"
        long_task = asyncio.create_task(
            agent.ainvoke(
                {"messages": [HumanMessage(content="long_running")]},
                config={"configurable": {"thread_id": long_tid, "user_id": "u_evict"}},
                context=AgentContext(session_id=long_tid, user_id="u_evict"),
            )
        )
        await asyncio.wait_for(running_entered.wait(), timeout=2.0)

        try:
            # 2) 완료 세션 12건 연속 실행
            for idx in range(12):
                sid = f"done_{idx:02d}"
                await agent.ainvoke(
                    {"messages": [HumanMessage(content=sid)]},
                    config={"configurable": {"thread_id": sid, "user_id": "u_evict"}},
                    context=AgentContext(session_id=sid, user_id="u_evict"),
                )
            tracer.flush()

            self.assertIn(long_tid, tracer._active_runs)
            self.assertEqual(tracer._active_runs[long_tid]["status"], "RUNNING")

            completed_keys = sorted(
                k for k, v in tracer._active_runs.items() if v.get("status") != "RUNNING"
            )
            self.assertEqual(completed_keys, ["done_07", "done_08", "done_09", "done_10", "done_11"])

            # 퇴거된 오래된 세션도 디스크 audit.jsonl 폴백으로 복원 가능해야 함
            snap_inst = tracer.get_progress_snapshot("done_00", user_id="u_evict")
            snap_cls = AgentLogTracer.get_session_snapshot("done_00", user_id="u_evict")
            for snap in (snap_inst, snap_cls):
                self.assertTrue(snap["found"])
                self.assertEqual(snap["source"], "disk")
                self.assertEqual(snap["status"], "SUCCESS")
                self.assertEqual(snap["total_tool_calls"], 1)
        finally:
            release_running.set()
            await asyncio.wait_for(long_task, timeout=2.0)
            tracer.flush()

    async def test_reused_agent_context_respects_thread_id_priority(self):
        """동일 AgentContext 인스턴스 재사용 시 thread_id가 최우선 적용되고 context.session_id가 오염되지 않는지 검증."""
        tracer = AgentLogTracer(disk_enabled=False)

        agent = create_agent(
            model=ScriptedToolChatModel(responder=lambda _m: AIMessage(content="ok")),
            tools=[],
            middleware=[tracer],
            context_schema=AgentContext,
        )

        reused_ctx = AgentContext(user_id="u_reuse")
        await agent.ainvoke(
            {"messages": [HumanMessage(content="1")]},
            config={"configurable": {"thread_id": "t1"}},
            context=reused_ctx,
        )
        await agent.ainvoke(
            {"messages": [HumanMessage(content="2")]},
            config={"configurable": {"thread_id": "t2"}},
            context=reused_ctx,
        )
        await agent.ainvoke(
            {"messages": [HumanMessage(content="3")]},
            context=AgentContext(user_id="u_reuse", session_id="ctx_only"),
        )

        self.assertEqual(reused_ctx.session_id, "unknown")
        self.assertEqual(set(tracer._active_runs.keys()), {"t1", "t2", "ctx_only"})

    async def test_anonymous_stop_hook_and_compaction_keep_single_session(self):
        """익명 실행 중 StopHooks/Compaction 하네스 기원 HumanMessage가 삽입되어도 단일 세션으로 유지되는지 검증."""

        class FakeStopHook(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.fired = False

            @hook_config(can_jump_to=["model"])
            def after_agent(self, state, runtime=None):
                if self.fired:
                    return None
                self.fired = True
                fb = HumanMessage(
                    content="fix your code",
                    additional_kwargs=origin_kwargs(ORIGIN_STOP_HOOK),
                )
                return {"messages": [fb], "jump_to": "model"}

        tracer = AgentLogTracer(disk_enabled=False)
        agent = create_agent(
            model=ScriptedToolChatModel(responder=lambda m: AIMessage(content=f"reply#{len(m)}")),
            tools=[],
            middleware=[tracer, FakeStopHook()],
            context_schema=AgentContext,
        )
        await agent.ainvoke({"messages": [HumanMessage(content="write code")]})

        self.assertEqual(len(tracer._active_runs), 1)
        only_run = next(iter(tracer._active_runs.values()))
        self.assertEqual(only_run["status"], "SUCCESS")
        self.assertEqual(only_run["phase"], "completed")
        self.assertEqual(only_run["llm_step_count"], 2)
        self.assertEqual(
            [e.get("response_snippet") for e in only_run["events"] if e.get("type") == "model_call"],
            ["reply#1", "reply#3"],
        )
        self.assertEqual(tracer._anon_turn_sessions, {})

        user_msg = HumanMessage(content="real user prompt", id="user-anchor-1")
        compaction_msg = HumanMessage(
            content="summary",
            id="compaction-ignored-id",
            additional_kwargs=origin_kwargs(ORIGIN_COMPACTION_SUMMARY),
        )
        stop_hook_msg = HumanMessage(
            content="fix",
            id="stop-hook-ignored-id",
            additional_kwargs=origin_kwargs(ORIGIN_STOP_HOOK),
        )
        self.assertEqual(
            _extract_turn_anchor_id({"messages": [user_msg, compaction_msg, stop_hook_msg]}),
            "user-anchor-1",
        )
        self.assertEqual(
            _extract_turn_anchor_id({"messages": [compaction_msg, user_msg, stop_hook_msg]}),
            "user-anchor-1",
        )

    async def test_hitl_after_model_pause_shows_awaiting_tools_and_resumes(self):
        """HumanInTheLoopMiddleware.after_model 승인 대기 시 phase='awaiting_tools' 표기 및 재개 후 완료 검증."""
        tracer = AgentLogTracer(disk_enabled=False)

        @tool
        def roll(n: int) -> str:
            """주사위 테스트 도구."""
            return f"rolled {n}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="done")
            return AIMessage(
                content="",
                tool_calls=[{"name": "roll", "args": {"n": 2}, "id": "c_hitl", "type": "tool_call"}],
            )

        conn = await aiosqlite.connect(str(Path(self.tmp.name) / "hitl_b.db"), check_same_thread=False)
        try:
            cp = AsyncSqliteSaver(conn)
            await cp.setup()
            agent = create_agent(
                model=ScriptedToolChatModel(responder=responder),
                tools=[roll],
                middleware=[
                    tracer,
                    HumanInTheLoopMiddleware(interrupt_on={"roll": {"allowed_decisions": ["approve"]}}),
                ],
                checkpointer=cp,
                context_schema=AgentContext,
            )
            cfg = {"configurable": {"thread_id": "hitl_B"}}
            ctx = AgentContext(session_id="hitl_B")
            await agent.ainvoke({"messages": [HumanMessage(content="roll")]}, config=cfg, context=ctx)

            s1 = tracer.get_progress_snapshot("hitl_B")
            self.assertEqual(s1["status"], "RUNNING")
            self.assertEqual(s1["phase"], "awaiting_tools")
            self.assertEqual(s1["turn_id"], 1)
            self.assertEqual(s1["llm_step_count"], 1)
            self.assertEqual(s1["total_tool_calls"], 0)

            await agent.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config=cfg,
                context=ctx,
            )
            s2 = tracer.get_progress_snapshot("hitl_B")
            self.assertEqual(s2["status"], "SUCCESS")
            self.assertEqual(s2["phase"], "completed")
            self.assertEqual(s2["turn_id"], 1)
            self.assertEqual(s2["total_tool_calls"], 1)
            self.assertEqual(s2["llm_step_count"], 2)
        finally:
            await conn.close()

    async def test_tool_interrupt_resets_status_to_running_on_resume(self):
        """도구 내부 interrupt() 재개 시 before_agent 없이도 status가 INTERRUPTED에서 RUNNING으로 복귀하는지 검증."""
        tracer = AgentLogTracer(disk_enabled=False)
        resumed_inside = asyncio.Event()
        release = asyncio.Event()

        @tool
        async def ask(q: str) -> str:
            """인터럽트 재개 테스트 도구."""
            ans = interrupt({"q": q})
            resumed_inside.set()
            await release.wait()
            return f"answer {ans}"

        def responder(messages: List[BaseMessage]) -> AIMessage:
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="done")
            return AIMessage(
                content="",
                tool_calls=[{"name": "ask", "args": {"q": "ok?"}, "id": "c_intr", "type": "tool_call"}],
            )

        conn = await aiosqlite.connect(str(Path(self.tmp.name) / "intr_c.db"), check_same_thread=False)
        try:
            cp = AsyncSqliteSaver(conn)
            await cp.setup()
            agent = create_agent(
                model=ScriptedToolChatModel(responder=responder),
                tools=[ask],
                middleware=[tracer],
                checkpointer=cp,
                context_schema=AgentContext,
            )
            cfg = {"configurable": {"thread_id": "intr_C"}}
            ctx = AgentContext(session_id="intr_C")
            await agent.ainvoke({"messages": [HumanMessage(content="go")]}, config=cfg, context=ctx)

            s1 = tracer.get_progress_snapshot("intr_C")
            self.assertEqual(s1["status"], "INTERRUPTED")
            self.assertEqual(s1["phase"], "interrupted")

            task = asyncio.create_task(
                agent.ainvoke(Command(resume="yes"), config=cfg, context=ctx)
            )
            await asyncio.wait_for(resumed_inside.wait(), timeout=5.0)
            s2 = tracer.get_progress_snapshot("intr_C")
            self.assertEqual(s2["status"], "RUNNING")
            self.assertEqual(s2["phase"], "running_tool")
            self.assertEqual([t["tool_name"] for t in s2["active_tools"]], ["ask"])

            release.set()
            await asyncio.wait_for(task, timeout=5.0)
            s3 = tracer.get_progress_snapshot("intr_C")
            self.assertEqual(s3["status"], "SUCCESS")
            self.assertEqual(s3["phase"], "completed")
            self.assertEqual(s3["active_tools"], [])
        finally:
            await conn.close()

    async def test_stale_active_sessions_evicted_by_ttl(self):
        """stale_active_ttl_s를 초과한 방치 RUNNING/INTERRUPTED 세션과 익명 앵커가 정리되는지 검증."""
        tracer = AgentLogTracer(disk_enabled=False, stale_active_ttl_s=60.0)
        tracer.start_turn("stale_running", "q1")
        tracer.start_turn("stale_interrupted", "q2")
        tracer.start_turn("fresh_running", "q3")

        old_ts = time.time() - 120.0
        for sid in ("stale_running", "stale_interrupted"):
            tracer._active_runs[sid]["updated_at"] = old_ts
            tracer._active_runs[sid]["start_time"] = old_ts
        tracer._active_runs["stale_interrupted"]["status"] = "INTERRUPTED"
        tracer._active_runs["stale_interrupted"]["phase"] = "interrupted"
        tracer._anon_turn_sessions["anchor_stale"] = "stale_running"

        agent = create_agent(
            model=ScriptedToolChatModel(responder=lambda _m: AIMessage(content="ok")),
            tools=[],
            middleware=[tracer],
            context_schema=AgentContext,
        )
        await agent.ainvoke(
            {"messages": [HumanMessage(content="new")]},
            config={"configurable": {"thread_id": "new_session"}},
        )

        self.assertNotIn("stale_running", tracer._active_runs)
        self.assertNotIn("stale_interrupted", tracer._active_runs)
        self.assertNotIn("anchor_stale", tracer._anon_turn_sessions)
        self.assertIn("fresh_running", tracer._active_runs)
        self.assertIn("new_session", tracer._active_runs)

        tracer_disabled = AgentLogTracer(disk_enabled=False, stale_active_ttl_s=0)
        tracer_disabled.start_turn("old_running", "q")
        tracer_disabled._active_runs["old_running"]["updated_at"] = time.time() - 999999.0
        tracer_disabled.start_turn("another", "q2")
        self.assertIn("old_running", tracer_disabled._active_runs)


if __name__ == "__main__":
    unittest.main()
