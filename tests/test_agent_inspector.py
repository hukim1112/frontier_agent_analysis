import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiosqlite
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.context import AgentContext
from app.middleware.error_control.self_recovery import ToolErrorHandlerMiddleware
from app.middleware.observability import AgentLogTracer
from app.middleware.prompt import PromptAssembler, create_prompt_assembler_middleware
from tests.agent_inspector import (
    close_agent_resources,
    extract_agent_middlewares,
    extract_agent_tools,
    extract_prompt_assembler,
    find_agent_middleware,
)


class _DummyChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy-inspector"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


class AgentInspectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_tools_middlewares_assembler_and_close_resources(self):
        """CompiledStateGraph에서 도구, 노드형/래퍼형 미들웨어, PromptAssembler, DB 커넥션을 정확히 추출/종료하는지 검증."""
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = await aiosqlite.connect(str(Path(tmpdir) / "cp.db"), check_same_thread=False)
            cp = AsyncSqliteSaver(conn)
            await cp.setup()

            @tool
            def sample_ping(x: str) -> str:
                """핑 도구."""
                return x

            tracer = AgentLogTracer(disk_enabled=False)
            assembler = PromptAssembler(system_rules="Test rules")
            prompt_mw = create_prompt_assembler_middleware(assembler, merge_system=True)
            err_mw = ToolErrorHandlerMiddleware(max_retries=0)

            agent = create_agent(
                model=_DummyChatModel(),
                tools=[sample_ping],
                middleware=[tracer, prompt_mw, err_mw],
                checkpointer=cp,
                context_schema=AgentContext,
            )

            tools = extract_agent_tools(agent)
            self.assertEqual([t.name for t in tools], ["sample_ping"])

            mws = extract_agent_middlewares(agent)
            mw_types = {type(m) for m in mws}
            self.assertIn(AgentLogTracer, mw_types)
            self.assertIn(type(prompt_mw), mw_types)
            self.assertIn(ToolErrorHandlerMiddleware, mw_types)

            self.assertIs(find_agent_middleware(agent, AgentLogTracer), tracer)
            self.assertIsNone(extract_agent_middlewares(None) or None)
            self.assertIs(extract_prompt_assembler(agent), assembler)

            self.assertTrue(getattr(conn, "_running", False))
            await close_agent_resources(agent)
            self.assertFalse(getattr(conn, "_running", True))

            # 이미 닫힌 커넥션에 대해 재호출해도 안전(idempotent)해야 함
            await close_agent_resources(agent)

    async def test_subagent_and_chatbot_factories_have_no_monkeypatched_attributes(self):
        """에이전트 팩토리 반환값에 테스트용 수동 속성들이 붙어 있지 않고 인트로스펙션으로 조회 가능한지 검증."""
        from app.agents import chatbot, main_agent
        from app.agents.analyst import agent as analyst_mod
        from app.agents.scraper import agent as scraper_mod
        from app.middleware.prompt import MCPServerRegistry

        forbidden_attrs = (
            "registered_tools",
            "registered_middlewares",
            "checkpointer_conn",
            "assembler",
            "memory_middleware",
            "episodic_store",
            "transcript_store",
            "semantic_store",
            "procedural_store",
            "mcp_registry",
            "log_tracer",
        )

        async def _noop_mcp_connect(registry_self):
            return registry_self

        with tempfile.TemporaryDirectory() as tmpdir:
            conn = await aiosqlite.connect(str(Path(tmpdir) / "shared_cp.db"), check_same_thread=False)
            main_exec = None
            try:
                cp = AsyncSqliteSaver(conn)
                await cp.setup()

                with patch.object(MCPServerRegistry, "connect", _noop_mcp_connect):
                    bot_exec = await chatbot.create_agent_executor(checkpointer=cp)
                    scraper_exec = await scraper_mod.create_agent_executor(checkpointer=cp)
                    analyst_exec = await analyst_mod.create_agent_executor(checkpointer=cp)
                    main_exec = await main_agent.create_agent_executor(checkpointer=cp)

                for exec_obj in (bot_exec, scraper_exec, analyst_exec, main_exec):
                    for attr in forbidden_attrs:
                        self.assertFalse(
                            hasattr(exec_obj, attr),
                            f"Unexpected monkey-patched attribute '{attr}' on {exec_obj}",
                        )
                for exec_obj in (bot_exec, scraper_exec, analyst_exec, main_exec):
                    self.assertGreater(len(extract_agent_tools(exec_obj)), 0)

                self.assertIsNotNone(extract_prompt_assembler(scraper_exec))
                self.assertIsNotNone(extract_prompt_assembler(analyst_exec))
                self.assertIsNotNone(find_agent_middleware(scraper_exec, AgentLogTracer))
                self.assertIsNotNone(find_agent_middleware(analyst_exec, AgentLogTracer))
            finally:
                if main_exec is not None:
                    await close_agent_resources(main_exec, review_timeout=1.0, shutdown_review_queue=True)
                if getattr(conn, "_running", False):
                    await conn.close()

    async def test_memory_middleware_aclose_all_instances(self):
        """MemoryMiddleware.aclose_all_instances()가 WeakSet을 통해 스토어/리뷰큐를 안전하게 종료하는지 검증."""
        from app.middleware.memory import (
            EpisodicStore,
            MemoryMiddleware,
            SemanticMemoryStore,
            TranscriptStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sem = SemanticMemoryStore(memory_dir=str(root / "mem"))
            sem.load_from_disk()
            tr = TranscriptStore(db_path=str(root / "transcript.db"))
            await tr.setup()
            epi = EpisodicStore(db_path=str(root / "episodic.db"), transcript_store=tr)
            await epi.setup()

            mw = MemoryMiddleware(semantic_store=sem, episodic_store=epi, transcript_store=tr)
            self.assertIsNotNone(getattr(tr, "_conn", None))
            self.assertIsNotNone(getattr(epi, "_conn", None))

            await MemoryMiddleware.aclose_all_instances(review_timeout=1.0)
            self.assertIsNone(getattr(tr, "_conn", None))
            self.assertIsNone(getattr(epi, "_conn", None))
            # 재호출 idempotency 확인
            await MemoryMiddleware.aclose_all_instances(review_timeout=1.0)
            del mw

    async def test_shared_checkpointer_recovers_after_close_agent_resources(self):
        """close_agent_resources()가 공유 checkpointer 커넥션을 닫아도 get_shared_checkpointer()가 복구하는지 검증."""
        from app.utils.database import close_shared_checkpointer, get_shared_checkpointer

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "shared_cp.db")
            cp1 = await get_shared_checkpointer(db_path=db_path)
            self.assertTrue(getattr(cp1.conn, "_running", False))

            agent = create_agent(
                model=_DummyChatModel(),
                tools=[],
                checkpointer=cp1,
                context_schema=AgentContext,
            )
            await close_agent_resources(agent)
            self.assertFalse(getattr(cp1.conn, "_running", True))

            cp2 = await get_shared_checkpointer(db_path=db_path)
            self.assertIsNot(cp2, cp1)
            self.assertTrue(getattr(cp2.conn, "_running", False))
            await close_shared_checkpointer()

    async def test_shared_checkpointer_lock_serialization_with_close(self):
        """get_shared_checkpointer()와 close_shared_checkpointer() 동시 호출 시 캐시와 커넥션 상태가 일관되게 직렬화되는지 검증."""
        from app.utils.database import session_store
        from app.utils.database.session_store import (
            _is_conn_alive,
            close_shared_checkpointer,
            get_shared_checkpointer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "race_cp.db")
            abs_db = session_store._resolve_db_path(db_path)
            loop_id = id(asyncio.get_running_loop())
            try:
                cp_raced, _ = await asyncio.gather(
                    get_shared_checkpointer(db_path=db_path),
                    close_shared_checkpointer(),
                )
                cached_conn = session_store._checkpointer_conn_by_key.get((loop_id, abs_db))
                if cached_conn is not None:
                    self.assertTrue(_is_conn_alive(cached_conn))
                    self.assertIs(cp_raced.conn, cached_conn)
                else:
                    self.assertFalse(_is_conn_alive(cp_raced.conn))

                _, cp_live = await asyncio.gather(
                    close_shared_checkpointer(),
                    get_shared_checkpointer(db_path=db_path),
                )
                self.assertTrue(_is_conn_alive(cp_live.conn))
                self.assertIs(
                    session_store._checkpointer_by_key.get((loop_id, abs_db)),
                    cp_live,
                )
            finally:
                await close_shared_checkpointer()
                self.assertEqual(
                    [k for k in session_store._checkpointer_conn_by_key if k[0] == loop_id],
                    [],
                )

    async def test_checkpoints_db_path_patch_unifies_session_crud_and_checkpointer(self):
        """CHECKPOINTS_DB_PATH만 패치해도 세션 CRUD와 공유 Checkpointer가 동일한 임시 DB 경로를 사용하는지 검증."""
        from app.utils.database import session_store
        from app.utils.database.session_store import (
            close_shared_checkpointer,
            create_session,
            get_session,
            get_shared_checkpointer,
        )

        orig_cp_path = session_store.CHECKPOINTS_DB_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = str(Path(tmpdir) / "patched_cp.db")
            abs_temp_db = session_store._resolve_db_path(temp_db)
            try:
                session_store.CHECKPOINTS_DB_PATH = temp_db
                created = await create_session("sess_patch_1", "chatbot", "Patched DB Session")
                self.assertEqual(created["session_id"], "sess_patch_1")

                fetched = await get_session("sess_patch_1")
                self.assertIsNotNone(fetched)
                self.assertEqual(fetched["title"], "Patched DB Session")

                cp = await get_shared_checkpointer()
                loop_id = id(asyncio.get_running_loop())
                self.assertIn((loop_id, abs_temp_db), session_store._checkpointer_by_key)
                self.assertIs(session_store._checkpointer_by_key[(loop_id, abs_temp_db)], cp)
                self.assertTrue(Path(abs_temp_db).exists())
            finally:
                await close_shared_checkpointer()
                session_store._initialized_db_paths.discard(abs_temp_db)
                session_store.CHECKPOINTS_DB_PATH = orig_cp_path


if __name__ == "__main__":
    unittest.main()
