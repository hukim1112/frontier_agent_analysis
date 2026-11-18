"""Regression and integration tests for app/tools/plan.py and tool-level planning guardrails."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field, ValidationError

import app.utils.workspace as workspace_mod
from app.context import AgentContext
from app.tools.plan import (
    check_plan_mode_tool_allowed,
    clear_plan_state,
    enter_plan,
    exit_plan,
    get_plan_history,
    get_plan_state,
    is_planning_mode_active,
    plan_session_scope,
    task_board,
    task_create,
    task_list,
    task_update,
)
from app.tools.supervisor_tools import invoke_sub_agent
from app.utils.workspace import resolve_session_workspace


class ScriptedToolModel(BaseChatModel):
    """Deterministic fake chat model returning pre-scripted AIMessage responses."""

    responses: list[AIMessage]
    call_index: int = 0
    seen_tool_messages: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content not in self.seen_tool_messages:
                self.seen_tool_messages.append(str(msg.content))
        reply = self.responses[self.call_index]
        self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


class PlanToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.users_root = Path(self.tmp.name) / "users"
        self.users_patch = patch.object(workspace_mod, "USERS_ROOT_DIR", self.users_root)
        self.users_patch.start()
        self.addCleanup(self.users_patch.stop)

    def test_claude_code_plan_mode_lifecycle_and_shared_paths(self):
        with plan_session_scope("sess_lifecycle", user_id="instructor_1"):
            enter_msg = enter_plan.invoke({"intent": "new", "goal": "Analyze market data", "plan_name": "Draft"})
            self.assertIn("[PLANNING MODE ENTERED]", enter_msg)
            self.assertIn("plan_state.json", enter_msg)
            self.assertTrue(is_planning_mode_active())

            state = get_plan_state()
            self.assertTrue(state["active"])
            self.assertEqual(state["phase"], "PLANNING")

            # Planning Mode 중 task_create 차단
            blocked = task_create.invoke({"task_description": "Should be blocked"})
            self.assertIn("[PLANNING MODE GUARDRAIL]", blocked)
            self.assertEqual(task_board.list_tasks(), [])

            # 빈 steps로 exit_plan 호출 시 거부 및 PLANNING 유지
            empty_exit = exit_plan.invoke({"plan_name": "Final Plan", "steps": ["  "]})
            self.assertIn("[ERROR]", empty_exit)
            self.assertTrue(is_planning_mode_active())

            # 유효한 steps로 exit_plan 호출 -> EXECUTING 전환 및 파일 경로 안내
            exit_msg = exit_plan.invoke({
                "plan_name": "Market Analysis",
                "steps": ["Inspect JSON schema", "Compute value score", "Write report"],
            })
            self.assertIn("[PLANNING MODE EXITED]", exit_msg)
            self.assertIn("plan_state.json", exit_msg)
            self.assertIn("task_state.json", exit_msg)
            self.assertFalse(is_planning_mode_active())

            exec_state = get_plan_state()
            self.assertEqual(exec_state["phase"], "EXECUTING")
            self.assertEqual(len(exec_state["steps"]), 3)

            # 이제 task_create 허용
            t1_msg = task_create.invoke({"task_description": "Delegate market analysis"})
            self.assertIn("[TASK CREATED] Task #1", t1_msg)
            self.assertIn("task_state.json", t1_msg)

            # task_list는 plan 요약 + tasks + 파일 경로가 담긴 유효한 JSON 반환
            board_payload = json.loads(task_list.invoke({}))
            self.assertTrue(board_payload["plan_file"].endswith("plan_state.json"))
            self.assertTrue(board_payload["task_file"].endswith("task_state.json"))
            self.assertEqual(board_payload["plan"]["phase"], "EXECUTING")
            self.assertEqual(board_payload["plan"]["plan_name"], "Market Analysis")
            self.assertEqual(len(board_payload["tasks"]), 1)

            # 모든 태스크 COMPLETED 시 plan_state["phase"]도 COMPLETED로 자동 동기화
            upd_msg = task_update.invoke({"task_id": 1, "status": "COMPLETED"})
            self.assertIn("[TASK UPDATED] Task #1 status changed to 'COMPLETED'", upd_msg)
            self.assertIn("task_state.json", upd_msg)
            self.assertEqual(get_plan_state()["phase"], "COMPLETED")

    def test_sanitize_unification_for_email_user_id(self):
        email_uid = "hyoun@example.com"
        expected_dir = resolve_session_workspace(
            user_id=email_uid,
            session_id="sess_email",
            ensure_exists=False,
        ).plan_dir

        with plan_session_scope("sess_email", user_id=email_uid):
            enter_plan.invoke({"intent": "new", "goal": "Check email sanitization"})
            exit_plan.invoke({"plan_name": "Email Test", "steps": ["Step 1"]})
            task_create.invoke({"task_description": "Verify workspace path"})

        self.assertTrue((expected_dir / "plan_state.json").exists())
        self.assertTrue((expected_dir / "task_state.json").exists())

    def test_multiturn_archiving_vs_backtracking_and_status_normalization(self):
        with plan_session_scope("sess_turns", user_id="u_turns"):
            # Turn 1: 정상 완료
            enter_plan.invoke({"intent": "new", "goal": "Turn 1 goal"})
            exit_plan.invoke({"plan_name": "Turn 1 Plan", "steps": ["Scrape quotes"]})
            task_create.invoke({"task_description": "Scrape page 1"})
            task_update.invoke({"task_id": 1, "status": "COMPLETED"})
            self.assertEqual(get_plan_state()["phase"], "COMPLETED")

            # Turn 2: 완료 후 새 enter_plan -> Turn 1 아카이빙(new_plan) + task_state 리셋
            msg_turn2 = enter_plan.invoke({"intent": "new", "goal": "Turn 2 goal"})
            self.assertIn("archived to plan_history.json", msg_turn2)
            self.assertEqual(task_board.list_tasks(), [])

            history = get_plan_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["reason"], "new_plan")
            self.assertEqual(history[0]["plan"]["plan_name"], "Turn 1 Plan")
            self.assertEqual(len(history[0]["tasks"]), 1)

            exit_plan.invoke({"plan_name": "Turn 2 Plan", "steps": ["Read primary catalog"]})
            task_create.invoke({"task_description": "Analyze primary catalog"})
            task_update.invoke({"task_id": 1, "status": "BLOCKED"})

            # Turn 2 중 백트래킹: intent='revise'로 재진입 -> 기존 Task #1(BLOCKED) 보존
            msg_backtrack = enter_plan.invoke({"intent": "revise", "goal": "Switch to backup catalog"})
            self.assertIn("preserved 1 existing task(s)", msg_backtrack)
            tasks_during_backtrack = task_board.list_tasks()
            self.assertEqual(len(tasks_during_backtrack), 1)
            self.assertEqual(tasks_during_backtrack[0]["status"], "BLOCKED")

            history = get_plan_history()
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["reason"], "backtrack_replan")

            exit_plan.invoke({"plan_name": "Turn 2 Revised", "steps": ["Read backup catalog"]})
            task_create.invoke({"task_description": "Analyze backup catalog"})

            # 'done' 소문자 변형도 'COMPLETED'로 정규화
            norm_msg = task_update.invoke({"task_id": 2, "status": "done"})
            self.assertIn("'COMPLETED'", norm_msg)
            self.assertEqual(task_board.list_tasks()[1]["status"], "COMPLETED")

            # 잘못된 status는 거부
            with self.assertRaises(ValidationError):
                task_update.invoke({"task_id": 2, "status": "INVALID_STATE"})

            # Turn 3: 마지막 태스크가 COMPLETED이므로 Turn 2 전체 아카이빙 후 Task #1부터 재시작
            enter_plan.invoke({"intent": "new", "goal": "Turn 3 goal"})
            self.assertEqual(task_board.list_tasks(), [])
            history = get_plan_history()
            self.assertEqual(len(history), 3)
            self.assertEqual(history[2]["reason"], "new_plan")
            self.assertEqual([t["status"] for t in history[2]["tasks"]], ["BLOCKED", "COMPLETED"])

    def test_exit_plan_requires_planning_mode(self):
        with plan_session_scope("sess_strict_exit", user_id="u_strict"):
            # 계획 모드가 아닐 때(IDLE) exit_plan 거부
            idle_exit = exit_plan.invoke({"plan_name": "Skip", "steps": ["s1"]})
            self.assertIn("[ERROR] Not in Planning Mode", idle_exit)
            self.assertEqual(get_plan_state()["phase"], "IDLE")

            # 완료된 턴 이후 enter_plan 없이 exit_plan 호출 시 거부 + 기존 계획/태스크/히스토리 불변
            enter_plan.invoke({"intent": "new", "goal": "turn1"})
            exit_plan.invoke({"plan_name": "Turn1", "steps": ["s1"]})
            task_create.invoke({"task_description": "t1"})
            task_update.invoke({"task_id": 1, "status": "COMPLETED"})
            skipped = exit_plan.invoke({"plan_name": "Turn2", "steps": ["new"]})
            self.assertIn("[ERROR] Not in Planning Mode", skipped)
            self.assertEqual(get_plan_state()["plan_name"], "Turn1")
            self.assertEqual(len(task_board.list_tasks()), 1)
            self.assertEqual(get_plan_history(), [])

    def test_enter_plan_intent_new_abandons_blocked_and_revise_requires_plan(self):
        with plan_session_scope("sess_intent", user_id="u_intent"):
            # intent는 필수 인자
            with self.assertRaises(ValidationError):
                enter_plan.invoke({"goal": "missing intent"})

            # 수정할 계획이 없으면 revise 거부
            no_plan = enter_plan.invoke({"intent": "revise", "goal": "nothing to revise"})
            self.assertIn("[ERROR] There is no current plan to revise", no_plan)
            self.assertFalse(is_planning_mode_active())

            # BLOCKED 상태로 남겨진 계획을 버리고 새 요청 시작 -> abandoned로 보관 후 리셋
            enter_plan.invoke({"intent": "new", "goal": "old"})
            exit_plan.invoke({"plan_name": "Old", "steps": ["old step"]})
            task_create.invoke({"task_description": "old task"})
            task_update.invoke({"task_id": 1, "status": "BLOCKED"})

            msg = enter_plan.invoke({"intent": "new", "goal": "brand new unrelated request"})
            self.assertIn("reason: abandoned", msg)
            state = get_plan_state()
            self.assertEqual(state["plan_name"], "")
            self.assertEqual(state["steps"], [])
            self.assertEqual(state["goal"], "brand new unrelated request")
            self.assertEqual(task_board.list_tasks(), [])
            history = get_plan_history()
            self.assertEqual(history[-1]["reason"], "abandoned")
            self.assertEqual(history[-1]["tasks"][0]["status"], "BLOCKED")

    def test_single_shared_lock_prevents_lock_order_inversion(self):
        from app.tools import plan as plan_mod
        self.assertIs(plan_mod.task_board._lock, plan_mod._plan_lock)
        self.assertIs(plan_mod.TaskBoardState()._lock, plan_mod._plan_lock)

    def test_edge_cases_revise_completed_default_goal_abandoned_without_tasks_and_exit_order(self):
        # 1) ①-(b): 기본 인자(goal="") 진입 직후 revise 호출 허용 + 빈 PLANNING 껍데기 replace 시 히스토리 미기록
        with plan_session_scope("sess_edge_empty", user_id="u_edge"):
            enter_plan.invoke({"intent": "new"})
            rev_msg = enter_plan.invoke({"intent": "revise", "goal": "Refined goal"})
            self.assertIn("[PLANNING MODE ENTERED]", rev_msg)
            self.assertEqual(get_plan_state()["goal"], "Refined goal")
            self.assertEqual(get_plan_history(), [])

        with plan_session_scope("sess_edge_shell", user_id="u_edge"):
            enter_plan.invoke({"intent": "new"})
            enter_plan.invoke({"intent": "new", "goal": "Fresh"})
            self.assertEqual(get_plan_history(), [])

        # 2) ②: 태스크 생성 없이 EXECUTING에서 버려진 계획 -> reason='abandoned'
        with plan_session_scope("sess_edge_abandoned", user_id="u_edge"):
            enter_plan.invoke({"intent": "new", "goal": "Draft"})
            exit_plan.invoke({"plan_name": "Unexecuted", "steps": ["s1"]})
            msg_exec_abandoned = enter_plan.invoke({"intent": "new", "goal": "Next"})
            self.assertIn("reason: abandoned", msg_exec_abandoned)
            history = get_plan_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["reason"], "abandoned")
            self.assertEqual(history[0]["plan"]["plan_name"], "Unexecuted")

            # 내용이 있는 PLANNING 초안을 버리고 새 계획 시작 시에도 reason='abandoned'
            msg_plan_abandoned = enter_plan.invoke({"intent": "new", "goal": "Replace draft"})
            self.assertIn("reason: abandoned", msg_plan_abandoned)
            history = get_plan_history()
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["reason"], "abandoned")

        # 3) ①-(a): COMPLETED 상태에서 intent='revise' 거부 + 기존 상태 보존
        with plan_session_scope("sess_edge_completed", user_id="u_edge"):
            enter_plan.invoke({"intent": "new", "goal": "Do task"})
            exit_plan.invoke({"plan_name": "Done Plan", "steps": ["step 1"]})
            task_create.invoke({"task_description": "work"})
            task_update.invoke({"task_id": 1, "status": "COMPLETED"})
            self.assertEqual(get_plan_state()["phase"], "COMPLETED")

            err = enter_plan.invoke({"intent": "revise", "goal": "Follow-up"})
            self.assertIn("[ERROR] The previous plan is already completed", err)
            self.assertEqual(get_plan_state()["phase"], "COMPLETED")
            self.assertEqual(get_plan_state()["plan_name"], "Done Plan")
            self.assertEqual(len(task_board.list_tasks()), 1)

            # intent='new'로 전환 시 정상 아카이빙 및 새 계획 시작
            ok = enter_plan.invoke({"intent": "new", "goal": "Follow-up"})
            self.assertIn("reason: new_plan", ok)
            self.assertEqual(task_board.list_tasks(), [])

        # 4) ④: IDLE 상태에서 빈 steps로 exit_plan 호출 시 'Not in Planning Mode' 우선 반환
        with plan_session_scope("sess_edge_idle_exit", user_id="u_edge"):
            idle_empty = exit_plan.invoke({"plan_name": "X", "steps": []})
            self.assertIn("[ERROR] Not in Planning Mode", idle_empty)

        # 5) 백트래킹 복구 완료 시(Task #1 BLOCKED, Task #2 COMPLETED) plan_state['phase'] -> 'COMPLETED'
        with plan_session_scope("sess_edge_backtrack_complete", user_id="u_edge"):
            enter_plan.invoke({"intent": "new", "goal": "Recover"})
            exit_plan.invoke({"plan_name": "Try 1", "steps": ["a"]})
            task_create.invoke({"task_description": "primary"})
            task_update.invoke({"task_id": 1, "status": "BLOCKED"})
            enter_plan.invoke({"intent": "revise", "goal": "Use fallback"})
            exit_plan.invoke({"plan_name": "Try 2", "steps": ["b"]})
            task_create.invoke({"task_description": "backup"})
            task_update.invoke({"task_id": 2, "status": "COMPLETED"})
            self.assertEqual(get_plan_state()["phase"], "COMPLETED")

    def test_scope_nesting_isolation_and_readonly_no_disk_side_effects(self):
        # 읽고 조회만 할 때는 디렉터리를 생성하지 않아야 함
        ghost_dir = resolve_session_workspace(
            user_id="ghost_user",
            session_id="never_created",
            ensure_exists=False,
        ).plan_dir
        with plan_session_scope("never_created", user_id="ghost_user"):
            self.assertFalse(is_planning_mode_active())
            self.assertEqual(get_plan_state()["phase"], "IDLE")
            self.assertEqual(get_plan_history(), [])
            empty_msg = task_list.invoke({})
            self.assertIn("No active plan or tasks", empty_msg)
        self.assertFalse(ghost_dir.parent.exists())

        # 중첩 스코프: 서브에이전트 스코프 내부에서 parent_session_id=None인 메인 스코프 진입 시
        # 외부 parent_session_id를 잘못 상속하지 않아야 함
        sup_dir = resolve_session_workspace(user_id="u1", session_id="sup_1").plan_dir
        sub_dir = resolve_session_workspace(
            user_id="u1", session_id="sub_1", parent_session_id="sup_1"
        ).plan_dir

        with plan_session_scope("sub_1", user_id="u1", parent_session_id="sup_1"):
            enter_plan.invoke({"intent": "new", "goal": "Subagent local plan"})
            with plan_session_scope("sup_1", user_id="u1", parent_session_id=None):
                enter_plan.invoke({"intent": "new", "goal": "Supervisor plan"})
                exit_plan.invoke({"plan_name": "Sup Plan", "steps": ["Step 1"]})
            # 내부 스코프 종료 후 외부 서브에이전트 스코프 정상 복원 확인
            self.assertTrue(is_planning_mode_active())
            self.assertEqual(get_plan_state()["goal"], "Subagent local plan")

        self.assertTrue((sup_dir / "plan_state.json").exists())
        self.assertTrue((sub_dir / "plan_state.json").exists())
        sup_in_sub = sup_dir.parent.parent / "subagents" / "sup_1"
        self.assertFalse(sup_in_sub.exists())

    def test_agent_loop_sync_blocks_task_create_and_subagent_until_exit_plan(self):
        executed = {"glob": 0, "subagent": 0}

        @tool
        def glob_search(pattern: str) -> str:
            """Discover files during read-only planning."""
            executed["glob"] += 1
            return f"FILES:{pattern}"

        @tool
        def custom_invoke_sub_agent(instruction: str) -> str:
            """Delegate task to a sub-agent (with tool-level planning guardrail)."""
            if msg := check_plan_mode_tool_allowed("invoke_sub_agent"):
                return msg
            executed["subagent"] += 1
            return f"DELEGATED:{instruction}"

        model = ScriptedToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "enter_plan", "args": {"intent": "new", "goal": "Delegate safely"}, "id": "c1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "glob_search", "args": {"pattern": "*.json"}, "id": "c2"},
                        {"name": "task_create", "args": {"task_description": "Blocked task"}, "id": "c3"},
                        {"name": "custom_invoke_sub_agent", "args": {"instruction": "Blocked call"}, "id": "c4"},
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "exit_plan",
                            "args": {"plan_name": "Delegation Plan", "steps": ["Delegate task"]},
                            "id": "c5",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "task_create", "args": {"task_description": "Allowed task"}, "id": "c6"},
                        {"name": "custom_invoke_sub_agent", "args": {"instruction": "Allowed call"}, "id": "c7"},
                    ],
                ),
                AIMessage(content="All done."),
            ]
        )

        agent = create_agent(
            model=model,
            tools=[enter_plan, exit_plan, task_create, task_list, task_update, glob_search, custom_invoke_sub_agent],
            system_prompt="Test Supervisor",
            context_schema=AgentContext,
        )

        ctx = AgentContext(agent_name="supervisor", session_id="sync_guard_01", user_id="guard_user")
        result = agent.invoke(
            {"messages": [("user", "Plan and delegate")]},
            config={"configurable": {"thread_id": "sync_guard_01", "user_id": "guard_user"}},
            context=ctx,
        )

        self.assertEqual(executed["glob"], 1)
        self.assertEqual(executed["subagent"], 1)
        self.assertGreaterEqual(
            sum("[PLANNING MODE GUARDRAIL]" in m for m in model.seen_tool_messages),
            2,
        )
        with plan_session_scope("sync_guard_01", user_id="guard_user"):
            tasks = task_board.list_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["description"], "Allowed task")
        self.assertEqual(result["messages"][-1].content, "All done.")

    async def test_supervisor_invoke_sub_agent_guardrail_direct_and_async_agent(self):
        fake_delegation_calls = []

        class StubAgentClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_agents(self):
                return [{"name": "scraper", "description": "Scraper", "roles": ["main", "sub"]}]

            async def get_messages(self, session_id: str):
                return []

            async def async_invoke(self, agent_name: str, message: str, thread_id: str = None, **kwargs):
                fake_delegation_calls.append({"agent_name": agent_name, "thread_id": thread_id})
                return {"type": "ai", "content": "[TASK REPORT]\n- Status: SUCCESS"}

        with patch("app.tools.supervisor_tools.AsyncAgentClient", StubAgentClient):
            runtime = SimpleNamespace(
                execution_info=SimpleNamespace(thread_id="sup_guard_01"),
                context=SimpleNamespace(agent_name="supervisor", user_id="guard_u1"),
            )

            # 1) 직접 invoke_sub_agent 호출 시 Planning Mode면 서버 조회 전에 즉시 차단
            with plan_session_scope("sup_guard_01", user_id="guard_u1"):
                enter_plan.invoke({"intent": "new", "goal": "Test direct guardrail"})
                blocked_res = await invoke_sub_agent.coroutine(
                    task_instruction="Scrape site",
                    target_file_list=["artifacts/data/out.json"],
                    subagent_role="scraper",
                    runtime=runtime,
                )
                self.assertIn("[PLANNING MODE GUARDRAIL]", blocked_res)
                self.assertEqual(fake_delegation_calls, [])
                clear_plan_state()

            # 2) Async create_agent에서 도구 레벨 가드레일로 차단 후 exit_plan 뒤 위임 성공
            model = ScriptedToolModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "enter_plan", "args": {"intent": "new", "goal": "Scrape safely"}, "id": "a1"}],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "invoke_sub_agent",
                                "args": {
                                    "task_instruction": "Scrape site",
                                    "target_file_list": ["out.json"],
                                    "subagent_role": "scraper",
                                },
                                "id": "a2",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "exit_plan",
                                "args": {"plan_name": "Scrape Plan", "steps": ["Run scraper"]},
                                "id": "a3",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "invoke_sub_agent",
                                "args": {
                                    "task_instruction": "Scrape site",
                                    "target_file_list": ["out.json"],
                                    "subagent_role": "scraper",
                                },
                                "id": "a4",
                            }
                        ],
                    ),
                    AIMessage(content="Delegation completed."),
                ]
            )

            agent = create_agent(
                model=model,
                tools=[enter_plan, exit_plan, task_create, task_list, task_update, invoke_sub_agent],
                system_prompt="Async Test Supervisor",
                context_schema=AgentContext,
            )

            ctx = AgentContext(agent_name="supervisor", session_id="sup_async_01", user_id="guard_u1")
            res = await agent.ainvoke(
                {"messages": [("user", "Scrape site")]},
                config={"configurable": {"thread_id": "sup_async_01", "user_id": "guard_u1"}},
                context=ctx,
            )

            self.assertEqual(len(fake_delegation_calls), 1)
            self.assertTrue(any("[PLANNING MODE GUARDRAIL]" in m for m in model.seen_tool_messages))
            self.assertEqual(res["messages"][-1].content, "Delegation completed.")

    def test_plan_tools_aci_spec_and_tool_call_error_status(self):
        for tool_obj in (enter_plan, exit_plan, task_create, task_list, task_update):
            self.assertTrue(tool_obj.handle_tool_error, f"{tool_obj.name} handle_tool_error")
            desc = tool_obj.description or ""
            self.assertNotIn("Args:", desc, f"{tool_obj.name} duplicated Args:")
            self.assertNotIn("Returns:", desc, f"{tool_obj.name} duplicated Returns:")
            self.assertFalse(
                desc.startswith((" ", "\t", "\n")),
                f"{tool_obj.name} description has leading whitespace",
            )

        with plan_session_scope("sess_aci_errors", user_id="u_aci"):
            enter_plan.invoke({"intent": "new", "goal": "Verify tool error status"})

            blocked_msg = task_create.invoke(
                {
                    "type": "tool_call",
                    "id": "call_plan_blocked",
                    "name": task_create.name,
                    "args": {"task_description": "Blocked while planning"},
                }
            )
            self.assertIsInstance(blocked_msg, ToolMessage)
            self.assertEqual(blocked_msg.status, "error")
            self.assertIn("[PLANNING MODE GUARDRAIL]", str(blocked_msg.content))

            exit_plan.invoke({"plan_name": "ACI Plan", "steps": ["Step 1"]})
            task_create.invoke({"task_description": "Real Task #1"})

            missing_msg = task_update.invoke(
                {
                    "type": "tool_call",
                    "id": "call_missing_task",
                    "name": task_update.name,
                    "args": {"task_id": 99, "status": "COMPLETED"},
                }
            )
            self.assertIsInstance(missing_msg, ToolMessage)
            self.assertEqual(missing_msg.status, "error")
            self.assertIn("Valid task IDs on board: [1]", str(missing_msg.content))


if __name__ == "__main__":
    unittest.main()
