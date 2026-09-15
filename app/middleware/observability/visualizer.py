"""
===============================================================================
[AAWS Middleware] Hierarchical Visualizer Middleware (visualizer.py)
===============================================================================
Supervisor(부모)와 Worker(자식) 에이전트의 다계층 실행 흐름을 터미널/노트북에
비침습적(Non-invasive)으로 시각화하는 LangChain 표준 AgentMiddleware 구현체.
===============================================================================
"""

import json
from langchain.agents.middleware import AgentMiddleware


class HierarchicalVisualizerMiddleware(AgentMiddleware):
    """Supervisor와 Worker 에이전트의 실행 과정을 계층적으로 시각화하는 미들웨어"""

    def __init__(self, agent_role: str = "Supervisor", is_subagent: bool = False):
        super().__init__()
        self.agent_role = agent_role
        self.is_subagent = is_subagent

    def before_agent(self, state, runtime) -> dict | None:
        """[Hook 1] 에이전트 실행 시작"""
        if self.is_subagent:
            print(f"\n    ┌── 👷 [Worker: {self.agent_role}] 격리 실행 루프 시작 ──────────")
        else:
            print(f"\n👑 [Supervisor: {self.agent_role}] 오케스트레이션 루프 시작")
        return None

    def after_model(self, state, runtime) -> dict | None:
        """[Hook 2] 모델 응답 직후 - 사고 과정(Thought) 출력"""
        if state.get("messages"):
            last_msg = state["messages"][-1]
            # 도구 호출이 포함된 경우 모델의 사고 텍스트 추출
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                thought = last_msg.content
                if thought and isinstance(thought, str) and thought.strip():
                    clean_thought = thought.strip().replace("\n", " ")
                    if len(clean_thought) > 100:
                        clean_thought = clean_thought[:97] + "..."
                    indent = "    │ 💭 [Worker Thought]" if self.is_subagent else "  💭 [Supervisor Thought]"
                    print(f"{indent} {clean_thought}")
        return None

    def wrap_tool_call(self, request, handler):
        """[Hook 3] 도구 실행 가로채기 - 도구 인자 및 결과 로깅"""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        args_str = json.dumps(tool_args, ensure_ascii=False)
        if len(args_str) > 85:
            args_str = args_str[:82] + "..."

        if self.is_subagent:
            print(f"    │ 🛠️  [Worker Tool Call] `{tool_name}`({args_str})")
        else:
            print(f"  🛠️  [Supervisor Tool Call] ➔ `{tool_name}`({args_str})")

        # 1. 실제 도구 핸들러 실행
        result = handler(request)

        # 2. 도구 실행 결과 추출 및 관찰 로깅
        res_content = str(result.content if hasattr(result, "content") else result).replace("\n", " ")
        if len(res_content) > 100:
            res_content = res_content[:97] + "..."

        if self.is_subagent:
            print(f"    │ 👁️  [Worker Observation] `{tool_name}`: {res_content}")
        else:
            print(f"  ✅ [Supervisor Tool Result] ➔ `{tool_name}`: {res_content}")

        return result

    def after_agent(self, state, runtime) -> dict | None:
        """[Hook 4] 에이전트 실행 완료"""
        if self.is_subagent:
            print(f"    └─────────────────────────────────────────────────────────────")
        else:
            print(f"\n👑 [Supervisor] 오케스트레이션 완료!\n")
        return None
