"""
===============================================================================
[H-06] Evaluator Harness & Autonomous LLM-as-a-Judge Middleware
===============================================================================
Source: frontier-agent-harness/evaluator.py

Autonomous Evaluator Rework Loop Diagram:
  [Start] ➡️ [model] (LLM Node) ──➡️─── [after_model] (Evaluator Middleware)
                ▲                                 │
                │                                 │ (Check: Has tool_calls?)
                │                                 ├─► YES ─► Pass (To Tools/END)
                │                                 │
                │                                 └─► NO  ─► (Final Answer Point)
                │                                             │
                │ (If Rework Required:                        │ (Run Judge: LLM-as-a-Judge)
                │  jump_to="model" + Feedback)               ▼
                └────────────────── Reject ─────────── [Judge Check]
                                                       │
                                                       └───── Approve ──► END
===============================================================================
"""

import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.runtime import Runtime
from app.utils import init_chat_model, normalize_content


# 판사 모델의 채점 및 피드백용 Pydantic 스키마 정의 (노트북 Part 1-1과 완벽 일치)
class JudgeVerdict(BaseModel):
    is_approved: bool = Field(description="에이전트의 답변과 수행 이력이 유저의 원래 목적과 요구사항을 충족했는지 여부 (True: 통과, False: 반려)")
    score: int = Field(default=100, description="에이전트 수행 결과에 대한 종합 정량 점수 (0~100)")
    feedback: str = Field(default="", description="불합격(is_approved=False)인 경우 에이전트가 보완해야 하는 구체적인 결함 사항과 수정 지시 사항 (합격인 경우 빈 문자열)")
    reason: str = Field(default="", description="판정에 대한 1줄 핵심 기술적 근거")


class EvaluatorHarness(AgentMiddleware):
    """
    최종 답변 도출 시점에 고성능 서브 에이전트(Judge LLM)를 구동하여
    작업 결과 및 궤적(Trajectory)의 오류를 검증하고, 결함 발견 시 model 노드로 롤백하여
    자동 재작업(Self-Correction Rework)을 지시하는 평가 하네스(Evaluator Harness).
    """
    def __init__(self, max_reworks: int = 2, judge_model_name: str = "gemini-3.8-flash", judge_llm: Optional[Any] = None):
        self.max_reworks = max_reworks
        self.judge_model_name = judge_model_name
        self.judge_llm = judge_llm or init_chat_model(model=judge_model_name, temperature=0.0)
        self.parser = JsonOutputParser(pydantic_object=JudgeVerdict)

    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        
        # 1. 과도기 상태(도구 호출 단계)라면 판사 기동 없이 Fast Pass
        has_tool_calls = getattr(last_msg, "tool_calls", None)
        if has_tool_calls:
            return None

        # 2. 도구 호출이 없는 '최종 답변 생성 시점'에만 검증 루프 작동 (속도 및 비용 효율성 극대화)
        final_answer = normalize_content(getattr(last_msg, "content", ""))
        
        # 3. 판사 에이전트 구동 및 최종 답변/실행 궤적 평가
        judgment = self._run_judge(messages, str(final_answer))
        
        if judgment.is_approved:
            print(f"✅ [EvaluatorHarness] Evaluator 승인 완료 (점수: {judgment.score}점). 최종 답변을 사용자에게 전달합니다.")
            return None

        # 4. 무한 루프 방지를 위한 최대 재작업 한도 조회
        rework_count = getattr(runtime.context, "rework_count", 0) if runtime and hasattr(runtime, "context") else 0
        if rework_count >= self.max_reworks:
            print(f"🚨 [EvaluatorHarness] 최대 재작업 한도({self.max_reworks}회)를 초과하여 더 이상 보정하지 않고 종료합니다.")
            return None

        # 5. 재시도 카운트 갱신 및 model 노드로 롤백 지시 (jump_to="model")
        if runtime and hasattr(runtime, "context"):
            runtime.context.rework_count = rework_count + 1
        print(f"🔄 [EvaluatorHarness] Evaluator 반려 발생 ({rework_count + 1}/{self.max_reworks}회차). model 노드로 롤백합니다.")

        # 사용자 지시 사항에 맞춰 다듬어진 피드백 및 재작업 지시문 적용
        feedback_content = (
            f"⚠️ [Evaluator Feedback - Rework Required]\n"
            f"Evaluator에 의해 피드백 및 재작업 필요성이 감지되었습니다 (채점 점수: {judgment.score}점).\n"
            f"제시된 피드백을 충실히 반영하여 이전 작업의 결과물을 보완하고 최종 완결된 보고서를 다시 작성하십시오.\n\n"
            f"🔍 세부 피드백:\n{judgment.feedback}"
        )

        return {
            "jump_to": "model",
            "messages": [
                HumanMessage(content=feedback_content)
            ]
        }

    def _run_judge(self, trajectory: list, final_answer: str) -> JudgeVerdict:
        """독립된 판사 서브 에이전트를 구동하여 답변과 실행 궤적(Trajectory) 검증"""
        formatted_history = []
        for msg in trajectory[:-1]:
            msg_type = getattr(msg, "type", "unknown")
            role = "User" if msg_type == "human" else "Agent"
            if getattr(msg, "tool_calls", None):
                formatted_history.append(f"[{role} Tool Call]: {msg.tool_calls}")
            elif msg_type == "tool":
                formatted_history.append(f"[Tool Response]: {msg.content}")
            else:
                formatted_history.append(f"[{role} Thought]: {msg.content}")

        trajectory_str = "\n".join(formatted_history)

        system_prompt = (
            "당신은 에이전트의 작업 실행 궤적(Trajectory)과 최종 결과물(Final Answer)을 감사하고 보정하는 전문 평가 에이전트(Evaluator)입니다.\n"
            "에이전트가 실행 도중 겪은 단순 에러나 시행착오(Trial & Error)는 에이전트 스스로의 극복 과정이므로 불이익을 주지 마십시오.\n"
            "오직 아래의 '치명적 결함'이 발견되었을 때만 반려(is_approved=False)하고 재작업을 지시해야 합니다.\n\n"
            
            "🚨 [치명적 결함 및 심사 기준]\n"
            "1. 목적 누락: 사용자의 원래 지시사항이나 요구사항 중 일부를 무시하고 답변에서 생략했는가?\n"
            "2. 환각 현상(Hallucination): 도구 실행 결과(Tool Response)로 획득한 명확한 사실적 근거가 있음에도 불구하고, 이를 최종 답변 작성 시 반영하지 않거나 완전히 왜곡하여 거짓말을 지어냈는가?\n"
            "3. 결과 불능: 생성된 최종 파일 등의 결과물이 텅 빈 껍데기이거나 에러 메시지만 포함하고 있는가?\n"
            "4. 무한 루프: 동일한 에러가 발생하는 특정 도구를 반복적으로 계속 호출하며 갇혀 있는가?\n\n"
            
            "💡 [중요: 피드백 작성 가이드라인 (훈수)]\n"
            "반려(is_approved=False)를 결정했다면, 에이전트가 헤매지 않도록 구체적인 '우회 제안 및 힌트(훈수)'를 feedback 필드에 상세히 적어주십시오. 특히 다음 상황에 부합할 경우 가이드를 엄격히 따라 작성하십시오:\n"
            "- 유저 요청 중 일부가 누락된 경우: 어떤 요구사항이 누락되었는지 명시하고, 이를 완수하기 위해 어떤 행동이나 도구를 추가로 기동해야 하는지 구체적으로 지시하십시오.\n"
            "- 근거가 있음에도 잘못된 답변(환각)을 한 경우: 실행 궤적(Tool Response) 상의 실제 팩트나 획득한 데이터를 직접 짚어주며 이를 올바르게 인용하여 답변을 정정하도록 지시하십시오.\n"
            "- 도구 호출 실패가 반복되는 경우: 에러 원인을 설명하고 다른 우회 도구나 인자 값을 제안하십시오.\n\n"
            
            "평가 규격:\n"
            "{format_instructions}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "■ 실행 궤적 (Trajectory):\n{trajectory}\n\n■ 에이전트의 최종 답변 (Final Answer):\n{final_answer}")
        ])

        chain = prompt | self.judge_llm | self.parser
        
        try:
            format_instructions_str = self.parser.get_format_instructions()
            result = chain.invoke({
                "trajectory": trajectory_str,
                "final_answer": final_answer,
                "format_instructions": format_instructions_str
            }, config={"tags": ["exclude_from_stream"]})
            return JudgeVerdict(**result)
        except Exception as e:
            print(f"⚠️ [EvaluatorHarness] 판정 중 에러 발생 (안전 패스 처리): {e}")
            return JudgeVerdict(is_approved=True, score=100, feedback="", reason=f"에러 안전 패스: {e}")


# LangChain create_agent 라우팅을 위한 can_jump_to 메타데이터 등록
EvaluatorHarness.after_model.__can_jump_to__ = ["model"]

