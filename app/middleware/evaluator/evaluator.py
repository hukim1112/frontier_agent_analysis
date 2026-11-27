"""
===============================================================================
[H-06] Evaluator Harness & Autonomous LLM-as-a-Judge Middleware (v2.0)
===============================================================================
Source: app/middleware/evaluator/evaluator.py

에이전트의 실행 궤적(Trajectory)과 최종 답변(Final Answer)을 심사하는
v2.0 런타임 평가 하네스 미들웨어:

1. 두 가지 운영 모드 지원:
   - mode="rework": @hook_config(can_jump_to=["model"]) 기반 after_agent / aafter_agent 훅.
     결함 발견 시 ORIGIN_EVALUATOR_FEEDBACK 표식을 가진 HumanMessage를 주입하고
     model 노드로 되돌려 자가 교정(Self-Correction Rework)을 지시합니다.
     (토큰 스트리밍 UX 충돌이 없는 백그라운드 서브에이전트·고위험 배치 작업에 적합)
   - mode="shadow": 사용자 응답을 지연시키지 않고(0ms latency) 백그라운드 비동기 태스크로
     샘플링 채점(sample_rate) 후 eval_audit_stream.jsonl에 감사 결과를 적재합니다.
     (실시간 SSE 스트리밍을 사용하는 인터랙티브 main_agent 온라인 평가에 적합)

2. v2.0 하네스 표준 호환:
   - CoT-First JudgeVerdict 스키마 (evidence/reason 선행 → is_approved/score 판정)
   - with_structured_output(JudgeVerdict) + config={"tags": ["exclude_from_stream"]}
   - last_user_turn_start 기반 현재 턴 궤적 슬라이싱 (멀티턴 오염 방지)
   - ORIGIN_EVALUATOR_FEEDBACK 마커로 TranscriptStore / MemoryMiddleware 턴 경계 보호
===============================================================================
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.utils import init_chat_model, normalize_content
from app.utils.message_markers import (
    EVALUATOR_FEEDBACK_TAG,
    ORIGIN_EVALUATOR_FEEDBACK,
    is_evaluator_feedback,
    is_user_turn_start,
    last_user_turn_start,
    message_origin,
    origin_kwargs,
)

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"


# =============================================================================
# CoT-First Structured Judge Schema
# =============================================================================

class JudgeVerdict(BaseModel):
    """판사 모델의 CoT-First 구조화 판정 스키마.

    근거(evidence)와 판정 사유(reason)를 승인 여부(is_approved) 및 점수(score)보다
    먼저 생성하여 자동회귀 LLM의 성급한 판정 및 사후 합리화 편향을 줄입니다.
    """

    evidence: str = Field(
        default="",
        description="도구 실행 결과 및 최종 답변에서 확인한 핵심 사실/누락/오류 인용 (1~2문장)",
    )
    reason: str = Field(
        default="",
        description="위 근거에 기반한 최종 판정 요약 (1~2문장)",
    )
    is_approved: bool = Field(
        description="치명적 결함 없이 사용자 요청의 핵심 목적을 충족했는지 여부 (True: 승인, False: 반려)"
    )
    score: int = Field(
        default=100,
        ge=0,
        le=100,
        description="에이전트 수행 결과에 대한 종합 정량 점수 (0~100)",
    )
    feedback: str = Field(
        default="",
        description="불합격(is_approved=False) 시 에이전트가 다음 턴에 즉시 반영할 구체적 결함 및 수정 지시(훈수). 승인 시 빈 문자열",
    )


# =============================================================================
# Current-Turn Message & Trajectory Extraction Helpers
# =============================================================================

def extract_current_turn_messages(messages: Sequence[Any]) -> List[Any]:
    """전체 대화 이력에서 현재(마지막) 사용자 턴에 해당하는 메시지 구간만 슬라이싱합니다."""
    msg_list = list(messages or [])
    if not msg_list:
        return []
    turn_idx = last_user_turn_start(msg_list)
    return msg_list[turn_idx:] if turn_idx is not None else msg_list


def extract_user_query(messages: Sequence[Any]) -> str:
    """현재 턴을 시작한 실제 사용자 발화 원문을 반환합니다 (하네스 내부 피드백/알림 제외)."""
    turn_msgs = extract_current_turn_messages(messages)
    for msg in turn_msgs:
        if is_user_turn_start(msg):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            return normalize_content(content).strip()
    return ""


def extract_last_ai_answer(messages: Sequence[Any]) -> str:
    """현재 턴 메시지에서 역순으로 실제 AI 최종 답변(content가 비어있지 않은 AIMessage)을 추출합니다."""
    turn_msgs = extract_current_turn_messages(messages)
    for msg in reversed(turn_msgs):
        is_ai = isinstance(msg, AIMessage) or (
            isinstance(msg, dict) and str(msg.get("role") or msg.get("type") or "").lower() in ("ai", "assistant")
        ) or getattr(msg, "type", None) == "ai"
        if not is_ai:
            continue
        raw_content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = normalize_content(raw_content).strip()
        if text:
            return text
    return ""


def extract_tool_contexts(messages: Sequence[Any]) -> List[str]:
    """현재 턴에서 실행된 ToolMessage들의 관측 결과를 문자열 리스트로 추출합니다."""
    turn_msgs = extract_current_turn_messages(messages)
    contexts: List[str] = []
    for msg in turn_msgs:
        is_tool = isinstance(msg, ToolMessage) or (
            isinstance(msg, dict) and str(msg.get("role") or msg.get("type") or "").lower() == "tool"
        ) or getattr(msg, "type", None) == "tool"
        if not is_tool:
            continue
        raw_content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = normalize_content(raw_content).strip()
        if text:
            contexts.append(text)
    return contexts


def format_turn_trajectory(messages: Sequence[Any], max_tool_chars: int = 1500) -> str:
    """현재 턴의 도구 호출(tool_calls), 도구 응답(ToolMessage), 중간 추론을 판사 모델용 텍스트로 포맷합니다."""
    turn_msgs = extract_current_turn_messages(messages)
    if not turn_msgs:
        return "(No messages)"

    lines: List[str] = []
    for msg in turn_msgs:
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            origin = message_origin(msg)
            content = normalize_content(getattr(msg, "content", "")).strip()
            if origin == ORIGIN_EVALUATOR_FEEDBACK:
                lines.append(f"[Previous Evaluator Feedback]: {content}")
            elif origin is None:
                lines.append(f"[User Request]: {content}")
            continue

        is_tool = isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool"
        if is_tool:
            tool_name = getattr(msg, "name", None) or "tool"
            status = getattr(msg, "status", None) or "ok"
            content = normalize_content(getattr(msg, "content", "")).strip()
            if len(content) > max_tool_chars:
                content = content[:max_tool_chars] + " ...(truncated)"
            lines.append(f"[Tool Response ({tool_name}, status={status})]: {content}")
            continue

        is_ai = isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai"
        if is_ai:
            tool_calls = getattr(msg, "tool_calls", None) or []
            content = normalize_content(getattr(msg, "content", "")).strip()
            if tool_calls:
                calls_fmt = ", ".join(
                    f"{tc.get('name', 'tool')}({json.dumps(tc.get('args', {}), ensure_ascii=False, default=str)})"
                    for tc in tool_calls
                )
                lines.append(f"[Agent Tool Call]: {calls_fmt}")
            elif content:
                lines.append(f"[Agent Draft/Response]: {content[:max_tool_chars]}")

    return "\n".join(lines) if lines else "(No tools called)"


# =============================================================================
# EvaluatorHarness Middleware (v2.0)
# =============================================================================

_RUNTIME_JUDGE_SYSTEM_PROMPT = """당신은 AI 에이전트의 작업 실행 궤적(Trajectory)과 최종 결과물(Final Answer)을 감사하고 보정하는 전문 평가 에이전트(Evaluator)입니다.
에이전트가 실행 도중 겪은 단순 에러나 시행착오(예: 손상된 파일 발견 후 백업 파일 탐색)는 스스로 극복했다면 불이익을 주지 마십시오.
오직 아래의 '치명적 결함'이 발견되었을 때만 반려(is_approved=False)하고 재작업을 지시해야 합니다.

🚨 [치명적 결함 및 심사 기준]
1. 목적 누락: 사용자의 원래 지시사항이나 요구사항(필수 항목, 계산 근거, 파일 저장 등) 중 일부를 답변에서 누락했는가?
2. 환각 현상(Hallucination): 도구 실행 결과(Tool Response)로 획득한 사실적 근거와 다르거나, 없는 수치·파일 내용을 지어냈는가?
3. 결과 불능: 생성된 최종 답변이나 산출물이 텅 비었거나 오류 설명 없이 종료되었는가?
4. 무한 루프/포기: 해결 가능한 도구 오류를 우회하지 않고 동일한 실패를 반복하다 끝났는가?
{custom_criteria_section}
💡 [작성 순서 (CoT-First)]
- 먼저 `evidence`에 도구 실행 결과와 최종 답변 대조 근거를 1~2문장으로 적으세요.
- 이어서 `reason`에 판정 사유를 적고, `is_approved`와 `score`(0~100)를 결정하세요.
- 반려(`is_approved=False`)인 경우 `feedback` 필드에 에이전트가 다음 턴에 즉시 수정할 수 있도록 누락된 요구사항이나 실제 도구 결과 값을 구체적으로 지시하세요. 승인 시 `feedback`은 빈 문자열로 두세요."""


class EvaluatorHarness(AgentMiddleware):
    """
    에이전트 실행 종료 직전(after_agent / aafter_agent) 궤적과 최종 답변을 검증하는 평가 하네스.

    Args:
        mode:
          - "rework": 결함 감지 시 ORIGIN_EVALUATOR_FEEDBACK 피드백을 주입하고 jump_to="model"로 재작업 지시.
          - "shadow": 사용자 응답을 지연시키지 않고 백그라운드 비동기로 샘플링 평가 및 JSONL 적재.
        max_reworks: mode="rework"일 때 한 사용자 턴에서 허용할 최대 재작업 횟수 (기본 2회).
        judge_model_name: 판사 LLM 모델명 (기본 "gpt-5.4-mini", 에이전트와 다른 계열 권장).
        judge_llm: 외부에서 주입한 판사 LLM 인스턴스 (지정 시 judge_model_name보다 우선).
        sample_rate: mode="shadow"일 때 평가 수행 확률 (0.0 ~ 1.0, 기본 1.0).
        log_dir: 평가 감사 스트림(`eval_audit_stream.jsonl`) 저장 디렉토리 (None이면 디스크 적재 생략).
        custom_criteria: 기본 4대 결함 기준 외에 추가할 도메인별 필수 점검 기준 문자열.
        verbose: 콘솔 진행 로그 출력 여부.
        strict: True면 판사 LLM 호출 실패 시 예외를 전파하고, False면 안전 패스(is_approved=True) 처리.
    """

    def __init__(
        self,
        mode: Literal["rework", "shadow"] = "rework",
        max_reworks: int = 2,
        judge_model_name: str = DEFAULT_JUDGE_MODEL,
        judge_llm: Optional[Any] = None,
        sample_rate: float = 1.0,
        log_dir: Optional[str] = "./artifacts/eval_logs",
        custom_criteria: Optional[str] = None,
        verbose: bool = True,
        strict: bool = False,
    ):
        super().__init__()
        if mode not in ("rework", "shadow"):
            raise ValueError(f"Unsupported EvaluatorHarness mode '{mode}'. Must be 'rework' or 'shadow'.")
        self.mode = mode
        self.max_reworks = max(0, int(max_reworks))
        self.judge_model_name = judge_model_name
        self.judge_llm = judge_llm
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self.log_dir = log_dir
        self.custom_criteria = (custom_criteria or "").strip()
        self.verbose = verbose
        self.strict = strict

        self._io_lock = threading.RLock()
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._pending_threads: set[threading.Thread] = set()

        self.last_verdict: Optional[JudgeVerdict] = None
        self.verdict_history: List[Dict[str, Any]] = []

    def _get_judge_llm(self) -> Any:
        if self.judge_llm is None:
            self.judge_llm = init_chat_model(model=self.judge_model_name, temperature=0.0)
        return self.judge_llm

    def _build_judge_messages(
        self,
        user_query: str,
        trajectory_str: str,
        final_answer: str,
    ) -> List[BaseMessage]:
        custom_section = (
            f"\n📌 [추가 필수 심사 기준]\n{self.custom_criteria}\n"
            if self.custom_criteria
            else ""
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", _RUNTIME_JUDGE_SYSTEM_PROMPT),
            (
                "user",
                "■ 사용자 원래 요청:\n{user_query}\n\n"
                "■ 현재 턴 실행 궤적 (Trajectory):\n{trajectory}\n\n"
                "■ 에이전트의 최종 답변 (Final Answer):\n{final_answer}",
            ),
        ])
        return prompt.format_messages(
            custom_criteria_section=custom_section,
            user_query=user_query or "(질문 정보 없음)",
            trajectory=trajectory_str,
            final_answer=final_answer,
        )

    @staticmethod
    def _coerce_verdict(raw: Any) -> JudgeVerdict:
        if isinstance(raw, JudgeVerdict):
            return raw
        if isinstance(raw, dict):
            return JudgeVerdict(**raw)
        if hasattr(raw, "model_dump"):
            return JudgeVerdict(**raw.model_dump())
        raise TypeError(f"Unsupported judge output type: {type(raw)}")

    @staticmethod
    def _count_consecutive_reworks(messages: Sequence[Any]) -> int:
        """현재 사용자 턴에서 이미 주입된 Evaluator 재작업 피드백 횟수를 셉니다."""
        count = 0
        for msg in reversed(messages):
            if is_user_turn_start(msg):
                break
            if is_evaluator_feedback(msg):
                count += 1
        return count

    @staticmethod
    def _resolve_session_id(state: Any, runtime: Any) -> str:
        exec_info = getattr(runtime, "execution_info", None) if runtime is not None else None
        if exec_info and getattr(exec_info, "thread_id", None):
            return str(exec_info.thread_id).strip()

        try:
            from langchain_core.runnables.config import var_child_runnable_config

            rcfg = var_child_runnable_config.get()
            if isinstance(rcfg, dict):
                tid = (rcfg.get("configurable") or {}).get("thread_id")
                if tid and str(tid).strip():
                    return str(tid).strip()
        except Exception:
            pass

        cfg = getattr(runtime, "config", None) if runtime is not None else None
        if isinstance(cfg, dict):
            tid = (cfg.get("configurable") or {}).get("thread_id")
            if tid and str(tid).strip():
                return str(tid).strip()

        ctx = getattr(runtime, "context", None) if runtime is not None else None
        if ctx is not None:
            sid = getattr(ctx, "session_id", None)
            if sid and str(sid).strip() and sid != "unknown":
                return str(sid).strip()

        if isinstance(state, dict):
            scfg = state.get("configurable")
            if isinstance(scfg, dict) and scfg.get("thread_id"):
                return str(scfg["thread_id"]).strip()
        return f"session_{int(time.time())}"

    @staticmethod
    def _should_skip_last_message(messages: Sequence[Any]) -> Optional[str]:
        """평가 대상이 아닌 메시지(빈 리스트, 비-AI 메시지, 도구 호출 대기, 가드레일 차단)인 경우 None이 아닌 이유 반환."""
        if not messages:
            return "empty_messages"
        last_msg = messages[-1]
        is_ai = isinstance(last_msg, AIMessage) or getattr(last_msg, "type", None) == "ai"
        if not is_ai:
            return "not_ai_message"
        if getattr(last_msg, "tool_calls", None):
            return "pending_tool_calls"
        kwargs = getattr(last_msg, "additional_kwargs", None) or {}
        if kwargs.get("guardrail_blocked"):
            return "guardrail_blocked"
        content = normalize_content(getattr(last_msg, "content", "")).strip()
        if not content:
            return "empty_answer"
        return None

    def _record_verdict(
        self,
        *,
        session_id: str,
        user_query: str,
        final_answer: str,
        verdict: JudgeVerdict,
        rework_attempt: int,
        latency_ms: int,
    ) -> None:
        entry = {
            "event": "evaluator_verdict",
            "mode": self.mode,
            "session_id": session_id,
            "rework_attempt": rework_attempt,
            "user_query": user_query,
            "final_answer_preview": final_answer[:300],
            "is_approved": verdict.is_approved,
            "score": verdict.score,
            "evidence": verdict.evidence,
            "reason": verdict.reason,
            "feedback": verdict.feedback,
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self._io_lock:
            self.last_verdict = verdict
            self.verdict_history.append(entry)
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                log_file = os.path.join(self.log_dir, "eval_audit_stream.jsonl")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _run_judge(self, turn_messages: Sequence[Any], final_answer: str) -> JudgeVerdict:
        user_query = extract_user_query(turn_msgs := list(turn_messages))
        trajectory_str = format_turn_trajectory(turn_msgs[:-1] if len(turn_msgs) > 1 else turn_msgs)
        judge_msgs = self._build_judge_messages(user_query, trajectory_str, final_answer)
        try:
            structured = self._get_judge_llm().with_structured_output(JudgeVerdict)
            raw = structured.invoke(judge_msgs, config={"tags": ["exclude_from_stream"]})
            return self._coerce_verdict(raw)
        except Exception as e:
            if self.strict:
                raise
            if self.verbose:
                print(f"⚠️ [EvaluatorHarness] 판정 중 에러 발생 (안전 패스 처리): {e}")
            return JudgeVerdict(
                evidence="",
                reason=f"에러 안전 패스: {e}",
                is_approved=True,
                score=100,
                feedback="",
            )

    async def _arun_judge(self, turn_messages: Sequence[Any], final_answer: str) -> JudgeVerdict:
        user_query = extract_user_query(turn_msgs := list(turn_messages))
        trajectory_str = format_turn_trajectory(turn_msgs[:-1] if len(turn_msgs) > 1 else turn_msgs)
        judge_msgs = self._build_judge_messages(user_query, trajectory_str, final_answer)
        try:
            structured = self._get_judge_llm().with_structured_output(JudgeVerdict)
            raw = await structured.ainvoke(judge_msgs, config={"tags": ["exclude_from_stream"]})
            return self._coerce_verdict(raw)
        except Exception as e:
            if self.strict:
                raise
            if self.verbose:
                print(f"⚠️ [EvaluatorHarness] 비동기 판정 중 에러 발생 (안전 패스 처리): {e}")
            return JudgeVerdict(
                evidence="",
                reason=f"에러 안전 패스: {e}",
                is_approved=True,
                score=100,
                feedback="",
            )

    def _build_rework_update(self, verdict: JudgeVerdict, reworks: int) -> Dict[str, Any]:
        feedback_content = (
            f"⚠️ {EVALUATOR_FEEDBACK_TAG} Rework Required ({reworks + 1}/{self.max_reworks})\n"
            f"Evaluator 검증 결과 보완이 필요합니다 (점수: {verdict.score}점, 사유: {verdict.reason}).\n"
            f"아래 피드백을 충실히 반영하여 답변(필요 시 도구 재호출 포함)을 수정하십시오.\n\n"
            f"🔍 세부 피드백:\n{verdict.feedback}"
        )
        return {
            "jump_to": "model",
            "messages": [
                HumanMessage(
                    content=feedback_content,
                    additional_kwargs=origin_kwargs(ORIGIN_EVALUATOR_FEEDBACK),
                )
            ],
        }

    def _run_shadow_sync(
        self,
        *,
        session_id: str,
        turn_messages: List[Any],
        user_query: str,
        final_answer: str,
        reworks: int,
    ) -> None:
        t0 = time.time()
        verdict = self._run_judge(turn_messages, final_answer)
        latency_ms = int((time.time() - t0) * 1000)
        self._record_verdict(
            session_id=session_id,
            user_query=user_query,
            final_answer=final_answer,
            verdict=verdict,
            rework_attempt=reworks,
            latency_ms=latency_ms,
        )
        if self.verbose:
            status = "PASS" if verdict.is_approved else "FLAGGED"
            print(f"📊 [EvaluatorHarness:Shadow] {status} ({verdict.score}점, {latency_ms}ms) | {verdict.reason}")

    async def _run_shadow_async(
        self,
        *,
        session_id: str,
        turn_messages: List[Any],
        user_query: str,
        final_answer: str,
        reworks: int,
    ) -> None:
        t0 = time.time()
        verdict = await self._arun_judge(turn_messages, final_answer)
        latency_ms = int((time.time() - t0) * 1000)
        self._record_verdict(
            session_id=session_id,
            user_query=user_query,
            final_answer=final_answer,
            verdict=verdict,
            rework_attempt=reworks,
            latency_ms=latency_ms,
        )
        if self.verbose:
            status = "PASS" if verdict.is_approved else "FLAGGED"
            print(f"📊 [EvaluatorHarness:Shadow] {status} ({verdict.score}점, {latency_ms}ms) | {verdict.reason}")

    def wait_pending(self, timeout: float = 15.0) -> None:
        """동기 shadow 스레드가 모두 종료될 때까지 대기합니다."""
        deadline = time.time() + timeout
        for t in list(self._pending_threads):
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)

    async def await_pending(self, timeout: float = 15.0) -> None:
        """비동기 shadow 태스크 및 동기 스레드가 모두 완료될 때까지 대기합니다."""
        tasks = [t for t in list(self._pending_tasks) if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        if self._pending_threads:
            await asyncio.to_thread(self.wait_pending, timeout)

    # =========================================================================
    # Agent Lifecycle Hooks (Sync + Async, jump_to="model" supported)
    # =========================================================================

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: AgentState, runtime: Optional[Runtime] = None) -> Dict[str, Any] | None:
        messages = list((state or {}).get("messages") or [])
        if self._should_skip_last_message(messages) is not None:
            return None

        turn_messages = extract_current_turn_messages(messages)
        user_query = extract_user_query(turn_messages)
        final_answer = normalize_content(getattr(messages[-1], "content", "")).strip()
        session_id = self._resolve_session_id(state, runtime)
        reworks = self._count_consecutive_reworks(messages)

        if self.mode == "shadow":
            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                return None

            def _worker() -> None:
                try:
                    self._run_shadow_sync(
                        session_id=session_id,
                        turn_messages=list(turn_messages),
                        user_query=user_query,
                        final_answer=final_answer,
                        reworks=reworks,
                    )
                finally:
                    with self._io_lock:
                        self._pending_threads.discard(threading.current_thread())

            thread = threading.Thread(target=_worker, daemon=True, name="EvaluatorShadowWorker")
            with self._io_lock:
                self._pending_threads.add(thread)
            thread.start()
            return None

        t0 = time.time()
        judgment = self._run_judge(turn_messages, final_answer)
        latency_ms = int((time.time() - t0) * 1000)
        self._record_verdict(
            session_id=session_id,
            user_query=user_query,
            final_answer=final_answer,
            verdict=judgment,
            rework_attempt=reworks,
            latency_ms=latency_ms,
        )

        if judgment.is_approved:
            if self.verbose:
                print(f"✅ [EvaluatorHarness] 승인 완료 ({judgment.score}점, {latency_ms}ms): {judgment.reason}")
            return None

        if reworks >= self.max_reworks:
            if self.verbose:
                print(f"🚨 [EvaluatorHarness] 최대 재작업 한도({self.max_reworks}회) 도달 — 현재 답변으로 종료합니다.")
            return None

        if self.verbose:
            print(f"🔄 [EvaluatorHarness] 반려 발생 ({reworks + 1}/{self.max_reworks}회차, {judgment.score}점) → model 노드로 롤백")
        return self._build_rework_update(judgment, reworks)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: AgentState, runtime: Optional[Runtime] = None) -> Dict[str, Any] | None:
        messages = list((state or {}).get("messages") or [])
        if self._should_skip_last_message(messages) is not None:
            return None

        turn_messages = extract_current_turn_messages(messages)
        user_query = extract_user_query(turn_messages)
        final_answer = normalize_content(getattr(messages[-1], "content", "")).strip()
        session_id = self._resolve_session_id(state, runtime)
        reworks = self._count_consecutive_reworks(messages)

        if self.mode == "shadow":
            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                return None

            task = asyncio.create_task(
                self._run_shadow_async(
                    session_id=session_id,
                    turn_messages=list(turn_messages),
                    user_query=user_query,
                    final_answer=final_answer,
                    reworks=reworks,
                ),
                name=f"eval-shadow-{session_id}",
            )
            with self._io_lock:
                self._pending_tasks.add(task)
            task.add_done_callback(lambda t: self._pending_tasks.discard(t))
            return None

        t0 = time.time()
        judgment = await self._arun_judge(turn_messages, final_answer)
        latency_ms = int((time.time() - t0) * 1000)
        self._record_verdict(
            session_id=session_id,
            user_query=user_query,
            final_answer=final_answer,
            verdict=judgment,
            rework_attempt=reworks,
            latency_ms=latency_ms,
        )

        if judgment.is_approved:
            if self.verbose:
                print(f"✅ [EvaluatorHarness] 승인 완료 ({judgment.score}점, {latency_ms}ms): {judgment.reason}")
            return None

        if reworks >= self.max_reworks:
            if self.verbose:
                print(f"🚨 [EvaluatorHarness] 최대 재작업 한도({self.max_reworks}회) 도달 — 현재 답변으로 종료합니다.")
            return None

        if self.verbose:
            print(f"🔄 [EvaluatorHarness] 반려 발생 ({reworks + 1}/{self.max_reworks}회차, {judgment.score}점) → model 노드로 롤백")
        return self._build_rework_update(judgment, reworks)
