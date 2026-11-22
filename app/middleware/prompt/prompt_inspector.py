"""Notebook/CLI prompt inspection helpers for PromptAssembler outputs."""

import re
import unicodedata
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.utils.message_markers import message_origin


def _disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content or [])


def _summarize_message(msg: Any, text: str) -> str:
    origin = message_origin(msg)
    if isinstance(msg, SystemMessage):
        if "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in text:
            parts = [
                label for cond, label in (
                    ("Layer 1" in text, "L1 Constitution"),
                    ("Layer 2" in text, "L2 Tool/Skill"),
                    ("__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in text, "⚡BOUNDARY"),
                ) if cond
            ]
            return " + ".join(parts)
        parts = [
            label for cond, label in (
                ("Layer 3" in text, "L3 Session"),
                ("Layer 4" in text, "L4 MemoryRules/Env"),
                ("Layer 5" in text, "L5 Resource"),
                ("# MCP Server Instructions" in text, "MCP지침(uncached)"),
                ("gitStatus:" in text, "gitStatus"),
            ) if cond
        ]
        return " + ".join(parts) if parts else text.splitlines()[0][:52]

    if origin == "user_context":
        heads = re.findall(r"^#\s+([A-Za-z0-9_]+)", text, re.MULTILINE)
        return " · ".join(f"#{h}" for h in heads) if heads else "<system-reminder>"

    if origin == "skill_listing":
        skills = [ln[2:].split(":", 1)[0].strip() for ln in text.splitlines() if ln.startswith("- ")]
        if len(skills) > 1:
            return f"스킬 {len(skills)}개 목록 ({skills[0]} 외 {len(skills) - 1}개)"
        if len(skills) == 1:
            return f"신규 스킬 1개 ({skills[0]})"

    if origin == "mcp_instructions_delta":
        added = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
        removed = []
        if "disconnected" in text:
            tail = text.split("disconnected.", 1)[-1].split(":", 1)[-1]
            removed = [ln.strip() for ln in tail.splitlines() if ln.strip() and not ln.strip().startswith("<")]
        segs = []
        if added:
            segs.append(f"+연결: {added}")
        if removed:
            segs.append(f"-해제: {removed}")
        return ", ".join(segs) or "MCP delta"

    if origin == "episodic_memory":
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("<") and ln.strip() != "# episodicMemory"
        ]
        if lines:
            preview = (lines[0][:38] + "…") if len(lines[0]) > 38 else lines[0]
            return f"# episodicMemory · {preview}"
        return "# episodicMemory · "

    if origin == "date_change":
        m = re.search(r"Today's date is now \d{4}-\d{2}-\d{2}\.", text)
        if m:
            return f"The date has changed. {m.group(0)}"
        for ln in text.splitlines():
            if "Today's date is now" in ln:
                return ln.strip()

    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        calls = ", ".join(f"{tc['name']}({tc.get('args', {})})" for tc in msg.tool_calls)
        return f"🔧 {calls[:54]}"

    flat = " ⏎ ".join(ln.strip() for ln in text.splitlines() if ln.strip())
    return flat[:52] + ("…" if len(flat) > 52 else "")


def _format_body_preview(msg: Any, max_skill_lines: int = 3) -> str:
    text = _msg_text(msg)
    if message_origin(msg) != "skill_listing":
        return text
    lines = text.splitlines()
    bullet_idx = [i for i, ln in enumerate(lines) if ln.startswith("- ")]
    if len(bullet_idx) <= max_skill_lines:
        return text
    head = lines[: bullet_idx[max_skill_lines - 1] + 1]
    hidden = len(bullet_idx) - max_skill_lines
    return "\n".join(head + [f"... (외 {hidden}개 스킬 생략)", "</system-reminder>"])


def show_prompt_map(
    messages: Sequence[Any],
    title: str,
    tools: Optional[Sequence[Any]] = None,
    prev_messages: Optional[Sequence[Any]] = None,
    highlight_indices: Iterable[int] = (),
    show_bodies: Iterable[int] = (),
    max_skill_lines: int = 3,
) -> None:
    """조립된 요청의 tools 순서와 메시지 계층 구조를 한눈에 출력합니다."""
    print("=" * 84)
    print(f"📌 {title}")
    print("=" * 84)
    if tools is not None:
        names = [t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "") for t in tools]
        builtin = [n for n in names if not n.startswith("mcp__")]
        mcp = [n for n in names if n.startswith("mcp__")]
        print(f"🧰 ① tools 칸 (총 {len(names)}개): 내장 {builtin} → MCP {mcp}")

    highlight_set = {idx if idx >= 0 else len(messages) + idx for idx in highlight_indices}
    use_badge_col = (prev_messages is not None) or bool(highlight_set)

    prefix_alive = True
    cached_count = 0
    cached_chars = 0
    for i, m in enumerate(messages):
        text = _msg_text(m)
        origin = message_origin(m)
        if isinstance(m, SystemMessage):
            if "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in text:
                cached = " +cache" if (getattr(m, "additional_kwargs", None) or {}).get("cache_control") else ""
                role_tag = f"System [L1~L2 정적 🔒{cached}]"
            else:
                role_tag = "System [L3~L5 동적 🔄]"
        elif isinstance(m, ToolMessage):
            role_tag = f"Tool({m.name})"
        elif isinstance(m, AIMessage):
            role_tag = "AI"
        else:
            role_tag = f"Human<{origin}>" if origin else "Human"

        is_cached = False
        if prev_messages is not None:
            if prefix_alive and i < len(prev_messages) and text == _msg_text(prev_messages[i]):
                is_cached = True
                cached_count += 1
                cached_chars += len(text)
            else:
                prefix_alive = False

        summary = _summarize_message(m, text)
        badge = ""
        if use_badge_col:
            if is_cached:
                badge_label = "[캐시 ✅]"
            elif i in highlight_set:
                if origin == "skill_listing":
                    badge_label = "[✨스킬]"
                elif origin == "mcp_instructions_delta":
                    badge_label = "[✨MCP]"
                else:
                    badge_label = "[✨추가]"
                summary = f"◀ {summary}"
            elif prev_messages is not None:
                badge_label = "[변경 ❌]" if i < len(prev_messages) else "[신규 🆕]"
            else:
                badge_label = ""
            badge = _pad(badge_label, 10) + " "

        print(f"  [{i:02d}] {badge}{_pad(role_tag, 32)} {len(text):>6,}자 | {summary}")

    if prev_messages is not None:
        if cached_count > 0:
            tail = (
                f" · 뒤 {len(messages) - cached_count}개 신규 (이 중 스킬·MCP 알림 {len(highlight_set)}개 ✨)"
                if highlight_set
                else f" · 뒤 {len(messages) - cached_count}개 신규 추가"
            )
            print(f"➡️  앞 [00]~[{cached_count - 1:02d}] 프리픽스 {cached_count}개 메시지 100% 바이트 일치 ({cached_chars:,}자 캐시 재사용){tail}")
        else:
            print(f"➡️  프리픽스 일치 메시지 없음 (총 {len(messages)}개 중 {len(messages)}개 신규 계산)")

    if show_bodies:
        print("\n--- 상세 내용 확인 ---")
        for raw_idx in show_bodies:
            idx = raw_idx if raw_idx >= 0 else len(messages) + raw_idx
            m = messages[idx]
            origin = message_origin(m) or type(m).__name__
            print(f"\n[{idx:02d} {origin}]")
            print(_format_body_preview(m, max_skill_lines=max_skill_lines))


def show_system_skeleton(messages: Sequence[Any]) -> None:
    """Static/Dynamic 두 SystemMessage 내부의 실제 계층·모듈 골격을 요약합니다."""
    s_txt, d_txt = _msg_text(messages[0]), _msg_text(messages[1])
    has_env_leak = any(k in s_txt for k in ("Session ID:", "Working Directory", "Today's date"))
    skill_policy = "포함 (목록 없음)" if "## 📦 Skill Execution Policy" in s_txt else "없음"
    env_lines = [
        ln
        for ln in d_txt.splitlines()
        if ln.startswith("- Working Directory") or ln.startswith("- Host OS") or ln.startswith("- Session ID")
    ]
    scratch_ln = next((ln for ln in d_txt.splitlines() if "dedicated scratch directory" in ln), "")
    git_lines = [ln for ln in d_txt[d_txt.find("gitStatus:"):].splitlines() if ln.strip()][:3] if "gitStatus:" in d_txt else []

    print("\n" + "=" * 84)
    print("🧱 System Prompt 골격 해부 ([00] 정적 vs [01] 동적)")
    print("=" * 84)
    print(f"[00] 정적 SystemMessage ({len(s_txt):,}자) — Global Prefix Cache 대상")
    print("  • Layer 1 (Global Constitution) : [0] 정체성 · [1] 시스템 수칙 · [2] 과잉설계 금지 · [3] 위험 확인 · [5~6] 톤/효율")
    print(f"  • Layer 2 (Tool Guide & Skills) : [4] 전용 도구 우선 원칙 + 스킬 사용법 ({skill_policy})")
    print(f"  • Boundary [7]                  : __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ (세션값 혼입 없음: {not has_env_leak})")
    print(f"[01] 동적 SystemMessage ({len(d_txt):,}자) — 세션별 1회 계산 + ④ gitStatus")
    print("  • Layer 3 (Session Interaction) : [8] ! <cmd>, 슬래시 커맨드, Explore 위임")
    print("  • Layer 4 (Environment & Memory): [9] # auto memory (사용 규칙만 고정, 데이터 분리)")
    for ln in env_lines:
        print(f"      {ln}")
    print("  • Layer 5 (Resource Control)    : [14] Scratchpad · [15~16] 결과 정리 공지 (MCP는 Delta 알림)")
    if scratch_ln:
        print(f"      - {scratch_ln}")
    if git_lines:
        print(f"  • ④ System Context (맨 끝)      : {' | '.join(git_lines)}")
    print("💡 원문 전체가 궁금하면 print(messages_t1[0].content) / print(messages_t1[1].content)로 확인하세요.")


def build_live_request(
    mw: Any,
    state: dict,
    tools: Optional[Sequence[Any]] = None,
    context: Any = None,
    model: str = "demo",
) -> Any:
    """wrap_model_call이 모델 API 직전에 조립한 최종 ModelRequest를 그대로 반환합니다."""
    req = ModelRequest(
        messages=state["messages"],
        model=model,
        tool_choice="auto",
        model_settings={},
        state=state,
        tools=list(tools or []),
        runtime=SimpleNamespace(context=context, config=None, execution_info=None),
    )
    return mw.wrap_model_call(req, lambda r: r)


def simulate_turn(
    mw: Any,
    state: dict,
    label: str,
    prev_messages: Optional[Sequence[Any]] = None,
    tools: Optional[Sequence[Any]] = None,
    context: Any = None,
) -> Any:
    """before_model(Delta · 에피소드 기억 알림 저장) + wrap_model_call(전체 메시지 조립) 결과를 한눈에 출력합니다."""
    runtime = SimpleNamespace(context=context, config=None, execution_info=None)
    update = mw.before_model(state, runtime=runtime) or {}
    added = update.pop("messages", [])
    state.update(update)
    state["messages"] = state["messages"] + added

    out = build_live_request(mw, state, tools or [], context)
    added_indices = list(range(len(out.messages) - len(added), len(out.messages))) if added else []

    show_prompt_map(
        out.messages,
        f"{label} (전체 {len(out.messages)}개 · 신규 알림 {len(added)}개)",
        tools=out.tools if tools else None,
        prev_messages=prev_messages,
        highlight_indices=added_indices,
        show_bodies=added_indices,
    )
    return out


__all__ = ["show_prompt_map", "show_system_skeleton", "build_live_request", "simulate_turn"]
