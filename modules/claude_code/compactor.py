import os
import re
import json
import uuid
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import wrap_model_call

# =============================================================================
# 1. SnipCompactor (Age-based Old Tool Output Truncation)
# =============================================================================
class SnipCompactor:
    """Replaces verbose tool results older than age_threshold turns with concise 1-line stubs."""
    def __init__(self, age_threshold: int = 2):
        self.age_threshold = age_threshold

    def compact(self, messages: list) -> tuple[list, bool]:
        dialogue_indices = [i for i, m in enumerate(messages) if not isinstance(m, SystemMessage)]
        total_dialogue = len(dialogue_indices)
        if total_dialogue <= self.age_threshold:
            return messages, False

        modified = False
        new_messages = list(messages)
        cutoff_dialogue_idx = total_dialogue - self.age_threshold

        for idx in range(cutoff_dialogue_idx):
            msg_pos = dialogue_indices[idx]
            msg = new_messages[msg_pos]
            
            if isinstance(msg, ToolMessage) and len(str(msg.content)) > 80:
                tool_name = getattr(msg, "name", "tool") or "tool"
                snipped_content = f"[Tool result snipped: Executed '{tool_name}' successfully ({len(str(msg.content))} chars)]"
                new_messages[msg_pos] = ToolMessage(content=snipped_content, tool_call_id=msg.tool_call_id, name=tool_name)
                modified = True

        return new_messages, modified


# =============================================================================
# 2. MicroCompactor (Size-based Disk Swap for Large Tool Outputs)
# =============================================================================
class MicroCompactor:
    """Swaps large tool outputs exceeding max_chars to local disk swap files."""
    def __init__(self, max_chars: int = 5000, swap_dir: str = "./.claude/swaps"):
        self.max_chars = max_chars
        self.swap_dir = swap_dir
        os.makedirs(self.swap_dir, exist_ok=True)

    def compact(self, messages: list) -> tuple[list, bool]:
        modified = False
        new_messages = []

        for msg in messages:
            content_str = str(msg.content) if msg.content else ""
            if isinstance(msg, ToolMessage) and len(content_str) > self.max_chars:
                swap_filename = f"swap_{uuid.uuid4().hex[:8]}.txt"
                swap_path = os.path.join(self.swap_dir, swap_filename)
                with open(swap_path, "w", encoding="utf-8") as f:
                    f.write(content_str)
                
                tool_name = getattr(msg, "name", "tool") or "tool"
                swap_stub = f"[Output ({len(content_str)} chars) microcompacted to disk: {swap_path}]"
                new_messages.append(ToolMessage(content=swap_stub, tool_call_id=msg.tool_call_id, name=tool_name))
                modified = True
            else:
                new_messages.append(msg)

        return new_messages, modified


# =============================================================================
# 3. ContextCollapse (Grouping Intermediate Exploration Work Blocks)
# =============================================================================
class ContextCollapse:
    """Folds 3+ consecutive exploration/research tool calls into 1 collapsed snapshot SystemMessage, optionally saving raw transcript to swap file for full agent restoration."""
    def __init__(self, min_consecutive: int = 3, swap_dir: str = "./.claude/swaps"):
        self.min_consecutive = min_consecutive
        self.swap_dir = swap_dir
        if self.swap_dir and not os.path.exists(self.swap_dir):
            os.makedirs(self.swap_dir, exist_ok=True)

    def compact(self, messages: list) -> tuple[list, bool]:
        new_messages = []
        i = 0
        n = len(messages)
        modified = False

        while i < n:
            msg = messages[i]
            if isinstance(msg, ToolMessage) or (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)):
                group = []
                while i < n and (isinstance(messages[i], ToolMessage) or (isinstance(messages[i], AIMessage) and getattr(messages[i], "tool_calls", None))):
                    group.append(messages[i])
                    i += 1
                
                if len(group) >= self.min_consecutive:
                    tool_names = set()
                    raw_lines = []
                    for m in group:
                        if isinstance(m, ToolMessage) and getattr(m, "name", None):
                            tool_names.add(m.name)
                            raw_lines.append(f"[ToolResult:{m.name}] {m.content}")
                        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                tool_names.add(tc.get("name", "tool"))
                            raw_lines.append(f"[AIThought] {m.content}")
                    
                    tools_str = ", ".join(sorted(tool_names)) if tool_names else "exploration tools"
                    
                    # Save raw snapshot to disk for 100% restoration if needed by agent
                    snapshot_path_str = ""
                    if self.swap_dir:
                        import time
                        snap_filename = f"collapse_snap_{int(time.time()*1000)}.txt"
                        snap_full_path = os.path.join(self.swap_dir, snap_filename)
                        try:
                            with open(snap_full_path, "w", encoding="utf-8") as sf:
                                sf.write("\n\n".join(raw_lines))
                            snapshot_path_str = f" | Raw Snapshot: {snap_full_path}"
                        except Exception:
                            pass

                    collapse_msg = SystemMessage(
                        content=f"[Context Collapsed: {len(group)} research steps ({tools_str}) performed in workspace{snapshot_path_str}]"
                    )
                    new_messages.append(collapse_msg)
                    modified = True
                else:
                    new_messages.extend(group)
            else:
                new_messages.append(msg)
                i += 1

        return new_messages, modified


# =============================================================================
# 4. AutoCompactor (Proactive Token Threshold Summary + Amnesia Guard)
# =============================================================================
class AutoCompactor:
    """Proactively summarizes history when token count exceeds threshold, preserving L1-L5 system messages."""
    def __init__(self, llm, threshold_tokens: int = 8000, amnesia_guard=None):
        self.llm = llm
        self.threshold_tokens = threshold_tokens
        self.amnesia_guard = amnesia_guard

    def get_token_count(self, messages: list) -> int:
        try:
            return self.llm.get_num_tokens_from_messages(messages)
        except Exception:
            return sum(len(str(m.content)) for m in messages) // 4

    def compact_if_needed(self, messages: list, force: bool = False) -> tuple[list, bool]:
        tokens = self.get_token_count(messages)
        if not force and tokens <= self.threshold_tokens:
            return messages, False

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        dialogue_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(dialogue_messages) <= 1:
            return messages, False

        to_compact = dialogue_messages[:-1]
        last_message = dialogue_messages[-1]

        history_text = ""
        for m in to_compact:
            role = "AI" if m.type == "ai" else "User" if m.type == "human" else m.type.upper()
            history_text += f"{role}: {m.content}\n"

        summary_prompt = (
            "You are a context compaction engine. Please summarize the following conversation history.\n"
            "Structure your summary into 4 clear sections:\n"
            "1. Primary Goal & Intent\n"
            "2. Key Decisions & Architecture\n"
            "3. Code Modifications & Workspace Delta\n"
            "4. Current Task & Next Action\n\n"
            f"Conversation History:\n{history_text}"
        )
        
        try:
            from app.utils.message_utils import normalize_content
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = normalize_content(getattr(summary_response, "content", summary_response))
        except Exception as e:
            summary_text = f"Error generating summary: {e}. Raw message count: {len(to_compact)}"

        summary_msg = SystemMessage(content=f"Previous Conversation Summary:\n{summary_text}")

        recovery_attachments = []
        if self.amnesia_guard:
            recovery_attachments = self.amnesia_guard.create_recovery_attachments()

        compacted = system_messages + [summary_msg] + recovery_attachments + [last_message]
        return compacted, True


# =============================================================================
# 5. ReactiveCompactor (Silent Withholding & 20% Tail Slicing on 413 Error)
# =============================================================================
class ReactiveCompactor:
    """Emergency post-invocation firewall that catches 413 / overflow errors and slices 20% tail messages."""
    def __init__(self, slice_ratio: float = 0.20, amnesia_guard=None):
        self.slice_ratio = slice_ratio
        self.amnesia_guard = amnesia_guard

    def handle_overflow(self, messages: list) -> list:
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        dialogue_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(dialogue_messages) <= 2:
            if dialogue_messages:
                last_msg = dialogue_messages[-1]
                truncated_content = str(last_msg.content)[:1000] + "\n... [Reactive Compact: Truncated 413 payload]"
                return system_messages + [HumanMessage(content=truncated_content)]
            return messages

        slice_count = max(2, int(len(dialogue_messages) * self.slice_ratio))
        sliced_dialogue = dialogue_messages[slice_count:]

        reactive_summary = SystemMessage(
            content=f"[Reactive Compact (Silent Withholding)]: Emergency sliced oldest {slice_count} dialogue messages after API 413 overflow."
        )

        recovery_attachments = []
        if self.amnesia_guard:
            recovery_attachments = self.amnesia_guard.create_recovery_attachments()

        return system_messages + [reactive_summary] + recovery_attachments + sliced_dialogue


# =============================================================================
# 6. Combined 5-Stage Compactor Middleware Decorator (@wrap_model_call)
# =============================================================================
def create_compactor_middleware(llm, threshold_tokens: int = 8000, amnesia_guard=None, swap_dir: str = "./.claude/swaps"):
    """Creates a LangChain @wrap_model_call middleware executing Phase 1 (pre-call) and Phase 2 (reactive) compaction."""
    snip_compactor = SnipCompactor(age_threshold=2)
    micro_compactor = MicroCompactor(max_chars=5000, swap_dir=swap_dir)
    context_collapse = ContextCollapse(min_consecutive=3)
    auto_compactor = AutoCompactor(llm=llm, threshold_tokens=threshold_tokens, amnesia_guard=amnesia_guard)
    reactive_compactor = ReactiveCompactor(slice_ratio=0.20, amnesia_guard=amnesia_guard)

    @wrap_model_call
    def compactor_middleware(request, handler):
        msgs = list(request.messages)
        
        # Stage 1: Snip Compact
        msgs, _ = snip_compactor.compact(msgs)
        # Stage 2: Microcompact
        msgs, _ = micro_compactor.compact(msgs)
        # Stage 3: Context Collapse
        msgs, _ = context_collapse.compact(msgs)
        # Stage 4: Auto-Compact
        msgs, _ = auto_compactor.compact_if_needed(msgs)

        req = request.override(messages=msgs)

        try:
            return handler(req)
        except Exception as e:
            err_str = str(e).lower()
            if "413" in err_str or "prompt_too_long" in err_str or "context_length_exceeded" in err_str or "too many tokens" in err_str:
                print("\n⚡ [Reactive Compact Triggered] Catching 413 API Error. Performing Silent 20% Tail Slicing...")
                reactive_msgs = reactive_compactor.handle_overflow(msgs)
                retry_req = request.override(messages=reactive_msgs)
                return handler(retry_req)
            raise e

    return compactor_middleware
