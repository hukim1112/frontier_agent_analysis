"""
app/context.py — AgentContext (요청/세션 단위 런타임 실행 컨텍스트 DTO)
"""

from dataclasses import dataclass


@dataclass
class AgentContext:
    # ── 세션 / 사용자 / 워크스페이스 컨텍스트 ──
    agent_name: str = ""
    session_id: str = "unknown"
    parent_session_id: str = ""
    user_id: str = "default"
    workspace_dir: str = ""
    recalled_memory: str = ""          # 전체 회상 메모리 합본 (단독 @dynamic_prompt 및 하위 호환용)
    memory_snapshot: str = ""          # 세션 시작 시 동결된 장기기억 스냅샷 (MEMORY.md + USER.md + PROCEDURAL → Message 3)
    episodic_context: str = ""         # 현재 턴 질문으로 검색된 에피소드 기억 (EPISODIC MEMORY → 대화 끝 <system-reminder>)

    # ── HITL 제어 ──
    hitl_enabled: bool = False

    # ── 계층형 메모리 미들웨어 제어 플래그 ──
    semantic_memory_enabled: bool = True
    semantic_learning_enabled: bool = True
    episodic_memory_enabled: bool = True
    episodic_learning_enabled: bool = True
    procedural_memory_enabled: bool = True
    procedural_learning_enabled: bool = True
    transcript_enabled: bool = True

    # ── 노트북(03_hermes_memory_architecture.ipynb) 하위 호환 필드 ──
    logging_enabled: bool = False
    auto_recall_episodic: bool = False
    memory_learning_enabled: bool = True
    memory_dir: str = ""
    episodic_db_path: str = ""
