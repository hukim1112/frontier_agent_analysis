from dataclasses import dataclass

@dataclass
class AgentContext:
    logging_enabled: bool = True
    log_path: str = "./artifacts/agent_audit_trail.json"
    response_mode: str = "chat"
    hitl_enabled: bool = False
    debug_mode: bool = False
    
    # 🌟 메모리 연동 및 프롬프트 캐싱 가드레일용 환경 변수 선언
    session_id: str = "unknown"
    user_permission: str = "GUEST"
    active_project: str = "UNKNOWN"
    
    # 🧠 Hermes Memory System 제어 (기본값 True로 세팅하여 메모리 자동 주입 및 학습 활성화)
    episodic_memory_enabled: bool = True     # L2: 에피소드 메모리 자동 인출
    semantic_memory_enabled: bool = True     # L3: MEMORY.md/USER.md 프리페치
    memory_learning_enabled: bool = True     # Closed Loop: after_agent 학습
    episodic_db_path: str = "./app/database/episodic.db"
    memory_dir: str = "./artifacts/memory"    # MEMORY.md, USER.md 저장 위치
    
    # 동적 주입 필드 (런타임에 MemoryMiddleware가 채움)
    recalled_memory: str = ""

