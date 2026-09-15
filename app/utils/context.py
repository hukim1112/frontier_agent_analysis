from dataclasses import dataclass

@dataclass
class AgentContext:
    session_id: str = "unknown"
    logging_enabled: bool = False
    hitl_enabled: bool = False
    recalled_memory: str = ""

    # 🧠 계층형 메모리 미들웨어 제어 플래그
    semantic_memory_enabled: bool = False
    episodic_memory_enabled: bool = False
    auto_recall_episodic: bool = False
    memory_learning_enabled: bool = False
    memory_dir: str = ""
    episodic_db_path: str = ""

