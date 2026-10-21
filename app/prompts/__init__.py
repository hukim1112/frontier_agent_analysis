"""
Backward-compatible prompt exports (re-exported from modular agent packages).
"""

from app.agents.main_agent.prompt import SUPERVISOR_SYSTEM_PROMPT
from app.agents.chatbot import CHATBOT_SYSTEM_PROMPT
from app.agents.scraper.prompt import SCRAPER_SYSTEM_PROMPT
from app.agents.analyst.prompt import ANALYST_SYSTEM_PROMPT

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "CHATBOT_SYSTEM_PROMPT",
    "SCRAPER_SYSTEM_PROMPT",
    "ANALYST_SYSTEM_PROMPT",
]
