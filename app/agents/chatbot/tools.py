"""
===============================================================================
[Chatbot Agent Tools] 범용 기본 도구 통합 바인딩
===============================================================================
Common(파일/노트북/셸/검색/웹 9종) = 9종
===============================================================================
"""

from app.tools.common import (
    file_read, file_edit, file_writer, notebook_edit,
    bash_command, grep_search, glob_search,
    web_fetch, web_search,
)

tools_chatbot = [
    file_read, file_edit, file_writer, notebook_edit,
    bash_command, grep_search, glob_search,
    web_fetch, web_search,
]
