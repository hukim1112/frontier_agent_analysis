"""
===============================================================================
[Scraper Agent Tools] 네비게이팅 및 코딩 통합 도구 바인딩
===============================================================================
Navigator(탐색/인터랙션/브라우징) 6종 + Common(파일/코딩/검색) 8종 = 14종
===============================================================================
"""

from app.tools.navigator import (
    extract_dom_skeleton, get_page_section,
    verify_selectors, interact_page, take_screenshot,
    browse_web,
)
from app.tools.common import (
    file_writer, file_read, file_edit,
    grep_search, glob_search,
    bash_command, web_search, web_fetch,
)

tools_scraper = [
    # L1 + L2 네비게이팅 도구
    extract_dom_skeleton, get_page_section,
    verify_selectors, interact_page, take_screenshot,
    # L3 browser-use 에이전트
    browse_web,
    # 코딩 및 파일 탐색 도구 (common.py 재사용)
    file_writer, file_read, file_edit,
    grep_search, glob_search,
    bash_command, web_search, web_fetch,
]

