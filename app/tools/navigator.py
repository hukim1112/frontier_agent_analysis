"""
===============================================================================
[Phase 3] Navigator Tools — Playwright 기반 네비게이팅 도구 6종
===============================================================================
PlaywrightManager 싱글턴(CDP 공유) + 6개 도구:
  - extract_dom_skeleton (L1): DOM 구조 맵
  - get_page_section (L1): 스코프드 HTML 정밀 분석  
  - verify_selectors (L1): CSS 셀렉터 검증
  - interact_page (L2): 경량 인터랙션
  - take_screenshot (유틸): 시각적 확인/디버깅
  - browse_web (L3): browser-use 에이전트 자율 탐색

아키텍처 (CDP URL 공유):
  Chrome (--remote-debugging-port) ← Playwright (connect_over_cdp)
                                   ← browser-use (BrowserSession cdp_url)
  동일 Chrome 인스턴스를 Playwright와 browser-use가 CDP로 공유하여
  로그인 세션 등의 상태가 L1/L2/L3 도구 간 자동 공유됩니다.

레퍼런스 패턴 차용:
  - playwright-skill: 쿠키 배너 처리, WSL no-sandbox, locale 설정
  - AAWS navigator.py: _build_skeleton 개선 이식
  - browser-use Playwright Integration 공식 예제: CDP 공유 패턴
"""

import os
import re
import json
import inspect
import asyncio
import shutil
import tempfile
import platform
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from urllib.parse import urlsplit
from datetime import datetime
from typing import Annotated, Any, Optional, Tuple

from bs4 import BeautifulSoup, Tag
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, BeforeValidator, Field


def _coerce_json_string(value: Any) -> Any:
    """네이티브 dict/list 인자가 전달되면 JSON 문자열로 자동 변환합니다 (Pokayoke)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


JsonStringParam = Annotated[str, BeforeValidator(_coerce_json_string)]


# 현재 실행 중인 스크래퍼/에이전트 세션(thread_id) 및 멀티탭(tab_id) 식별자 ContextVar
_DEFAULT_BROWSER_SESSION = "default"
_DEFAULT_BROWSER_TAB = "default"
_browser_session_var: ContextVar[Optional[str]] = ContextVar(
    "_browser_session_var", default=None
)
_browser_tab_var: ContextVar[Optional[str]] = ContextVar(
    "_browser_tab_var", default=None
)


def _current_browser_session_id(session_id: Optional[str] = None) -> str:
    """현재 실행 중인 스크래퍼/에이전트의 브라우저 세션 ID(thread_id)를 반환합니다."""
    if session_id and str(session_id).strip():
        return str(session_id).strip()
    ctx_sid = _browser_session_var.get()
    if ctx_sid:
        return ctx_sid
    try:
        cfg = var_child_runnable_config.get()
        if isinstance(cfg, dict):
            tid = (cfg.get("configurable") or {}).get("thread_id")
            if tid and str(tid).strip():
                return str(tid).strip()
    except Exception:
        pass
    return _DEFAULT_BROWSER_SESSION


def _current_browser_tab_id(tab_id: Optional[str] = None) -> str:
    """현재 실행 중인 도구 호출의 탭 ID(tab_id)를 반환합니다."""
    if tab_id and str(tab_id).strip():
        return str(tab_id).strip()
    ctx_tid = _browser_tab_var.get()
    if ctx_tid:
        return ctx_tid
    return _DEFAULT_BROWSER_TAB


def _current_browser_slot_key(
    session_id: Optional[str] = None,
    tab_id: Optional[str] = None,
) -> str:
    """(session_id, tab_id) 복합 슬롯 키를 반환합니다."""
    sid = _current_browser_session_id(session_id)
    tid = _current_browser_tab_id(tab_id)
    if sid == _DEFAULT_BROWSER_SESSION and tid == _DEFAULT_BROWSER_TAB:
        return _DEFAULT_BROWSER_SESSION
    return f"{sid}::{tid}"


@contextmanager
def browser_session_scope(session_id: str, tab_id: str = _DEFAULT_BROWSER_TAB):
    """명시적으로 브라우저 세션(thread_id)과 탭(tab_id)을 지정하는 컨텍스트 매니저."""
    tok_s = _browser_session_var.set(_current_browser_session_id(session_id))
    tok_t = _browser_tab_var.set(_current_browser_tab_id(tab_id))
    try:
        yield
    finally:
        _browser_tab_var.reset(tok_t)
        _browser_session_var.reset(tok_s)


# =============================================================================
# PlaywrightManager 싱글턴 — 브라우저 생명주기 및 세션·멀티탭 격리 관리
# =============================================================================

# 쿠키 배너 자동 처리 셀렉터 (playwright-skill helpers.js:67-88 패턴 이식 + 한국 대응)
COOKIE_BANNER_SELECTORS = [
    'button:has-text("Accept")',
    'button:has-text("Accept all")',
    'button:has-text("OK")',
    'button:has-text("Got it")',
    'button:has-text("I agree")',
    # 한국 사이트 대응
    'button:has-text("동의")',
    'button:has-text("모두 동의")',
    'button:has-text("확인")',
    '.cookie-accept', '#cookie-accept',
    '[data-testid="cookie-accept"]',
]

# 에러 페이지 감지 시그널 (문맥 기반 판별용 강한 오류 구문)
STRONG_ERROR_SIGNALS = (
    "페이지를 찾을 수 없습니다",
    "page not found",
    "404 not found",
    "403 forbidden",
    "access denied",
    "접근이 거부",
    "서비스 점검 중",
    "502 bad gateway",
    "503 service unavailable",
)
# 단문(< 1000자) 전면 차단 인터스티셜 전용 시그널 (일반 reCAPTCHA/로그인 문구 오탐 배제)
ANTI_BOT_SIGNALS = (
    "verify you are human",
    "checking if the site connection is secure",
    "attention required! | cloudflare",
    "429 too many requests",
    "비정상적인 접근이 감지되어",
)
ERROR_PAGE_SIGNALS = list(STRONG_ERROR_SIGNALS)

# 스크린샷 저장 기본 디렉토리
SCREENSHOT_DIR = "./artifacts/screenshots"


logger = logging.getLogger(__name__)

# CDP 디버깅 포트 (browser-use 기본값 9242 사용, 다른 도구/개발자와 충돌 방지)
CDP_DEBUG_PORT = 9242


class PlaywrightManager:
    """CDP 공유 기반 Playwright 브라우저 생명주기 및 멀티 스크래퍼·멀티탭 관리자.
    
    Chrome을 remote-debugging-port와 함께 기동하고, Playwright가
    connect_over_cdp()로 연결합니다. 동일한 cdp_url을 browser-use의
    BrowserSession에도 전달하여 Playwright와 browser-use가 같은
    Chrome 인스턴스를 공유합니다.
    
    또한 (session_id, tab_id) 복합 슬롯별로 독립된 탭(Page)과 Lock을 관리하여:
    - 여러 스크래퍼가 동시에 실행될 때 각자 독립된 탭에서 간섭 없이 병렬 작업하고,
    - 단일 스크래퍼 내부에서도 tab_id('default', 'list', 'detail' 등)를 분리해
      여러 탭을 동시에 열거나 상태를 유지한 채 오갈 수 있습니다.
    """
    _instance: Optional["PlaywrightManager"] = None
    _playwright = None
    _browser = None
    _chrome_process = None
    _user_data_dir: Optional[str] = None
    _cdp_port: int = CDP_DEBUG_PORT

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._chrome_process = None
        self._user_data_dir = None
        self._pages_by_session: dict[str, object] = {}
        self._target_ids_by_session: dict[str, Optional[str]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._watched_contexts = set()
        self._watched_pages = set()
        self._launch_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._tab_alloc_lock = asyncio.Lock()
        self._browser_use_lock = asyncio.Lock()
        self._suppress_page_event = False

    @property
    def _active_page(self):
        slot = _current_browser_slot_key()
        return self._pages_by_session.get(slot)

    @_active_page.setter
    def _active_page(self, page):
        slot = _current_browser_slot_key()
        if page is None:
            self._pages_by_session.pop(slot, None)
        else:
            self._watch_page(page)
            self._pages_by_session[slot] = page

    @property
    def _active_target_id(self) -> Optional[str]:
        slot = _current_browser_slot_key()
        return self._target_ids_by_session.get(slot)

    @_active_target_id.setter
    def _active_target_id(self, target_id: Optional[str]):
        slot = _current_browser_slot_key()
        if target_id is None:
            self._target_ids_by_session.pop(slot, None)
        else:
            self._target_ids_by_session[slot] = target_id

    def _get_session_lock(
        self,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ) -> asyncio.Lock:
        slot = _current_browser_slot_key(session_id, tab_id)
        if slot == _DEFAULT_BROWSER_SESSION:
            return self._operation_lock
        return self._session_locks.setdefault(slot, asyncio.Lock())
    
    @classmethod
    async def get_instance(cls) -> "PlaywrightManager":
        """싱글턴 인스턴스를 반환합니다. 최초 호출 시 브라우저를 시작합니다."""
        if cls._instance is None:
            cls._instance = cls()
        await cls._instance._ensure_browser()
        return cls._instance
    
    @property
    def cdp_url(self) -> str:
        """browser-use가 동일 Chrome에 연결하기 위한 CDP URL.
        
        browse_web 도구에서 BrowserSession(cdp_url=manager.cdp_url)로 사용합니다.
        """
        return f"http://localhost:{self._cdp_port}"
    
    async def _launch(self):
        """Chrome을 CDP 포트와 함께 기동하고, Playwright로 연결합니다.
        
        공식 browser-use Playwright Integration 패턴 적용:
        1. .env 로드 (HEADLESS, DISPLAY 등 환경변수 반영)
        2. Playwright 시작 → chromium.executable_path로 Chrome 경로 확보
        3. Chrome subprocess 기동 (--remote-debugging-port)
        4. Playwright connect_over_cdp()로 연결
        """
        from playwright.async_api import async_playwright
        from dotenv import load_dotenv
        
        # 0. .env 로드 — Chrome 기동 전에 HEADLESS, DISPLAY 등 환경변수 반영
        load_dotenv()
        
        # 1. Playwright 시작 (Chrome 경로 확보를 위해 먼저 시작)
        self._playwright = await async_playwright().start()
        
        # 2. Chrome을 CDP 디버깅 포트와 함께 기동
        chrome_exe = self._playwright.chromium.executable_path
        self._chrome_process = await self._start_chrome_with_cdp(
            self._cdp_port, chrome_exe
        )
        
        # 3. Playwright를 동일 Chrome에 CDP로 연결
        self._browser = await self._playwright.chromium.connect_over_cdp(
            self.cdp_url
        )
        logger.info(f"🔗 PlaywrightManager: Chrome CDP 연결 완료 ({self.cdp_url})")
    
    async def _start_chrome_with_cdp(
        self, port: int, chrome_exe: str
    ) -> asyncio.subprocess.Process:
        """Chrome을 remote debugging 포트와 함께 기동합니다.
        
        Args:
            port: CDP 디버깅 포트 번호
            chrome_exe: Chrome 실행 파일 경로 (Playwright가 제공)
            
        Returns:
            Chrome subprocess Process 인스턴스
        """
        self._user_data_dir = tempfile.mkdtemp(prefix="bu_cdp_")
        
        if not os.path.exists(chrome_exe):
            raise RuntimeError(
                f"❌ Chrome을 찾을 수 없습니다: {chrome_exe}\n"
                "'python -m playwright install chromium' 을 실행하세요."
            )
        
        # Headless/Headed 모드 결정 (HEADLESS 환경변수, 기본값: true)
        headless = os.environ.get("HEADLESS", "true").lower() == "true"
        
        # Chrome 기동 인자
        chrome_args = [
            chrome_exe,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self._user_data_dir}",
            "--no-sandbox",              # 컨테이너/WSL 환경 필수
            "--disable-dev-shm-usage",   # Docker/WSL 메모리 제한 대응
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-gpu",
            # 멀티탭/멀티스크래퍼 환경에서 백그라운드 탭 렌더링·타이머 절전(Throttling) 방지
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--disable-background-timer-throttling",
            # locale/UA 설정 (new_context 대신 Chrome 프로세스 레벨에서 적용)
            "--lang=ko",
            "--accept-lang=ko-KR,ko",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
            "about:blank",
        ]
        
        if headless:
            chrome_args.insert(3, "--headless=new")
            logger.info("🖥️ Chrome 모드: headless (HEADLESS=true)")
        else:
            # headed 모드: 초기 Chrome 창 크기를 Playwright 컨텍스트와 동일하게 맞춤
            chrome_args.extend([
                "--window-size=1280,720",
                "--window-position=0,0",
            ])
            display = os.environ.get("DISPLAY", "")
            logger.info(f"🖥️ Chrome 모드: headed (HEADLESS=false, DISPLAY={display or '시스템 기본'})")
        
        process = await asyncio.create_subprocess_exec(
            *chrome_args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # CDP 엔드포인트 준비 대기
        try:
            await self._wait_for_cdp_ready(port, timeout=15)
        except RuntimeError:
            # Chrome이 시작 실패한 경우 stderr에서 원인 확인
            stderr_data = await process.stderr.read()
            stderr_msg = stderr_data.decode(errors="replace").strip()
            raise RuntimeError(
                f"❌ Chrome CDP가 15초 내 준비되지 않았습니다 (port: {port})\n"
                f"Chrome stderr: {stderr_msg[:500]}"
            )
        logger.info(f"🚀 Chrome CDP 기동 완료 (PID: {process.pid}, port: {port})")
        return process
    
    async def _wait_for_cdp_ready(self, port: int, timeout: int = 15):
        """CDP 엔드포인트가 응답할 때까지 대기합니다.
        
        Args:
            port: CDP 포트 번호
            timeout: 최대 대기 시간(초)
            
        Raises:
            RuntimeError: 타임아웃 시
        """
        import aiohttp
        for attempt in range(timeout):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{port}/json/version",
                        timeout=aiohttp.ClientTimeout(total=1)
                    ) as resp:
                        if resp.status == 200:
                            return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise RuntimeError(
            f"❌ Chrome CDP가 {timeout}초 내 준비되지 않았습니다 (port: {port})"
        )
    
    async def _ensure_browser(self):
        """Serialize startup and reconnect without discarding the Chrome session."""
        async with self._launch_lock:
            if self._browser is not None and self._browser.is_connected():
                try:
                    # is_connected() can remain True after the Playwright driver
                    # stops, until its disconnect event has been processed.
                    probe = await self._browser.new_browser_cdp_session()
                    await probe.detach()
                    return
                except Exception:
                    logger.info("Playwright CDP 연결을 복구합니다.")
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    logger.debug("Playwright driver already stopped", exc_info=True)
                self._playwright = None
            self._browser = None
            self._pages_by_session.clear()
            self._watched_contexts.clear()
            self._watched_pages.clear()
            if self._chrome_process is not None and self._chrome_process.returncode is None:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                try:
                    self._browser = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
                except Exception:
                    await self._playwright.stop()
                    self._playwright = None
                    raise
            else:
                self._target_ids_by_session.clear()
                await self._launch()

    def _on_page_popup(self, opener_page, popup_page):
        """새 창/탭(target='_blank', window.open)이 열렸을 때 해당 팝업을 연 슬롯만 새 탭으로 갱신합니다."""
        self._watch_page(popup_page)
        for slot, current_page in list(self._pages_by_session.items()):
            if current_page is opener_page:
                self._target_ids_by_session.pop(slot, None)
                self._pages_by_session[slot] = popup_page

    def _watch_page(self, page):
        if page is not None and page not in self._watched_pages:
            page.on("popup", lambda popup, opener=page: self._on_page_popup(opener, popup))
            self._watched_pages.add(page)

    def _remember_page(
        self,
        page,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ):
        if self._suppress_page_event and session_id is None and tab_id is None:
            return
        slot = _current_browser_slot_key(session_id, tab_id)
        if page is not self._pages_by_session.get(slot):
            self._target_ids_by_session.pop(slot, None)
        self._watch_page(page)
        self._pages_by_session[slot] = page

    def _watch_context(self, context):
        if context not in self._watched_contexts:
            self._watched_contexts.add(context)

    @staticmethod
    async def page_target_id(page) -> str:
        """Use CDP target IDs to identify the same tab across both clients."""
        session = await page.context.new_cdp_session(page)
        try:
            info = await session.send("Target.getTargetInfo")
            return info["targetInfo"]["targetId"]
        finally:
            await session.detach()

    async def adopt_target(
        self,
        target_id: str,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ):
        """Select browser-use's final tab, including after a CDP reconnect."""
        slot = _current_browser_slot_key(session_id, tab_id)
        await self._ensure_browser()
        for context in self._browser.contexts:
            self._watch_context(context)
            for page in context.pages:
                if not page.is_closed() and await self.page_target_id(page) == target_id:
                    self._watch_page(page)
                    self._pages_by_session[slot] = page
                    self._target_ids_by_session[slot] = target_id
                    return page
        return None

    async def focus_browser_session(
        self,
        browser_session,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ):
        """Hand the current Playwright tab to browser-use without navigating."""
        from browser_use.browser.events import SwitchTabEvent

        page = await self.get_page(wait_ms=0, session_id=session_id, tab_id=tab_id)
        target_id = await self.page_target_id(page)
        await browser_session.start()
        event = browser_session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
        await event
        await event.event_result(raise_if_any=True, raise_if_none=False)

    async def sync_browser_session(
        self,
        browser_session,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ):
        """Call before stop(), which clears browser-use's focused target ID."""
        target_id = browser_session.agent_focus_target_id
        if target_id:
            return await self.adopt_target(target_id, session_id=session_id, tab_id=tab_id)
        return None

    @staticmethod
    def _same_url(left: str, right: str) -> bool:
        # Browsers add '/' to an empty HTTP path. Preserve query and fragment
        # differences so an explicit navigation still has its intended meaning.
        def normalized(value):
            parts = urlsplit(value)
            if parts.scheme in ("http", "https"):
                parts = parts._replace(path=parts.path or "/")
            return parts.geturl()
        return normalized(left) == normalized(right)

    async def get_page(
        self,
        url: str = None,
        wait_ms: int = 3000,
        reload: bool = False,
        session_id: Optional[str] = None,
        tab_id: Optional[str] = None,
    ):
        """Return the active tab for the current (session_id, tab_id) slot, creating an isolated tab when needed.

        Empty URL continues the current document. The same URL preserves DOM,
        input, scroll and sessionStorage; a different URL navigates this tab.
        Set reload=True to explicitly refresh the document.
        """
        slot = _current_browser_slot_key(session_id, tab_id)
        await self._ensure_browser()
        page = self._pages_by_session.get(slot)
        target_id = self._target_ids_by_session.get(slot)
        if page is None and target_id:
            page = await self.adopt_target(target_id, session_id=session_id, tab_id=tab_id)
        if page is None or page.is_closed():
            async with self._tab_alloc_lock:
                page = self._pages_by_session.get(slot)
                if page is None or page.is_closed():
                    context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
                    self._watch_context(context)
                    assigned_pages = {
                        p for k, p in self._pages_by_session.items()
                        if k != slot and p is not None and not p.is_closed()
                    }
                    other_target_ids = {
                        tid for k, tid in self._target_ids_by_session.items()
                        if k != slot and tid
                    }
                    unassigned_pages = []
                    for p in context.pages:
                        if p.is_closed() or p in assigned_pages:
                            continue
                        if other_target_ids:
                            try:
                                if await self.page_target_id(p) in other_target_ids:
                                    continue
                            except Exception:
                                pass
                        unassigned_pages.append(p)
                    if unassigned_pages and not assigned_pages and not other_target_ids:
                        page = unassigned_pages[-1]
                    else:
                        self._suppress_page_event = True
                        try:
                            page = await context.new_page()
                        finally:
                            self._suppress_page_event = False
                    self._remember_page(page, session_id=session_id, tab_id=tab_id)
        clean_url = url.strip() if isinstance(url, str) else ""
        if clean_url and not clean_url.lower().startswith(("http://", "https://", "about:", "file:", "data:")):
            raise ValueError(
                f"잘못된 URL 형식입니다 ('{clean_url}'): 'http://' 또는 'https://'를 포함한 전체 URL을 입력하세요."
            )
        safe_wait_ms = min(30000, max(0, int(wait_ms or 0)))

        page = self._pages_by_session.get(slot)
        self._watch_context(page.context)
        navigated = False
        if clean_url and not self._same_url(clean_url, page.url):
            await page.goto(clean_url, wait_until="domcontentloaded", timeout=30000)
            navigated = True
        elif reload:
            await page.reload(wait_until="domcontentloaded", timeout=30000)
            navigated = True
        if safe_wait_ms:
            await page.wait_for_timeout(safe_wait_ms)
        if navigated:
            await self._dismiss_cookie_banner(page)
        if page is self._pages_by_session.get(slot):
            self._target_ids_by_session[slot] = await self.page_target_id(page)
        return page
    
    async def _dismiss_cookie_banner(self, page, timeout_ms: int = 2000):
        """쿠키 동의 배너를 자동으로 닫습니다.
        
        playwright-skill helpers.js:67-88 패턴을 Python으로 이식.
        여러 일반적인 셀렉터를 순차 시도하여 첫 매칭 시 클릭합니다.
        """
        per_selector_timeout = timeout_ms // len(COOKIE_BANNER_SELECTORS)
        for selector in COOKIE_BANNER_SELECTORS:
            try:
                locator = page.locator(selector).first
                if await locator.is_visible(timeout=per_selector_timeout):
                    await locator.click(timeout=per_selector_timeout)
                    print("  🍪 쿠키 배너 자동 닫기 완료")
                    return True
            except Exception:
                continue
        return False

    async def close_session_tabs(self, session_id: str) -> int:
        """특정 세션(sub_thread_id)에 할당된 모든 브라우저 탭과 슬롯 상태를 안전하게 정리합니다.

        공유 Chrome 인스턴스가 종료되지 않도록 컨텍스트의 마지막 남은 탭인 경우 닫는 대신 about:blank로 초기화합니다.
        """
        sid = (session_id or "").strip()
        if not sid or sid == _DEFAULT_BROWSER_SESSION:
            return 0

        prefix = f"{sid}::"
        matching_slots = sorted(
            {
                k
                for k in (
                    list(self._pages_by_session.keys())
                    + list(self._target_ids_by_session.keys())
                    + list(self._session_locks.keys())
                )
                if k.startswith(prefix)
            }
        )
        if not matching_slots:
            return 0

        cleaned = 0
        for slot in matching_slots:
            page = self._pages_by_session.pop(slot, None)
            self._target_ids_by_session.pop(slot, None)
            self._session_locks.pop(slot, None)
            if page is None:
                continue
            self._watched_pages.discard(page)
            try:
                if page.is_closed():
                    cleaned += 1
                    continue
                context = getattr(page, "context", None)
                open_pages = [p for p in (getattr(context, "pages", None) or []) if not p.is_closed()]
                if len(open_pages) > 1:
                    await page.close()
                else:
                    await page.goto("about:blank", wait_until="domcontentloaded", timeout=5000)
                cleaned += 1
            except Exception as err:
                logger.warning(f"⚠️ Failed to clean up browser tab for slot '{slot}': {err}")
        return cleaned

    async def close(self):
        """브라우저를 종료하고 리소스를 해제합니다.
        
        Chrome subprocess와 임시 디렉토리도 정리합니다.
        """
        self._pages_by_session.clear()
        self._target_ids_by_session.clear()
        self._session_locks.clear()
        self._watched_contexts.clear()
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        # Chrome subprocess 정리
        if self._chrome_process:
            try:
                self._chrome_process.terminate()
                await asyncio.wait_for(self._chrome_process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._chrome_process.kill()
                except ProcessLookupError:
                    pass
            self._chrome_process = None
            logger.info("🛑 Chrome subprocess 종료 완료")
        # 임시 user data 디렉토리 정리
        if self._user_data_dir and os.path.exists(self._user_data_dir):
            try:
                shutil.rmtree(self._user_data_dir, ignore_errors=True)
            except Exception:
                pass
            self._user_data_dir = None
        PlaywrightManager._instance = None


def _shared_browser_tool(function):
    """Isolate browser tabs by (session_id, tab_id) and serialize tool calls targeting the same tab."""
    @wraps(function)
    async def wrapped(*args, **kwargs):
        try:
            manager = await PlaywrightManager.get_instance()
        except Exception as exc:
            raise ToolException(f"[Error] 브라우저 준비 실패: {exc}") from exc
        sid = _current_browser_session_id()
        tid = _current_browser_tab_id(kwargs.get("tab_id"))
        tok_s = _browser_session_var.set(sid)
        tok_t = _browser_tab_var.set(tid)
        try:
            slot_lock = manager._get_session_lock(sid, tid)
            async with slot_lock:
                return await function(*args, **kwargs)
        finally:
            _browser_tab_var.reset(tok_t)
            _browser_session_var.reset(tok_s)
    return wrapped


def _format_validation_error(error: Exception) -> str:
    """Pydantic 인자 검증 오류를 에이전트가 수정할 수 있는 안내 문자열로 포맷팅합니다."""
    details: list[str] = []
    errors_fn = getattr(error, "errors", None)
    if callable(errors_fn):
        try:
            for item in errors_fn():
                loc = ".".join(str(p) for p in item.get("loc", ())) or "(root)"
                details.append(f"- `{loc}`: {item.get('msg', 'invalid value')}")
        except Exception:
            details = []
    if not details:
        details = [f"- {error}"]
    return (
        "[Error] 도구 인자 검증 실패:\n"
        + "\n".join(details)
        + "\n→ 도구 스키마의 파라미터 이름과 타입을 확인한 뒤 다시 호출하세요."
    )


def _selector_error_hint(selector: str, exc: Exception) -> str:
    """CSS/Playwright 셀렉터 오류 유형별 수정 힌트를 반환합니다."""
    hints: list[str] = []
    sel = selector or ""
    if re.search(r"#\d", sel):
        hints.append(
            "힌트: 숫자로 시작하는 HTML id는 CSS에서 '#123' 대신 '[id=\"123\"]' 형식으로 작성해야 합니다."
        )
    if ":contains(" in sel:
        hints.append(
            "힌트: ':contains()' 대신 Playwright의 ':has-text(\"텍스트\")' 또는 표준 CSS 셀렉터를 사용하세요."
        )
    if re.search(r"(?<!\\)-\[[^\]=]+(?<!\\)\]", sel):
        hints.append(
            "힌트: Tailwind 임의값 클래스(예: pb-[50px])는 CSS 셀렉터에서 '\\\\[' 와 '\\\\]'로 "
            "이스케이프하거나 '[class*=\"pb-[50px]\"]' 속성 셀렉터를 사용하세요."
        )
    if sel.startswith(("/", "xpath=")) and ("/text()" in sel or re.search(r"/@[\w-]+", sel)):
        hints.append(
            "힌트: XPath의 '/text()' 또는 '/@속성' 대신 CSS 셀렉터를 사용하세요. "
            "verify_selectors에서 속성은 '셀렉터::attr(속성명)'으로 조회합니다."
        )
    return " ".join(hints)


def _interaction_error_hint(selector: str, exc: Exception) -> str:
    """Playwright 인터랙션 실패 시 셀렉터·오버레이·숨김 요소·타임아웃 복구 힌트를 반환합니다."""
    base = _selector_error_hint(selector, exc) if selector else ""
    extra: list[str] = [base] if base else []
    msg = str(exc).lower()
    if "intercepts pointer events" in msg:
        extra.append(
            "힌트: 팝업/배너/오버레이가 요소를 가리고 있습니다. 닫기 버튼을 먼저 클릭하거나 모달 상태를 확인하세요."
        )
    elif "not visible" in msg:
        extra.append(
            "힌트: 매칭된 요소가 숨김(hidden) 상태입니다. 화면에 보이는 구체적인 셀렉터를 지정하거나 상위 메뉴/탭을 먼저 여세요."
        )
    elif "timeout" in msg and selector:
        extra.append(
            "힌트: 해당 셀렉터의 요소가 없거나 아직 렌더링되지 않았습니다. extract_dom_skeleton(url='')으로 현재 탭의 실제 셀렉터를 확인하세요."
        )
    return " ".join(extra)


def _detect_error_page(page_text: str, page_title: str = "") -> Optional[str]:
    """문맥(제목 및 본문 길이)을 고려해 실제 HTTP/안티봇 차단 에러 페이지만 감지합니다."""
    title_lower = (page_title or "").strip().lower()
    stripped_text = (page_text or "").strip()
    sample_lower = stripped_text[:800].lower()

    if len(stripped_text) < 1000:
        anti_bot_matched = [
            sig for sig in ANTI_BOT_SIGNALS if sig in title_lower or sig in sample_lower
        ]
        if anti_bot_matched:
            return (
                f"[Error] 안티봇/CAPTCHA 또는 요청 제한(Rate Limit) 화면이 감지되었습니다 "
                f"(제목: '{page_title}'): {', '.join(anti_bot_matched)}. "
                "→ 우회나 반복 재시도를 중단하고 현재 상태를 보존한 채 사용자에게 보고하세요."
            )

    if len(stripped_text) < 1500:
        matched_title = [sig for sig in STRONG_ERROR_SIGNALS if sig in title_lower]
        if matched_title:
            return (
                f"[Error] 에러 페이지 감지 (제목: '{page_title}'): {', '.join(matched_title)}. "
                "→ URL이 올바른지, 또는 접근 권한/요청 차단 여부를 확인하세요."
            )
        if re.search(r"\b(403|404|502|503)\b", title_lower) and any(
            w in title_lower for w in ("error", "forbidden", "not found", "오류", "거부")
        ):
            return (
                f"[Error] HTTP 에러 페이지 감지 (제목: '{page_title}'). "
                "→ URL이 올바른지, 또는 접근 권한/차단 여부를 확인하세요."
            )

    if len(stripped_text) < 1200:
        matched_body = [sig for sig in STRONG_ERROR_SIGNALS if sig in sample_lower]
        if matched_body:
            return (
                f"[Error] 에러 페이지 감지: {', '.join(matched_body)} (페이지 제목: '{page_title}'). "
                "→ URL이 올바른지 확인하거나 browse_web / web_search 등 대안을 검토하세요."
            )
    return None


def _detect_title_error_warning(page_text: str, page_title: str = "") -> str:
    """장문(>= 1500자) 페이지 제목에 에러 문구가 포함된 경우 차단 대신 주의 배너를 반환합니다."""
    stripped_text = (page_text or "").strip()
    if len(stripped_text) < 1500:
        return ""
    title_lower = (page_title or "").strip().lower()
    has_signal = any(sig in title_lower for sig in STRONG_ERROR_SIGNALS) or bool(
        re.search(r"\b(403|404|429|502|503)\b", title_lower)
        and any(w in title_lower for w in ("error", "forbidden", "not found", "오류", "거부"))
    )
    if has_signal:
        return (
            f"⚠️ [주의] 페이지 제목('{page_title}')에 에러 관련 문구가 포함되어 있습니다. "
            "커스텀 에러 페이지인지 실제 콘텐츠인지 아래 내용을 확인하세요.\n"
        )
    return ""


def _check_blank_uninitialized_page(url: str, page_url: str, html_content: str, tab_id: str) -> None:
    """아무 URL도 열리지 않은 초기 빈 탭(about:blank)에서 url='' 호출 시 안내 오류를 발생시킵니다."""
    if (not url or not url.strip()) and page_url == "about:blank" and len(html_content) < 300:
        probe_soup = BeautifulSoup(html_content, "html.parser")
        has_body_tags = bool(probe_soup.body and probe_soup.body.find(True))
        if not has_body_tags and not probe_soup.get_text(strip=True):
            raise ToolException(
                f"[Error] 현재 탭(tab_id='{tab_id}')에 열린 웹페이지가 없습니다 (about:blank). "
                "먼저 'url' 인자에 이동할 전체 주소(http:// 또는 https://)를 지정해 호출하세요."
            )


async def _select_root_element(
    soup: BeautifulSoup,
    page,
    root_selector: str,
    strip_tags: Tuple[str, ...],
) -> Tuple[Optional[Tag], str]:
    """표준 CSS(BeautifulSoup) 선택을 우선 시도하고, Playwright 확장 문법(:has-text 등)이면 라이브 로케이터로 폴백합니다."""
    clean_sel = (root_selector or "").strip()
    if not clean_sel:
        raise ToolException(
            "[Error] 'root_selector'는 비어 있을 수 없습니다. (예: 'body', 'main', 'div.product_list')"
        )
    try:
        return soup.select_one(clean_sel), ""
    except Exception:
        try:
            loc = page.locator(clean_sel)
            match_count = await loc.count()
            if match_count == 0:
                return None, ""
            outer_html = await loc.first.evaluate("el => el.outerHTML")
        except Exception as loc_err:
            hint = _selector_error_hint(clean_sel, loc_err)
            hint_msg = f"\n→ {hint}" if hint else ""
            raise ToolException(
                f"[Error] 잘못된 root_selector 문법입니다 ('{clean_sel}'): {loc_err}{hint_msg}\n"
                "→ 표준 CSS 셀렉터(예: 'body', 'main', 'div.product_list', '[id=\"123\"]')를 사용하거나 "
                "root_selector='body'로 전체 스켈레톤을 먼저 확인하세요."
            ) from loc_err
        frag_soup = BeautifulSoup(outer_html, "html.parser")
        for tag in frag_soup(list(strip_tags)):
            tag.decompose()
        fallback_note = ""
        if match_count > 1:
            fallback_note = (
                f"💡 확장 셀렉터 매칭 {match_count}건 중 첫 번째(최상위) 요소를 선택했습니다. "
                "상위 래퍼 전체가 잡혔다면 'div' 대신 구체적인 하위 태그·클래스"
                "(예: 'li:has-text(...)', 'dl.spec_item:has-text(...)')로 좁히세요."
            )
        return next((c for c in frag_soup.children if isinstance(c, Tag)), None), fallback_note


# =============================================================================
# DOM 스켈레톤 빌더 — AAWS _build_skeleton 개선 이식
# =============================================================================

# 스크래핑에 유용한 data-* 속성 패턴
USEFUL_DATA_ATTRS = re.compile(
    r"^data-(id|price|sale|type|category|name|value|index|product|item|sku|rating|count|url|href|src)$",
    re.IGNORECASE
)


def _build_skeleton(
    element, 
    depth: int = 0, 
    max_depth: int = 8, 
    sibling_limit: int = 5
) -> list[str]:
    """HTML 요소를 재귀적으로 순회하며 구조 맵을 생성합니다.
    
    AAWS navigator.py _build_skeleton 대비 개선:
    - max_depth 기본값 6→8 (SPA 대응)
    - 변형 탐지: 1번째 vs 2번째만 → 1번째/중간/마지막 3개 샘플링
    - data-* 주요 속성 포함 (data-id, data-price 등)
    - sibling_limit 실제 제한 로직 구현
    """
    if depth > max_depth or not hasattr(element, 'children'):
        return []
    
    lines = []
    indent = "  " * depth
    
    # 자식 요소를 시그니처(tag+class)별로 그룹화
    children = [c for c in element.children if isinstance(c, Tag)]
    signature_groups = {}  # {signature: [elements]}
    order = []  # 등장 순서 보존
    
    for child in children:
        cls = " ".join(child.get("class", []))
        sig = f"{child.name}.{cls}" if cls else child.name
        if sig not in signature_groups:
            signature_groups[sig] = []
            order.append(sig)
        signature_groups[sig].append(child)
    
    # sibling_limit 적용: 서로 다른 시그니처 그룹 수 제한
    displayed_groups = 0
    
    for sig in order:
        if displayed_groups >= sibling_limit and len(order) > sibling_limit:
            remaining = len(order) - displayed_groups
            lines.append(f"{indent}  ... (+ {remaining}개 형제 그룹 생략)")
            break
        
        displayed_groups += 1
        group = signature_groups[sig]
        representative = group[0]
        count = len(group)
        
        # 주요 속성 수집 (href, src, id + data-* 속성)
        attrs = []
        for attr_name in ["href", "src", "id"]:
            val = representative.get(attr_name, "")
            if val:
                display_val = val[:80] + ("..." if len(val) > 80 else "")
                attrs.append(f'{attr_name}="{display_val}"')
        
        # data-* 유용한 속성 수집 (개선: 스크래핑에 중요한 data 속성)
        for attr_name, attr_val in representative.attrs.items():
            if isinstance(attr_name, str) and USEFUL_DATA_ATTRS.match(attr_name):
                display_val = str(attr_val)[:50]
                attrs.append(f'{attr_name}="{display_val}"')
        
        # 직접 텍스트 샘플
        direct_text = ""
        child_tags = [c for c in representative.children if isinstance(c, Tag)]
        if representative.string or len(child_tags) <= 2:
            direct_text = representative.get_text(strip=True)[:60]
        
        # 자식 수
        child_count = len(child_tags)
        
        # 한 줄 생성
        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        text_str = f' → "{direct_text}"' if direct_text and len(direct_text) > 1 else ""
        count_str = f" (×{count})" if count > 1 else ""
        children_str = f" (children: {child_count})" if child_count > 0 else ""
        
        line = f"{indent}├─ {sig}{count_str}{attr_str}{children_str}{text_str}"
        lines.append(line)
        
        # 반복 그룹 처리
        if count > 1:
            # 대표 요소(첫 번째) 재귀 펼침
            child_lines = _build_skeleton(representative, depth + 1, max_depth, sibling_limit)
            lines.extend(child_lines)
            
            # 개선: 3-point 변형 탐지 (1번째, 중간, 마지막)
            sample_indices = []
            if count > 1:
                sample_indices.append(1)  # 2번째
            if count > 3:
                sample_indices.append(count // 2)  # 중간
            if count > 2:
                sample_indices.append(count - 1)  # 마지막
            # 중복 제거
            sample_indices = sorted(set(sample_indices))
            
            first_child_sigs = [
                f"{c.name}.{' '.join(c.get('class', []))}" 
                for c in representative.children if isinstance(c, Tag)
            ]
            
            for sample_idx in sample_indices:
                if sample_idx >= len(group):
                    continue
                sample_el = group[sample_idx]
                sample_child_sigs = [
                    f"{c.name}.{' '.join(c.get('class', []))}" 
                    for c in sample_el.children if isinstance(c, Tag)
                ]
                if sample_child_sigs != first_child_sigs:
                    position = "중간" if sample_idx == count // 2 else f"{sample_idx + 1}번째"
                    lines.append(f"{indent}  ⚠️ 구조 변형 발견 — {position} {sig}의 자식 구조가 다름:")
                    variant_lines = _build_skeleton(sample_el, depth + 1, max_depth, sibling_limit)
                    lines.extend(variant_lines)
                    break  # 변형 하나만 보여줌
        else:
            child_lines = _build_skeleton(representative, depth + 1, max_depth, sibling_limit)
            lines.extend(child_lines)
    
    return lines


# =============================================================================
# Tool 1: extract_dom_skeleton (Level 1 — 구조 파악)
# =============================================================================

class ExtractDomSkeletonInput(BaseModel):
    url: str = Field(default="", description="대상 URL (빈 문자열이면 현재 탭 상태 유지)")
    root_selector: str = Field(default="body", description="분석 시작 루트 요소의 CSS 셀렉터 (기본: body)")
    max_depth: int = Field(default=8, description="DOM 탐색 최대 깊이 (기본: 8, SPA 대응)")
    wait_ms: int = Field(default=3000, description="페이지 로드 후 JS 대기 시간(ms) (기본: 3000)")
    reload: bool = Field(default=False, description="현재 문서를 명시적으로 새로고침할 때만 True")
    tab_id: str = Field(
        default="default",
        description="브라우저 탭 식별자 (기본: 'default'). 단일 스크래퍼가 여러 탭(예: 'list', 'detail')을 동시에 열어 상태를 유지하거나 병렬 탐색할 때 지정하세요."
    )

@tool(args_schema=ExtractDomSkeletonInput)
@_shared_browser_tool
async def extract_dom_skeleton(url: str = "", root_selector: str = "body", max_depth: int = 8, wait_ms: int = 3000, reload: bool = False, tab_id: str = "default") -> str:
    """페이지의 DOM 트리 구조를 간결한 구조 맵(태그/클래스/ID/data속성/자식수/샘플텍스트)으로 반환합니다.

    url을 생략하면 현재 공유 탭을 분석합니다. 같은 URL의 상태도 유지하며,
    명시적으로 새로고침하려면 reload=True를 지정하세요.
    이 구조 맵으로 스크래핑 대상 영역을 파악한 뒤, get_page_section으로 해당 영역의 실제 HTML을 확인하세요.
    """
    print(f"\n🦴 [extract_dom_skeleton] {url} (root: {root_selector}, depth: {max_depth}, wait: {wait_ms}ms)")
    safe_max_depth = min(15, max(1, int(max_depth)))

    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url, wait_ms, reload=reload, tab_id=tab_id)
    except Exception as e:
        raise ToolException(f"[Error] 페이지 로드 실패: {e}\n→ URL이 올바른지 확인하세요.") from e

    try:
        html_content = await page.content()
        page_url = page.url
    except Exception as e:
        raise ToolException(f"[Error] HTML 수집 실패: {e}") from e

    _check_blank_uninitialized_page(url, page_url, html_content, tab_id)

    soup = BeautifulSoup(html_content, "html.parser")

    # 에러 페이지 조기 감지
    page_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    page_text = soup.get_text(strip=True)
    error_msg = _detect_error_page(page_text, page_title)
    if error_msg:
        raise ToolException(error_msg)
    title_warning = _detect_title_error_warning(page_text, page_title)

    # 비콘텐츠 태그 제거 (광고 등 특정 클래스는 절대 제거하지 않음)
    strip_tags = ("script", "style", "noscript", "svg", "path", "link", "meta")
    for tag in soup(list(strip_tags)):
        tag.decompose()

    # 루트 요소 탐색 (표준 CSS 실패 시 Playwright 로케이터 폴백 + 문법 힌트)
    root, fallback_note = await _select_root_element(soup, page, root_selector, strip_tags)
    if not root:
        available = [
            t.name + ('.' + '.'.join(t.get('class', [])) if t.get('class') else '')
            for t in (soup.body or soup).children
            if isinstance(t, Tag)
        ][:15]
        return (
            f"[Warning] '{root_selector}' 셀렉터에 해당하는 요소를 찾지 못했습니다.\n"
            f"사용 가능한 최상위 요소: {available}"
        )

    # 스켈레톤 생성
    skeleton_lines = _build_skeleton(root, depth=0, max_depth=safe_max_depth)
    if not skeleton_lines:
        leaf_attrs = []
        for attr_name in ("id", "href", "src"):
            val = root.get(attr_name, "")
            if val:
                leaf_attrs.append(f'{attr_name}="{str(val)[:80]}"')
        for attr_name, attr_val in root.attrs.items():
            if isinstance(attr_name, str) and USEFUL_DATA_ATTRS.match(attr_name):
                leaf_attrs.append(f'{attr_name}="{str(attr_val)[:50]}"')
        direct_text = root.get_text(strip=True)[:100]
        if leaf_attrs or direct_text:
            attr_str = f" [{', '.join(leaf_attrs)}]" if leaf_attrs else ""
            text_str = f' → "{direct_text}"' if direct_text else ""
            skeleton_lines = [
                f"└─ (하위 태그 없음 — 단일 요소){attr_str}{text_str} (원문은 get_page_section으로 확인)"
            ]
        else:
            skeleton_lines = [
                "└─ (비어 있음 — 하위 태그 및 텍스트 없음. JS 렌더링 대기(wait_ms 증가) 또는 iframe/Shadow DOM 여부를 확인하세요)"
            ]

    root_cls = " ".join(root.get("class", []))
    root_sig = f"{root.name}.{root_cls}" if root_cls else root.name
    header_lines = [
        f"{title_warning}🦴 DOM Skeleton: {page_url}",
        f"📍 Root: {root_sig}",
    ]
    if fallback_note:
        header_lines.append(fallback_note)
    header_lines.append("─" * 60)
    header = "\n".join(header_lines)

    skeleton_text = header + "\n" + "\n".join(skeleton_lines)

    # 크기 제한 안전장치
    if len(skeleton_text) > 15000:
        skeleton_text = skeleton_text[:15000] + "\n\n⚠️ [출력 잘림] max_depth를 줄이거나 root_selector를 더 구체적으로 지정하세요."

    print(f"  → 스켈레톤 생성 완료: {len(skeleton_lines)}줄, {len(skeleton_text)}자")
    return skeleton_text


# =============================================================================
# Tool 2: get_page_section (Level 1 — 스코프드 HTML 정밀 분석)
# =============================================================================

class GetPageSectionInput(BaseModel):
    url: str = Field(default="", description="대상 URL (빈 문자열이면 현재 탭 상태 유지)")
    root_selector: str = Field(description="추출할 영역의 CSS 셀렉터 (예: '.product_list', '#content'). 텍스트 기반 매칭 시 Playwright ':has-text(...)'도 폴백 지원")
    max_chars: int = Field(default=30000, description="반환 HTML 최대 문자 수 (200~100000, 기본: 30000 ≈ ~7.5K 토큰)")
    wait_ms: int = Field(default=3000, description="페이지 로드 후 JS 대기 시간(ms) (기본: 3000)")
    reload: bool = Field(default=False, description="현재 문서를 명시적으로 새로고침할 때만 True")
    tab_id: str = Field(
        default="default",
        description="브라우저 탭 식별자 (기본: 'default'). 단일 스크래퍼가 여러 탭(예: 'list', 'detail')을 동시에 열어 상태를 유지하거나 병렬 탐색할 때 지정하세요."
    )

@tool(args_schema=GetPageSectionInput)
@_shared_browser_tool
async def get_page_section(url: str = "", root_selector: str = "body", max_chars: int = 30000, wait_ms: int = 3000, reload: bool = False, tab_id: str = "default") -> str:
    """특정 영역의 정제된 HTML 원문(script/style/svg/noscript 제거됨)을 반환합니다.

    인터랙션 이후에는 url을 생략해 현재 공유 탭의 DOM을 읽으세요.
    같은 URL은 상태를 유지하며, reload=True일 때만 명시적으로 새로고침합니다.
    extract_dom_skeleton으로 구조를 파악한 뒤 data-* 속성, 광고 혼합 패턴, 숨겨진 필드 등을 분석할 때 사용합니다.
    """
    print(f"\n📄 [get_page_section] {url} (selector: {root_selector}, max: {max_chars}자)")
    safe_max_chars = min(100000, max(200, int(max_chars)))

    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url, wait_ms, reload=reload, tab_id=tab_id)
    except Exception as e:
        raise ToolException(f"[Error] 페이지 로드 실패: {e}") from e

    try:
        html_content = await page.content()
        page_url = page.url
    except Exception as e:
        raise ToolException(f"[Error] HTML 수집 실패: {e}") from e

    _check_blank_uninitialized_page(url, page_url, html_content, tab_id)

    soup = BeautifulSoup(html_content, "html.parser")

    # 에러 페이지 조기 감지
    page_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    page_text = soup.get_text(strip=True)
    error_msg = _detect_error_page(page_text, page_title)
    if error_msg:
        raise ToolException(error_msg)
    title_warning = _detect_title_error_warning(page_text, page_title)

    # 비콘텐츠 태그 선제 제거 (미매칭 시 후보 목록 오염 방지)
    strip_tags = ("script", "style", "svg", "noscript", "link", "meta")
    for tag in soup(list(strip_tags)):
        tag.decompose()

    # 대상 영역 추출 (표준 CSS 실패 시 Playwright 로케이터 폴백 + 문법 힌트)
    target, fallback_note = await _select_root_element(soup, page, root_selector, strip_tags)
    if not target:
        available = [
            t.name + ('.' + '.'.join(t.get('class', [])) if t.get('class') else '')
            for t in (soup.body or soup).children
            if isinstance(t, Tag)
        ][:15]
        return (
            f"[Warning] '{root_selector}' 셀렉터에 해당하는 요소를 찾지 못했습니다.\n"
            f"→ extract_dom_skeleton으로 구조를 다시 확인하세요.\n"
            f"최상위 요소 후보: {available}"
        )

    # 정제된 HTML 반환 (prettify 1회 계산)
    cleaned_html = target.prettify()
    total_chars = len(cleaned_html)

    if total_chars > safe_max_chars:
        cleaned_html = cleaned_html[:safe_max_chars] + f"\n\n⚠️ [HTML 잘림 — 전체 {total_chars}자 중 {safe_max_chars}자만 표시] root_selector를 더 구체적으로 지정하세요."

    header_lines = [
        f"{title_warning}📄 Page Section: {page_url}",
        f"📍 Selector: {root_selector}",
        f"📏 크기: {len(cleaned_html)}자",
    ]
    if fallback_note:
        header_lines.append(fallback_note)
    header_lines.append("─" * 60)
    header = "\n".join(header_lines) + "\n"

    print(f"  → HTML 추출 완료: {len(cleaned_html)}자")
    return header + cleaned_html


# =============================================================================
# Tool 3: verify_selectors (Level 1 — 셀렉터 검증)
# =============================================================================

class VerifySelectorsInput(BaseModel):
    url: str = Field(default="", description="대상 URL (빈 문자열이면 현재 탭 상태 유지)")
    selectors_json: JsonStringParam = Field(description="JSON 문자열 셀렉터 딕셔너리. 텍스트: '{\"title\": \"a.title\"}', 속성: '{\"link\": \"a.title::attr(href)\"}'")
    max_samples: int = Field(default=5, description="각 셀렉터당 반환할 최대 샘플 수 (1~50, 기본: 5)")
    wait_ms: int = Field(default=2000, description="페이지 로드 후 대기 시간(ms) (기본: 2000)")
    reload: bool = Field(default=False, description="현재 문서를 명시적으로 새로고침할 때만 True")
    tab_id: str = Field(
        default="default",
        description="브라우저 탭 식별자 (기본: 'default'). 단일 스크래퍼가 여러 탭(예: 'list', 'detail')을 동시에 열어 상태를 유지하거나 병렬 탐색할 때 지정하세요."
    )

@tool(args_schema=VerifySelectorsInput)
@_shared_browser_tool
async def verify_selectors(url: str = "", selectors_json: str = "{}", max_samples: int = 5, wait_ms: int = 2000, reload: bool = False, tab_id: str = "default") -> str:
    """CSS 셀렉터 딕셔너리가 실제로 데이터를 가져오는지 브라우저로 일괄 검증하고 OK/FAILED와 샘플을 반환합니다.

    url을 생략하면 현재 공유 탭을 검증하며, 같은 URL의 상태를 유지합니다. 새로고침은 reload=True로 요청하세요.
    기본적으로 요소의 텍스트를 반환하며, 속성(href, src 등)이 필요하면 '셀렉터::attr(속성명)' 구문을 사용하세요.
    """
    print(f"\n🔍 [verify_selectors] {url}")

    try:
        sel_dict = json.loads(selectors_json)
    except json.JSONDecodeError as e:
        raise ToolException(f"[Error] 유효한 JSON 문자열이어야 합니다: {e}") from e

    if not isinstance(sel_dict, dict) or not sel_dict:
        raise ToolException(
            "[Error] selectors_json은 비어 있지 않은 JSON 객체 문자열"
            "(예: '{\"title\": \"h2.name\", \"link\": \"a::attr(href)\"}')이어야 합니다."
        )

    safe_max_samples = min(50, max(1, int(max_samples)))

    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url, wait_ms, reload=reload, tab_id=tab_id)
    except Exception as e:
        raise ToolException(f"[Error] 페이지 로드 실패: {e}") from e

    try:
        html_content = await page.content()
        _check_blank_uninitialized_page(url, page.url, html_content, tab_id)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"[Error] HTML 상태 확인 실패: {e}") from e

    soup_check = BeautifulSoup(html_content, "html.parser")
    page_title = soup_check.title.string.strip() if soup_check.title and soup_check.title.string else ""
    error_msg = _detect_error_page(soup_check.get_text(strip=True), page_title)
    if error_msg:
        raise ToolException(error_msg)

    output_lines: list[str] = []
    error_lines: list[str] = []
    error_count = 0
    total_matched = 0

    for key, sel in sel_dict.items():
        if not isinstance(sel, str) or not sel.strip():
            err_msg = f"[Error] {key}: 셀렉터 문자열이 비어 있습니다."
            error_lines.append(err_msg)
            output_lines.append(err_msg)
            error_count += 1
            continue

        actual_sel = sel.strip()
        attr_name = ""

        # ::text / ::attr() 커스텀 구문 파싱
        if actual_sel.endswith("::text"):
            actual_sel = actual_sel[:-6].strip()
            if not actual_sel:
                err_msg = f"[Error] {key} ({sel}): '::text' 앞의 CSS 셀렉터가 비어 있습니다."
                error_lines.append(err_msg)
                output_lines.append(err_msg)
                error_count += 1
                continue
        elif "::attr(" in actual_sel:
            match = re.match(r"^(.*?)::attr\((.*?)\)\s*$", actual_sel)
            if not match or not match.group(1).strip() or not match.group(2).strip():
                err_msg = (
                    f"[Error] {key} ({sel}): '셀렉터::attr(속성명)' 형식이 올바르지 않습니다 "
                    "(예: 'a.title::attr(href)')."
                )
                error_lines.append(err_msg)
                output_lines.append(err_msg)
                error_count += 1
                continue
            actual_sel = match.group(1).strip()
            attr_name = match.group(2).strip()

        try:
            elements = await page.query_selector_all(actual_sel)
            samples = []
            for el in elements[:safe_max_samples]:
                if attr_name:
                    val = await el.get_attribute(attr_name)
                else:
                    val = await el.text_content()
                if val and val.strip():
                    samples.append(val.strip())

            match_count = len(elements)
            total_matched += match_count
            if samples:
                output_lines.append(f"[✅ OK] {key} ({sel}): 매칭 {match_count}건 (샘플 {len(samples)}건)")
                for i, s in enumerate(samples):
                    output_lines.append(f"  샘플{i+1}: {s[:100]}")
            else:
                output_lines.append(f"[⚠️ FAILED] {key} ({sel}): 매칭 {match_count}건 (비어 있지 않은 샘플 0건)")
                if match_count > 0 and not attr_name:
                    output_lines.append(
                        "  → 요소는 매칭되었으나 텍스트가 비어 있습니다. href/src/value 등 속성은 '셀렉터::attr(속성명)'으로 확인하세요."
                    )
        except Exception as sel_err:
            error_count += 1
            hint = _selector_error_hint(actual_sel, sel_err)
            err_line = f"[Error] {key} ({sel}): 브라우저 검증 실패 ({sel_err})"
            if hint:
                err_line += f"\n  → {hint}"
            error_lines.append(err_line)
            output_lines.append(err_line)

    if error_count == len(sel_dict):
        raise ToolException("\n".join(error_lines))

    if error_count < len(sel_dict) and total_matched == 0:
        try:
            iframe_count = len(await page.query_selector_all("iframe"))
        except Exception:
            iframe_count = 0
        if iframe_count > 0:
            output_lines.append(
                f"💡 모든 셀렉터가 0건 매칭되었으며 현재 페이지에 iframe({iframe_count}개)이 존재합니다. 대상 데이터가 iframe 내부에 있는지 확인하세요."
            )

    return "\n".join(output_lines)


# =============================================================================
# Tool 4: interact_page (Level 2 — 경량 인터랙션)
# =============================================================================

class InteractPageInput(BaseModel):
    url: str = Field(default="", description="대상 URL (빈 문자열이면 현재 페이지에서 계속)")
    actions_json: JsonStringParam = Field(description=(
        '수행할 액션 리스트 JSON. 지원 액션: '
        '[{"action": "click", "selector": "button.load_more"}], '
        '[{"action": "fill", "selector": "input#search", "value": "검색어"}], '
        '[{"action": "press", "selector": "input#search", "value": "Enter"}], '
        '[{"action": "select", "selector": "select#category", "value": "electronics"}], '
        '[{"action": "scroll", "direction": "down", "amount": 3000}], '
        '[{"action": "wait", "selector": ".results", "timeout_ms": 5000}]'
    ))
    wait_ms: int = Field(default=2000, description="액션 후 결과 대기 시간(ms, 0~30000 범위, 기본: 2000)")
    reload: bool = Field(default=False, description="현재 문서를 명시적으로 새로고침할 때만 True")
    tab_id: str = Field(
        default="default",
        description="브라우저 탭 식별자 (기본: 'default'). 단일 스크래퍼가 여러 탭(예: 'list', 'detail')을 동시에 열어 상태를 유지하거나 병렬 탐색할 때 지정하세요."
    )

@tool(args_schema=InteractPageInput)
@_shared_browser_tool
async def interact_page(url: str = "", actions_json: str = "[]", wait_ms: int = 2000, reload: bool = False, tab_id: str = "default") -> str:
    """페이지에서 클릭, 입력, 키 입력, 드롭다운 선택, 스크롤, 대기 인터랙션을 수행합니다.

    url을 생략하면 활성 탭에서 계속하며, 작업 후 탭을 유지하므로 L1 도구와 스크린샷에서
    같은 상태를 읽을 수 있습니다. 새로고침은 reload=True로 요청합니다.

    단순 인터랙션을 browser-use 없이 1~2초 만에 처리하고, 수행 전후 URL·DOM 변화를 보고합니다.
    """
    print(f"\n🎯 [interact_page] {url or '(현재 페이지)'}")

    try:
        actions = json.loads(actions_json)
    except json.JSONDecodeError as e:
        raise ToolException(f"[Error] 유효한 JSON 액션 리스트여야 합니다: {e}") from e

    if not isinstance(actions, list):
        actions = [actions]
    if not actions:
        raise ToolException("[Error] actions_json에 최소 1개 이상의 액션을 지정해야 합니다.")
    if not all(isinstance(a, dict) for a in actions):
        raise ToolException("[Error] actions_json의 각 요소는 객체(dict)여야 합니다.")

    clamped_wait_ms = max(0, min(int(wait_ms), 30000))

    try:
        manager = await PlaywrightManager.get_instance()
        clean_target_url = (url or "").strip()
        existing_page = manager._pages_by_session.get(_current_browser_slot_key(tab_id=tab_id))
        existing_url = existing_page.url if (existing_page and not existing_page.is_closed()) else ""
        will_navigate = bool(reload or (clean_target_url and not manager._same_url(clean_target_url, existing_url)))
        page = await manager.get_page(
            clean_target_url if clean_target_url else None,
            wait_ms=clamped_wait_ms if will_navigate else 0,
            reload=reload,
            tab_id=tab_id,
        )
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"[Error] 페이지 로드 실패: {e}") from e

    # 인터랙션 전 상태 스냅샷 및 빈 탭 조기 감지
    initial_url = page.url
    try:
        initial_html = await page.content()
        initial_content_length = len(initial_html)
    except Exception:
        initial_html = ""
        initial_content_length = 0
    _check_blank_uninitialized_page(clean_target_url, initial_url, initial_html, tab_id)

    action_results = []
    succeeded_count = 0

    try:
        for action_spec in actions:
            # Follow a popup opened by the preceding action.
            page = await manager.get_page(wait_ms=0, tab_id=tab_id)
            action_type = str(action_spec.get("action", "")).lower().strip()
            selector = str(action_spec.get("selector", "") or "").strip()

            if action_type in ("click", "fill", "select", "wait") and not selector:
                action_results.append(f"  ❌ {action_type} 실패: 'selector' 필드가 비어 있습니다.")
                continue

            try:
                if action_type == "click":
                    await page.click(selector, timeout=5000)
                    action_results.append(f"  ✅ click '{selector}' 성공")
                    succeeded_count += 1

                elif action_type == "fill":
                    value = str(action_spec.get("value", ""))
                    await page.fill(selector, value, timeout=5000)
                    action_results.append(f"  ✅ fill '{selector}' = '{value}' 성공")
                    succeeded_count += 1

                elif action_type == "press":
                    key = str(action_spec.get("value") or action_spec.get("key") or "Enter")
                    if selector:
                        await page.press(selector, key, timeout=5000)
                        action_results.append(f"  ✅ press '{selector}' key='{key}' 성공")
                    else:
                        await page.keyboard.press(key)
                        action_results.append(f"  ✅ press page key='{key}' 성공")
                    succeeded_count += 1

                elif action_type == "select":
                    value = str(action_spec.get("value", ""))
                    await page.select_option(selector, value, timeout=5000)
                    action_results.append(f"  ✅ select '{selector}' = '{value}' 성공")
                    succeeded_count += 1

                elif action_type == "scroll":
                    direction = str(action_spec.get("direction", "down") or "down").lower().strip()
                    raw_amount = action_spec.get("amount")
                    amount = 1000 if raw_amount is None else int(raw_amount)
                    if direction == "down":
                        await page.evaluate("(y) => window.scrollBy(0, y)", abs(amount))
                        action_results.append(f"  ✅ scroll down {abs(amount)}px 성공")
                        succeeded_count += 1
                    elif direction == "up":
                        await page.evaluate("(y) => window.scrollBy(0, y)", -abs(amount))
                        action_results.append(f"  ✅ scroll up {abs(amount)}px 성공")
                        succeeded_count += 1
                    elif direction == "bottom":
                        await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                        action_results.append("  ✅ scroll bottom (페이지 끝) 성공")
                        succeeded_count += 1
                    elif direction == "top":
                        await page.evaluate("() => window.scrollTo(0, 0)")
                        action_results.append("  ✅ scroll top (페이지 상단) 성공")
                        succeeded_count += 1
                    else:
                        action_results.append(
                            f"  ❌ scroll 실패: 지원하지 않는 direction='{direction}' (지원: down, up, bottom, top)"
                        )

                elif action_type == "wait":
                    raw_timeout = action_spec.get("timeout_ms")
                    timeout = max(0, min(5000 if raw_timeout is None else int(raw_timeout), 30000))
                    await page.wait_for_selector(selector, timeout=timeout)
                    action_results.append(f"  ✅ wait '{selector}' 요소 출현 확인")
                    succeeded_count += 1

                else:
                    action_results.append(
                        f"  ⚠️ 미지원 액션: '{action_type}' (지원: click, fill, press, select, scroll, wait)"
                    )

            except Exception as e:
                hint = _interaction_error_hint(selector, e)
                hint_suffix = f" ({hint})" if hint else ""
                action_results.append(f"  ❌ {action_type} '{selector}' 실패: {e}{hint_suffix}")

        if succeeded_count == 0:
            raise ToolException(
                "[Error] 요청한 모든 인터랙션 액션이 실패했습니다.\n" + "\n".join(action_results)
            )

        # 액션 후 대기
        if clamped_wait_ms > 0:
            await page.wait_for_timeout(clamped_wait_ms)

        # 인터랙션 후 상태
        page = await manager.get_page(wait_ms=0, tab_id=tab_id)
        final_url = page.url
        try:
            final_content_length = len(await page.content())
        except Exception:
            final_content_length = 0

        url_changed = final_url != initial_url
        content_changed = abs(final_content_length - initial_content_length) > 100

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"[Error] 인터랙션 중 오류: {e}") from e

    # 결과 요약
    summary_parts = [
        f"🎯 [interact_page 결과]",
        f"  URL: {final_url}" + (" ← URL 변경됨!" if url_changed else ""),
        f"  DOM 변경: {'✅ 감지됨' if content_changed else '❌ 변경 없음'} "
        f"(이전: {initial_content_length}자 → 이후: {final_content_length}자)",
        f"  수행된 액션 ({succeeded_count}/{len(actions)}개 성공):",
    ]
    if succeeded_count < len(actions):
        summary_parts.insert(
            3,
            f"  ⚠️ [부분 실패] 전체 {len(actions)}개 중 {len(actions) - succeeded_count}개 액션이 실패했습니다.",
        )
    summary_parts.extend(action_results)

    return "\n".join(summary_parts)


# =============================================================================
# Tool 5: take_screenshot (유틸리티 — 시각적 확인/디버깅)
# =============================================================================

class TakeScreenshotInput(BaseModel):
    url: str = Field(default="", description="스크린샷 대상 URL (빈 문자열이면 현재 페이지)")
    selector: str = Field(default="", description="특정 요소만 캡처할 CSS/Playwright 셀렉터 (빈 문자열이면 전체 페이지)")
    full_page: bool = Field(default=True, description="전체 페이지 캡처 여부 (기본: True)")
    filename: str = Field(default="", description="저장 파일명 (빈 문자열이면 타임스탬프 자동 생성, SCREENSHOT_DIR 기준 상대 경로)")
    reload: bool = Field(default=False, description="현재 문서를 명시적으로 새로고침할 때만 True")
    tab_id: str = Field(
        default="default",
        description="브라우저 탭 식별자 (기본: 'default'). 단일 스크래퍼가 여러 탭(예: 'list', 'detail')을 동시에 열어 상태를 유지하거나 병렬 탐색할 때 지정하세요."
    )

@tool(args_schema=TakeScreenshotInput)
@_shared_browser_tool
async def take_screenshot(url: str = "", selector: str = "", full_page: bool = True, filename: str = "", reload: bool = False, tab_id: str = "default") -> str:
    """현재 페이지 또는 특정 요소의 스크린샷 PNG 파일을 저장하고 절대 경로를 반환합니다.

    url을 생략하면 현재 공유 탭을 캡처하고 탭을 유지합니다. 같은 URL도 상태를 보존하며,
    명시적 새로고침은 reload=True로 요청합니다. 페이지 분석, 인터랙션 전후 비교, 디버깅에 사용합니다.
    """
    print(f"\n📸 [take_screenshot] {url or '(현재 페이지)'}")

    base_dir = os.path.abspath(SCREENSHOT_DIR)
    os.makedirs(base_dir, exist_ok=True)

    # 파일명 정규화 (./, SCREENSHOT_DIR, artifacts/screenshots/ 접두사 및 상위 경로 이탈 방지)
    default_name = f"screenshot-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
    raw_name = (filename or "").replace("\\", "/").strip()
    while raw_name.startswith("./"):
        raw_name = raw_name[2:]
    normalized_dir = os.path.normpath(SCREENSHOT_DIR).replace("\\", "/")
    while normalized_dir.startswith("./"):
        normalized_dir = normalized_dir[2:]
    for prefix in tuple(dict.fromkeys((f"{normalized_dir}/", "artifacts/screenshots/", "screenshots/"))):
        if prefix != "/" and raw_name.startswith(prefix):
            raw_name = raw_name[len(prefix):]
    cleaned_parts = []
    for p in raw_name.split("/"):
        p = re.sub(r"^[A-Za-z]:", "", p).strip()
        if p and p not in (".", ".."):
            cleaned_parts.append(p)
    parts = cleaned_parts or [default_name]
    if not parts[-1].lower().endswith(".png"):
        parts[-1] += ".png"

    abs_filepath = os.path.abspath(os.path.join(base_dir, *parts))
    if os.path.commonpath([base_dir, abs_filepath]) != base_dir:
        raise ToolException("[Error] 잘못된 스크린샷 저장 경로입니다.")
    os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)

    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url if url else None, wait_ms=2000, reload=reload, tab_id=tab_id)
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"[Error] 페이지 로드 실패: {e}") from e

    if not url or not url.strip():
        try:
            _check_blank_uninitialized_page(url, page.url, await page.content(), tab_id)
        except ToolException:
            raise
        except Exception:
            pass

    try:
        # 멀티탭 환경에서 백그라운드 탭의 Compositor 프레임 생성이 멈춰 타임아웃되는 현상 방지
        await page.bring_to_front()
        if selector:
            loc = page.locator(selector)
            if await loc.count() == 0:
                raise ToolException(
                    f"[Error] '{selector}' 셀렉터에 해당하는 요소를 찾지 못했습니다. "
                    "extract_dom_skeleton(url='')으로 현재 탭의 셀렉터를 다시 확인하세요."
                )
            await loc.first.screenshot(path=abs_filepath, timeout=5000)
        else:
            await page.screenshot(path=abs_filepath, full_page=full_page)
    except ToolException:
        raise
    except Exception as e:
        hint = _selector_error_hint(selector, e) if selector else ""
        hint_suffix = f" ({hint})" if hint else ""
        raise ToolException(f"[Error] 스크린샷 저장 실패: {e}{hint_suffix}") from e

    file_size = os.path.getsize(abs_filepath)
    print(f"  → 스크린샷 저장: {abs_filepath} ({file_size:,} bytes)")

    return f"📸 스크린샷 저장 완료: {abs_filepath} ({file_size:,} bytes)"


# =============================================================================
# Tool 6: browse_web (Level 3 — browser-use 에이전트 자율 탐색)
# =============================================================================

# browser-use 에이전트에게 전달되는 행동 가이드라인 (AAWS 프롬프트 이식)
BROWSER_AGENT_GUIDE = """[당신의 역할]
당신은 데이터 스크래핑 파이프라인의 일부로, 상위 에이전트(Scraper)의 지시를 받아 브라우저를 조작합니다.
스크래핑 작업에서는 속도와 정확성이 핵심이므로, 아래 원칙을 따르세요.

[효율적 행동 원칙]
1. evaluate는 단독 사용: evaluate()는 다른 액션과 같은 스텝에 넣지 마세요. URL 확인은 최초 1회만 단독 스텝으로 실행하세요.
2. 읽기 > 클릭: 텍스트나 URL 확보가 목적이라면 불필요한 클릭(페이지 전환 유발)을 피하세요. 현재 <browser_state>에 이미 표시된 텍스트는 추가 도구 호출 없이 바로 읽고, <browser_state>에 보이지 않는 속성(href, src 등)이나 DOM 구조 확인이 필요할 때만 find_elements 또는 search_page로 조회하세요.
3. 반복 금지: 동일한 액션을 2번 이상 반복하지 마세요. 같은 셀렉터, 같은 evaluate 코드, 같은 URL로의 navigate 모두 해당됩니다.
4. 광고/트래킹 URL 구별: bridge, redirect, ad 등의 키워드가 포함된 URL은 무시하고 실제 콘텐츠 URL 패턴을 찾으세요.
5. 빠른 실패 보고: 페이지 내용이 목적과 전혀 다르면 즉시 실패(success=False)를 보고하여 스텝 예산 낭비를 막으세요.
6. 지시받은 작업만 수행: 상위 에이전트가 요청한 작업만 수행하세요. 요청되지 않은 부가 작업을 하지 마세요.
7. 예상과 다른 페이지 도착 시: navigate 후 기대한 페이지가 아니라면 같은 URL로 재시도하지 말고 현재 상태를 있는 그대로 보고하세요.
"""


class BrowseWebInput(BaseModel):
    task: str = Field(
        description="브라우저가 자율적으로 수행해야 할 자연어 지시사항 "
                    "(예: '로그인 버튼을 찾아 누르고 대시보드로 이동해줘')"
    )
    url: str = Field(
        default="",
        description="시작 URL (빈 문자열이면 현재 페이지에서 계속)"
    )
    max_steps: int = Field(
        default=15,
        description="최대 탐색 단계 수 (1~50 범위, 기본 15, 복잡한 작업은 25~30)"
    )
    tab_id: str = Field(
        default="default",
        description="브라우저 탭 식별자 (기본: 'default'). 단일 스크래퍼가 여러 탭(예: 'list', 'detail')을 동시에 열어 상태를 유지하거나 병렬 탐색할 때 지정하세요."
    )


@tool(args_schema=BrowseWebInput)
@_shared_browser_tool
async def browse_web(task: str, url: str = "", max_steps: int = 15, tab_id: str = "default") -> str:
    """[Level 3 - 최후의 수단] browser-use 에이전트를 가동하여 고난이도 웹 작업을 자율 수행합니다.

    Level 1(정적 분석)과 Level 2(interact_page)로 해결할 수 없는 경우에만 호출하세요.
    동일 Chrome 인스턴스를 공유하므로, 로그인 등의 세션 상태가 이후 L1/L2 도구 호출에서도 유지됩니다.

    적합한 상황: obfuscated DOM(랜덤 클래스명 SPA) 비전 탐색, Shadow DOM/iframe 중첩 구조,
    미지의 UI 자율 탐색. 현재 공유 탭을 이어받고 마지막 활성 탭을 L1/L2에 인계합니다.
    CAPTCHA·2FA 등 사람의 개입이 필요한 단계는 상태를 보존하고 보고하세요.
    """
    task_clean = (task or "").strip()
    if not task_clean:
        raise ToolException("[Error] task 지시사항은 비어 있을 수 없습니다.")

    clamped_max_steps = max(1, min(int(max_steps), 50))
    print(f"\n🌐 [browse_web] task='{task_clean[:80]}...' url='{url}' max_steps={clamped_max_steps}")

    browser_session = None
    manager = None
    try:
        # 1. PlaywrightManager에서 CDP URL 획득 (같은 Chrome 공유)
        manager = await PlaywrightManager.get_instance()
        cdp_url = manager.cdp_url

        async with manager._browser_use_lock:
            try:
                # 2. 초기 탭 이동/검증 후 browser-use BrowserSession을 동일 Chrome에 연결
                from browser_use import Agent, BrowserSession
                page = await manager.get_page(url or None, wait_ms=0, tab_id=tab_id)
                browser_session = BrowserSession(cdp_url=cdp_url, keep_alive=True)
                await manager.focus_browser_session(browser_session, tab_id=tab_id)

                # 3. LLM 설정 (browser-use 전용 래퍼)
                #    .env에서 OPENAI_API_KEY를 로드하여 browser-use ChatOpenAI에 전달
                from dotenv import load_dotenv
                load_dotenv()
                from browser_use import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4.1-mini")

                # 4. task에 가이드라인 프리펜드 (다른 5개 브라우저 도구와 동일한 기본 탭·상태 유지 원칙 적용)
                full_task = BROWSER_AGENT_GUIDE + "\n"
                full_task += (
                    f"[현재 포커스된 탭 URL]\n{page.url}\n"
                    f"[작업 지시]\n{task_clean}\n\n"
                    "[탭 및 상태 관리 원칙]\n"
                    "1. 기본 탭 작업 원칙: 작업 지시(task)에서 명시적으로 '새 탭 열기'를 요구하지 않는 한, "
                    "새 탭(new_tab=True)을 열지 말고 현재 포커스된 탭(new_tab=False)에서 모든 이동과 조작을 수행하세요.\n"
                    "2. 상태 유지 vs 페이지 이동 구분:\n"
                    "   - 현재 페이지 내에서 후속 작업(검색, 필터, 클릭, 스크롤 등)을 이어갈 때는 불필요한 새로고침을 피해 입력값·스크롤·로그인 세션을 유지하세요.\n"
                    "   - 작업 지시에 따라 다른 URL이나 사이트로 이동해야 할 때는 이전 페이지를 보존하려고 새 탭을 열지 말고, 현재 탭에서 바로 해당 URL로 이동(navigate with new_tab=False)하세요.\n"
                    "3. 작업 완료 후에도 후속 도구(get_page_section 등)가 결과를 이어서 분석할 수 있도록 현재 탭을 닫지 말고 열어 두세요."
                )

                # 5. Agent 실행 (이미 PlaywrightManager가 탭 포커스 및 초기 이동을 처리했으므로 directly_open_url=False)
                agent = Agent(
                    task=full_task,
                    llm=llm,
                    browser_session=browser_session,
                    use_vision=True,
                    max_actions_per_step=3,
                    directly_open_url=False,
                )
                # browser-use 0.11.x 버그 보정: find_elements / search_page 등이
                # long_term_memory와 extracted_content를 동시에 반환할 때
                # include_extracted_content_only_once=False(기본값)이면 MessageManager의
                # if/elif 분기에서 extracted_content가 누락되어 LLM에 전달되지 않는 현상 수정
                msg_mgr = getattr(agent, "_message_manager", None)
                orig_update_history = getattr(msg_mgr, "_update_agent_history_description", None)
                if callable(orig_update_history):
                    def _patched_update_history(model_output=None, result=None, step_info=None):
                        if result:
                            for r in result:
                                if (
                                    getattr(r, "extracted_content", None)
                                    and getattr(r, "long_term_memory", None)
                                    and not getattr(r, "is_done", False)
                                ):
                                    r.include_extracted_content_only_once = True
                        return orig_update_history(
                            model_output=model_output, result=result, step_info=step_info
                        )
                    msg_mgr._update_agent_history_description = _patched_update_history

                history = await agent.run(max_steps=clamped_max_steps)

                # 6. 결과 포맷팅 (탭 인계와 세션 연결 정리는 finally에서 수행)
                steps = len(history)
                success = history.is_successful()
                urls = history.urls()
                final_url = urls[-1] if urls else "알 수 없음"
                raw_result = history.final_result()
                if raw_result:
                    result_text = raw_result
                elif success:
                    result_text = "결과 반환 없음"
                else:
                    last_err = ""
                    errors_fn = getattr(history, "errors", None)
                    if callable(errors_fn):
                        try:
                            errs = [e for e in (errors_fn() or []) if e]
                            if errs:
                                last_err = f" (마지막 사유: {str(errs[-1])[:200]})"
                        except Exception:
                            pass
                    if steps >= clamped_max_steps:
                        result_text = f"⚠️ 작업 미완료 상태로 종료됨 (최대 스텝 {clamped_max_steps} 도달){last_err}"
                    else:
                        result_text = f"⚠️ 작업 미완료 상태로 종료됨{last_err}"

                summary = (
                    f"[browse_web 실행 결과]\n"
                    f"  스텝: {steps}/{clamped_max_steps}, 성공: {'✅' if success else '❌'}\n"
                    f"  최종 URL: {final_url}\n"
                    f"  💡 동일 Chrome 세션 유지 — L1/L2 도구에서 인증 상태 접근 가능\n\n"
                    f"{result_text}"
                )

                print(f"  → browse_web 완료: {steps}스텝, {'성공' if success else '실패'}")
                return summary
            finally:
                if browser_session is not None:
                    try:
                        await manager.sync_browser_session(browser_session, tab_id=tab_id)
                    except Exception as exc:
                        logger.warning("browser-use의 마지막 탭 인계 실패: %s", exc)
                    try:
                        await browser_session.stop()
                    except Exception as exc:
                        logger.warning("browser-use 연결 정리 실패: %s", exc)
    except ToolException:
        raise
    except Exception as e:
        error_msg = f"[Error] browse_web 실행 실패: {e}"
        print(f"  ❌ {error_msg}")
        raise ToolException(error_msg) from e


def _finalize_navigator_tool(t):
    """브라우저 도구 공통 메타데이터(docstring 정제, 에러 핸들링)를 일관 적용합니다."""
    if getattr(t, "description", None):
        t.description = inspect.cleandoc(t.description)
    t.handle_tool_error = True
    t.handle_validation_error = _format_validation_error
    return t


for _nav_tool in (
    extract_dom_skeleton,
    get_page_section,
    verify_selectors,
    interact_page,
    take_screenshot,
    browse_web,
):
    _finalize_navigator_tool(_nav_tool)

