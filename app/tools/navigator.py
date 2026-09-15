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
import asyncio
import shutil
import tempfile
import platform
import logging
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup, Tag
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# =============================================================================
# PlaywrightManager 싱글턴 — 브라우저 생명주기 관리
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

# 에러 페이지 감지 시그널 (AAWS get_page_structure 로직 이관)
ERROR_PAGE_SIGNALS = [
    "페이지를 찾을 수 없습니다", "404", "not found", "page not found",
    "오류가 발생", "error", "접근이 거부", "403 forbidden", "서비스 점검",
]

# 스크린샷 저장 기본 디렉토리
SCREENSHOT_DIR = "./artifacts/screenshots"


logger = logging.getLogger(__name__)

# CDP 디버깅 포트 (browser-use 기본값 9242 사용, 다른 도구/개발자와 충돌 방지)
CDP_DEBUG_PORT = 9242


class PlaywrightManager:
    """CDP 공유 기반 Playwright 브라우저 생명주기 관리자.
    
    Chrome을 remote-debugging-port와 함께 기동하고, Playwright가
    connect_over_cdp()로 연결합니다. 동일한 cdp_url을 browser-use의
    BrowserSession에도 전달하여 Playwright와 browser-use가 같은
    Chrome 인스턴스를 공유합니다.
    
    이를 통해:
    - browser-use(L3)가 수행한 로그인 등의 세션 상태를
      L1/L2 도구가 즉시 활용 가능
    - 브라우저 인스턴스 공유로 메모리 효율성 확보
    """
    _instance: Optional["PlaywrightManager"] = None
    _playwright = None
    _browser = None
    _chrome_process = None
    _user_data_dir: Optional[str] = None
    _cdp_port: int = CDP_DEBUG_PORT
    
    @classmethod
    async def get_instance(cls) -> "PlaywrightManager":
        """싱글턴 인스턴스를 반환합니다. 최초 호출 시 브라우저를 시작합니다."""
        if cls._instance is None:
            cls._instance = cls()
        if cls._instance._browser is None:
            await cls._instance._launch()
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
    
    async def get_page(self, url: str = None, wait_ms: int = 3000):
        """기존 브라우저 창에서 새 탭을 열고 URL로 이동합니다.
        
        Chrome의 default context(기본 창)에서 탭을 생성하므로,
        browser-use가 수행한 로그인 세션(쿠키)이 자동으로 공유됩니다.
        
        Args:
            url: 이동할 URL (None이면 빈 페이지)
            wait_ms: 페이지 로드 후 JS 대기 시간(ms)
            
        Returns:
            Playwright Page 인스턴스
        """
        if self._browser is None:
            await self._launch()
        
        # Stale 연결 감지: browser-use stop() 등으로 CDP 연결이 끊어진 경우 자동 재연결
        if not self._browser.is_connected():
            logger.warning("⚠️ PlaywrightManager: 브라우저 연결 끊김 감지 → 자동 재연결 시도")
            # 기존 리소스 정리
            try:
                if self._playwright:
                    await self._playwright.stop()
            except Exception:
                pass
            self._browser = None
            self._playwright = None
            # Chrome subprocess는 아직 살아있을 수 있으므로 CDP 재연결
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.connect_over_cdp(
                    self.cdp_url
                )
                logger.info("🔗 PlaywrightManager: CDP 재연결 성공")
            except Exception as e:
                logger.warning(f"⚠️ CDP 재연결 실패, Chrome 재시작: {e}")
                self._playwright = None
                self._browser = None
                await self._launch()
        
        # 기존 default context 사용 (같은 Chrome 창 내 새 탭)
        if self._browser.contexts:
            context = self._browser.contexts[0]
        else:
            # fallback: default context가 없는 경우 (발생 가능성 낮음)
            context = await self._browser.new_context()
        
        page = await context.new_page()
        
        if url:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(wait_ms)
            # 쿠키 배너 자동 처리
            await self._dismiss_cookie_banner(page)
        
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
    
    async def close(self):
        """브라우저를 종료하고 리소스를 해제합니다.
        
        Chrome subprocess와 임시 디렉토리도 정리합니다.
        """
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
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


def _detect_error_page(page_text: str, page_title: str = "") -> Optional[str]:
    """에러 페이지를 조기 감지합니다 (AAWS get_page_structure 로직 이관).
    
    Returns:
        에러 메시지 문자열 (에러 아니면 None)
    """
    text_sample = page_text[:500].lower()
    detected = [sig for sig in ERROR_PAGE_SIGNALS if sig in text_sample]
    if detected:
        return (
            f"[Error] 에러 페이지 감지: {', '.join(detected)}. "
            f"페이지 제목: '{page_title}'. "
            f"→ URL이 올바른지 확인하세요."
        )
    return None


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
    url: str = Field(description="분석할 웹페이지 URL")
    root_selector: str = Field(default="body", description="분석 시작 루트 요소의 CSS 셀렉터 (기본: body)")
    max_depth: int = Field(default=8, description="DOM 탐색 최대 깊이 (기본: 8, SPA 대응)")
    wait_ms: int = Field(default=3000, description="페이지 로드 후 JS 대기 시간(ms) (기본: 3000)")

@tool(args_schema=ExtractDomSkeletonInput)
async def extract_dom_skeleton(url: str, root_selector: str = "body", max_depth: int = 8, wait_ms: int = 3000) -> str:
    """페이지의 DOM 트리 구조를 간결한 스켈레톤(구조 맵)으로 반환합니다.

    HTML 원문 대신, 태그/클래스/ID/data속성/자식수/샘플텍스트만 추출한
    경량 트리(~5-15KB)를 제공합니다. 이 구조 맵으로 스크래핑 대상 영역을 파악한 뒤,
    get_page_section으로 해당 영역의 실제 HTML을 확인하세요.

    Args:
        url: 분석할 웹페이지 URL
        root_selector: 분석 시작 루트 요소의 CSS 셀렉터 (기본: body)
        max_depth: DOM 탐색 최대 깊이 (기본: 8, SPA 대응)
        wait_ms: 페이지 로드 후 JS 대기 시간(ms) (기본: 3000)
    
    Returns:
        DOM 스켈레톤 구조 맵 텍스트. 반복 패턴, 구조 변형, 주요 속성을 한눈에 파악할 수 있습니다.
    """
    print(f"\n🦴 [extract_dom_skeleton] {url} (root: {root_selector}, depth: {max_depth}, wait: {wait_ms}ms)")
    
    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url, wait_ms)
    except Exception as e:
        return f"[Error] 페이지 로드 실패: {e}\n→ URL이 올바른지 확인하세요."
    
    try:
        html_content = await page.content()
    except Exception as e:
        return f"[Error] HTML 수집 실패: {e}"
    finally:
        await page.close()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 에러 페이지 조기 감지
    page_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    page_text = soup.get_text(strip=True)
    error_msg = _detect_error_page(page_text, page_title)
    if error_msg:
        return error_msg
    
    # 비콘텐츠 태그 제거 (광고 등 특정 클래스는 절대 제거하지 않음)
    for tag in soup(["script", "style", "noscript", "svg", "path", "link", "meta"]):
        tag.decompose()
    
    # 루트 요소 탐색
    root = soup.select_one(root_selector)
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
    skeleton_lines = _build_skeleton(root, depth=0, max_depth=max_depth)
    
    root_cls = " ".join(root.get("class", []))
    root_sig = f"{root.name}.{root_cls}" if root_cls else root.name
    header = f"🦴 DOM Skeleton: {url}\n📍 Root: {root_sig}\n{'─' * 60}"
    
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
    url: str = Field(description="대상 웹페이지 URL")
    root_selector: str = Field(description="추출할 영역의 CSS 셀렉터 (예: '.product_list', '#content')")
    max_chars: int = Field(default=30000, description="반환 HTML 최대 문자 수 (기본: 30000 ≈ ~7.5K 토큰)")
    wait_ms: int = Field(default=3000, description="페이지 로드 후 JS 대기 시간(ms) (기본: 3000)")

@tool(args_schema=GetPageSectionInput)
async def get_page_section(url: str, root_selector: str, max_chars: int = 30000, wait_ms: int = 3000) -> str:
    """특정 영역의 정제된 HTML 원문을 반환합니다.

    extract_dom_skeleton으로 구조를 파악한 뒤, 특정 영역의 실제 HTML을 보고
    data-* 속성, 광고 혼합 패턴, 숨겨진 필드 등을 직접 분석하여
    CSS 셀렉터를 결정할 때 사용합니다.

    Args:
        url: 대상 웹페이지 URL
        root_selector: 추출할 영역의 CSS 셀렉터 (예: '.product_list', '#content')
        max_chars: 반환 HTML 최대 문자 수 (기본: 30000 ≈ ~7.5K 토큰)
        wait_ms: 페이지 로드 후 JS 대기 시간(ms) (기본: 3000)
    
    Returns:
        정제된 HTML 원문. script/style/svg/noscript 태그가 제거된 상태입니다.
    """
    print(f"\n📄 [get_page_section] {url} (selector: {root_selector}, max: {max_chars}자)")
    
    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url, wait_ms)
    except Exception as e:
        return f"[Error] 페이지 로드 실패: {e}"
    
    try:
        html_content = await page.content()
    except Exception as e:
        return f"[Error] HTML 수집 실패: {e}"
    finally:
        await page.close()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 에러 페이지 조기 감지
    page_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    page_text = soup.get_text(strip=True)
    error_msg = _detect_error_page(page_text, page_title)
    if error_msg:
        return error_msg
    
    # 대상 영역 추출
    target = soup.select_one(root_selector)
    if not target:
        # 후보 셀렉터 제안
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
    
    # 비콘텐츠 태그 제거
    for tag in target(["script", "style", "svg", "noscript"]):
        tag.decompose()
    
    # 정제된 HTML 반환
    cleaned_html = target.prettify()
    
    if len(cleaned_html) > max_chars:
        cleaned_html = cleaned_html[:max_chars] + f"\n\n⚠️ [HTML 잘림 — 전체 {len(target.prettify())}자 중 {max_chars}자만 표시] root_selector를 더 구체적으로 지정하세요."
    
    header = (
        f"📄 Page Section: {url}\n"
        f"📍 Selector: {root_selector}\n"
        f"📏 크기: {len(cleaned_html)}자\n"
        f"{'─' * 60}\n"
    )
    
    print(f"  → HTML 추출 완료: {len(cleaned_html)}자")
    return header + cleaned_html


# =============================================================================
# Tool 3: verify_selectors (Level 1 — 셀렉터 검증)
# =============================================================================

class VerifySelectorsInput(BaseModel):
    url: str = Field(description="검증할 웹페이지 URL")
    selectors_json: str = Field(description="JSON 문자열 셀렉터 딕셔너리. 텍스트: '{\"title\": \"a.title\"}', 속성: '{\"link\": \"a.title::attr(href)\"}'")
    max_samples: int = Field(default=5, description="각 셀렉터당 반환할 최대 샘플 수 (기본: 5)")
    wait_ms: int = Field(default=2000, description="페이지 로드 후 대기 시간(ms) (기본: 2000)")

@tool(args_schema=VerifySelectorsInput)
async def verify_selectors(url: str, selectors_json: str, max_samples: int = 5, wait_ms: int = 2000) -> str:
    """CSS 셀렉터가 실제로 데이터를 가져오는지 브라우저로 검증합니다.

    기본적으로 요소의 텍스트를 반환하며, 속성(href, src 등)이 필요하면 ::attr() 구문을 사용하세요.

    Args:
        url: 검증할 웹페이지 URL
        selectors_json: JSON 문자열 셀렉터 딕셔너리.
            텍스트: '{"title": "a.title"}'
            속성: '{"link": "a.title::attr(href)", "image": "img.thumb::attr(src)"}'
        max_samples: 각 셀렉터당 반환할 최대 샘플 수 (기본: 5)
        wait_ms: 페이지 로드 후 대기 시간(ms) (기본: 2000)
    
    Returns:
        각 셀렉터의 검증 결과 (OK/FAILED)와 샘플 데이터
    """
    print(f"\n🔍 [verify_selectors] {url}")
    
    try:
        sel_dict = json.loads(selectors_json)
    except json.JSONDecodeError as e:
        return f"[Error] 유효한 JSON 문자열이어야 합니다: {e}"
    
    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url, wait_ms)
    except Exception as e:
        return f"[Error] 페이지 로드 실패: {e}"
    
    results = {}
    try:
        for key, sel in sel_dict.items():
            actual_sel = sel
            attr_name = ""
            
            # ::attr() 커스텀 구문 파싱
            if "::attr(" in sel:
                match = re.search(r'(.*?)::attr\((.*?)\)', sel)
                if match:
                    actual_sel = match.group(1).strip()
                    attr_name = match.group(2).strip()
            
            elements = await page.query_selector_all(actual_sel)
            samples = []
            for el in elements[:max_samples]:
                if attr_name:
                    val = await el.get_attribute(attr_name)
                else:
                    val = await el.text_content()
                if val:
                    samples.append(val.strip())
            
            results[key] = samples
    except Exception as e:
        return f"[Error] 브라우저 검증 중 오류: {e}"
    finally:
        await page.close()
    
    # 결과 포맷팅
    output_lines = []
    for key, samples in results.items():
        sel = sel_dict[key]
        if samples:
            output_lines.append(f"[✅ OK] {key} ({sel}): 매칭 {len(samples)}건")
            for i, s in enumerate(samples):
                output_lines.append(f"  샘플{i+1}: {s[:100]}")
        else:
            output_lines.append(f"[⚠️ FAILED] {key} ({sel}): 매칭 0건")
    
    return "\n".join(output_lines)


# =============================================================================
# Tool 4: interact_page (Level 2 — 경량 인터랙션)
# =============================================================================

class InteractPageInput(BaseModel):
    url: str = Field(default="", description="대상 URL (빈 문자열이면 현재 페이지에서 계속)")
    actions_json: str = Field(description=(
        '수행할 액션 리스트 JSON. 예: '
        '\'[{"action": "click", "selector": "button.load_more"}]\' '
        '\'[{"action": "fill", "selector": "input#search", "value": "검색어"}]\' '
        '\'[{"action": "scroll", "direction": "down", "amount": 3000}]\' '
        '\'[{"action": "select", "selector": "select#category", "value": "electronics"}]\' '
        '\'[{"action": "wait", "selector": ".results", "timeout_ms": 5000}]\''
    ))
    wait_ms: int = Field(default=2000, description="액션 후 결과 대기 시간(ms) (기본: 2000)")

@tool(args_schema=InteractPageInput)
async def interact_page(url: str, actions_json: str, wait_ms: int = 2000) -> str:
    """페이지에서 클릭, 입력, 스크롤 등 경량 인터랙션을 수행합니다.

    더보기 버튼 클릭, 검색어 입력, 드롭다운 선택, 페이지 스크롤 등
    단순 인터랙션을 browser-use 없이 1~2초 만에 처리합니다.

    Args:
        url: 대상 URL (빈 문자열이면 현재 페이지에서 계속)
        actions_json: 수행할 액션 리스트 JSON. 지원 액션:
            - click: {"action": "click", "selector": "button.more"}
            - fill: {"action": "fill", "selector": "input#q", "value": "검색어"}
            - select: {"action": "select", "selector": "select#cat", "value": "opt1"}
            - scroll: {"action": "scroll", "direction": "down", "amount": 3000}
            - wait: {"action": "wait", "selector": ".results", "timeout_ms": 5000}
        wait_ms: 액션 후 결과 대기 시간(ms) (기본: 2000)

    Returns:
        수행 결과 요약 (현재 URL, 각 액션 성공/실패, DOM 변경 감지)
    """
    print(f"\n🎯 [interact_page] {url or '(현재 페이지)'}")
    
    try:
        actions = json.loads(actions_json)
    except json.JSONDecodeError as e:
        return f"[Error] 유효한 JSON 액션 리스트여야 합니다: {e}"
    
    if not isinstance(actions, list):
        actions = [actions]
    
    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url if url else None, wait_ms=2000)
    except Exception as e:
        return f"[Error] 페이지 로드 실패: {e}"
    
    # 인터랙션 전 상태 스냅샷
    initial_url = page.url
    try:
        initial_content_length = len(await page.content())
    except Exception:
        initial_content_length = 0
    
    action_results = []
    
    try:
        for i, action_spec in enumerate(actions):
            action_type = action_spec.get("action", "").lower()
            selector = action_spec.get("selector", "")
            
            try:
                if action_type == "click":
                    await page.click(selector, timeout=5000)
                    action_results.append(f"  ✅ click '{selector}' 성공")
                    
                elif action_type == "fill":
                    value = action_spec.get("value", "")
                    await page.fill(selector, value, timeout=5000)
                    action_results.append(f"  ✅ fill '{selector}' = '{value}' 성공")
                    
                elif action_type == "select":
                    value = action_spec.get("value", "")
                    await page.select_option(selector, value, timeout=5000)
                    action_results.append(f"  ✅ select '{selector}' = '{value}' 성공")
                    
                elif action_type == "scroll":
                    direction = action_spec.get("direction", "down")
                    amount = action_spec.get("amount", 1000)
                    scroll_y = amount if direction == "down" else -amount
                    await page.evaluate(f"window.scrollBy(0, {scroll_y})")
                    action_results.append(f"  ✅ scroll {direction} {amount}px 성공")
                    
                elif action_type == "wait":
                    timeout = action_spec.get("timeout_ms", 5000)
                    await page.wait_for_selector(selector, timeout=timeout)
                    action_results.append(f"  ✅ wait '{selector}' 요소 출현 확인")
                    
                else:
                    action_results.append(f"  ⚠️ 미지원 액션: '{action_type}'")
                    
            except Exception as e:
                action_results.append(f"  ❌ {action_type} '{selector}' 실패: {e}")
        
        # 액션 후 대기
        await page.wait_for_timeout(wait_ms)
        
        # 인터랙션 후 상태
        final_url = page.url
        try:
            final_content_length = len(await page.content())
        except Exception:
            final_content_length = 0
        
        url_changed = final_url != initial_url
        content_changed = abs(final_content_length - initial_content_length) > 100
        
    except Exception as e:
        return f"[Error] 인터랙션 중 오류: {e}"
    finally:
        await page.close()
    
    # 결과 요약
    summary_parts = [
        f"🎯 [interact_page 결과]",
        f"  URL: {final_url}" + (" ← URL 변경됨!" if url_changed else ""),
        f"  DOM 변경: {'✅ 감지됨' if content_changed else '❌ 변경 없음'} "
        f"(이전: {initial_content_length}자 → 이후: {final_content_length}자)",
        f"  수행된 액션 ({len(actions)}개):",
    ]
    summary_parts.extend(action_results)
    
    return "\n".join(summary_parts)


# =============================================================================
# Tool 5: take_screenshot (유틸리티 — 시각적 확인/디버깅)
# =============================================================================

class TakeScreenshotInput(BaseModel):
    url: str = Field(default="", description="스크린샷 대상 URL (빈 문자열이면 현재 페이지)")
    selector: str = Field(default="", description="특정 요소만 캡처할 CSS 셀렉터 (빈 문자열이면 전체 페이지)")
    full_page: bool = Field(default=True, description="전체 페이지 캡처 여부 (기본: True)")
    filename: str = Field(default="", description="저장 파일명 (빈 문자열이면 타임스탬프 자동 생성)")

@tool(args_schema=TakeScreenshotInput)
async def take_screenshot(url: str = "", selector: str = "", full_page: bool = True, filename: str = "") -> str:
    """현재 페이지 또는 특정 영역의 스크린샷을 저장합니다.

    페이지 분석, 인터랙션 전후 비교, 디버깅, 사용자 보고에 사용합니다.
    저장된 이미지 경로를 반환합니다.

    Args:
        url: 스크린샷 대상 URL (빈 문자열이면 현재 페이지)
        selector: 특정 요소만 캡처할 CSS 셀렉터 (빈 문자열이면 전체)
        full_page: 전체 페이지 캡처 여부 (기본: True)
        filename: 저장 파일명 (빈 문자열이면 타임스탬프 자동 생성)

    Returns:
        저장된 스크린샷 파일의 절대 경로
    """
    print(f"\n📸 [take_screenshot] {url or '(현재 페이지)'}")
    
    # 저장 디렉토리 생성
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    
    # 파일명 생성 (playwright-skill takeScreenshot 패턴: 타임스탬프)
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"screenshot-{timestamp}.png"
    if not filename.endswith(".png"):
        filename += ".png"
    
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    abs_filepath = os.path.abspath(filepath)
    
    try:
        manager = await PlaywrightManager.get_instance()
        page = await manager.get_page(url if url else None, wait_ms=2000)
    except Exception as e:
        return f"[Error] 페이지 로드 실패: {e}"
    
    try:
        if selector:
            # 특정 요소만 캡처
            element = await page.query_selector(selector)
            if element:
                await element.screenshot(path=abs_filepath)
            else:
                return f"[Error] '{selector}' 셀렉터에 해당하는 요소를 찾지 못했습니다."
        else:
            # 전체 페이지 캡처
            await page.screenshot(path=abs_filepath, full_page=full_page)
    except Exception as e:
        return f"[Error] 스크린샷 저장 실패: {e}"
    finally:
        await page.close()
    
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
2. 읽기 > 클릭: 텍스트나 URL 확보가 목적이라면, 클릭(페이지 전환 유발) 대신 find_elements로 속성을 직접 읽으세요.
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
        description="최대 탐색 단계 수 (기본 15, 복잡한 작업은 25~30)"
    )


@tool(args_schema=BrowseWebInput)
async def browse_web(task: str, url: str = "", max_steps: int = 15) -> str:
    """[Level 3 - 최후의 수단] browser-use 에이전트를 가동하여 고난이도 웹 작업을 자율 수행합니다.

    Level 1(정적 분석)과 Level 2(interact_page)로 해결할 수 없는 경우에만 호출하세요.
    동일 Chrome 인스턴스를 공유하므로, 로그인 등의 세션 상태가
    이후 L1/L2 도구 호출에서도 유지됩니다.
    
    적합한 상황:
    - CAPTCHA, 봇 탐지 챌린지
    - obfuscated DOM (랜덤 클래스명 SPA)에서 비전 기반 탐색
    - Shadow DOM / iframe 중첩 구조
    - 복잡한 다단계 인증 (2FA)
    - 미지의 UI에서 자율적 탐색이 필요한 경우

    Args:
        task: 브라우저가 자율적으로 수행해야 할 자연어 지시사항
        url: 시작 URL (빈 문자열이면 현재 페이지에서 계속)
        max_steps: 최대 탐색 단계 수 (기본 15)

    Returns:
        수행 결과 요약 (스텝 수, 성공 여부, 최종 URL, 에이전트 결과)
    """
    print(f"\n🌐 [browse_web] task='{task[:80]}...' url='{url}' max_steps={max_steps}")
    
    try:
        # 1. PlaywrightManager에서 CDP URL 획득 (같은 Chrome 공유)
        manager = await PlaywrightManager.get_instance()
        cdp_url = manager.cdp_url
        
        # 2. browser-use BrowserSession을 동일 Chrome에 연결
        from browser_use import Agent, BrowserSession
        browser_session = BrowserSession(cdp_url=cdp_url)
        
        # 3. LLM 설정 (browser-use 전용 래퍼)
        #    .env에서 OPENAI_API_KEY를 로드하여 browser-use ChatOpenAI에 전달
        from dotenv import load_dotenv
        load_dotenv()
        from browser_use import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4.1-mini")
        
        # 4. task에 가이드라인 프리펜드
        full_task = BROWSER_AGENT_GUIDE + "\n"
        if url:
            full_task += f"[작업]\n'{url}'에 접속하여 다음을 수행하세요: {task}"
        else:
            full_task += f"[작업]\n현재 페이지에서 다음을 수행하세요: {task}"
        
        # 5. Agent 실행
        agent = Agent(
            task=full_task,
            llm=llm,
            browser_session=browser_session,
            use_vision=True,
            max_actions_per_step=3,
        )
        history = await agent.run(max_steps=max_steps)
        
        # 6. BrowserSession 정리 (Chrome은 유지, 세션 리소스만 정리)
        await browser_session.stop()
        
        # 7. 결과 포맷팅
        steps = len(history)
        success = history.is_successful()
        urls = history.urls()
        final_url = urls[-1] if urls else "알 수 없음"
        result_text = history.final_result() or "결과 반환 없음"
        
        summary = (
            f"[browse_web 실행 결과]\n"
            f"  스텝: {steps}/{max_steps}, 성공: {'✅' if success else '❌'}\n"
            f"  최종 URL: {final_url}\n"
            f"  💡 동일 Chrome 세션 유지 — L1/L2 도구에서 인증 상태 접근 가능\n\n"
            f"{result_text}"
        )
        
        print(f"  → browse_web 완료: {steps}스텝, {'성공' if success else '실패'}")
        return summary
        
    except Exception as e:
        error_msg = f"[Error] browse_web 실행 실패: {e}"
        print(f"  ❌ {error_msg}")
        return error_msg
