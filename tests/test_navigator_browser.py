"""Local Chromium/CDP regression tests; enable with RUN_BROWSER_TESTS=1.

No external websites or model API calls are needed. browser-use's actual
BrowserSession is used, with only its LLM/Agent run replaced by local actions.
"""

import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from langchain_core.messages import ToolMessage

from app.tools import navigator


HTML = """<!doctype html><html><head><title>Local collection fixture</title></head>
<body><input id="term"><button id="add" onclick="appendItem()">Add item</button>
<button id="record" onclick="recorded.push(document.querySelector('#term').value)">Record</button>
<ul id="items"></ul><div style="height:4000px">Scroll fixture</div>
<script>
let recorded = [];
let count = 0;
function appendItem() {
  const item = document.createElement('li');
  item.className = 'item'; item.textContent = 'item-' + (++count);
  document.querySelector('#items').append(item);
}
for (let i=0; i<12; i++) appendItem();
sessionStorage.loads = String(Number(sessionStorage.loads || 0) + 1);
</script></body></html>"""


@unittest.skipUnless(os.environ.get("RUN_BROWSER_TESTS") == "1", "Set RUN_BROWSER_TESTS=1 for local Chromium/CDP tests")
class SharedBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="navigator-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {
            "HEADLESS": "true", "ANONYMIZED_TELEMETRY": "false",
            "BROWSER_USE_CLOUD_SYNC": "false", "BROWSER_USE_LOGGING_LEVEL": "error",
            "BROWSER_USE_CONFIG_DIR": str(self.root / "browser-use"),
            "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        app = web.Application()

        async def fixture(request):
            return web.Response(text=HTML, content_type="text/html")

        app.router.add_get("/{tail:.*}", fixture)
        self.server = web.AppRunner(app)
        await self.server.setup()
        self.addAsyncCleanup(self.server.cleanup)
        site = web.TCPSite(self.server, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/"

        self.manager = navigator.PlaywrightManager()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.manager._cdp_port = sock.getsockname()[1]
        self.instance = patch.object(navigator.PlaywrightManager, "_instance", self.manager)
        self.instance.start()
        self.addCleanup(self.instance.stop)
        self.addAsyncCleanup(self.manager.close)
        self.screenshots = patch.object(navigator, "SCREENSHOT_DIR", str(self.root / "screenshots"))
        self.screenshots.start()
        self.addCleanup(self.screenshots.stop)
        self.page = await self.manager.get_page(self.url, wait_ms=0)

    async def test_interaction_survives_analysis_verification_and_screenshot(self):
        result = await navigator.interact_page.ainvoke({
            "url": self.url,
            "actions_json": json.dumps([
                {"action": "fill", "selector": "#term", "value": "preserved"},
                {"action": "click", "selector": "#add"},
                {"action": "scroll", "direction": "down", "amount": 650},
            ]), "wait_ms": 0,
        })
        self.assertNotIn("[Error]", result)
        await self.page.wait_for_function("window.scrollY > 0")
        scroll = await self.page.evaluate("scrollY")
        loads = await self.page.evaluate("sessionStorage.loads")
        skeleton = await navigator.extract_dom_skeleton.ainvoke({"root_selector": "#items", "wait_ms": 0})
        section = await navigator.get_page_section.ainvoke({"root_selector": "#items", "wait_ms": 0})
        self.assertIn("×13", skeleton)
        self.assertIn("item-13", section)
        self.assertIn(self.url, skeleton)
        self.assertIn(self.url, section)
        verification = await navigator.verify_selectors.ainvoke({"selectors_json": '{"items":".item"}', "max_samples": 3, "wait_ms": 0})
        self.assertIn("매칭 13건 (샘플 3건)", verification)
        screenshot = await navigator.take_screenshot.ainvoke({"filename": "preserved"})
        self.assertNotIn("[Error]", screenshot)
        self.assertTrue((self.root / "screenshots/preserved.png").is_file())
        self.assertIs(await self.manager.get_page(self.url.rstrip("/"), wait_ms=0), self.page)
        self.assertEqual(await self.page.locator("#term").input_value(), "preserved")
        self.assertEqual(await self.page.evaluate("sessionStorage.loads"), loads)
        self.assertEqual(await self.page.evaluate("scrollY"), scroll)
        self.assertEqual(len(self.page.context.pages), 1)

    async def test_explicit_reload_navigation_and_closed_tab_recovery(self):
        await self.page.locator("#term").fill("keep until reload")
        result = await navigator.verify_selectors.ainvoke({"selectors_json": '{"bad":"["}', "wait_ms": 0})
        self.assertIn("[Error]", result)
        self.assertFalse(self.page.is_closed())
        self.assertEqual(await self.page.locator("#term").input_value(), "keep until reload")
        await navigator.get_page_section.ainvoke({"root_selector": "#items", "reload": True, "wait_ms": 0})
        self.assertEqual(await self.page.locator("#term").input_value(), "")
        self.assertEqual(await self.page.evaluate("sessionStorage.loads"), "2")
        self.assertIs(await self.manager.get_page(self.url + "other", wait_ms=0), self.page)
        self.assertEqual(self.page.url, self.url + "other")
        await self.page.close()
        replacement = await self.manager.get_page(self.url, wait_ms=0)
        self.assertFalse(replacement.is_closed())
        self.assertIsNot(replacement, self.page)

    async def test_concurrent_interaction_sequences_do_not_interleave(self):
        async def record(value):
            return await navigator.interact_page.ainvoke({"actions_json": json.dumps([
                {"action": "fill", "selector": "#term", "value": value},
                {"action": "click", "selector": "#record"},
            ]), "wait_ms": 0})
        outputs = await asyncio.gather(record("alpha"), record("beta"))
        self.assertTrue(all("[Error]" not in r for r in outputs))
        self.assertEqual(await self.page.evaluate("recorded"), ["alpha", "beta"])

    async def test_cdp_reconnect_recovers_same_tab_and_document(self):
        await self.page.locator("#term").fill("survives-reconnect")
        target_id = await self.manager.page_target_id(self.page)
        await self.manager._playwright.stop()
        restored = await self.manager.get_page(wait_ms=0)
        self.assertEqual(await self.manager.page_target_id(restored), target_id)
        self.assertEqual(await restored.locator("#term").input_value(), "survives-reconnect")
        self.assertEqual(await restored.evaluate("sessionStorage.loads"), "1")

    async def test_browser_use_handoff_tracks_tab_identity_on_success_and_failure(self):
        from browser_use import BrowserSession
        from browser_use.browser.events import SwitchTabEvent

        other = await self.page.context.new_page()
        await other.goto(self.url)
        original_id = await self.manager.page_target_id(self.page)
        other_id = await self.manager.page_target_id(other)
        await self.manager.adopt_target(original_id)
        await self.page.locator("#term").fill("before-browser-use")
        test = self
        sessions = []

        def make_session(**kwargs):
            session = BrowserSession(**kwargs, enable_default_extensions=False, downloads_path=str(self.root / "downloads"))
            sessions.append(session)
            return session

        class History:
            def __len__(self): return 1
            def is_successful(self): return True
            def urls(self): return [test.url]
            def final_result(self): return "local browser-use action complete"

        class LocalAgent:
            def __init__(self, browser_session, **kwargs):
                self.session = browser_session

            async def run(self, max_steps):
                test.assertEqual(self.session.agent_focus_target_id, original_id)
                actor = await self.session.get_current_page()
                test.assertEqual(await actor.evaluate("() => document.querySelector('#term').value"), "before-browser-use")
                event = self.session.event_bus.dispatch(SwitchTabEvent(target_id=other_id))
                await event
                await event.event_result(raise_if_any=True, raise_if_none=False)
                actor = await self.session.get_current_page()
                await actor.evaluate("() => { document.querySelector('#term').value = 'after-browser-use'; return true; }")
                if max_steps == 1:
                    raise RuntimeError("local fixture interruption")
                return History()

        for max_steps in (2, 1):
            await self.manager.adopt_target(original_id)
            with patch("browser_use.BrowserSession", side_effect=make_session), patch("browser_use.Agent", LocalAgent), patch("browser_use.ChatOpenAI", return_value=object()):
                result = await navigator.browse_web.ainvoke({"task": "local fixture", "max_steps": max_steps})
            self.assertEqual("[Error]" in result, max_steps == 1)
            current = await self.manager.get_page(wait_ms=0)
            self.assertIs(current, other)
            self.assertEqual(await current.locator("#term").input_value(), "after-browser-use")
            self.assertFalse(self.page.is_closed())
            self.assertFalse(other.is_closed())
            self.assertEqual(len(self.page.context.pages), 2)
            self.assertIsNone(sessions[-1].agent_focus_target_id)

    async def test_multi_scraper_sessions_isolate_tabs_and_state_across_interleaved_turns(self):
        """두 개의 스크래퍼(서로 다른 thread_id)가 동시에 여러 턴에 걸쳐 브라우저 도구를 호출할 때
        각자 독립된 탭(Page)에서 URL, DOM 상태, 입력값, 스크롤 등이 섞이지 않고 유지되는지 검증."""
        cfg_alpha = {"configurable": {"thread_id": "sup_01_scraper_alpha"}}
        cfg_beta = {"configurable": {"thread_id": "sup_01_scraper_beta"}}
        url_alpha = self.url + "alpha-market"
        url_beta = self.url + "beta-market"

        # 1턴: 두 스크래퍼가 동시에 서로 다른 URL로 이동 후 각자 다른 횟수만큼 아이템 추가 및 값 기록
        res_a1, res_b1 = await asyncio.gather(
            navigator.interact_page.ainvoke(
                {
                    "url": url_alpha,
                    "actions_json": json.dumps([
                        {"action": "fill", "selector": "#term", "value": "alpha-secret"},
                        {"action": "click", "selector": "#add"},
                        {"action": "click", "selector": "#record"},
                    ]),
                    "wait_ms": 0,
                },
                config=cfg_alpha,
            ),
            navigator.interact_page.ainvoke(
                {
                    "url": url_beta,
                    "actions_json": json.dumps([
                        {"action": "fill", "selector": "#term", "value": "beta-secret"},
                        {"action": "click", "selector": "#add"},
                        {"action": "click", "selector": "#add"},
                        {"action": "click", "selector": "#record"},
                    ]),
                    "wait_ms": 0,
                },
                config=cfg_beta,
            ),
        )
        self.assertNotIn("[Error]", res_a1)
        self.assertNotIn("[Error]", res_b1)

        # 2턴: url="" (빈 문자열)로 각자의 현재 탭 상태를 이어받아 동시에 스켈레톤 및 섹션 추출
        (skel_a, sec_a, ver_a), (skel_b, sec_b, ver_b) = await asyncio.gather(
            asyncio.gather(
                navigator.extract_dom_skeleton.ainvoke({"root_selector": "#items", "wait_ms": 0}, config=cfg_alpha),
                navigator.get_page_section.ainvoke({"root_selector": "#items", "wait_ms": 0}, config=cfg_alpha),
                navigator.verify_selectors.ainvoke({"selectors_json": '{"items":".item"}', "wait_ms": 0}, config=cfg_alpha),
            ),
            asyncio.gather(
                navigator.extract_dom_skeleton.ainvoke({"root_selector": "#items", "wait_ms": 0}, config=cfg_beta),
                navigator.get_page_section.ainvoke({"root_selector": "#items", "wait_ms": 0}, config=cfg_beta),
                navigator.verify_selectors.ainvoke({"selectors_json": '{"items":".item"}', "wait_ms": 0}, config=cfg_beta),
            ),
        )

        # Alpha는 초기 12개 + 1개 = 13개 (url_alpha), Beta는 초기 12개 + 2개 = 14개 (url_beta)여야 함
        self.assertIn(url_alpha, skel_a)
        self.assertIn("×13", skel_a)
        self.assertIn("item-13", sec_a)
        self.assertNotIn("item-14", sec_a)
        self.assertIn("매칭 13건", ver_a)

        self.assertIn(url_beta, skel_b)
        self.assertIn("×14", skel_b)
        self.assertIn("item-14", sec_b)
        self.assertIn("매칭 14건", ver_b)

        # 두 세션의 실제 Page 객체가 서로 다른 독립 탭인지 검증
        page_alpha = await self.manager.get_page(wait_ms=0, session_id="sup_01_scraper_alpha")
        page_beta = await self.manager.get_page(wait_ms=0, session_id="sup_01_scraper_beta")
        self.assertIsNot(page_alpha, page_beta)
        self.assertIsNot(page_alpha, self.page)
        self.assertEqual(await page_alpha.locator("#term").input_value(), "alpha-secret")
        self.assertEqual(await page_beta.locator("#term").input_value(), "beta-secret")
        self.assertEqual(await page_alpha.evaluate("recorded"), ["alpha-secret"])
        self.assertEqual(await page_beta.evaluate("recorded"), ["beta-secret"])

        # CDP 재연결 후에도 두 세션 각각의 탭과 입력 상태가 독립적으로 복구되는지 검증
        tid_alpha = await self.manager.page_target_id(page_alpha)
        tid_beta = await self.manager.page_target_id(page_beta)
        await self.manager._playwright.stop()

        restored_alpha = await self.manager.get_page(wait_ms=0, session_id="sup_01_scraper_alpha")
        restored_beta = await self.manager.get_page(wait_ms=0, session_id="sup_01_scraper_beta")
        self.assertEqual(await self.manager.page_target_id(restored_alpha), tid_alpha)
        self.assertEqual(await self.manager.page_target_id(restored_beta), tid_beta)
        self.assertEqual(await restored_alpha.locator("#term").input_value(), "alpha-secret")
        self.assertEqual(await restored_beta.locator("#term").input_value(), "beta-secret")

    async def test_single_scraper_multi_tab_parallel_and_state_preservation(self):
        """단일 스크래퍼(동일한 thread_id)가 tab_id('list', 'detail')를 다르게 지정하여
        동시에 두 개의 탭에서 작업하고, 'detail' 탭을 다녀온 뒤에도 'list' 탭의 상태가 보존되는지 검증."""
        cfg_single = {"configurable": {"thread_id": "sup_01_scraper_single"}}
        url_list = self.url + "search-list"
        url_detail = self.url + "item-detail-99"

        # 1. 하나의 스크래퍼가 'list' 탭과 'detail' 탭을 동시에 조작
        res_list, res_detail = await asyncio.gather(
            navigator.interact_page.ainvoke(
                {
                    "url": url_list,
                    "tab_id": "list",
                    "actions_json": json.dumps([
                        {"action": "fill", "selector": "#term", "value": "list-filter-kept"},
                        {"action": "click", "selector": "#add"},
                        {"action": "scroll", "direction": "down", "amount": 500},
                    ]),
                    "wait_ms": 0,
                },
                config=cfg_single,
            ),
            navigator.interact_page.ainvoke(
                {
                    "url": url_detail,
                    "tab_id": "detail",
                    "actions_json": json.dumps([
                        {"action": "fill", "selector": "#term", "value": "detail-spec-view"},
                        {"action": "click", "selector": "#add"},
                        {"action": "click", "selector": "#add"},
                        {"action": "click", "selector": "#add"},
                    ]),
                    "wait_ms": 0,
                },
                config=cfg_single,
            ),
        )
        self.assertNotIn("[Error]", res_list)
        self.assertNotIn("[Error]", res_detail)

        # 2. 'detail' 탭을 url=""로 읽은 뒤 다시 'list' 탭을 url=""로 읽었을 때 각각의 상태가 온전히 유지되어야 함
        sec_detail, sec_list = await asyncio.gather(
            navigator.get_page_section.ainvoke(
                {"root_selector": "#items", "tab_id": "detail", "wait_ms": 0},
                config=cfg_single,
            ),
            navigator.get_page_section.ainvoke(
                {"root_selector": "#items", "tab_id": "list", "wait_ms": 0},
                config=cfg_single,
            ),
        )
        self.assertIn(url_detail, sec_detail)
        self.assertIn("item-15", sec_detail)
        self.assertIn(url_list, sec_list)
        self.assertIn("item-13", sec_list)
        self.assertNotIn("item-14", sec_list)

        page_list = await self.manager.get_page(wait_ms=0, session_id="sup_01_scraper_single", tab_id="list")
        page_detail = await self.manager.get_page(wait_ms=0, session_id="sup_01_scraper_single", tab_id="detail")
        self.assertIsNot(page_list, page_detail)
        self.assertEqual(await page_list.locator("#term").input_value(), "list-filter-kept")
        self.assertEqual(await page_detail.locator("#term").input_value(), "detail-spec-view")

    async def test_non_default_slot_popup_does_not_overwrite_default_slot(self):
        """비기본 슬롯(session_id + tab_id)에서 target='_blank' 팝업이 열릴 때,
        해당 슬롯만 새 팝업 탭으로 갱신되고 기본('default') 슬롯은 덮어써지지 않는지 검증."""
        cfg_other = {"configurable": {"thread_id": "sup_01_scraper_popup"}}
        page_other = await self.manager.get_page(
            self.url, wait_ms=0, session_id="sup_01_scraper_popup", tab_id="detail"
        )
        await page_other.set_content(
            f'<a id="popup-other" href="{self.url}popup-from-other" target="_blank">Popup Other</a>'
        )

        res = await navigator.interact_page.ainvoke(
            {
                "tab_id": "detail",
                "actions_json": json.dumps([{"action": "click", "selector": "#popup-other"}]),
                "wait_ms": 0,
            },
            config=cfg_other,
        )
        self.assertIn("click '#popup-other' 성공", res)
        self.assertIn(self.url + "popup-from-other", res)

        sec_other = await navigator.get_page_section.ainvoke(
            {"root_selector": "#items", "tab_id": "detail", "wait_ms": 0},
            config=cfg_other,
        )
        self.assertIn(self.url + "popup-from-other", sec_other)
        # default 슬롯의 활성 탭은 여전히 self.page여야 함
        self.assertIs(await self.manager.get_page(wait_ms=0), self.page)

    async def test_playwright_pseudo_selector_fallback_and_invalid_selector_error(self):
        section = await navigator.get_page_section.ainvoke(
            {"root_selector": "button:has-text('Add item')", "wait_ms": 0}
        )
        self.assertIn("Add item", section)

        skeleton = await navigator.extract_dom_skeleton.ainvoke(
            {"root_selector": "button:has-text('Add item')", "wait_ms": 0}
        )
        self.assertIn("Add item", skeleton)

        direct_err = await navigator.get_page_section.ainvoke(
            {"root_selector": "[", "wait_ms": 0}
        )
        self.assertIn("[Error]", direct_err)

        tool_msg = await navigator.get_page_section.ainvoke(
            {
                "type": "tool_call",
                "id": "call_invalid_sel",
                "name": navigator.get_page_section.name,
                "args": {"root_selector": "[", "wait_ms": 0},
            }
        )
        self.assertIsInstance(tool_msg, ToolMessage)
        self.assertEqual(tool_msg.status, "error")
        self.assertTrue(str(tool_msg.content).startswith("[Error]"))

    async def test_verify_selectors_isolates_partial_errors_and_press_action(self):
        verified = await navigator.verify_selectors.ainvoke(
            {
                "selectors_json": json.dumps(
                    {
                        "items": ".item::text",
                        "empty_attr": "a::attr()",
                        "bad_numeric_id": "button#125796",
                    }
                ),
                "wait_ms": 0,
            }
        )
        self.assertIn("[✅ OK] items (.item::text)", verified)
        self.assertIn("item-1", verified)
        self.assertIn("[Error] empty_attr", verified)
        self.assertIn("[Error] bad_numeric_id", verified)
        self.assertIn('[id="123"]', verified)

        interacted = await navigator.interact_page.ainvoke(
            {
                "actions_json": json.dumps(
                    [
                        {"action": "fill", "selector": "#term", "value": "laptop"},
                        {"action": "press", "selector": "#term", "value": "Enter"},
                        {"action": "scroll", "direction": "bottom", "amount": None},
                        {"action": "scroll", "direction": "sideways"},
                    ]
                ),
                "wait_ms": 0,
            }
        )
        self.assertIn("fill '#term' = 'laptop' 성공", interacted)
        self.assertIn("press '#term' key='Enter' 성공", interacted)
        self.assertIn("scroll bottom (페이지 끝) 성공", interacted)
        self.assertIn("지원하지 않는 direction='sideways'", interacted)
        self.assertIn("⚠️ [부분 실패]", interacted)

        all_fail = await navigator.interact_page.ainvoke(
            {
                "actions_json": json.dumps(
                    [{"action": "click", "selector": "button#125796"}]
                ),
                "wait_ms": 0,
            }
        )
        self.assertTrue(all_fail.startswith("[Error]"))
        self.assertIn('[id="123"]', all_fail)

        tool_msg = await navigator.interact_page.ainvoke(
            {
                "type": "tool_call",
                "id": "call_all_fail",
                "name": navigator.interact_page.name,
                "args": {
                    "actions_json": json.dumps(
                        [{"action": "click", "selector": "button#125796"}]
                    ),
                    "wait_ms": 0,
                },
            }
        )
        self.assertIsInstance(tool_msg, ToolMessage)
        self.assertEqual(tool_msg.status, "error")
        self.assertTrue(str(tool_msg.content).startswith("[Error]"))


class NavigatorSpecAndUnitTests(unittest.TestCase):
    """브라우저 실행 없이 ACI·LangChain Tool 스펙과 헬퍼 함수를 검증합니다."""

    def test_all_navigator_tools_follow_langchain_aci_spec(self):
        tools = [
            navigator.extract_dom_skeleton,
            navigator.get_page_section,
            navigator.verify_selectors,
            navigator.interact_page,
            navigator.take_screenshot,
            navigator.browse_web,
        ]
        for tool_obj in tools:
            self.assertTrue(tool_obj.handle_tool_error, f"{tool_obj.name} handle_tool_error")
            self.assertTrue(
                callable(tool_obj.handle_validation_error),
                f"{tool_obj.name} handle_validation_error",
            )
            desc = tool_obj.description or ""
            self.assertNotIn("Args:", desc, f"{tool_obj.name} duplicated Args:")
            self.assertNotIn("Returns:", desc, f"{tool_obj.name} duplicated Returns:")
            self.assertFalse(
                desc.startswith((" ", "\t", "\n")),
                f"{tool_obj.name} description has leading whitespace",
            )

    def test_tool_call_validation_error_uses_formatted_error_status(self):
        msg = asyncio.run(
            navigator.browse_web.ainvoke(
                {
                    "type": "tool_call",
                    "id": "call_val_err",
                    "name": navigator.browse_web.name,
                    "args": {"max_steps": "not-an-int"},
                }
            )
        )
        self.assertIsInstance(msg, ToolMessage)
        self.assertEqual(msg.status, "error")
        self.assertTrue(str(msg.content).startswith("[Error]"))

    def test_detect_error_page_distinguishes_normal_long_pages_from_http_errors(self):
        long_normal_page = (
            "게이밍 노트북 추천 목록 404건, error 없이 정상 비교 가능. " * 60
        )
        self.assertGreaterEqual(len(long_normal_page), 1500)
        self.assertIsNone(
            navigator._detect_error_page(long_normal_page, page_title="다나와 노트북 가격비교")
        )

        long_tech_article = (
            "Troubleshooting HTTP status codes and server configuration in nginx. " * 35
        )
        self.assertGreaterEqual(len(long_tech_article), 1500)
        self.assertIsNone(
            navigator._detect_error_page(
                long_tech_article, page_title="How to fix 403 Forbidden error"
            )
        )
        warning = navigator._detect_title_error_warning(
            long_tech_article, page_title="How to fix 403 Forbidden error"
        )
        self.assertIn("⚠️", warning)

        short_recaptcha_footer = "회원가입 및 로그인 페이지입니다. Google reCAPTCHA 개인정보 보호정책 및 이용약관이 적용됩니다."
        self.assertIsNone(
            navigator._detect_error_page(short_recaptcha_footer, page_title="로그인")
        )

        short_antibot = navigator._detect_error_page(
            "Verify you are human before continuing to this site.",
            page_title="Just a moment...",
        )
        self.assertIsNotNone(short_antibot)
        self.assertTrue(str(short_antibot).startswith("[Error]"))
        self.assertIn("중단", str(short_antibot))

        short_error = navigator._detect_error_page(
            "404 Not Found - The requested page could not be found.",
            page_title="404 Not Found",
        )
        self.assertIsNotNone(short_error)
        self.assertTrue(str(short_error).startswith("[Error]"))

    def test_selector_error_hint_covers_common_syntax_pitfalls(self):
        dummy_exc = ValueError("invalid selector")
        hint_num = navigator._selector_error_hint("button#125796", dummy_exc)
        self.assertIn('[id="123"]', hint_num)

        hint_contains = navigator._selector_error_hint("div:contains('RAM')", dummy_exc)
        self.assertIn(":has-text", hint_contains)

        hint_tw = navigator._selector_error_hint(".pb-[50px]", dummy_exc)
        self.assertIn("Tailwind", hint_tw)
        self.assertIn("\\\\[", hint_tw)

        hint_xpath = navigator._selector_error_hint("//div[@class='price']/text()", dummy_exc)
        self.assertIn("CSS", hint_xpath)
        self.assertIn("::attr(", hint_xpath)

    def test_json_string_param_pokayoke_coerces_native_dict_and_list(self):
        parsed_selectors = navigator.VerifySelectorsInput(
            selectors_json={"title": "h1", "price": "span.price"}
        )
        self.assertEqual(json.loads(parsed_selectors.selectors_json), {"title": "h1", "price": "span.price"})

        parsed_actions = navigator.InteractPageInput(
            actions_json=[{"action": "click", "selector": "button#submit"}]
        )
        self.assertEqual(
            json.loads(parsed_actions.actions_json),
            [{"action": "click", "selector": "button#submit"}],
        )

        # Emitted JSON Schema stays type='string' for Gemini/OpenAI compatibility
        ver_schema = navigator.VerifySelectorsInput.model_json_schema()
        act_schema = navigator.InteractPageInput.model_json_schema()
        self.assertEqual(ver_schema["properties"]["selectors_json"]["type"], "string")
        self.assertEqual(act_schema["properties"]["actions_json"]["type"], "string")
