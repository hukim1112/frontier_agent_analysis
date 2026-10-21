"""Inspect an existing AAWS Chrome over CDP without launching or closing it."""

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DEFAULT_CDP_URL = "http://127.0.0.1:9242"
MAX_BODY_BYTES = 256_000
MAX_RESPONSES = 12


def diagnostic_url(url):
    """Keep route and parameter names; omit credentials, values and fragments."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port:
        host += f":{parts.port}"
    query = urlencode([(key, "[omitted]") for key, _ in parse_qsl(parts.query)])
    return urlunsplit((parts.scheme, host, parts.path, query, ""))


def json_shape(value, depth=4):
    """Describe structure, not scalar values; this is not a dataset export."""
    if isinstance(value, dict):
        keys = list(value)[:16]
        result = {"type": "object", "keys": keys, "key_count": len(value)}
        if depth:
            result["fields"] = {key: json_shape(value[key], depth - 1) for key in keys}
        return result
    if isinstance(value, list):
        result = {"type": "array", "length": len(value)}
        if value and depth:
            result["first_item"] = json_shape(value[0], depth - 1)
        return result
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    return {"type": "string"}


async def list_targets(browser):
    targets = []
    for context in browser.contexts:
        for page in context.pages:
            if page.is_closed():
                continue
            session = await context.new_cdp_session(page)
            try:
                info = (await session.send("Target.getTargetInfo"))["targetInfo"]
            finally:
                await session.detach()
            targets.append((page, {
                "target_id": info["targetId"],
                "url": diagnostic_url(page.url),
                "title": await page.title(),
            }))
    return targets


def choose_target(targets, target_id=None):
    if target_id:
        matches = [(page, info) for page, info in targets if info["target_id"] == target_id]
    else:
        matches = [(page, info) for page, info in targets if page.url.startswith(("http://", "https://"))]
    if len(matches) != 1:
        raise ValueError("A unique existing page is required. Run 'tabs', then pass --target-id.")
    return matches[0]


async def inspect_page(page):
    frames = [{"name": frame.name, "url": diagnostic_url(frame.url)} for frame in page.frames]
    scripts = await page.evaluate(
        """limit => Array.from(document.querySelectorAll(
            'script#__NEXT_DATA__, script#__NUXT_DATA__, script[type="application/ld+json"]'
        )).slice(0, 12).map(el => {
            const text = el.textContent || '';
            return {id: el.id, type: el.type, chars: text.length,
                    text: text.length <= limit ? text : null};
        })""",
        MAX_BODY_BYTES,
    )
    for script in scripts:
        raw = script.pop("text")
        if raw is None:
            script["note"] = "Too large for this probe; inspect the required data path separately."
        else:
            try:
                script["json_shape"] = json_shape(json.loads(raw))
            except (ValueError, RecursionError):
                script["note"] = "Not plain JSON; no JavaScript was executed."
    return {"frames": frames, "embedded_data": scripts}


async def capture_network(page, *, click=None, scroll=False, reload=False, duration_ms=3000):
    records, pending = [], []
    ignored = 0
    action = {"kind": "observe", "ok": True}

    async def describe_body(response, record):
        try:
            size = response.headers.get("content-length", "")
            if size.isdigit() and int(size) > MAX_BODY_BYTES:
                record["body_note"] = "Body exceeds inspection limit."
                return
            body = await asyncio.wait_for(response.body(), timeout=2)
            if len(body) > MAX_BODY_BYTES:
                record["body_note"] = "Body exceeds inspection limit."
                return
            record["json_shape"] = json_shape(json.loads(body))
        except Exception as exc:
            record["body_note"] = f"JSON body unavailable: {type(exc).__name__}"

    def on_response(response):
        nonlocal ignored
        request = response.request
        if request.resource_type not in {"document", "xhr", "fetch"}:
            return
        if len(records) >= MAX_RESPONSES:
            ignored += 1
            return
        headers = response.headers
        record = {
            "url": diagnostic_url(response.url),
            "method": request.method,
            "resource_type": request.resource_type,
            "status": response.status,
            "content_type": headers.get("content-type", ""),
            "retry_after": headers.get("retry-after"),
        }
        if request.post_data:
            try:
                record["request_body_shape"] = json_shape(json.loads(request.post_data))
            except (ValueError, RecursionError):
                record["request_body_note"] = "Non-JSON request body; values omitted."
        records.append(record)
        if "json" in record["content_type"].lower():
            pending.append(asyncio.create_task(describe_body(response, record)))

    def on_failed(request):
        nonlocal ignored
        if request.resource_type not in {"document", "xhr", "fetch"}:
            return
        if len(records) >= MAX_RESPONSES:
            ignored += 1
            return
        records.append({
            "url": diagnostic_url(request.url), "method": request.method,
            "resource_type": request.resource_type, "status": None,
            "network_failure": request.failure,
        })

    page.on("response", on_response)
    page.on("requestfailed", on_failed)
    try:
        try:
            if click:
                action["kind"] = "click"
                await page.locator(click).click(timeout=5000)
            elif scroll:
                action["kind"] = "scroll"
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
            elif reload:
                action["kind"] = "reload"
                await page.reload(wait_until="domcontentloaded", timeout=8000)
            await asyncio.sleep(duration_ms / 1000)
        except Exception as exc:
            action.update(ok=False, error=type(exc).__name__)
    finally:
        page.remove_listener("response", on_response)
        page.remove_listener("requestfailed", on_failed)
        if pending:
            await asyncio.gather(*pending)
    return {
        "ok": action["ok"], "action": action, "responses": records,
        "omitted_responses": ignored,
        "scope": "Selected page and its frames during this window; not past traffic, popups or WebSockets.",
    }


async def probe(mode, *, cdp_url=DEFAULT_CDP_URL, target_id=None, **capture_options):
    from playwright.async_api import async_playwright

    # The context manager stops only this Playwright client. Chrome belongs to AAWS.
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(cdp_url, timeout=8000)
        targets = await list_targets(browser)
        if mode == "tabs":
            return {"ok": True, "tabs": [info for _, info in targets]}
        page, target = choose_target(targets, target_id)
        result = {"ok": True, "target": target}
        if mode == "inspect":
            result.update(await inspect_page(page))
        elif mode == "network":
            result.update(await capture_network(page, **capture_options))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # Never page.close(), context.close(), browser.close() or launch() here.
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tabs", "inspect", "network"))
    parser.add_argument("--cdp-url", default=DEFAULT_CDP_URL)
    parser.add_argument("--target-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration-ms", type=int, default=3000)
    trigger = parser.add_mutually_exclusive_group()
    trigger.add_argument("--click", help="Existing selector for one task-authorized interaction.")
    trigger.add_argument("--scroll", action="store_true")
    trigger.add_argument("--reload", action="store_true", help="Explicitly reload the selected page.")
    args = parser.parse_args(argv)
    if not 0 <= args.duration_ms <= 15000:
        parser.error("--duration-ms must be between 0 and 15000.")
    if args.mode != "network" and (args.click or args.scroll or args.reload):
        parser.error("Triggers are only supported in network mode.")
    try:
        result = asyncio.run(probe(
            args.mode, cdp_url=args.cdp_url, target_id=args.target_id,
            click=args.click, scroll=args.scroll, reload=args.reload,
            duration_ms=args.duration_ms,
        ))
        exit_code = 0 if result["ok"] else 1
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        exit_code = 2
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
        print(json.dumps({"ok": result["ok"], "report": str(args.output)}, ensure_ascii=False))
    else:
        print(content)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
