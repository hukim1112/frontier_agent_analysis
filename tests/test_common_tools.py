"""Regression & ACI verification suite for app/tools/common.py (9 essential tools)."""

from __future__ import annotations

import email.message
import http.server
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

import app.tools as app_tools
import app.tools.common as common_tools
from app.tools.common import (
    bash_command,
    file_edit,
    file_read,
    file_writer,
    glob_search,
    grep_search,
    notebook_edit,
    web_fetch,
    web_search,
)

ALL_COMMON_TOOLS = [
    file_read,
    file_edit,
    file_writer,
    notebook_edit,
    bash_command,
    grep_search,
    glob_search,
    web_fetch,
    web_search,
]


# =============================================================================
# 1. LangChain Tool Specification & Export Contract
# =============================================================================


def test_tool_search_removed_and_nine_tools_exported():
    assert not hasattr(common_tools, "tool_search")
    assert not hasattr(common_tools, "ToolSearchInput")
    assert not hasattr(app_tools, "tool_search")
    assert "tool_search" not in app_tools.__all__
    assert all(getattr(app_tools, t.name) is t for t in ALL_COMMON_TOOLS)
    for removed_bundle in ("tools_chatbot", "tools_supervisor", "tools_analyst", "tools_scraper"):
        assert not hasattr(app_tools, removed_bundle)
        assert removed_bundle not in app_tools.__all__
    from app.agents.chatbot.tools import tools_chatbot
    assert [t.name for t in tools_chatbot] == [t.name for t in ALL_COMMON_TOOLS]
    assert len(ALL_COMMON_TOOLS) == 9


def test_langchain_openai_tool_schemas_have_no_docstring_duplication():
    for t in ALL_COMMON_TOOLS:
        schema = convert_to_openai_tool(t)
        assert schema["type"] == "function"
        fn = schema["function"]
        desc = fn["description"]

        assert "Args:" not in desc, f"{t.name} description should not duplicate Args:"
        assert "Returns:" not in desc, f"{t.name} description should not duplicate Returns:"
        assert "Usage:" in desc, f"{t.name} should include Usage: guidance"

        props = fn["parameters"]["properties"]
        assert props, f"{t.name} should declare parameters"
        for param_name, param_schema in props.items():
            assert param_schema.get("description", "").strip(), (
                f"{t.name}.{param_name} must have a non-empty Field description"
            )

    grep_schema = convert_to_openai_tool(grep_search)["function"]["parameters"]
    assert grep_schema["properties"]["output_mode"]["enum"] == [
        "files_with_matches",
        "content",
        "count",
    ]


# =============================================================================
# 2. Bash Command: Timeout Process-Tree Termination, Partial Output & Semantics
# =============================================================================


def test_bash_command_timeout_kills_child_process_and_returns_partial_output(tmp_path):
    pid_file = tmp_path / "child.pid"
    child_py = tmp_path / "child.py"
    parent_py = tmp_path / "parent.py"

    child_py.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    parent_py.write_text(
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(child_py)!r}])\n"
        "deadline = time.time() + 5\n"
        f"while not os.path.exists({str(pid_file)!r}) and time.time() < deadline:\n"
        "    time.sleep(0.02)\n"
        "print('PARTIAL_STDOUT_MARKER', flush=True)\n"
        "print('PARTIAL_STDERR_MARKER', file=sys.stderr, flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    cmd = f"{sys.executable} {str(parent_py)!r}"

    t0 = time.monotonic()
    res = bash_command.invoke({"command": cmd, "timeout_seconds": 1})
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, f"bash_command hung too long on timeout: {elapsed:.2f}s"
    assert "Bash Error:" in res
    assert "timed out" in res
    assert "PARTIAL_STDOUT_MARKER" in res
    assert "PARTIAL_STDERR_MARKER" in res
    assert "timeout_seconds" in res

    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    if os.name == "posix":
        alive = True
        for _ in range(10):
            try:
                os.kill(child_pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                alive = False
                break
            except PermissionError:
                alive = True
                break
        assert not alive, f"Child process PID {child_pid} survived bash_command timeout!"


def test_bash_command_head_tail_truncation_and_exit_semantics(tmp_path):
    res = bash_command.invoke(
        {
            "command": (
                f"{sys.executable} -c \"print('HEAD_LINE'); "
                "print('x' * 500); print('TAIL_TRACEBACK_END')\""
            ),
            "max_stdout_length": 150,
        }
    )
    assert "HEAD_LINE" in res
    assert "TAIL_TRACEBACK_END" in res
    assert "STDOUT TRUNCATED" in res

    target = tmp_path / "sample.txt"
    target.write_text("hello\n", encoding="utf-8")
    grep_res = bash_command.invoke({"command": f"grep NEVER_MATCHES_XYZ {str(target)!r}"})
    assert "[Exit Code: 1" in grep_res
    assert "No matches found" in grep_res

    empty_res = bash_command.invoke({"command": "   "})
    assert "Bash Error:" in empty_res


# =============================================================================
# 3. File Operations (file_read, file_edit, file_writer, notebook_edit)
# =============================================================================


def test_file_read_pagination_bounds_binary_and_external_tmp(tmp_path):
    empty_f = tmp_path / "empty.txt"
    empty_f.write_text("", encoding="utf-8")
    assert "Empty file: 0 lines" in file_read.invoke({"file_path": str(empty_f)})

    multi_f = tmp_path / "lines.txt"
    multi_f.write_text("\n".join(f"line_{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    page1 = file_read.invoke({"file_path": str(multi_f), "offset": 1, "limit": 3})
    assert "(Lines 1-3 of 10)" in page1
    assert "offset=4 to continue reading" in page1

    oob = file_read.invoke({"file_path": str(multi_f), "offset": 50})
    assert "FileRead Error:" in oob
    assert "10" in oob

    bin_f = tmp_path / "data.bin"
    bin_f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    assert "binary file" in file_read.invoke({"file_path": str(bin_f)}).lower()

    long_f = tmp_path / "minified.json"
    long_f.write_text('{"k": "' + ("a" * 3000) + '"}\n', encoding="utf-8")
    long_res = file_read.invoke({"file_path": str(long_f)})
    assert "line truncated" in long_res

    assert "directory" in file_read.invoke({"file_path": str(tmp_path)}).lower()

    missing_res = file_read.invoke({"file_path": str(tmp_path / "linez.txt")})
    assert "lines.txt" in missing_res

    # Compaction swap compatibility: files under /tmp must remain readable
    swap_like = Path("/tmp/test_aaws_common_read_swap.txt")
    try:
        swap_like.write_text("swap row 1\nswap row 2\n", encoding="utf-8")
        assert "swap row 2" in file_read.invoke({"file_path": str(swap_like)})
    finally:
        if swap_like.exists():
            swap_like.unlink()


def test_file_edit_crlf_replace_all_hints_and_safety(tmp_path):
    crlf_file = tmp_path / "win_script.py"
    crlf_file.write_bytes(b"line1 = 1\r\nline2 = 'dup'\r\nline3 = 'dup'\r\n")

    # Multi-match without replace_all reports matching lines
    dup_err = file_edit.invoke(
        {
            "file_path": str(crlf_file),
            "target_content": "'dup'",
            "replacement_content": "'fixed'",
        }
    )
    assert "FileEdit Error:" in dup_err
    assert "lines: 2, 3" in dup_err
    assert "replace_all=True" in dup_err

    # Line-number prefix mistake detection
    prefix_err = file_edit.invoke(
        {
            "file_path": str(crlf_file),
            "target_content": "     1 | line1 = 1",
            "replacement_content": "line1 = 2",
        }
    )
    assert "line-number prefixes" in prefix_err

    # replace_all=True succeeds and preserves CRLF
    ok_res = file_edit.invoke(
        {
            "file_path": str(crlf_file),
            "target_content": "'dup'",
            "replacement_content": "'fixed'",
            "replace_all": True,
        }
    )
    assert ok_res.startswith("SUCCESS:")
    assert "--- Updated Snippet ---" in ok_res
    raw_after = crlf_file.read_bytes()
    assert b"\r\n" in raw_after
    assert raw_after == b"line1 = 1\r\nline2 = 'fixed'\r\nline3 = 'fixed'\r\n"

    # No leftover temp files
    assert not list(tmp_path.glob(".tmp_*"))

    # Empty target_content, identical replacement, and .ipynb guard
    assert "cannot be empty" in file_edit.invoke(
        {"file_path": str(crlf_file), "target_content": "", "replacement_content": "x"}
    )
    assert "identical" in file_edit.invoke(
        {"file_path": str(crlf_file), "target_content": "line1", "replacement_content": "line1"}
    )
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text('{"cells": []}', encoding="utf-8")
    assert "notebook_edit" in file_edit.invoke(
        {"file_path": str(nb_path), "target_content": "cells", "replacement_content": "x"}
    )


def test_file_writer_overwrite_guard_and_notebook_edit(tmp_path):
    target = tmp_path / "nested" / "app.py"
    res1 = file_writer.invoke({"file_path": str(target), "content": "print(1)\n"})
    assert res1.startswith("SUCCESS:")
    assert target.read_text(encoding="utf-8") == "print(1)\n"

    guard_res = file_writer.invoke(
        {"file_path": str(target), "content": "print(2)\n", "overwrite": False}
    )
    assert "FileWrite Error:" in guard_res
    assert "file_edit" in guard_res
    assert "overwrite=True" in guard_res

    assert "directory" in file_writer.invoke({"file_path": str(tmp_path), "content": "x"}).lower()

    nb = tmp_path / "lab.ipynb"
    nb.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
                    {
                        "cell_type": "code",
                        "execution_count": 7,
                        "metadata": {},
                        "outputs": [{"output_type": "stream", "text": ["old"]}],
                        "source": ["x = 1\n"],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    oob_nb = notebook_edit.invoke(
        {"notebook_path": str(nb), "cell_index": 2, "new_code": "y = 2"}
    )
    assert "0 to 1" in oob_nb

    ok_nb = notebook_edit.invoke(
        {"notebook_path": str(nb), "cell_index": 1, "new_code": "x = 42\nprint(x)"}
    )
    assert ok_nb.startswith("SUCCESS:")
    updated_nb = json.loads(nb.read_text(encoding="utf-8"))
    assert updated_nb["cells"][1]["outputs"] == []
    assert updated_nb["cells"][1]["execution_count"] is None
    assert updated_nb["cells"][1]["source"] == ["x = 42\n", "print(x)"]


# =============================================================================
# 4. Search Tools (glob_search, grep_search)
# =============================================================================


def test_glob_search_dotfile_inclusion_vcs_exclusion_and_mtime_sort(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "env_langchain_1315").mkdir()
    (tmp_path / "env_langchain_1315" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (tmp_path / "env_langchain_1315" / "site.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "env_config").mkdir()
    (tmp_path / "env_config" / "settings.py").write_text("y=1\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "artifact.py").write_text("z=1\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    ci_yml = tmp_path / ".github" / "workflows" / "ci.yml"
    ci_yml.write_text("name: CI\n", encoding="utf-8")
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.pyc\n", encoding="utf-8")
    old_py = tmp_path / "old.py"
    old_py.write_text("print('old')\n", encoding="utf-8")
    new_py = tmp_path / "new.py"
    new_py.write_text("print('new')\n", encoding="utf-8")

    os.utime(old_py, (1_700_000_000, 1_700_000_000))
    os.utime(new_py, (1_800_000_000, 1_800_000_000))

    res = glob_search.invoke({"pattern": "**/*", "search_path": str(tmp_path)})
    assert ".gitignore" in res
    assert "ci.yml" in res
    assert ".git/config" not in res and ".git\\config" not in res
    assert "site.py" not in res, "real virtualenv (pyvenv.cfg) must be excluded"
    assert "settings.py" in res, "ordinary env_* folders must stay visible"
    assert "artifact.py" in res, "build/ folders must stay visible"
    assert res.index("new.py") < res.index("old.py")

    limited = glob_search.invoke({"pattern": "**/*", "search_path": str(tmp_path), "head_limit": 2})
    assert "more files omitted" in limited

    err = glob_search.invoke({"pattern": "*.py", "search_path": str(tmp_path / "missing_dir")})
    assert err.startswith("Glob Error:")


def test_grep_search_modes_case_sensitivity_and_binary_skip(tmp_path):
    f1 = tmp_path / "alpha.py"
    f1.write_text("TargetToken = 1\ntargettoken = 2\nTargetToken = 3\n", encoding="utf-8")
    bin_f = tmp_path / "blob.bin"
    bin_f.write_bytes(b"\x00TargetToken\x00")

    ci_res = grep_search.invoke(
        {"pattern": "TargetToken", "search_path": str(tmp_path), "output_mode": "count"}
    )
    assert "3 total occurrences across 1 files" in ci_res

    cs_res = grep_search.invoke(
        {
            "pattern": "TargetToken",
            "search_path": str(tmp_path),
            "output_mode": "content",
            "case_sensitive": True,
            "head_limit": 1,
        }
    )
    assert "TargetToken = 1" in cs_res
    assert "targettoken = 2" not in cs_res
    assert "more matches exist" in cs_res

    bad_re = grep_search.invoke({"pattern": "[unclosed", "search_path": str(tmp_path)})
    assert "Invalid regex" in bad_re


# =============================================================================
# 5. Web Tools (web_fetch, web_search)
# =============================================================================


class _LocalTestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/page":
            body = (
                "<html><head><style>body{color:red;}</style>"
                "<script>const secret = 'JS_NOISE';</script></head>"
                "<body><h1>Welcome</h1><p>Readable body text.</p></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redirect-file":
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.end_headers()
        elif self.path == "/redirect-ftp":
            self.send_response(302)
            self.send_header("Location", "ftp://example.com/secret.txt")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        elif self.path == "/pdf":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            self.wfile.write(b"%PDF-1.4")
        else:
            self.send_response(404)
            self.end_headers()


def test_web_fetch_scheme_guard_html_cleaning_and_http_errors():
    assert "Unsupported URL scheme" in web_fetch.invoke({"url": "file:///etc/passwd"})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LocalTestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ok = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/page"})
        assert "Welcome" in ok
        assert "Readable body text." in ok
        assert "JS_NOISE" not in ok
        assert "color:red" not in ok

        redir_err = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/redirect-file"})
        assert "WebFetch Error: Redirect blocked" in redir_err

        ftp_redir_err = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/redirect-ftp"})
        assert "WebFetch Error: Redirect blocked" in ftp_redir_err
        assert "ftp" in ftp_redir_err

        pdf_err = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/pdf"})
        assert "binary content (application/pdf)" in pdf_err

        not_found = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/missing"})
        assert "HTTP 404" in not_found
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_web_search_warning_isolation_and_error_reporting(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    class _FailingTavily:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, query):
            raise RuntimeError("Tavily quota exceeded")

    class _FailingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=5):
            raise RuntimeError("DDG rate limit")

    class _DummyTavilyMod:
        TavilySearch = _FailingTavily

    class _DummyDDGSMod:
        DDGS = _FailingDDGS

    filters_before = list(warnings.filters)
    with patch.dict(
        sys.modules,
        {"langchain_tavily": _DummyTavilyMod(), "ddgs": _DummyDDGSMod()},
    ):
        err = web_search.invoke({"query": "python aci test", "max_results": 3})

    assert list(warnings.filters) == filters_before, "web_search must not mutate global warnings.filters"
    assert "Tavily quota exceeded" in err
    assert "DDG rate limit" in err

    class _SuccessDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=5):
            return [{"title": "Example", "href": "https://example.com", "body": "Snippet"}]

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with patch.dict(sys.modules, {"ddgs": type("M", (), {"DDGS": _SuccessDDGS})()}):
        ok = web_search.invoke({"query": "example", "max_results": 2})
    assert "1. [Example](https://example.com): Snippet" in ok


# =============================================================================
# 6. Regression Tests for Review Findings (duplication, kill, perms, symlink, status)
# =============================================================================


def test_bash_timeout_partial_output_is_not_duplicated():
    # Marker text only appears in output ("ONCE_2_MARKER"), not in the command string.
    res = bash_command.invoke({"command": "echo ONCE_$((1+1))_MARKER; sleep 30", "timeout_seconds": 1})
    assert "timed out" in res
    assert res.count("ONCE_2_MARKER") == 1, res


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
def test_bash_timeout_kills_sigterm_ignoring_descendant(tmp_path):
    pid_file = tmp_path / "trap.pid"
    script = tmp_path / "trap.sh"
    script.write_text(
        "#!/bin/bash\n"
        f"(trap '' TERM; echo $BASHPID > {pid_file}; sleep 30) &\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    t0 = time.monotonic()
    res = bash_command.invoke({"command": f"bash {script}", "timeout_seconds": 1})
    elapsed = time.monotonic() - t0
    assert "timed out" in res
    assert elapsed < 3.0, f"timeout handling too slow: {elapsed:.2f}s"

    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    alive = True
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            alive = False
            break
    if alive:
        os.kill(child_pid, 9)
    assert not alive, "SIGTERM-ignoring descendant survived the timeout"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_atomic_write_preserves_permissions_and_symlinks(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("echo a\n", encoding="utf-8")
    os.chmod(script, 0o755)

    file_edit.invoke({"file_path": str(script), "target_content": "echo a", "replacement_content": "echo b"})
    assert (os.stat(script).st_mode & 0o777) == 0o755

    file_writer.invoke({"file_path": str(script), "content": "echo c\n"})
    assert (os.stat(script).st_mode & 0o777) == 0o755

    new_file = tmp_path / "new.txt"
    file_writer.invoke({"file_path": str(new_file), "content": "hello\n"})
    assert (os.stat(new_file).st_mode & 0o777) == (0o666 & ~common_tools._PROCESS_UMASK)

    real = tmp_path / "real.txt"
    real.write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    os.symlink(real, link)
    file_edit.invoke({"file_path": str(link), "target_content": "x = 1", "replacement_content": "x = 2"})
    assert link.is_symlink(), "symlink must be preserved"
    assert real.read_text(encoding="utf-8") == "x = 2\n"


def test_file_edit_mixed_line_endings_keep_untouched_lines(tmp_path):
    mixed = tmp_path / "mixed.txt"
    mixed.write_bytes(b"alpha\r\nbeta\ngamma\r\n")
    res = file_edit.invoke({"file_path": str(mixed), "target_content": "beta", "replacement_content": "BETA"})
    assert res.startswith("SUCCESS:")
    assert mixed.read_bytes() == b"alpha\r\nBETA\ngamma\r\n"


def _call(tool_obj, args, call_id="call_1"):
    return tool_obj.invoke({"type": "tool_call", "id": call_id, "name": tool_obj.name, "args": args})


def test_error_status_contract_for_agent_loops(tmp_path):
    from langchain_core.messages import ToolMessage

    ok_file = tmp_path / "ok.txt"
    ok_file.write_text("hello\n", encoding="utf-8")

    ok_msg = _call(file_read, {"file_path": str(ok_file)})
    assert isinstance(ok_msg, ToolMessage) and ok_msg.status == "success"

    err_msg = _call(file_writer, {"file_path": str(tmp_path), "content": "x"})
    assert err_msg.status == "error"
    assert err_msg.content.startswith("FileWrite Error:")

    fail_msg = _call(bash_command, {"command": "echo boom >&2; exit 3"})
    assert fail_msg.status == "error"
    assert "[Exit Code: 3" in fail_msg.content and "boom" in fail_msg.content

    grep_msg = _call(bash_command, {"command": f"grep NOPE_XYZ {str(ok_file)!r}"})
    assert grep_msg.status == "success", "grep exit 1 (no matches) is not a failure"

    timeout_msg = _call(bash_command, {"command": "sleep 5", "timeout_seconds": 1})
    assert timeout_msg.status == "error"

    invalid_msg = _call(file_read, {"offset": 3})
    assert invalid_msg.status == "error"
    assert "Tool Argument Error" in invalid_msg.content
    assert "file_path" in invalid_msg.content


def test_direct_invoke_returns_error_string_without_raising(tmp_path):
    res = file_read.invoke({"file_path": str(tmp_path / "missing.txt")})
    assert isinstance(res, str) and res.startswith("FileRead Error:")
    res2 = bash_command.invoke({"command": "exit 2"})
    assert isinstance(res2, str) and "[Exit Code: 2" in res2


def test_descriptions_have_no_docstring_indentation():
    for t in ALL_COMMON_TOOLS:
        assert "\n    " not in t.description, f"{t.name} description still contains docstring indentation"
        assert t.handle_tool_error is True


# =============================================================================
# 7. Regression Tests for Second-Pass Fixes (quotes/subshell, mixed EOL, 3xx, prune)
# =============================================================================


def test_bash_command_quoted_pipe_and_subshell_exit_semantics(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("12345\n", encoding="utf-8")

    # Quoted '|' inside regex must NOT split the command into `"888'"`.
    msg_or = _call(bash_command, {"command": f"grep -E '999|888' {str(target)!r}"})
    assert msg_or.status == "success", msg_or.content
    assert "No matches found" in msg_or.content

    # Subshell wrapping must still identify 'grep'.
    msg_sub = _call(bash_command, {"command": f"(grep NOPE_XYZ {str(target)!r})"})
    assert msg_sub.status == "success", msg_sub.content
    assert "No matches found" in msg_sub.content

    # Redirection with 2>&1 must not confuse '&' with a background separator.
    msg_redir = _call(bash_command, {"command": f"grep NOPE_XYZ {str(target)!r} 2>&1"})
    assert msg_redir.status == "success", msg_redir.content
    assert "No matches found" in msg_redir.content

    # Negated grep (! grep ...) or explicit non-zero exit must remain an error.
    msg_neg = _call(bash_command, {"command": f"! grep 12345 {str(target)!r}"})
    assert msg_neg.status == "error"

    msg_exit1 = _call(
        bash_command,
        {"command": f"{sys.executable} -c \"import sys; sys.exit(1)\""},
    )
    assert msg_exit1.status == "error"


def test_file_edit_mixed_line_endings_multiline_target(tmp_path):
    mixed = tmp_path / "mixed_multi.txt"
    mixed.write_bytes(b"line1\r\nline2\r\nline3\nline4\r\n")

    res = file_edit.invoke(
        {
            "file_path": str(mixed),
            "target_content": "line1\nline2",
            "replacement_content": "LINE1\nLINE2",
        }
    )
    assert res.startswith("SUCCESS:"), res
    assert mixed.read_bytes() == b"LINE1\r\nLINE2\r\nline3\nline4\r\n"


def test_web_fetch_distinguishes_redirect_loop_from_blocked_scheme():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LocalTestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        loop_err = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/loop"})
        assert "HTTP 302 redirect error" in loop_err
        assert "Only http/https redirects are followed" not in loop_err

        redir_err = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/redirect-file"})
        assert "WebFetch Error: Redirect blocked" in redir_err

        ftp_err = web_fetch.invoke({"url": f"http://127.0.0.1:{port}/redirect-ftp"})
        assert "WebFetch Error: Redirect blocked" in ftp_err
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_glob_search_prunes_excluded_dirs_without_walking_them(tmp_path, monkeypatch):
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "pack.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "env_langchain_1315" / "lib").mkdir(parents=True)
    (tmp_path / "env_langchain_1315" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (tmp_path / "env_langchain_1315" / "lib" / "mod.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "artifact.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "top_a.py").write_text("a=1\n", encoding="utf-8")
    (tmp_path / "top_b.py").write_text("b=1\n", encoding="utf-8")

    visited_dirs = []
    real_walk = os.walk

    def tracking_walk(top, *args, **kwargs):
        for root, dirs, files in real_walk(top, *args, **kwargs):
            visited_dirs.append(os.path.abspath(root))
            yield root, dirs, files

    monkeypatch.setattr(os, "walk", tracking_walk)

    rec = glob_search.invoke({"pattern": "**/*.py", "search_path": str(tmp_path)})
    assert "top_a.py" in rec and "artifact.py" in rec
    assert "pack.py" not in rec and "mod.py" not in rec
    assert not any(".git" in Path(d).parts or "env_langchain_1315" in Path(d).parts for d in visited_dirs)

    visited_dirs.clear()
    top_only = glob_search.invoke({"pattern": "*.py", "search_path": str(tmp_path)})
    assert "top_a.py" in top_only and "top_b.py" in top_only
    assert "artifact.py" not in top_only
    assert visited_dirs == [os.path.abspath(tmp_path)]
