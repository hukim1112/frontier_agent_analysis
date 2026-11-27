"""
===============================================================================
[AAWS Common Tools] Production-Grade 9 Essential ACI Tool Engine (common.py)
===============================================================================
Categories:
  1. File Operations: file_read, file_edit, file_writer, notebook_edit
  2. Shell: bash_command
  3. Search: grep_search, glob_search
  4. Web: web_fetch, web_search

Error Contract:
  - Failures raise `ToolException` with an actionable message.
  - Every tool sets `handle_tool_error=True`, so:
      * direct `.invoke({...})` calls return the error message as a plain string (no crash),
      * agent loops receive `ToolMessage(status="error")` so tracers/evaluators can count failures.
  - Invalid arguments (schema validation errors) are also returned as error observations.
===============================================================================
"""

import difflib
import glob
import inspect
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field


# =============================================================================
# Shared Internal Helpers (Path, Atomic I/O, Truncation, Binary & Process Tree)
# =============================================================================

# Directory names that are always skipped by grep_search / glob_search.
# Python virtual environments with other names are detected via `pyvenv.cfg`.
VCS_EXCLUDE_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".tox",
}

_BINARY_CONTENT_TYPE_PREFIXES = ("image/", "audio/", "video/")
_BINARY_CONTENT_TYPES = {
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/octet-stream",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_RAW_TEXT_CONTENT_TYPES = {
    "application/json",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/xml",
    "text/xml",
}


def _read_process_umask() -> int:
    """Reads the process umask once at import time (used for permissions of newly created files)."""
    try:
        mask = os.umask(0)
        os.umask(mask)
        return mask
    except Exception:
        return 0o022


_PROCESS_UMASK = _read_process_umask()


def _is_excluded_dir(dir_path: str, cache: Optional[Dict[str, bool]] = None) -> bool:
    """Returns True for VCS/cache directories or Python virtual environments (contains pyvenv.cfg)."""
    if cache is not None and dir_path in cache:
        return cache[dir_path]
    name = os.path.basename(os.path.normpath(dir_path))
    excluded = name in VCS_EXCLUDE_DIRS or os.path.isfile(os.path.join(dir_path, "pyvenv.cfg"))
    if cache is not None:
        cache[dir_path] = excluded
    return excluded


def _has_excluded_dir_part(
    target_path: str,
    base_root: Optional[str] = None,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Checks directory segments only (never file basenames or ancestors above base_root)."""
    target_p = Path(os.path.abspath(target_path))
    dir_p = target_p if target_p.is_dir() else target_p.parent

    if base_root:
        root_p = Path(os.path.abspath(base_root))
        try:
            rel_p = dir_p.relative_to(root_p)
        except ValueError:
            rel_p = None
        if rel_p is not None:
            current = root_p
            for part in rel_p.parts:
                current = current / part
                if _is_excluded_dir(str(current), cache):
                    return True
            return False

    return any(part in VCS_EXCLUDE_DIRS for part in dir_p.parts)


def _safe_relpath(path_str: str, start: Optional[str] = None) -> str:
    """Returns a cwd-relative path for files under cwd; absolute paths otherwise (no '../../..' chains)."""
    try:
        rel = os.path.relpath(path_str, start or os.getcwd())
    except ValueError:
        return os.path.abspath(path_str)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return os.path.abspath(path_str)
    return rel


def _is_binary_file(abs_path: str, sample_size: int = 8192) -> bool:
    """Detects binary files by inspecting the leading bytes for NUL bytes."""
    try:
        with open(abs_path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        return False


def _atomic_write_text(abs_path: str, content: str, encoding: str = "utf-8") -> None:
    """Atomically writes text via a unique temp file.

    - Follows symlinks so the link itself is preserved and its target is updated.
    - Preserves the existing file's permission bits (e.g. executable scripts stay 0755).
    - New files get the default permissions implied by the process umask (not tempfile's 0600).
    """
    target_path = os.path.realpath(abs_path)
    parent = os.path.dirname(target_path) or "."
    os.makedirs(parent, exist_ok=True)

    target_mode: Optional[int]
    if os.path.exists(target_path):
        try:
            target_mode = stat.S_IMODE(os.stat(target_path).st_mode)
        except OSError:
            target_mode = None
    else:
        target_mode = 0o666 & ~_PROCESS_UMASK

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=parent,
            prefix=".tmp_",
            suffix=".part",
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(content)
            tmp_file.flush()
            try:
                os.fsync(tmp_file.fileno())
            except OSError:
                pass
        if target_mode is not None:
            try:
                os.chmod(tmp_path, target_mode)
            except OSError:
                pass
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _truncate_head_tail(text: str, max_chars: int, label: str = "OUTPUT") -> str:
    """Preserves both head and tail context when truncating long output strings."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    head_part = text[:head]
    tail_part = text[-tail:] if tail > 0 else ""
    return (
        f"{head_part}\n\n"
        f"... [{label} TRUNCATED: {omitted} characters omitted] ...\n\n"
        f"{tail_part}"
    )


def _suggest_similar_files(abs_path: str, limit: int = 3) -> List[str]:
    """Returns close filename matches from the target's parent directory if available."""
    parent = os.path.dirname(abs_path)
    if not parent or not os.path.isdir(parent):
        return []
    try:
        entries = sorted(os.listdir(parent))
    except OSError:
        return []
    return difflib.get_close_matches(os.path.basename(abs_path), entries, n=limit, cutoff=0.5)


def _build_edit_mismatch_hint(
    file_path: str,
    norm_content: str,
    norm_target: str,
) -> str:
    """Builds diagnostic guidance when file_edit fails to find an exact target match."""
    hints: List[str] = []

    if any(re.match(r"^\s*\d+\s*\|\s?", ln) for ln in norm_target.splitlines()):
        hints.append(
            "Hint: 'target_content' appears to include file_read line-number prefixes (e.g. '     1 | '). "
            "Strip line numbers and '|' prefixes so only actual file text remains."
        )

    lines = norm_content.splitlines()
    probe = next((ln.strip() for ln in norm_target.splitlines() if ln.strip()), norm_target.strip())

    best_idx = 0
    best_ratio = 0.0
    if probe and lines:
        scan_limit = min(len(lines), 5000)
        for idx in range(scan_limit):
            cand = lines[idx].strip()
            if not cand:
                continue
            if probe in cand:
                best_idx = idx
                best_ratio = 1.0
                break
            ratio = difflib.SequenceMatcher(None, probe[:200], cand[:200]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx

    if lines:
        if best_ratio >= 0.45:
            start = max(0, best_idx - 2)
            end = min(len(lines), best_idx + 3)
            label = f"Closest region (around line {best_idx + 1}):"
        else:
            start = 0
            end = min(len(lines), 5)
            label = "File preview (first lines):"
        preview = "\n".join(f"{i + 1:6d} | {lines[i]}" for i in range(start, end))
    else:
        label = "File preview:"
        preview = "(empty file)"

    hint_block = ("\n" + "\n".join(hints)) if hints else ""
    return (
        f"FileEdit Error: Exact 'target_content' match not found in '{file_path}'.{hint_block}\n"
        f"Use file_read on '{file_path}' to verify exact whitespace, indentation, and content before retrying.\n"
        f"{label}\n{preview}"
    )


def _decode_stream_fragment(data: Any) -> str:
    """Safely decodes bytes/str fragments captured from subprocess streams or TimeoutExpired."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminates a subprocess and all its descendants across POSIX and Windows.

    POSIX: SIGTERM to the whole process group, short grace period, then ALWAYS SIGKILL
    the group so descendants that ignore SIGTERM (or outlive the shell) are reaped too.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            pass
        try:
            proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
    else:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _extract_last_base_command(command: str) -> str:
    """Heuristically extracts the final executable name in a chained/piped shell command."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        raw_segments = [seg.strip() for seg in re.split(r"&&|\|\||;|\|", command) if seg.strip()]
        tokens = raw_segments[-1].split() if raw_segments else []

    if not tokens:
        return ""

    segments: List[List[str]] = []
    current: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in {"(", ")", "{", "}"}:
            i += 1
            continue
        if tok.endswith((">", "<")) and nxt == "&":
            combined = tok + nxt
            i += 2
            if i < len(tokens) and tokens[i] not in {"|", "||", "&&", ";", "&", "(", ")"}:
                combined += tokens[i]
                i += 1
            current.append(combined)
            continue
        if tok == "&" and nxt.startswith(">"):
            current.append(tok + nxt)
            i += 2
            continue
        if tok in {"|", "||", "&&", ";", "&"}:
            if current:
                segments.append(current)
                current = []
            i += 1
            continue
        current.append(tok)
        i += 1
    if current:
        segments.append(current)

    if not segments:
        return ""

    last_tokens = segments[-1]
    skip_next = False
    for tok in last_tokens:
        if skip_next:
            skip_next = False
            continue
        if re.fullmatch(r"\d*(?:>>?|<<?|[<>]&?|&>>?)", tok):
            skip_next = True
            continue
        if re.match(r"^(?:\d*>>?|\d*<<?|&>>?|\d*>&\d+)", tok):
            continue
        if "=" in tok and not tok.startswith(("-", "=")):
            var_name = tok.split("=", 1)[0]
            if var_name.isidentifier():
                continue
        if tok in {"sudo", "env", "time", "command", "nohup"}:
            continue
        if tok == "!":
            return "!"
        return os.path.basename(tok)
    return ""


def _interpret_exit_code(command: str, returncode: int) -> Tuple[bool, Optional[str]]:
    """Returns (is_error, note) with semantic context for shell exit codes."""
    if returncode == 0:
        return False, None
    if returncode < 0:
        return True, f"Process was terminated by signal {-returncode}."
    if returncode == 127:
        return True, "Command not found in PATH or script path does not exist."
    if returncode == 126:
        return True, "Permission denied or target file is not executable."

    base_cmd = _extract_last_base_command(command)
    if base_cmd in {"grep", "rg", "egrep", "fgrep"} and returncode == 1:
        return False, "No matches found (exit code 1 is normal for grep/rg when 0 lines match)."
    if base_cmd == "diff" and returncode == 1:
        return False, "Files differ (exit code 1 is normal when differences exist)."
    if base_cmd in {"test", "["} and returncode == 1:
        return False, "Condition evaluated to false (exit code 1 is expected for false condition)."
    if base_cmd == "find" and returncode == 1:
        return False, "Partial traversal: some directories could not be accessed."
    return True, (
        f"Command failed with exit code {returncode}. "
        "Inspect STDERR/STDOUT to diagnose and fix the cause before retrying."
    )


def _format_validation_error(error: Exception) -> str:
    """Formats Pydantic argument validation errors as an actionable observation for the agent."""
    details: List[str] = []
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
        "Tool Argument Error: Invalid arguments for this tool.\n"
        + "\n".join(details)
        + "\nCheck the parameter names, required fields, and value types in the tool schema, then retry."
    )


# =============================================================================
# Category 1: File Operations (FileRead, FileEdit, FileWriter, NotebookEdit)
# =============================================================================


class FileReadInput(BaseModel):
    file_path: str = Field(description="Relative or absolute path of the file to read from disk.")
    offset: int = Field(
        default=1,
        description="1-indexed line number to start reading from (>= 1). Defaults to 1.",
    )
    limit: int = Field(
        default=250,
        description="Maximum number of lines to read in a single call (1 to 2000). Defaults to 250.",
    )
    show_line_numbers: bool = Field(
        default=True,
        description="Whether to prefix each line with line numbers (e.g. '     1 | ...'). Defaults to True.",
    )


@tool(args_schema=FileReadInput)
def file_read(file_path: str, offset: int = 1, limit: int = 250, show_line_numbers: bool = True) -> str:
    """Reads lines from a local text file with optional line numbers and pagination.

    Usage:
    - Always prefer file_read over bash_command (`cat`, `head`, `tail`) when inspecting files.
    - Use `offset` and `limit` to paginate large files instead of reading everything at once.
    - Rejects directories and binary files with diagnostic errors.
    """
    if not file_path or not file_path.strip():
        raise ToolException("FileRead Error: 'file_path' cannot be empty.")

    abs_path = os.path.abspath(os.path.expanduser(file_path.strip()))
    if not os.path.exists(abs_path):
        suggestions = _suggest_similar_files(abs_path)
        hint = (
            f" Did you mean one of: {', '.join(suggestions)}?"
            if suggestions
            else " Use glob_search to locate the file."
        )
        raise ToolException(
            f"FileRead Error: File '{file_path}' (resolved to '{abs_path}') does not exist.{hint}"
        )
    if os.path.isdir(abs_path):
        raise ToolException(
            f"FileRead Error: Path '{file_path}' is a directory, not a file. "
            "Use glob_search to list files in this directory."
        )
    if _is_binary_file(abs_path):
        raise ToolException(
            f"FileRead Error: File '{file_path}' appears to be a binary file and cannot be displayed as text."
        )

    safe_offset = max(1, int(offset))
    safe_limit = min(2000, max(1, int(limit)))

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if total_lines == 0:
            return f"[File: {abs_path} (Empty file: 0 lines)]"

        if safe_offset > total_lines:
            raise ToolException(
                f"FileRead Error: offset={safe_offset} exceeds total line count ({total_lines}) of '{file_path}'. "
                f"Specify an offset between 1 and {total_lines}."
            )

        start_idx = safe_offset - 1
        end_idx = min(total_lines, start_idx + safe_limit)
        sliced_lines = lines[start_idx:end_idx]

        max_line_chars = 2000
        max_total_chars = 50000
        output_lines: List[str] = []
        total_chars = 0
        actual_end_idx = start_idx
        truncated_by_size = False

        for idx, raw_line in enumerate(sliced_lines, start=start_idx + 1):
            has_newline = raw_line.endswith("\n")
            line_body = raw_line.rstrip("\r\n")
            if len(line_body) > max_line_chars:
                omitted = len(line_body) - max_line_chars
                line_body = line_body[:max_line_chars] + f"... [line truncated, {omitted} chars omitted]"
            formatted = f"{idx:6d} | {line_body}" if show_line_numbers else line_body
            if has_newline or idx < end_idx:
                formatted += "\n"

            if output_lines and total_chars + len(formatted) > max_total_chars:
                truncated_by_size = True
                break

            output_lines.append(formatted)
            total_chars += len(formatted)
            actual_end_idx = idx

        content = "".join(output_lines)
        header = f"[File: {abs_path} (Lines {start_idx + 1}-{actual_end_idx} of {total_lines})]\n"
        result = header + content

        if actual_end_idx < total_lines:
            size_note = " (output size limit reached)" if truncated_by_size else ""
            result = result.rstrip("\n") + (
                f"\n... [{total_lines - actual_end_idx} more lines remaining{size_note}. "
                f"Call file_read with offset={actual_end_idx + 1} to continue reading.]"
            )
        return result

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"FileRead Error: Failed to read file '{file_path}': {str(e)}") from e


class FileEditInput(BaseModel):
    file_path: str = Field(description="Relative or absolute path to the existing file to edit.")
    target_content: str = Field(
        description="Exact existing text block to replace (copied verbatim without file_read line-number prefixes)."
    )
    replacement_content: str = Field(description="New text block to substitute in place of target_content.")
    replace_all: bool = Field(
        default=False,
        description="If True, replaces all matching occurrences. If False, requires target_content to match uniquely. Defaults to False.",
    )


@tool(args_schema=FileEditInput)
def file_edit(
    file_path: str,
    target_content: str,
    replacement_content: str,
    replace_all: bool = False,
) -> str:
    """Replaces exact text blocks in an existing file while preserving original line endings.

    Usage:
    - Always call file_read first to inspect the exact indentation and content before editing.
    - Prefer file_edit over file_writer for modifying existing files; never include file_read line-number prefixes ('     1 | ') in target_content.
    - For Jupyter notebooks (.ipynb), use notebook_edit instead of file_edit.
    """
    if not file_path or not file_path.strip():
        raise ToolException("FileEdit Error: 'file_path' cannot be empty.")

    abs_path = os.path.abspath(os.path.expanduser(file_path.strip()))
    if not os.path.exists(abs_path):
        suggestions = _suggest_similar_files(abs_path)
        hint = f" Did you mean one of: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolException(
            f"FileEdit Error: Target file '{file_path}' does not exist.{hint} Use file_writer to create a new file."
        )
    if os.path.isdir(abs_path):
        raise ToolException(f"FileEdit Error: Path '{file_path}' is a directory, not a file.")
    if abs_path.lower().endswith(".ipynb"):
        raise ToolException(
            f"FileEdit Error: '{file_path}' is a Jupyter Notebook. "
            "Use notebook_edit to modify Jupyter Notebook (.ipynb) cells safely."
        )
    if _is_binary_file(abs_path):
        raise ToolException(f"FileEdit Error: '{file_path}' appears to be a binary file and cannot be text-edited.")
    if target_content == "":
        raise ToolException(
            "FileEdit Error: 'target_content' cannot be empty. "
            "Provide the exact existing text block to replace, or use file_writer to write a new file."
        )
    if target_content == replacement_content:
        raise ToolException(
            "FileEdit Error: No changes made because 'target_content' and 'replacement_content' are identical."
        )

    try:
        with open(abs_path, "rb") as f:
            raw_bytes = f.read()

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ToolException(f"FileEdit Error: File '{file_path}' is not valid UTF-8 ({e}).") from e

        lf_count = raw_bytes.count(b"\n")
        crlf_count = raw_bytes.count(b"\r\n")
        default_eol = "\r\n" if (crlf_count > 0 and crlf_count >= (lf_count / 2)) else "\n"

        norm_target = target_content.replace("\r\n", "\n")
        norm_replacement = replacement_content.replace("\r\n", "\n")

        direct_exact = False
        if norm_target == norm_replacement:
            if target_content != replacement_content and target_content in text:
                direct_exact = True
                matches = list(re.finditer(re.escape(target_content), text))
            else:
                raise ToolException(
                    "FileEdit Error: No changes made because 'target_content' and 'replacement_content' are identical."
                )
        else:
            pattern_str = r"\r?\n".join(re.escape(part) for part in norm_target.split("\n"))
            matches = list(re.finditer(pattern_str, text))

        if not matches:
            raise ToolException(_build_edit_mismatch_hint(file_path, text.replace("\r\n", "\n"), norm_target))

        match_count = len(matches)
        if match_count > 1 and not replace_all:
            match_lines = [str(text.count("\n", 0, m.start()) + 1) for m in matches[:10]]
            more_note = ", ..." if match_count > len(match_lines) else ""
            raise ToolException(
                f"FileEdit Error: 'target_content' matches {match_count} locations in '{file_path}' "
                f"(lines: {', '.join(match_lines)}{more_note}). "
                "Provide more surrounding lines to match uniquely, or set replace_all=True to replace all occurrences."
            )

        selected = matches if replace_all else [matches[0]]
        pieces: List[str] = []
        last_end = 0
        for m in selected:
            pieces.append(text[last_end:m.start()])
            if direct_exact:
                pieces.append(replacement_content)
            else:
                matched_str = m.group(0)
                if "\r\n" in matched_str:
                    eol = "\r\n"
                elif "\n" in matched_str:
                    eol = "\n"
                else:
                    nl_pos = text.find("\n", m.end())
                    if nl_pos > 0 and text[nl_pos - 1] == "\r":
                        eol = "\r\n"
                    elif nl_pos != -1:
                        eol = "\n"
                    else:
                        eol = default_eol
                pieces.append(norm_replacement.replace("\n", eol))
            last_end = m.end()
        pieces.append(text[last_end:])
        final_text = "".join(pieces)

        _atomic_write_text(abs_path, final_text)

        display_lines = final_text.replace("\r\n", "\n").splitlines()
        edit_start_line = text.count("\n", 0, matches[0].start()) + 1
        rep_line_count = max(1, len(norm_replacement.splitlines()))
        snip_start = max(1, edit_start_line - 3)
        snip_end = min(len(display_lines), edit_start_line + rep_line_count - 1 + 3)
        snippet = "\n".join(f"{ln:6d} | {display_lines[ln - 1]}" for ln in range(snip_start, snip_end + 1))

        return (
            f"SUCCESS: File '{abs_path}' edited successfully "
            f"({len(selected)} replacement(s); replaced {len(norm_target)} characters with {len(norm_replacement)} characters).\n"
            f"--- Updated Snippet ---\n{snippet}"
        )

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"FileEdit Error: Failed to edit '{file_path}': {str(e)}") from e


class FileWriterInput(BaseModel):
    file_path: str = Field(
        description="Relative or absolute destination file path. Parent directories are created automatically."
    )
    content: str = Field(description="Complete text content to write into the file.")
    overwrite: bool = Field(
        default=True,
        description="If True, overwrites an existing file. If False, fails when target file exists. Defaults to True.",
    )


@tool(args_schema=FileWriterInput)
def file_writer(file_path: str, content: str, overwrite: bool = True) -> str:
    """Creates a new file or atomically overwrites an existing file with text content.

    Usage:
    - Use file_writer for creating new files or complete rewrites; prefer file_edit for partial edits to existing files.
    - Prefer file_writer over bash_command shell redirection (`echo >`, `cat <<EOF`).
    """
    if not file_path or not file_path.strip():
        raise ToolException("FileWrite Error: 'file_path' cannot be empty.")

    abs_path = os.path.abspath(os.path.expanduser(file_path.strip()))
    if os.path.isdir(abs_path):
        raise ToolException(f"FileWrite Error: Path '{file_path}' is an existing directory, not a file.")
    if os.path.exists(abs_path) and not overwrite:
        raise ToolException(
            f"FileWrite Error: File '{file_path}' already exists and overwrite=False. "
            "Use file_edit for targeted changes to existing files, or call file_writer with overwrite=True to replace the entire file."
        )

    try:
        _atomic_write_text(abs_path, content)
        return f"SUCCESS: Written {len(content)} characters ({len(content.splitlines())} lines) to '{abs_path}'."
    except Exception as e:
        raise ToolException(f"FileWrite Error: Failed to write to '{file_path}': {str(e)}") from e


class NotebookEditInput(BaseModel):
    notebook_path: str = Field(description="Relative or absolute path to the .ipynb Jupyter Notebook file.")
    cell_index: int = Field(description="0-indexed integer specifying which cell in the notebook to modify.")
    new_code: str = Field(description="Complete new source code or markdown text for the target cell.")


@tool(args_schema=NotebookEditInput)
def notebook_edit(notebook_path: str, cell_index: int, new_code: str) -> str:
    """Replaces the source of a specific cell in a Jupyter Notebook (.ipynb) file.

    Usage:
    - Always use notebook_edit instead of file_edit when editing .ipynb files.
    - Read the notebook with file_read first to confirm valid 0-indexed cell positions.
    - Clears stale execution outputs and execution_count automatically when modifying code cells.
    """
    if not notebook_path or not notebook_path.strip():
        raise ToolException("NotebookEdit Error: 'notebook_path' cannot be empty.")

    abs_path = os.path.abspath(os.path.expanduser(notebook_path.strip()))
    if not os.path.exists(abs_path):
        suggestions = _suggest_similar_files(abs_path)
        hint = f" Did you mean one of: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolException(f"NotebookEdit Error: Notebook '{notebook_path}' does not exist.{hint}")
    if os.path.isdir(abs_path):
        raise ToolException(f"NotebookEdit Error: Path '{notebook_path}' is a directory, not a notebook file.")

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        if not isinstance(nb_data, dict) or not isinstance(nb_data.get("cells"), list):
            raise ToolException(
                f"NotebookEdit Error: File '{notebook_path}' is not a valid Jupyter Notebook JSON (missing 'cells' list)."
            )

        cells = nb_data["cells"]
        if len(cells) == 0:
            raise ToolException(f"NotebookEdit Error: Notebook '{notebook_path}' has 0 cells.")

        if cell_index < 0 or cell_index >= len(cells):
            raise ToolException(
                f"NotebookEdit Error: cell_index={cell_index} out of bounds. "
                f"Notebook has {len(cells)} cells (valid 0-indexed range: 0 to {len(cells) - 1}). "
                "Read the notebook first to verify cell indices."
            )

        target_cell = cells[cell_index]
        target_cell["source"] = new_code.splitlines(keepends=True)
        cell_type = target_cell.get("cell_type", "unknown")
        if cell_type == "code":
            target_cell["outputs"] = []
            target_cell["execution_count"] = None

        _atomic_write_text(abs_path, json.dumps(nb_data, indent=2, ensure_ascii=False) + "\n")
        return (
            f"SUCCESS: Cell #{cell_index} ({cell_type}) in notebook '{abs_path}' updated successfully "
            f"({len(new_code)} characters)."
        )

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"NotebookEdit Error: Failed to edit notebook '{notebook_path}': {str(e)}") from e


# =============================================================================
# Category 2: Shell (Bash)
# =============================================================================


class BashCommandInput(BaseModel):
    command: str = Field(description="Shell command string to execute in bash/sh.")
    timeout_seconds: int = Field(
        default=30,
        description="Maximum execution time in seconds (1 to 600) before terminating the process tree. Defaults to 30.",
    )
    max_stdout_length: int = Field(
        default=4000,
        description="Maximum character length of stdout to return (100 to 50000). Defaults to 4000.",
    )


@tool(args_schema=BashCommandInput)
def bash_command(command: str, timeout_seconds: int = 30, max_stdout_length: int = 4000) -> str:
    """Executes a shell command on the local system and returns its exit code, duration, stdout, and stderr.

    Usage:
    - Prefer dedicated tools over shell equivalents: use file_read (not cat/head/tail), file_edit (not sed/awk), file_writer (not echo >/cat <<EOF), grep_search (not grep/rg), and glob_search for finding files by wildcard pattern.
    - Use bash_command when you need hierarchical directory trees or detailed listings (`find . -maxdepth 3 -not -path '*/.git*'`, `tree`, `ls -la`) — always bound depth and exclude noise directories (`.git`, `node_modules`, `__pycache__`, `.venv`) to protect context.
    - Each invocation runs in an isolated subshell: working directory changes (`cd`) and environment exports (`export`) do NOT persist across calls; chain dependent commands with `&&`.
    - NEVER run interactive commands waiting for stdin or long-lived foreground servers.
    - Increase `timeout_seconds` (up to 600) for legitimate heavy builds or test suites.
    - A non-zero exit code is reported as a failed tool call (except benign codes such as grep exit 1 = no matches).
    """
    if not command or not command.strip():
        raise ToolException("Bash Error: 'command' cannot be empty.")

    safe_timeout = min(600, max(1, int(timeout_seconds)))
    stdout_limit = min(50000, max(100, int(max_stdout_length)))
    stderr_limit = 4000

    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "cwd": os.getcwd(),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc: Optional[subprocess.Popen] = None
    start_time = time.time()

    try:
        proc = subprocess.Popen(command, **popen_kwargs)
        stdout_raw, stderr_raw = proc.communicate(timeout=safe_timeout)
        duration = round(time.time() - start_time, 2)

        stdout_clean = (stdout_raw or "").strip()
        stderr_clean = (stderr_raw or "").strip()
        is_error, note = _interpret_exit_code(command, proc.returncode)

        result_parts = [
            f"[Command: '{command}']",
            f"[Exit Code: {proc.returncode} | Execution Time: {duration}s]",
        ]
        if note:
            result_parts.append(f"[Note: {note}]")
        if stdout_clean:
            result_parts.append("--- STDOUT ---\n" + _truncate_head_tail(stdout_clean, stdout_limit, "STDOUT"))
        if stderr_clean:
            result_parts.append("--- STDERR ---\n" + _truncate_head_tail(stderr_clean, stderr_limit, "STDERR"))
        if not stdout_clean and not stderr_clean:
            result_parts.append("(Command executed silently with no output)")

        output = "\n".join(result_parts)
        if is_error:
            raise ToolException(output)
        return output

    except subprocess.TimeoutExpired as exc:
        duration = round(time.time() - start_time, 2)
        partial_out = _decode_stream_fragment(exc.stdout)
        partial_err = _decode_stream_fragment(exc.stderr)

        pid = proc.pid if proc is not None else "unknown"
        if proc is not None:
            _terminate_process_tree(proc)
            try:
                # Retrying communicate() returns ALL output collected so far (including what
                # exc.stdout already holds), so prefer it instead of concatenating (no duplication).
                full_out, full_err = proc.communicate(timeout=2)
                full_out_s = _decode_stream_fragment(full_out)
                full_err_s = _decode_stream_fragment(full_err)
                if len(full_out_s) >= len(partial_out):
                    partial_out = full_out_s
                if len(full_err_s) >= len(partial_err):
                    partial_err = full_err_s
            except subprocess.TimeoutExpired:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            except (OSError, ValueError):
                pass

        partial_limit = min(stdout_limit, 2000)
        partial_out_clean = partial_out.strip()
        partial_err_clean = partial_err.strip()

        parts = [
            f"Bash Error: Command '{command}' timed out after {safe_timeout} seconds (elapsed {duration}s) "
            f"and its process tree (PID {pid}) was terminated.",
            "Recovery Hints: (1) If the task legitimately needs more time, retry with a higher timeout_seconds (up to 600); "
            "(2) Check partial output below for hangs, infinite loops, or prompts waiting for input (use non-interactive flags like -y); "
            "(3) Never launch persistent servers or watchers in the foreground.",
        ]
        if partial_out_clean:
            parts.append(
                "--- PARTIAL STDOUT (before timeout) ---\n"
                + _truncate_head_tail(partial_out_clean, partial_limit, "PARTIAL STDOUT")
            )
        if partial_err_clean:
            parts.append(
                "--- PARTIAL STDERR (before timeout) ---\n"
                + _truncate_head_tail(partial_err_clean, min(stderr_limit, 2000), "PARTIAL STDERR")
            )
        if not partial_out_clean and not partial_err_clean:
            parts.append("(No stdout/stderr output captured before timeout)")
        raise ToolException("\n".join(parts)) from None

    except ToolException:
        raise

    except Exception as e:
        if proc is not None and proc.poll() is None:
            _terminate_process_tree(proc)
        raise ToolException(f"Bash Error: Failed to execute command '{command}': {str(e)}") from e


# =============================================================================
# Category 3: Search (Grep, Glob)
# =============================================================================


class GrepSearchInput(BaseModel):
    pattern: str = Field(description="Regular expression pattern string to match within file contents.")
    search_path: str = Field(
        default=".",
        description="Directory or file path to search within. Defaults to current directory ('.').",
    )
    output_mode: Literal["files_with_matches", "content", "count"] = Field(
        default="files_with_matches",
        description="Output format: 'files_with_matches' (paths only), 'content' (matching lines with line numbers), or 'count' (match summary). Defaults to 'files_with_matches'.",
    )
    head_limit: int = Field(
        default=50,
        description="Maximum number of matching files or lines to return (1 to 500). Defaults to 50.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="If True, performs case-sensitive regex matching. Defaults to False (case-insensitive).",
    )


@tool(args_schema=GrepSearchInput)
def grep_search(
    pattern: str,
    search_path: str = ".",
    output_mode: str = "files_with_matches",
    head_limit: int = 50,
    case_sensitive: bool = False,
) -> str:
    """Searches file contents using regular expressions while excluding VCS, cache, and virtual environment directories.

    Usage:
    - Always use grep_search instead of invoking `grep` or `rg` via bash_command.
    - Skips binary files and files larger than 2 MB automatically.
    - Use `output_mode='files_with_matches'` first to locate relevant files, then `'content'` to inspect matching lines.
    """
    if not pattern:
        raise ToolException("Grep Error: 'pattern' cannot be empty.")
    if output_mode not in ("files_with_matches", "content", "count"):
        raise ToolException(
            f"Grep Error: Invalid output_mode '{output_mode}'. "
            "Choose one of: 'files_with_matches', 'content', 'count'."
        )

    abs_root = os.path.abspath(os.path.expanduser((search_path or ".").strip()))
    if not os.path.exists(abs_root):
        raise ToolException(f"Grep Error: Search path '{search_path}' does not exist.")

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as err:
        raise ToolException(f"Grep Error: Invalid regex pattern '{pattern}': {err}") from err

    safe_limit = min(500, max(1, int(head_limit)))
    max_grep_file_bytes = 2 * 1024 * 1024
    max_match_line_chars = 300

    matching_files: List[str] = []
    content_matches: List[str] = []
    total_match_count = 0
    truncated = False

    if os.path.isfile(abs_root):
        if _is_binary_file(abs_root):
            raise ToolException(f"Grep Error: File '{search_path}' appears to be a binary file.")
        target_files = [abs_root]
    else:
        target_files = []
        for root, dirs, files in os.walk(abs_root):
            dirs[:] = sorted(d for d in dirs if not _is_excluded_dir(os.path.join(root, d)))
            for fname in sorted(files):
                target_files.append(os.path.join(root, fname))

    stop_scanning = False
    for fpath in target_files:
        if stop_scanning:
            break
        try:
            if os.path.getsize(fpath) > max_grep_file_bytes:
                continue
            if _is_binary_file(fpath):
                continue
        except OSError:
            continue

        try:
            rel_path = _safe_relpath(fpath)
            file_had_match = False
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    if regex.search(line):
                        if not file_had_match:
                            file_had_match = True
                            if output_mode == "files_with_matches":
                                if len(matching_files) >= safe_limit:
                                    truncated = True
                                    stop_scanning = True
                                    break
                                matching_files.append(rel_path)
                                break
                            matching_files.append(rel_path)

                        total_match_count += 1
                        if output_mode == "content":
                            if len(content_matches) >= safe_limit:
                                truncated = True
                                stop_scanning = True
                                break
                            line_clean = line.rstrip("\r\n").strip()
                            if len(line_clean) > max_match_line_chars:
                                line_clean = line_clean[:max_match_line_chars] + "..."
                            content_matches.append(f"{rel_path}:{line_no}:{line_clean}")
        except OSError:
            continue

    if output_mode == "content":
        if not content_matches:
            return f"No matches found for pattern '{pattern}' in '{search_path}'."
        if truncated:
            header = (
                f"Showing first {safe_limit} matching lines for '{pattern}' "
                "(more matches exist; narrow 'search_path' or increase 'head_limit'):\n"
            )
        else:
            header = f"Found {len(content_matches)} matches for '{pattern}':\n"
        return header + "\n".join(content_matches)

    if output_mode == "count":
        return f"Found {total_match_count} total occurrences across {len(matching_files)} files for '{pattern}'."

    if not matching_files:
        return f"No files matching pattern '{pattern}' in '{search_path}'."
    if truncated:
        header = (
            f"Showing first {safe_limit} matching files for '{pattern}' "
            "(more files match; narrow 'search_path' or increase 'head_limit'):\n"
        )
    else:
        header = f"Found {len(matching_files)} matching files for '{pattern}':\n"
    return header + "\n".join(matching_files)


def _glob_segment_to_regex(seg: str) -> str:
    """Converts a single glob path segment into a non-slash regex fragment."""
    out: List[str] = []
    i = 0
    n = len(seg)
    while i < n:
        ch = seg[i]
        if ch == "*":
            while i < n and seg[i] == "*":
                i += 1
            out.append("[^/]*")
            continue
        if ch == "?":
            out.append("[^/]")
            i += 1
            continue
        if ch == "[":
            j = i + 1
            if j < n and seg[j] == "!":
                j += 1
            if j < n and seg[j] == "]":
                j += 1
            while j < n and seg[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape("["))
                i += 1
                continue
            inner = seg[i + 1 : j]
            if inner.startswith("!"):
                inner = "^" + inner[1:]
            elif inner.startswith("^"):
                inner = "\\" + inner
            out.append(f"[{inner}]")
            i = j + 1
            continue
        out.append(re.escape(ch))
        i += 1
    return "".join(out)


def _glob_walk_pruned(
    abs_root: str,
    clean_pattern: str,
    exclusion_cache: Dict[str, bool],
) -> Optional[List[str]]:
    """Matches relative glob patterns via directory-pruned os.walk; returns None for fallback."""
    norm_pat = clean_pattern.replace("\\", "/")
    while norm_pat.startswith("./"):
        norm_pat = norm_pat[2:]
    include_dirs = clean_pattern.endswith(("/", "\\"))
    norm_pat = norm_pat.rstrip("/")

    segments = [s for s in norm_pat.split("/") if s and s != "."]
    if os.path.isabs(clean_pattern) or ".." in segments:
        return None
    if not segments:
        return []

    prefix_parts: List[str] = []
    for seg in segments[:-1]:
        if any(ch in seg for ch in "*?["):
            break
        prefix_parts.append(seg)

    walk_root = abs_root
    for part in prefix_parts:
        walk_root = os.path.join(walk_root, part)
        if not os.path.isdir(walk_root) or _is_excluded_dir(walk_root, exclusion_cache):
            return []

    rem_segments = segments[len(prefix_parts) :]
    collapsed: List[str] = []
    for seg in rem_segments:
        if seg == "**" and collapsed and collapsed[-1] == "**":
            continue
        collapsed.append(seg)

    max_depth: Optional[int] = None if "**" in collapsed else len(collapsed)

    regex_parts: List[str] = ["\\A"]
    for idx, seg in enumerate(collapsed):
        if seg == "**":
            if len(collapsed) == 1:
                regex_parts.append(".+")
            elif idx == 0:
                regex_parts.append("(?:.+/)?")
            elif idx == len(collapsed) - 1:
                regex_parts.append("/.+")
            else:
                regex_parts.append("/(?:.+/)?")
        else:
            if idx > 0 and collapsed[idx - 1] != "**":
                regex_parts.append("/")
            regex_parts.append(_glob_segment_to_regex(seg))
    regex_parts.append("\\Z")
    compiled = re.compile("".join(regex_parts))

    matched: List[str] = []
    for root, dirs, files in os.walk(walk_root):
        rel_root = os.path.relpath(root, walk_root)
        if rel_root == ".":
            rel_prefix = ""
            depth = 0
        else:
            rel_prefix = Path(rel_root).as_posix() + "/"
            depth = len(Path(rel_root).parts)

        cand_dirs = [
            d for d in sorted(dirs) if not _is_excluded_dir(os.path.join(root, d), exclusion_cache)
        ]

        if include_dirs:
            for d in cand_dirs:
                rel_d = rel_prefix + d
                if compiled.match(rel_d):
                    matched.append(os.path.join(root, d))
        else:
            for fname in sorted(files):
                rel_f = rel_prefix + fname
                if compiled.match(rel_f):
                    matched.append(os.path.join(root, fname))

        if max_depth is not None and depth + 1 >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = cand_dirs

    return matched


class GlobSearchInput(BaseModel):
    pattern: str = Field(
        description="Glob wildcard pattern string to match file names and paths (e.g. '**/*.py' or 'src/**/*.ts')."
    )
    search_path: str = Field(
        default=".",
        description="Base directory path to execute the search from. Defaults to current directory ('.').",
    )
    head_limit: int = Field(
        default=100,
        description="Maximum number of matching file paths to return (1 to 500). Defaults to 100.",
    )


@tool(args_schema=GlobSearchInput)
def glob_search(pattern: str, search_path: str = ".", head_limit: int = 100) -> str:
    """Finds file paths matching a glob wildcard pattern sorted by most recently modified.

    Usage:
    - Prefer glob_search over shell `find` when locating files by name or wildcard pattern (e.g. `**/*.py`, `app/**/*.md`).
    - Supports recursive patterns (`**/*.py`), dotfiles (`.gitignore`, `.github/...`), and trailing-slash directory listing (`*/`, `*/*/` — useful in read-only planning mode), while automatically excluding VCS, cache, and virtual environment directories.
    - For full hierarchical directory trees or detailed metadata (`ls -la`, `tree`, `find -maxdepth`), use bash_command instead.
    """
    if not pattern or not pattern.strip():
        raise ToolException("Glob Error: 'pattern' cannot be empty.")

    clean_pattern = pattern.strip()
    abs_root = os.path.abspath(os.path.expanduser((search_path or ".").strip()))
    if not os.path.exists(abs_root):
        raise ToolException(f"Glob Error: Search path '{search_path}' does not exist.")
    if os.path.isfile(abs_root):
        raise ToolException(f"Glob Error: Search path '{search_path}' is a file, not a directory.")

    safe_limit = min(500, max(1, int(head_limit)))

    try:
        exclusion_cache: Dict[str, bool] = {}
        pruned_matches = _glob_walk_pruned(abs_root, clean_pattern, exclusion_cache)
        if pruned_matches is not None:
            filtered_abs = list(dict.fromkeys(os.path.abspath(m) for m in pruned_matches))
        else:
            full_pattern = clean_pattern if os.path.isabs(clean_pattern) else os.path.join(abs_root, clean_pattern)
            matches = glob.glob(full_pattern, recursive=True, include_hidden=True)
            include_dirs = clean_pattern.endswith(("/", "\\"))

            filtered_abs = []
            seen: set = set()
            for m in matches:
                abs_m = os.path.abspath(m)
                if abs_m in seen:
                    continue
                if not include_dirs and not os.path.isfile(abs_m):
                    continue
                if _has_excluded_dir_part(abs_m, base_root=abs_root, cache=exclusion_cache):
                    continue
                seen.add(abs_m)
                filtered_abs.append(abs_m)

        if not filtered_abs:
            return f"No files matched glob pattern '{pattern}' in '{search_path}'."

        def _mtime_key(p: str) -> Tuple[float, str]:
            try:
                return (-os.path.getmtime(p), p)
            except OSError:
                return (0.0, p)

        filtered_abs.sort(key=_mtime_key)
        total_count = len(filtered_abs)
        displayed = [_safe_relpath(m) for m in filtered_abs[:safe_limit]]

        if total_count > safe_limit:
            return (
                f"Found {total_count} files matching '{pattern}' (showing {safe_limit} most recently modified):\n"
                + "\n".join(displayed)
                + f"\n... [{total_count - safe_limit} more files omitted. Narrow your glob pattern or increase head_limit.]"
            )
        return (
            f"Found {total_count} files matching '{pattern}' (sorted by most recently modified):\n"
            + "\n".join(displayed)
        )

    except ToolException:
        raise
    except Exception as e:
        raise ToolException(
            f"Glob Error: Failed to run glob search '{pattern}' in '{search_path}': {str(e)}"
        ) from e


# =============================================================================
# Category 4: Web (WebFetch, WebSearch)
# =============================================================================


def _validate_http_url(raw_url: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned = (raw_url or "").strip()
    if not cleaned:
        return None, "WebFetch Error: 'url' cannot be empty."
    parsed = urllib.parse.urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None, (
            f"WebFetch Error: Unsupported URL scheme '{scheme or 'none'}' in '{raw_url}'. "
            "Only 'http://' and 'https://' URLs are allowed."
        )
    if not parsed.netloc:
        return None, f"WebFetch Error: Invalid URL '{raw_url}' (missing host)."
    return cleaned, None


class _BlockedRedirectError(urllib.error.URLError):
    """Raised when a redirect points to a non-HTTP(S) URL."""


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Ensures HTTP redirects cannot switch to non-HTTP/HTTPS schemes (e.g., ftp://)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        scheme = (urllib.parse.urlparse(resolved_url).scheme or "").lower()
        if scheme not in ("http", "https"):
            raise _BlockedRedirectError(
                f"'{req.full_url}' redirected to '{resolved_url}' (scheme '{scheme or 'none'}'). "
                "Only http/https redirects are followed."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTMLTextExtractor(HTMLParser):
    """Extracts readable text from HTML while ignoring script/style/svg subtrees."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "aside",
        "nav",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "hr",
        "pre",
        "blockquote",
    }

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_l = tag.lower()
        if tag_l in self.SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag_l in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
        elif self._skip_depth == 0 and tag_l in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if self._skip_depth == 0 and tag.lower() in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", data)
        if cleaned.strip():
            self._chunks.append(cleaned)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = [re.sub(r" +", " ", ln).strip() for ln in raw.splitlines()]
        non_empty = [ln for ln in lines if ln]
        return "\n".join(non_empty)


def _get_header_value(headers: Any, name: str) -> str:
    if headers is None:
        return ""
    if hasattr(headers, "get"):
        return str(headers.get(name, "") or "")
    return ""


def _parse_content_type_and_charset(headers: Any) -> Tuple[str, str]:
    raw_ct = _get_header_value(headers, "Content-Type")
    content_type = raw_ct.split(";", 1)[0].strip().lower() if raw_ct else ""
    charset = "utf-8"
    if headers is not None and hasattr(headers, "get_content_charset"):
        detected = headers.get_content_charset()
        if detected:
            charset = detected
    elif raw_ct:
        m = re.search(r"charset=([^\s;\"']+)", raw_ct, flags=re.IGNORECASE)
        if m:
            charset = m.group(1).strip()
    return content_type, charset


class WebFetchInput(BaseModel):
    url: str = Field(description="Full HTTP or HTTPS web URL string to fetch.")
    timeout_seconds: int = Field(
        default=10,
        description="HTTP request timeout in seconds (1 to 60). Defaults to 10.",
    )
    max_length: int = Field(
        default=4000,
        description="Maximum character length of extracted text to return (200 to 50000). Defaults to 4000.",
    )


@tool(args_schema=WebFetchInput)
def web_fetch(url: str, timeout_seconds: int = 10, max_length: int = 4000) -> str:
    """Fetches an HTTP/HTTPS URL and extracts readable plain text with script/style noise removed.

    Usage:
    - Use web_search first to discover URLs, then use web_fetch to read a specific target page.
    - Authenticated, login-gated, or JavaScript-only SPAs may fail or return incomplete text; fall back to web_search snippets or dedicated browser tools if blocked.
    - Rejects non-HTTP(S) URLs and binary responses (images, PDFs, archives).
    """
    clean_url, url_err = _validate_http_url(url)
    if url_err or not clean_url:
        raise ToolException(url_err or "WebFetch Error: Invalid URL.")

    safe_timeout = min(60, max(1, int(timeout_seconds)))
    safe_max_len = min(50000, max(200, int(max_length)))
    max_download_bytes = 2 * 1024 * 1024

    try:
        req = urllib.request.Request(
            clean_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        opener = urllib.request.build_opener(_SafeRedirectHandler())
        with opener.open(req, timeout=safe_timeout) as response:
            headers = getattr(response, "headers", None)
            content_type, charset = _parse_content_type_and_charset(headers)

            if content_type.startswith(_BINARY_CONTENT_TYPE_PREFIXES) or content_type in _BINARY_CONTENT_TYPES:
                raise ToolException(
                    f"WebFetch Error: URL '{clean_url}' returned binary content ({content_type}), "
                    "which cannot be extracted as plain text."
                )

            raw_bytes = response.read(max_download_bytes + 1)
            download_truncated = len(raw_bytes) > max_download_bytes
            raw_bytes = raw_bytes[:max_download_bytes]
            try:
                decoded_text = raw_bytes.decode(charset, errors="replace")
            except LookupError:
                decoded_text = raw_bytes.decode("utf-8", errors="replace")

        if content_type in _RAW_TEXT_CONTENT_TYPES:
            extracted = decoded_text.strip()
        else:
            parser = HTMLTextExtractor()
            parser.feed(decoded_text)
            parser.close()
            extracted = parser.get_text()

        header = f"[Fetched URL: {clean_url}]\n"
        if download_truncated:
            header += (
                "[Note: Response exceeded 2 MB; only the first 2 MB were parsed, "
                "so the end of the shown text is not the end of the document.]\n"
            )

        if not extracted:
            return header + "(Page returned no readable text; it may require JavaScript rendering or authentication.)"

        return header + _truncate_head_tail(extracted, safe_max_len, "WEBFETCH")

    except ToolException:
        raise
    except _BlockedRedirectError as e:
        raise ToolException(f"WebFetch Error: Redirect blocked: {e.reason}") from e
    except urllib.error.HTTPError as e:
        reason_str = str(e.reason)
        if 300 <= e.code < 400:
            if "not allowed" in reason_str.lower():
                raise ToolException(
                    f"WebFetch Error: Redirect blocked while fetching '{clean_url}' ({reason_str}). "
                    "Only http/https redirects are followed."
                ) from e
            raise ToolException(
                f"WebFetch Error: HTTP {e.code} redirect error while fetching '{clean_url}' ({reason_str}). "
                "The URL may have a redirect loop or invalid Location header; try web_search instead."
            ) from e
        if e.code in (401, 403):
            advice = "Access denied or authentication/anti-bot check required. Try web_search snippets or an authorized source."
        elif e.code == 404:
            advice = "Page not found. Verify the URL or discover the updated URL via web_search."
        elif e.code == 429:
            advice = "Rate limited by remote server. Wait or use web_search instead."
        elif 500 <= e.code < 600:
            advice = "Remote server error; retry later or check another source."
        else:
            advice = "Verify the URL and retry if needed."
        raise ToolException(
            f"WebFetch Error: HTTP {e.code} ({e.reason}) while fetching '{clean_url}'. {advice}"
        ) from e
    except (TimeoutError, socket.timeout) as e:
        raise ToolException(
            f"WebFetch Error: Request to '{clean_url}' timed out after {safe_timeout} seconds. "
            "Retry with a higher timeout_seconds (up to 60) or use web_search."
        ) from e
    except urllib.error.URLError as e:
        raise ToolException(f"WebFetch Error: Failed to reach '{clean_url}': {e.reason}") from e
    except Exception as e:
        raise ToolException(f"WebFetch Error: Failed to fetch '{clean_url}': {str(e)}") from e


def _format_search_results(items: Any, source_label: str, query: str) -> Optional[str]:
    """Formats Tavily or DuckDuckGo search result payloads into numbered markdown lines."""
    if isinstance(items, dict) and "results" in items:
        items = items["results"]
    if not isinstance(items, list) or not items:
        return None

    formatted: List[str] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            title = str(item.get("title", "Result")).strip() or "Result"
            url = str(item.get("url") or item.get("href") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or item.get("body") or "").strip()
            if url:
                formatted.append(f"{idx}. [{title}]({url}): {snippet}")
            else:
                formatted.append(f"{idx}. [{title}]: {snippet}")
        else:
            formatted.append(f"{idx}. {str(item)}")

    if not formatted:
        return None
    return f"[{source_label} for '{query}']\n" + "\n".join(formatted)


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string to look up on the web.")
    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return (1 to 10). Defaults to 5.",
    )


@tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Searches the web using Tavily Search API with automatic fallback to DuckDuckGo.

    Usage:
    - Use web_search to locate current facts, documentation, or candidate URLs before calling web_fetch.
    - Include explicit keywords and current year/version terms when searching for recent topics.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        raise ToolException("WebSearch Error: 'query' cannot be empty.")

    safe_max = min(10, max(1, int(max_results)))
    tavily_error: Optional[str] = None

    # 1. Attempt Tavily Search first when API key is configured
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            tavily = TavilySearch(max_results=safe_max)
            raw_res = tavily.invoke(clean_query)
            formatted = _format_search_results(raw_res, "Tavily Search Results", clean_query)
            if formatted:
                return formatted
            tavily_error = "Tavily returned no results"
        except Exception as e:
            tavily_error = f"{type(e).__name__}: {e}"

    # 2. Fallback to DuckDuckGo Search (DDGS) with localized warning suppression
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                search_results = list(ddgs.text(clean_query, max_results=safe_max))
    except Exception as e:
        ddg_error = f"{type(e).__name__}: {e}"
        if tavily_error:
            raise ToolException(
                f"WebSearch Error: Both Tavily ({tavily_error}) and DuckDuckGo ({ddg_error}) "
                f"searches failed for query '{clean_query}'."
            ) from e
        raise ToolException(
            f"WebSearch Error: DuckDuckGo search failed for query '{clean_query}': {ddg_error}"
        ) from e

    formatted = _format_search_results(
        search_results,
        "DuckDuckGo Search Results (Fallback)",
        clean_query,
    )
    if formatted:
        return formatted

    if tavily_error:
        return (
            f"WebSearch Result for '{clean_query}': No snippets extracted "
            f"(Tavily error: {tavily_error})."
        )
    return f"WebSearch Result for '{clean_query}': No snippets extracted."


# =============================================================================
# Tool Finalization (Error Status Contract & Clean Descriptions)
# =============================================================================


def _finalize_tool(t):
    """Applies the shared error contract and strips docstring indentation from descriptions."""
    t.description = inspect.cleandoc(t.description or "")
    t.handle_tool_error = True
    t.handle_validation_error = _format_validation_error
    return t


for _common_tool in (
    file_read,
    file_edit,
    file_writer,
    notebook_edit,
    bash_command,
    grep_search,
    glob_search,
    web_fetch,
    web_search,
):
    _finalize_tool(_common_tool)
