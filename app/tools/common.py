"""
===============================================================================
[Harness Module 03-3] Claude Code Production-Grade 8 Essential Tool Category Engine
===============================================================================
"""

import os
import sys
import re
import glob
import json
import time
import subprocess
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# =============================================================================
# Category 1: File Operations (FileRead, FileEdit, FileWriter, NotebookEdit)
# =============================================================================

class FileReadInput(BaseModel):
    file_path: str = Field(description="Relative or absolute path of the file to read from disk.")
    offset: int = Field(default=1, description="Line number to start reading from (1-indexed). Defaults to 1.")
    limit: int = Field(default=250, description="Maximum number of lines to read in a single call. Defaults to 250.")
    show_line_numbers: bool = Field(default=True, description="Whether to prefix each line with line numbers (e.g. '     1 | ...'). Defaults to True.")

@tool(args_schema=FileReadInput)
def file_read(file_path: str, offset: int = 1, limit: int = 250, show_line_numbers: bool = True) -> str:
    """Reads lines from a file on the local filesystem with optional line numbers and pagination.

    Args:
        file_path: Relative or absolute path of the file to read from disk.
        offset: Line number to start reading from (1-indexed). Defaults to 1.
        limit: Maximum number of lines to read in a single call. Defaults to 250.
        show_line_numbers: Whether to prefix each output line with line numbers. Defaults to True.

    Returns:
        Formatted string containing header with file metadata and line range, followed by file content lines.
    """
    abs_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(abs_path):
        return f"FileRead Error: File '{file_path}' (resolved to '{abs_path}') does not exist."
    if os.path.isdir(abs_path):
        return f"FileRead Error: Path '{file_path}' is a directory, not a file."

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        start_idx = max(0, offset - 1)
        end_idx = min(total_lines, start_idx + limit)
        
        sliced_lines = lines[start_idx:end_idx]
        output_lines = []
        
        for idx, line in enumerate(sliced_lines, start=start_idx + 1):
            if show_line_numbers:
                output_lines.append(f"{idx:6d} | {line}")
            else:
                output_lines.append(line)
                
        content = "".join(output_lines)
        header = f"[File: {abs_path} (Lines {start_idx+1}-{end_idx} of {total_lines})]\n"
        return header + content
        
    except Exception as e:
        return f"FileRead Error: Failed to read file '{file_path}': {str(e)}"


class FileEditInput(BaseModel):
    file_path: str = Field(description="Relative or absolute path to the target file to edit.")
    target_content: str = Field(description="The exact multi-line string block to find and replace. Must match uniquely in the file.")
    replacement_content: str = Field(description="The exact string block to substitute in place of target_content.")

@tool(args_schema=FileEditInput)
def file_edit(file_path: str, target_content: str, replacement_content: str) -> str:
    """Replaces an exact matching block of text in a file with replacement_content.

    Args:
        file_path: Relative or absolute path to the target file to edit.
        target_content: The exact multi-line string block to find and replace. Must match uniquely in the file.
        replacement_content: The exact string block to substitute in place of target_content.

    Returns:
        Success message with replacement byte sizes, or error message if file/match fails.
    """
    abs_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(abs_path):
        return f"FileEdit Error: Target file '{file_path}' does not exist."

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        if target_content not in content:
            preview = content[:200] + "..." if len(content) > 200 else content
            return (
                f"FileEdit Error: Exact target_content match not found in '{file_path}'.\n"
                f"File preview:\n{preview}"
            )

        match_count = content.count(target_content)
        if match_count > 1:
            return f"FileEdit Error: target_content matches {match_count} locations in '{file_path}'. Provide a unique block."

        new_content = content.replace(target_content, replacement_content, 1)

        # Atomic Write
        temp_path = abs_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(temp_path, abs_path)

        return f"SUCCESS: File '{abs_path}' edited successfully (replaced {len(target_content)} bytes with {len(replacement_content)} bytes)."

    except Exception as e:
        return f"FileEdit Error: Failed to edit '{file_path}': {str(e)}"


class FileWriterInput(BaseModel):
    file_path: str = Field(description="Relative or absolute destination file path. Parent directories created automatically.")
    content: str = Field(description="Complete text string content to write into the file.")
    overwrite: bool = Field(default=True, description="If True, overwrites existing file. If False, fails if target file exists. Defaults to True.")

@tool(args_schema=FileWriterInput)
def file_writer(file_path: str, content: str, overwrite: bool = True) -> str:
    """Creates a new file or overwrites an existing file with the provided text content.

    Args:
        file_path: Relative or absolute destination file path. Parent directories are created automatically.
        content: Complete text string content to write into the file.
        overwrite: If True, overwrites existing file. If False, fails if target file already exists. Defaults to True.

    Returns:
        Success message with character count written, or error message on failure.
    """
    abs_path = os.path.abspath(os.path.expanduser(file_path))
    if os.path.exists(abs_path) and not overwrite:
        return f"FileWrite Error: File '{file_path}' already exists and overwrite=False."

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"SUCCESS: Written {len(content)} characters to '{abs_path}'."
    except Exception as e:
        return f"FileWrite Error: Failed to write to '{file_path}': {str(e)}"


class NotebookEditInput(BaseModel):
    notebook_path: str = Field(description="Relative or absolute path to the .ipynb notebook file.")
    cell_index: int = Field(description="0-indexed integer specifying which cell in the notebook to modify.")
    new_code: str = Field(description="Complete new source code or markdown text for the target cell.")

@tool(args_schema=NotebookEditInput)
def notebook_edit(notebook_path: str, cell_index: int, new_code: str) -> str:
    """Edits a specific code or markdown cell inside a Jupyter Notebook (.ipynb) file.

    Args:
        notebook_path: Relative or absolute path to the .ipynb notebook file.
        cell_index: 0-indexed integer specifying which cell to modify.
        new_code: Complete new source code or markdown text for the target cell.

    Returns:
        Success message confirming cell update, or error message on out-of-bounds index or invalid JSON.
    """
    abs_path = os.path.abspath(os.path.expanduser(notebook_path))
    if not os.path.exists(abs_path):
        return f"NotebookEdit Error: Notebook '{notebook_path}' does not exist."

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        if cell_index < 0 or cell_index >= len(cells):
            return f"NotebookEdit Error: Cell index {cell_index} out of bounds (Notebook has {len(cells)} cells)."

        cells[cell_index]["source"] = new_code.splitlines(keepends=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=2, ensure_ascii=False)

        return f"SUCCESS: Cell #{cell_index} in notebook '{abs_path}' updated successfully."
    except Exception as e:
        return f"NotebookEdit Error: Failed to edit notebook '{notebook_path}': {str(e)}"


# =============================================================================
# Category 2: Shell (Bash)
# =============================================================================

class BashCommandInput(BaseModel):
    command: str = Field(description="The shell command line string to execute in bash/sh.")
    timeout_seconds: int = Field(default=30, description="Maximum execution time in seconds before process is terminated. Defaults to 30.")
    max_stdout_length: int = Field(default=4000, description="Maximum character length of stdout to return to prevent prompt bloat. Defaults to 4000. Max 50000.")

@tool(args_schema=BashCommandInput)
def bash_command(command: str, timeout_seconds: int = 30, max_stdout_length: int = 4000) -> str:
    """Executes a real shell command on local system, capturing stdout, stderr, and exit code.

    Args:
        command: The shell command line string to execute in bash/sh.
        timeout_seconds: Maximum execution time in seconds before process is terminated. Defaults to 30.
        max_stdout_length: Maximum character length of stdout to return to prevent prompt bloat. Defaults to 4000. Maximum allowed is 50000.

    Returns:
        Formatted execution summary including exit code, duration, stdout, and stderr outputs.
    """
    stdout_limit = min(50000, max(100, max_stdout_length))
    
    try:
        start_time = time.time()
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd()
        )
        duration = round(time.time() - start_time, 2)
        
        stdout_clean = process.stdout.strip()
        stderr_clean = process.stderr.strip()
        
        result_parts = [
            f"[Command: '{command}']",
            f"[Exit Code: {process.returncode} | Execution Time: {duration}s]"
        ]
        
        if stdout_clean:
            if len(stdout_clean) > stdout_limit:
                stdout_clean = stdout_clean[:stdout_limit] + f"\n... [STDOUT TRUNCATED {len(stdout_clean)-stdout_limit} bytes]"
            result_parts.append(f"--- STDOUT ---\n{stdout_clean}")
            
        if stderr_clean:
            if len(stderr_clean) > 2000:
                stderr_clean = stderr_clean[:2000] + f"\n... [STDERR TRUNCATED {len(stderr_clean)-2000} bytes]"
            result_parts.append(f"--- STDERR ---\n{stderr_clean}")
            
        if not stdout_clean and not stderr_clean:
            result_parts.append("(Command executed silently with no output)")

        return "\n".join(result_parts)

    except subprocess.TimeoutExpired:
        return f"Bash Error: Command '{command}' timed out after {timeout_seconds} seconds."
    except Exception as e:
        return f"Bash Error: Failed to execute command '{command}': {str(e)}"


# =============================================================================
# Category 3: Search (Grep, Glob, ToolSearch)
# =============================================================================
VCS_EXCLUDE_DIRS = {".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "env_langchain_123"}

class GrepSearchInput(BaseModel):
    pattern: str = Field(description="Regular expression pattern string to match within file contents.")
    search_path: str = Field(default=".", description="Directory or file path to search within. Defaults to current directory ('.').")
    output_mode: str = Field(default="files_with_matches", description="Output format mode - 'files_with_matches' (paths only), 'content' (lines with match), or 'count' (match summary). Defaults to 'files_with_matches'.")
    head_limit: int = Field(default=50, description="Maximum number of matching files or lines to return. Defaults to 50.")

@tool(args_schema=GrepSearchInput)
def grep_search(pattern: str, search_path: str = ".", output_mode: str = "files_with_matches", head_limit: int = 50) -> str:
    """Searches file contents using regex matching while excluding VCS and environment directories.

    Args:
        pattern: Regular expression pattern string to match within file contents.
        search_path: Directory or file path to search within. Defaults to current directory ('.').
        output_mode: Output format mode - 'files_with_matches' (file paths only), 'content' (lines with match), or 'count' (match summary). Defaults to 'files_with_matches'.
        head_limit: Maximum number of matching files or lines to return. Defaults to 50.

    Returns:
        String listing matching file paths, line occurrences, or count summary.
    """
    abs_root = os.path.abspath(os.path.expanduser(search_path))
    if not os.path.exists(abs_root):
        return f"Grep Error: Search path '{search_path}' does not exist."

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except Exception as err:
        return f"Grep Error: Invalid regex pattern '{pattern}': {err}"

    matching_files = []
    content_matches = []
    total_match_count = 0

    if os.path.isfile(abs_root):
        target_files = [abs_root]
    else:
        target_files = []
        for root, dirs, files in os.walk(abs_root):
            dirs[:] = [d for d in dirs if d not in VCS_EXCLUDE_DIRS]
            for f in files:
                target_files.append(os.path.join(root, f))

    for fpath in target_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            rel_path = os.path.relpath(fpath, os.getcwd())
            file_had_match = False
            
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    file_had_match = True
                    total_match_count += 1
                    if output_mode == "content" and len(content_matches) < head_limit:
                        content_matches.append(f"{rel_path}:{line_no}:{line.strip()}")

            if file_had_match:
                matching_files.append(rel_path)
                if output_mode == "files_with_matches" and len(matching_files) >= head_limit:
                    break
        except Exception:
            continue

    if output_mode == "content":
        if not content_matches:
            return f"No matches found for pattern '{pattern}'."
        return f"Found {total_match_count} matches for '{pattern}':\n" + "\n".join(content_matches)

    elif output_mode == "count":
        return f"Found {total_match_count} total occurrences across {len(matching_files)} files."

    else:
        if not matching_files:
            return f"No files matching pattern '{pattern}'."
        return f"Found {len(matching_files)} matching files:\n" + "\n".join(matching_files[:head_limit])


class GlobSearchInput(BaseModel):
    pattern: str = Field(description="Glob wildcard pattern string to match file names and paths (e.g. '**/*.py' or 'src/**/*.ts').")
    search_path: str = Field(default=".", description="Base directory path to execute search from. Defaults to current directory ('.').")

@tool(args_schema=GlobSearchInput)
def glob_search(pattern: str, search_path: str = ".") -> str:
    """Finds file paths matching a glob wildcard pattern (e.g. '**/*.py' or 'src/**/*.ts').

    Args:
        pattern: Glob wildcard pattern string to match file names and paths.
        search_path: Base directory path to execute search from. Defaults to current directory ('.').

    Returns:
        List of relative matching file paths, or a message indicating no matches found.
    """
    abs_root = os.path.abspath(os.path.expanduser(search_path))
    full_pattern = os.path.join(abs_root, pattern)
    
    matches = glob.glob(full_pattern, recursive=True)
    clean_matches = [
        os.path.relpath(m, os.getcwd()) for m in matches 
        if not any(ex in m for ex in VCS_EXCLUDE_DIRS)
    ]
    
    if not clean_matches:
        return f"No files matched glob pattern '{pattern}' in '{search_path}'."
        
    return f"Found {len(clean_matches)} files matching '{pattern}':\n" + "\n".join(clean_matches[:100])


class ToolSearchInput(BaseModel):
    query: str = Field(description="Keyword or search phrase to match against available tool names and capability descriptions.")

@tool(args_schema=ToolSearchInput)
def tool_search(query: str) -> str:
    """Searches registered agent tools by keyword or capability description.

    Args:
        query: Keyword or search phrase to match against available tool names and capability descriptions.

    Returns:
        Formatted string listing matching tools and their capability summaries.
    """
    tools_catalog = {
        "file_read": "Reads file contents with line offset and limit.",
        "file_edit": "Performs exact string matching replacement in local files.",
        "file_writer": "Creates or overwrites files on local disk.",
        "notebook_edit": "Edits cells inside Jupyter notebooks (.ipynb).",
        "bash_command": "Executes shell commands with timeout and exit codes.",
        "grep_search": "Searches file contents using regex.",
        "glob_search": "Finds files matching glob patterns.",
        "tool_search": "Searches registered agent tools catalog.",
        "web_fetch": "Fetches and parses text/markdown from web URLs.",
        "web_search": "Performs web search prioritizing Tavily API with DuckDuckGo fallback.",
    }
    
    query_lower = query.lower()
    matched = [f"{tname}: {desc}" for tname, desc in tools_catalog.items() if query_lower in tname.lower() or query_lower in desc.lower()]
    
    if not matched:
        return f"No tools matching query '{query}'. Available tools: {list(tools_catalog.keys())}"
        
    return "Matched Tools:\n" + "\n".join(matched)


# =============================================================================
# Category 4: Web (WebFetch, WebSearch)
# =============================================================================

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_chunks = []
    def handle_data(self, data):
        cleaned = data.strip()
        if cleaned:
            self.text_chunks.append(cleaned)
    def get_text(self):
        return " ".join(self.text_chunks)


class WebFetchInput(BaseModel):
    url: str = Field(description="Full HTTP or HTTPS web URL string to fetch.")

@tool(args_schema=WebFetchInput)
def web_fetch(url: str) -> str:
    """Fetches raw web content from an HTTP/HTTPS URL and extracts readable plain text.

    Args:
        url: Full HTTP or HTTPS web URL string to fetch.

    Returns:
        Extracted plain text content from the HTML body, truncated to 4000 characters if excessive.
    """
    try:
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html_bytes = response.read()
            html_text = html_bytes.decode("utf-8", errors="ignore")

        parser = HTMLTextExtractor()
        parser.feed(html_text)
        extracted = parser.get_text()

        if len(extracted) > 4000:
            extracted = extracted[:4000] + f"\n... [WEBFETCH TRUNCATED {len(extracted)-4000} bytes]"

        return f"[Fetched URL: {url}]\n" + extracted

    except Exception as e:
        return f"WebFetch Error: Failed to fetch '{url}': {str(e)}"


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string to look up on the web.")

@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Performs web search prioritizing Tavily Search API with automatic fallback to DuckDuckGo if Tavily fails or is unavailable.

    Args:
        query: Search query string to look up on the web.

    Returns:
        Formatted list of top web search result titles, URLs, and text snippets.
    """
    # 1. Attempt Tavily Search first
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_tavily import TavilySearch
            tavily = TavilySearch(max_results=5)
            raw_res = tavily.invoke(query)
            
            if isinstance(raw_res, dict) and "results" in raw_res:
                results = raw_res["results"]
                if results:
                    formatted = []
                    for idx, item in enumerate(results, start=1):
                        title = item.get("title", "Result").strip()
                        url = item.get("url", "")
                        content = item.get("content", item.get("snippet", "")).strip()
                        if url:
                            formatted.append(f"{idx}. [{title}]({url}): {content}")
                        else:
                            formatted.append(f"{idx}. [{title}]: {content}")
                    return f"[Tavily Search Results for '{query}']\n" + "\n".join(formatted)
            elif isinstance(raw_res, list) and raw_res:
                formatted = []
                for idx, item in enumerate(raw_res, start=1):
                    if isinstance(item, dict):
                        title = item.get("title", "Result").strip()
                        url = item.get("url", "")
                        content = item.get("content", item.get("snippet", "")).strip()
                        if url:
                            formatted.append(f"{idx}. [{title}]({url}): {content}")
                        else:
                            formatted.append(f"{idx}. [{title}]: {content}")
                    else:
                        formatted.append(f"{idx}. {str(item)}")
                return f"[Tavily Search Results for '{query}']\n" + "\n".join(formatted)
        except Exception:
            # Fallback to DuckDuckGo if Tavily call fails or rate limits
            pass

    # 2. Fallback to DuckDuckGo Search (DDGS)
    import warnings
    warnings.filterwarnings("ignore")
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5))

        for idx, r in enumerate(search_results):
            clean_title = r.get("title", "Result").strip()
            clean_url = r.get("href", r.get("url", ""))
            clean_snippet = r.get("body", "").strip()
            if clean_url:
                results.append(f"{idx+1}. [{clean_title}]({clean_url}): {clean_snippet}")
            else:
                results.append(f"{idx+1}. [{clean_title}]: {clean_snippet}")

        if not results:
            return f"WebSearch Result for '{query}': No snippets extracted."

        return f"[DuckDuckGo Search Results (Fallback) for '{query}']\n" + "\n".join(results)

    except Exception as e:
        return f"WebSearch Error: Both Tavily and DuckDuckGo searches failed for query '{query}': {str(e)}"
