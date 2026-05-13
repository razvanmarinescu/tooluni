from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older environments
    from duckduckgo_search import DDGS


# DDGS holds non-thread-safe state in its HTTP client / cookie jar; concurrent
# DDGS() sessions from different worker threads deadlock indefinitely. A
# module-level lock serializes the call. Each invocation only takes 1–3s, so
# the throughput cost is negligible compared to the LLM call that follows it.
_DDGS_LOCK = threading.Lock()


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "do",
    "for",
    "from",
    "given",
    "how",
    "in",
    "include",
    "into",
    "is",
    "it",
    "its",
    "may",
    "must",
    "not",
    "of",
    "on",
    "or",
    "our",
    "response",
    "should",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "use",
    "via",
    "want",
    "we",
    "what",
    "which",
    "with",
    "your",
}


def _run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def _compact_query(query: str, max_terms: int = 18) -> str:
    cleaned = re.sub(r"\s+", " ", query.strip())
    if not cleaned:
        return query

    first_line = cleaned.split("\n", 1)[0].strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
    candidate = first_sentence or first_line

    words = re.findall(r"[A-Za-z0-9+\-_/]{2,}", candidate)
    filtered: list[str] = []
    seen: set[str] = set()
    for word in words:
        normalized = word.lower()
        if normalized in STOPWORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        filtered.append(word)
        if len(filtered) >= max_terms:
            break

    if filtered:
        return " ".join(filtered)
    return candidate[:240]


def web_search_context(query: str, max_results: int = 5) -> dict[str, Any]:
    search_query = _compact_query(query)
    try:
        with _DDGS_LOCK:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))
    except Exception as exc:  # pragma: no cover - network errors are environment-specific
        return {"query": search_query, "results": [], "error": str(exc)}

    normalized = []
    for result in results:
        normalized.append(
            {
                "title": result.get("title", ""),
                "href": result.get("href", ""),
                "body": result.get("body", ""),
            }
        )
    return {"query": search_query, "results": normalized, "error": None}


def format_web_context(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"Web search was attempted but failed: {payload['error']}"

    results = payload.get("results", [])
    if not results:
        return "No web search results were retrieved."

    lines = ["Web search context:"]
    if payload.get("query"):
        lines.append(f"Query: {payload['query']}")
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}")
        if result.get("body"):
            lines.append(f"   Snippet: {result['body']}")
        if result.get("href"):
            lines.append(f"   URL: {result['href']}")
    return "\n".join(lines)


def build_web_search_trace(payload: dict[str, Any], injected_text: str) -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "query": payload.get("query"),
            "error": payload.get("error"),
            "raw_results": payload.get("results", []),
            "injected_text": injected_text,
        }
    ]


def render_pretty_web_trace(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return ""

    lines: list[str] = []
    for event in trace:
        step = event.get("step", "?")
        lines.append(f"Step {step}: web search")
        if event.get("query"):
            lines.append(f"  Query: {event['query']}")
        if event.get("error"):
            lines.append(f"  Error: {event['error']}")
        results = event.get("raw_results") or []
        lines.append(f"  Result count: {len(results)}")
        for index, result in enumerate(results, start=1):
            title = result.get("title") or ""
            href = result.get("href") or ""
            body = result.get("body") or ""
            lines.append(f"  {index}. {title}")
            if href:
                lines.append(f"     URL: {href}")
            if body:
                lines.append(f"     Snippet: {body}")
    return "\n".join(lines)


def render_markdown_web_trace(
    *,
    question: str,
    dataset_index: int,
    submission_id: str,
    model_name: str,
    trace: list[dict[str, Any]],
    raw_response: dict[str, Any] | None,
    final_response_text: str | None,
) -> str:
    """Per-question markdown for the ``web_tools`` tier: DDGS query and
    results that were injected into the prompt, then the model's final
    output (and thinking blocks, when present)."""
    out: list[str] = []
    out.append(f"# q{dataset_index:02d} — {model_name} (web_tools)")
    out.append("")
    out.append(f"_submission: `{submission_id}`_")
    out.append("")
    out.append("## Question")
    out.append("")
    out.append(question.strip())
    out.append("")

    for event in trace or []:
        out.append(f"## Web search (step {event.get('step', '?')})")
        out.append("")
        q = event.get("query")
        if q:
            out.append(f"**Query:** `{q}`")
            out.append("")
        if event.get("error"):
            out.append(f"**Error:** {event['error']}")
            out.append("")
        results = event.get("raw_results") or []
        out.append(f"**Result count:** {len(results)}")
        out.append("")
        for index, result in enumerate(results, start=1):
            title = result.get("title") or "(no title)"
            href = result.get("href") or ""
            body = (result.get("body") or "").strip()
            if href:
                out.append(f"{index}. [{title}]({href})")
            else:
                out.append(f"{index}. {title}")
            if body:
                snippet = body if len(body) <= 600 else body[:600].rstrip() + "…"
                out.append(f"   > {snippet}")
        out.append("")

    # Pull thinking + text from the final model response, if available.
    if raw_response is not None:
        final = raw_response.get("final") if isinstance(raw_response, dict) else None
        if isinstance(final, dict):
            content = final.get("content") or final.get("output") or []
            thinking_pieces: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    txt = (block.get("thinking") or "").strip()
                    if txt:
                        thinking_pieces.append(txt)
                    elif block.get("signature"):
                        thinking_pieces.append(
                            "_(adaptive thinking occurred; content not exposed by API)_"
                        )
                elif btype == "redacted_thinking":
                    thinking_pieces.append("_(redacted thinking block)_")
                elif btype == "reasoning":
                    summary = block.get("summary") or block.get("text") or ""
                    if summary:
                        thinking_pieces.append(str(summary))
            if thinking_pieces:
                out.append("## Thinking")
                out.append("")
                for piece in thinking_pieces:
                    out.append("> " + piece.replace("\n", "\n> "))
                    out.append("")

    if final_response_text:
        out.append("## Final answer")
        out.append("")
        out.append(final_response_text.strip())
        out.append("")

    return "\n".join(out)


def _normalize_tool_results(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "items", "matches", "tools"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    return []


def tooluniverse_context(query: str, cwd: Path | None = None, max_results: int = 5) -> dict[str, Any]:
    search_query = _compact_query(query)
    find_command = [
        "uvx",
        "--env-file",
        ".env",
        "--from",
        "tooluniverse",
        "tu",
        "find",
        "--raw",
        search_query,
        "--limit",
        str(max_results),
    ]
    find_result = _run_command(find_command, cwd=cwd)
    if find_result.returncode != 0:
        return {
            "query": search_query,
            "results": [],
            "tool_details": [],
            "error": find_result.stderr.strip() or find_result.stdout.strip() or "ToolUniverse find failed",
        }

    try:
        find_payload = json.loads(find_result.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        return {
            "query": search_query,
            "results": [],
            "tool_details": [],
            "error": f"Could not parse ToolUniverse output: {exc}",
        }

    tools = _normalize_tool_results(find_payload)
    tool_names = []
    for item in tools:
        for key in ("name", "tool_name", "id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                tool_names.append(value)
                break

    tool_names = tool_names[: max_results if max_results > 0 else len(tool_names)]
    tool_details: list[dict[str, Any]] = []
    if tool_names:
        info_command = [
            "uvx",
            "--env-file",
            ".env",
            "--from",
            "tooluniverse",
            "tu",
            "info",
            "--raw",
            "--detail",
            "brief",
            *tool_names,
        ]
        info_result = _run_command(info_command, cwd=cwd)
        if info_result.returncode == 0:
            try:
                info_payload = json.loads(info_result.stdout.strip() or "[]")
                tool_details = _normalize_tool_results(info_payload)
            except json.JSONDecodeError:
                tool_details = []

    return {"query": search_query, "results": tools, "tool_details": tool_details, "error": None}


def format_tooluniverse_context(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"ToolUniverse lookup was attempted but failed: {payload['error']}"

    results = payload.get("results", [])
    details = payload.get("tool_details", [])
    if not results:
        return "No ToolUniverse matches were retrieved."

    detail_by_name: dict[str, dict[str, Any]] = {}
    for detail in details:
        name = detail.get("name") or detail.get("tool_name") or detail.get("id")
        if isinstance(name, str) and name:
            detail_by_name[name] = detail

    lines = ["ToolUniverse context:"]
    if payload.get("query"):
        lines.append(f"Query: {payload['query']}")
    for index, result in enumerate(results, start=1):
        name = result.get("name") or result.get("tool_name") or result.get("id") or f"tool-{index}"
        description = (
            result.get("description")
            or result.get("summary")
            or detail_by_name.get(name, {}).get("description")
            or detail_by_name.get(name, {}).get("summary")
            or ""
        )
        category = result.get("category") or result.get("categories") or detail_by_name.get(name, {}).get("category")

        lines.append(f"{index}. {name}")
        if category:
            lines.append(f"   Category: {category}")
        if description:
            lines.append(f"   Description: {description}")
    return "\n".join(lines)
