from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


OPENAI_TOOLUNIVERSE_TOOLS = [
    {
        "type": "function",
        "name": "tooluniverse_find_tools",
        "description": "Search ToolUniverse for relevant tools by natural-language query before executing a tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query for relevant tools."},
                "limit": {"type": "integer", "description": "Maximum number of tools to return.", "default": 10},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional category filters.",
                },
                "use_advanced_search": {
                    "type": "boolean",
                    "description": "Whether to use ToolUniverse advanced search.",
                    "default": True,
                },
                "search_method": {
                    "type": "string",
                    "description": "Search strategy, usually 'auto'.",
                    "default": "auto",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "tooluniverse_get_tool_info",
        "description": "Get full ToolUniverse schema/details for one or more tools before execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_names": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "A single tool name or a list of tool names.",
                },
                "detail_level": {
                    "type": "string",
                    "description": "Either 'description' or 'full'.",
                    "default": "full",
                },
            },
            "required": ["tool_names"],
        },
    },
    {
        "type": "function",
        "name": "tooluniverse_execute_tool",
        "description": "Execute a specific ToolUniverse tool. Use this for actual external data retrieval, not just discovery.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Exact ToolUniverse tool name to execute."},
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments as a JSON object. Pass the exact parameters required by the chosen ToolUniverse tool.",
                    "default": {},
                    "additionalProperties": True,
                },
            },
            "required": ["tool_name", "arguments"],
        },
    },
    {
        "type": "function",
        "name": "tooluniverse_grep_tools",
        "description": "Search ToolUniverse tools by text pattern in the name, description, type, or category.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text pattern to search for."},
                "field": {"type": "string", "description": "Optional field to search in."},
                "search_mode": {
                    "type": "string",
                    "description": "Either 'text' or 'regex'.",
                    "default": "text",
                },
                "limit": {"type": "integer", "description": "Maximum number of results.", "default": 20},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional category filters.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "type": "function",
        "name": "tooluniverse_list_tools",
        "description": "List ToolUniverse tools or categories. Prefer find_tools or grep_tools for focused discovery.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "Output mode such as names, categories, or summary.", "default": "names"},
                "categories": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "description": "Maximum number of tools to return.", "default": 20},
                "offset": {"type": "integer", "description": "Offset for pagination.", "default": 0},
            },
            "required": [],
        },
    },
]

ANTHROPIC_TOOLUNIVERSE_TOOLS = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": tool["parameters"],
    }
    for tool in OPENAI_TOOLUNIVERSE_TOOLS
]

MODEL_TOOL_TO_MCP_TOOL = {
    "tooluniverse_find_tools": "find_tools",
    "tooluniverse_get_tool_info": "get_tool_info",
    "tooluniverse_execute_tool": "execute_tool",
    "tooluniverse_grep_tools": "grep_tools",
    "tooluniverse_list_tools": "list_tools",
}

TRACE_DISPLAY_NAME = {
    "tooluniverse_find_tools": "tooluniverse.find_tools",
    "tooluniverse_get_tool_info": "tooluniverse.get_tool_info",
    "tooluniverse_execute_tool": "tooluniverse.execute_tool",
    "tooluniverse_grep_tools": "tooluniverse.grep_tools",
    "tooluniverse_list_tools": "tooluniverse.list_tools",
}

TOOLUNIVERSE_SYSTEM_PROMPT = (
    "You have real ToolUniverse access through MCP-backed tools. "
    "You must use ToolUniverse tools before giving your final answer, and at least one call must use tooluniverse_execute_tool to retrieve actual external data. "
    "Use tooluniverse_find_tools or tooluniverse_get_tool_info when you need to discover or inspect tools, then use tooluniverse_execute_tool for the substantive lookups. "
    "Prefer a small number of high-value tool calls, keep arguments precise, and synthesize the results directly for the user."
)


@dataclass
class ToolUniverseExecution:
    model_result: Any
    trace_event: dict[str, Any]


class ToolUniverseMCPClient:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        env = {"PYTHONIOENCODING": "utf-8"}
        env.update({key: value for key, value in os.environ.items() if value is not None})
        self.server_params = StdioServerParameters(
            command="uvx",
            args=["tooluniverse"],
            env=env,
            cwd=str(project_root),
        )
        self._stdio_context = None
        self._session_context = None
        self._read_stream = None
        self._write_stream = None
        self.session: ClientSession | None = None
        self.step_counter = 0

    async def __aenter__(self) -> ToolUniverseMCPClient:
        self._stdio_context = stdio_client(self.server_params)
        self._read_stream, self._write_stream = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(self._read_stream, self._write_stream)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._session_context is not None:
                with contextlib.suppress(Exception):
                    await self._session_context.__aexit__(exc_type, exc, tb)
            if self._stdio_context is not None:
                with contextlib.suppress(Exception):
                    await self._stdio_context.__aexit__(exc_type, exc, tb)
        finally:
            self.session = None

    async def execute_model_tool(self, model_tool_name: str, arguments: dict[str, Any]) -> ToolUniverseExecution:
        if self.session is None:
            raise RuntimeError("ToolUniverse MCP session is not initialized.")

        mcp_tool_name = MODEL_TOOL_TO_MCP_TOOL[model_tool_name]
        mcp_arguments = self._prepare_mcp_arguments(model_tool_name, arguments)
        result = await self.session.call_tool(mcp_tool_name, mcp_arguments)
        raw_result = result.model_dump(mode="json")
        normalized = self._normalize_result(raw_result)
        self.step_counter += 1
        ok = self._is_successful_result(raw_result, normalized)
        trace_event = {
            "step": self.step_counter,
            "display_name": TRACE_DISPLAY_NAME[model_tool_name],
            "model_tool_name": model_tool_name,
            "mcp_tool_name": mcp_tool_name,
            "arguments": mcp_arguments,
            "original_model_arguments": arguments,
            "ok": ok,
            "result": normalized,
            "raw_result": raw_result,
        }
        return ToolUniverseExecution(model_result=normalized, trace_event=trace_event)

    @staticmethod
    def _normalize_result(raw_result: dict[str, Any]) -> Any:
        structured = raw_result.get("structuredContent")
        if structured is not None:
            return ToolUniverseMCPClient._deep_parse_jsonish(structured)

        content = raw_result.get("content") or []
        if len(content) == 1 and content[0].get("type") == "text":
            text = content[0].get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return raw_result

    @staticmethod
    def _prepare_mcp_arguments(model_tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if model_tool_name != "tooluniverse_execute_tool":
            return arguments

        if "arguments" in arguments:
            tool_arguments = arguments["arguments"]
        elif "arguments_json" in arguments:
            tool_arguments = arguments["arguments_json"]
        else:
            tool_arguments = {
                key: value
                for key, value in arguments.items()
                if key not in {"tool_name", "arguments", "arguments_json"}
            }

        if isinstance(tool_arguments, str):
            try:
                tool_arguments = json.loads(tool_arguments)
            except json.JSONDecodeError:
                tool_arguments = {"_raw": tool_arguments}

        if not isinstance(tool_arguments, dict):
            raise ValueError("tooluniverse_execute_tool arguments must decode to an object.")

        return {
            "tool_name": arguments.get("tool_name", ""),
            "arguments": tool_arguments,
        }

    @staticmethod
    def _is_successful_result(raw_result: dict[str, Any], normalized: Any) -> bool:
        if raw_result.get("isError"):
            return False
        if isinstance(normalized, dict) and normalized.get("status") == "error":
            return False
        return True

    @staticmethod
    def _deep_parse_jsonish(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ToolUniverseMCPClient._deep_parse_jsonish(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ToolUniverseMCPClient._deep_parse_jsonish(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{":
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return value
                return ToolUniverseMCPClient._deep_parse_jsonish(parsed)
        return value


def render_pretty_tool_trace(trace_events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in trace_events:
        lines.append(
            f"• Called {event['display_name']}({json.dumps(event['arguments'], ensure_ascii=True)})"
        )
        rendered = json.dumps(event["result"], ensure_ascii=True, indent=2)
        rendered_lines = rendered.splitlines() or [rendered]
        lines.append(f"  └ {rendered_lines[0]}")
        for line in rendered_lines[1:]:
            lines.append(f"    {line}")
    return "\n".join(lines)