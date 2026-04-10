from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from .context import format_tooluniverse_context, format_web_context, tooluniverse_context, web_search_context
from .tooluniverse_mcp import (
    ANTHROPIC_TOOLUNIVERSE_TOOLS,
    OPENAI_TOOLUNIVERSE_TOOLS,
    TOOLUNIVERSE_SYSTEM_PROMPT,
    ToolUniverseMCPClient,
    render_pretty_tool_trace,
)


DEFAULT_SYSTEM_PROMPT = (
    "You answer scientific experimental-design and research questions directly. "
    "Do not mention hidden reasoning. Be concise but complete, and use provided context when it is relevant. "
    "Prefer a compact, well-structured answer over an exhaustive essay."
)

DEFAULT_OUTPUT_TOKEN_BUDGET = 5000
RETRY_OUTPUT_TOKEN_BUDGET = 8000
MAX_TOOLUNIVERSE_TOOL_CALLS = 30


def load_local_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)
    if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTROPIC_API_KEY"]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_name: str
    display_name: str


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(provider="openai", model_name="gpt-5.4", display_name="gpt-5.4"),
        ModelSpec(provider="anthropic", model_name="claude-sonnet-4-6", display_name="claude-sonnet-4.6"),
    ]


def default_tiers() -> list[str]:
    return ["internal_only", "web_tools", "tooluniverse"]


def build_user_prompt(question: str, tier: str, context_text: str | None) -> str:
    lines = ["Question:", question.strip()]
    if tier != "internal_only" and context_text:
        lines.extend(["", context_text.strip(), "", "Use the context above when it helps, but do not rely on it blindly."])
    return "\n".join(lines)


class AnswerRunner:
    def __init__(self, project_root: Path):
        load_local_env(project_root)
        self.project_root = project_root
        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._openai_client

    @property
    def anthropic_client(self) -> Anthropic:
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._anthropic_client

    def gather_context(self, question: str, tier: str) -> dict[str, Any]:
        if tier == "internal_only":
            return {"type": tier, "text": None, "raw": None}
        if tier == "web_tools":
            payload = web_search_context(question)
            return {"type": tier, "text": format_web_context(payload), "raw": payload}
        if tier == "tooluniverse":
            payload = tooluniverse_context(question, cwd=self.project_root)
            return {"type": tier, "text": format_tooluniverse_context(payload), "raw": payload}
        raise ValueError(f"Unsupported tier: {tier}")

    def generate(self, model: ModelSpec, tier: str, question: str) -> dict[str, Any]:
        try:
            if tier == "tooluniverse":
                context_payload = {
                    "type": tier,
                    "text": "ToolUniverse MCP tools are available at runtime.",
                    "raw": {"mode": "mcp_tool_calls", "transport": "stdio", "core_tools": [tool["name"] for tool in OPENAI_TOOLUNIVERSE_TOOLS]},
                }
                if model.provider == "openai":
                    text, raw, trace = self._generate_openai_tooluniverse(model.model_name, question)
                elif model.provider == "anthropic":
                    text, raw, trace = self._generate_anthropic_tooluniverse(model.model_name, question)
                else:
                    raise ValueError(f"Unsupported provider: {model.provider}")
            else:
                context_payload = self.gather_context(question, tier)
                user_prompt = build_user_prompt(question, tier, context_payload.get("text"))
                trace = []
                if model.provider == "openai":
                    text, raw = self._generate_openai(model.model_name, user_prompt)
                elif model.provider == "anthropic":
                    text, raw = self._generate_anthropic(model.model_name, user_prompt)
                else:
                    raise ValueError(f"Unsupported provider: {model.provider}")
        except Exception as exc:  # pragma: no cover - depends on external APIs
            return {
                "provider": model.provider,
                "model_name": model.model_name,
                "display_name": model.display_name,
                "tier": tier,
                "response_text": None,
                "error": str(exc),
                "context": context_payload,
                "raw_response": None,
                "tool_trace": [],
            }

        return {
            "provider": model.provider,
            "model_name": model.model_name,
            "display_name": model.display_name,
            "tier": tier,
            "response_text": text,
            "error": None,
            "context": context_payload,
            "raw_response": raw,
            "tool_trace": trace,
        }

    def _generate_openai(self, model_name: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        last_text = ""
        for budget in (DEFAULT_OUTPUT_TOKEN_BUDGET, RETRY_OUTPUT_TOKEN_BUDGET):
            response = self.openai_client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=budget,
            )
            raw = response.model_dump(mode="json")
            attempts.append({"max_output_tokens": budget, "response": raw})
            last_text = response.output_text
            if not self._openai_response_incomplete(raw):
                return last_text, {"attempts": attempts, "final": raw}

        raise RuntimeError("OpenAI response was truncated after retrying with a larger token budget.")

    def _generate_anthropic(self, model_name: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        last_text = ""
        for budget in (DEFAULT_OUTPUT_TOKEN_BUDGET, RETRY_OUTPUT_TOKEN_BUDGET):
            response = self.anthropic_client.messages.create(
                model=model_name,
                system=DEFAULT_SYSTEM_PROMPT,
                max_tokens=budget,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.model_dump(mode="json")
            attempts.append({"max_tokens": budget, "response": raw})
            parts = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            last_text = "".join(parts)
            if not self._anthropic_response_incomplete(raw):
                return last_text, {"attempts": attempts, "final": raw}

        raise RuntimeError("Anthropic response was truncated after retrying with a larger token budget.")

    @staticmethod
    def _openai_response_incomplete(raw: dict[str, Any]) -> bool:
        if raw.get("status") == "incomplete":
            return True
        return bool(raw.get("incomplete_details"))

    @staticmethod
    def _anthropic_response_incomplete(raw: dict[str, Any]) -> bool:
        return raw.get("stop_reason") == "max_tokens"

    def _generate_openai_tooluniverse(self, model_name: str, question: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        return anyio.run(self._generate_openai_tooluniverse_async, model_name, question)

    async def _generate_openai_tooluniverse_async(self, model_name: str, question: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        raw_turns: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []
        execute_tool_calls = 0

        async with ToolUniverseMCPClient(self.project_root) as tool_client:
            response = self.openai_client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": TOOLUNIVERSE_SYSTEM_PROMPT},
                    {"role": "user", "content": question.strip()},
                ],
                tools=OPENAI_TOOLUNIVERSE_TOOLS,
                tool_choice="required",
                max_output_tokens=DEFAULT_OUTPUT_TOKEN_BUDGET,
            )
            raw_turns.append(response.model_dump(mode="json"))

            while True:
                current_raw = raw_turns[-1]
                tool_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
                if tool_calls:
                    outputs = []
                    for tool_call in tool_calls:
                        args = json.loads(tool_call.arguments or "{}")
                        execution = await tool_client.execute_model_tool(tool_call.name, args)
                        trace_events.append(execution.trace_event)
                        if execution.trace_event["mcp_tool_name"] == "execute_tool" and execution.trace_event["ok"]:
                            execute_tool_calls += 1
                        outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.call_id,
                                "output": json.dumps(execution.model_result, ensure_ascii=True),
                            }
                        )
                    if len(trace_events) > MAX_TOOLUNIVERSE_TOOL_CALLS:
                        raise RuntimeError("Exceeded maximum ToolUniverse tool calls in OpenAI tool loop.")
                    response = self.openai_client.responses.create(
                        model=model_name,
                        previous_response_id=response.id,
                        input=outputs,
                        tools=OPENAI_TOOLUNIVERSE_TOOLS,
                        max_output_tokens=DEFAULT_OUTPUT_TOKEN_BUDGET,
                    )
                    raw_turns.append(response.model_dump(mode="json"))
                    continue

                if execute_tool_calls == 0:
                    response = self.openai_client.responses.create(
                        model=model_name,
                        previous_response_id=response.id,
                        input=[
                            {
                                "role": "user",
                                "content": "You must use tooluniverse_execute_tool at least once to gather real ToolUniverse data before finalizing your answer.",
                            }
                        ],
                        tools=OPENAI_TOOLUNIVERSE_TOOLS,
                        tool_choice="required",
                        max_output_tokens=DEFAULT_OUTPUT_TOKEN_BUDGET,
                    )
                    raw_turns.append(response.model_dump(mode="json"))
                    continue

                if self._openai_response_incomplete(current_raw):
                    raise RuntimeError("OpenAI ToolUniverse answer was truncated.")
                return response.output_text, {"turns": raw_turns, "pretty_tool_trace": render_pretty_tool_trace(trace_events)}, trace_events

    def _generate_anthropic_tooluniverse(self, model_name: str, question: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        return anyio.run(self._generate_anthropic_tooluniverse_async, model_name, question)

    async def _generate_anthropic_tooluniverse_async(self, model_name: str, question: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        raw_turns: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []
        execute_tool_calls = 0
        messages: list[dict[str, Any]] = [{"role": "user", "content": question.strip()}]
        tool_choice: dict[str, Any] | None = {"type": "any"}
        tools_enabled = True
        forced_execute_reminder_sent = False
        forced_finalize_sent = False

        async with ToolUniverseMCPClient(self.project_root) as tool_client:
            while True:
                response = self.anthropic_client.messages.create(
                    model=model_name,
                    system=TOOLUNIVERSE_SYSTEM_PROMPT,
                    max_tokens=DEFAULT_OUTPUT_TOKEN_BUDGET,
                    messages=messages,
                    tools=ANTHROPIC_TOOLUNIVERSE_TOOLS if tools_enabled else None,
                    tool_choice=tool_choice,
                )
                current_raw = response.model_dump(mode="json")
                raw_turns.append(current_raw)

                tool_uses = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
                if tool_uses:
                    messages.append({"role": "assistant", "content": current_raw["content"]})
                    tool_results = []
                    for block in tool_uses:
                        args = dict(block.input)
                        execution = await tool_client.execute_model_tool(block.name, args)
                        trace_events.append(execution.trace_event)
                        if execution.trace_event["mcp_tool_name"] == "execute_tool" and execution.trace_event["ok"]:
                            execute_tool_calls += 1
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(execution.model_result, ensure_ascii=True),
                            }
                        )
                    messages.append({"role": "user", "content": tool_results})

                    if len(trace_events) >= MAX_TOOLUNIVERSE_TOOL_CALLS and execute_tool_calls > 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": "You already have enough ToolUniverse evidence. Do not call more tools. Write the final answer now.",
                            }
                        )
                        tools_enabled = False
                        tool_choice = None
                        forced_finalize_sent = True
                        continue

                    if len(trace_events) >= 12 and execute_tool_calls == 0 and not forced_execute_reminder_sent:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Stop tool discovery. On the next tool step, call tooluniverse_execute_tool exactly once with concrete arguments for a real data source, "
                                    "for example PubMed_search_articles with a query about elephant TP53, or an NCBI/Ensembl gene lookup for TP53 in elephant. After that, finalize the answer."
                                ),
                            }
                        )
                        forced_execute_reminder_sent = True
                        tool_choice = {"type": "any"}
                        continue

                    if len(trace_events) > MAX_TOOLUNIVERSE_TOOL_CALLS:
                        raise RuntimeError("Exceeded maximum ToolUniverse tool calls in Anthropic tool loop.")

                    tool_choice = {"type": "auto"}
                    continue

                if execute_tool_calls == 0:
                    messages.append({"role": "assistant", "content": current_raw["content"]})
                    messages.append(
                        {
                            "role": "user",
                            "content": "You must use tooluniverse_execute_tool at least once to gather real ToolUniverse data before finalizing your answer.",
                        }
                    )
                    tool_choice = {"type": "any"}
                    tools_enabled = True
                    continue

                if forced_finalize_sent:
                    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
                    return "".join(text_parts), {"turns": raw_turns, "pretty_tool_trace": render_pretty_tool_trace(trace_events)}, trace_events

                if self._anthropic_response_incomplete(current_raw):
                    raise RuntimeError("Anthropic ToolUniverse answer was truncated.")

                text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
                return "".join(text_parts), {"turns": raw_turns, "pretty_tool_trace": render_pretty_tool_trace(trace_events)}, trace_events
