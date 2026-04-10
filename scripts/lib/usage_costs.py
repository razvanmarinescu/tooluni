from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Pricing:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    cache_write_5m_per_million: float | None = None
    cache_write_1h_per_million: float | None = None
    cache_hit_per_million: float | None = None


PRICING_BY_MODEL: dict[str, Pricing] = {
    "claude-opus-4.6": Pricing(5.0, 25.0, cache_write_5m_per_million=6.25, cache_write_1h_per_million=10.0, cache_hit_per_million=0.50),
    "claude-opus-4-6": Pricing(5.0, 25.0, cache_write_5m_per_million=6.25, cache_write_1h_per_million=10.0, cache_hit_per_million=0.50),
    "claude-opus-4.5": Pricing(5.0, 25.0, cache_write_5m_per_million=6.25, cache_write_1h_per_million=10.0, cache_hit_per_million=0.50),
    "claude-opus-4-5": Pricing(5.0, 25.0, cache_write_5m_per_million=6.25, cache_write_1h_per_million=10.0, cache_hit_per_million=0.50),
    "claude-opus-4.1": Pricing(15.0, 75.0, cache_write_5m_per_million=18.75, cache_write_1h_per_million=30.0, cache_hit_per_million=1.50),
    "claude-opus-4-1": Pricing(15.0, 75.0, cache_write_5m_per_million=18.75, cache_write_1h_per_million=30.0, cache_hit_per_million=1.50),
    "claude-opus-4": Pricing(15.0, 75.0, cache_write_5m_per_million=18.75, cache_write_1h_per_million=30.0, cache_hit_per_million=1.50),
    "claude-opus-3": Pricing(15.0, 75.0, cache_write_5m_per_million=18.75, cache_write_1h_per_million=30.0, cache_hit_per_million=1.50),
    "claude-sonnet-4.6": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-sonnet-4-6": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-sonnet-4.5": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-sonnet-4-5": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-sonnet-4": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-sonnet-3.7": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-sonnet-3-7": Pricing(3.0, 15.0, cache_write_5m_per_million=3.75, cache_write_1h_per_million=6.0, cache_hit_per_million=0.30),
    "claude-haiku-4.5": Pricing(1.0, 5.0, cache_write_5m_per_million=1.25, cache_write_1h_per_million=2.0, cache_hit_per_million=0.10),
    "claude-haiku-4-5": Pricing(1.0, 5.0, cache_write_5m_per_million=1.25, cache_write_1h_per_million=2.0, cache_hit_per_million=0.10),
    "claude-haiku-3.5": Pricing(0.80, 4.0, cache_write_5m_per_million=1.0, cache_write_1h_per_million=1.6, cache_hit_per_million=0.08),
    "claude-haiku-3-5": Pricing(0.80, 4.0, cache_write_5m_per_million=1.0, cache_write_1h_per_million=1.6, cache_hit_per_million=0.08),
    "claude-haiku-3": Pricing(0.25, 1.25, cache_write_5m_per_million=0.30, cache_write_1h_per_million=0.50, cache_hit_per_million=0.03),
    "gpt-5.4": Pricing(2.50, 15.0, cached_input_per_million=0.25),
    "gpt-5-4": Pricing(2.50, 15.0, cached_input_per_million=0.25),
    "gpt-5.4-mini": Pricing(0.75, 4.50, cached_input_per_million=0.075),
    "gpt-5.4 mini": Pricing(0.75, 4.50, cached_input_per_million=0.075),
    "gpt-5-4-mini": Pricing(0.75, 4.50, cached_input_per_million=0.075),
    "gpt-5.4-nano": Pricing(0.20, 1.25, cached_input_per_million=0.02),
    "gpt-5.4 nano": Pricing(0.20, 1.25, cached_input_per_million=0.02),
    "gpt-5-4-nano": Pricing(0.20, 1.25, cached_input_per_million=0.02),
}


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _collect_usage_dicts(node: Any) -> list[dict[str, Any]]:
    usages: list[dict[str, Any]] = []
    if isinstance(node, dict):
        usage = node.get("usage")
        if isinstance(usage, dict):
            usages.append(usage)
        for value in node.values():
            usages.extend(_collect_usage_dicts(value))
    elif isinstance(node, list):
        for item in node:
            usages.extend(_collect_usage_dicts(item))
    return usages


def summarize_usage(raw_payload: Any, provider: str) -> dict[str, int]:
    summary = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_creation_ephemeral_5m_input_tokens": 0,
        "cache_creation_ephemeral_1h_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    for usage in _collect_usage_dicts(raw_payload):
        summary["input_tokens"] += _to_int(usage.get("input_tokens"))
        summary["output_tokens"] += _to_int(usage.get("output_tokens"))
        summary["cache_creation_input_tokens"] += _to_int(usage.get("cache_creation_input_tokens"))
        summary["cache_read_input_tokens"] += _to_int(usage.get("cache_read_input_tokens"))
        cache_creation = usage.get("cache_creation") or {}
        if isinstance(cache_creation, dict):
            summary["cache_creation_ephemeral_5m_input_tokens"] += _to_int(cache_creation.get("ephemeral_5m_input_tokens"))
            summary["cache_creation_ephemeral_1h_input_tokens"] += _to_int(cache_creation.get("ephemeral_1h_input_tokens"))
        if provider == "openai":
            input_details = usage.get("input_tokens_details") or {}
            if isinstance(input_details, dict):
                summary["cached_input_tokens"] += _to_int(input_details.get("cached_tokens"))
    return summary


def estimate_cost_usd(provider: str, model_name: str, usage: dict[str, int] | None) -> float | None:
    if not usage:
        return None
    pricing = PRICING_BY_MODEL.get(model_name)
    if pricing is None:
        pricing = PRICING_BY_MODEL.get(model_name.lower())
    if pricing is None:
        return None

    output_tokens = usage.get("output_tokens", 0)
    if provider == "openai":
        cached_input_tokens = usage.get("cached_input_tokens", 0)
        input_tokens = max(usage.get("input_tokens", 0) - cached_input_tokens, 0)
        total = (input_tokens / 1_000_000.0) * pricing.input_per_million
        if pricing.cached_input_per_million is not None:
            total += (cached_input_tokens / 1_000_000.0) * pricing.cached_input_per_million
        total += (output_tokens / 1_000_000.0) * pricing.output_per_million
        return round(total, 6)

    cache_creation_5m = usage.get("cache_creation_ephemeral_5m_input_tokens", 0)
    cache_creation_1h = usage.get("cache_creation_ephemeral_1h_input_tokens", 0)
    cache_creation_total = usage.get("cache_creation_input_tokens", 0)
    if cache_creation_total > (cache_creation_5m + cache_creation_1h):
        cache_creation_5m += cache_creation_total - (cache_creation_5m + cache_creation_1h)

    total = (usage.get("input_tokens", 0) / 1_000_000.0) * pricing.input_per_million
    if pricing.cache_write_5m_per_million is not None:
        total += (cache_creation_5m / 1_000_000.0) * pricing.cache_write_5m_per_million
    if pricing.cache_write_1h_per_million is not None:
        total += (cache_creation_1h / 1_000_000.0) * pricing.cache_write_1h_per_million
    if pricing.cache_hit_per_million is not None:
        total += (usage.get("cache_read_input_tokens", 0) / 1_000_000.0) * pricing.cache_hit_per_million
    total += (output_tokens / 1_000_000.0) * pricing.output_per_million
    return round(total, 6)


def build_usage_metrics(provider: str, model_name: str, raw_payload: Any) -> dict[str, Any]:
    usage = summarize_usage(raw_payload, provider)
    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    metrics: dict[str, Any] = {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": total_tokens,
        "cached_input_tokens": usage["cached_input_tokens"],
        "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
        "cache_creation_ephemeral_5m_input_tokens": usage["cache_creation_ephemeral_5m_input_tokens"],
        "cache_creation_ephemeral_1h_input_tokens": usage["cache_creation_ephemeral_1h_input_tokens"],
        "cache_read_input_tokens": usage["cache_read_input_tokens"],
    }
    metrics["estimated_cost_usd"] = estimate_cost_usd(provider, model_name, usage)
    return metrics


def combine_usage_metrics(prefix: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {
            f"{prefix}_input_tokens": None,
            f"{prefix}_output_tokens": None,
            f"{prefix}_total_tokens": None,
            f"{prefix}_cached_input_tokens": None,
            f"{prefix}_cache_creation_input_tokens": None,
            f"{prefix}_cache_creation_ephemeral_5m_input_tokens": None,
            f"{prefix}_cache_creation_ephemeral_1h_input_tokens": None,
            f"{prefix}_cache_read_input_tokens": None,
            f"{prefix}_estimated_cost_usd": None,
        }
    return {f"{prefix}_{key}": value for key, value in metrics.items()}