from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_path() -> Path:
    return project_root() / "47-submissions-clean.json"


def load_items(dataset_path: Path | None = None) -> list[dict[str, Any]]:
    path = dataset_path or default_dataset_path()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return list(data["questions"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset format in {path}")


def _normalize_criterion_items(values: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, str):
            criterion = value.strip()
            if not criterion:
                continue
            normalized.append({"criterion": criterion, "weight": 1.0, "raw_weight": 1.0})
            continue
        if not isinstance(value, dict):
            continue
        criterion = str(value.get("criterion") or value.get("text") or "").strip()
        if not criterion:
            continue
        weight = value.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            weight = 1.0
        normalized.append({"criterion": criterion, "weight": float(weight), "raw_weight": float(weight)})

    total_weight = sum(item["weight"] for item in normalized)
    if total_weight > 0:
        for item in normalized:
            item["weight"] = item["weight"] / total_weight
    return normalized


def get_question_text(item: dict[str, Any]) -> str:
    return str(item.get("prompt") or item.get("question") or "")


def infer_rubric_style(raw: dict[str, Any] | None) -> str:
    rubric = raw if isinstance(raw, dict) else {}
    has_expected_section = any(key in rubric for key in ("expected-criteria", "expected_criteria"))
    has_prohibited_section = any(key in rubric for key in ("prohibited-criteria", "prohibited_criteria"))
    if has_expected_section or has_prohibited_section:
        return "legacy_structured"
    if isinstance(rubric.get("criteria"), list):
        return "weighted_positive"
    return "legacy_structured"


def normalize_criteria(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("refinedCriteria") or item.get("criteria") or item.get("rubric") or {}
    expected_raw = raw.get("expected-criteria") or raw.get("expected_criteria") or raw.get("criteria") or []
    prohibited_raw = raw.get("prohibited-criteria") or raw.get("prohibited_criteria") or []
    expected_items = _normalize_criterion_items(list(expected_raw))
    prohibited_items = _normalize_criterion_items(list(prohibited_raw))
    rubric_style = infer_rubric_style(raw)
    return {
        "raw": raw,
        "expected": [item["criterion"] for item in expected_items],
        "prohibited": [item["criterion"] for item in prohibited_items],
        "expected_items": expected_items,
        "prohibited_items": prohibited_items,
        "rubric_style": rubric_style,
        "has_structured_rubric": bool(expected_items or prohibited_items),
    }


def select_items(
    items: list[dict[str, Any]],
    start_index: int = 1,
    end_index: int | None = None,
    limit: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if start_index < 1:
        raise ValueError("start_index must be >= 1")

    final_index = end_index or len(items)
    if final_index < start_index:
        raise ValueError("end_index must be >= start_index")
    if final_index > len(items):
        raise ValueError(f"end_index must be <= {len(items)}")

    selected = list(enumerate(items[start_index - 1 : final_index], start=start_index))
    if limit is not None:
        selected = selected[:limit]
    return selected
