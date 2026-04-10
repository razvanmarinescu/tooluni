from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_path() -> Path:
    return project_root() / "48-submissions-clean.json"


def load_items(dataset_path: Path | None = None) -> list[dict[str, Any]]:
    path = dataset_path or default_dataset_path()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_criteria(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("refinedCriteria") or item.get("criteria") or {}
    expected = raw.get("expected-criteria") or raw.get("expected_criteria") or []
    prohibited = raw.get("prohibited-criteria") or raw.get("prohibited_criteria") or []
    return {
        "raw": raw,
        "expected": list(expected),
        "prohibited": list(prohibited),
        "has_structured_rubric": bool(expected or prohibited),
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
