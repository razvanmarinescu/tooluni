from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if path.exists() and path.stat().st_size > 0:
            handle.write("\n")
        handle.write(json.dumps(row, ensure_ascii=True, indent=2) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    length = len(content)

    while index < length:
        while index < length and content[index].isspace():
            index += 1
        if index >= length:
            break
        row, next_index = decoder.raw_decode(content, index)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object in {path}, found {type(row).__name__}.")
        rows.append(row)
        index = next_index

    return rows


def _runner_key(row: dict[str, Any]) -> str:
    return f"{row['display_name']} | {row['tier']}"


def write_summary_csv(path: Path, judgments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_index",
        "submission_id",
        "display_name",
        "provider",
        "model_name",
        "tier",
        "has_structured_rubric",
        "response_error",
        "judge_error",
        "expected_coverage",
        "prohibited_rate",
        "final_score",
        "average_holistic_score",
        "summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in judgments:
            scores = row.get("judgment", {}).get("scores", {})
            writer.writerow(
                {
                    "dataset_index": row.get("dataset_index"),
                    "submission_id": row.get("submission_id"),
                    "display_name": row.get("display_name"),
                    "provider": row.get("provider"),
                    "model_name": row.get("model_name"),
                    "tier": row.get("tier"),
                    "has_structured_rubric": row.get("has_structured_rubric"),
                    "response_error": row.get("response_error"),
                    "judge_error": row.get("judge_error"),
                    "expected_coverage": scores.get("expected_coverage"),
                    "prohibited_rate": scores.get("prohibited_rate"),
                    "final_score": scores.get("final_score"),
                    "average_holistic_score": scores.get("average_holistic_score"),
                    "summary": row.get("judgment", {}).get("summary", ""),
                }
            )


def write_summary_markdown(path: Path, judgments: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[_runner_key(row)].append(row)

    lines = ["# Evaluation Summary", ""]
    if not judgments:
        lines.append("No judgments were produced.")
    else:
        lines.append("## Aggregate")
        lines.append("")
        lines.append("| Runner | Items | Mean final score | Mean expected coverage | Total prohibited rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key in sorted(grouped):
            rows = grouped[key]
            final_scores = [r.get("judgment", {}).get("scores", {}).get("final_score") for r in rows]
            final_scores = [value for value in final_scores if isinstance(value, (int, float))]
            expected_coverages = [r.get("judgment", {}).get("scores", {}).get("expected_coverage") for r in rows]
            expected_coverages = [value for value in expected_coverages if isinstance(value, (int, float))]
            prohibited_rates = [r.get("judgment", {}).get("scores", {}).get("prohibited_rate") for r in rows]
            prohibited_rates = [value for value in prohibited_rates if isinstance(value, (int, float))]
            lines.append(
                "| {key} | {count} | {final_score} | {expected_coverage} | {prohibited_rate} |".format(
                    key=key,
                    count=len(rows),
                    final_score=f"{mean(final_scores):.2f}" if final_scores else "n/a",
                    expected_coverage=f"{mean(expected_coverages):.3f}" if expected_coverages else "n/a",
                    prohibited_rate=f"{mean(prohibited_rates):.3f}" if prohibited_rates else "n/a",
                )
            )

        rubric_light = [row for row in judgments if row.get("judgment", {}).get("no_structured_rubric")]
        if rubric_light:
            lines.extend(["", "## Rubric-Light Items", "", f"Count: {len(rubric_light)}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
