from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .usage_costs import build_usage_metrics


def _judgment_payload(row: dict[str, Any]) -> dict[str, Any]:
    judgment = row.get("judgment")
    return judgment if isinstance(judgment, dict) else {}


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


def _runner_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("display_name", "")), str(row.get("tier", "")))


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def _rubric_style(row: dict[str, Any], response_row: dict[str, Any] | None = None) -> str:
    judgment_style = _judgment_payload(row).get("rubric_style")
    if isinstance(judgment_style, str) and judgment_style:
        return judgment_style
    if response_row is not None:
        criteria = response_row.get("criteria")
        if isinstance(criteria, dict):
            criteria_style = criteria.get("rubric_style")
            if isinstance(criteria_style, str) and criteria_style:
                return criteria_style
            raw = criteria.get("raw")
            if isinstance(raw, dict):
                has_expected_section = any(key in raw for key in ("expected-criteria", "expected_criteria"))
                has_prohibited_section = any(key in raw for key in ("prohibited-criteria", "prohibited_criteria"))
                if has_expected_section or has_prohibited_section:
                    return "legacy_structured"
                if isinstance(raw.get("criteria"), list):
                    return "weighted_positive"
    if _judgment_payload(row).get("no_structured_rubric"):
        return "rubric_light"
    return "legacy_structured"


def _summary_style(judgments: list[dict[str, Any]], responses: list[dict[str, Any]]) -> str:
    response_index = {_row_identity(row): row for row in responses}
    styles = {
        _rubric_style(row, response_index.get(_row_identity(row)))
        for row in judgments
        if not row.get("judge_error")
    }
    if styles == {"weighted_positive"}:
        return "weighted_positive"
    return "legacy_structured"


def _build_answer_metrics(row: dict[str, Any], response_row: dict[str, Any] | None) -> dict[str, Any]:
    answer_input_tokens = row.get("answer_input_tokens")
    if answer_input_tokens is None and response_row is not None:
        raw_response = response_row.get("raw_response")
        if raw_response is not None:
            return build_usage_metrics(str(response_row.get("provider", "")), str(response_row.get("model_name", "")), raw_response)
    return {
        "input_tokens": row.get("answer_input_tokens"),
        "output_tokens": row.get("answer_output_tokens"),
        "total_tokens": row.get("answer_total_tokens"),
        "cached_input_tokens": row.get("answer_cached_input_tokens"),
        "cache_creation_input_tokens": row.get("answer_cache_creation_input_tokens"),
        "cache_creation_ephemeral_5m_input_tokens": row.get("answer_cache_creation_ephemeral_5m_input_tokens"),
        "cache_creation_ephemeral_1h_input_tokens": row.get("answer_cache_creation_ephemeral_1h_input_tokens"),
        "cache_read_input_tokens": row.get("answer_cache_read_input_tokens"),
        "estimated_cost_usd": row.get("answer_estimated_cost_usd"),
    }


def _build_judge_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": row.get("judge_input_tokens"),
        "output_tokens": row.get("judge_output_tokens"),
        "total_tokens": row.get("judge_total_tokens"),
        "cached_input_tokens": row.get("judge_cached_input_tokens"),
        "cache_creation_input_tokens": row.get("judge_cache_creation_input_tokens"),
        "cache_creation_ephemeral_5m_input_tokens": row.get("judge_cache_creation_ephemeral_5m_input_tokens"),
        "cache_creation_ephemeral_1h_input_tokens": row.get("judge_cache_creation_ephemeral_1h_input_tokens"),
        "cache_read_input_tokens": row.get("judge_cache_read_input_tokens"),
        "estimated_cost_usd": row.get("judge_estimated_cost_usd"),
    }


def _metric_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))


def write_summary_csv(path: Path, judgments: list[dict[str, Any]], responses: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    response_index = {_row_identity(row): row for row in responses}
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
        "rubric_style",
        "rubric_score",
        "expected_coverage",
        "prohibited_rate",
        "final_score",
        "average_holistic_score",
        "answer_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "answer_estimated_cost_usd",
        "judge_input_tokens",
        "judge_output_tokens",
        "judge_total_tokens",
        "judge_estimated_cost_usd",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_estimated_cost_usd",
        "summary",
        "generation_duration_seconds",
        "judge_duration_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in judgments:
            scores = _judgment_payload(row).get("scores", {})
            response_row = response_index.get(_row_identity(row))
            answer_metrics = _build_answer_metrics(row, response_row)
            judge_metrics = _build_judge_metrics(row)
            total_input_tokens = sum(
                value for value in [answer_metrics.get("input_tokens"), judge_metrics.get("input_tokens")] if isinstance(value, (int, float))
            )
            total_output_tokens = sum(
                value for value in [answer_metrics.get("output_tokens"), judge_metrics.get("output_tokens")] if isinstance(value, (int, float))
            )
            total_tokens = sum(
                value for value in [answer_metrics.get("total_tokens"), judge_metrics.get("total_tokens")] if isinstance(value, (int, float))
            )
            total_estimated_cost = sum(
                value for value in [answer_metrics.get("estimated_cost_usd"), judge_metrics.get("estimated_cost_usd")] if isinstance(value, (int, float))
            )
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
                    "rubric_style": _rubric_style(row, response_row),
                    "rubric_score": scores.get("rubric_score"),
                    "expected_coverage": scores.get("expected_coverage"),
                    "prohibited_rate": scores.get("prohibited_rate"),
                    "final_score": scores.get("final_score"),
                    "average_holistic_score": scores.get("average_holistic_score"),
                    "answer_input_tokens": answer_metrics.get("input_tokens"),
                    "answer_output_tokens": answer_metrics.get("output_tokens"),
                    "answer_total_tokens": answer_metrics.get("total_tokens"),
                    "answer_estimated_cost_usd": answer_metrics.get("estimated_cost_usd"),
                    "judge_input_tokens": judge_metrics.get("input_tokens"),
                    "judge_output_tokens": judge_metrics.get("output_tokens"),
                    "judge_total_tokens": judge_metrics.get("total_tokens"),
                    "judge_estimated_cost_usd": judge_metrics.get("estimated_cost_usd"),
                    "total_input_tokens": total_input_tokens or None,
                    "total_output_tokens": total_output_tokens or None,
                    "total_tokens": total_tokens or None,
                    "total_estimated_cost_usd": round(total_estimated_cost, 6) if total_estimated_cost else None,
                    "summary": _judgment_payload(row).get("summary", ""),
                    "generation_duration_seconds": row.get("generation_duration_seconds"),
                    "judge_duration_seconds": row.get("judge_duration_seconds"),
                }
            )


def write_summary_markdown(path: Path, judgments: list[dict[str, Any]], responses: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[_runner_key(row)].append(row)
    response_index = {_row_identity(row): row for row in responses}
    summary_style = _summary_style(judgments, responses)

    lines = ["# Evaluation Summary", ""]
    if not judgments:
        lines.append("No judgments were produced.")
    else:
        lines.append("## Aggregate")
        lines.append("")
        if summary_style == "weighted_positive":
            lines.append("| Runner | Tier | Items | Mean final score | Mean rubric score | Total input tok | Total output tok | Total tok | Est answer $ | Est judge $ | Est total $ | Mean answer s | Mean judge s |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        else:
            lines.append("| Runner | Tier | Items | Mean final score | Mean expected coverage | Mean prohibited rate | Total input tok | Total output tok | Total tok | Est answer $ | Est judge $ | Est total $ | Mean answer s | Mean judge s |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key in sorted(grouped):
            display_name, tier = key
            rows = grouped[key]
            generation_durations = [r.get("generation_duration_seconds") for r in rows]
            generation_durations = [value for value in generation_durations if isinstance(value, (int, float))]
            judge_durations = [r.get("judge_duration_seconds") for r in rows]
            judge_durations = [value for value in judge_durations if isinstance(value, (int, float))]
            final_scores = [_judgment_payload(r).get("scores", {}).get("final_score") for r in rows]
            final_scores = [value for value in final_scores if isinstance(value, (int, float))]
            expected_coverages = [_judgment_payload(r).get("scores", {}).get("expected_coverage") for r in rows]
            expected_coverages = [value for value in expected_coverages if isinstance(value, (int, float))]
            prohibited_rates = [_judgment_payload(r).get("scores", {}).get("prohibited_rate") for r in rows]
            prohibited_rates = [value for value in prohibited_rates if isinstance(value, (int, float))]
            rubric_scores = [_judgment_payload(r).get("scores", {}).get("rubric_score") for r in rows]
            rubric_scores = [value for value in rubric_scores if isinstance(value, (int, float))]
            answer_metrics = [_build_answer_metrics(row, response_index.get(_row_identity(row))) for row in rows]
            judge_metrics = [_build_judge_metrics(row) for row in rows]
            total_input_tokens = sum(
                value
                for metrics in answer_metrics + judge_metrics
                for value in [_metric_number(metrics.get("input_tokens"))]
                if value is not None
            )
            total_output_tokens = sum(
                value
                for metrics in answer_metrics + judge_metrics
                for value in [_metric_number(metrics.get("output_tokens"))]
                if value is not None
            )
            total_tokens = sum(
                value
                for metrics in answer_metrics + judge_metrics
                for value in [_metric_number(metrics.get("total_tokens"))]
                if value is not None
            )
            answer_cost = sum(value for metrics in answer_metrics for value in [_metric_number(metrics.get("estimated_cost_usd"))] if value is not None)
            judge_cost = sum(value for metrics in judge_metrics for value in [_metric_number(metrics.get("estimated_cost_usd"))] if value is not None)
            if summary_style == "weighted_positive":
                lines.append(
                    "| {display_name} | {tier} | {count} | {final_score} | {rubric_score} | {total_input_tokens} | {total_output_tokens} | {total_tokens} | {answer_cost} | {judge_cost} | {total_cost} | {generation_duration} | {judge_duration} |".format(
                        display_name=display_name,
                        tier=tier,
                        count=len(rows),
                        final_score=f"{mean(final_scores):.2f}" if final_scores else "n/a",
                        rubric_score=f"{mean(rubric_scores):.3f}" if rubric_scores else "n/a",
                        total_input_tokens=f"{int(total_input_tokens):,}" if total_input_tokens else "n/a",
                        total_output_tokens=f"{int(total_output_tokens):,}" if total_output_tokens else "n/a",
                        total_tokens=f"{int(total_tokens):,}" if total_tokens else "n/a",
                        answer_cost=f"{answer_cost:.4f}" if answer_cost else "n/a",
                        judge_cost=f"{judge_cost:.4f}" if judge_cost else "n/a",
                        total_cost=f"{answer_cost + judge_cost:.4f}" if (answer_cost or judge_cost) else "n/a",
                        generation_duration=f"{mean(generation_durations):.2f}" if generation_durations else "n/a",
                        judge_duration=f"{mean(judge_durations):.2f}" if judge_durations else "n/a",
                    )
                )
            else:
                lines.append(
                    "| {display_name} | {tier} | {count} | {final_score} | {expected_coverage} | {prohibited_rate} | {total_input_tokens} | {total_output_tokens} | {total_tokens} | {answer_cost} | {judge_cost} | {total_cost} | {generation_duration} | {judge_duration} |".format(
                        display_name=display_name,
                        tier=tier,
                        count=len(rows),
                        final_score=f"{mean(final_scores):.2f}" if final_scores else "n/a",
                        expected_coverage=f"{mean(expected_coverages):.3f}" if expected_coverages else "n/a",
                        prohibited_rate=f"{mean(prohibited_rates):.3f}" if prohibited_rates else "n/a",
                        total_input_tokens=f"{int(total_input_tokens):,}" if total_input_tokens else "n/a",
                        total_output_tokens=f"{int(total_output_tokens):,}" if total_output_tokens else "n/a",
                        total_tokens=f"{int(total_tokens):,}" if total_tokens else "n/a",
                        answer_cost=f"{answer_cost:.4f}" if answer_cost else "n/a",
                        judge_cost=f"{judge_cost:.4f}" if judge_cost else "n/a",
                        total_cost=f"{answer_cost + judge_cost:.4f}" if (answer_cost or judge_cost) else "n/a",
                        generation_duration=f"{mean(generation_durations):.2f}" if generation_durations else "n/a",
                        judge_duration=f"{mean(judge_durations):.2f}" if judge_durations else "n/a",
                    )
                )

        if summary_style == "weighted_positive":
            lines.extend(
                [
                    "",
                    "## Formulas",
                    "",
                    "Benchling weighted-rubric strategy:",
                    "Each question provides only positive rubric criteria with raw weights. The harness normalizes those weights within the question so the weighted rubric score is always on a 0 to 1 scale.",
                    "",
                    "Per-criterion score:",
                    "$$",
                    "s(c) = \\begin{cases}",
                    "1.0 & \\text{if met} \\\\",
                    "0.5 & \\text{if partial} \\\\",
                    "0.0 & \\text{if missed} \\\\",
                    "0.25 & \\text{if unclear}",
                    "\\end{cases}",
                    "$$",
                    "",
                    "Weight normalization:",
                    "$$",
                    "w(c) = \\frac{w_{raw}(c)}{\\sum_{c' \\in C} w_{raw}(c')}",
                    "$$",
                    "$$",
                    "\\sum_{c \\in C} w(c) = 1",
                    "$$",
                    "",
                    "Item-level rubric score:",
                    "$$",
                    "R = \\sum_{c \\in C} w(c) \\cdot s(c)",
                    "$$",
                    "$$",
                    "\\text{rubric score} = R",
                    "$$",
                    "",
                    "Final score:",
                    "$$",
                    "\\text{final score} = 100 \\cdot R",
                    "$$",
                    "No prohibited-criteria adjustment or cap is applied for this rubric style.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## Formulas",
                    "",
                    "Per-criterion expected-item score:",
                    "$$",
                    "s_{expected}(c) = \\begin{cases}",
                    "1.0 & \\text{if met} \\\\",
                    "0.5 & \\text{if partial} \\\\",
                    "0.0 & \\text{if missed} \\\\",
                    "0.25 & \\text{if unclear}",
                    "\\end{cases}",
                    "$$",
                    "",
                    "Per-criterion prohibited-item score:",
                    "$$",
                    "s_{prohibited}(c) = \\begin{cases}",
                    "0.0 & \\text{if not violated} \\\\",
                    "1.0 & \\text{if violated} \\\\",
                    "0.5 & \\text{if unclear}",
                    "\\end{cases}",
                    "$$",
                    "",
                    "Item-level aggregates:",
                    "$$",
                    "w(c) = \\frac{w_{raw}(c)}{\\sum_{c' \\in C_{section}} w_{raw}(c')}",
                    "$$",
                    "$$",
                    "E = \\sum_{c \\in C_{expected}} w(c) \\cdot s_{expected}(c)",
                    "$$",
                    "$$",
                    "\\sum_{c \\in C_{expected}} w(c) = 1",
                    "$$",
                    "$$",
                    "\\text{expected coverage} = E",
                    "$$",
                    "$$",
                    "P = \\sum_{c \\in C_{prohibited}} w(c) \\cdot s_{prohibited}(c)",
                    "$$",
                    "$$",
                    "\\sum_{c \\in C_{prohibited}} w(c) = 1",
                    "$$",
                    "$$",
                    "\\text{prohibited rate} = P",
                    "$$",
                    "$$",
                    "\\text{if } C_{prohibited} = \\varnothing, \\text{ then prohibited rate} = 0",
                    "$$",
                    "$$",
                    "\\text{prohibited compliance} = 1 - \\text{prohibited rate}",
                    "$$",
                    "",
                    "Final score:",
                    "$$",
                    "\\text{final score} = 100 \\cdot \\left(0.8 \\cdot \\text{expected coverage} + 0.2 \\cdot \\text{prohibited compliance}\\right)",
                    "$$",
                    "For datasets with prohibited criteria, if $P > 0$, then the final score is capped at $74.0$.",
                ]
            )

        run_answer_input_tokens = 0.0
        run_answer_output_tokens = 0.0
        run_answer_total_tokens = 0.0
        run_answer_cost = 0.0
        run_judge_input_tokens = 0.0
        run_judge_output_tokens = 0.0
        run_judge_total_tokens = 0.0
        run_judge_cost = 0.0
        for row in judgments:
            answer_metrics = _build_answer_metrics(row, response_index.get(_row_identity(row)))
            judge_metrics = _build_judge_metrics(row)
            run_answer_input_tokens += _metric_number(answer_metrics.get("input_tokens")) or 0.0
            run_answer_output_tokens += _metric_number(answer_metrics.get("output_tokens")) or 0.0
            run_answer_total_tokens += _metric_number(answer_metrics.get("total_tokens")) or 0.0
            run_answer_cost += _metric_number(answer_metrics.get("estimated_cost_usd")) or 0.0
            run_judge_input_tokens += _metric_number(judge_metrics.get("input_tokens")) or 0.0
            run_judge_output_tokens += _metric_number(judge_metrics.get("output_tokens")) or 0.0
            run_judge_total_tokens += _metric_number(judge_metrics.get("total_tokens")) or 0.0
            run_judge_cost += _metric_number(judge_metrics.get("estimated_cost_usd")) or 0.0

        lines.extend(
            [
                "",
                "## Cost Totals",
                "",
                "| Scope | Total input tok | Total output tok | Total tok | Estimated cost $ |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Answers | {int(run_answer_input_tokens):,} | {int(run_answer_output_tokens):,} | {int(run_answer_total_tokens):,} | {run_answer_cost:.4f} |",
                f"| Judging | {int(run_judge_input_tokens):,} | {int(run_judge_output_tokens):,} | {int(run_judge_total_tokens):,} | {run_judge_cost:.4f} |",
                f"| Combined | {int(run_answer_input_tokens + run_judge_input_tokens):,} | {int(run_answer_output_tokens + run_judge_output_tokens):,} | {int(run_answer_total_tokens + run_judge_total_tokens):,} | {run_answer_cost + run_judge_cost:.4f} |",
            ]
        )

        item_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in judgments:
            if isinstance(row.get("dataset_index"), int):
                item_groups[row["dataset_index"]].append(row)

        lines.extend(["", "## Timing By Item", "", "| Dataset index | Answer time total s | Judge time total s | Completed runners |", "| --- | ---: | ---: | ---: |"])
        for dataset_index in sorted(item_groups):
            rows = item_groups[dataset_index]
            generation_total = sum(
                value for value in (row.get("generation_duration_seconds") for row in rows) if isinstance(value, (int, float))
            )
            judge_total = sum(
                value for value in (row.get("judge_duration_seconds") for row in rows) if isinstance(value, (int, float))
            )
            completed_runners = sum(1 for row in rows if not row.get("response_error"))
            lines.append(
                f"| {dataset_index} | {generation_total:.2f} | {judge_total:.2f} | {completed_runners} |"
            )

        rubric_light = [row for row in judgments if _judgment_payload(row).get("no_structured_rubric")]
        if rubric_light:
            lines.extend(["", "## Rubric-Light Items", "", f"Count: {len(rubric_light)}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
