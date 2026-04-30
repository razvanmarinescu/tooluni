from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .judge import EXPECTED_WEIGHT, PROHIBITED_WEIGHT
from .usage_costs import build_usage_metrics


def _recompute_final_score(scores: dict[str, Any]) -> None:
    """Recompute `final_score` in place from stored coverage/rate using the
    currently-configured EXPECTED_WEIGHT / PROHIBITED_WEIGHT. No cap.

    Safe to call repeatedly and on rows from older runs: the previously stored
    final_score is simply overwritten from the components we already have.
    """
    expected_coverage = scores.get("expected_coverage")
    prohibited_rate = scores.get("prohibited_rate")
    if not isinstance(expected_coverage, (int, float)):
        return
    has_prohibited = isinstance(scores.get("prohibited_max"), (int, float)) and scores.get("prohibited_max") > 0
    if has_prohibited and isinstance(prohibited_rate, (int, float)):
        compliance = 1.0 - prohibited_rate
    else:
        compliance = 1.0
    scores["final_score"] = 100.0 * (EXPECTED_WEIGHT * expected_coverage + PROHIBITED_WEIGHT * compliance)


def _judgment_payload(row: dict[str, Any]) -> dict[str, Any]:
    judgment = row.get("judgment")
    if not isinstance(judgment, dict):
        return {}
    scores = judgment.get("scores")
    if isinstance(scores, dict):
        _recompute_final_score(scores)
    return judgment


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


def _per_question_macro_values(rows: list[dict[str, Any]], score_key: str) -> list[float]:
    """Group rows by dataset_index, take the mean of the given score within
    each question, and return the list of per-question means. Used for
    macro-averaging in the aggregate table.
    """
    by_index: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        value = _judgment_payload(row).get("scores", {}).get(score_key)
        if isinstance(value, (int, float)):
            by_index[row.get("dataset_index")].append(float(value))
    return [mean(values) for values in by_index.values() if values]


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def _summary_csv_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    dataset_index = row.get("dataset_index")
    dataset_sort = dataset_index if isinstance(dataset_index, int) else float("inf")
    return (
        dataset_sort,
        str(row.get("tier") or ""),
        str(row.get("display_name") or ""),
        str(row.get("provider") or ""),
        str(row.get("model_name") or ""),
    )


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
        for row in sorted(judgments, key=_summary_csv_sort_key):
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
        lines.append(
            "Means are **macro-averaged**: each runner's score on a question is the "
            "mean across all runs of that question, and the reported mean is the average "
            "of those per-question means. When a runner has a single run per question this "
            "equals the plain mean."
        )
        lines.append("")
        if summary_style == "weighted_positive":
            lines.append("| Runner | Tier | Questions | Runs | Mean final score | Mean rubric score | Total input tok | Total output tok | Total tok | Est answer $ | Est judge $ | Est total $ | Mean answer s | Mean judge s |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        else:
            lines.append("| Runner | Tier | Questions | Runs | Mean final score | Mean expected coverage | Mean prohibited rate | Total input tok | Total output tok | Total tok | Est answer $ | Est judge $ | Est total $ | Mean answer s | Mean judge s |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key in sorted(grouped):
            display_name, tier = key
            rows = grouped[key]
            generation_durations = [r.get("generation_duration_seconds") for r in rows]
            generation_durations = [value for value in generation_durations if isinstance(value, (int, float))]
            judge_durations = [r.get("judge_duration_seconds") for r in rows]
            judge_durations = [value for value in judge_durations if isinstance(value, (int, float))]
            # Macro-average: first average each metric within a dataset_index
            # (across that question's runs), then average the per-question means.
            # When every question has a single run this degenerates to the plain
            # mean, so behavior is unchanged for the common case.
            final_scores = _per_question_macro_values(rows, "final_score")
            expected_coverages = _per_question_macro_values(rows, "expected_coverage")
            prohibited_rates = _per_question_macro_values(rows, "prohibited_rate")
            rubric_scores = _per_question_macro_values(rows, "rubric_score")
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
            n_questions = len({r.get("dataset_index") for r in rows if r.get("dataset_index") is not None})
            if summary_style == "weighted_positive":
                lines.append(
                    "| {display_name} | {tier} | {questions} | {runs} | {final_score} | {rubric_score} | {total_input_tokens} | {total_output_tokens} | {total_tokens} | {answer_cost} | {judge_cost} | {total_cost} | {generation_duration} | {judge_duration} |".format(
                        display_name=display_name,
                        tier=tier,
                        questions=n_questions,
                        runs=len(rows),
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
                    "| {display_name} | {tier} | {questions} | {runs} | {final_score} | {expected_coverage} | {prohibited_rate} | {total_input_tokens} | {total_output_tokens} | {total_tokens} | {answer_cost} | {judge_cost} | {total_cost} | {generation_duration} | {judge_duration} |".format(
                        display_name=display_name,
                        tier=tier,
                        questions=n_questions,
                        runs=len(rows),
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
                    f"\\text{{final score}} = 100 \\cdot \\left({EXPECTED_WEIGHT:.1f} \\cdot \\text{{expected coverage}} + {PROHIBITED_WEIGHT:.1f} \\cdot \\text{{prohibited compliance}}\\right)",
                    "$$",
                    f"Weights: expected coverage ({EXPECTED_WEIGHT:.1f}) / prohibited compliance ({PROHIBITED_WEIGHT:.1f}). "
                    "No score cap is applied — prohibited violations are expressed entirely through the weight.",
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

        lines.extend(["", "## Rubric Scores By Question", ""])
        lines.append(
            "One sub-table per dataset question. Each sub-table lists every "
            "run that was scored for that question and ends with a **Mean** "
            "row (average across the runs in that question group)."
        )
        lines.append("")
        response_by_key = {_row_identity(r): r for r in responses}

        def _fmt(value: Any, decimals: int) -> str:
            return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "n/a"

        for dataset_index in sorted(item_groups):
            item_rows = sorted(item_groups[dataset_index], key=_summary_csv_sort_key)
            question = ""
            for row in item_rows:
                response_row = response_by_key.get(_row_identity(row))
                if response_row and response_row.get("question"):
                    question = str(response_row.get("question")).strip()
                    break
                if row.get("question"):
                    question = str(row.get("question")).strip()
                    break
            preview = question.replace("\n", " ").replace("|", "\\|")
            if len(preview) > 140:
                preview = preview[:137] + "..."

            lines.append(f"### Question {dataset_index}")
            if preview:
                lines.append("")
                lines.append(f"> {preview}")
            lines.append("")
            if summary_style == "weighted_positive":
                lines.append("| Runner | Model | Tier | Final score | Rubric score |")
                lines.append("| --- | --- | --- | ---: | ---: |")
                finals: list[float] = []
                rubric_vals: list[float] = []
                for row in item_rows:
                    scores = _judgment_payload(row).get("scores", {})
                    final_score = scores.get("final_score")
                    rubric_score = scores.get("rubric_score")
                    if isinstance(final_score, (int, float)):
                        finals.append(float(final_score))
                    if isinstance(rubric_score, (int, float)):
                        rubric_vals.append(float(rubric_score))
                    lines.append(
                        f"| {row.get('display_name') or ''} | {row.get('model_name') or ''} | "
                        f"{row.get('tier') or ''} | {_fmt(final_score, 2)} | {_fmt(rubric_score, 3)} |"
                    )
                lines.append(
                    "| **Mean** | | | **{final}** | **{rubric}** |".format(
                        final=_fmt(mean(finals) if finals else None, 2),
                        rubric=_fmt(mean(rubric_vals) if rubric_vals else None, 3),
                    )
                )
            else:
                lines.append("| Runner | Model | Tier | Final score | Expected coverage | Prohibited rate |")
                lines.append("| --- | --- | --- | ---: | ---: | ---: |")
                finals = []
                coverages: list[float] = []
                prohibiteds: list[float] = []
                for row in item_rows:
                    scores = _judgment_payload(row).get("scores", {})
                    final_score = scores.get("final_score")
                    expected_coverage = scores.get("expected_coverage")
                    prohibited_rate = scores.get("prohibited_rate")
                    if isinstance(final_score, (int, float)):
                        finals.append(float(final_score))
                    if isinstance(expected_coverage, (int, float)):
                        coverages.append(float(expected_coverage))
                    if isinstance(prohibited_rate, (int, float)):
                        prohibiteds.append(float(prohibited_rate))
                    lines.append(
                        f"| {row.get('display_name') or ''} | {row.get('model_name') or ''} | "
                        f"{row.get('tier') or ''} | {_fmt(final_score, 2)} | "
                        f"{_fmt(expected_coverage, 3)} | {_fmt(prohibited_rate, 3)} |"
                    )
                lines.append(
                    "| **Mean** | | | **{final}** | **{coverage}** | **{prohibited}** |".format(
                        final=_fmt(mean(finals) if finals else None, 2),
                        coverage=_fmt(mean(coverages) if coverages else None, 3),
                        prohibited=_fmt(mean(prohibiteds) if prohibiteds else None, 3),
                    )
                )
            lines.append("")

        rubric_light = [row for row in judgments if _judgment_payload(row).get("no_structured_rubric")]
        if rubric_light:
            lines.extend(["", "## Rubric-Light Items", "", f"Count: {len(rubric_light)}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
