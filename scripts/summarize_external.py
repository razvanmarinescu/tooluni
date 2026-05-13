#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Build aggregate + per-question summary tables for the externally-sourced
Biomni and Potato judgment runs.

Reads:
  runs/biomni/judgments.jsonl
  runs/potato/judgments.jsonl
  genetic_benchmark_v1/47-submissions-clean.json  (for question previews)

Writes:
  runs/external_summary.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.dataset import load_items  # noqa: E402
from lib.judge import EXPECTED_WEIGHT, PROHIBITED_WEIGHT  # noqa: E402
from lib.reporting import _recompute_final_score, load_jsonl  # noqa: E402


DATASET = PROJECT_ROOT / "genetic_benchmark_v1" / "47-submissions-clean.json"
BIOMNI_JUDGMENTS = PROJECT_ROOT / "runs" / "biomni" / "judgments.jsonl"
POTATO_JUDGMENTS = PROJECT_ROOT / "runs" / "potato" / "judgments.jsonl"
OUTPUT_MD = PROJECT_ROOT / "runs" / "external_summary.md"


def _scores(row: dict[str, Any]) -> dict[str, Any]:
    judgment = row.get("judgment") or {}
    if not isinstance(judgment, dict):
        return {}
    scores = judgment.get("scores", {})
    if isinstance(scores, dict):
        # Recompute final_score with the currently-configured weights so this
        # summary stays consistent with runs/{biomni,potato}/summary.md.
        _recompute_final_score(scores)
    return scores


def _numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def per_question_means(rows: list[dict[str, Any]]) -> dict[int, dict[str, float | None]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        idx = row.get("dataset_index")
        if isinstance(idx, int):
            grouped[idx].append(row)

    out: dict[int, dict[str, float | None]] = {}
    for idx, row_list in grouped.items():
        finals = [_numeric(_scores(r).get("final_score")) for r in row_list]
        finals = [v for v in finals if v is not None]
        coverages = [_numeric(_scores(r).get("expected_coverage")) for r in row_list]
        coverages = [v for v in coverages if v is not None]
        prohibs = [_numeric(_scores(r).get("prohibited_rate")) for r in row_list]
        prohibs = [v for v in prohibs if v is not None]
        out[idx] = {
            "n_runs": len(row_list),
            "final_score": mean(finals) if finals else None,
            "expected_coverage": mean(coverages) if coverages else None,
            "prohibited_rate": mean(prohibs) if prohibs else None,
        }
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    # Macro-average across questions, matching write_summary_markdown in
    # lib/reporting.py. When every question has 1 run this is equivalent to
    # the plain mean.
    per_q = per_question_means(rows)
    finals = [v["final_score"] for v in per_q.values() if isinstance(v["final_score"], (int, float))]
    coverages = [v["expected_coverage"] for v in per_q.values() if isinstance(v["expected_coverage"], (int, float))]
    prohibs = [v["prohibited_rate"] for v in per_q.values() if isinstance(v["prohibited_rate"], (int, float))]
    return {
        "n_runs": len(rows),
        "n_questions": len({r.get("dataset_index") for r in rows if isinstance(r.get("dataset_index"), int)}),
        "n_judged": len(finals),
        "final_score": mean(finals) if finals else None,
        "expected_coverage": mean(coverages) if coverages else None,
        "prohibited_rate": mean(prohibs) if prohibs else None,
    }


def fmt(value: float | None, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "n/a"


def main() -> None:
    biomni_rows = [r for r in load_jsonl(BIOMNI_JUDGMENTS) if not r.get("judge_error")]
    potato_rows = [r for r in load_jsonl(POTATO_JUDGMENTS) if not r.get("judge_error")]
    items = load_items(DATASET)
    prompts = {i: (item.get("prompt") or "").strip() for i, item in enumerate(items, start=1)}

    biomni_agg = aggregate(biomni_rows)
    potato_agg = aggregate(potato_rows)
    biomni_per_q = per_question_means(biomni_rows)
    potato_per_q = per_question_means(potato_rows)

    lines: list[str] = []
    lines.append("# External evaluation: Biomni vs Potato")
    lines.append("")
    lines.append("Judge: gpt-5.4 (pinned, see `scripts/judge_eval.py`).")
    lines.append("Rubric: legacy structured (expected + prohibited criteria from `47-submissions-clean.json`).")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| System | Questions | Runs | Mean final score | Mean expected coverage | Mean prohibited rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| Biomni | {biomni_agg['n_questions']} | {biomni_agg['n_runs']} | "
        f"{fmt(biomni_agg['final_score'])} | {fmt(biomni_agg['expected_coverage'], 3)} | "
        f"{fmt(biomni_agg['prohibited_rate'], 3)} |"
    )
    lines.append(
        f"| Potato | {potato_agg['n_questions']} | {potato_agg['n_runs']} | "
        f"{fmt(potato_agg['final_score'])} | {fmt(potato_agg['expected_coverage'], 3)} | "
        f"{fmt(potato_agg['prohibited_rate'], 3)} |"
    )
    lines.append("")
    lines.append(
        f"`final_score = 100 * ({EXPECTED_WEIGHT} * expected_coverage + "
        f"{PROHIBITED_WEIGHT} * (1 - prohibited_rate))`. No score cap."
    )
    lines.append("")

    # --- Per-question side-by-side ---
    all_indices = sorted(set(biomni_per_q) | set(potato_per_q))

    lines.append("## Per-question scores")
    lines.append("")
    lines.append(
        "| # | Question | Biomni runs | Biomni final | Biomni cov | Biomni pro | "
        "Potato runs | Potato final | Potato cov | Potato pro |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx in all_indices:
        b = biomni_per_q.get(idx)
        p = potato_per_q.get(idx)
        question = prompts.get(idx, "")
        preview = question.replace("\n", " ").replace("|", "\\|")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        row = [
            str(idx),
            preview,
            str(b["n_runs"]) if b else "-",
            fmt(b["final_score"]) if b else "-",
            fmt(b["expected_coverage"], 3) if b else "-",
            fmt(b["prohibited_rate"], 3) if b else "-",
            str(p["n_runs"]) if p else "-",
            fmt(p["final_score"]) if p else "-",
            fmt(p["expected_coverage"], 3) if p else "-",
            fmt(p["prohibited_rate"], 3) if p else "-",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # --- Biomni-only per-question (full 47) ---
    lines.append("")
    lines.append("## Biomni per-question (full list)")
    lines.append("")
    lines.append("| # | Question | Final score | Expected coverage | Prohibited rate |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for idx in sorted(biomni_per_q):
        b = biomni_per_q[idx]
        question = prompts.get(idx, "")
        preview = question.replace("\n", " ").replace("|", "\\|")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        lines.append(
            f"| {idx} | {preview} | {fmt(b['final_score'])} | "
            f"{fmt(b['expected_coverage'], 3)} | {fmt(b['prohibited_rate'], 3)} |"
        )

    # --- Potato per-run breakdown (so run variability is visible) ---
    lines.append("")
    lines.append("## Potato per-question, per-run")
    lines.append("")
    lines.append("| # | Model (run) | Final | Coverage | Prohibited |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    potato_by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in potato_rows:
        if isinstance(row.get("dataset_index"), int):
            potato_by_idx[row["dataset_index"]].append(row)
    for idx in sorted(potato_by_idx):
        for row in sorted(potato_by_idx[idx], key=lambda r: str(r.get("model_name"))):
            scores = _scores(row)
            lines.append(
                f"| {idx} | {row.get('model_name')} | "
                f"{fmt(_numeric(scores.get('final_score')))} | "
                f"{fmt(_numeric(scores.get('expected_coverage')), 3)} | "
                f"{fmt(_numeric(scores.get('prohibited_rate')), 3)} |"
            )

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
