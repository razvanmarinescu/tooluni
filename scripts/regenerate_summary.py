#!/Users/razvan/research/evals/tooluni/.venv/bin/python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from lib.reporting import load_jsonl, write_summary_csv, write_summary_markdown


# `uniform` scoring: every criterion (expected + prohibited) carries the
# same weight; the score is 100 × (satisfied + 0.5·unclear) / total. We map
# our per-criterion vocabulary onto a single satisfaction axis below.
UNIFORM_EXPECTED_SAT = {"met": 1.0, "partial": 0.5, "unclear": 0.5, "missed": 0.0}
UNIFORM_PROHIBITED_SAT = {"not_violated": 1.0, "unclear": 0.5, "violated": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate summary files for an existing run.")
    parser.add_argument("--run-id", required=True, help="Run directory name under runs/, e.g. 00013")
    parser.add_argument(
        "--drop-indices",
        default="",
        help="Comma-separated dataset_index values to exclude from the summary "
        "(applied to both responses and judgments). Example: --drop-indices 20,29.",
    )
    parser.add_argument(
        "--drop-note",
        default="",
        help="Optional explanation appended to summary.md describing why "
        "questions were dropped.",
    )
    parser.add_argument(
        "--score-method",
        choices=["80-20", "uniform"],
        default="80-20",
        help="Final-score formula. '80-20' (default) = our coverage-vs-compliance weighted "
        "blend (80% expected coverage, 20% prohibited compliance). 'uniform' = Diego-style "
        "100·(satisfied + 0.5·unclear)/N treating every criterion (expected + prohibited) "
        "equally. Per-criterion statuses are NOT re-judged, only re-aggregated; safe to run "
        "repeatedly with different settings.",
    )
    parser.add_argument(
        "--output-stem",
        default="summary",
        help="Basename for the generated files (default: 'summary' → summary.csv + summary.md). "
        "Use this to write side-by-side variants — e.g. --output-stem summary_uniform.",
    )
    return parser.parse_args()


def _recompute_uniform_score(judgment: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the judgment with scores.final_score replaced by the
    uniform-weight formula. Other score fields are left intact for
    transparency (expected_coverage and prohibited_rate retain their
    original 80/20 semantics so callers can audit the difference)."""
    if not isinstance(judgment, dict):
        return judgment
    expected = judgment.get("expected") or []
    prohibited = judgment.get("prohibited") or []
    total = len(expected) + len(prohibited)
    if total == 0:
        return judgment
    sat_points = 0.0
    for c in expected:
        sat_points += UNIFORM_EXPECTED_SAT.get(str(c.get("status") or ""), 0.0)
    for c in prohibited:
        sat_points += UNIFORM_PROHIBITED_SAT.get(str(c.get("status") or ""), 0.0)
    new = copy.deepcopy(judgment)
    scores = new.setdefault("scores", {})
    scores["final_score"] = 100.0 * sat_points / total
    scores["uniform_satisfied_points"] = sat_points
    scores["uniform_total_criteria"] = total
    # Marker read by lib.reporting._recompute_final_score so the writer
    # preserves our uniform-style final_score instead of overwriting it.
    scores["_score_method"] = "uniform"
    return new


def _apply_score_method(judgments: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method != "uniform":
        return judgments
    out = []
    for j in judgments:
        if j.get("judge_error") or not j.get("judgment"):
            out.append(j)
            continue
        new_j = dict(j)
        new_j["judgment"] = _recompute_uniform_score(j["judgment"])
        out.append(new_j)
    return out


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / args.run_id
    responses_path = run_dir / "responses.jsonl"
    judgments_path = run_dir / "judgments.jsonl"

    responses = load_jsonl(responses_path)
    judgments = load_jsonl(judgments_path)

    drop = {int(x) for x in args.drop_indices.split(",") if x.strip()}
    if drop:
        before_r, before_j = len(responses), len(judgments)
        responses = [r for r in responses if r.get("dataset_index") not in drop]
        judgments = [j for j in judgments if j.get("dataset_index") not in drop]
        print(
            f"Dropped indices {sorted(drop)}: "
            f"responses {before_r}->{len(responses)}, "
            f"judgments {before_j}->{len(judgments)}"
        )

    judgments_for_summary = _apply_score_method(judgments, args.score_method)
    if args.score_method == "uniform":
        print(
            "Using uniform scoring: final = 100·(satisfied + 0.5·unclear)/N across "
            "expected and prohibited criteria. expected_coverage and prohibited_rate columns "
            "retain their original 80/20 semantics."
        )

    stem = args.output_stem
    summary_md_path = run_dir / f"{stem}.md"
    summary_csv_path = run_dir / f"{stem}.csv"
    write_summary_csv(summary_csv_path, judgments_for_summary, responses)
    write_summary_markdown(summary_md_path, judgments_for_summary, responses)

    note_lines: list[str] = []
    if drop:
        note_lines.extend([
            "## Excluded questions",
            "",
            f"Dropped from this summary: **q{', q'.join(str(i) for i in sorted(drop))}** "
            f"({len(drop)} question{'s' if len(drop) != 1 else ''}).",
            "",
        ])
        if args.drop_note:
            note_lines.append(args.drop_note)
            note_lines.append("")
    note_lines.append("## Scoring method")
    note_lines.append("")
    note_lines.append(f"Scoring method: **`{args.score_method}`**")
    note_lines.append("")
    if args.score_method == "uniform":
        note_lines.extend([
            "Every rubric criterion — expected *and* prohibited — is treated as one item with "
            "equal weight. Each criterion contributes a satisfaction value in `{0.0, 0.5, 1.0}` "
            "based on its aggregated status:",
            "",
            "| Section | Status | Satisfaction |",
            "|---|---|---:|",
            "| expected | `met` | 1.0 |",
            "| expected | `partial` (split vote) | 0.5 |",
            "| expected | `unclear` (single-judge \"don't know\") | 0.5 |",
            "| expected | `missed` | 0.0 |",
            "| prohibited | `not_violated` | 1.0 |",
            "| prohibited | `unclear` (split vote) | 0.5 |",
            "| prohibited | `violated` | 0.0 |",
            "",
            "Final score:",
            "",
            "```",
            "final = 100 × (sum of satisfaction across all criteria) / N",
            "       where N = |expected_items| + |prohibited_items|",
            "```",
            "",
            "Implications: a rubric with 3 prohibited items and 17 expected items weights "
            "prohibited compliance at 3/20 = 15% of the final score. Both criterion sections "
            "contribute proportionally to their count, not by a fixed mix.",
        ])
    else:
        note_lines.extend([
            "Expected and prohibited criteria are aggregated separately, then mixed 80/20.",
            "",
            "**Per-criterion score maps** (with `weight` from the rubric, normalized within "
            "each section so the section weights sum to 1):",
            "",
            "| Section | Status | Score |",
            "|---|---|---:|",
            "| expected | `met` | 1.0 |",
            "| expected | `partial` | 0.5 |",
            "| expected | `unclear` | 0.25 |",
            "| expected | `missed` | 0.0 |",
            "| prohibited | `not_violated` | 0.0 |",
            "| prohibited | `unclear` | 0.5 |",
            "| prohibited | `violated` | 1.0 |",
            "",
            "**Section aggregates:**",
            "",
            "```",
            "expected_coverage  = Σ_c  w(c) · score_expected(c)        # 0–1, higher = better",
            "prohibited_rate    = Σ_c  w(c) · score_prohibited(c)      # 0–1, lower  = better",
            "prohibited_compliance = 1 − prohibited_rate",
            "```",
            "",
            "**Final score:**",
            "",
            "```",
            "final = 100 × ( 0.8 · expected_coverage + 0.2 · prohibited_compliance )",
            "```",
            "",
            "Implications: prohibited compliance is capped at a fixed 20% weight regardless of "
            "how many prohibited items the rubric has. A rubric with 1 prohibited item and 20 "
            "expected items contributes the same 20% from prohibited compliance as a rubric with "
            "5 prohibited items and 10 expected items.",
        ])
    note_lines.append("")
    if note_lines:
        existing = summary_md_path.read_text(encoding="utf-8")
        # Insert all preamble notes right after the H1 title, before the aggregate table.
        marker = "## Aggregate"
        if marker in existing:
            head, tail = existing.split(marker, 1)
            summary_md_path.write_text(
                head + "\n".join(note_lines) + "\n" + marker + tail,
                encoding="utf-8",
            )
        else:
            summary_md_path.write_text(
                existing + "\n\n" + "\n".join(note_lines), encoding="utf-8"
            )

    print(f"Regenerated summary for {run_dir} (score_method={args.score_method}, files: {stem}.csv, {stem}.md)")


if __name__ == "__main__":
    main()