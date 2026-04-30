#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Judge a responses.jsonl file using the pinned gpt-5.4 judge.

Behaves like scripts/judge_eval.py but:
  * skips rows already present in judgments.jsonl (resume on restart)
  * catches provider-side refusals (HTTP 400 safety filter) and records a
    judge_error entry instead of crashing the whole run
  * writes summary.csv / summary.md at the end
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.judge import Judge  # noqa: E402
from lib.reporting import append_jsonl, load_jsonl, write_summary_csv, write_summary_markdown  # noqa: E402
from lib.usage_costs import combine_usage_metrics  # noqa: E402


FIXED_JUDGE_MODEL = "gpt-5.4"


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def _base_judgment_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_index": row.get("dataset_index"),
        "submission_id": row.get("submission_id"),
        "provider": row.get("provider"),
        "model_name": row.get("model_name"),
        "display_name": row.get("display_name"),
        "tier": row.get("tier"),
        "rubric_style": row.get("rubric_style") or (row.get("criteria") or {}).get("rubric_style"),
        "has_structured_rubric": row.get("has_structured_rubric"),
        "question": row.get("question"),
        "response_error": row.get("response_error"),
    }


def _answer_metrics(row: dict[str, Any]) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge responses.jsonl into judgments.jsonl, tolerant of provider-side refusals.")
    parser.add_argument("responses", type=Path)
    parser.add_argument("judgments", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    judge = Judge(root, model_name=FIXED_JUDGE_MODEL)

    responses = load_jsonl(args.responses)
    existing = {_identity(row) for row in load_jsonl(args.judgments)}
    remaining = [row for row in responses if _identity(row) not in existing]
    print(f"responses: {len(responses)}  already judged: {len(existing)}  to judge: {len(remaining)}")

    for i, row in enumerate(remaining, start=1):
        base = _base_judgment_fields(row)
        if row.get("response_error") or not row.get("response_text"):
            append_jsonl(
                args.judgments,
                {
                    **base,
                    "judge_error": "Skipped because response generation failed.",
                    "judgment": None,
                    **combine_usage_metrics("answer", _answer_metrics(row)),
                    **combine_usage_metrics("judge", None),
                    "judge_model_name": FIXED_JUDGE_MODEL,
                },
            )
            continue

        try:
            judgment = judge.judge_response(row["question"], row["criteria"], row["response_text"])
            judge_meta = judgment.pop("_meta", {}) if isinstance(judgment, dict) else {}
            append_jsonl(
                args.judgments,
                {
                    **base,
                    "judge_error": None,
                    "judgment": judgment,
                    **combine_usage_metrics("answer", _answer_metrics(row)),
                    **combine_usage_metrics("judge", judge_meta.get("usage_metrics")),
                    "judge_model_name": judge_meta.get("judge_model_name", FIXED_JUDGE_MODEL),
                },
            )
            print(f"[{i}/{len(remaining)}] scored index={row.get('dataset_index')} model={row.get('model_name')}")
        except Exception as exc:  # includes openai.BadRequestError for safety refusals
            msg = str(exc).replace("\n", " ")[:500]
            print(f"[{i}/{len(remaining)}] judge failure index={row.get('dataset_index')} model={row.get('model_name')}: {msg}", file=sys.stderr)
            append_jsonl(
                args.judgments,
                {
                    **base,
                    "judge_error": msg,
                    "judgment": None,
                    **combine_usage_metrics("answer", _answer_metrics(row)),
                    **combine_usage_metrics("judge", None),
                    "judge_model_name": FIXED_JUDGE_MODEL,
                },
            )

    all_judgments = load_jsonl(args.judgments)
    all_responses = load_jsonl(args.responses)
    write_summary_csv(args.judgments.with_name("summary.csv"), all_judgments, all_responses)
    write_summary_markdown(args.judgments.with_name("summary.md"), all_judgments, all_responses)
    print(f"done. judgments: {len(all_judgments)}")


if __name__ == "__main__":
    main()
