#!/Users/razvan/research/evals/tooluni/.venv/bin/python
from __future__ import annotations

import argparse
from pathlib import Path

from lib.judge import Judge
from lib.reporting import append_jsonl, load_jsonl, write_summary_csv, write_summary_markdown
from lib.usage_costs import combine_usage_metrics


FIXED_JUDGE_MODEL = "gpt-5.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge existing response rows.")
    parser.add_argument("responses", type=Path)
    parser.add_argument("judgments", type=Path)
    parser.add_argument("--judge-model", default=FIXED_JUDGE_MODEL, help=f"Ignored. Judge model is pinned to {FIXED_JUDGE_MODEL}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.judge_model = FIXED_JUDGE_MODEL
    root = Path(__file__).resolve().parents[1]
    judge = Judge(root, model_name=args.judge_model)
    responses = load_jsonl(args.responses)
    for row in responses:
        if row.get("response_error") or not row.get("response_text"):
            continue
        judgment = judge.judge_response(row["question"], row["criteria"], row["response_text"])
        judge_meta = judgment.pop("_meta", {}) if isinstance(judgment, dict) else {}
        append_jsonl(
            args.judgments,
            {
                "dataset_index": row.get("dataset_index"),
                "submission_id": row.get("submission_id"),
                "provider": row.get("provider"),
                "model_name": row.get("model_name"),
                "display_name": row.get("display_name"),
                "tier": row.get("tier"),
                "has_structured_rubric": row.get("has_structured_rubric"),
                "question": row.get("question"),
                "response_error": row.get("response_error"),
                "judge_error": None,
                "judgment": judgment,
                **combine_usage_metrics(
                    "answer",
                    {
                        "input_tokens": row.get("answer_input_tokens"),
                        "output_tokens": row.get("answer_output_tokens"),
                        "total_tokens": row.get("answer_total_tokens"),
                        "cached_input_tokens": row.get("answer_cached_input_tokens"),
                        "cache_creation_input_tokens": row.get("answer_cache_creation_input_tokens"),
                        "cache_creation_ephemeral_5m_input_tokens": row.get("answer_cache_creation_ephemeral_5m_input_tokens"),
                        "cache_creation_ephemeral_1h_input_tokens": row.get("answer_cache_creation_ephemeral_1h_input_tokens"),
                        "cache_read_input_tokens": row.get("answer_cache_read_input_tokens"),
                        "estimated_cost_usd": row.get("answer_estimated_cost_usd"),
                    },
                ),
                **combine_usage_metrics("judge", judge_meta.get("usage_metrics")),
                "judge_model_name": judge_meta.get("judge_model_name", args.judge_model),
            },
        )
    for row in responses:
        if not row.get("response_error") and row.get("response_text"):
            continue
        append_jsonl(
            args.judgments,
            {
                "dataset_index": row.get("dataset_index"),
                "submission_id": row.get("submission_id"),
                "provider": row.get("provider"),
                "model_name": row.get("model_name"),
                "display_name": row.get("display_name"),
                "tier": row.get("tier"),
                "has_structured_rubric": row.get("has_structured_rubric"),
                "question": row.get("question"),
                "response_error": row.get("response_error"),
                "judge_error": "Skipped because response generation failed.",
                "judgment": None,
                **combine_usage_metrics(
                    "answer",
                    {
                        "input_tokens": row.get("answer_input_tokens"),
                        "output_tokens": row.get("answer_output_tokens"),
                        "total_tokens": row.get("answer_total_tokens"),
                        "cached_input_tokens": row.get("answer_cached_input_tokens"),
                        "cache_creation_input_tokens": row.get("answer_cache_creation_input_tokens"),
                        "cache_creation_ephemeral_5m_input_tokens": row.get("answer_cache_creation_ephemeral_5m_input_tokens"),
                        "cache_creation_ephemeral_1h_input_tokens": row.get("answer_cache_creation_ephemeral_1h_input_tokens"),
                        "cache_read_input_tokens": row.get("answer_cache_read_input_tokens"),
                        "estimated_cost_usd": row.get("answer_estimated_cost_usd"),
                    },
                ),
                **combine_usage_metrics("judge", None),
                "judge_model_name": args.judge_model,
            },
        )
    judgments = load_jsonl(args.judgments)
    write_summary_csv(args.judgments.with_name("summary.csv"), judgments, responses)
    write_summary_markdown(args.judgments.with_name("summary.md"), judgments, responses)


if __name__ == "__main__":
    main()