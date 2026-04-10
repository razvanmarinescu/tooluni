#!/Users/razvan/research/evals/tooluni/.venv/bin/python
from __future__ import annotations

import argparse
from pathlib import Path

from lib.judge import Judge
from lib.reporting import append_jsonl, load_jsonl, write_summary_csv, write_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge existing response rows.")
    parser.add_argument("responses", type=Path)
    parser.add_argument("judgments", type=Path)
    parser.add_argument("--judge-model", default="gpt-5.4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    judge = Judge(root, model_name=args.judge_model)
    responses = load_jsonl(args.responses)
    for row in responses:
        if row.get("response_error") or not row.get("response_text"):
            continue
        judgment = judge.judge_response(row["question"], row["criteria"], row["response_text"])
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
            },
        )
    judgments = load_jsonl(args.judgments)
    write_summary_csv(args.judgments.with_name("summary.csv"), judgments)
    write_summary_markdown(args.judgments.with_name("summary.md"), judgments)


if __name__ == "__main__":
    main()