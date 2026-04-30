#!/Users/razvan/research/evals/tooluni/.venv/bin/python
from __future__ import annotations

import argparse
from pathlib import Path

from lib.reporting import load_jsonl, write_summary_csv, write_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate summary files for an existing run.")
    parser.add_argument("--run-id", required=True, help="Run directory name under runs/, e.g. 00013")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / args.run_id
    responses_path = run_dir / "responses.jsonl"
    judgments_path = run_dir / "judgments.jsonl"

    responses = load_jsonl(responses_path)
    judgments = load_jsonl(judgments_path)

    write_summary_csv(run_dir / "summary.csv", judgments, responses)
    write_summary_markdown(run_dir / "summary.md", judgments, responses)
    print(f"Regenerated summary for {run_dir}")


if __name__ == "__main__":
    main()