#!/Users/razvan/research/evals/tooluni/.venv/bin/python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lib.dataset import default_dataset_path, load_items, normalize_criteria, project_root, select_items
from lib.judge import Judge
from lib.reporting import append_jsonl, load_jsonl, write_summary_csv, write_summary_markdown
from lib.runners import AnswerRunner, default_model_specs, default_tiers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ToolUniverse evaluation harness.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--judge-model", default="gpt-5.4")
    return parser.parse_args()


def response_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def judgment_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def next_incremental_run_name(root: Path) -> str:
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    numeric_names = []
    for child in runs_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            numeric_names.append(int(child.name))

    next_value = max(numeric_names, default=0) + 1
    return f"{next_value:05d}"


def build_output_paths(root: Path, run_name: str | None) -> dict[str, Path]:
    resolved_run_name = run_name or next_incremental_run_name(root)
    run_dir = root / "runs" / resolved_run_name
    return {
        "run_name": resolved_run_name,
        "run_dir": run_dir,
        "responses": run_dir / "responses.jsonl",
        "judgments": run_dir / "judgments.jsonl",
        "tool_traces": run_dir / "tool_traces.jsonl",
        "summary_csv": run_dir / "summary.csv",
        "summary_md": run_dir / "summary.md",
        "meta": run_dir / "meta.json",
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    items = load_items(args.dataset)
    selected_items = select_items(items, start_index=args.start_index, end_index=args.end_index, limit=args.limit)
    paths = build_output_paths(root, args.run_name)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["meta"].write_text(
        json.dumps(
            {
                "run_name": paths["run_name"],
                "dataset": str(args.dataset),
                "start_index": args.start_index,
                "end_index": args.end_index,
                "limit": args.limit,
                "judge_model": args.judge_model,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    existing_responses = {response_identity(row) for row in load_jsonl(paths["responses"])} if args.resume else set()
    existing_judgments = {judgment_identity(row) for row in load_jsonl(paths["judgments"])} if args.resume else set()

    runner = AnswerRunner(root)
    judge = Judge(root, model_name=args.judge_model)
    model_specs = default_model_specs()
    tiers = default_tiers()

    response_rows: list[dict[str, Any]] = []
    judgment_rows: list[dict[str, Any]] = []

    for dataset_index, item in selected_items:
        question = item.get("prompt", "").strip()
        criteria = normalize_criteria(item)
        for model in model_specs:
            for tier in tiers:
                base_row = {
                    "dataset_index": dataset_index,
                    "submission_id": item.get("id"),
                    "provider": model.provider,
                    "model_name": model.model_name,
                    "display_name": model.display_name,
                    "tier": tier,
                    "has_structured_rubric": criteria["has_structured_rubric"],
                }

                identity = response_identity(base_row)
                response_row: dict[str, Any] | None = None
                if not args.judge_only:
                    if identity in existing_responses:
                        continue
                    result = runner.generate(model, tier, question)
                    response_row = {
                        **base_row,
                        "question": question,
                        "criteria": criteria,
                        "clarity_selections": item.get("claritySelections") or {},
                        "response_text": result.get("response_text"),
                        "response_error": result.get("error"),
                        "context": result.get("context"),
                        "raw_response": result.get("raw_response"),
                        "tool_trace_available": bool(result.get("tool_trace")),
                        "tool_trace_call_count": len(result.get("tool_trace") or []),
                        "tool_trace_file": paths["tool_traces"].name if result.get("tool_trace") else None,
                    }
                    append_jsonl(paths["responses"], response_row)
                    response_rows.append(response_row)
                    if result.get("tool_trace"):
                        append_jsonl(
                            paths["tool_traces"],
                            {
                                **base_row,
                                "question": question,
                                "tool_trace": result.get("tool_trace"),
                                "pretty_trace": result.get("raw_response", {}).get("pretty_tool_trace", ""),
                            },
                        )

                judgment_key = judgment_identity(base_row)
                if judgment_key in existing_judgments:
                    continue

                if response_row is None:
                    matches = [
                        row
                        for row in load_jsonl(paths["responses"])
                        if response_identity(row) == identity
                    ]
                    response_row = matches[-1] if matches else None
                if response_row is None:
                    continue

                if response_row.get("response_error") or not response_row.get("response_text"):
                    judgment_row = {
                        **base_row,
                        "question": question,
                        "response_error": response_row.get("response_error"),
                        "judge_error": "Skipped because response generation failed.",
                        "judgment": None,
                    }
                else:
                    try:
                        judgment = judge.judge_response(question, criteria, response_row["response_text"])
                        judgment_row = {
                            **base_row,
                            "question": question,
                            "response_error": response_row.get("response_error"),
                            "judge_error": None,
                            "judgment": judgment,
                        }
                    except Exception as exc:  # pragma: no cover - depends on external APIs
                        judgment_row = {
                            **base_row,
                            "question": question,
                            "response_error": response_row.get("response_error"),
                            "judge_error": str(exc),
                            "judgment": None,
                        }

                append_jsonl(paths["judgments"], judgment_row)
                judgment_rows.append(judgment_row)

    all_judgments = load_jsonl(paths["judgments"])
    write_summary_csv(paths["summary_csv"], all_judgments)
    write_summary_markdown(paths["summary_md"], all_judgments)
    print(f"Run complete: {paths['run_dir']}")


if __name__ == "__main__":
    main()
