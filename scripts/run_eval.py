#!/Users/razvan/research/evals/tooluni/.venv/bin/python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Any

from lib.context import render_markdown_web_trace
from lib.dataset import default_dataset_path, get_question_text, load_items, normalize_criteria, project_root, select_items
from lib.judge import HarnessJudge, Judge
from lib.reporting import append_jsonl, load_jsonl, write_summary_csv, write_summary_markdown
from lib.runners import AnswerRunner, default_model_specs, default_tiers
from lib.tooluniverse_mcp import render_markdown_tool_trace
from lib.usage_costs import combine_usage_metrics


FIXED_JUDGE_MODEL = "gpt-5.4"


def format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("#" * width) + "]"
    filled = min(width, int(width * completed / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


class ProgressReporter:
    def __init__(self, total_items: int, total_runs: int):
        self.total_items = total_items
        self.total_runs = total_runs
        self.completed_items = 0
        self.completed_runs = 0
        self.interactive = sys.stdout.isatty()
        self.terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
        self.lock = threading.Lock()
        self.active_runs: dict[str, dict[str, str | int]] = {}
        self._last_line_was_progress = False
        self._last_message_length = 0

    def _render(self, status: str | None = None) -> None:
        bar = format_progress_bar(self.completed_runs, self.total_runs)
        active_labels = []
        for active in sorted(self.active_runs.values(), key=lambda value: str(value["job_id"])):
            active_labels.append(
                f"{active['model_name']}@{active['item_position']}:{active['stage']}"
            )
        active_summary = ", ".join(active_labels[:3]) if active_labels else "idle"
        if len(active_labels) > 3:
            active_summary += f", +{len(active_labels) - 3} more"
        message = (
            f"{bar} {self.completed_runs}/{self.total_runs} runs"
            f" | {self.completed_items}/{self.total_items} q"
            f" | active {active_summary}"
        )
        if status:
            message += f" | {status}"
        if self.interactive:
            clipped = message[: max(1, self.terminal_width - 1)]
            padding = " " * max(0, self._last_message_length - len(clipped))
            print(f"\r{clipped}{padding}", end="", flush=True)
            self._last_line_was_progress = True
            self._last_message_length = len(clipped)
        else:
            print(message, flush=True)

    def update(
        self,
        *,
        job_id: str,
        item_position: int,
        dataset_index: int,
        model_name: str,
        tier: str,
        stage: str,
    ) -> None:
        with self.lock:
            self.active_runs[job_id] = {
                "job_id": job_id,
                "item_position": item_position,
                "dataset_index": dataset_index,
                "model_name": model_name,
                "tier": tier,
                "stage": stage,
            }
            self._render()

    def finish_run(self, *, job_id: str, item_completed: bool) -> None:
        with self.lock:
            self.completed_runs += 1
            self.active_runs.pop(job_id, None)
            if item_completed:
                self.completed_items += 1
            self._render(status="completed")

    def clear_run(self, job_id: str) -> None:
        with self.lock:
            self.active_runs.pop(job_id, None)
            self._render()

    def end_line(self) -> None:
        if self.interactive and self._last_line_was_progress:
            print()
            self._last_line_was_progress = False
            self._last_message_length = 0

    def print_status(self, message: str) -> None:
        self.end_line()
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ToolUniverse evaluation harness.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", nargs="+", help="Optional list of model names to run, e.g. gpt-5.4 claude-haiku-4-5 claude-opus-4-6 claude-sonnet-4-6")
    parser.add_argument("--tiers", nargs="+", help="Optional list of tiers to run, e.g. tooluniverse")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--run-name-suffix", default="", help="Optional suffix for auto-generated run names, e.g. _benchling")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for model runs.")
    parser.add_argument("--judge-model", default=FIXED_JUDGE_MODEL, help=f"Ignored when --verifier=single-verifier. Judge model is pinned to {FIXED_JUDGE_MODEL}.")
    parser.add_argument(
        "--verifier",
        choices=["single-verifier", "verifier-harness"],
        default="single-verifier",
        help="Which judging strategy to use. 'single-verifier' = one GPT-5.4 judge. "
        "'verifier-harness' = 4-way majority vote across 2x GPT-5.5 + 2x Sonnet-4.6 (Diego's pattern).",
    )
    return parser.parse_args()


def response_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def judgment_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset_index"), row.get("provider"), row.get("model_name"), row.get("tier"))


def next_incremental_run_name(root: Path, suffix: str = "") -> str:
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    numeric_names = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^(\d{5})(?:_.+)?$", child.name)
        if match:
            numeric_names.append(int(match.group(1)))

    next_value = max(numeric_names, default=0) + 1
    return f"{next_value:05d}{suffix}"


def build_output_paths(root: Path, run_name: str | None, run_name_suffix: str = "") -> dict[str, Path]:
    resolved_run_name = run_name or next_incremental_run_name(root, suffix=run_name_suffix)
    run_dir = root / "runs" / resolved_run_name
    return {
        "run_name": resolved_run_name,
        "run_dir": run_dir,
        "responses": run_dir / "responses.jsonl",
        "judgments": run_dir / "judgments.jsonl",
        "web_traces": run_dir / "web_traces.jsonl",
        "tool_traces": run_dir / "tool_traces.jsonl",
        "tooluniverse_traces_dir": run_dir / "tooluniverse_traces",
        "web_traces_dir": run_dir / "web_traces",
        "summary_csv": run_dir / "summary.csv",
        "summary_md": run_dir / "summary.md",
        "meta": run_dir / "meta.json",
    }


THREAD_LOCAL = threading.local()


def get_thread_runner(root: Path) -> AnswerRunner:
    runner = getattr(THREAD_LOCAL, "runner", None)
    if runner is None:
        runner = AnswerRunner(root)
        THREAD_LOCAL.runner = runner
    return runner


def get_thread_judge(root: Path, model_name: str, verifier: str = "single-verifier"):
    judge = getattr(THREAD_LOCAL, "judge", None)
    judge_model_name = getattr(THREAD_LOCAL, "judge_model_name", None)
    cache_key = (verifier, model_name)
    if judge is None or judge_model_name != cache_key:
        if verifier == "verifier-harness":
            judge = HarnessJudge(root)
        else:
            judge = Judge(root, model_name=model_name)
        THREAD_LOCAL.judge = judge
        THREAD_LOCAL.judge_model_name = cache_key
    return judge


def build_response_row(
    *,
    base_row: dict[str, Any],
    question: str,
    criteria: dict[str, Any],
    clarity_selections: dict[str, Any],
    generation_started_at: dt.datetime,
    generation_duration_seconds: float,
    result: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        **base_row,
        "question": question,
        "criteria": criteria,
        "clarity_selections": clarity_selections,
        "generation_started_at": generation_started_at.isoformat(),
        "generation_duration_seconds": generation_duration_seconds,
        "response_text": result.get("response_text"),
        "response_error": result.get("error"),
        "context": result.get("context"),
        "raw_response": result.get("raw_response"),
        "web_trace_available": bool(result.get("web_trace")),
        "web_trace_call_count": len(result.get("web_trace") or []),
        "web_trace_file": paths["web_traces"].name if result.get("web_trace") else None,
        "tool_trace_available": bool(result.get("tool_trace")),
        "tool_trace_call_count": len(result.get("tool_trace") or []),
        "tool_trace_file": paths["tool_traces"].name if result.get("tool_trace") else None,
        **combine_usage_metrics("answer", result.get("usage_metrics")),
    }


def build_judgment_row(
    *,
    base_row: dict[str, Any],
    question: str,
    response_row: dict[str, Any],
    args: argparse.Namespace,
    judge: Judge,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    if response_row.get("response_error") or not response_row.get("response_text"):
        return {
            **base_row,
            "question": question,
            "response_error": response_row.get("response_error"),
            "judge_error": "Skipped because response generation failed.",
            "judgment": None,
        }

    try:
        judge_started_at = dt.datetime.now(dt.timezone.utc)
        judge_started_perf = time.perf_counter()
        judgment = judge.judge_response(question, criteria, response_row["response_text"])
        judge_duration_seconds = round(time.perf_counter() - judge_started_perf, 3)
        judge_meta = judgment.pop("_meta", {}) if isinstance(judgment, dict) else {}
        return {
            **base_row,
            "question": question,
            "response_error": response_row.get("response_error"),
            "generation_started_at": response_row.get("generation_started_at"),
            "generation_duration_seconds": response_row.get("generation_duration_seconds"),
            "judge_started_at": judge_started_at.isoformat(),
            "judge_duration_seconds": judge_duration_seconds,
            "judge_error": None,
            "judgment": judgment,
            **combine_usage_metrics("answer", {
                "input_tokens": response_row.get("answer_input_tokens"),
                "output_tokens": response_row.get("answer_output_tokens"),
                "total_tokens": response_row.get("answer_total_tokens"),
                "cached_input_tokens": response_row.get("answer_cached_input_tokens"),
                "cache_creation_input_tokens": response_row.get("answer_cache_creation_input_tokens"),
                "cache_creation_ephemeral_5m_input_tokens": response_row.get("answer_cache_creation_ephemeral_5m_input_tokens"),
                "cache_creation_ephemeral_1h_input_tokens": response_row.get("answer_cache_creation_ephemeral_1h_input_tokens"),
                "cache_read_input_tokens": response_row.get("answer_cache_read_input_tokens"),
                "estimated_cost_usd": response_row.get("answer_estimated_cost_usd"),
            }),
            **combine_usage_metrics("judge", judge_meta.get("usage_metrics")),
            "judge_model_name": judge_meta.get("judge_model_name", args.judge_model),
            # Optional: only present when the verifier-harness runs. Captures
            # per-judge timing, retry attempts, and any errors so a future
            # post-hoc inspection of judgments.jsonl can see what happened.
            **(
                {"harness_calls": judge_meta["harness_calls"]}
                if judge_meta.get("harness_calls")
                else {}
            ),
        }
    except Exception as exc:  # pragma: no cover - depends on external APIs
        return {
            **base_row,
            "question": question,
            "response_error": response_row.get("response_error"),
            "generation_started_at": response_row.get("generation_started_at"),
            "generation_duration_seconds": response_row.get("generation_duration_seconds"),
            "judge_started_at": None,
            "judge_duration_seconds": None,
            "judge_error": str(exc),
            "judgment": None,
            **combine_usage_metrics("answer", {
                "input_tokens": response_row.get("answer_input_tokens"),
                "output_tokens": response_row.get("answer_output_tokens"),
                "total_tokens": response_row.get("answer_total_tokens"),
                "cached_input_tokens": response_row.get("answer_cached_input_tokens"),
                "cache_creation_input_tokens": response_row.get("answer_cache_creation_input_tokens"),
                "cache_creation_ephemeral_5m_input_tokens": response_row.get("answer_cache_creation_ephemeral_5m_input_tokens"),
                "cache_creation_ephemeral_1h_input_tokens": response_row.get("answer_cache_creation_ephemeral_1h_input_tokens"),
                "cache_read_input_tokens": response_row.get("answer_cache_read_input_tokens"),
                "estimated_cost_usd": response_row.get("answer_estimated_cost_usd"),
            }),
            **combine_usage_metrics("judge", None),
            "judge_model_name": args.judge_model,
        }


def _markdown_filename(dataset_index: int) -> str:
    return f"q{int(dataset_index):02d}.md"


def _write_tool_trace_markdown(
    *,
    paths: dict[str, Path],
    base_row: dict[str, Any],
    job: dict[str, Any],
    result: dict[str, Any],
) -> None:
    raw_response = result.get("raw_response") or {}
    raw_turns = raw_response.get("turns") if isinstance(raw_response, dict) else None
    trace_events = result.get("tool_trace") or []
    if not raw_turns and not trace_events:
        return
    md = render_markdown_tool_trace(
        question=job["question"],
        dataset_index=base_row["dataset_index"],
        submission_id=base_row.get("submission_id") or "",
        model_name=base_row.get("model_name") or "?",
        tier=base_row.get("tier") or "?",
        raw_turns=raw_turns or [],
        trace_events=trace_events,
        final_response_text=result.get("response_text"),
    )
    out_dir = paths["tooluniverse_traces_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _markdown_filename(base_row["dataset_index"])).write_text(md, encoding="utf-8")


def _write_web_trace_markdown(
    *,
    paths: dict[str, Path],
    base_row: dict[str, Any],
    job: dict[str, Any],
    result: dict[str, Any],
) -> None:
    web_trace = result.get("web_trace") or []
    if not web_trace:
        return
    md = render_markdown_web_trace(
        question=job["question"],
        dataset_index=base_row["dataset_index"],
        submission_id=base_row.get("submission_id") or "",
        model_name=base_row.get("model_name") or "?",
        trace=web_trace,
        raw_response=result.get("raw_response"),
        final_response_text=result.get("response_text"),
    )
    out_dir = paths["web_traces_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _markdown_filename(base_row["dataset_index"])).write_text(md, encoding="utf-8")


def process_job(
    *,
    root: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    progress: ProgressReporter,
    job: dict[str, Any],
) -> dict[str, Any]:
    base_row = job["base_row"]
    response_row = job.get("existing_response_row")
    job_id = job["job_id"]

    try:
        if job["needs_generation"]:
            progress.update(
                job_id=job_id,
                item_position=job["item_position"],
                dataset_index=job["dataset_index"],
                model_name=job["model_name"],
                tier=job["tier"],
                stage="generating",
            )
            runner = get_thread_runner(root)
            generation_started_at = dt.datetime.now(dt.timezone.utc)
            generation_started_perf = time.perf_counter()
            result = runner.generate(job["model_spec"], job["tier"], job["question"])
            generation_duration_seconds = round(time.perf_counter() - generation_started_perf, 3)
            response_row = build_response_row(
                base_row=base_row,
                question=job["question"],
                criteria=job["criteria"],
                clarity_selections=job["clarity_selections"],
                generation_started_at=generation_started_at,
                generation_duration_seconds=generation_duration_seconds,
                result=result,
                paths=paths,
            )
            web_trace_row = None
            if result.get("web_trace"):
                web_trace_row = {
                    **base_row,
                    "question": job["question"],
                    "web_trace": result.get("web_trace"),
                    "pretty_trace": result.get("raw_response", {}).get("pretty_web_trace", ""),
                }
                _write_web_trace_markdown(
                    paths=paths,
                    base_row=base_row,
                    job=job,
                    result=result,
                )
            tool_trace_row = None
            if result.get("tool_trace"):
                tool_trace_row = {
                    **base_row,
                    "question": job["question"],
                    "tool_trace": result.get("tool_trace"),
                    "pretty_trace": result.get("raw_response", {}).get("pretty_tool_trace", ""),
                }
                _write_tool_trace_markdown(
                    paths=paths,
                    base_row=base_row,
                    job=job,
                    result=result,
                )
        else:
            web_trace_row = None
            tool_trace_row = None

        judgment_row = None
        if job["needs_judgment"] and response_row is not None:
            progress.update(
                job_id=job_id,
                item_position=job["item_position"],
                dataset_index=job["dataset_index"],
                model_name=job["model_name"],
                tier=job["tier"],
                stage="judging",
            )
            judge = get_thread_judge(root, args.judge_model, verifier=getattr(args, "verifier", "single-verifier"))
            judgment_row = build_judgment_row(
                base_row=base_row,
                question=job["question"],
                response_row=response_row,
                args=args,
                judge=judge,
                criteria=job["criteria"],
            )

        return {
            "job": job,
            "response_row": response_row,
            "judgment_row": judgment_row,
            "web_trace_row": web_trace_row,
            "tool_trace_row": tool_trace_row,
        }
    finally:
        progress.clear_run(job_id)


def main() -> None:
    args = parse_args()
    args.judge_model = FIXED_JUDGE_MODEL
    root = project_root()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    run_started_perf = time.perf_counter()
    items = load_items(args.dataset)
    selected_items = select_items(items, start_index=args.start_index, end_index=args.end_index, limit=args.limit)
    paths = build_output_paths(root, args.run_name, args.run_name_suffix)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)

    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    existing_response_rows = {
        response_identity(row): row
        for row in load_jsonl(paths["responses"])
    }
    existing_judgments = {judgment_identity(row) for row in load_jsonl(paths["judgments"])} if args.resume else set()
    model_specs = default_model_specs()
    if args.models:
        allowed_models = set(args.models)
        model_specs = [model for model in model_specs if model.model_name in allowed_models or model.display_name in allowed_models]
    tiers = default_tiers()
    if args.tiers:
        allowed_tiers = set(args.tiers)
        tiers = [tier for tier in tiers if tier in allowed_tiers]

    if not model_specs:
        raise ValueError("No model specs remain after applying --models filters.")
    if not tiers:
        raise ValueError("No tiers remain after applying --tiers filters.")

    jobs: list[dict[str, Any]] = []
    item_job_counts: dict[int, int] = {}

    for item_position, (dataset_index, item) in enumerate(selected_items, start=1):
        question = get_question_text(item).strip()
        criteria = normalize_criteria(item)
        clarity_selections = item.get("claritySelections") or {}
        for model in model_specs:
            for tier in tiers:
                base_row = {
                    "dataset_index": dataset_index,
                    "submission_id": item.get("id"),
                    "provider": model.provider,
                    "model_name": model.model_name,
                    "display_name": model.display_name,
                    "tier": tier,
                    "rubric_style": criteria.get("rubric_style"),
                    "has_structured_rubric": criteria["has_structured_rubric"],
                }
                identity = response_identity(base_row)
                existing_response_row = existing_response_rows.get(identity)
                needs_generation = not args.judge_only and existing_response_row is None
                needs_judgment = identity not in existing_judgments
                if not needs_generation and not needs_judgment:
                    continue
                job_id = f"{dataset_index}:{model.model_name}:{tier}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "item_position": item_position,
                        "dataset_index": dataset_index,
                        "model_name": model.model_name,
                        "tier": tier,
                        "model_spec": model,
                        "question": question,
                        "criteria": criteria,
                        "clarity_selections": clarity_selections,
                        "base_row": base_row,
                        "identity": identity,
                        "existing_response_row": existing_response_row,
                        "needs_generation": needs_generation,
                        "needs_judgment": needs_judgment,
                    }
                )
                item_job_counts[dataset_index] = item_job_counts.get(dataset_index, 0) + 1

    progress = ProgressReporter(
        total_items=len(item_job_counts) if item_job_counts else len(selected_items),
        total_runs=len(jobs),
    )

    paths["meta"].write_text(
        json.dumps(
            {
                "run_name": paths["run_name"],
                "dataset": str(args.dataset),
                "start_index": args.start_index,
                "end_index": args.end_index,
                "limit": args.limit,
                "models": [m.model_name for m in model_specs],
                "tiers": tiers,
                "workers": args.workers,
                "judge_model": args.judge_model,
                "verifier": getattr(args, "verifier", "single-verifier"),
                "run_started_at": run_started_at.isoformat(),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    response_rows: list[dict[str, Any]] = []
    judgment_rows: list[dict[str, Any]] = []

    completed_jobs_by_item: dict[int, int] = {dataset_index: 0 for dataset_index in item_job_counts}

    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_job,
                    root=root,
                    args=args,
                    paths=paths,
                    progress=progress,
                    job=job,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                result = future.result()
                job = result["job"]
                response_row = result["response_row"]
                judgment_row = result["judgment_row"]
                if job["needs_generation"] and response_row is not None:
                    append_jsonl(paths["responses"], response_row)
                    response_rows.append(response_row)
                    existing_response_rows[job["identity"]] = response_row
                if result["web_trace_row"] is not None:
                    append_jsonl(paths["web_traces"], result["web_trace_row"])
                if result["tool_trace_row"] is not None:
                    append_jsonl(paths["tool_traces"], result["tool_trace_row"])
                if judgment_row is not None:
                    append_jsonl(paths["judgments"], judgment_row)
                    judgment_rows.append(judgment_row)
                completed_jobs_by_item[job["dataset_index"]] += 1
                item_completed = completed_jobs_by_item[job["dataset_index"]] == item_job_counts[job["dataset_index"]]
                progress.finish_run(job_id=job["job_id"], item_completed=item_completed)

    all_responses = load_jsonl(paths["responses"])
    all_judgments = load_jsonl(paths["judgments"])
    write_summary_csv(paths["summary_csv"], all_judgments, all_responses)
    write_summary_markdown(paths["summary_md"], all_judgments, all_responses)
    run_completed_at = dt.datetime.now(dt.timezone.utc)
    run_duration_seconds = round(time.perf_counter() - run_started_perf, 3)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    meta["run_completed_at"] = run_completed_at.isoformat()
    meta["run_duration_seconds"] = run_duration_seconds
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    progress.end_line()
    print(f"Run complete: {paths['run_dir']}")


if __name__ == "__main__":
    main()
