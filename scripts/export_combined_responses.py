#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Export each Biomni / Potato response to its own .md file.

Two folders are produced:
  combined_responses/             — response text only
  combined_responses_judgements/  — response text plus the judge's verdict
                                    (per-criterion statuses + scores)

Biomni filenames embed the source folder p-NN. Potato filenames embed both
the source folder query_NNN and the run number, so the +1 offset between
Potato's folder numbering and our dataset_index is visible at a glance.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.reporting import load_jsonl  # noqa: E402

OUTPUT_RESPONSES_DIR = PROJECT_ROOT / "combined_responses"
OUTPUT_JUDGEMENTS_DIR = PROJECT_ROOT / "combined_responses_judgements"


QUERY_RE = re.compile(r"query_(\d+)_")


def _slugify(text: str, length: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:length].strip("_")


def _query_id_from_source(source_path: str) -> str:
    """Extract 'query<NNN>' from a Potato source path like
    'data/Potato.../query_022_when_following_the_protocol_to_different/run_1'."""
    for part in (source_path or "").split("/"):
        m = QUERY_RE.match(part)
        if m:
            return f"query{int(m.group(1)):03d}"
    return "queryNA"


def _run_id_from_model(model_name: str) -> str:
    return (model_name or "potato").replace("potato-", "").replace("_", "")


def _build_response_section(row: dict[str, Any]) -> tuple[str, str]:
    """Return (header_block, body) for a response row.

    Body is the same response text we pass to the judge.
    """
    idx = row["dataset_index"]
    if row.get("display_name") == "Biomni":
        sources = row.get("source_files") or ["answer.txt"]
        header = (
            f"<!-- biomi q{idx} | source folder: {row.get('source_path')} | "
            f"included files: {', '.join(sources)} -->\n\n"
            f"# Biomi response — Q{idx}: {row.get('question', '').strip()}\n"
        )
    else:
        run = _run_id_from_model(row.get("model_name", ""))
        query = _query_id_from_source(row.get("source_path", ""))
        header = (
            f"<!-- potato q{idx} | source folder: {row.get('source_path')} | "
            f"{query} | {run} -->\n\n"
            f"# Potato response — Q{idx} ({query} / {run}): "
            f"{row.get('question', '').strip()}\n"
        )
    return header, row["response_text"]


def _format_judgment(row: dict[str, Any]) -> str:
    if row.get("judge_error"):
        return f"_Judge error_: `{row.get('judge_error')}`\n"
    judgment = row.get("judgment")
    if not isinstance(judgment, dict):
        return "_No judgment recorded._\n"
    scores = judgment.get("scores", {}) or {}
    expected = judgment.get("expected") or []
    prohibited = judgment.get("prohibited") or []
    holistic = judgment.get("holistic") or {}
    summary = (judgment.get("summary") or "").strip()

    lines: list[str] = ["## Judgment", ""]
    lines.append(f"- **Final score**: {scores.get('final_score', 'n/a')}")
    if "expected_coverage" in scores:
        lines.append(f"- **Expected coverage**: {scores.get('expected_coverage')}")
    if "prohibited_rate" in scores:
        lines.append(f"- **Prohibited rate**: {scores.get('prohibited_rate')}")
    if holistic:
        holistic_str = ", ".join(f"{k}={v}" for k, v in holistic.items())
        lines.append(f"- **Holistic**: {holistic_str}")
    if summary:
        lines.append("")
        lines.append(f"> {summary}")

    if expected:
        lines.extend(["", "### Expected criteria", ""])
        for entry in expected:
            criterion = (entry.get("criterion") or "").strip()
            status = entry.get("status", "?")
            evidence = (entry.get("evidence") or "").strip().replace("\n", " ")
            lines.append(f"- **[{status}]** {criterion}")
            if evidence:
                lines.append(f"  - _evidence_: {evidence}")

    if prohibited:
        lines.extend(["", "### Prohibited criteria", ""])
        for entry in prohibited:
            criterion = (entry.get("criterion") or "").strip()
            status = entry.get("status", "?")
            evidence = (entry.get("evidence") or "").strip().replace("\n", " ")
            lines.append(f"- **[{status}]** {criterion}")
            if evidence:
                lines.append(f"  - _evidence_: {evidence}")

    return "\n".join(lines) + "\n"


def _filename_for(row: dict[str, Any]) -> str:
    idx = row["dataset_index"]
    slug = _slugify(row.get("question") or "", length=40)
    if row.get("display_name") == "Biomni":
        src = (row.get("source_path") or "").rsplit("/", 1)[-1] or "unknown"
        return f"biomi_q{idx:02d}_{src}_{slug}.md"
    run = _run_id_from_model(row.get("model_name", ""))
    query = _query_id_from_source(row.get("source_path", ""))
    return f"potato_q{idx:02d}_{query}_{run}_{slug}.md"


def main() -> None:
    OUTPUT_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JUDGEMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Wipe stale files so renamed ones don't linger.
    for d in (OUTPUT_RESPONSES_DIR, OUTPUT_JUDGEMENTS_DIR):
        for stale in d.glob("*.md"):
            stale.unlink()

    biomni_resp = load_jsonl(PROJECT_ROOT / "runs" / "biomni" / "responses.jsonl")
    biomni_judge = load_jsonl(PROJECT_ROOT / "runs" / "biomni" / "judgments.jsonl")
    potato_resp = load_jsonl(PROJECT_ROOT / "runs" / "potato" / "responses.jsonl")
    potato_judge = load_jsonl(PROJECT_ROOT / "runs" / "potato" / "judgments.jsonl")

    def _key(r: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        return (r.get("dataset_index"), r.get("provider"), r.get("model_name"), r.get("tier"))

    judgement_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for j in biomni_judge + potato_judge:
        judgement_by_key[_key(j)] = j

    n_resp = 0
    n_judge = 0
    for row in biomni_resp + potato_resp:
        filename = _filename_for(row)
        header, body = _build_response_section(row)
        # Plain response file
        (OUTPUT_RESPONSES_DIR / filename).write_text(header + "\n---\n\n" + body, encoding="utf-8")
        n_resp += 1
        # Response + judgement file
        judgment_md = _format_judgment(judgement_by_key.get(_key(row), {}))
        combined = header + "\n---\n\n" + judgment_md + "\n---\n\n## Response text\n\n" + body
        (OUTPUT_JUDGEMENTS_DIR / filename).write_text(combined, encoding="utf-8")
        n_judge += 1

    print(f"wrote {n_resp} files to {OUTPUT_RESPONSES_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"wrote {n_judge} files to {OUTPUT_JUDGEMENTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
