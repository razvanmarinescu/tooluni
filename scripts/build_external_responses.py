#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Build responses.jsonl files for externally-produced answer sets
(Biomni and Potato) so they can be scored by scripts/judge_eval.py.

Biomni layout: data/Biomi/biomni_outputs/p-NN/{prompt.txt, answer.txt}
Potato layout: data/Potato_Artifacts_Gene_Editing_Mar 26 2026/query_NNN_<slug>/run_N/final_report.md

Responses are matched to entries in genetic_benchmark_v1/47-submissions-clean.json
by prompt text (biomni) or by slug prefix of the question prompt (potato).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.dataset import load_items, normalize_criteria  # noqa: E402
from lib.reporting import append_jsonl  # noqa: E402


DATASET = PROJECT_ROOT / "genetic_benchmark_v1" / "47-submissions-clean.json"
BIOMNI_DIR = PROJECT_ROOT / "data" / "Biomi" / "biomni_outputs"
POTATO_DIR = PROJECT_ROOT / "data" / "Potato_Artifacts_Gene_Editing_Mar 26 2026"
RUNS_DIR = PROJECT_ROOT / "runs"


def _slugify(text: str, length: int = 40) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:length]


def _flatten(text: str) -> str:
    """Lowercase and drop every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_item_for_prompt(prompt: str, items: list[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    p = prompt.strip()
    for i, item in enumerate(items, start=1):
        if (item.get("prompt") or "").strip() == p:
            return i, item
    # Try prefix match
    slug = _slugify(p, length=40)
    for i, item in enumerate(items, start=1):
        item_slug = _slugify(item.get("prompt") or "", length=40)
        if item_slug == slug:
            return i, item
    return None


def find_item_for_slug(
    query_num: int,
    query_slug: str,
    items: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]] | None:
    """query_slug comes from folder name, e.g. 'design_a_protocol_to_precisely_insert_a'.

    Potato folder slugs drop all non-alphanumeric characters (so 'pre-clinical'
    shows up as 'preclinical'). And many prompts share the same 40-char prefix.
    We compare the alphanumeric-only flattened form. For ties, we pick the
    dataset entry closest to the query number (which roughly tracks position).
    """
    query_flat = _flatten(query_slug)
    matches: list[tuple[int, dict[str, Any]]] = []
    for i, item in enumerate(items, start=1):
        item_flat = _flatten(item.get("prompt") or "")
        if item_flat.startswith(query_flat):
            matches.append((i, item))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # Tie-break: pick the one whose 1-indexed position is nearest the query
    # number (folder numbering loosely tracks dataset order).
    matches.sort(key=lambda pair: (abs(pair[0] - query_num), pair[0]))
    return matches[0]


def build_response_row(
    *,
    dataset_index: int,
    item: dict[str, Any],
    response_text: str,
    display_name: str,
    model_name: str,
    tier: str,
    source_path: str,
) -> dict[str, Any]:
    criteria = normalize_criteria(item)
    return {
        "dataset_index": dataset_index,
        "submission_id": item.get("id"),
        "provider": "external",
        "model_name": model_name,
        "display_name": display_name,
        "tier": tier,
        "rubric_style": criteria.get("rubric_style"),
        "has_structured_rubric": criteria["has_structured_rubric"],
        "question": (item.get("prompt") or "").strip(),
        "criteria": criteria,
        "response_text": response_text,
        "response_error": None,
        "source_path": source_path,
    }


def _collect_biomni_response_text(folder: Path) -> tuple[str, list[str]]:
    """Return (response_text, list-of-source-files-included).

    answer.txt is the agent's summary; any markdown/text files under
    files/mnt_results/ are the longer protocol reports the summary points
    to. Concatenating them gives the full work the rubric should be
    judging.
    """
    parts: list[str] = []
    sources: list[str] = []

    answer_file = folder / "answer.txt"
    if answer_file.exists():
        parts.append(f"# answer.txt\n\n{answer_file.read_text(encoding='utf-8').strip()}")
        sources.append("answer.txt")

    reports_dir = folder / "files" / "mnt_results"
    if reports_dir.exists():
        for report in sorted(reports_dir.iterdir()):
            if not report.is_file():
                continue
            if report.suffix.lower() not in (".md", ".txt"):
                continue
            parts.append(f"# {report.name}\n\n{report.read_text(encoding='utf-8').strip()}")
            sources.append(f"files/mnt_results/{report.name}")

    return ("\n\n---\n\n".join(parts), sources)


def build_biomni(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for sub in sorted(BIOMNI_DIR.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("p-"):
            continue
        prompt_file = sub / "prompt.txt"
        if not prompt_file.exists():
            continue
        prompt = prompt_file.read_text(encoding="utf-8")
        response_text, sources = _collect_biomni_response_text(sub)
        if not response_text:
            continue
        matched = find_item_for_prompt(prompt, items)
        if not matched:
            unmatched.append(sub.name)
            continue
        dataset_index, item = matched
        row = build_response_row(
            dataset_index=dataset_index,
            item=item,
            response_text=response_text,
            display_name="Biomni",
            model_name="biomni",
            tier="external",
            source_path=str(sub.relative_to(PROJECT_ROOT)),
        )
        row["source_files"] = sources
        rows.append(row)
    if unmatched:
        print(f"[biomni] unmatched folders: {unmatched}", file=sys.stderr)
    return rows


def build_potato(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    folder_re = re.compile(r"^query_(\d+)_(.+)$")
    for query_dir in sorted(POTATO_DIR.iterdir()):
        if not query_dir.is_dir():
            continue
        m = folder_re.match(query_dir.name)
        if not m:
            continue
        query_num = int(m.group(1))
        query_slug = m.group(2)
        matched = find_item_for_slug(query_num, query_slug, items)
        if not matched:
            unmatched.append(query_dir.name)
            continue
        dataset_index, item = matched
        for run_dir in sorted(query_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            report = run_dir / "final_report.md"
            if not report.exists():
                continue
            response_text = report.read_text(encoding="utf-8")
            rows.append(
                build_response_row(
                    dataset_index=dataset_index,
                    item=item,
                    response_text=response_text,
                    display_name="Potato",
                    # Vary model_name per run so (dataset_index, provider,
                    # model_name, tier) stays unique, but keep display_name
                    # and tier constant so the markdown summary aggregates
                    # them as one runner.
                    model_name=f"potato-{run_dir.name}",
                    tier="external",
                    source_path=str(run_dir.relative_to(PROJECT_ROOT)),
                )
            )
    if unmatched:
        print(f"[potato] unmatched folders: {unmatched}", file=sys.stderr)
    return rows


def write_responses(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    for row in rows:
        append_jsonl(output_path, row)


def main() -> None:
    items = load_items(DATASET)

    biomni_rows = build_biomni(items)
    print(f"biomni rows: {len(biomni_rows)}")
    write_responses(biomni_rows, RUNS_DIR / "biomni" / "responses.jsonl")

    potato_rows = build_potato(items)
    print(f"potato rows: {len(potato_rows)}")
    write_responses(potato_rows, RUNS_DIR / "potato" / "responses.jsonl")


if __name__ == "__main__":
    main()
