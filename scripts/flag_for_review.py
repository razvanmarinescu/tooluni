#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Surface questions that are worth a manual sanity check.

Heuristics applied (each row picks up zero or more flags):

  LOW          mean final score < 40
  LOW_RUN      a single run has final < 30 (even if mean is OK)
  SHORT_RESP   response_text is < 3000 bytes (often a clarification request,
               not an actual answer — e.g. Biomni Q12)
  CLARIFY      response asks questions back to the user rather than answering
               (looks for "clarifying question", "would you be able",
               "I need to ask", "once I have your answers", etc.)
  HIGH_VAR     stdev of final score across runs > 10 (judge instability or
               real spread between runs)
  GAP          | Biomni mean - Potato mean | > 25 on the same question
  HOL_MISMATCH judge gave high holistic ratings (factuality + completeness
               >= 8) but rubric coverage < 0.4 — judge may be praising prose
               while missing the rubric
  FALLBACK     judge fell back to a non-OpenAI model (e.g. safety refusal)
  UNCLEAR      >= 30% of rubric items came back "unclear"

Output: a ranked markdown table written to runs/review_candidates.md.
"""
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.dataset import load_items  # noqa: E402
from lib.reporting import load_jsonl  # noqa: E402

DATASET = PROJECT_ROOT / "genetic_benchmark_v1" / "47-submissions-clean.json"
OUT = PROJECT_ROOT / "runs" / "review_candidates.md"


def _scores(row: dict[str, Any]) -> dict[str, Any]:
    j = row.get("judgment") or {}
    return j.get("scores", {}) if isinstance(j, dict) else {}


def _holistic(row: dict[str, Any]) -> dict[str, Any]:
    j = row.get("judgment") or {}
    return j.get("holistic", {}) if isinstance(j, dict) else {}


def _verdicts(row: dict[str, Any]) -> list[dict[str, Any]]:
    j = row.get("judgment") or {}
    if not isinstance(j, dict):
        return []
    return list(j.get("expected") or []) + list(j.get("prohibited") or [])


def _unclear_fraction(row: dict[str, Any]) -> float:
    verdicts = _verdicts(row)
    if not verdicts:
        return 0.0
    n_unclear = sum(1 for v in verdicts if str(v.get("status", "")) == "unclear")
    return n_unclear / len(verdicts)


def _resp_size(resp_row: dict[str, Any] | None) -> int:
    if not resp_row:
        return 0
    return len(resp_row.get("response_text") or "")


_CLARIFY_PHRASES = (
    "clarifying question",
    "i need to ask",
    "need a few clarifying",
    "would you be able to provide",
    "once i have your answers",
    "could you clarify",
    "could you provide more",
    "before i can",
    "to give you the best",
    "to provide a meaningful",
)


def _looks_like_clarification(resp_row: dict[str, Any] | None) -> bool:
    if not resp_row:
        return False
    text = (resp_row.get("response_text") or "").lower()
    if not text:
        return False
    return any(phrase in text for phrase in _CLARIFY_PHRASES)


def main() -> None:
    biomni_resp = {r.get("dataset_index"): r for r in load_jsonl(PROJECT_ROOT / "runs/biomni/responses.jsonl")}
    biomni_judge = load_jsonl(PROJECT_ROOT / "runs/biomni/judgments.jsonl")
    potato_resp = load_jsonl(PROJECT_ROOT / "runs/potato/responses.jsonl")
    potato_judge = load_jsonl(PROJECT_ROOT / "runs/potato/judgments.jsonl")

    # Index potato responses (one per run)
    potato_resp_by_key = {(r["dataset_index"], r["model_name"]): r for r in potato_resp}

    items = load_items(DATASET)
    questions = {i: (item.get("prompt") or "").strip() for i, item in enumerate(items, start=1)}

    # Group by dataset_index
    biomni_by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    potato_by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for j in biomni_judge:
        if isinstance(j.get("dataset_index"), int):
            biomni_by_q[j["dataset_index"]].append(j)
    for j in potato_judge:
        if isinstance(j.get("dataset_index"), int):
            potato_by_q[j["dataset_index"]].append(j)

    all_indices = sorted(set(biomni_by_q) | set(potato_by_q))

    rows_out: list[dict[str, Any]] = []
    for idx in all_indices:
        biomni_rows = biomni_by_q.get(idx, [])
        potato_rows = potato_by_q.get(idx, [])

        b_finals = [_scores(r).get("final_score") for r in biomni_rows]
        b_finals = [v for v in b_finals if isinstance(v, (int, float))]
        p_finals = [_scores(r).get("final_score") for r in potato_rows]
        p_finals = [v for v in p_finals if isinstance(v, (int, float))]

        b_mean = statistics.mean(b_finals) if b_finals else None
        p_mean = statistics.mean(p_finals) if p_finals else None
        p_stdev = statistics.stdev(p_finals) if len(p_finals) >= 2 else 0.0

        b_resp_size = _resp_size(biomni_resp.get(idx))
        # For potato, take the smallest run's response size (a single short run is suspicious)
        potato_sizes = [
            _resp_size(potato_resp_by_key.get((r["dataset_index"], r.get("model_name"))))
            for r in potato_rows
        ]
        p_min_size = min(potato_sizes) if potato_sizes else 0

        flags: list[str] = []
        # LOW (combined mean across both systems' available means)
        means = [m for m in (b_mean, p_mean) if m is not None]
        worst_mean = min(means) if means else None
        if worst_mean is not None and worst_mean < 40:
            flags.append("LOW")
        # LOW_RUN
        for f in b_finals + p_finals:
            if f < 30:
                flags.append("LOW_RUN")
                break
        # SHORT_RESP
        if (biomni_rows and b_resp_size and b_resp_size < 3000) or (potato_rows and p_min_size and p_min_size < 3000):
            flags.append("SHORT_RESP")
        # CLARIFY
        if _looks_like_clarification(biomni_resp.get(idx)):
            flags.append("CLARIFY")
        else:
            for r in potato_rows:
                if _looks_like_clarification(potato_resp_by_key.get((r["dataset_index"], r.get("model_name")))):
                    flags.append("CLARIFY")
                    break
        # HIGH_VAR (Potato only — Biomni has 1 run)
        if p_stdev > 10:
            flags.append("HIGH_VAR")
        # GAP
        if b_mean is not None and p_mean is not None and abs(b_mean - p_mean) > 25:
            flags.append("GAP")
        # HOL_MISMATCH (any row)
        for r in biomni_rows + potato_rows:
            scores = _scores(r)
            cov = scores.get("expected_coverage")
            h = _holistic(r)
            if isinstance(cov, (int, float)) and cov < 0.4:
                fac = h.get("factuality")
                comp = h.get("completeness")
                if isinstance(fac, (int, float)) and isinstance(comp, (int, float)) and (fac + comp) >= 8:
                    flags.append("HOL_MISMATCH")
                    break
        # FALLBACK
        for r in biomni_rows + potato_rows:
            jm = (r.get("judge_model_name") or "")
            if jm and "gpt" not in jm.lower():
                flags.append("FALLBACK")
                break
        # UNCLEAR
        for r in biomni_rows + potato_rows:
            if _unclear_fraction(r) >= 0.30:
                flags.append("UNCLEAR")
                break

        if not flags:
            continue

        rows_out.append(
            {
                "idx": idx,
                "question": questions.get(idx, "")[:90],
                "biomni_mean": b_mean,
                "potato_mean": p_mean,
                "potato_runs": len(potato_rows),
                "potato_stdev": p_stdev,
                "biomni_resp_bytes": b_resp_size,
                "potato_min_bytes": p_min_size,
                "flags": sorted(set(flags)),
            }
        )

    # Rank: by number of flags first, then by worst final score asc
    def _rank_key(r: dict[str, Any]) -> tuple[int, float, int]:
        finals = [v for v in (r["biomni_mean"], r["potato_mean"]) if v is not None]
        worst = min(finals) if finals else 1e9
        return (-len(r["flags"]), worst, r["idx"])

    rows_out.sort(key=_rank_key)

    lines = [
        "# Manual review candidates",
        "",
        f"Generated from `runs/biomni/judgments.jsonl` and `runs/potato/judgments.jsonl`. ",
        f"{len(rows_out)} questions flagged out of {len(all_indices)}.",
        "",
        "## Flag legend",
        "",
        "- **LOW** — worst mean final score across the two systems is below 40.",
        "- **LOW_RUN** — at least one individual run scored below 30.",
        "- **SHORT_RESP** — a response was below 3000 bytes (often a clarification request, not an answer).",
        "- **CLARIFY** — the response asks questions back to the user instead of answering (e.g. 'I need to ask a few clarifying questions').",
        "- **HIGH_VAR** — Potato run-to-run stdev exceeds 10 points.",
        "- **GAP** — Biomni mean and Potato mean differ by more than 25 points on the same question.",
        "- **HOL_MISMATCH** — judge gave factuality+completeness >= 8 but rubric coverage was < 0.4 (judge praising prose while missing the rubric).",
        "- **FALLBACK** — the judge fell back to a non-OpenAI model (gpt-5.4 refused).",
        "- **UNCLEAR** — at least 30% of rubric verdicts came back 'unclear'.",
        "",
        "## Candidates",
        "",
        "| Q | Flags | Biomni mean | Potato runs / mean / stdev | Biomni size | Potato min size | Question |",
        "| ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for r in rows_out:
        b_str = f"{r['biomni_mean']:.1f}" if r['biomni_mean'] is not None else "-"
        if r['potato_mean'] is not None:
            p_str = f"{r['potato_runs']} / {r['potato_mean']:.1f} / {r['potato_stdev']:.1f}"
        else:
            p_str = "-"
        lines.append(
            f"| {r['idx']} | {' '.join(r['flags'])} | {b_str} | {p_str} | "
            f"{r['biomni_resp_bytes']} | {r['potato_min_bytes']} | {r['question'].replace('|', '\\|')} |"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} — {len(rows_out)} flagged questions")


if __name__ == "__main__":
    main()
