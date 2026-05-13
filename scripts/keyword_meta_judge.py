#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Keyword-based meta-judge.

For each criterion in each judgment we:
  1. Extract anchor terms from the criterion (acronyms, specific reagents,
     gene/cell-line names, named entities, multi-word noun phrases).
  2. Score the criterion against the response: fraction of anchors found.
  3. Pick a *meta-verdict*:
       expected:    score == 1.0 → met, 0 < score < 1.0 → partial, 0 → missed
       prohibited:  score == 1.0 → violated, score == 0 → not_violated, else uncertain
  4. Compare to the judge's actual verdict and flag disagreements.

Criteria with no extractable specific anchor are skipped (`abstained`).
This is heuristic and meant to verify a percentage of the judges, not all of
them. We emit:
  combined_responses_judgements_meta/<filename>.md  (per response: judge vs meta)
  runs/meta_judge_summary.md                        (aggregate stats + chart)
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the anchor extractor + matcher from the audit so behavior is consistent.
import importlib.util  # noqa: E402

_audit_spec = importlib.util.spec_from_file_location("audit_module", str(PROJECT_ROOT / "scripts" / "audit_judge_verdicts.py"))
_audit_mod = importlib.util.module_from_spec(_audit_spec)
assert _audit_spec.loader is not None
_audit_spec.loader.exec_module(_audit_mod)

extract_anchors = _audit_mod.extract_anchors
anchor_in_response = _audit_mod.anchor_in_response
compute_generic_anchors = _audit_mod.compute_generic_anchors

from lib.reporting import _recompute_final_score, load_jsonl  # noqa: E402

PER_RESPONSE_DIR = PROJECT_ROOT / "combined_responses_judgements_meta"
SUMMARY_PATH = PROJECT_ROOT / "runs" / "meta_judge_summary.md"


def meta_verdict_expected(score: float | None) -> str:
    if score is None:
        return "abstain"
    if score >= 0.999:
        return "met"
    if score <= 0.001:
        return "missed"
    return "partial"


def meta_verdict_prohibited(score: float | None) -> str:
    if score is None:
        return "abstain"
    if score >= 0.999:
        return "violated"
    if score <= 0.001:
        return "not_violated"
    return "uncertain"


def evaluate_one(
    *,
    response_text: str,
    judgment: dict[str, Any],
    generic_anchors: set[str],
) -> dict[str, Any]:
    response_lower = response_text.lower()
    expected_items = list(judgment.get("expected") or [])
    prohibited_items = list(judgment.get("prohibited") or [])

    rows: list[dict[str, Any]] = []
    for section, items, meta_fn in (
        ("expected", expected_items, meta_verdict_expected),
        ("prohibited", prohibited_items, meta_verdict_prohibited),
    ):
        for item in items:
            criterion = (item.get("criterion") or "").strip()
            judge_status = (item.get("status") or "").strip()
            evidence = (item.get("evidence") or "").strip().replace("\n", " ")
            anchors = extract_anchors(criterion)
            specific = [a for a in anchors if a.lower() not in generic_anchors]
            if not specific:
                rows.append(
                    {
                        "section": section,
                        "criterion": criterion,
                        "judge_status": judge_status,
                        "judge_evidence": evidence,
                        "specific_anchors": [],
                        "anchors_found": [],
                        "score": None,
                        "meta_status": "abstain",
                        "agree": None,
                    }
                )
                continue
            present = [a for a in specific if anchor_in_response(a, response_lower)]
            score = len(present) / len(specific)
            meta_status = meta_fn(score)
            # Treat partial / unclear as agreeing with anything that isn't a
            # clear contradiction, since both signal "in between".
            if judge_status in ("partial", "unclear"):
                agree = meta_status not in (
                    "met",
                    "missed",
                    "violated",
                    "not_violated",
                ) or _partial_consistent(judge_status, meta_status)
            else:
                agree = (judge_status == meta_status)
            rows.append(
                {
                    "section": section,
                    "criterion": criterion,
                    "judge_status": judge_status,
                    "judge_evidence": evidence,
                    "specific_anchors": specific,
                    "anchors_found": present,
                    "score": score,
                    "meta_status": meta_status,
                    "agree": agree,
                }
            )
    return {"items": rows}


def _partial_consistent(judge: str, meta: str) -> bool:
    # Treat "partial" + "partial" or "uncertain" as compatible.
    return (judge == "partial" and meta in ("partial", "met", "missed")) or (
        judge == "unclear" and meta in ("partial", "uncertain", "abstain")
    )


def _filename_for(row: dict[str, Any]) -> str:
    import re

    idx = row["dataset_index"]

    def _slug(text: str, length: int = 40) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
        return s[:length].strip("_")

    slug = _slug(row.get("question") or "", length=40)
    if row.get("display_name") == "Biomni":
        src = (row.get("source_path") or "").rsplit("/", 1)[-1] or "unknown"
        return f"biomi_q{idx:02d}_{src}_{slug}.md"
    model = (row.get("model_name") or "potato").replace("potato-", "").replace("_", "")
    query = "queryNA"
    for part in (row.get("source_path") or "").split("/"):
        m = re.match(r"query_(\d+)_", part)
        if m:
            query = f"query{int(m.group(1)):03d}"
            break
    return f"potato_q{idx:02d}_{query}_{model}_{slug}.md"


def _format_meta_inline(meta_row: dict[str, Any]) -> str:
    """Return the inline meta-judge line that goes right under a criterion."""
    if meta_row["meta_status"] == "abstain":
        return "  - _meta_: abstain (no specific anchor extractable)"
    mark = "✓" if meta_row["agree"] else "✗"
    score = f"{meta_row['score']:.2f}" if meta_row["score"] is not None else "-"
    anchors = ", ".join(meta_row["specific_anchors"]) or "—"
    found = f"{len(meta_row['anchors_found'])}/{len(meta_row['specific_anchors'])}"
    return (
        f"  - _meta_: {meta_row['section']} "
        f"{meta_row['judge_status']} → {meta_row['meta_status']} {mark} "
        f"({score}, {found}: {anchors})"
    )


def _format_response_section(row: dict[str, Any]) -> str:
    idx = row["dataset_index"]
    if row.get("display_name") == "Biomni":
        sources = row.get("source_files") or ["answer.txt"]
        header = (
            f"<!-- biomi q{idx} | source folder: {row.get('source_path')} | "
            f"included files: {', '.join(sources)} -->\n\n"
            f"# Biomi response — Q{idx}: {row.get('question', '').strip()}\n"
        )
    else:
        header = (
            f"<!-- potato q{idx} | source folder: {row.get('source_path')} -->\n\n"
            f"# Potato response — Q{idx}: {row.get('question', '').strip()}\n"
        )
    return header


def _format_judgment_block(judgment: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> str:
    """Render the judge's verdict with the keyword meta-judge result inlined
    immediately under each criterion (right after the evidence line)."""
    if not judgment:
        return "_(no judgment)_\n"
    scores = judgment.get("scores", {}) or {}
    holistic = judgment.get("holistic", {}) or {}
    summary = (judgment.get("summary") or "").strip()
    expected = judgment.get("expected") or []
    prohibited = judgment.get("prohibited") or []

    # Meta rows are emitted in the same order as expected then prohibited, so
    # we can pair them by sequential walk.
    meta_rows = list((meta or {}).get("items") or [])
    meta_iter = iter(meta_rows)

    out = ["## Judge verdict", ""]
    out.append(f"- Final score: {scores.get('final_score', 'n/a')}")
    if "expected_coverage" in scores:
        out.append(f"- Expected coverage: {scores.get('expected_coverage')}")
    if "prohibited_rate" in scores:
        out.append(f"- Prohibited rate: {scores.get('prohibited_rate')}")
    if holistic:
        out.append(f"- Holistic: {', '.join(f'{k}={v}' for k, v in holistic.items())}")
    if meta_rows:
        n_eval = sum(1 for r in meta_rows if r["meta_status"] != "abstain")
        n_dis = sum(1 for r in meta_rows if r["agree"] is False)
        out.append(
            f"- Meta-judge: {n_eval}/{len(meta_rows)} criteria evaluated, "
            f"{n_dis} disagreement(s)"
        )
    if summary:
        out.extend(["", f"> {summary}"])

    def _emit_section(title: str, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        out.extend(["", title, ""])
        for entry in entries:
            criterion = (entry.get("criterion") or "").strip()
            status = entry.get("status", "?")
            evidence = (entry.get("evidence") or "").strip().replace("\n", " ")
            out.append(f"- **[{status}]** {criterion}")
            if evidence:
                out.append(f"  - _evidence_: {evidence}")
            try:
                meta_row = next(meta_iter)
            except StopIteration:
                meta_row = None
            if meta_row is not None:
                out.append(_format_meta_inline(meta_row))

    _emit_section("### Expected criteria", expected)
    _emit_section("### Prohibited criteria", prohibited)

    return "\n".join(out) + "\n"


def main() -> None:
    PER_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in PER_RESPONSE_DIR.glob("*.md"):
        stale.unlink()

    biomni_resp = {
        (r["dataset_index"], r.get("model_name"), r.get("tier")): r
        for r in load_jsonl(PROJECT_ROOT / "runs" / "biomni" / "responses.jsonl")
    }
    biomni_judge = load_jsonl(PROJECT_ROOT / "runs" / "biomni" / "judgments.jsonl")
    potato_resp = {
        (r["dataset_index"], r.get("model_name"), r.get("tier")): r
        for r in load_jsonl(PROJECT_ROOT / "runs" / "potato" / "responses.jsonl")
    }
    potato_judge = load_jsonl(PROJECT_ROOT / "runs" / "potato" / "judgments.jsonl")

    all_resp_rows = list(biomni_resp.values()) + list(potato_resp.values())
    generic_anchors = compute_generic_anchors(all_resp_rows, threshold=0.5)

    # Aggregate counters
    total_criteria = 0
    evaluated_criteria = 0
    agreements = Counter()  # by (judge_status, meta_status)
    per_question_eval: dict[int, int] = defaultdict(int)
    per_question_disagree: dict[int, int] = defaultdict(int)
    anchor_count_distribution: Counter[int] = Counter()  # specific anchors per evaluated criterion
    examples_by_kind: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for runner, judgments, resp_index in (
        ("Biomni", biomni_judge, biomni_resp),
        ("Potato", potato_judge, potato_resp),
    ):
        for j in judgments:
            judgment = j.get("judgment")
            if not isinstance(judgment, dict):
                continue
            key = (j.get("dataset_index"), j.get("model_name"), j.get("tier"))
            resp = resp_index.get(key)
            if not resp:
                continue
            # Recompute final_score under the currently-configured weights so
            # the rendered meta file matches lib/reporting outputs.
            scores = judgment.get("scores")
            if isinstance(scores, dict):
                _recompute_final_score(scores)
            meta = evaluate_one(
                response_text=resp.get("response_text") or "",
                judgment=judgment,
                generic_anchors=generic_anchors,
            )
            for r in meta["items"]:
                total_criteria += 1
                if r["meta_status"] == "abstain":
                    continue
                evaluated_criteria += 1
                anchor_count_distribution[len(r["specific_anchors"])] += 1
                kind = (r["judge_status"], r["meta_status"])
                agreements[kind] += 1
                per_question_eval[j["dataset_index"]] += 1
                if r["agree"] is False:
                    per_question_disagree[j["dataset_index"]] += 1
                # Stash a few representative examples for the summary
                if len(examples_by_kind[kind]) < 2:
                    examples_by_kind[kind].append(
                        {
                            "runner": runner,
                            "model": j.get("model_name"),
                            "dataset_index": j["dataset_index"],
                            "section": r["section"],
                            "criterion": r["criterion"],
                            "specific_anchors": r["specific_anchors"],
                            "anchors_found": r["anchors_found"],
                            "score": r["score"],
                            "judge_status": r["judge_status"],
                            "meta_status": r["meta_status"],
                        }
                    )

            filename = _filename_for(resp)
            body = (
                _format_response_section(resp)
                + "\n---\n\n"
                + _format_judgment_block(judgment, meta)
                + "\n---\n\n## Response text\n\n"
                + resp["response_text"]
            )
            (PER_RESPONSE_DIR / filename).write_text(body, encoding="utf-8")

    n_agree = sum(c for (jr, mr), c in agreements.items() if jr == mr)
    n_partial_compat = sum(
        c
        for (jr, mr), c in agreements.items()
        if jr in ("partial", "unclear") and mr in ("partial", "uncertain", "abstain")
    )
    n_disagree = evaluated_criteria - n_agree - n_partial_compat
    pct_agree = 100 * n_agree / evaluated_criteria if evaluated_criteria else 0.0

    # ---- Summary ----
    lines = ["# Keyword meta-judge summary", ""]
    lines.append(
        f"Cross-checked **{evaluated_criteria} of {total_criteria}** rubric items "
        f"(items with no specific extractable anchor were abstained from)."
    )
    lines.append("")

    # How it works
    lines.append("## How the meta-judge works")
    lines.append("")
    lines.append(
        "It is a pure keyword/anchor matcher — no LLM call. "
        "Implemented in `scripts/keyword_meta_judge.py`, sharing extraction "
        "logic with `scripts/audit_judge_verdicts.py`."
    )
    lines.append("")
    lines.append(
        "**Step 1 — extract anchors from the criterion text.** A regex grabs:"
    )
    lines.append("")
    lines.append("- All-caps acronyms (`WB`, `RNP`, `CBE`, `LNP`, `NHEJ`, …) of length ≥ 2")
    lines.append("- Mixed-case identifiers with embedded digits (`H1299`, `BL21`, `BE4max`, `OCT4B`)")
    lines.append("- Greek-letter strain names (`DH5α`)")
    lines.append("- Two-word capitalized phrases after stripping the leading verb (`Joshua Schiffman`, `Maltose Binding`, `Vincent Lynch`)")
    lines.append("")
    lines.append(
        "**Step 2 — drop generic anchors.** Anchors appearing in more than 50% "
        "of all responses (e.g. `WB`, `RNP`, `Cas9`, `crispr`) are filtered out — "
        "their presence isn't specific evidence for any one criterion. "
        "Criteria left with no specific anchors abstain (no meta-verdict)."
    )
    lines.append("")
    lines.append(
        "**Step 3 — match against the response.** A normalized lookup folds "
        "Greek letters to ASCII (`PKCδ` → `PKCD`) and strips internal hyphens "
        "(`B-27` → `B27`), then checks each anchor against the lowercased "
        "response. A small synonym map covers common acronyms (e.g. "
        "`FLAG` ↔ `flag-tag`/`3xflag`, `MBP` ↔ `maltose binding protein`, "
        "`LIPOFECTAMINE` ↔ `lipofection`)."
    )
    lines.append("")
    lines.append(
        "**Step 4 — derive a meta-verdict** from the fraction of specific "
        "anchors found in the response:"
    )
    lines.append("")
    lines.append("| Section | All anchors found | Some found | None found |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| expected | met | partial | missed |")
    lines.append("| prohibited | violated | uncertain | not_violated |")
    lines.append("")
    lines.append(
        "**Step 5 — compare to the actual judge.** Diagonals on the confusion "
        "matrix below are agreements; off-diagonal cells are disagreements."
    )
    lines.append("")
    lines.append(
        "**Known limitations.** It can't distinguish 'we use X' from 'we don't use X' "
        "(both contain X), so prohibited `not_violated → violated` is a frequent "
        "false alarm. It can't tell that a specific concentration is wrong (e.g. "
        "criterion says 10 µM, response says 7.5 µM — both contain the anchor). "
        "Conceptual criteria with no specific keyword always abstain."
    )
    lines.append("")

    # Keywords-per-criterion distribution
    if anchor_count_distribution:
        max_n = max(anchor_count_distribution.values())
        bar_width = 30
        lines.append("## Anchors per evaluated criterion")
        lines.append("")
        lines.append(
            "Distribution of how many specific anchors the meta-judge extracts "
            "from each evaluated criterion. Most criteria yield 1 anchor; a "
            "long tail with 2+ comes from criteria that name multiple "
            "reagents, gene/cell-line names, or numbered identifiers in a "
            "single sentence."
        )
        lines.append("")
        total_eval = sum(anchor_count_distribution.values())
        lines.append("```")
        lines.append(f"{'count':<7}{'criteria':<10}{'%':<6}")
        for n in sorted(anchor_count_distribution):
            c = anchor_count_distribution[n]
            bar = "█" * int(round(bar_width * c / max_n))
            pct = 100 * c / total_eval
            lines.append(f"{n:<7}{c:<10}{pct:<5.1f}% {bar}")
        lines.append("```")
        lines.append("")

    # Worked examples for each (judge, meta) bucket
    lines.append("## Examples")
    lines.append("")
    lines.append(
        "Two representative rows for each judge/meta bucket. `score` is the "
        "fraction of specific anchors found; `agree` is whether judge and meta "
        "match exactly."
    )
    lines.append("")

    def _ex_block(title: str, kinds: list[tuple[str, str]]) -> None:
        rows: list[dict[str, Any]] = []
        for kind in kinds:
            rows.extend(examples_by_kind.get(kind, []))
        if not rows:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            "| Runner | Q | Section | Judge → Meta | Score | Anchors (found / total) | Criterion |"
        )
        lines.append("| --- | ---: | --- | --- | ---: | --- | --- |")
        for r in rows:
            anchors = (
                f"{len(r['anchors_found'])}/{len(r['specific_anchors'])}: "
                + ", ".join(r['specific_anchors'])
            )
            crit = r['criterion'].replace("|", "\\|")
            if len(crit) > 100:
                crit = crit[:97] + "..."
            score_txt = "-" if r["score"] is None else f"{r['score']:.2f}"
            lines.append(
                f"| {r['runner']} | Q{r['dataset_index']} | {r['section']} | "
                f"{r['judge_status']} → {r['meta_status']} | {score_txt} | {anchors} | {crit} |"
            )
        lines.append("")

    _ex_block(
        "Strict agreement (positive)",
        [("met", "met"), ("violated", "violated")],
    )
    _ex_block(
        "Strict agreement (negative)",
        [("missed", "missed"), ("not_violated", "not_violated")],
    )
    _ex_block(
        "judge=missed, meta=met (most actionable disagreement)",
        [("missed", "met")],
    )
    _ex_block(
        "judge=not_violated, meta=violated (meta-judge false alarm — anchor in 'we don't use X' context)",
        [("not_violated", "violated")],
    )
    _ex_block(
        "judge=partial, meta=met",
        [("partial", "met")],
    )

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Strict agreement (judge_status == meta_status): **{n_agree} ({pct_agree:.1f}%)**")
    lines.append(f"- Compatible partial / unclear: {n_partial_compat}")
    lines.append(f"- Disagreement: **{n_disagree}**")
    lines.append("")

    # Confusion matrix
    judge_statuses = ["met", "partial", "missed", "unclear", "not_violated", "violated"]
    meta_statuses = ["met", "partial", "missed", "violated", "not_violated", "uncertain"]
    seen_judge = sorted({jr for (jr, mr) in agreements})
    seen_meta = sorted({mr for (jr, mr) in agreements})
    lines.append("## Confusion matrix (judge vs meta-judge)")
    lines.append("")
    header = "| judge ↓ / meta → | " + " | ".join(seen_meta) + " | total |"
    sep = "| --- |" + " ---: |" * (len(seen_meta) + 1)
    lines.append(header)
    lines.append(sep)
    col_totals = {m: 0 for m in seen_meta}
    grand = 0
    for jr in seen_judge:
        row_total = 0
        cells = []
        for mr in seen_meta:
            n = agreements.get((jr, mr), 0)
            row_total += n
            col_totals[mr] += n
            grand += n
            cells.append(str(n))
        lines.append(f"| {jr} | " + " | ".join(cells) + f" | {row_total} |")
    lines.append("| **total** | " + " | ".join(str(col_totals[m]) for m in seen_meta) + f" | {grand} |")
    lines.append("")
    lines.append(
        "Diagonal cells (e.g. met/met, missed/missed) are agreements; off-diagonal cells are "
        "disagreements. The meta-judge has no `unclear` state and uses `uncertain` only for "
        "partial-prohibited matches."
    )
    lines.append("")

    # Per-question disagreement chart (text bar)
    lines.append("## Per-question disagreement count")
    lines.append("")
    lines.append("Bar = number of criteria where judge and meta-judge disagree.")
    lines.append("")
    lines.append("```")
    if per_question_disagree:
        max_d = max(per_question_disagree.values())
    else:
        max_d = 0
    bar_width = 30
    for idx in sorted(per_question_eval):
        d = per_question_disagree.get(idx, 0)
        e = per_question_eval[idx]
        bar = "█" * int(round(bar_width * d / max_d)) if max_d else ""
        lines.append(f"Q{idx:02d}  {bar:<{bar_width}}  {d:>3} / {e:<3}  ({100*d/e:.0f}% disagree)")
    lines.append("```")
    lines.append("")

    # Top disagreement questions
    ranked = sorted(per_question_disagree.items(), key=lambda kv: (-kv[1], kv[0]))
    lines.append("## Top disagreement questions")
    lines.append("")
    lines.append("| Q | Disagreements | Evaluated | Disagree % |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for idx, n in ranked[:15]:
        e = per_question_eval[idx]
        lines.append(f"| {idx} | {n} | {e} | {100*n/e:.0f}% |")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"per-question files: {PER_RESPONSE_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  evaluated: {evaluated_criteria}/{total_criteria} criteria")
    print(f"  strict agreement: {n_agree} ({pct_agree:.1f}%)")
    print(f"  disagreements: {n_disagree}")


if __name__ == "__main__":
    main()
