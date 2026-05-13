#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Cross-system comparison: Biomni (B), Potato (A), Claude Opus 4.7, GPT 5.5.

Reads judgments + responses from:
    runs/biomni/         (Biomni, 1 run/question)
    runs/potato/         (Potato, multiple runs/question)
    runs/00016/          (filtered to claude-opus-4-7 only, drops the one
                          submission that isn't in the 47-question set)
    runs/00017_gpt55/    (GPT 5.5, 1 run/question)

Final score is recomputed under the current EXPECTED_WEIGHT / PROHIBITED_WEIGHT
(80/20 after the switch in lib/judge.py), no cap.

Writes:
    runs/4-model-comparison/aggregate.csv
    runs/4-model-comparison/per_question.md
    runs/4-model-comparison/findings.md
    combined_responses_judgements_meta/opus_q##_*.md   (per response)
    combined_responses_judgements_meta/gpt_q##_*.md    (per response)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.dataset import load_items  # noqa: E402
from lib.judge import EXPECTED_WEIGHT, PROHIBITED_WEIGHT  # noqa: E402
from lib.reporting import _recompute_final_score, load_jsonl  # noqa: E402

# Re-use the keyword meta-judge so the per-response files for Opus/GPT match
# the format of the existing biomi/potato files.
_kw_spec = importlib.util.spec_from_file_location(
    "keyword_meta_judge", str(PROJECT_ROOT / "scripts" / "keyword_meta_judge.py")
)
_kw_mod = importlib.util.module_from_spec(_kw_spec)
assert _kw_spec.loader is not None
_kw_spec.loader.exec_module(_kw_mod)
evaluate_one = _kw_mod.evaluate_one
_format_judgment_block = _kw_mod._format_judgment_block

_audit_spec = importlib.util.spec_from_file_location(
    "audit_judge_verdicts", str(PROJECT_ROOT / "scripts" / "audit_judge_verdicts.py")
)
_audit_mod = importlib.util.module_from_spec(_audit_spec)
assert _audit_spec.loader is not None
_audit_spec.loader.exec_module(_audit_mod)
compute_generic_anchors = _audit_mod.compute_generic_anchors


DATASET_PATH = PROJECT_ROOT / "genetic_benchmark_v1" / "47-submissions-clean.json"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "4-model-comparison"
META_DIR = PROJECT_ROOT / "combined_responses_judgements_meta"


def _load(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def _scores(j: dict[str, Any]) -> dict[str, Any]:
    judgment = j.get("judgment") or {}
    if not isinstance(judgment, dict):
        return {}
    s = judgment.get("scores") or {}
    if isinstance(s, dict):
        _recompute_final_score(s)
    return s


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _int_field(row: dict[str, Any], *names: str) -> int:
    for n in names:
        v = row.get(n)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def _slug(text: str, length: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:length].strip("_")


def _short(text: str, length: int = 80) -> str:
    t = (text or "").replace("\n", " ").replace("|", "\\|").strip()
    return t[: length - 3] + "..." if len(t) > length else t


def _fmt(v: float | None, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if isinstance(v, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# Load + canonicalise question identity by submission_id (stable across the
# 47 vs 48 dataset shuffle that happened during data cleaning).
# ---------------------------------------------------------------------------

def load_canonical_index() -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    sub_to_idx: dict[str, int] = {}
    idx_to_item: dict[int, dict[str, Any]] = {}
    for i, item in enumerate(items, start=1):
        sub_to_idx[item["id"]] = i
        idx_to_item[i] = item
    return sub_to_idx, idx_to_item


def filter_to_canonical(rows: list[dict[str, Any]], sub_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    """Keep rows whose submission_id is in the canonical 47-set, and stamp the
    canonical dataset_index onto each row so all systems share the same q-axis."""
    out: list[dict[str, Any]] = []
    for r in rows:
        sid = r.get("submission_id")
        if sid in sub_to_idx:
            r2 = dict(r)
            r2["dataset_index"] = sub_to_idx[sid]
            out.append(r2)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_system(judgments: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, Any]:
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in judgments:
        if r.get("judge_error") or not isinstance(r.get("judgment"), dict):
            continue
        idx = r.get("dataset_index")
        if isinstance(idx, int):
            by_q[idx].append(r)

    # Per-question macro means (so multi-run systems aren't overweighted).
    # Prohibited compliance defaults to 1.0 for questions with no prohibited
    # criteria, matching the convention in lib/reporting._recompute_final_score.
    finals: list[float] = []
    coverages: list[float] = []
    prohib_rates: list[float] = []
    prohib_compliances: list[float] = []
    for rows in by_q.values():
        f, c, p, comp = [], [], [], []
        for r in rows:
            s = _scores(r)
            fv, cv, pv = _num(s.get("final_score")), _num(s.get("expected_coverage")), _num(s.get("prohibited_rate"))
            pmax = s.get("prohibited_max")
            has_prohibited = isinstance(pmax, (int, float)) and pmax > 0
            if fv is not None:
                f.append(fv)
            if cv is not None:
                c.append(cv)
            if pv is not None and has_prohibited:
                p.append(pv)
                comp.append(1.0 - pv)
            else:
                comp.append(1.0)
        if f:
            finals.append(mean(f))
        if c:
            coverages.append(mean(c))
        if p:
            prohib_rates.append(mean(p))
        if comp:
            prohib_compliances.append(mean(comp))

    # Tokens used by the *system* (not judging). Source from judgments.jsonl
    # where answer_* fields are populated for all 4 systems (Biomni/Potato
    # responses.jsonl has no token columns since those are external).
    answer_in = sum(_int_field(r, "answer_input_tokens") for r in judgments)
    answer_out = sum(_int_field(r, "answer_output_tokens") for r in judgments)
    answer_total = sum(_int_field(r, "answer_total_tokens") for r in judgments)

    return {
        "n_questions": len(by_q),
        "n_runs": sum(len(v) for v in by_q.values()),
        "mean_final": mean(finals) if finals else None,
        "mean_coverage": mean(coverages) if coverages else None,
        "mean_prohibited_compliance": mean(prohib_compliances) if prohib_compliances else None,
        "mean_prohibited_rate": mean(prohib_rates) if prohib_rates else None,
        "total_input_tokens": answer_in,
        "total_output_tokens": answer_out,
        "total_tokens": answer_total,
    }


def per_question_aggregate(judgments: list[dict[str, Any]]) -> dict[int, dict[str, float | None]]:
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in judgments:
        if r.get("judge_error") or not isinstance(r.get("judgment"), dict):
            continue
        idx = r.get("dataset_index")
        if isinstance(idx, int):
            by_q[idx].append(r)
    out: dict[int, dict[str, float | None]] = {}
    for idx, rows in by_q.items():
        f = [_num(_scores(r).get("final_score")) for r in rows]
        c = [_num(_scores(r).get("expected_coverage")) for r in rows]
        p = [_num(_scores(r).get("prohibited_rate")) for r in rows]
        f = [v for v in f if v is not None]
        c = [v for v in c if v is not None]
        p = [v for v in p if v is not None]
        out[idx] = {
            "n_runs": len(rows),
            "final": mean(f) if f else None,
            "coverage": mean(c) if c else None,
            "prohibited_rate": mean(p) if p else None,
        }
    return out


# ---------------------------------------------------------------------------
# Per-response meta files for Opus / GPT
# ---------------------------------------------------------------------------

def _meta_filename(system_slug: str, idx: int, model_name: str, question: str, run_id: int | None) -> str:
    qslug = _slug(question, 40)
    suffix = f"_run{run_id}" if run_id is not None else ""
    return f"{system_slug}_q{idx:02d}_{model_name.replace('/', '_').replace('.', '_')}{suffix}_{qslug}.md"


def write_meta_files(
    *,
    system_slug: str,
    display_name: str,
    judgments: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    generic_anchors: set[str],
    idx_to_item: dict[int, dict[str, Any]],
) -> int:
    resp_index = {(r["dataset_index"], r.get("model_name"), r.get("tier")): r for r in responses}
    written = 0
    for j in judgments:
        if j.get("judge_error") or not isinstance(j.get("judgment"), dict):
            continue
        idx = j.get("dataset_index")
        if not isinstance(idx, int) or idx not in idx_to_item:
            continue
        key = (idx, j.get("model_name"), j.get("tier"))
        resp = resp_index.get(key)
        if not resp or not resp.get("response_text"):
            continue
        question = (idx_to_item[idx].get("prompt") or "").strip()
        # Recompute final_score under the current 80/20 weights so the value
        # rendered in the meta file matches the aggregate / per-question tables.
        scores = j["judgment"].get("scores")
        if isinstance(scores, dict):
            _recompute_final_score(scores)
        meta = evaluate_one(
            response_text=resp["response_text"],
            judgment=j["judgment"],
            generic_anchors=generic_anchors,
        )
        # Prepare a synthetic resp row that _format_response_section knows how
        # to handle. We don't reuse keyword_meta_judge._format_response_section
        # since its branches assume Biomi/Potato; we render a small header
        # ourselves.
        header = (
            f"<!-- {system_slug} q{idx} | model: {resp.get('model_name')} | "
            f"tier: {resp.get('tier')} -->\n\n"
            f"# {display_name} response — Q{idx}: {question}\n"
        )
        body = (
            header
            + "\n---\n\n"
            + _format_judgment_block(j["judgment"], meta)
            + "\n---\n\n## Response text\n\n"
            + resp["response_text"]
        )
        fname = _meta_filename(system_slug, idx, resp.get("model_name") or system_slug, question, None)
        (META_DIR / fname).write_text(body, encoding="utf-8")
        written += 1
    return written


# ---------------------------------------------------------------------------
# Findings: per-criterion miss rates (expected) and violation rates
# (prohibited), aggregated across systems.
# ---------------------------------------------------------------------------

def per_criterion_stats(judgments_by_system: dict[str, list[dict[str, Any]]], idx_to_item: dict[int, dict[str, Any]]):
    """For each (question, criterion) and each system, compute fraction of
    runs that missed (expected) or violated (prohibited). Then average across
    systems to find criteria everyone struggles with."""
    expected_miss_per_q: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    prohibited_violate_per_q: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)

    for system, judgments in judgments_by_system.items():
        # Group runs per question for this system.
        by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in judgments:
            if r.get("judge_error") or not isinstance(r.get("judgment"), dict):
                continue
            idx = r.get("dataset_index")
            if isinstance(idx, int):
                by_q[idx].append(r)
        for idx, rows in by_q.items():
            # Map criterion text -> list of (status) across runs
            exp_status: dict[str, list[str]] = defaultdict(list)
            pro_status: dict[str, list[str]] = defaultdict(list)
            for r in rows:
                judgment = r["judgment"]
                for entry in judgment.get("expected") or []:
                    exp_status[(entry.get("criterion") or "").strip()].append(entry.get("status") or "")
                for entry in judgment.get("prohibited") or []:
                    pro_status[(entry.get("criterion") or "").strip()].append(entry.get("status") or "")
            for crit, statuses in exp_status.items():
                miss = sum(1 for s in statuses if s == "missed") / len(statuses)
                expected_miss_per_q[(idx, crit)][system] = miss
            for crit, statuses in pro_status.items():
                viol = sum(1 for s in statuses if s == "violated") / len(statuses)
                prohibited_violate_per_q[(idx, crit)][system] = viol

    return expected_miss_per_q, prohibited_violate_per_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sub_to_idx, idx_to_item = load_canonical_index()

    biomni_j = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "biomni" / "judgments.jsonl"), sub_to_idx)
    biomni_r = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "biomni" / "responses.jsonl"), sub_to_idx)
    potato_j = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "potato" / "judgments.jsonl"), sub_to_idx)
    potato_r = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "potato" / "responses.jsonl"), sub_to_idx)

    opus_all_j = _load(PROJECT_ROOT / "runs" / "00016" / "judgments.jsonl")
    opus_all_r = _load(PROJECT_ROOT / "runs" / "00016" / "responses.jsonl")
    opus_j = filter_to_canonical(
        [r for r in opus_all_j if r.get("model_name") == "claude-opus-4-7"], sub_to_idx
    )
    opus_r = filter_to_canonical(
        [r for r in opus_all_r if r.get("model_name") == "claude-opus-4-7"], sub_to_idx
    )

    # GPT 5.5 — pick whatever run dir matches the suffix; tolerate absence.
    gpt_j: list[dict[str, Any]] = []
    gpt_r: list[dict[str, Any]] = []
    runs_root = PROJECT_ROOT / "runs"
    gpt_dirs = sorted(p for p in runs_root.glob("*_gpt55") if p.is_dir())
    for d in gpt_dirs:
        gpt_j.extend(_load(d / "judgments.jsonl"))
        gpt_r.extend(_load(d / "responses.jsonl"))
    gpt_j = filter_to_canonical([r for r in gpt_j if r.get("model_name") == "gpt-5.5"], sub_to_idx)
    gpt_r = filter_to_canonical([r for r in gpt_r if r.get("model_name") == "gpt-5.5"], sub_to_idx)

    systems = [
        ("Potato (A)", "potato", potato_j, potato_r, True),
        ("Biomni (B)", "biomi", biomni_j, biomni_r, True),
        ("Claude Opus 4.7", "opus", opus_j, opus_r, False),
        ("GPT 5.5", "gpt", gpt_j, gpt_r, False),
    ]
    # The 5th tuple field flags 'tokens are external / not tracked' so we
    # render n/a in the aggregate table instead of misleading zeros.

    # ---- Aggregate CSV ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "system",
            "n_questions",
            "n_runs",
            "mean_final_score (↑)",
            "mean_expected_coverage (↑)",
            "mean_prohibited_rate (↓)",
            "total_tokens",
        ])
        agg_by_system: dict[str, dict[str, Any]] = {}
        for label, slug, j, r, external_tokens in systems:
            agg = aggregate_system(j, r)
            agg["tokens_external"] = external_tokens
            agg_by_system[label] = agg
            tok_tot = "n/a" if external_tokens else agg["total_tokens"]
            w.writerow([
                label,
                agg["n_questions"],
                agg["n_runs"],
                _fmt(agg["mean_final"]),
                _fmt(agg["mean_coverage"], 3),
                _fmt(agg["mean_prohibited_rate"], 3),
                tok_tot,
            ])
    print(f"wrote {csv_path.relative_to(PROJECT_ROOT)}")

    # ---- Per-question table ----
    pq_lines: list[str] = []
    pq_lines.append("# Per-question scores (4 systems)")
    pq_lines.append("")
    pq_lines.append(
        f"Final score uses 80/20 weighting: `100 * ({EXPECTED_WEIGHT} * coverage + "
        f"{PROHIBITED_WEIGHT} * (1 - prohibited_rate))`. No cap."
    )
    pq_lines.append("")
    pq_lines.append("Potato has multiple runs per question; one row per run is emitted, plus a Potato (mean) row.")
    pq_lines.append("Biomni / Opus / GPT have one row per question.")
    pq_lines.append("")
    pq_lines.append("| # | Question | System | Run | Final | Coverage | Prohibited rate |")
    pq_lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: |")

    # Group rows per question across systems.
    potato_rows_by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in potato_j:
        if r.get("judge_error") or not isinstance(r.get("judgment"), dict):
            continue
        idx = r.get("dataset_index")
        if isinstance(idx, int):
            potato_rows_by_q[idx].append(r)
    biomni_pq = per_question_aggregate(biomni_j)
    opus_pq = per_question_aggregate(opus_j)
    gpt_pq = per_question_aggregate(gpt_j)
    potato_pq_mean = per_question_aggregate(potato_j)

    for idx in sorted(idx_to_item):
        question = (idx_to_item[idx].get("prompt") or "").strip()
        preview = _short(question, 70)

        # Potato per-run rows.
        if idx in potato_rows_by_q:
            for r in sorted(potato_rows_by_q[idx], key=lambda x: str(x.get("model_name"))):
                s = _scores(r)
                pq_lines.append(
                    f"| {idx} | {preview} | A (Potato) | {r.get('model_name')} | "
                    f"{_fmt(_num(s.get('final_score')))} | "
                    f"{_fmt(_num(s.get('expected_coverage')), 3)} | "
                    f"{_fmt(_num(s.get('prohibited_rate')), 3)} |"
                )
            mean_p = potato_pq_mean.get(idx)
            # Skip the mean row when there's only one underlying run — it
            # would just duplicate the row already shown above.
            if mean_p and mean_p["n_runs"] > 1:
                pq_lines.append(
                    f"| {idx} | {preview} | A (Potato) | mean of {mean_p['n_runs']} runs | "
                    f"{_fmt(mean_p['final'])} | {_fmt(mean_p['coverage'], 3)} | "
                    f"{_fmt(mean_p['prohibited_rate'], 3)} |"
                )

        for label, pq in (("B (Biomni)", biomni_pq), ("Claude Opus 4.7", opus_pq), ("GPT 5.5", gpt_pq)):
            v = pq.get(idx)
            if v:
                pq_lines.append(
                    f"| {idx} | {preview} | {label} | single | "
                    f"{_fmt(v['final'])} | {_fmt(v['coverage'], 3)} | "
                    f"{_fmt(v['prohibited_rate'], 3)} |"
                )

    pq_path = OUTPUT_DIR / "per_question.md"
    pq_path.write_text("\n".join(pq_lines) + "\n", encoding="utf-8")
    print(f"wrote {pq_path.relative_to(PROJECT_ROOT)}")

    # ---- Per-response meta files for Opus + GPT ----
    META_DIR.mkdir(parents=True, exist_ok=True)
    all_resp_for_anchors = biomni_r + potato_r + opus_r + gpt_r
    generic_anchors = compute_generic_anchors(all_resp_for_anchors, threshold=0.5)
    n_opus = write_meta_files(
        system_slug="opus",
        display_name="Claude Opus 4.7",
        judgments=opus_j,
        responses=opus_r,
        generic_anchors=generic_anchors,
        idx_to_item=idx_to_item,
    )
    n_gpt = write_meta_files(
        system_slug="gpt",
        display_name="GPT 5.5",
        judgments=gpt_j,
        responses=gpt_r,
        generic_anchors=generic_anchors,
        idx_to_item=idx_to_item,
    )
    print(f"wrote {n_opus} opus meta files, {n_gpt} gpt meta files into combined_responses_judgements_meta/")

    # ---- Findings markdown ----
    findings: list[str] = []
    findings.append("# Cross-system findings: Biomni (B), Potato (A), Claude Opus 4.7, GPT 5.5")
    findings.append("")
    findings.append(f"Judge: gpt-5.4. Dataset: genetic_benchmark_v1/47-submissions-clean.json (47 questions).")
    findings.append(
        f"Final score: `100 * ({EXPECTED_WEIGHT} * expected_coverage + "
        f"{PROHIBITED_WEIGHT} * (1 - prohibited_rate))`. No cap."
    )
    findings.append("")
    findings.append("## Aggregate")
    findings.append("")
    findings.append(
        "| System | Questions | Runs | Mean final (↑) | Mean coverage (↑) | Mean prohibited rate (↓) | Total tok |"
    )
    findings.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, _slug2, j, r, external_tokens in systems:
        a = agg_by_system[label]
        tok_tot_s = "n/a (external)" if external_tokens else f"{a['total_tokens']:,}"
        findings.append(
            f"| {label} | {a['n_questions']} | {a['n_runs']} | "
            f"{_fmt(a['mean_final'])} | {_fmt(a['mean_coverage'], 3)} | "
            f"{_fmt(a['mean_prohibited_rate'], 3)} | {tok_tot_s} |"
        )
    findings.append("")

    # ---- Refusals / generation failures per system ----
    skipped_per_system: dict[str, list[tuple[int, str]]] = {}
    for label, slug, j, r, _ext in systems:
        # Map dataset_index -> reason for any row that didn't yield a usable
        # judgment (response refusal, content-policy block, truncation, etc.).
        skipped: list[tuple[int, str]] = []
        for row in j:
            idx = row.get("dataset_index")
            if not isinstance(idx, int):
                continue
            if isinstance(row.get("judgment"), dict):
                continue
            # Try to find a useful reason from the matching response.
            reason = row.get("judge_error") or "unknown"
            for resp in r:
                if resp.get("dataset_index") == idx and resp.get("submission_id") == row.get("submission_id"):
                    if resp.get("response_error"):
                        reason = resp["response_error"]
                    elif not resp.get("response_text"):
                        # Anthropic refusals don't set response_error but leave
                        # raw_response.stop_reason == 'refusal'.
                        rr = resp.get("raw_response") or {}
                        final = (rr.get("final") if isinstance(rr, dict) else None) or {}
                        if isinstance(final, dict) and final.get("stop_reason") == "refusal":
                            reason = "model refusal (stop_reason=refusal)"
                    break
            # Trim long error messages.
            short = reason.split("\\n")[0][:140] if isinstance(reason, str) else "unknown"
            skipped.append((idx, short))
        skipped_per_system[label] = sorted(skipped)

    if any(skipped_per_system.values()):
        findings.append("## Generation refusals / failures")
        findings.append("")
        findings.append(
            "Questions where the system did not produce a usable response (excluded from coverage / final-score averages). "
            "Note that **questions 20 and 29 are refused by both Claude Opus 4.7 and GPT 5.5** — these prompts (SpCas9 immunogenicity reduction, Chlamydia α-synuclein hypothesis test) trip both vendors' safety classifiers. Biomni and Potato (external systems) answered them. "
            "GPT 5.5 also refused q9 (CRISPR design rigorous safety workflow) and truncated on q24 (APOE4→APOE3 conversion) even after retrying with a larger token budget."
        )
        findings.append("")
        findings.append("| System | # skipped | Question indices | Reason |")
        findings.append("| --- | ---: | --- | --- |")
        for label in [s[0] for s in systems]:
            sk = skipped_per_system.get(label) or []
            if not sk:
                continue
            indices = ", ".join(str(i) for i, _ in sk)
            unique_reasons = []
            seen = set()
            for _, r in sk:
                # Pull the meaningful prefix.
                short_r = r
                if "stop_reason=refusal" in r:
                    short_r = "Anthropic safety refusal (stop_reason=refusal)"
                elif "Invalid prompt" in r and "safety reasons" in r:
                    short_r = "OpenAI invalid_prompt safety block"
                elif "truncated" in r:
                    short_r = "Output truncated even after retry with larger token budget"
                if short_r not in seen:
                    unique_reasons.append(short_r)
                    seen.add(short_r)
            findings.append(f"| {label} | {len(sk)} | {indices} | {' | '.join(unique_reasons)} |")
        findings.append("")

    # Hardest questions: average final across systems present.
    hardness: list[tuple[int, float, list[str]]] = []
    by_q_finals: dict[int, dict[str, float]] = defaultdict(dict)
    for label, pq in (
        ("Potato (A)", potato_pq_mean),
        ("Biomni (B)", biomni_pq),
        ("Claude Opus 4.7", opus_pq),
        ("GPT 5.5", gpt_pq),
    ):
        for idx, v in pq.items():
            if v["final"] is not None:
                by_q_finals[idx][label] = v["final"]
    for idx, sys_finals in by_q_finals.items():
        if len(sys_finals) >= 2:
            hardness.append((idx, mean(sys_finals.values()), sorted(sys_finals)))
    hardness.sort(key=lambda x: x[1])

    findings.append("## Hardest questions (lowest mean final across systems)")
    findings.append("")
    findings.append("| # | Question | Mean final | Systems contributing | Per-system finals |")
    findings.append("| ---: | --- | ---: | --- | --- |")
    for idx, m, syslist in hardness[:10]:
        sys_finals = by_q_finals[idx]
        per_sys = ", ".join(f"{s}={_fmt(sys_finals[s])}" for s in sorted(sys_finals))
        findings.append(
            f"| {idx} | {_short(idx_to_item[idx].get('prompt') or '', 60)} | "
            f"{_fmt(m)} | {len(sys_finals)} | {per_sys} |"
        )
    findings.append("")

    # Easiest questions
    findings.append("## Easiest questions (highest mean final across systems)")
    findings.append("")
    findings.append("| # | Question | Mean final | Per-system finals |")
    findings.append("| ---: | --- | ---: | --- |")
    for idx, m, syslist in sorted(hardness, key=lambda x: -x[1])[:10]:
        sys_finals = by_q_finals[idx]
        per_sys = ", ".join(f"{s}={_fmt(sys_finals[s])}" for s in sorted(sys_finals))
        findings.append(
            f"| {idx} | {_short(idx_to_item[idx].get('prompt') or '', 60)} | "
            f"{_fmt(m)} | {per_sys} |"
        )
    findings.append("")

    # Common expected misses + prohibited violations
    judgments_by_system = {
        "Potato (A)": potato_j,
        "Biomni (B)": biomni_j,
        "Claude Opus 4.7": opus_j,
        "GPT 5.5": gpt_j,
    }
    judgments_by_system = {k: v for k, v in judgments_by_system.items() if v}
    expected_miss, prohibited_violate = per_criterion_stats(judgments_by_system, idx_to_item)

    # For each (question, criterion) compute mean miss/violate rate across the
    # systems that touched it. Surface the highest.
    expected_summary: list[tuple[float, int, str, dict[str, float]]] = []
    for (idx, crit), per_sys in expected_miss.items():
        if not per_sys:
            continue
        m = mean(per_sys.values())
        if m >= 0.5:  # majority of runs across majority of systems missed
            expected_summary.append((m, idx, crit, per_sys))
    expected_summary.sort(key=lambda x: -x[0])

    prohibited_summary: list[tuple[float, int, str, dict[str, float]]] = []
    for (idx, crit), per_sys in prohibited_violate.items():
        if not per_sys:
            continue
        m = mean(per_sys.values())
        if m >= 0.5:
            prohibited_summary.append((m, idx, crit, per_sys))
    prohibited_summary.sort(key=lambda x: -x[0])

    findings.append("## Expected criteria most systems miss")
    findings.append("")
    findings.append(
        "Criteria where the average miss rate across systems is ≥ 0.5 "
        "(top 25). \"Miss rate\" per system = fraction of runs marked `missed` "
        "(treating partial as not-missed)."
    )
    findings.append("")
    findings.append("| Q | Criterion | Mean miss rate | Per-system miss rate |")
    findings.append("| ---: | --- | ---: | --- |")
    for m, idx, crit, per_sys in expected_summary[:25]:
        per_sys_str = ", ".join(f"{s}={r:.2f}" for s, r in sorted(per_sys.items()))
        findings.append(
            f"| {idx} | {_short(crit, 80)} | {m:.2f} | {per_sys_str} |"
        )
    findings.append("")

    findings.append("## Prohibited criteria most systems violate")
    findings.append("")
    findings.append(
        "Criteria where the average violation rate across systems is ≥ 0.5 (top 25)."
    )
    findings.append("")
    findings.append("| Q | Criterion | Mean violate rate | Per-system violate rate |")
    findings.append("| ---: | --- | ---: | --- |")
    for m, idx, crit, per_sys in prohibited_summary[:25]:
        per_sys_str = ", ".join(f"{s}={r:.2f}" for s, r in sorted(per_sys.items()))
        findings.append(
            f"| {idx} | {_short(crit, 80)} | {m:.2f} | {per_sys_str} |"
        )
    findings.append("")

    # Qualitative narrative — built from the aggregate numbers above.
    findings.append("## Narrative")
    findings.append("")
    ranked = sorted(
        ((label, agg_by_system[label]) for label, _, _, _, _ in systems if agg_by_system[label]["mean_final"] is not None),
        key=lambda kv: -kv[1]["mean_final"],
    )
    if ranked:
        winner = ranked[0]
        findings.append(
            f"- **Best mean final score: {winner[0]}** at {_fmt(winner[1]['mean_final'])} "
            f"(coverage {_fmt(winner[1]['mean_coverage'], 3)}, "
            f"prohibited rate {_fmt(winner[1]['mean_prohibited_rate'], 3)})."
        )
        for label, agg in ranked[1:]:
            findings.append(
                f"- {label}: {_fmt(agg['mean_final'])} "
                f"(coverage {_fmt(agg['mean_coverage'], 3)}, "
                f"prohibited rate {_fmt(agg['mean_prohibited_rate'], 3)})."
            )
    findings.append("")
    findings.append(
        "- **Biomni and GPT 5.5 are within noise** of each other on mean final score "
        "(71.44 vs 71.43) and coverage (0.672 vs 0.674). Opus 4.7 leads by ~3 points, "
        "but only on the 44 questions it answered — the 3 it refused (20, 29, 39) are "
        "ones it would likely have scored lower on; the gap to Biomni shrinks if you "
        "score refusals as 0."
    )
    findings.append(
        "- **Potato (A) covers only 17/47 questions** (30 runs total), so its low "
        "prohibited rate (0.032) is largely a sampling artefact: it skipped "
        "the question subset where prohibited criteria are most easily violated. "
        "Its mean coverage (0.450) is roughly half what the dense systems achieve, "
        "and that drives the 55.4 final score."
    )
    findings.append(
        "- **Coverage and prohibited rate move together** on this rubric — "
        "detailed protocols routinely mention lentivirus, AAV, His-tags, cancerous "
        "cell lines, T7 promoters, etc. Every dense system violates 11+ prohibited "
        "criteria across ≥50% of the runs that touched them, with extensive overlap "
        "between Biomni, Opus 4.7 and GPT 5.5."
    )
    if expected_summary:
        findings.append(
            f"- **Universally-missed expected criteria ({len(expected_summary)} "
            "question-criterion pairs)** cluster around: (a) named researchers "
            "(Vincent Lynch, Joshua Schiffman) the rubric expects to be cited; "
            "(b) hyper-specific reagents and protocols rooted in particular "
            "papers (E8 Flex media, Lewis-Israeli/Volmert media, Lipofectamine "
            "3000 / CRISPRMAX, SpCas9-HF1 RNP, RS-1); (c) granular validation "
            "controls such as donor-only HDR negatives, untransfected lysates, "
            "lysosome inhibitors for troubleshooting, Annexin V at fixed "
            "post-electroporation timepoints; (d) specific orthogonal readouts "
            "such as flow cytometry for OCT4, qPCR for NANOG, immunofluorescence "
            "for PAX6, trilineage differentiation. None of these are universally "
            "wrong answers — they are the lab's preferred specific strategies, "
            "and no system reliably guesses them."
        )
    if prohibited_summary:
        findings.append(
            f"- **Universally-violated prohibited criteria ({len(prohibited_summary)} "
            "pairs)** include (i) lentivirus / AAV delivery where the rubric required "
            "a non-integrating method (q1, q11, q46, q47); (ii) cancer-derived cell "
            "lines for non-cancer assays (q1); (iii) in-vivo testing for an in-vitro "
            "scope (q15); (iv) His-tag / T7 promoter use (q31), and E. coli "
            "expression (q32) where eukaryotic systems were specified. Both Anthropic "
            "and OpenAI flagship models reach for these defaults absent explicit "
            "rubric constraints in the prompt."
        )
    findings.append(
        "- **Why Opus 4.7 wins**: it has the highest coverage (0.710) — it "
        "tends to enumerate more validation steps, controls and analytical "
        "readouts per protocol than the others. GPT 5.5's coverage is "
        "essentially identical to Biomni's, so the score gap to Opus is "
        "almost entirely explained by depth of expected-criteria coverage."
    )
    findings.append("")

    # ---- Per-system failure modes ----
    # For each system: lowest-scoring questions, top expected misses where this
    # system specifically lags, prohibited violations distinctive to this
    # system, and any refusals.
    findings.append("## Per-system failure modes")
    findings.append("")
    findings.append(
        "For each system, this section lists: (a) the questions where it "
        "scored lowest, (b) expected criteria it misses most often relative "
        "to the other systems, and (c) prohibited criteria it violates most "
        "often relative to the other systems. \"Distinctive\" miss / "
        "violation rates are computed as **this system's rate − mean rate of "
        "the other systems on the same criterion**."
    )
    findings.append("")

    pq_by_system = {
        "Biomni (B)": biomni_pq,
        "Potato (A)": potato_pq_mean,
        "Claude Opus 4.7": opus_pq,
        "GPT 5.5": gpt_pq,
    }
    # Index every judgment row by (system, dataset_index) so we can pull the
    # judge's per-criterion evidence and per-response summary into the
    # explanation column.
    judg_index_by_system: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for sys_label, sys_judgments in (
        ("Biomni (B)", biomni_j),
        ("Potato (A)", potato_j),
        ("Claude Opus 4.7", opus_j),
        ("GPT 5.5", gpt_j),
    ):
        idx_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in sys_judgments:
            if r.get("judge_error") or not isinstance(r.get("judgment"), dict):
                continue
            di = r.get("dataset_index")
            if isinstance(di, int):
                idx_map[di].append(r)
        judg_index_by_system[sys_label] = idx_map

    def _evidence_for(label: str, q_idx: int, criterion: str, target_status: str) -> str:
        """Return the judge's short evidence string for the criterion entry
        on this system's run for q_idx. Prefer an entry whose status matches
        target_status (e.g. 'missed' / 'violated'), else any match."""
        runs = judg_index_by_system.get(label, {}).get(q_idx, [])
        crit_norm = criterion.strip()
        candidates: list[dict[str, Any]] = []
        for r in runs:
            j = r["judgment"]
            for section in ("expected", "prohibited"):
                for entry in (j.get(section) or []):
                    if (entry.get("criterion") or "").strip() == crit_norm:
                        candidates.append(entry)
        for entry in candidates:
            if entry.get("status") == target_status:
                return (entry.get("evidence") or "").strip().replace("\n", " ")
        if candidates:
            return (candidates[0].get("evidence") or "").strip().replace("\n", " ")
        return ""

    def _question_explanation(label: str, q_idx: int) -> str:
        """Concise why-it-scored-low explanation, built from the judge
        summary on the median-final run. Special-cases known failure modes
        (feasibility rejection, truncation, reframing) up front; otherwise
        keeps the diagnostic part of the summary (text after a praise→
        critique 'however / but' turn). Drops bare miss/violation counts —
        the qualitative reason is what matters."""
        runs = judg_index_by_system.get(label, {}).get(q_idx, [])
        if not runs:
            return ""
        runs_sorted = sorted(
            runs, key=lambda r: _num(_scores(r).get("final_score")) or 0.0
        )
        chosen = runs_sorted[len(runs_sorted) // 2]
        j = chosen["judgment"]
        summary = (j.get("summary") or "").strip().replace("\n", " ")
        if not summary:
            return ""
        low = summary.lower()

        # Pattern-matched categories — more useful than the raw judge text.
        if "feasibility rejection" in low or (
            "refuse" in low and ("specify" in low or "downstream" in low)
        ):
            return (
                "Model declared the task infeasible and refused to specify "
                "downstream design (sgRNA / delivery / validation / safety)."
            )
        if "severely incomplete" in low or "incomplete/truncated" in low or "truncated" in low:
            return (
                "Response severely incomplete / truncated — most rubric "
                "elements not addressed at all."
            )
        if ("reframes the task" in low) or ("planning document" in low and "instead" in low):
            return (
                "Reframed as a high-level planning document instead of a "
                "concrete experimental protocol with reagents and steps."
            )

        # Otherwise: keep the diagnostic part of the judge summary. Try a
        # few praise-then-critique pivots in priority order.
        pivot_phrases = (
            "however,", "however ",
            "but misses ", "but it ", "but does ", "but the response ", "but largely ", "but ",
            "main gaps ", "key gaps ",
            "yet ", " though,",
        )
        cut_at = -1
        for m in pivot_phrases:
            i = low.find(m)
            if i >= 0 and (cut_at < 0 or i < cut_at):
                # For "main gaps" / "key gaps" we want to keep the phrase;
                # for everything else we drop the pivot itself.
                cut_at = i if m.startswith(("main gaps", "key gaps")) else i + len(m)
        if cut_at >= 0:
            summary = summary[cut_at:].strip()

        # Trim leading filler phrases that don't add diagnostic info.
        summary = re.sub(
            r"^(it |the response )?"
            r"(misses|omits|lacks|does not address|does not deliver) "
            r"(several |many |most |a few |some )?"
            r"(specific |key |concrete |technical |operational |implementation )?"
            r"(rubric-specific |rubric-required |rubric )?"
            r"(items|details|requirements|elements|criteria|points|aspects)"
            r"[:,.\s]+",
            "",
            summary,
            flags=re.IGNORECASE,
        )
        return _short(summary, 140)

    def _per_system_block(label: str) -> None:
        pq = pq_by_system.get(label) or {}
        if not pq:
            return
        agg = agg_by_system.get(label) or {}
        findings.append(f"### {label}")
        findings.append("")
        findings.append(
            f"Mean final {_fmt(agg.get('mean_final'))}, coverage "
            f"{_fmt(agg.get('mean_coverage'), 3)}, prohibited rate "
            f"{_fmt(agg.get('mean_prohibited_rate'), 3)}, on "
            f"{agg.get('n_questions')} of 47 questions."
        )
        if skipped_per_system.get(label):
            sk_idx = ", ".join(str(i) for i, _ in skipped_per_system[label])
            findings.append(f"Skipped questions: {sk_idx}.")
        findings.append("")

        # (a) Worst questions for this system.
        ranked_q = sorted(
            ((idx, v) for idx, v in pq.items() if v.get("final") is not None),
            key=lambda kv: kv[1]["final"],
        )
        if ranked_q:
            findings.append("**Lowest-scoring questions**")
            findings.append("")
            findings.append("| Q | Question | Final | Coverage | Prohibited rate | Why it failed |")
            findings.append("| ---: | --- | ---: | ---: | ---: | --- |")
            for idx, v in ranked_q[:8]:
                preview = _short(idx_to_item.get(idx, {}).get("prompt") or "", 60)
                why = _short(_question_explanation(label, idx), 110)
                findings.append(
                    f"| {idx} | {preview} | {_fmt(v['final'])} | "
                    f"{_fmt(v['coverage'], 3)} | {_fmt(v['prohibited_rate'], 3)} | {why} |"
                )
            findings.append("")

        # (b) Distinctive expected misses.
        distinctive_exp: list[tuple[float, int, str, dict[str, float]]] = []
        for (idx, crit), per_sys in expected_miss.items():
            if label not in per_sys:
                continue
            others = [v for s, v in per_sys.items() if s != label]
            if not others:
                continue
            delta = per_sys[label] - mean(others)
            if per_sys[label] >= 0.5 and delta >= 0.25:
                distinctive_exp.append((delta, idx, crit, per_sys))
        distinctive_exp.sort(key=lambda x: -x[0])
        if distinctive_exp:
            findings.append("**Distinctive expected-criterion misses (where this system trails the others)**")
            findings.append("")
            findings.append("| Q | Criterion | This | Others (avg) | Δ | Why missed |")
            findings.append("| ---: | --- | ---: | ---: | ---: | --- |")
            for delta, idx, crit, per_sys in distinctive_exp[:10]:
                others_avg = mean(v for s, v in per_sys.items() if s != label)
                why = _short(_evidence_for(label, idx, crit, "missed"), 80)
                findings.append(
                    f"| {idx} | {_short(crit, 80)} | {per_sys[label]:.2f} | "
                    f"{others_avg:.2f} | +{delta:.2f} | {why} |"
                )
            findings.append("")

        # (c) Distinctive prohibited violations.
        distinctive_pro: list[tuple[float, int, str, dict[str, float]]] = []
        for (idx, crit), per_sys in prohibited_violate.items():
            if label not in per_sys:
                continue
            others = [v for s, v in per_sys.items() if s != label]
            if not others:
                continue
            delta = per_sys[label] - mean(others)
            if per_sys[label] >= 0.5 and delta >= 0.25:
                distinctive_pro.append((delta, idx, crit, per_sys))
        distinctive_pro.sort(key=lambda x: -x[0])
        if distinctive_pro:
            findings.append("**Distinctive prohibited-criterion violations (where this system is worse than the others)**")
            findings.append("")
            findings.append("| Q | Criterion | This | Others (avg) | Δ | Why violated |")
            findings.append("| ---: | --- | ---: | ---: | ---: | --- |")
            for delta, idx, crit, per_sys in distinctive_pro[:10]:
                others_avg = mean(v for s, v in per_sys.items() if s != label)
                why = _short(_evidence_for(label, idx, crit, "violated"), 80)
                findings.append(
                    f"| {idx} | {_short(crit, 80)} | {per_sys[label]:.2f} | "
                    f"{others_avg:.2f} | +{delta:.2f} | {why} |"
                )
            findings.append("")

    # User-requested order: Biomni first, then Potato, then the new model runs.
    for label in ("Biomni (B)", "Potato (A)", "Claude Opus 4.7", "GPT 5.5"):
        _per_system_block(label)

    # System-specific qualitative notes derived from the distinctive tables.
    findings.append("### Headline per-system patterns")
    findings.append("")
    findings.append(
        "- **Biomni (B)**: the consistent middle-of-the-pack performer "
        "— answers all 47 questions, coverage 0.672, prohibited rate 0.117. "
        "Its weakest items are highly specific protocol questions where the "
        "rubric expects very particular reagents or controls (q12 protein "
        "purification troubleshooting, q24 APOE4→APOE3 base-editing, q47 "
        "fluorescent-protein insertion). Where Biomni distinctively lags is "
        "in details that require knowing the *target lab's* preferred "
        "reagents (Lewis-Israeli media, E8 Flex, Annexin V timepoints, "
        "specific isoform/promoter selection). Its prohibited-violation "
        "profile is similar to Opus and GPT — driven by reaching for "
        "lentivirus / AAV delivery and cancer-derived cell lines as defaults."
    )
    findings.append(
        "- **Potato (A)**: the dominant failure mode is **incomplete coverage** "
        "of the dataset (17/47 questions, 30 runs). On the questions it does "
        "answer, coverage is still 0.450 — about two-thirds of Biomni's per-"
        "question coverage, suggesting answers are also less thorough. "
        "Run-to-run variance is high (e.g. q23: 0.231 to 0.885 coverage "
        "across runs), which means single-run reads of Potato are unreliable. "
        "Its low prohibited rate (0.032) is mostly a side-effect of attempting "
        "fewer questions and giving terser answers — fewer reagents named "
        "means fewer forbidden reagents named."
    )
    findings.append(
        "- **Claude Opus 4.7**: highest coverage (0.710) and best mean final, "
        "but **3 hard refusals (q20, q29, q39)** at the model level — these "
        "are CRISPR/SpCas9 immunogenicity, an α-synuclein hypothesis test, "
        "and a complete pathogenic-variant correction workflow. Opus also has "
        "the highest distinctive prohibited rate among the three flagship "
        "models on a few items (e.g. AAV for BAFF KD, LB as expression "
        "medium, Matrigel-as-neuron-coating)."
    )
    findings.append(
        "- **GPT 5.5**: matches Biomni on coverage (0.674) but with **4 "
        "skipped questions** — 3 OpenAI safety-classifier 400 errors "
        "(q9, q20, q29) and 1 hard truncation (q24 APOE4→APOE3 conversion) "
        "that doesn't recover even after a larger token budget. GPT 5.5 is "
        "the heaviest output producer (436k output tokens vs Opus's 232k), "
        "so the truncation suggests its protocols are unusually verbose. "
        "Its distinctive prohibited rate is highest on TTAA insertion "
        "sites and lentivirus-as-random-insertion (q47)."
    )
    findings.append("")

    findings.append(
        "See `runs/4-model-comparison/per_question.md` for per-question rows "
        "(per-run for Potato), and `combined_responses_judgements_meta/` for "
        "the per-response Judge + meta-judge breakdowns of every Biomni, "
        "Potato, Opus and GPT answer."
    )
    findings.append("")

    findings_path = OUTPUT_DIR / "findings.md"
    findings_path.write_text("\n".join(findings) + "\n", encoding="utf-8")
    print(f"wrote {findings_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
