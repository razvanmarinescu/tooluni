#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Generate Beamer slide decks for the Potato (A) and Biomni (B) reviews.

Produces:
    presentations/figures/system_comparison.pdf   (matplotlib bar chart)
    presentations/potato_review.tex
    presentations/biomni_review.tex

Then attempts to compile both .tex files with pdflatex (twice, for refs).
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib.reporting import _recompute_final_score, load_jsonl  # noqa: E402

# Reuse the loader / canonicalisation from the comparison builder so the
# numbers in the slides match findings.md exactly.
_cmp_spec = importlib.util.spec_from_file_location(
    "cmp_module", str(PROJECT_ROOT / "scripts" / "build_4_model_comparison.py")
)
_cmp_mod = importlib.util.module_from_spec(_cmp_spec)
assert _cmp_spec.loader is not None
_cmp_spec.loader.exec_module(_cmp_mod)
load_canonical_index = _cmp_mod.load_canonical_index
filter_to_canonical = _cmp_mod.filter_to_canonical
aggregate_system = _cmp_mod.aggregate_system
per_question_aggregate = _cmp_mod.per_question_aggregate

OUT_DIR = PROJECT_ROOT / "presentations"
FIG_DIR = OUT_DIR / "figures"


def _load(p: Path) -> list[dict[str, Any]]:
    return load_jsonl(p) if p.exists() else []


def _scores(j: dict[str, Any]) -> dict[str, Any]:
    judgment = j.get("judgment") or {}
    if not isinstance(judgment, dict):
        return {}
    s = judgment.get("scores") or {}
    if isinstance(s, dict):
        _recompute_final_score(s)
    return s


def _latex_escape(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "→": r"$\rightarrow$",
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "δ": r"$\delta$",
        "Δ": r"$\Delta$",
        "µ": r"$\mu$",
        "≥": r"$\geq$",
        "≤": r"$\leq$",
        "×": r"$\times$",
        "—": "---",
        "–": "--",
        "“": "``",
        "”": "''",
        "‘": "`",
        "’": "'",
        "⅔": "2/3",
        "⅓": "1/3",
        "½": "1/2",
        "°": r"$^{\circ}$",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _short_q(text: str, length: int = 220) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[: length - 3] + "..." if len(t) > length else t


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_comparison_figure(
    agg_by_system: dict[str, dict[str, Any]],
    *,
    target_label: str,
    out_name: str,
) -> Path:
    """Three-panel bar chart for one target system + the flagship LLMs.
    Excludes the *other* external system so each deck doesn't mention it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Build the system order: target first, then the flagship LLMs only.
    flagship = ["Claude Opus 4.7", "GPT 5.5"]
    order = [target_label] + flagship
    short = {
        "Potato (A)": "Potato",
        "Biomni (B)": "Biomni",
        "Claude Opus 4.7": "Opus 4.7",
        "GPT 5.5": "GPT 5.5",
    }
    labels = [short[k] for k in order]
    finals = [agg_by_system[k]["mean_final"] or 0.0 for k in order]
    covs = [(agg_by_system[k]["mean_coverage"] or 0.0) * 100 for k in order]
    pros = [(agg_by_system[k]["mean_prohibited_rate"] or 0.0) * 100 for k in order]
    base_color = "#9aa0a6"
    highlight = {"Potato (A)": "#e07b39", "Biomni (B)": "#3d7eff"}
    colors = [highlight.get(k, base_color) for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, vals, title, ylabel, ylim in (
        (axes[0], finals, "Mean final score", "score (0–100)", (0, 100)),
        (axes[1], covs, "Mean expected coverage", "% covered", (0, 100)),
        (axes[2], pros, "Mean prohibited rate", "% violated", (0, 30)),
    ):
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ylim[1] * 0.015,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        "47-question genetic-engineering benchmark — system comparison",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Best / worst task selection
# ---------------------------------------------------------------------------

def pick_best_worst(judgments: list[dict[str, Any]], idx_to_item: dict[int, dict[str, Any]]):
    """Return ((best_idx, best_runs), (worst_idx, worst_runs)) for a system,
    using per-question final-score means. Each entry is (judgment_row,
    rendered_summary)."""
    pq = per_question_aggregate(judgments)
    candidates = [(idx, v) for idx, v in pq.items() if v["final"] is not None]
    if not candidates:
        return None, None
    candidates_sorted_high = sorted(candidates, key=lambda kv: -kv[1]["final"])
    candidates_sorted_low = sorted(candidates, key=lambda kv: kv[1]["final"])
    best_idx, best_v = candidates_sorted_high[0]
    worst_idx, worst_v = candidates_sorted_low[0]
    # Find the matching judgment rows for these q indices.
    runs_by_idx: dict[int, list[dict[str, Any]]] = {}
    for r in judgments:
        if r.get("judge_error") or not isinstance(r.get("judgment"), dict):
            continue
        runs_by_idx.setdefault(r.get("dataset_index"), []).append(r)
    return (best_idx, best_v, runs_by_idx.get(best_idx, [])), (
        worst_idx,
        worst_v,
        runs_by_idx.get(worst_idx, []),
    )


def summarize_run_judgment(judgment_row: dict[str, Any], k_misses: int = 3, k_violations: int = 2) -> dict[str, Any]:
    j = judgment_row.get("judgment") or {}
    expected = j.get("expected") or []
    prohibited = j.get("prohibited") or []
    n_exp = len(expected)
    n_miss = sum(1 for e in expected if e.get("status") == "missed")
    n_partial = sum(1 for e in expected if e.get("status") == "partial")
    n_pro = len(prohibited)
    n_viol = sum(1 for e in prohibited if e.get("status") == "violated")
    n_met = sum(1 for e in expected if e.get("status") == "met")
    miss_examples = [
        (e.get("criterion") or "").strip()
        for e in expected
        if e.get("status") == "missed"
    ][:k_misses]
    met_examples = [
        (e.get("criterion") or "").strip()
        for e in expected
        if e.get("status") == "met"
    ][:k_misses]
    viol_examples = [
        (e.get("criterion") or "").strip()
        for e in prohibited
        if e.get("status") == "violated"
    ][:k_violations]
    summary = (j.get("summary") or "").strip().replace("\n", " ")
    # Trim to the diagnostic part.
    for marker in ("However,", "However ", "But "):
        if marker in summary:
            summary = summary.split(marker, 1)[1].strip()
            break
    return {
        "summary": summary,
        "n_exp": n_exp,
        "n_met": n_met,
        "n_miss": n_miss,
        "n_partial": n_partial,
        "n_pro": n_pro,
        "n_viol": n_viol,
        "miss_examples": miss_examples,
        "met_examples": met_examples,
        "viol_examples": viol_examples,
        "scores": _scores(judgment_row),
    }


# ---------------------------------------------------------------------------
# Beamer rendering
# ---------------------------------------------------------------------------

PREAMBLE = r"""
\documentclass[aspectratio=169,11pt]{beamer}
\usetheme{default}
\usecolortheme{seagull}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\definecolor{accent}{HTML}{%(accent_hex)s}
\setbeamercolor{frametitle}{fg=accent}
\setbeamercolor{title}{fg=accent}
\title{%(title)s}
\subtitle{%(subtitle)s}
\author{Victoria Gruber}
\date{2026-05-08}
"""


def render_deck(
    *,
    title: str,
    subtitle: str,
    accent_hex: str,
    label: str,
    display_label: str,
    agg_by_system: dict[str, dict[str, Any]],
    best_payload: dict[str, Any],
    worst_payload: dict[str, Any],
    failure_bullets: list[str],
    success_bullets: list[str],
    top_questions: list[tuple[int, dict[str, float | None], str]],
    figure_path: Path,
) -> str:
    e = _latex_escape
    out: list[str] = []
    out.append(PREAMBLE % {"title": e(title), "subtitle": e(subtitle), "accent_hex": accent_hex})
    out.append(r"\begin{document}")
    out.append(r"\frame{\titlepage}")

    # --- Slide 2: Aggregate comparison ---
    out.append(r"\begin{frame}{Aggregate comparison}")
    out.append(r"\centering")
    out.append(rf"\includegraphics[width=0.92\linewidth]{{{figure_path.relative_to(OUT_DIR).as_posix()}}}")
    out.append(r"\vspace{0.4em}")
    out.append(r"\begin{itemize}\setlength{\itemsep}{0pt}")
    a = agg_by_system[label]
    flagship = [k for k in ("Claude Opus 4.7", "GPT 5.5") if agg_by_system.get(k, {}).get("mean_final") is not None]
    others_avg_final = mean(agg_by_system[k]["mean_final"] for k in flagship)
    others_avg_cov = mean(agg_by_system[k]["mean_coverage"] for k in flagship)
    out.append(
        rf"\item {e(display_label)}: mean final {a['mean_final']:.1f}, coverage {a['mean_coverage']:.3f}, "
        rf"prohibited rate {a['mean_prohibited_rate']:.3f}, on {a['n_questions']}/47 questions"
    )
    out.append(
        rf"\item Flagship LLM average (Opus 4.7, GPT 5.5): "
        rf"final {others_avg_final:.1f}, coverage {others_avg_cov:.3f}"
    )
    out.append(
        rf"\item Gap: {a['mean_final'] - others_avg_final:+.1f} points on final score"
    )
    out.append(r"\end{itemize}")
    out.append(r"\end{frame}")

    # --- Slide 3: Where you succeeded ---
    out.append(rf"\begin{{frame}}{{Where {e(display_label)} succeeded}}")
    out.append(r"\begin{itemize}\setlength{\itemsep}{2pt}")
    for b in success_bullets:
        out.append(rf"\item {e(b)}")
    out.append(r"\end{itemize}")
    if top_questions:
        out.append(r"\vspace{0.3em}\textbf{\small Highest-scoring questions}\par")
        out.append(r"\begin{tabular}{rrl}")
        out.append(r"\toprule")
        out.append(r"Q & Final & Topic \\")
        out.append(r"\midrule")
        for idx, v, prompt in top_questions:
            out.append(rf"{idx} & {v['final']:.1f} & {e(_short_q(prompt, 70))} \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
    out.append(r"\end{frame}")

    # --- Slide 4: Where you specifically failed ---
    out.append(rf"\begin{{frame}}{{Where {e(display_label)} specifically failed}}")
    out.append(r"\begin{itemize}\setlength{\itemsep}{2pt}")
    for b in failure_bullets:
        if isinstance(b, dict):
            out.append(rf"\item {e(b['text'])}")
            if b.get("subs"):
                out.append(r"\begin{itemize}\setlength{\itemsep}{0pt}")
                for s in b["subs"]:
                    out.append(rf"\item {e(s)}")
                out.append(r"\end{itemize}")
        else:
            out.append(rf"\item {e(b)}")
    out.append(r"\end{itemize}")
    out.append(r"\end{frame}")

    # --- Slide 4: Best task ---
    bidx, bv, bruns = best_payload["idx"], best_payload["per_q"], best_payload["runs"]
    bsum = best_payload["summary"]
    out.append(r"\begin{frame}{Best task}")
    out.append(rf"\textbf{{Q{bidx}}} (final {bv['final']:.1f}, coverage {bv['coverage']:.3f}, "
               rf"prohibited rate {bv['prohibited_rate']:.3f})\par")
    out.append(rf"\small {e(_short_q(best_payload['prompt'], 320))}\par")
    out.append(r"\vspace{0.3em}\textbf{What went right}")
    out.append(r"\begin{itemize}\setlength{\itemsep}{0pt}")
    out.append(rf"\item {bsum['n_met']}/{bsum['n_exp']} expected criteria met, "
               rf"{bsum['n_viol']}/{bsum['n_pro']} prohibited violated")
    for ex in bsum["met_examples"][:2]:
        out.append(rf"\item {e(_short_q(ex, 110))}")
    out.append(r"\end{itemize}")
    out.append(r"\end{frame}")

    # --- Slide 5: Worst task ---
    widx, wv, wruns = worst_payload["idx"], worst_payload["per_q"], worst_payload["runs"]
    wsum = worst_payload["summary"]
    out.append(r"\begin{frame}{Worst task}")
    out.append(rf"\textbf{{Q{widx}}} (final {wv['final']:.1f}, coverage {wv['coverage']:.3f}, "
               rf"prohibited rate {wv['prohibited_rate']:.3f})\par")
    out.append(rf"\small {e(_short_q(worst_payload['prompt'], 320))}\par")
    out.append(r"\vspace{0.3em}\textbf{What failed}")
    out.append(r"\begin{itemize}\setlength{\itemsep}{0pt}")
    out.append(rf"\item {wsum['n_miss']}/{wsum['n_exp']} expected criteria missed "
               rf"(+{wsum['n_partial']} partial); {wsum['n_viol']}/{wsum['n_pro']} prohibited violated")
    for ex in wsum["miss_examples"][:2]:
        out.append(rf"\item missed: {e(_short_q(ex, 100))}")
    for ex in wsum["viol_examples"][:1]:
        out.append(rf"\item violated: {e(_short_q(ex, 100))}")
    if wsum["summary"]:
        out.append(rf"\item judge: {e(_short_q(wsum['summary'], 130))}")
    out.append(r"\end{itemize}")
    out.append(r"\end{frame}")

    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sub_to_idx, idx_to_item = load_canonical_index()

    biomni_j = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "biomni" / "judgments.jsonl"), sub_to_idx)
    potato_j = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "potato" / "judgments.jsonl"), sub_to_idx)
    biomni_r = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "biomni" / "responses.jsonl"), sub_to_idx)
    potato_r = filter_to_canonical(_load(PROJECT_ROOT / "runs" / "potato" / "responses.jsonl"), sub_to_idx)
    opus_all_j = _load(PROJECT_ROOT / "runs" / "00016" / "judgments.jsonl")
    opus_all_r = _load(PROJECT_ROOT / "runs" / "00016" / "responses.jsonl")
    opus_j = filter_to_canonical([r for r in opus_all_j if r.get("model_name") == "claude-opus-4-7"], sub_to_idx)
    opus_r = filter_to_canonical([r for r in opus_all_r if r.get("model_name") == "claude-opus-4-7"], sub_to_idx)
    gpt_dirs = sorted(p for p in (PROJECT_ROOT / "runs").glob("*_gpt55") if p.is_dir())
    gpt_j_raw, gpt_r_raw = [], []
    for d in gpt_dirs:
        gpt_j_raw.extend(_load(d / "judgments.jsonl"))
        gpt_r_raw.extend(_load(d / "responses.jsonl"))
    gpt_j = filter_to_canonical([r for r in gpt_j_raw if r.get("model_name") == "gpt-5.5"], sub_to_idx)
    gpt_r = filter_to_canonical([r for r in gpt_r_raw if r.get("model_name") == "gpt-5.5"], sub_to_idx)

    agg_by_system = {
        "Potato (A)": aggregate_system(potato_j, potato_r),
        "Biomni (B)": aggregate_system(biomni_j, biomni_r),
        "Claude Opus 4.7": aggregate_system(opus_j, opus_r),
        "GPT 5.5": aggregate_system(gpt_j, gpt_r),
    }

    fig_path_potato = make_comparison_figure(
        agg_by_system, target_label="Potato (A)", out_name="system_comparison_potato.pdf"
    )
    fig_path_biomni = make_comparison_figure(
        agg_by_system, target_label="Biomni (B)", out_name="system_comparison_biomni.pdf"
    )
    print(f"wrote {fig_path_potato.relative_to(PROJECT_ROOT)}")
    print(f"wrote {fig_path_biomni.relative_to(PROJECT_ROOT)}")

    # Per-system best/worst payload assembly.
    def assemble_payload(judgments, label, idx, per_q):
        runs = [
            r for r in judgments
            if r.get("dataset_index") == idx and isinstance(r.get("judgment"), dict)
        ]
        # Pick the run with the highest final for "best", the lowest for "worst".
        # Caller selects which.
        return {
            "idx": idx,
            "per_q": per_q,
            "runs": runs,
            "prompt": (idx_to_item[idx].get("prompt") or ""),
        }

    success_bullets = {
        "Potato (A)": [
            "Best single-run result: Q45 (CRISPRa T-cell exhaustion screen) — final 75.4, coverage 0.692, no prohibited violations.",
            "Q28 (split-DdCBE for mitochondrial G-to-A correction) averages 74.0 across 2 runs, within 14 points of the flagship LLMs on the same item.",
            "Genuinely low prohibited rate (0.032) on the questions attempted — when Potato answers, it rarely reaches for off-rubric reagents.",
            "Multi-run consistency on some items (Q6 across 4 runs all land 56–68, Q25 across 3 runs all 56–66).",
        ],
        "Biomni (B)": [
            "Perfect score on Q28 (split-DdCBE for ND4): 10/10 expected criteria met, 0 prohibited violated.",
            "10 questions at final ≥ 85, including Q4 (89.1), Q17 (89.1), Q41 (96.0), Q22 (90.5), Q19 (89.4).",
            "Strong at enumerating standard validation readouts (Western blot, qPCR, IF, FACS, lineage markers, off-target sequencing).",
            "Outperforms the flagship LLMs on questions that reward broad protocol thoroughness over single-reagent specificity (Q41, Q19, Q22).",
        ],
    }

    failure_bullets = {
        "Potato (A)": [
            {
                "text": "Refused to specify downstream tasks when no valid target could be confirmed:",
                "subs": [
                    "Q39 (HBB ABE correction) — ruled the ABE chemistry incompatible with the requested A-to-G reversion, then declined to specify sgRNA, delivery, validation, and safety package.",
                    "Q10 run 3 (TwinPE 1 kb knock-in into hepatocyte safe harbour) — sequence retrieval failed for SHS231/CLYBL, so pegRNA design, predicted-junction modelling, and downstream assays were all skipped.",
                ],
            },
            "On the questions answered, mean expected coverage is 0.450, ~⅔ of what the flagship LLMs (Opus 4.7, GPT 5.5) achieve on the same items.",
            {
                "text": "High run-to-run variance: e.g. Q23 coverage spans 0.231–0.885 across 4 runs.",
                "subs": [
                    "Same pattern on Q10: run 3 abandoned the workflow at sequence retrieval (final 40), while run 4 produced a complete TwinPE design under stated assumptions (final 60).",
                ],
            },
            "Consistent gaps on standard validation steps — Western blot for dCas9-KRAB (Q3), pegRNA chemical modifications (Q6), Brachyury flow cytometry (Q8).",
            "Low prohibited rate (0.032) is partly an artefact of terser answers and the refused subset, not careful constraint following.",
        ],
        "Biomni (B)": [
            "All 47 questions answered, but coverage trails Opus 4.7 by 0.04 (0.672 vs 0.710).",
            "Lab-specific reagent gaps: misses RS-1 (Q2), 30 °C cold-shock for big-protein synthesis (Q12, Q16), CRE-LOX BAFF KD (Q11).",
            "Doesn't recognise rubric prohibitions on lab-specific media — uses mTeSR and E8 Flex where Lewis-Israeli/Volmert-only was required (Q5, Q30).",
            "Distinctive prohibited slips: Matrigel-for-neurons (Q43), B27 for NPCs (Q44), AAV at AAVS1 safe-harbour for NGN2 (Q43), primary hepatocytes (Q40).",
            "Strong on questions that reward enumeration of validation steps (Q4, Q17, Q28); weakest on troubleshooting protocols that hinge on a single specific reagent change (Q12).",
        ],
    }

    decks = {
        "potato": {
            "label": "Potato (A)",
            "display_label": "Potato",
            "title": "Potato evaluation review",
            "subtitle": "47-question genetic-engineering benchmark, vs flagship LLMs",
            "accent_hex": "C2540F",
            "judgments": potato_j,
            "figure": fig_path_potato,
        },
        "biomni": {
            "label": "Biomni (B)",
            "display_label": "Biomni",
            "title": "Biomni evaluation review",
            "subtitle": "47-question genetic-engineering benchmark, vs flagship LLMs",
            "accent_hex": "1F4FB0",
            "judgments": biomni_j,
            "figure": fig_path_biomni,
        },
    }

    for slug, cfg in decks.items():
        per_q = per_question_aggregate(cfg["judgments"])
        # Best = highest final; Worst = lowest final.
        ranked = sorted(
            ((idx, v) for idx, v in per_q.items() if v["final"] is not None),
            key=lambda kv: kv[1]["final"],
        )
        worst_idx, worst_v = ranked[0]
        best_idx, best_v = ranked[-1]

        def _pick_run(idx: int, prefer: str) -> dict[str, Any]:
            rows = [
                r for r in cfg["judgments"]
                if r.get("dataset_index") == idx and isinstance(r.get("judgment"), dict)
            ]
            if not rows:
                return {}
            rows_sorted = sorted(
                rows, key=lambda r: (_scores(r).get("final_score") or 0.0)
            )
            return rows_sorted[-1] if prefer == "best" else rows_sorted[0]

        best_run = _pick_run(best_idx, "best")
        worst_run = _pick_run(worst_idx, "worst")

        best_payload = {
            "idx": best_idx,
            "per_q": best_v,
            "runs": [best_run] if best_run else [],
            "prompt": idx_to_item[best_idx].get("prompt") or "",
            "summary": summarize_run_judgment(best_run) if best_run else {
                "summary": "", "n_exp": 0, "n_met": 0, "n_miss": 0, "n_partial": 0,
                "n_pro": 0, "n_viol": 0, "miss_examples": [], "met_examples": [],
                "viol_examples": [], "scores": {},
            },
        }
        worst_payload = {
            "idx": worst_idx,
            "per_q": worst_v,
            "runs": [worst_run] if worst_run else [],
            "prompt": idx_to_item[worst_idx].get("prompt") or "",
            "summary": summarize_run_judgment(worst_run) if worst_run else {
                "summary": "", "n_exp": 0, "n_met": 0, "n_miss": 0, "n_partial": 0,
                "n_pro": 0, "n_viol": 0, "miss_examples": [], "met_examples": [],
                "viol_examples": [], "scores": {},
            },
        }

        # Top N questions for the success slide (highest final, exclude the
        # single chosen "best task" so the same row doesn't appear twice).
        top_q = []
        for idx, v in sorted(per_q.items(), key=lambda kv: -(kv[1]["final"] or 0.0)):
            if v["final"] is None or idx == best_idx:
                continue
            top_q.append((idx, v, idx_to_item[idx].get("prompt") or ""))
            if len(top_q) >= 4:
                break

        tex = render_deck(
            title=cfg["title"],
            subtitle=cfg["subtitle"],
            accent_hex=cfg["accent_hex"],
            label=cfg["label"],
            display_label=cfg["display_label"],
            agg_by_system=agg_by_system,
            best_payload=best_payload,
            worst_payload=worst_payload,
            failure_bullets=failure_bullets[cfg["label"]],
            success_bullets=success_bullets[cfg["label"]],
            top_questions=top_q,
            figure_path=cfg["figure"],
        )
        tex_path = OUT_DIR / f"{slug}_review.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print(f"wrote {tex_path.relative_to(PROJECT_ROOT)}")

        # Compile twice with pdflatex (silent).
        for _ in range(2):
            res = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(OUT_DIR),
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(f"!! pdflatex failed for {slug}:")
                print(res.stdout[-2000:])
                break
        pdf_path = OUT_DIR / f"{slug}_review.pdf"
        if pdf_path.exists():
            print(f"compiled {pdf_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
