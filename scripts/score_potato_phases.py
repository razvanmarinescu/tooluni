"""Bucket potato judgments into the 8 SlotValue phases and write artefacts.

Reads runs/potato/judgments.jsonl, classifies each rubric criterion into one
of the 8 phases (biological_context ... validation), and emits separate
markdown tables and matplotlib charts for expected and prohibited criteria.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES: list[tuple[str, str]] = [
    ("biological_context", "Phase 1"),
    ("target_selection", "Phase 2"),
    ("cell_model", "Phase 3"),
    ("editing_system", "Phase 4"),
    ("construct_design", "Phase 5"),
    ("delivery", "Phase 6"),
    ("safety_and_controls", "Phase 7"),
    ("validation", "Phase 8"),
]
PHASE_NAMES = [p for p, _ in PHASES]
BUCKETS = PHASE_NAMES + ["other"]

SLOT_KEYWORDS: dict[str, list[str]] = {
    "biological_context": [
        "research by", "paradox", "disease", "pathway", "mechanism",
        "prior work", "background", "mentions ",
    ],
    "target_selection": [
        "isoform", "ortholog", "ncbi", "target region", "tss",
        "transcription start site", "target gene", "locus", "retrogene",
    ],
    "cell_model": [
        "cell line", "cell type", "cell model", "ipsc", "ipscs",
        "hpsc", "hpscs", "hesc", "pluripotent", "hek",
        "293ft", "saos-2", "h1299", "cardiomyocyte", "organoid",
        "assembloid", "t cell", "neuron", "astrocyte", "hepatocyte",
        "fibroblast", "myoblast", "matrigel",
        "differentiat", "suspension cell",
    ],
    "editing_system": [
        "crispri", "crispra", "dcas9", "krab", "base edit", "abe ",
        "cbe ", "adenine base editor", "cytidine base editor",
        "prime edit", "cas9", "cas12", "spcas9", "nickase",
        "high-fidelity cas9", "inducible system", "tet-on",
        "overexpress", "transposon", "crisproff", "talen", "zfn",
    ],
    "construct_design": [
        "grna", "sgrna", "homology arm", "donor template", "donor",
        "plasmid donor", "ssodn", "lssdna", "pbs", "primer binding site",
        "silent mutation", "pam", "vector", "backbone", "codon optim",
        "flag-tag", "flag tag", "linker", "reporter", "scaffold",
        "hu6", "pegrna", "nicking sgrna",
    ],
    "delivery": [
        "electroporation", "nucleofection", "nucleofector",
        "lipofect", "lentivir", "aav", "adeno-associated", "lnp",
        "lipid nanoparticle", "transfect", "transduction", "deliver",
        "stereotactic", "intrathecal", "mrna electroporation",
        "rnp electroporation",
    ],
    "safety_and_controls": [
        "off-target", "guide-seq", "circle-seq", "digenome", "karyotyp",
        "genomic stability", "non-targeting", "negative control",
        "positive control", "sgrna control", "empty vector", "donor-only",
        "pluripotency", "trilineage", "toxicity", "translocation",
        "genotoxic", "bystander", "off target", "immunogenicity",
        " control", "ntc ",
    ],
    "validation": [
        "qpcr", "rt-qpcr", "western blot", "western", "flow cytom",
        "facs", "junction pcr", "sanger", "ngs",
        "next-generation sequencing", "deep sequencing", "amplicon",
        "t7e1", "surveyor", "annexin", "colony formation",
        "immunofluoresc", "icc", "microscopy", "reporter assay",
        "luciferase", "rna-seq", "tide", "staining",
    ],
}

EXPECTED_SCORE = {"met": 1.0, "partial": 0.5, "missed": 0.0, "unclear": 0.25}
PROHIBITED_SCORE = {"not_violated": 0.0, "violated": 1.0, "unclear": 0.5}


def classify(text: str) -> str:
    """Pick the best-matching phase for a criterion text. 'other' if no hit."""
    t = text.lower()
    scores: Counter[str] = Counter()
    for slot, patterns in SLOT_KEYWORDS.items():
        for kw in patterns:
            if len(kw) <= 5 and kw.strip().isalpha():
                if re.search(rf"\b{re.escape(kw)}\b", t):
                    scores[slot] += 1
            else:
                if kw in t:
                    scores[slot] += 1
    if not scores:
        return "other"
    best = max(scores.values())
    for slot in PHASE_NAMES:
        if scores[slot] == best:
            return slot
    return "other"


def parse_records(path: Path) -> list[dict]:
    """Pretty-printed JSON-objects-per-record file. Walk braces to split."""
    text = path.read_text()
    records: list[dict] = []
    buf = ""
    depth = 0
    in_str = False
    esc = False
    for ch in text:
        buf += ch
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and buf.strip():
                records.append(json.loads(buf))
                buf = ""
    return records


def aggregate(records: list[dict]) -> tuple[dict, dict]:
    """Return (expected_stats, prohibited_stats) keyed by phase.

    Each entry: {count, mean_score, status_breakdown: Counter}.
    """
    exp_by: dict[str, list[float]] = defaultdict(list)
    exp_status: dict[str, Counter] = defaultdict(Counter)
    pro_by: dict[str, list[float]] = defaultdict(list)
    pro_status: dict[str, Counter] = defaultdict(Counter)

    for rec in records:
        j = rec.get("judgment") or {}
        for c in j.get("expected", []):
            phase = classify(c["criterion"])
            status = c["status"]
            exp_by[phase].append(EXPECTED_SCORE.get(status, 0.0))
            exp_status[phase][status] += 1
        for c in j.get("prohibited", []):
            phase = classify(c["criterion"])
            status = c["status"]
            pro_by[phase].append(PROHIBITED_SCORE.get(status, 0.0))
            pro_status[phase][status] += 1

    def pack(by_phase: dict[str, list[float]], status_by_phase: dict[str, Counter]) -> dict:
        out = {}
        for phase in BUCKETS:
            scores = by_phase.get(phase, [])
            n = len(scores)
            mean = (sum(scores) / n) if scores else None
            if n > 1 and mean is not None:
                var = sum((s - mean) ** 2 for s in scores) / (n - 1)
                se = math.sqrt(var / n)
                ci95 = 1.96 * se
            else:
                ci95 = 0.0
            out[phase] = {
                "count": n,
                "mean": mean,
                "ci95": ci95,
                "status": dict(status_by_phase.get(phase, Counter())),
            }
        return out

    return pack(exp_by, exp_status), pack(pro_by, pro_status)


def write_table(stats: dict, path: Path, *, kind: str) -> None:
    if kind == "expected":
        header = (
            "| Phase | Criteria | Mean coverage | Met | Partial | Unclear | Missed |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        keys = ["met", "partial", "unclear", "missed"]
    else:
        header = (
            "| Phase | Criteria | Mean violation rate | Violated | Unclear | Not violated |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        )
        keys = ["violated", "unclear", "not_violated"]

    rows = []
    for phase in BUCKETS:
        s = stats[phase]
        if s["count"] == 0:
            mean_str = "—"
        else:
            mean_str = f"{s['mean']:.3f}"
        cells = [phase, str(s["count"]), mean_str]
        for k in keys:
            cells.append(str(s["status"].get(k, 0)))
        rows.append("| " + " | ".join(cells) + " |")

    title = "Expected criteria — by phase" if kind == "expected" else "Prohibited criteria — by phase"
    body = (
        f"# {title}\n\n"
        f"Source: `runs/potato/judgments.jsonl` (30 runs, 17 questions).\n"
        f"Phase assignment is keyword-based (`old/protocol_gym/criteria.py`); "
        f"criteria with no keyword match fall into `other`.\n\n"
        f"{header}" + "\n".join(rows) + "\n"
    )
    path.write_text(body)


def write_chart(stats: dict, path: Path, *, kind: str) -> None:
    fig, (ax_score, ax_count) = plt.subplots(1, 2, figsize=(13, 5))

    phases = BUCKETS
    means = [stats[p]["mean"] if stats[p]["mean"] is not None else 0.0 for p in phases]
    cis = [stats[p]["ci95"] for p in phases]
    counts = [stats[p]["count"] for p in phases]

    if kind == "expected":
        score_label = "Mean coverage"
        title_score = "Expected coverage by phase"
    else:
        score_label = "Proportion of true reads"
        title_score = "Prohibited violation rate by phase"

    bar_color = "#4c78a8"
    bars = ax_score.bar(phases, means, color=bar_color,
                        yerr=cis, capsize=4,
                        error_kw={"ecolor": "#333", "elinewidth": 1.2})
    ax_score.set_ylim(0, 1)
    ax_score.set_ylabel(score_label, fontsize=9)
    ax_score.set_title(title_score)
    ax_score.tick_params(axis="x", labelrotation=35)
    for lbl in ax_score.get_xticklabels():
        lbl.set_horizontalalignment("right")
    for bar, m, ci, n in zip(bars, means, cis, counts):
        top = m + ci
        if n == 0:
            ax_score.text(bar.get_x() + bar.get_width() / 2, 0.02, "n=0",
                          ha="center", fontsize=8, color="#888")
        else:
            ax_score.text(bar.get_x() + bar.get_width() / 2, top + 0.02,
                          f"{m:.2f} ±{ci:.2f}\n(n={n})", ha="center", fontsize=8)

    nonzero = [(p, c) for p, c in zip(phases, counts) if c > 0]
    if nonzero:
        labels, sizes = zip(*nonzero)
        wedges, texts, autotexts = ax_count.pie(
            sizes, labels=labels, autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 9},
        )
        ax_count.set_title(f"Criterion distribution across phases (n={sum(sizes)})")
        # nudge biological_context label up by ~1.5x text height to declutter
        # the top of the pie where it sits next to target_selection.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label, txt in zip(labels, texts):
            if label == "biological_context":
                bbox_disp = txt.get_window_extent(renderer=renderer)
                bbox_data = bbox_disp.transformed(ax_count.transData.inverted())
                dy = 1.5 * (bbox_data.y1 - bbox_data.y0)
                x, y = txt.get_position()
                txt.set_position((x, y + dy))
                break
    else:
        ax_count.axis("off")

    fig.suptitle(f"Potato run — {kind} criteria by phase", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    judgments_path = out_dir / "judgments.jsonl"
    records = parse_records(judgments_path)
    print(f"loaded {len(records)} records")

    exp_stats, pro_stats = aggregate(records)

    write_table(exp_stats, out_dir / "phase_expected.md", kind="expected")
    write_table(pro_stats, out_dir / "phase_prohibited.md", kind="prohibited")
    write_chart(exp_stats, out_dir / "phase_expected.png", kind="expected")
    write_chart(pro_stats, out_dir / "phase_prohibited.png", kind="prohibited")

    print("wrote:")
    for p in ["phase_expected.md", "phase_prohibited.md",
              "phase_expected.png", "phase_prohibited.png"]:
        print(f"  {out_dir/p}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/potato")
    main(target.resolve())
