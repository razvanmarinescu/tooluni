#!/Users/razvan/research/evals/tooluni/.venv/bin/python
"""Cross-check the judge's per-criterion verdicts against the response text.

The audit is intentionally narrow: we only flag a verdict when a *specific*
anchor term from the criterion is unambiguously present or absent in the
response. We deliberately skip:

  * the prohibited section's `not_violated` verdicts — the heuristic there
    is too weak (prohibited terms regularly appear in "we don't use X"
    contexts, which my regex can't distinguish from "use X").
  * anchors that turn out to be generic (appear in >50% of all responses).
  * conceptual criteria with no extractable anchor.

What we flag:

       expected     met       + 0 specific anchors found  → POSSIBLE_FALSE_MET
       expected     missed    + ≥1 specific anchor found  → POSSIBLE_FALSE_MISSED
       prohibited   violated  + 0 specific anchors found  → POSSIBLE_FALSE_VIOLATED

This is a triage list, not a verdict.
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

OUT = PROJECT_ROOT / "runs" / "judge_audit.md"


# Anchors that almost any biology answer will contain — ignore them.
ANCHOR_STOPWORDS = {
    "DNA", "RNA", "MRNA", "CDNA", "GDNA", "PCR", "DSB",
    "ATP", "GTP", "CTP", "UTP", "AMP", "ADP",
    "AND", "FOR", "WITH", "FROM", "INTO", "USING", "USE", "USES",
    "THE", "BUT", "NOT", "ALL", "ANY", "ONE", "TWO", "THREE",
    "WT", "KO", "OR", "VS", "NO", "YES", "MOI", "SOC",
    "NCBI",  # too common in biomedical writing — skip to reduce noise
    "AAV", "CMV", "EF1A",  # only flag if very specific
    "P53",  # gene; too common, but watch in synonym map
}


# Synonym/paraphrase map for common acronyms — extended as needed.
SYNONYMS: dict[str, list[str]] = {
    "WB": ["western blot", "western blotting", "wb."],
    "QPCR": ["qpcr", "rt-qpcr", "quantitative pcr", "real-time pcr", "real time pcr", "rtqpcr"],
    "RT-QPCR": ["rt-qpcr", "qpcr", "quantitative pcr"],
    "RT-PCR": ["rt-pcr", "reverse transcription pcr"],
    "NGS": ["next-generation sequencing", "next generation sequencing"],
    "FLAG": ["flag-tag", "flag tag", "3xflag", "flag epitope", "n-flag", "c-flag"],
    "HA": ["ha-tag", "ha tag", "ha epitope"],
    "NLS": ["nuclear localization signal"],
    "NES": ["nuclear export signal"],
    "ELISA": ["elisa"],
    "FACS": ["facs", "flow cytometry", "flow-cytometry"],
    "IHC": ["immunohistochemistry"],
    "IF": ["immunofluorescence"],
    "TALED": ["taled"],
    "DDCBE": ["ddcbe"],
    "ABE": ["adenine base editor", "adenine base editing"],
    "CBE": ["cytidine base editor", "cytosine base editor", "cytidine base editing"],
    "HDR": ["hdr", "homology-directed repair", "homology directed repair"],
    "NHEJ": ["nhej", "non-homologous end joining"],
    "MMEJ": ["mmej"],
    "GFP": ["gfp", "green fluorescent protein"],
    "MCHERRY": ["mcherry"],
    "MCERRY": ["mcerry", "mcherry"],  # common typo in dataset
    "RNP": ["rnp", "ribonucleoprotein"],
    "AAV": ["aav", "adeno-associated virus"],
    "LNP": ["lnp", "lipid nanoparticle"],
    "RFLP": ["rflp"],
    "DNASE": ["dnase", "dnase i"],
    "TRIS": ["tris"],
    "MBP": ["mbp", "maltose binding protein", "maltose-binding protein"],
    "BL21": ["bl21", "bl21(de3)", "rosetta"],
    "ROSETTA": ["rosetta", "bl21"],
    "DH5A": ["dh5α", "dh5alpha", "dh5a"],
    "OCT4": ["oct4", "pou5f1"],
    "BSA": ["bsa", "bovine serum albumin"],
    "PFA": ["pfa", "paraformaldehyde"],
    "IPSC": ["ipsc", "induced pluripotent stem"],
    "HPSC": ["hpsc", "human pluripotent stem"],
    "ESC": ["esc"],
    "PEI": ["pei", "polyethylenimine"],
    "LIPOFECTAMINE": ["lipofectamine", "lipofection"],
    "NUCLEOFECTION": ["nucleofection", "nucleofected", "nucleofector"],
    "ELECTROPORATION": ["electroporation", "electroporated"],
    "INDUCIBLE": ["inducible", "tet-on", "doxycycline-inducible", "tet on"],
    "SUMO": ["sumo", "pet-sumo"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


_LEADING_VERB_RE = re.compile(
    r"^(suggests?|recommends?|proposes?|uses?|using|designs?|mentions?|claims?|states?|"
    r"includes?|adds?|considers?|requires?|advises?|notes?|describes?|"
    r"correctly|incorrectly)\s+",
    flags=re.IGNORECASE,
)


def _strip_leading_verb(text: str) -> str:
    return _LEADING_VERB_RE.sub("", text, count=1).strip()


def extract_anchors(criterion: str) -> list[str]:
    """Pull specific terms out of criterion text. Skips generic stopwords."""
    anchors: list[str] = []
    body = _strip_leading_verb(criterion)

    # Capitalized 2- to 6-letter acronyms (allow trailing digits/hyphens).
    for m in re.finditer(r"\b[A-Z][A-Z0-9-]{1,8}\b", criterion):
        token = m.group(0).strip("-")
        if len(token) < 2:
            continue
        if token.upper() in ANCHOR_STOPWORDS:
            continue
        if token not in anchors:
            anchors.append(token)

    # Mixed-case identifiers with embedded digits (cell lines, strains, primers).
    for m in re.finditer(r"\b[A-Za-z]+\d+[A-Za-z]*\b", criterion):
        token = m.group(0)
        if token.upper() in ANCHOR_STOPWORDS or any(token == a for a in anchors):
            continue
        if len(token) >= 3:
            anchors.append(token)

    # Greek letters used in strain/cell names (e.g. DH5α).
    for m in re.finditer(r"\bDH5[αa]\w*\b", criterion, flags=re.IGNORECASE):
        token = m.group(0)
        if token not in anchors:
            anchors.append(token)

    # Two-word capitalized names (e.g. "Vincent Lynch", "Maltose Binding") —
    # but extract from the body (after dropping a leading "Suggests"/"Uses"/...)
    # so we don't capture verb tokens.
    for m in re.finditer(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", body):
        phrase = m.group(0)
        if phrase.lower().startswith(("the ", "a ", "an ", "and ", "but ", "for ", "if ", "this ", "that ")):
            continue
        # Skip leading-verb-only matches sneaking through the body filter.
        if _LEADING_VERB_RE.match(phrase + " "):
            continue
        if phrase not in anchors:
            anchors.append(phrase)

    return anchors


_GREEK_FOLDING = str.maketrans({
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e",
    "κ": "k", "λ": "l", "μ": "u", "π": "p", "σ": "s",
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E",
})


def _normalize(text: str) -> str:
    """Lowercase, fold Greek letters, drop hyphens. 'B-27' and 'B27' collide,
    'PKCδ' and 'PKCD' collide."""
    return text.translate(_GREEK_FOLDING).lower().replace("-", "")


def anchor_in_response(anchor: str, response_lower: str) -> bool:
    response_norm = _normalize(response_lower)
    if anchor.lower() in response_lower:
        return True
    if _normalize(anchor) in response_norm:
        return True
    canonical = anchor.upper().replace("-", "")
    if canonical in SYNONYMS:
        for syn in SYNONYMS[canonical]:
            if syn in response_lower or _normalize(syn) in response_norm:
                return True
    needle = anchor.lower()
    if needle.endswith("s") and needle[:-1] in response_lower:
        return True
    return False


def audit_one(
    *,
    source_path: str,
    dataset_index: int,
    response_text: str,
    judgment: dict[str, Any],
    runner: str,
    model: str,
    generic_anchors: set[str],
) -> list[dict[str, Any]]:
    response_lower = response_text.lower()
    flags: list[dict[str, Any]] = []

    def _check(section: str, items: list[dict[str, Any]]) -> None:
        for item in items:
            criterion = (item.get("criterion") or "").strip()
            status = (item.get("status") or "").strip()
            if not criterion or status in ("partial", "unclear"):
                continue
            anchors = extract_anchors(criterion)
            if not anchors:
                continue
            # Only consider *specific* anchors — drop the universally-common ones.
            specific_anchors = [a for a in anchors if a.lower() not in generic_anchors]
            if not specific_anchors:
                continue
            present = [a for a in specific_anchors if anchor_in_response(a, response_lower)]
            all_present = len(present) == len(specific_anchors)
            none_present = len(present) == 0

            if section == "expected":
                # FALSE_MISSED: judge said `missed`, but every specific anchor
                # the criterion mentions is in the response. Stronger signal
                # than "any anchor present" — much fewer false positives.
                if status == "met" and none_present:
                    flag_kind = "POSSIBLE_FALSE_MET"
                elif status == "missed" and all_present:
                    flag_kind = "POSSIBLE_FALSE_MISSED"
                else:
                    continue
            else:  # prohibited — only the over-reach case
                if status == "violated" and none_present:
                    flag_kind = "POSSIBLE_FALSE_VIOLATED"
                else:
                    continue

            flags.append(
                {
                    "runner": runner,
                    "model": model,
                    "dataset_index": dataset_index,
                    "source_path": source_path,
                    "section": section,
                    "criterion": criterion,
                    "judge_status": status,
                    "anchors_extracted": anchors,
                    "specific_anchors": specific_anchors,
                    "anchors_found_in_response": present,
                    "judge_evidence": (item.get("evidence") or "").strip(),
                    "flag": flag_kind,
                }
            )

    _check("expected", list(judgment.get("expected") or []))
    _check("prohibited", list(judgment.get("prohibited") or []))
    return flags


def compute_generic_anchors(responses: list[dict[str, Any]], threshold: float = 0.5) -> set[str]:
    """An anchor is 'generic' when it shows up in more than `threshold` of all
    responses. We use the lowercased anchor as the cache key."""
    n = len(responses) or 1
    counts: dict[str, int] = {}
    # Build a small candidate vocabulary by extracting anchors from every
    # criterion appearing in any judgment.
    all_anchors: set[str] = set()
    for r in responses:
        criteria = (r.get("criteria") or {})
        items = (criteria.get("expected_items") or []) + (criteria.get("prohibited_items") or [])
        for item in items:
            for anchor in extract_anchors(item.get("criterion") or ""):
                all_anchors.add(anchor)

    for anchor in all_anchors:
        present = sum(1 for r in responses if anchor_in_response(anchor, (r.get("response_text") or "").lower()))
        if present / n > threshold:
            counts[anchor.lower()] = present
    return set(counts.keys())


def main() -> None:
    biomni_resp = {(r["dataset_index"], r.get("model_name"), r.get("tier")): r for r in load_jsonl(PROJECT_ROOT / "runs" / "biomni" / "responses.jsonl")}
    biomni_judge = load_jsonl(PROJECT_ROOT / "runs" / "biomni" / "judgments.jsonl")
    potato_resp = {(r["dataset_index"], r.get("model_name"), r.get("tier")): r for r in load_jsonl(PROJECT_ROOT / "runs" / "potato" / "responses.jsonl")}
    potato_judge = load_jsonl(PROJECT_ROOT / "runs" / "potato" / "judgments.jsonl")

    all_responses = list(biomni_resp.values()) + list(potato_resp.values())
    generic_anchors = compute_generic_anchors(all_responses, threshold=0.5)

    flags: list[dict[str, Any]] = []
    for runner, judgments, resp_index in (("Biomni", biomni_judge, biomni_resp), ("Potato", potato_judge, potato_resp)):
        for j in judgments:
            judgment = j.get("judgment")
            if not isinstance(judgment, dict):
                continue
            key = (j.get("dataset_index"), j.get("model_name"), j.get("tier"))
            resp = resp_index.get(key)
            if not resp:
                continue
            flags.extend(
                audit_one(
                    source_path=resp.get("source_path") or "",
                    dataset_index=j.get("dataset_index"),
                    response_text=resp.get("response_text") or "",
                    judgment=judgment,
                    runner=runner,
                    model=str(j.get("model_name") or ""),
                    generic_anchors=generic_anchors,
                )
            )

    # Group by question for readability
    by_question: dict[int, list[dict[str, Any]]] = {}
    for f in flags:
        by_question.setdefault(f["dataset_index"], []).append(f)

    counts_by_kind: dict[str, int] = {}
    for f in flags:
        counts_by_kind[f["flag"]] = counts_by_kind.get(f["flag"], 0) + 1

    lines = ["# Judge verdict audit", ""]
    lines.append(
        f"Heuristic anchor-based audit of `runs/{{biomni,potato}}/judgments.jsonl`. "
        f"Total flags: {len(flags)} across {len(by_question)} questions."
    )
    lines.append("")
    lines.append("## Flag counts by kind")
    lines.append("")
    lines.append("| Kind | Count |")
    lines.append("| --- | ---: |")
    for kind, count in sorted(counts_by_kind.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {kind} | {count} |")
    lines.append("")
    lines.append(
        "**POSSIBLE_FALSE_MISSED** is the most actionable: judge marked an "
        "expected criterion as `missed`, but a specific anchor term from "
        "the criterion (acronym / cell-line name / reagent) does appear in "
        "the response. These are the cases most likely to be judge errors "
        "rather than rubric ambiguity."
    )
    lines.append("")

    lines.append("## Per-question flags")
    for idx in sorted(by_question):
        lines.append("")
        lines.append(f"### Q{idx}")
        lines.append("")
        lines.append("| Runner | Model | Section | Status | Flag | Anchors | Found | Criterion | Judge evidence |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for f in sorted(by_question[idx], key=lambda x: (x["runner"], x["model"], x["section"])):
            anchors = ", ".join(f["anchors_extracted"])
            found = ", ".join(f["anchors_found_in_response"]) or "(none)"
            crit = f["criterion"].replace("|", "\\|")
            evidence = (f["judge_evidence"] or "").replace("|", "\\|")[:150]
            lines.append(
                f"| {f['runner']} | {f['model']} | {f['section']} | {f['judge_status']} | {f['flag']} | "
                f"{anchors} | {found} | {crit} | {evidence} |"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} — {len(flags)} flags across {len(by_question)} questions")
    for kind, count in sorted(counts_by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
