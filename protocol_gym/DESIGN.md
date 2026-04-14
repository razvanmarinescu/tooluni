# Protocol Gym: RL Environment for Gene-Editing Protocol Design

## 1. Overview

Protocol Gym is a reinforcement learning environment where an LLM agent designs
gene-editing experimental protocols by progressively filling a structured
protocol document. The agent has access to tools (web search, ToolUniverse,
code execution, molecular simulation) and must decide **which slot to work on
next** and **what tool to use** to gather the information needed.

The environment is episodic: each episode presents a gene-editing question from
the benchmark, and the agent builds a protocol step by step until it submits.
Reward is computed from the expert rubric (expected + prohibited criteria).

```
                    +-----------+
                    |  Question |
                    +-----------+
                          |
                          v
               +--------------------+
               |  Empty Protocol    |  <-- initial state
               |  (all slots None)  |
               +--------------------+
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
     [use_tool]    [fill_slot]    [revise_slot]   <-- actions
            |             |             |
            v             v             v
               +--------------------+
               | Partial Protocol   |  <-- intermediate states
               | (some slots filled)|
               +--------------------+
                          |
                          v
                      [submit]         <-- terminal action
                          |
                          v
                 +----------------+
                 | Rubric Scoring |
                 +----------------+
```

---

## 2. The Protocol State

### 2.1 Design philosophy: structured slots, free-form values

After analyzing all 588 expected criteria and 190 prohibited criteria across the
48 benchmark questions, we find that criteria span a predictable set of
**protocol phases** but demand **highly specific, open-ended answers** within
each phase (e.g., "RS-1 at 10-20 uM", "junction PCR with one primer outside the
homology arm"). 

Therefore:

- **Slots are categorical** (fixed set of protocol phases). This gives the RL
  agent a tractable action space for "which part of the protocol to work on."
- **Slot values are free-form text + optional structured metadata**. This lets
  the agent express the specificity the rubric demands. We do NOT constrain
  values to enums because:
  1. The 48 questions span too many modalities (CRISPR KO, CRISPRi, base
     editing, prime editing, overexpression, protein purification, organoid
     culture) for any fixed enum to cover.
  2. Criteria test for specific reagent names, concentrations, temperatures,
     and timing that cannot be enumerated in advance.
  3. Forcing enums would make the environment unable to distinguish a good
     answer from a great one.

However, each slot carries **structured annotations** alongside the free text to
enable cheaper intermediate reward signals without running the full LLM judge.

### 2.2 Protocol slots

The protocol state `S` has **8 top-level slots**, derived from the natural
structure of gene-editing protocols and validated against the criterion
categories in the benchmark:

```python
@dataclass
class ProtocolState:
    """The full state visible to the agent at each step."""

    # --- Fixed per episode (set at reset, never changed by agent) ---
    question: str                    # The gene-editing prompt
    clarity: dict[str, str]          # claritySelections from the benchmark
    episode_budget: int              # Max steps remaining

    # --- Mutable slots (the agent fills these) ---
    hypothesis_generation:SlotValue  # Phase 0
    biological_context: SlotValue    # Phase 1
    target_selection: SlotValue      # Phase 2
    cell_model: SlotValue            # Phase 3
    editing_system: SlotValue        # Phase 4
    construct_design: SlotValue      # Phase 5
    delivery: SlotValue              # Phase 6
    safety_and_controls: SlotValue   # Phase 7
    validation: SlotValue            # Phase 8

    # --- Agent working memory ---
    scratchpad: str                  # Free-form notes from tool calls
    tool_history: list[ToolCall]     # Record of all tools used
    steps_taken: int                 # Counts toward budget
```

Each `SlotValue` is:

```python
@dataclass
class SlotValue:
    """One slot in the protocol."""
    text: str | None              # Free-form content (the actual protocol text)
    confidence: float             # Agent's self-assessed confidence [0, 1]
    sources: list[str]            # URLs, paper DOIs, tool names that informed this
    revision_count: int           # How many times this slot has been rewritten
    last_updated_step: int        # Step number of most recent edit
```

### 2.3 Slot definitions and what they capture

| # | Slot | What the agent writes here | Rubric criteria this maps to (examples) |
|---|------|---------------------------|----------------------------------------|
| 1 | **biological_context** | The biological question being addressed, disease background, relevant prior work, key references | "Mentions research by Vincent Lynch", "Correctly identifies Peto's Paradox" |
| 2 | **target_selection** | Which gene(s)/locus/lncRNA/protein to target, isoform choice, rationale for target | "Suggest using NCBI to check TP53 isoforms", "Targets sgRNA within -50 to +200 bp of TSS" |
| 3 | **editing_system** | Which editing modality (CRISPR-Cas9, CRISPRi/a, base editor, prime editor, transposon, overexpression, RNAi, etc.) and why | "Justifies use of CRISPRi (dCas9-KRAB) to avoid indels in non-coding elements", "ABE choice" |
| 4 | **construct_design** | gRNA sequences/design rules, donor templates, homology arm lengths, vector backbone, pegRNA design, reporter constructs | "Homology arms of 500-1000 bp", "Silent mutations to prevent re-cutting", "Optimizes PBS length" |
| 5 | **delivery** | Delivery method, reagents, timing, cell handling during delivery | "Selects electroporation for PE mRNA", "RNP delivery", concentrations, timing protocols |
| 6 | **cell_model** | Cell type/line, culture conditions, organism, differentiation protocol if relevant | "Suggest using H1299", "p53-null human cells", iPSC maintenance, organoid assembly |
| 7 | **validation** | Assays to confirm editing, functional readouts, quantification methods, phenotypic assays | "Junction PCR", "RT-qPCR", "flow cytometry", "Western blot", "NGS", colony formation, karyotyping |
| 8 | **safety_and_controls** | Off-target analysis, controls (negative, positive, non-targeting sgRNA), toxicity monitoring, genomic stability, prohibited approaches avoided | "GUIDE-seq", "non-targeting sgRNA control", "donor-only negative control", karyotyping, pluripotency checks |

### 2.4 Why these 8 slots and not more/fewer

**Not fewer**: Collapsing (e.g., merging delivery + cell_model) would lose the
ability to give per-phase reward signals. The rubric criteria naturally cluster
into these 8 groups -- we verified this by tagging all 588 expected criteria.

**Not more**: Finer granularity (e.g., separate slots for "gRNA design" vs.
"donor template" vs. "vector") would make the action space needlessly large for
questions that don't involve all sub-slots (e.g., CRISPRi has no donor
template). The free-text value inside each slot handles sub-structure naturally.

### 2.5 Slot value format -- why free text, not enums

Consider the delivery slot. Across the 48 questions, delivery methods include:

- Electroporation (Lonza 4D-Nucleofector, Neon, Bio-Rad GenePulser Xcell)
- Nucleofection (specific programs for iPSCs vs. T cells vs. neurons)
- Lipofection (Lipofectamine 2000, 3000, RNAiMAX)
- Lentiviral transduction (with MOI specifications)
- AAV (serotype selection: AAV9 for CNS, AAV8 for liver, etc.)
- mRNA electroporation
- RNP electroporation
- Intrastromal injection, stereotactic injection, intrathecal delivery
- Naked plasmid + selection

An enum would need 30+ entries and still miss edge cases. Worse, the rubric
doesn't just test "which method" -- it tests specifics like "use RNP to avoid
prolonged Cas9 expression" or "electroporate PE mRNA for high efficiency in
hard-to-transfect primary myoblasts." These are free-text judgments.

**However**, to enable cheap intermediate rewards, we extract lightweight
**tags** from each slot value using a small classifier (keyword matching or a
fast LLM call). For example, after the agent writes in the delivery slot, we
extract:

```python
tags = {
    "method": "electroporation",       # normalized keyword
    "format": "RNP",                   # what's being delivered
    "device": "Lonza 4D-Nucleofector", # if mentioned
    "cell_handling": "Y-27632 pre-treatment",
}
```

These tags feed into the fast reward estimator (Section 4.2) but are NOT part of
the agent's action space -- the agent always writes free text.

---

## 3. Action Space and Tool Use

### 3.1 Action space (MDP-level)

The action space has exactly **three actions**:

| Action | Description |
|--------|-------------|
| `fill_slot(slot_name, text, confidence, sources)` | Write content into an empty slot |
| `revise_slot(slot_name, text, confidence, sources)` | Overwrite an existing slot (increments revision_count) |
| `submit()` | Terminal action -- finalize the protocol for scoring |

These are the only actions the RL algorithm sees and assigns credit to. Each
action transitions the `ProtocolState` in a meaningful, reward-relevant way.

### 3.2 Tools are NOT actions -- they are part of the agent's policy

Tools (`web_search`, `tooluniverse`, `run_code`, `run_simulation`) are **not in
the action space**. They are part of the **agent's internal reasoning loop**
that runs *between* RL actions. Conceptually:

```
┌─────────────────────────────────────────────────────────────┐
│  One RL step                                                │
│                                                             │
│   observation (ProtocolState)                               │
│        │                                                    │
│        v                                                    │
│   ┌──────────────────────────────────────┐                  │
│   │  LLM policy (inner loop)             │                  │
│   │                                      │                  │
│   │   think -> call tool -> observe      │                  │
│   │   think -> call tool -> observe      │                  │
│   │   think -> call tool -> observe      │                  │
│   │   ...                                │                  │
│   │   emit: fill_slot("delivery", "...") │                  │
│   └──────────────────────────────────────┘                  │
│        │                                                    │
│        v                                                    │
│   action (fill_slot / revise_slot / submit)                 │
│        │                                                    │
│        v                                                    │
│   reward, next observation                                  │
└─────────────────────────────────────────────────────────────┘
```

This is the same pattern as your current Tier-3 evaluation: the LLM loops
through tool calls until it emits a final text answer. Here, the "final text
answer" is replaced by one of the three MDP actions.

**Why this framing is better than making tools part of the action space:**

1. **Credit assignment**: The RL algorithm should not have to assign credit to
   individual tool calls -- we want it to assign credit to the slot content
   that gets written. Tool choice is a tactical decision that the LLM policy
   handles internally.
2. **Standard LLM RL pattern**: This matches how RLHF / RL on tool-using LLMs
   is typically done (GRPO, RLAIF, etc.) -- the action is the final structured
   output, not the intermediate chain-of-thought.
3. **Cleaner MDP**: The state space stays compact. Tool results live in the
   `scratchpad` (which IS part of the state), but the action space stays at 3.

**What tools look like to the agent**: tools are exposed the same way as in
your current `AnswerRunner.generate()` -- as MCP / function-calling tools the
LLM invokes during generation. The LLM's inner tool-use loop continues until
it emits a valid `{fill_slot | revise_slot | submit}` action, at which point
the RL step completes.

### 3.3 Tool budget and cost (still tracked)

Even though tools are not MDP actions, their use is tracked and costed:

- Each RL step has an **inner tool budget** (e.g., max 10 tool calls before the
  agent is forced to emit a protocol action).
- The total number of tool calls in the step is folded into that step's reward
  as `R_tool` (Section 4.3). The RL algorithm therefore *does* learn to
  control tool use, but it does so through the lens of "how much tool use was
  needed to produce this slot fill," rather than by scoring each call
  individually.
- Total cost accounting (tokens, wall time, dollars) is accumulated across the
  whole episode for reporting.

### 3.4 Tool catalog

| Tool | Purpose | Examples |
|------|---------|----------|
| `web_search(query)` | Find papers, protocols, reagent datasheets | "RS-1 concentration iPSC HDR", "AAV9 serotype CNS tropism" |
| `tooluniverse(tool, args)` | Call any of the 1000+ ToolUniverse tools (existing MCP) | PubChem compound lookup, OpenTargets gene-disease, UniProt protein info |
| `run_code(python)` | Execute Python locally for sequence analysis, calculations, data wrangling | Biopython for gRNA design, primer Tm calculation, reading VCF files |
| `run_simulation(config)` | Invoke a domain-specific simulator wrapper (see 3.5) | Cas-OFFinder off-target search, ESMFold structure prediction, docking |

### 3.5 What `run_simulation` means concretely

These simulators run **locally on the user's laptop** (or on a lightweight
compute backend), and their results **do inform what the agent writes into
slots**. They are wrapped as deterministic Python functions that the LLM
invokes with a structured config. Example simulations:

| Simulator | What it does | Which slot it typically informs |
|-----------|--------------|--------------------------------|
| **Cas-OFFinder / CRISPOR** (local binary) | Scans a target genome for off-target sites for a given gRNA sequence, returns list of potential off-targets with mismatch counts | `safety_and_controls`, `construct_design` |
| **ESMFold** (local inference, ~few GB GPU) | Predicts 3D protein structure from sequence in seconds | `target_selection` (for isoform choice), `construct_design` (for tag placement) |
| **AlphaFold2** (local if GPU available, else API) | High-accuracy protein structure prediction | Same as ESMFold but higher quality |
| **AutoDock Vina** (local binary) | Molecular docking of small molecules to protein pockets | `construct_design` (inducible systems -- will doxycycline bind the engineered TetR?), safety assessment of small-molecule enhancers |
| **ViennaRNA / NUPACK** (local Python) | RNA secondary structure prediction (folding, minimum free energy) | `construct_design` (gRNA scaffold folding, pegRNA 3' extension folding, stability of PBS-RT) |
| **Primer3 / MELTING** (local Python) | Primer Tm, hairpin, dimer prediction | `validation` (designing junction-PCR and qPCR primers) |
| **PyRosetta** (local Python, heavier) | Protein design, mutation scoring, ddG calculation | `editing_system` (when designing mini-Cas9 variants), `construct_design` |
| **Off-target scoring (MIT / Doench)** (local Python) | Compute on-target efficiency and off-target scores for candidate gRNAs | `construct_design` (ranking gRNA candidates) |
| **Cell-count / dilution math** (just Python) | Serial dilution plate planning, seeding density calculations | `delivery`, `cell_model` |
| **Short MD relaxation** (OpenMM, local CPU-minutes) | Brief energy minimization / thermal equilibration to sanity-check a designed construct | `construct_design` (rare; only for protein engineering questions) |

**How results inform slot filling**: after a simulation returns, its structured
output appears in the `scratchpad` (e.g., "Cas-OFFinder found 3 off-target
sites with <=3 mismatches for guide X: sites at chr1:123, chr5:456, chr11:789").
The LLM then uses that information when composing its next slot write --
producing, say, a `safety_and_controls` value that says "we selected guide X
from a panel of 5 candidates; Cas-OFFinder screening against hg38 found 3
potential off-target sites with <=3 mismatches, which we will validate by
amplicon deep sequencing," and lists the simulator call as a source.

**Not all simulators need to be installed up front.** We recommend a tiered
rollout:

- **Tier A (ship day 1)**: Primer3, ViennaRNA, Cas-OFFinder, Python numerics.
  All of these are lightweight, install in seconds, and cover most of what the
  48 benchmark questions actually need.
- **Tier B (ship when needed)**: ESMFold, AutoDock Vina, PyRosetta. These are
  heavier but the environment can gate them behind a config flag so episodes
  only use them when a question type demands it.
- **Tier C (optional)**: AlphaFold2, OpenMM MD. Expensive; use only for a
  small subset of protein-engineering questions.

### 3.6 Example: one RL step with internal tool use

Agent sees a state where `construct_design` is empty for an HDR knock-in
question. Internally it may do:

```
[LLM] thought: I need to design a donor template for GAPDH C-terminal GFP tagging.
[LLM] tool_call: run_code("from Bio import Entrez; fetch GAPDH mRNA sequence...")
[LLM] observation: 1425 bp mRNA, stop codon at position 1008
[LLM] tool_call: run_simulation({"type": "cas_offinder", "guide_region": "...TGA...", "genome": "hg38"})
[LLM] observation: 2 candidate gRNAs identified, best has 0 off-targets <=3mm
[LLM] tool_call: web_search("RS-1 concentration iPSC HDR donor")
[LLM] observation: 10-20 uM RS-1 reported in PMID:28983828
[LLM] emit RL action: fill_slot("construct_design", "Use dsDNA plasmid donor with 800 bp homology arms flanking the eGFP insert. sgRNA targets 3 bp upstream of GAPDH stop codon (sequence: ...); Cas-OFFinder finds 0 off-targets with <=3 mismatches in hg38. Include silent mutations in the PAM to prevent re-cutting...", confidence=0.8, sources=["cas_offinder", "PMID:28983828"])
```

The RL algorithm sees one action (`fill_slot`) and one step reward. It does
not try to assign credit to individual tool calls.

---

## 4. Reward Functions

### 4.1 Terminal reward (primary signal)

Computed at `submit()` using two parallel stages -- a **rubric score** and a
**verification score** -- combined into the final terminal reward.

The rubric score is what the existing eval pipeline already produces: "did
you cover what the experts listed?" The verification score is new: "is what
you said actually true?" Both are necessary because the rubric cannot anticipate
every hallucinated fact, implausible concentration, or inconsistent claim the
agent might emit, while the verifiers cannot substitute for the expert
judgement encoded in the rubric.

#### 4.1.1 Rubric score (unchanged from current eval)

```
expected_coverage = sum(weight_i * score_i) for each expected criterion
    where score_i in {met=1.0, partial=0.5, missed=0.0, unclear=0.25}

prohibited_rate = sum(weight_j * score_j) for each prohibited criterion
    where score_j in {violated=1.0, not_violated=0.0, unclear=0.5}

prohibited_compliance = 1.0 - prohibited_rate

R_rubric = 0.8 * expected_coverage + 0.2 * prohibited_compliance
```

Capped at 0.74 if any prohibited criterion is violated. The judge
(**Gemini 3.1 Flash-Lite Preview**, fixed) receives the concatenation of all
non-empty slot values as the "model response." This is the same model family
as the fast scorer (§4.2), but invoked with a different, more elaborate
prompt designed for end-of-episode rubric grading across all criteria at
once rather than per-criterion checks.

**Cost**: one LLM judge call per episode. At Gemini 3.1 Flash-Lite Preview
pricing ($0.25/M input, $1.50/M output), a terminal judgment with ~3000
input tokens (concatenated slots + full rubric) and ~1500 output tokens
(structured JSON with per-criterion evidence) is ~$0.003 per episode.

**Note on using the same model at both tiers**: reusing Gemini 3.1
Flash-Lite for both the per-step fast scorer and the terminal judge has
three practical advantages:

1. **Cost**: terminal judging is ~50x cheaper than using a frontier model
   like gpt-5.4. For 10k training episodes, terminal cost drops from
   ~$500 to ~$30.
2. **Consistency**: step and terminal scores come from the same model,
   reducing the chance that an RL policy "solves" the fast scorer but
   fails the terminal judge due to model disagreement.
3. **Operational simplicity**: one API dependency, one provider, one
   pricing sheet, one set of rate limits.

The tradeoff is that Flash-Lite is less capable than gpt-5.4 at holistic
grading of long multi-section responses. This is mitigated by (a) giving
the terminal judge a more careful prompt with explicit evidence
requirements per criterion, (b) validating the judge against a
human-graded sample before training starts (§8 item 2), and (c) keeping
gpt-5.4 or Claude Opus available as an **audit judge** that spot-checks
~5% of training episodes to detect drift between Flash-Lite and a
stronger grader.

#### 4.1.2 Verification score (new)

The rubric-only reward has four known blind spots:

1. **Hallucinated facts outside the rubric.** Criteria check that the model
   mentions H1299; they do not catch a stray false claim like "HEK293 is
   p53-null" or a fabricated PMID citation.
2. **Quantitative plausibility.** Rubrics check that RS-1 is mentioned; they
   do not check that the concentration is sane ("RS-1 at 500 uM" vs. "10 uM").
3. **Internal consistency of designed artifacts.** If the agent emits a gRNA
   sequence, pegRNA design, or PCR primer, nothing in the rubric checks
   whether it actually targets the claimed locus, folds correctly, or has
   plausible Tm.
4. **Cross-entity consistency.** "Use HEK293 as a p53-null line" is
   internally inconsistent. Rubric does not flag this unless a specific
   criterion forbids HEK293 for p53 work.

`R_verify` addresses these via a deterministic, detection-based stage. It
starts at **1.0** and loses credit only when a verifier finds a concrete
error. It **never awards credit for unverified claims**, so the agent cannot
reward-hack by flooding the protocol with easy-to-verify trivia.

```
R_verify = max(0.0, 1.0 - sum(severity_k * violation_k))
```

where `violation_k ∈ {0, 1}` is whether verifier `k` flagged a problem and
`severity_k ∈ [0, 1]` is its configured weight.

**Tiered verifier catalog**:

| Tier | Verifier | What it detects | Implementation | Severity |
|------|----------|-----------------|----------------|----------|
| Cheap | **Citation existence** | Fabricated PMIDs / DOIs | NCBI `esummary` / CrossRef API | 0.15 per fake citation |
| Cheap | **Reagent concentration sanity** | 10x-out-of-range working concentrations | Regex-parse `<reagent> at <X> uM\|nM\|mg/mL`; look up against curated table (PubChem SID + hand-curated ranges for ~30 recurring small molecules in the corpus: RS-1, M3814, Y-27632, doxycycline, puromycin, etc.) | 0.10 per violation |
| Cheap | **Cell line property check** | False cell line claims ("HEK293 is p53-null") | Cellosaurus API for genotype/phenotype of named line | 0.15 per false claim |
| Cheap | **Gene name resolution** | Made-up or wrong-species gene symbols | NCBI Gene / HGNC lookup | 0.10 per unresolved symbol |
| Medium | **gRNA target verification** | gRNA sequence does not target the claimed locus, or off-target count contradicts model's claim | Cas-OFFinder locally against hg38 / target genome | 0.20 per wrong-target gRNA; 0.10 per off-target miscount |
| Medium | **Primer sanity** | Egregious Tm mismatch, hairpin, primer-dimer | Primer3 / MELTING locally | 0.05 per broken primer |
| Medium | **RNA folding for pegRNA / gRNA scaffold** | MFE structure disrupts scaffold or PBS-RTT pairing | ViennaRNA locally | 0.10 per broken design |

**Not included (rejected as overkill for reward purposes)**:

- End-to-end simulated protocol execution (stochastic kinetics of
  transfection/editing/selection). Interesting research; signal-to-noise
  too low for a reward in this budget.
- ESMFold / AlphaFold structure prediction on every designed construct.
  Expensive, slow, rarely needed as a reward signal (but the agent can
  still invoke these as tools during generation -- see §3.5).
- Live unstructured literature search. Non-deterministic, duplicates what
  targeted API checks do more reliably.

**Parseability is a prerequisite, not a reward signal.** If the agent writes
prose with no extractable PMIDs, sequences, concentrations, or cell line
names, the verifiers find nothing to check and `R_verify` stays at 1.0. This
is intentional: we do not want to penalize the agent for being cautious
about verifiable claims. We only penalize demonstrably wrong claims.

**Cost**: per-episode, roughly 1-3 s for the cheap tier plus 10-30 s if the
medium tier fires. Verifier calls are parallelizable -- all cheap-tier
checks run concurrently via async API calls.

#### 4.1.3 Why not give the judge itself tool access?

A natural alternative is to give the rubric judge a tool-calling loop so it
can search the web / call ToolUniverse / run code while scoring. We
explicitly reject this path for three reasons:

1. **Non-deterministic rewards.** Web search results change; the same
   response scores differently across days. RL training is already
   high-variance; adding reward-stochasticity hurts sample efficiency and
   makes training-run diffs meaningless.
2. **Auditability.** A judge with unbounded tool use can justify the same
   score with different evidence each run. You cannot confidently compare
   two training runs' reward signals or debug a regression.
3. **Duplicates what targeted verifiers do better.** Deterministic API
   calls to PubMed / Cellosaurus / local simulators are strictly more
   reliable than "LLM decides to search and hopefully finds the right
   result." Verifier tools are fixed, typed, cached, and unit-testable;
   judge-initiated tool use is none of those.

The judge stays deterministic and rubric-focused. Verification is a
separate, structured stage.

#### 4.1.4 Combined terminal reward

```
R_terminal = alpha * R_rubric + beta * R_verify
```

with starting weights `alpha = 0.7, beta = 0.3`. Rationale:

- Rubric carries expert-encoded priorities and is our ground truth for "what
  a good protocol looks like." It should dominate.
- Verification is strictly detection of error, starts at 1.0, and only
  moves downward. A weight of 0.3 on it means a fully hallucinated
  protocol (all verifiers trip) loses at most 0.3 from terminal reward --
  substantial but not dominant. Enough pressure to deter hallucination
  without overriding rubric signal when the agent genuinely captures the
  expert priorities and makes one verifiable slip.
- Prohibited-violation cap (0.74) from §4.1.1 still applies to the
  combined reward.

Weights are tunable. Monitor: if `R_verify` is consistently near 1.0 across
training (no detected violations), either the agent learned to avoid
verifiable claims entirely (watch for vagueness) or the verifier catalog
is under-specified. If `R_verify` is consistently near 0, the agent is
hallucinating heavily and `beta` may be too small.

#### 4.1.5 Reference data for verifiers

Several verifiers in §4.1.2 need curated reference data that is not produced
by an API at query time. This subsection specifies how that data is
sourced, what it looks like on disk, and how it is maintained.

**Reagent concentration reference table**

The concentration sanity verifier needs working ranges for the ~30
recurring small molecules in the corpus (RS-1, M3814, Y-27632,
doxycycline, puromycin, blasticidin, hygromycin, doxorubicin, etoposide,
Accutase, Lipofectamine variants, etc.). There is **no single public
dataset** with "recommended working concentration in cellular experiments"
covering all reagent classes we care about -- different classes live in
different databases. We therefore do a **hybrid pull**:

| Reagent class | Source | Access |
|---|---|---|
| Research probes (RS-1, M3814, Y-27632, etc.) | [Chemical Probes Portal](https://www.chemicalprobes.org/) | Expert-reviewed recommended in-cell concentrations. No public API / bulk download as of this writing; either request a data dump (cite the 2024 NAR paper's stated open-access stance) or scrape the ~15 probe pages we specifically need. |
| Research probes (broader coverage, potency only) | [Probes & Drugs Portal](https://www.probes-drugs.org/download) | Bulk download: CSV, XLSX, SDF, PostgreSQL / SQLite dumps. Targets & Activities file has potency data. License: CC BY-SA 4.0. No API. |
| Biochemical potency (IC50, EC50, Ki, Kd) | [ChEMBL](https://www.ebi.ac.uk/chembl/api/data/docs) | REST API + `chembl_webresource_client` Python package. Free, no auth. Note: biochemical != cellular working concentration; use as a sanity bound (working conc should be >= biochemical IC50, <1000x IC50 or selectivity is suspect). |
| Selection antibiotics (puromycin, blasticidin, hygromycin) | Hand-curated from established ranges | 1 Python dict entry per reagent per cell type (iPSCs, T cells, primary, etc.). Well-established; ~5 lines per reagent. |
| Dissociation reagents (Accutase, TrypLE) | N/A -- remove from verifier | Working concentration is not the failure mode for these; drop from scope. |
| Established cytotoxics (doxorubicin, etoposide) | [DrugBank](https://www.drugbank.ca/) + ChEMBL | DrugBank for clinical ranges; ChEMBL for research cellular IC50. |

**Effort estimate**: ChEMBL API pull script (~2 h) + Probes & Drugs XLSX
download and parse (~30 min) + Chemical Probes Portal scrape for the 15
specific probes we need (~3 h) + hand-filling antibiotics and cytotoxics
(~1-2 h). **Total: ~1 day**, with ~80% of the data coming from maintained
public databases rather than ad-hoc literature reading.

**On-disk artifact**: `protocol_gym/reagent_ranges.yaml` with schema:

```yaml
- name: RS-1
  aliases: [RAD51-stimulatory compound 1, RAD51 activator]
  cas: 312756-74-4
  working_concentration:
    min_uM: 5
    max_uM: 20
    typical_uM: 10
  context:
    cell_types: [iPSC, hPSC, HEK293, primary]
    application: HDR enhancement
  sources:
    - chemical_probes_portal: <URL>
    - chembl: CHEMBL269732
  biochemical_ic50_uM: 8  # for sanity floor; from ChEMBL
  notes: Combination with NHEJ inhibitor (M3814) common.
```

The YAML is loaded once at verifier init, kept in version control, and
re-validated via the verifier coverage audit (§8 item 8) whenever it
changes.

**Other reference data** used by verifiers is fetched live from public
APIs at query time (PubMed / CrossRef for citations, Cellosaurus for cell
lines, HGNC / NCBI Gene for gene names) and does not require local
curation, only response caching.

### 4.2 Intermediate reward: slot-level criterion matching (shaped reward)

To provide denser signal without waiting for terminal reward, we compute a
**fast approximate reward** after each `fill_slot` or `revise_slot` action.

#### Step 1: Pre-classify criteria to slots (done once per episode at reset)

Each expected/prohibited criterion is mapped to the protocol slot it most
naturally evaluates. This mapping is computed once at episode start using a
lightweight classifier:

```python
SLOT_KEYWORDS = {
    "biological_context": ["research by", "paradox", "hypothesis", "disease", "pathway", "mechanism"],
    "target_selection":   ["gene", "isoform", "locus", "TSS", "target", "NCBI", "ortholog"],
    "editing_system":     ["CRISPRi", "base edit", "prime edit", "Cas9", "Cas12", "dCas9", "modality"],
    "construct_design":   ["gRNA", "sgRNA", "pegRNA", "homology arm", "donor", "plasmid", "vector", "PBS", "silent mutation"],
    "delivery":           ["electroporation", "nucleofection", "lipofect", "lentivir", "AAV", "RNP", "delivery", "transfect"],
    "cell_model":         ["cell line", "iPSC", "hPSC", "HEK", "neuron", "cardiomyocyte", "organoid", "culture", "differentiat"],
    "validation":         ["qPCR", "Western", "flow cytom", "FACS", "junction PCR", "NGS", "sequenc", "assay", "stain"],
    "safety_and_controls":["off-target", "GUIDE-seq", "control", "karyotyp", "toxicity", "safety", "non-targeting", "prohibited"],
}
```

For criteria that match multiple slots or none, fall back to an LLM
classification call (cached per question, so this is a one-time cost).

**Empirical note on LLM fallback necessity.** We ran the keyword classifier
above over all 588 expected criteria from the 48 benchmark questions (see
`protocol_gym/analyze_slot_misses.py`). **~30% of criteria failed to match
any slot-specific keyword** and would fall through to a `global` bucket. Many
of these are genuinely cross-slot ("ensure sufficient biological replicates",
"provide publication references") but others are just vocabulary our keyword
list misses. For a real deployment, the LLM fallback is not an edge case --
it must classify roughly one-third of criteria. Budget: one
Gemini 3.1 Flash-Lite Preview call per unmatched criterion at `reset()`,
cached by criterion text across all episodes for the same question
(~$0.0003/question amortized at $0.25/M input, $1.50/M output).

#### Step 2: After each slot write, score the matched criteria

After the agent writes to slot `k`, we evaluate ONLY the criteria assigned to
slot `k`. This scorer must read for **meaning, with polarity awareness** --
not just surface keyword overlap. Keyword matching is tempting because it is
cheap, but it is strictly wrong for this task: "Use H1299 cells" and "Don't
use H1299, it's unsuitable" both contain the keyword "H1299" at equal weight
and embed at >0.85 cosine similarity, yet only the first satisfies the
criterion "Suggest using H1299 cell type." Surface overlap cannot
distinguish correct from contradicting claims.

We therefore implement the fast scorer as a **small-LLM call**
(Gemini 3.1 Flash-Lite Preview) that returns the same status ontology as
the terminal judge (§4.1.1):

```python
def fast_criterion_check(criterion: Criterion, slot_value: str,
                         question: str) -> CriterionJudgment:
    """Read the slot value for meaning and judge the criterion.

    Returns one of {met, partial, missed, inapplicable} with a short
    evidence string. Same ontology as the terminal judge, so scores
    translate directly.
    """
    prompt = f"""You are evaluating whether a slot in a gene-editing protocol
satisfies a specific rubric criterion. Be polarity-aware: if the slot argues
AGAINST doing what the criterion asks, that is MISSED, not MET.

QUESTION CONTEXT: {question}

SLOT CONTENT:
{slot_value or "[empty]"}

CRITERION TO CHECK:
{criterion.text}

Return JSON: {{"status": "met|partial|missed|inapplicable",
               "evidence": "<one short sentence>"}}

- met: the slot clearly and correctly satisfies the criterion.
- partial: slot addresses the criterion but with caveats, missing details, or only tangentially.
- missed: slot does not satisfy the criterion (including cases where it argues against it).
- inapplicable: the criterion's premise does not apply to this slot's content (use for prereq-failure cases; see §4.2 Step 2b).
"""
    resp = gemini_flash_lite_call(prompt, max_output_tokens=60, temperature=0.0)
    return parse_judgment(resp)


STATUS_TO_SCORE = {
    "met":          1.0,
    "partial":      0.5,
    "missed":       0.0,
    "inapplicable": None,   # excluded from the denominator; see prereq gating
}
```

**Why this is the right design, not just keyword + embedding:**

1. **Polarity awareness**: the LLM sees "Don't use H1299" and scores MISSED,
   not MET. A keyword or cosine check cannot do this.
2. **Concept-level judgment**: "the right target is selected" requires reading
   the claimed target and comparing it to what the criterion asks for. The
   LLM can do this directly; keyword matching cannot.
3. **Consistent ontology with the terminal judge**: the same `met/partial/
   missed/inapplicable` vocabulary appears at both step-level and
   terminal-level, and both tiers use the same model
   (Gemini 3.1 Flash-Lite Preview) with different prompts. Step rewards
   and terminal rewards are directly comparable -- no artifact of using
   different scoring models or scoring schemes.
4. **Handles prereq gating natively**: `inapplicable` returns None and the
   criterion is excluded from the denominator, which is exactly what the
   prereq-gate mechanism in §4.2 Step 2b wants.

**Cost**: Gemini 3.1 Flash-Lite Preview at $0.25/M input + $1.50/M output,
~200 input tokens + 20 output tokens per check ≈ **$0.00008 per criterion**
. A slot write typically
triggers 1-3 mapped criteria, so ~$0.0002 per step. Across ~30 steps per
episode, this is **~$0.007 per episode**. For 10k training episodes, total
fast-scorer cost is **~$70**. If prompt caching is enabled (Gemini's
context-cache tier at $0.025/M cached input), the scorer cost drops further
because the judge instructions and the question context are reused across
every criterion check within the same episode. The free tier is also an
option for non-production experimentation (zero cost, but outputs are used
to improve Google's products -- not appropriate for sensitive data).

**Pre-filter for efficiency (optional)**: keyword + embedding overlap *is*
useful as a cheap pre-filter to skip obviously-unrelated criteria without
calling the LLM. Specifically:

```python
def should_skip_llm_check(criterion, slot_value) -> bool:
    if not slot_value:
        return True   # empty slot -> trivially MISSED, no call needed
    keyword_overlap = keyword_match(criterion, slot_value)
    embed_sim = cosine(embed(criterion.text), embed(slot_value))
    return keyword_overlap == 0 and embed_sim < 0.30
```

When this returns True, we assign MISSED without an LLM call. Empirically
(to be validated during build) this cuts Gemini Flash-Lite calls by
~40-50% because most slot writes don't touch most criteria. The pre-filter
can only produce MISSED judgments -- it never promotes a weak match to MET
without LLM confirmation -- so it does not introduce the polarity errors
that killed the original keyword-only design.

**Caching**: judgments are cached on `(criterion.text, slot_value)` pairs
within an episode. If the agent revises a different slot, unchanged slots'
criteria do not re-bill. If the agent revises the same slot with identical
content (unlikely but possible), we return the cached judgment for free.

The slot-level reward is:

```
R_slot(k) = sum(weight_i * status_score_i * prereq_gate_i)
              for criteria assigned to slot k where status_i != "inapplicable"
          - lambda_revision * revision_count(k)    # penalize excessive rewriting
```

where:
- `status_score_i` comes from `STATUS_TO_SCORE` applied to the fast-scorer
  judgment (met=1.0, partial=0.5, missed=0.0; inapplicable criteria are
  excluded from the sum entirely, same convention as §4.1.1).
- `prereq_gate_i` is defined in Step 2b.

Note: the fast scorer and the terminal judge both use Gemini 3.1 Flash-Lite
Preview with the same status ontology, so `R_step` deltas computed at
write-time are on the same scale as the eventual `R_rubric` contribution
at submit -- and from the same model, which further reduces the chance of
step/terminal disagreement. This means intermediate reward is a true
lower-variance estimate of terminal reward, not an incommensurable
heuristic.

#### Step 2b: Prerequisite gating (cross-slot conditioning)

**Problem**: many criteria are phase-local in *where they live* but depend on
content in *other* slots being coherent. Example: the criterion "Performs
junction PCR with one primer outside the homology arm" lives in `validation`
but is only meaningful if `construct_design` actually describes a donor
template with homology arms. If `construct_design` says "CRISPRi-KRAB" (no
donor template at all), awarding credit for mentioning junction PCR rewards
nonsense -- the criterion's premise doesn't apply.

Without gating, the agent can *reward-hack* by sprinkling keyword-rich
validation text regardless of whether the upstream construct justifies it.

**Mechanism**: each criterion carries an optional **prerequisite predicate**
computed at mapping time. Before the fast scorer awards credit, it checks that
the prerequisite holds in the current state:

```python
@dataclass
class Criterion:
    text: str
    slot: str                     # which slot it lives in (§4.2 Step 1)
    weight: float
    prereq: PrereqPredicate | None  # must hold for credit to count
```

A `PrereqPredicate` is a cheap lambda over `ProtocolState` expressed as a
small set of composable checks:

```python
prereq = HasKeyword(slot="construct_design", any_of=["donor", "homology arm", "ssODN"])
prereq = HasKeyword(slot="editing_system", any_of=["Cas9", "base edit", "prime edit"])
prereq = And(prereq_A, prereq_B)
```

Prerequisites are generated at `reset()` by the same classifier that assigns
criteria to slots. The rule is: for each criterion, look at the keywords that
made it land in its slot, and infer which upstream slot content those keywords
presuppose. Where the rule is ambiguous, the LLM fallback (same call used for
slot assignment) returns a prereq too.

**Gating application**:

```python
def fast_criterion_check_gated(
    crit: Criterion, state: ProtocolState
) -> CriterionJudgment:
    # Short-circuit: if prereq is missing, mark inapplicable without any
    # LLM call. This is cheaper than asking Gemini Flash-Lite to figure it out.
    if crit.prereq is not None and not crit.prereq.holds(state):
        return CriterionJudgment(
            status="inapplicable",
            evidence="Prerequisite not yet satisfied; deferring scoring.",
        )
    return fast_criterion_check(
        crit, state.slot(crit.slot).text or "", state.question,
    )
```

Inapplicable judgments contribute zero to `R_slot` (they are excluded from
the sum, not treated as missed=0). This means deferring scoring is neutral,
not punitive -- the agent does not lose reward for not-yet-scorable
criteria, and will gain reward when the prereq is satisfied and the
criterion is re-scored.

**Deferred scoring is not punitive**. A criterion whose prereq is not yet
satisfied contributes 0 (same as unfilled). Once the agent fills the
prerequisite slot, the criterion becomes scorable and a later revision of the
downstream slot can earn the credit. This turns the DAG in §5.4 from a *soft
prior* into a *functional constraint*: you cannot earn `validation` credit
for donor-based assays until `construct_design` actually describes a donor.

**Examples of prereq pairs derived from the 48 benchmark questions**:

| Criterion (lives in) | Prereq (checked in) |
|---|---|
| "Junction PCR with primer outside HA" (validation) | construct_design has {"donor", "homology arm"} |
| "Non-targeting sgRNA control" (safety) | editing_system has {"Cas9", "dCas9", "base edit", "prime edit"} |
| "FACS-sort GFP+ cells" (validation) | construct_design has {"GFP", "fluorescent"} |
| "GUIDE-seq off-target" (safety) | editing_system has {"Cas9", "Cas12"} and construct_design has {"sgRNA", "gRNA"} |
| "qPCR with elephant-specific primers" (validation) | target_selection mentions a non-human ortholog |
| "Karyotype post-editing" (safety) | cell_model mentions {"iPSC", "hPSC", "ESC"} or similar |
| "Cold-shock 32C post-electroporation" (construct/delivery) | delivery has {"electroporation"} |

The gating table is data -- we build it once from the rubric corpus, extend
it as new question types are added, and store it in
`protocol_gym/prereqs.yaml`.

#### Step 3: Compute step reward as the improvement

```
R_step = R_slot(k)_after - R_slot(k)_before
```

This means the agent only gets positive reward for *improving* the protocol, not
for rewriting a slot with equivalent content.

**Interaction with prereq gating**: writing to a slot may newly *unlock*
credit for criteria living in *other* slots (by satisfying their prereq).
Those unlocks should also flow back as step reward. In practice, after any
`fill_slot` or `revise_slot`, we recompute `R_slot` for *every* slot whose
criteria's prereq involves the slot just written, and sum the deltas. This
gives the agent immediate positive signal for, say, "you filled
construct_design, which just turned on 3 pending validation criteria."

#### Step 4: Verifier early firing (intermediate hallucination penalty)

The cheap-tier verifiers from §4.1.2 (citation existence, concentration
sanity, cell-line property check, gene-name resolution) are fast enough to
run on every slot write, not just at submit. If a newly-written slot
contains a parseable verifiable claim, run the relevant verifiers
immediately:

```
R_verify_step(k) = -sum(severity_v * newly_violated_v)
```

where `newly_violated_v` counts only violations introduced by the most
recent slot write (not previously-flagged ones, to avoid double-charging).

This gives the agent an immediate negative signal at the exact step the
hallucination occurs, rather than pooling all verification feedback at
submit. The same violation set is used at submit to compute `R_verify` --
no double-counting of severity.

Medium-tier verifiers (gRNA target verification via Cas-OFFinder, primer
sanity, RNA folding) run only at submit because they are slower and often
require context from multiple slots (e.g., the gRNA in construct_design
needs the claimed gene from target_selection to check).

### 4.3 Tool-use cost penalty

Every tool action incurs a small negative reward to encourage efficiency:

```
R_tool = -c_tool  (per tool call)
```

Where `c_tool` varies by tool type:

| Tool | Cost `c_tool` | Rationale |
|------|---------------|-----------|
| `web_search` | 0.001 | Cheap, fast |
| `tooluniverse` | 0.002 | API call |
| `run_code` | 0.003 | Local compute |
| `run_simulation` | 0.01 | Expensive compute |

These are intentionally small relative to rubric reward (which is 0-1 scale).
The agent should never skip a useful tool call to save cost, but should avoid
calling 30 tools when 5 would suffice.

### 4.4 Budget exhaustion penalty

If the agent hits the step budget without submitting:

```
R_budget_exhausted = -0.2
```

The protocol is auto-submitted in its current state, and terminal reward is
computed on whatever has been filled. This teaches the agent to manage its time.

### 4.5 Prohibited-criteria early warning

After each `fill_slot` or `revise_slot`, we also run the fast check against
**prohibited** criteria. If a slot value triggers a prohibited criterion:

```
R_prohibited_warning = -0.1 per triggered prohibited criterion
```

This gives the agent an immediate signal to revise, rather than discovering the
violation only at terminal scoring (where it caps the score at 0.74).

### 4.6 Coherence reward (cross-slot consistency)

After every slot write, we run a lightweight cross-slot consistency check:

```python
def coherence_check(state: ProtocolState) -> float:
    """Check that filled slots don't contradict each other."""
    penalties = 0.0

    # Example rules (extensible):
    # If editing_system says "CRISPRi" but construct_design mentions "donor template"
    # If delivery says "AAV" but cell_model says "in vitro iPSCs" with no in-vivo context
    # If cell_model says "suspension cells" but delivery says "lipofection"

    # Implemented as keyword-pair contradiction detector:
    CONTRADICTIONS = [
        ("editing_system:CRISPRi",  "construct_design:donor template"),
        ("editing_system:CRISPRi",  "construct_design:homology arm"),
        ("delivery:lipofection",    "cell_model:T cell"),
        ("delivery:lipofection",    "cell_model:primary neuron"),
        # ... extensible list derived from prohibited criteria patterns
    ]

    for (slot_a_pattern, slot_b_pattern) in CONTRADICTIONS:
        slot_a, keyword_a = slot_a_pattern.split(":")
        slot_b, keyword_b = slot_b_pattern.split(":")
        val_a = getattr(state, slot_a).text or ""
        val_b = getattr(state, slot_b).text or ""
        if keyword_a.lower() in val_a.lower() and keyword_b.lower() in val_b.lower():
            penalties += 0.05

    return -penalties
```

### 4.7 Length and boilerplate penalties (anti-reward-hacking)

The fast scorer (§4.2) is deliberately cheap -- it is a mixture of keyword
match and embedding similarity. This creates two reward-hacking attack
surfaces the agent can exploit to inflate intermediate reward without
producing a good protocol:

1. **Boilerplate fills**: sprinkle every slot with rubric-tasting phrases
   ("include appropriate controls", "validate via qPCR and Western blot",
   "perform off-target analysis by GUIDE-seq") that pass surface keyword
   checks but are not substantively connected to the question. These phrases
   will score as "met" by a keyword matcher and moderately by an embedding
   matcher, but would be caught as shallow by the terminal LLM judge.
2. **Length inflation**: pad each slot with as much text as possible to
   maximize the probability of at least one keyword match per criterion. This
   also dilutes the scratchpad readable by the next step.

#### 4.7.1 Terminal judge anchors the intermediate signal

The first line of defense is structural: the terminal reward is computed by a
full LLM judge (§4.1, Gemini 3.1 Flash-Lite Preview) that reads for meaning, not keywords. The
combined reward weights (Section 4.9) keep terminal reward dominant, so a
policy that maximizes intermediate reward at the cost of terminal reward will
lose. Specifically:

- Expected terminal reward scale is ~0.4-0.8 per episode.
- Total intermediate shaped reward across an episode should cap at well
  below the terminal scale (target: <0.3).

If during training we observe that intermediate reward rises while terminal
reward stagnates or falls, that is a direct reward-hacking signature. The
mitigation is to dial down the intermediate-reward weight until terminal
reward resumes improving. We recommend this as a monitored quantity in the
training loop, not a one-time hyperparameter.

#### 4.7.2 Per-slot soft length penalty

We add a length penalty that is zero for moderately-sized slot values and
grows polynomially past a threshold:

```
R_length(slot) = -alpha_len * max(0, words(slot) - L_soft)^2 / L_soft^2
```

with `L_soft = 400 words` and `alpha_len = 0.05`. A 400-word slot incurs
zero penalty. A 600-word slot incurs `-0.05 * (200/400)^2 = -0.0125`. An
800-word slot incurs `-0.05 * (400/400)^2 = -0.05`. A 1200-word slot
incurs `-0.2`, which starts to dominate any plausible keyword gain from the
extra tokens.

The cap is also reinforced in the **observation format** (§5.3): when
rendering state back to the agent, slot values longer than `L_soft` are
truncated in the observation with a `[... N words truncated ...]` marker.
This prevents padding from occupying scratchpad space and, more importantly,
prevents the agent from using long slots as a side channel to carry
information across steps.

#### 4.7.3 Semantic-density check (optional, stronger)

For higher-cost robustness, we can add an occasional density check: sample
one slot per episode at random, run a minimal
Gemini 3.1 Flash-Lite Preview call with the
prompt "Is this slot value substantively connected to the question, or is
it generic boilerplate?" and apply a `-0.1` penalty on a "boilerplate"
verdict. This costs roughly one cheap LLM call per episode and empirically
suppresses boilerplate regression. Disabled by default; enable only if
monitoring flags reward hacking.

#### 4.7.4 Keyword-diversity cap

A simpler, cheaper defense: cap how much credit a slot can accumulate from
any single surface-form keyword. If a criterion's key terms are
`["RS-1", "10 uM", "RAD51"]` and the slot text contains "RS-1 RS-1 RS-1"
because the agent stuffed the keyword, only the first occurrence counts.
`extract_key_terms` deduplicates before scoring. This blocks the most naive
padding strategy for free.

### 4.8 Reward summary table

| Reward component | When | Scale | Purpose |
|-----------------|------|-------|---------|
| `R_terminal` | At submit | [0, 1] | `alpha * R_rubric + beta * R_verify` (§4.1.4) |
| `R_step` | After fill/revise | [-1, 1] | Dense per-slot improvement signal (gated by prereqs, §4.2) |
| `R_verify_step` | After fill/revise | [-0.3, 0] | Early firing of cheap verifiers (§4.2 Step 4) |
| `R_tool` | After tool use | [-0.01, 0] | Efficiency pressure |
| `R_budget_exhausted` | At budget limit | -0.2 | Time management |
| `R_prohibited_warning` | After fill/revise | [-0.1*n, 0] | Early rubric-violation detection |
| `R_coherence` | After fill/revise | [-0.05*n, 0] | Cross-slot consistency |
| `R_length` | After fill/revise | [-0.2, 0] | Anti-padding (§4.7.2) |
| `R_density` | Sampled per episode | {-0.1, 0} | Anti-boilerplate (optional, §4.7.3) |

**Total step reward**:

```
R(t) = R_step(t) + R_verify_step(t) + R_tool(t) + R_prohibited_warning(t)
       + R_coherence(t) + R_length(t) + R_density(t)
       + R_terminal * 1[t = T]
       + R_budget_exhausted * 1[budget exhausted]
```

### 4.9 Uneven slot weighting (motivated empirically)

Not all slots are equally valuable to shape. In run 00013 we computed per-slot
miss rates for Claude Opus 4.6 across the three existing tiers
(`protocol_gym/analyze_slot_misses.py`). The result:

```
slot                 internal_only   web_tools     tooluniverse
biological_context    11.1% miss      11.1%         14.3%
target_selection      18.2%            9.1%         30.0%
editing_system        20.8%           16.0%         18.2%
construct_design      15.0%           15.0%         16.7%
delivery              25.0%           24.0%         17.4%
cell_model            16.0%           12.5%          8.7%
validation             5.4%            5.4%          9.1%
safety_and_controls   18.8%           12.9%         20.7%
```

("Miss" here means >=1 expected criterion mapped to the slot and none scored
met/partial.)

**Interpretation driving uneven weighting**:

- `validation` is near-saturated (5-9% miss). Shaped reward on this slot adds
  little headroom; the model already knows to list qPCR + WB + flow.
- `delivery`, `safety_and_controls`, and `editing_system` have 17-25% miss
  rates. These are the slots where dense reward has the most room to lift
  performance.
- `target_selection` *regresses* under tool use (9% → 30%). This is likely a
  tool-induced tangent -- the model goes deep into tool outputs and forgets
  the isoform / TSS-window question. This slot needs reward signal *and*
  observation-level intervention (see §5.3).
- `biological_context` has few mapped criteria (N=7-9 applicable per tier).
  Shaping here is high-variance; keep its weight small.

**Per-slot shaping-reward weights `w_shape[slot]`**: applied as a multiplier
on `R_step` contributions from that slot:

```
w_shape = {
    "biological_context":  0.5,   # small N, high variance
    "target_selection":    1.5,   # regresses with tools; needs extra signal
    "editing_system":      1.2,   # moderate miss rate
    "construct_design":    1.0,   # baseline
    "delivery":            1.5,   # biggest tier-1/2 miss
    "cell_model":          0.8,   # fairly well-handled
    "validation":          0.5,   # near-saturated; reward adds little
    "safety_and_controls": 1.5,   # consistent miss across tiers
}
```

These are starting points, not learned values. They will be re-calibrated
empirically as the model trains: recompute the miss-rate matrix against the
RL-trained policy periodically (every N training steps) and shift shaping
weight toward slots that are *currently* the weakest under the *current*
policy. Treat it as a slow outer loop at the granularity of checkpoint
evaluation, not per-episode.

**Terminal reward weights are NOT changed**. The rubric judge's own
criterion weights are what they are -- uneven shaping only biases the
*path* the agent takes, never the ground truth of what a good protocol
looks like.

---

## 5. Episode Dynamics

### 5.1 Reset

```python
def reset(question_id: str) -> ProtocolState:
    """Start a new episode."""
    item = load_question(question_id)

    # Classify every rubric criterion to a slot (+ generate its prereq).
    # This is the mapping used by the fast scorer (§4.2).
    classified = classify_criteria(
        item["criteria"]["expected-criteria"],
        item["criteria"]["prohibited-criteria"],
    )

    # Determine per-slot applicability for THIS question.
    # A slot is OPTIONAL if zero criteria map to it.
    applicable = {slot: False for slot in SLOT_ORDER}
    for crit in classified.expected + classified.prohibited:
        if crit.slot in applicable:
            applicable[crit.slot] = True

    return ProtocolState(
        question=item["prompt"],
        clarity=item["claritySelections"],
        episode_budget=30,
        biological_context=SlotValue(),
        target_selection=SlotValue(),
        editing_system=SlotValue(),
        construct_design=SlotValue(),
        delivery=SlotValue(),
        cell_model=SlotValue(),
        validation=SlotValue(),
        safety_and_controls=SlotValue(),
        scratchpad="",
        tool_history=[],
        steps_taken=0,
        classified_criteria=classified,   # used by fast scorer
        slot_applicable=applicable,        # used by applicability logic
    )
```

**Per-episode slot applicability (addresses "not all questions need all
slots")**. Some questions in the 48 benchmark set are genuinely not full
protocol-design questions: protein purification (items 12, 16, 31, 32), a
culture-substrate question (item 33), a differentiation-only NGN2 protocol
(item 43), a pharmacoepidemiology question about Zofran (items 19, 34). For
these, forcing all 8 slots to be filled produces worse answers, not better.

We let the rubric decide. At `reset()`, we classify all expected and
prohibited criteria into slots. A slot is marked **applicable** iff at least
one criterion maps to it, otherwise **optional**.

Applicability controls three things during the episode:

1. **Shaped reward**: only applicable slots contribute to `R_step`. Filling
   an optional slot does not produce shaped reward, but also does not cost
   (except via `R_length` if the agent writes excessively).
2. **Coherence and prereq checks**: still run across *all* slots, applicable
   or not. An agent that fills an optional slot with contradictory content
   still pays the coherence penalty.
3. **Observation rendering**: optional slots are tagged `[OPTIONAL]` in the
   observation (§5.3) so the agent can see they are not required, and are
   listed after applicable slots.

The agent can also emit an explicit `mark_na(slot)` micro-action (technically
a variant of `fill_slot` with empty text + an `"N/A"` flag). This is
useful documentation but carries no reward bonus; it is only a clean way to
signal "I considered this slot and determined it doesn't apply" for
interpretability and for the terminal judge to see.

**On the terminal judge side**: the judge receives only the filled slots
(applicable + any the agent chose to fill). Missing optional slots are not
penalized because their criteria don't exist. Missing applicable slots are
penalized via the standard `missed=0` rule in expected-coverage.

### 5.2 Step

```python
def step(state: ProtocolState, action: Action) -> tuple[ProtocolState, float, bool]:
    """Execute one RL action. Tool use happens INSIDE the policy, before this
    function is called -- by the time we get here, the agent has already
    produced its next protocol action. The tool calls made during that inner
    loop are recorded on `action.tool_calls` so we can account for their cost."""
    reward = 0.0
    done = False

    # Cost of tool calls used during the LLM's inner loop to produce this action
    for tc in action.tool_calls:
        state.scratchpad += f"\n[{tc.type}] {tc.summary}"
        state.tool_history.append(tc)
        reward += R_tool(tc.type)   # small negative per tool call

    if action.type == "fill_slot":
        before = fast_slot_score(state, action.slot)
        set_slot(state, action.slot, action.value)
        after = fast_slot_score(state, action.slot)
        reward += (after - before)                      # R_step
        reward += prohibited_check(state, action.slot)  # R_prohibited_warning
        reward += coherence_check(state)                # R_coherence

    elif action.type == "revise_slot":
        before = fast_slot_score(state, action.slot)
        revise_slot(state, action.slot, action.value)
        after = fast_slot_score(state, action.slot)
        reward += (after - before)
        reward += prohibited_check(state, action.slot)
        reward += coherence_check(state)

    elif action.type == "submit":
        reward += terminal_reward(state)
        done = True

    state.steps_taken += 1
    if state.steps_taken >= state.episode_budget and not done:
        reward += R_budget_exhausted
        reward += terminal_reward(state)  # auto-submit
        done = True

    return state, reward, done
```

### 5.2.1 Step-by-step execution model

The pseudocode above elides the driver loop that *wraps* `env.step()`. This
subsection makes the end-to-end mechanics of a single step explicit.

**Invariant**: one step = one LLM turn = exactly one emitted protocol action.

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP t                                                                  │
│                                                                         │
│ 1. Driver renders ProtocolState → observation (§5.3)                    │
│                                                                         │
│ 2. Driver invokes LLM policy with:                                      │
│      - observation                                                      │
│      - system prompt (defines the 3 MDP actions, slot names, format)    │
│      - tool interface (web_search, tooluniverse, run_code, run_sim)     │
│                                                                         │
│ 3. LLM inner loop (runs until it emits a valid protocol action):        │
│      think → call tool → observe result → think → call tool → ...       │
│      think → emit structured action                                     │
│                                                                         │
│    Bounded by (whichever fires first):                                  │
│      - max inner tool calls per step   (default 10)                     │
│      - max output tokens per step      (default 4000)                   │
│      - wall-clock timeout              (default 180 s)                  │
│                                                                         │
│ 4. LLM returns ONE structured Action, exactly one of:                   │
│      {"action":"fill_slot",   "slot": <name>, "text": ..., ...}         │
│      {"action":"revise_slot", "slot": <name>, "text": ..., ...}         │
│      {"action":"submit"}                                                │
│                                                                         │
│    With `tool_calls` attached: the list of tool invocations made        │
│    during this turn's inner loop, used by env.step() to bill R_tool.    │
│                                                                         │
│ 5. env.step(action) mutates state, computes step reward, returns        │
│    (new_state, reward, done)                                            │
│                                                                         │
│ 6. t += 1, loop back to step 1 unless done                              │
└────────────────────────────────────────────────────────────────────────┘
```

#### Slot choice is the agent's, not the environment's

The environment does **not** pre-assign which slot to fill on step `t`.
The LLM picks the target slot as part of the action. This means:

- The agent can fill slots in any order.
- The agent can return to a previously-filled slot via `revise_slot`.
- The DAG in §5.4 is a soft prior, not an enforced schedule.

#### One slot per step, not multiple

The action space allows exactly one `fill_slot` / `revise_slot` per step.
Multi-slot atomic writes are **not** supported. Rationale:

1. **Credit assignment**: every reward delta maps to a known single slot
   write. Multi-slot actions would require decomposing observed reward
   across writes, which adds variance.
2. **Fast-scorer timing**: the Gemini 3.1 Flash-Lite scorer (§4.2) runs on
   the single just-written slot's criteria. Single-slot steps keep the
   per-step cost bounded and predictable.
3. **Tool cost attribution**: the tool calls made during the inner loop
   are attributable to exactly one subsequent slot write.
4. **MDP cleanliness**: fixed action shape `{fill_slot | revise_slot |
   submit} × slot_name × text` simplifies both the policy's output
   grammar and any policy-gradient algorithm we later apply.

If the agent wants to fill multiple slots, it does so across consecutive
steps. The scratchpad persists across steps, so the agent does **not** need
to re-run the same tools for each fill -- information gathered during
step `t` is available when writing a different slot on step `t+1`.

#### What happens when an inner-loop bound fires before an action is emitted

If the tool-call cap, token cap, or wall-clock timeout fires during step
3 above *before* the LLM has emitted a valid structured action, the
driver sends one **forcing prompt**:

```
System: Inner-loop budget exhausted. Emit a valid protocol action now
        (fill_slot / revise_slot / submit). Do not call further tools.
```

The LLM is given one more bounded generation (1000 tokens, 60 s). Possible
outcomes:

| Outcome | Handling |
|---|---|
| Valid action emitted | Proceed to step 5 as normal. |
| Still no valid action | Log the step as `NOOP`. `steps_taken` increments, a `-0.05` noop penalty applies, `R_tool` for the consumed calls is still billed, ProtocolState is otherwise unchanged. |
| LLM emits malformed JSON | Driver attempts one salvage parse (regex-based). On failure: same as no-valid-action case. |

The NOOP penalty is small on purpose. It is enough to discourage looping
forever on tool use, but not so large that a single bad step destroys the
episode. The agent still has `episode_budget - steps_taken` remaining
steps to recover.

#### Revisions are cheap but not free

`revise_slot` overwrites the previous text and increments
`revision_count` on the SlotValue. `R_step` penalizes excessive rewriting
via `lambda_revision * revision_count` (§4.2). This discourages thrashing
(write → revise → write → revise the same slot with near-identical
content) but still lets net-improving revisions go positive.

#### Why this shape matches how the current eval already runs

This is structurally the same as the Tier-3 tool-using flow in your
existing `AnswerRunner.generate()` (see `scripts/lib/runners.py`): the LLM
loops through tool calls until it emits a final text answer, bounded by
max tool calls and max output tokens. Protocol Gym just replaces "final
text answer" with "one of three structured protocol actions" and wraps the
whole thing in a gym-style `step()` interface. The tool-calling MCP
plumbing and bounding logic transfer directly.

### 5.3 Observation (what the agent sees)

At each step, the agent receives the full state rendered as a structured prompt:

```
## Question
{state.question}

## Current Protocol State
### 1. Biological Context  [APPLICABLE]
{state.biological_context.text or "[EMPTY]"}
[confidence: {state.biological_context.confidence}]

### 2. Target Selection    [APPLICABLE]
{state.target_selection.text or "[EMPTY]"}
...

### 8. Safety & Controls   [APPLICABLE]
{state.safety_and_controls.text or "[EMPTY]"}

## Scratchpad (recent tool results)
{last 3 entries from state.scratchpad}

## Focus Reminder
{one-line reminder listing still-empty APPLICABLE slots}

## Budget
Steps used: {state.steps_taken}/{state.episode_budget}
Applicable slots filled: {count_filled_applicable}/{count_applicable}
```

**Length-truncated slot rendering**. Any slot text longer than `L_soft`
words (§4.7.2) is rendered as the first `L_soft` words followed by
`[... N words truncated ...]`. The full text is retained in state for
terminal judging, but the agent's next-step context only sees the
truncated view. This caps the observation's growth and prevents the agent
from padding slots as a scratchpad side-channel.

**Optional slot rendering**. Slots marked `[OPTIONAL]` (no mapped criteria)
are listed after applicable ones, visually de-emphasized. Their header
reads `### 9. <name>  [OPTIONAL -- no rubric criteria mapped]`.

**Focus Reminder (addresses target_selection regression under tools)**.
The §4.9 analysis showed `target_selection` miss rate jumps from 9% to 30%
when the model has tool access -- classic tool-induced tangent. To mitigate
without changing the reward landscape, the observation carries a one-line
"focus reminder" that names the still-empty applicable slots. The reminder
is:

- **Inactive** for the first 3 steps (don't nag prematurely).
- **Active when** `steps_taken >= 3` and at least one applicable slot is
  still empty. Format: `"Empty applicable slots: target_selection,
  safety_and_controls. Consider writing one before further tool use."`
- **Escalates** if `tool_calls_since_last_fill >= 5`. Format:
  `"You have called {N} tools without updating any slot. Consider emitting
  fill_slot before further exploration."`

This is *observation engineering*, not reward engineering. It does not
change the MDP's reward function; it just surfaces information already in
the state in a more salient way. If RL with this reminder produces
meaningfully better behavior than without, we keep it; if not, we drop it
to minimize observation clutter.

### 5.4 Slot dependency structure

Some slots are logically upstream of others. While the agent CAN fill them in
any order, the natural dependency flow is:

```
biological_context ──> target_selection ──> editing_system
                                                │
                                    ┌───────────┼───────────┐
                                    v           v           v
                            construct_design  delivery  cell_model
                                    │           │           │
                                    └───────────┼───────────┘
                                                v
                                           validation
                                                │
                                                v
                                       safety_and_controls
```

We do NOT enforce this ordering -- the agent is free to fill slots in any
sequence. But the coherence reward (Section 4.6) will naturally penalize
contradictions that arise from filling downstream slots before upstream ones
(e.g., designing a donor template before choosing the editing system).

---

## 6. Data Integration

### 6.1 genetic_benchmark_v1 (48 questions)

Primary training/evaluation set. Each question provides:
- `prompt` -> `state.question`
- `claritySelections` -> `state.clarity`
- `criteria.expected-criteria` -> terminal reward rubric (expected)
- `criteria.prohibited-criteria` -> terminal reward rubric (prohibited)
- `modelResponse` -> expert demonstrations for behavioral cloning warmstart

### 6.2 benchling (4 questions)

Protocol-adaptation questions with weighted rubrics. These are particularly
well-suited for this environment because they explicitly test cascading
reasoning about protocol changes. They use the weighted-positive rubric format:

```
R_terminal = sum(criterion_weight * score) / sum(all_weights)
```

### 6.3 Data split

| Split | Source | Size | Use |
|-------|--------|------|-----|
| Train | genetic_benchmark_v1 | 38 questions | RL training |
| Val | genetic_benchmark_v1 | 10 questions | Hyperparameter tuning |
| Test | benchling | 4 questions | Held-out evaluation |

The benchling questions are held out because they test a qualitatively different
skill (protocol adaptation vs. protocol design from scratch), making them a good
generalization test.

---

## 7. Implementation Plan

### 7.1 Files

```
protocol_gym/
    DESIGN.md              # This document
    env.py                 # ProtocolEnv(gymnasium.Env) -- core environment
    state.py               # ProtocolState, SlotValue dataclasses
    actions.py             # Action types and parsing
    rewards.py             # All reward functions
    criteria_mapper.py     # Maps rubric criteria to slots
    fast_scorer.py         # Keyword + embedding based fast criterion check
    tools.py               # Tool execution wrappers (web, tooluniverse, code, sim)
    render.py              # Observation rendering for the LLM agent
    config.py              # Hyperparameters (budgets, cost weights, etc.)
    __init__.py
    tests/
        test_state.py
        test_rewards.py
        test_env.py
```

### 7.2 Dependencies

- `gymnasium` -- standard RL environment interface
- `sentence-transformers` -- for embedding-based semantic scoring
- Existing `scripts/lib/` -- for judge, dataset loading, tooluniverse MCP

### 7.3 Build order

1. `state.py` + `actions.py` -- data structures
2. `criteria_mapper.py` + `fast_scorer.py` -- reward infrastructure
3. `rewards.py` -- all reward functions
4. `tools.py` -- tool execution wrappers
5. `env.py` -- gymnasium Env tying it all together
6. `render.py` -- LLM-readable observation formatting
7. Tests

---

## 8. Open Questions

1. **Episode budget**: 30 steps is a guess. Analyzing existing Tier 3 tool
   traces from run 00013 would give an empirical distribution of how many tool
   calls Opus 4.6 actually makes per question. Should calibrate to ~2x the
   median.

2. **Judge calibration against human-graded ground truth**: both the
   fast scorer and the terminal judge now use Gemini 3.1 Flash-Lite
   Preview, so inter-tier model disagreement is no longer the primary
   concern -- the concern is whether Flash-Lite agrees with a human
   expert well enough to drive RL training. Validation protocol:
   - **Human-grade a calibration set**: sample ~200 (criterion,
     slot_value) pairs from the 48 expert responses and have a biologist
     grade them in the 4-way ontology {met, partial, missed,
     inapplicable}. This is the ground-truth reference.
   - **Fast scorer vs. human**: require Cohen's kappa >= 0.70 on the
     4-way classification. Prompt-engineer systematic biases out before
     training (e.g. over-calling "partial" when humans say "missed").
   - **Terminal judge vs. human**: same threshold on ~50 full-protocol
     judgments where a biologist has graded the complete response
     against the rubric.
   - **Audit judge spot-check**: during training, sample ~5% of
     episodes and re-judge with a stronger model (gpt-5.4 or Claude
     Opus 4.6). If the stronger judge disagrees with Flash-Lite
     systematically and in a direction that correlates with policy
     reward, the training signal is being exploited -- pause training
     and recalibrate.
   - **Pre-filter safety**: the keyword + embedding pre-filter (§4.2
     Step 2) needs its own check -- confirm it only produces MISSED
     judgments that humans also call MISSED (false-MISSED rate < 5%).
     Any pre-filter that silently downgrades a real MET to MISSED is
     worse than no pre-filter at all.

3. **Prereq table coverage**: §4.2 introduces prerequisite gating, and lists
   example prereq pairs derived from the 48 benchmark questions. We need to
   build out `protocol_gym/prereqs.yaml` by passing the full criterion corpus
   through an LLM (one-time) and hand-auditing the generated prereqs. Expect
   ~100-150 distinct prereq rules across the corpus.

4. **Episode structure -- decided**: one question per episode. Each
   episode begins at `reset()` with a single question, ends at `submit()`
   (or `episode_budget` exhaustion), and no cross-question state carries
   between episodes. Rationale: simpler MDP, cleaner credit assignment,
   and the `claritySelections` metadata already lets us stratify
   difficulty at evaluation time without needing a curriculum-style
   multi-question episode structure. We sample questions uniformly
   during training but can weight by difficulty dimensions if we observe
   systematic underperformance on any stratum.

5. **Reward scale balancing**: The relative weights between terminal reward,
   step reward, tool cost, length penalties, and uneven slot weighting all
   need empirical tuning. Terminal reward should dominate to avoid reward
   hacking on intermediate signals (§4.7.1). Monitor the ratio of cumulative
   intermediate shaped reward to terminal reward across training -- if
   intermediate rises while terminal stagnates, that is the signature of
   reward hacking and the intermediate weight should be cut.

6. **Applicability classifier drift**: the §5.1 applicability decision
   depends on the keyword classifier assigning criteria to slots. If the
   classifier misses a criterion (sends it to `global`), a slot that is
   genuinely in-scope might look optional. We mitigate by treating `global`
   criteria as applying to *all* slots (any slot that has no specific
   criteria but shares global coverage is still applicable). But this is a
   source of noise worth measuring.

7. **Does the focus reminder actually help?** §5.3 adds an observation-level
   focus reminder to address the `target_selection` regression under tools.
   Whether this helps versus clutters the observation is an ablation to run
   after initial training is working.

8. **Verifier coverage audit**: the verifier catalog in §4.1.2 is a
   starting point derived from the failure modes we anticipate. Before
   training, run every verifier against the 48 expert `modelResponse`
   fields as a calibration check.

   **Procedure**:

   ```
   for each expert response in genetic_benchmark_v1.modelResponse:
       1. Slot-segment the response using the §4.2 classifier.
       2. Run EVERY verifier in the §4.1.2 catalog:
            - citation existence (PubMed / CrossRef API)
            - concentration sanity (regex extract + reference table lookup)
            - cell line property (Cellosaurus API)
            - gene name resolution (HGNC / NCBI Gene)
            - gRNA target verification (Cas-OFFinder) if sequences present
            - primer sanity (Primer3) if primers present
            - RNA folding (ViennaRNA) if scaffold/pegRNA present
       3. Record every flagged violation with its severity and the
          verifier that raised it.
       4. Compute R_verify(response) = max(0, 1 - sum(severity_k * flag_k))
   ```

   **Expected outcome**: every expert response should score R_verify >=
   0.95. Experts are the ground-truth signal -- if a verifier flags
   expert content, one of three things is true:

   - *True positive*: the expert genuinely made a mistake (typo'd PMID,
     misremembered cell-line property). Rare. Keep the flag; document it.
   - *Verifier too strict*: e.g., reference table says "RS-1: 10-20 uM"
     but the expert correctly uses "7.5 uM" per Pinder 2015. Widen the
     reference range.
   - *Verifier reference data wrong or incomplete*: e.g., gene-name
     resolver doesn't know the TP53RTG1 retrogene alias. Fix the
     resolver (add alias tables, fall back to CrossRef / bioRxiv for
     citations, etc.).

   **Output artifact**: per-verifier calibration report with columns
   (total_checks, flagged, false_positives_after_review,
   true_positives_after_review, status, fix_commit). Stored as
   `protocol_gym/verifier_audit.md` and updated every time the catalog
   changes.

   **Go/no-go gate**: no verifier is enabled as training signal until
   its expert-response false-positive rate is below 2%. A verifier that
   flags real expert work during the audit will also flag real correct
   work produced by a trained agent -- which means it is not correcting
   hallucinations, it is penalizing correctness. Strictly worse than no
   verifier.

   **Why pre-training, not during**: during RL the agent's output
   distribution drifts. If a verifier is mis-calibrated, we cannot
   separate "agent is hallucinating more" from "agent is producing
   content the verifier mishandles." The pre-training audit gives a
   fixed, known-good reference distribution to calibrate against.

9. **Verifier-induced vagueness**: a subtle failure mode -- if `R_verify`
    penalizes wrong specific claims but not vague ones, the agent may
    learn to be deliberately unspecific ("a suitable concentration of
    RS-1") to avoid verifier risk. The rubric reward should counteract
    this (criteria often demand specific values), but watch for it in
    training: compare average slot specificity (e.g., count of
    extractable numeric claims) against baseline across training. If
    specificity falls while `R_verify` stays high, we have a problem
    and need to add a specificity floor to the rubric judge.
