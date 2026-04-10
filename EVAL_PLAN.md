# Evaluation Harness Plan

## Goal

Build a reproducible evaluation script that:

1. Loads every question from `48-submissions-clean.json`.
2. Runs multiple response-generation configurations for each question.
3. Uses two base answer models:
  - `gpt-5.4` with low thinking or no thinking.
  - `claude-opus-4.6` with no thinking.
4. Runs each base model through three tool tiers:
  - `internal_only`: LLM with no external tools.
  - `web_tools`: LLM with web search and standard non-ToolUniverse tools.
  - `tooluniverse`: LLM with ToolUniverse access.
5. This yields six answering configurations per question.
6. Runs a separate judge LLM on each response independently.
7. Scores each response against the rubric embedded in the dataset item, using:
  - `refinedCriteria` when present.
  - otherwise `criteria`.
8. Produces machine-readable outputs plus a summary table for comparison across modes.
9. Starts with only the first two dataset questions for the initial run.

This plan covers only the design. It does not implement the harness yet.

## Confirmed Decisions

These choices are now fixed for the first implementation pass:

1. Answering models:
  - `gpt-5.4` low-thinking or no-thinking mode
  - `claude-opus-4.6` no-thinking mode
2. Tool tiers for each answering model:
  - `internal_only`
  - `web_tools`
  - `tooluniverse`
3. Initial execution scope:
  - first two questions only
4. Rubric exposure to answering models:
  - no, keep rubric judge-only in the first benchmark
5. `claritySelections` exposure to answering models:
  - no in the first benchmark
  - keep them as metadata for analysis only
6. Empty-rubric items:
  - report separately from the main rubric leaderboard
7. Code organization:
  - small multi-file harness, not a monolithic script
8. Judge model for first pass:
  - use `gpt-5.4` low-thinking or no-thinking as the primary structured judge
  - keep optional support for later cross-checking with `claude-opus-4.6`

## Recommended Output Structure

I would implement the pipeline around a few explicit artifacts:

- `runs/<timestamp>/responses.jsonl`
  - one row per question x mode
- `runs/<timestamp>/judgments.jsonl`
  - one row per judged response
- `runs/<timestamp>/summary.csv`
  - flat table for analysis
- `runs/<timestamp>/summary.md`
  - human-readable aggregate report

Each response row should include:

- dataset index
- submission id
- prompt
- criteria used
- clarity selections metadata
- mode
- model name
- provider name
- tool tier
- raw response text
- token usage if available
- latency if available
- any tool trace metadata if available

Each judgment row should include:

- dataset index
- submission id
- mode
- judge model name
- per-criterion decisions
- expected coverage score
- prohibited violation count
- final ranking tuple or composite score
- short rationale

## Proposed Files To Implement Later

- `scripts/run_eval.py`
  - main orchestration entry point
- `scripts/judge_eval.py`
  - optional standalone judging entry point if we want two-pass execution
- `scripts/lib/dataset.py`
  - load items, normalize criteria, select prompt text
- `scripts/lib/runners.py`
  - adapters for the six model-plus-tier configurations
- `scripts/lib/judge.py`
  - judge prompt builder and schema validation
- `scripts/lib/reporting.py`
  - aggregate scores, emit csv and markdown summaries

If we want to keep the repo small, this can also be done as a single script first, but I would still separate the logic internally into clearly named functions.

## Execution Flow

For each dataset item:

1. Read `prompt`.
2. Read criteria from `refinedCriteria`; if empty, fall back to `criteria`.
3. Read `claritySelections` and store them as metadata.
4. Generate six responses:
  - `gpt-5.4` x `internal_only`
  - `gpt-5.4` x `web_tools`
  - `gpt-5.4` x `tooluniverse`
  - `claude-opus-4.6` x `internal_only`
  - `claude-opus-4.6` x `web_tools`
  - `claude-opus-4.6` x `tooluniverse`
5. Store raw responses immediately so the run is resumable.
6. Judge each response independently against that item's rubric.
7. Store judgments immediately.
8. After all items complete, compute aggregate metrics by model and by tool tier.

The harness should support:

- `--limit` for quick testing
- `--start-index` and `--end-index`
- `--resume` to skip already completed rows
- `--judge-only` to re-score existing responses
- `--modes internal_only,web_tools,tooluniverse`

## Prompting Strategy For The Three Answering Modes

All three answering modes should receive the same user question and the same high-level instruction about the task. The only difference should be tool availability.

Base generation prompt should include:

- the raw user prompt
- a short instruction to answer the user directly and comprehensively

I would not expose the rubric to the answering models during the main run if the goal is to test real capability. If the goal is rubric optimization, that can be a second benchmark variant later.

I would also not expose `claritySelections` to the answering models in the first benchmark.

## What `claritySelections` Are

`claritySelections` are dataset-side annotations describing how explicit or underspecified the question is along a few dimensions, for example:

- whether the biological question is explicit or open
- whether the target genes are explicit, implicit, or open
- whether the organism is explicit or open
- whether the editing modality is explicit or open
- whether the validation technique is explicit or open
- whether the timeline is explicit or open

These are useful because they let us stratify results later. For example, we can compare model performance on:

- prompts with explicit target genes versus open-ended ones
- prompts with explicit timelines versus underspecified timelines
- prompts that strongly constrain the method versus prompts that require the model to infer missing details

For the first benchmark pass, they should be treated as analysis metadata, not as extra information given to the answer models. Otherwise the answering models would be seeing annotator-side structure that a real user did not provide.

### Mode 1: `internal_only`

- No tools.
- Pure model knowledge.

### Mode 2: `web_tools`

- Web search and standard retrieval tools.
- Explicitly no ToolUniverse tools.

### Mode 3: `tooluniverse`

- ToolUniverse-enabled environment.
- Same base instruction, but with access to ToolUniverse tools.

Each of these three tiers will be run for both `gpt-5.4` and `claude-opus-4.6`.

## Judge Design

The judge should evaluate one response at a time, not compare all three responses in the same prompt initially. That avoids anchoring bias and keeps the scoring attributable.

Each judge prompt should contain:

1. The original question.
2. The rubric object for that item.
3. The candidate response.
4. A required JSON schema for the judgment.

The judge should return structured fields like:

```json
{
  "expected": [
    {
      "criterion": "Suggest using H1299 cell type, very important in p53 research.",
      "status": "met",
      "confidence": 0.97,
      "evidence": "H1299 is explicitly recommended as a TP53-null line."
    }
  ],
  "prohibited": [
    {
      "criterion": "Suggest lentivirus to deliver the plasmid.",
      "status": "violated",
      "confidence": 0.99,
      "evidence": "The answer recommends lentiviral transduction."
    }
  ],
  "holistic": {
    "factuality": 4,
    "completeness": 4,
    "clarity": 4
  },
  "summary": "Strong answer overall, but it violates a prohibited delivery-method criterion.",
  "scores": {
    "expected_points": 8.5,
    "expected_max": 13,
    "prohibited_violations": 1
  }
}
```

### Criterion Status Values

For expected criteria:

- `met`
- `partial`
- `missed`
- `unclear`

For prohibited criteria:

- `not_violated`
- `violated`
- `unclear`

I would treat `unclear` as a separate bucket in the raw output and resolve it conservatively during aggregation.

## Best Scoring Protocol

I do not recommend a simple `+1 for expected, -1 for prohibited` scheme as the primary metric.

That approach is too brittle because:

1. Items have different numbers of expected and prohibited criteria.
2. A single prohibited violation is often more important than missing one minor expected detail.
3. Some criteria are only partially satisfied.
4. Raw additive scores are hard to compare across items.

### Recommended Primary Protocol: Two-Level Scoring

Use a criterion-level structured judgment and derive two core metrics:

1. `expected_coverage`
2. `prohibited_violations`

#### Expected Coverage

For each expected criterion:

- `met = 1.0`
- `partial = 0.5`
- `missed = 0.0`
- `unclear = 0.25`

Then compute:

`expected_coverage = sum(points) / number_of_expected_criteria`

This yields a normalized value from `0.0` to `1.0`.

#### Prohibited Violations

For each prohibited criterion:

- `violated = 1`
- `not_violated = 0`
- `unclear = 0.5`

Then compute:

- `prohibited_count = sum(violations)`
- `prohibited_rate = prohibited_count / number_of_prohibited_criteria` when prohibited criteria exist

### Recommended Ranking Rule

For ranking model outputs, I would use lexicographic ordering instead of a single raw score:

1. Fewer prohibited violations is better.
2. Higher expected coverage is better.
3. Higher holistic factuality is better.
4. Higher holistic completeness is better.

This is the most robust default because it respects explicit "do not do this" constraints.

### Recommended Dashboard Composite

If a single number is needed for summaries, compute:

`final_score = 100 * (0.8 * expected_coverage + 0.2 * prohibited_compliance)`

where:

- `prohibited_compliance = 1 - prohibited_rate`

Then apply a cap:

- if any prohibited criterion is violated, cap `final_score` at `74`

Reasoning:

- the weighted score remains interpretable and normalized
- the cap ensures that a response with explicit prohibited content cannot outrank a clean response with similar coverage
- the separate raw metrics remain available, so the cap is not hiding information

### Why This Is Better Than +1 / -1

- It handles partial matches.
- It normalizes across questions.
- It preserves the special role of prohibited constraints.
- It gives both an interpretable pair of core metrics and a usable one-number summary.

## Handling Missing Or Empty Rubrics

Some items may have empty `criteria` and `refinedCriteria` objects.

For those items, the judge should fall back to a rubric-light review with:

- factuality: 1 to 5
- completeness: 1 to 5
- directness: 1 to 5
- actionability: 1 to 5

These items should be flagged as `no_structured_rubric = true` so they are not mixed blindly with rubric-scored items in the aggregate analysis.

### What "Reported Separately" Means

An empty-rubric item is a question where the dataset does not define explicit expected or prohibited criteria. In those cases, there is no item-specific checklist to score against.

Because of that, these items should not be mixed into the main rubric leaderboard. Instead:

- keep a main leaderboard for rubric-bearing items only
- keep a separate rubric-light report for empty-rubric items
- still judge them for factuality, completeness, directness, and actionability
- report their results in a separate section of the markdown summary and, if needed, a separate CSV

This avoids distorting the main benchmark with items that do not have comparable scoring structure.

## Aggregation Across The Full Dataset

For each mode, the final report should include:

- mean expected coverage
- median expected coverage
- total prohibited violations
- count of items with at least one prohibited violation
- mean composite score
- number of rubric-bearing items
- number of rubric-light items

I would also add per-criterion frequency tables, for example:

- which expected criteria are most often missed
- which prohibited criteria are most often violated

That makes the evaluation useful for model improvement, not just ranking.

## Reproducibility And Reliability

I would implement several guardrails:

- fixed judge prompt template
- structured JSON output from judge
- retry on invalid JSON
- save raw judge text alongside parsed JSON
- optional second-pass re-judge for low-confidence cases

If we want stronger reliability later, we can add:

- `n=2` or `n=3` judge runs per response
- majority vote on criterion labels
- confidence-weighted averaging

I would not start with multi-judge consensus on the first version because it will increase cost and latency substantially.

## Implementation Decision I Would Use

If I implement this next, I will build:

1. A resumable Python harness that loops through `48-submissions-clean.json`.
2. Runner backends for two answer models across three tool tiers.
3. An initial run configuration restricted to the first two dataset questions.
4. One structured judge pass per response using `gpt-5.4` as the primary judge.
5. A scoring layer using:
   - normalized expected coverage
   - prohibited violation count and rate
   - lexicographic ranking
   - capped composite score for dashboards
6. CSV and markdown reports for quick inspection.

## Open Questions Before Implementation

The main unresolved coding questions after this update are narrower:

1. What exact API wrappers or SDKs are available locally for `gpt-5.4` and `claude-opus-4.6`?
2. How should the `web_tools` and `tooluniverse` tiers be invoked in practice from the harness?
3. Do you want a second optional judge model in v1, or keep single-judge only?
4. What exact output schema do you want in the final CSV and markdown reports?

## Recommendation Summary

My recommendation is:

- do not use raw `+1 / -1` scoring as the primary metric
- score expected criteria with partial credit
- track prohibited criteria separately and treat them as ranking-critical
- rank by:
  1. prohibited violations
  2. expected coverage
  3. holistic quality
- keep a capped composite score only for dashboards and summaries

This will give you a more stable and more defensible benchmark than a flat additive score.