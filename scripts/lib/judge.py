from __future__ import annotations

import json
import os
import random
import sys
import time as _time
from pathlib import Path
from typing import Any

import anthropic
import openai
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from .dataset import infer_rubric_style
from .usage_costs import build_usage_metrics


FALLBACK_JUDGE_MODEL = "claude-sonnet-4-6"

# Output-token budget for every judge call. Sized to accommodate the
# largest rubric in the dataset (≈25 criteria) with verbose evidence
# strings. GPT-5.5 in particular emits long evidence; 2500 truncated the
# JSON mid-array on rubrics with >20 criteria (e.g. q27, q30).
JUDGE_MAX_OUTPUT_TOKENS = 6000


EXPECTED_SCORES = {"met": 1.0, "partial": 0.5, "missed": 0.0, "unclear": 0.25}
PROHIBITED_SCORES = {"not_violated": 0.0, "violated": 1.0, "unclear": 0.5}

# Weights applied to expected coverage and prohibited compliance when
# combining them into the final 0-100 score. Must sum to 1.0.
EXPECTED_WEIGHT = 0.8
PROHIBITED_WEIGHT = 0.2


def load_local_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)


def build_judge_prompt(question: str, criteria: dict[str, Any], response_text: str) -> str:
    return (
        "You are scoring one model response against a rubric.\n"
        "Return JSON only. Do not add markdown or commentary outside the JSON object.\n\n"
        "For every structured rubric item, return exactly one matching judgment entry and copy the rubric criterion text verbatim when possible.\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Rubric JSON:\n"
        f"{json.dumps(criteria, indent=2, ensure_ascii=True)}\n\n"
        "Candidate response:\n"
        f"{response_text.strip()}\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "expected": [{"criterion": str, "status": "met|partial|missed|unclear", "evidence": str}],\n'
        '  "prohibited": [{"criterion": str, "status": "not_violated|violated|unclear", "evidence": str}],\n'
        '  "holistic": {"factuality": 1-5, "completeness": 1-5, "clarity": 1-5},\n'
        '  "summary": str\n'
        "}\n"
    )


def build_rubric_light_prompt(question: str, response_text: str) -> str:
    return (
        "You are scoring one model response when no item-specific rubric is available.\n"
        "Return JSON only.\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Candidate response:\n"
        f"{response_text.strip()}\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "holistic": {"factuality": 1-5, "completeness": 1-5, "directness": 1-5, "actionability": 1-5},\n'
        '  "summary": str\n'
        "}\n"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class Judge:
    def __init__(self, project_root: Path, model_name: str = "gpt-5.4", fallback_model_name: str | None = FALLBACK_JUDGE_MODEL):
        load_local_env(project_root)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self._anthropic_client: Anthropic | None = None

    @property
    def anthropic_client(self) -> Anthropic:
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._anthropic_client

    def judge_response(self, question: str, criteria: dict[str, Any], response_text: str) -> dict[str, Any]:
        has_structured_rubric = bool(criteria.get("expected") or criteria.get("prohibited"))
        rubric_style = str(criteria.get("rubric_style") or infer_rubric_style(criteria.get("raw")))
        prompt = (
            build_judge_prompt(question, criteria, response_text)
            if has_structured_rubric
            else build_rubric_light_prompt(question, response_text)
        )
        provider = "openai"
        judge_model_name = self.model_name
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            )
            raw = response.model_dump(mode="json")
            output_text = response.output_text
        except BadRequestError as exc:
            if self.fallback_model_name is None or not _is_safety_refusal(exc):
                raise
            provider = "anthropic"
            judge_model_name = self.fallback_model_name
            output_text, raw = self._call_anthropic_fallback(prompt)
        parsed = extract_json_object(output_text)
        if has_structured_rubric:
            if rubric_style == "weighted_positive":
                judgment = finalize_weighted_positive_judgment(parsed, criteria)
            else:
                judgment = finalize_structured_judgment(parsed, criteria)
        else:
            judgment = finalize_rubric_light_judgment(parsed)
        judgment["rubric_style"] = rubric_style
        judgment["_meta"] = {
            "judge_model_name": judge_model_name,
            "usage_metrics": build_usage_metrics(provider, judge_model_name, raw),
        }
        return judgment

    def _call_anthropic_fallback(self, prompt: str) -> tuple[str, dict[str, Any]]:
        response = self.anthropic_client.messages.create(
            model=self.fallback_model_name,
            system="Return strict JSON only.",
            max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.model_dump(mode="json")
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return text, raw


HARNESS_DEFAULT_CONFIG: list[tuple[str, str, int]] = [
    # (provider, model_name, repeats) — matches Diego's verifier harness:
    # 2 GPT-5.5 + 2 Sonnet-4.6, voting per criterion.
    ("openai", "gpt-5.5", 2),
    ("anthropic", "claude-sonnet-4-6", 2),
]


def _vote_expected(satisfied_count: int, total: int) -> str:
    if total <= 0:
        return "unclear"
    if satisfied_count > total / 2:
        return "met"
    if satisfied_count < total / 2:
        return "missed"
    return "partial"


def _vote_prohibited(satisfied_count: int, total: int) -> str:
    if total <= 0:
        return "unclear"
    if satisfied_count > total / 2:
        return "not_violated"
    if satisfied_count < total / 2:
        return "violated"
    return "unclear"


class HarnessJudge:
    """Multi-LLM voting verifier (Diego's pattern).

    Runs the same scoring prompt across N (provider, model) × repeat
    judges. For each rubric criterion we count "satisfied" votes
    (status == "met" for expected items, "not_violated" for prohibited
    items) and aggregate via majority voting:

      - **>half satisfied** → met / not_violated
      - **=half satisfied** → partial (expected) / unclear (prohibited)
      - **<half satisfied** → missed / violated

    The aggregated per-criterion statuses are then passed through the
    same legacy_structured / weighted_positive finalizer used by the
    single-judge path, so summary.csv / summary.md numbers stay
    comparable across the two verifier modes.
    """

    def __init__(
        self,
        project_root: Path,
        judges: list[tuple[str, str, int]] | None = None,
    ):
        load_local_env(project_root)
        self.project_root = project_root
        self.judges = judges or HARNESS_DEFAULT_CONFIG
        self._openai: OpenAI | None = None
        self._anthropic: Anthropic | None = None

    @property
    def openai_client(self) -> OpenAI:
        if self._openai is None:
            self._openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._openai

    @property
    def anthropic_client(self) -> Anthropic:
        if self._anthropic is None:
            self._anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._anthropic

    # Retry policy for transient provider failures (overloaded, rate limit,
    # transient 5xx, network) and for empty/unparseable outputs. We retry up
    # to MAX_ATTEMPTS times with jittered exponential backoff.
    _MAX_ATTEMPTS = 4
    _BASE_BACKOFF_S = 2.0

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        # Provider-specific transient errors.
        retryable_types = (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,  # 5xx
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
        if isinstance(exc, retryable_types):
            return True
        # Overloaded errors (Anthropic 529) and any other 5xx — match by
        # status code on the generic APIStatusError base class.
        for status in (getattr(exc, "status_code", None), getattr(exc, "status", None)):
            if isinstance(status, int) and (status == 429 or 500 <= status < 600):
                return True
        # Newer Anthropic SDKs raise an OverloadedError subclass; the message
        # carries "Overloaded" — match defensively in case the class is gone
        # but the API still returns a generic APIStatusError.
        msg = str(exc).lower()
        if "overloaded" in msg or "rate limit" in msg or "try again" in msg:
            return True
        return False

    def _call_single(self, provider: str, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
        if provider == "openai":
            response = self.openai_client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            )
            return response.output_text, response.model_dump(mode="json")
        if provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model=model,
                system="Return strict JSON only.",
                max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
            return text, response.model_dump(mode="json")
        raise ValueError(f"Unsupported harness judge provider: {provider!r}")

    def _call_and_parse(
        self, provider: str, model: str, prompt: str, label: str
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Call one judge with retries, returning (parsed_json, raw_response, attempt_errors).

        Retries cover both transient API errors (rate limit, overloaded,
        timeout, 5xx) and empty/unparseable JSON responses. Each retry waits
        ``_BASE_BACKOFF_S * 2**attempt`` seconds plus jitter.

        Per-attempt timing is also emitted to stderr (one line per attempt)
        so a tail of run.log shows what each judge is doing in real time.

        Raises the last exception (or RuntimeError describing the last parse
        failure) if all attempts fail.
        """
        attempt_errors: list[str] = []
        last_exc: Exception | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            attempt_t0 = _time.perf_counter()
            try:
                output_text, raw = self._call_single(provider, model, prompt)
            except Exception as exc:  # provider call failed
                last_exc = exc
                retryable = self._is_retryable(exc)
                dur = _time.perf_counter() - attempt_t0
                msg = (
                    f"[harness] {label} attempt {attempt + 1}/{self._MAX_ATTEMPTS} "
                    f"FAIL after {dur:.1f}s: {exc.__class__.__name__}: "
                    f"{str(exc)[:200]} (retryable={retryable})"
                )
                print(msg, file=sys.stderr, flush=True)
                attempt_errors.append(msg)
                if not retryable or attempt == self._MAX_ATTEMPTS - 1:
                    break
                _time.sleep(self._BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1.0))
                continue
            dur = _time.perf_counter() - attempt_t0
            # Got a response — try to parse it.
            try:
                parsed = extract_json_object(output_text)
                if attempt > 0 or dur > 30.0:
                    print(
                        f"[harness] {label} attempt {attempt + 1}/{self._MAX_ATTEMPTS} OK after {dur:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                return parsed, raw, attempt_errors
            except Exception as exc:
                last_exc = exc
                msg = (
                    f"[harness] {label} attempt {attempt + 1}/{self._MAX_ATTEMPTS} "
                    f"parse-error after {dur:.1f}s: {str(exc)[:200]} "
                    f"(output_len={len(output_text)})"
                )
                print(msg, file=sys.stderr, flush=True)
                attempt_errors.append(msg)
                # Retry on parse error — model may have produced junk.
                if attempt == self._MAX_ATTEMPTS - 1:
                    break
                _time.sleep(self._BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1.0))
                continue
        # If we got here, every attempt failed.
        msg = f"{label}: " + " | ".join(attempt_errors)
        if last_exc is not None:
            raise RuntimeError(msg) from last_exc
        raise RuntimeError(msg)

    def judge_response(self, question: str, criteria: dict[str, Any], response_text: str) -> dict[str, Any]:
        has_structured_rubric = bool(criteria.get("expected") or criteria.get("prohibited"))
        rubric_style = str(criteria.get("rubric_style") or infer_rubric_style(criteria.get("raw")))
        if not has_structured_rubric:
            # Holistic rubric: fall back to a single judge (voting has no
            # meaningful per-criterion structure to aggregate over).
            single = Judge(project_root=self.project_root)
            return single.judge_response(question, criteria, response_text)

        prompt = build_judge_prompt(question, criteria, response_text)

        # Collect per-criterion votes keyed by criterion text (case-/whitespace-folded).
        # Each vote is {"label": str, "status": str, "evidence": str}.
        votes_expected: dict[str, list[dict[str, str]]] = {}
        votes_prohibited: dict[str, list[dict[str, str]]] = {}

        # Track usage across calls (accumulator has the same shape as
        # build_usage_metrics output, with values summed).
        accumulated_usage_metrics: dict[str, Any] = {}
        judge_model_names: list[str] = []
        per_call_meta: list[dict[str, Any]] = []
        any_call_succeeded = False
        last_error: str | None = None

        for provider, model, repeats in self.judges:
            for rep in range(1, repeats + 1):
                label = f"{model}:rep{rep}"
                judge_model_names.append(label)
                try:
                    parsed, raw, attempt_errors = self._call_and_parse(provider, model, prompt, label)
                except Exception as exc:
                    last_error = f"{label}: {exc.__class__.__name__}: {str(exc)[:400]}"
                    per_call_meta.append({"label": label, "error": last_error})
                    continue
                any_call_succeeded = True
                # Accumulate usage cost for this call
                accumulated_usage_metrics = _add_usage(accumulated_usage_metrics, provider, model, raw)
                per_call_meta.append({
                    "label": label,
                    "provider": provider,
                    "model": model,
                    "repeat": rep,
                    "retry_attempts": attempt_errors,  # may be empty if first attempt succeeded
                })
                for item in parsed.get("expected", []) or []:
                    if not isinstance(item, dict):
                        continue
                    crit = (item.get("criterion") or "").strip()
                    if not crit:
                        continue
                    votes_expected.setdefault(crit, []).append({
                        "label": label,
                        "status": str(item.get("status") or ""),
                        "evidence": str(item.get("evidence") or "")[:500],
                    })
                for item in parsed.get("prohibited", []) or []:
                    if not isinstance(item, dict):
                        continue
                    crit = (item.get("criterion") or "").strip()
                    if not crit:
                        continue
                    votes_prohibited.setdefault(crit, []).append({
                        "label": label,
                        "status": str(item.get("status") or ""),
                        "evidence": str(item.get("evidence") or "")[:500],
                    })

        if not any_call_succeeded:
            raise RuntimeError(f"All harness judges failed; last error: {last_error}")

        # Build aggregated expected/prohibited arrays in the order the
        # rubric defines them, so finalize_*_judgment matches weights
        # positionally. The normalized `criteria` dict carries the
        # expected_items / prohibited_items lists at the top level.
        rubric_expected = criteria.get("expected_items") or criteria.get("expected") or []
        rubric_prohibited = criteria.get("prohibited_items") or criteria.get("prohibited") or []
        expected_aggregated: list[dict[str, Any]] = []
        for spec in rubric_expected:
            crit_text = (spec.get("criterion") if isinstance(spec, dict) else "") or ""
            crit_text = crit_text.strip()
            vs = votes_expected.get(crit_text, [])
            sat = sum(1 for v in vs if v["status"] == "met")
            status = _vote_expected(sat, len(vs))
            expected_aggregated.append({
                "criterion": crit_text,
                "status": status,
                "satisfied_votes": sat,
                "total_votes": len(vs),
                "votes": vs,
                "evidence": " | ".join(f"[{v['label']} {v['status']}] {v['evidence']}" for v in vs)[:1500],
            })

        prohibited_aggregated: list[dict[str, Any]] = []
        for spec in rubric_prohibited:
            crit_text = (spec.get("criterion") if isinstance(spec, dict) else "") or ""
            crit_text = crit_text.strip()
            vs = votes_prohibited.get(crit_text, [])
            sat = sum(1 for v in vs if v["status"] == "not_violated")
            status = _vote_prohibited(sat, len(vs))
            prohibited_aggregated.append({
                "criterion": crit_text,
                "status": status,
                "satisfied_votes": sat,
                "total_votes": len(vs),
                "votes": vs,
                "evidence": " | ".join(f"[{v['label']} {v['status']}] {v['evidence']}" for v in vs)[:1500],
            })

        aggregated = {
            "expected": expected_aggregated,
            "prohibited": prohibited_aggregated,
            "holistic": {},  # voting over 1-5 numeric scales not implemented; left empty
            "summary": "Aggregated via verifier-harness (voting across multiple judges).",
        }
        if rubric_style == "weighted_positive":
            judgment = finalize_weighted_positive_judgment(aggregated, criteria)
        else:
            judgment = finalize_structured_judgment(aggregated, criteria)

        judgment["rubric_style"] = rubric_style
        judgment["_meta"] = {
            "judge_model_name": "verifier-harness:" + ",".join(judge_model_names),
            "harness_calls": per_call_meta,
            "usage_metrics": accumulated_usage_metrics,
        }
        return judgment


def _add_usage(acc: dict[str, Any], provider: str, model: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Sum usage metrics across multiple judge calls."""
    one = build_usage_metrics(provider, model, raw)
    if not one:
        return acc
    out = dict(acc) if acc else {}
    for k, v in one.items():
        if isinstance(v, (int, float)) and v is not None:
            out[k] = (out.get(k) or 0) + v
        else:
            out.setdefault(k, v)
    return out


def _is_safety_refusal(exc: BadRequestError) -> bool:
    message = str(exc).lower()
    return "safety" in message or "invalid_prompt" in message


def _criterion_weight_maps(criteria: dict[str, Any], key: str) -> tuple[dict[str, float], list[float]]:
    items = criteria.get(key) or []
    weights_by_text: dict[str, float] = {}
    positional_weights: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("criterion") or "").strip()
        weight = item.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            weight = 1.0
        positional_weights.append(float(weight))
        if text:
            weights_by_text[text] = float(weight)
    return weights_by_text, positional_weights


def _weighted_points(items: list[dict[str, Any]], score_map: dict[str, float], weights_by_text: dict[str, float], positional_weights: list[float]) -> tuple[float, float]:
    points = 0.0
    total_weight = 0.0
    for index, item in enumerate(items):
        criterion = str(item.get("criterion") or "").strip()
        weight = weights_by_text.get(criterion)
        if weight is None:
            weight = positional_weights[index] if index < len(positional_weights) else 1.0
        total_weight += weight
        points += weight * score_map.get(item.get("status", "unclear"), 0.25)
    return points, total_weight


def finalize_structured_judgment(data: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    expected = data.get("expected") or []
    prohibited = data.get("prohibited") or []
    holistic = data.get("holistic") or {}

    expected_weight_map, expected_positional_weights = _criterion_weight_maps(criteria, "expected_items")
    prohibited_weight_map, prohibited_positional_weights = _criterion_weight_maps(criteria, "prohibited_items")

    expected_points, expected_max = _weighted_points(expected, EXPECTED_SCORES, expected_weight_map, expected_positional_weights)
    expected_coverage = expected_points / expected_max if expected_max else None

    prohibited_points, prohibited_max = _weighted_points(prohibited, PROHIBITED_SCORES, prohibited_weight_map, prohibited_positional_weights)
    prohibited_rate = prohibited_points / prohibited_max if prohibited_max else 0.0
    prohibited_compliance = 1.0 - prohibited_rate if prohibited_max else 1.0
    final_score = 100.0 * ((EXPECTED_WEIGHT * (expected_coverage or 0.0)) + (PROHIBITED_WEIGHT * prohibited_compliance))

    return {
        "expected": expected,
        "prohibited": prohibited,
        "holistic": holistic,
        "summary": data.get("summary", ""),
        "rubric_style": "legacy_structured",
        "scores": {
            "expected_points": expected_points,
            "expected_max": expected_max,
            "expected_coverage": expected_coverage,
            "prohibited_points": prohibited_points,
            "prohibited_max": prohibited_max,
            "prohibited_rate": prohibited_rate,
            "prohibited_compliance": prohibited_compliance,
            "final_score": final_score,
        },
    }


def finalize_weighted_positive_judgment(data: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    expected = data.get("expected") or []
    holistic = data.get("holistic") or {}

    expected_weight_map, expected_positional_weights = _criterion_weight_maps(criteria, "expected_items")
    rubric_points, rubric_max = _weighted_points(expected, EXPECTED_SCORES, expected_weight_map, expected_positional_weights)
    rubric_score = rubric_points / rubric_max if rubric_max else None
    final_score = 100.0 * rubric_score if rubric_score is not None else None

    return {
        "expected": expected,
        "prohibited": [],
        "holistic": holistic,
        "summary": data.get("summary", ""),
        "rubric_style": "weighted_positive",
        "scores": {
            "rubric_points": rubric_points,
            "rubric_max": rubric_max,
            "rubric_score": rubric_score,
            "final_score": final_score,
        },
    }


def finalize_rubric_light_judgment(data: dict[str, Any]) -> dict[str, Any]:
    holistic = data.get("holistic") or {}
    score_values = []
    for key in ("factuality", "completeness", "directness", "actionability"):
        value = holistic.get(key)
        if isinstance(value, (int, float)):
            score_values.append(float(value))

    average_score = sum(score_values) / len(score_values) if score_values else None
    return {
        "holistic": holistic,
        "summary": data.get("summary", ""),
        "rubric_style": "rubric_light",
        "scores": {
            "average_holistic_score": average_score,
        },
        "no_structured_rubric": True,
    }
