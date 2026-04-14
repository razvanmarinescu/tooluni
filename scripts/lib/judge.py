from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .dataset import infer_rubric_style
from .usage_costs import build_usage_metrics


EXPECTED_SCORES = {"met": 1.0, "partial": 0.5, "missed": 0.0, "unclear": 0.25}
PROHIBITED_SCORES = {"not_violated": 0.0, "violated": 1.0, "unclear": 0.5}


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
    def __init__(self, project_root: Path, model_name: str = "gpt-5.4"):
        load_local_env(project_root)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    def judge_response(self, question: str, criteria: dict[str, Any], response_text: str) -> dict[str, Any]:
        has_structured_rubric = bool(criteria.get("expected") or criteria.get("prohibited"))
        rubric_style = str(criteria.get("rubric_style") or infer_rubric_style(criteria.get("raw")))
        prompt = (
            build_judge_prompt(question, criteria, response_text)
            if has_structured_rubric
            else build_rubric_light_prompt(question, response_text)
        )
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=2500,
        )
        raw = response.model_dump(mode="json")
        parsed = extract_json_object(response.output_text)
        if has_structured_rubric:
            if rubric_style == "weighted_positive":
                judgment = finalize_weighted_positive_judgment(parsed, criteria)
            else:
                judgment = finalize_structured_judgment(parsed, criteria)
        else:
            judgment = finalize_rubric_light_judgment(parsed)
        judgment["rubric_style"] = rubric_style
        judgment["_meta"] = {
            "judge_model_name": self.model_name,
            "usage_metrics": build_usage_metrics("openai", self.model_name, raw),
        }
        return judgment


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
    final_score = 100.0 * ((0.8 * (expected_coverage or 0.0)) + (0.2 * prohibited_compliance))
    if prohibited_points > 0:
        final_score = min(final_score, 74.0)

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
