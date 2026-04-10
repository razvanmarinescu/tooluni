from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .usage_costs import build_usage_metrics


EXPECTED_SCORES = {"met": 1.0, "partial": 0.5, "missed": 0.0, "unclear": 0.25}
PROHIBITED_SCORES = {"not_violated": 0.0, "violated": 1.0, "unclear": 0.5}


def load_local_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)


def build_judge_prompt(question: str, criteria: dict[str, Any], response_text: str) -> str:
    return (
        "You are scoring one model response against a rubric.\n"
        "Return JSON only. Do not add markdown or commentary outside the JSON object.\n\n"
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
            judgment = finalize_structured_judgment(parsed)
        else:
            judgment = finalize_rubric_light_judgment(parsed)
        judgment["_meta"] = {
            "judge_model_name": self.model_name,
            "usage_metrics": build_usage_metrics("openai", self.model_name, raw),
        }
        return judgment


def finalize_structured_judgment(data: dict[str, Any]) -> dict[str, Any]:
    expected = data.get("expected") or []
    prohibited = data.get("prohibited") or []
    holistic = data.get("holistic") or {}

    expected_points = 0.0
    for item in expected:
        expected_points += EXPECTED_SCORES.get(item.get("status", "unclear"), 0.25)

    expected_max = len(expected)
    expected_coverage = expected_points / expected_max if expected_max else None

    prohibited_points = 0.0
    for item in prohibited:
        prohibited_points += PROHIBITED_SCORES.get(item.get("status", "unclear"), 0.5)

    prohibited_max = len(prohibited)
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
        "scores": {
            "average_holistic_score": average_score,
        },
        "no_structured_rubric": True,
    }
