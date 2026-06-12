#!/usr/bin/env python3
"""Download and convert UK theory test questions into eval question-group format."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.request import urlopen

from generated.eval_question_models import (
    Model as EvalQuestionGroupsModel,
)
from generated.eval_question_models import (
    NonEmptyOptionIds,
)
from generated.eval_question_models import (
    Option as EvalOption,
)
from generated.eval_question_models import (
    Question as EvalQuestionModel,
)
from generated.eval_question_models import (
    QuestionGroup as EvalQuestionGroupModel,
)
from utils.repo import REPO_ROOT

DATASET_NAME = "UK_THEORY_TEST"
LANGUAGE = "en"
SOURCE_URL = "https://github.com/PeiDashuai/LLMs_Nav/blob/main/UK_THEORY_TEST.json"
RAW_URL = (
    "https://raw.githubusercontent.com/PeiDashuai/LLMs_Nav/main/UK_THEORY_TEST.json"
)

DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "evals" / "pei2024application" / "uk_theory_test.json"
)


def _normalize_question_number(raw_value: object, fallback_index: int) -> str:
    if isinstance(raw_value, int):
        return str(raw_value)
    if isinstance(raw_value, str):
        cleaned = re.sub(r"[^\d]+", "", raw_value)
        if cleaned:
            return str(int(cleaned))
    return str(fallback_index)


def _normalize_correct_answers(raw_value: object) -> list[str]:
    values: list[str]
    if isinstance(raw_value, list):
        values = [str(item) for item in raw_value]
    elif isinstance(raw_value, str):
        values = [raw_value]
    else:
        values = []

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().upper()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _load_source_questions() -> list[dict[str, object]]:
    with urlopen(RAW_URL, timeout=60) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Expected root JSON array from UK theory dataset")
    return [item for item in payload if isinstance(item, dict)]


def _build_group(source_questions: list[dict[str, object]]) -> EvalQuestionGroupModel:
    questions: list[EvalQuestionModel] = []
    used_ids: set[str] = set()

    for index, row in enumerate(source_questions, start=1):
        question_text = str(row.get("question_text", "")).strip()
        if not question_text:
            continue

        raw_number = row.get("question_number")
        normalized_number = _normalize_question_number(raw_number, index)
        base_id = f"pei2024application-uk-{normalized_number.zfill(4)}"
        question_id = base_id
        suffix = 2
        while question_id in used_ids:
            question_id = f"{base_id}-dup{suffix}"
            suffix += 1
        used_ids.add(question_id)

        raw_choices = row.get("choices", [])
        options: list[EvalOption] = []
        if isinstance(raw_choices, list):
            for choice in raw_choices:
                if not isinstance(choice, dict):
                    continue
                choice_dict = cast(dict[str, object], choice)
                option_id = str(choice_dict.get("choice_letter", "")).strip().upper()
                option_text = str(choice_dict.get("choice_text", "")).strip()
                if not option_id or not option_text:
                    continue
                options.append(EvalOption(id=option_id, text=option_text, images=[]))

        if not options:
            continue

        valid_option_ids = {option.id for option in options}
        correct_option_ids = [
            option_id
            for option_id in _normalize_correct_answers(row.get("correct_answer"))
            if option_id in valid_option_ids
        ]
        if not correct_option_ids:
            continue

        questions.append(
            EvalQuestionModel(
                id=question_id,
                questionText=question_text,
                metadata={
                    "dataset": DATASET_NAME,
                    "language": LANGUAGE,
                    "originalQuestionNumber": str(raw_number),
                    "normalizedQuestionNumber": normalized_number,
                    "originalIndex": index,
                },
                images=[],
                source={
                    "repository": "PeiDashuai/LLMs_Nav",
                    "sourceFile": "UK_THEORY_TEST.json",
                    "originalQuestionNumber": normalized_number,
                },
                options=options,
                correctOptionIds=NonEmptyOptionIds(root=correct_option_ids),
            )
        )

    return EvalQuestionGroupModel(
        id="pei2024application-uk",
        metadata={
            "dataset": DATASET_NAME,
            "language": LANGUAGE,
            "questionCount": len(questions),
            "convertedAtUtc": datetime.now(UTC).isoformat(),
        },
        source={
            "provider": "github",
            "repository": "PeiDashuai/LLMs_Nav",
            "sourceUrl": SOURCE_URL,
            "rawUrl": RAW_URL,
            "sourceFile": "UK_THEORY_TEST.json",
            "paperTag": "pei2024application",
        },
        questions=questions,
    )


def main(output_path: Path = DEFAULT_OUTPUT) -> None:
    source_questions = _load_source_questions()
    group = _build_group(source_questions)
    payload = EvalQuestionGroupsModel(root=[group]).model_dump(
        mode="json", exclude_none=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(group.questions)} questions to {output_path}")


if __name__ == "__main__":
    main()
