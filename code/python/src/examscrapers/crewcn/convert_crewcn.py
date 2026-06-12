#!/usr/bin/env python3
"""Convert CrewCN exam questions into eval question-group format.

Reads the raw flat-array JSON from data/evals/mcq/CREWCN_EXAM_QUESTIONS.json
and writes QuestionGroup-schema JSON to data/evals/crewcn/.

Usage:
    cd code/python
    uv run python -m examscrapers.crewcn.convert_crewcn
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

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

SOURCE_FILE = REPO_ROOT / "data" / "evals" / "mcq" / "CREWCN_EXAM_QUESTIONS.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "evals" / "crewcn" / "crewcn_exam_questions.json"


def _load_source_questions() -> list[dict]:
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Expected root JSON array from CrewCN dataset")
    return [item for item in payload if isinstance(item, dict)]


def _build_groups(source_questions: list[dict]) -> list[EvalQuestionGroupModel]:
    # Group questions by source file_name
    by_source: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(source_questions, start=1):
        source = row.get("source", {})
        file_name = (
            source.get("file_name", "unknown")
            if isinstance(source, dict)
            else "unknown"
        )
        by_source[file_name].append((idx, row))

    groups: list[EvalQuestionGroupModel] = []
    used_ids: set[str] = set()

    for file_name, rows in by_source.items():
        # Create a slug from the file name for the group id
        slug = (
            Path(file_name)
            .stem.lower()
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )
        group_id = f"crewcn-{slug}"

        questions: list[EvalQuestionModel] = []

        for original_index, row in rows:
            question_text = str(row.get("question", "")).strip()
            if not question_text:
                continue

            # Parse options from {A: text, B: text, ...} dict
            raw_options = row.get("options", {})
            if not isinstance(raw_options, dict):
                continue

            options: list[EvalOption] = []
            for option_id in sorted(raw_options.keys()):
                option_text = str(raw_options[option_id]).strip()
                if not option_text:
                    continue
                options.append(EvalOption(id=option_id.upper(), text=option_text))

            if not options:
                continue

            # Parse answer
            raw_answer = row.get("answer", "")
            if not raw_answer or not isinstance(raw_answer, str):
                continue
            correct_ids = [
                a.strip().upper() for a in raw_answer.split(",") if a.strip()
            ]
            valid_option_ids = {o.id for o in options}
            correct_ids = [a for a in correct_ids if a in valid_option_ids]
            if not correct_ids:
                continue

            # Generate unique question id
            raw_id = row.get("id", original_index)
            base_id = (
                f"crewcn-{raw_id:04d}"
                if isinstance(raw_id, int)
                else f"crewcn-{raw_id}"
            )
            question_id = base_id
            suffix = 2
            while question_id in used_ids:
                question_id = f"{base_id}-dup{suffix}"
                suffix += 1
            used_ids.add(question_id)

            source_info = row.get("source", {})

            questions.append(
                EvalQuestionModel(
                    id=question_id,
                    questionText=question_text,
                    metadata={
                        "dataset": "CREWCN_EXAM_QUESTIONS",
                        "originalId": row.get("id"),
                        "originalIndex": original_index,
                        "hint": row.get("hint"),
                        "explanation": row.get("explanation"),
                    },
                    messages=[],
                    images=[],
                    source={
                        k: str(v)
                        for k, v in (
                            source_info if isinstance(source_info, dict) else {}
                        ).items()
                        if v is not None
                    },
                    options=options,
                    correctOptionIds=NonEmptyOptionIds(root=correct_ids),
                )
            )

        if not questions:
            continue

        # Build source dict from first question's source
        first_source = rows[0][1].get("source", {})
        group_source = {
            "provider": "crewcn",
            "sourceFile": "CREWCN_EXAM_QUESTIONS.json",
        }
        if isinstance(first_source, dict):
            if first_source.get("parent_url"):
                group_source["parentUrl"] = str(first_source["parent_url"])
            if first_source.get("exam_title"):
                group_source["examTitle"] = str(first_source["exam_title"])
            if first_source.get("file_name"):
                group_source["fileName"] = str(first_source["file_name"])

        groups.append(
            EvalQuestionGroupModel(
                id=group_id,
                metadata={
                    "dataset": "CREWCN_EXAM_QUESTIONS",
                    "sourceFileName": file_name,
                    "questionCount": len(questions),
                    "convertedAtUtc": datetime.now(UTC).isoformat(),
                },
                source=group_source,
                questions=questions,
            )
        )

    return groups


def main(output_path: Path = DEFAULT_OUTPUT) -> None:
    source_questions = _load_source_questions()
    groups = _build_groups(source_questions)

    total_questions = sum(len(g.questions) for g in groups)
    payload = EvalQuestionGroupsModel(root=groups).model_dump(
        mode="json", exclude_none=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {total_questions} questions in {len(groups)} groups to {output_path}")


if __name__ == "__main__":
    main()
