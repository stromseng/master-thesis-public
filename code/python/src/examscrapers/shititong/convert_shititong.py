#!/usr/bin/env python3
"""Convert Shititong deduped exam questions into eval question-group format.

Reads the deduped JSON from data/evals/mcq/SHITITONG_*_DEDUPED.json
and writes QuestionGroup-schema JSON to data/evals/shititong/.

Only MCQ-compatible types are converted (single_choice / 单选题,
multiple_choice / 多选题, true_false / 判断题).
Fill-in-blank and short-answer types are skipped.

Splits each dataset into two files:
- *_text.json: Questions without <img> tags (text-only model)
- *_vision.json: Questions with <img> tags (vision model)

Usage:
    cd code/python
    uv run python -m examscrapers.shititong.convert_shititong --target english-deduped
    uv run python -m examscrapers.shititong.convert_shititong --target chinese-deduped
    uv run python -m examscrapers.shititong.convert_shititong --target all
"""

from __future__ import annotations

import argparse
import json
import re
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
from generated.eval_question_models import (
    QuestionImage,
)
from utils.repo import REPO_ROOT

MCQ_DIR = REPO_ROOT / "data" / "evals" / "mcq"
OUTPUT_DIR = REPO_ROOT / "data" / "evals" / "shititong"

# Question types that fit the MCQ schema
MCQ_TYPES_ZH = {"单选题", "多选题", "判断题"}
MCQ_TYPES_EN = {"single_choice", "multiple_choice", "true_false"}
MCQ_TYPES = MCQ_TYPES_ZH | MCQ_TYPES_EN

TARGETS: dict[str, tuple[str, str, str, str]] = {
    # target_name: (source_filename, output_text_filename, output_vision_filename, language)
    "english-deduped": (
        "SHITITONG_ENGLISH_DEDUPED.json",
        "shititong_english_deduped_text.json",
        "shititong_english_deduped_vision.json",
        "en",
    ),
    "chinese-deduped": (
        "SHITITONG_CHINESE_DEDUPED.json",
        "shititong_chinese_deduped_text.json",
        "shititong_chinese_deduped_vision.json",
        "zh",
    ),
}


# Matches <img> tags in various formats:
#   <img src = URL />           (self-closing, unquoted)
#   <img src="URL">             (quoted, not self-closing)
#   <img src="URL" alt="text">  (with extra attributes)
_IMG_TAG_RE = re.compile(r"<img\s[^>]*>", re.IGNORECASE)
_IMG_SRC_RE = re.compile(
    r"""src\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+?)\s*/>|(\S+?)(?:\s|>))"""
)


def _extract_images(text: str) -> tuple[str, list[QuestionImage], bool]:
    """Extract <img> tags from text, returning cleaned text, image list, and validity flag.

    Handles multiple <img> formats (quoted/unquoted src, self-closing or not).
    Null/empty/placeholder src values are stripped but not added to the images list.
    Returns has_invalid=True if any non-empty src is a local/unfetchable reference
    (not http:// or https://), signalling the caller to drop the question.
    """
    images: list[QuestionImage] = []
    has_invalid = False

    for tag_match in _IMG_TAG_RE.finditer(text):
        tag = tag_match.group(0)
        src_match = _IMG_SRC_RE.search(tag)
        if src_match:
            # Pick whichever capture group matched
            src = next((g for g in src_match.groups() if g is not None), "")
            src = src.strip().rstrip("/").strip().strip("\"'")
            if not src or src.lower() in ("null", "..."):
                continue
            if not src.startswith(("http://", "https://")):
                has_invalid = True
            else:
                images.append(QuestionImage(uri=src))

    # Strip all img tags and surrounding <br/> tags
    cleaned = _IMG_TAG_RE.sub("", text)
    cleaned = re.sub(r"<br\s*/>\s*$", "", cleaned)
    cleaned = cleaned.strip()

    return cleaned, images, has_invalid


def _has_images(question: EvalQuestionModel) -> bool:
    """Check if question has any images (on question or options)."""
    if question.images:
        return True
    for option in question.options:
        if option.images:
            return True
    return False


def _load_source_questions(source_path: Path) -> tuple[list[dict], dict]:
    """Load questions from a shititong JSON file.

    Returns (questions_list, file_metadata).
    """
    with source_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        questions = payload.get("questions", [])
        metadata = {k: v for k, v in payload.items() if k != "questions"}
        return questions, metadata
    if isinstance(payload, list):
        return payload, {}
    raise ValueError(f"Unexpected root type in {source_path.name}")


def _build_groups(
    source_questions: list[dict],
    dataset_name: str,
    language: str,
    source_filename: str,
) -> list[EvalQuestionGroupModel]:
    # Group questions by exam_title from source
    by_exam: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(source_questions, start=1):
        if not isinstance(row, dict):
            continue
        # Filter to MCQ types only
        q_type = row.get("type", "")
        if q_type and q_type not in MCQ_TYPES:
            continue
        source = row.get("source", {})
        exam_title = (
            source.get("exam_title", "unknown")
            if isinstance(source, dict)
            else "unknown"
        )
        by_exam[exam_title].append((idx, row))

    groups: list[EvalQuestionGroupModel] = []
    used_ids: set[str] = set()
    id_prefix = f"shititong-{language}"

    for exam_title, rows in by_exam.items():
        # Create slug from exam title
        slug = (
            exam_title[:60]
            .lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace(".", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
        )
        group_id = f"{id_prefix}-{slug}"

        questions: list[EvalQuestionModel] = []

        for original_index, row in rows:
            raw_question_text = str(row.get("question", "")).strip()
            if not raw_question_text:
                continue

            # Extract images from question text
            question_text, question_images, q_has_invalid = _extract_images(
                raw_question_text
            )
            if q_has_invalid:
                continue  # drop question with unfetchable local image refs

            # Parse options from {A: text, B: text, ...} dict
            raw_options = row.get("options", {})
            if not isinstance(raw_options, dict) or not raw_options:
                continue

            options: list[EvalOption] = []
            skip_question = False
            for option_id in sorted(raw_options.keys()):
                option_text = str(raw_options[option_id]).strip()
                if not option_text:
                    continue
                # Extract images from option text
                cleaned_option_text, option_images, opt_has_invalid = _extract_images(
                    option_text
                )
                if opt_has_invalid:
                    skip_question = True
                    break
                options.append(
                    EvalOption(
                        id=option_id.upper(),
                        text=cleaned_option_text,
                        images=option_images if option_images else None,
                    )
                )

            if skip_question or not options:
                continue

            # Parse answer - comma-separated ("A,B") or concatenated ("ABC")
            raw_answer = row.get("answer", "")
            if not raw_answer or not isinstance(raw_answer, str):
                continue
            valid_option_ids = {o.id for o in options}
            # Try comma-separated first
            correct_ids = [
                a.strip().upper() for a in raw_answer.split(",") if a.strip()
            ]
            correct_ids = [a for a in correct_ids if a in valid_option_ids]
            # Fall back to splitting each character (e.g. "ABC" -> ["A","B","C"])
            if not correct_ids:
                correct_ids = [
                    c.upper()
                    for c in raw_answer.strip()
                    if c.upper() in valid_option_ids
                ]
            if not correct_ids:
                continue

            # Generate unique question id
            raw_id = row.get("id", original_index)
            base_id = (
                f"{id_prefix}-{raw_id:06d}"
                if isinstance(raw_id, int)
                else f"{id_prefix}-{raw_id}"
            )
            question_id = base_id
            suffix = 2
            while question_id in used_ids:
                question_id = f"{base_id}-dup{suffix}"
                suffix += 1
            used_ids.add(question_id)

            source_info = row.get("source", {})

            metadata: dict = {
                "dataset": dataset_name,
                "language": language,
                "originalId": row.get("id"),
                "originalIndex": original_index,
                "questionType": row.get("type"),
            }
            if row.get("hint"):
                metadata["hint"] = row["hint"]
            if row.get("explanation"):
                metadata["explanation"] = row["explanation"]
            if row.get("difficulty"):
                metadata["difficulty"] = row["difficulty"]
            if row.get("category_id"):
                metadata["categoryId"] = row["category_id"]
            if row.get("original_id"):
                metadata["shititongId"] = row["original_id"]

            source_dict: dict[str, str] = {
                "provider": "shititong",
                "sourceFile": source_filename,
            }
            if isinstance(source_info, dict):
                if source_info.get("exam_url"):
                    source_dict["examUrl"] = str(source_info["exam_url"])
                if source_info.get("exam_title"):
                    source_dict["examTitle"] = str(source_info["exam_title"])

            questions.append(
                EvalQuestionModel(
                    id=question_id,
                    questionText=question_text,
                    metadata=metadata,
                    images=question_images if question_images else None,
                    source=source_dict,
                    options=options,
                    correctOptionIds=NonEmptyOptionIds(root=correct_ids),
                )
            )

        if not questions:
            continue

        first_source = rows[0][1].get("source", {})
        group_source: dict[str, str] = {
            "provider": "shititong",
            "sourceFile": source_filename,
        }
        if isinstance(first_source, dict) and first_source.get("exam_title"):
            group_source["examTitle"] = str(first_source["exam_title"])

        groups.append(
            EvalQuestionGroupModel(
                id=group_id,
                metadata={
                    "dataset": dataset_name,
                    "language": language,
                    "examTitle": exam_title,
                    "questionCount": len(questions),
                    "convertedAtUtc": datetime.now(UTC).isoformat(),
                },
                source=group_source,
                questions=questions,
            )
        )

    return groups


def convert_target(target_name: str) -> None:
    source_filename, output_text_filename, output_vision_filename, language = TARGETS[
        target_name
    ]
    source_path = MCQ_DIR / source_filename
    output_text_path = OUTPUT_DIR / output_text_filename
    output_vision_path = OUTPUT_DIR / output_vision_filename
    dataset_name = Path(source_filename).stem

    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        return

    source_questions, file_metadata = _load_source_questions(source_path)
    print(f"Loaded {len(source_questions)} raw questions from {source_filename}")

    groups = _build_groups(source_questions, dataset_name, language, source_filename)
    total_questions = sum(len(g.questions) for g in groups)

    # Split groups into text-only and vision based on presence of <img> tags
    text_groups: list[EvalQuestionGroupModel] = []
    vision_groups: list[EvalQuestionGroupModel] = []

    for group in groups:
        text_questions: list[EvalQuestionModel] = []
        vision_questions: list[EvalQuestionModel] = []

        for question in group.questions:
            if _has_images(question):
                vision_questions.append(question)
            else:
                text_questions.append(question)

        # Create separate groups for text and vision if they have questions
        if text_questions:
            text_group = EvalQuestionGroupModel(
                id=group.id,
                metadata={
                    **group.metadata,
                    "questionCount": len(text_questions),
                    "convertedAtUtc": datetime.now(UTC).isoformat(),
                },
                source=group.source,
                questions=text_questions,
            )
            text_groups.append(text_group)

        if vision_questions:
            vision_group = EvalQuestionGroupModel(
                id=f"{group.id}-vision",
                metadata={
                    **group.metadata,
                    "questionCount": len(vision_questions),
                    "convertedAtUtc": datetime.now(UTC).isoformat(),
                },
                source=group.source,
                questions=vision_questions,
            )
            vision_groups.append(vision_group)

    # Write text-only questions
    text_payload = EvalQuestionGroupsModel(root=text_groups).model_dump(
        mode="json", exclude_none=True
    )
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    with output_text_path.open("w", encoding="utf-8") as f:
        json.dump(text_payload, f, ensure_ascii=False, indent=2)

    text_count = sum(len(g.questions) for g in text_groups)
    print(
        f"Wrote {text_count} text-only questions in {len(text_groups)} groups to {output_text_path}"
    )

    # Write vision questions
    vision_payload = EvalQuestionGroupsModel(root=vision_groups).model_dump(
        mode="json", exclude_none=True
    )
    with output_vision_path.open("w", encoding="utf-8") as f:
        json.dump(vision_payload, f, ensure_ascii=False, indent=2)

    vision_count = sum(len(g.questions) for g in vision_groups)
    print(
        f"Wrote {vision_count} vision questions in {len(vision_groups)} groups to {output_vision_path}"
    )

    skipped = len(source_questions) - total_questions
    if skipped:
        print(f"(skipped {skipped} non-MCQ or invalid questions)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Shititong MCQ data to eval schema"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=[*TARGETS.keys(), "all"],
        help="Which dataset to convert",
    )
    args = parser.parse_args()

    if args.target == "all":
        for target_name in TARGETS:
            print(f"\n--- Converting {target_name} ---")
            convert_target(target_name)
    else:
        convert_target(args.target)


if __name__ == "__main__":
    main()
