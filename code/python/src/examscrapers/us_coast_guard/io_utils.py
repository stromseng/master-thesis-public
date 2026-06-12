from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from generated.eval_question_models import (
    Model as EvalQuestionGroupsModel,
    Question as EvalQuestionModel,
    QuestionGroup as EvalQuestionGroupModel,
)
from utils.repo import REPO_ROOT

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "evals_to_parse" / "scraped_coast_guard"
DEFAULT_MARKDOWN_ROOT_DIR = (
    REPO_ROOT / "data" / "evals_to_parse" / "coast_guard_markdown"
)
DEFAULT_MARKDOWN_EXAMS_DIR = DEFAULT_MARKDOWN_ROOT_DIR / "exams"
DEFAULT_EVAL_DIR = REPO_ROOT / "data" / "evals" / "us_coast_guard"
DEFAULT_EVAL_IMAGES_DIR = DEFAULT_EVAL_DIR / "images"
DEFAULT_INTERMEDIATE_DIR = DEFAULT_EVAL_DIR / "intermediate"


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_exam_group(
    exam_stem: str,
    exam_code: str,
    exam_title: str,
    exam_file: str,
    questions: list[EvalQuestionModel],
) -> list[dict[str, Any]]:
    group = EvalQuestionGroupModel(
        id=f"uscg-{exam_code}",
        metadata={
            "examCode": exam_code,
            "examTitle": exam_title,
            "questionCount": len(questions),
        },
        source={
            "provider": "uscg",
            "examFile": exam_file,
            "examStem": exam_stem,
        },
        questions=questions,
    )
    return EvalQuestionGroupsModel(root=[group]).model_dump(
        mode="json", exclude_none=True
    )


def load_existing_exam_meta(final_path: Path, exam_stem: str) -> dict[str, str]:
    exam_code = exam_stem.split("_", maxsplit=1)[0]
    default = {
        "exam_code": exam_code,
        "exam_title": exam_stem.replace("_", " "),
        "exam_file": f"{exam_stem}.pdf",
    }
    if not final_path.exists():
        return default

    try:
        payload = read_json(final_path)
    except Exception:
        return default

    if not isinstance(payload, list) or not payload:
        return default

    first = payload[0]
    if not isinstance(first, dict):
        return default

    metadata_obj = first.get("metadata")
    source_obj = first.get("source")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    source = source_obj if isinstance(source_obj, dict) else {}

    return {
        "exam_code": str(metadata.get("examCode") or default["exam_code"]),
        "exam_title": str(metadata.get("examTitle") or default["exam_title"]),
        "exam_file": str(source.get("examFile") or default["exam_file"]),
    }
