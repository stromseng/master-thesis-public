#!/usr/bin/env python3
"""Build USCG eval JSON from markdown using deterministic extraction."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any, cast

from rapidfuzz import fuzz
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from examscrapers.us_coast_guard.deterministic_markdown_parser import (
    parse_exam_markdown,
)
from examscrapers.us_coast_guard.io_utils import (
    DEFAULT_EVAL_DIR,
    DEFAULT_EVAL_IMAGES_DIR,
    DEFAULT_INTERMEDIATE_DIR,
    DEFAULT_MARKDOWN_EXAMS_DIR,
    load_existing_exam_meta,
    now_utc_iso,
    serialize_exam_group,
    write_json,
)
from generated.eval_question_models import (
    Model as EvalQuestionGroupsModel,
    Option as EvalOption,
    Question as EvalQuestionModel,
    QuestionGroup as EvalQuestionGroupModel,
    QuestionImage,
)

console = Console()
logger = logging.getLogger("uscg_markdown_to_eval")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class MarkdownInput:
    exam_stem: str
    markdown_path: Path
    source_images_dir: Path | None = None


@dataclass
class DedupCluster:
    canonical: EvalQuestionModel
    question_blob: str
    options_blob: str
    answer_key: str
    illustration_ids: set[str]
    source_ids: set[str] = field(default_factory=set)
    source_exam_stems: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)


@dataclass
class ParsedExamWork:
    item: MarkdownInput
    exam_meta: dict[str, str]
    questions: list[EvalQuestionModel]
    illustration_mapping: dict[str, list[str]]
    parse_report: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rebuild USCG eval JSON files from deterministic markdown parsing.",
    )
    parser.add_argument(
        "--markdown-exams-dir",
        type=Path,
        default=DEFAULT_MARKDOWN_EXAMS_DIR,
        help="Directory containing exam markdown folders (or legacy flat q*.md files).",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR,
        help="Directory for per-exam q*.json outputs.",
    )
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_EVAL_IMAGES_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR / "markdown_to_eval_report.json",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=DEFAULT_INTERMEDIATE_DIR / "markdown_to_eval_progress.json",
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--exam-stem", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dedupe-threshold",
        type=int,
        default=97,
        help="Fuzzy threshold [0-100] for merged deduplication.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    return payload


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_utc_iso()
    write_json(path, payload)


def _ensure_entries(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries_obj = state.get("entries")
    if not isinstance(entries_obj, dict):
        entries_obj = {}
        state["entries"] = entries_obj
    return cast(dict[str, dict[str, Any]], entries_obj)


def iter_markdown_inputs(markdown_exams_dir: Path) -> list[MarkdownInput]:
    # Legacy flat mode: q*.md files directly in the directory.
    flat_files = sorted(markdown_exams_dir.glob("q*.md"))
    if flat_files:
        return [
            MarkdownInput(
                exam_stem=path.stem, markdown_path=path, source_images_dir=None
            )
            for path in flat_files
        ]

    items: list[MarkdownInput] = []
    for exam_dir in sorted(
        path for path in markdown_exams_dir.iterdir() if path.is_dir()
    ):
        exam_stem = exam_dir.name
        markdown_path = exam_dir / f"{exam_stem}.md"
        if not markdown_path.exists():
            fallback = sorted(exam_dir.glob("*.md"))
            if not fallback:
                continue
            markdown_path = fallback[0]
        source_images = exam_dir / "images"
        items.append(
            MarkdownInput(
                exam_stem=exam_stem,
                markdown_path=markdown_path,
                source_images_dir=source_images if source_images.exists() else None,
            )
        )
    return items


def _normalize_text(text: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", text.strip().lower())
    return normalized


def _extract_illustration_ids(question: EvalQuestionModel) -> set[str]:
    ids: set[str] = set()
    for image in question.images or []:
        if image.id:
            ids.add(image.id.upper())
    return ids


def _options_blob(options: list[EvalOption]) -> str:
    parts = [
        f"{option.id}:{_normalize_text(option.text)}"
        for option in sorted(options, key=lambda o: o.id)
    ]
    return " | ".join(parts)


def _answer_key(question: EvalQuestionModel) -> str:
    return "|".join(question.correctOptionIds.root)


def _question_blob(question: EvalQuestionModel) -> str:
    return (
        f"{_normalize_text(question.questionText)} || {_options_blob(question.options)}"
    )


def _question_score(question: EvalQuestionModel) -> tuple[int, int, int]:
    image_count = len(question.images or [])
    option_total = sum(len(option.text) for option in question.options)
    return (
        int(image_count > 0),
        image_count,
        len(question.questionText) + option_total,
    )


def _available_image_names(source_images_dir: Path | None) -> set[str]:
    if source_images_dir is None or not source_images_dir.exists():
        return set()
    return {path.name for path in source_images_dir.iterdir() if path.is_file()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id_token(illustration_id: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9_-]+", "_", illustration_id.upper())
    return cleaned.strip("_") or "ILLUSTRATION"


def _build_id_based_filename(illustration_id: str, index: int, suffix: str) -> str:
    token = _safe_id_token(illustration_id)
    extension = suffix.lower() if suffix else ".jpg"
    if index <= 1:
        return f"{token}{extension}"
    return f"{token}_{index}{extension}"


def process_markdown_file(
    *,
    item: MarkdownInput,
    intermediate_dir: Path,
    dry_run: bool,
) -> tuple[dict[str, Any], ParsedExamWork | None]:
    exam_stem = item.exam_stem
    final_path = intermediate_dir / f"{exam_stem}.json"

    if dry_run:
        return (
            {
                "exam_stem": exam_stem,
                "status": "dry_run",
                "markdown_file": item.markdown_path.as_posix(),
                "final_file": final_path.as_posix(),
            },
            None,
        )

    markdown = item.markdown_path.read_text(encoding="utf-8", errors="ignore")
    exam_meta = load_existing_exam_meta(final_path=final_path, exam_stem=exam_stem)
    available_images = _available_image_names(item.source_images_dir)
    parsed = parse_exam_markdown(
        markdown_text=markdown,
        exam_stem=exam_stem,
        exam_meta=exam_meta,
        images_dir=item.source_images_dir or item.markdown_path.parent,
        available_images=available_images,
    )

    entry = {
        "exam_stem": exam_stem,
        "status": "ok",
        "questions_detected": parsed.report["questions_detected"],
        "questions_emitted": parsed.report["questions_emitted"],
        "questions_repaired": parsed.report["questions_repaired"],
        "questions_skipped_missing_answer": parsed.report[
            "questions_skipped_missing_answer"
        ],
        "questions_skipped_incomplete_options": parsed.report[
            "questions_skipped_incomplete_options"
        ],
        "duplicate_candidates_discarded": parsed.report[
            "duplicate_candidates_discarded"
        ],
        "illustration_ids_detected": parsed.report["illustration_ids_detected"],
        "illustration_ids_resolved": parsed.report["illustration_ids_resolved"],
        "questions_with_images": parsed.report["questions_with_images"],
        "questions_with_illustration_keyword_missing_images": parsed.report.get(
            "questions_with_illustration_keyword_missing_images", 0
        ),
        "questions_with_detected_illustration_ids_missing_images": parsed.report.get(
            "questions_with_detected_illustration_ids_missing_images", 0
        ),
        "questions_invalid_missing_visual_reference": parsed.report.get(
            "questions_invalid_missing_visual_reference", 0
        ),
        "illustration_image_link_gap_examples": parsed.report.get(
            "detected_illustration_ids_missing_images_examples", []
        )[:20],
        "invalid_missing_visual_reference_examples": parsed.report.get(
            "invalid_missing_visual_reference_examples", []
        )[:20],
        "illustration_images_detected": sum(
            len(image_names) for image_names in parsed.illustration_mapping.values()
        ),
        "issues": parsed.report.get("issues", [])[:80],
        "markdown_file": item.markdown_path.as_posix(),
        "final_file": final_path.as_posix(),
    }
    return (
        entry,
        ParsedExamWork(
            item=item,
            exam_meta=exam_meta,
            questions=parsed.questions,
            illustration_mapping=parsed.illustration_mapping,
            parse_report=parsed.report,
        ),
    )


def _find_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def dedupe_and_copy_illustration_images(
    parsed_works: list[ParsedExamWork],
    images_dir: Path,
    force: bool,
) -> tuple[dict[tuple[str, str, str], str], dict[str, list[str]], dict[str, int]]:
    if force and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    key_to_filename: dict[tuple[str, str, str], str] = {}
    id_to_filenames: dict[str, list[str]] = {}
    id_to_hashes: dict[str, dict[str, str]] = {}
    copied_files = 0
    reused_files = 0
    missing_sources = 0

    for work in sorted(parsed_works, key=lambda item: item.item.exam_stem):
        source_dir = work.item.source_images_dir
        if source_dir is None:
            continue
        for illustration_id, image_names in work.illustration_mapping.items():
            id_upper = illustration_id.upper()
            hash_to_name = id_to_hashes.setdefault(id_upper, {})
            filenames = id_to_filenames.setdefault(id_upper, [])
            for image_name in image_names:
                source_name = Path(image_name).name
                source_path = source_dir / source_name
                if not source_path.exists() or not source_path.is_file():
                    missing_sources += 1
                    continue

                digest = _sha256_file(source_path)
                existing_name = hash_to_name.get(digest)
                if existing_name is None:
                    proposed = _build_id_based_filename(
                        id_upper, len(filenames) + 1, source_path.suffix
                    )
                    target_path = _find_unique_path(images_dir / proposed)
                    shutil.copy2(source_path, target_path)
                    existing_name = target_path.name
                    hash_to_name[digest] = existing_name
                    filenames.append(existing_name)
                    copied_files += 1
                else:
                    reused_files += 1

                key_to_filename[(work.item.exam_stem, id_upper, source_name)] = (
                    existing_name
                )

    stats = {
        "images_copied": copied_files,
        "images_reused": reused_files,
        "image_sources_missing": missing_sources,
        "illustration_ids_with_images": sum(
            1 for names in id_to_filenames.values() if names
        ),
        "illustration_files_total": sum(
            len(names) for names in id_to_filenames.values()
        ),
    }
    return key_to_filename, id_to_filenames, stats


def remap_question_images_to_deduped_filenames(
    work: ParsedExamWork,
    key_to_filename: dict[tuple[str, str, str], str],
    id_to_filenames: dict[str, list[str]],
) -> int:
    remapped = 0
    for question in work.questions:
        if not question.images:
            continue
        new_images: list[QuestionImage] = []
        seen: set[str] = set()
        for image in question.images:
            image_id = (image.id or "").upper()
            source_name = Path(image.uri).name
            filename = key_to_filename.get((work.item.exam_stem, image_id, source_name))
            if (
                filename is None
                and image_id in id_to_filenames
                and id_to_filenames[image_id]
            ):
                filename = id_to_filenames[image_id][0]
            if filename is None:
                continue
            if filename in seen:
                continue
            seen.add(filename)
            new_images.append(
                QuestionImage(
                    id=image_id or image.id,
                    uri=filename,
                    caption=image.caption,
                )
            )
            remapped += 1
        question.images = new_images or None
    return remapped


def write_exam_output(intermediate_dir: Path, work: ParsedExamWork) -> None:
    exam_stem = work.item.exam_stem
    payload = serialize_exam_group(
        exam_stem=exam_stem,
        exam_code=work.exam_meta["exam_code"],
        exam_title=work.exam_meta["exam_title"],
        exam_file=work.exam_meta["exam_file"],
        questions=work.questions,
    )
    write_json(intermediate_dir / f"{exam_stem}.json", payload)


def dedupe_questions(
    questions: list[EvalQuestionModel],
    threshold: int,
) -> tuple[list[EvalQuestionModel], dict[str, int]]:
    clusters: list[DedupCluster] = []
    exact_index: dict[tuple[str, str], int] = {}
    exact_merges = 0
    fuzzy_merges = 0

    for question in questions:
        q_blob = _question_blob(question)
        o_blob = _options_blob(question.options)
        answer_key = _answer_key(question)
        exact_key = (q_blob, answer_key)
        source_id = question.id
        source_exam = ""
        if isinstance(question.source, dict):
            source_exam = str(question.source.get("examStem") or "")

        if exact_key in exact_index:
            cluster = clusters[exact_index[exact_key]]
            cluster.source_ids.add(source_id)
            if source_exam:
                cluster.source_exam_stems.add(source_exam)
            cluster.methods.add("exact")
            if _question_score(question) > _question_score(cluster.canonical):
                cluster.canonical = question
                cluster.question_blob = q_blob
                cluster.options_blob = o_blob
                cluster.illustration_ids = _extract_illustration_ids(question)
            exact_merges += 1
            continue

        matched_cluster: DedupCluster | None = None
        for cluster in clusters:
            if cluster.answer_key != answer_key:
                continue
            question_sim = fuzz.ratio(q_blob, cluster.question_blob)
            if question_sim < threshold:
                continue
            options_sim = fuzz.ratio(o_blob, cluster.options_blob)
            shared_illustration = bool(
                _extract_illustration_ids(question) & cluster.illustration_ids
            )
            if options_sim < threshold and not shared_illustration:
                continue
            matched_cluster = cluster
            break

        if matched_cluster is None:
            new_cluster = DedupCluster(
                canonical=question,
                question_blob=q_blob,
                options_blob=o_blob,
                answer_key=answer_key,
                illustration_ids=_extract_illustration_ids(question),
                source_ids={source_id},
                source_exam_stems={source_exam} if source_exam else set(),
                methods={"exact"},
            )
            exact_index[exact_key] = len(clusters)
            clusters.append(new_cluster)
            continue

        matched_cluster.source_ids.add(source_id)
        if source_exam:
            matched_cluster.source_exam_stems.add(source_exam)
        matched_cluster.methods.add("fuzzy")
        if _question_score(question) > _question_score(matched_cluster.canonical):
            matched_cluster.canonical = question
            matched_cluster.question_blob = q_blob
            matched_cluster.options_blob = o_blob
            matched_cluster.illustration_ids = _extract_illustration_ids(question)
        fuzzy_merges += 1

    deduped: list[EvalQuestionModel] = []
    for cluster in clusters:
        question = cluster.canonical.model_copy(deep=True)
        metadata_obj = question.metadata if isinstance(question.metadata, dict) else {}
        if len(cluster.source_ids) > 1:
            metadata_obj["dedupSources"] = sorted(cluster.source_ids)
            metadata_obj["dedupExamStems"] = sorted(cluster.source_exam_stems)
            if cluster.methods == {"exact"}:
                metadata_obj["dedupMethod"] = "exact"
            elif cluster.methods == {"fuzzy"}:
                metadata_obj["dedupMethod"] = "fuzzy"
            else:
                metadata_obj["dedupMethod"] = "mixed"
            metadata_obj["dedupClusterSize"] = len(cluster.source_ids)
        question.metadata = metadata_obj
        deduped.append(question)

    deduped.sort(key=lambda q: q.id)
    stats = {
        "input_questions": len(questions),
        "deduped_questions": len(deduped),
        "removed_duplicates": len(questions) - len(deduped),
        "exact_merges": exact_merges,
        "fuzzy_merges": fuzzy_merges,
    }
    return deduped, stats


def write_merged_output(
    final_dir: Path, deduped_questions: list[EvalQuestionModel]
) -> dict[str, int]:
    def _question_is_multimodal(question: EvalQuestionModel) -> bool:
        if question.images:
            return True
        for option in question.options:
            if option.images:
                return True
        return False

    def _write_group(
        path: Path, group_id: str, questions: list[EvalQuestionModel]
    ) -> None:
        group = EvalQuestionGroupModel(
            id=group_id,
            metadata={
                "provider": "uscg",
                "questionCount": len(questions),
                "deduplicated": True,
            },
            source={"provider": "uscg", "examFile": "multiple", "examStem": "all"},
            questions=questions,
        )
        payload = EvalQuestionGroupsModel(root=[group]).model_dump(
            mode="json",
            exclude_none=True,
        )
        write_json(path, payload)

    merged_group = EvalQuestionGroupModel(
        id="uscg-merged-deduped",
        metadata={
            "provider": "uscg",
            "questionCount": len(deduped_questions),
            "deduplicated": True,
        },
        source={"provider": "uscg", "examFile": "multiple", "examStem": "all"},
        questions=deduped_questions,
    )
    payload = EvalQuestionGroupsModel(root=[merged_group]).model_dump(
        mode="json",
        exclude_none=True,
    )
    write_json(final_dir / "all_questions.json", payload)
    multimodal_questions = [q for q in deduped_questions if _question_is_multimodal(q)]
    text_only_questions = [
        q for q in deduped_questions if not _question_is_multimodal(q)
    ]

    _write_group(
        final_dir / "all_questions_text_only.json",
        "uscg-merged-deduped-text-only",
        text_only_questions,
    )
    _write_group(
        final_dir / "all_questions_multimodal.json",
        "uscg-merged-deduped-multimodal",
        multimodal_questions,
    )

    return {
        "all_questions_total": len(deduped_questions),
        "all_questions_text_only": len(text_only_questions),
        "all_questions_multimodal": len(multimodal_questions),
    }


def cleanup_legacy_per_exam_outputs(final_dir: Path) -> int:
    removed = 0
    for path in final_dir.glob("q*.json"):
        if not path.is_file():
            continue
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.max_parallel < 1:
        raise RuntimeError("--max-parallel must be >= 1")

    markdown_exams_dir = args.markdown_exams_dir.resolve()
    intermediate_dir = args.intermediate_dir.resolve()
    final_dir = args.final_dir.resolve()
    images_dir = args.images_dir.resolve()
    report_path = args.report_path.resolve()
    progress_path = args.progress_path.resolve()

    if not markdown_exams_dir.exists():
        raise RuntimeError(
            f"Markdown exams directory does not exist: {markdown_exams_dir}"
        )

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    items = iter_markdown_inputs(markdown_exams_dir)
    if args.exam_stem:
        requested = set(args.exam_stem)
        items = [item for item in items if item.exam_stem in requested]
    if args.skip_existing and not args.force:
        items = [
            item
            for item in items
            if not (intermediate_dir / f"{item.exam_stem}.json").exists()
        ]
    if args.limit_files is not None:
        items = items[: args.limit_files]

    progress_state: dict[str, Any] = (
        {"version": 1, "entries": {}}
        if args.reset_progress
        else load_progress(progress_path)
    )
    progress_state["version"] = 1
    progress_state.setdefault("started_at", now_utc_iso())
    progress_state["mode"] = "dry_run" if args.dry_run else "write"
    progress_state["markdown_exams_dir"] = markdown_exams_dir.as_posix()
    progress_state["intermediate_dir"] = intermediate_dir.as_posix()
    progress_state["final_dir"] = final_dir.as_posix()
    progress_state["images_dir"] = images_dir.as_posix()

    if args.resume and not args.force:
        entries = _ensure_entries(progress_state)
        done_ok = {
            stem for stem, entry in entries.items() if entry.get("status") == "ok"
        }
        before = len(items)
        items = [item for item in items if item.exam_stem not in done_ok]
        logger.info(
            "Resume enabled: skipping %s completed exam(s).", before - len(items)
        )

    if not items:
        logger.info("No markdown files selected.")
        return

    logger.info(
        "Selected %s exam markdown file(s). dry_run=%s force=%s max_parallel=%s",
        len(items),
        args.dry_run,
        args.force,
        args.max_parallel,
    )
    logger.info("Progress checkpoint: %s", progress_path)

    report_entries: list[dict[str, Any]] = []
    report_index: dict[str, dict[str, Any]] = {}
    parsed_works: dict[str, ParsedExamWork] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building eval JSON", total=len(items))

        def _record_entry(entry: dict[str, Any]) -> None:
            exam_stem = str(entry.get("exam_stem", "unknown"))
            report_entries.append(entry)
            report_index[exam_stem] = entry
            entries = _ensure_entries(progress_state)
            prev = entries.get(exam_stem, {})
            attempts = int(prev.get("attempts", 0)) + 1
            entries[exam_stem] = {
                **entry,
                "attempts": attempts,
                "updated_at": now_utc_iso(),
            }
            save_progress(progress_path, progress_state)

            if entry.get("status") == "ok":
                logger.info(
                    "Completed %s: emitted=%s images=%s",
                    exam_stem,
                    entry.get("questions_emitted", 0),
                    entry.get("questions_with_images", 0),
                )
            elif entry.get("status") == "error":
                logger.error("Failed %s: %s", exam_stem, entry.get("error"))

        def _run_one(
            item: MarkdownInput,
        ) -> tuple[dict[str, Any], ParsedExamWork | None]:
            return process_markdown_file(
                item=item,
                intermediate_dir=intermediate_dir,
                dry_run=args.dry_run,
            )

        if args.max_parallel <= 1:
            for item in items:
                progress.update(task, description=f"Processing {item.exam_stem}")
                try:
                    entry, work = _run_one(item)
                    if work is not None:
                        parsed_works[item.exam_stem] = work
                except Exception as error:  # noqa: BLE001
                    logger.exception("Failed processing %s", item.exam_stem)
                    entry = {
                        "exam_stem": item.exam_stem,
                        "status": "error",
                        "error": str(error),
                        "markdown_file": item.markdown_path.as_posix(),
                        "final_file": (
                            intermediate_dir / f"{item.exam_stem}.json"
                        ).as_posix(),
                    }
                _record_entry(entry)
                progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
                future_to_item = {pool.submit(_run_one, item): item for item in items}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    progress.update(task, description=f"Completed {item.exam_stem}")
                    try:
                        entry, work = future.result()
                        if work is not None:
                            parsed_works[item.exam_stem] = work
                    except Exception as error:  # noqa: BLE001
                        logger.exception("Failed processing %s", item.exam_stem)
                        entry = {
                            "exam_stem": item.exam_stem,
                            "status": "error",
                            "error": str(error),
                            "markdown_file": item.markdown_path.as_posix(),
                            "final_file": (
                                intermediate_dir / f"{item.exam_stem}.json"
                            ).as_posix(),
                        }
                    _record_entry(entry)
                    progress.advance(task)

    dedupe_stats = {
        "input_questions": 0,
        "deduped_questions": 0,
        "removed_duplicates": 0,
        "exact_merges": 0,
        "fuzzy_merges": 0,
    }
    image_stats = {
        "images_copied": 0,
        "images_reused": 0,
        "image_sources_missing": 0,
        "illustration_ids_with_images": 0,
        "illustration_files_total": 0,
        "question_images_remapped": 0,
    }
    modality_stats = {
        "all_questions_total": 0,
        "all_questions_text_only": 0,
        "all_questions_multimodal": 0,
    }
    if not args.dry_run:
        parsed_work_list = sorted(
            parsed_works.values(), key=lambda work: work.item.exam_stem
        )
        key_to_filename, id_to_filenames, image_stats_raw = (
            dedupe_and_copy_illustration_images(
                parsed_works=parsed_work_list,
                images_dir=images_dir,
                force=args.force,
            )
        )
        image_stats.update(image_stats_raw)

        all_questions: list[EvalQuestionModel] = []
        for work in parsed_work_list:
            remapped_count = remap_question_images_to_deduped_filenames(
                work=work,
                key_to_filename=key_to_filename,
                id_to_filenames=id_to_filenames,
            )
            image_stats["question_images_remapped"] += remapped_count
            entry = report_index.get(work.item.exam_stem)
            if entry is not None:
                entry["question_images_remapped"] = remapped_count
                entry["question_images_after_remap"] = sum(
                    len(question.images or []) for question in work.questions
                )
                entry["final_file"] = (
                    intermediate_dir / f"{work.item.exam_stem}.json"
                ).as_posix()
            write_exam_output(intermediate_dir=intermediate_dir, work=work)
            all_questions.extend(work.questions)

        removed_legacy_outputs = 0
        if args.force:
            removed_legacy_outputs = cleanup_legacy_per_exam_outputs(final_dir)
        image_stats["legacy_final_qjson_removed"] = removed_legacy_outputs

        deduped_questions, dedupe_stats = dedupe_questions(
            all_questions, args.dedupe_threshold
        )
        modality_stats = write_merged_output(final_dir, deduped_questions)

    summary = {
        "mode": "dry_run" if args.dry_run else "write",
        "files_selected": len(items),
        "ok": sum(1 for entry in report_entries if entry.get("status") == "ok"),
        "errors": sum(1 for entry in report_entries if entry.get("status") == "error"),
        "dry_run": sum(
            1 for entry in report_entries if entry.get("status") == "dry_run"
        ),
        "images": image_stats,
        "dedupe": dedupe_stats,
        "modalities": modality_stats,
        "entries": report_entries,
    }

    def _sum_entry_metric(key: str) -> int:
        total = 0
        for entry in report_entries:
            value = entry.get(key, 0)
            if isinstance(value, bool):
                total += int(value)
            elif isinstance(value, int):
                total += value
            elif isinstance(value, str) and value.isdigit():
                total += int(value)
        return total

    final_metrics = {
        "questions_detected": _sum_entry_metric("questions_detected"),
        "questions_emitted": _sum_entry_metric("questions_emitted"),
        "questions_repaired": _sum_entry_metric("questions_repaired"),
        "questions_skipped_missing_answer": _sum_entry_metric(
            "questions_skipped_missing_answer"
        ),
        "questions_skipped_incomplete_options": _sum_entry_metric(
            "questions_skipped_incomplete_options"
        ),
        "questions_with_images": _sum_entry_metric("questions_with_images"),
        "questions_with_illustration_keyword_missing_images": _sum_entry_metric(
            "questions_with_illustration_keyword_missing_images"
        ),
        "questions_with_detected_illustration_ids_missing_images": _sum_entry_metric(
            "questions_with_detected_illustration_ids_missing_images"
        ),
        "questions_invalid_missing_visual_reference": _sum_entry_metric(
            "questions_invalid_missing_visual_reference"
        ),
        "illustration_ids_detected": _sum_entry_metric("illustration_ids_detected"),
        "illustration_ids_resolved": _sum_entry_metric("illustration_ids_resolved"),
        "illustration_images_detected": _sum_entry_metric(
            "illustration_images_detected"
        ),
    }
    summary["totals"] = final_metrics
    write_json(report_path, summary)

    logger.info(
        "Done. files=%s ok=%s errors=%s dry_run=%s deduped=%s",
        summary["files_selected"],
        summary["ok"],
        summary["errors"],
        summary["dry_run"],
        dedupe_stats["deduped_questions"],
    )
    logger.info(
        "Question totals: detected=%s emitted=%s repaired=%s skipped_missing_answer=%s "
        "skipped_incomplete_options=%s",
        final_metrics["questions_detected"],
        final_metrics["questions_emitted"],
        final_metrics["questions_repaired"],
        final_metrics["questions_skipped_missing_answer"],
        final_metrics["questions_skipped_incomplete_options"],
    )
    logger.info(
        "Visual-reference totals: questions_with_images=%s keyword_gaps=%s "
        "id_gaps=%s invalid_missing_visual_reference=%s ids_detected=%s ids_resolved=%s "
        "illustration_images_detected=%s",
        final_metrics["questions_with_images"],
        final_metrics["questions_with_illustration_keyword_missing_images"],
        final_metrics["questions_with_detected_illustration_ids_missing_images"],
        final_metrics["questions_invalid_missing_visual_reference"],
        final_metrics["illustration_ids_detected"],
        final_metrics["illustration_ids_resolved"],
        final_metrics["illustration_images_detected"],
    )
    logger.info(
        "Merged modality totals: all=%s text_only=%s multimodal=%s",
        modality_stats.get("all_questions_total", 0),
        modality_stats.get("all_questions_text_only", 0),
        modality_stats.get("all_questions_multimodal", 0),
    )
    logger.info(
        "Dedupe/Image totals: input_questions=%s deduped_questions=%s removed_duplicates=%s "
        "exact_merges=%s fuzzy_merges=%s images_copied=%s images_reused=%s "
        "image_sources_missing=%s illustration_ids_with_images=%s illustration_files_total=%s "
        "question_images_remapped=%s",
        dedupe_stats.get("input_questions", 0),
        dedupe_stats.get("deduped_questions", 0),
        dedupe_stats.get("removed_duplicates", 0),
        dedupe_stats.get("exact_merges", 0),
        dedupe_stats.get("fuzzy_merges", 0),
        image_stats.get("images_copied", 0),
        image_stats.get("images_reused", 0),
        image_stats.get("image_sources_missing", 0),
        image_stats.get("illustration_ids_with_images", 0),
        image_stats.get("illustration_files_total", 0),
        image_stats.get("question_images_remapped", 0),
    )
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
