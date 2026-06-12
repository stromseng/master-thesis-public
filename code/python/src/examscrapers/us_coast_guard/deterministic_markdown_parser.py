from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from generated.eval_question_models import (
    NonEmptyOptionIds,
    Option as EvalOption,
    Question as EvalQuestionModel,
    QuestionImage,
)

CHOICE_LETTERS = ("A", "B", "C", "D")
PAGE_BOUNDARY_RE = re.compile(r"\n\{(\d+)\}-+\n")
QUESTION_START_BOLD_RE = re.compile(r"^\s*-?\s*\*\*(\d{1,3})\.\*\*\s*(.*)$")
QUESTION_START_PLAIN_RE = re.compile(r"^\s*(\d{1,3})\.(?:\s+|(?=[A-Za-z])|$)(.*)$")
INLINE_QUESTION_MARKER_RE = re.compile(r"(?:-\s*)?\*\*\d{1,3}\.\*\*")
ANSWER_RE = re.compile(r"Correct\s+answer\s*:\s*([A-D])\b", re.IGNORECASE)
OPTION_MARKER_RE = re.compile(r"(?<![A-Z0-9])([A-D])\.\s*")
IMAGE_MARKDOWN_RE = re.compile(r"!\[\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_TABLE_TAG_RE = re.compile(r"<(?:table|tr|td|th)\b", re.IGNORECASE)
HEADING_WITH_BOLD_RE = re.compile(r"^#{1,6}\s*\*\*(.+?)\*\*\s*$")
BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")
ILLUSTRATION_ID_RE = re.compile(
    r"\b(?:D\d{3}[A-Z]{2}|[A-Z][A-Z0-9]{1,2}-\d{4}|[A-Z]{2,4}\d{3,4}[A-Z]{0,2})\b",
    re.IGNORECASE,
)


@dataclass
class QuestionDraft:
    question_number: int
    question_text: str
    options: dict[str, str]
    correct_answer: str
    illustration_ids: list[str]
    repairs: int = 0


@dataclass
class ParseIssue:
    reason: str
    question_number: int | None
    snippet: str


@dataclass
class ParsedExamResult:
    questions: list[EvalQuestionModel]
    illustration_mapping: dict[str, list[str]]
    report: dict[str, Any]


@dataclass
class _LineRecord:
    raw: str
    canonical: str
    page_number: int
    line_number: int


@dataclass
class _QuestionParseResult:
    draft: QuestionDraft | None
    reason: str | None
    repairs: int


@dataclass
class _QuestionExtractionState:
    issues: list[ParseIssue] = field(default_factory=list)
    questions_detected: int = 0
    questions_repaired: int = 0
    questions_skipped_missing_answer: int = 0
    questions_skipped_incomplete_options: int = 0
    duplicate_candidates_discarded: int = 0


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _has_embedded_table_markup(text: str) -> bool:
    if not text:
        return False
    if HTML_TABLE_TAG_RE.search(text):
        return True

    # Markdown table fallback when pipe-table syntax survives extraction.
    has_pipe_row = bool(re.search(r"\|[^|]+\|[^|]+\|", text))
    has_separator = bool(re.search(r"\|\s*:?-{3,}", text))
    return has_pipe_row and has_separator


def extract_illustration_ids(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in ILLUSTRATION_ID_RE.findall(text.upper()):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def _detect_question_category(question_text: str) -> str:
    upper = question_text.upper()
    if "INLAND ONLY" in upper:
        return "inland"
    if "INTERNATIONAL ONLY" in upper:
        return "international"
    if "BOTH" in upper and "INLAND" in upper and "INTERNATIONAL" in upper:
        return "both"
    return "unknown"


def _clarify_inland_references(question_text: str) -> str:
    # Prevent duplicating an existing "US INLAND" phrase while clarifying
    # standalone "INLAND" references.
    return re.sub(r"(?<!US )\bINLAND\b", "US INLAND", question_text)


def _is_table_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    non_empty = [cell for cell in cells if cell.strip()]
    if not non_empty:
        return False
    return all(re.fullmatch(r"[:\- ]+", cell) for cell in non_empty)


def _split_cell(cell: str) -> list[str]:
    text = cell.replace("<br>", "\n")
    return [segment.strip() for segment in text.splitlines() if segment.strip()]


def _pair_table_option_lines(tokens: list[str]) -> list[str]:
    if len(tokens) < 6:
        return tokens

    def _is_label(value: str) -> bool:
        return value in {"A.", "B.", "C.", "D."}

    # Pattern: A. B. C. D. <textA> <textB> <textC> <textD>
    prefix_labels = [token for token in tokens[:4] if _is_label(token)]
    if len(prefix_labels) >= 2 and tokens[: len(prefix_labels)] == prefix_labels:
        suffix = tokens[len(prefix_labels) :]
        if len(suffix) >= len(prefix_labels):
            paired = [
                f"{label[0]}. {text}"
                for label, text in zip(prefix_labels, suffix, strict=False)
            ]
            remainder = suffix[len(prefix_labels) :]
            return paired + remainder

    # Pattern: A. <textA> B. <textB> ...
    if len(tokens) % 2 == 0 and all(
        _is_label(tokens[i]) for i in range(0, len(tokens), 2)
    ):
        return [f"{tokens[i][0]}. {tokens[i + 1]}" for i in range(0, len(tokens), 2)]

    return tokens


def _normalize_markdown_lines(markdown_text: str) -> list[_LineRecord]:
    segments = PAGE_BOUNDARY_RE.split(markdown_text)
    pages: list[tuple[int, str]] = []
    if len(segments) == 1:
        pages.append((0, segments[0]))
    else:
        prefix = segments[0]
        if prefix.strip():
            pages.append((0, prefix))
        for index in range(1, len(segments), 2):
            page_no = int(segments[index])
            page_text = segments[index + 1] if index + 1 < len(segments) else ""
            pages.append((page_no, page_text))

    records: list[_LineRecord] = []
    for page_number, page_text in pages:
        for line_number, raw_line in enumerate(page_text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            canonical_lines: list[str] = []
            if stripped.startswith("|") and "|" in stripped:
                raw_cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                if _is_table_separator_row(raw_cells):
                    continue

                token_buffer: list[str] = []
                for cell in raw_cells:
                    token_buffer.extend(_split_cell(cell))
                token_buffer = _pair_table_option_lines(token_buffer)
                canonical_lines.extend(token_buffer)
            else:
                canonical_lines.extend(_split_cell(stripped))

            for canonical in canonical_lines:
                normalized = _clean_whitespace(canonical)
                if not normalized:
                    continue
                records.append(
                    _LineRecord(
                        raw=raw_line,
                        canonical=normalized,
                        page_number=page_number,
                        line_number=line_number,
                    )
                )

    return records


def _split_inline_question_segments(line: str) -> list[str]:
    matches = list(INLINE_QUESTION_MARKER_RE.finditer(line))
    if not matches:
        return [line]

    if len(matches) == 1 and matches[0].start() == 0:
        return [line]

    segments: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        chunk = line[match.start() : end].strip()
        if chunk:
            segments.append(chunk)

    if segments:
        return segments
    return [line]


def _parse_question_start(line: str) -> tuple[int, str] | None:
    match_bold = QUESTION_START_BOLD_RE.match(line)
    if match_bold:
        return int(match_bold.group(1)), _clean_whitespace(match_bold.group(2))

    match_plain = QUESTION_START_PLAIN_RE.match(line)
    if match_plain:
        question_number = int(match_plain.group(1))
        remainder = _clean_whitespace(match_plain.group(2))
        if question_number <= 0:
            return None
        if not remainder:
            if re.fullmatch(rf"\s*{question_number}\.\s*", line):
                return question_number, ""
            return None
        # Reject numeric/coordinate/table rows that are not natural-language stems.
        if not re.search(r"[A-Za-z]", remainder):
            return None
        return question_number, remainder

    return None


def _strip_question_prefix(text: str, question_number: int) -> str:
    patterns = (
        rf"^\s*-?\s*\*\*{question_number}\.\*\*\s*",
        rf"^\s*{question_number}\.\s*",
    )
    value = text
    for pattern in patterns:
        value = re.sub(pattern, "", value, count=1)
    return value


def _parse_question_block(
    question_number: int, lines: list[str]
) -> _QuestionParseResult:
    merged_text = _clean_whitespace(" ".join(lines))
    merged_text = _strip_question_prefix(merged_text, question_number)
    if not merged_text:
        return _QuestionParseResult(
            draft=None,
            reason="incomplete_options",
            repairs=0,
        )

    answer_match = ANSWER_RE.search(merged_text)
    if answer_match is None:
        return _QuestionParseResult(
            draft=None,
            reason="missing_answer",
            repairs=0,
        )

    correct_answer = answer_match.group(1).upper()
    option_matches = list(OPTION_MARKER_RE.finditer(merged_text))
    if not option_matches:
        return _QuestionParseResult(
            draft=None,
            reason="incomplete_options",
            repairs=0,
        )

    first_option_start = option_matches[0].start()
    text_end = min(first_option_start, answer_match.start())
    question_text = _clean_whitespace(merged_text[:text_end])

    repairs = 0
    if answer_match.start() < first_option_start:
        repairs += 1

    parsed_order: list[str] = []
    options: dict[str, str] = {}
    for index, match in enumerate(option_matches):
        option_letter = match.group(1)
        parsed_order.append(option_letter)
        start = match.end()
        end = (
            option_matches[index + 1].start()
            if index + 1 < len(option_matches)
            else len(merged_text)
        )
        option_text = merged_text[start:end]
        option_text = re.sub(
            r"Correct\s+answer\s*:\s*[A-D]\b.*$", "", option_text, flags=re.IGNORECASE
        )
        option_text = _clean_whitespace(option_text)
        option_text = re.sub(r"\s*-\s*$", "", option_text).strip()
        if option_text and option_letter not in options:
            options[option_letter] = option_text

    if parsed_order != ["A", "B", "C", "D"]:
        repairs += 1

    if not question_text:
        question_text = _clean_whitespace(
            _strip_question_prefix(merged_text[: answer_match.start()], question_number)
        )

    if not question_text:
        return _QuestionParseResult(
            draft=None,
            reason="incomplete_options",
            repairs=repairs,
        )

    if any(letter not in options or not options[letter] for letter in CHOICE_LETTERS):
        return _QuestionParseResult(
            draft=None,
            reason="incomplete_options",
            repairs=repairs,
        )

    illustration_ids = extract_illustration_ids(
        " ".join([question_text, *(options[letter] for letter in CHOICE_LETTERS)])
    )

    return _QuestionParseResult(
        draft=QuestionDraft(
            question_number=question_number,
            question_text=question_text,
            options={letter: options[letter] for letter in CHOICE_LETTERS},
            correct_answer=correct_answer,
            illustration_ids=illustration_ids,
            repairs=repairs,
        ),
        reason=None,
        repairs=repairs,
    )


def _score_draft(draft: QuestionDraft) -> tuple[int, int, int, int]:
    complete = int(all(draft.options.get(letter, "") for letter in CHOICE_LETTERS))
    valid_answer = int(draft.correct_answer in CHOICE_LETTERS)
    return (
        complete,
        valid_answer,
        len(draft.question_text),
        -draft.repairs,
    )


def _extract_questions_with_report(
    markdown_text: str,
) -> tuple[list[QuestionDraft], _QuestionExtractionState]:
    state = _QuestionExtractionState()
    records = _normalize_markdown_lines(markdown_text)
    canonical_lines: list[str] = []
    for record in records:
        canonical_lines.extend(_split_inline_question_segments(record.canonical))

    raw_starts: list[tuple[int, int]] = []
    for index, line in enumerate(canonical_lines):
        parsed = _parse_question_start(line)
        if parsed is None:
            continue
        raw_starts.append((index, parsed[0]))

    starts: list[tuple[int, int]] = []
    for start_index, (line_index, question_number) in enumerate(raw_starts):
        next_line_index = (
            raw_starts[start_index + 1][0]
            if start_index + 1 < len(raw_starts)
            else len(canonical_lines)
        )
        span = canonical_lines[line_index:next_line_index]
        span_text = _clean_whitespace(" ".join(span))
        if ANSWER_RE.search(span_text) is None:
            continue
        starts.append((line_index, question_number))

    state.questions_detected = len(starts)

    drafts_by_number: dict[int, list[QuestionDraft]] = {}
    for start_index, (line_index, question_number) in enumerate(starts):
        next_line_index = (
            starts[start_index + 1][0]
            if start_index + 1 < len(starts)
            else len(canonical_lines)
        )
        span = canonical_lines[line_index:next_line_index]
        parsed = _parse_question_block(question_number, span)
        if parsed.draft is None:
            snippet = _clean_whitespace(" ".join(span))[:260]
            state.issues.append(
                ParseIssue(
                    reason=parsed.reason or "unknown",
                    question_number=question_number,
                    snippet=snippet,
                )
            )
            if parsed.reason == "missing_answer":
                state.questions_skipped_missing_answer += 1
            else:
                state.questions_skipped_incomplete_options += 1
            continue

        if parsed.repairs > 0:
            state.questions_repaired += 1

        drafts_by_number.setdefault(question_number, []).append(parsed.draft)

    final_drafts: list[QuestionDraft] = []
    for question_number in sorted(drafts_by_number):
        candidates = drafts_by_number[question_number]
        if len(candidates) > 1:
            state.duplicate_candidates_discarded += len(candidates) - 1
        best = max(candidates, key=_score_draft)
        final_drafts.append(best)

    return final_drafts, state


def extract_questions(markdown_text: str) -> list[QuestionDraft]:
    drafts, _ = _extract_questions_with_report(markdown_text)
    return drafts


def _extract_ids_from_appendix_header(raw_line: str) -> list[str]:
    stripped = raw_line.strip()
    if not stripped:
        return []

    canonical = _clean_whitespace(stripped)
    text_only = _clean_whitespace(HTML_TAG_RE.sub(" ", stripped))
    if _parse_question_start(canonical) is not None:
        return []
    if "CORRECT ANSWER:" in canonical.upper():
        return []

    if re.search(r"\*\*\d{1,3}\.\*\*", stripped):
        return []

    heading_match = HEADING_WITH_BOLD_RE.match(stripped)
    if heading_match:
        return extract_illustration_ids(heading_match.group(1))

    bold_match = BOLD_LINE_RE.match(stripped)
    if bold_match:
        return extract_illustration_ids(bold_match.group(1))

    headingless = re.sub(r"^#{1,6}\s*", "", stripped).strip()
    candidate_ids = extract_illustration_ids(headingless)
    if candidate_ids:
        remainder = headingless.upper()
        for illustration_id in candidate_ids:
            remainder = remainder.replace(illustration_id, " ")
        remainder = re.sub(r"[\s:;,\-\[\]()/\\]+", "", remainder)
        if not remainder:
            return candidate_ids

    # PaddleOCR markdown often wraps appendix IDs in centered HTML divs:
    # <div style="...">RA-0043</div>
    if text_only and text_only != canonical:
        candidate_ids = extract_illustration_ids(text_only)
        if candidate_ids:
            remainder = text_only.upper()
            for illustration_id in candidate_ids:
                remainder = remainder.replace(illustration_id, " ")
            remainder = re.sub(r"[\s:;,\-\[\]()/\\]+", "", remainder)
            if not remainder:
                return candidate_ids

    return []


def _extract_image_name(raw_line: str) -> str | None:
    markdown_match = IMAGE_MARKDOWN_RE.search(raw_line)
    if markdown_match:
        name = Path(markdown_match.group(1).strip()).name
        return name or None

    html_match = HTML_IMAGE_RE.search(raw_line)
    if html_match:
        name = Path(html_match.group(1).strip()).name
        return name or None

    return None


def extract_illustration_mapping(markdown_text: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    active_ids: list[str] = []

    for raw_line in markdown_text.splitlines():
        ids = _extract_ids_from_appendix_header(raw_line)
        if ids:
            active_ids = ids
            for illustration_id in ids:
                mapping.setdefault(illustration_id, [])
            continue

        image_name = _extract_image_name(raw_line)
        if image_name and active_ids:
            if image_name:
                for illustration_id in active_ids:
                    if image_name not in mapping.setdefault(illustration_id, []):
                        mapping[illustration_id].append(image_name)
            continue

        canonical = _clean_whitespace(raw_line)
        if not canonical:
            continue
        if (
            _parse_question_start(canonical) is not None
            or "Correct answer:" in canonical
        ):
            active_ids = []

    return mapping


def build_eval_questions(
    question_drafts: list[QuestionDraft],
    exam_stem: str,
    exam_meta: dict[str, str],
    illustration_mapping: dict[str, list[str]],
) -> list[EvalQuestionModel]:
    result: list[EvalQuestionModel] = []

    for draft in question_drafts:
        options = [
            EvalOption(id=letter, text=draft.options[letter])
            for letter in CHOICE_LETTERS
        ]
        clarified_question_text = _clarify_inland_references(draft.question_text)
        images: list[QuestionImage] = []
        seen_uris: set[str] = set()

        for illustration_id in draft.illustration_ids:
            for image_name in illustration_mapping.get(illustration_id, []):
                uri = Path(image_name).name
                if uri in seen_uris:
                    continue
                seen_uris.add(uri)
                images.append(QuestionImage(id=illustration_id, uri=uri))

        question = EvalQuestionModel(
            id=f"uscg-{exam_stem}-{draft.question_number:04d}",
            questionText=clarified_question_text,
            metadata={
                "category": _detect_question_category(clarified_question_text),
                "examCode": exam_meta.get("exam_code", ""),
                "examTitle": exam_meta.get("exam_title", ""),
                "originalQuestionNumber": draft.question_number,
            },
            source={
                "examFile": exam_meta.get("exam_file", ""),
                "examStem": exam_stem,
            },
            options=options,
            correctOptionIds=NonEmptyOptionIds(root=[draft.correct_answer]),
            images=images or None,
        )
        result.append(question)

    return result


def parse_exam_markdown(
    markdown_text: str,
    exam_stem: str,
    exam_meta: dict[str, str],
    images_dir: Path,
    available_images: set[str] | None = None,
) -> ParsedExamResult:
    drafts, question_state = _extract_questions_with_report(markdown_text)

    raw_mapping = extract_illustration_mapping(markdown_text)
    if available_images is None:
        exam_images_dir = images_dir / exam_stem
        if not exam_images_dir.exists() and images_dir.exists():
            exam_images_dir = images_dir
        available_images = (
            {path.name for path in exam_images_dir.glob("*") if path.is_file()}
            if exam_images_dir.exists()
            else set()
        )

    resolved_mapping: dict[str, list[str]] = {}
    for illustration_id, image_names in raw_mapping.items():
        resolved = [name for name in image_names if name in available_images]
        if resolved:
            resolved_mapping[illustration_id] = resolved

    built_questions = build_eval_questions(
        question_drafts=drafts,
        exam_stem=exam_stem,
        exam_meta=exam_meta,
        illustration_mapping=resolved_mapping,
    )

    illustration_keyword_missing_images: list[dict[str, Any]] = []
    detected_illustration_ids_missing_images: list[dict[str, Any]] = []
    invalid_missing_visual_reference_examples: list[dict[str, Any]] = []
    filtered_questions: list[EvalQuestionModel] = []
    invalid_missing_visual_reference_count = 0

    for draft, question in zip(drafts, built_questions, strict=False):
        has_table_markup = _has_embedded_table_markup(question.questionText)

        if not question.images:
            if "ILLUSTRATION" in question.questionText.upper():
                illustration_keyword_missing_images.append(
                    {
                        "id": question.id,
                        "question_number": draft.question_number,
                        "question_text": question.questionText,
                    }
                )
            if draft.illustration_ids:
                example = {
                    "id": question.id,
                    "question_number": draft.question_number,
                    "illustration_ids": draft.illustration_ids,
                    "question_text": question.questionText,
                    "has_embedded_table_markup": has_table_markup,
                    "invalid_missing_visual_reference": not has_table_markup,
                }
                detected_illustration_ids_missing_images.append(example)

                # If an ID-like reference is present but neither an image nor an
                # embedded table is available, drop the question as invalid.
                if not has_table_markup:
                    invalid_missing_visual_reference_count += 1
                    invalid_missing_visual_reference_examples.append(example)
                    continue

        filtered_questions.append(question)

    questions = filtered_questions

    report = {
        "questions_detected": question_state.questions_detected,
        "questions_emitted": len(questions),
        "questions_repaired": question_state.questions_repaired,
        "questions_skipped_missing_answer": question_state.questions_skipped_missing_answer,
        "questions_skipped_incomplete_options": question_state.questions_skipped_incomplete_options,
        "duplicate_candidates_discarded": question_state.duplicate_candidates_discarded,
        "illustration_ids_detected": len(raw_mapping),
        "illustration_ids_resolved": len(resolved_mapping),
        "questions_with_images": sum(1 for question in questions if question.images),
        "questions_with_illustration_keyword_missing_images": len(
            illustration_keyword_missing_images
        ),
        "questions_with_detected_illustration_ids_missing_images": len(
            detected_illustration_ids_missing_images
        ),
        "questions_invalid_missing_visual_reference": (
            invalid_missing_visual_reference_count
        ),
        "illustration_keyword_missing_images_examples": illustration_keyword_missing_images[
            :50
        ],
        "detected_illustration_ids_missing_images_examples": (
            detected_illustration_ids_missing_images[:50]
        ),
        "invalid_missing_visual_reference_examples": (
            invalid_missing_visual_reference_examples[:50]
        ),
        "issues": [
            {
                "reason": issue.reason,
                "question_number": issue.question_number,
                "snippet": issue.snippet,
            }
            for issue in question_state.issues[:200]
        ],
    }

    return ParsedExamResult(
        questions=questions,
        illustration_mapping=resolved_mapping,
        report=report,
    )
