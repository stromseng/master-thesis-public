"""
MCQ Deduplication Script

Filters duplicate/near-duplicate questions from MCQ datasets using:
0. LLM OCR Repair (Chinese only) - Fix garbled question text using LLM with options/hints as context
1. Exact fuzzy dedup (0.99) - Remove guaranteed near-exact duplicates cheaply
2. LLM Quality Filter - Remove garbage/OCR-error/context-dependent questions
3. Embedding similarity - Dense embeddings + FAISS similarity search (default 0.70 threshold, GPU if available)
4. Fuzzy matching (optional) - rapidfuzz token_set_ratio (default 0.70 threshold)
5. LLM Duplicate Verification - Review flagged duplicates, recover false positives

Usage (local):
    cd code/python && uv run python -m scripts.idun.dedup_mcq --limit 100 --dry-run
    cd code/python && uv run python -m scripts.idun.dedup_mcq --chinese --limit 20 --dry-run --stages llm_repair,llm_quality

Usage (IDUN):
    just idun submit --script dedup_mcq.py
    just idun submit --script dedup_mcq.py --args "--chinese"
"""

# Complete chinese dataset result
#                            Deduplication Summary
# ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
# ┃ Stage                 ┃ Before ┃                        Removed ┃  After ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
# │ filter (简答题)        │ 257396 │                             48 │ 257348 │
# │ filter (null answer)  │ 257348 │                           2475 │ 254873 │
# │ filter (null options) │ 254873 │                           1817 │ 253056 │
# │ llm_repair            │ 253056 │ 125649 repaired, 120417 failed │ 253056 │
# │ llm_quality           │ 253056 │                          92026 │ 161030 │
# │ embedding             │ 161030 │                         126107 │  34923 │
# │ llm_verify            │  34923 │                         +23732 │  58655 │
# │ Total                 │ 257396 │                         198741 │  58655 │
# └───────────────────────┴────────┴────────────────────────────────┴────────┘

# Complete english dataset result
#                Deduplication Summary
# ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
# ┃ Stage                 ┃ Before ┃ Removed ┃ After ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
# │ filter (null answer)  │  21144 │     624 │ 20520 │
# │ filter (null options) │  20520 │       1 │ 20519 │
# │ exact_fuzzy           │  20519 │    5771 │ 14748 │
# │ llm_quality           │  14748 │      65 │ 14683 │
# │ embedding             │  14683 │   11125 │  3558 │
# │ fuzzy                 │   3558 │      71 │  3487 │
# │ llm_verify            │   3487 │   +6153 │  9640 │
# │ Total                 │  21144 │   11504 │  9640 │
# └───────────────────────┴────────┴─────────┴───────┘

from __future__ import annotations

import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests
import typer
from dotenv import load_dotenv
from rapidfuzz import fuzz
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.table import Table

# Load environment variables
load_dotenv()

# IDUN Embedding API configuration
IDUN_EMBEDDING_URL = "https://llm.hpc.ntnu.no/v1/embeddings"
IDUN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# IDUN Chat API configuration (for LLM-as-judge)
IDUN_CHAT_URL = "https://llm.hpc.ntnu.no/v1/chat/completions"
IDUN_CHAT_MODEL = "openai/gpt-oss-120b"

# Type mapping: Chinese -> English
TYPE_MAP = {
    "单选题": "single_choice",
    "判断题": "true_false",
    "多选题": "multiple_choice",
    "填空题": "fill_in_the_blank",
}

app = typer.Typer(help="MCQ deduplication tool")
console = Console()


def is_tty() -> bool:
    """Check if stdout is a TTY (interactive terminal)."""
    return sys.stdout.isatty()


class SimpleProgress:
    """Simple progress tracker for non-TTY environments (e.g., log files).

    Adaptive update intervals:
    - First minute: every 5 seconds
    - 1-10 minutes: every 1 minute
    - After 10 minutes: every 10% progress
    """

    def __init__(self):
        self.tasks: dict[TaskID, dict[str, Any]] = {}
        self.next_id: TaskID = TaskID(0)
        self.start_time = datetime.now()

    def add_task(self, description: str, total: int) -> TaskID:
        task_id = self.next_id
        self.next_id = TaskID(int(self.next_id) + 1)
        now = datetime.now()
        self.tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
            "last_percent": -1,
            "start_time": now,
            "last_update_time": now,
            "first_update_done": False,
        }
        console.print(f"[cyan]{description}[/cyan] (total: {total})")
        return task_id

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _should_update(self, task: dict[str, Any]) -> bool:
        """Determine if we should print an update based on elapsed time."""
        # Always print on first update (immediate feedback that processing started)
        if not task["first_update_done"]:
            return True

        now = datetime.now()
        elapsed_total = (now - task["start_time"]).total_seconds()
        elapsed_since_update = (now - task["last_update_time"]).total_seconds()
        percent = (
            int(100 * task["completed"] / task["total"]) if task["total"] > 0 else 0
        )

        # Always update at 100%
        if percent == 100:
            return True

        # First minute: update every 10 seconds
        if elapsed_total < 60:
            return elapsed_since_update >= 10

        # 1-10 minutes: update every minute
        if elapsed_total < 600:
            return elapsed_since_update >= 60

        # After 10 minutes: update every 10%
        return percent // 10 > task["last_percent"] // 10

    def advance(self, task_id: TaskID, amount: int = 1) -> None:
        task = self.tasks[task_id]
        task["completed"] += amount

        if not self._should_update(task):
            return

        now = datetime.now()
        elapsed = (now - task["start_time"]).total_seconds()
        percent = (
            int(100 * task["completed"] / task["total"]) if task["total"] > 0 else 0
        )

        # Calculate ETA
        if task["completed"] > 0 and task["total"] > 0:
            rate = task["completed"] / elapsed  # items per second
            remaining = task["total"] - task["completed"]
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "??:??"

        elapsed_str = self._format_time(elapsed)

        console.print(
            f"  Progress: {task['completed']}/{task['total']} ({percent}%) "
            f"| Elapsed: {elapsed_str} | ETA: {eta_str}"
        )

        task["last_percent"] = percent
        task["last_update_time"] = now
        task["first_update_done"] = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@contextmanager
def make_progress() -> Iterator[Progress | SimpleProgress]:
    """Create a progress bar (fancy for TTY, simple for non-TTY)."""
    if is_tty():
        # Interactive terminal: use rich progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            console=console,
            speed_estimate_period=600,
        )
        with progress:
            yield progress
    else:
        # Non-TTY (log file): use simple text updates
        yield SimpleProgress()


# Paths - relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "evals" / "mcq"
INPUT_FILE = DATA_DIR / "SHITITONG_ENGLISH_QUESTIONS.json"
OUTPUT_FILE = DATA_DIR / "SHITITONG_ENGLISH_DEDUPED.json"
REPORT_FILE = DATA_DIR / "dedup_report.json"

CHINESE_INPUT_FILE = DATA_DIR / "SHITITONG_CHINESE_QUESTIONS.json"
CHINESE_OUTPUT_FILE = DATA_DIR / "SHITITONG_CHINESE_DEDUPED.json"
CHINESE_REPORT_FILE = DATA_DIR / "dedup_report_chinese.json"

DEFAULT_STAGES_ENGLISH = "llm_quality,embedding,llm_verify"
DEFAULT_STAGES_CHINESE = "llm_repair,llm_quality,embedding,llm_verify"


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip leading numbers, collapse whitespace."""
    # Remove leading question numbers like "1.", "2.", "123."
    text = re.sub(r"^\d+\.\s*", "", text.strip())
    # Lowercase
    text = text.lower()
    # Collapse multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation like ( )
    text = re.sub(r"\s*\(\s*\)\s*$", "", text)
    return text.strip()


def is_null_value(value: Any) -> bool:
    """Check if a value is null (string 'null', None, or empty)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "null" or value.strip() == ""
    return False


def has_valid_answer(q: dict[str, Any]) -> bool:
    """Check if question has a valid (non-null) answer."""
    return not is_null_value(q.get("answer"))


def has_valid_options(q: dict[str, Any]) -> bool:
    """Check if question has at least one valid (non-null) option."""
    options = q.get("options", {})
    if not options:
        return False
    return any(not is_null_value(v) for v in options.values())


def get_options_length(q: dict[str, Any]) -> int:
    """Get total length of all non-null options."""
    options = q.get("options", {})
    return sum(len(str(v)) for v in options.values() if not is_null_value(v))


def get_question_length(q: dict[str, Any]) -> int:
    """Get length of question text."""
    return len(q.get("question", ""))


def filter_invalid_questions(
    questions: list[dict[str, Any]],
    verbose: bool = False,
    chinese: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Filter out questions with null answers, all-null options, and non-MCQ types.

    Returns (valid_questions, filter_stats).
    """
    console.print("\n[bold]Pre-filter: Removing Invalid Questions[/bold]")

    valid = []
    null_answer_count = 0
    null_options_count = 0
    short_answer_count = 0

    for q in questions:
        # Drop 简答题 (short answer) for Chinese data - not MCQ format
        if chinese and q.get("type") == "简答题":
            short_answer_count += 1
            if verbose:
                console.print(
                    f"  [dim]Short answer: ID {q.get('id')} - {q.get('question', '')[:50]}...[/dim]"
                )
            continue
        if not has_valid_answer(q):
            null_answer_count += 1
            if verbose:
                console.print(
                    f"  [dim]Null answer: ID {q.get('id')} - {q.get('question', '')[:50]}...[/dim]"
                )
            continue
        if not has_valid_options(q):
            null_options_count += 1
            if verbose:
                console.print(
                    f"  [dim]All null options: ID {q.get('id')} - {q.get('question', '')[:50]}...[/dim]"
                )
            continue
        valid.append(q)

    if short_answer_count:
        console.print(
            f"  Removed [red]{short_answer_count}[/red] short answer (简答题)"
        )
    console.print(f"  Removed [red]{null_answer_count}[/red] with null answer")
    console.print(f"  Removed [red]{null_options_count}[/red] with all null options")
    console.print(f"  Kept [green]{len(valid)}[/green] valid questions")

    stats = {
        "null_answer_removed": null_answer_count,
        "null_options_removed": null_options_count,
        "short_answer_removed": short_answer_count,
        "after_filter": len(valid),
    }
    return valid, stats


def llm_quality_filter(
    questions: list[dict[str, Any]],
    model: str = IDUN_CHAT_MODEL,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    LLM Quality Filter stage: Remove garbage/OCR-error questions.

    Uses LLM to judge if each question is a valid exam question.
    Questions with images/pictures are kept (they will be split to a separate database later).
    Returns (kept_questions, removed_questions).
    """
    console.print(f"\n[bold]Stage: LLM Quality Filter[/bold] (model: {model})")

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    prompt_template = """Is this exam question usable? Answer with ONLY "GOOD" or "BAD" on the first line.

{question_text}

GOOD = readable, understandable (typos/formatting OK, image refs OK)
BAD = unreadable, empty, severe OCR errors, nonsensical

Most questions are GOOD. Only mark BAD if truly broken."""

    # Batch size balances throughput vs progress visibility (~50s per batch)
    # ThreadPoolExecutor with max_workers=30 maintains sliding window
    batch_size = 200

    with make_progress() as progress:
        task = progress.add_task("Checking question quality...", total=len(questions))

        for batch_start in range(0, len(questions), batch_size):
            batch = questions[batch_start : batch_start + batch_size]
            batch_texts = [
                prompt_template.format(question_text=format_question_for_llm(q))
                for q in batch
            ]

            responses = llm_judge_batch(
                batch_texts,
                model=model,
                max_tokens=100,
            )

            for q, response in zip(batch, responses):
                if response is None:
                    # API error, keep to be safe
                    kept.append(q)
                    continue

                # Parse simple GOOD/BAD response
                response_upper = response.upper().strip()
                first_line = response_upper.split("\n")[0].strip()

                # Check first line first, then whole response
                if first_line.startswith("GOOD") or first_line == "GOOD":
                    verdict = "GOOD"
                    reason = response.strip()[:100]
                elif first_line.startswith("BAD") or first_line == "BAD":
                    verdict = "BAD"
                    reason = response.strip()[:100]
                elif "GOOD" in response_upper and "BAD" not in response_upper:
                    verdict = "GOOD"
                    reason = response.strip()[:100]
                elif "BAD" in response_upper and "GOOD" not in response_upper:
                    verdict = "BAD"
                    reason = response.strip()[:100]
                else:
                    # Ambiguous or unparseable, keep to be safe
                    kept.append(q)
                    continue

                if verdict == "BAD":
                    removed.append(
                        {
                            "id": q.get("id"),
                            "question": q.get("question"),
                            "options": q.get("options"),
                            "answer": q.get("answer"),
                            "llm_reason": reason,
                            "llm_verdict": verdict,
                        }
                    )
                    if verbose:
                        console.print(
                            f"  [dim]Removed: ID {q.get('id')} - {reason}[/dim]"
                        )
                else:
                    kept.append(q)

            progress.advance(task, len(batch))

    console.print(
        f"  Removed [red]{len(removed)}[/red] low-quality questions, kept [green]{len(kept)}[/green]"
    )

    return kept, removed


def _is_english_question(q: dict[str, Any]) -> bool:
    """Check if a question is primarily in English (not Chinese)."""
    text = q.get("question", "")
    # Count Chinese characters (CJK Unified Ideographs range)
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # If less than 10% of non-whitespace chars are Chinese, it's English
    non_ws = len(text.replace(" ", ""))
    if non_ws == 0:
        return False
    return chinese_chars / non_ws < 0.1


def _is_valid_chinese_repair(text: str) -> bool:
    """Check if text is a valid Chinese repair (not English reasoning)."""
    text = text.strip()
    if not text or text.upper() == "UNCHANGED":
        return True
    # Must have a reasonable proportion of Chinese characters
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if chinese_chars < 4:
        return False
    # Reject if it starts with common English reasoning patterns
    lower = text.lower()
    reasoning_prefixes = (
        "we need",
        "we could",
        "the question",
        "the original",
        "based on",
        "this is",
        "let me",
        "i need",
        "the given",
        "the text",
        "maybe",
        "could be",
        "probably",
        "so the",
        "it seems",
        "the correct",
        "note:",
        "given the",
        "thus",
        'or "',
        "here is",
        "the ocr",
    )
    if any(lower.startswith(p) for p in reasoning_prefixes):
        return False
    return True


def _extract_chinese_repair(response: str) -> str | None:
    """
    Extract the corrected Chinese question from an LLM response.

    Reasoning models often output chain-of-thought in English before the
    actual Chinese answer. This function aggressively finds Chinese text.
    Returns None if no usable Chinese text found.
    """
    text = response.strip()

    # If response is just "UNCHANGED", return as-is
    if text.upper() == "UNCHANGED":
        return "UNCHANGED"

    # Strategy 1: Look for CORRECTED: marker (our new format)
    if "CORRECTED:" in text.upper():
        parts = text.upper().split("CORRECTED:", 1)
        if len(parts) == 2:
            # Find the actual CORRECTED: in original case
            idx = text.upper().index("CORRECTED:")
            candidate = text[idx + 10 :].strip()  # 10 = len("CORRECTED:")
            lines = candidate.split("\n")
            chinese_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # Stop at obvious English reasoning
                if stripped.lower().startswith(
                    ("note:", "explanation:", "we need", "the question", "based on")
                ):
                    break
                # Include lines with substantial Chinese content
                chinese_chars = sum(1 for c in stripped if "\u4e00" <= c <= "\u9fff")
                if chinese_chars > 3:
                    chinese_lines.append(stripped)
            if chinese_lines:
                return "\n".join(chinese_lines)

    # Strategy 2: Find longest contiguous Chinese segment
    lines = text.split("\n")
    all_segments: list[list[str]] = []
    current_segment: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_segment:
                all_segments.append(current_segment)
                current_segment = []
            continue

        # Count Chinese characters
        chinese_chars = sum(1 for c in stripped if "\u4e00" <= c <= "\u9fff")
        total_chars = len(stripped.replace(" ", ""))

        # If line is mostly Chinese (>30% Chinese chars and at least 5 chars)
        if chinese_chars >= 5 and (
            total_chars == 0 or chinese_chars / total_chars > 0.3
        ):
            current_segment.append(stripped)
        else:
            # Check if this is a short connector line (like "或者:" or "即:")
            if chinese_chars > 0 and len(stripped) < 10:
                current_segment.append(stripped)
            else:
                # Non-Chinese line, save current segment
                if current_segment:
                    all_segments.append(current_segment)
                    current_segment = []

    if current_segment:
        all_segments.append(current_segment)

    # Find the longest segment by total Chinese character count
    if all_segments:
        best_segment = max(
            all_segments,
            key=lambda seg: sum(
                sum(1 for c in line if "\u4e00" <= c <= "\u9fff") for line in seg
            ),
        )
        result = "\n".join(best_segment)

        # Strip common prefixes
        result = re.sub(
            r"^(corrected|answer|修正|答案|纠正|题目)\s*[:：]\s*",
            "",
            result,
            flags=re.IGNORECASE,
        )
        return result.strip() if result.strip() else None

    return None


def llm_repair_ocr(
    questions: list[dict[str, Any]],
    model: str = IDUN_CHAT_MODEL,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    LLM OCR Repair stage: Attempt to fix OCR-corrupted Chinese question text.

    Uses LLM with clean options + hint as context to reconstruct garbled questions.
    Stores original_question on repaired questions for audit trail.
    Skips questions that are already in English.
    Returns (repaired_questions, repair_stats).
    """
    console.print(f"\n[bold]Stage: LLM OCR Repair[/bold] (model: {model})")

    prompt_template = """你是OCR修复专家。修复下面中文题目的OCR错误。根据选项、提示和答案重构正确的题目文本。

严格要求：
1. 第一行必须是: CORRECTED:
2. 第二行开始输出修正后的完整中文题目
3. 绝对不要输出任何英文单词或解释
4. 不要输出思考过程
5. 不要添加其他说明文字
6. 如果题目已正确无需修复，只输出一行: UNCHANGED

题目（含OCR错误）：{question}
选项：
{options}
提示：{hint}
答案：{answer}

输出格式示例：
CORRECTED:
修正后的完整题目文本在这里"""

    json_response_format = None  # Don't use JSON format - reasoning models ignore it

    repaired_count = 0
    unchanged_count = 0
    failed_count = 0
    skipped_english_count = 0
    sample_repairs: list[dict[str, Any]] = []

    # Large batch size to minimize waiting between batches
    # ThreadPoolExecutor with max_workers=30 maintains sliding window of concurrent requests
    batch_size = 200

    # Separate English questions (skip repair)
    to_repair: list[tuple[int, dict[str, Any]]] = []
    for idx, q in enumerate(questions):
        if _is_english_question(q):
            skipped_english_count += 1
        else:
            to_repair.append((idx, q))

    if skipped_english_count:
        console.print(
            f"  Skipping [blue]{skipped_english_count}[/blue] English questions"
        )

    with make_progress() as progress:
        task = progress.add_task("Repairing OCR errors...", total=len(to_repair))

        for batch_start in range(0, len(to_repair), batch_size):
            batch = to_repair[batch_start : batch_start + batch_size]
            batch_prompts = []
            for _idx, q in batch:
                options = q.get("options", {})
                options_str = "\n".join(
                    f"  {k}: {v}" for k, v in sorted(options.items())
                )
                hint = q.get("hint") or "（无提示）"
                batch_prompts.append(
                    prompt_template.format(
                        question=q.get("question", ""),
                        options=options_str,
                        hint=hint,
                        answer=q.get("answer", ""),
                    )
                )

            responses = llm_judge_batch(
                batch_prompts,
                model=model,
                max_tokens=300,
                response_format=json_response_format,
            )

            for (_idx, q), response in zip(batch, responses):
                if response is None:
                    failed_count += 1
                    continue

                # Extract using new CORRECTED: format
                extracted = None
                response_text = response.strip()

                # Strategy 1: Look for CORRECTED: prefix
                if "CORRECTED:" in response_text:
                    # Extract everything after CORRECTED:
                    parts = response_text.split("CORRECTED:", 1)
                    if len(parts) == 2:
                        candidate = parts[1].strip()
                        # Stop at English reasoning if present
                        lines = candidate.split("\n")
                        chinese_lines = []
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            # Stop at English reasoning markers
                            if stripped.lower().startswith(
                                (
                                    "note:",
                                    "explanation:",
                                    "we need",
                                    "the question",
                                    "based on",
                                )
                            ):
                                break
                            chinese_lines.append(stripped)
                        if chinese_lines:
                            extracted = "\n".join(chinese_lines)

                # Strategy 2: Check for UNCHANGED
                if not extracted and response_text.upper() == "UNCHANGED":
                    extracted = "UNCHANGED"

                # Strategy 3: Fallback to old JSON extraction
                if not extracted:
                    parsed = parse_llm_json(response)
                    if parsed is not None:
                        candidate = parsed.get("corrected") or parsed.get("question")
                        if candidate and _is_valid_chinese_repair(candidate):
                            extracted = candidate

                # Strategy 4: General Chinese text extraction fallback
                if not extracted:
                    extracted = _extract_chinese_repair(response)

                # Validate the extracted text is actually Chinese, not reasoning
                if extracted and extracted.upper() != "UNCHANGED":
                    if not _is_valid_chinese_repair(extracted):
                        extracted = None

                if not extracted:
                    failed_count += 1
                    if verbose:
                        console.print(
                            f"  [dim]Failed ID {q.get('id')}: could not extract Chinese text[/dim]"
                        )
                    continue

                extracted = extracted.strip()
                # Strip stray quotes from JSON extraction
                extracted = extracted.strip("\"'")
                # Strip leading question number prefix (e.g. "1.", "140.")
                extracted = re.sub(r"^\d+[.、]\s*", "", extracted)
                # Strip "Or ..." alternative answers at the end
                extracted = re.split(r'\nOr "', extracted)[0].strip()

                if extracted.upper() == "UNCHANGED":
                    unchanged_count += 1
                else:
                    original = q["question"]
                    q["original_question"] = original
                    q["question"] = extracted
                    repaired_count += 1

                    if len(sample_repairs) < 20:
                        sample_repairs.append(
                            {
                                "id": q.get("id"),
                                "original": original,
                                "repaired": extracted,
                                "hint": q.get("hint"),
                            }
                        )

                    if verbose:
                        console.print(
                            f"  [dim]Repaired ID {q.get('id')}: {original[:40]}... → {extracted[:40]}...[/dim]"
                        )

            progress.advance(task, len(batch))

    console.print(f"  Repaired [green]{repaired_count}[/green] questions")
    console.print(f"  Unchanged [blue]{unchanged_count}[/blue] questions")
    console.print(f"  Skipped (English) [blue]{skipped_english_count}[/blue] questions")
    console.print(f"  Failed [red]{failed_count}[/red] questions")

    stats = {
        "name": "llm_repair",
        "model": model,
        "total": len(questions),
        "repaired_count": repaired_count,
        "unchanged_count": unchanged_count,
        "skipped_english": skipped_english_count,
        "failed_count": failed_count,
        "sample_repairs": sample_repairs,
    }

    return questions, stats


def create_comparison_text(q: dict[str, Any]) -> str:
    """Combine question + options + answer for comparison.

    This ensures questions with generic stems like "Which is true?" are only
    matched when their options are also similar.
    """
    question = normalize_text(q.get("question", ""))
    options = q.get("options", {})
    # Sort options by key for consistent ordering
    options_text = " ".join(
        f"{k}:{normalize_text(str(v))}" for k, v in sorted(options.items())
    )
    answer = str(q.get("answer", "")).upper()
    return f"{question} | {options_text} | {answer}"


def format_question_for_llm(q: dict[str, Any]) -> str:
    """Format a question for LLM prompts."""
    options = q.get("options", {})
    options_str = "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))
    return f"Question: {q.get('question', '')}\nOptions:\n{options_str}\nAnswer: {q.get('answer', '')}"


def llm_judge(
    prompt: str,
    model: str = IDUN_CHAT_MODEL,
    max_tokens: int = 100,
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    """
    Call IDUN LLM API for judging tasks.

    Returns the response text, or None if API call fails.
    If response_format is provided, it is passed to the API for structured output.
    """
    api_key = os.getenv("LITE_LLM_API_KEY")
    if not api_key:
        console.print("[yellow]LITE_LLM_API_KEY not set[/yellow]")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format is not None:
        payload["response_format"] = response_format

    try:
        response = requests.post(
            IDUN_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        # Some models (reasoning models) may return content as None with reasoning_content
        if content is None:
            content = message.get("reasoning_content", "")
        return content.strip() if content else None
    except requests.RequestException as e:
        console.print(f"[red]LLM API error: {e}[/red]")
        return None
    except (KeyError, IndexError, TypeError) as e:
        console.print(f"[red]LLM API response parse error: {e}[/red]")
        return None


def llm_judge_batch(
    prompts: list[str],
    model: str = IDUN_CHAT_MODEL,
    max_tokens: int = 100,
    temperature: float = 0,
    max_workers: int = 30,
    response_format: dict[str, Any] | None = None,
) -> list[str | None]:
    """
    Call IDUN LLM API for multiple prompts concurrently.

    Returns a list of responses (same order as prompts). None for failed calls.
    """
    results: list[str | None] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                llm_judge, prompt, model, max_tokens, temperature, response_format
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                console.print(f"[red]Batch LLM error: {e}[/red]")
                results[idx] = None

    return results


def parse_llm_json(response: str | None) -> dict[str, Any] | None:
    """Parse JSON from LLM response, extracting JSON object from anywhere in text."""
    if response is None:
        return None
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```"]
        text = "\n".join(lines)

    # Try parsing the whole text first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from within the text
    # Look for {...} pattern
    import re

    json_pattern = r"\{[^{}]*\}"
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def load_questions(file_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load questions from JSON file, return metadata and questions list."""
    console.print(f"Loading questions from [cyan]{file_path}[/cyan]...")
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    metadata = {k: v for k, v in data.items() if k != "questions"}
    console.print(f"Loaded [green]{len(questions)}[/green] questions")
    return metadata, questions


def choose_better_question(
    q1: dict[str, Any], q2: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Choose the better question to keep from a duplicate pair.

    Criteria (in order):
    1. Longer total options length
    2. Longer question text

    Returns (kept, removed).
    """
    opts_len_1 = get_options_length(q1)
    opts_len_2 = get_options_length(q2)

    if opts_len_1 > opts_len_2:
        return q1, q2
    elif opts_len_2 > opts_len_1:
        return q2, q1

    # Same options length, use question length
    q_len_1 = get_question_length(q1)
    q_len_2 = get_question_length(q2)

    if q_len_2 > q_len_1:
        return q2, q1

    # Default: keep q1 (first encountered)
    return q1, q2


def fuzzy_dedup(
    questions: list[dict[str, Any]],
    threshold: float = 0.98,
    verbose: bool = False,
    label: str = "Fuzzy Matching",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fuzzy string matching deduplication.

    Uses rapidfuzz token_set_ratio for efficient fuzzy matching.
    When duplicates are found, keeps the question with longer options (then question length).
    Returns (kept_questions, removed_pairs).
    """
    # Convert threshold from 0-1 to 0-100 for rapidfuzz
    threshold_pct = threshold * 100
    console.print(f"\n[bold]Stage: {label}[/bold] (threshold: {threshold})")

    # Pre-compute comparison texts
    comparison_texts = [create_comparison_text(q) for q in questions]

    # Track which indices to keep (start with all)
    keep_mask = [True] * len(questions)
    removed_pairs: list[dict[str, Any]] = []

    with make_progress() as progress:
        task = progress.add_task("Comparing questions...", total=len(questions))

        for i in range(len(questions)):
            if not keep_mask[i]:
                progress.advance(task)
                continue

            text_i = comparison_texts[i]

            # Compare with remaining questions
            for j in range(i + 1, len(questions)):
                if not keep_mask[j]:
                    continue

                text_j = comparison_texts[j]

                # Use token_set_ratio for better handling of word order differences
                similarity_pct = fuzz.token_set_ratio(text_i, text_j)
                similarity = similarity_pct / 100.0  # Convert to 0-1 scale

                if similarity_pct >= threshold_pct:
                    # Choose the better question to keep
                    kept_q, removed_q = choose_better_question(
                        questions[i], questions[j]
                    )

                    # Mark the removed one
                    if kept_q is questions[j]:
                        # j is better, mark i for removal and continue with j
                        keep_mask[i] = False
                        kept_idx, removed_idx = j, i
                    else:
                        # i is better (or equal), mark j for removal
                        keep_mask[j] = False
                        kept_idx, removed_idx = i, j

                    removed_pairs.append(
                        {
                            "similarity": similarity,
                            "reason": "fuzzy_match",
                            "kept": {
                                "id": kept_q.get("id"),
                                "question": kept_q.get("question"),
                                "options": kept_q.get("options"),
                                "answer": kept_q.get("answer"),
                            },
                            "removed": {
                                "id": removed_q.get("id"),
                                "question": removed_q.get("question"),
                                "options": removed_q.get("options"),
                                "answer": removed_q.get("answer"),
                            },
                        }
                    )

                    if verbose:
                        console.print(
                            f"  [dim]Duplicate ({similarity:.4f}): ID {removed_q.get('id')} ← kept ID {kept_q.get('id')}[/dim]"
                        )

                    # If i was marked for removal, break the inner loop
                    if not keep_mask[i]:
                        break

            progress.advance(task)

    # Filter to kept questions
    kept_questions = [q for q, keep in zip(questions, keep_mask) if keep]

    console.print(
        f"  Removed [red]{len(removed_pairs)}[/red] duplicates, kept [green]{len(kept_questions)}[/green]"
    )

    return kept_questions, removed_pairs


def get_idun_embeddings(
    texts: list[str],
    batch_size: int = 100,
) -> list[np.ndarray] | None:
    """
    Get embeddings from IDUN embedding API (Qwen3-Embedding-8B).

    Returns list of embeddings or None if API is unavailable.
    """
    api_key = os.getenv("LITE_LLM_API_KEY")
    if not api_key:
        console.print(
            "[yellow]LITE_LLM_API_KEY not set, falling back to local embeddings[/yellow]"
        )
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[np.ndarray] = []

    with make_progress() as progress:
        task = progress.add_task("Embedding via IDUN API...", total=len(texts))

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            payload = {
                "model": IDUN_EMBEDDING_MODEL,
                "input": batch,
            }

            try:
                response = requests.post(
                    IDUN_EMBEDDING_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                # Extract embeddings from OpenAI-compatible response format
                batch_embeddings = [
                    np.array(item["embedding"])
                    for item in sorted(data["data"], key=lambda x: x["index"])
                ]
                all_embeddings.extend(batch_embeddings)
                progress.advance(task, len(batch))

            except requests.RequestException as e:
                console.print(f"[red]IDUN API error: {e}[/red]")
                return None

    return all_embeddings


def get_local_embeddings(
    texts: list[str],
    batch_size: int = 100,
) -> list[np.ndarray] | None:
    """
    Get embeddings using local FastEmbed model.

    Returns list of embeddings or None if FastEmbed is unavailable.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError:
        console.print("[red]fastembed not available[/red]")
        return None

    console.print("  Loading local embedding model...")
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_embeddings: list[np.ndarray] = []

    with make_progress() as progress:
        task = progress.add_task("Embedding locally...", total=len(texts))

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = list(model.embed(batch))
            all_embeddings.extend(embeddings)
            progress.advance(task, len(batch))

    return all_embeddings


def _faiss_cosine_search(
    normalized: np.ndarray,
    threshold: float,
    k: int = 256,
    search_batch_size: int = 8192,
) -> list[tuple[int, int, float]]:
    """Find all pairs with cosine similarity >= threshold using FAISS.

    Vectors must already be L2-normalized (inner product = cosine similarity).
    Returns list of (i, j, similarity) tuples where i < j.
    Uses GPU if available, falls back to CPU.
    """
    import faiss

    n, d = normalized.shape
    data = normalized.astype(np.float32)

    # Build index (inner product on normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(d)

    # Try GPU
    use_gpu = False
    try:
        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            use_gpu = True
            console.print(f"  [green]FAISS using GPU ({ngpus} available)[/green]")
    except (AttributeError, Exception) as e:
        console.print(f"  [yellow]FAISS GPU unavailable ({e}), using CPU[/yellow]")

    if not use_gpu:
        console.print("  FAISS using CPU")

    index.add(data)

    # Clamp k to number of vectors (can't ask for more neighbors than exist)
    effective_k = min(k, n)

    # Search in batches
    pairs: list[tuple[int, int, float]] = []
    k_warning = False

    with make_progress() as progress:
        task = progress.add_task("FAISS similarity search...", total=n)

        for batch_start in range(0, n, search_batch_size):
            batch_end = min(batch_start + search_batch_size, n)
            queries = data[batch_start:batch_end]

            similarities, indices = index.search(queries, effective_k)

            for local_idx in range(batch_end - batch_start):
                global_i = batch_start + local_idx

                for rank in range(effective_k):
                    j = int(indices[local_idx, rank])
                    sim = float(similarities[local_idx, rank])

                    if j == global_i or j < 0 or sim < threshold:
                        continue
                    # Only keep pairs where i < j to avoid duplicates
                    if global_i < j:
                        pairs.append((global_i, j, sim))

                # Safety check: k-th result still above threshold
                if similarities[local_idx, effective_k - 1] >= threshold:
                    k_warning = True

            progress.advance(task, batch_end - batch_start)

    if k_warning:
        console.print(
            f"  [yellow]Warning: k={effective_k} may be too small. "
            f"Some queries returned {effective_k} results all above threshold. "
            f"Consider re-running with --faiss-k {effective_k * 2}.[/yellow]"
        )

    return pairs


def _numpy_cosine_search(
    normalized: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Fallback O(n^2) cosine similarity search using numpy.

    Used when FAISS is not installed.
    """
    n = normalized.shape[0]
    pairs: list[tuple[int, int, float]] = []

    with make_progress() as progress:
        task = progress.add_task("Numpy similarity search (slow)...", total=n)

        for i in range(n):
            similarities = np.dot(normalized[i], normalized[i + 1 :].T)
            duplicate_indices = np.where(similarities >= threshold)[0] + i + 1

            for j in duplicate_indices:
                pairs.append((i, int(j), float(similarities[j - i - 1])))

            progress.advance(task)

    return pairs


def embedding_dedup(
    questions: list[dict[str, Any]],
    threshold: float = 0.97,
    verbose: bool = False,
    batch_size: int = 100,
    use_local: bool = False,
    faiss_k: int = 256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Embedding similarity deduplication using FAISS.

    Uses IDUN embedding API (Qwen3-Embedding-8B) by default, falls back to local FastEmbed.
    Similarity search uses FAISS IndexFlatIP (GPU if available, else CPU).
    Returns (kept_questions, removed_pairs).
    """
    console.print(
        f"\n[bold]Stage: Embedding Similarity[/bold] (threshold: {threshold})"
    )

    # Pre-compute comparison texts
    comparison_texts = [create_comparison_text(q) for q in questions]
    console.print(f"  Embedding {len(comparison_texts)} questions...")

    # Try IDUN API first, fall back to local
    if use_local:
        console.print("  Using local embeddings (--local flag)")
        all_embeddings = get_local_embeddings(comparison_texts, batch_size)
    else:
        console.print(f"  Using IDUN API ({IDUN_EMBEDDING_MODEL})...")
        all_embeddings = get_idun_embeddings(comparison_texts, batch_size)
        if all_embeddings is None:
            console.print("  [yellow]Falling back to local embeddings...[/yellow]")
            all_embeddings = get_local_embeddings(comparison_texts, batch_size)

    if all_embeddings is None:
        console.print(
            "[red]No embedding method available, skipping embedding stage[/red]"
        )
        return questions, []

    if len(all_embeddings) != len(questions):
        console.print(
            f"[red]Embedding count mismatch: {len(all_embeddings)} vs {len(questions)}[/red]"
        )
        return questions, []

    # Convert to numpy and normalize for cosine similarity
    embeddings_matrix = np.vstack(all_embeddings)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings_matrix / norms

    # FAISS similarity search (replaces O(n²) numpy loop)
    console.print("  Running FAISS similarity search...")
    try:
        candidate_pairs = _faiss_cosine_search(normalized, threshold, k=faiss_k)
    except ImportError:
        console.print(
            "  [yellow]FAISS not installed, falling back to numpy similarity[/yellow]"
        )
        candidate_pairs = _numpy_cosine_search(normalized, threshold)

    console.print(f"  Found {len(candidate_pairs)} candidate pairs above threshold")

    # Sort pairs by descending similarity (resolve highest-confidence duplicates first)
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    # Process pairs using keep_mask pattern
    keep_mask = [True] * len(questions)
    removed_pairs: list[dict[str, Any]] = []

    for i, j, sim_score in candidate_pairs:
        if not keep_mask[i] or not keep_mask[j]:
            continue

        kept_q, removed_q = choose_better_question(questions[i], questions[j])

        if kept_q is questions[j]:
            keep_mask[i] = False
        else:
            keep_mask[j] = False

        removed_pairs.append(
            {
                "similarity": sim_score,
                "reason": "embedding_similarity",
                "kept": {
                    "id": kept_q.get("id"),
                    "question": kept_q.get("question"),
                    "options": kept_q.get("options"),
                    "answer": kept_q.get("answer"),
                },
                "removed": {
                    "id": removed_q.get("id"),
                    "question": removed_q.get("question"),
                    "options": removed_q.get("options"),
                    "answer": removed_q.get("answer"),
                },
            }
        )

        if verbose:
            console.print(
                f"  [dim]Duplicate ({sim_score:.3f}): ID {removed_q.get('id')} -> {kept_q.get('id')}[/dim]"
            )

    # Filter to kept questions
    kept_questions = [q for q, keep in zip(questions, keep_mask) if keep]

    console.print(
        f"  Removed [red]{len(removed_pairs)}[/red] duplicates, kept [green]{len(kept_questions)}[/green]"
    )

    return kept_questions, removed_pairs


def llm_verify_duplicates(
    questions: list[dict[str, Any]],
    removed_pairs: list[dict[str, Any]],
    verify_threshold: float = 0.97,
    model: str = IDUN_CHAT_MODEL,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    LLM Duplicate Verification stage: Verify flagged duplicates and recover false positives.

    Reviews pairs where:
    - Embedding similarity is 70%-97% (borderline cases)
    - All fuzzy matches (can have false positives)

    Returns (updated_questions, confirmed_pairs, recovered_pairs).
    """
    console.print(f"\n[bold]Stage: LLM Duplicate Verification[/bold] (model: {model})")

    # Separate pairs to verify vs skip
    to_verify: list[dict[str, Any]] = []
    confirmed_high_similarity: list[dict[str, Any]] = []

    def get_answer_content(q: dict[str, Any]) -> str:
        """Get the actual answer text, not just the letter."""
        answer_key = str(q.get("answer", "")).upper()
        options = q.get("options", {})
        answer_text = str(options.get(answer_key, "")).lower().strip()
        # Normalize: remove punctuation, extra spaces
        import re

        answer_text = re.sub(r"[;/,\s]+", " ", answer_text).strip()
        return answer_text

    for pair in removed_pairs:
        similarity = pair.get("similarity", 0)
        reason = pair.get("reason", "")

        # Compare actual answer content, not just letter
        kept_answer_content = get_answer_content(pair.get("kept", {}))
        removed_answer_content = get_answer_content(pair.get("removed", {}))
        answers_match = kept_answer_content == removed_answer_content

        # Trust pairs where answer content matches AND similarity is high enough
        # This catches cases where options are reordered (different letter, same content)
        content_match_threshold = 0.85  # Lower threshold when answer content matches
        if answers_match and similarity >= content_match_threshold:
            confirmed_high_similarity.append(
                {
                    **pair,
                    "recovered": False,
                    "llm_verified": False,  # Skipped - same answer content + high similarity
                    "llm_reason": None,
                }
            )
        elif reason == "embedding_similarity" and similarity >= verify_threshold:
            # Very high similarity (>97%), trust even without answer match check
            confirmed_high_similarity.append(
                {
                    **pair,
                    "recovered": False,
                    "llm_verified": False,  # Skipped - trusted due to very high similarity
                    "llm_reason": None,
                }
            )
        else:
            # Verify: lower similarity, different answer content, or fuzzy matches
            to_verify.append(pair)

    console.print(
        f"  Skipping {len(confirmed_high_similarity)} high-similarity pairs (>{verify_threshold:.0%} + same answer)"
    )
    console.print(f"  Verifying {len(to_verify)} borderline pairs...")

    if not to_verify:
        console.print("  No pairs to verify")
        return questions, removed_pairs, []

    prompt_template = """Are these two exam questions duplicates or different? Answer ONLY "DUPLICATE" or "DIFFERENT" on the first line.

Q1: {q1_text}

Q2: {q2_text}

DUPLICATE = same question, reordered options (answer content same even if letter differs)
DIFFERENT = different questions, topics, or images"""

    confirmed_pairs: list[dict[str, Any]] = list(confirmed_high_similarity)
    recovered_pairs: list[dict[str, Any]] = []

    # Batch size balances throughput vs progress visibility (~50s per batch)
    # ThreadPoolExecutor with max_workers=30 maintains sliding window
    batch_size = 200

    with make_progress() as progress:
        task = progress.add_task("Verifying duplicates...", total=len(to_verify))

        for batch_start in range(0, len(to_verify), batch_size):
            batch = to_verify[batch_start : batch_start + batch_size]
            batch_prompts = []
            for pair in batch:
                q1_text = format_question_for_llm(pair["kept"])
                q2_text = format_question_for_llm(pair["removed"])
                batch_prompts.append(
                    prompt_template.format(q1_text=q1_text, q2_text=q2_text)
                )

            responses = llm_judge_batch(
                batch_prompts,
                model=model,
                max_tokens=150,
            )

            for pair, response in zip(batch, responses):
                if response is None:
                    # API error, trust original decision (keep as duplicate)
                    confirmed_pairs.append(
                        {
                            **pair,
                            "recovered": False,
                            "llm_verified": False,
                            "llm_reason": None,
                            "llm_verdict": None,
                        }
                    )
                    continue

                # Parse DUPLICATE/DIFFERENT response
                response_upper = response.upper().strip()
                first_line = response_upper.split("\n")[0].strip()
                last_line = response_upper.split("\n")[-1].strip()

                # Check first line for clear verdict
                if first_line.startswith("DIFFERENT") or first_line == "DIFFERENT":
                    verdict = "DIFFERENT"
                    reason = "first_line"
                elif first_line.startswith("DUPLICATE") or first_line == "DUPLICATE":
                    verdict = "DUPLICATE"
                    reason = "first_line"
                # Check last line (some models put verdict at end)
                elif last_line.startswith("DIFFERENT") or last_line == "DIFFERENT":
                    verdict = "DIFFERENT"
                    reason = "last_line"
                elif last_line.startswith("DUPLICATE") or last_line == "DUPLICATE":
                    verdict = "DUPLICATE"
                    reason = "last_line"
                # Count occurrences - if one appears more, use that
                elif response_upper.count("DIFFERENT") > response_upper.count(
                    "DUPLICATE"
                ):
                    verdict = "DIFFERENT"
                    reason = "count_inferred"
                elif response_upper.count("DUPLICATE") > response_upper.count(
                    "DIFFERENT"
                ):
                    verdict = "DUPLICATE"
                    reason = "count_inferred"
                else:
                    # Truly ambiguous, keep as duplicate (conservative)
                    confirmed_pairs.append(
                        {
                            **pair,
                            "recovered": False,
                            "llm_verified": True,
                            "llm_reason": first_line[:50],
                            "llm_verdict": "AMBIGUOUS",
                        }
                    )
                    continue

                if verdict == "DIFFERENT":
                    recovered_pairs.append(
                        {
                            **pair,
                            "recovered": True,
                            "llm_verified": True,
                            "llm_reason": reason,
                            "llm_verdict": verdict,
                        }
                    )
                    if verbose:
                        console.print(
                            f"  [dim]Recovered: ID {pair['removed'].get('id')} - {reason}[/dim]"
                        )
                else:
                    confirmed_pairs.append(
                        {
                            **pair,
                            "recovered": False,
                            "llm_verified": True,
                            "llm_reason": reason,
                            "llm_verdict": verdict,
                        }
                    )

            progress.advance(task, len(batch))

    # Restore recovered questions to the list
    recovered_questions = []
    for pair in recovered_pairs:
        # Reconstruct the full question object from the removed data
        recovered_q = {
            "id": pair["removed"].get("id"),
            "question": pair["removed"].get("question"),
            "options": pair["removed"].get("options"),
            "answer": pair["removed"].get("answer"),
        }
        recovered_questions.append(recovered_q)

    updated_questions = questions + recovered_questions

    # Sort by similarity descending (highest first, lowest at bottom for easier validation)
    confirmed_pairs.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    recovered_pairs.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    console.print(f"  Confirmed [red]{len(confirmed_pairs)}[/red] duplicates")
    console.print(f"  Recovered [green]{len(recovered_pairs)}[/green] false positives")
    console.print(f"  Final count: [green]{len(updated_questions)}[/green] questions")

    return updated_questions, confirmed_pairs, recovered_pairs


def normalize_question_types(questions: list[dict[str, Any]]) -> None:
    """
    Normalize question types and answers in-place.

    - Converts Chinese type names to English using TYPE_MAP
    - Converts answer to array for multiple_choice questions
    """
    for q in questions:
        # Convert type to English
        old_type = q.get("type")
        if old_type in TYPE_MAP:
            q["type"] = TYPE_MAP[old_type]

        # Convert answer to array for multiple choice
        if q.get("type") == "multiple_choice":
            answer = q.get("answer", "")
            if isinstance(answer, str):
                # Split "ABC" into ["A", "B", "C"]
                q["answer"] = list(answer.upper().replace(" ", "").replace(",", ""))


def save_results(
    metadata: dict[str, Any],
    questions: list[dict[str, Any]],
    report: dict[str, Any],
    output_file: Path,
    report_file: Path,
    dry_run: bool = False,
) -> None:
    """Save deduplicated questions and report."""
    if dry_run:
        console.print("\n[yellow]Dry run - not saving files[/yellow]")
        return

    # Normalize types (Chinese -> English) and convert multiple choice answers to arrays
    normalize_question_types(questions)

    # Compute type statistics
    by_type: dict[str, int] = {}
    for q in questions:
        qtype = q.get("type", "unknown")
        by_type[qtype] = by_type.get(qtype, 0) + 1

    # Save deduplicated questions
    output_data = {
        **metadata,
        "statistics": {
            "total_questions": len(questions),
            "by_type": by_type,
        },
        "deduplication": {
            "original_count": report["original_count"],
            "final_count": len(questions),
            "removed_count": report["original_count"] - len(questions),
            "timestamp": datetime.now().isoformat(),
        },
        "questions": questions,
    }

    console.print(f"\nSaving deduplicated questions to [cyan]{output_file}[/cyan]...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Sort removed_pairs by similarity (highest first) for easier review
    for stage in report.get("stages", []):
        if "removed_pairs" in stage:
            stage["removed_pairs"] = sorted(
                stage["removed_pairs"],
                key=lambda x: x.get("similarity", 0),
                reverse=True,
            )

    # Save report
    console.print(f"Saving report to [cyan]{report_file}[/cyan]...")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_summary(report: dict[str, Any]) -> None:
    """Print summary table."""
    table = Table(title="Deduplication Summary")
    table.add_column("Stage", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("Removed", justify="right", style="red")
    table.add_column("After", justify="right", style="green")

    current = report["original_count"]

    # Show filter stage if present
    filtered = report.get("filtered", {})
    if filtered:
        short_answer = filtered.get("short_answer_removed", 0)
        null_answer = filtered.get("null_answer_removed", 0)
        null_options = filtered.get("null_options_removed", 0)
        after_filter = filtered.get("after_filter", current)
        if short_answer:
            table.add_row(
                "filter (简答题)",
                str(current),
                str(short_answer),
                str(current - short_answer),
            )
            current -= short_answer
        table.add_row(
            "filter (null answer)",
            str(current),
            str(null_answer),
            str(current - null_answer),
        )
        current -= null_answer
        table.add_row(
            "filter (null options)",
            str(current),
            str(null_options),
            str(after_filter),
        )
        current = after_filter

    for stage in report.get("stages", []):
        stage_name = stage["name"]

        if stage_name == "llm_repair":
            # Repair stage doesn't change count, show repair stats
            repaired = stage.get("repaired_count", 0)
            failed = stage.get("failed_count", 0)
            table.add_row(
                stage_name,
                str(current),
                f"[green]{repaired} repaired[/green], [red]{failed} failed[/red]",
                str(current),
            )
        elif stage_name == "llm_quality":
            # Quality filter removes questions
            removed = stage.get("removed_count", 0)
            after = current - removed
            table.add_row(stage_name, str(current), str(removed), str(after))
            current = after
        elif stage_name == "llm_verify":
            # Verification recovers false positives (negative removal)
            recovered = stage.get("recovered_count", 0)
            after = current + recovered
            # Show recovered as negative removal (green)
            table.add_row(
                stage_name,
                str(current),
                f"[green]+{recovered}[/green]",
                str(after),
            )
            current = after
        else:
            # Standard dedup stages (fuzzy, embedding)
            removed = len(stage.get("removed_pairs", []))
            after = current - removed
            table.add_row(stage_name, str(current), str(removed), str(after))
            current = after

    table.add_row(
        "[bold]Total[/bold]",
        str(report["original_count"]),
        str(report["original_count"] - report["final_count"]),
        str(report["final_count"]),
        style="bold",
    )

    console.print()
    console.print(table)


@app.command()
def main(
    input_file: Path = typer.Option(
        None, "--input", "-i", help="Input JSON file (auto-set by --chinese)"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output JSON file (auto-set by --chinese)"
    ),
    report_file: Path = typer.Option(
        None, "--report", "-r", help="Report JSON file (auto-set by --chinese)"
    ),
    stages: str = typer.Option(
        None,
        "--stages",
        "-s",
        help="Comma-separated stages: llm_repair, exact_fuzzy, llm_quality, embedding, fuzzy, llm_verify",
    ),
    chinese: bool = typer.Option(
        False, "--chinese", help="Process Chinese dataset instead of English"
    ),
    fuzzy_threshold: float = typer.Option(
        0.70, "--fuzzy-threshold", help="Fuzzy match threshold (0-1, default: 0.70)"
    ),
    embedding_threshold: float = typer.Option(
        0.70,
        "--embedding-threshold",
        help="Embedding similarity threshold (0-1, default: 0.70)",
    ),
    verify_threshold: float = typer.Option(
        0.97,
        "--verify-threshold",
        help="Skip LLM verification for embedding pairs above this (default: 0.97)",
    ),
    llm_model: str = typer.Option(
        IDUN_CHAT_MODEL, "--llm-model", help="LLM model for quality/verification stages"
    ),
    limit: int = typer.Option(
        0, "--limit", "-l", help="Limit to N random questions for testing (0 = all)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report only, don't save files"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show duplicate pairs"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    local: bool = typer.Option(
        False, "--local", help="Use local FastEmbed instead of IDUN API"
    ),
    faiss_k: int = typer.Option(
        256, "--faiss-k", help="FAISS max neighbors per query (default: 256)"
    ),
) -> None:
    """Deduplicate MCQ questions using fuzzy matching, embedding similarity, and LLM-as-judge."""

    # Set defaults based on --chinese flag
    if input_file is None:
        input_file = CHINESE_INPUT_FILE if chinese else INPUT_FILE
    if output_file is None:
        output_file = CHINESE_OUTPUT_FILE if chinese else OUTPUT_FILE
    if report_file is None:
        report_file = CHINESE_REPORT_FILE if chinese else REPORT_FILE
    if stages is None:
        stages = DEFAULT_STAGES_CHINESE if chinese else DEFAULT_STAGES_ENGLISH

    lang = "Chinese" if chinese else "English"
    console.print(f"[bold]MCQ Deduplication Tool[/bold] ({lang})\n")

    # Parse stages
    stage_list = [s.strip().lower() for s in stages.split(",")]
    valid_stages = {
        "exact_fuzzy",
        "fuzzy",
        "embedding",
        "llm_quality",
        "llm_verify",
        "llm_repair",
    }
    for s in stage_list:
        if s not in valid_stages:
            console.print(
                f"[red]Invalid stage: {s}. Valid stages: {valid_stages}[/red]"
            )
            raise typer.Exit(1)

    if "llm_repair" in stage_list and not chinese:
        console.print(
            "[yellow]Warning: llm_repair stage is designed for Chinese data[/yellow]"
        )

    # Load questions
    metadata, questions = load_questions(input_file)

    # Sample if limit is set
    if limit > 0 and limit < len(questions):
        console.print(
            f"[yellow]Sampling {limit} random questions (seed={seed})[/yellow]"
        )
        random.seed(seed)
        questions = random.sample(questions, limit)

    original_count = len(questions)

    # Initialize report
    embedding_model = (
        "sentence-transformers/all-MiniLM-L6-v2" if local else IDUN_EMBEDDING_MODEL
    )
    report: dict[str, Any] = {
        "original_count": original_count,
        "final_count": original_count,
        "language": lang.lower(),
        "config": {
            "stages": stage_list,
            "stage_order": " → ".join(stage_list),
            "fuzzy_threshold": fuzzy_threshold,
            "embedding_threshold": embedding_threshold,
            "verify_threshold": verify_threshold,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
        },
        "stages": [],
        "filtered": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Track all removed pairs for llm_verify stage
    all_removed_pairs: list[dict[str, Any]] = []

    # Pre-filter: remove invalid questions (null answers, all-null options, non-MCQ types)
    questions, filter_stats = filter_invalid_questions(
        questions, verbose=verbose, chinese=chinese
    )
    report["filtered"] = filter_stats

    # Run stages in the order specified
    for stage in stage_list:
        if stage == "llm_repair":
            questions, repair_stats = llm_repair_ocr(
                questions, model=llm_model, verbose=verbose
            )
            report["stages"].append(repair_stats)
        elif stage == "exact_fuzzy":
            questions, removed = fuzzy_dedup(
                questions, threshold=0.99, verbose=verbose, label="Exact Fuzzy"
            )
            report["stages"].append(
                {
                    "name": "exact_fuzzy",
                    "threshold": 0.99,
                    "removed_pairs": removed,
                }
            )
        elif stage == "llm_quality":
            questions, removed = llm_quality_filter(
                questions, model=llm_model, verbose=verbose
            )
            report["stages"].append(
                {
                    "name": "llm_quality",
                    "model": llm_model,
                    "removed_count": len(removed),
                    "removed_questions": removed,
                }
            )
        elif stage == "fuzzy":
            questions, removed = fuzzy_dedup(
                questions, threshold=fuzzy_threshold, verbose=verbose
            )
            all_removed_pairs.extend(removed)
            report["stages"].append(
                {
                    "name": "fuzzy",
                    "threshold": fuzzy_threshold,
                    "removed_pairs": removed,
                }
            )
        elif stage == "embedding":
            questions, removed = embedding_dedup(
                questions,
                threshold=embedding_threshold,
                verbose=verbose,
                use_local=local,
                faiss_k=faiss_k,
            )
            all_removed_pairs.extend(removed)
            report["stages"].append(
                {
                    "name": "embedding",
                    "threshold": embedding_threshold,
                    "removed_pairs": removed,
                }
            )
        elif stage == "llm_verify":
            if not all_removed_pairs:
                console.print(
                    "\n[yellow]No duplicate pairs to verify, skipping llm_verify[/yellow]"
                )
                continue
            questions, confirmed, recovered = llm_verify_duplicates(
                questions,
                all_removed_pairs,
                verify_threshold=verify_threshold,
                model=llm_model,
                verbose=verbose,
            )
            report["stages"].append(
                {
                    "name": "llm_verify",
                    "model": llm_model,
                    "verify_threshold": verify_threshold,
                    "confirmed_count": len(confirmed),
                    "recovered_count": len(recovered),
                    "confirmed_pairs": confirmed,  # All pairs with recovered: false
                    "recovered_pairs": recovered,  # All pairs with recovered: true
                }
            )

    report["final_count"] = len(questions)

    # Print summary
    print_summary(report)

    # Save results
    save_results(metadata, questions, report, output_file, report_file, dry_run)

    console.print("\n[green]Done![/green]")


if __name__ == "__main__":
    app()
