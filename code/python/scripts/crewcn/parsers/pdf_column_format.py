"""
Parser for PDF documents with embedded answer format.

Handles:
- Latest Sailing English (Chinese and English versions) - 2012.pdf

Format:
- Questions numbered like: 0001., 0036., etc.
- Answer embedded in question: ___D___ or ____A____
- Options on separate lines: A．xxx B．yyy (full-width period)
- Chinese translations interspersed
"""

import re
from pathlib import Path

import pymupdf

from .base import Question


class PDFColumnFormatParser:
    """Parser for PDF with embedded answers and full-width option markers."""

    # Question pattern: 4 digits followed by period
    QUESTION_PATTERN = re.compile(r"^(\d{4})\.\s*(.+)", re.MULTILINE)

    # Embedded answer pattern: ___X___ or ____X____
    ANSWER_PATTERN = re.compile(r"_+([A-D])_+")

    # Option line pattern: starts with A./B./C./D. (full or half width)
    OPTION_LINE_PATTERN = re.compile(r"^([A-D])[.．、]\s*(.+)$")

    # Inline options pattern (multiple on one line)
    # Match option letter, then content up to next option letter+period
    # Note: sometimes there's no space before next option (e.g., "参考C．")
    INLINE_OPTIONS_PATTERN = re.compile(r"([A-D])[.．、]\s*(.+?)(?=[A-D][.．、]|$)")

    def __init__(self, file_path: str, exam_title: str):
        self.file_path = file_path
        self.exam_title = exam_title
        self.doc = pymupdf.open(file_path)

    def parse(self) -> list[Question]:
        """Parse PDF and extract questions."""
        # Extract all text from PDF
        self.full_text = ""
        for page in self.doc:
            self.full_text += page.get_text("text") + "\n"

        return self._parse_text(self.full_text)

    def count_answer_patterns(self) -> int:
        """Count how many ANSWER_PATTERN matches are in the full text."""
        if not hasattr(self, "full_text"):
            return 0
        return len(self.ANSWER_PATTERN.findall(self.full_text))

    def _parse_text(self, text: str) -> list[Question]:
        """Parse full text into questions."""
        questions: list[Question] = []
        lines = text.split("\n")

        current_question: dict | None = None
        question_id = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and headers
            if not line or "crewcn.com" in line.lower():
                i += 1
                continue

            # Check if this starts a new question (4-digit number)
            q_match = re.match(r"^(\d{4})\.\s*(.+)", line)
            if q_match:
                # Save previous question
                if current_question:
                    q = self._finalize_question(current_question)
                    if q:
                        questions.append(q)

                # Start new question
                question_id += 1
                q_text = q_match.group(2)

                # Extract embedded answer
                answer_match = self.ANSWER_PATTERN.search(q_text)
                answer = answer_match.group(1) if answer_match else ""

                # Clean question text
                q_text = self.ANSWER_PATTERN.sub("______", q_text)

                current_question = {
                    "id": question_id,
                    "q_num": q_match.group(1),
                    "text": q_text,
                    "answer": answer,
                    "options": {},
                }
                i += 1
                continue

            # Check if this is an option line
            if current_question:
                # Check if line contains option patterns
                option_markers = re.findall(r"[A-D][.．、]", line)

                if len(option_markers) > 1:
                    # Multiple options on this line - use inline pattern
                    inline_matches = list(self.INLINE_OPTIONS_PATTERN.finditer(line))
                    for match in inline_matches:
                        letter = match.group(1).upper()
                        content = match.group(2).strip()
                        if letter not in current_question["options"]:
                            current_question["options"][letter] = content
                    i += 1
                    continue
                elif len(option_markers) == 1:
                    # Single option line (A．xxx)
                    opt_match = self.OPTION_LINE_PATTERN.match(line)
                    if opt_match:
                        letter = opt_match.group(1).upper()
                        content = opt_match.group(2).strip()
                        if letter not in current_question["options"]:
                            current_question["options"][letter] = content
                    else:
                        # Option not at start - still extract it
                        inline_matches = list(
                            self.INLINE_OPTIONS_PATTERN.finditer(line)
                        )
                        for match in inline_matches:
                            letter = match.group(1).upper()
                            content = match.group(2).strip()
                            if letter not in current_question["options"]:
                                current_question["options"][letter] = content
                    i += 1
                    continue

                # Check if line contains answer (continuation of question)
                if not current_question["answer"]:
                    answer_match = self.ANSWER_PATTERN.search(line)
                    if answer_match:
                        current_question["answer"] = answer_match.group(1)
                        # Add to question text
                        clean_line = self.ANSWER_PATTERN.sub("______", line)
                        current_question["text"] += " " + clean_line

            i += 1

        # Don't forget last question
        if current_question:
            q = self._finalize_question(current_question)
            if q:
                questions.append(q)

        return questions

    def _finalize_question(self, q_data: dict) -> Question | None:
        """Finalize and validate a question."""
        text = q_data["text"].strip()
        options = q_data["options"]
        answer = q_data["answer"]

        # Validate: need question text, answer, and at least 2 options
        if not text or not answer or len(options) < 2:
            return None

        return Question(
            id=q_data["id"],
            question=text,
            options=options,
            answer=answer,
            source={
                "parent_url": "https://www.crewcn.com/download/?Smallclassname=13",
                "exam_title": self.exam_title,
                "file_name": Path(self.file_path).name,
            },
        )

    def to_json(self, questions: list[Question]) -> list[dict]:
        """Convert questions to JSON format."""
        return [
            {
                "id": q.id,
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "hint": q.hint,
                "explanation": q.explanation,
                "source": q.source,
            }
            for q in questions
        ]
