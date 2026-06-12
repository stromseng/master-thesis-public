"""
Parser for documents with answer embedded in question line.

Handles:
- Latest Sailing English (Chinese and English versions) - 2012

Formats:
1. Answer after question number: 0001.  D	are published for...
2. Answer embedded in text: 0036. Attention is \tD  the advice...
"""

import re

from .base import BaseParser, Question


class EmbeddedAnswerParser(BaseParser):
    # Pattern 1: 4 digits, period, spaces, answer letter at START, then question text
    QUESTION_PATTERN_START = re.compile(r"^(\d{4})\.\s+([A-D])\s+(.+)", re.IGNORECASE)

    # Pattern 2: Generic question line (4 digits, period, then text)
    QUESTION_PATTERN_GENERIC = re.compile(r"^(\d{4})\.\s*(.+)", re.IGNORECASE)

    # Patterns to find embedded answer in text:
    # - tab + letter + tab/space: \tD\t or \tD
    # - spaces + letter + space + period: "  A ．"
    # - letter at end with tab/period: "D\t．" or " D ．"
    EMBEDDED_ANSWER_PATTERNS = [
        re.compile(r"\t([A-D])[\t\s]", re.IGNORECASE),  # tab+letter+tab/space
        re.compile(r"\s{2,}([A-D])\s*[.．、\[]", re.IGNORECASE),  # spaces+letter+punct
        re.compile(r"\s([A-D])\t[.．]", re.IGNORECASE),  # space+letter+tab+period
        re.compile(
            r"\s([A-D])\s*[.．]\s*(?:\[\d+\])?\s*$", re.IGNORECASE
        ),  # letter at end
    ]

    # Pattern for options (various formats)
    OPTION_PATTERN = re.compile(
        r"([A-D])[.．、]\s*(.+?)(?=[A-D][.．、]|$)", re.IGNORECASE
    )

    def parse(self) -> list[Question]:
        questions: list[Question] = []
        paragraphs = self.get_all_text()

        current_question: dict | None = None
        current_options: dict[str, str] = {}
        question_id = 0

        i = 0
        while i < len(paragraphs):
            text = paragraphs[i]

            # Try pattern 1: answer right after question number
            q_match = self.QUESTION_PATTERN_START.match(text)
            if q_match:
                # Save previous question if exists
                if current_question and current_options:
                    question_id += 1
                    questions.append(
                        Question(
                            id=question_id,
                            question=current_question["text"],
                            options=current_options,
                            answer=current_question["answer"],
                            source=self._make_source(),
                        )
                    )

                question_num = q_match.group(1)
                answer = q_match.group(2).upper()
                question_text = q_match.group(3).strip()

                # Clean up question text - remove trailing [number] references
                question_text = re.sub(r"\s*\[\d+\]\s*$", "", question_text)

                current_question = {
                    "num": question_num,
                    "answer": answer,
                    "text": question_text,
                }
                current_options = {}

                i += 1
                continue

            # Try pattern 2: question number then embedded answer in text
            q_match2 = self.QUESTION_PATTERN_GENERIC.match(text)
            if q_match2:
                question_text = q_match2.group(2)

                # Try all embedded answer patterns
                answer = None
                answer_match = None
                matched_pattern = None
                for pattern in self.EMBEDDED_ANSWER_PATTERNS:
                    answer_match = pattern.search(question_text)
                    if answer_match:
                        answer = answer_match.group(1).upper()
                        matched_pattern = pattern
                        break

                if answer:
                    # Save previous question if exists
                    if current_question and current_options:
                        question_id += 1
                        questions.append(
                            Question(
                                id=question_id,
                                question=current_question["text"],
                                options=current_options,
                                answer=current_question["answer"],
                                source=self._make_source(),
                            )
                        )

                    question_num = q_match2.group(1)

                    # Remove the embedded answer from question text
                    assert matched_pattern is not None
                    question_text = matched_pattern.sub(
                        " ______ ", question_text
                    ).strip()

                    # Clean up question text
                    question_text = re.sub(r"\s*\[\d+\]\s*$", "", question_text)
                    question_text = re.sub(
                        r"\s+", " ", question_text
                    )  # Normalize spaces
                    question_text = question_text.rstrip(
                        " ．.、"
                    )  # Remove trailing punctuation

                    current_question = {
                        "num": question_num,
                        "answer": answer,
                        "text": question_text,
                    }
                    current_options = {}

                    i += 1
                    continue

            # If we have a current question, look for options
            if current_question:
                options = self._extract_options(text)
                if options:
                    current_options.update(options)

            i += 1

        # Don't forget the last question
        if current_question and current_options:
            question_id += 1
            questions.append(
                Question(
                    id=question_id,
                    question=current_question["text"],
                    options=current_options,
                    answer=current_question["answer"],
                    source=self._make_source(),
                )
            )

        return questions

    def _extract_options(self, text: str) -> dict[str, str]:
        """Extract options from text."""
        options = {}

        # Try to find options with A．, B．, etc. format
        matches = self.OPTION_PATTERN.findall(text)
        for letter, content in matches:
            # Clean up content - remove Chinese translations after English
            content = content.strip()
            options[letter.upper()] = content

        return options
