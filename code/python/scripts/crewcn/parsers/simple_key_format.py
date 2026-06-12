"""
Parser for 3300 document where options are on separate lines without prefixes.

Handles:
- Latest Sailing English Question Bank for Seamen's Exam 3300

Format:
[802]The Vessel is 	 with CO2 system...
fitted
contained
held
made
KEY:	b

Key observations from debug analysis:
- Only ~135 KEY lines in entire document
- Questions often don't have [number] prefix
- New question starts immediately after KEY (same line or next line)
- Last option and KEY can be on same line: "option text KEY: B"
- New question can be on same line after KEY explanation
"""

import re

from .base import BaseParser, Question


class SimpleKeyFormatParser(BaseParser):
    # Pattern to match question start: [number]
    QUESTION_NUM_PATTERN = re.compile(r"\[(\d+)\]")

    # Pattern to match KEY: answer (with optional text before/after)
    KEY_PATTERN = re.compile(r"KEY:\s*([A-Da-d])", re.IGNORECASE)

    # Letters for options
    OPTION_LETTERS = ["A", "B", "C", "D"]

    def parse(self) -> list[Question]:
        questions: list[Question] = []
        paragraphs = self.get_all_text()

        # Join all text for better pattern matching
        self.full_text = "\n".join(paragraphs)

        # Iterate line by line, accumulating content between KEYs
        current_content: list[str] = []
        question_id = 0

        for text in paragraphs:
            key_match = self.KEY_PATTERN.search(text)

            if key_match:
                # Found KEY - process accumulated content
                answer = key_match.group(1).upper()

                # Include text before KEY in current content
                before_key = text[: key_match.start()].strip()
                if before_key:
                    current_content.append(before_key)

                # Try to extract question from accumulated content
                if current_content:
                    q = self._extract_question_from_content(
                        current_content, answer, question_id + 1
                    )
                    if q:
                        question_id += 1
                        questions.append(q)

                # Reset for next question
                current_content = []

                # Check for new question starting after KEY on same line
                after_key = text[key_match.end() :].strip()
                # Remove Chinese explanation if present (non-ASCII chars before [number])
                new_q_match = self.QUESTION_NUM_PATTERN.search(after_key)
                if new_q_match:
                    # New question starts on this line
                    q_text = after_key[new_q_match.start() :]
                    current_content.append(q_text)
                elif self._looks_like_question_start(after_key):
                    current_content.append(after_key)
            else:
                # No KEY - accumulate content
                if text.strip():
                    current_content.append(text.strip())

        return questions

    def count_key_patterns(self) -> int:
        """Count how many KEY patterns are in the full text."""
        if not hasattr(self, "full_text"):
            return 0
        return len(self.KEY_PATTERN.findall(self.full_text))

    def _extract_question_from_content(
        self, lines: list[str], answer: str, qid: int
    ) -> Question | None:
        """Extract a question from accumulated lines."""
        if not lines:
            return None

        # Find where question text ends and options begin
        # Question usually has [number] prefix or is longer/contains blanks
        question_text = ""
        option_lines: list[str] = []
        found_question = False

        for i, line in enumerate(lines):
            # Check for [number] pattern
            num_match = self.QUESTION_NUM_PATTERN.search(line)
            if num_match:
                # This line contains question start
                # Remove [number] prefix
                q_start = line[num_match.end() :].strip()
                if q_start:
                    question_text = q_start
                    found_question = True
                continue

            if not found_question:
                # First substantial line might be the question
                if self._looks_like_question_start(line):
                    question_text = line
                    found_question = True
                    continue
                elif len(lines) > 4:
                    # If we have many lines, first one is probably the question
                    question_text = line
                    found_question = True
                    continue

            # After question, remaining lines are options
            if found_question and line:
                option_lines.append(line)

        # If still no question found, use first line
        if not question_text and lines:
            question_text = lines[0]
            option_lines = lines[1:]

        # Build options from option_lines
        options = self._build_options(option_lines)

        # Validate
        if not question_text or len(question_text) < 5:
            return None
        if len(options) < 2:
            return None
        if answer not in options:
            return None

        return Question(
            id=qid,
            question=question_text.replace("\t", " ").strip(),
            options=options,
            answer=answer,
            source=self._make_source(),
        )

    def _build_options(self, lines: list[str]) -> dict[str, str]:
        """Build options dict from lines."""
        options = {}

        # Filter out empty/header lines
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line or self._is_header_line(line):
                continue

            # Expand tab-separated values
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                clean_lines.extend(parts)
            else:
                clean_lines.append(line)

        # Remove option letter prefixes if present
        processed = []
        for line in clean_lines:
            # Remove A. B. C. D. prefix
            clean = re.sub(r"^[A-D][.．、]\s*", "", line)
            if clean:
                processed.append(clean)

        # Take up to 4 options
        for idx, line in enumerate(processed[:4]):
            if idx < len(self.OPTION_LETTERS):
                options[self.OPTION_LETTERS[idx]] = line

        return options

    def _is_header_line(self, text: str) -> bool:
        """Check if line is a header/title to skip."""
        if len(text) < 2:
            return True
        if text.startswith("第") and "章" in text:
            return True
        if text.startswith("http"):
            return True
        return False

    def _looks_like_question_start(self, text: str) -> bool:
        """Check if text looks like a question start."""
        if not text or len(text) < 10:
            return False
        # Has a blank to fill
        if "\t" in text or "____" in text or "	" in text:
            return True
        # Ends with question mark
        if "?" in text:
            return True
        # Contains typical question words at start
        if any(
            text.lower().startswith(w)
            for w in [
                "the ",
                "a ",
                "an ",
                "what",
                "which",
                "how",
                "when",
                "where",
                "why",
            ]
        ):
            return True
        # Long enough and looks like a sentence
        if len(text) > 40 and text[0].isupper():
            return True
        return False
