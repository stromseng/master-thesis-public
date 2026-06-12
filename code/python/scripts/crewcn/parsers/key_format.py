"""
Parser for Dalian Maritime University 2580 Question Bank (markdown format).

The DOCX uses Word's list numbering feature where option letters and question
numbers are NOT in the raw text. We convert to markdown first using pandoc,
which renders the list numbers as text.

Format variations in markdown:
- Question numbers: "1.", "\\*4+", "\\[25\\]", "\\[ 2566 \\]"
- Options: 1 per line, 2 per line, or all 4 on one line
- KEY: may be on same line as last option or separate line
- Some KEY lines are blockquoted with ">"
- "同上" (same as above) entries inherit from previous question

Strategy:
1. Find all question number patterns to get question boundaries
2. For each question, determine if it's "同上" or a real question
3. For real questions, extract text, options, and KEY
4. For 同上 questions, copy from previous question
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    id: int
    question: str
    options: dict[str, str]
    answer: str
    hint: str | None = None
    explanation: str | None = None
    source: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "hint": self.hint,
            "explanation": self.explanation,
            "source": self.source,
        }


class KeyFormatParser:
    """Parser for markdown files converted from Dalian 2580 DOCX."""

    PARENT_URL = "https://www.crewcn.com/download/?Smallclassname=13"

    # Pattern to match question numbers (various formats)
    # Matches: "1.", "\*4+", "*4+", "\[25\]", "[25]", "\[ 2566 \]"
    QUESTION_NUM_PATTERN = re.compile(
        r"(?:^|\n|\s)"
        r"(?:"
        r"\\\*\s*(\d+)\s*\+|"  # \*4+ format (pandoc escaped) -> group 1
        r"\*\s*(\d+)\s*\+|"  # *4+ format (unescaped) -> group 2
        r"\\\[\s*(\d+)\s*\\\]|"  # \[25\] format (pandoc escaped) -> group 3
        r"\[\s*(\d+)\s*\]|"  # [25] format (unescaped) -> group 4
        r"(\d+)\.\s"  # "1. " format -> group 5
        r")",
        re.MULTILINE,
    )

    # Pattern to match KEY: answer
    KEY_PATTERN = re.compile(r"KEY:\s*([A-Da-d])", re.IGNORECASE)

    # Pattern to find options A, B, C, D
    OPTION_PATTERN = re.compile(
        r"(?<![A-Za-z])([A-D])[.．、]\s*",
        re.MULTILINE,
    )

    def __init__(self, file_path: str, exam_title: str | None = None):
        self.file_path = Path(file_path)
        self.exam_title = exam_title or self.file_path.stem
        self.full_text = ""

    def _make_source(self) -> dict[str, str]:
        """Create source metadata for questions."""
        return {
            "parent_url": self.PARENT_URL,
            "exam_title": self.exam_title,
            "file_name": self.file_path.name,
        }

    def parse(self, debug_questions: list[int] | None = None) -> list[Question]:
        """Parse the markdown file and return list of questions."""
        self.full_text = self.file_path.read_text(encoding="utf-8")

        # Find all question number matches
        q_matches = list(self.QUESTION_NUM_PATTERN.finditer(self.full_text))

        if debug_questions:
            print(f"  Found {len(q_matches)} question number patterns")

        questions: list[Question] = []

        for i, match in enumerate(q_matches):
            # Get question number from whichever group matched
            q_num = (
                match.group(1)
                or match.group(2)
                or match.group(3)
                or match.group(4)
                or match.group(5)
            )
            if not q_num:
                continue

            q_num = int(q_num)

            # Get content from this match to the next (or end)
            start = match.end()
            if i + 1 < len(q_matches):
                end = q_matches[i + 1].start()
            else:
                end = len(self.full_text)

            content = self.full_text[start:end].strip()

            # Skip "同上" (same as above) entries - these are duplicates
            if self._is_same_as_above(content):
                continue

            # Parse regular question
            question_text, options, answer, explanation = self._parse_question(content)

            # Skip if no valid answer found
            if not answer:
                continue

            # Skip if question too short
            if not question_text or len(question_text.strip()) < 3:
                continue

            question_id = len(questions) + 1
            q = Question(
                id=question_id,
                question=question_text,
                options=options,
                answer=answer,
                explanation=explanation,
                source=self._make_source(),
            )
            questions.append(q)

            if debug_questions and question_id in debug_questions:
                print(f"\n  [DEBUG] Question {question_id} (orig #{q_num}):")
                print(f"    Content: {content[:100]}...")
                print(f"    Question: {question_text[:80]}...")
                print(f"    Options: {options}")
                print(f"    Answer: {answer}")

        return questions

    def count_key_patterns(self) -> int:
        """Count how many KEY patterns are in the full text."""
        if not self.full_text:
            return 0
        return len(self.KEY_PATTERN.findall(self.full_text))

    def _is_same_as_above(self, content: str) -> bool:
        """Check if content is a '同上' (same as above) reference."""
        # Look for 同上 at the start of content (after optional whitespace/punctuation)
        cleaned = re.sub(r"^[\s:：,，.。;；>]+", "", content)
        return cleaned.startswith("同上")

    def _parse_question(
        self, content: str
    ) -> tuple[str, dict[str, str], str | None, str | None]:
        """
        Parse content into question text, options, answer, and explanation.

        Returns: (question_text, options, answer, explanation)
        """
        # Remove blockquote markers
        content = re.sub(r"^>\s*", "", content, flags=re.MULTILINE)

        # Remove HTML comments (pandoc artifacts)
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        # Remove image references
        content = re.sub(r"!\[.*?\]\(.*?\)\{[^}]*\}", "", content)
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

        # Find KEY pattern
        key_match = self.KEY_PATTERN.search(content)
        if not key_match:
            return "", {}, None, None

        answer = key_match.group(1).upper()

        # Content before KEY is question + options
        before_key = content[: key_match.start()].strip()

        # Content after KEY is explanation (until next question number or end)
        after_key = content[key_match.end() :].strip()
        explanation = self._extract_explanation(after_key)

        # Parse question text and options from before_key
        question_text, options = self._parse_content(before_key)

        return question_text, options, answer, explanation

    def _extract_explanation(self, text: str) -> str | None:
        """Extract Chinese explanation from text after KEY: X."""
        text = text.strip()
        if not text:
            return None

        # Explanation ends at a new question number pattern or double newline
        # Look for question number patterns that would indicate next question
        next_q = self.QUESTION_NUM_PATTERN.search(text)
        if next_q:
            text = text[: next_q.start()].strip()

        # Take first paragraph (up to double newline)
        parts = re.split(r"\n\s*\n", text)
        explanation = parts[0].strip() if parts else ""

        # Normalize whitespace
        explanation = re.sub(r"\s+", " ", explanation).strip()

        if len(explanation) < 2:
            return None

        return explanation

    def _parse_content(self, content: str) -> tuple[str, dict[str, str]]:
        """
        Parse content into question text and options.
        """
        # Normalize whitespace
        content = re.sub(r"\n{2,}", "\n", content)
        content = content.strip()

        # Remove leading punctuation
        content = re.sub(r"^[\s:：,，.。;；]+", "", content).strip()

        options: dict[str, str] = {}

        # Find first option marker
        first_option = self.OPTION_PATTERN.search(content)

        if not first_option:
            # No options found
            question_text = re.sub(r"\s+", " ", content).strip()
            return question_text, options

        # Question text is everything before first option
        question_text = content[: first_option.start()].strip()
        question_text = re.sub(r"\s+", " ", question_text).strip()

        # Options text is everything from first option onwards
        options_text = content[first_option.start() :]

        # Normalize options text
        options_text = re.sub(r"\s+", " ", options_text).strip()

        # Find all option markers
        markers = list(self.OPTION_PATTERN.finditer(options_text))

        for i, marker in enumerate(markers):
            letter = marker.group(1).upper()
            start = marker.end()

            if i + 1 < len(markers):
                end = markers[i + 1].start()
            else:
                end = len(options_text)

            opt_text = options_text[start:end].strip()
            if opt_text:
                options[letter] = opt_text

        return question_text, options

    def to_json(self, questions: list[Question]) -> list[dict]:
        """Convert questions to JSON-serializable format."""
        return [q.to_dict() for q in questions]
