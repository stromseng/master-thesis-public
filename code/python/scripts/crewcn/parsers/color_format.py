"""
Parser for documents using colored text (orange/red) to indicate correct answers.

Handles:
- Seamen's 62nd Second and Third Class Navigation English Test Questions

Format:
- All questions start with 【单选】
- Question text is on the same line (may continue to next line if long)
- Each option A, B, C, D has its own line
- After D option, next line is 【单选】 (next question)
- Correct answer is highlighted in orange (RGB FF6600) or red (RGB FF0000)
"""

import re

from docx.shared import RGBColor

from .base import BaseParser, Question


class ColorFormatParser(BaseParser):
    # Colors used for correct answers
    ORANGE_COLOR = RGBColor(0xFF, 0x66, 0x00)  # R: 255, G: 102, B: 0
    RED_COLOR = RGBColor(0xFF, 0x00, 0x00)  # R: 255, G: 0, B: 0 (same as 【单选】)
    ANSWER_COLORS = {ORANGE_COLOR, RED_COLOR}

    # Pattern for question marker
    QUESTION_MARKER = "【单选】"

    # Pattern for option line: starts with A./B./C./D.
    OPTION_LINE_PATTERN = re.compile(r"^([A-D])[.．、](.+)$", re.IGNORECASE)

    def parse(self) -> list[Question]:
        questions: list[Question] = []
        question_id = 0

        # Collect all paragraphs with their text and color info
        paragraphs: list[
            tuple[str, bool, list[bool]]
        ] = []  # (text, has_answer_color, char_colors)

        for para in self.doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Build character-level color map
            char_colors: list[bool] = []
            has_answer_color = False
            for run in para.runs:
                is_answer_color = run.font.color.rgb in self.ANSWER_COLORS
                if is_answer_color:
                    has_answer_color = True
                for _ in run.text:
                    char_colors.append(is_answer_color)

            paragraphs.append((text, has_answer_color, char_colors))

        # Process paragraphs
        i = 0
        while i < len(paragraphs):
            text, _, _ = paragraphs[i]

            # Look for question start
            if self.QUESTION_MARKER not in text:
                i += 1
                continue

            # Found a question - extract question text
            question_text = text.replace(self.QUESTION_MARKER, "").strip()

            # Check if there's an option on the same line (e.g., "... A.infected")
            inline_option_match = re.search(r"\s+([A-D])[.．、](\S+.*)$", question_text)
            if inline_option_match:
                # Remove inline option from question text
                question_text = question_text[: inline_option_match.start()].strip()

            i += 1

            # If question text continues on next line (before options start)
            while i < len(paragraphs):
                next_text, _, _ = paragraphs[i]
                # Check if this is an option line or next question
                if (
                    self.OPTION_LINE_PATTERN.match(next_text)
                    or self.QUESTION_MARKER in next_text
                ):
                    break
                # Check if it has multiple options (like "B.xxx C.yyy")
                if re.search(r"[A-D][.．、]", next_text):
                    break
                # It's continuation of question text
                question_text += " " + next_text
                i += 1

            # Now collect options (each on its own line)
            options: dict[str, str] = {}
            answer: str | None = None

            # If we had inline option, add it first
            if inline_option_match:
                letter = inline_option_match.group(1).upper()
                content = inline_option_match.group(2).strip()
                options[letter] = content

            # Collect remaining options
            while i < len(paragraphs) and len(options) < 4:
                opt_text, has_answer_color, char_colors = paragraphs[i]

                # Stop if we hit next question
                if self.QUESTION_MARKER in opt_text:
                    break

                # First check if line has multiple options (e.g., "B.xxx C.yyy D.zzz")
                option_pattern = re.compile(r"[A-D][.．、]", re.IGNORECASE)
                option_matches = list(option_pattern.finditer(opt_text))

                if len(option_matches) > 1:
                    # Multiple options on this line - use multi-option handler
                    multi_opts = self._extract_multi_options_with_color(
                        opt_text, char_colors
                    )
                    if multi_opts:
                        for letter, content, is_answer in multi_opts:
                            options[letter] = content
                            if is_answer:
                                answer = letter
                    i += 1
                elif len(option_matches) == 1:
                    # Single option - check if it starts at beginning of line
                    opt_match = self.OPTION_LINE_PATTERN.match(opt_text)
                    if opt_match:
                        letter = opt_match.group(1).upper()
                        content = opt_match.group(2).strip()
                        options[letter] = content

                        if has_answer_color:
                            answer = letter
                    else:
                        # Option not at start - use multi-option handler anyway
                        multi_opts = self._extract_multi_options_with_color(
                            opt_text, char_colors
                        )
                        if multi_opts:
                            for letter, content, is_answer in multi_opts:
                                options[letter] = content
                                if is_answer:
                                    answer = letter
                    i += 1
                else:
                    # No options found, skip
                    i += 1

            # Clean up question text
            question_text = re.sub(r"\[\d+\]\s*$", "", question_text).strip()
            question_text = question_text.replace("\t", " ").strip()

            # Save question if valid
            if question_text and options:
                question_id += 1
                questions.append(
                    Question(
                        id=question_id,
                        question=question_text,
                        options=options,
                        answer=answer or "",
                        source=self._make_source(),
                    )
                )

        return questions

    def count_colored_answers(self) -> int:
        """Count how many colored option letters (A./B./C./D.) are in the document (orange or red)."""
        count = 0
        option_pattern = re.compile(r"[A-D][.．、]", re.IGNORECASE)

        for para in self.doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Build character-level color map
            char_colors: list[bool] = []
            for run in para.runs:
                is_answer_color = run.font.color.rgb in self.ANSWER_COLORS
                for _ in run.text:
                    char_colors.append(is_answer_color)

            # Find option patterns and check if they're colored
            for match in option_pattern.finditer(text):
                letter_pos = match.start()
                if letter_pos < len(char_colors) and char_colors[letter_pos]:
                    count += 1

        return count

    def _extract_multi_options_with_color(
        self, text: str, char_colors: list[bool]
    ) -> list[tuple[str, str, bool]] | None:
        """Extract multiple options from a single line with character-level color detection."""
        # Find all option patterns
        pattern = re.compile(r"([A-D])[.．、]", re.IGNORECASE)
        matches = list(pattern.finditer(text))

        if not matches:
            return None

        results = []
        for i, match in enumerate(matches):
            letter = match.group(1).upper()
            start = match.end()

            # End is either next option or end of string
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            content = text[start:end].strip()
            if content:
                # Check if this option's letter is orange
                letter_pos = match.start()
                is_answer = letter_pos < len(char_colors) and char_colors[letter_pos]
                results.append((letter, content, is_answer))

        return results if results else None
