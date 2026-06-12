#!/usr/bin/env python3
"""
Extract maritime exam questions from DOCX files.

Usage:
    uv run python scripts/crewcn/extract_questions.py [--debug]
"""

import argparse
import json
from pathlib import Path

from parsers import (  # type: ignore[unresolved-import]  # implicit relative, resolved at runtime
    ColorFormatParser,
    KeyFormatParser,
    Question,
    SimpleKeyFormatParser,
)
from parsers.pdf_column_format import PDFColumnFormatParser  # type: ignore[unresolved-import]


def validate_questions(
    questions: list[Question], debug: bool = False
) -> list[Question]:
    """Validate and filter questions."""
    valid = []
    invalid_count = 0

    for q in questions:
        # Must have question text
        if not q.question or len(q.question.strip()) < 5:
            if debug:
                print(
                    f"  Skipping: empty or short question: {q.question[:50] if q.question else 'None'}"
                )
            invalid_count += 1
            continue

        # Must have at least 2 options
        if len(q.options) < 2:
            if debug:
                print(
                    f"  Skipping: too few options ({len(q.options)}): {q.question[:50]}"
                )
            invalid_count += 1
            continue

        # Must have an answer
        if not q.answer:
            if debug:
                print(f"  Skipping: no answer: {q.question[:50]}")
            invalid_count += 1
            continue

        # Answer must be valid option
        if q.answer not in q.options:
            if debug:
                print(
                    f"  Skipping: answer {q.answer} not in options {list(q.options.keys())}: {q.question[:50]}"
                )
            invalid_count += 1
            continue

        valid.append(q)

    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} invalid questions")

    return valid


def save_questions(questions: list[Question], output_file: Path, parser) -> None:
    """Save questions to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            parser.to_json(questions),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract questions from CrewCN DOCX files"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    files_dir = script_dir / "files"
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    all_questions: list[Question] = []
    stats: dict[str, dict] = {}

    print("Extracting questions from DOCX and PDF files...\n")

    # =========================================================================
    # FILE 1: Latest Sailing English (PDF) - 3-column PDF layout
    # =========================================================================
    pdf_file = (
        files_dir / "Latest Sailing English (Chinese and English versions) - 2012.pdf"
    )

    if pdf_file.exists():
        print(f"Processing: {pdf_file.name}")
        print("  Using: PDFColumnFormatParser")

        try:
            exam_title = "Latest Sailing English (Chinese/English)"
            doc_parser = PDFColumnFormatParser(str(pdf_file), exam_title)
            questions = doc_parser.parse()
            print(
                f"  Answer patterns found in text: {doc_parser.count_answer_patterns()}"
            )
            print(f"  Extracted: {len(questions)} questions")

            valid_questions = validate_questions(questions, args.debug)
            print(f"  Valid: {len(valid_questions)} questions")

            output_file = output_dir / f"{pdf_file.stem}.json"
            save_questions(valid_questions, output_file, doc_parser)

            all_questions.extend(valid_questions)
            stats[pdf_file.name] = {
                "extracted": len(questions),
                "valid": len(valid_questions),
                "parser": "PDFColumnFormatParser",
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

        print()
    else:
        print(f"SKIPPED: {pdf_file.name} (file not found)")
        print()

    # =========================================================================
    # FILE 2: Dalian Maritime University 2580 (Markdown) - KEY: X format
    # =========================================================================
    # DOCX uses Word's list numbering (not in raw text). Convert to markdown
    # with pandoc to preserve the numbering.
    dalian_docx = (
        files_dir
        / "Dalian Maritime University English 2580 Question Bank English Comparative Study Full Version - 2012.docx"
    )
    dalian_file = files_dir / "dalian_2580.md"

    # Auto-convert DOCX to markdown if needed
    if not dalian_file.exists() and dalian_docx.exists():
        import subprocess

        print(f"Converting {dalian_docx.name} to markdown...")
        try:
            subprocess.run(
                ["pandoc", str(dalian_docx), "-o", str(dalian_file)],
                check=True,
                capture_output=True,
            )
            print("  Conversion successful")
        except FileNotFoundError:
            print("  ERROR: pandoc not installed. Install with: brew install pandoc")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: pandoc conversion failed: {e.stderr.decode()}")

    if dalian_file.exists():
        print(f"Processing: {dalian_file.name}")
        print("  Using: KeyFormatParser (markdown)")

        try:
            exam_title = "Dalian Maritime University English 2580"
            doc_parser = KeyFormatParser(str(dalian_file), exam_title)
            # Debug specific questions to check parsing
            debug_qs = [1, 2, 3, 4, 5] if args.debug else []
            questions = doc_parser.parse(debug_questions=debug_qs)
            print(f"  KEY patterns found in text: {doc_parser.count_key_patterns()}")
            print(f"  Extracted: {len(questions)} questions")

            valid_questions = validate_questions(questions, args.debug)
            print(f"  Valid: {len(valid_questions)} questions")

            output_file = (
                output_dir
                / "Dalian Maritime University English 2580 Question Bank English Comparative Study Full Version - 2012.json"
            )
            save_questions(valid_questions, output_file, doc_parser)

            all_questions.extend(valid_questions)
            stats[dalian_file.name] = {
                "extracted": len(questions),
                "valid": len(valid_questions),
                "parser": "KeyFormatParser",
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

        print()
    else:
        print(f"SKIPPED: {dalian_file.name} (file not found)")
        print()

    # =========================================================================
    # FILE 3: Latest Sailing English 3300 (DOCX) - Simple KEY: X format
    # =========================================================================
    sailing_3300_file = (
        files_dir / "Latest Sailing English Question Bank for Seamen's Exam 3300.docx"
    )

    if sailing_3300_file.exists():
        print(f"Processing: {sailing_3300_file.name}")
        print("  Using: SimpleKeyFormatParser")

        try:
            exam_title = "Latest Sailing English 3300"
            doc_parser = SimpleKeyFormatParser(str(sailing_3300_file), exam_title)
            questions = doc_parser.parse()
            print(f"  KEY patterns found in text: {doc_parser.count_key_patterns()}")
            print(f"  Extracted: {len(questions)} questions")

            valid_questions = validate_questions(questions, args.debug)
            print(f"  Valid: {len(valid_questions)} questions")

            output_file = output_dir / f"{sailing_3300_file.stem}.json"
            save_questions(valid_questions, output_file, doc_parser)

            all_questions.extend(valid_questions)
            stats[sailing_3300_file.name] = {
                "extracted": len(questions),
                "valid": len(valid_questions),
                "parser": "SimpleKeyFormatParser",
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

        print()
    else:
        print(f"SKIPPED: {sailing_3300_file.name} (file not found)")
        print()

    # =========================================================================
    # FILE 4: Seamen's 62nd (DOCX) - Orange text = answer
    # =========================================================================
    seamen_62nd_file = (
        files_dir
        / "Seamen's 62nd Second and Third Class Navigation English Test Questions.docx"
    )

    if seamen_62nd_file.exists():
        print(f"Processing: {seamen_62nd_file.name}")
        print("  Using: ColorFormatParser")

        try:
            exam_title = "Seamen's 62nd Navigation English"
            doc_parser = ColorFormatParser(str(seamen_62nd_file), exam_title)
            questions = doc_parser.parse()
            print(
                f"  Colored answer patterns found: {doc_parser.count_colored_answers()}"
            )
            print(f"  Extracted: {len(questions)} questions")

            valid_questions = validate_questions(questions, args.debug)
            print(f"  Valid: {len(valid_questions)} questions")

            output_file = output_dir / f"{seamen_62nd_file.stem}.json"
            save_questions(valid_questions, output_file, doc_parser)

            all_questions.extend(valid_questions)
            stats[seamen_62nd_file.name] = {
                "extracted": len(questions),
                "valid": len(valid_questions),
                "parser": "ColorFormatParser",
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()

        print()
    else:
        print(f"SKIPPED: {seamen_62nd_file.name} (file not found)")
        print()

    # =========================================================================
    # MERGE ALL QUESTIONS
    # =========================================================================

    # Reassign IDs sequentially
    for i, q in enumerate(all_questions, 1):
        q.id = i

    # Save merged output
    merged_file = output_dir / "all-questions.json"
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(
            [q.to_dict() for q in all_questions],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 50)
    print(f"Total questions extracted: {len(all_questions)}")
    print(f"Saved to: {merged_file}")
    print("\nPer-file statistics:")
    for filename, stat in stats.items():
        print(f"  {filename[:50]}...")
        print(
            f"    Parser: {stat['parser']}, Extracted: {stat['extracted']}, Valid: {stat['valid']}"
        )


if __name__ == "__main__":
    main()
