#!/usr/bin/env python3
"""Extract random questions from MCQ evaluation files to understand required book topics."""

import json
import random
from pathlib import Path
from collections import defaultdict

MCQ_DIR = Path(__file__).parent.parent / "data" / "evals" / "mcq"

def load_questions(file_path: Path) -> list[dict]:
    """Load questions from JSON file, handling different schemas."""
    with open(file_path) as f:
        data = json.load(f)

    # Handle SHITITONG format (has metadata wrapper)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data

def format_question(q: dict, source_file: str) -> str:
    """Format a question for display."""
    lines = [f"[{source_file}]"]

    # Get question text
    question_text = q.get("question") or q.get("question_text", "")
    lines.append(f"Q: {question_text}")

    # Get options
    options = q.get("options") or q.get("choices", [])
    if isinstance(options, dict):
        for letter, text in options.items():
            lines.append(f"  {letter}: {text}")
    elif isinstance(options, list):
        for opt in options:
            letter = opt.get("choice_letter", "?")
            text = opt.get("choice_text", "")
            lines.append(f"  {letter}: {text}")

    # Get answer
    answer = q.get("answer") or q.get("correct_answer", "")
    if isinstance(answer, list):
        answer = ", ".join(answer)
    lines.append(f"Answer: {answer}")

    # Get source info if available
    source = q.get("source", {})
    if isinstance(source, dict):
        exam_title = source.get("exam_title", "")
        if exam_title:
            lines.append(f"Exam: {exam_title}")

    return "\n".join(lines)

def extract_topics(questions: list[dict]) -> dict[str, int]:
    """Extract topic/exam titles and count questions per topic."""
    topics = defaultdict(int)
    for q in questions:
        source = q.get("source", {})
        if isinstance(source, dict):
            exam_title = source.get("exam_title", "Unknown")
            topics[exam_title] += 1
    return dict(topics)

def main():
    files = list(MCQ_DIR.glob("*.json"))

    print("=" * 80)
    print("MCQ FILE STATISTICS")
    print("=" * 80)

    all_questions = {}
    all_topics = {}

    for file_path in sorted(files):
        questions = load_questions(file_path)
        all_questions[file_path.stem] = questions
        topics = extract_topics(questions)
        all_topics[file_path.stem] = topics

        print(f"\n{file_path.name}:")
        print(f"  Total questions: {len(questions)}")
        if topics:
            print("  Topics/Exams:")
            for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:10]:
                print(f"    - {topic}: {count} questions")

    print("\n" + "=" * 80)
    print("RANDOM SAMPLE QUESTIONS (5 per file)")
    print("=" * 80)

    for file_name, questions in all_questions.items():
        print(f"\n{'─' * 40}")
        print(f"FROM: {file_name}")
        print("─" * 40)

        sample_size = min(5, len(questions))
        sample = random.sample(questions, sample_size)

        for i, q in enumerate(sample, 1):
            print(f"\n[Sample {i}]")
            print(format_question(q, file_name))

    print("\n" + "=" * 80)
    print("UNIQUE TOPICS/EXAMS ACROSS ALL FILES")
    print("=" * 80)

    combined_topics = defaultdict(int)
    for file_topics in all_topics.values():
        for topic, count in file_topics.items():
            combined_topics[topic] += count

    print("\nAll exam/topic sources:")
    for topic, count in sorted(combined_topics.items(), key=lambda x: -x[1]):
        if topic and topic != "Unknown":
            print(f"  [{count:5d}] {topic}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
