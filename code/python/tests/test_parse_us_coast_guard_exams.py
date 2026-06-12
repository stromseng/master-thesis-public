from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examscrapers.us_coast_guard.deterministic_markdown_parser import (
    extract_illustration_mapping,
    extract_questions,
    parse_exam_markdown,
)


def _exam_meta() -> dict[str, str]:
    return {
        "exam_file": "q999_demo.pdf",
        "exam_stem": "q999_demo",
        "exam_code": "q999",
        "exam_title": "Q999 Demo",
    }


def test_extract_questions_bullet_and_unbulleted_blocks() -> None:
    markdown = """
- **1.** INLAND ONLY Which signal is correct?
  - A. One short blast
  - B. Two short blasts
  - C. Three short blasts
  - D. Four short blasts
Correct answer: B

**2.** INTERNATIONAL ONLY What does vessel A show?
- A. Masthead light
- B. Stern light
- C. Sidelight
- D. All-around light
Correct answer: D
"""
    drafts = extract_questions(markdown)

    assert [draft.question_number for draft in drafts] == [1, 2]
    assert drafts[0].correct_answer == "B"
    assert drafts[1].correct_answer == "D"
    assert drafts[0].options["A"] == "One short blast"
    assert drafts[1].options["D"] == "All-around light"


def test_extract_questions_collapsed_inline_multi_question() -> None:
    markdown = (
        "Q test **6.** Which light is shown in illustration D041RR? "
        "A. Red B. Green C. White D. Yellow Correct answer: C "
        "**7.** Which sound signal applies? A. One short B. Two short "
        "C. Three short D. Four short Correct answer: B"
    )

    drafts = extract_questions(markdown)
    assert [draft.question_number for draft in drafts] == [6, 7]
    assert drafts[0].illustration_ids == ["D041RR"]
    assert drafts[1].correct_answer == "B"


def test_extract_questions_table_rows() -> None:
    markdown = """
| 1. | | Which component is shown in the illustration? Illustration SG-0004 |
|    | A.<br>B.<br>C.<br>D. | Valve<br>Pump<br>Compressor<br>Turbine |
|    | | Correct answer: D |
"""

    drafts = extract_questions(markdown)
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.question_number == 1
    assert draft.correct_answer == "D"
    assert draft.options == {
        "A": "Valve",
        "B": "Pump",
        "C": "Compressor",
        "D": "Turbine",
    }
    assert draft.illustration_ids == ["SG-0004"]


def test_extract_illustration_mapping_from_heading_and_bold_lines() -> None:
    markdown = """
### **D025DG**
![](_page_10_Figure_1.jpeg)
![](_page_10_Figure_2.jpeg)

**EL-0115 Watertight Door Controller**
![](_page_11_Figure_4.jpeg)

## D041RR
<div style="text-align: center;"><img src="images/_page_12_Figure_6.jpeg" alt="Image" /></div>

D084RR
<img src="imgs/_page_13_Figure_1.jpeg" />
"""

    mapping = extract_illustration_mapping(markdown)
    assert mapping == {
        "D025DG": ["_page_10_Figure_1.jpeg", "_page_10_Figure_2.jpeg"],
        "EL-0115": ["_page_11_Figure_4.jpeg"],
        "D041RR": ["_page_12_Figure_6.jpeg"],
        "D084RR": ["_page_13_Figure_1.jpeg"],
    }


def test_parse_exam_markdown_attaches_all_existing_images(tmp_path: Path) -> None:
    exam_stem = "q999_demo"
    images_root = tmp_path / "images"
    exam_images_dir = images_root / exam_stem
    exam_images_dir.mkdir(parents=True, exist_ok=True)

    (exam_images_dir / "_page_10_Figure_1.jpeg").write_bytes(b"img-1")
    (exam_images_dir / "_page_10_Figure_2.jpeg").write_bytes(b"img-2")

    markdown = """
- **1.** Which fitting is shown in illustration D025DG below?
  - A. A
  - B. B
  - C. C
  - D. D
Correct answer: A

### **D025DG**
![](_page_10_Figure_1.jpeg)
![](_page_10_Figure_2.jpeg)
"""

    result = parse_exam_markdown(
        markdown_text=markdown,
        exam_stem=exam_stem,
        exam_meta=_exam_meta(),
        images_dir=images_root,
    )

    assert len(result.questions) == 1
    question = result.questions[0]
    assert question.images is not None
    assert [image.uri for image in question.images] == [
        "_page_10_Figure_1.jpeg",
        "_page_10_Figure_2.jpeg",
    ]
    assert result.report["illustration_ids_detected"] == 1
    assert result.report["illustration_ids_resolved"] == 1
    assert result.report["questions_with_images"] == 1


def test_parse_exam_markdown_reports_missing_image_mapping(tmp_path: Path) -> None:
    exam_stem = "q999_demo"
    images_root = tmp_path / "images"
    (images_root / exam_stem).mkdir(parents=True, exist_ok=True)

    markdown = """
- **1.** Which fitting is shown in illustration D025DG below?
  - A. A
  - B. B
  - C. C
  - D. D
Correct answer: A

### **D025DG**
![](_page_10_Figure_1.jpeg)
"""

    result = parse_exam_markdown(
        markdown_text=markdown,
        exam_stem=exam_stem,
        exam_meta=_exam_meta(),
        images_dir=images_root,
    )

    assert len(result.questions) == 1
    assert result.questions[0].images is None
    assert result.report["illustration_ids_detected"] == 1
    assert result.report["illustration_ids_resolved"] == 0
    assert result.report["questions_with_images"] == 0


def test_parse_exam_markdown_clarifies_inland_to_us_inland(tmp_path: Path) -> None:
    exam_stem = "q999_demo"
    images_root = tmp_path / "images"
    (images_root / exam_stem).mkdir(parents=True, exist_ok=True)

    markdown = """
- **1.** INLAND ONLY When operating in INLAND waters near US INLAND routes, what action is required?
  - A. Maintain course
  - B. Alter course
  - C. Reduce speed
  - D. Sound danger signal
Correct answer: C
"""

    result = parse_exam_markdown(
        markdown_text=markdown,
        exam_stem=exam_stem,
        exam_meta=_exam_meta(),
        images_dir=images_root,
    )

    assert len(result.questions) == 1
    question = result.questions[0]
    assert question.questionText.startswith("US INLAND ONLY ")
    assert " in US INLAND waters " in question.questionText
    assert "US US INLAND" not in question.questionText
    assert question.metadata["category"] == "inland"
