# MCQ Dataset Conversion to EvalQuestion Schema

Convert raw MCQ JSON files from `data/evals/mcq/` into the canonical `QuestionGroup` schema defined in `code/ts/evals/question-schema.ts`.

## Status

| Source File | Questions | Converted | Script | Output |
|-------------|-----------|-----------|--------|--------|
| `CREWCN_EXAM_QUESTIONS.json` | 3,061 | Yes | `examscrapers/crewcn/convert_crewcn.py` | `data/evals/crewcn/` |
| `SHITITONG_ENGLISH_DEDUPED.json` | 9,640 deduped / 9,545 converted | **Yes** | `examscrapers/shititong/convert_shititong.py --target english-deduped` | `data/evals/shititong/` (split: 8,519 text / 1,026 vision) |
| `SHITITONG_CHINESE_DEDUPED.json` | 58,655 deduped / 53,802 converted | **Yes** | `examscrapers/shititong/convert_shititong.py --target chinese-deduped` | `data/evals/shititong/` (split: 53,545 text / 257 vision) |
| `UK_THEORY_TEST.json` | 814 | Yes | `examscrapers/pei2024application/convert_uk_theory_test.py` | `data/evals/pei2024application/` |
| `ZH_THEORY_TEST.json` | 786 | Yes | `examscrapers/pei2024application/convert_zh_theory_test.py` | `data/evals/pei2024application/` |

**Note:** Only deduped Shititong datasets are converted. Each is split into two files:
- `*_text.json`: Questions without `<img>` tags (for text-only models)
- `*_vision.json`: Questions with `<img>` tags (for vision models)

Non-deduped source files (`SHITITONG_ENGLISH_QUESTIONS.json`, `SHITITONG_CHINESE_QUESTIONS.json`) are not converted. Question counts reflect schema-compatible entries only: non-MCQ types, rows with missing or unusable answers, answer-option mismatches, and invalid local image references are skipped.

## Quick Start

```bash
cd code/python

# CrewCN
uv run python -m examscrapers.crewcn.convert_crewcn

# Shititong (deduped only, splits into text/vision)
uv run python -m examscrapers.shititong.convert_shititong --target english-deduped
uv run python -m examscrapers.shititong.convert_shititong --target chinese-deduped
uv run python -m examscrapers.shititong.convert_shititong --target all

# UK/ZH Theory Tests
uv run python -m examscrapers.pei2024application.convert_uk_theory_test
uv run python -m examscrapers.pei2024application.convert_zh_theory_test
```

## Target Schema

Every converted file must be a JSON array of `QuestionGroup` objects:

```json
[
  {
    "id": "crewcn-sailing-english",
    "metadata": { "dataset": "...", "questionCount": 100, "convertedAtUtc": "..." },
    "source": { "provider": "crewcn", "sourceFile": "..." },
    "questions": [
      {
        "id": "crewcn-0001",
        "questionText": "What is the meaning of...",
        "metadata": {},
        "options": [
          { "id": "A", "text": "Option A" },
          { "id": "B", "text": "Option B" }
        ],
        "correctOptionIds": ["A"]
      }
    ]
  }
]
```

See `code/ts/evals/question-schema.ts` for the canonical Effect Schema definition.
See `code/python/src/generated/eval_question_models.py` for the generated Pydantic models.

## Source Format Differences

### CrewCN (`CREWCN_EXAM_QUESTIONS.json`)

Flat array. Options are `{ "A": "text", "B": "text", ... }` dict instead of array.

```json
{
  "id": 1,
  "question": "______ are published for the correction of...",
  "options": { "A": "Admiralty Sailing Directions", "B": "...", "C": "...", "D": "..." },
  "answer": "D",
  "hint": null,
  "explanation": null,
  "source": { "parent_url": "...", "exam_title": "...", "file_name": "..." }
}
```

Conversion groups questions by `source.file_name` to create one `QuestionGroup` per source document.

### Shititong (`SHITITONG_*_DEDUPED.json`)

Wrapper object with `questions` array. Same option dict format as CrewCN. Includes question type field.

```json
{
  "exported_at": "...",
  "source": "shititong.cn",
  "language": "english",
  "statistics": { "total_questions": 21144, "by_type": { "单选题": 19797, ... } },
  "questions": [
    {
      "id": 1,
      "question": "The fresh water jacket cooling system forms a _____ circuit.",
      "options": { "A": "open", "B": "closed", "C": "opening", "D": "closing" },
      "answer": "B",
      "type": "单选题",
      "hint": null,
      "explanation": null,
      "difficulty": null,
      "category_id": "241947",
      "original_id": "0005decd-...",
      "source": { "exam_url": "...", "exam_title": "...", "found_by_keywords": [...] }
    }
  ]
}
```

Only MCQ types are converted: `单选题` / `single_choice`, `多选题` / `multiple_choice`, `判断题` / `true_false`.
Fill-in-blank (`填空题`) and short-answer (`简答题`) are skipped.
Conversion groups questions by `source.exam_title`.

**Vision/Text Split:** Each dataset is split into two outputs:
- Text file: Questions without `<img>` tags (for text-only models)
- Vision file: Questions with `<img>` tags (for vision-capable models)

## Notes

- The canonical schema is defined in TypeScript (`code/ts/evals/question-schema.ts`)
- Pydantic models are auto-generated from it (see `docs/EVAL_QUESTION_PYDANTIC_CODEGEN.md`)
- All converters follow the same pattern as `convert_uk_theory_test.py`
- `dedup_report.json` in `data/evals/mcq/` is not a question file
