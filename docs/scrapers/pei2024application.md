# pei2024application

Converts Pei et al. (2024) maritime MCQ datasets from [GitHub](https://github.com/PeiDashuai/LLMs_Nav/tree/main) into the canonical eval `QuestionGroup` format.

- Source files:
  - `UK_THEORY_TEST.json`
  - `ZH_THEORY_TEST.json`
- Converter scripts:
  - `code/python/src/thesis/examscrapers/pei2024application/convert_uk_theory_test.py`
  - `code/python/src/thesis/examscrapers/pei2024application/convert_zh_theory_test.py`
- Output files:
  - `data/evals/pei2024application/uk_theory_test.json`
  - `data/evals/pei2024application/zh_theory_test.json`

Notes:

- Field mapping is based on `code/ts/evals/mcq/mcq.ts` (`question_number`, `question_text`, `choices`, `correct_answer`).
- Output schema follows `code/ts/evals/question-schema.ts`.
- Chinese source has 706 rows; 4 invalid rows are skipped during conversion (final: 702).
