# raynormaritime

Scrapes Raynor Maritime Navigation Rules questions into the canonical eval `QuestionGroup[]` format using deterministic `FilterQuest` ID requests.

- Scraper script:
  - `code/python/src/thesis/examscrapers/raynormaritime/scrape_questions.py`
- Default output file:
  - `data/evals/raynor/raynor_international_questions.json`
- Images directory:
  - `data/evals/raynor/images`

Classification:

- `INLAND ONLY` -> group `raynor-inland`
- `BOTH INTERNATIONAL AND INLAND` -> group `raynor-both`
- `INTERNATIONAL ONLY` -> group `raynor-international-only`

ID format:

- Question IDs are normalized to `raynor-<group>-<4-digit-id>`.
  - Examples: `raynor-inland-0004`, `raynor-both-4585`, `raynor-international-only-0397`
- Per-question `source` includes only the original Raynor ID:
  - `{"originalQuestionId": "0004"}`

Run commands:

- Full scrape (default `0000..9999`):
  - `uv run --project code/python python code/python/src/thesis/examscrapers/raynormaritime/scrape_questions.py`
- Smoke test (stop after 20 accepted questions):
  - `uv run --project code/python python code/python/src/thesis/examscrapers/raynormaritime/scrape_questions.py --max-accepted 20`
- Bounded ID range:
  - `uv run --project code/python python code/python/src/thesis/examscrapers/raynormaritime/scrape_questions.py --start-id 0 --end-id 500`

Notes:

- The scraper does not load or resume from old JSON; each run starts fresh.
- Invalid/blank IDs are skipped.
- Option text prefixes (`A.`, `B.`, etc.) are stripped.
- If an image URI is remote and not present locally, it is downloaded into `data/evals/raynor/images` and referenced as `images/<filename>` when possible.
