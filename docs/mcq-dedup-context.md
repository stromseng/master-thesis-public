# MCQ Deduplication - Context for Continuation

## Summary

Created a script to filter duplicate/near-duplicate questions from MCQ datasets (English: 21k, Chinese: 257k questions from shititong.cn). Supports both datasets via `--chinese` flag.

## What Was Built

**Script:** `code/python/scripts/idun/dedup_mcq.py`

**Pre-filtering:**
- Removes questions with `"answer": "null"` (string null, not valid answer)
- Removes questions where all options are `"null"` (garbage data)
- (Chinese only) Removes 简答题 (short answer) questions — not MCQ format

**Six-stage pipeline:** (order configurable via `--stages`)
0. **LLM OCR Repair** (Chinese only) - Fix garbled question text using LLM with clean options/hints as context
1. **Exact Fuzzy** (0.99) - Remove near-exact duplicates cheaply
2. **LLM Quality Filter** - Remove garbage/OCR-error questions using `openai/gpt-oss-120b`
3. **Embedding similarity** - Qwen3-Embedding-8B + cosine similarity (default 0.70 threshold)
4. **Fuzzy matching** - rapidfuzz token_set_ratio (default 0.70 threshold)
5. **LLM Duplicate Verification** - Review flagged duplicates, recover false positives

**Duplicate selection:** When duplicates are found, keeps the "better" question:
1. Longer total options length (non-null options)
2. Tiebreaker: longer question text

**Key design decision:** Compare `question + options + answer` combined, not just question text. This prevents false positives on generic stems like "Which is true?".

## How to Run

```bash
# English: Full pipeline with LLM stages (default)
cd code/python && uv run python -m scripts.idun.dedup_mcq --limit 100 --dry-run

# English: Skip LLM stages, just embedding + fuzzy
cd code/python && uv run python -m scripts.idun.dedup_mcq --stages embedding,fuzzy --limit 1000 --dry-run

# Chinese: Full pipeline with OCR repair + dedup
cd code/python && uv run python -m scripts.idun.dedup_mcq --chinese --limit 20 --dry-run

# Chinese: Skip LLM stages, just dedup
cd code/python && uv run python -m scripts.idun.dedup_mcq --chinese --stages exact_fuzzy,embedding,fuzzy --limit 100 --dry-run

# Use local embeddings instead of IDUN API
cd code/python && uv run python -m scripts.idun.dedup_mcq --limit 100 --dry-run --local

# Full run on IDUN
just idun submit --script scripts/idun/dedup_mcq.py
just idun submit --script scripts/idun/dedup_mcq.py --args "--chinese"
```

**CLI Flags:**
- `--chinese` - Process Chinese dataset instead of English
- `--limit N` - Sample N random questions for testing (default: 0 = all)
- `--stages` - Stages to run in order (English default: `exact_fuzzy,llm_quality,embedding,fuzzy,llm_verify`; Chinese default: `llm_repair,exact_fuzzy,llm_quality,embedding,fuzzy,llm_verify`)
- `--dry-run` - Don't save files, just show stats
- `--verbose` - Show each duplicate pair found
- `--fuzzy-threshold N` - Fuzzy match threshold 0-1 (default: 0.70)
- `--embedding-threshold N` - Cosine similarity threshold 0-1 (default: 0.70)
- `--verify-threshold N` - Skip LLM verification for embedding pairs above this (default: 0.97)
- `--llm-model` - LLM model for quality/verification stages (default: `openai/gpt-oss-120b`)
- `--local` - Use local FastEmbed instead of IDUN API

## Relevant Files

| File | Purpose |
|------|---------|
| `code/python/scripts/idun/dedup_mcq.py` | Main dedup script |
| `data/evals/mcq/SHITITONG_ENGLISH_QUESTIONS.json` | Input: 21,144 English MCQs |
| `data/evals/mcq/SHITITONG_ENGLISH_DEDUPED.json` | Output: deduplicated English dataset |
| `data/evals/mcq/dedup_report.json` | Output: English report with removed pairs |
| `data/evals/mcq/SHITITONG_CHINESE_QUESTIONS.json` | Input: 257,396 Chinese MCQs |
| `data/evals/mcq/SHITITONG_CHINESE_DEDUPED.json` | Output: deduplicated Chinese dataset |
| `data/evals/mcq/dedup_report_chinese.json` | Output: Chinese report with removed/repaired pairs |
| `docs/mcq-deduplication-plan.md` | Full planning document |
| `code/python/pyproject.toml` | Added `rapidfuzz>=3.0.0` dependency |

## Data Structure

```json
{
  "question": "1. The fresh water jacket cooling system forms a circuit.( )",
  "options": {"A": "open", "B": "closed", "C": "opening", "D": "closing"},
  "answer": "B",
  "type": "单选题",
  "category_id": "241947",
  "original_id": "...",
  "source": {...}
}
```

Question types:
- 单选题 (single choice): 19,797
- 填空题 (fill in blank): 580
- 判断题 (true/false): 530
- 多选题 (multiple choice): 237

## Algorithm Details

**Pre-filtering (removes garbage data):**
```python
def is_null_value(value):
    """Check if value is 'null' string, None, or empty."""
    if value is None: return True
    if isinstance(value, str): return value.strip().lower() == "null" or value.strip() == ""
    return False

# Filter out: answer == "null" OR all options == "null"
```

**Comparison key:**
```python
def create_comparison_text(q):
    question = normalize_text(q["question"])  # lowercase, strip numbers, collapse whitespace
    options = " ".join(sorted(f"{k}:{v}" for k, v in q["options"].items()))
    answer = q["answer"]
    return f"{question} | {options} | {answer}"
```

**Duplicate selection (when duplicates found, keep the better one):**
```python
def choose_better_question(q1, q2):
    # 1. Prefer longer total options length
    # 2. Tiebreaker: longer question text
    # 3. Default: keep first encountered
```

**Fuzzy matching:** O(n²) pairwise comparison using `fuzz.token_set_ratio`. For 21k questions this is slow (~223M comparisons).

**Embedding similarity:**
1. Embed all texts with FastEmbed (batches of 100)
2. Normalize vectors
3. For each question, compute cosine similarity with all later questions
4. Mark as duplicate if similarity >= threshold

## Performance Notes

- Fuzzy matching 21k questions: Very slow locally (O(n²) = ~223M comparisons)
- Embedding 21k questions: ~4 minutes with FastEmbed on CPU
- Recommendation: Run on IDUN with `--limit` for testing, full run for production

## Embedding Details

**Default: IDUN API (Qwen3-Embedding-8B)**
- Uses IDUN's LLM API at `https://llm.hpc.ntnu.no/v1/embeddings`
- Model: `Qwen/Qwen3-Embedding-8B` (larger, higher quality embeddings)
- Requires `LITE_LLM_API_KEY` environment variable (from `.env`)
- Falls back to local embeddings if API is unavailable

**Fallback: Local FastEmbed**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- First run downloads model to `~/.cache/fastembed/` (~90MB)
- Force local with `--local` flag

## LLM-as-Judge Stages

**LLM Quality Filter (`llm_quality`):**
- Runs on ALL questions before deduplication
- Removes garbage questions with OCR errors, gibberish, or malformed text
- Uses `openai/gpt-oss-120b` via IDUN LiteLLM API (`https://llm.hpc.ntnu.no/v1`)
- Prompt asks LLM to rate question as GOOD or BAD

**LLM Duplicate Verification (`llm_verify`):**
- Runs after embedding/fuzzy stages
- Reviews flagged duplicate pairs and recovers false positives
- Skips high-similarity embedding pairs (>97%) - trusts as true duplicates
- Verifies all fuzzy pairs and borderline embedding pairs (70-97%)
- If LLM says DIFFERENT, restores the removed question to the dataset

## Report Format (for reviewing removed pairs)

Each stage in `dedup_report.json` contains its own data format:

```json
{
  "original_count": 1000,
  "final_count": 900,
  "config": {
    "stages": ["llm_quality", "embedding", "fuzzy", "llm_verify"],
    "fuzzy_threshold": 0.70,
    "embedding_threshold": 0.70,
    "verify_threshold": 0.97,
    "llm_model": "openai/gpt-oss-120b"
  },
  "stages": [
    {
      "name": "llm_quality",
      "model": "openai/gpt-oss-120b",
      "removed_count": 5,
      "removed_questions": [{"id": 123, "question": "...", "llm_response": "BAD"}]
    },
    {
      "name": "embedding",
      "threshold": 0.70,
      "removed_pairs": [{"similarity": 0.85, "kept": {...}, "removed": {...}}]
    },
    {
      "name": "fuzzy",
      "threshold": 0.70,
      "removed_pairs": [{"similarity": 0.92, "kept": {...}, "removed": {...}}]
    },
    {
      "name": "llm_verify",
      "model": "openai/gpt-oss-120b",
      "verify_threshold": 0.97,
      "confirmed_count": 50,
      "recovered_count": 3,
      "recovered_pairs": [{"kept": {...}, "removed": {...}, "llm_response": "DIFFERENT"}]
    }
  ]
}
```

## Chinese Dataset & OCR Repair

**Chinese dataset:** 257,396 questions from `SHITITONG_CHINESE_QUESTIONS.json`
- ~81.5% of question texts have OCR artifacts (characters with similar shapes swapped)
- Options, hints, and answers are mostly clean
- Has `hint` field not present in English data — provides a clean summary of each question
- Question types: 单选题 (178k), 多选题 (35k), 判断题 (43k), 填空题 (916), 简答题 (48)

**Common OCR errors:** 绳→船, 路→舶, 曳→拖, 雷责→雷达, etc.

**LLM OCR Repair stage (`llm_repair`):**
- Sends ALL Chinese questions to LLM (corruption too pervasive for heuristic filtering)
- Prompt provides: corrupted question + clean options + clean hint + answer
- LLM returns corrected question text or "UNCHANGED"
- Stores `original_question` field on repaired questions for audit trail
- Runs before quality filter so repaired questions survive the pipeline

**Repaired question has extra field:**
```json
{
  "question": "航海雷达探测目标的距离是根据...",
  "original_question": "航海雷责探辆护障绳整吊路辆蚀...",
  "options": {...},
  "hint": "航海雷达测距离，脉冲传播时间定"
}
```

---

## IDUN Context

- Script location: `code/python/scripts/idun/` (runs as module with overlay system)
- GPU: H100 available, but this script is mostly CPU-bound (embedding benefits from GPU)
- See `docs/idun_usage/IDUN.md` for IDUN CLI usage

## Dependencies Added

```toml
# code/python/pyproject.toml
"rapidfuzz>=3.0.0",
```

Already available: `fastembed`, `numpy`, `typer`, `rich`, `requests`, `python-dotenv`
