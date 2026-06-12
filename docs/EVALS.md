# Evaluation Framework

The evaluation system benchmarks LLMs on maritime domain MCQ exams, measuring precision, recall, and F1 across multiple datasets, languages, and modalities.

## Datasets

| Dataset | Questions | Language | Modality | Focus |
| --- | --- | --- | --- | --- |
| **Shititong** | ~9.5k EN + ~53.8k ZH (deduped) | EN, ZH | Text, Vision | Mixed maritime (engine, navigation, safety) |
| **US Coast Guard** | 3,967 | EN | Text, Multimodal | Calculations, chart-based scenarios |
| **CrewCN** | 3,061 | ZH | Text | Chinese crew exams |
| **PEI2024** | 1,516 (814 UK + 702 ZH) | EN, ZH | Text | Officer certification, regulations |
| **Raynor** | 1,258 | EN | Text, Multimodal | COLREGs navigation rules (22.6% diagrams) |
| **NavReas** | 92 | EN | Multimodal | Spatial navigation reasoning |

See [evals_overview.md](evals_overview.md) for detailed breakdowns of each dataset.

## Question Schema

All datasets use a canonical `QuestionGroup` schema defined in `code/ts/evals/question-schema.ts`:

```typescript
QuestionGroup {
  name: string
  questions: Question[]
}

Question {
  id: string                    // e.g. "raynor-4285"
  questionText: string
  options: QuestionOption[]     // { id, text }
  correctOptionIds: string[]    // supports multi-answer
  images?: QuestionImage[]      // for multimodal questions
  metadata?: unknown
  source?: string
}
```

Pydantic models are auto-generated via `bun run gen:question-pydantic` (see [EVAL_QUESTION_PYDANTIC_CODEGEN.md](EVAL_QUESTION_PYDANTIC_CODEGEN.md)).

## Metrics

The evaluator (`code/ts/evals/evaluator.ts`) scores each question with:

| Metric | Formula | Description |
| --- | --- | --- |
| **Precision** | TP / predicted count | Penalizes false positives |
| **Recall** | TP / expected count | Penalizes false negatives |
| **F1** | Harmonic mean of P & R | Balanced metric |

Supports both single-answer and multi-answer MCQ formats.

## Running Evals

### Single Evaluation

```bash
EVAL_MODEL="mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4" \
EVAL_PROVIDER=litellm \
  bun code/ts/evals/pei2024/uk_theory_test.ts
```

### RAG Evaluation

```bash
EVAL_MODEL="moonshotai/Kimi-K2.5" EVAL_PROVIDER=litellm \
  bun code/ts/evals/pei2024/rag.eval.ts
```

### Batch Runner

```bash
bun code/ts/evals/run_batch.ts
```

The batch runner (`code/ts/evals/run_batch.ts`) executes all configured **(model x script)** combinations:
- Sequential script execution per model (avoids rate limiting)
- Configurable model list (LiteLLM and vLLM endpoints)
- Retry logic with exponential backoff (3 retries)
- Progress bar tracking

### Environment Variables

| Variable | Values | Description |
| --- | --- | --- |
| `EVAL_PROVIDER` | `litellm`, `vllm` | Which LLM backend to use |
| `EVAL_MODEL` | Model ID string | Which model to evaluate |
| `EVAL_CONCURRENCY` | Integer `>= 1` (default: `2`) | Number of concurrent evaluation tasks per run |
| `RESUME_EXPERIMENT_ID` | Phoenix experiment ID | Resume an interrupted experiment instead of starting fresh |

## Eval Scripts by Dataset

### PEI2024 (`code/ts/evals/pei2024/`)
- `uk_theory_test.ts` - UK Marine Engine Officer exam (baseline)
- `zh_theory_test.ts` - Chinese Maritime exam (baseline)
- `rag.eval.ts` - RAG-enhanced evaluation

### Shititong (`code/ts/evals/shititong/`)
- `en_text.ts` - English text-only
- `en_vision.ts` - English with images
- `zh_text.ts` - Chinese text-only
- `zh_vision.ts` - Chinese with images

### US Coast Guard (`code/ts/evals/us_coast_guard/`)
- `basic.ts` - Text-only
- `basic_multimodal.ts` - Text + images
- `rag.eval.ts` - RAG-enhanced

### CrewCN (`code/ts/evals/crew/`)
- `basic.ts` - Baseline evaluation
- `rag.eval.ts` - RAG-enhanced

### Raynor (`code/ts/evals/raynor/`)
- `basic.ts` - Text-only navigation rules
- `basic_multimodal.ts` - With diagrams

### NavReas (`code/ts/evals/navreas/`)
- `navreas.eval.ts` - Text-based
- `navreas_rag.eval.ts` - RAG-enhanced

## Resuming Interrupted Experiments

If an experiment is interrupted (crash, rate limit exhaustion, timeout, Ctrl+C), it can be resumed without re-running already completed examples. Set `RESUME_EXPERIMENT_ID` to the Phoenix experiment ID:

```bash
RESUME_EXPERIMENT_ID=<experiment-id> bun code/ts/evals/crew/rag.eval.ts
```

This uses Phoenix's `resumeExperiment` API which:
1. Queries the server for incomplete `(example, repetition)` pairs
2. Re-runs the task only for those pairs (completed runs are skipped)
3. Resumes evaluations for any runs missing evaluator results
4. Shows a progress bar tracking only the remaining incomplete runs

The experiment ID can be found in the Phoenix UI (port 6006) or in the server logs from the original run.

For batch runs, set the env per-script in the batch config:
```json5
{ script: "evals/crew/rag.eval.ts", env: { RESUME_EXPERIMENT_ID: "<experiment-id>" } }
```

If all runs are already complete, the resume exits immediately with a log message.

## Experiment Tracking

Results are tracked in **Arize Phoenix** (port 6006):
- Each eval run creates a Phoenix experiment
- Metrics (precision/recall/F1) stored per question
- Metadata includes model ID, retrieval config, embedding details
- See [phoenix-model-pricing.md](phoenix-model-pricing.md) for cost tracking

## Data Sources & Scrapers

| Source | Scraper | Docs |
| --- | --- | --- |
| Shititong.cn | `code/ts/scripts/shititong/` | [scrapers/shititong_exams.md](scrapers/shititong_exams.md) |
| US Coast Guard | `code/python/src/examscrapers/us_coast_guard/` | - |
| CrewCN | `code/python/scripts/crewcn/` | [scrapers/crewcn.md](scrapers/crewcn.md) |
| Raynor Maritime | `code/python/src/examscrapers/raynormaritime/` | [scrapers/raynormaritime.md](scrapers/raynormaritime.md) |
| PEI2024 | `code/python/src/examscrapers/pei2024/` | [scrapers/pei2024application.md](scrapers/pei2024application.md) |
| NavReas | `code/python/src/examscrapers/navreas/` | - |

Deduplication pipeline: [mcq-dedup-context.md](mcq-dedup-context.md)
Format conversion: [scrapers/mcq_conversion.md](scrapers/mcq_conversion.md)
