# Benchmarking and Augmenting LLM Capabilities in the Maritime Domain

Master's thesis in Informatics, Norwegian University of Science and Technology (NTNU), 2026.

**[Read the full thesis on NTNU Open](https://ntnuopen.ntnu.no/)** *(link available after publication)*

This thesis evaluates whether current large language models can answer maritime domain questions, and whether retrieval-augmented generation (RAG) improves that performance. The evaluation suite contains 139,266 question instances across 21 datasets and derived variants: 79,688 multiple-choice questions and 59,578 derived open-ended questions in English and Mandarin Chinese. The models span 4B to 1T+ total parameters and cover dense and mixture-of-experts architectures.

The central finding is deliberately cautious: strong maritime multiple-choice performance does not yet translate into consistently reliable free-form maritime answering. RAG improves the multiple-choice setting most for weaker models and corpus-covered regulatory questions, including cross-lingual transfer from an English corpus to Chinese questions, but RAG was not evaluated on the open-ended variants. The results therefore support maritime RAG as promising knowledge-support infrastructure, not as evidence of production-ready safety-critical decision support.

## Research Questions

1. **RQ1:** How well do current LLMs perform on existing maritime benchmarks?
2. **RQ2:** Does a RAG system improve performance on these benchmarks?
3. **RQ3:** How do model size and inference cost relate to performance across maritime task types?

## Repository Overview

This repository contains the full codebase for the thesis:

- **Dataset collection and curation pipeline** — scrapers, converters, OCR repair, relevance filtering, and deduplication for 6 maritime exam sources
- **Hybrid RAG pipeline** — Qwen3-Embedding-8B dense retrieval + BM25 sparse retrieval with reciprocal rank fusion via Qdrant
- **Evaluation harness** — batch-capable MCQ, multimodal, RAG, and open-ended evaluation with Phoenix tracing and cost tracking
- **HPC integration** — vLLM serving and SLURM job management on NTNU's IDUN cluster

Built with TypeScript (Bun, Effect-TS) and Python (FastAPI, uv), orchestrated via Docker Compose and Just.

## Key Results

- **Multiple-choice baseline** ranges from 52.6% to 77.5% weighted accuracy across the 13-model text-only baseline; MMLU-Pro strongly predicts maritime MCQ performance (Spearman rho = 0.97)
- **Open-ended evaluation** lowers scores to 31.4% to 68.2% across the 9 evaluated models, widens the gap between model tiers, and leaves only Kimi K2.6 approaching a practical free-form pass level
- **RAG gains are real but bounded**: across the three English-primary RAG-tested datasets, retrieval improves 28 of 39 model-dataset pairs with an average gain of +2.2 percentage points; the largest individual gain is +14.0pp
- **Cross-lingual RAG works in a narrow matched setting**: retrieval from an English-only corpus improves Chinese PEI2024 performance by +6.1pp on average, where the Chinese questions test the same international regulatory content
- **Scaling and specialization**: Llamarine improves strongly over its Llama 3.1 70B base model, but remains below newer general-purpose models; this is evidence about one QLoRA fine-tuning setup, not all fine-tuning
- **Cost and model size**: within the Qwen 3.5 family, MCQ returns flatten beyond 27B parameters; MoE models can reach high accuracy with far fewer active parameters, but open-ended results prevent small-model MCQ pass marks from being treated as deployment readiness

## Models Evaluated

The full text-only multiple-choice baseline covers 13 models spanning 4B to 1T total parameters across dense and mixture-of-experts (MoE) architectures. The open-ended and multimodal evaluations cover smaller subsets due to compute constraints. For MoE models, "Active" shows the routed parameters per forward pass.

| Model | Total Params | Active Params | Architecture |
|-------|-------------|---------------|--------------|
| Kimi K2.6 | 1000B | 32B | MoE |
| Kimi K2.5 | 1000B | 32B | MoE |
| Qwen3.5-397B-A17B | 397B | 17B | MoE |
| Qwen3.5-122B-A10B | 122B | 10B | MoE |
| GPT-OSS-120B | 120B | 5B | MoE |
| Llama 3.1 70B | 70B | 70B | Dense |
| Llamarine 70B | 70B | 70B | Dense |
| Gemma 4 31B IT | 31B | 31B | Dense |
| Qwen3.6-27B | 27B | 27B | Dense |
| Qwen3.5-27B | 27B | 27B | Dense |
| Gemma 4 26B IT | 26B | 4B | MoE |
| Qwen3.5-9B | 9B | 9B | Dense |
| Qwen3.5-4B | 4B | 4B | Dense |

## Reproducing the Experiments

> **Note:** This codebase was developed on NTNU's IDUN HPC cluster. The IDUN CLI tooling (`code/python/src/idun/`) and SLURM job scripts are tightly coupled to that environment and are included as reference rather than portable utilities. The core evaluation and RAG pipelines run independently on any machine with Docker and a GPU, but replicating the exact HPC workflow requires access to a compatible SLURM cluster.

### Prerequisites

[Docker](https://docs.docker.com/get-docker/), [Bun](https://bun.sh/), [uv](https://docs.astral.sh/uv/), and [Just](https://github.com/casey/just) must be installed. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for full details.

### Step 1: Environment Setup

```bash
cp .env.template .env              # Configure API keys and service URLs
cd code/python && uv sync && cd ../..
bun install
```

Edit `.env` to set at minimum `LITE_LLM_API_KEY`, `QDRANT_URL`, and `PHOENIX_HOST`. See [.env.template](.env.template) for all available variables.

### Step 2: Dataset Preparation

#### Scraping Raw Data

Each of the 6 exam sources has its own scraper. Per-source instructions are in [docs/scrapers/](docs/scrapers/):

| Dataset | Scraper docs |
|---------|-------------|
| Shititong | [docs/scrapers/shititong_exams.md](docs/scrapers/shititong_exams.md) |
| US Coast Guard | Scraped from [USCG NMC](https://www.dco.uscg.mil/nmc/examinations/) via `code/python/src/examscrapers/us_coast_guard/` |
| CrewCN | [docs/scrapers/crewcn.md](docs/scrapers/crewcn.md) |
| PEI2024 | [docs/scrapers/pei2024application.md](docs/scrapers/pei2024application.md) |
| Raynor | [docs/scrapers/raynormaritime.md](docs/scrapers/raynormaritime.md) |
| NavReas | Downloaded from [MO-RISE/navreas-dataset](https://github.com/MO-RISE/navreas-dataset) via `code/python/src/examscrapers/navreas/` |

#### MCQ and Open-Ended Conversion

Raw scraped data must be converted to the canonical MCQ JSON schema before evaluation. Open-ended variants are derived from the text-only MCQ datasets by removing answer options and preserving reference correct and incorrect answers for judge-based scoring. See [docs/scrapers/mcq_conversion.md](docs/scrapers/mcq_conversion.md) for format details.

```bash
cd code/python

uv run python -m examscrapers.crewcn.convert_crewcn
uv run python -m examscrapers.shititong.convert_shititong --target all
uv run python -m examscrapers.pei2024application.convert_uk_theory_test
uv run python -m examscrapers.pei2024application.convert_zh_theory_test
```

Output lands in `data/evals/`.

#### Deduplication

Several sources contain duplicate or near-duplicate questions. The deduplication pipeline must be run before evaluation to avoid inflated scores. See [docs/mcq-dedup-context.md](docs/mcq-dedup-context.md) for the full pipeline.

### Step 3: Start Infrastructure

```bash
just dev    # Starts Qdrant, Phoenix, FastAPI, and TS client watcher
```

This runs `docker compose up` (Qdrant on port 6333, Phoenix on port 6006, FastAPI on port 8001) and watches for API schema changes. If using remote NTNU-hosted services, configure the URLs in `.env` and use `just dev-python` for local FastAPI only.

### Step 4: Index Documents for RAG

The RAG corpus consists of 23 maritime reference books and regulatory documents (COLREGs, SOLAS, MARPOL, STCW, etc.). See [docs/rag/rag-corpus.md](docs/rag/rag-corpus.md) for the full list.

#### Convert PDFs to Markdown

Source PDFs in `data/rag/raw/` must first be converted to Markdown using [Marker](https://github.com/VikParuchuri/marker):

```bash
cd code/python
uv run python scripts/convert_pdfs.py           # Convert all PDFs
uv run python scripts/convert_pdfs.py --test     # Test mode (smallest PDF only)
```

Output is written to `data/rag/processed/`.

#### Embed and Index

```bash
DENSE_METHOD=litellm DENSE_MODEL=Qwen/Qwen3-Embedding-8B CHUNKING_METHOD=recursive \
  bun code/ts/scripts/index-documents.ts
```

This embeds and upserts documents with dense Qwen3-Embedding-8B vectors and sparse BM25 vectors into Qdrant. The reported RAG experiments use recursive chunking, top-5 retrieval, and reciprocal rank fusion; ColBERT late-interaction reranking was explored but not used in the final results. See [docs/indexing.md](docs/indexing.md) for the full indexing pipeline and configuration options.

### Step 5: Run Evaluations

The evaluation harness uses the [Vercel AI SDK](https://ai-sdk.dev/) for model abstraction. Models and providers are selected via environment variables, so switching between models requires no code changes:

```bash
# Single eval
EVAL_MODEL="moonshotai/Kimi-K2.5" EVAL_PROVIDER=litellm \
  bun code/ts/evals/pei2024/uk_theory_test.ts

# RAG eval
EVAL_MODEL="moonshotai/Kimi-K2.5" EVAL_PROVIDER=litellm \
  bun code/ts/evals/pei2024/rag.eval.ts

# Open-ended eval
EVAL_MODEL="moonshotai/Kimi-K2.6" EVAL_PROVIDER=litellm \
  bun code/ts/evals/pei2024/uk_open_ended.ts

# Batch all configured (model x script) combinations
bun code/ts/evals/run_batch.ts
```

`EVAL_PROVIDER` selects the backend (`litellm` for NTNU's HPC cluster, `vllm` for local serving) and `EVAL_MODEL` sets the model ID. MCQ runs are scored by question-level F1, while open-ended runs are scored by an LLM-as-a-judge factuality classifier. Results are tracked in Phoenix (http://localhost:6006). See [docs/EVALS.md](docs/EVALS.md) for the full evaluation framework, metrics, and dataset documentation.

### IDUN HPC CLI

The project includes an interactive CLI for managing jobs and vLLM inference servers on NTNU's IDUN HPC cluster:

```bash
just idun              # Interactive TUI menu
just idun submit       # Job submission wizard (script, GPUs, VRAM, duration)
just idun list         # List jobs with actions (cancel, SSH, logs, kill)
just idun vllm serve   # Start vLLM/SGLang inference server
just idun vllm status  # Show running servers with health checks
just idun vllm connect # SSH tunnel to a running server
just idun monitor      # Real-time CPU/GPU resource monitor
just idun ssh          # Open shell on compute node
just idun logs         # Tail job stdout/stderr
just idun script       # Run a one-off script on the cluster
just idun artifacts    # Download run artifacts from scripts or jobs
```

Built with Typer and InquirerPy. See [docs/idun_usage/IDUN.md](docs/idun_usage/IDUN.md) for cluster setup and usage details.

## Datasets

The benchmark has three tracks:

- **Text-only MCQ:** 77,119 questions across 7 datasets
- **Multimodal MCQ:** 2,569 questions across 7 datasets
- **Derived open-ended:** 59,578 questions across 7 datasets, converted from the text-only MCQ sources

Together these form 139,266 evaluated question instances across 21 datasets and derived variants.

### Multiple-Choice Sources

| Dataset | Questions | Language | Modality | Focus |
|---------|-----------|----------|----------|-------|
| Shititong | 63,347 (8,519 EN text + 1,026 EN multimodal + 53,545 ZH text + 257 ZH multimodal) | EN, ZH | Text, Multimodal | Mixed maritime (engine, navigation, safety) |
| US Coast Guard | 10,414 (9,504 text + 910 multimodal) | EN | Text, Multimodal | Calculations, chart-based scenarios |
| CrewCN | 3,061 | EN with ZH glosses | Text | Crew certification exams |
| PEI2024 | 1,516 (814 EN + 702 ZH) | EN, ZH | Text | Officer certification, regulations |
| Raynor | 1,258 (974 text + 284 multimodal) | EN | Text, Multimodal | COLREGs navigation rules |
| NavReas | 92 | EN | Multimodal | Spatial navigation reasoning |
| **MCQ total** | **79,688** | | | |

The raw Shititong corpus contains ~278k questions (257k ZH + 21k EN). After OCR repair, deduplication, and filtering of malformed entries, the evaluated subset is 63,347. See [docs/evals_overview.md](docs/evals_overview.md) for detailed breakdowns per dataset, and [docs/mcq-dedup-context.md](docs/mcq-dedup-context.md) for the deduplication pipeline.

### Open-Ended Variants

The open-ended datasets are derived from the text-only MCQ sources by removing the answer options. They are evaluated without RAG using an LLM-as-a-judge factuality classifier, so their scores are not directly comparable to the deterministic MCQ answer-key scores.

| Open-ended dataset | Questions | Derived from |
|--------------------|-----------|--------------|
| CrewCN open-ended | 2,936 | CrewCN |
| PEI2024 UK open-ended | 802 | PEI2024 UK |
| PEI2024 ZH open-ended | 681 | PEI2024 ZH |
| Raynor open-ended | 963 | Raynor text-only |
| Shititong EN open-ended | 7,979 | Shititong EN text |
| Shititong ZH open-ended | 36,903 | Shititong ZH text |
| US Coast Guard open-ended | 9,314 | US Coast Guard text-only |
| **Open-ended total** | **59,578** | |

## Project Structure

```
.
├── code/ts/                        # TypeScript layer (Effect-TS + Bun)
│   ├── evals/                      # Evaluation scripts per dataset
│   │   ├── run_batch.ts            # Batch runner (model x script combinations)
│   │   ├── evaluator.ts            # Precision/Recall/F1 scoring
│   │   └── {dataset}/              # Per-dataset eval scripts (MCQ, RAG, multimodal, open-ended)
│   └── src/services/               # Core services (Qdrant, Retrieval, LLM, Embedding)
├── code/python/                    # Python layer (FastAPI + ML)
│   └── src/
│       ├── ts_api/                 # FastAPI sidecar (embeddings, chunking)
│       ├── examscrapers/           # Dataset scrapers & converters
│       └── idun/                   # IDUN HPC CLI
├── data/
│   ├── evals/                      # Evaluation datasets (JSON)
│   └── rag/                        # RAG source documents (raw PDFs + processed)
├── docs/                           # Documentation
├── typst/                          # Thesis document (Typst)
├── docker-compose.yml              # Qdrant + Phoenix + FastAPI
└── justfile                        # Command runner
```

### Architecture

```
┌──────────────────────────────────────────────┐
│  TypeScript (Effect-TS + Bun)                │
│  Orchestration, evals, LLM calls, retrieval  │
└──────────────────┬───────────────────────────┘
                   │ OpenAPI (auto-generated client)
┌──────────────────▼───────────────────────────┐
│  Python (FastAPI)                             │
│  Embeddings, chunking, document processing   │
└──────────────────┬───────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Qdrant      Phoenix     LiteLLM/vLLM
  (vectors)   (tracing)    (LLM serving)
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Dev setup, commands, architecture, services |
| [docs/EVALS.md](docs/EVALS.md) | Evaluation framework, datasets, metrics |
| [docs/indexing.md](docs/indexing.md) | RAG document indexing pipeline |
| [docs/evals_overview.md](docs/evals_overview.md) | Detailed breakdown of all evaluation datasets |
| [docs/ts-python-rpc.md](docs/ts-python-rpc.md) | TypeScript-Python bridge architecture |
| [docs/phoenix-model-pricing.md](docs/phoenix-model-pricing.md) | Model cost tracking in Phoenix |
| [docs/idun_usage/IDUN.md](docs/idun_usage/IDUN.md) | IDUN HPC cluster usage (vLLM, SLURM) |
| [docs/skyhigh.md](docs/skyhigh.md) | Skyhigh server hosting (Qdrant + Phoenix) |
| [docs/mcq-dedup-context.md](docs/mcq-dedup-context.md) | MCQ deduplication pipeline |
| [docs/rag/rag-corpus.md](docs/rag/rag-corpus.md) | RAG corpus: all reference books and conventions |
| [docs/scrapers/](docs/scrapers/) | Per-dataset scraper documentation |

## Development

```bash
just check             # Run all checks (lint + format check + typecheck) for Python and TypeScript
just fmt               # Format both Python and TypeScript
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full development setup.

## Citation

```bibtex
@mastersthesis{stromseng2026maritime,
  title   = {Benchmarking and Augmenting LLM Capabilities in the Maritime Domain},
  author  = {Str{\o}mseng, Magnus Alexander and Alfnes, Sondre},
  school  = {Norwegian University of Science and Technology},
  year    = {2026},
  month   = jun,
  type    = {Master's thesis}
}
```

## Authors

- **Magnus Alexander Strømseng**
- **Sondre Alfnes**
- Supervisor: Benjamin Kille
- Co-supervisors: Børge Rokseth, Rudolf Mester

## License

This project is licensed under the [MIT License](LICENSE).
