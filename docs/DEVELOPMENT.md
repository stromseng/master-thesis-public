# Development Guide

## Prerequisites

- [Docker](https://www.docker.com/) - Qdrant vector database + Phoenix observability
- [Bun](https://bun.sh/) - JavaScript runtime and toolkit
- [uv](https://github.com/astral-sh/uv) - Python package manager
- [just](https://github.com/casey/just) - Command runner

## Environment Setup

```bash
cp .env.template .env
```

Required variables in `.env`:

| Variable | Description |
| --- | --- |
| `IDUN_USER` | NTNU username for IDUN HPC |
| `LITE_LLM_API_KEY` | API key for LiteLLM (NTNU HPC gateway) |
| `LITE_LLM_ENDPOINT` | LiteLLM endpoint (default: `<your-endpoint>`) |
| `QDRANT_URL` | Qdrant endpoint (default: `http://<your-server>:6333`) |
| `PHOENIX_HOST` | Phoenix UI URL (default: `http://<your-server>:6006`) |
| `PHOENIX_COLLECTOR_ENDPOINT` | OpenTelemetry GRPC endpoint (default: `http://<your-server>:4317`) |
| `HUGGINGFACE_TOKEN` | For downloading models on IDUN |
| `USER_EMAIL` | For SLURM job notifications |

## Install Dependencies

```bash
cd code/python && uv sync      # Python dependencies
bun install                    # TypeScript dependencies (from root)
```

### Pre-commit hooks

```bash
bunx @j178/prek install
bun install
```

## Start Dev Environment

```bash
just dev
```

This starts:
- **Qdrant** vector database (ports 6333/6334)
- **Phoenix** observability (port 6006, GRPC on 4317)
- **FastAPI** sidecar server (port 8001)
- **TS client** auto-generation watcher

Use `just dev-python` to skip Docker (when Qdrant/Phoenix are already running or hosted elsewhere).

## Commands

### Root

| Command | Description |
| --- | --- |
| `just dev` | Start full dev environment |
| `just dev-python` | Start FastAPI + TS client watcher only |
| `just check` | Run all linting and type checks (Python + TS) |
| `just fmt` | Format all code (Python + TS) |
| `just idun <args>` | Run IDUN HPC CLI commands |

### TypeScript (`just ts <recipe>`)

| Command | Description |
| --- | --- |
| `just ts check` | lint + fmt-check + typecheck |
| `just ts typecheck` | TypeScript type checking |
| `just ts lint` | Lint with oxlint |
| `just ts fmt` | Format with oxfmt |
| `just ts test` | Run tests with Vitest |
| `bun run gen:api-client` | Regenerate Python API client |
| `bun run gen:question-pydantic` | Regenerate Pydantic models from TS schema |

### Python (`just py <recipe>`)

| Command | Description |
| --- | --- |
| `just py check` | lint + fmt-check + typecheck |
| `just py lint` | Lint with Ruff |
| `just py fmt` | Format with Ruff |
| `just py typecheck` | Type check with ty |

## Architecture

The system uses a **TypeScript + Python bridge** architecture:

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

### TypeScript Services (`code/ts/src/services/`)

| Service | Purpose |
| --- | --- |
| `Qdrant.ts` | Vector database indexing, search, collection management |
| `Retrieval.ts` | Hybrid retrieval with Dense + Sparse + ColBERT reranking |
| `LLMProvider.ts` | LLM provider abstraction (LiteLLM for NTNU HPC, vLLM for local) |
| `LanguageModel.ts` | LLM invocation via Vercel AI SDK |
| `Embedding.ts` | Embedding model management (Dense, Sparse, Late) |
| `Chunking.ts` | Document chunking (6 methods via Python API) |
| `PhoenixClient.ts` | Experiment tracking and dataset management |
| `PythonApiClient.ts` | Auto-generated OpenAPI client for FastAPI |

### Python API Endpoints (`code/python/src/ts_api/`)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/embed/dense` | POST | Dense embeddings (BGE-small) |
| `/embed/sparse` | POST | Sparse BM25 embeddings |
| `/embed/late` | POST | Late-interaction ColBERT embeddings |
| `/embed/*/batch` | POST | Batch variants of above |
| `/embed/*/resolve` | POST | Get vector config (dimensions, etc.) |
| `/embedding-config` | GET | Full embedding configuration |

### Embedding Models

| Type | Model | Vector Name | Use |
| --- | --- | --- | --- |
| Dense | `BAAI/bge-small-en-v1.5` | `dense-bge-small` | Semantic similarity |
| Sparse | `Qdrant/bm25` | `sparse-bm25` | Keyword matching |
| Late | `Qdrant/ColBERT-onnx` | `late-colbert` | Token-level reranking |

All embeddings use FastEmbed (ONNX) for fast local inference.

### Chunking Methods

| Method | Description |
| --- | --- |
| `recursive` (default) | Hierarchical splitting, 2048 tokens |
| `fast` | Simple byte-based, 4096 bytes |
| `semantic` | Embedding-based breakpoints |
| `late` | ColBERT-aware chunking |
| `neural` | Neural network splitting |
| `slumber` | Hybrid (Chonkie library) |

### RAG Pipeline

The Standard RAG pipeline uses hybrid retrieval with Reciprocal Rank Fusion (RRF):

1. **Query** is embedded in parallel using Dense, Sparse, and Late-interaction models
2. **Qdrant** prefetches candidates from each vector type
3. **RRF fusion** combines rankings (or ColBERT reranks if late-interaction is available)
4. **Top-K contexts** are injected into the LLM prompt as `<source>` blocks
5. **LLM generates** an answer grounded in the retrieved context

Configuration per eval:
- `limit`: Final result count (default: 10)
- `prefetchLimit`: Intermediate candidates before fusion (default: 20)

### Indexing Documents to Qdrant

RAG evals require documents to already be indexed in Qdrant. Quick start:

```bash
# Default: fastembed (all-MiniLM-L6-v2) + recursive chunking
bun code/ts/scripts/index-documents.ts

# LiteLLM (Qwen3-Embedding-8B) + recursive chunking
DENSE_METHOD=litellm bun code/ts/scripts/index-documents.ts
```

Collection names are auto-generated based on embedding + chunking config (e.g. `documents_recursive_{hash}`). Changing the embedding model or chunking method creates a new collection automatically.

See [indexing.md](indexing.md) for full details on env vars, collection naming, and the pipeline.

See [skyhigh.md](skyhigh.md) for details on the hosted Qdrant instance.

### LLM Providers

**LiteLLM** (HPC gateway at `<your-endpoint>`):

| Model ID | Description |
| --- | --- |
| `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` | Mistral Large 3 |
| `openai/gpt-oss-120b` | GPT-OSS 120B |
| `NorwAI/NorwAI-Magistral-24B-reasoning` | NorwAI Magistral 24B |

**vLLM** (local/IDUN): Configure with `EVAL_PROVIDER=vllm` and optional port.

## Typst Submodule

The thesis document (`typst/`) lives in a separate repo ([master-thesis-typst](https://github.com/stromseng/master-thesis-typst)) and is cloned into `typst/` as an independent git repo. The parent repo gitignores this directory. This allows the Typst web editor to work with a smaller repo.

### First-time setup (after cloning)

```bash
git clone https://github.com/stromseng/master-thesis-typst.git typst
```

### Pull changes from the Typst web editor

```bash
cd typst && git pull
```

### Push local typst changes

```bash
cd typst
git add .
git commit -m "update chapter"
git push
```

## Configuration Files

| File | Purpose |
| --- | --- |
| `.env` | API keys and secrets |
| `docker-compose.yml` | Qdrant + Phoenix services |
| `code/ts/openapi-ts.config.ts` | API client generation config |
| `code/ts/evals/question-schema.ts` | Canonical evaluation question schema |
