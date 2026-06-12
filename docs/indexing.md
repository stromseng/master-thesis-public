# Document Indexing

## Overview

The indexing pipeline converts markdown documents into vector embeddings stored in Qdrant. Documents are chunked, embedded using multiple models, and upserted into a Qdrant collection for hybrid retrieval.

## Prerequisites

- **Python FastAPI server** running locally (`just dev-python`) — required for chunking and BM25 sparse embeddings
- **Qdrant** — either local (`just dev`) or the example remote endpoint in the checked-in defaults
- **LiteLLM API key** in `.env` — only needed when using `DENSE_METHOD=litellm`

## Step-by-Step

### 1. Place source documents

Put markdown files in `data/rag/processed/`:

```
data/rag/processed/
├── chapter1.md
├── chapter2.md
└── appendix.md
```

To convert PDFs to markdown first:

```bash
cd code/python
uv run python scripts/convert_pdfs.py
```

### 2. Start the Python FastAPI server

The server handles chunking and BM25 sparse embedding (runs on `localhost:8001`):

```bash
just dev-python
```

Keep this running in a separate terminal.

### 3. Run the indexing script

```bash
# Default: fastembed (all-MiniLM-L6-v2) + BM25 sparse + recursive chunking
bun code/ts/scripts/index-documents.ts

# LiteLLM (Qwen3-Embedding-8B) + BM25 sparse + recursive chunking
DENSE_METHOD=litellm bun code/ts/scripts/index-documents.ts

# Semantic chunking instead of recursive
CHUNKING_METHOD=semantic DENSE_METHOD=litellm bun code/ts/scripts/index-documents.ts

# Full override
DENSE_METHOD=litellm DENSE_MODEL=Qwen/Qwen3-Embedding-8B CHUNKING_METHOD=recursive bun code/ts/scripts/index-documents.ts
```

### 4. Verify in Qdrant

Check the collection was created:

- **Dashboard**: http://<your-server>:6333/dashboard#/collections
- **API**: `curl http://<your-server>:6333/collections`

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DENSE_METHOD` | `fastembed` | Dense embedding backend: `fastembed` (local ONNX) or `litellm` (NTNU HPC API) |
| `DENSE_MODEL` | method-dependent | Override dense model name (e.g. `BAAI/bge-large-en-v1.5` or `Qwen/Qwen3-Embedding-8B`) |
| `CHUNKING_METHOD` | `recursive` | Chunking strategy: `recursive` or `semantic` |
| `RETRIEVAL_LIMIT` | `5` | Number of top-K results returned at query time (eval scripts only) |

Default models per method:

| Method | Default Model | Vector Size |
| --- | --- | --- |
| `fastembed` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `litellm` | `Qwen/Qwen3-Embedding-8B` | 4096 |

## Architecture

```
┌─────────────────────────────────────────────┐
│  index-documents.ts (Bun)                   │
│  Orchestrates chunking → embedding → upsert │
└────────────┬──────────────┬─────────────────┘
             │              │
    ┌────────▼────────┐     │  Dense embedding
    │  Python FastAPI  │     │  (fastembed local
    │  localhost:8001  │     │   OR litellm API)
    │  - Chunking      │     │
    │  - BM25 sparse   │     │
    └─────────────────┘     │
                            │
              ┌─────────────▼─────────────────┐
              │  Qdrant (example.com:6333)     │
              │  - Creates collection           │
              │  - Upserts points with vectors  │
              │  - Tracks indexed files (_hashes)│
              └─────────────────────────────────┘
```

## Pipeline Steps

1. **Scan** `data/rag/processed/` for `**/*.md` files
2. **Skip** files already indexed (tracked in the `_hashes` companion collection)
3. **Chunk** each file using the configured method (via Python FastAPI at `localhost:8001`)
4. **Embed** chunks — Dense (fastembed or LiteLLM) + Sparse (BM25 via Python API) in parallel
5. **Upsert** points to Qdrant with automatic batching
6. **Mark** file as indexed in the `_hashes` collection

## Qdrant Collection Naming

Collections are **auto-generated** based on the embedding configuration. The naming formula is:

```
documents_{chunkingMethod}_{schemaHash}
```

Where `schemaHash` is the first 12 characters of a SHA-256 hash of the canonical embedding descriptor strings, sorted alphabetically and joined with `|`.

Each embedding descriptor has the format:
- **Dense/Late**: `{kind}:{method}:{modelName}:{vectorName}:{vectorSize}:cosine:{mode}`
- **Sparse**: `{kind}:{method}:{modelName}:{vectorName}:sparse:idf`

This means **changing the embedding model or chunking method automatically creates a new collection**, leaving the previous collection untouched.

### Example Collection Names

| Configuration | Collection Name |
| --- | --- |
| fastembed (MiniLM) + BM25, recursive | `documents_recursive_{hash1}` |
| litellm (Qwen3) + BM25, recursive | `documents_recursive_{hash2}` |
| litellm (Qwen3) + BM25, semantic | `documents_semantic_{hash2}` |
| fastembed (MiniLM) + BM25 + ColBERT, recursive | `documents_recursive_{hash3}` |

Each collection also has a companion `{collectionName}_hashes` collection that tracks which files have been indexed (for deduplication).

### Inspecting Collections

- **Qdrant Dashboard**: http://<your-server>:6333/dashboard#/collections
- **API**: `curl http://<your-server>:6333/collections`

## Running Evals After Indexing

RAG eval scripts use the same env vars for embedding configuration. The retrieval layer **must match** the embedding configuration used during indexing.

```bash
# Default (fastembed) — matches default indexing
bun evals/crew/rag.eval.ts

# LiteLLM — matches DENSE_METHOD=litellm indexing
DENSE_METHOD=litellm bun evals/crew/rag.eval.ts

# Override retrieval limit (default: 5)
RETRIEVAL_LIMIT=10 bun evals/crew/rag.eval.ts
```

## Python Alternatives

```bash
cd code/python
uv run python scripts/convert_pdfs.py              # Convert PDFs to markdown only
uv run python -m scripts.chunk_and_index            # Convert PDFs + chunk + index
uv run python -m scripts.index_markdown             # Index pre-processed markdown
```
