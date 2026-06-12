## Python Source

This is the Python layer of the maritime RAG system. It provides FastAPI endpoints for embeddings, chunking, and Qdrant search, consumed by the TypeScript orchestration layer.

### Setup

```bash
cd code/python
uv sync
```

### Key modules

| Module | Purpose |
|--------|---------|
| `ts_api/` | FastAPI sidecar serving endpoints for the TS layer |
| `examscrapers/` | Dataset scrapers (Shititong, Coast Guard, CrewCN, etc.) |
| `idun/` | IDUN HPC cluster CLI and job management |
| `utils/` | Shared utilities |

### Running scripts

Scripts are Python modules -- run them with `uv run -m` from `code/python/`:

```bash
uv run python -m scripts.index_markdown      # Index processed markdown to Qdrant
uv run python -m scripts.chunk_and_index      # Convert PDFs, chunk, and index
```
