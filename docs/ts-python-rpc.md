# TS ↔ Python API bridge

## What it does
- Exposes a FastAPI sidecar in `code/python/src/ts_api/ts_api.py`
- Generates a typesafe TS client with `@hey-api/openapi-ts`

## Commands
- Start FastAPI server manually: `cd code/python && uv run uvicorn ts_api:app --reload --port 8001 --app-dir src`
- Watch + regenerate client manually: `cd code/ts && bun run gen:api-client:watch`
- Run both together from repo root: `just dev-python`
- One-off client generation: `bun run gen:api-client` (from `code/ts/`)

## How it works
- FastAPI serves OpenAPI at `http://localhost:8001/openapi.json`
- `@hey-api/openapi-ts` pulls the schema and writes to `code/ts/src/generated/python-api`
- `chokidar-cli` watches `code/python/src/**/*.py` and reruns codegen on change
