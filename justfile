# Python recipies
mod py 'code/python'
# Typescript recipies
mod ts 'code/ts'

# Run all checks (Python + TypeScript)
[working-directory: "./"]
check:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "Running Python checks..."
    (cd code/python && just check)

    echo "Running TypeScript checks..."
    (cd code/ts && just check)

# Format both Python and TypeScript
[working-directory: "./"]
fmt:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "Formatting Python code..."
    (cd code/python && just fmt)

    echo "Formatting TypeScript code..."
    (cd code/ts && just fmt)

[working-directory: './code/python']
idun *args:
    PYTHONPATH=src uv run python -m idun {{args}}


# Run all dev services (docker services + TS client watcher)
dev:
    #!/usr/bin/env bash
    set -euo pipefail

    cleanup() {
        echo ""
        echo "Shutting down..."
        kill $watch_pid 2>/dev/null || true
        docker compose down
        echo "Done."
    }
    trap cleanup EXIT

    # Start all services in background (includes API via docker compose)
    docker compose up --build &
    docker_pid=$!

    (cd code/ts && bun run gen:api-client:watch) &
    watch_pid=$!

    # Wait for any process to exit
    wait

dev-python:
    #!/usr/bin/env bash
    set -euo pipefail
    cleanup() {
        echo ""
        echo "Shutting down..."
        kill $api_pid $watch_pid 2>/dev/null || true
        echo "Done."
    }
    trap cleanup EXIT

    (cd code/python && uv run uvicorn ts_api:app --reload --port 8001 --app-dir src) &
    api_pid=$!

    (cd code/ts && bun run gen:api-client:watch) &
    watch_pid=$!

    # Wait for any process to exit
    wait
