FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy workspace root files + member pyproject for dependency install
COPY pyproject.toml uv.lock ./
COPY code/python/pyproject.toml code/python/

# Install dependencies (cached layer)
RUN uv sync --frozen --no-install-project --no-dev --package thesis

# Copy source
COPY code/python/src/ code/python/src/

WORKDIR /app/code/python

EXPOSE 8001

CMD ["uv", "run", "--package", "thesis", "uvicorn", "ts_api:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "src"]
