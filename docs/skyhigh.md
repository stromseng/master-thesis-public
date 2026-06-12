# Skyhigh Hosting

The project uses a hosted VM (`<your-server>`) to run Qdrant and Phoenix as shared services. This is the default backend for RAG evaluations — you do **not** need to run Qdrant or Phoenix locally unless you want to.

## Hosted Services

| Service | URL | Purpose |
| --- | --- | --- |
| **Qdrant** | `http://<your-server>:6333` | Vector database for RAG retrieval |
| **Qdrant Dashboard** | `http://<your-server>:6333/dashboard#/collections` | Web UI for inspecting collections |
| **Phoenix** | `http://<your-server>:6006` | Experiment tracking and observability |
| **Phoenix GRPC** | `http://<your-server>:4317` | OpenTelemetry trace collection |

## SSH Access

Configure an SSH alias (e.g. `<your-ssh-alias>`) for the server in `~/.ssh/config`:

```bash
ssh <your-ssh-alias>           # connect to server
ssh -A <your-ssh-alias>        # forward SSH keys (for git pull on server)
```

## Server Management

The server runs the same `docker-compose.yml` from the repo root:

```bash
ssh -A <your-ssh-alias>
cd <your-home-dir>/master-thesis
sudo -s
docker compose up -d     # start/restart services
docker compose logs -f   # view logs
docker compose down      # stop services
```

## Local vs. Remote

The `.env.template` includes placeholders for `QDRANT_URL`, `PHOENIX_HOST`, and `PHOENIX_COLLECTOR_ENDPOINT`. To run locally instead:

1. Start local services: `just dev` (runs Docker Compose with Qdrant + Phoenix)
2. Update `.env` to use localhost:
   ```
   QDRANT_URL=http://127.0.0.1:6333
   PHOENIX_HOST=http://127.0.0.1:6006
   PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
   ```

In the TypeScript code, `Qdrant.skyhigh` and `Qdrant.localhost` are static layers that hardcode example and local URLs. The eval and indexing scripts currently use `Qdrant.skyhigh`, so replace the example URL before running against a real remote server.
