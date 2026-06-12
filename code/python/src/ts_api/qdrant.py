"""Qdrant indexing and search operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

from qdrant_client import QdrantClient, models

from ts_api.api_models import DocumentChunk


@dataclass
class SearchResult:
    """A search result with score and chunk data."""

    score: float
    chunk: DocumentChunk


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding configuration for hybrid search."""

    dense_model_name: str
    sparse_model_name: str
    late_model_name: str
    dense_vector_name: str
    sparse_vector_name: str
    late_vector_name: str
    dense_vector_size: int
    late_vector_size: int


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
    sparse_model_name="Qdrant/bm25",
    late_model_name="colbert-ir/colbertv2.0",
    dense_vector_name="all-MiniLM-L6-v2",
    sparse_vector_name="bm25",
    late_vector_name="colbertv2.0",
    dense_vector_size=384,
    late_vector_size=128,
)
DEFAULT_COLLECTION_BASE = "documents"


def _normalize_collection_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
        .lower()
    )


def get_embedding_config(
    embedding_config: EmbeddingConfig | None = None,
) -> EmbeddingConfig:
    """Return an embedding config, falling back to defaults."""
    return embedding_config or DEFAULT_EMBEDDING_CONFIG


def get_collection_name(
    base_name: str = DEFAULT_COLLECTION_BASE,
    embedding_config: EmbeddingConfig | None = None,
) -> str:
    """Build a collection name from embedding config."""
    config = get_embedding_config(embedding_config)
    suffix = "_".join(
        [
            _normalize_collection_part(config.dense_vector_name),
            _normalize_collection_part(config.sparse_vector_name),
            _normalize_collection_part(config.late_vector_name),
        ]
    )
    return f"{base_name}_{suffix}"


DEFAULT_COLLECTION = get_collection_name()


def create_memory_client(
    collection_name: str | None = None,
    ensure_collection_exists: bool = True,
    embedding_config: EmbeddingConfig | None = None,
) -> QdrantClient:
    """Create an in-memory Qdrant client."""
    config = get_embedding_config(embedding_config)
    resolved_collection = collection_name or get_collection_name(
        embedding_config=config
    )
    client = QdrantClient(":memory:")
    if ensure_collection_exists:
        ensure_collection(
            client,
            collection_name=resolved_collection,
            embedding_config=config,
        )
    return client


def create_localhost_client(
    collection_name: str | None = None,
    ensure_collection_exists: bool = True,
    url: str = "http://localhost:6333",
    api_key: str | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> QdrantClient:
    """Create a localhost Qdrant client."""
    config = get_embedding_config(embedding_config)
    resolved_collection = collection_name or get_collection_name(
        embedding_config=config
    )
    client = QdrantClient(url=url, api_key=api_key)
    if ensure_collection_exists:
        ensure_collection(
            client,
            collection_name=resolved_collection,
            embedding_config=config,
        )
    return client


def create_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_config: EmbeddingConfig | None = None,
) -> None:
    """Create the hybrid collection with dense, sparse, late vectors."""
    config = get_embedding_config(embedding_config)
    _ = client.create_collection(
        collection_name=collection_name,
        vectors_config={
            config.dense_vector_name: models.VectorParams(
                size=client.get_embedding_size(config.dense_model_name),
                distance=models.Distance.COSINE,
            ),
            config.late_vector_name: models.VectorParams(
                size=client.get_embedding_size(config.late_model_name),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            config.sparse_vector_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )


def collection_exists(
    client: QdrantClient,
    collection_name: str,
) -> bool:
    """Check if the collection exists."""
    collections = client.get_collections().collections
    return any(c.name == collection_name for c in collections)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_config: EmbeddingConfig | None = None,
) -> None:
    """Create collection if it doesn't exist."""
    if not collection_exists(client, collection_name=collection_name):
        create_collection(
            client,
            collection_name=collection_name,
            embedding_config=embedding_config,
        )


def delete_collection(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """Delete the collection if it exists."""
    if collection_exists(client, collection_name=collection_name):
        _ = client.delete_collection(collection_name)


def index_chunks(
    client: QdrantClient,
    chunks: list[DocumentChunk],
    collection_name: str,
    batch_size: int = 100,
    embedding_config: EmbeddingConfig | None = None,
) -> Iterator[tuple[str, int, int]]:
    """Index chunks yielding progress updates.

    Yields:
        Tuples of (stage, current, total) for progress tracking.
        Stages: "embedding", "uploading"
    """
    if not chunks:
        return

    config = get_embedding_config(embedding_config)
    total = len(chunks)
    points: list[models.PointStruct] = []
    uploaded = 0

    for i, chunk in enumerate(chunks, start=1):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                payload=dict(chunk.to_dict()),
                vector={
                    config.dense_vector_name: models.Document(
                        text=chunk.text,
                        model=config.dense_model_name,
                    ),
                    config.sparse_vector_name: models.Document(
                        text=chunk.text,
                        model=config.sparse_model_name,
                    ),
                    config.late_vector_name: models.Document(
                        text=chunk.text,
                        model=config.late_model_name,
                    ),
                },
            )
        )
        yield ("embedding", i, total)

        if len(points) >= batch_size:
            _ = client.upsert(collection_name=collection_name, points=points)
            uploaded += len(points)
            yield ("uploading", uploaded, total)
            points = []

    if points:
        _ = client.upsert(collection_name=collection_name, points=points)
        uploaded += len(points)
        yield ("uploading", uploaded, total)


def search(
    client: QdrantClient,
    query: str,
    collection_name: str,
    limit: int = 5,
    embedding_config: EmbeddingConfig | None = None,
    prefetch_limit: int = 20,
) -> list[SearchResult]:
    """Search for documents using hybrid retrieval with reranking."""
    config = get_embedding_config(embedding_config)
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=config.dense_model_name),
                using=config.dense_vector_name,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=config.sparse_model_name),
                using=config.sparse_vector_name,
                limit=prefetch_limit,
            ),
        ],
        query=models.Document(text=query, model=config.late_model_name),
        using=config.late_vector_name,
        with_payload=True,
        limit=limit,
    )

    search_results: list[SearchResult] = []
    for point in results.points:
        if point.payload is None:
            continue
        chunk = DocumentChunk.from_dict(point.payload)
        search_results.append(SearchResult(score=float(point.score), chunk=chunk))

    return search_results


def count(
    client: QdrantClient,
    collection_name: str,
) -> int:
    """Get the number of points in the collection."""
    info = client.get_collection(collection_name)
    return int(info.points_count or 0)
