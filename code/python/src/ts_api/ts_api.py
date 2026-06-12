from __future__ import annotations

from fastapi import FastAPI

from ts_api.qdrant import DEFAULT_COLLECTION_BASE, DEFAULT_EMBEDDING_CONFIG
from ts_api.chunking import (
    ChunkRequest,
    ChunkResponse,
    DocumentChunkResponse,
    FastChunkRequest,
    LateChunkRequest,
    NeuralChunkRequest,
    RecursiveChunkRequest,
    SemanticChunkRequest,
    SlumberChunkRequest,
    get_fast_chunker,
    get_late_chunker,
    get_neural_chunker,
    get_recursive_chunker,
    get_semantic_chunker,
    get_slumber_chunker,
)
from ts_api.embeddings import (
    BatchDenseEmbedRequest,
    BatchDenseEmbedResponse,
    BatchLateEmbedRequest,
    BatchLateEmbedResponse,
    BatchSparseEmbedRequest,
    BatchSparseEmbedResponse,
    DenseEmbeddingConfig,
    DenseEmbedRequest,
    DenseEmbedResponse,
    DenseVectorConfig,
    EmbeddingConfigResponse,
    LateEmbeddingConfig,
    LateEmbedRequest,
    LateEmbedResponse,
    LateVectorConfig,
    SparseEmbedding,
    SparseEmbeddingConfig,
    SparseEmbedRequest,
    SparseEmbedResponse,
    SparseVectorConfig,
    get_dense_model,
    get_late_model,
    get_sparse_model,
    resolve_dense_embedding_config,
    resolve_embedding_config,
    resolve_late_embedding_config,
    resolve_sparse_embedding_config,
)

app = FastAPI(title="TS API Demo")


@app.post("/embed/dense", response_model=DenseEmbedResponse, operation_id="embedDense")
def embed_dense(payload: DenseEmbedRequest) -> DenseEmbedResponse:
    """Generate dense embedding for text."""
    model = get_dense_model(payload.config)
    embeddings = list(model.embed([payload.text]))
    return DenseEmbedResponse(
        embedding=embeddings[0].tolist(),
        model=payload.config.model_name,
    )


@app.post(
    "/embed/sparse", response_model=SparseEmbedResponse, operation_id="embedSparse"
)
def embed_sparse(payload: SparseEmbedRequest) -> SparseEmbedResponse:
    """Generate sparse (BM25) embedding for text."""
    model = get_sparse_model(payload.config)
    embeddings = list(model.embed([payload.text]))
    sparse = embeddings[0]
    return SparseEmbedResponse(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
        model=payload.config.model_name,
    )


@app.post("/embed/late", response_model=LateEmbedResponse, operation_id="embedLate")
def embed_late(payload: LateEmbedRequest) -> LateEmbedResponse:
    """Generate late interaction (ColBERT) embedding for text."""
    model = get_late_model(payload.config)
    embeddings = list(model.embed([payload.text]))
    return LateEmbedResponse(
        embeddings=[e.tolist() for e in embeddings[0]],
        model=payload.config.model_name,
    )


@app.post(
    "/embed/dense/batch",
    response_model=BatchDenseEmbedResponse,
    operation_id="embedDenseBatch",
)
def embed_dense_batch(payload: BatchDenseEmbedRequest) -> BatchDenseEmbedResponse:
    """Generate dense embeddings for multiple texts in a single batch."""
    model = get_dense_model(payload.config)
    embeddings = list(model.embed(payload.texts))
    return BatchDenseEmbedResponse(
        embeddings=[e.tolist() for e in embeddings],
        model=payload.config.model_name,
    )


@app.post(
    "/embed/sparse/batch",
    response_model=BatchSparseEmbedResponse,
    operation_id="embedSparseBatch",
)
def embed_sparse_batch(payload: BatchSparseEmbedRequest) -> BatchSparseEmbedResponse:
    """Generate sparse (BM25) embeddings for multiple texts in a single batch."""
    model = get_sparse_model(payload.config)
    embeddings = list(model.embed(payload.texts))
    return BatchSparseEmbedResponse(
        embeddings=[
            SparseEmbedding(indices=e.indices.tolist(), values=e.values.tolist())
            for e in embeddings
        ],
        model=payload.config.model_name,
    )


@app.post(
    "/embed/late/batch",
    response_model=BatchLateEmbedResponse,
    operation_id="embedLateBatch",
)
def embed_late_batch(payload: BatchLateEmbedRequest) -> BatchLateEmbedResponse:
    """Generate late interaction (ColBERT) embeddings for multiple texts in a single batch."""
    model = get_late_model(payload.config)
    embeddings = list(model.embed(payload.texts))
    return BatchLateEmbedResponse(
        embeddings=[[token.tolist() for token in e] for e in embeddings],
        model=payload.config.model_name,
    )


@app.post(
    "/embed/dense/resolve",
    response_model=DenseVectorConfig,
    operation_id="resolveDenseEmbeddingConfig",
)
def resolve_dense_embedding_config_endpoint(
    payload: DenseEmbeddingConfig,
) -> DenseVectorConfig:
    """Resolve dense embedding descriptor and vector size."""
    return resolve_dense_embedding_config(payload)


@app.post(
    "/embed/sparse/resolve",
    response_model=SparseVectorConfig,
    operation_id="resolveSparseEmbeddingConfig",
)
def resolve_sparse_embedding_config_endpoint(
    payload: SparseEmbeddingConfig,
) -> SparseVectorConfig:
    """Resolve sparse embedding descriptor."""
    return resolve_sparse_embedding_config(payload)


@app.post(
    "/embed/late/resolve",
    response_model=LateVectorConfig,
    operation_id="resolveLateEmbeddingConfig",
)
def resolve_late_embedding_config_endpoint(
    payload: LateEmbeddingConfig,
) -> LateVectorConfig:
    """Resolve late embedding descriptor and vector size."""
    return resolve_late_embedding_config(payload)


@app.get(
    "/embedding-config",
    response_model=EmbeddingConfigResponse,
    operation_id="getEmbeddingConfig",
)
def get_embedding_config_endpoint() -> EmbeddingConfigResponse:
    """Get the default resolved embedding configuration."""

    return resolve_embedding_config(
        dense=DenseEmbeddingConfig(
            method="fastembed",
            model_name=DEFAULT_EMBEDDING_CONFIG.dense_model_name,
            vector_name=DEFAULT_EMBEDDING_CONFIG.dense_vector_name,
        ),
        sparse=SparseEmbeddingConfig(
            method="fastembed",
            model_name=DEFAULT_EMBEDDING_CONFIG.sparse_model_name,
            vector_name=DEFAULT_EMBEDDING_CONFIG.sparse_vector_name,
        ),
        late=LateEmbeddingConfig(
            method="fastembed",
            model_name=DEFAULT_EMBEDDING_CONFIG.late_model_name,
            vector_name=DEFAULT_EMBEDDING_CONFIG.late_vector_name,
        ),
        collection_name_prefix=DEFAULT_COLLECTION_BASE,
    )


@app.post("/chunk", response_model=ChunkResponse, operation_id="chunkFile")
def chunk_file(payload: ChunkRequest) -> ChunkResponse:
    """Chunk text using the specified method."""
    match payload:
        case FastChunkRequest():
            chunker = get_fast_chunker(payload.chunk_size, payload.delimiters)
        case RecursiveChunkRequest():
            chunker = get_recursive_chunker(
                payload.chunk_size, payload.min_characters_per_chunk
            )
        case SemanticChunkRequest():
            chunker = get_semantic_chunker(
                payload.threshold,
                payload.chunk_size,
                payload.similarity_window,
                payload.skip_window,
            )
        case LateChunkRequest():
            chunker = get_late_chunker(
                payload.chunk_size, payload.min_characters_per_chunk
            )
        case NeuralChunkRequest():
            chunker = get_neural_chunker(payload.min_characters_per_chunk)
        case SlumberChunkRequest():
            chunker = get_slumber_chunker(
                payload.chunk_size,
                payload.candidate_size,
                payload.min_characters_per_chunk,
            )

    chunks = chunker.chunk(payload.content)
    return ChunkResponse(
        chunks=[
            DocumentChunkResponse(
                text=chunk.text,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                token_count=chunk.token_count,
                context=chunk.context,
            )
            for chunk in chunks
        ]
    )
