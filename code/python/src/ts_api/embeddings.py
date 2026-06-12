"""Embedding models, config resolution, and model factories for TS API."""

from __future__ import annotations

from typing import Literal

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel, Field


class FastEmbedDenseConfig(BaseModel):
    method: Literal["fastembed"] = "fastembed"
    model_name: str
    vector_name: str


class FastEmbedSparseConfig(BaseModel):
    method: Literal["fastembed"] = "fastembed"
    model_name: str
    vector_name: str


class FastEmbedLateConfig(BaseModel):
    method: Literal["fastembed"] = "fastembed"
    model_name: str
    vector_name: str


DenseEmbeddingConfig = FastEmbedDenseConfig
SparseEmbeddingConfig = FastEmbedSparseConfig
LateEmbeddingConfig = FastEmbedLateConfig


class DenseVectorConfig(BaseModel):
    model_name: str
    vector_name: str
    vector_size: int


class SparseVectorConfig(BaseModel):
    model_name: str
    vector_name: str


class LateVectorConfig(BaseModel):
    model_name: str
    vector_name: str
    vector_size: int


class EmbeddingConfigResponse(BaseModel):
    dense: DenseVectorConfig
    sparse: SparseVectorConfig
    late: LateVectorConfig
    collection_name_prefix: str


class DenseEmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    config: DenseEmbeddingConfig


class BatchDenseEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    config: DenseEmbeddingConfig


class SparseEmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    config: SparseEmbeddingConfig


class BatchSparseEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    config: SparseEmbeddingConfig


class LateEmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    config: LateEmbeddingConfig


class BatchLateEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    config: LateEmbeddingConfig


class DenseEmbedResponse(BaseModel):
    embedding: list[float]
    model: str


class BatchDenseEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str


class SparseEmbedResponse(BaseModel):
    indices: list[int]
    values: list[float]
    model: str


class SparseEmbedding(BaseModel):
    indices: list[int]
    values: list[float]


class BatchSparseEmbedResponse(BaseModel):
    embeddings: list[SparseEmbedding]
    model: str


class LateEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str


class BatchLateEmbedResponse(BaseModel):
    embeddings: list[list[list[float]]]
    model: str


_dense_model_cache: dict[str, TextEmbedding] = {}
_sparse_model_cache: dict[str, SparseTextEmbedding] = {}
_late_model_cache: dict[str, LateInteractionTextEmbedding] = {}
_vector_size_cache: dict[str, int] = {}


def get_dense_model(config: DenseEmbeddingConfig) -> TextEmbedding:
    key = f"{config.method}:{config.model_name}"
    if key not in _dense_model_cache:
        _dense_model_cache[key] = TextEmbedding(config.model_name)
    return _dense_model_cache[key]


def get_sparse_model(config: SparseEmbeddingConfig) -> SparseTextEmbedding:
    key = f"{config.method}:{config.model_name}"
    if key not in _sparse_model_cache:
        _sparse_model_cache[key] = SparseTextEmbedding(config.model_name)
    return _sparse_model_cache[key]


def get_late_model(config: LateEmbeddingConfig) -> LateInteractionTextEmbedding:
    key = f"{config.method}:{config.model_name}"
    if key not in _late_model_cache:
        _late_model_cache[key] = LateInteractionTextEmbedding(config.model_name)
    return _late_model_cache[key]


def get_dense_vector_size(config: DenseEmbeddingConfig) -> int:
    key = f"dense:{config.method}:{config.model_name}"
    if key not in _vector_size_cache:
        model = get_dense_model(config)
        embeddings = list(model.embed(["vector size probe"]))
        if not embeddings:
            msg = (
                f"Unable to determine dense vector size for model: {config.model_name}"
            )
            raise ValueError(msg)
        _vector_size_cache[key] = int(embeddings[0].shape[0])
    return _vector_size_cache[key]


def get_late_vector_size(config: LateEmbeddingConfig) -> int:
    key = f"late:{config.method}:{config.model_name}"
    if key not in _vector_size_cache:
        model = get_late_model(config)
        embeddings = list(model.embed(["vector size probe"]))
        if not embeddings or len(embeddings[0]) == 0:
            msg = f"Unable to determine late vector size for model: {config.model_name}"
            raise ValueError(msg)
        _vector_size_cache[key] = int(embeddings[0][0].shape[0])
    return _vector_size_cache[key]


def resolve_dense_embedding_config(payload: DenseEmbeddingConfig) -> DenseVectorConfig:
    return DenseVectorConfig(
        model_name=payload.model_name,
        vector_name=payload.vector_name,
        vector_size=get_dense_vector_size(payload),
    )


def resolve_sparse_embedding_config(
    payload: SparseEmbeddingConfig,
) -> SparseVectorConfig:
    return SparseVectorConfig(
        model_name=payload.model_name,
        vector_name=payload.vector_name,
    )


def resolve_late_embedding_config(payload: LateEmbeddingConfig) -> LateVectorConfig:
    return LateVectorConfig(
        model_name=payload.model_name,
        vector_name=payload.vector_name,
        vector_size=get_late_vector_size(payload),
    )


def resolve_embedding_config(
    dense: DenseEmbeddingConfig,
    sparse: SparseEmbeddingConfig,
    late: LateEmbeddingConfig,
    collection_name_prefix: str = "documents",
) -> EmbeddingConfigResponse:
    return EmbeddingConfigResponse(
        dense=resolve_dense_embedding_config(dense),
        sparse=resolve_sparse_embedding_config(sparse),
        late=resolve_late_embedding_config(late),
        collection_name_prefix=collection_name_prefix,
    )
