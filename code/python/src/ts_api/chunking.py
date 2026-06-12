"""Chunking models and factory functions for multi-method text chunking."""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# Request models for each chunking method
class FastChunkRequest(BaseModel):
    method: Literal["fast"] = "fast"
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(4096, ge=100)  # bytes
    delimiters: str = Field("\n.?")


class RecursiveChunkRequest(BaseModel):
    method: Literal["recursive"] = "recursive"
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(2048, ge=100)  # tokens
    min_characters_per_chunk: int = Field(24, ge=1)


class SemanticChunkRequest(BaseModel):
    method: Literal["semantic"] = "semantic"
    content: str = Field(..., min_length=1)
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    chunk_size: int = Field(2048, ge=100)
    similarity_window: int = Field(3, ge=1)
    skip_window: int = Field(0, ge=0)


class LateChunkRequest(BaseModel):
    method: Literal["late"] = "late"
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(2048, ge=100)
    min_characters_per_chunk: int = Field(24, ge=1)


class NeuralChunkRequest(BaseModel):
    method: Literal["neural"] = "neural"
    content: str = Field(..., min_length=1)
    min_characters_per_chunk: int = Field(10, ge=1)


class SlumberChunkRequest(BaseModel):
    method: Literal["slumber"] = "slumber"
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(1024, ge=100)
    candidate_size: int = Field(128, ge=1)
    min_characters_per_chunk: int = Field(24, ge=1)


# Discriminated union of all chunk request types
ChunkRequest = Annotated[
    FastChunkRequest
    | RecursiveChunkRequest
    | SemanticChunkRequest
    | LateChunkRequest
    | NeuralChunkRequest
    | SlumberChunkRequest,
    Field(discriminator="method"),
]


# Response models
class DocumentChunkResponse(BaseModel):
    text: str
    start_index: int
    end_index: int
    token_count: int
    context: str | None = None


class ChunkResponse(BaseModel):
    chunks: list[DocumentChunkResponse]


# Chunker cache for lazy initialization
_chunker_cache: dict[str, Any] = {}


def get_fast_chunker(chunk_size: int, delimiters: str) -> Any:
    """Get or create a FastChunker with the given parameters."""
    from chonkie import FastChunker

    key = f"fast:{chunk_size}:{delimiters}"
    if key not in _chunker_cache:
        _chunker_cache[key] = FastChunker(chunk_size=chunk_size, delimiters=delimiters)
    return _chunker_cache[key]


def get_recursive_chunker(chunk_size: int, min_chars: int) -> Any:
    """Get or create a RecursiveChunker with the given parameters."""
    from chonkie import RecursiveChunker

    key = f"recursive:{chunk_size}:{min_chars}"
    if key not in _chunker_cache:
        _chunker_cache[key] = RecursiveChunker(
            chunk_size=chunk_size,
            min_characters_per_chunk=min_chars,
        )
    return _chunker_cache[key]


def get_semantic_chunker(
    threshold: float, chunk_size: int, similarity_window: int, skip_window: int
) -> Any:
    """Get or create a SemanticChunker with the given parameters."""
    from chonkie import SemanticChunker

    key = f"semantic:{threshold}:{chunk_size}:{similarity_window}:{skip_window}"
    if key not in _chunker_cache:
        _chunker_cache[key] = SemanticChunker(
            embedding_model="minishlab/potion-base-32M",
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            skip_window=skip_window,
        )
    return _chunker_cache[key]


def get_late_chunker(chunk_size: int, min_chars: int) -> Any:
    """Get or create a LateChunker with the given parameters."""
    from chonkie import LateChunker

    key = f"late:{chunk_size}:{min_chars}"
    if key not in _chunker_cache:
        _chunker_cache[key] = LateChunker(
            embedding_model="nomic-ai/modernbert-embed-base",
            chunk_size=chunk_size,
            min_characters_per_chunk=min_chars,
        )
    return _chunker_cache[key]


def get_neural_chunker(min_chars: int) -> Any:
    """Get or create a NeuralChunker with the given parameters."""
    from chonkie import NeuralChunker

    key = f"neural:{min_chars}"
    if key not in _chunker_cache:
        _chunker_cache[key] = NeuralChunker(
            model="mirth/chonky_modernbert_base_1",
            device_map="cpu",
            min_characters_per_chunk=min_chars,
        )
    return _chunker_cache[key]


def get_slumber_chunker(chunk_size: int, candidate_size: int, min_chars: int) -> Any:
    """Get or create a SlumberChunker with the given parameters."""
    from chonkie import SlumberChunker
    from chonkie.genie import OpenAIGenie

    key = f"slumber:{chunk_size}:{candidate_size}:{min_chars}"
    if key not in _chunker_cache:
        genie = OpenAIGenie(
            model="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://idun.hpc.ntnu.no/litellm/v1"
            ),
        )
        _chunker_cache[key] = SlumberChunker(
            genie=genie,
            chunk_size=chunk_size,
            candidate_size=candidate_size,
            min_characters_per_chunk=min_chars,
        )
    return _chunker_cache[key]
