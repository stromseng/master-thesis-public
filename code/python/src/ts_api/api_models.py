"""Shared data models for document processing pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""

    source: str
    headings: list[str]
    captions: list[str] | None = None
    page_numbers: list[int] | None = None


class DocumentChunkPayload(TypedDict, total=False):
    """Serialized payload for a document chunk."""

    text: str
    source: str
    headings: list[str]
    captions: list[str] | None
    page_numbers: list[int] | None


@dataclass
class DocumentChunk:
    """A chunk of a document with its text and metadata."""

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> DocumentChunkPayload:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "source": self.metadata.source,
            "headings": self.metadata.headings,
            "captions": self.metadata.captions,
            "page_numbers": self.metadata.page_numbers,
        }

    @staticmethod
    def from_dict(data: DocumentChunkPayload | Mapping[str, object]) -> DocumentChunk:
        """Create a DocumentChunk from a dictionary."""
        payload: Mapping[str, object] = data
        captions_value = payload.get("captions")
        page_numbers_value = payload.get("page_numbers")
        headings_value = payload.get("headings")
        captions = (
            [str(item) for item in cast(list[object], captions_value)]
            if isinstance(captions_value, list)
            else None
        )
        page_numbers = (
            [int(str(item)) for item in cast(list[object], page_numbers_value)]
            if isinstance(page_numbers_value, list)
            else None
        )
        headings = (
            [str(item) for item in cast(list[object], headings_value)]
            if isinstance(headings_value, list)
            else []
        )
        return DocumentChunk(
            text=str(payload.get("text", "")),
            metadata=ChunkMetadata(
                source=str(payload.get("source", "")),
                headings=headings,
                captions=captions,
                page_numbers=page_numbers,
            ),
        )
