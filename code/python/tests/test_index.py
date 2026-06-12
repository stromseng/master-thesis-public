from __future__ import annotations

from uuid import uuid4

from ts_api.qdrant import (
    collection_exists,
    count,
    create_memory_client,
    delete_collection,
    index_chunks,
    search,
)
from ts_api.api_models import ChunkMetadata, DocumentChunk


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text="alpha beta gamma",
            metadata=ChunkMetadata(source="doc-1", headings=["Intro"]),
        ),
        DocumentChunk(
            text="delta epsilon zeta",
            metadata=ChunkMetadata(source="doc-2", headings=["Body"]),
        ),
    ]


def test_index_and_search_roundtrip() -> None:
    collection = f"test-collection-{uuid4().hex}"
    client = create_memory_client(collection_name=collection)
    chunks = make_chunks()

    _ = list(index_chunks(client, chunks, collection_name=collection))

    assert count(client, collection_name=collection) == len(chunks)

    results = search(client, "beta", collection_name=collection, limit=2)
    assert results
    assert any("beta" in result.chunk.text for result in results)


def test_delete_collection() -> None:
    collection = f"test-collection-{uuid4().hex}"
    client = create_memory_client(collection_name=collection)

    assert collection_exists(client, collection_name=collection)
    delete_collection(client, collection_name=collection)
    assert not collection_exists(client, collection_name=collection)
