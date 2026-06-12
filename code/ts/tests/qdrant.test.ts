import { describe, expect, it } from "@effect/vitest";
import { Effect, Layer } from "effect";
import type { QdrantClient } from "@qdrant/js-client-rest";
import { Qdrant, QdrantTestCollection } from "../src/services/Qdrant";
import type { CollectionConfig } from "../src/services/Qdrant";
import { Chunking } from "../src/services/Chunking";
import { EmbeddingProvider } from "../src/services/embeddings";
import { makeDenseTestService } from "./support/layers";

describe("qdrant integration", () => {
  const testCollectionConfig = {
    vectors: {
      size: 2,
      distance: "Cosine",
    },
  } satisfies CollectionConfig;

  const embeddingDependencies = Layer.mergeAll(
    Chunking.Test,
    EmbeddingProvider.make(makeDenseTestService("qdrant-test"), undefined, undefined),
  );
  const qdrantLayer = Qdrant.localhost.pipe(Layer.provide(embeddingDependencies));
  const testCollectionLayer = QdrantTestCollection.layer({
    config: testCollectionConfig,
  }).pipe(Layer.provide(qdrantLayer));
  const testLayer = Layer.mergeAll(embeddingDependencies, qdrantLayer, testCollectionLayer);

  it.scopedLive("supports basic ops", () =>
    Effect.gen(function* () {
      const pointsCount = yield* Effect.gen(function* () {
        const collectionName = yield* QdrantTestCollection;
        const qdrant = yield* Qdrant;
        yield* qdrant.health();

        yield* qdrant.withClient((client: QdrantClient) =>
          client.upsert(collectionName, {
            wait: true,
            points: [
              {
                id: 1,
                vector: [0.1, 0.2],
                payload: { city: "Trondheim" },
              },
            ],
          }),
        );

        const info = yield* qdrant.withClient((client: QdrantClient) =>
          client.getCollection(collectionName),
        );

        return info.points_count ?? 0;
      }).pipe(Effect.provide(testLayer), Effect.orDie);

      expect(pointsCount).toBe(1);
    }),
  );
});
