import { Effect, Layer, Option, Schema } from "effect";
import { describe, expect, it } from "@effect/vitest";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Chunking } from "../src/services/Chunking";
import { EmbeddingProvider } from "../src/services/embeddings";
import {
  CollectionSchemaMismatchError,
  DocumentChunkPayload,
  NoEmbeddingServicesConfiguredError,
  Qdrant,
  QDRANT_LOCALHOST_URL,
  getCollectionName,
} from "../src/services/Qdrant";
import { makeDenseTestService } from "./support/layers";

class QdrantCollectionSetupError extends Schema.TaggedError<QdrantCollectionSetupError>()(
  "QdrantCollectionSetupError",
  {
    collectionName: Schema.String,
    url: Schema.String,
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

const deleteCollectionIfExists = (collectionName: string) =>
  Effect.tryPromise({
    try: async () => {
      const client = new QdrantClient({ url: QDRANT_LOCALHOST_URL });
      await client.deleteCollection(collectionName);
    },
    catch: () => undefined,
  }).pipe(Effect.orDie);

describe("Qdrant localhost integration", () => {
  it.scopedLive("fails when no embedding services are configured", () =>
    Effect.gen(function* () {
      const emptyProvider = Layer.succeed(
        EmbeddingProvider,
        EmbeddingProvider.of({
          dense: Option.none(),
          sparse: Option.none(),
          late: Option.none(),
        }),
      );
      const error = yield* Effect.flip(getCollectionName).pipe(
        Effect.provide(Layer.mergeAll(Chunking.Test, emptyProvider)),
      );
      expect(error._tag).toBe("NoEmbeddingServicesConfiguredError");
      expect(error).toBeInstanceOf(NoEmbeddingServicesConfiguredError);
    }),
  );

  it.scopedLive("fails fast on existing collection schema mismatch", () =>
    Effect.gen(function* () {
      const suffix = crypto.randomUUID().slice(0, 8);
      const dependencyLayer = Layer.mergeAll(
        Chunking.Test,
        EmbeddingProvider.make(makeDenseTestService(suffix), undefined, undefined),
      );
      const qdrantLayer = Qdrant.localhost.pipe(Layer.provide(dependencyLayer));
      yield* Effect.gen(function* () {
        const qdrant = yield* Qdrant;
        yield* qdrant.health();
      }).pipe(Effect.provide(qdrantLayer));

      const collectionName = yield* getCollectionName.pipe(Effect.provide(dependencyLayer));

      yield* deleteCollectionIfExists(collectionName);
      yield* Effect.addFinalizer(() => deleteCollectionIfExists(collectionName));

      yield* Effect.tryPromise({
        try: async () => {
          const client = new QdrantClient({ url: QDRANT_LOCALHOST_URL });
          await client.createCollection(collectionName, {
            vectors: {
              [`dense-test-${suffix}`]: {
                size: 8,
                distance: "Cosine",
              },
            },
          });
        },
        catch: (cause) =>
          new QdrantCollectionSetupError({
            collectionName,
            url: QDRANT_LOCALHOST_URL,
            reason: `Unable to create test collection ${collectionName}`,
            cause,
          }),
      });

      const error = yield* Effect.scoped(Effect.flip(Layer.build(qdrantLayer)));

      expect(error).toBeInstanceOf(CollectionSchemaMismatchError);
    }),
  );

  it.scopedLive("creates schema-versioned collection and stores chunk_key payload", () =>
    Effect.gen(function* () {
      const suffix = crypto.randomUUID().slice(0, 8);
      const dependencyLayer = Layer.mergeAll(
        Chunking.Test,
        EmbeddingProvider.make(makeDenseTestService(suffix), undefined, undefined),
      );
      const qdrantLayer = Qdrant.localhost.pipe(Layer.provide(dependencyLayer));
      const fullLayer = Layer.mergeAll(qdrantLayer, dependencyLayer);
      yield* Effect.gen(function* () {
        const qdrant = yield* Qdrant;
        yield* qdrant.health();
      }).pipe(Effect.provide(qdrantLayer));

      const collectionName = yield* getCollectionName.pipe(Effect.provide(dependencyLayer));
      yield* deleteCollectionIfExists(collectionName);
      yield* Effect.addFinalizer(() => deleteCollectionIfExists(collectionName));

      expect(/^documents_test_[a-f0-9]{12}$/.test(collectionName)).toBe(true);

      const chunk = new DocumentChunkPayload({
        text: "Maritime safety bulletin",
        start_index: 0,
        end_index: 24,
        token_count: 3,
        context: null,
        source: `source-${suffix}.md`,
      });

      const points = yield* Effect.gen(function* () {
        const qdrant = yield* Qdrant;
        yield* qdrant.indexChunks([chunk]);

        return yield* qdrant.withClient((client) =>
          client.scroll(collectionName, {
            limit: 10,
            with_payload: true,
            with_vector: false,
          }),
        );
      }).pipe(Effect.provide(fullLayer));

      expect(points.points?.length).toBe(1);

      const payload = (points.points?.[0]?.payload ?? {}) as Record<string, unknown>;
      expect(typeof payload.chunk_key).toBe("string");
      expect((payload.chunk_key as string).length).toBe(64);
      expect(payload.source).toBe(`source-${suffix}.md`);
    }),
  );
});
