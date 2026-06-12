import { QdrantClient } from "@qdrant/js-client-rest";
import { Chunk, Config, Context, Effect, Layer, Option, Schema, Schedule, Stream } from "effect";
import { Chunking } from "./Chunking";
import { EmbeddingProvider, type EmbeddingProviderService, type SparseVector } from "./embeddings";
import { wrapClientCall } from "./utils/wrapClientCall";

const COLLECTION_NAME_PREFIX = "documents";
export const QDRANT_LOCALHOST_URL = "http://127.0.0.1:6333";
export const QDRANT_SKYHIGH_URL = "http://example.com:6333";

/**
 * Schema for document chunk payload - mirrors Chonkie's Chunk dataclass, except we add source file name.
 * Used for Qdrant point payloads.
 */

export class DocumentChunkPayload extends Schema.Class<DocumentChunkPayload>(
  "DocumentChunkPayload",
)({
  text: Schema.String,
  start_index: Schema.Number,
  end_index: Schema.Number,
  token_count: Schema.Number,
  context: Schema.NullOr(Schema.String),
  source: Schema.String,
  chunk_key: Schema.optional(Schema.String),
}) {}
/**
 * Schema for a Qdrant search result point with document chunk payload.
 */

export class QdrantSearchPoint extends Schema.Class<QdrantSearchPoint>("QdrantSearchPoint")({
  id: Schema.Union(Schema.String, Schema.Number),
  score: Schema.Number,
  payload: DocumentChunkPayload,
}) {}
/**
 * Schema for decoding Qdrant query response points.
 */

export const QdrantSearchPoints = Schema.Array(QdrantSearchPoint);

// ==========================================================================
// Errors
// ==========================================================================

export class QdrantError extends Schema.TaggedError<QdrantError>()("QdrantError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

export class NoEmbeddingServicesConfiguredError extends Schema.TaggedError<NoEmbeddingServicesConfiguredError>()(
  "NoEmbeddingServicesConfiguredError",
  {},
) {}

export class CollectionSchemaMismatchError extends Schema.TaggedError<CollectionSchemaMismatchError>()(
  "CollectionSchemaMismatchError",
  {
    collectionName: Schema.String,
    reason: Schema.String,
    expected: Schema.Unknown,
    actual: Schema.Unknown,
  },
) {}

export class CollectionIntrospectionError extends Schema.TaggedError<CollectionIntrospectionError>()(
  "CollectionIntrospectionError",
  {
    collectionName: Schema.String,
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class CollectionCreationError extends Schema.TaggedError<CollectionCreationError>()(
  "CollectionCreationError",
  {
    collectionName: Schema.String,
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

const formatDetails = (value: unknown) => {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
};

const previewDetails = (value: unknown) => {
  const text = formatDetails(value);
  return text.length > 500 ? `${text.slice(0, 500)}...` : text;
};

// ==========================================================================
// Indexing
// ==========================================================================

const EMBEDDING_BATCH_SIZE = 50; // Max texts per embedding API call
const EMBEDDING_BATCH_CONCURRENCY = 10; // Sliding window: start next batch as soon as one finishes

// Exponential backoff: 1s, 2s, 4s (3 retries)
const retrySchedule = Schedule.exponential("1 second").pipe(Schedule.intersect(Schedule.recurs(3)));

export class IndexingError extends Schema.TaggedError<IndexingError>()("IndexingError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

export interface IndexChunksOptions {
  readonly onProgress?: (processed: number, total: number) => Effect.Effect<void>;
}

type DensePointVector = number[];
type SparsePointVector = {
  indices: number[];
  values: number[];
};
type LatePointVector = number[][];
type PointVectorValue = DensePointVector | SparsePointVector | LatePointVector;
type PointVectors = Record<string, PointVectorValue>;

interface DensePointVectorEntry {
  readonly kind: "dense";
  readonly vectorName: string;
  readonly value: DensePointVector;
}

interface SparsePointVectorEntry {
  readonly kind: "sparse";
  readonly vectorName: string;
  readonly value: SparsePointVector;
}

interface LatePointVectorEntry {
  readonly kind: "late";
  readonly vectorName: string;
  readonly value: LatePointVector;
}

type PointVectorEntry = DensePointVectorEntry | SparsePointVectorEntry | LatePointVectorEntry;

const pointVectorsFromEntries = (entries: readonly PointVectorEntry[]): PointVectors => {
  const vectors: PointVectors = {};
  for (const entry of entries) {
    vectors[entry.vectorName] = entry.value;
  }
  return vectors;
};

interface DenseDescriptor {
  readonly kind: "dense";
  readonly method: string;
  readonly modelName: string;
  readonly vectorName: string;
  readonly vectorSize: number;
}

interface SparseDescriptor {
  readonly kind: "sparse";
  readonly method: string;
  readonly modelName: string;
  readonly vectorName: string;
}

interface LateDescriptor {
  readonly kind: "late";
  readonly method: string;
  readonly modelName: string;
  readonly vectorName: string;
  readonly vectorSize: number;
}

type EmbeddingDescriptor = DenseDescriptor | SparseDescriptor | LateDescriptor;

const embeddingDescriptorsFromProvider = (
  provider: EmbeddingProviderService,
): readonly EmbeddingDescriptor[] => {
  const { dense, sparse, late } = provider;
  const descriptors: EmbeddingDescriptor[] = [];
  if (Option.isSome(dense)) {
    descriptors.push({ kind: "dense", ...dense.value.descriptor });
  }
  if (Option.isSome(sparse)) {
    descriptors.push({ kind: "sparse", ...sparse.value.descriptor });
  }
  if (Option.isSome(late)) {
    descriptors.push({ kind: "late", ...late.value.descriptor });
  }
  return descriptors;
};

interface CollectionSchemaSpec {
  readonly vectors: Record<string, { size: number; distance: "Cosine"; multivector?: boolean }>;
  readonly sparseVectors: readonly string[];
}

const NormalizedVectorConfigSchema = Schema.Struct({
  size: Schema.Number,
  distance: Schema.Literal("Cosine"),
  multivector: Schema.optional(Schema.Boolean),
});

const NormalizedCollectionSchemaSpecSchema = Schema.Struct({
  vectors: Schema.Record({ key: Schema.String, value: NormalizedVectorConfigSchema }),
  sparseVectors: Schema.Array(Schema.String),
});

type NormalizedCollectionSchemaSpec = Schema.Schema.Type<
  typeof NormalizedCollectionSchemaSpecSchema
>;

const normalizedCollectionSchemaSpecEquivalence = Schema.equivalence(
  NormalizedCollectionSchemaSpecSchema,
);

const normalizeSchemaSpec = (schema: CollectionSchemaSpec): NormalizedCollectionSchemaSpec => ({
  vectors: Object.fromEntries(
    Object.entries(schema.vectors)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([name, config]) => [name, { ...config }]),
  ),
  sparseVectors: [...schema.sparseVectors].sort(),
});

const buildCollectionSchema = (
  descriptors: readonly EmbeddingDescriptor[],
): CollectionSchemaSpec => {
  const vectors: Record<string, { size: number; distance: "Cosine"; multivector?: boolean }> = {};
  const sparseVectors: string[] = [];

  for (const descriptor of descriptors) {
    if (descriptor.kind === "sparse") {
      sparseVectors.push(descriptor.vectorName);
      continue;
    }

    vectors[descriptor.vectorName] = {
      size: descriptor.vectorSize,
      distance: "Cosine",
      ...(descriptor.kind === "late" ? { multivector: true } : {}),
    };
  }

  return { vectors, sparseVectors };
};

const buildSchemaId = (descriptors: readonly EmbeddingDescriptor[]): string => {
  const canonical = descriptors
    .map((descriptor) => {
      if (descriptor.kind === "sparse") {
        return `${descriptor.kind}:${descriptor.method}:${descriptor.modelName}:${descriptor.vectorName}:sparse:idf`;
      }
      return `${descriptor.kind}:${descriptor.method}:${descriptor.modelName}:${descriptor.vectorName}:${descriptor.vectorSize}:cosine:${descriptor.kind === "late" ? "multivector-max-sim" : "single"}`;
    })
    .sort()
    .join("|");

  return new Bun.CryptoHasher("sha256").update(canonical).digest("hex").slice(0, 12);
};

const QdrantCollectionVectorConfigSchema = Schema.Struct({
  size: Schema.optional(Schema.Number),
  distance: Schema.optional(Schema.String),
  multivector_config: Schema.optional(Schema.Unknown),
});

const QdrantCollectionInfoSchema = Schema.Struct({
  config: Schema.optional(
    Schema.Struct({
      params: Schema.optional(
        Schema.Struct({
          vectors: Schema.optional(
            Schema.Record({ key: Schema.String, value: QdrantCollectionVectorConfigSchema }),
          ),
          sparse_vectors: Schema.optional(
            Schema.Record({ key: Schema.String, value: Schema.Unknown }),
          ),
        }),
      ),
    }),
  ),
});

const actualSchemaFromCollection = Effect.fn("Qdrant.actualSchemaFromCollection")(function* (
  collectionInfo: unknown,
) {
  const payload = yield* Schema.decodeUnknown(QdrantCollectionInfoSchema)(collectionInfo);
  const vectors = payload.config?.params?.vectors;
  const sparseVectorsObject = payload.config?.params?.sparse_vectors ?? {};

  const denseLateVectors: Record<
    string,
    { size: number; distance: "Cosine"; multivector?: boolean }
  > = {};
  if (vectors) {
    for (const [name, value] of Object.entries(vectors)) {
      if (typeof value.size !== "number") {
        continue;
      }
      denseLateVectors[name] = {
        size: value.size,
        distance: "Cosine",
        ...(value.multivector_config ? { multivector: true } : {}),
      };
    }
  }

  const sparseVectors = Object.keys(sparseVectorsObject);
  return { vectors: denseLateVectors, sparseVectors };
});

interface QdrantPoint {
  id: string;
  payload: Record<string, unknown>;
  vector: PointVectors;
}

const MAX_UPSERT_BYTES = 30 * 1024 * 1024;

const estimateJsonBytes = (value: unknown): number =>
  Buffer.byteLength(JSON.stringify(value), "utf8");

const buildUpsertBatches = (points: readonly QdrantPoint[]): QdrantPoint[][] => {
  const baseEnvelopeBytes = estimateJsonBytes({ points: [] });
  const batches: QdrantPoint[][] = [];
  let current: QdrantPoint[] = [];
  let currentBytes = baseEnvelopeBytes;

  for (const point of points) {
    const pointBytes = estimateJsonBytes(point);
    if (pointBytes + baseEnvelopeBytes > MAX_UPSERT_BYTES) {
      throw new Error(
        `Single point payload too large (${pointBytes} bytes) for Qdrant upsert limit`,
      );
    }
    if (currentBytes + pointBytes > MAX_UPSERT_BYTES && current.length > 0) {
      batches.push(current);
      current = [];
      currentBytes = baseEnvelopeBytes;
    }
    current.push(point);
    currentBytes += pointBytes;
  }

  if (current.length > 0) {
    batches.push(current);
  }

  return batches;
};

// Batch embed all chunks and create points
const embedChunksBatch = Effect.fn("Qdrant.embedChunksBatch")(function* (
  chunks: readonly DocumentChunkPayload[],
  onProgress?: (processed: number, total: number) => Effect.Effect<void>,
) {
  const { dense, sparse, late } = yield* EmbeddingProvider;
  const chunking = yield* Chunking;

  const texts = chunks.map((c) => c.text + (c.context ?? ""));
  const total = chunks.length;
  let processed = 0;

  // Create a stream of text batches using grouped
  const batchStream = Stream.fromIterable(texts).pipe(Stream.grouped(EMBEDDING_BATCH_SIZE));

  // Process each batch with embedding services
  const embeddingsStream = batchStream.pipe(
    Stream.mapEffect(
      (batchChunk) =>
        Effect.gen(function* () {
          const batchTexts = [...Chunk.toReadonlyArray(batchChunk)];

          // Call all 3 embedding services in parallel for this batch
          const batchEmbeddings = yield* Effect.all(
            {
              dense: Option.isSome(dense)
                ? dense.value.embedBatch(batchTexts).pipe(
                    Effect.retry(retrySchedule),
                    Effect.map(Option.some),
                    Effect.mapError(
                      (e) =>
                        new IndexingError({
                          reason: "Dense batch embedding failed",
                          cause: e,
                        }),
                    ),
                  )
                : Effect.succeed(Option.none<number[][]>()),
              sparse: Option.isSome(sparse)
                ? sparse.value.embedBatch(batchTexts).pipe(
                    Effect.retry(retrySchedule),
                    Effect.map(Option.some),
                    Effect.mapError(
                      (e) =>
                        new IndexingError({
                          reason: "Sparse batch embedding failed",
                          cause: e,
                        }),
                    ),
                  )
                : Effect.succeed(Option.none<SparseVector[]>()),
              late: Option.isSome(late)
                ? late.value.embedBatch(batchTexts).pipe(
                    Effect.retry(retrySchedule),
                    Effect.map(Option.some),
                    Effect.mapError(
                      (e) =>
                        new IndexingError({
                          reason: "Late batch embedding failed",
                          cause: e,
                        }),
                    ),
                  )
                : Effect.succeed(Option.none<number[][][]>()),
            },
            { concurrency: "unbounded" },
          );

          processed += batchTexts.length;
          if (onProgress) yield* onProgress(processed, total);

          return { batchTexts, batchEmbeddings };
        }),
      { concurrency: EMBEDDING_BATCH_CONCURRENCY },
    ),
  );

  // Collect all batch results
  const allBatchResults = yield* Stream.runCollect(embeddingsStream);

  // Build points from all batch results
  const points: QdrantPoint[] = [];
  let globalIndex = 0;

  for (const { batchTexts, batchEmbeddings } of allBatchResults) {
    for (let i = 0; i < batchTexts.length; i++) {
      const chunkData = chunks[globalIndex]!;
      const vectorEntries: PointVectorEntry[] = [];

      const denseEmb = Option.isSome(batchEmbeddings.dense) ? batchEmbeddings.dense.value[i] : null;
      if (denseEmb && denseEmb.length > 0) {
        if (Option.isSome(dense)) {
          vectorEntries.push({
            kind: "dense",
            vectorName: dense.value.descriptor.vectorName,
            value: denseEmb,
          });
        }
      }

      const sparseEmb = Option.isSome(batchEmbeddings.sparse)
        ? batchEmbeddings.sparse.value[i]
        : null;
      if (sparseEmb && sparseEmb.indices.length > 0) {
        if (Option.isSome(sparse)) {
          vectorEntries.push({
            kind: "sparse",
            vectorName: sparse.value.descriptor.vectorName,
            value: {
              indices: sparseEmb.indices,
              values: sparseEmb.values,
            },
          });
        }
      }

      const lateEmb = Option.isSome(batchEmbeddings.late) ? batchEmbeddings.late.value[i] : null;
      if (lateEmb && lateEmb.length > 0) {
        if (Option.isSome(late)) {
          vectorEntries.push({
            kind: "late",
            vectorName: late.value.descriptor.vectorName,
            value: lateEmb,
          });
        }
      }

      const contextText = chunkData.context ?? "";
      const textHash = new Bun.CryptoHasher("sha256").update(chunkData.text).digest("hex");
      const chunkKey = new Bun.CryptoHasher("sha256")
        .update(
          `${chunkData.source}|${chunking.method}|${chunkData.start_index}|${chunkData.end_index}|${textHash}`,
        )
        .digest("hex");

      points.push({
        id: crypto.randomUUID(),
        payload: {
          text: chunkData.text,
          start_index: chunkData.start_index,
          end_index: chunkData.end_index,
          token_count: chunkData.token_count,
          context: contextText,
          source: chunkData.source,
          chunk_key: chunkKey,
        },
        vector: pointVectorsFromEntries(vectorEntries),
      });

      globalIndex++;
    }
  }

  return points;
});

// ==========================================================================
// Service Definition
// ==========================================================================

export const getCollectionName = Effect.gen(function* () {
  const provider = yield* EmbeddingProvider;
  const descriptors = embeddingDescriptorsFromProvider(provider);
  const chunking = yield* Chunking;

  if (descriptors.length === 0) {
    return yield* new NoEmbeddingServicesConfiguredError();
  }

  const schemaId = buildSchemaId(descriptors);
  return `${COLLECTION_NAME_PREFIX}_${chunking.method}_${schemaId}`;
});

export const getHashCollectionName = Effect.gen(function* () {
  const collectionName = yield* getCollectionName;
  return `${collectionName}_hashes`;
});

// Hash filename to create a deterministic point ID for O(1) lookup
const hashFilename = (filename: string): string =>
  new Bun.CryptoHasher("md5").update(filename).digest("hex");

const qdrantUrlConfig = Config.string("QDRANT_URL").pipe(
  Config.orElse(() => Config.succeed("http://127.0.0.1:6333")),
);

type ConstructorArgs<T extends new (...args: any[]) => any> = T extends new (
  ...args: infer A
) => infer _R
  ? A
  : never;

export type CollectionConfig = Parameters<QdrantClient["createCollection"]>[1];

export interface TestCollectionOptions {
  readonly collectionName?: string;
  readonly config: CollectionConfig;
}

const makeQdrantClient = (options: ConstructorArgs<typeof QdrantClient>[0]) =>
  Effect.try({
    try: () => new QdrantClient(options),
    catch: (error) =>
      QdrantError.make({
        reason: `Failed to construct Qdrant client: ${previewDetails(error)}`,
        cause: error,
      }),
  });

const makeInitializedQdrantImpl = (options: ConstructorArgs<typeof QdrantClient>[0]) =>
  Effect.gen(function* () {
    const client = yield* makeQdrantClient(options);
    const impl = makeQdrantImpl(client);
    yield* impl.ensureCollection();
    return impl;
  });

const makeQdrantImpl = (client: QdrantClient) => {
  const withClient = Effect.fn("Qdrant.withClient")(function* <T>(fn: (client: QdrantClient) => T) {
    return yield* wrapClientCall(
      () => fn(client),
      (error) =>
        QdrantError.make({
          reason: "Synchronous error in Qdrant.withClient",
          cause: error,
        }),
      (error) =>
        QdrantError.make({
          reason: "Asynchronous error in Qdrant.withClient",
          cause: error,
        }),
    );
  });

  const health = Effect.fn("Qdrant.health")(function* () {
    yield* withClient((client) => client.getCollections());
  });

  const ensureCollection = Effect.fn("Qdrant.ensureCollection")(function* () {
    const provider = yield* EmbeddingProvider;
    const descriptors = embeddingDescriptorsFromProvider(provider);

    if (descriptors.length === 0) {
      return yield* new NoEmbeddingServicesConfiguredError();
    }

    const expectedSchema = buildCollectionSchema(descriptors);
    const expectedNormalized = normalizeSchemaSpec(expectedSchema);
    const collectionName = yield* getCollectionName;

    yield* Effect.annotateCurrentSpan("collectionName", collectionName);

    const collections = yield* withClient((client) => client.getCollections());
    const exists = collections.collections.some((c: { name: string }) => c.name === collectionName);

    if (!exists) {
      yield* withClient((client) =>
        client.createCollection(collectionName, {
          vectors: Object.fromEntries(
            Object.entries(expectedSchema.vectors).map(([name, vectorConfig]) => [
              name,
              {
                size: vectorConfig.size,
                distance: vectorConfig.distance,
                ...(vectorConfig.multivector
                  ? { multivector_config: { comparator: "max_sim" } }
                  : {}),
              },
            ]),
          ),
          sparse_vectors: Object.fromEntries(
            expectedSchema.sparseVectors.map((name) => [name, { modifier: "idf" }]),
          ),
        }),
      ).pipe(
        Effect.mapError(
          (cause) =>
            new CollectionCreationError({
              collectionName,
              reason: "Failed creating collection with expected schema",
              cause,
            }),
        ),
      );
      yield* Effect.log("Created collection", { collectionName });
      return;
    }

    const collectionInfo = yield* withClient((client) => client.getCollection(collectionName)).pipe(
      Effect.mapError(
        (cause) =>
          new CollectionIntrospectionError({
            collectionName,
            reason: "Failed to inspect existing collection schema",
            cause,
          }),
      ),
    );

    const actualSchema = yield* actualSchemaFromCollection(collectionInfo).pipe(
      Effect.mapError(
        (cause) =>
          new CollectionIntrospectionError({
            collectionName,
            reason: `Failed to decode existing collection schema: ${previewDetails(cause)}`,
            cause,
          }),
      ),
    );
    const actualNormalized = normalizeSchemaSpec(actualSchema);
    if (!normalizedCollectionSchemaSpecEquivalence(actualNormalized, expectedNormalized)) {
      return yield* new CollectionSchemaMismatchError({
        collectionName,
        reason: "Existing collection schema does not match active embedding descriptors",
        expected: expectedNormalized,
        actual: actualNormalized,
      });
    }
  });

  const indexChunks = Effect.fn("Qdrant.indexChunks")(function* (
    chunks: readonly DocumentChunkPayload[],
    options?: IndexChunksOptions,
  ) {
    const sources = [...new Set(chunks.map((c) => c.source))];
    yield* Effect.annotateCurrentSpan("chunkCount", chunks.length);
    yield* Effect.annotateCurrentSpan("sources", sources);

    yield* ensureCollection();

    const provider = yield* EmbeddingProvider;
    const descriptors = embeddingDescriptorsFromProvider(provider);

    if (descriptors.length === 0) {
      return yield* new IndexingError({
        reason: "No embedding services configured",
        cause: new Error("No embedding services configured"),
      });
    }

    const collectionName = yield* getCollectionName;

    // Delete old points for sources being re-indexed
    for (const source of sources) {
      yield* withClient((client) =>
        client.delete(collectionName, {
          filter: { must: [{ key: "source", match: { value: source } }] },
        }),
      ).pipe(
        Effect.mapError(
          (e) =>
            new IndexingError({
              reason: `Failed to delete old points for source: ${source}`,
              cause: e,
            }),
        ),
      );
    }

    // Use batch embedding for better performance
    const points = yield* embedChunksBatch(chunks, options?.onProgress);

    const batches = yield* Effect.try({
      try: () => buildUpsertBatches(points),
      catch: (error) =>
        new IndexingError({
          reason: "Failed to build Qdrant upsert batches",
          cause: error,
        }),
    });
    yield* Effect.annotateCurrentSpan("batchCount", batches.length);

    // Upsert to Qdrant with retry
    yield* Effect.forEach(
      batches,
      (batch, index) =>
        withClient((client) => client.upsert(collectionName, { points: batch })).pipe(
          Effect.retry(retrySchedule),
          Effect.mapError(
            (e) =>
              new IndexingError({
                reason: `Failed to upsert points to Qdrant (batch ${index + 1}/${batches.length})`,
                cause: e,
              }),
          ),
        ),
      { concurrency: 1 },
    );
  });

  const ensureHashCollection = Effect.fn("Qdrant.ensureHashCollection")(function* () {
    const hashCollectionName = yield* getHashCollectionName;

    const collections = yield* withClient((client) => client.getCollections());
    const exists = collections.collections.some(
      (c: { name: string }) => c.name === hashCollectionName,
    );

    if (!exists) {
      yield* withClient((client) =>
        client.createCollection(hashCollectionName, {
          vectors: {},
        }),
      );
      yield* Effect.log("Created hash collection", { collectionName: hashCollectionName });
    }
  });

  const checkFileIndexed = Effect.fn("Qdrant.checkFileIndexed")(function* (filename: string) {
    yield* Effect.annotateCurrentSpan("filename", filename);

    yield* ensureHashCollection();
    const hashCollectionName = yield* getHashCollectionName;
    const pointId = hashFilename(filename);

    const existingPoints = yield* withClient((client) =>
      client.retrieve(hashCollectionName, {
        ids: [pointId],
        with_payload: false,
      }),
    ).pipe(Effect.catchAll(() => Effect.succeed([] as Array<unknown>)));

    return existingPoints.length > 0;
  });

  const markFileIndexed = Effect.fn("Qdrant.markFileIndexed")(function* (filename: string) {
    yield* Effect.annotateCurrentSpan("filename", filename);

    yield* ensureHashCollection();
    const hashCollectionName = yield* getHashCollectionName;
    const pointId = hashFilename(filename);

    yield* withClient((client) =>
      client.upsert(hashCollectionName, {
        points: [
          {
            id: pointId,
            payload: { source: filename },
            vector: {},
          },
        ],
      }),
    );
  });

  return { withClient, health, indexChunks, ensureCollection, checkFileIndexed, markFileIndexed };
};

export class Qdrant extends Effect.Service<Qdrant>()("@app/Qdrant", {
  effect: Effect.gen(function* () {
    const url = yield* qdrantUrlConfig;
    return yield* makeInitializedQdrantImpl({ url });
  }),
}) {
  static layer = (options: ConstructorArgs<typeof QdrantClient>[0]) =>
    Layer.effect(
      Qdrant,
      makeInitializedQdrantImpl(options).pipe(Effect.map((impl) => new Qdrant(impl))),
    );

  static fromEnv = Layer.effect(
    Qdrant,
    Effect.gen(function* () {
      const url = yield* qdrantUrlConfig;
      const impl = yield* makeInitializedQdrantImpl({ url });
      return new Qdrant(impl);
    }),
  );

  static localhost = Qdrant.layer({ url: QDRANT_LOCALHOST_URL });

  static skyhigh = Qdrant.layer({
    url: QDRANT_SKYHIGH_URL,
  });
}

export class QdrantTestCollection extends Context.Tag("@app/QdrantTestCollection")<
  QdrantTestCollection,
  string
>() {
  // Always generates unique name: `${prefix}_${timestamp}_${random}`
  // prefix defaults to "test" if not provided
  static readonly make = Effect.fn("QdrantTestCollection.make")(function* (
    options: TestCollectionOptions,
  ) {
    const qdrant = yield* Qdrant;
    const prefix = options.collectionName ?? "test";
    const collectionName = `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;

    return yield* Effect.acquireRelease(
      Effect.gen(function* () {
        yield* qdrant.withClient((client) =>
          client.createCollection(collectionName, options.config),
        );

        return QdrantTestCollection.of(collectionName);
      }),
      () => Effect.ignore(qdrant.withClient((client) => client.deleteCollection(collectionName))),
    );
  });

  static readonly layer = (options: TestCollectionOptions) =>
    Layer.scoped(QdrantTestCollection, QdrantTestCollection.make(options));

  static readonly withLocalhost = (options: TestCollectionOptions) =>
    Layer.merge(
      Qdrant.localhost,
      QdrantTestCollection.layer(options).pipe(Layer.provide(Qdrant.localhost)),
    );

  static readonly withSkyhigh = (options: TestCollectionOptions) =>
    Layer.merge(
      Qdrant.skyhigh,
      QdrantTestCollection.layer(options).pipe(Layer.provide(Qdrant.skyhigh)),
    );
}
