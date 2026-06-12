import { Effect, Layer, Option, Schema } from "effect";
import { Chunking } from "../src/services/Chunking";
import { PythonApiClient } from "../src/services/PythonApiClient";
import { Qdrant, getCollectionName } from "../src/services/Qdrant";
import {
  DenseEmbedding,
  EmbeddingProvider,
  LateEmbedding,
  SparseEmbedding,
  type SparseVector,
} from "../src/services/embeddings";

class MigrationError extends Schema.TaggedError<MigrationError>()("MigrationError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

type ReusePolicy = "strict" | "relaxed";

type CliOptions = {
  readonly from: string;
  readonly to?: string;
  readonly batchSize: number;
  readonly reusePolicy: ReusePolicy;
  readonly forceRecompute: ReadonlySet<string>;
};

type Stats = {
  processed: number;
  failed: number;
  reused: Record<string, number>;
  recomputed: Record<string, number>;
};

type TargetDescriptor =
  | {
      kind: "dense";
      vectorName: string;
      modelName: string;
      method: string;
      vectorSize: number;
    }
  | {
      kind: "sparse";
      vectorName: string;
      modelName: string;
      method: string;
    }
  | {
      kind: "late";
      vectorName: string;
      modelName: string;
      method: string;
      vectorSize: number;
    };

type PointVectors = Record<string, number[] | number[][] | SparseVector>;

const parseArgs = (): CliOptions => {
  const argv = process.argv.slice(2);
  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) {
      return undefined;
    }
    return argv[index + 1];
  };

  const from = get("--from");
  if (!from) {
    throw new Error("Missing required --from <collection>");
  }

  const to = get("--to");
  const batchSize = Number(get("--batch-size") ?? "128");
  const reusePolicyRaw = (get("--reuse-policy") ?? "strict") as ReusePolicy;
  const forceRaw = get("--force-recompute") ?? "";

  if (!Number.isFinite(batchSize) || batchSize <= 0) {
    throw new Error("--batch-size must be a positive number");
  }
  if (reusePolicyRaw !== "strict" && reusePolicyRaw !== "relaxed") {
    throw new Error("--reuse-policy must be strict or relaxed");
  }

  return {
    from,
    to,
    batchSize,
    reusePolicy: reusePolicyRaw,
    forceRecompute: new Set(
      forceRaw
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    ),
  };
};

const parseChunkingMethodFromCollectionName = (collectionName: string): string => {
  const parts = collectionName.split("_");
  return parts[1] ?? "recursive";
};

const computeChunkKey = (payload: Record<string, unknown>, chunkingMethod: string): string => {
  const source = typeof payload.source === "string" ? payload.source : "unknown";
  const start = typeof payload.start_index === "number" ? payload.start_index : -1;
  const end = typeof payload.end_index === "number" ? payload.end_index : -1;
  const text = typeof payload.text === "string" ? payload.text : "";
  const textHash = new Bun.CryptoHasher("sha256").update(text).digest("hex");
  return new Bun.CryptoHasher("sha256")
    .update(`${source}|${chunkingMethod}|${start}|${end}|${textHash}`)
    .digest("hex");
};

const denseCompatible = (
  vector: unknown,
  descriptor: Extract<TargetDescriptor, { kind: "dense" }>,
) => Array.isArray(vector) && vector.length === descriptor.vectorSize;

const sparseCompatible = (vector: unknown, policy: ReusePolicy) => {
  if (!vector || typeof vector !== "object") {
    return false;
  }
  const sparse = vector as SparseVector;
  if (!Array.isArray(sparse.indices) || !Array.isArray(sparse.values)) {
    return false;
  }
  return policy === "relaxed" ? true : sparse.indices.length === sparse.values.length;
};

const lateCompatible = (
  vector: unknown,
  descriptor: Extract<TargetDescriptor, { kind: "late" }>,
) => {
  if (!Array.isArray(vector) || vector.length === 0) {
    return false;
  }
  const rows = vector as number[][];
  return Array.isArray(rows[0]) && rows[0]?.length === descriptor.vectorSize;
};

const canReuseVector = (
  sourceVector: unknown,
  descriptor: TargetDescriptor,
  policy: ReusePolicy,
): sourceVector is number[] | number[][] | SparseVector => {
  if (descriptor.kind === "dense") {
    return policy === "strict"
      ? denseCompatible(sourceVector, descriptor)
      : Array.isArray(sourceVector);
  }
  if (descriptor.kind === "sparse") {
    return sparseCompatible(sourceVector, policy);
  }
  return policy === "strict"
    ? lateCompatible(sourceVector, descriptor)
    : Array.isArray(sourceVector);
};

const program = Effect.gen(function* () {
  const args = yield* Effect.try({
    try: parseArgs,
    catch: (cause) => new MigrationError({ reason: "Failed to parse CLI args", cause }),
  });

  const qdrant = yield* Qdrant;
  const dense = yield* Effect.serviceOption(DenseEmbedding);
  const sparse = yield* Effect.serviceOption(SparseEmbedding);
  const late = yield* Effect.serviceOption(LateEmbedding);

  const targetDescriptors: TargetDescriptor[] = [];
  if (Option.isSome(dense)) {
    targetDescriptors.push({ kind: "dense", ...dense.value.descriptor });
  }
  if (Option.isSome(sparse)) {
    targetDescriptors.push({ kind: "sparse", ...sparse.value.descriptor });
  }
  if (Option.isSome(late)) {
    targetDescriptors.push({ kind: "late", ...late.value.descriptor });
  }

  if (targetDescriptors.length === 0) {
    return yield* new MigrationError({
      reason: "At least one embedding service must be configured",
      cause: new Error("No embeddings configured"),
    });
  }

  const activeCollection = yield* getCollectionName;
  if (args.to && args.to !== activeCollection) {
    return yield* new MigrationError({
      reason:
        `Destination collection "${args.to}" does not match the active embedding collection "${activeCollection}". ` +
        "Use the default destination or reconfigure embeddings to match the desired collection.",
      cause: new Error("Destination collection schema may not match active embeddings"),
    });
  }

  const destinationCollection = args.to ?? activeCollection;
  const chunkingMethod = parseChunkingMethodFromCollectionName(destinationCollection);
  const stats: Stats = { processed: 0, failed: 0, reused: {}, recomputed: {} };

  let offset: string | number | undefined;
  let sourceTotal = 0;

  while (true) {
    const scroll = yield* qdrant.withClient((client) =>
      client.scroll(args.from, {
        limit: args.batchSize,
        with_payload: true,
        with_vector: true,
        offset,
      }),
    );

    const points = (scroll.points ?? []) as Array<{
      id: string | number;
      payload?: Record<string, unknown>;
      vector?: Record<string, unknown>;
    }>;

    if (points.length === 0) {
      break;
    }

    sourceTotal += points.length;
    const upsertPoints: Array<{
      id: string | number;
      payload: Record<string, unknown>;
      vector: PointVectors;
    }> = [];

    for (const point of points) {
      const result = yield* Effect.either(
        Effect.gen(function* () {
          const payload = point.payload ?? {};
          const vectors = point.vector ?? {};
          const text = typeof payload.text === "string" ? payload.text : "";
          const context = typeof payload.context === "string" ? payload.context : "";
          const inputText = `${text}${context}`;
          const nextVectors: PointVectors = {};

          for (const descriptor of targetDescriptors) {
            const force = args.forceRecompute.has(descriptor.vectorName);
            const sourceVector = vectors[descriptor.vectorName];

            const canReuse = !force && canReuseVector(sourceVector, descriptor, args.reusePolicy);

            if (canReuse) {
              nextVectors[descriptor.vectorName] = sourceVector;
              stats.reused[descriptor.vectorName] = (stats.reused[descriptor.vectorName] ?? 0) + 1;
              continue;
            }

            if (descriptor.kind === "dense" && Option.isSome(dense)) {
              nextVectors[descriptor.vectorName] = yield* dense.value.embed(inputText);
            } else if (descriptor.kind === "sparse" && Option.isSome(sparse)) {
              nextVectors[descriptor.vectorName] = yield* sparse.value.embed(inputText);
            } else if (descriptor.kind === "late" && Option.isSome(late)) {
              nextVectors[descriptor.vectorName] = yield* late.value.embed(inputText);
            }

            stats.recomputed[descriptor.vectorName] =
              (stats.recomputed[descriptor.vectorName] ?? 0) + 1;
          }

          return {
            id: point.id,
            payload: {
              ...payload,
              chunk_key:
                typeof payload.chunk_key === "string"
                  ? payload.chunk_key
                  : computeChunkKey(payload, chunkingMethod),
            },
            vector: nextVectors,
          };
        }),
      );

      if (result._tag === "Right") {
        upsertPoints.push(result.right);
        stats.processed += 1;
      } else {
        stats.failed += 1;
      }
    }

    if (upsertPoints.length > 0) {
      yield* qdrant.withClient((client) =>
        client.upsert(destinationCollection, {
          points: upsertPoints,
        }),
      );
    }

    offset = (scroll as { next_page_offset?: string | number }).next_page_offset;
    if (!offset) {
      break;
    }

    console.log(`Processed ${stats.processed} points so far`);
  }

  const destinationCount = yield* qdrant.withClient((client) =>
    client.count(destinationCollection, {
      exact: true,
    }),
  );

  const sample = yield* qdrant.withClient((client) =>
    client.scroll(destinationCollection, {
      limit: 25,
      with_payload: true,
      with_vector: false,
    }),
  );
  const sampleMissingChunkKeys = (sample.points ?? []).filter((point: { payload?: unknown }) => {
    const payload = (point.payload ?? {}) as Record<string, unknown>;
    return typeof payload.chunk_key !== "string";
  }).length;

  console.log("\nMigration report");
  console.log(`from: ${args.from}`);
  console.log(`to: ${destinationCollection}`);
  console.log(`reuse policy: ${args.reusePolicy}`);
  console.log(`processed: ${stats.processed}`);
  console.log(`failed: ${stats.failed}`);
  console.log(`source count observed: ${sourceTotal}`);
  console.log(`destination count: ${destinationCount.count}`);
  console.log(`sample missing chunk_key: ${sampleMissingChunkKeys}`);
  console.log("reuse counters:", stats.reused);
  console.log("recompute counters:", stats.recomputed);
});

const qdrantLayer = Qdrant.skyhigh;
const embeddingLayer = EmbeddingProvider.make(
  DenseEmbedding.Default,
  SparseEmbedding.Default,
  LateEmbedding.Default,
);
const chunkingLayer = Chunking.Recursive();

const layer = Layer.mergeAll(
  qdrantLayer.pipe(Layer.provide(embeddingLayer), Layer.provide(chunkingLayer)),
  embeddingLayer,
  chunkingLayer,
).pipe(Layer.provideMerge(PythonApiClient.Default));

Effect.runPromise(program.pipe(Effect.provide(layer)));
