import { type QdrantClient } from "@qdrant/js-client-rest";
import { Context, Effect, Layer, Option, Schema } from "effect";
import {
  DenseEmbedding,
  EmbeddingProvider,
  LateEmbedding,
  SparseEmbedding,
  type SparseVector,
} from "./embeddings";
import { Qdrant, QdrantSearchPoints, getCollectionName } from "./Qdrant";

export class RetrievalError extends Schema.TaggedError<RetrievalError>()("RetrievalError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

export class NoServicesConfiguredError extends Schema.TaggedError<NoServicesConfiguredError>()(
  "NoServicesConfiguredError",
  {},
) {}

export interface RetrievalConfig {
  readonly limit: number;
  readonly prefetchLimit: number;
}

interface Embeddings {
  dense: number[] | undefined;
  sparse: SparseVector | undefined;
  late: number[][] | undefined;
}

interface Services {
  dense: Option.Option<typeof DenseEmbedding.Service>;
  sparse: Option.Option<typeof SparseEmbedding.Service>;
  late: Option.Option<typeof LateEmbedding.Service>;
}

type DenseQueryVector = number[];
type LateQueryVector = number[][];
type SparseQueryVector = {
  indices: number[];
  values: number[];
};
type QueryVector = DenseQueryVector | SparseQueryVector;

interface PrefetchQueryRequest {
  readonly query: QueryVector;
  readonly using: string;
  readonly limit: number;
}

interface BaseQueryRequest {
  readonly with_payload: true;
  readonly limit: number;
}

interface SingleVectorQueryRequest extends BaseQueryRequest {
  readonly kind: "single";
  readonly query: QueryVector;
  readonly using: string;
}

interface FusionQueryRequest extends BaseQueryRequest {
  readonly kind: "fusion";
  readonly prefetch: [PrefetchQueryRequest, PrefetchQueryRequest, ...PrefetchQueryRequest[]];
  readonly query: { readonly fusion: "rrf" };
}

interface RerankQueryRequest extends BaseQueryRequest {
  readonly kind: "rerank";
  readonly prefetch: [PrefetchQueryRequest, ...PrefetchQueryRequest[]];
  readonly query: LateQueryVector;
  readonly using: string;
}

type QueryRequestPlan = SingleVectorQueryRequest | FusionQueryRequest | RerankQueryRequest;
type QdrantQueryRequest = Parameters<QdrantClient["query"]>[1];

const toQdrantQueryRequest = (request: QueryRequestPlan): QdrantQueryRequest => {
  switch (request.kind) {
    case "single":
      return {
        query: request.query,
        using: request.using,
        with_payload: request.with_payload,
        limit: request.limit,
      } satisfies QdrantQueryRequest;
    case "fusion":
      return {
        prefetch: request.prefetch,
        query: request.query,
        with_payload: request.with_payload,
        limit: request.limit,
      } satisfies QdrantQueryRequest;
    case "rerank":
      return {
        prefetch: request.prefetch,
        query: request.query,
        using: request.using,
        with_payload: request.with_payload,
        limit: request.limit,
      } satisfies QdrantQueryRequest;
  }
};

const buildQueryParams: (
  embeddings: Embeddings,
  services: Services,
  limit: number,
  prefetchLimit: number,
) => Effect.Effect<QueryRequestPlan, NoServicesConfiguredError> = Effect.fn("buildQueryParams")(
  function* (embeddings: Embeddings, services: Services, limit: number, prefetchLimit: number) {
    const prefetch: PrefetchQueryRequest[] = [];

    // Check for non-empty arrays (empty arrays are truthy but useless for search)
    if (embeddings.dense && embeddings.dense.length > 0 && Option.isSome(services.dense)) {
      prefetch.push({
        query: embeddings.dense,
        using: services.dense.value.descriptor.vectorName,
        limit: prefetchLimit,
      });
    }

    if (
      embeddings.sparse &&
      embeddings.sparse.indices.length > 0 &&
      Option.isSome(services.sparse)
    ) {
      prefetch.push({
        query: {
          indices: embeddings.sparse.indices,
          values: embeddings.sparse.values,
        },
        using: services.sparse.value.descriptor.vectorName,
        limit: prefetchLimit,
      });
    }

    // Late interaction as final reranker (requires prefetch)
    if (
      embeddings.late &&
      embeddings.late.length > 0 &&
      prefetch.length > 0 &&
      Option.isSome(services.late)
    ) {
      const [first, ...rest] = prefetch;
      if (!first) {
        return yield* new NoServicesConfiguredError();
      }
      return {
        kind: "rerank",
        prefetch: [first, ...rest],
        query: embeddings.late,
        using: services.late.value.descriptor.vectorName,
        with_payload: true,
        limit,
      };
    }

    // No late interaction - use fusion or single vector
    if (prefetch.length > 1) {
      const [first, second, ...rest] = prefetch;
      if (!first || !second) {
        return yield* new NoServicesConfiguredError();
      }
      return {
        kind: "fusion",
        prefetch: [first, second, ...rest],
        query: { fusion: "rrf" },
        with_payload: true,
        limit,
      };
    }

    // Single vector search
    if (prefetch.length === 1 && prefetch[0]) {
      return {
        kind: "single",
        query: prefetch[0].query,
        using: prefetch[0].using,
        with_payload: true,
        limit,
      };
    }

    return yield* new NoServicesConfiguredError();
  },
);

const DEFAULT_CONFIG: RetrievalConfig = {
  limit: 10,
  prefetchLimit: 20,
};

export interface RetrievalImpl {
  readonly config: RetrievalConfig;
  readonly collectionName: string;
  readonly search: (
    query: string,
  ) => Effect.Effect<typeof QdrantSearchPoints.Type, RetrievalError | NoServicesConfiguredError>;
}

const make = (config: RetrievalConfig) =>
  Effect.gen(function* () {
    const { limit, prefetchLimit } = config;
    const qdrant = yield* Qdrant;
    const collectionName = yield* getCollectionName;

    // Optional dependencies
    const provider = yield* EmbeddingProvider;
    const { dense, sparse, late } = provider;

    yield* Effect.annotateCurrentSpan({
      "retrieval.servicesAvailable": {
        dense: Option.isSome(dense),
        sparse: Option.isSome(sparse),
        late: Option.isSome(late),
      },
    });

    const services: Services = { dense, sparse, late };

    return {
      config,
      collectionName,
      search: Effect.fn("Retrieval.search")(function* (query: string) {
        // Get embeddings for enabled strategies in parallel using struct pattern
        const embeddingEffects = {
          dense: Option.isSome(dense)
            ? dense.value.embed(query).pipe(
                Effect.map(Option.some),
                Effect.mapError(
                  (e) =>
                    new RetrievalError({
                      reason: "Dense embedding failed",
                      cause: e,
                    }),
                ),
              )
            : Effect.succeed(Option.none<number[]>()),
          sparse: Option.isSome(sparse)
            ? sparse.value.embed(query).pipe(
                Effect.map(Option.some),
                Effect.mapError(
                  (e) =>
                    new RetrievalError({
                      reason: "Sparse embedding failed",
                      cause: e,
                    }),
                ),
              )
            : Effect.succeed(Option.none<SparseVector>()),
          late: Option.isSome(late)
            ? late.value.embed(query).pipe(
                Effect.map(Option.some),
                Effect.mapError(
                  (e) =>
                    new RetrievalError({
                      reason: "Late embedding failed",
                      cause: e,
                    }),
                ),
              )
            : Effect.succeed(Option.none<number[][]>()),
        };

        // Run all embeddings in parallel and collect results
        const embeddingResults = yield* Effect.all(embeddingEffects, { concurrency: "unbounded" });

        const embeddings: Embeddings = {
          dense: Option.getOrUndefined(embeddingResults.dense),
          sparse: Option.getOrUndefined(embeddingResults.sparse),
          late: Option.getOrUndefined(embeddingResults.late),
        };

        // Build query based on what's available
        const queryRequestPlan = yield* buildQueryParams(
          embeddings,
          services,
          limit,
          prefetchLimit,
        );
        const queryParams = toQdrantQueryRequest(queryRequestPlan);

        // Execute search
        const results = yield* qdrant
          .withClient((client) => client.query(collectionName, queryParams))
          .pipe(
            Effect.mapError(
              (e) =>
                new RetrievalError({
                  reason: "Qdrant query failed",
                  cause: e,
                }),
            ),
          );

        yield* Effect.annotateCurrentSpan({
          "qdrant.pointsFound": results.points?.length ?? 0,
        });

        // Decode results using schema
        return yield* Schema.decodeUnknown(QdrantSearchPoints)(results.points).pipe(
          Effect.mapError(
            (e) =>
              new RetrievalError({
                reason: "Failed to decode Qdrant results",
                cause: e,
              }),
          ),
        );
      }),
    } satisfies RetrievalImpl;
  });

export class Retrieval extends Context.Tag("@app/Retrieval")<Retrieval, RetrievalImpl>() {
  static readonly layer = (config: Partial<RetrievalConfig> = {}) =>
    Layer.effect(
      Retrieval,
      make({
        limit: config.limit ?? DEFAULT_CONFIG.limit,
        prefetchLimit: config.prefetchLimit ?? DEFAULT_CONFIG.prefetchLimit,
      }),
    );

  static readonly Default = Retrieval.layer();
}
