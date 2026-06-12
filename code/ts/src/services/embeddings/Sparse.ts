import { Context, Effect, Layer, Schema } from "effect";
import {
  embedSparse,
  embedSparseBatch,
  resolveSparseEmbeddingConfig,
} from "../../generated/python-api/sdk.gen";
import { DEFAULT_SPARSE_MODEL_NAME, DEFAULT_SPARSE_VECTOR_NAME } from "./defaults";
import { PYTHON_API_TIMEOUT } from "../pythonApi";
import { PythonApiClient } from "../PythonApiClient";

export class SparseResolveError extends Schema.TaggedError<SparseResolveError>()(
  "SparseResolveError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class SparseResolveTimeoutError extends Schema.TaggedError<SparseResolveTimeoutError>()(
  "SparseResolveTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class SparseResolveMissingDescriptorError extends Schema.TaggedError<SparseResolveMissingDescriptorError>()(
  "SparseResolveMissingDescriptorError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class SparseResolveDescriptorMismatchError extends Schema.TaggedError<SparseResolveDescriptorMismatchError>()(
  "SparseResolveDescriptorMismatchError",
  {
    expectedModelName: Schema.String,
    expectedVectorName: Schema.String,
    actualModelName: Schema.String,
    actualVectorName: Schema.String,
  },
) {}

export class SparseEmbedError extends Schema.TaggedError<SparseEmbedError>()("SparseEmbedError", {
  modelName: Schema.String,
  vectorName: Schema.String,
  cause: Schema.Defect,
}) {}

export class SparseEmbedTimeoutError extends Schema.TaggedError<SparseEmbedTimeoutError>()(
  "SparseEmbedTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class SparseEmbedMissingDataError extends Schema.TaggedError<SparseEmbedMissingDataError>()(
  "SparseEmbedMissingDataError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class SparseEmbedBatchError extends Schema.TaggedError<SparseEmbedBatchError>()(
  "SparseEmbedBatchError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export class SparseEmbedBatchTimeoutError extends Schema.TaggedError<SparseEmbedBatchTimeoutError>()(
  "SparseEmbedBatchTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class SparseEmbedBatchMissingDataError extends Schema.TaggedError<SparseEmbedBatchMissingDataError>()(
  "SparseEmbedBatchMissingDataError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export class SparseEmbedBatchInvalidPayloadError extends Schema.TaggedError<SparseEmbedBatchInvalidPayloadError>()(
  "SparseEmbedBatchInvalidPayloadError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export type SparseEmbeddingError =
  | SparseResolveError
  | SparseResolveTimeoutError
  | SparseResolveMissingDescriptorError
  | SparseResolveDescriptorMismatchError
  | SparseEmbedError
  | SparseEmbedTimeoutError
  | SparseEmbedMissingDataError
  | SparseEmbedBatchError
  | SparseEmbedBatchTimeoutError
  | SparseEmbedBatchMissingDataError
  | SparseEmbedBatchInvalidPayloadError;

export interface SparseVector {
  indices: number[];
  values: number[];
}

export type SparseEmbeddingMethod = "fastembed" | "test";

export interface SparseEmbeddingDescriptor {
  readonly method: SparseEmbeddingMethod;
  readonly modelName: string;
  readonly vectorName: string;
}

export interface SparseEmbeddingService {
  readonly descriptor: SparseEmbeddingDescriptor;
  readonly embed: (text: string) => Effect.Effect<SparseVector, SparseEmbeddingError>;
  readonly embedBatch: (texts: string[]) => Effect.Effect<SparseVector[], SparseEmbeddingError>;
}

export interface FastembedSparseConfig {
  readonly modelName?: string;
  readonly vectorName?: string;
}

const makeFastembedSparse = (config?: FastembedSparseConfig) =>
  Effect.gen(function* () {
    const { client } = yield* PythonApiClient;
    const modelName = config?.modelName ?? DEFAULT_SPARSE_MODEL_NAME;
    const vectorName = config?.vectorName ?? DEFAULT_SPARSE_VECTOR_NAME;

    const resolved = yield* Effect.tryPromise(() =>
      resolveSparseEmbeddingConfig({
        client,
        body: {
          method: "fastembed",
          model_name: modelName,
          vector_name: vectorName,
        },
      }),
    ).pipe(
      Effect.mapError(
        (error) =>
          new SparseResolveError({
            modelName,
            vectorName,
            cause: error,
          }),
      ),
      Effect.timeoutFail({
        duration: PYTHON_API_TIMEOUT,
        onTimeout: () =>
          new SparseResolveTimeoutError({
            modelName,
            vectorName,
            timeout: PYTHON_API_TIMEOUT,
            cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
          }),
      }),
    );

    const resolvedSparse = resolved.data;
    if (!resolvedSparse) {
      return yield* new SparseResolveMissingDescriptorError({
        modelName,
        vectorName,
        cause: resolved.error ?? new Error("Resolve response missing data"),
      });
    }
    if (resolvedSparse.model_name !== modelName || resolvedSparse.vector_name !== vectorName) {
      return yield* new SparseResolveDescriptorMismatchError({
        expectedModelName: modelName,
        expectedVectorName: vectorName,
        actualModelName: resolvedSparse.model_name,
        actualVectorName: resolvedSparse.vector_name,
      });
    }

    const embed = Effect.fn("SparseEmbedding.embed.fastembed")(function* (text: string) {
      const response = yield* Effect.tryPromise(() =>
        embedSparse({
          client,
          body: {
            text,
            config: {
              method: "fastembed",
              model_name: modelName,
              vector_name: vectorName,
            },
          },
        }),
      ).pipe(
        Effect.mapError(
          (error) =>
            new SparseEmbedError({
              modelName,
              vectorName,
              cause: error,
            }),
        ),
        Effect.timeoutFail({
          duration: PYTHON_API_TIMEOUT,
          onTimeout: () =>
            new SparseEmbedTimeoutError({
              modelName,
              vectorName,
              timeout: PYTHON_API_TIMEOUT,
              cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
            }),
        }),
      );

      const indices = response.data?.indices;
      const values = response.data?.values;

      if (!indices || !values) {
        return yield* new SparseEmbedMissingDataError({
          modelName,
          vectorName,
          cause: response.error ?? new Error("Sparse embed response missing data"),
        });
      }

      return { indices, values };
    });

    const embedBatch = Effect.fn("SparseEmbedding.embedBatch.fastembed")(function* (
      texts: string[],
    ) {
      const response = yield* Effect.tryPromise(() =>
        embedSparseBatch({
          client,
          body: {
            texts,
            config: {
              method: "fastembed",
              model_name: modelName,
              vector_name: vectorName,
            },
          },
        }),
      ).pipe(
        Effect.mapError(
          (error) =>
            new SparseEmbedBatchError({
              modelName,
              vectorName,
              batchSize: texts.length,
              cause: error,
            }),
        ),
        Effect.timeoutFail({
          duration: PYTHON_API_TIMEOUT,
          onTimeout: () =>
            new SparseEmbedBatchTimeoutError({
              modelName,
              vectorName,
              batchSize: texts.length,
              timeout: PYTHON_API_TIMEOUT,
              cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
            }),
        }),
      );

      const embeddings = response.data?.embeddings;
      if (!embeddings) {
        return yield* new SparseEmbedBatchMissingDataError({
          modelName,
          vectorName,
          batchSize: texts.length,
          cause: response.error ?? new Error("Sparse embed batch response missing data"),
        });
      }

      const normalized: SparseVector[] = [];
      for (const embedding of embeddings) {
        if (!embedding.indices || !embedding.values) {
          return yield* new SparseEmbedBatchInvalidPayloadError({
            modelName,
            vectorName,
            batchSize: texts.length,
            cause: response.error ?? new Error("Sparse embed batch response contains invalid data"),
          });
        }
        normalized.push({ indices: embedding.indices, values: embedding.values });
      }

      return normalized;
    });

    return SparseEmbedding.of({
      descriptor: {
        method: "fastembed",
        modelName,
        vectorName,
      },
      embed,
      embedBatch,
    });
  });

export class SparseEmbedding extends Context.Tag("@app/SparseEmbedding")<
  SparseEmbedding,
  SparseEmbeddingService
>() {
  static readonly Fastembed = (config?: FastembedSparseConfig) =>
    Layer.effect(SparseEmbedding, makeFastembedSparse(config));

  static readonly Default = SparseEmbedding.Fastembed();

  static readonly Test = Layer.succeed(
    SparseEmbedding,
    SparseEmbedding.of({
      descriptor: {
        method: "test",
        modelName: "test-sparse",
        vectorName: "sparse-test",
      },
      embed: (text) =>
        Effect.succeed({
          indices: [0],
          values: [text.length],
        }),
      embedBatch: (texts) =>
        Effect.succeed(
          texts.map((text, idx) => ({
            indices: [idx],
            values: [text.length],
          })),
        ),
    }),
  );
}
