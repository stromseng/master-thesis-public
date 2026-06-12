import { Context, Effect, Layer, Schema } from "effect";
import {
  embedLate,
  embedLateBatch,
  resolveLateEmbeddingConfig,
} from "../../generated/python-api/sdk.gen";
import { DEFAULT_LATE_MODEL_NAME, DEFAULT_LATE_VECTOR_NAME } from "./defaults";
import { PYTHON_API_TIMEOUT } from "../pythonApi";
import { PythonApiClient } from "../PythonApiClient";

export class LateResolveError extends Schema.TaggedError<LateResolveError>()("LateResolveError", {
  modelName: Schema.String,
  vectorName: Schema.String,
  cause: Schema.Defect,
}) {}

export class LateResolveTimeoutError extends Schema.TaggedError<LateResolveTimeoutError>()(
  "LateResolveTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class LateResolveMissingVectorSizeError extends Schema.TaggedError<LateResolveMissingVectorSizeError>()(
  "LateResolveMissingVectorSizeError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class LateEmbedError extends Schema.TaggedError<LateEmbedError>()("LateEmbedError", {
  modelName: Schema.String,
  vectorName: Schema.String,
  cause: Schema.Defect,
}) {}

export class LateEmbedTimeoutError extends Schema.TaggedError<LateEmbedTimeoutError>()(
  "LateEmbedTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class LateEmbedMissingDataError extends Schema.TaggedError<LateEmbedMissingDataError>()(
  "LateEmbedMissingDataError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class LateEmbedBatchError extends Schema.TaggedError<LateEmbedBatchError>()(
  "LateEmbedBatchError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export class LateEmbedBatchTimeoutError extends Schema.TaggedError<LateEmbedBatchTimeoutError>()(
  "LateEmbedBatchTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class LateEmbedBatchMissingDataError extends Schema.TaggedError<LateEmbedBatchMissingDataError>()(
  "LateEmbedBatchMissingDataError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export type LateEmbeddingError =
  | LateResolveError
  | LateResolveTimeoutError
  | LateResolveMissingVectorSizeError
  | LateEmbedError
  | LateEmbedTimeoutError
  | LateEmbedMissingDataError
  | LateEmbedBatchError
  | LateEmbedBatchTimeoutError
  | LateEmbedBatchMissingDataError;

export type LateEmbeddingMethod = "fastembed" | "test";

export interface LateEmbeddingDescriptor {
  readonly method: LateEmbeddingMethod;
  readonly modelName: string;
  readonly vectorName: string;
  readonly vectorSize: number;
}

export interface LateEmbeddingService {
  readonly descriptor: LateEmbeddingDescriptor;
  readonly embed: (text: string) => Effect.Effect<number[][], LateEmbeddingError>;
  readonly embedBatch: (texts: string[]) => Effect.Effect<number[][][], LateEmbeddingError>;
}

export interface FastembedLateConfig {
  readonly modelName?: string;
  readonly vectorName?: string;
}

const makeFastembedLate = (config?: FastembedLateConfig) =>
  Effect.gen(function* () {
    const { client } = yield* PythonApiClient;
    const modelName = config?.modelName ?? DEFAULT_LATE_MODEL_NAME;
    const vectorName = config?.vectorName ?? DEFAULT_LATE_VECTOR_NAME;

    const resolved = yield* Effect.tryPromise(() =>
      resolveLateEmbeddingConfig({
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
          new LateResolveError({
            modelName,
            vectorName,
            cause: error,
          }),
      ),
      Effect.timeoutFail({
        duration: PYTHON_API_TIMEOUT,
        onTimeout: () =>
          new LateResolveTimeoutError({
            modelName,
            vectorName,
            timeout: PYTHON_API_TIMEOUT,
            cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
          }),
      }),
    );

    const vectorSize = resolved.data?.vector_size;
    if (!vectorSize) {
      return yield* new LateResolveMissingVectorSizeError({
        modelName,
        vectorName,
        cause: resolved.error ?? new Error("Resolve response missing data"),
      });
    }

    const embed = Effect.fn("LateEmbedding.embed.fastembed")(function* (text: string) {
      const response = yield* Effect.tryPromise(() =>
        embedLate({
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
            new LateEmbedError({
              modelName,
              vectorName,
              cause: error,
            }),
        ),
        Effect.timeoutFail({
          duration: PYTHON_API_TIMEOUT,
          onTimeout: () =>
            new LateEmbedTimeoutError({
              modelName,
              vectorName,
              timeout: PYTHON_API_TIMEOUT,
              cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
            }),
        }),
      );

      const embeddings = response.data?.embeddings;
      if (!embeddings) {
        return yield* new LateEmbedMissingDataError({
          modelName,
          vectorName,
          cause: response.error ?? new Error("Late embed response missing data"),
        });
      }

      return embeddings;
    });

    const embedBatch = Effect.fn("LateEmbedding.embedBatch.fastembed")(function* (texts: string[]) {
      const response = yield* Effect.tryPromise(() =>
        embedLateBatch({
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
            new LateEmbedBatchError({
              modelName,
              vectorName,
              batchSize: texts.length,
              cause: error,
            }),
        ),
        Effect.timeoutFail({
          duration: PYTHON_API_TIMEOUT,
          onTimeout: () =>
            new LateEmbedBatchTimeoutError({
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
        return yield* new LateEmbedBatchMissingDataError({
          modelName,
          vectorName,
          batchSize: texts.length,
          cause: response.error ?? new Error("Late embed batch response missing data"),
        });
      }

      return embeddings;
    });

    return LateEmbedding.of({
      descriptor: {
        method: "fastembed",
        modelName,
        vectorName,
        vectorSize,
      },
      embed,
      embedBatch,
    });
  });

export class LateEmbedding extends Context.Tag("@app/LateEmbedding")<
  LateEmbedding,
  LateEmbeddingService
>() {
  static readonly Fastembed = (config?: FastembedLateConfig) =>
    Layer.effect(LateEmbedding, makeFastembedLate(config));

  static readonly Default = LateEmbedding.Fastembed();

  static readonly Test = Layer.succeed(
    LateEmbedding,
    LateEmbedding.of({
      descriptor: {
        method: "test",
        modelName: "test-late",
        vectorName: "late-test",
        vectorSize: 2,
      },
      embed: (_text) => Effect.succeed([[0, 1]]),
      embedBatch: (texts) => Effect.succeed(texts.map(() => [[0, 1]])),
    }),
  );
}
