import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { embed, embedMany } from "ai";
import { Config, Context, Duration, Effect, Layer, Redacted, Schedule, Schema } from "effect";
import {
  embedDense,
  embedDenseBatch,
  resolveDenseEmbeddingConfig,
} from "../../generated/python-api/sdk.gen";
import {
  DEFAULT_DENSE_MODEL_NAME,
  DEFAULT_DENSE_VECTOR_NAME,
  DEFAULT_LITELLM_DENSE_MODEL_NAME,
  DEFAULT_LITELLM_DENSE_VECTOR_NAME,
} from "./defaults";
import { PYTHON_API_TIMEOUT } from "../pythonApi";
import { PythonApiClient } from "../PythonApiClient";

export class DenseResolveError extends Schema.TaggedError<DenseResolveError>()(
  "DenseResolveError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class DenseResolveTimeoutError extends Schema.TaggedError<DenseResolveTimeoutError>()(
  "DenseResolveTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class DenseResolveMissingVectorSizeError extends Schema.TaggedError<DenseResolveMissingVectorSizeError>()(
  "DenseResolveMissingVectorSizeError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class DenseEmbedError extends Schema.TaggedError<DenseEmbedError>()("DenseEmbedError", {
  modelName: Schema.String,
  vectorName: Schema.String,
  cause: Schema.Defect,
}) {}

export class DenseEmbedTimeoutError extends Schema.TaggedError<DenseEmbedTimeoutError>()(
  "DenseEmbedTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class DenseEmbedMissingDataError extends Schema.TaggedError<DenseEmbedMissingDataError>()(
  "DenseEmbedMissingDataError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class DenseEmbedBatchError extends Schema.TaggedError<DenseEmbedBatchError>()(
  "DenseEmbedBatchError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export class DenseEmbedBatchTimeoutError extends Schema.TaggedError<DenseEmbedBatchTimeoutError>()(
  "DenseEmbedBatchTimeoutError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    timeout: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class DenseEmbedBatchMissingDataError extends Schema.TaggedError<DenseEmbedBatchMissingDataError>()(
  "DenseEmbedBatchMissingDataError",
  {
    modelName: Schema.String,
    vectorName: Schema.String,
    batchSize: Schema.Number,
    cause: Schema.Defect,
  },
) {}

export type DenseEmbeddingError =
  | DenseResolveError
  | DenseResolveTimeoutError
  | DenseResolveMissingVectorSizeError
  | DenseEmbedError
  | DenseEmbedTimeoutError
  | DenseEmbedMissingDataError
  | DenseEmbedBatchError
  | DenseEmbedBatchTimeoutError
  | DenseEmbedBatchMissingDataError;

export type DenseEmbeddingMethod = "fastembed" | "litellm" | "test";

export interface DenseEmbeddingDescriptor {
  readonly method: DenseEmbeddingMethod;
  readonly modelName: string;
  readonly vectorName: string;
  readonly vectorSize: number;
}

export interface DenseEmbeddingService {
  readonly descriptor: DenseEmbeddingDescriptor;
  readonly embed: (text: string) => Effect.Effect<number[], DenseEmbeddingError>;
  readonly embedBatch: (texts: string[]) => Effect.Effect<number[][], DenseEmbeddingError>;
}

export interface FastembedDenseConfig {
  readonly modelName?: string;
  readonly vectorName?: string;
}

export interface LiteLLMDenseConfig {
  readonly modelName?: string;
  readonly vectorName?: string;
}

const LITELLM_EMBED_TIMEOUT = "15 minutes" as const;
const LITELLM_RATE_LIMIT_WAIT = "2 minutes" as const;
const LITELLM_RATE_LIMIT_RETRIES = 5;

const isRateLimitError = (error: unknown): boolean => {
  const msg = String(error);
  return msg.includes("Rate limit") || msg.includes("rate_limit") || msg.includes("429");
};

const rateLimitRetryPolicy = Schedule.intersect(
  Schedule.recurs(LITELLM_RATE_LIMIT_RETRIES),
  Schedule.fixed(Duration.decode(LITELLM_RATE_LIMIT_WAIT)),
);

const makeLiteLLMDense = (config?: LiteLLMDenseConfig) =>
  Effect.gen(function* () {
    const apiKey = yield* Config.redacted("LITE_LLM_API_KEY");
    const baseUrl = yield* Config.string("LITE_LLM_BASE_URL").pipe(
      Config.orElse(() => Config.succeed("https://llm.hpc.ntnu.no/v1")),
    );

    const modelName = config?.modelName ?? DEFAULT_LITELLM_DENSE_MODEL_NAME;
    const vectorName = config?.vectorName ?? DEFAULT_LITELLM_DENSE_VECTOR_NAME;

    const provider = yield* Effect.try({
      try: () =>
        createOpenAICompatible({
          name: "litellm-embed",
          apiKey: Redacted.value(apiKey),
          baseURL: baseUrl,
        }),
      catch: (cause) => new DenseResolveError({ modelName, vectorName, cause }),
    });

    // Probe the model to determine vector size
    const probeResult = yield* Effect.tryPromise({
      try: () =>
        embed({
          model: provider.embeddingModel(modelName),
          value: "probe",
        }),
      catch: (cause) => new DenseResolveError({ modelName, vectorName, cause }),
    }).pipe(
      Effect.timeoutFail({
        duration: LITELLM_EMBED_TIMEOUT,
        onTimeout: () =>
          new DenseResolveTimeoutError({
            modelName,
            vectorName,
            timeout: LITELLM_EMBED_TIMEOUT,
            cause: new Error(`Probe embedding timed out after ${LITELLM_EMBED_TIMEOUT}`),
          }),
      }),
    );

    const vectorSize = probeResult.embedding.length;
    if (!vectorSize) {
      return yield* new DenseResolveMissingVectorSizeError({
        modelName,
        vectorName,
        cause: new Error("Probe embedding returned empty vector"),
      });
    }

    yield* Effect.log(`LiteLLM dense embedding resolved: ${modelName} (${vectorSize} dims)`);

    // Semaphore serializes LiteLLM API calls to stay under rate limit (20 req/min).
    // Multiple batches can run concurrently in the pipeline, but dense API calls
    // go through one at a time with a small delay to pace requests.
    const apiSemaphore = yield* Effect.makeSemaphore(1);
    const LITELLM_PACING_DELAY = "3 seconds" as const;
    const rateLimitedCall = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
      apiSemaphore.withPermits(1)(
        Effect.flatMap(effect, (a) => Effect.as(Effect.sleep(LITELLM_PACING_DELAY), a)),
      );

    return DenseEmbedding.of({
      descriptor: {
        method: "litellm",
        modelName,
        vectorName,
        vectorSize,
      },
      embed: Effect.fn("DenseEmbedding.embed.litellm")(function* (text: string) {
        const result = yield* rateLimitedCall(
          Effect.tryPromise({
            try: () =>
              embed({
                model: provider.embeddingModel(modelName),
                value: text,
              }),
            catch: (cause) => new DenseEmbedError({ modelName, vectorName, cause }),
          }),
        ).pipe(
          Effect.tapError((err) =>
            isRateLimitError(err.cause)
              ? Effect.log(`Rate limited (embed), waiting ${LITELLM_RATE_LIMIT_WAIT}...`)
              : Effect.void,
          ),
          Effect.retry({
            while: (err) => isRateLimitError(err.cause),
            schedule: rateLimitRetryPolicy,
          }),
          Effect.timeoutFail({
            duration: LITELLM_EMBED_TIMEOUT,
            onTimeout: () =>
              new DenseEmbedTimeoutError({
                modelName,
                vectorName,
                timeout: LITELLM_EMBED_TIMEOUT,
                cause: new Error(`Operation exceeded timeout: ${LITELLM_EMBED_TIMEOUT}`),
              }),
          }),
        );

        return result.embedding;
      }),
      embedBatch: Effect.fn("DenseEmbedding.embedBatch.litellm")(function* (texts: string[]) {
        const result = yield* rateLimitedCall(
          Effect.tryPromise({
            try: () =>
              embedMany({
                model: provider.embeddingModel(modelName),
                values: texts,
              }),
            catch: (cause) =>
              new DenseEmbedBatchError({
                modelName,
                vectorName,
                batchSize: texts.length,
                cause,
              }),
          }),
        ).pipe(
          Effect.tapError((err) =>
            isRateLimitError(err.cause)
              ? Effect.log(
                  `Rate limited (embedBatch ${texts.length} texts), waiting ${LITELLM_RATE_LIMIT_WAIT}...`,
                )
              : Effect.void,
          ),
          Effect.retry({
            while: (err) => isRateLimitError(err.cause),
            schedule: rateLimitRetryPolicy,
          }),
          Effect.timeoutFail({
            duration: LITELLM_EMBED_TIMEOUT,
            onTimeout: () =>
              new DenseEmbedBatchTimeoutError({
                modelName,
                vectorName,
                batchSize: texts.length,
                timeout: LITELLM_EMBED_TIMEOUT,
                cause: new Error(`Operation exceeded timeout: ${LITELLM_EMBED_TIMEOUT}`),
              }),
          }),
        );

        return result.embeddings;
      }),
    });
  });

const makeFastembedDense = (config?: FastembedDenseConfig) =>
  Effect.gen(function* () {
    const { client } = yield* PythonApiClient;
    const modelName = config?.modelName ?? DEFAULT_DENSE_MODEL_NAME;
    const vectorName = config?.vectorName ?? DEFAULT_DENSE_VECTOR_NAME;
    const resolved = yield* Effect.tryPromise(() =>
      resolveDenseEmbeddingConfig({
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
          new DenseResolveError({
            modelName,
            vectorName,
            cause: error,
          }),
      ),
      Effect.timeoutFail({
        duration: PYTHON_API_TIMEOUT,
        onTimeout: () =>
          new DenseResolveTimeoutError({
            modelName,
            vectorName,
            timeout: PYTHON_API_TIMEOUT,
            cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
          }),
      }),
    );

    const vectorSize = resolved.data?.vector_size;
    if (!vectorSize) {
      return yield* new DenseResolveMissingVectorSizeError({
        modelName,
        vectorName,
        cause: resolved.error ?? new Error("Resolve response missing data"),
      });
    }

    return DenseEmbedding.of({
      descriptor: {
        method: "fastembed",
        modelName,
        vectorName,
        vectorSize,
      },
      embed: Effect.fn("DenseEmbedding.embed.fastembed")(function* (text: string) {
        const response = yield* Effect.tryPromise(() =>
          embedDense({
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
              new DenseEmbedError({
                modelName,
                vectorName,
                cause: error,
              }),
          ),
          Effect.timeoutFail({
            duration: PYTHON_API_TIMEOUT,
            onTimeout: () =>
              new DenseEmbedTimeoutError({
                modelName,
                vectorName,
                timeout: PYTHON_API_TIMEOUT,
                cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
              }),
          }),
        );

        const embedding = response.data?.embedding;
        if (!embedding) {
          return yield* new DenseEmbedMissingDataError({
            modelName,
            vectorName,
            cause: response.error ?? new Error("Dense embed response missing data"),
          });
        }

        return embedding;
      }),
      embedBatch: Effect.fn("DenseEmbedding.embedBatch.fastembed")(function* (texts: string[]) {
        const response = yield* Effect.tryPromise(() =>
          embedDenseBatch({
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
              new DenseEmbedBatchError({
                modelName,
                vectorName,
                batchSize: texts.length,
                cause: error,
              }),
          ),
          Effect.timeoutFail({
            duration: PYTHON_API_TIMEOUT,
            onTimeout: () =>
              new DenseEmbedBatchTimeoutError({
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
          return yield* new DenseEmbedBatchMissingDataError({
            modelName,
            vectorName,
            batchSize: texts.length,
            cause: response.error ?? new Error("Dense embed batch response missing data"),
          });
        }

        return embeddings;
      }),
    });
  });

export class DenseEmbedding extends Context.Tag("@app/DenseEmbedding")<
  DenseEmbedding,
  DenseEmbeddingService
>() {
  static readonly Fastembed = (config?: FastembedDenseConfig) =>
    Layer.effect(DenseEmbedding, makeFastembedDense(config));

  static readonly LiteLLM = (config?: LiteLLMDenseConfig) =>
    Layer.effect(DenseEmbedding, makeLiteLLMDense(config));

  static readonly Default = DenseEmbedding.Fastembed();

  static readonly Test = Layer.succeed(
    DenseEmbedding,
    DenseEmbedding.of({
      descriptor: {
        method: "test",
        modelName: "test-dense",
        vectorName: "dense-test",
        vectorSize: 4,
      },
      embed: (text) => Effect.succeed([text.length, 1, 2, 3]),
      embedBatch: (texts) => Effect.succeed(texts.map((t) => [t.length, 1, 2, 3])),
    }),
  );
}
