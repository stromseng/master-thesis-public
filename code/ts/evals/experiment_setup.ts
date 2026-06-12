import { createOrGetDataset as createOrGetDatasetPhoenix } from "@arizeai/phoenix-client/datasets";
import {
  asExperimentEvaluator,
  evaluateExperiment,
  getExperimentInfo,
  getExperimentEvaluators,
  resumeEvaluation,
  resumeExperiment,
  runExperiment,
} from "@arizeai/phoenix-client/experiments";
import type {
  ResumeEvaluationParams,
  ResumeExperimentParams,
} from "@arizeai/phoenix-client/experiments";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { RetryError } from "ai";
import {
  OpenInferenceBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor,
} from "@arizeai/openinference-vercel";
import * as NodeSdk from "@effect/opentelemetry/NodeSdk";
import { context, propagation, ROOT_CONTEXT, trace } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import type { SpanProcessor } from "@opentelemetry/sdk-trace-base";
import type { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import type { Example } from "@arizeai/phoenix-client/types/datasets";
import type { EvaluatorParams } from "@arizeai/phoenix-client/types/experiments";
import * as Resource from "@effect/opentelemetry/Resource";
import * as Tracer from "@effect/opentelemetry/Tracer";
import * as Progress from "effective-progress";
import { Config, Duration, Effect, Layer, Option, Random, Ref, Runtime, Schedule } from "effect";
import type { JsonObject } from "type-fest";
import {
  DEFAULT_EVAL_CONNECTIVITY_POLL_INTERVAL,
  DEFAULT_EVAL_MAX_RETRIES,
  DEFAULT_EVAL_RETRY_SCHEDULE,
  EvalConcurrencyConfig,
} from "./constants";
import { LanguageModel } from "../src/services/LanguageModel";
import { PhoenixClient } from "../src/services/PhoenixClient";
import { Chunking } from "../src/services/Chunking";
import { Retrieval } from "../src/services/Retrieval";
import { EmbeddingProvider } from "../src/services/embeddings";

export type { Example, EvaluatorParams };
export { asExperimentEvaluator, PhoenixClient };

export type PhoenixOtelLayerParams = {
  projectName: string;
  collectorUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  batch?: boolean;
  global?: boolean;
};

const collectorUrlConfig = Config.string("PHOENIX_COLLECTOR_ENDPOINT").pipe(
  Config.orElse(() => Config.succeed("http://localhost:4317")),
);

const getCollectorUrl = Effect.fn("PhoenixOtel.getCollectorUrl")(function* (collectorUrl?: string) {
  if (collectorUrl) {
    return collectorUrl;
  }

  return yield* collectorUrlConfig;
});

const makeSpanProcessor = ({
  collectorUrl,
  apiKey,
  headers = {},
  batch = true,
}: Omit<PhoenixOtelLayerParams, "collectorUrl"> & { collectorUrl: string }): SpanProcessor => {
  const exporterHeaders: Record<string, string> = { ...headers };
  if (apiKey) {
    exporterHeaders.Authorization = `Bearer ${apiKey}`;
  }

  const exporter = new OTLPTraceExporter({
    url: collectorUrl,
    headers: exporterHeaders,
  });

  return batch
    ? new OpenInferenceBatchSpanProcessor({ exporter })
    : new OpenInferenceSimpleSpanProcessor({ exporter });
};

const resetOtelGlobals = Effect.sync(() => {
  trace.disable();
  context.disable();
  propagation.disable();
});

export const PhoenixOtelLayer = (params: PhoenixOtelLayerParams) => {
  return Layer.unwrapEffect(
    Effect.gen(function* () {
      const resolvedCollectorUrl = yield* getCollectorUrl(params.collectorUrl);

      const resourceAttributes = Resource.configToAttributes({
        serviceName: params.projectName,
        attributes: {
          [SEMRESATTRS_PROJECT_NAME]: params.projectName,
        },
      });

      const resourceLayer = Resource.layerFromEnv(resourceAttributes);

      const tracerProviderLayer = NodeSdk.layerTracerProvider(
        makeSpanProcessor({ ...params, collectorUrl: resolvedCollectorUrl }),
      ).pipe(Layer.provide(resourceLayer));

      const registerGlobalLayer =
        (params.global ?? true)
          ? Layer.scopedDiscard(
              Effect.acquireRelease(
                Effect.gen(function* () {
                  const provider = (yield* Tracer.OtelTracerProvider) as NodeTracerProvider;

                  // OTel globals are process-wide singletons. The eval runner creates a scoped
                  // provider per script, so the global registration must be scoped as well.
                  yield* resetOtelGlobals;
                  yield* Effect.sync(() => provider.register());
                }),
                () => resetOtelGlobals,
              ),
            ).pipe(Layer.provide(tracerProviderLayer))
          : Layer.empty;

      const tracerLayer = Tracer.layer.pipe(
        Layer.provide(Layer.mergeAll(resourceLayer, tracerProviderLayer)),
      );

      return Layer.mergeAll(resourceLayer, tracerProviderLayer, registerGlobalLayer, tracerLayer);
    }),
  );
};

export const PhoenixOtelLayerSkyhigh = (projectName: string) =>
  PhoenixOtelLayer({
    projectName,
    collectorUrl: "http://example.com:4317",
  });

export const EvalTaskLayerSkyhigh = Layer.mergeAll(
  PhoenixClient.skyhigh,
  PhoenixOtelLayerSkyhigh("evals"),
);

export const createOrGetDataset = Effect.fn("createDataset")(function* (
  params: Omit<Parameters<typeof createOrGetDatasetPhoenix>[0], "client">,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) => createOrGetDatasetPhoenix({ ...params, client }));
});

export const runPhoenixExperiment = Effect.fn("runPhoenixExperiment")(function* (
  params: Omit<Parameters<typeof runExperiment>[0], "client">,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) =>
    context.with(ROOT_CONTEXT, () => runExperiment({ ...params, client })),
  );
});

export const evaluatePhoenixExperiment = Effect.fn("evaluatePhoenixExperiment")(function* (
  params: Omit<Parameters<typeof evaluateExperiment>[0], "client">,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) =>
    context.with(ROOT_CONTEXT, () => evaluateExperiment({ ...params, client })),
  );
});

const ResumeExperimentId = Config.string("RESUME_EXPERIMENT_ID").pipe(Config.option);

const getPhoenixExperimentInfo = Effect.fn("getPhoenixExperimentInfo")(function* (
  experimentId: string,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) => getExperimentInfo({ client, experimentId }));
});

const resumePhoenixExperiment = Effect.fn("resumePhoenixExperiment")(function* (
  params: Omit<ResumeExperimentParams, "client">,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) =>
    context.with(ROOT_CONTEXT, () => resumeExperiment({ ...params, client })),
  );
});

const resumePhoenixEvaluation = Effect.fn("resumePhoenixEvaluation")(function* (
  params: Omit<ResumeEvaluationParams, "client">,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) =>
    context.with(ROOT_CONTEXT, () => resumeEvaluation({ ...params, client })),
  );
});

const rateLimitResetAtRegex =
  /Limit resets at:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\s+UTC)/i;

const findRetryError = (error: unknown): RetryError | null => {
  let current: unknown = error;
  for (let i = 0; i < 8; i += 1) {
    if (RetryError.isInstance(current)) {
      return current;
    }
    if (typeof current !== "object" || current === null || !("cause" in current)) {
      return null;
    }
    current = (current as { cause?: unknown }).cause;
  }
  return null;
};

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

const isConnectivityRetryError = (error: unknown): boolean => {
  const retryError = findRetryError(error);
  if (!retryError) {
    return false;
  }

  return [
    retryError.message,
    getErrorMessage(retryError.lastError),
    ...retryError.errors.map(getErrorMessage),
  ].some((message) => /cannot connect to api|unable to connect/i.test(message));
};

const getRateLimitDelay = <E>(error: E) =>
  Effect.gen(function* () {
    const retryError = findRetryError(error);
    if (!retryError) {
      return yield* Effect.fail(error);
    }

    const text = retryError.message;
    if (!/rate limit exceeded/i.test(text)) {
      return yield* Effect.fail(error);
    }

    const resetAtMatch = text.match(rateLimitResetAtRegex);
    const resetAtRaw = resetAtMatch?.[1];
    if (!resetAtRaw) {
      yield* Effect.logError(
        `Rate limit error has no 'Limit resets at' timestamp. Full message: ${text}`,
      );
      return yield* Effect.fail(error);
    }

    const resetAt = Date.parse(resetAtRaw);
    if (Number.isNaN(resetAt)) {
      yield* Effect.logError(
        `Rate limit error has unparseable reset timestamp: '${resetAtRaw}'. Full message: ${text}`,
      );
      return yield* Effect.fail(error);
    }

    // Small buffer to avoid retrying right before the limiter fully resets.
    return Duration.millis(Math.max(1_000, resetAt - Date.now() + 1_000));
  });

export const waitAndRetryOnRateLimit = Effect.fn("waitAndRetryOnRateLimit")(
  <A, E, R>(effect: Effect.Effect<A, E, R>, maxRetries?: number): Effect.Effect<A, E, R> => {
    const go = (remaining: number | undefined): Effect.Effect<A, E, R> =>
      effect.pipe(
        Effect.catchAll((error) =>
          Effect.gen(function* () {
            if (remaining !== undefined && remaining <= 0) {
              return yield* Effect.fail(error as E);
            }
            const delay = yield* getRateLimitDelay(error);

            const resumeAt = new Date(Date.now() + Duration.toMillis(delay)).toISOString();
            yield* Effect.logWarning("Rate limit exceeded, waiting before retry", {
              delay: Duration.format(delay),
              resumeAt,
              error: error instanceof Error ? error.message : String(error),
            });
            yield* Effect.sleep(delay);
            return yield* go(remaining !== undefined ? remaining - 1 : undefined);
          }),
        ),
      );
    return go(maxRetries);
  },
);

export const waitAndRetryOnConnectivityFailure = Effect.fn("waitAndRetryOnConnectivityFailure")(
  <A, E, R>(
    effect: Effect.Effect<A, E, R>,
    options?: { pollInterval?: Duration.DurationInput; maxRetries?: number },
  ): Effect.Effect<A, E, R> => {
    const pollInterval = Duration.decode(
      options?.pollInterval ?? DEFAULT_EVAL_CONNECTIVITY_POLL_INTERVAL,
    );

    const go = (remaining: number | undefined): Effect.Effect<A, E, R> =>
      effect.pipe(
        Effect.catchAll((error) =>
          Effect.gen(function* () {
            if (!isConnectivityRetryError(error)) {
              return yield* Effect.fail(error as E);
            }
            if (remaining !== undefined && remaining <= 0) {
              return yield* Effect.fail(error as E);
            }

            yield* Effect.logWarning("Language model connection failed, polling before retry", {
              delay: Duration.format(pollInterval),
              error: getErrorMessage(error),
            });
            yield* Effect.sleep(pollInterval);
            return yield* go(remaining !== undefined ? remaining - 1 : undefined);
          }),
        ),
      );
    return go(options?.maxRetries);
  },
);

/**
 * Combined retry policy for eval tasks:
 * rate-limit backoff → connectivity polling → log errors → exponential retry.
 *
 * Set MAX_TASK_RETRIES env var to cap all retry mechanisms (rate-limit,
 * connectivity, schedule). When exceeded the error propagates so Phoenix
 * records a failed run and can compute accuracy on the rest.
 */
export const taskRetryPolicy = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> =>
  Effect.gen(function* () {
    const maxTaskRetries = process.env.MAX_TASK_RETRIES
      ? Number.parseInt(process.env.MAX_TASK_RETRIES, 10)
      : undefined;

    const maxRetries = maxTaskRetries ?? DEFAULT_EVAL_MAX_RETRIES;

    const retrySchedule =
      maxTaskRetries !== undefined
        ? Schedule.exponential("1 second").pipe(
            Schedule.delayed((delay) => Duration.min(delay, Duration.minutes(30))),
            Schedule.addDelayEffect(() =>
              Random.nextIntBetween(0, Duration.toMillis(Duration.minutes(5)) + 1).pipe(
                Effect.map(Duration.millis),
              ),
            ),
            Schedule.intersect(Schedule.recurs(Math.max(0, maxTaskRetries - 1))),
          )
        : DEFAULT_EVAL_RETRY_SCHEDULE;

    const attempt = yield* Ref.make(0);
    const startTime = yield* Ref.make(performance.now());
    const runAttempt = Effect.gen(function* () {
      const now = performance.now();
      yield* Ref.update(attempt, (n) => n + 1);
      yield* Ref.set(startTime, now);
      return yield* effect;
    });

    return yield* runAttempt.pipe(
      (eff) => waitAndRetryOnRateLimit(eff, maxTaskRetries),
      (eff) => waitAndRetryOnConnectivityFailure(eff, { maxRetries: maxTaskRetries }),
      Effect.tapErrorCause((cause) =>
        Effect.all([Ref.get(attempt), Ref.get(startTime)]).pipe(
          Effect.andThen(([n, start]) => {
            const elapsed = ((performance.now() - start) / 1000).toFixed(1);
            const willRetry = n <= maxRetries;
            const log = willRetry ? Effect.logWarning : Effect.logError;
            return log(`Eval task error${willRetry ? ", will retry" : ", giving up"}`, {
              cause,
              elapsed: `${elapsed}s`,
              attempt: n,
            });
          }),
        ),
      ),
      Effect.retry(retrySchedule),
    );
  });

// Returns dataset info or null if not found
export const getDatasetByName = Effect.fn("getDatasetByName")(function* (name: string) {
  const phoenix = yield* PhoenixClient;
  const response = yield* phoenix.use((client) =>
    client.GET("/v1/datasets", { params: { query: { name } } }),
  );
  const datasets = response.data?.data ?? [];
  return datasets[0] as { id: string; example_count: number } | null;
});

// Builds an experiment name from metadata.
// Examples:
//   "Kimi-K2.5"
//   "Kimi-K2.5 (RAG k=10, e5-large-instruct+bm42+colbertv2.5, recursive)"
const buildExperimentName = (metadata: ExperimentMetadata): string => {
  const modelId = metadata.model.id;
  if (!("retrieval" in metadata)) return modelId;

  const parts: string[] = [];

  parts.push(`k=${metadata.retrieval.limit}`);

  const embeddingNames: string[] = [];
  if (metadata.embeddings.dense) embeddingNames.push(metadata.embeddings.dense.model);
  if (metadata.embeddings.sparse) embeddingNames.push(metadata.embeddings.sparse.model);
  if (metadata.embeddings.late) embeddingNames.push(metadata.embeddings.late.model);
  if (embeddingNames.length > 0) parts.push(embeddingNames.join("+"));

  if (metadata.chunking) parts.push(metadata.chunking.method);

  return `${modelId} (RAG ${parts.join(", ")})`;
};

// Wraps runPhoenixExperiment with metadata extraction, logging, and a progress bar.
// Automatically extracts experiment metadata from the environment (LanguageModel, Retrieval, etc.)
// and reads EVAL_CONCURRENCY config, so callers don't need to do this manually.
// The experiment name is built automatically from metadata unless overridden.
export const runPhoenixExperimentWithProgress = Effect.fn("runPhoenixExperimentWithProgress")(
  function* (
    params: Omit<
      Parameters<typeof runExperiment>[0],
      "client" | "experimentMetadata" | "concurrency" | "experimentName"
    > & {
      total: number;
      experimentName?: string | ((metadata: ExperimentMetadata) => string);
    },
  ) {
    const { total, task, experimentName: experimentNameOrFn, ...rest } = params;

    // Resume mode: if RESUME_EXPERIMENT_ID is set, resume the existing experiment
    // instead of starting a new one.
    const resumeId = yield* ResumeExperimentId;
    if (Option.isSome(resumeId)) {
      const evalConcurrency = yield* EvalConcurrencyConfig;
      const experimentId = resumeId.value;

      const info = yield* getPhoenixExperimentInfo(experimentId);
      const requestedDatasetId = "datasetId" in params.dataset ? params.dataset.datasetId : null;
      if (requestedDatasetId !== null && requestedDatasetId !== info.datasetId) {
        return yield* Effect.die(
          [
            `RESUME_EXPERIMENT_ID ${experimentId} belongs to dataset ${info.datasetId},`,
            `but this script is configured for dataset ${requestedDatasetId}.`,
            "Use the eval script that created the experiment, or update the resume experiment id.",
          ].join(" "),
        );
      }
      const incompleteCount = info.exampleCount * info.repetitions - info.successfulRunCount;

      if (incompleteCount === 0) {
        yield* Effect.log("No incomplete runs found. Experiment is already complete.", {
          experimentId,
        });
        if (params.evaluators && params.evaluators.length > 0) {
          yield* Effect.log("Checking for incomplete evaluations", { experimentId });
          yield* resumePhoenixEvaluation({
            experimentId,
            evaluators: params.evaluators as readonly any[],
            concurrency: evalConcurrency,
            useBatchSpanProcessor: rest.useBatchSpanProcessor ?? false,
            setGlobalTracerProvider: rest.setGlobalTracerProvider ?? false,
          });
        }
        return;
      }

      yield* Effect.log("Resuming experiment", {
        experimentId,
        incompleteCount,
        totalExamples: info.exampleCount,
        successfulRuns: info.successfulRunCount,
      });

      yield* Progress.task(
        Effect.gen(function* () {
          const progress = yield* Progress.Progress;
          const taskId = yield* Progress.Task;
          const runtime = yield* Effect.runtime<never>();
          const runSync = Runtime.runSync(runtime);

          const wrappedTask: typeof task = async (example) => {
            try {
              const result = await task(example);
              runSync(progress.incrementSucceeded(taskId, 1));
              return result;
            } catch (error) {
              runSync(progress.incrementFailed(taskId, 1));
              throw error;
            }
          };

          yield* resumePhoenixExperiment({
            experimentId,
            task: wrappedTask,
            evaluators: params.evaluators as readonly any[],
            concurrency: evalConcurrency,
            useBatchSpanProcessor: rest.useBatchSpanProcessor ?? false,
            setGlobalTracerProvider: rest.setGlobalTracerProvider ?? false,
          });
        }),
        { description: `Resuming experiment: ${experimentId}`, total: incompleteCount },
      );

      return;
    }

    const metadata = yield* extractExperimentMetadata;
    const evalConcurrency = yield* EvalConcurrencyConfig;
    const experimentName = experimentNameOrFn
      ? typeof experimentNameOrFn === "function"
        ? experimentNameOrFn(metadata)
        : experimentNameOrFn
      : buildExperimentName(metadata);

    yield* Effect.log("Experiment metadata", metadata);

    yield* Progress.task(
      Effect.gen(function* () {
        const progress = yield* Progress.Progress;
        const taskId = yield* Progress.Task;
        const runtime = yield* Effect.runtime<never>();
        const runSync = Runtime.runSync(runtime);

        let completed = 0;
        const wrappedTask: typeof task = async (example) => {
          try {
            const result = await task(example);
            completed += 1;
            // Bridge async callback completion into the Effect progress runtime.
            runSync(progress.incrementSucceeded(taskId, 1));
            return result;
          } catch (error) {
            completed += 1;
            runSync(progress.incrementFailed(taskId, 1));
            throw error;
          }
        };

        yield* runPhoenixExperiment({
          ...rest,
          experimentName,
          experimentMetadata: metadata,
          concurrency: evalConcurrency,
          task: wrappedTask,
        });
      }),
      { description: `Running experiment: ${experimentName}`, total },
    );
  },
);

export const runPhoenixExperimentOnlyWithProgress = Effect.fn(
  "runPhoenixExperimentOnlyWithProgress",
)(function* (
  params: Omit<
    Parameters<typeof runExperiment>[0],
    "client" | "experimentMetadata" | "concurrency" | "experimentName" | "evaluators"
  > & {
    total: number;
    experimentName?: string | ((metadata: ExperimentMetadata) => string);
  },
) {
  const { total, task, experimentName: experimentNameOrFn, ...rest } = params;

  const resumeId = yield* ResumeExperimentId;
  if (Option.isSome(resumeId)) {
    const evalConcurrency = yield* EvalConcurrencyConfig;
    const experimentId = resumeId.value;
    const info = yield* getPhoenixExperimentInfo(experimentId);
    const incompleteCount = info.exampleCount * info.repetitions - info.successfulRunCount;

    if (incompleteCount === 0) {
      yield* Effect.log("No incomplete runs found. Experiment is already complete.", {
        experimentId,
      });
      return experimentId;
    }

    yield* Progress.task(
      Effect.gen(function* () {
        const progress = yield* Progress.Progress;
        const taskId = yield* Progress.Task;
        const runtime = yield* Effect.runtime<never>();
        const runSync = Runtime.runSync(runtime);

        const wrappedTask: typeof task = async (example) => {
          try {
            const result = await task(example);
            runSync(progress.incrementSucceeded(taskId, 1));
            return result;
          } catch (error) {
            runSync(progress.incrementFailed(taskId, 1));
            throw error;
          }
        };

        yield* resumePhoenixExperiment({
          experimentId,
          task: wrappedTask,
          evaluators: [],
          concurrency: evalConcurrency,
          useBatchSpanProcessor: rest.useBatchSpanProcessor ?? false,
          setGlobalTracerProvider: rest.setGlobalTracerProvider ?? false,
        });
      }),
      { description: `Resuming experiment: ${experimentId}`, total: incompleteCount },
    );

    return experimentId;
  }

  const metadata = yield* extractExperimentMetadata;
  const evalConcurrency = yield* EvalConcurrencyConfig;
  const experimentName = experimentNameOrFn
    ? typeof experimentNameOrFn === "function"
      ? experimentNameOrFn(metadata)
      : experimentNameOrFn
    : buildExperimentName(metadata);

  yield* Effect.log("Experiment metadata", metadata);

  const ranExperiment = yield* Progress.task(
    Effect.gen(function* () {
      const progress = yield* Progress.Progress;
      const taskId = yield* Progress.Task;
      const runtime = yield* Effect.runtime<never>();
      const runSync = Runtime.runSync(runtime);

      const wrappedTask: typeof task = async (example) => {
        try {
          const result = await task(example);
          runSync(progress.incrementSucceeded(taskId, 1));
          return result;
        } catch (error) {
          runSync(progress.incrementFailed(taskId, 1));
          throw error;
        }
      };

      return yield* runPhoenixExperiment({
        ...rest,
        experimentName,
        experimentMetadata: metadata,
        concurrency: evalConcurrency,
        task: wrappedTask,
        evaluators: [],
      });
    }),
    { description: `Running experiment: ${experimentName}`, total },
  );

  return ranExperiment.id;
});

export const evaluatePhoenixExperimentWithProgress = Effect.fn(
  "evaluatePhoenixExperimentWithProgress",
)(function* (
  params: Omit<
    Parameters<typeof resumeEvaluation>[0],
    "client" | "experimentId" | "concurrency"
  > & {
    experimentId: string;
  },
) {
  const { experimentId, evaluators, ...rest } = params;
  const evalConcurrency = yield* EvalConcurrencyConfig;
  const info = yield* getPhoenixExperimentInfo(experimentId);
  const normalizedEvaluators = getExperimentEvaluators(
    Array.isArray(evaluators) ? [...evaluators] : [evaluators],
  );

  const evaluatorNames = normalizedEvaluators.map((e) => e.name);
  yield* Effect.log("[evaluate-only] phase starting", {
    experimentId,
    successfulRunCount: info.successfulRunCount,
    exampleCount: info.exampleCount,
    failedRunCount: info.failedRunCount,
    missingRunCount: info.missingRunCount,
    evaluatorNames,
    concurrency: evalConcurrency,
  });

  const phoenix = yield* PhoenixClient;
  const preflight = yield* phoenix.use((client) =>
    client.GET("/v1/experiments/{experiment_id}/incomplete-evaluations", {
      params: {
        path: { experiment_id: experimentId },
        query: { limit: 1, evaluation_name: evaluatorNames },
      },
    }),
  );
  const incompleteFirstPage = preflight.data?.data?.length ?? 0;
  const hasMorePages = preflight.data?.next_cursor != null;

  yield* Effect.log("[evaluate-only] incomplete evaluations pre-flight", {
    incompleteFirstPage,
    hasMorePages,
    allComplete: incompleteFirstPage === 0,
  });

  if (incompleteFirstPage === 0) {
    yield* Effect.log("[evaluate-only] all evaluations already complete — nothing to do");
    return;
  }

  return yield* Progress.task(
    Effect.gen(function* () {
      const progress = yield* Progress.Progress;
      const taskId = yield* Progress.Task;
      const runtime = yield* Effect.runtime<never>();
      const runSync = Runtime.runSync(runtime);
      const wrappedEvaluators = normalizedEvaluators.map((evaluator) => ({
        ...evaluator,
        evaluate: async (args: EvaluatorParams) => {
          try {
            const result = await evaluator.evaluate(args);
            runSync(progress.incrementSucceeded(taskId, 1));
            return result;
          } catch (error) {
            runSync(progress.incrementFailed(taskId, 1));
            throw error;
          }
        },
      }));

      yield* Effect.log("[evaluate-only] calling resumePhoenixEvaluation");
      yield* resumePhoenixEvaluation({
        ...rest,
        experimentId,
        evaluators: wrappedEvaluators,
        concurrency: evalConcurrency,
      });
      yield* Effect.log("[evaluate-only] resumePhoenixEvaluation completed successfully");
    }),
    {
      description: `Resuming evaluations: ${experimentId}`,
      total: info.successfulRunCount * normalizedEvaluators.length,
      countDisplay: "processedOnly",
    },
  );
});

const extractExperimentMetadata = Effect.gen(function* () {
  const languageModel = yield* LanguageModel;
  const modelId = typeof languageModel === "string" ? languageModel : languageModel.modelId;
  const retrievalOpt = yield* Effect.serviceOption(Retrieval);
  const embeddingProviderOpt = yield* Effect.serviceOption(EmbeddingProvider);
  const chunkingOpt = yield* Effect.serviceOption(Chunking);
  const denseOpt = Option.isSome(embeddingProviderOpt)
    ? embeddingProviderOpt.value.dense
    : Option.none();
  const sparseOpt = Option.isSome(embeddingProviderOpt)
    ? embeddingProviderOpt.value.sparse
    : Option.none();
  const lateOpt = Option.isSome(embeddingProviderOpt)
    ? embeddingProviderOpt.value.late
    : Option.none();

  const metadata = (() => {
    const base = {
      model: {
        id: modelId,
      },
    };

    if (Option.isSome(retrievalOpt)) {
      const retrievalConfig = retrievalOpt.value.config;

      return {
        ...base,
        retrieval: {
          limit: retrievalConfig.limit,
          prefetchLimit: retrievalConfig.prefetchLimit,
          collection: retrievalOpt.value.collectionName,
        },
        chunking: Option.isSome(chunkingOpt) ? { method: chunkingOpt.value.method } : null,
        embeddings: {
          dense: Option.isSome(denseOpt)
            ? {
                method: denseOpt.value.descriptor.method,
                model: denseOpt.value.descriptor.modelName,
                vector: denseOpt.value.descriptor.vectorName,
              }
            : null,
          sparse: Option.isSome(sparseOpt)
            ? {
                method: sparseOpt.value.descriptor.method,
                model: sparseOpt.value.descriptor.modelName,
                vector: sparseOpt.value.descriptor.vectorName,
              }
            : null,
          late: Option.isSome(lateOpt)
            ? {
                method: lateOpt.value.descriptor.method,
                model: lateOpt.value.descriptor.modelName,
                vector: lateOpt.value.descriptor.vectorName,
              }
            : null,
        },
      };
    }

    return base;
  })() satisfies JsonObject;

  return metadata;
});

type ExperimentMetadata = Effect.Effect.Success<typeof extractExperimentMetadata>;
