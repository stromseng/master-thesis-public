import type { Layer } from "effect";
import { Config, Duration, Effect, Random, Schedule } from "effect";
import { Chunking } from "../src/services/Chunking";
import { DenseEmbedding } from "../src/services/embeddings/Dense";
import type { PythonApiClient } from "../src/services/PythonApiClient";

const DEFAULT_EVAL_CONCURRENCY = (process.env.EVAL_PROVIDER ?? "litellm") === "vllm" ? 10 : 5;
export const EvalConcurrencyConfig = Config.integer("EVAL_CONCURRENCY").pipe(
  Config.orElse(() => Config.succeed(DEFAULT_EVAL_CONCURRENCY)),
  Config.validate({
    message: "EVAL_CONCURRENCY must be an integer >= 1",
    validation: (n): n is number => n >= 1,
  }),
);

const DEFAULT_EVAL_MAX_ATTEMPTS = 10;
const DEFAULT_EVAL_MAX_RETRY_DELAY = Duration.minutes(30);
const DEFAULT_EVAL_MAX_RETRY_JITTER = Duration.minutes(5);
export const DEFAULT_EVAL_CONNECTIVITY_POLL_INTERVAL = Duration.seconds(10);

export const DEFAULT_EVAL_MAX_RETRIES = DEFAULT_EVAL_MAX_ATTEMPTS - 1;

export const DEFAULT_EVAL_RETRY_SCHEDULE = Schedule.exponential("1 second").pipe(
  Schedule.delayed((delay) => Duration.min(delay, DEFAULT_EVAL_MAX_RETRY_DELAY)),
  Schedule.addDelayEffect(() =>
    Random.nextIntBetween(0, Duration.toMillis(DEFAULT_EVAL_MAX_RETRY_JITTER) + 1).pipe(
      Effect.map(Duration.millis),
    ),
  ),
  Schedule.intersect(Schedule.recurs(DEFAULT_EVAL_MAX_RETRIES)),
);

/**
 * Resolve the dense embedding layer from env vars:
 *   DENSE_METHOD=litellm|fastembed (default: litellm)
 *   DENSE_MODEL=<model-name>       (optional override)
 */
export const resolveDenseLayer = (): Layer.Layer<DenseEmbedding, any, PythonApiClient> => {
  const method = process.env.DENSE_METHOD ?? "litellm";
  const model = process.env.DENSE_MODEL;
  const config = model ? { modelName: model, vectorName: model.split("/").pop()! } : undefined;

  return method === "litellm" ? DenseEmbedding.LiteLLM(config) : DenseEmbedding.Fastembed(config);
};

/**
 * Resolve the chunking layer from env vars:
 *   CHUNKING_METHOD=recursive|semantic (default: recursive)
 */
export const resolveChunkingLayer = (): Layer.Layer<Chunking, any, PythonApiClient> => {
  const method = process.env.CHUNKING_METHOD ?? "recursive";
  return method === "semantic" ? Chunking.Semantic() : Chunking.Recursive();
};

/** Resolve retrieval limit from RETRIEVAL_LIMIT env var (default: 5) */
export const resolveRetrievalLimit = (): number =>
  Number.parseInt(process.env.RETRIEVAL_LIMIT ?? "5", 10);
