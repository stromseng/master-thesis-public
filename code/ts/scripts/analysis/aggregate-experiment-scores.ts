// Usage:
//   bun scripts/analysis/aggregate-experiment-scores.ts
//   bun scripts/analysis/aggregate-experiment-scores.ts --dataset crewcn
//   bun scripts/analysis/aggregate-experiment-scores.ts --dataset crewcn --dataset pei2024-uk
//   bun scripts/analysis/aggregate-experiment-scores.ts --model openai/gpt-4.1
//   bun scripts/analysis/aggregate-experiment-scores.ts --model openai/gpt-4.1 --model moonshotai/Kimi-K2.5
//   bun scripts/analysis/aggregate-experiment-scores.ts --exclude-metadata-keyword RAG
//   bun scripts/analysis/aggregate-experiment-scores.ts --exclude-metadata-keyword RAG --exclude-metadata-keyword bm25
//   bun scripts/analysis/aggregate-experiment-scores.ts --prefer-scored
//   bun scripts/analysis/aggregate-experiment-scores.ts --output aggregate-scores-text-no-rag.json
//   bun scripts/analysis/aggregate-experiment-scores.ts --concurrency 12
//   bun scripts/analysis/aggregate-experiment-scores.ts --annotation question-f1
//   bun scripts/analysis/aggregate-experiment-scores.ts --no-standard-error
//   bun scripts/analysis/aggregate-experiment-scores.ts --base-url http://127.0.0.1:6006
//
// Fetches all Phoenix datasets and experiments, aggregates annotation scores
// per dataset and across all selected datasets grouped by identical experiment
// metadata, and writes a JSON report to:
//   data/analysis/aggregate-scores/
//
// Grouping is based on canonicalized metadata, so runs for the same model with
// different retrieval/chunking/embedding settings stay separate.
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { Effect, Logger, Schema } from "effect";
import * as Progress from "effective-progress";
import { CREWCN_DATASET_NAME } from "../../evals/crew/crew";
import { PEI2024_UK_DATASET_NAME, PEI2024_ZH_DATASET_NAME } from "../../evals/pei2024/pei2024";
import { RAYNOR_DATASET_NAME_V2 } from "../../evals/raynor/raynor";
import {
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
} from "../../evals/shititong/shititong";
import { US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2 } from "../../evals/us_coast_guard/us_coast_guard";
import { PhoenixClient, type PhoenixClientImpl } from "../../src/services/PhoenixClient";
import { dataPath } from "../../src/utils/repo";

type CliArgs = {
  annotationName: string;
  concurrency: number;
  datasets: string[];
  models: string[];
  excludedMetadataKeywords: string[];
  preferScored: boolean;
  computeStandardError: boolean;
  outputFileName?: string;
  baseUrl?: string;
};

type DatasetInfo = {
  id: string;
  name: string;
  example_count: number;
};

type ExperimentInfo = {
  id: string;
  dataset_id: string;
  metadata: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
};

type CostBreakdown = {
  tokens?: number | null;
  cost?: number | null;
};

type CostSummary = {
  total?: CostBreakdown | null;
  prompt?: CostBreakdown | null;
  completion?: CostBreakdown | null;
};

type ExperimentAnnotationSummary = {
  annotationName: string;
  minScore: number | null;
  maxScore: number | null;
  meanScore: number | null;
  count: number;
  errorCount: number;
};

type ExperimentAggregate = {
  runCount: number;
  averageRunLatencyMs: number | null;
  costSummary: CostSummary | null;
  annotationSummaries: readonly ExperimentAnnotationSummary[];
};

type ScoreAccumulator = {
  runCount: number;
  scoredRunCount: number;
  sum: number;
  sumSquares: number | null;
  min: number;
  max: number;
};

type UsageAccumulator = {
  latencyRunCount: number;
  totalLatencyMs: number;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  totalCostUsd: number;
  promptCostUsd: number;
  completionCostUsd: number;
};

type GroupAccumulator = ScoreAccumulator &
  UsageAccumulator & {
    modelId: string;
    metadataKey: string;
    metadata: Record<string, unknown>;
    groupLabel: string;
    experiments: ExperimentOutput[];
  };

type DatasetAccumulator = {
  datasetId: string;
  datasetName: string;
  exampleCount: number;
  experimentCount: number;
  selectedExperimentCount: number;
  skippedByModelFilter: number;
  skippedByMetadataKeywordFilter: number;
  skippedSupersededByMetadata: number;
  skippedMissingExperimentExports: number;
  groups: Map<string, GroupAccumulator>;
};

type GlobalGroupAccumulator = ScoreAccumulator &
  UsageAccumulator & {
    modelId: string;
    metadataKey: string;
    metadata: Record<string, unknown>;
    groupLabel: string;
    datasetNames: Set<string>;
    experimentIds: Set<string>;
    experiments: ExperimentOutput[];
  };

type ScoreSummary = {
  runCount: number;
  scoredRunCount: number;
  meanScore: number | null;
  scoreVariance: number | null;
  scoreStandardDeviation: number | null;
  scoreStandardError: number | null;
  minScore: number | null;
  maxScore: number | null;
};

type UsageSummary = {
  latencyRunCount: number;
  totalLatencyMs: number;
  averageLatencyMs: number | null;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  totalCostUsd: number;
  promptCostUsd: number;
  completionCostUsd: number;
  meanTotalTokens: number | null;
  meanPromptTokens: number | null;
  meanCompletionTokens: number | null;
  meanTotalCostUsd: number | null;
};

type ExperimentOutput = ScoreSummary &
  UsageSummary & {
    experimentId: string;
    datasetId: string;
    datasetName: string;
  };

type GroupOutput = ScoreSummary &
  UsageSummary & {
    modelId: string;
    groupLabel: string;
    metadataKey: string;
    metadata: Record<string, unknown>;
    weightedMeanScore: number | null;
    unweightedMeanScore: number | null;
    unweightedScoreStandardError: number | null;
    datasetCount: number;
    datasets: string[];
    experimentCount: number;
    experiments: ExperimentOutput[];
  };

type PreparedExperiment = {
  experimentId: string;
  modelId: string;
  metadataKey: string;
  metadata: Record<string, unknown>;
  groupLabel: string;
  scores: ScoreAccumulator;
  usage: UsageAccumulator;
  output: ExperimentOutput;
};

class WriteOutputError extends Schema.TaggedError<WriteOutputError>()("WriteOutputError", {
  cause: Schema.Defect,
  outputPath: Schema.String,
}) {}

const TEXT_DATASETS = [
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
  SHITITONG_EN_TEXT_DATASET_NAME,
  RAYNOR_DATASET_NAME_V2,
  CREWCN_DATASET_NAME,
  PEI2024_UK_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
  PEI2024_ZH_DATASET_NAME,
];

const parseArgs = (): CliArgs => {
  const argv = process.argv.slice(2);

  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) return undefined;
    return argv[index + 1];
  };
  const has = (name: string): boolean => argv.includes(name);

  const datasets: string[] = [];
  const models: string[] = [];
  const excludedMetadataKeywords: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--dataset" && i + 1 < argv.length) {
      datasets.push(argv[i + 1]!);
    }
    if (argv[i] === "--model" && i + 1 < argv.length) {
      models.push(argv[i + 1]!);
    }
    if (argv[i] === "--exclude-metadata-keyword" && i + 1 < argv.length) {
      excludedMetadataKeywords.push(argv[i + 1]!);
    }
  }

  const concurrency = Number(get("--concurrency") ?? "8");
  if (!Number.isFinite(concurrency) || concurrency < 1) {
    throw new Error("Invalid --concurrency value. Expected a positive number.");
  }

  return {
    annotationName: get("--annotation") ?? "question-f1",
    concurrency,
    datasets,
    models,
    excludedMetadataKeywords,
    preferScored: has("--prefer-scored"),
    computeStandardError: !has("--no-standard-error"),
    outputFileName: get("--output"),
    baseUrl: get("--base-url"),
  };
};

const ARGS = parseArgs();
const INCLUDED_MODEL_IDS = [...ARGS.models];
const INCLUDED_MODEL_ID_SET = new Set(INCLUDED_MODEL_IDS);
const EXCLUDED_METADATA_KEYWORDS = [...ARGS.excludedMetadataKeywords];
const EXCLUDED_METADATA_KEYWORD_SET = new Set(
  EXCLUDED_METADATA_KEYWORDS.map((keyword) => keyword.toLowerCase()),
);

/** Models unconditionally excluded from aggregation. */
const HARD_EXCLUDED_MODEL_IDS = new Set([
  "Qwen/Qwen3.5-35B-A3B",
  "Qwen/Qwen3.5-35B-A3B-FP8",
  "Qwen/Qwen3.5-2B",
  "Qwen/Qwen3-4B-Instruct-2507",
  "Qwen/Qwen3-4B-Thinking-2507",
  "mistralai/Ministral-3-14B-Instruct-2512",
  "mistralai/Ministral-3-14B-Reasoning-2512",
  "qwen3.5-397b-a17b@q8_0",
  "qwen3.5-397b-a17b@q8_k_xl",
  "Qwen/Qwen3.5-122B-A10B",
  "google/gemma-4-E4B-it",
  "MiniMaxAI/MiniMax-M2.7",
]);

/** Model IDs that are merged into a canonical alias for grouping. */
const MODEL_ID_ALIASES: Record<string, string> = {
  "Qwen/Qwen3.5-27B-FP8": "Qwen/Qwen3.5-27B",
};

const normalizeModelId = (modelId: string): string => MODEL_ID_ALIASES[modelId] ?? modelId;

const withNormalizedModelId = (
  canonicalMetadata: Record<string, unknown>,
  normalizedModelId: string,
): Record<string, unknown> => {
  const model = canonicalMetadata.model;
  if (!isRecord(model) || model.id === normalizedModelId) return canonicalMetadata;
  return { ...canonicalMetadata, model: { ...model, id: normalizedModelId } };
};

const DATASET_FETCH_CONCURRENCY = 4;

const makeScoreAccumulator = (): ScoreAccumulator => ({
  runCount: 0,
  scoredRunCount: 0,
  sum: 0,
  sumSquares: 0,
  min: Number.POSITIVE_INFINITY,
  max: Number.NEGATIVE_INFINITY,
});

const makeUsageAccumulator = (): UsageAccumulator => ({
  latencyRunCount: 0,
  totalLatencyMs: 0,
  totalTokens: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalCostUsd: 0,
  promptCostUsd: 0,
  completionCostUsd: 0,
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const GraphqlCostBreakdownSchema = Schema.Struct({
  tokens: Schema.NullOr(Schema.Number),
  cost: Schema.NullOr(Schema.Number),
});

const GraphqlExperimentAnnotationSummarySchema = Schema.Struct({
  annotationName: Schema.String,
  minScore: Schema.NullOr(Schema.Number),
  maxScore: Schema.NullOr(Schema.Number),
  meanScore: Schema.NullOr(Schema.Number),
  count: Schema.Number,
  errorCount: Schema.Number,
});

const GraphqlExperimentAggregateSchema = Schema.Struct({
  node: Schema.NullOr(
    Schema.Struct({
      __typename: Schema.Literal("Experiment"),
      runCount: Schema.Number,
      averageRunLatencyMs: Schema.NullOr(Schema.Number),
      costSummary: Schema.NullOr(
        Schema.Struct({
          total: GraphqlCostBreakdownSchema,
          prompt: GraphqlCostBreakdownSchema,
          completion: GraphqlCostBreakdownSchema,
        }),
      ),
      annotationSummaries: Schema.Array(GraphqlExperimentAnnotationSummarySchema),
    }),
  ),
});

type GraphqlExperimentAggregateResponse = Schema.Schema.Type<
  typeof GraphqlExperimentAggregateSchema
>;

const PageInfoSchema = Schema.Struct({
  hasNextPage: Schema.Boolean,
  endCursor: Schema.NullOr(Schema.String),
});

const GraphqlAnnotationSchema = Schema.Struct({
  name: Schema.String,
  score: Schema.NullOr(Schema.Number),
  error: Schema.NullOr(Schema.String),
});

const GraphqlExperimentRunsSchema = Schema.Struct({
  node: Schema.NullOr(
    Schema.Struct({
      __typename: Schema.Literal("Experiment"),
      runs: Schema.Struct({
        pageInfo: PageInfoSchema,
        edges: Schema.Array(
          Schema.Struct({
            node: Schema.Struct({
              annotations: Schema.Struct({
                edges: Schema.Array(
                  Schema.Struct({
                    node: GraphqlAnnotationSchema,
                  }),
                ),
              }),
            }),
          }),
        ),
      }),
    }),
  ),
});

type GraphqlExperimentRunsResponse = Schema.Schema.Type<typeof GraphqlExperimentRunsSchema>;

const RUN_PAGE_SIZE = 1000;
const ANNOTATION_PAGE_SIZE = 20;

const EXPERIMENT_AGGREGATE_QUERY = `
  query ExperimentAggregateForAggregation($experimentId: ID!) {
    node(id: $experimentId) {
      __typename
      ... on Experiment {
        runCount
        averageRunLatencyMs
        costSummary {
          total {
            tokens
            cost
          }
          prompt {
            tokens
            cost
          }
          completion {
            tokens
            cost
          }
        }
        annotationSummaries {
          annotationName
          minScore
          maxScore
          meanScore
          count
          errorCount
        }
      }
    }
  }
`;

const EXPERIMENT_RUN_SCORES_QUERY = `
  query ExperimentRunScoresForAggregation(
    $experimentId: ID!
    $first: Int!
    $after: String
    $annotationFirst: Int!
  ) {
    node(id: $experimentId) {
      __typename
      ... on Experiment {
        runs(first: $first, after: $after) {
          pageInfo {
            hasNextPage
            endCursor
          }
          edges {
            node {
              annotations(first: $annotationFirst) {
                edges {
                  node {
                    name
                    score
                    error
                  }
                }
              }
            }
          }
        }
      }
    }
  }
`;

const canonicalizeJson = (value: unknown): unknown => {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (typeof value === "bigint") {
    return value.toString();
  }

  if (Array.isArray(value)) {
    return value.map((item) => (item === undefined ? null : canonicalizeJson(item)));
  }

  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, canonicalizeJson(item)]),
    );
  }

  return String(value);
};

const stableStringify = (value: unknown): string => JSON.stringify(canonicalizeJson(value));

const parseTimestampMs = (value: string | undefined): number => {
  if (!value) return Number.NEGATIVE_INFINITY;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
};

const getExperimentRecencyMs = (experiment: ExperimentInfo): number =>
  Math.max(parseTimestampMs(experiment.updated_at), parseTimestampMs(experiment.created_at));

const compareExperimentRecency = (left: ExperimentInfo, right: ExperimentInfo): number => {
  const leftRecency = getExperimentRecencyMs(left);
  const rightRecency = getExperimentRecencyMs(right);
  if (leftRecency !== rightRecency) {
    return leftRecency > rightRecency ? 1 : -1;
  }

  const leftUpdated = parseTimestampMs(left.updated_at);
  const rightUpdated = parseTimestampMs(right.updated_at);
  if (leftUpdated !== rightUpdated) {
    return leftUpdated > rightUpdated ? 1 : -1;
  }

  const leftCreated = parseTimestampMs(left.created_at);
  const rightCreated = parseTimestampMs(right.created_at);
  if (leftCreated !== rightCreated) {
    return leftCreated > rightCreated ? 1 : -1;
  }

  return left.id.localeCompare(right.id);
};

const hasScoredAnnotation = (aggregate: ExperimentAggregate, annotationName: string): boolean => {
  const summary = aggregate.annotationSummaries.find(
    (item) => item.annotationName === annotationName,
  );
  return typeof summary?.meanScore === "number" && summary.count > 0;
};

const extractModelId = (metadata: Record<string, unknown>): string => {
  const model = metadata.model;
  if (!isRecord(model)) return "unknown";
  return typeof model.id === "string" ? model.id : "unknown";
};

const extractMetadataModels = (metadata: Record<string, unknown>): string[] => {
  const embeddings = metadata.embeddings;
  if (!isRecord(embeddings)) return [];

  const names: string[] = [];
  for (const key of ["dense", "sparse", "late"] as const) {
    const config = embeddings[key];
    if (!isRecord(config)) continue;
    if (typeof config.model === "string") {
      names.push(config.model);
    }
  }
  return names;
};

const describeMetadataGroup = (metadata: Record<string, unknown>): string => {
  const modelId = extractModelId(metadata);
  const retrieval = metadata.retrieval;
  if (!isRecord(retrieval)) return modelId;

  const parts: string[] = ["RAG"];

  if (typeof retrieval.limit === "number") {
    parts.push(`k=${retrieval.limit}`);
  }
  if (typeof retrieval.prefetchLimit === "number") {
    parts.push(`prefetch=${retrieval.prefetchLimit}`);
  }
  if (typeof retrieval.collection === "string" && retrieval.collection.length > 0) {
    parts.push(`collection=${retrieval.collection}`);
  }

  const embeddingModels = extractMetadataModels(metadata);
  if (embeddingModels.length > 0) {
    parts.push(embeddingModels.join("+"));
  }

  const chunking = metadata.chunking;
  if (isRecord(chunking) && typeof chunking.method === "string") {
    parts.push(`chunking=${chunking.method}`);
  }

  return `${modelId} (${parts.join(", ")})`;
};

const shouldExcludeMetadata = (metadata: Record<string, unknown>): boolean => {
  if (EXCLUDED_METADATA_KEYWORD_SET.size === 0) return false;

  const haystacks = [stableStringify(metadata), describeMetadataGroup(metadata)].map((text) =>
    text.toLowerCase(),
  );

  return [...EXCLUDED_METADATA_KEYWORD_SET].some((keyword) =>
    haystacks.some((haystack) => haystack.includes(keyword)),
  );
};

const recordScoreSummary = (
  accumulator: ScoreAccumulator,
  runCount: number,
  summary: ExperimentAnnotationSummary | undefined,
): void => {
  accumulator.runCount += runCount;

  if (!summary || typeof summary.meanScore !== "number" || summary.count <= 0) {
    return;
  }

  accumulator.scoredRunCount += summary.count;
  accumulator.sum += summary.meanScore * summary.count;
  // GraphQL annotation summaries do not expose variance; this fallback can
  // only preserve exact uncertainty when every recorded score is identical.
  if (
    typeof summary.minScore === "number" &&
    typeof summary.maxScore === "number" &&
    summary.minScore === summary.maxScore
  ) {
    accumulator.sumSquares =
      (accumulator.sumSquares ?? 0) + summary.minScore * summary.minScore * summary.count;
  } else {
    accumulator.sumSquares = null;
  }
  if (typeof summary.minScore === "number") {
    accumulator.min = Math.min(accumulator.min, summary.minScore);
  }
  if (typeof summary.maxScore === "number") {
    accumulator.max = Math.max(accumulator.max, summary.maxScore);
  }
};

const recordScore = (accumulator: ScoreAccumulator, score: number): void => {
  accumulator.scoredRunCount += 1;
  accumulator.sum += score;
  accumulator.sumSquares = (accumulator.sumSquares ?? 0) + score * score;
  accumulator.min = Math.min(accumulator.min, score);
  accumulator.max = Math.max(accumulator.max, score);
};

const recordUsageSummary = (
  accumulator: UsageAccumulator,
  summary: CostSummary | null | undefined,
): void => {
  if (!summary) return;

  for (const [key, targetTokenKey, targetCostKey] of [
    ["total", "totalTokens", "totalCostUsd"],
    ["prompt", "promptTokens", "promptCostUsd"],
    ["completion", "completionTokens", "completionCostUsd"],
  ] as const) {
    const breakdown = summary[key];
    if (!breakdown) continue;

    if (typeof breakdown.tokens === "number") {
      accumulator[targetTokenKey] += breakdown.tokens;
    }
    if (typeof breakdown.cost === "number") {
      accumulator[targetCostKey] += breakdown.cost;
    }
  }
};

const recordLatencySummary = (
  accumulator: UsageAccumulator,
  averageRunLatencyMs: number | null | undefined,
  runCount: number,
): void => {
  if (typeof averageRunLatencyMs !== "number" || runCount <= 0) return;

  accumulator.latencyRunCount += runCount;
  accumulator.totalLatencyMs += averageRunLatencyMs * runCount;
};

const toScoreSummary = (
  accumulator: ScoreAccumulator,
  options?: { computeStandardError?: boolean },
): ScoreSummary => {
  const n = accumulator.scoredRunCount;
  const mean = n > 0 ? accumulator.sum / n : null;
  const computeStandardError = options?.computeStandardError ?? true;
  const variance =
    computeStandardError && n > 1 && accumulator.sumSquares !== null
      ? Math.max(0, (accumulator.sumSquares - (accumulator.sum * accumulator.sum) / n) / (n - 1))
      : computeStandardError && n === 1
        ? 0
        : null;
  const standardDeviation = variance === null ? null : Math.sqrt(variance);

  return {
    runCount: accumulator.runCount,
    scoredRunCount: n,
    meanScore: mean,
    scoreVariance: variance,
    scoreStandardDeviation: standardDeviation,
    scoreStandardError:
      standardDeviation === null || n === 0 ? null : standardDeviation / Math.sqrt(n),
    minScore: n > 0 ? accumulator.min : null,
    maxScore: n > 0 ? accumulator.max : null,
  };
};

const toUsageSummary = (accumulator: UsageAccumulator, runCount: number): UsageSummary => ({
  latencyRunCount: accumulator.latencyRunCount,
  totalLatencyMs: accumulator.totalLatencyMs,
  averageLatencyMs:
    accumulator.latencyRunCount > 0
      ? accumulator.totalLatencyMs / accumulator.latencyRunCount
      : null,
  totalTokens: accumulator.totalTokens,
  promptTokens: accumulator.promptTokens,
  completionTokens: accumulator.completionTokens,
  totalCostUsd: accumulator.totalCostUsd,
  promptCostUsd: accumulator.promptCostUsd,
  completionCostUsd: accumulator.completionCostUsd,
  meanTotalTokens: runCount > 0 ? accumulator.totalTokens / runCount : null,
  meanPromptTokens: runCount > 0 ? accumulator.promptTokens / runCount : null,
  meanCompletionTokens: runCount > 0 ? accumulator.completionTokens / runCount : null,
  meanTotalCostUsd: runCount > 0 ? accumulator.totalCostUsd / runCount : null,
});

const computeUnweightedMeanScore = (
  experiments: readonly Pick<ExperimentOutput, "meanScore">[],
): number | null => {
  const scoredExperiments = experiments.filter(
    (experiment): experiment is Pick<ExperimentOutput, "meanScore"> & { meanScore: number } =>
      typeof experiment.meanScore === "number",
  );

  if (scoredExperiments.length === 0) return null;

  return (
    scoredExperiments.reduce((sum, experiment) => sum + experiment.meanScore, 0) /
    scoredExperiments.length
  );
};

const computeUnweightedScoreStandardError = (
  experiments: readonly Pick<ExperimentOutput, "meanScore" | "scoreStandardError">[],
  options?: { computeStandardError?: boolean },
): number | null => {
  if (options?.computeStandardError === false) return null;

  const scoredExperiments = experiments.filter(
    (
      experiment,
    ): experiment is Pick<ExperimentOutput, "meanScore" | "scoreStandardError"> & {
      meanScore: number;
      scoreStandardError: number;
    } =>
      typeof experiment.meanScore === "number" && typeof experiment.scoreStandardError === "number",
  );

  if (scoredExperiments.length === 0) return null;

  const sumVariance = scoredExperiments.reduce(
    (sum, experiment) => sum + experiment.scoreStandardError * experiment.scoreStandardError,
    0,
  );
  return Math.sqrt(sumVariance) / scoredExperiments.length;
};

const mergeScoreAccumulator = (target: ScoreAccumulator, source: ScoreAccumulator): void => {
  target.runCount += source.runCount;
  target.scoredRunCount += source.scoredRunCount;
  target.sum += source.sum;
  target.sumSquares =
    target.sumSquares === null || source.sumSquares === null
      ? null
      : target.sumSquares + source.sumSquares;

  if (source.scoredRunCount > 0) {
    target.min = Math.min(target.min, source.min);
    target.max = Math.max(target.max, source.max);
  }
};

const mergeUsageAccumulator = (target: UsageAccumulator, source: UsageAccumulator): void => {
  target.latencyRunCount += source.latencyRunCount;
  target.totalLatencyMs += source.totalLatencyMs;
  target.totalTokens += source.totalTokens;
  target.promptTokens += source.promptTokens;
  target.completionTokens += source.completionTokens;
  target.totalCostUsd += source.totalCostUsd;
  target.promptCostUsd += source.promptCostUsd;
  target.completionCostUsd += source.completionCostUsd;
};

const listAllDatasets = Effect.fn("listAllDatasets")(function* (phoenix: PhoenixClientImpl) {
  const datasets: DatasetInfo[] = [];
  let cursor: string | undefined;

  while (true) {
    const response = yield* phoenix.use((client) =>
      client.GET("/v1/datasets", {
        params: { query: { limit: 100, ...(cursor ? { cursor } : {}) } },
      }),
    );

    const data = response.data?.data ?? [];
    for (const dataset of data) {
      datasets.push({
        id: dataset.id,
        name: dataset.name,
        example_count: dataset.example_count,
      });
    }

    const nextCursor = response.data?.next_cursor;
    if (!nextCursor || data.length === 0) break;
    cursor = nextCursor;
  }

  return datasets;
});

const listExperiments = Effect.fn("listExperiments")(function* (
  phoenix: PhoenixClientImpl,
  datasetId: string,
) {
  const experiments: ExperimentInfo[] = [];
  let cursor: string | undefined;

  while (true) {
    const response = yield* phoenix.use((client) =>
      client.GET("/v1/datasets/{dataset_id}/experiments", {
        params: {
          path: { dataset_id: datasetId },
          query: { limit: 50, ...(cursor ? { cursor } : {}) },
        },
      }),
    );

    const data = response.data?.data ?? [];
    for (const experiment of data) {
      experiments.push({
        id: experiment.id,
        dataset_id: experiment.dataset_id,
        metadata: (experiment.metadata ?? {}) as Record<string, unknown>,
        created_at: experiment.created_at,
        updated_at: experiment.updated_at,
      });
    }

    const nextCursor = response.data?.next_cursor;
    if (!nextCursor || data.length === 0) break;
    cursor = nextCursor;
  }

  return experiments;
});

const getExperimentAggregateGraphql = Effect.fn("getExperimentAggregateGraphql")(function* (
  phoenix: PhoenixClientImpl,
  experimentId: string,
) {
  const response: GraphqlExperimentAggregateResponse = yield* phoenix.graphql({
    operationName: "ExperimentAggregateForAggregation",
    query: EXPERIMENT_AGGREGATE_QUERY,
    variables: {
      experimentId,
    },
    schema: GraphqlExperimentAggregateSchema,
  });

  if (response.node === null) {
    yield* Effect.logWarning(`Experiment ${experimentId}: missing GraphQL node, skipping`);
    return null;
  }

  return {
    runCount: response.node.runCount,
    averageRunLatencyMs: response.node.averageRunLatencyMs,
    costSummary: response.node.costSummary,
    annotationSummaries: response.node.annotationSummaries,
  } as const;
});

const getExperimentScoreAccumulatorGraphql = Effect.fn("getExperimentScoreAccumulatorGraphql")(
  function* (phoenix: PhoenixClientImpl, experimentId: string, annotationName: string) {
    const scores = makeScoreAccumulator();
    let after: string | undefined;

    while (true) {
      const response: GraphqlExperimentRunsResponse = yield* phoenix.graphql({
        operationName: "ExperimentRunScoresForAggregation",
        query: EXPERIMENT_RUN_SCORES_QUERY,
        variables: {
          experimentId,
          first: RUN_PAGE_SIZE,
          after,
          annotationFirst: ANNOTATION_PAGE_SIZE,
        },
        schema: GraphqlExperimentRunsSchema,
      });

      if (response.node === null) {
        yield* Effect.logWarning(`Experiment ${experimentId}: missing GraphQL node, skipping`);
        break;
      }

      for (const edge of response.node.runs.edges) {
        scores.runCount += 1;
        const annotation = edge.node.annotations.edges
          .map((annotationEdge) => annotationEdge.node)
          .find((candidate) => candidate.name === annotationName);
        if (annotation && !annotation.error && typeof annotation.score === "number") {
          recordScore(scores, annotation.score);
        }
      }

      if (
        !response.node.runs.pageInfo.hasNextPage ||
        response.node.runs.pageInfo.endCursor === null
      ) {
        break;
      }
      after = response.node.runs.pageInfo.endCursor;
    }

    return scores;
  },
);

const writeOutput = Effect.fn("writeOutput")(function* (outputPath: string, output: unknown) {
  mkdirSync(dirname(outputPath), { recursive: true });
  yield* Effect.tryPromise({
    try: () => Bun.write(outputPath, `${JSON.stringify(output, null, 2)}\n`),
    catch: (cause) => new WriteOutputError({ cause, outputPath }),
  });
});

const collectDatasetAggregate = Effect.fn("collectDatasetAggregate")(function* (
  phoenix: PhoenixClientImpl,
  dataset: DatasetInfo,
  annotationName: string,
  concurrency: number,
) {
  const experiments = yield* listExperiments(phoenix, dataset.id);
  yield* Effect.log(`[${dataset.name}] ${experiments.length} experiments`);

  const datasetAccumulator: DatasetAccumulator = {
    datasetId: dataset.id,
    datasetName: dataset.name,
    exampleCount: dataset.example_count,
    experimentCount: experiments.length,
    selectedExperimentCount: 0,
    skippedByModelFilter: 0,
    skippedByMetadataKeywordFilter: 0,
    skippedSupersededByMetadata: 0,
    skippedMissingExperimentExports: 0,
    groups: new Map(),
  };

  const filteredExperiments = experiments
    .map((experiment) => {
      const modelId = normalizeModelId(extractModelId(experiment.metadata));
      const canonicalMetadata = withNormalizedModelId(
        canonicalizeJson(experiment.metadata) as Record<string, unknown>,
        modelId,
      );

      return {
        experiment,
        modelId,
        metadataKey: stableStringify(canonicalMetadata),
        metadata: canonicalMetadata,
        groupLabel: describeMetadataGroup(canonicalMetadata),
      };
    })
    .filter((prepared) => {
      if (HARD_EXCLUDED_MODEL_IDS.has(prepared.modelId)) {
        datasetAccumulator.skippedByModelFilter += 1;
        return false;
      }

      if (INCLUDED_MODEL_ID_SET.size > 0 && !INCLUDED_MODEL_ID_SET.has(prepared.modelId)) {
        datasetAccumulator.skippedByModelFilter += 1;
        return false;
      }

      if (shouldExcludeMetadata(prepared.metadata)) {
        datasetAccumulator.skippedByMetadataKeywordFilter += 1;
        return false;
      }

      return true;
    });

  const selectedExperiments = ARGS.preferScored
    ? yield* Effect.gen(function* () {
        const enrichedExperiments = yield* Progress.forEach(
          filteredExperiments,
          (prepared) =>
            Effect.gen(function* () {
              const aggregate = yield* getExperimentAggregateGraphql(
                phoenix,
                prepared.experiment.id,
              );
              if (aggregate === null) {
                datasetAccumulator.skippedMissingExperimentExports += 1;
                return null;
              }
              return {
                ...prepared,
                hasScoredAnnotation: hasScoredAnnotation(aggregate, annotationName),
              };
            }),
          { description: `${dataset.name}: selecting scored experiments`, concurrency },
        );

        const latestExperimentsByMetadata = new Map<
          string,
          NonNullable<(typeof enrichedExperiments)[number]>
        >();
        for (const prepared of enrichedExperiments) {
          if (prepared === null) continue;

          const existing = latestExperimentsByMetadata.get(prepared.metadataKey);
          if (!existing) {
            latestExperimentsByMetadata.set(prepared.metadataKey, prepared);
            continue;
          }

          if (prepared.hasScoredAnnotation !== existing.hasScoredAnnotation) {
            if (prepared.hasScoredAnnotation) {
              latestExperimentsByMetadata.set(prepared.metadataKey, prepared);
            }
            continue;
          }

          if (compareExperimentRecency(prepared.experiment, existing.experiment) > 0) {
            latestExperimentsByMetadata.set(prepared.metadataKey, prepared);
          }
        }

        return [...latestExperimentsByMetadata.values()];
      })
    : (() => {
        const latestExperimentsByMetadata = new Map<string, (typeof filteredExperiments)[number]>();
        for (const prepared of filteredExperiments) {
          const existing = latestExperimentsByMetadata.get(prepared.metadataKey);
          if (!existing) {
            latestExperimentsByMetadata.set(prepared.metadataKey, prepared);
            continue;
          }

          if (compareExperimentRecency(prepared.experiment, existing.experiment) > 0) {
            latestExperimentsByMetadata.set(prepared.metadataKey, prepared);
          }
        }

        return [...latestExperimentsByMetadata.values()];
      })();
  datasetAccumulator.selectedExperimentCount = selectedExperiments.length;
  datasetAccumulator.skippedSupersededByMetadata =
    filteredExperiments.length - selectedExperiments.length;

  const preparedExperiments = yield* Progress.forEach(
    selectedExperiments,
    (prepared) =>
      Effect.gen(function* () {
        const aggregate = yield* getExperimentAggregateGraphql(phoenix, prepared.experiment.id);

        if (aggregate === null) {
          datasetAccumulator.skippedMissingExperimentExports += 1;
          return null;
        }

        const usage = makeUsageAccumulator();
        const annotationSummary = aggregate.annotationSummaries.find(
          (summary) => summary.annotationName === annotationName,
        );
        if (!annotationSummary) {
          yield* Effect.logWarning(
            `Experiment ${prepared.experiment.id}: missing annotation summary "${annotationName}"`,
          );
        }

        const scores = makeScoreAccumulator();
        if (ARGS.computeStandardError) {
          const runLevelScores = yield* getExperimentScoreAccumulatorGraphql(
            phoenix,
            prepared.experiment.id,
            annotationName,
          );
          mergeScoreAccumulator(scores, runLevelScores);
          if (scores.runCount === 0) {
            scores.runCount = aggregate.runCount;
          }
          if (scores.scoredRunCount === 0 && annotationSummary) {
            recordScoreSummary(scores, 0, annotationSummary);
          }
        } else {
          recordScoreSummary(scores, aggregate.runCount, annotationSummary);
        }
        recordLatencySummary(usage, aggregate.averageRunLatencyMs, aggregate.runCount);
        recordUsageSummary(usage, aggregate.costSummary);
        const scoreSummary = toScoreSummary(scores, {
          computeStandardError: ARGS.computeStandardError,
        });

        return {
          experimentId: prepared.experiment.id,
          modelId: prepared.modelId,
          metadataKey: prepared.metadataKey,
          metadata: prepared.metadata,
          groupLabel: prepared.groupLabel,
          scores,
          usage,
          output: {
            experimentId: prepared.experiment.id,
            datasetId: dataset.id,
            datasetName: dataset.name,
            ...scoreSummary,
            ...toUsageSummary(usage, scores.runCount),
          },
        } satisfies PreparedExperiment;
      }),
    {
      description: ARGS.computeStandardError
        ? `${dataset.name}: fetching run-level scores`
        : `${dataset.name}: fetching experiment summaries`,
      concurrency,
    },
  );

  for (const prepared of preparedExperiments) {
    if (prepared === null) continue;

    const existingGroup = datasetAccumulator.groups.get(prepared.metadataKey);
    const group =
      existingGroup ??
      (() => {
        const nextGroup: GroupAccumulator = {
          ...makeScoreAccumulator(),
          ...makeUsageAccumulator(),
          modelId: prepared.modelId,
          metadataKey: prepared.metadataKey,
          metadata: prepared.metadata,
          groupLabel: prepared.groupLabel,
          experiments: [],
        };
        datasetAccumulator.groups.set(prepared.metadataKey, nextGroup);
        return nextGroup;
      })();

    mergeScoreAccumulator(group, prepared.scores);
    mergeUsageAccumulator(group, prepared.usage);
    group.experiments.push(prepared.output);
  }

  return datasetAccumulator;
});

const buildOutput = (
  outputPath: string,
  experiments: readonly ExperimentOutput[],
  globalGroups: readonly GroupOutput[],
  annotationName: string,
  selectedDatasets: readonly string[],
  selectedModels: readonly string[],
  excludedMetadataKeywords: readonly string[],
  processedDatasets: readonly {
    id: string;
    name: string;
    exampleCount: number;
    experimentCount: number;
    selectedExperimentCount: number;
  }[],
  skippedByModelFilter: number,
  skippedByMetadataKeywordFilter: number,
  skippedSupersededByMetadata: number,
  skippedMissingExperimentExports: number,
) => {
  const summary = globalGroups.reduce(
    (accumulator, group) => {
      accumulator.globalGroupCount += 1;
      accumulator.experimentCount += group.experimentCount;
      accumulator.runCount += group.runCount;
      accumulator.scoredRunCount += group.scoredRunCount;
      accumulator.latencyRunCount += group.latencyRunCount;
      accumulator.totalLatencyMs += group.totalLatencyMs;
      accumulator.totalTokens += group.totalTokens;
      accumulator.promptTokens += group.promptTokens;
      accumulator.completionTokens += group.completionTokens;
      accumulator.totalCostUsd += group.totalCostUsd;
      accumulator.promptCostUsd += group.promptCostUsd;
      accumulator.completionCostUsd += group.completionCostUsd;
      return accumulator;
    },
    {
      datasetCount: processedDatasets.length,
      experimentCount: 0,
      globalGroupCount: 0,
      runCount: 0,
      scoredRunCount: 0,
      latencyRunCount: 0,
      totalLatencyMs: 0,
      totalTokens: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalCostUsd: 0,
      promptCostUsd: 0,
      completionCostUsd: 0,
      skippedByModelFilter,
      skippedByMetadataKeywordFilter,
      skippedSupersededByMetadata,
      skippedMissingExperimentExports,
    },
  );

  return {
    generatedAt: new Date().toISOString(),
    outputPath,
    annotationName,
    datasetFilter: selectedDatasets,
    modelFilter: selectedModels,
    excludedMetadataKeywords,
    processedDatasets,
    summary: {
      ...summary,
      averageLatencyMs:
        summary.latencyRunCount > 0 ? summary.totalLatencyMs / summary.latencyRunCount : null,
      meanTotalTokens: summary.runCount > 0 ? summary.totalTokens / summary.runCount : null,
      meanPromptTokens: summary.runCount > 0 ? summary.promptTokens / summary.runCount : null,
      meanCompletionTokens:
        summary.runCount > 0 ? summary.completionTokens / summary.runCount : null,
      meanTotalCostUsd: summary.runCount > 0 ? summary.totalCostUsd / summary.runCount : null,
    },
    experiments,
    globalGroups,
  };
};

const program = Effect.gen(function* () {
  const phoenix = yield* PhoenixClient;

  yield* Effect.log(`Aggregating "${ARGS.annotationName}" scores across Phoenix datasets`);
  yield* Effect.log(`Experiment fetch concurrency: ${ARGS.concurrency}`);
  yield* Effect.log(`Dataset fetch concurrency: ${DATASET_FETCH_CONCURRENCY}`);
  yield* Effect.log(
    `Standard error calculation: ${ARGS.computeStandardError ? "enabled" : "disabled"}`,
  );
  const datasetFilter = ARGS.datasets.length > 0 ? ARGS.datasets : TEXT_DATASETS;
  yield* Effect.log(
    ARGS.datasets.length > 0
      ? `Dataset filter enabled (${datasetFilter.length} datasets)`
      : `Using default text datasets (${datasetFilter.length} datasets)`,
  );
  for (const datasetName of datasetFilter) {
    yield* Effect.log(`  - ${datasetName}`);
  }

  if (INCLUDED_MODEL_ID_SET.size > 0) {
    yield* Effect.log(`Model filter enabled (${INCLUDED_MODEL_ID_SET.size} models)`);
    for (const modelId of INCLUDED_MODEL_IDS) {
      yield* Effect.log(`  - ${modelId}`);
    }
  }
  if (EXCLUDED_METADATA_KEYWORD_SET.size > 0) {
    yield* Effect.log(
      `Metadata keyword exclusion enabled (${EXCLUDED_METADATA_KEYWORD_SET.size} keywords)`,
    );
    for (const keyword of EXCLUDED_METADATA_KEYWORDS) {
      yield* Effect.log(`  - ${keyword}`);
    }
  }

  const allDatasets = yield* listAllDatasets(phoenix);
  const selectedDatasets = allDatasets.filter((dataset) => datasetFilter.includes(dataset.name));

  const missingDatasets = datasetFilter.filter(
    (name) => !selectedDatasets.some((dataset) => dataset.name === name),
  );
  if (missingDatasets.length > 0) {
    yield* Effect.logWarning(`Missing datasets: ${missingDatasets.join(", ")}`);
  }

  yield* Effect.log(
    `Processing ${selectedDatasets.length}/${datasetFilter.length} requested datasets`,
  );

  const globalGroups = new Map<string, GlobalGroupAccumulator>();
  const experimentOutputs: ExperimentOutput[] = [];
  const processedDatasets: {
    id: string;
    name: string;
    exampleCount: number;
    experimentCount: number;
    selectedExperimentCount: number;
  }[] = [];
  let skippedByModelFilter = 0;
  let skippedByMetadataKeywordFilter = 0;
  let skippedSupersededByMetadata = 0;
  let skippedMissingExperimentExports = 0;

  const datasetAggregates = yield* Progress.forEach(
    [...selectedDatasets].sort((a, b) => a.name.localeCompare(b.name)),
    (dataset) =>
      collectDatasetAggregate(phoenix, dataset, ARGS.annotationName, ARGS.concurrency).pipe(
        Effect.withSpan("aggregateDataset", {
          attributes: {
            datasetId: dataset.id,
            datasetName: dataset.name,
          },
        }),
      ),
    { description: "Aggregating datasets", concurrency: DATASET_FETCH_CONCURRENCY },
  );

  for (const aggregate of datasetAggregates) {
    processedDatasets.push({
      id: aggregate.datasetId,
      name: aggregate.datasetName,
      exampleCount: aggregate.exampleCount,
      experimentCount: aggregate.experimentCount,
      selectedExperimentCount: aggregate.selectedExperimentCount,
    });
    skippedByModelFilter += aggregate.skippedByModelFilter;
    skippedByMetadataKeywordFilter += aggregate.skippedByMetadataKeywordFilter;
    skippedSupersededByMetadata += aggregate.skippedSupersededByMetadata;
    skippedMissingExperimentExports += aggregate.skippedMissingExperimentExports;

    for (const group of aggregate.groups.values()) {
      const existingGroup = globalGroups.get(group.metadataKey);
      const globalGroup =
        existingGroup ??
        (() => {
          const nextGroup: GlobalGroupAccumulator = {
            ...makeScoreAccumulator(),
            ...makeUsageAccumulator(),
            modelId: group.modelId,
            metadataKey: group.metadataKey,
            metadata: group.metadata,
            groupLabel: group.groupLabel,
            datasetNames: new Set(),
            experimentIds: new Set(),
            experiments: [],
          };
          globalGroups.set(group.metadataKey, nextGroup);
          return nextGroup;
        })();

      mergeScoreAccumulator(globalGroup, group);
      mergeUsageAccumulator(globalGroup, group);
      globalGroup.datasetNames.add(aggregate.datasetName);
      for (const experiment of group.experiments) {
        globalGroup.experimentIds.add(experiment.experimentId);
        globalGroup.experiments.push(experiment);
        experimentOutputs.push(experiment);
      }
    }

    yield* Effect.log(
      `[${aggregate.datasetName}] selected ${aggregate.selectedExperimentCount} latest experiments from ${aggregate.experimentCount - aggregate.skippedByModelFilter - aggregate.skippedByMetadataKeywordFilter} filtered experiments into ${aggregate.groups.size} metadata groups`,
    );
  }

  const groups = [...globalGroups.values()]
    .map((group) => {
      const experiments = [...group.experiments].sort((a, b) => {
        const datasetOrder = a.datasetName.localeCompare(b.datasetName);
        if (datasetOrder !== 0) return datasetOrder;
        return a.experimentId.localeCompare(b.experimentId);
      });

      const scoreSummary = toScoreSummary(group, {
        computeStandardError: ARGS.computeStandardError,
      });

      return {
        modelId: group.modelId,
        groupLabel: group.groupLabel,
        metadataKey: group.metadataKey,
        metadata: group.metadata,
        weightedMeanScore: scoreSummary.meanScore,
        unweightedMeanScore: computeUnweightedMeanScore(experiments),
        unweightedScoreStandardError: computeUnweightedScoreStandardError(experiments, {
          computeStandardError: ARGS.computeStandardError,
        }),
        datasetCount: group.datasetNames.size,
        datasets: [...group.datasetNames].sort((a, b) => a.localeCompare(b)),
        experimentCount: group.experimentIds.size,
        experiments,
        ...scoreSummary,
        ...toUsageSummary(group, group.runCount),
      };
    })
    .sort((a, b) => {
      const modelOrder = a.modelId.localeCompare(b.modelId);
      if (modelOrder !== 0) return modelOrder;

      const labelOrder = a.groupLabel.localeCompare(b.groupLabel);
      if (labelOrder !== 0) return labelOrder;

      return a.metadataKey.localeCompare(b.metadataKey);
    });

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputFileName = ARGS.outputFileName ?? `aggregate-scores-${timestamp}.json`;
  const outputPath = dataPath("analysis", "aggregate-scores", outputFileName);
  const output = buildOutput(
    outputPath,
    [...experimentOutputs].sort((a, b) => {
      const datasetOrder = a.datasetName.localeCompare(b.datasetName);
      if (datasetOrder !== 0) return datasetOrder;
      return a.experimentId.localeCompare(b.experimentId);
    }),
    groups,
    ARGS.annotationName,
    [...datasetFilter].sort((a, b) => a.localeCompare(b)),
    [...INCLUDED_MODEL_IDS].sort((a, b) => a.localeCompare(b)),
    [...EXCLUDED_METADATA_KEYWORDS].sort((a, b) => a.localeCompare(b)),
    processedDatasets.sort((a, b) => a.name.localeCompare(b.name)),
    skippedByModelFilter,
    skippedByMetadataKeywordFilter,
    skippedSupersededByMetadata,
    skippedMissingExperimentExports,
  );

  yield* writeOutput(outputPath, output);
  yield* Effect.log(`Wrote aggregate scores to ${outputPath}`);
});

const layer = ARGS.baseUrl
  ? PhoenixClient.layer({ options: { baseUrl: ARGS.baseUrl } })
  : PhoenixClient.skyhigh;

Effect.runPromise(program.pipe(Effect.provide(layer), Effect.provide(Logger.pretty))).catch(
  console.error,
);
