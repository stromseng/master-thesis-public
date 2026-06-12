// Usage:
//   bun scripts/analyze-failure-modes.ts
//   bun scripts/analyze-failure-modes.ts --multimodal
//   bun scripts/analyze-failure-modes.ts --limit 10
//   bun scripts/analyze-failure-modes.ts --concurrency 3
//   bun scripts/analyze-failure-modes.ts --dataset pei2024-uk
//   bun scripts/analyze-failure-modes.ts --dataset pei2024-uk --dataset crewcn
//   bun scripts/analyze-failure-modes.ts --model moonshotai/Kimi-K2.5
//   bun scripts/analyze-failure-modes.ts --all-models
//   bun scripts/analyze-failure-modes.ts --all-models --dry-run
//   bun scripts/analyze-failure-modes.ts --include-rag
//   bun scripts/analyze-failure-modes.ts --output data/analysis/failure-mode-analysis-run.json
//   bun scripts/analyze-failure-modes.ts --resume data/analysis/failure-mode-analysis-run.json
//   bun scripts/analyze-failure-modes.ts --exclude-metadata-keyword RAG
//   bun scripts/analyze-failure-modes.ts --exclude-metadata-keyword RAG --exclude-metadata-keyword bm25
//
// Environment variables (same as eval scripts):
//   EVAL_PROVIDER  — "litellm" (default) or "vllm"
//   EVAL_MODEL     — explicit model ID (auto-detected for vllm if omitted)
//   VLLM_BASE_URL / VLLM_PORT — vLLM endpoint config
//
// Fetches all experiment runs where F1=0 from Phoenix, asks an LLM to
// categorize why each answer was wrong, and produces a JSON report with
// a final summary.
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { jsonSchema } from "ai";
import { encode } from "gpt-tokenizer";
import { Cause, Effect, Layer, Logger, Schema } from "effect";
import * as JSONSchema from "effect/JSONSchema";
import * as S from "effect/Schema";
import * as Progress from "effective-progress";
import { CREWCN_DATASET_NAME } from "../../evals/crew/crew";
import { taskRetryPolicy } from "../../evals/experiment_setup";
import {
  navreasDatasets,
  toDatasetName as toNavreasDatasetName,
} from "../../evals/navreas/navreas";
import {
  RAYNOR_DATASET_NAME_MULTIMODAL_V2,
  RAYNOR_DATASET_NAME_V2,
} from "../../evals/raynor/raynor";
import { PEI2024_UK_DATASET_NAME, PEI2024_ZH_DATASET_NAME } from "../../evals/pei2024/pei2024";
import {
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
  SHITITONG_ZH_VISION_DATASET_NAME,
} from "../../evals/shititong/shititong";
import {
  US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
} from "../../evals/us_coast_guard/us_coast_guard";
import { PhoenixClient, type PhoenixClientImpl } from "../../src/services/PhoenixClient";
import {
  LanguageModel,
  LanguageModelError,
  EvalLanguageModelLayer,
  generateObject,
  generateText,
} from "../../src/services/LanguageModel";
import { dataPath } from "../../src/utils/repo";

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

const parseArgs = () => {
  const argv = process.argv.slice(2);
  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) return undefined;
    return argv[index + 1];
  };

  const datasets: string[] = [];
  const models: string[] = [];
  const excludeMetadataKeywords: string[] = [];
  const allModels = argv.includes("--all-models");
  const dryRun = argv.includes("--dry-run");
  const multimodal = argv.includes("--multimodal");
  const includeRag = argv.includes("--include-rag");
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--dataset" && i + 1 < argv.length) {
      datasets.push(argv[i + 1]!);
    }
    if (argv[i] === "--model" && i + 1 < argv.length) {
      models.push(argv[i + 1]!);
    }
    if (argv[i] === "--exclude-metadata-keyword" && i + 1 < argv.length) {
      excludeMetadataKeywords.push(argv[i + 1]!);
    }
  }

  return {
    limit: Number(get("--limit") ?? "0"), // 0 = unlimited
    concurrency: Number(get("--concurrency") ?? "20"),
    datasets,
    models,
    allModels,
    dryRun,
    multimodal,
    includeRag,
    output: get("--output"),
    resume: get("--resume"),
    excludeMetadataKeywords,
  };
};

const ARGS = parseArgs();

// ---------------------------------------------------------------------------
// Source datasets
// ---------------------------------------------------------------------------

const TEXT_DATASETS = [
  PEI2024_UK_DATASET_NAME,
  PEI2024_ZH_DATASET_NAME,
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
  RAYNOR_DATASET_NAME_V2,
  CREWCN_DATASET_NAME,
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
];

const MULTIMODAL_DATASETS = [
  US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
  RAYNOR_DATASET_NAME_MULTIMODAL_V2,
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_VISION_DATASET_NAME,
  ...navreasDatasets.map(toNavreasDatasetName),
];

const ANALYSIS_MODE = ARGS.multimodal ? "multimodal" : "text";
type AnalysisMode = typeof ANALYSIS_MODE;
const DEFAULT_SOURCE_DATASETS = ARGS.multimodal ? MULTIMODAL_DATASETS : TEXT_DATASETS;

const DEFAULT_MODEL_IDS = [
  "moonshotai/Kimi-K2.5",
  "Qwen/Qwen3.5-397B-A17B-FP8",
  "Qwen/Qwen3.5-122B-A10B-FP8",
  "openai/gpt-oss-120b",
  "pentagoniac/llamarine",
  "meta-llama/Llama-3.1-70B",
  "Qwen/Qwen3.5-27B",
  "Qwen/Qwen3.5-9B",
  "Qwen/Qwen3.5-4B",
] as const;

const INCLUDED_MODEL_IDS = [...(ARGS.allModels ? DEFAULT_MODEL_IDS : []), ...ARGS.models].filter(
  (modelId, index, ids) => ids.indexOf(modelId) === index,
);
const EXCLUDED_METADATA_KEYWORD_SET = new Set(
  ARGS.excludeMetadataKeywords.map((k) => k.toLowerCase()),
);

const shouldExcludeMetadata = (metadata: Record<string, unknown>): boolean => {
  if (EXCLUDED_METADATA_KEYWORD_SET.size === 0) return false;
  const haystack = JSON.stringify(metadata).toLowerCase();
  return [...EXCLUDED_METADATA_KEYWORD_SET].some((keyword) => haystack.includes(keyword));
};

const isRagExperimentMetadata = (metadata: Record<string, unknown>): boolean =>
  isRecord(metadata.retrieval);

class WriteOutputError extends Schema.TaggedError<WriteOutputError>()("WriteOutputError", {
  cause: Schema.Defect,
}) {}

class ReadCheckpointError extends Schema.TaggedError<ReadCheckpointError>()("ReadCheckpointError", {
  cause: Schema.Defect,
  path: Schema.String,
}) {}

class ResumeCheckpointMismatchError extends Schema.TaggedError<ResumeCheckpointMismatchError>()(
  "ResumeCheckpointMismatchError",
  {
    path: Schema.String,
    mismatches: Schema.Array(Schema.String),
  },
) {}

// ---------------------------------------------------------------------------
// Failure mode categories
// ---------------------------------------------------------------------------

const BASE_FAILURE_MODE_DEFINITIONS = [
  {
    id: "dataset_or_ground_truth_issue",
    description:
      "The failure is caused by a problem in the dataset or answer key, such as a wrong reference answer, malformed options, missing context, or other data-quality issues rather than a model mistake",
  },
  {
    id: "knowledge_gap",
    description:
      "Model lacks the domain knowledge needed to answer correctly, including regulatory rules, factual knowledge, thresholds, conventions, equipment requirements, or technical facts",
  },
  {
    id: "interpretation_or_confusion",
    description:
      "Model misreads the question or confuses similar concepts, terms, options, signals, procedures, or contextual cues",
  },
  {
    id: "reasoning_error",
    description:
      "Model has the needed information in principle but fails in calculation, comparison, sequencing, spatial reasoning, or multi-step inference",
  },
  {
    id: "model_output_error",
    description:
      "Model produced off-topic, hallucinated, placeholder, or unparseable output instead of attempting to answer",
  },
  {
    id: "other",
    description:
      "None of the above categories fit — provide a custom category name in customFailureMode",
  },
] as const;

const MULTIMODAL_FAILURE_MODE_DEFINITIONS = [
  {
    id: "image_understanding_error",
    description:
      "Model appears to misunderstand, miss, or misuse the visual information required by the question, based on its chosen answer and explanation",
  },
] as const;

const FAILURE_MODE_DEFINITIONS =
  ANALYSIS_MODE === "multimodal"
    ? [...BASE_FAILURE_MODE_DEFINITIONS, ...MULTIMODAL_FAILURE_MODE_DEFINITIONS]
    : BASE_FAILURE_MODE_DEFINITIONS;

const FAILURE_MODES = FAILURE_MODE_DEFINITIONS.map((d) => d.id);
type FailureMode = (typeof FAILURE_MODE_DEFINITIONS)[number]["id"];
const FAILURE_MODE_DEFINITION_EXPORT = FAILURE_MODE_DEFINITIONS.map((d) => ({
  id: d.id,
  description: d.description,
}));

/** Resolve the effective failure mode label for aggregation. */
const resolveFailureMode = (analysis: FailureAnalysis): string =>
  analysis.failureMode === "other" && analysis.customFailureMode
    ? `other: ${analysis.customFailureMode}`
    : analysis.failureMode;

// ---------------------------------------------------------------------------
// Failure analysis schema
// ---------------------------------------------------------------------------

const FailureAnalysisSchema = S.Struct({
  failureMode: S.Literal(
    ...(FAILURE_MODES as unknown as readonly [FailureMode, ...FailureMode[]]),
  ).annotations({
    description:
      "The failure mode category. Use one of the predefined values when it fits. Use 'other' only when none of the predefined categories apply.",
  }),
  customFailureMode: S.optional(S.String).annotations({
    description:
      "A short custom category name when failureMode is 'other', e.g. 'chart datum confusion'. Omit when using a predefined category.",
  }),
  explanation: S.String.annotations({
    description: "Why the model likely chose the wrong answer",
  }),
  difficultyFactors: S.Array(S.String).annotations({
    description: "What makes this question hard",
  }),
  requiredKnowledge: S.Array(S.String).annotations({
    description: "Specific knowledge or reasoning needed to answer correctly",
  }),
});

export type FailureAnalysis = {
  failureMode: FailureMode;
  customFailureMode?: string;
  explanation: string;
  difficultyFactors: string[];
  requiredKnowledge: string[];
};

const failureAnalysisJsonSchema = jsonSchema<FailureAnalysis>(
  JSONSchema.make(FailureAnalysisSchema),
);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type DatasetInfo = { id: string; name: string; example_count: number };
type ExperimentAnnotationSummary = {
  annotationName: string;
  minScore: number | null;
  maxScore: number | null;
  meanScore: number | null;
  count: number;
  errorCount: number;
};
type ExperimentInfo = {
  id: string;
  dataset_id: string;
  metadata: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
  annotationSummaries: readonly ExperimentAnnotationSummary[];
};

type ExperimentAnnotation = {
  name: string;
  score: number | null;
};

type ExperimentRun = {
  example_id: string;
  output: Record<string, unknown> | null;
  example_input: Record<string, unknown>;
  example_output: Record<string, unknown>;
  error: string | null;
  annotations: ExperimentAnnotation[];
};

export type FailedRun = {
  sourceDataset: string;
  exampleId: string;
  questionId: string;
  questionKey: string;
  modelId: string;
  questionText: string;
  questionImages: RunImage[];
  options: RunOption[];
  modelAnswer: string[];
  modelReason: string;
  correctAnswer: string[];
};

export type AnalysisResult = {
  sourceDataset: string;
  exampleId: string;
  questionId: string;
  questionKey: string;
  modelId: string;
  questionText: string;
  questionImages: RunImage[];
  options: RunOption[];
  modelAnswer: string[];
  modelReason: string;
  correctAnswer: string[];
  analysis: FailureAnalysis | null;
  error?: string;
};

type RunImage = {
  uri: string;
  caption?: string;
};

type RunOption = {
  id: string;
  text: string;
  images: RunImage[];
};

type DatasetModelPairSummary = {
  sourceDataset: string;
  modelId: string;
  experimentId: string;
};

const analysisKey = (item: { sourceDataset: string; exampleId: string; modelId: string }) =>
  `${item.sourceDataset}::${item.exampleId}::${item.modelId}`;

const questionKeyOf = (sourceDataset: string, questionId: string) =>
  `${sourceDataset}::${questionId}`;

const sortDatasetModelPairs = (
  pairs: readonly DatasetModelPairSummary[],
): DatasetModelPairSummary[] =>
  [...pairs].sort(
    (left, right) =>
      left.sourceDataset.localeCompare(right.sourceDataset) ||
      left.modelId.localeCompare(right.modelId),
  );

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

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

const selectLatestExperimentsByModel = (
  experiments: readonly ExperimentInfo[],
  includedModelIds: readonly string[],
) => {
  const includedModelIdSet = new Set(includedModelIds);
  let skippedByModelFilter = 0;
  let skippedByRagFilter = 0;
  let skippedByMetadataFilter = 0;
  const latestExperimentByModel = new Map<string, ExperimentInfo>();

  for (const experiment of experiments) {
    const modelId = (experiment.metadata as { model?: { id?: string } })?.model?.id ?? "unknown";

    if (includedModelIdSet.size > 0 && !includedModelIdSet.has(modelId)) {
      skippedByModelFilter++;
      continue;
    }

    if (!ARGS.includeRag && isRagExperimentMetadata(experiment.metadata)) {
      skippedByRagFilter++;
      continue;
    }

    if (shouldExcludeMetadata(experiment.metadata)) {
      skippedByMetadataFilter++;
      continue;
    }

    const current = latestExperimentByModel.get(modelId);
    if (!current || compareExperimentRecency(experiment, current) > 0) {
      latestExperimentByModel.set(modelId, experiment);
    }
  }

  return {
    skippedByModelFilter,
    skippedByRagFilter,
    skippedByMetadataFilter,
    latestExperimentByModel,
  };
};

// ---------------------------------------------------------------------------
// Phoenix helpers
// ---------------------------------------------------------------------------

const GRAPHQL_PAGE_INFO_SCHEMA = Schema.Struct({
  hasNextPage: Schema.Boolean,
  endCursor: Schema.NullOr(Schema.String),
});

const GRAPHQL_EXPERIMENT_ANNOTATION_SUMMARY_SCHEMA = Schema.Struct({
  annotationName: Schema.String,
  minScore: Schema.NullOr(Schema.Number),
  maxScore: Schema.NullOr(Schema.Number),
  meanScore: Schema.NullOr(Schema.Number),
  count: Schema.Number,
  errorCount: Schema.Number,
});

const GRAPHQL_DATASETS_SCHEMA = Schema.Struct({
  datasets: Schema.Struct({
    pageInfo: GRAPHQL_PAGE_INFO_SCHEMA,
    edges: Schema.Array(
      Schema.Struct({
        node: Schema.Struct({
          id: Schema.String,
          name: Schema.String,
          exampleCount: Schema.Number,
        }),
      }),
    ),
  }),
});

type GraphqlDatasetsResponse = Schema.Schema.Type<typeof GRAPHQL_DATASETS_SCHEMA>;

const GRAPHQL_DATASET_EXPERIMENTS_SCHEMA = Schema.Struct({
  node: Schema.NullOr(
    Schema.Struct({
      __typename: Schema.Literal("Dataset"),
      experiments: Schema.Struct({
        pageInfo: GRAPHQL_PAGE_INFO_SCHEMA,
        edges: Schema.Array(
          Schema.Struct({
            node: Schema.Struct({
              id: Schema.String,
              metadata: Schema.Unknown,
              createdAt: Schema.String,
              updatedAt: Schema.String,
              annotationSummaries: Schema.Array(GRAPHQL_EXPERIMENT_ANNOTATION_SUMMARY_SCHEMA),
            }),
          }),
        ),
      }),
    }),
  ),
});

type GraphqlDatasetExperimentsResponse = Schema.Schema.Type<
  typeof GRAPHQL_DATASET_EXPERIMENTS_SCHEMA
>;

const GRAPHQL_EXPERIMENT_RUNS_SCHEMA = Schema.Struct({
  node: Schema.NullOr(
    Schema.Struct({
      __typename: Schema.Literal("Experiment"),
      runs: Schema.Struct({
        pageInfo: GRAPHQL_PAGE_INFO_SCHEMA,
        edges: Schema.Array(
          Schema.Struct({
            node: Schema.Struct({
              example: Schema.Struct({
                id: Schema.String,
                revision: Schema.Struct({
                  input: Schema.Unknown,
                  output: Schema.Unknown,
                }),
              }),
              output: Schema.NullOr(Schema.Unknown),
              error: Schema.NullOr(Schema.String),
              annotations: Schema.Struct({
                edges: Schema.Array(
                  Schema.Struct({
                    node: Schema.Struct({
                      name: Schema.String,
                      score: Schema.NullOr(Schema.Number),
                    }),
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

type GraphqlExperimentRunsResponse = Schema.Schema.Type<typeof GRAPHQL_EXPERIMENT_RUNS_SCHEMA>;

const GRAPHQL_DATASETS_QUERY = `
  query FailureModeDatasets($first: Int!, $after: String) {
    datasets(first: $first, after: $after) {
      pageInfo {
        hasNextPage
        endCursor
      }
      edges {
        node {
          id
          name
          exampleCount
        }
      }
    }
  }
`;

const GRAPHQL_DATASET_EXPERIMENTS_QUERY = `
  query FailureModeDatasetExperiments($datasetId: ID!, $first: Int!, $after: String) {
    node(id: $datasetId) {
      __typename
      ... on Dataset {
        experiments(first: $first, after: $after) {
          pageInfo {
            hasNextPage
            endCursor
          }
          edges {
            node {
              id
              metadata
              createdAt
              updatedAt
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
      }
    }
  }
`;

const GRAPHQL_EXPERIMENT_RUNS_QUERY = `
  query FailureModeExperimentRuns($experimentId: ID!, $first: Int!, $after: String) {
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
              output
              error
              example {
                id
                revision {
                  input
                  output
                }
              }
              annotations(first: 50) {
                edges {
                  node {
                    name
                    score
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

const GRAPHQL_DATASET_PAGE_SIZE = 100;
const GRAPHQL_EXPERIMENT_PAGE_SIZE = 100;
const GRAPHQL_RUN_PAGE_SIZE = 100;

const listAllDatasets = Effect.fn("listAllDatasets")(function* (phoenix: PhoenixClientImpl) {
  const datasets: DatasetInfo[] = [];
  let after: string | undefined;

  while (true) {
    const response: GraphqlDatasetsResponse = yield* phoenix.graphql({
      operationName: "FailureModeDatasets",
      query: GRAPHQL_DATASETS_QUERY,
      variables: {
        first: GRAPHQL_DATASET_PAGE_SIZE,
        after,
      },
      schema: GRAPHQL_DATASETS_SCHEMA,
    });

    for (const edge of response.datasets.edges) {
      datasets.push({
        id: edge.node.id,
        name: edge.node.name,
        example_count: edge.node.exampleCount,
      });
    }

    if (!response.datasets.pageInfo.hasNextPage || response.datasets.pageInfo.endCursor === null)
      break;
    after = response.datasets.pageInfo.endCursor;
  }

  return datasets;
});

const listExperiments = Effect.fn("listExperiments")(function* (
  phoenix: PhoenixClientImpl,
  datasetId: string,
) {
  const experiments: ExperimentInfo[] = [];
  let after: string | undefined;

  while (true) {
    const response: GraphqlDatasetExperimentsResponse = yield* phoenix.graphql({
      operationName: "FailureModeDatasetExperiments",
      query: GRAPHQL_DATASET_EXPERIMENTS_QUERY,
      variables: {
        datasetId,
        first: GRAPHQL_EXPERIMENT_PAGE_SIZE,
        after,
      },
      schema: GRAPHQL_DATASET_EXPERIMENTS_SCHEMA,
    });

    if (response.node === null) {
      yield* Effect.logWarning(`Dataset ${datasetId}: missing GraphQL node, skipping`);
      return experiments;
    }

    for (const edge of response.node.experiments.edges) {
      const metadata = isRecord(edge.node.metadata) ? edge.node.metadata : {};
      experiments.push({
        id: edge.node.id,
        dataset_id: datasetId,
        metadata,
        created_at: edge.node.createdAt,
        updated_at: edge.node.updatedAt,
        annotationSummaries: edge.node.annotationSummaries,
      });
    }

    if (
      !response.node.experiments.pageInfo.hasNextPage ||
      response.node.experiments.pageInfo.endCursor === null
    ) {
      break;
    }
    after = response.node.experiments.pageInfo.endCursor;
  }

  return experiments;
});

const listExperimentRuns = Effect.fn("listExperimentRuns")(function* (
  phoenix: PhoenixClientImpl,
  experimentId: string,
) {
  const runs: ExperimentRun[] = [];
  let after: string | undefined;

  while (true) {
    const response: GraphqlExperimentRunsResponse = yield* phoenix.graphql({
      operationName: "FailureModeExperimentRuns",
      query: GRAPHQL_EXPERIMENT_RUNS_QUERY,
      variables: {
        experimentId,
        first: GRAPHQL_RUN_PAGE_SIZE,
        after,
      },
      schema: GRAPHQL_EXPERIMENT_RUNS_SCHEMA,
    });

    if (response.node === null) {
      yield* Effect.logWarning(`Experiment ${experimentId}: missing GraphQL node, skipping`);
      return runs;
    }

    for (const edge of response.node.runs.edges) {
      const output = isRecord(edge.node.output) ? edge.node.output : null;
      const exampleInput = isRecord(edge.node.example.revision.input)
        ? edge.node.example.revision.input
        : {};
      const exampleOutput = isRecord(edge.node.example.revision.output)
        ? edge.node.example.revision.output
        : {};
      runs.push({
        example_id: edge.node.example.id,
        output,
        example_input: exampleInput,
        example_output: exampleOutput,
        error: edge.node.error,
        annotations: edge.node.annotations.edges.map((annotation) => ({
          name: annotation.node.name,
          score: annotation.node.score,
        })),
      });
    }

    if (!response.node.runs.pageInfo.hasNextPage || response.node.runs.pageInfo.endCursor === null)
      break;
    after = response.node.runs.pageInfo.endCursor;
  }

  return runs;
});

const F1_ANNOTATION_NAME = "question-f1";
const DATASET_FETCH_CONCURRENCY = 4;

/** Read the stored F1 score from Phoenix annotations. Returns null if missing. */
const getRunF1 = (run: ExperimentRun): number | null => {
  const annotations: readonly ExperimentAnnotation[] = Array.isArray(run.annotations)
    ? run.annotations
    : [];
  const f1Annotation = annotations.find((a) => a.name === F1_ANNOTATION_NAME);
  return f1Annotation?.score ?? null;
};

// ---------------------------------------------------------------------------
// Step 1: Collect all F1=0 failures
// ---------------------------------------------------------------------------

const collectDatasetFailedRuns = Effect.fn("collectDatasetFailedRuns")(function* (
  phoenix: PhoenixClientImpl,
  dataset: DatasetInfo,
  limitPerDataset: number,
  includedModelIds: readonly string[] = [],
  concurrency: number,
) {
  const experiments = yield* listExperiments(phoenix, dataset.id);
  yield* Effect.log(`[${dataset.name}] ${experiments.length} experiments`);

  if (experiments.length === 0) return [] as FailedRun[];

  const {
    skippedByModelFilter,
    skippedByRagFilter,
    skippedByMetadataFilter,
    latestExperimentByModel,
  } = selectLatestExperimentsByModel(experiments, includedModelIds);

  const selectedExperiments = [...latestExperimentByModel.entries()];
  const selectedFailedRuns = yield* Effect.forEach(
    selectedExperiments,
    ([modelId, experiment]) =>
      Effect.gen(function* () {
        const f1Summary = experiment.annotationSummaries.find(
          (summary) => summary.annotationName === F1_ANNOTATION_NAME,
        );
        if (f1Summary !== undefined && (f1Summary.minScore ?? Number.POSITIVE_INFINITY) > 0) {
          yield* Effect.logWarning(
            `[${dataset.name}]   "${modelId}": latest experiment ${experiment.id} has no F1=0 failures by annotation summary; analysis will skip this dataset/model pair`,
          );
          return [] as FailedRun[];
        }

        const runs = yield* listExperimentRuns(phoenix, experiment.id).pipe(
          Effect.catchAll((error) => {
            return Effect.logWarning(
              `[${dataset.name}] Skipping experiment ${experiment.id}: ${error}`,
            ).pipe(Effect.map(() => [] as ExperimentRun[]));
          }),
        );

        let f1ZeroCount = 0;
        const failedRuns: FailedRun[] = [];
        for (const run of runs) {
          const f1 = getRunF1(run);
          if (f1 !== 0) continue; // skip null (no annotation) and f1 > 0

          f1ZeroCount++;
          const input = run.example_input as {
            id?: string;
            questionText?: string;
            images?: Array<{ uri?: string; caption?: string }>;
            options?: Array<{
              id?: string;
              text?: string;
              images?: Array<{ uri?: string; caption?: string }>;
            }>;
          };
          const referenceOutput = run.example_output as {
            correctOptionIds?: string[];
          };
          const normalizeImages = (images: Array<{ uri?: string; caption?: string }> = []) =>
            images
              .filter(
                (image): image is { uri: string; caption?: string } =>
                  typeof image.uri === "string",
              )
              .map((image) => ({
                uri: image.uri,
                ...(typeof image.caption === "string" ? { caption: image.caption } : {}),
              }));
          const options: RunOption[] = Array.isArray(input.options)
            ? input.options
                .filter(
                  (
                    option,
                  ): option is {
                    id: string;
                    text?: string;
                    images?: Array<{ uri?: string; caption?: string }>;
                  } => typeof option.id === "string",
                )
                .map((option) => ({
                  id: option.id,
                  text: typeof option.text === "string" ? option.text : "",
                  images: normalizeImages(option.images),
                }))
            : [];

          const modelAnswer = Array.isArray(run.output?.answerIds)
            ? (run.output.answerIds as string[])
            : typeof run.output?.answerId === "string"
              ? [run.output.answerId]
              : [];
          const modelReason = typeof run.output?.reason === "string" ? run.output.reason : "";
          const correctAnswer = Array.isArray(referenceOutput.correctOptionIds)
            ? referenceOutput.correctOptionIds.filter((id): id is string => typeof id === "string")
            : [];
          const questionId =
            typeof input.id === "string" && input.id.trim().length > 0 ? input.id : run.example_id;

          failedRuns.push({
            sourceDataset: dataset.name,
            exampleId: run.example_id,
            questionId,
            questionKey: questionKeyOf(dataset.name, questionId),
            modelId,
            questionText: input.questionText ?? "",
            questionImages: normalizeImages(input.images),
            options,
            modelAnswer,
            modelReason,
            correctAnswer,
          });
        }

        yield* Effect.log(
          `[${dataset.name}]   "${modelId}": using latest experiment ${experiment.id}, ${runs.length} runs, ${f1ZeroCount} F1=0`,
        );
        if (f1ZeroCount === 0) {
          yield* Effect.logWarning(
            `[${dataset.name}]   "${modelId}": latest experiment ${experiment.id} has no F1=0 failures after run inspection; analysis will skip this dataset/model pair`,
          );
        }

        return failedRuns;
      }),
    { concurrency },
  );

  if (skippedByModelFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByModelFilter} experiments not in included model IDs`,
    );
  }

  if (skippedByRagFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByRagFilter} RAG experiments; pass --include-rag to include them`,
    );
  }

  if (skippedByMetadataFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByMetadataFilter} experiments by metadata keyword filter`,
    );
  }

  const datasetFailed = selectedFailedRuns.flat();

  // Dedupe by (exampleId, modelId) within this dataset
  const seen = new Set<string>();
  const deduped = datasetFailed.filter((run) => {
    const key = `${run.exampleId}::${run.modelId}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  // Apply per-dataset limit
  const limited = limitPerDataset > 0 ? deduped.slice(0, limitPerDataset) : deduped;
  if (limitPerDataset > 0 && deduped.length > limitPerDataset) {
    yield* Effect.log(
      `[${dataset.name}] Limiting to ${limitPerDataset}/${deduped.length} failures`,
    );
  }

  yield* Effect.log(`[${dataset.name}] ${limited.length} unique failures collected`);
  return limited;
});

const collectDatasetAnalysisTargets = Effect.fn("collectDatasetAnalysisTargets")(function* (
  phoenix: PhoenixClientImpl,
  dataset: DatasetInfo,
  includedModelIds: readonly string[] = [],
) {
  const experiments = yield* listExperiments(phoenix, dataset.id);
  yield* Effect.log(`[${dataset.name}] ${experiments.length} experiments`);

  if (experiments.length === 0) return [] as DatasetModelPairSummary[];

  const {
    skippedByModelFilter,
    skippedByRagFilter,
    skippedByMetadataFilter,
    latestExperimentByModel,
  } = selectLatestExperimentsByModel(experiments, includedModelIds);

  if (skippedByModelFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByModelFilter} experiments not in included model IDs`,
    );
  }

  if (skippedByRagFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByRagFilter} RAG experiments; pass --include-rag to include them`,
    );
  }

  if (skippedByMetadataFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByMetadataFilter} experiments by metadata keyword filter`,
    );
  }

  const targets: DatasetModelPairSummary[] = [];
  for (const [modelId, experiment] of latestExperimentByModel.entries()) {
    const f1Summary = experiment.annotationSummaries.find(
      (summary) => summary.annotationName === F1_ANNOTATION_NAME,
    );
    if (f1Summary !== undefined && (f1Summary.minScore ?? Number.POSITIVE_INFINITY) > 0) {
      yield* Effect.logWarning(
        `[${dataset.name}]   "${modelId}": latest experiment ${experiment.id} has no F1=0 failures by annotation summary; dry run will skip this dataset/model pair`,
      );
      continue;
    }
    if (f1Summary === undefined || f1Summary.minScore === null) {
      yield* Effect.logWarning(
        `[${dataset.name}]   "${modelId}": latest experiment ${experiment.id} has no usable question-f1 summary; dry run is conservatively including this dataset/model pair`,
      );
    } else {
      yield* Effect.log(
        `[${dataset.name}]   "${modelId}": latest experiment ${experiment.id} has F1=0 failures by annotation summary`,
      );
    }
    targets.push({
      sourceDataset: dataset.name,
      modelId,
      experimentId: experiment.id,
    });
  }

  return targets;
});

const collectFailedRuns = Effect.fn("collectFailedRuns")(function* (
  phoenix: PhoenixClientImpl,
  sourceNames: string[],
  limitPerDataset: number,
  includedModelIds: readonly string[] = [],
  concurrency: number,
) {
  const allDatasets = yield* listAllDatasets(phoenix);
  const sources = allDatasets.filter((d) => sourceNames.includes(d.name));

  const missing = sourceNames.filter((n) => !sources.some((d) => d.name === n));
  if (missing.length > 0) {
    yield* Effect.logWarning(`Missing datasets: ${missing.join(", ")}`);
  }

  yield* Effect.log(`Found ${sources.length}/${sourceNames.length} source datasets`);
  const failedByDataset = yield* Effect.forEach(
    sources,
    (dataset) =>
      collectDatasetFailedRuns(phoenix, dataset, limitPerDataset, includedModelIds, concurrency),
    { concurrency: DATASET_FETCH_CONCURRENCY },
  );

  return failedByDataset.flat();
});

const collectAnalysisTargets = Effect.fn("collectAnalysisTargets")(function* (
  phoenix: PhoenixClientImpl,
  sourceNames: string[],
  includedModelIds: readonly string[] = [],
) {
  const allDatasets = yield* listAllDatasets(phoenix);
  const sources = allDatasets.filter((d) => sourceNames.includes(d.name));

  const missing = sourceNames.filter((n) => !sources.some((d) => d.name === n));
  if (missing.length > 0) {
    yield* Effect.logWarning(`Missing datasets: ${missing.join(", ")}`);
  }

  yield* Effect.log(`Found ${sources.length}/${sourceNames.length} source datasets`);
  const targetsByDataset = yield* Effect.forEach(
    sources,
    (dataset) => collectDatasetAnalysisTargets(phoenix, dataset, includedModelIds),
    { concurrency: DATASET_FETCH_CONCURRENCY },
  );

  return sortDatasetModelPairs(targetsByDataset.flat());
});

// ---------------------------------------------------------------------------
// Step 2: Analyze each failure with LLM
// ---------------------------------------------------------------------------

const failureModeList = FAILURE_MODE_DEFINITIONS.map((d) => `  - "${d.id}": ${d.description}`).join(
  "\n",
);

const ANALYSIS_SYSTEM_PROMPT_BASE = `You are an expert in maritime education and AI evaluation. Your task is to analyze why an AI model answered a maritime multiple-choice question incorrectly.

For each failed question, determine:
1. The failure mode category from the predefined list below. If none fit, use "other" and provide a customFailureMode string.
2. Why the model likely chose the wrong answer
3. What makes this question difficult
4. What specific knowledge or reasoning was needed

## Failure mode categories

${failureModeList}

## Examples

Example 1:
Question: "According to COLREG Rule 19, a vessel hearing a fog signal apparently forward of her beam shall..."
Model answered: "Reduce speed to bare steerageway" (B)
Correct answer: "Take all way off and navigate with extreme caution" (C)
Analysis: failureMode="knowledge_gap", explanation="The model confused the general obligation to reduce speed with the specific requirement when a fog signal is heard forward of the beam. Both are mentioned in Rule 19 but apply to different situations.", difficultyFactors=["Two options reference valid COLREG actions", "Requires distinguishing between general and specific obligations"], requiredKnowledge=["COLREG Rule 19(e) specific requirements", "Distinction between Rule 19(b) general speed and 19(e) fog signal response"]

Example 2:
Question: "What is the minimum number of lifebuoys required on a cargo ship of 200m in length?"
Model answered: "12" (C)
Correct answer: "14" (D)
Analysis: failureMode="knowledge_gap", explanation="The model likely applied the wrong threshold from the SOLAS table. Ships 200m and above require 14 lifebuoys, not 12.", difficultyFactors=["Requires exact recall of regulatory table values", "Adjacent numeric options are close"], requiredKnowledge=["SOLAS Chapter III lifebuoy requirements table", "Ship length thresholds for safety equipment"]

Example 3:
Question: "A vessel is overtaking another in a narrow channel. What sound signal should the overtaking vessel make?"
Model answered: "Two prolonged blasts followed by one short blast" (A)
Correct answer: "Two prolonged blasts followed by two short blasts" (B)
Analysis: failureMode="interpretation_or_confusion", explanation="The model confused the signal for 'I intend to overtake on your starboard side' (2 prolonged + 1 short) with 'I intend to overtake on your port side' (2 prolonged + 2 short). The question context implied port-side overtaking.", difficultyFactors=["Signal patterns differ by only one short blast", "Requires contextual inference of overtaking side"], requiredKnowledge=["COLREG Rule 34(c) overtaking signals", "Narrow channel overtaking procedures"]

Prefer using a predefined category whenever it reasonably fits. Only use "other" when you are confident none of the predefined categories apply.`;

const ANALYSIS_SYSTEM_PROMPT_TEXT = ANALYSIS_SYSTEM_PROMPT_BASE;

const ANALYSIS_SYSTEM_PROMPT_MULTIMODAL = `${ANALYSIS_SYSTEM_PROMPT_BASE}

For multimodal questions, infer image-related failures only from the model's chosen answer and explanation. Use "image_understanding_error" when the failure appears to come from misunderstanding or misusing the visual information required by the question. Do not assume access to the image itself, and do not use an image-related category unless the response clearly suggests the visual component was the issue.`;

const formatOption = (option: RunOption) => {
  const baseText = option.text || (option.images.length > 0 ? "[See image]" : "");
  return `  ${option.id}. ${baseText}`;
};

const buildAnalysisPrompt = (run: FailedRun) => `Analyze this failed maritime MCQ response:

**Source Dataset:** ${run.sourceDataset}
**Model:** ${run.modelId}
**Analysis Mode:** ${ANALYSIS_MODE}

**Question:** ${run.questionText}

**Options:**
${run.options.map(formatOption).join("\n")}

**Model's Answer:** ${run.modelAnswer.join(", ")}
**Model's Reasoning:** ${run.modelReason || "(none)"}
**Correct Answer:** ${run.correctAnswer.join(", ")}

Why did the model get this wrong? Categorize the failure mode.`;

const analyzeFailure = (run: FailedRun) =>
  generateObject({
    schema: failureAnalysisJsonSchema,
    system:
      ANALYSIS_MODE === "multimodal"
        ? ANALYSIS_SYSTEM_PROMPT_MULTIMODAL
        : ANALYSIS_SYSTEM_PROMPT_TEXT,
    prompt: buildAnalysisPrompt(run),
  }).pipe(Effect.map((result) => result.object as FailureAnalysis));

// ---------------------------------------------------------------------------
// Step 3: Run analyses with progress tracking
// ---------------------------------------------------------------------------

/** Interval (in completed items) between incremental saves. */
const SAVE_EVERY = 1;

const analyzeAllFailures = (
  failedRuns: FailedRun[],
  concurrency: number,
  onResult: (result: AnalysisResult) => Effect.Effect<void>,
) =>
  Progress.forEach(
    failedRuns,
    (run) =>
      analyzeFailure(run).pipe(
        taskRetryPolicy,
        Effect.map(
          (analysis): AnalysisResult => ({
            ...run,
            analysis,
          }),
        ),
        Effect.catchAll((error) =>
          Effect.succeed({
            ...run,
            analysis: null,
            error: error instanceof Error ? error.message : String(error),
          } satisfies AnalysisResult),
        ),
        Effect.tap(onResult),
      ),
    {
      description: "Analyzing failures",
      concurrency,
    },
  );

// ---------------------------------------------------------------------------
// Step 4: Aggregate failure mode stats
// ---------------------------------------------------------------------------

export type FailureModeStats = {
  /** Counts per resolved failure mode label, sorted descending. */
  overall: Array<{ mode: string; count: number; percentage: number }>;
  /** Counts per failure mode, broken down by dataset. */
  byDataset: Record<string, Array<{ mode: string; count: number }>>;
  /** Counts per failure mode, broken down by model. */
  byModel: Record<string, Array<{ mode: string; count: number }>>;
  /** Counts per failure mode, broken down by dataset and then model. */
  byDatasetModel: Record<string, Record<string, Array<{ mode: string; count: number }>>>;
};

export const computeFailureModeStats = (analyses: readonly AnalysisResult[]): FailureModeStats => {
  const successful = analyses.filter((a) => a.analysis !== null);

  // Overall counts
  const overallCounts = new Map<string, number>();
  const byDatasetCounts = new Map<string, Map<string, number>>();
  const byModelCounts = new Map<string, Map<string, number>>();
  const byDatasetModelCounts = new Map<string, Map<string, Map<string, number>>>();

  for (const a of successful) {
    const mode = resolveFailureMode(a.analysis!);

    overallCounts.set(mode, (overallCounts.get(mode) ?? 0) + 1);

    if (!byDatasetCounts.has(a.sourceDataset)) byDatasetCounts.set(a.sourceDataset, new Map());
    const dsCounts = byDatasetCounts.get(a.sourceDataset)!;
    dsCounts.set(mode, (dsCounts.get(mode) ?? 0) + 1);

    if (!byModelCounts.has(a.modelId)) byModelCounts.set(a.modelId, new Map());
    const mCounts = byModelCounts.get(a.modelId)!;
    mCounts.set(mode, (mCounts.get(mode) ?? 0) + 1);

    if (!byDatasetModelCounts.has(a.sourceDataset)) {
      byDatasetModelCounts.set(a.sourceDataset, new Map());
    }
    const datasetModelCounts = byDatasetModelCounts.get(a.sourceDataset)!;
    if (!datasetModelCounts.has(a.modelId)) {
      datasetModelCounts.set(a.modelId, new Map());
    }
    const modelWithinDatasetCounts = datasetModelCounts.get(a.modelId)!;
    modelWithinDatasetCounts.set(mode, (modelWithinDatasetCounts.get(mode) ?? 0) + 1);
  }

  const total = successful.length;
  const sortedEntries = (m: Map<string, number>) =>
    [...m.entries()].sort((a, b) => b[1] - a[1]).map(([mode, count]) => ({ mode, count }));

  const overall = sortedEntries(overallCounts).map((e) => ({
    ...e,
    percentage: total > 0 ? Math.round((e.count / total) * 1000) / 10 : 0,
  }));

  const byDataset: Record<string, Array<{ mode: string; count: number }>> = {};
  for (const [ds, counts] of byDatasetCounts) byDataset[ds] = sortedEntries(counts);

  const byModel: Record<string, Array<{ mode: string; count: number }>> = {};
  for (const [model, counts] of byModelCounts) byModel[model] = sortedEntries(counts);

  const byDatasetModel: Record<string, Record<string, Array<{ mode: string; count: number }>>> = {};
  for (const [dataset, modelCounts] of byDatasetModelCounts) {
    byDatasetModel[dataset] = {};
    for (const [model, counts] of modelCounts) {
      byDatasetModel[dataset][model] = sortedEntries(counts);
    }
  }

  return { overall, byDataset, byModel, byDatasetModel };
};

// ---------------------------------------------------------------------------
// Step 5: Per-failure-mode recursive summarization
// ---------------------------------------------------------------------------

/** Max input tokens per summarization LLM call. */
const SUMMARY_MAX_INPUT_TOKENS = 50_000;

const SUMMARY_SYSTEM_PROMPT =
  "Summarize maritime MCQ failure analyses. Be concise — use short paragraphs, no bullet filler.";

export type SummaryTextGenerator = typeof generateText;

/** Format a single analysis for inclusion in a summary prompt. */
const formatAnalysisForSummary = (a: AnalysisResult, i: number) =>
  `${i + 1}. [${a.sourceDataset}|${a.modelId}] "${a.questionText.slice(0, 120)}…" Model:${a.modelAnswer.join(",")} Correct:${a.correctAnswer.join(",")} — ${a.analysis!.explanation}`;

/** Build the summarize-chunk prompt text (without the analyses lines). */
const chunkPromptPrefix = (modeLabel: string, modeDescription: string, count: number) =>
  `${count} failures for "${modeLabel}" (${modeDescription}):\n\n`;

const chunkPromptSuffix = "\n\nSummarize: common patterns, affected datasets, key knowledge gaps.";

const mergePromptPrefix = (modeLabel: string, modeDescription: string, count: number) =>
  `${count} partial summaries for "${modeLabel}" (${modeDescription}):\n\n`;

const mergePromptSuffix =
  "\n\nMerge these into one concise summary. Deduplicate repeated points and keep dataset-level distinctions when they matter.";

const compressPromptPrefix = (modeLabel: string, modeDescription: string) =>
  `Compress this partial summary for "${modeLabel}" (${modeDescription}) while preserving the key dataset patterns and major knowledge gaps:\n\n`;

const formatPartialSummaryForMerge = (summary: string, i: number) => `--- ${i + 1} ---\n${summary}`;

const chunkItemsByTokenBudget = <T>(
  items: readonly T[],
  renderItem: (item: T, indexInChunk: number) => string,
  prefix: string,
  suffix: string,
  maxInputTokens = SUMMARY_MAX_INPUT_TOKENS,
): T[][] => {
  const systemTokens = encode(SUMMARY_SYSTEM_PROMPT).length;
  const overhead = systemTokens + encode(prefix).length + encode(suffix).length;
  const budget = Math.max(1, maxInputTokens - overhead);

  const chunks: T[][] = [];
  let currentChunk: T[] = [];
  let currentTokens = 0;

  for (const item of items) {
    const renderedItem = renderItem(item, currentChunk.length);
    const itemTokens = encode(renderedItem).length + 1; // +1 for newline
    if (currentChunk.length > 0 && currentTokens + itemTokens > budget) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentTokens = 0;
    }
    currentChunk.push(item);
    currentTokens += itemTokens;
  }

  if (currentChunk.length > 0) chunks.push(currentChunk);
  return chunks;
};

const partialSummaryNeedsCompression = (
  modeLabel: string,
  modeDescription: string,
  summary: string,
  maxInputTokens = SUMMARY_MAX_INPUT_TOKENS,
): boolean => {
  const prompt = `${compressPromptPrefix(modeLabel, modeDescription)}${summary}`.trimEnd();
  return encode(SUMMARY_SYSTEM_PROMPT).length + encode(prompt).length > maxInputTokens;
};

/**
 * Split analyses into chunks that each fit within SUMMARY_MAX_INPUT_TOKENS
 * when formatted as a summarize-chunk prompt.
 */
export const chunkByTokenBudget = (
  modeLabel: string,
  modeDescription: string,
  analyses: readonly AnalysisResult[],
  maxInputTokens = SUMMARY_MAX_INPUT_TOKENS,
): AnalysisResult[][] => {
  return chunkItemsByTokenBudget(
    analyses,
    formatAnalysisForSummary,
    chunkPromptPrefix(modeLabel, modeDescription, analyses.length),
    chunkPromptSuffix,
    maxInputTokens,
  );
};

/** Summarize a single chunk of analyses for one failure mode. */
const summarizeChunk = (
  modeLabel: string,
  modeDescription: string,
  chunk: readonly AnalysisResult[],
  textGenerator: SummaryTextGenerator = generateText,
) =>
  textGenerator({
    system: SUMMARY_SYSTEM_PROMPT,
    prompt: `${chunkPromptPrefix(modeLabel, modeDescription, chunk.length)}${chunk.map(formatAnalysisForSummary).join("\n")}${chunkPromptSuffix}`,
  }).pipe(Effect.map((r) => r.text));

/** Merge multiple partial summaries for one failure mode into a single summary. */
const mergeSummaries = (
  modeLabel: string,
  modeDescription: string,
  partialSummaries: readonly string[],
  textGenerator: SummaryTextGenerator = generateText,
) =>
  textGenerator({
    system: SUMMARY_SYSTEM_PROMPT,
    prompt: `${mergePromptPrefix(modeLabel, modeDescription, partialSummaries.length)}${partialSummaries.map(formatPartialSummaryForMerge).join("\n\n")}${mergePromptSuffix}`,
  }).pipe(Effect.map((r) => r.text));

const compressPartialSummary = (
  modeLabel: string,
  modeDescription: string,
  summary: string,
  textGenerator: SummaryTextGenerator = generateText,
) =>
  textGenerator({
    system: SUMMARY_SYSTEM_PROMPT,
    prompt: `${compressPromptPrefix(modeLabel, modeDescription)}${summary}`,
  }).pipe(Effect.map((r) => r.text));

export const chunkPartialSummariesByTokenBudget = (
  modeLabel: string,
  modeDescription: string,
  partialSummaries: readonly string[],
  maxInputTokens = SUMMARY_MAX_INPUT_TOKENS,
): string[][] =>
  chunkItemsByTokenBudget(
    partialSummaries,
    formatPartialSummaryForMerge,
    mergePromptPrefix(modeLabel, modeDescription, partialSummaries.length),
    mergePromptSuffix,
    maxInputTokens,
  );

export const mergeSummariesRecursively = (
  modeLabel: string,
  modeDescription: string,
  partialSummaries: readonly string[],
  maxInputTokens = SUMMARY_MAX_INPUT_TOKENS,
  textGenerator: SummaryTextGenerator = generateText,
): Effect.Effect<string, LanguageModelError, LanguageModel> =>
  Effect.gen(function* () {
    if (partialSummaries.length === 0) return "";

    const normalizedPartialSummaries: string[] = [];
    for (const partialSummary of partialSummaries) {
      if (
        !partialSummaryNeedsCompression(modeLabel, modeDescription, partialSummary, maxInputTokens)
      ) {
        normalizedPartialSummaries.push(partialSummary);
        continue;
      }

      yield* Effect.log(
        `  [${modeLabel}] compressing oversized partial summary before recursive merge`,
      );
      normalizedPartialSummaries.push(
        yield* compressPartialSummary(modeLabel, modeDescription, partialSummary, textGenerator),
      );
    }

    if (normalizedPartialSummaries.length === 1) {
      return normalizedPartialSummaries[0]!;
    }

    const chunks = chunkPartialSummariesByTokenBudget(
      modeLabel,
      modeDescription,
      normalizedPartialSummaries,
      maxInputTokens,
    );

    if (chunks.length === 1) {
      return yield* mergeSummaries(modeLabel, modeDescription, chunks[0]!, textGenerator);
    }

    if (chunks.every((chunk) => chunk.length === 1)) {
      yield* Effect.log(
        `  [${modeLabel}] merge chunks are all singletons; compressing partial summaries before retry`,
      );
      const compressedSummaries: string[] = [];
      for (const partialSummary of normalizedPartialSummaries) {
        compressedSummaries.push(
          yield* compressPartialSummary(modeLabel, modeDescription, partialSummary, textGenerator),
        );
      }
      return yield* mergeSummariesRecursively(
        modeLabel,
        modeDescription,
        compressedSummaries,
        maxInputTokens,
        textGenerator,
      );
    }

    yield* Effect.log(
      `  [${modeLabel}] ${normalizedPartialSummaries.length} partial summaries -> ${chunks.length} merge chunks`,
    );

    const reducedSummaries: string[] = [];
    for (const chunk of chunks) {
      if (chunk.length === 1) {
        reducedSummaries.push(chunk[0]!);
        continue;
      }
      reducedSummaries.push(
        yield* mergeSummaries(modeLabel, modeDescription, chunk, textGenerator),
      );
    }

    return yield* mergeSummariesRecursively(
      modeLabel,
      modeDescription,
      reducedSummaries,
      maxInputTokens,
      textGenerator,
    );
  });

/**
 * Recursively summarize analyses for a single failure mode.
 * Splits into token-budget-bounded chunks, summarizes each, then merges.
 */
export const recursiveSummarizeMode = (
  modeLabel: string,
  modeDescription: string,
  analyses: readonly AnalysisResult[],
  maxInputTokens = SUMMARY_MAX_INPUT_TOKENS,
  textGenerator: SummaryTextGenerator = generateText,
): Effect.Effect<string, LanguageModelError, LanguageModel> =>
  Effect.gen(function* () {
    const chunks = chunkByTokenBudget(modeLabel, modeDescription, analyses, maxInputTokens);

    if (chunks.length === 1) {
      return yield* summarizeChunk(modeLabel, modeDescription, chunks[0]!, textGenerator);
    }

    yield* Effect.log(`  [${modeLabel}] ${analyses.length} analyses -> ${chunks.length} chunks`);

    const partialSummaries: string[] = [];
    for (const chunk of chunks) {
      const summary = yield* summarizeChunk(modeLabel, modeDescription, chunk, textGenerator);
      partialSummaries.push(summary);
    }

    return yield* mergeSummariesRecursively(
      modeLabel,
      modeDescription,
      partialSummaries,
      maxInputTokens,
      textGenerator,
    );
  });

type FailureModeSummary = {
  mode: string;
  description: string;
  count: number;
  summary: string;
};

type CheckpointMetadata = {
  analysisModel?: string;
  analysisMode?: "text" | "multimodal";
  datasetsAnalyzed?: readonly string[];
  sourceDatasets?: readonly string[];
  includedModelIds?: readonly string[];
  excludedMetadataKeywords?: readonly string[];
  includeRag?: boolean;
  limitPerDataset?: number;
};

/** Group analyses by resolved failure mode and summarize each group. */
const summarizeByFailureMode = (analyses: readonly AnalysisResult[]) =>
  Effect.gen(function* () {
    const successful = analyses.filter((a) => a.analysis !== null);
    if (successful.length === 0) return [] as FailureModeSummary[];

    // Group by resolved failure mode
    const grouped = new Map<string, AnalysisResult[]>();
    for (const a of successful) {
      const mode = resolveFailureMode(a.analysis!);
      if (!grouped.has(mode)) grouped.set(mode, []);
      grouped.get(mode)!.push(a);
    }

    // Build a description lookup: predefined modes use their definition,
    // "other: X" modes get a generic description.
    const descriptionOf = (mode: string): string => {
      const def = FAILURE_MODE_DEFINITIONS.find((d) => d.id === mode);
      if (def) return def.description;
      return `Custom failure mode: ${mode}`;
    };

    // Sort by count descending
    const sortedModes = [...grouped.entries()].sort((a, b) => b[1].length - a[1].length);

    const summaries: FailureModeSummary[] = [];

    for (const [mode, group] of sortedModes) {
      const description = descriptionOf(mode);
      yield* Effect.log(`Summarizing "${mode}" (${group.length} analyses)...`);
      const summary = yield* recursiveSummarizeMode(mode, description, group);
      summaries.push({ mode, description, count: group.length, summary });
    }

    return summaries;
  });

// ---------------------------------------------------------------------------
// Step 5: Write output
// ---------------------------------------------------------------------------

const buildOutput = (
  analyses: readonly AnalysisResult[],
  stats: FailureModeStats | null,
  failureModeSummaries: readonly FailureModeSummary[],
  datasetsAnalyzed: readonly string[],
  analysisModelId: string,
) => {
  const modelsFound = [...new Set(analyses.map((a) => a.modelId))].sort();
  const successfulCount = analyses.filter((a) => a.analysis !== null).length;
  const failedCount = analyses.filter((a) => a.analysis === null).length;

  return {
    metadata: {
      timestamp: new Date().toISOString(),
      analysisModel: analysisModelId,
      analysisMode: ANALYSIS_MODE as AnalysisMode,
      summaryModel: analysisModelId,
      totalFailedRuns: analyses.length,
      successfulAnalyses: successfulCount,
      failedAnalyses: failedCount,
      datasetsAnalyzed,
      sourceDatasets: [
        ...new Set(ARGS.datasets.length > 0 ? ARGS.datasets : DEFAULT_SOURCE_DATASETS),
      ].sort(),
      modelsFound,
      includedModelIds: INCLUDED_MODEL_IDS,
      excludedMetadataKeywords: [...EXCLUDED_METADATA_KEYWORD_SET],
      includeRag: ARGS.includeRag,
      limitPerDataset: ARGS.limit,
      failureModeCategories: FAILURE_MODES,
    },
    failureModeDefinitions: FAILURE_MODE_DEFINITION_EXPORT,
    failureModeStats: stats,
    failureModeSummaries: failureModeSummaries.map((s) => ({
      mode: s.mode,
      description: s.description,
      count: s.count,
      summary: s.summary,
    })),
    analyses: analyses.map((a) => ({
      sourceDataset: a.sourceDataset,
      exampleId: a.exampleId,
      questionId: a.questionId,
      questionKey: a.questionKey,
      modelId: a.modelId,
      questionText: a.questionText,
      questionImages: Array.isArray(a.questionImages) ? a.questionImages : [],
      options: a.options.map((option) => ({
        id: option.id,
        text: option.text,
        images: Array.isArray(option.images) ? option.images : [],
      })),
      modelAnswer: a.modelAnswer,
      modelReason: a.modelReason,
      correctAnswer: a.correctAnswer,
      analysis: a.analysis,
      ...(a.error ? { error: a.error } : {}),
    })),
  };
};

/** Build a lightweight output containing only what Typst needs. */
const buildTypstOutput = (output: ReturnType<typeof buildOutput>) => ({
  metadata: {
    totalFailedRuns: output.metadata.totalFailedRuns,
    modelsFound: output.metadata.modelsFound,
    datasetsAnalyzed: output.metadata.datasetsAnalyzed,
  },
  failureModeDefinitions: output.failureModeDefinitions,
  failureModeStats: output.failureModeStats
    ? {
        overall: output.failureModeStats.overall,
        byDataset: output.failureModeStats.byDataset,
        byModel: output.failureModeStats.byModel,
        byDatasetModel: output.failureModeStats.byDatasetModel,
      }
    : null,
});

const writeJson = (path: string, data: unknown) =>
  Effect.tryPromise({
    try: () => {
      const json = JSON.stringify(data, null, 2)
        .replaceAll("\u2028", " ")
        .replaceAll("\u2029", " ");
      return Bun.write(path, json);
    },
    catch: (cause) => new WriteOutputError({ cause }),
  });

const saveOutput = (outputPath: string, output: ReturnType<typeof buildOutput>) =>
  writeJson(outputPath, output);

/** Derive the typst output path from the full output path. */
const typstOutputPath = (fullPath: string) => fullPath.replace(/\.json$/, "-typst.json");

const loadCheckpoint = (path: string) =>
  Effect.try({
    try: () => JSON.parse(readFileSync(path, "utf8")) as ReturnType<typeof buildOutput>,
    catch: (cause) => new ReadCheckpointError({ cause, path }),
  });

const normalizeStringArray = (values: readonly string[] = []) => [...new Set(values)].sort();

export const validateResumeCheckpoint = (
  checkpointPath: string,
  checkpointMetadata: CheckpointMetadata | undefined,
  current: {
    analysisModelId: string;
    analysisMode: AnalysisMode;
    sourceDatasets: readonly string[];
    includedModelIds: readonly string[];
    excludedMetadataKeywords: readonly string[];
    includeRag: boolean;
    limitPerDataset: number;
  },
) =>
  Effect.gen(function* () {
    if (!checkpointMetadata) {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} has no metadata; skipping resume compatibility checks`,
      );
      return;
    }

    const mismatches: string[] = [];

    if (
      typeof checkpointMetadata.analysisModel === "string" &&
      checkpointMetadata.analysisModel !== current.analysisModelId
    ) {
      mismatches.push(
        `analysis model differs: checkpoint=${checkpointMetadata.analysisModel}, current=${current.analysisModelId}`,
      );
    }

    if (
      typeof checkpointMetadata.analysisMode === "string" &&
      checkpointMetadata.analysisMode !== current.analysisMode
    ) {
      mismatches.push(
        `analysis mode differs: checkpoint=${checkpointMetadata.analysisMode}, current=${current.analysisMode}`,
      );
    } else if (typeof checkpointMetadata.analysisMode !== "string") {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store analysisMode; skipping that compatibility check`,
      );
    }

    if (Array.isArray(checkpointMetadata.includedModelIds)) {
      const checkpointModels = normalizeStringArray(checkpointMetadata.includedModelIds);
      const currentModels = normalizeStringArray(current.includedModelIds);
      if (JSON.stringify(checkpointModels) !== JSON.stringify(currentModels)) {
        mismatches.push(
          `included models differ: checkpoint=${checkpointModels.join(", ") || "(none)"}, current=${currentModels.join(", ") || "(none)"}`,
        );
      }
    } else {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store includedModelIds; skipping that compatibility check`,
      );
    }

    if (Array.isArray(checkpointMetadata.excludedMetadataKeywords)) {
      const checkpointExcluded = normalizeStringArray(
        checkpointMetadata.excludedMetadataKeywords.map((value) => value.toLowerCase()),
      );
      const currentExcluded = normalizeStringArray(
        current.excludedMetadataKeywords.map((value) => value.toLowerCase()),
      );
      if (JSON.stringify(checkpointExcluded) !== JSON.stringify(currentExcluded)) {
        mismatches.push(
          `excluded metadata keywords differ: checkpoint=${checkpointExcluded.join(", ") || "(none)"}, current=${currentExcluded.join(", ") || "(none)"}`,
        );
      }
    } else {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store excludedMetadataKeywords; skipping that compatibility check`,
      );
    }

    if (typeof checkpointMetadata.includeRag === "boolean") {
      if (checkpointMetadata.includeRag !== current.includeRag) {
        mismatches.push(
          `RAG inclusion differs: checkpoint=${checkpointMetadata.includeRag}, current=${current.includeRag}`,
        );
      }
    } else {
      mismatches.push(
        `resume checkpoint ${checkpointPath} does not store includeRag; regenerate it so RAG experiment filtering is explicit`,
      );
    }

    if (Array.isArray(checkpointMetadata.sourceDatasets)) {
      const checkpointDatasets = normalizeStringArray(checkpointMetadata.sourceDatasets);
      const currentDatasets = normalizeStringArray(current.sourceDatasets);
      if (JSON.stringify(checkpointDatasets) !== JSON.stringify(currentDatasets)) {
        mismatches.push(
          `source datasets differ: checkpoint=${checkpointDatasets.join(", ")}, current=${currentDatasets.join(", ")}`,
        );
      }
    } else {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store sourceDatasets; skipping that compatibility check`,
      );
    }

    if (
      typeof checkpointMetadata.limitPerDataset === "number" &&
      checkpointMetadata.limitPerDataset !== current.limitPerDataset
    ) {
      mismatches.push(
        `per-dataset limit differs: checkpoint=${checkpointMetadata.limitPerDataset}, current=${current.limitPerDataset}`,
      );
    } else if (typeof checkpointMetadata.limitPerDataset !== "number") {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store limitPerDataset; skipping that compatibility check`,
      );
    }

    if (mismatches.length > 0) {
      return yield* new ResumeCheckpointMismatchError({
        path: checkpointPath,
        mismatches,
      });
    }
  });

const saveCheckpoint = (
  outputPath: string,
  analyses: readonly AnalysisResult[],
  datasetsAnalyzed: readonly string[],
  analysisModelId: string,
) => {
  const output = buildOutput(
    analyses,
    analyses.length > 0 ? computeFailureModeStats(analyses) : null,
    [],
    datasetsAnalyzed,
    analysisModelId,
  );

  return saveOutput(outputPath, output).pipe(
    Effect.zipRight(writeJson(typstOutputPath(outputPath), buildTypstOutput(output))),
  );
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const program = Effect.gen(function* () {
  const phoenix = yield* PhoenixClient;
  const sourceNames = ARGS.datasets.length > 0 ? ARGS.datasets : DEFAULT_SOURCE_DATASETS;
  if (INCLUDED_MODEL_IDS.length > 0) {
    yield* Effect.log(`Included model ID filter enabled (${INCLUDED_MODEL_IDS.length} IDs):`);
    for (const modelId of INCLUDED_MODEL_IDS) {
      yield* Effect.log(`  - ${modelId}`);
    }
  }
  if (EXCLUDED_METADATA_KEYWORD_SET.size > 0) {
    yield* Effect.log(
      `Excluding experiments with metadata keywords: ${[...EXCLUDED_METADATA_KEYWORD_SET].join(", ")}`,
    );
  }
  if (ARGS.includeRag) {
    yield* Effect.log("RAG experiments are included (--include-rag)");
  } else {
    yield* Effect.log(
      "RAG experiments are excluded by default; pass --include-rag to include them",
    );
  }
  yield* Effect.log(
    `Analyzing ${ANALYSIS_MODE} failure modes for ${sourceNames.length} dataset(s)` +
      (ARGS.limit > 0 ? ` (limit: ${ARGS.limit} per dataset)` : "") +
      ` with concurrency ${ARGS.concurrency}` +
      (ARGS.dryRun ? " [dry run]" : ""),
  );

  if (ARGS.dryRun && ARGS.resume) {
    yield* Effect.log(`Ignoring --resume in dry-run mode`);
  }
  if (ARGS.dryRun && ARGS.output) {
    yield* Effect.log(`Ignoring --output in dry-run mode`);
  }

  if (ARGS.dryRun) {
    const modelIdsToProcess = INCLUDED_MODEL_IDS.length > 0 ? INCLUDED_MODEL_IDS : [null];
    const totalModels = modelIdsToProcess.length;
    const dryRunTargets: DatasetModelPairSummary[] = [];

    for (const [index, modelId] of modelIdsToProcess.entries()) {
      const modelLabel = modelId ?? "all models";
      yield* Effect.log(
        `\n=== Dry Run ${index + 1}/${totalModels}: Checking analyzable pairs for ${modelLabel} ===`,
      );
      const targets = yield* collectAnalysisTargets(phoenix, sourceNames, modelId ? [modelId] : []);
      dryRunTargets.push(...targets);
    }

    const pairs = sortDatasetModelPairs(dryRunTargets);
    yield* Effect.log(`\nDry run complete: ${pairs.length} dataset/model pairs would be analyzed`);
    for (const pair of pairs) {
      yield* Effect.log(`  [${pair.sourceDataset}] ${pair.modelId} via ${pair.experimentId}`);
    }
    if (pairs.length === 0) {
      yield* Effect.log("No dataset/model pairs would be analyzed.");
    }
    return;
  }

  const model = yield* LanguageModel;
  const analysisModelId = typeof model === "string" ? model : model.modelId;

  // Prepare output path and incremental save state
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputPath =
    ARGS.resume ??
    ARGS.output ??
    dataPath(
      "analysis",
      `failure-mode-analysis${ANALYSIS_MODE === "multimodal" ? "-multimodal" : ""}-${timestamp}.json`,
    );
  mkdirSync(dirname(outputPath), { recursive: true });
  if (ARGS.resume && ARGS.output && ARGS.resume !== ARGS.output) {
    yield* Effect.log(
      `Ignoring --output because --resume was provided; resuming in-place at ${ARGS.resume}`,
    );
  }
  if (ARGS.resume && !existsSync(ARGS.resume)) {
    return yield* new ReadCheckpointError({
      cause: new Error(`Checkpoint file does not exist: ${ARGS.resume}`),
      path: ARGS.resume,
    });
  }

  const checkpoint = ARGS.resume ? yield* loadCheckpoint(ARGS.resume) : null;
  if (ARGS.resume) {
    yield* validateResumeCheckpoint(ARGS.resume, checkpoint?.metadata, {
      analysisModelId,
      analysisMode: ANALYSIS_MODE,
      sourceDatasets: sourceNames,
      includedModelIds: INCLUDED_MODEL_IDS,
      excludedMetadataKeywords: [...EXCLUDED_METADATA_KEYWORD_SET],
      includeRag: ARGS.includeRag,
      limitPerDataset: ARGS.limit,
    });
  }
  const checkpointAnalyses = Array.isArray(checkpoint?.analyses) ? checkpoint.analyses : [];
  const resumedSuccessfulResults = checkpointAnalyses.filter((result) => result.analysis !== null);
  const completedAnalysisKeys = new Set(resumedSuccessfulResults.map(analysisKey));
  const collectedResults: AnalysisResult[] = [...resumedSuccessfulResults];
  const datasetsAnalyzed = () => [...new Set(collectedResults.map((a) => a.sourceDataset))].sort();
  let completedCount = resumedSuccessfulResults.length;
  let lastSavedAt = resumedSuccessfulResults.length;
  let totalPlannedAnalyses = resumedSuccessfulResults.length;

  if (checkpoint) {
    const checkpointTotal = checkpointAnalyses.length;
    const checkpointErrors = checkpointTotal - resumedSuccessfulResults.length;
    yield* Effect.log(
      `Resuming from ${ARGS.resume}: ${resumedSuccessfulResults.length} completed analyses restored` +
        (checkpointErrors > 0
          ? `, ${checkpointErrors} incomplete/error analyses will be retried`
          : ""),
    );
  }

  const onResult = (result: AnalysisResult) =>
    Effect.gen(function* () {
      collectedResults.push(result);
      if (result.analysis !== null) {
        completedAnalysisKeys.add(analysisKey(result));
      }
      completedCount++;
      if (completedCount - lastSavedAt >= SAVE_EVERY) {
        lastSavedAt = completedCount;
        yield* saveCheckpoint(
          outputPath,
          collectedResults,
          datasetsAnalyzed(),
          analysisModelId,
        ).pipe(
          Effect.tapError((e) => Effect.logWarning(`Incremental save failed: ${e}`)),
          Effect.catchAll(() => Effect.void),
        );
        yield* Effect.log(`Incremental save (${completedCount}/${totalPlannedAnalyses})`);
      }
    });
  const modelIdsToProcess = INCLUDED_MODEL_IDS.length > 0 ? INCLUDED_MODEL_IDS : [null];
  const totalModels = modelIdsToProcess.length;
  let totalFailedRuns = 0;

  for (const [index, modelId] of modelIdsToProcess.entries()) {
    const modelLabel = modelId ?? "all models";
    yield* Effect.log(
      `\n=== Step 1.${index + 1}: Collecting F1=0 failures from Phoenix for ${modelLabel} (${index + 1}/${totalModels}) ===`,
    );
    const failedRuns = yield* collectFailedRuns(
      phoenix,
      sourceNames,
      ARGS.limit,
      modelId ? [modelId] : [],
      ARGS.concurrency,
    );
    totalFailedRuns += failedRuns.length;
    yield* Effect.log(`[${modelLabel}] ${failedRuns.length} failures collected`);

    const pendingFailedRuns = failedRuns.filter(
      (run) => !completedAnalysisKeys.has(analysisKey(run)),
    );
    const resumedCount = failedRuns.length - pendingFailedRuns.length;
    totalPlannedAnalyses += pendingFailedRuns.length;

    if (resumedCount > 0) {
      yield* Effect.log(`[${modelLabel}] ${resumedCount} failures already analyzed, skipping`);
    }

    if (pendingFailedRuns.length === 0) {
      yield* Effect.log(`[${modelLabel}] No remaining failures to analyze, skipping`);
      continue;
    }

    yield* Effect.log(`\n=== Step 2.${index + 1}: Analyzing failures for ${modelLabel} ===`);
    yield* analyzeAllFailures(pendingFailedRuns, ARGS.concurrency, onResult);
  }

  yield* Effect.log(`\nTotal F1=0 failures collected: ${totalFailedRuns}`);

  if (collectedResults.length === 0) {
    yield* Effect.log("No failures found. Nothing to analyze.");
    return;
  }

  const results: readonly AnalysisResult[] = collectedResults;
  const successCount = results.filter((a) => a.analysis !== null).length;
  const errorCount = results.filter((a) => a.analysis === null).length;
  yield* Effect.log(`\nAnalysis complete: ${successCount} successful, ${errorCount} failed`);

  // Save after all analyses complete (before summaries)
  yield* saveCheckpoint(outputPath, results, datasetsAnalyzed(), analysisModelId);
  yield* Effect.log(`Saved analyses to ${outputPath}`);

  // Step 4: Aggregate stats
  yield* Effect.log("\n=== Step 3: Aggregating failure mode stats ===");
  const stats = computeFailureModeStats(results);
  for (const entry of stats.overall) {
    yield* Effect.log(`  ${entry.mode}: ${entry.count} (${entry.percentage}%)`);
  }

  // Save with stats
  yield* saveOutput(
    outputPath,
    buildOutput(results, stats, [], datasetsAnalyzed(), analysisModelId),
  );

  // Step 5: Summarize each failure mode
  yield* Effect.log("\n=== Step 4: Summarizing each failure mode ===");
  const failureModeSummaries = yield* summarizeByFailureMode(results);
  for (const s of failureModeSummaries) {
    yield* Effect.log(`  "${s.mode}" (${s.count}) — done`);
  }

  // Final save with everything
  yield* Effect.log("\n=== Step 5: Writing final output ===");
  const finalOutput = buildOutput(
    results,
    stats,
    failureModeSummaries,
    datasetsAnalyzed(),
    analysisModelId,
  );
  yield* saveOutput(outputPath, finalOutput);
  yield* Effect.log(`Full report written to ${outputPath}`);

  // Write lightweight Typst export (metadata + stats only)
  const typstPath = typstOutputPath(outputPath);
  yield* writeJson(typstPath, buildTypstOutput(finalOutput));
  yield* Effect.log(`Typst export written to ${typstPath}`);

  yield* Effect.log("\nDone!");
});

const layer = Layer.mergeAll(PhoenixClient.skyhigh, EvalLanguageModelLayer);
const runtimeLayer = Layer.mergeAll(layer, Logger.pretty);

if (import.meta.main) {
  Effect.runPromise(
    program.pipe(
      Effect.provide(runtimeLayer),
      Effect.catchAllCause((cause) => Effect.sync(() => console.error(Cause.pretty(cause)))),
    ),
  );
}
