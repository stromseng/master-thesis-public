#!/usr/bin/env bun
// Remap already-recorded open-ended correctness scores from stored judge choices.
//
// Usage:
//   bun scripts/phoenix/remap-open-ended-correctness-scores.ts
//   bun scripts/phoenix/remap-open-ended-correctness-scores.ts --apply
//   bun scripts/phoenix/remap-open-ended-correctness-scores.ts --experiment-id <id> --limit 10 --apply
//   bun scripts/phoenix/remap-open-ended-correctness-scores.ts --experiment-id <id> --base-url http://127.0.0.1:6006

import { getDatasetInfo } from "@arizeai/phoenix-client/datasets";
import { listExperiments } from "@arizeai/phoenix-client/experiments";
import * as Progress from "effective-progress";
import { Effect, Fiber, Layer, Logger, Queue, Ref, Schema } from "effect";
import { openEndedDatasetDefinition as crewOpenEndedDataset } from "../../evals/crew/basic_open_ended";
import { openEndedDatasetDefinition as pei2024UkOpenEndedDataset } from "../../evals/pei2024/uk_open_ended";
import { openEndedDatasetDefinition as pei2024ZhOpenEndedDataset } from "../../evals/pei2024/zh_open_ended";
import { openEndedDatasetDefinition as raynorOpenEndedDataset } from "../../evals/raynor/basic_open_ended";
import { openEndedDatasetDefinition as shititongEnOpenEndedDataset } from "../../evals/shititong/en_text_open_ended";
import { openEndedDatasetDefinition as shititongZhOpenEndedDataset } from "../../evals/shititong/zh_text_open_ended";
import { openEndedDatasetDefinition as usCoastGuardOpenEndedDataset } from "../../evals/us_coast_guard/basic_open_ended";
import {
  PHOENIX_SKYHIGH_BASE_URL,
  PhoenixClient as PhoenixClientService,
  type PhoenixClientImpl,
} from "../../src/services/PhoenixClient";
import type { FactualityChoice, FactualityScore } from "../../src/services/Judge";

const DEFAULT_ANNOTATION_NAME = "open-ended-correctness";
const DEFAULT_CONCURRENCY = 10;
const DEFAULT_PAGE_SIZE = 100;
const ANNOTATION_PAGE_SIZE = 100;
const REMAP_METADATA_KEY = "scoreRemap";
const REMAP_METADATA_VERSION = 1;
const OPEN_ENDED_DATASETS = [
  crewOpenEndedDataset,
  pei2024UkOpenEndedDataset,
  pei2024ZhOpenEndedDataset,
  raynorOpenEndedDataset,
  shititongEnOpenEndedDataset,
  shititongZhOpenEndedDataset,
  usCoastGuardOpenEndedDataset,
] as const;
const HELP_TEXT = [
  "Usage:",
  "  bun scripts/phoenix/remap-open-ended-correctness-scores.ts [--apply]",
  "  bun scripts/phoenix/remap-open-ended-correctness-scores.ts --experiment-id <id> [--apply]",
  "",
  "Options:",
  "  --experiment-id <id>    Experiment to update. Repeatable. If omitted, all known open-ended datasets are used.",
  "  --dataset <name>        Open-ended dataset to discover experiments from. Repeatable; overrides defaults.",
  "  --annotation <name>     Annotation name. Default: open-ended-correctness.",
  "  --limit <n>             Maximum changed evaluations per experiment.",
  "  --concurrency <n>       Concurrent Phoenix scans/upserts. Default: 10.",
  "  --base-url <url>        Phoenix base URL.",
  "  --apply                 Write score updates. Omit for dry-run.",
].join("\n");

type CliArgs = {
  experimentIds: string[];
  datasetNames: string[];
  annotationName: string;
  apply: boolean;
  limit: number | null;
  concurrency: number;
  baseUrl: string;
};

type RemappedResult = {
  score: FactualityScore;
  label: "perfect" | "none" | "partial";
  explanation: string | null;
  metadata: Record<string, unknown>;
  traceId: string | null;
};

class CliArgsError extends Schema.TaggedError<CliArgsError>()("CliArgsError", {
  reason: Schema.String,
}) {}

class UpsertEvaluationError extends Schema.TaggedError<UpsertEvaluationError>()(
  "UpsertEvaluationError",
  {
    experimentRunId: Schema.String,
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

const parsePositiveInteger = (name: string, value: string): Effect.Effect<number, CliArgsError> =>
  Effect.gen(function* () {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed < 1) {
      return yield* new CliArgsError({ reason: `${name} must be a positive integer.` });
    }
    return parsed;
  });

const parseArgs = Effect.fn("remapOpenEndedCorrectnessScores.parseArgs")(function* (
  argv: readonly string[],
) {
  const experimentIds: string[] = [];
  const datasetNames: string[] = [];
  let annotationName = DEFAULT_ANNOTATION_NAME;
  let apply = false;
  let limit: number | null = null;
  let concurrency = DEFAULT_CONCURRENCY;
  let baseUrl = PHOENIX_SKYHIGH_BASE_URL;

  const readValue = (index: number, flag: string): Effect.Effect<string, CliArgsError> =>
    Effect.gen(function* () {
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) {
        return yield* new CliArgsError({ reason: `Missing value after ${flag}.` });
      }
      return value;
    });

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--experiment-id":
      case "--experiment": {
        experimentIds.push(yield* readValue(i, arg));
        i += 1;
        break;
      }
      case "--dataset":
      case "--dataset-name": {
        datasetNames.push(yield* readValue(i, arg));
        i += 1;
        break;
      }
      case "--annotation": {
        annotationName = yield* readValue(i, arg);
        i += 1;
        break;
      }
      case "--apply": {
        apply = true;
        break;
      }
      case "--limit": {
        limit = yield* parsePositiveInteger("--limit", yield* readValue(i, arg));
        i += 1;
        break;
      }
      case "--concurrency": {
        concurrency = yield* parsePositiveInteger("--concurrency", yield* readValue(i, arg));
        i += 1;
        break;
      }
      case "--base-url": {
        baseUrl = yield* readValue(i, arg);
        i += 1;
        break;
      }
      case "--help": {
        return yield* new CliArgsError({
          reason: HELP_TEXT,
        });
      }
      default:
        return yield* new CliArgsError({ reason: `Unknown argument '${arg}'.` });
    }
  }

  return {
    experimentIds,
    datasetNames:
      datasetNames.length > 0 ? datasetNames : OPEN_ENDED_DATASETS.map(({ name }) => name),
    annotationName,
    apply,
    limit,
    concurrency,
    baseUrl,
  } satisfies CliArgs;
});

const PageInfoSchema = Schema.Struct({
  hasNextPage: Schema.Boolean,
  endCursor: Schema.NullOr(Schema.String),
});

const AnnotationSchema = Schema.Struct({
  name: Schema.String,
  label: Schema.NullOr(Schema.String),
  score: Schema.NullOr(Schema.Number),
  explanation: Schema.NullOr(Schema.String),
  error: Schema.NullOr(Schema.String),
  metadata: Schema.Unknown,
  startTime: Schema.String,
  endTime: Schema.String,
  traceId: Schema.NullOr(Schema.String),
});

const ExperimentRunsResponseSchema = Schema.Struct({
  node: Schema.NullOr(
    Schema.Struct({
      __typename: Schema.Literal("Experiment"),
      runs: Schema.Struct({
        pageInfo: PageInfoSchema,
        edges: Schema.Array(
          Schema.Struct({
            node: Schema.Struct({
              id: Schema.String,
              example: Schema.Struct({
                id: Schema.String,
              }),
              annotations: Schema.Struct({
                edges: Schema.Array(
                  Schema.Struct({
                    node: AnnotationSchema,
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

type GraphqlExperimentRunsResponse = Schema.Schema.Type<typeof ExperimentRunsResponseSchema>;
type GraphqlAnnotation = Schema.Schema.Type<typeof AnnotationSchema>;

type RemapTarget = {
  experimentId: string;
  experimentRunId: string;
  exampleId: string;
  previousAnnotation: GraphqlAnnotation;
  choice: FactualityChoice;
  remapped: RemappedResult;
};

const UPSERT_DONE = Symbol("upsertDone");
type UpsertQueueItem = RemapTarget | typeof UPSERT_DONE;

const EXPERIMENT_RUNS_QUERY = `
  query RemapOpenEndedCorrectnessScores(
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
              id
              example {
                id
              }
              annotations(first: $annotationFirst) {
                edges {
                  node {
                    name
                    label
                    score
                    explanation
                    error
                    metadata
                    startTime
                    endTime
                    traceId
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

const CHOICE_SCORES = {
  A: 0.5,
  B: 1,
  C: 1,
  D: 0,
  E: 1,
} as const satisfies Record<FactualityChoice, FactualityScore>;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const toMetadata = (value: unknown): Record<string, unknown> => (isRecord(value) ? value : {});

const scoreLabel = (score: FactualityScore): RemappedResult["label"] => {
  if (score === 1) return "perfect";
  if (score === 0) return "none";
  return "partial";
};

const remapMetadata = (
  metadata: unknown,
  choice: FactualityChoice,
  score: FactualityScore,
  reason: string,
): Record<string, unknown> => {
  const base = toMetadata(metadata);
  const judgment = base.judgment;
  const scoreRemap = {
    script: "remap-open-ended-correctness-scores",
    version: REMAP_METADATA_VERSION,
    scoringScheme: "three-level-factuality",
    choice,
    score,
  };

  if (!isRecord(judgment)) {
    return {
      ...base,
      judgment: {
        reason,
        choice,
        score,
      },
      [REMAP_METADATA_KEY]: scoreRemap,
    };
  }

  return {
    ...base,
    judgment: {
      ...judgment,
      reason: typeof judgment.reason === "string" ? judgment.reason : reason,
      choice,
      score,
    },
    [REMAP_METADATA_KEY]: scoreRemap,
  };
};

const decodeGlobalTraceId = (traceId: string): string | null => {
  try {
    const decoded = atob(traceId);
    const match = decoded.match(/^Trace:(.+)$/);
    return match?.[1] ?? null;
  } catch {
    return null;
  }
};

const toRawTraceId = (traceId: string | null): string | null => {
  if (traceId === null) return null;
  const onceDecoded = decodeGlobalTraceId(traceId);
  if (onceDecoded === null) return traceId;
  return decodeGlobalTraceId(onceDecoded) ?? onceDecoded;
};

const isDoubleEncodedTraceId = (traceId: string | null): boolean => {
  if (traceId === null) return false;
  const onceDecoded = decodeGlobalTraceId(traceId);
  if (onceDecoded === null) return false;
  return decodeGlobalTraceId(onceDecoded) !== null;
};

const hasCurrentScoreRemapMetadata = (
  annotation: GraphqlAnnotation,
  choice: FactualityChoice,
  score: FactualityScore,
): boolean => {
  const metadata = toMetadata(annotation.metadata);
  const scoreRemap = metadata[REMAP_METADATA_KEY];
  if (!isRecord(scoreRemap)) return false;
  return (
    scoreRemap.script === "remap-open-ended-correctness-scores" &&
    scoreRemap.version === REMAP_METADATA_VERSION &&
    scoreRemap.scoringScheme === "three-level-factuality" &&
    scoreRemap.choice === choice &&
    scoreRemap.score === score
  );
};

const hasCurrentJudgmentMetadata = (
  annotation: GraphqlAnnotation,
  choice: FactualityChoice,
  score: FactualityScore,
): boolean => {
  const judgment = toMetadata(annotation.metadata).judgment;
  if (!isRecord(judgment)) return false;
  return (
    judgment.choice === choice &&
    judgment.score === score &&
    typeof judgment.reason === "string" &&
    judgment.reason.length > 0
  );
};

const parseChoice = (choice: unknown): FactualityChoice | null => {
  if (choice === "A" || choice === "B" || choice === "C" || choice === "D" || choice === "E") {
    return choice;
  }
  return null;
};

const isLegacyScore = (score: number, expected: number): boolean =>
  Math.abs(score - expected) < 1e-9;

const inferChoiceFromLegacyScore = (
  score: number | null,
  explanation: string | null,
): FactualityChoice | null => {
  if (score === null) return null;
  if (isLegacyScore(score, 0.4) || isLegacyScore(score, 0.5)) return "A";
  if (isLegacyScore(score, 0.6)) return "B";
  if (isLegacyScore(score, 0)) return "D";
  if (!isLegacyScore(score, 1)) return null;

  const normalizedExplanation = explanation?.toLowerCase() ?? "";
  if (
    normalizedExplanation.includes("don't matter") ||
    normalizedExplanation.includes("do not matter")
  ) {
    return "E";
  }
  return "C";
};

const getStoredChoice = (annotation: GraphqlAnnotation): FactualityChoice | null => {
  const metadata = toMetadata(annotation.metadata);
  const judgment = metadata.judgment;
  if (isRecord(judgment)) {
    const choice = parseChoice(judgment.choice);
    if (choice !== null) return choice;
  }
  const explanationChoice = annotation.explanation?.match(/\bjudgment=([ABCDE])(?::|$)/)?.[1];
  return (
    parseChoice(explanationChoice) ??
    inferChoiceFromLegacyScore(annotation.score, annotation.explanation)
  );
};

const isChanged = (
  annotation: GraphqlAnnotation,
  choice: FactualityChoice,
  remapped: RemappedResult,
): boolean =>
  annotation.score !== remapped.score ||
  annotation.label !== remapped.label ||
  isDoubleEncodedTraceId(annotation.traceId) ||
  !hasCurrentJudgmentMetadata(annotation, choice, remapped.score) ||
  !hasCurrentScoreRemapMetadata(annotation, choice, remapped.score);

const toRemappedResult = (
  annotation: GraphqlAnnotation,
  choice: FactualityChoice,
): RemappedResult => {
  const score = CHOICE_SCORES[choice];
  return {
    score,
    label: scoreLabel(score),
    explanation: annotation.explanation,
    metadata: remapMetadata(annotation.metadata, choice, score, annotation.explanation ?? ""),
    traceId: toRawTraceId(annotation.traceId),
  };
};

const listRemapTargets = Effect.fn("remapOpenEndedCorrectnessScores.listRemapTargets")(function* (
  phoenix: PhoenixClientImpl,
  args: CliArgs,
  experimentId: string,
) {
  const targets: RemapTarget[] = [];
  let missingChoiceCount = 0;
  let unchangedCount = 0;
  let after: string | undefined;

  while (true) {
    const response: GraphqlExperimentRunsResponse = yield* phoenix.graphql({
      operationName: "RemapOpenEndedCorrectnessScores",
      query: EXPERIMENT_RUNS_QUERY,
      variables: {
        experimentId,
        first: DEFAULT_PAGE_SIZE,
        after,
        annotationFirst: ANNOTATION_PAGE_SIZE,
      },
      schema: ExperimentRunsResponseSchema,
    });

    if (response.node === null) {
      yield* Effect.logWarning(`Experiment ${experimentId}: missing GraphQL node, skipping`);
      return { targets, missingChoiceCount, unchangedCount };
    }

    for (const edge of response.node.runs.edges) {
      const annotation = edge.node.annotations.edges
        .map((annotationEdge) => annotationEdge.node)
        .find((candidate) => candidate.name === args.annotationName);

      if (!annotation || annotation.error !== null) continue;

      const choice = getStoredChoice(annotation);
      if (choice === null) {
        missingChoiceCount += 1;
        continue;
      }

      const remapped = toRemappedResult(annotation, choice);
      if (!isChanged(annotation, choice, remapped)) {
        unchangedCount += 1;
        continue;
      }

      targets.push({
        experimentId,
        experimentRunId: edge.node.id,
        exampleId: edge.node.example.id,
        previousAnnotation: annotation,
        choice,
        remapped,
      });

      if (args.limit !== null && targets.length >= args.limit) {
        return { targets, missingChoiceCount, unchangedCount };
      }
    }

    if (!response.node.runs.pageInfo.hasNextPage || response.node.runs.pageInfo.endCursor === null)
      break;
    after = response.node.runs.pageInfo.endCursor;
  }

  return { targets, missingChoiceCount, unchangedCount };
});

const upsertEvaluationResult = Effect.fn("remapOpenEndedCorrectnessScores.upsertResult")(function* (
  phoenix: PhoenixClientImpl,
  target: RemapTarget,
) {
  yield* phoenix
    .use((client) =>
      client.POST("/v1/experiment_evaluations", {
        body: {
          experiment_run_id: target.experimentRunId,
          name: target.previousAnnotation.name,
          annotator_kind: "CODE",
          start_time: target.previousAnnotation.startTime,
          end_time: target.previousAnnotation.endTime,
          result: {
            score: target.remapped.score,
            label: target.remapped.label,
            explanation: target.remapped.explanation,
          },
          metadata: target.remapped.metadata,
          error: null,
          trace_id: target.remapped.traceId,
        },
      }),
    )
    .pipe(
      Effect.mapError(
        (cause) =>
          new UpsertEvaluationError({
            experimentRunId: target.experimentRunId,
            reason: "Failed to upsert remapped evaluation result",
            cause,
          }),
      ),
    );
});

const runUpsertWorker = Effect.fn("remapOpenEndedCorrectnessScores.runUpsertWorker")(function* (
  phoenix: PhoenixClientImpl,
  queue: Queue.Queue<UpsertQueueItem>,
  updatedCount: Ref.Ref<number>,
  errors: Ref.Ref<UpsertEvaluationError[]>,
  workerIndex: number,
) {
  while (true) {
    const item = yield* Queue.take(queue);
    if (item === UPSERT_DONE) return;

    yield* upsertEvaluationResult(phoenix, item).pipe(
      Effect.tap(() => Ref.update(updatedCount, (count) => count + 1)),
      Effect.catchAll((error) =>
        Effect.gen(function* () {
          yield* Ref.update(errors, (current) => [...current, error]);
          yield* Effect.logError("Failed to upsert remapped evaluation result", {
            workerIndex,
            experimentRunId: item.experimentRunId,
            error,
          });
        }),
      ),
    );
  }
});

const dedupe = <A>(items: readonly A[]): A[] => [...new Set(items)];

const discoverExperimentIds = Effect.fn("remapOpenEndedCorrectnessScores.discoverExperimentIds")(
  function* (phoenix: PhoenixClientImpl, args: CliArgs) {
    if (args.experimentIds.length > 0) return dedupe(args.experimentIds);

    const experimentIdsByDataset = yield* Effect.forEach(
      args.datasetNames,
      (datasetName) =>
        Effect.gen(function* () {
          const dataset = yield* phoenix.use((client) =>
            getDatasetInfo({ client, dataset: { datasetName } }),
          );
          const experiments = yield* phoenix.use((client) =>
            listExperiments({ client, datasetId: dataset.id }),
          );
          yield* Effect.logInfo("Discovered open-ended dataset experiments", {
            datasetName,
            datasetId: dataset.id,
            experiments: experiments.length,
          });
          return experiments.map((experiment) => experiment.id);
        }),
      { concurrency: args.concurrency },
    );

    return dedupe(experimentIdsByDataset.flat());
  },
);

const printDryRun = (
  targetsByExperiment: ReadonlyMap<string, readonly RemapTarget[]>,
  skippedByExperiment: ReadonlyMap<string, { missingChoiceCount: number; unchangedCount: number }>,
): void => {
  let total = 0;
  for (const [experimentId, targets] of targetsByExperiment) {
    total += targets.length;
    const skipped = skippedByExperiment.get(experimentId);
    console.log(`Experiment ${experimentId}: ${targets.length} evaluation update(s)`);
    console.log(
      `  skipped: ${skipped?.unchangedCount ?? 0} unchanged, ${skipped?.missingChoiceCount ?? 0} missing stored judge choice`,
    );
    for (const target of targets.slice(0, 10)) {
      console.log(
        [
          `  - run=${target.experimentRunId}`,
          `example=${target.exampleId}`,
          `choice=${target.choice}`,
          `score=${target.previousAnnotation.score} -> ${target.remapped.score}`,
          `label=${target.previousAnnotation.label ?? "null"} -> ${target.remapped.label}`,
        ].join(" "),
      );
    }
    if (targets.length > 10) {
      console.log(`  ... ${targets.length - 10} more`);
    }
  }
  console.log(`Dry run complete. ${total} evaluation(s) would be upserted. Pass --apply to write.`);
};

const collectTargetsByExperiment = Effect.fn(
  "remapOpenEndedCorrectnessScores.collectTargetsByExperiment",
)(function* (args: CliArgs) {
  const phoenix = yield* PhoenixClientService;
  const targetsByExperiment = new Map<string, RemapTarget[]>();
  const skippedByExperiment = new Map<
    string,
    { missingChoiceCount: number; unchangedCount: number }
  >();
  const experimentIds = yield* discoverExperimentIds(phoenix, args);

  yield* Effect.logInfo("Scanning Phoenix experiments for open-ended correctness annotations", {
    experiments: experimentIds.length,
    concurrency: args.concurrency,
  });

  const scanResults = yield* Progress.forEach(
    experimentIds,
    (experimentId) =>
      listRemapTargets(phoenix, args, experimentId).pipe(
        Effect.map(({ targets, missingChoiceCount, unchangedCount }) => ({
          experimentId,
          targets,
          missingChoiceCount,
          unchangedCount,
        })),
      ),
    {
      description: "Scanning Phoenix experiments",
      concurrency: args.concurrency,
    },
  );

  for (const { experimentId, targets, missingChoiceCount, unchangedCount } of scanResults) {
    targetsByExperiment.set(experimentId, targets);
    skippedByExperiment.set(experimentId, { missingChoiceCount, unchangedCount });
  }

  return { targetsByExperiment, skippedByExperiment };
});

const logRemapConfig = (args: CliArgs) =>
  Effect.logInfo("Preparing open-ended correctness score remap", {
    explicitExperiments: args.experimentIds.length,
    discoveryDatasets: args.experimentIds.length === 0 ? args.datasetNames : [],
    annotationName: args.annotationName,
    apply: args.apply,
    limit: args.limit,
    concurrency: args.concurrency,
    baseUrl: args.baseUrl,
  });

const runDryRemap = (args: CliArgs) =>
  Effect.gen(function* () {
    yield* logRemapConfig(args);
    const { targetsByExperiment, skippedByExperiment } = yield* collectTargetsByExperiment(args);
    printDryRun(targetsByExperiment, skippedByExperiment);
  });

const runApplyRemap = (args: CliArgs) =>
  Effect.scoped(
    Effect.gen(function* () {
      const phoenix = yield* PhoenixClientService;

      yield* logRemapConfig(args);
      const experimentIds = yield* discoverExperimentIds(phoenix, args);
      const queue = yield* Queue.bounded<UpsertQueueItem>(args.concurrency * 4);
      const updatedCount = yield* Ref.make(0);
      const upsertErrors = yield* Ref.make<UpsertEvaluationError[]>([]);

      yield* Effect.logInfo("Scanning Phoenix experiments and remapping matching evaluations", {
        experiments: experimentIds.length,
        scanConcurrency: args.concurrency,
        upsertConcurrency: args.concurrency,
      });

      const workerFibers = yield* Effect.forEach(
        Array.from({ length: args.concurrency }, (_, index) => index),
        (workerIndex) =>
          runUpsertWorker(phoenix, queue, updatedCount, upsertErrors, workerIndex).pipe(
            Effect.forkScoped,
          ),
        { concurrency: "unbounded" },
      );

      const scanResults = yield* Progress.forEach(
        experimentIds,
        (experimentId) =>
          listRemapTargets(phoenix, args, experimentId).pipe(
            Effect.tap(({ targets }) =>
              Effect.forEach(targets, (target) => Queue.offer(queue, target), {
                discard: true,
              }),
            ),
            Effect.map(({ targets, missingChoiceCount, unchangedCount }) => ({
              experimentId,
              targetCount: targets.length,
              missingChoiceCount,
              unchangedCount,
            })),
          ),
        {
          description: "Scanning Phoenix experiments",
          concurrency: args.concurrency,
        },
      );

      yield* Effect.forEach(workerFibers, () => Queue.offer(queue, UPSERT_DONE), {
        discard: true,
        concurrency: "unbounded",
      });
      yield* Fiber.joinAll(workerFibers);

      const failedUpserts = yield* Ref.get(upsertErrors);
      if (failedUpserts.length > 0) {
        yield* Effect.logError("Open-ended correctness score remap finished with upsert failures", {
          failures: failedUpserts.length,
        });
        return yield* failedUpserts[0]!;
      }

      const updated = yield* Ref.get(updatedCount);
      const targetCount = scanResults.reduce((sum, result) => sum + result.targetCount, 0);
      if (targetCount === 0) {
        yield* Effect.logInfo("No matching evaluations to update.");
        return;
      }

      yield* Effect.logInfo("Open-ended correctness score remap complete", {
        updated,
      });
    }),
  );

if (import.meta.main) {
  if (process.argv.slice(2).includes("--help")) {
    console.log(HELP_TEXT);
    process.exit(0);
  }

  Effect.runPromise(
    Effect.gen(function* () {
      const args = yield* parseArgs(process.argv.slice(2));
      const phoenixLayer = PhoenixClientService.layer({ options: { baseUrl: args.baseUrl } });
      const layer = Layer.mergeAll(Logger.pretty, phoenixLayer);

      if (!args.apply) {
        yield* runDryRemap(args).pipe(Effect.provide(layer));
        return;
      }

      yield* runApplyRemap(args).pipe(Effect.provide(layer));
    }),
  ).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
