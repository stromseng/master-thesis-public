#!/usr/bin/env bun
// Repair open-ended correctness evaluations that were recorded as successful
// score-zero "error" labels instead of Phoenix evaluation errors.
//
// Usage:
//   bun scripts/phoenix/repair-open-ended-evaluations.ts --experiment-id <id>
//   bun scripts/phoenix/repair-open-ended-evaluations.ts --experiment-id <id> --apply
//   bun scripts/phoenix/repair-open-ended-evaluations.ts --experiment-id <id> --limit 10 --apply
//   bun scripts/phoenix/repair-open-ended-evaluations.ts --experiment-id <id> --base-url http://127.0.0.1:6006

import { getExperimentInfo } from "@arizeai/phoenix-client/experiments";
import type {
  EvaluationResult,
  Evaluator,
  EvaluatorParams,
} from "@arizeai/phoenix-client/types/experiments";
import * as Progress from "effective-progress";
import { Either, Effect, Layer, Logger, Schema } from "effect";
import { makeOpenEndedCorrectnessEvaluator } from "../../evals/open-ended-evaluator";
import { JudgeLayer } from "../../src/services/Judge";
import { JudgeLanguageModelLayer } from "../../src/services/LanguageModel";
import {
  PHOENIX_SKYHIGH_BASE_URL,
  PhoenixClient as PhoenixClientService,
  type PhoenixClientImpl,
} from "../../src/services/PhoenixClient";

const DEFAULT_ANNOTATION_NAME = "open-ended-correctness";
const DEFAULT_LABEL = "error";
const DEFAULT_EXPLANATION_CONTAINS = "Failed to generate factuality judgment";
const DEFAULT_CONCURRENCY = 2;
const DEFAULT_PAGE_SIZE = 100;
const ANNOTATION_PAGE_SIZE = 100;

type CliArgs = {
  experimentIds: string[];
  annotationName: string;
  label: string;
  explanationContains: string | null;
  apply: boolean;
  limit: number | null;
  concurrency: number;
  baseUrl: string;
};

class CliArgsError extends Schema.TaggedError<CliArgsError>()("CliArgsError", {
  reason: Schema.String,
}) {}

class ReevaluateTargetError extends Schema.TaggedError<ReevaluateTargetError>()(
  "ReevaluateTargetError",
  {
    experimentRunId: Schema.String,
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

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

const parseArgs = Effect.fn("repairOpenEndedEvaluations.parseArgs")(function* (
  argv: readonly string[],
) {
  const experimentIds: string[] = [];
  let annotationName = DEFAULT_ANNOTATION_NAME;
  let label = DEFAULT_LABEL;
  let explanationContains: string | null = DEFAULT_EXPLANATION_CONTAINS;
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
      case "--annotation": {
        annotationName = yield* readValue(i, arg);
        i += 1;
        break;
      }
      case "--label": {
        label = yield* readValue(i, arg);
        i += 1;
        break;
      }
      case "--explanation-contains": {
        explanationContains = yield* readValue(i, arg);
        i += 1;
        break;
      }
      case "--include-all-error-labels": {
        explanationContains = null;
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
          reason: [
            "Usage:",
            "  bun scripts/phoenix/repair-open-ended-evaluations.ts --experiment-id <id> [--apply]",
            "",
            "Options:",
            "  --experiment-id <id>             Experiment to repair. Repeatable.",
            "  --annotation <name>              Annotation name. Default: open-ended-correctness.",
            "  --label <label>                  Label to repair. Default: error.",
            "  --explanation-contains <text>    Explanation substring to match.",
            "  --include-all-error-labels       Ignore explanation substring filtering.",
            "  --limit <n>                      Maximum matching runs per experiment.",
            "  --concurrency <n>                Concurrent judge calls. Default: 2.",
            "  --base-url <url>                 Phoenix base URL.",
            "  --apply                          Write repaired evaluations. Omit for dry-run.",
          ].join("\n"),
        });
      }
      default:
        return yield* new CliArgsError({ reason: `Unknown argument '${arg}'.` });
    }
  }

  if (experimentIds.length === 0) {
    return yield* new CliArgsError({ reason: "Missing required --experiment-id <id>." });
  }

  return {
    experimentIds,
    annotationName,
    label,
    explanationContains,
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
              output: Schema.NullOr(Schema.Unknown),
              error: Schema.NullOr(Schema.String),
              example: Schema.Struct({
                id: Schema.String,
                revision: Schema.Struct({
                  input: Schema.Unknown,
                  output: Schema.Unknown,
                  metadata: Schema.Unknown,
                }),
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

type RepairTarget = {
  experimentId: string;
  experimentRunId: string;
  exampleId: string;
  input: unknown;
  output: unknown;
  expected: unknown;
  metadata: Record<string, unknown>;
  previousAnnotation: GraphqlAnnotation;
};

const EXPERIMENT_RUNS_QUERY = `
  query RepairOpenEndedExperimentRuns(
    $experimentId: ID!
    $datasetVersionId: ID!
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
              output
              error
              example {
                id
                revision(datasetVersionId: $datasetVersionId) {
                  input
                  output
                  metadata
                }
              }
              annotations(first: $annotationFirst) {
                edges {
                  node {
                    name
                    label
                    score
                    explanation
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

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const toMetadata = (value: unknown): Record<string, unknown> => (isRecord(value) ? value : {});

const getErrorText = (error: unknown): string => {
  if (typeof error === "object" && error !== null && "reason" in error) {
    return String((error as { reason: unknown }).reason);
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

const matchesRepairFilter = (annotation: GraphqlAnnotation, args: CliArgs): boolean => {
  if (annotation.name !== args.annotationName) return false;
  if (annotation.label !== args.label) return false;
  if (args.explanationContains === null) return true;
  return annotation.explanation?.includes(args.explanationContains) ?? false;
};

const listRepairTargets = Effect.fn("repairOpenEndedEvaluations.listRepairTargets")(function* (
  phoenix: PhoenixClientImpl,
  args: CliArgs,
  experimentId: string,
) {
  const info = yield* phoenix.use((client) => getExperimentInfo({ client, experimentId }));
  const targets: RepairTarget[] = [];
  let after: string | undefined;

  while (true) {
    const response: GraphqlExperimentRunsResponse = yield* phoenix.graphql({
      operationName: "RepairOpenEndedExperimentRuns",
      query: EXPERIMENT_RUNS_QUERY,
      variables: {
        experimentId,
        datasetVersionId: info.datasetVersionId,
        first: DEFAULT_PAGE_SIZE,
        after,
        annotationFirst: ANNOTATION_PAGE_SIZE,
      },
      schema: ExperimentRunsResponseSchema,
    });

    if (response.node === null) {
      yield* Effect.logWarning(`Experiment ${experimentId}: missing GraphQL node, skipping`);
      return targets;
    }

    for (const edge of response.node.runs.edges) {
      const annotation = edge.node.annotations.edges
        .map((annotationEdge) => annotationEdge.node)
        .find((candidate) => matchesRepairFilter(candidate, args));

      if (!annotation) continue;

      targets.push({
        experimentId,
        experimentRunId: edge.node.id,
        exampleId: edge.node.example.id,
        input: edge.node.example.revision.input,
        output: edge.node.output,
        expected: edge.node.example.revision.output,
        metadata: toMetadata(edge.node.example.revision.metadata),
        previousAnnotation: annotation,
      });

      if (args.limit !== null && targets.length >= args.limit) {
        return targets;
      }
    }

    if (!response.node.runs.pageInfo.hasNextPage || response.node.runs.pageInfo.endCursor === null)
      break;
    after = response.node.runs.pageInfo.endCursor;
  }

  return targets;
});

const upsertEvaluationResult = Effect.fn("repairOpenEndedEvaluations.upsertResult")(function* (
  phoenix: PhoenixClientImpl,
  target: RepairTarget,
  result: EvaluationResult,
  startTime: Date,
  endTime: Date,
) {
  yield* phoenix
    .use((client) =>
      client.POST("/v1/experiment_evaluations", {
        body: {
          experiment_run_id: target.experimentRunId,
          name: target.previousAnnotation.name,
          annotator_kind: "CODE",
          start_time: startTime.toISOString(),
          end_time: endTime.toISOString(),
          result: {
            score: result.score ?? null,
            label: result.label ?? null,
            explanation: result.explanation ?? null,
          },
          metadata: result.metadata ?? {},
          error: null,
          trace_id: null,
        },
      }),
    )
    .pipe(
      Effect.mapError(
        (cause) =>
          new UpsertEvaluationError({
            experimentRunId: target.experimentRunId,
            reason: "Failed to upsert repaired evaluation result",
            cause,
          }),
      ),
    );
});

const upsertEvaluationError = Effect.fn("repairOpenEndedEvaluations.upsertError")(function* (
  phoenix: PhoenixClientImpl,
  target: RepairTarget,
  error: string,
  startTime: Date,
  endTime: Date,
) {
  yield* phoenix
    .use((client) =>
      client.POST("/v1/experiment_evaluations", {
        body: {
          experiment_run_id: target.experimentRunId,
          name: target.previousAnnotation.name,
          annotator_kind: "CODE",
          start_time: startTime.toISOString(),
          end_time: endTime.toISOString(),
          result: null,
          metadata: {},
          error,
          trace_id: null,
        },
      }),
    )
    .pipe(
      Effect.mapError(
        (cause) =>
          new UpsertEvaluationError({
            experimentRunId: target.experimentRunId,
            reason: "Failed to upsert failed evaluation result",
            cause,
          }),
      ),
    );
});

const repairTarget = Effect.fn("repairOpenEndedEvaluations.repairTarget")(function* (
  phoenix: PhoenixClientImpl,
  evaluator: Evaluator,
  target: RepairTarget,
) {
  const startTime = new Date();
  const evaluation = yield* Effect.tryPromise({
    try: () =>
      Promise.resolve(
        evaluator.evaluate({
          input: target.input as EvaluatorParams["input"],
          output: target.output as EvaluatorParams["output"],
          expected: target.expected as EvaluatorParams["expected"],
          metadata: target.metadata,
        }),
      ),
    catch: (cause) =>
      new ReevaluateTargetError({
        experimentRunId: target.experimentRunId,
        reason: "Open-ended correctness evaluator failed during repair",
        cause,
      }),
  }).pipe(Effect.either);
  const endTime = new Date();

  if (Either.isRight(evaluation)) {
    yield* upsertEvaluationResult(phoenix, target, evaluation.right, startTime, endTime);
    return {
      status: "repaired",
      target,
      result: evaluation.right,
    };
  }

  const error = getErrorText(evaluation.left);
  yield* upsertEvaluationError(phoenix, target, error, startTime, endTime);
  return {
    status: "failed",
    target,
    error,
  };
});

const printDryRun = (targetsByExperiment: ReadonlyMap<string, readonly RepairTarget[]>): void => {
  let total = 0;
  for (const [experimentId, targets] of targetsByExperiment) {
    total += targets.length;
    console.log(`Experiment ${experimentId}: ${targets.length} matching evaluation(s)`);
    for (const target of targets.slice(0, 10)) {
      console.log(
        `  - run=${target.experimentRunId} example=${target.exampleId} score=${target.previousAnnotation.score} explanation=${target.previousAnnotation.explanation ?? ""}`,
      );
    }
    if (targets.length > 10) {
      console.log(`  ... ${targets.length - 10} more`);
    }
  }
  console.log(`Dry run complete. ${total} evaluation(s) would be repaired. Pass --apply to write.`);
};

const appModelLayer = Layer.mergeAll(
  JudgeLayer.pipe(Layer.provide(JudgeLanguageModelLayer)),
  Logger.pretty,
);

const collectTargetsByExperiment = Effect.fn(
  "repairOpenEndedEvaluations.collectTargetsByExperiment",
)(function* (args: CliArgs) {
  const phoenix = yield* PhoenixClientService;
  const targetsByExperiment = new Map<string, RepairTarget[]>();

  for (const experimentId of args.experimentIds) {
    const targets = yield* listRepairTargets(phoenix, args, experimentId);
    targetsByExperiment.set(experimentId, targets);
  }

  return targetsByExperiment;
});

const logRepairConfig = (args: CliArgs) =>
  Effect.logInfo("Preparing open-ended evaluation repair", {
    experiments: args.experimentIds.length,
    annotationName: args.annotationName,
    label: args.label,
    explanationContains: args.explanationContains,
    apply: args.apply,
    limit: args.limit,
    concurrency: args.concurrency,
    baseUrl: args.baseUrl,
  });

const runDryRepair = (args: CliArgs) =>
  Effect.gen(function* () {
    yield* logRepairConfig(args);
    const targetsByExperiment = yield* collectTargetsByExperiment(args);
    printDryRun(targetsByExperiment);
  });

const runApplyRepair = (args: CliArgs) =>
  Effect.gen(function* () {
    const phoenix = yield* PhoenixClientService;
    const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof appModelLayer>>();
    const evaluator = makeOpenEndedCorrectnessEvaluator(runtime);

    yield* logRepairConfig(args);
    const targetsByExperiment = yield* collectTargetsByExperiment(args);
    const allTargets = [...targetsByExperiment.values()].flat();
    if (allTargets.length === 0) {
      yield* Effect.logInfo("No matching evaluations to repair.");
      return;
    }

    const outcomes = yield* Progress.forEach(
      allTargets,
      (target) => repairTarget(phoenix, evaluator, target),
      {
        description: "Repairing open-ended evaluations",
        concurrency: args.concurrency,
      },
    );

    const repaired = outcomes.filter((outcome) => outcome.status === "repaired");
    const failed = outcomes.filter((outcome) => outcome.status === "failed");

    yield* Effect.logInfo("Open-ended evaluation repair complete", {
      repaired: repaired.length,
      failed: failed.length,
      total: outcomes.length,
    });

    if (failed.length > 0) {
      for (const outcome of failed) {
        yield* Effect.logWarning("Repair evaluation failed and was recorded as Phoenix error", {
          experimentRunId: outcome.target.experimentRunId,
          exampleId: outcome.target.exampleId,
          error: outcome.error,
        });
      }
    }
  });

if (import.meta.main) {
  Effect.runPromise(
    Effect.gen(function* () {
      const args = yield* parseArgs(process.argv.slice(2));
      const phoenixLayer = PhoenixClientService.layer({ options: { baseUrl: args.baseUrl } });

      if (!args.apply) {
        yield* runDryRepair(args).pipe(Effect.provide(phoenixLayer));
        return;
      }

      yield* runApplyRepair(args).pipe(Effect.provide(Layer.mergeAll(appModelLayer, phoenixLayer)));
    }),
  ).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
