// Usage:
//   bun scripts/analysis/fetch-open-ended-score-histogram.ts
//   bun scripts/analysis/fetch-open-ended-score-histogram.ts --aggregate ../../typst/data/open-ended-aggregate.json
//   bun scripts/analysis/fetch-open-ended-score-histogram.ts --output ../../typst/data/open-ended-score-histogram.json
//   bun scripts/analysis/fetch-open-ended-score-histogram.ts --base-url http://127.0.0.1:6006
//
// Fetches per-run open-ended correctness scores for the experiments selected
// by the aggregate report and writes score-bin counts for Typst figures.
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Effect, Logger, Schema } from "effect";
import { PhoenixClient, type PhoenixClientImpl } from "../../src/services/PhoenixClient";
import { repoPath } from "../../src/utils/repo";

type CliArgs = {
  aggregatePath: string;
  annotationName: string;
  outputPath: string;
  concurrency: number;
  baseUrl?: string;
};

type AggregateExperiment = {
  experimentId: string;
  datasetName: string;
};

type ScoreBins = {
  zero: number;
  partial: number;
  one: number;
  other: number;
  missing: number;
};

type ExperimentHistogram = ScoreBins & {
  experimentId: string;
  datasetName: string;
  scored: number;
  total: number;
};

type ModelHistogram = ScoreBins & {
  modelId: string;
  weightedMeanScore: number | null;
  scored: number;
  total: number;
  experiments: readonly ExperimentHistogram[];
};

class ReadAggregateError extends Schema.TaggedError<ReadAggregateError>()("ReadAggregateError", {
  aggregatePath: Schema.String,
  cause: Schema.Defect,
}) {}

class WriteOutputError extends Schema.TaggedError<WriteOutputError>()("WriteOutputError", {
  outputPath: Schema.String,
  cause: Schema.Defect,
}) {}

const PAGE_SIZE = 1000;
const ANNOTATION_PAGE_SIZE = 20;

/** Model IDs that are merged into a canonical alias for grouping. */
const MODEL_ID_ALIASES: Record<string, string> = {
  "Qwen/Qwen3.5-27B-FP8": "Qwen/Qwen3.5-27B",
};

const normalizeModelId = (modelId: string): string => MODEL_ID_ALIASES[modelId] ?? modelId;

type MergedGroup = {
  modelId: string;
  weightedMeanScore: number | null;
  experiments: readonly AggregateExperiment[];
};

const mergeGroupsByNormalizedModelId = (
  groups: readonly {
    modelId: string;
    weightedMeanScore: number | null;
    experiments: readonly AggregateExperiment[];
  }[],
): MergedGroup[] => {
  const byNormalizedId = new Map<
    string,
    { modelId: string; scores: number[]; experiments: AggregateExperiment[] }
  >();

  for (const group of groups) {
    const normalizedId = normalizeModelId(group.modelId);
    let entry = byNormalizedId.get(normalizedId);
    if (!entry) {
      entry = { modelId: normalizedId, scores: [], experiments: [] };
      byNormalizedId.set(normalizedId, entry);
    }
    if (group.weightedMeanScore !== null) {
      entry.scores.push(group.weightedMeanScore);
    }
    for (const experiment of group.experiments) {
      entry.experiments.push(experiment);
    }
  }

  return [...byNormalizedId.values()].map((entry) => ({
    modelId: entry.modelId,
    weightedMeanScore:
      entry.scores.length > 0
        ? entry.scores.reduce((sum, value) => sum + value, 0) / entry.scores.length
        : null,
    experiments: entry.experiments,
  }));
};

const AggregateExperimentSchema = Schema.Struct({
  experimentId: Schema.String,
  datasetName: Schema.String,
});

const AggregateGroupSchema = Schema.Struct({
  modelId: Schema.String,
  weightedMeanScore: Schema.NullOr(Schema.Number),
  experiments: Schema.Array(AggregateExperimentSchema),
});

const AggregateReportSchema = Schema.Struct({
  generatedAt: Schema.optional(Schema.String),
  annotationName: Schema.optional(Schema.String),
  globalGroups: Schema.Array(AggregateGroupSchema),
});

const PageInfoSchema = Schema.Struct({
  hasNextPage: Schema.Boolean,
  endCursor: Schema.NullOr(Schema.String),
});

const AnnotationSchema = Schema.Struct({
  name: Schema.String,
  score: Schema.NullOr(Schema.Number),
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

type ExperimentRunsResponse = Schema.Schema.Type<typeof ExperimentRunsResponseSchema>;

const EXPERIMENT_RUNS_QUERY = `
  query OpenEndedScoreHistogramRuns(
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

const parseArgs = (): CliArgs => {
  const argv = process.argv.slice(2);
  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) return undefined;
    return argv[index + 1];
  };

  const concurrency = Number(get("--concurrency") ?? "6");
  if (!Number.isFinite(concurrency) || concurrency < 1) {
    throw new Error("Invalid --concurrency value. Expected a positive number.");
  }

  return {
    aggregatePath: resolve(
      get("--aggregate") ?? repoPath("typst", "data", "open-ended-aggregate.json"),
    ),
    annotationName: get("--annotation") ?? "open-ended-correctness",
    outputPath: resolve(
      get("--output") ?? repoPath("typst", "data", "open-ended-score-histogram.json"),
    ),
    concurrency,
    baseUrl: get("--base-url"),
  };
};

const emptyBins = (): ScoreBins => ({
  zero: 0,
  partial: 0,
  one: 0,
  other: 0,
  missing: 0,
});

const addBins = (target: ScoreBins, source: ScoreBins): void => {
  target.zero += source.zero;
  target.partial += source.partial;
  target.one += source.one;
  target.other += source.other;
  target.missing += source.missing;
};

const isScore = (value: number, expected: number): boolean => Math.abs(value - expected) < 1e-9;

const recordScore = (bins: ScoreBins, score: number | null): void => {
  if (score === null) {
    bins.missing += 1;
  } else if (isScore(score, 0)) {
    bins.zero += 1;
  } else if (isScore(score, 0.5)) {
    bins.partial += 1;
  } else if (isScore(score, 1)) {
    bins.one += 1;
  } else {
    bins.other += 1;
  }
};

const scoredCount = (bins: ScoreBins): number => bins.zero + bins.partial + bins.one + bins.other;

const readAggregateReport = Effect.fn("readAggregateReport")(function* (path: string) {
  const raw = yield* Effect.try({
    try: () => JSON.parse(readFileSync(path, "utf8")) as unknown,
    catch: (cause) => new ReadAggregateError({ aggregatePath: path, cause }),
  });

  return yield* Schema.decodeUnknown(AggregateReportSchema)(raw);
});

const fetchExperimentHistogram = Effect.fn("fetchOpenEndedHistogram.fetchExperimentHistogram")(
  function* (phoenix: PhoenixClientImpl, experiment: AggregateExperiment, annotationName: string) {
    const bins = emptyBins();
    let after: string | undefined;

    while (true) {
      const response: ExperimentRunsResponse = yield* phoenix.graphql({
        operationName: "OpenEndedScoreHistogramRuns",
        query: EXPERIMENT_RUNS_QUERY,
        variables: {
          experimentId: experiment.experimentId,
          first: PAGE_SIZE,
          after,
          annotationFirst: ANNOTATION_PAGE_SIZE,
        },
        schema: ExperimentRunsResponseSchema,
      });

      if (response.node === null) {
        yield* Effect.logWarning(
          `Experiment ${experiment.experimentId}: missing GraphQL node, skipping`,
        );
        break;
      }

      for (const edge of response.node.runs.edges) {
        const annotation = edge.node.annotations.edges
          .map((annotationEdge) => annotationEdge.node)
          .find((candidate) => candidate.name === annotationName);
        recordScore(bins, annotation && !annotation.error ? annotation.score : null);
      }

      if (
        !response.node.runs.pageInfo.hasNextPage ||
        response.node.runs.pageInfo.endCursor === null
      ) {
        break;
      }
      after = response.node.runs.pageInfo.endCursor;
    }

    return {
      experimentId: experiment.experimentId,
      datasetName: experiment.datasetName,
      ...bins,
      scored: scoredCount(bins),
      total: scoredCount(bins) + bins.missing,
    };
  },
);

const writeOutput = Effect.fn("writeOpenEndedHistogram.writeOutput")(function* (
  outputPath: string,
  output: unknown,
) {
  mkdirSync(dirname(outputPath), { recursive: true });
  yield* Effect.tryPromise({
    try: () => Bun.write(outputPath, `${JSON.stringify(output, null, 2)}\n`),
    catch: (cause) => new WriteOutputError({ outputPath, cause }),
  });
});

const args = parseArgs();

const program = Effect.gen(function* () {
  const phoenix = yield* PhoenixClient;
  const aggregate = yield* readAggregateReport(args.aggregatePath);

  yield* Effect.log(`Reading selected experiments from ${args.aggregatePath}`);
  yield* Effect.log(
    `Fetching "${args.annotationName}" score bins with concurrency ${args.concurrency}`,
  );

  const mergedGroups = mergeGroupsByNormalizedModelId(aggregate.globalGroups);

  const models = yield* Effect.forEach(
    mergedGroups,
    (group) =>
      Effect.gen(function* () {
        const experiments = yield* Effect.forEach(
          group.experiments,
          (experiment) =>
            Effect.gen(function* () {
              const histogram = yield* fetchExperimentHistogram(
                phoenix,
                experiment,
                args.annotationName,
              );
              yield* Effect.log(
                `[${group.modelId}] ${experiment.datasetName}: ${histogram.scored} scored, ${histogram.missing} missing`,
              );
              return histogram;
            }),
          { concurrency: args.concurrency },
        );

        const bins = emptyBins();
        for (const experiment of experiments) {
          addBins(bins, experiment);
        }

        return {
          modelId: group.modelId,
          weightedMeanScore: group.weightedMeanScore,
          ...bins,
          scored: scoredCount(bins),
          total: scoredCount(bins) + bins.missing,
          experiments: experiments.sort((a, b) => a.datasetName.localeCompare(b.datasetName)),
        } satisfies ModelHistogram;
      }),
    { concurrency: Math.max(1, Math.floor(args.concurrency / 2)) },
  );

  const output = {
    generatedAt: new Date().toISOString(),
    sourceAggregatePath: args.aggregatePath,
    sourceAggregateGeneratedAt: aggregate.generatedAt ?? null,
    annotationName: args.annotationName,
    scoreBins: [
      { key: "zero", score: 0, label: "0" },
      { key: "partial", score: 0.5, label: "0.5" },
      { key: "one", score: 1, label: "1" },
    ],
    models: models.sort((a, b) => {
      const left = a.weightedMeanScore ?? -1;
      const right = b.weightedMeanScore ?? -1;
      if (left !== right) return right - left;
      return a.modelId.localeCompare(b.modelId);
    }),
  };

  yield* writeOutput(args.outputPath, output);
  yield* Effect.log(`Wrote open-ended score histogram data to ${args.outputPath}`);
});

const layer = args.baseUrl
  ? PhoenixClient.layer({ options: { baseUrl: args.baseUrl } })
  : PhoenixClient.skyhigh;

Effect.runPromise(program.pipe(Effect.provide(layer), Effect.provide(Logger.pretty))).catch(
  (error) => {
    console.error(error);
    process.exitCode = 1;
  },
);
