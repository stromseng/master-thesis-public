// Usage:
//   bun scripts/phoenix/download-experiment-spans.ts
//   bun scripts/phoenix/download-experiment-spans.ts --experiment-id RXhwZXJpbWVudDoyMzQ=
//   bun scripts/phoenix/download-experiment-spans.ts --output data/phoenix/experiment-234-spans.json
//   bun scripts/phoenix/download-experiment-spans.ts --base-url http://127.0.0.1:6006
//
// Downloads every span for every trace attached to a Phoenix experiment run and
// writes one JSON document to disk.
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { Effect, Layer, Logger, Schema } from "effect";
import { PhoenixClient } from "../../src/services/PhoenixClient";
import { dataPath, repoPath } from "../../src/utils/repo";

const DEFAULT_EXPERIMENT_ID = "RXhwZXJpbWVudDoyMzQ=";
const DEFAULT_SPAN_PAGE_SIZE = 100;

type CliArgs = {
  experimentId: string;
  outputPath: string;
  baseUrl?: string;
  spanPageSize: number;
};

class WriteOutputError extends Schema.TaggedError<WriteOutputError>()("WriteOutputError", {
  cause: Schema.Defect,
  outputPath: Schema.String,
}) {}

const parseArgs = (): CliArgs => {
  const argv = process.argv.slice(2);

  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) return undefined;
    return argv[index + 1];
  };

  const experimentId = get("--experiment-id") ?? get("--experiment") ?? DEFAULT_EXPERIMENT_ID;
  const spanPageSizeRaw = get("--span-page-size");
  const spanPageSize = spanPageSizeRaw
    ? Number.parseInt(spanPageSizeRaw, 10)
    : DEFAULT_SPAN_PAGE_SIZE;
  if (!Number.isInteger(spanPageSize) || spanPageSize < 1) {
    throw new Error(`Invalid --span-page-size "${spanPageSizeRaw}". Expected a positive integer.`);
  }

  return {
    experimentId,
    outputPath: get("--output") ?? dataPath("phoenix", "experiment-234-spans.json"),
    baseUrl: get("--base-url"),
    spanPageSize,
  };
};

const PageInfoSchema = Schema.Struct({
  hasNextPage: Schema.Boolean,
  endCursor: Schema.NullOr(Schema.String),
});

const CostBreakdownSchema = Schema.Struct({
  tokens: Schema.NullOr(Schema.Number),
  cost: Schema.NullOr(Schema.Number),
});

const CostSummarySchema = Schema.Struct({
  prompt: CostBreakdownSchema,
  completion: CostBreakdownSchema,
  total: CostBreakdownSchema,
});

const SpanIoValueSchema = Schema.Struct({
  mimeType: Schema.String,
  truncatedValue: Schema.String,
  value: Schema.String,
});

const SpanAnnotationSchema = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  annotatorKind: Schema.String,
  label: Schema.NullOr(Schema.String),
  score: Schema.NullOr(Schema.Number),
  explanation: Schema.NullOr(Schema.String),
  metadata: Schema.Unknown,
  source: Schema.String,
  identifier: Schema.String,
  createdAt: Schema.String,
  updatedAt: Schema.String,
});

const SpanSchema = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  statusCode: Schema.String,
  statusMessage: Schema.String,
  startTime: Schema.String,
  endTime: Schema.NullOr(Schema.String),
  latencyMs: Schema.NullOr(Schema.Number),
  parentId: Schema.NullOr(Schema.String),
  spanKind: Schema.String,
  spanId: Schema.String,
  context: Schema.Struct({
    traceId: Schema.String,
    spanId: Schema.String,
  }),
  attributes: Schema.String,
  metadata: Schema.NullOr(Schema.String),
  numDocuments: Schema.NullOr(Schema.Number),
  tokenCountTotal: Schema.NullOr(Schema.Number),
  tokenCountPrompt: Schema.NullOr(Schema.Number),
  tokenCountCompletion: Schema.NullOr(Schema.Number),
  cumulativeTokenCountTotal: Schema.NullOr(Schema.Number),
  cumulativeTokenCountPrompt: Schema.NullOr(Schema.Number),
  cumulativeTokenCountCompletion: Schema.NullOr(Schema.Number),
  propagatedStatusCode: Schema.String,
  input: Schema.NullOr(SpanIoValueSchema),
  output: Schema.NullOr(SpanIoValueSchema),
  events: Schema.Array(
    Schema.Struct({
      name: Schema.String,
      message: Schema.String,
      timestamp: Schema.String,
      attributes: Schema.Unknown,
    }),
  ),
  spanAnnotations: Schema.Array(SpanAnnotationSchema),
  costSummary: Schema.NullOr(CostSummarySchema),
  costDetailSummaryEntries: Schema.Array(
    Schema.Struct({
      tokenType: Schema.String,
      isPrompt: Schema.Boolean,
      value: CostBreakdownSchema,
    }),
  ),
});

const ExperimentRunsResponseSchema = Schema.Struct({
  node: Schema.NullOr(
    Schema.Struct({
      __typename: Schema.Literal("Experiment"),
      id: Schema.String,
      name: Schema.String,
      runCount: Schema.Number,
      runs: Schema.Struct({
        pageInfo: PageInfoSchema,
        edges: Schema.Array(
          Schema.Struct({
            node: Schema.Struct({
              id: Schema.String,
              repetitionNumber: Schema.Number,
              traceId: Schema.NullOr(Schema.String),
              trace: Schema.NullOr(
                Schema.Struct({
                  id: Schema.String,
                  traceId: Schema.String,
                  projectId: Schema.String,
                  numSpans: Schema.Number,
                }),
              ),
              example: Schema.Struct({
                id: Schema.String,
              }),
            }),
          }),
        ),
      }),
    }),
  ),
});

const TraceSpansResponseSchema = Schema.Struct({
  getTraceByOtelId: Schema.NullOr(
    Schema.Struct({
      id: Schema.String,
      traceId: Schema.String,
      startTime: Schema.String,
      endTime: Schema.String,
      latencyMs: Schema.NullOr(Schema.Number),
      projectId: Schema.String,
      numSpans: Schema.Number,
      spans: Schema.Struct({
        pageInfo: PageInfoSchema,
        edges: Schema.Array(
          Schema.Struct({
            node: SpanSchema,
          }),
        ),
      }),
    }),
  ),
});

type ExperimentRunsResponse = Schema.Schema.Type<typeof ExperimentRunsResponseSchema>;
type ExperimentNode = NonNullable<ExperimentRunsResponse["node"]>;
type ExperimentRunNode = ExperimentNode["runs"]["edges"][number]["node"];
type TraceSpansResponse = Schema.Schema.Type<typeof TraceSpansResponseSchema>;
type TraceNode = NonNullable<
  Schema.Schema.Type<typeof TraceSpansResponseSchema>["getTraceByOtelId"]
>;
type SpanNode = Schema.Schema.Type<typeof SpanSchema>;

type RunTraceReference = {
  runId: string;
  exampleId: string;
  repetitionNumber: number;
  traceId: string;
  projectId: string | null;
  expectedSpanCount: number | null;
};

type DownloadOutput = {
  experimentId: string;
  experimentName: string;
  runCount: number;
  traceCount: number;
  spanCount: number;
  downloadedAt: string;
  traces: Array<
    Omit<TraceNode, "spans"> & {
      runs: RunTraceReference[];
      spans: SpanNode[];
    }
  >;
};

const EXPERIMENT_RUNS_QUERY = `
  query DownloadExperimentSpansRuns($experimentId: ID!, $first: Int!, $after: String) {
    node(id: $experimentId) {
      __typename
      ... on Experiment {
        id
        name
        runCount
        runs(first: $first, after: $after) {
          pageInfo {
            hasNextPage
            endCursor
          }
          edges {
            node {
              id
              repetitionNumber
              traceId
              trace {
                id
                traceId
                projectId
                numSpans
              }
              example {
                id
              }
            }
          }
        }
      }
    }
  }
`;

const TRACE_SPANS_QUERY = `
  query DownloadExperimentTraceSpans($traceId: String!, $first: Int!, $after: String) {
    getTraceByOtelId(traceId: $traceId) {
      id
      traceId
      startTime
      endTime
      latencyMs
      projectId
      numSpans
      spans(first: $first, after: $after, rootSpansOnly: false) {
        pageInfo {
          hasNextPage
          endCursor
        }
        edges {
          node {
            id
            name
            statusCode
            statusMessage
            startTime
            endTime
            latencyMs
            parentId
            spanKind
            spanId
            context {
              traceId
              spanId
            }
            attributes
            metadata
            numDocuments
            tokenCountTotal
            tokenCountPrompt
            tokenCountCompletion
            cumulativeTokenCountTotal
            cumulativeTokenCountPrompt
            cumulativeTokenCountCompletion
            propagatedStatusCode
            input {
              mimeType
              truncatedValue
              value
            }
            output {
              mimeType
              truncatedValue
              value
            }
            events {
              name
              message
              timestamp
              attributes
            }
            spanAnnotations {
              id
              name
              annotatorKind
              label
              score
              explanation
              metadata
              source
              identifier
              createdAt
              updatedAt
            }
            costSummary {
              prompt {
                tokens
                cost
              }
              completion {
                tokens
                cost
              }
              total {
                tokens
                cost
              }
            }
            costDetailSummaryEntries {
              tokenType
              isPrompt
              value {
                tokens
                cost
              }
            }
          }
        }
      }
    }
  }
`;

const fetchExperimentRuns = Effect.fn("downloadExperimentSpans.fetchExperimentRuns")(function* (
  experimentId: string,
) {
  const phoenix = yield* PhoenixClient;
  const runs: ExperimentRunNode[] = [];
  let experimentName = "";
  let runCount = 0;
  let after: string | null = null;

  do {
    const response: ExperimentRunsResponse = yield* phoenix.graphql({
      operationName: "DownloadExperimentSpansRuns",
      query: EXPERIMENT_RUNS_QUERY,
      schema: ExperimentRunsResponseSchema,
      variables: {
        experimentId,
        first: 100,
        after,
      },
    });

    if (response.node === null) {
      throw new Error(`Phoenix experiment ${experimentId} was not found`);
    }

    experimentName = response.node.name;
    runCount = response.node.runCount;
    runs.push(...response.node.runs.edges.map((edge) => edge.node));
    after = response.node.runs.pageInfo.endCursor;
    if (!response.node.runs.pageInfo.hasNextPage) {
      after = null;
    }
  } while (after !== null);

  return { experimentName, runCount, runs };
});

const fetchTraceSpans = Effect.fn("downloadExperimentSpans.fetchTraceSpans")(function* (
  traceId: string,
  pageSize: number,
) {
  const phoenix = yield* PhoenixClient;
  let trace: Omit<TraceNode, "spans"> | null = null;
  const spans: SpanNode[] = [];
  let after: string | null = null;

  do {
    const response: TraceSpansResponse = yield* phoenix.graphql({
      operationName: "DownloadExperimentTraceSpans",
      query: TRACE_SPANS_QUERY,
      schema: TraceSpansResponseSchema,
      variables: {
        traceId,
        first: pageSize,
        after,
      },
    });

    if (response.getTraceByOtelId === null) {
      throw new Error(`Phoenix trace ${traceId} was not found`);
    }

    const { spans: page, ...traceFields } = response.getTraceByOtelId;
    trace = traceFields;
    spans.push(...page.edges.map((edge) => edge.node));
    after = page.pageInfo.endCursor;
    if (!page.pageInfo.hasNextPage) {
      after = null;
    }
  } while (after !== null);

  if (trace === null) {
    throw new Error(`Phoenix trace ${traceId} returned no data`);
  }

  return { trace, spans };
});

const toTraceReferences = (runs: readonly ExperimentRunNode[]): RunTraceReference[] =>
  runs.flatMap((run) => {
    const traceId = run.trace?.traceId ?? run.traceId;
    if (!traceId) return [];
    return [
      {
        runId: run.id,
        exampleId: run.example.id,
        repetitionNumber: run.repetitionNumber,
        traceId,
        projectId: run.trace?.projectId ?? null,
        expectedSpanCount: run.trace?.numSpans ?? null,
      },
    ];
  });

const writeJson = Effect.fn("downloadExperimentSpans.writeJson")(function* (
  outputPath: string,
  output: DownloadOutput,
) {
  mkdirSync(dirname(outputPath), { recursive: true });
  yield* Effect.tryPromise({
    try: () => Bun.write(outputPath, `${JSON.stringify(output, null, 2)}\n`),
    catch: (cause) => new WriteOutputError({ cause, outputPath }),
  });
});

const args = parseArgs();
const outputPath = args.outputPath.startsWith("/") ? args.outputPath : repoPath(args.outputPath);
const layer = args.baseUrl
  ? PhoenixClient.layer({ options: { baseUrl: args.baseUrl } })
  : PhoenixClient.skyhigh;
const appLayer = Layer.mergeAll(layer, Logger.pretty);

const program = Effect.gen(function* () {
  const { experimentName, runCount, runs } = yield* fetchExperimentRuns(args.experimentId);
  const traceReferences = toTraceReferences(runs);
  const referencesByTraceId = Map.groupBy(traceReferences, (reference) => reference.traceId);
  const traces: DownloadOutput["traces"] = [];

  for (const [traceId, references] of referencesByTraceId) {
    yield* Effect.logInfo(`Downloading trace ${traceId}`, {
      runCount: references.length,
    });
    const { trace, spans } = yield* fetchTraceSpans(traceId, args.spanPageSize);
    traces.push({
      ...trace,
      runs: references,
      spans,
    });
  }

  const output: DownloadOutput = {
    experimentId: args.experimentId,
    experimentName,
    runCount,
    traceCount: traces.length,
    spanCount: traces.reduce((sum, trace) => sum + trace.spans.length, 0),
    downloadedAt: new Date().toISOString(),
    traces,
  };

  yield* writeJson(outputPath, output);
  return output;
});

Effect.runPromise(program.pipe(Effect.provide(appLayer))).then(
  (output) => {
    console.error(
      `Wrote ${output.spanCount} spans from ${output.traceCount} traces to ${outputPath}`,
    );
  },
  (error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Failed to download Phoenix experiment spans: ${message}`);
    process.exitCode = 1;
  },
);
