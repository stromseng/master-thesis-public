// Usage:
//   bun scripts/fetch-experiment-scores.ts --experiment-id RXhwZXJpbWVudDo1OA==
//   bun scripts/fetch-experiment-scores.ts --experiment-id RXhw... --annotation question-f1
//   bun scripts/fetch-experiment-scores.ts --experiment-id RXhw... --format csv
//   bun scripts/fetch-experiment-scores.ts --experiment-id RXhw... --base-url http://127.0.0.1:6006
//
// Fetches per-run evaluation scores from a Phoenix experiment export.
//
// Output:
// - Summary is written to stderr
// - Rows are written to stdout as JSON array (default) or CSV
import { Effect } from "effect";
import { PhoenixClient } from "../../src/services/PhoenixClient";

type CliArgs = {
  experimentId: string;
  annotationName?: string;
  format: "json" | "csv";
  includeNullScores: boolean;
  baseUrl?: string;
};

type ExperimentAnnotation = {
  name?: string;
  annotator_kind?: string;
  label?: string | null;
  score?: number | null;
  explanation?: string | null;
  trace_id?: string | null;
  error?: string | null;
  start_time?: string;
  end_time?: string;
};

type ExperimentRun = {
  example_id?: string;
  repetition_number?: number;
  trace_id?: string | null;
  error?: string | null;
  annotations?: ExperimentAnnotation[];
};

type ScoreRow = {
  example_id: string;
  repetition_number: number;
  annotation_name: string;
  annotator_kind: string | null;
  score: number | null;
  label: string | null;
  explanation: string | null;
  annotation_trace_id: string | null;
  run_trace_id: string | null;
  annotation_error: string | null;
  run_error: string | null;
};

const parseArgs = (): CliArgs => {
  const argv = process.argv.slice(2);

  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) return undefined;
    return argv[index + 1];
  };

  const has = (name: string): boolean => argv.includes(name);

  const experimentId = get("--experiment-id") ?? get("--experiment");
  if (!experimentId) {
    throw new Error("Missing required --experiment-id <id>");
  }

  const formatRaw = get("--format") ?? "json";
  if (formatRaw !== "json" && formatRaw !== "csv") {
    throw new Error(`Invalid --format "${formatRaw}". Expected "json" or "csv".`);
  }

  return {
    experimentId,
    annotationName: get("--annotation"),
    format: formatRaw,
    includeNullScores: has("--include-null-scores"),
    baseUrl: get("--base-url"),
  };
};

const fetchExperimentRuns = Effect.fn("fetchExperimentRuns")(function* (experimentId: string) {
  const phoenix = yield* PhoenixClient;
  const response = yield* phoenix.use((client) =>
    client.GET("/v1/experiments/{experiment_id}/json", {
      params: { path: { experiment_id: experimentId } },
    }),
  );

  if (response.error) {
    throw new Error(`Phoenix API error while fetching experiment ${experimentId}`);
  }

  const raw = response.data as unknown;
  const parsed = typeof raw === "string" ? (JSON.parse(raw) as unknown) : raw;

  if (!Array.isArray(parsed)) {
    throw new Error(`Unexpected /v1/experiments/{id}/json shape for experiment ${experimentId}`);
  }

  return parsed as ExperimentRun[];
});

const toRows = (
  runs: readonly ExperimentRun[],
  opts: { annotationName?: string; includeNullScores: boolean },
): ScoreRow[] => {
  const rows: ScoreRow[] = [];

  for (const run of runs) {
    const annotations = Array.isArray(run.annotations) ? run.annotations : [];
    for (const annotation of annotations) {
      const name = annotation.name ?? "";
      if (opts.annotationName && name !== opts.annotationName) {
        continue;
      }

      const score = annotation.score ?? null;
      if (!opts.includeNullScores && score === null) {
        continue;
      }

      rows.push({
        example_id: run.example_id ?? "",
        repetition_number: run.repetition_number ?? 0,
        annotation_name: name,
        annotator_kind: annotation.annotator_kind ?? null,
        score,
        label: annotation.label ?? null,
        explanation: annotation.explanation ?? null,
        annotation_trace_id: annotation.trace_id ?? null,
        run_trace_id: run.trace_id ?? null,
        annotation_error: annotation.error ?? null,
        run_error: run.error ?? null,
      });
    }
  }

  return rows;
};

const printSummary = (rows: readonly ScoreRow[]): void => {
  const grouped = new Map<
    string,
    {
      count: number;
      nonNullCount: number;
      nullCount: number;
      sum: number;
      min: number;
      max: number;
    }
  >();

  for (const row of rows) {
    const key = row.annotation_name || "(unnamed)";
    const current = grouped.get(key) ?? {
      count: 0,
      nonNullCount: 0,
      nullCount: 0,
      sum: 0,
      min: Number.POSITIVE_INFINITY,
      max: Number.NEGATIVE_INFINITY,
    };

    current.count += 1;
    if (row.score === null) {
      current.nullCount += 1;
    } else {
      current.nonNullCount += 1;
      current.sum += row.score;
      if (row.score < current.min) current.min = row.score;
      if (row.score > current.max) current.max = row.score;
    }
    grouped.set(key, current);
  }

  console.error(`Rows: ${rows.length}`);
  for (const [name, stats] of [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const mean = stats.nonNullCount > 0 ? stats.sum / stats.nonNullCount : null;
    const min = stats.nonNullCount > 0 ? stats.min : null;
    const max = stats.nonNullCount > 0 ? stats.max : null;
    console.error(
      [
        `- ${name}`,
        `count=${stats.count}`,
        `scored=${stats.nonNullCount}`,
        `null=${stats.nullCount}`,
        `mean=${mean === null ? "null" : mean.toFixed(4)}`,
        `min=${min === null ? "null" : min.toFixed(4)}`,
        `max=${max === null ? "null" : max.toFixed(4)}`,
      ].join(" "),
    );
  }
};

const csvEscape = (value: unknown): string => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (!text.includes(",") && !text.includes('"') && !text.includes("\n")) {
    return text;
  }
  return `"${text.replaceAll('"', '""')}"`;
};

const toCsv = (rows: readonly ScoreRow[]): string => {
  const headers = [
    "example_id",
    "repetition_number",
    "annotation_name",
    "annotator_kind",
    "score",
    "label",
    "explanation",
    "annotation_trace_id",
    "run_trace_id",
    "annotation_error",
    "run_error",
  ] as const;

  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(
      headers
        .map((header) => {
          const value = row[header];
          return csvEscape(value);
        })
        .join(","),
    );
  }
  return lines.join("\n");
};

const args = parseArgs();

const layer = args.baseUrl
  ? PhoenixClient.layer({ options: { baseUrl: args.baseUrl } })
  : PhoenixClient.skyhigh;

const program = Effect.gen(function* () {
  const runs = yield* fetchExperimentRuns(args.experimentId);
  const rows = toRows(runs, {
    annotationName: args.annotationName,
    includeNullScores: args.includeNullScores,
  });
  return rows;
});

Effect.runPromise(program.pipe(Effect.provide(layer)))
  .then((rows) => {
    printSummary(rows);
    if (args.format === "csv") {
      process.stdout.write(`${toCsv(rows)}\n`);
      return;
    }
    process.stdout.write(`${JSON.stringify(rows, null, 2)}\n`);
  })
  .catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Failed to fetch experiment scores: ${message}`);
    process.exitCode = 1;
  });
