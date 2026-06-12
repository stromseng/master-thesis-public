// Usage:
//   bun scripts/examples/fetch-experiment-run.ts --experiment-id RXhwZXJpbWVudDo2NzA= --question-id pei2024application-uk-0005
//   bun scripts/examples/fetch-experiment-run.ts --experiment-id RXhw... --question-id pei2024application-uk-0005 --base-url http://127.0.0.1:6006
//
// Fetches a single experiment run (question + model output + grading annotations)
// for a given source question id from a Phoenix experiment export.
import { Effect } from "effect";
import { PhoenixClient } from "../../src/services/PhoenixClient";

type CliArgs = {
  experimentId: string;
  questionId: string;
  baseUrl?: string;
};

type ExperimentRun = {
  example_id?: string;
  input?: {
    id?: string;
    questionText?: string;
    metadata?: { sourceQuestionId?: string };
  };
  reference_output?: {
    referenceCorrectAnswers?: string[];
    referenceIncorrectAnswers?: string[];
  };
  output?: { reason?: string; answer?: string };
  error?: string | null;
  annotations?: Array<{
    name?: string;
    label?: string | null;
    score?: number | null;
    explanation?: string | null;
  }>;
};

const parseArgs = (): CliArgs => {
  const argv = process.argv.slice(2);
  const get = (name: string): string | undefined => {
    const index = argv.indexOf(name);
    if (index < 0 || index + 1 >= argv.length) return undefined;
    return argv[index + 1];
  };

  const experimentId = get("--experiment-id") ?? get("--experiment");
  if (!experimentId) throw new Error("Missing required --experiment-id <id>");
  const questionId = get("--question-id") ?? get("--question");
  if (!questionId) throw new Error("Missing required --question-id <id>");

  return { experimentId, questionId, baseUrl: get("--base-url") };
};

const fetchExperimentRuns = Effect.fn("fetchExperimentRuns")(function* (experimentId: string) {
  const phoenix = yield* PhoenixClient;
  // The OpenAPI client percent-encodes the base64 id's trailing "=" which the
  // server rejects (405), so build the URL manually keeping the id raw.
  const raw = yield* phoenix.use(async (client) => {
    const baseUrl = client.config.baseUrl;
    if (!baseUrl) throw new Error("Phoenix client baseUrl is not configured");
    const response = await fetch(`${baseUrl}/v1/experiments/${experimentId}/json`);
    if (!response.ok) {
      throw new Error(
        `Phoenix API error ${response.status} ${response.statusText} for experiment ${experimentId}`,
      );
    }
    return (await response.json()) as unknown;
  });

  const parsed = typeof raw === "string" ? (JSON.parse(raw) as unknown) : raw;
  if (!Array.isArray(parsed)) {
    throw new Error(`Unexpected /v1/experiments/{id}/json shape for experiment ${experimentId}`);
  }
  return parsed as ExperimentRun[];
});

const args = parseArgs();
const layer = args.baseUrl
  ? PhoenixClient.layer({ options: { baseUrl: args.baseUrl } })
  : PhoenixClient.skyhigh;

const program = Effect.gen(function* () {
  const runs = yield* fetchExperimentRuns(args.experimentId);
  const run = runs.find(
    (r) =>
      r.input?.id === args.questionId || r.input?.metadata?.sourceQuestionId === args.questionId,
  );
  if (!run) {
    throw new Error(
      `No run found for question id "${args.questionId}" in experiment ${args.experimentId} (${runs.length} runs scanned)`,
    );
  }
  return run;
});

Effect.runPromise(program.pipe(Effect.provide(layer)))
  .then((run) => {
    process.stdout.write(`${JSON.stringify(run, null, 2)}\n`);
  })
  .catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Failed to fetch experiment run: ${message}`);
    process.exitCode = 1;
  });
