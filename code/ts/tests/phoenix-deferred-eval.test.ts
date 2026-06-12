/**
 * Tests the two-step Phoenix experiment workflow:
 * 1. Run an experiment WITHOUT evaluators (just task execution)
 * 2. Later, run evaluators on the completed experiment via `evaluateExperiment`
 */
import { describe, expect, it } from "@effect/vitest";
import { Effect } from "effect";
import { createOrGetDataset } from "@arizeai/phoenix-client/datasets";
import {
  asExperimentEvaluator,
  evaluateExperiment,
  getExperiment,
  runExperiment,
} from "@arizeai/phoenix-client/experiments";
import type { Example } from "@arizeai/phoenix-client/types/datasets";
import { PhoenixClient } from "../src/services/PhoenixClient";

const DATASET_NAME = `deferred-eval-test-${Date.now()}`;

const MOCK_EXAMPLES: {
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metadata: Record<string, unknown>;
}[] = [
  {
    input: { question: "What is the capital of Norway?" },
    output: { answer: "Oslo" },
    metadata: {},
  },
  {
    input: { question: "What is 2 + 2?" },
    output: { answer: "4" },
    metadata: {},
  },
  {
    input: { question: "What color is the sky?" },
    output: { answer: "Blue" },
    metadata: {},
  },
];

/** Simple mock task: echoes the question back as an "answer" */
const mockTask = async (example: Example) => {
  const question = example.input.question as string;
  return { answer: `Mock answer to: ${question}` };
};

/** Evaluator: checks if the output contains the word "Mock" */
const containsMockEvaluator = asExperimentEvaluator({
  name: "contains-mock",
  kind: "CODE",
  evaluate: async ({ output }) => {
    const text =
      typeof output === "object" && output !== null ? JSON.stringify(output) : String(output);
    const contains = text.includes("Mock");
    return {
      score: contains ? 1 : 0,
      label: contains ? "contains-mock" : "missing-mock",
      explanation: contains ? "Output contains 'Mock'" : "Output does not contain 'Mock'",
      metadata: {},
    };
  },
});

/** Evaluator: checks if the output matches the expected answer */
const exactMatchEvaluator = asExperimentEvaluator({
  name: "exact-match",
  kind: "CODE",
  evaluate: async ({ output, expected }) => {
    const outputAnswer =
      typeof output === "object" && output !== null
        ? (output as Record<string, unknown>).answer
        : output;
    const expectedAnswer = expected?.answer;
    const matches = outputAnswer === expectedAnswer;
    return {
      score: matches ? 1 : 0,
      label: matches ? "exact-match" : "no-match",
      explanation: matches
        ? "Output matches expected"
        : `Expected "${expectedAnswer}", got "${outputAnswer}"`,
      metadata: {},
    };
  },
});

describe("Phoenix deferred evaluation", () => {
  it.live(
    "can run experiment without evaluators, then evaluate separately",
    () =>
      Effect.gen(function* () {
        const phoenix = yield* PhoenixClient;

        // Step 1: Create a test dataset
        const { datasetId } = yield* phoenix.use((client) =>
          createOrGetDataset({
            client,
            name: DATASET_NAME,
            description: "Test dataset for deferred evaluation flow",
            examples: MOCK_EXAMPLES,
          }),
        );
        expect(datasetId).toBeDefined();

        // Step 2: Run experiment WITHOUT evaluators
        const experiment = yield* phoenix.use((client) =>
          runExperiment({
            client,
            dataset: { datasetId },
            task: mockTask,
            experimentName: `deferred-eval-test-${Date.now()}`,
            // No evaluators! This is the key part of the test.
            setGlobalTracerProvider: false,
          }),
        );

        expect(experiment.id).toBeDefined();
        const runIds = Object.keys(experiment.runs);
        expect(runIds.length).toBe(MOCK_EXAMPLES.length);

        // Verify all runs completed successfully
        for (const runId of runIds) {
          const run = experiment.runs[runId]!;
          expect(run.error).toBeNull();
          expect(run.output).toBeDefined();
        }

        // Verify no evaluation runs exist yet
        expect(experiment.evaluationRuns ?? []).toHaveLength(0);

        // Step 3: Re-fetch the experiment by ID (simulates a later session)
        const fetchedExperiment = yield* phoenix.use((client) =>
          getExperiment({ client, experimentId: experiment.id }),
        );
        expect(fetchedExperiment.id).toBe(experiment.id);

        // Step 4: Run evaluators on the completed experiment
        const evaluated = yield* phoenix.use((client) =>
          evaluateExperiment({
            client,
            experiment: fetchedExperiment,
            evaluators: [containsMockEvaluator, exactMatchEvaluator],
            setGlobalTracerProvider: false,
          }),
        );

        // Verify evaluation runs were created
        const evalRuns = evaluated.evaluationRuns ?? [];
        expect(evalRuns.length).toBeGreaterThan(0);

        // We expect 2 evaluators x 3 examples = 6 evaluation runs
        expect(evalRuns.length).toBe(MOCK_EXAMPLES.length * 2);

        // Check that both evaluator names are present
        const evalNames = new Set(evalRuns.map((r) => r.name));
        expect(evalNames.has("contains-mock")).toBe(true);
        expect(evalNames.has("exact-match")).toBe(true);

        // All "contains-mock" evaluations should score 1 (our mock task always includes "Mock")
        const containsMockRuns = evalRuns.filter((r) => r.name === "contains-mock");
        for (const run of containsMockRuns) {
          expect(run.result?.score).toBe(1);
        }

        // All "exact-match" evaluations should score 0 (mock task doesn't return exact expected answers)
        const exactMatchRuns = evalRuns.filter((r) => r.name === "exact-match");
        for (const run of exactMatchRuns) {
          expect(run.result?.score).toBe(0);
        }
      }).pipe(Effect.provide(PhoenixClient.skyhigh)),
    { timeout: 60_000 },
  );
});
