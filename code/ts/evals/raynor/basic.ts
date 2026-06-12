// Usage:
//   bun evals/raynor/basic.ts
//   EVAL_MODEL="moonshotai/Kimi-K2.5" bun evals/raynor/basic.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8001 bun evals/raynor/basic.ts
import type { ModelMessage, TextPart } from "ai";
import { Effect, Layer, Logger, Runtime } from "effect";
import {
  EvalTaskLayerSkyhigh,
  runPhoenixExperimentWithProgress,
  taskRetryPolicy,
} from "../experiment_setup";
import { EvalLanguageModelLayer, generateObject } from "../../src/services/LanguageModel";
import {
  buildSystemPrompt,
  getAnswerMode,
  getOutputJsonSchema,
  normalizeTaskOutput,
  questionMetricEvaluators,
  type AnswerMode,
  type QuestionDatasetExpected,
  type QuestionDatasetInput,
  type QuestionTaskOutput,
} from "../evaluator";
import { buildQuestionPrompt, getOrCreateRaynorDataset } from "./raynor";

const buildMessages = (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
): { messages: ModelMessage[] } => {
  const textPart: TextPart = { type: "text", text: buildQuestionPrompt(question) };

  return {
    messages: [
      { role: "system", content: buildSystemPrompt({ answerMode, rag: false }) },
      { role: "user", content: [textPart] },
    ],
  };
};

const runModel = Effect.fn("question.raynor.runModel")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const { messages } = buildMessages(question, answerMode);
  const result = yield* generateObject({ messages, schema: getOutputJsonSchema(answerMode) });

  return normalizeTaskOutput(result.object as Record<string, unknown>) as QuestionTaskOutput;
});

const modelLayer = Layer.mergeAll(EvalLanguageModelLayer, Logger.pretty);
export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh);

export const program = Effect.gen(function* () {
  const { datasetId, total } = yield* getOrCreateRaynorDataset();
  const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof taskLayer>>();
  yield* runPhoenixExperimentWithProgress({
    experimentDescription: "",
    total,
    dataset: { datasetId },
    useBatchSpanProcessor: false,
    setGlobalTracerProvider: false,
    task: (example) => {
      const input = example.input as QuestionDatasetInput;
      const expected = example.output as QuestionDatasetExpected | undefined;
      const answerMode = getAnswerMode(expected?.correctOptionIds ?? []);
      return Runtime.runPromise(runtime)(
        Effect.scoped(runModel(input, answerMode).pipe(taskRetryPolicy)),
      );
    },
    evaluators: [...questionMetricEvaluators],
  });
});

export const run = Effect.scoped(program.pipe(Effect.provide(taskLayer)));

if (import.meta.main) {
  Effect.runPromise(run);
}
