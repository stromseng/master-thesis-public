// Usage:
//   bun evals/pei2024/uk_theory_test.ts
//   EVAL_MODEL="moonshotai/Kimi-K2.5" bun evals/pei2024/uk_theory_test.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8001 bun evals/pei2024/uk_theory_test.ts
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
import { buildQuestionPrompt, getOrCreatePeiUkDataset } from "./pei2024";

const runModel = Effect.fn("question.pei2024_uk_theory_test.runModel")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const result = yield* generateObject({
    prompt: buildQuestionPrompt(question),
    system: buildSystemPrompt({ answerMode, rag: false }),
    schema: getOutputJsonSchema(answerMode),
  });

  return normalizeTaskOutput(result.object as Record<string, unknown>) as QuestionTaskOutput;
});

const modelLayer = Layer.mergeAll(EvalLanguageModelLayer, Logger.pretty);
export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh);

export const program = Effect.gen(function* () {
  const { datasetId, total } = yield* getOrCreatePeiUkDataset();
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
