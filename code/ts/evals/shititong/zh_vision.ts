// Usage:
//   bun evals/shititong/zh_vision.ts
//   EVAL_MULTIMODAL_MODEL="mistralai/Mistral-Large-3" bun evals/shititong/zh_vision.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8000 bun evals/shititong/zh_vision.ts
import type { ImagePart, ModelMessage, TextPart } from "ai";
import { Effect, Layer, Logger, Runtime } from "effect";
import {
  EvalTaskLayerSkyhigh,
  runPhoenixExperimentWithProgress,
  taskRetryPolicy,
} from "../experiment_setup";
import { EvalMultimodalLanguageModelLayer, generateObject } from "../../src/services/LanguageModel";
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
import { buildQuestionPrompt, getOrCreateShititongZhVisionDataset } from "./shititong";

/** Collect all image URIs from question-level and option-level images. */
const collectImageUris = (question: QuestionDatasetInput): string[] => {
  const uris: string[] = [];
  for (const img of question.images) {
    uris.push(img.uri);
  }
  for (const option of question.options) {
    for (const img of option.images) {
      uris.push(img.uri);
    }
  }
  return uris;
};

const buildMessages = (
  question: QuestionDatasetInput,
  imageParts: readonly ImagePart[],
  answerMode: AnswerMode,
): { messages: ModelMessage[] } => {
  const textPart: TextPart = { type: "text", text: buildQuestionPrompt(question) };

  return {
    messages: [
      { role: "system", content: buildSystemPrompt({ answerMode, rag: false }) },
      { role: "user", content: [textPart, ...imageParts] },
    ],
  };
};

const runModel = Effect.fn("question.shititong_zh_vision.runModel")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const imageUris = collectImageUris(question);
  const imageParts: ImagePart[] = imageUris.map((uri) => ({
    type: "image",
    image: new URL(uri.replace(/"+$/, "")),
  }));

  const { messages } = buildMessages(question, imageParts, answerMode);
  const result = yield* generateObject({ messages, schema: getOutputJsonSchema(answerMode) });

  return normalizeTaskOutput(result.object as Record<string, unknown>) as QuestionTaskOutput;
});

const modelLayer = Layer.mergeAll(EvalMultimodalLanguageModelLayer, Logger.pretty);
export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh);

export const program = Effect.gen(function* () {
  const { datasetId, total } = yield* getOrCreateShititongZhVisionDataset();
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
