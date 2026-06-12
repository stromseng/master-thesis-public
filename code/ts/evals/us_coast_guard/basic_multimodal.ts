// Usage:
//   bun evals/us_coast_guard/basic_multimodal.ts
//   EVAL_MULTIMODAL_MODEL="mistralai/Mistral-Large-3" bun evals/us_coast_guard/basic_multimodal.ts
//   EVAL_PROVIDER=vllm VLLM_PORT=8000 bun evals/us_coast_guard/basic_multimodal.ts
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
import {
  buildQuestionPrompt,
  getOrCreateUsCoastGuardMultimodalDataset,
  loadImage,
} from "./us_coast_guard";

const imageCache = new Map<string, Buffer>();

const getCachedImage = Effect.fn("question.us_coast_guard_multimodal.getCachedImage")(function* (
  uri: string,
) {
  const cached = imageCache.get(uri);
  if (cached) {
    return cached;
  }

  const loaded = yield* loadImage(uri);
  yield* Effect.sync(() => imageCache.set(uri, loaded));
  return loaded;
});

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

const runModel = Effect.fn("question.us_coast_guard_multimodal.runModel")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const imageParts = yield* Effect.all(
    question.images.map((image) =>
      getCachedImage(image.uri).pipe(
        Effect.map((imageData) => ({ type: "image", image: imageData }) satisfies ImagePart),
      ),
    ),
  );

  const { messages } = buildMessages(question, imageParts, answerMode);
  const result = yield* generateObject({ messages, schema: getOutputJsonSchema(answerMode) });

  return normalizeTaskOutput(result.object as Record<string, unknown>) as QuestionTaskOutput;
});

const modelLayer = Layer.mergeAll(EvalMultimodalLanguageModelLayer, Logger.pretty);
export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh);

export const program = Effect.gen(function* () {
  const { datasetId, total } = yield* getOrCreateUsCoastGuardMultimodalDataset();
  const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof taskLayer>>();
  imageCache.clear();

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
