// Usage:
//   bun evals/navreas/navreas.eval.ts
//   EVAL_MODEL="moonshotai/Kimi-K2.5" bun evals/navreas/navreas.eval.ts
//   EVAL_PROVIDER=vllm bun evals/navreas/navreas.eval.ts
import type { ImagePart, ModelMessage, TextPart } from "ai";
import * as Progress from "effective-progress";
import { Effect, Layer, Runtime } from "effect";
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
import {
  baseLayer,
  buildQuestionPrompt,
  getOrCreateNavreasDataset,
  loadImage,
  navreasDatasets,
} from "./navreas";

const imageCache = new Map<string, Buffer>();

const getCachedImage = Effect.fn("navreas.getCachedImage")(function* (uri: string) {
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
  const userContent: Array<ImagePart | TextPart> = [...imageParts, textPart];

  return {
    messages: [
      { role: "system", content: buildSystemPrompt({ answerMode, rag: false }) },
      { role: "user", content: userContent },
    ],
  };
};

const skipImages = process.env.SKIP_IMAGES === "true";

const runModel = Effect.fn("question.navreas.runModel")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const imageParts = skipImages
    ? []
    : yield* Effect.all(
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

const modelLayer = Layer.mergeAll(EvalLanguageModelLayer, baseLayer);
export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh);

export const program = Effect.gen(function* () {
  const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof taskLayer>>();

  yield* Progress.forEach(
    navreasDatasets,
    (dataset) =>
      Effect.gen(function* () {
        imageCache.clear();

        const { datasetId, total } = yield* getOrCreateNavreasDataset(dataset);

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
      }),
    { description: "NavREAS evaluations" },
  );

  yield* Effect.log("All evaluations complete");
});

export const run = Effect.scoped(program.pipe(Effect.provide(taskLayer)));

if (import.meta.main) {
  Effect.runPromise(run);
}
