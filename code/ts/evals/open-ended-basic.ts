import { Effect, Layer, Logger, Runtime } from "effect";
import * as S from "effect/Schema";
import {
  EvalTaskLayerSkyhigh,
  PhoenixClient,
  evaluatePhoenixExperimentWithProgress,
  runPhoenixExperimentOnlyWithProgress,
  runPhoenixExperimentWithProgress,
  taskRetryPolicy,
} from "./experiment_setup";
import { Judge, JudgeLayer } from "../src/services/Judge";
import {
  EvalLanguageModelLayer,
  JudgeLanguageModelLayer,
  generateObject,
} from "../src/services/LanguageModel";
import {
  buildOpenEndedQuestionPrompt,
  buildOpenEndedSystemPrompt,
  makeOpenEndedCorrectnessEvaluator,
  openEndedTaskOutputJsonSchema,
  openEndedTaskOutputSchema,
  type OpenEndedDatasetInput,
} from "./open-ended-evaluator";

const modelLayer = Layer.mergeAll(
  EvalLanguageModelLayer,
  JudgeLayer.pipe(Layer.provide(JudgeLanguageModelLayer)),
  Logger.pretty,
);

export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh);
export const experimentOnlyTaskLayer = Layer.mergeAll(
  EvalLanguageModelLayer,
  Logger.pretty,
  EvalTaskLayerSkyhigh,
);
export const evaluatorTaskLayer = Layer.mergeAll(
  JudgeLayer.pipe(Layer.provide(JudgeLanguageModelLayer)),
  Logger.pretty,
  EvalTaskLayerSkyhigh,
);

type DatasetLoader = Effect.Effect<{ datasetId: string; total: number }, unknown, PhoenixClient>;

const makeRunModel = (runModelName: string) =>
  Effect.fn(runModelName)(function* (question: OpenEndedDatasetInput) {
    const result = yield* generateObject({
      prompt: buildOpenEndedQuestionPrompt(question),
      system: buildOpenEndedSystemPrompt({ rag: false }),
      schema: openEndedTaskOutputJsonSchema,
    });

    return yield* S.decodeUnknown(openEndedTaskOutputSchema)(result.object);
  });

export const makeOpenEndedEvaluators = (runtime: Runtime.Runtime<Judge>) => [
  makeOpenEndedCorrectnessEvaluator(runtime),
];

export const makeOpenEndedProgram = ({
  getOrCreateDataset,
  runModelName,
}: {
  getOrCreateDataset: DatasetLoader;
  runModelName: string;
}) =>
  Effect.gen(function* () {
    const runModel = makeRunModel(runModelName);
    const { datasetId, total } = yield* getOrCreateDataset;
    const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof taskLayer>>();

    yield* runPhoenixExperimentWithProgress({
      experimentDescription: "",
      total,
      dataset: { datasetId },
      useBatchSpanProcessor: false,
      setGlobalTracerProvider: false,
      task: (example) => {
        const input = example.input as OpenEndedDatasetInput;
        return Runtime.runPromise(runtime)(Effect.scoped(runModel(input).pipe(taskRetryPolicy)));
      },
      evaluators: [makeOpenEndedCorrectnessEvaluator(runtime)],
    });
  });

export const makeOpenEndedExperimentOnlyProgram = ({
  getOrCreateDataset,
  runModelName,
}: {
  getOrCreateDataset: DatasetLoader;
  runModelName: string;
}) =>
  Effect.gen(function* () {
    const runModel = makeRunModel(runModelName);
    const { datasetId, total } = yield* getOrCreateDataset;
    const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof experimentOnlyTaskLayer>>();

    return yield* runPhoenixExperimentOnlyWithProgress({
      experimentDescription: "",
      total,
      dataset: { datasetId },
      useBatchSpanProcessor: false,
      setGlobalTracerProvider: false,
      task: (example) => {
        const input = example.input as OpenEndedDatasetInput;
        return Runtime.runPromise(runtime)(Effect.scoped(runModel(input).pipe(taskRetryPolicy)));
      },
    });
  });

export const makeOpenEndedEvaluateOnlyProgram = (experimentId: string) =>
  Effect.gen(function* () {
    const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof evaluatorTaskLayer>>();

    yield* evaluatePhoenixExperimentWithProgress({
      experimentId,
      evaluators: [makeOpenEndedCorrectnessEvaluator(runtime)],
      useBatchSpanProcessor: false,
      setGlobalTracerProvider: false,
    });
  });
