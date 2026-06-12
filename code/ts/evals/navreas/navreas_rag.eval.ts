// RAG eval for NavREAS navigation scene understanding questions.
//
// Usage:  bun evals/navreas/navreas_rag.eval.ts
//
// Prerequisites:
//   Python FastAPI server must be running (`just dev-python`) for BM25 sparse embeddings.
//
// Environment variables:
//   EVAL_MODEL        LLM to evaluate          (default: auto)
//   EVAL_PROVIDER     litellm | vllm            (default: litellm)
//   DENSE_METHOD      fastembed | litellm       (default: fastembed)
//   CHUNKING_METHOD   recursive | semantic      (default: recursive)
//   RETRIEVAL_LIMIT   number of chunks          (default: 5)
//   EVAL_CONCURRENCY  parallel tasks            (default: 2)
//
// Examples:
//   DENSE_METHOD=fastembed CHUNKING_METHOD=recursive bun evals/navreas/navreas_rag.eval.ts
//   DENSE_METHOD=fastembed CHUNKING_METHOD=semantic  bun evals/navreas/navreas_rag.eval.ts
//   DENSE_METHOD=litellm  CHUNKING_METHOD=recursive bun evals/navreas/navreas_rag.eval.ts
//   DENSE_METHOD=litellm  CHUNKING_METHOD=semantic  bun evals/navreas/navreas_rag.eval.ts
import type { ImagePart, ModelMessage, TextPart } from "ai";
import * as Progress from "effective-progress";
import { Effect, Layer, Runtime } from "effect";
import {
  EvalTaskLayerSkyhigh,
  runPhoenixExperimentWithProgress,
  taskRetryPolicy,
} from "../experiment_setup";
import { EvalLanguageModelLayer, generateObject } from "../../src/services/LanguageModel";
import { Retrieval } from "../../src/services/Retrieval";
import { EmbeddingProvider, LateEmbedding, SparseEmbedding } from "../../src/services/embeddings";
import { Qdrant, type QdrantSearchPoint } from "../../src/services/Qdrant";
import { PythonApiClient } from "../../src/services/PythonApiClient";
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
import { resolveChunkingLayer, resolveDenseLayer, resolveRetrievalLimit } from "../constants";

const formatContext = (results: readonly QdrantSearchPoint[]): string => {
  if (results.length === 0) return "(no context)";
  return results
    .map((r, i) => {
      return `Source ${i + 1}: ${r.payload.source}\n${r.payload.text}`;
    })
    .join("\n\n");
};

const imageCache = new Map<string, Buffer>();

const getCachedImage = Effect.fn("navreas.rag.getCachedImage")(function* (uri: string) {
  const cached = imageCache.get(uri);
  if (cached) {
    return cached;
  }

  const loaded = yield* loadImage(uri);
  yield* Effect.sync(() => imageCache.set(uri, loaded));
  return loaded;
});

const buildMessagesWithRag = (
  question: QuestionDatasetInput,
  imageParts: readonly ImagePart[],
  context: readonly QdrantSearchPoint[],
  answerMode: AnswerMode,
): { messages: ModelMessage[] } => {
  const textPart: TextPart = {
    type: "text",
    text: [
      "<context>",
      formatContext(context),
      "</context>",
      "",
      buildQuestionPrompt(question),
    ].join("\n"),
  };

  const userContent: Array<ImagePart | TextPart> = [...imageParts, textPart];
  const messages: ModelMessage[] = [
    { role: "system", content: buildSystemPrompt({ answerMode, rag: true }) },
    { role: "user", content: userContent },
  ];

  return { messages };
};

const runModelWithRag = Effect.fn("question.navreas.rag.runModelWithRag")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const retrieval = yield* Retrieval;
  const context = yield* retrieval.search(question.questionText);
  const imageParts = yield* Effect.all(
    question.images.map((image) =>
      getCachedImage(image.uri).pipe(
        Effect.map((imageData) => ({ type: "image", image: imageData }) satisfies ImagePart),
      ),
    ),
  );

  const { messages } = buildMessagesWithRag(question, imageParts, context, answerMode);
  const result = yield* generateObject({ messages, schema: getOutputJsonSchema(answerMode) });
  return normalizeTaskOutput(result.object as Record<string, unknown>) as QuestionTaskOutput;
});

// ============================================================================
// Layer composition
// ============================================================================

// Build retrieval layer with hybrid search (dense + sparse + late)
const embeddingProviderLayer = EmbeddingProvider.make(
  resolveDenseLayer(),
  SparseEmbedding.Default,
  LateEmbedding.Default,
);
const chunkingLayer = resolveChunkingLayer();

const retrievalLayer = Retrieval.layer({ limit: resolveRetrievalLimit() }).pipe(
  Layer.provide(embeddingProviderLayer),
  Layer.provide(Qdrant.localhost),
  Layer.provide(chunkingLayer),
);

// Config layers exposed for metadata extraction
const configLayers = Layer.mergeAll(embeddingProviderLayer, chunkingLayer);

const modelLayer = Layer.mergeAll(EvalLanguageModelLayer, baseLayer, retrievalLayer);
export const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh).pipe(
  Layer.provideMerge(configLayers),
  Layer.provideMerge(PythonApiClient.Default),
);

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
              Effect.scoped(runModelWithRag(input, answerMode).pipe(taskRetryPolicy)),
            );
          },
          evaluators: [...questionMetricEvaluators],
        });
      }),
    { description: "NavREAS RAG evaluations" },
  );

  yield* Effect.log("All RAG evaluations complete");
});

export const run = Effect.scoped(program.pipe(Effect.provide(taskLayer)));

if (import.meta.main) {
  Effect.runPromise(run);
}
