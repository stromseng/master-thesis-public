// RAG eval for CrewCN maritime exam questions.
//
// Usage:  bun evals/crew/rag.eval.ts
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
//   DENSE_METHOD=fastembed CHUNKING_METHOD=recursive bun evals/crew/rag.eval.ts
//   DENSE_METHOD=fastembed CHUNKING_METHOD=semantic  bun evals/crew/rag.eval.ts
//   DENSE_METHOD=litellm  CHUNKING_METHOD=recursive bun evals/crew/rag.eval.ts
//   DENSE_METHOD=litellm  CHUNKING_METHOD=semantic  bun evals/crew/rag.eval.ts
import { Effect, Layer, Logger, Runtime } from "effect";
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
import { buildQuestionPrompt, getOrCreateCrewDataset } from "./crew";
import { resolveChunkingLayer, resolveDenseLayer, resolveRetrievalLimit } from "../constants";

const formatContext = (results: readonly QdrantSearchPoint[]): string => {
  if (results.length === 0) return "(no context)";
  return results
    .map((result, index) => {
      return `<source index="${index + 1}" metadata="${result.payload.source}">\n${result.payload.text}\n</source>`;
    })
    .join("\n\n");
};

const buildRagPrompt = (
  question: QuestionDatasetInput,
  results: readonly QdrantSearchPoint[],
): string => {
  return [
    "<context>",
    formatContext(results),
    "</context>",
    "",
    buildQuestionPrompt(question),
  ].join("\n");
};

const runModelWithRag = Effect.fn("question.crew.rag.runModelWithRag")(function* (
  question: QuestionDatasetInput,
  answerMode: AnswerMode,
) {
  const retrieval = yield* Retrieval;
  const context = yield* retrieval.search(question.questionText);
  const prompt = buildRagPrompt(question, context);

  const result = yield* generateObject({
    prompt,
    system: buildSystemPrompt({ answerMode, rag: true }),
    schema: getOutputJsonSchema(answerMode),
  });

  return normalizeTaskOutput(result.object as Record<string, unknown>) as QuestionTaskOutput;
});

// ============================================================================
// Layer composition
// ============================================================================

const embeddingProviderLayer = EmbeddingProvider.make(resolveDenseLayer(), SparseEmbedding.Default);
const chunkingLayer = resolveChunkingLayer();
const qdrantLayer = Qdrant.skyhigh.pipe(
  Layer.provide(embeddingProviderLayer),
  Layer.provide(chunkingLayer),
);

const retrievalLayer = Retrieval.layer({ limit: resolveRetrievalLimit() }).pipe(
  Layer.provide(qdrantLayer),
  Layer.provide(chunkingLayer),
);

const configLayers = Layer.mergeAll(embeddingProviderLayer, chunkingLayer);

const modelLayer = Layer.mergeAll(EvalLanguageModelLayer, Logger.pretty, retrievalLayer);
const taskLayer = Layer.mergeAll(modelLayer, EvalTaskLayerSkyhigh).pipe(
  Layer.provideMerge(configLayers),
);
export const taskLayerWithPythonApi = taskLayer.pipe(Layer.provideMerge(PythonApiClient.Default));

export const program = Effect.gen(function* () {
  const { datasetId, total } = yield* getOrCreateCrewDataset();
  const runtime = yield* Effect.runtime<Layer.Layer.Success<typeof taskLayerWithPythonApi>>();

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
});

export const run = Effect.scoped(program.pipe(Effect.provide(taskLayerWithPythonApi)));

if (import.meta.main) {
  Effect.runPromise(run);
}
