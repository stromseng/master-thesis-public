/// Not to be used for Qdrant embeddings. Only used for scripts and tests.
import { embed, embedMany } from "ai";
import { Effect, Layer, Schema } from "effect";
import { LiteLLMModelId, LLMProvider } from "./LLMProvider";

// ============================================================================
// Errors
// ============================================================================

export class EmbeddingError extends Schema.TaggedError<EmbeddingError>()("EmbeddingError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

// ============================================================================
// Service Definition
// ============================================================================

export class Embedding extends Effect.Service<Embedding>()("@app/Embedding", {
  accessors: true,
  dependencies: [LLMProvider.IdunLiteLLM],
  effect: Effect.gen(function* () {
    const { provider } = yield* LLMProvider;

    const embedOne = Effect.fn("Embedding.embed")(function* (text: string, model?: string) {
      const modelId = model ?? LiteLLMModelId.Qwen3Embedding8B;

      const result = yield* Effect.tryPromise({
        try: () =>
          embed({
            model: provider.embeddingModel(modelId),
            value: text,
          }),
        catch: (cause) =>
          new EmbeddingError({
            reason: "Failed to generate embedding for text",
            cause,
          }),
      });

      return result.embedding;
    });

    const embedManyFn = Effect.fn("Embedding.embedMany")(function* (
      texts: readonly string[],
      model?: string,
    ) {
      const modelId = model ?? LiteLLMModelId.Qwen3Embedding8B;

      const result = yield* Effect.tryPromise({
        try: () =>
          embedMany({
            model: provider.embeddingModel(modelId),
            values: texts as string[],
          }),
        catch: (cause) =>
          new EmbeddingError({
            reason: `Failed to generate embeddings for ${texts.length} texts`,
            cause,
          }),
      });

      return result.embeddings;
    });

    return {
      embed: embedOne,
      embedMany: embedManyFn,
    };
  }),
}) {
  static readonly testLayer = Layer.succeed(
    Embedding,
    new Embedding({
      embed: (_text: string, _model?: string) =>
        Effect.succeed(Array.from({ length: 768 }, () => Math.random())),
      embedMany: (texts: readonly string[], _model?: string) =>
        Effect.succeed(texts.map(() => Array.from({ length: 768 }, () => Math.random()))),
    }),
  );
}
