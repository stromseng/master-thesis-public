import { Context, Effect, Layer, Schema } from "effect";
import { chunkFile } from "../generated/python-api/sdk.gen";
import type { DocumentChunkResponse } from "../generated/python-api/types.gen";
import { PYTHON_API_TIMEOUT } from "./pythonApi";
import { PythonApiClient } from "./PythonApiClient";

// ============================================================================
// Errors
// ============================================================================

export class ChunkingError extends Schema.TaggedError<ChunkingError>()("ChunkingError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

// ============================================================================
// Method Enum
// ============================================================================

export type ChunkingMethod =
  | "fast"
  | "recursive"
  | "semantic"
  | "late"
  | "neural"
  | "slumber"
  | "test";

// ============================================================================
// Service Definition
// ============================================================================

export interface ChunkingService {
  readonly method: ChunkingMethod;
  readonly chunk: (content: string) => Effect.Effect<DocumentChunkResponse[], ChunkingError>;
}

// ============================================================================
// Configuration Schemas
// ============================================================================

const FastConfigSchema = Schema.Struct({
  chunkSize: Schema.optional(Schema.Number),
  delimiters: Schema.optional(Schema.String),
});
export type FastConfig = typeof FastConfigSchema.Type;

const RecursiveConfigSchema = Schema.Struct({
  chunkSize: Schema.optional(Schema.Number),
  minCharactersPerChunk: Schema.optional(Schema.Number),
});
export type RecursiveConfig = typeof RecursiveConfigSchema.Type;

const SemanticConfigSchema = Schema.Struct({
  threshold: Schema.optional(Schema.Number),
  chunkSize: Schema.optional(Schema.Number),
  similarityWindow: Schema.optional(Schema.Number),
  skipWindow: Schema.optional(Schema.Number),
});
export type SemanticConfig = typeof SemanticConfigSchema.Type;

const LateConfigSchema = Schema.Struct({
  chunkSize: Schema.optional(Schema.Number),
  minCharactersPerChunk: Schema.optional(Schema.Number),
});
export type LateConfig = typeof LateConfigSchema.Type;

const NeuralConfigSchema = Schema.Struct({
  minCharactersPerChunk: Schema.optional(Schema.Number),
});
export type NeuralConfig = typeof NeuralConfigSchema.Type;

const SlumberConfigSchema = Schema.Struct({
  chunkSize: Schema.optional(Schema.Number),
  candidateSize: Schema.optional(Schema.Number),
  minCharactersPerChunk: Schema.optional(Schema.Number),
});
export type SlumberConfig = typeof SlumberConfigSchema.Type;

// ============================================================================
// Internal Helper
// ============================================================================

type ChunkBody = Parameters<typeof chunkFile>[0]["body"];

const makeChunker = (
  method: ChunkingMethod,
  buildBody: (content: string) => ChunkBody,
): Effect.Effect<ChunkingService, never, PythonApiClient> =>
  Effect.gen(function* () {
    const { client } = yield* PythonApiClient;

    return Chunking.of({
      method,
      chunk: (content: string): Effect.Effect<DocumentChunkResponse[], ChunkingError> =>
        Effect.gen(function* () {
          yield* Effect.annotateCurrentSpan({ contentLength: content.length, method });
          const response = yield* Effect.tryPromise(() =>
            chunkFile({ client, body: buildBody(content) }),
          ).pipe(
            Effect.mapError(
              (error) =>
                new ChunkingError({
                  reason: `Failed to chunk content using ${method} method`,
                  cause: error,
                }),
            ),
            Effect.timeoutFail({
              duration: PYTHON_API_TIMEOUT,
              onTimeout: () =>
                new ChunkingError({
                  reason: `Timed out while chunking content using ${method} method`,
                  cause: new Error(`Operation exceeded timeout: ${PYTHON_API_TIMEOUT}`),
                }),
            }),
          );
          if (!response.data?.chunks) {
            return yield* new ChunkingError({
              reason: `Failed to chunk content using ${method} method`,
              cause: response.error,
            });
          }
          return response.data.chunks;
        }).pipe(Effect.withSpan(`Chunking.chunk.${method}`)),
    });
  });

// ============================================================================
// Service with Static Layers
// ============================================================================

export class Chunking extends Context.Tag("@app/Chunking")<Chunking, ChunkingService>() {
  static readonly Fast = (config?: FastConfig) =>
    Layer.effect(
      Chunking,
      makeChunker("fast", (content) => ({
        method: "fast",
        content,
        chunk_size: config?.chunkSize,
        delimiters: config?.delimiters,
      })),
    );

  static readonly Recursive = (config?: RecursiveConfig) =>
    Layer.effect(
      Chunking,
      makeChunker("recursive", (content) => ({
        method: "recursive",
        content,
        chunk_size: config?.chunkSize,
        min_characters_per_chunk: config?.minCharactersPerChunk,
      })),
    );

  static readonly Semantic = (config?: SemanticConfig) =>
    Layer.effect(
      Chunking,
      makeChunker("semantic", (content) => ({
        method: "semantic",
        content,
        threshold: config?.threshold,
        chunk_size: config?.chunkSize,
        similarity_window: config?.similarityWindow,
        skip_window: config?.skipWindow,
      })),
    );

  static readonly Late = (config?: LateConfig) =>
    Layer.effect(
      Chunking,
      makeChunker("late", (content) => ({
        method: "late",
        content,
        chunk_size: config?.chunkSize,
        min_characters_per_chunk: config?.minCharactersPerChunk,
      })),
    );

  static readonly Neural = (config?: NeuralConfig) =>
    Layer.effect(
      Chunking,
      makeChunker("neural", (content) => ({
        method: "neural",
        content,
        min_characters_per_chunk: config?.minCharactersPerChunk,
      })),
    );

  static readonly Slumber = (config?: SlumberConfig) =>
    Layer.effect(
      Chunking,
      makeChunker("slumber", (content) => ({
        method: "slumber",
        content,
        chunk_size: config?.chunkSize,
        candidate_size: config?.candidateSize,
        min_characters_per_chunk: config?.minCharactersPerChunk,
      })),
    );

  static readonly Test = Layer.succeed(
    Chunking,
    Chunking.of({
      method: "test",
      chunk: (content) =>
        Effect.succeed([
          {
            text: content,
            start_index: 0,
            end_index: content.length,
            token_count: content.split(/\s+/).length,
            context: null,
          },
        ]),
    }),
  );
}
