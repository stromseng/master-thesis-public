import { describe, expect, it } from "@effect/vitest";
import { Effect, Layer, Option } from "effect";
import type { Client } from "../src/generated/python-api/client";
import {
  DenseEmbedding,
  EmbeddingProvider,
  LateEmbedding,
  SparseEmbedding,
  SparseResolveDescriptorMismatchError,
} from "../src/services/embeddings";
import { makePythonApiClientLayer } from "./support/layers";

const responseFields = {
  request: new Request("http://localhost/mock"),
  response: new Response(null, { status: 200 }),
};

const makeMockClient = (handlers: Record<string, (body: unknown) => unknown>): Client =>
  ({
    post: ({ url, body }: { url: string; body: unknown }) => {
      const handler = handlers[url];
      if (!handler) {
        throw new Error(`Unexpected POST URL: ${url}`);
      }
      return Promise.resolve({
        data: handler(body),
        error: undefined,
        ...responseFields,
      });
    },
  }) as unknown as Client;

const denseTestService = DenseEmbedding.of({
  descriptor: {
    method: "test",
    modelName: "provider-dense",
    vectorName: "provider-dense-vector",
    vectorSize: 4,
  },
  embed: (text: string) => Effect.succeed([text.length, 1, 2, 3]),
  embedBatch: (texts: string[]) => Effect.succeed(texts.map((text) => [text.length, 1, 2, 3])),
});

const sparseTestService = SparseEmbedding.of({
  descriptor: {
    method: "test",
    modelName: "provider-sparse",
    vectorName: "provider-sparse-vector",
  },
  embed: (text: string) =>
    Effect.succeed({
      indices: [0],
      values: [text.length],
    }),
  embedBatch: (texts: string[]) =>
    Effect.succeed(
      texts.map((text, idx) => ({
        indices: [idx],
        values: [text.length],
      })),
    ),
});

const lateTestService = LateEmbedding.of({
  descriptor: {
    method: "test",
    modelName: "provider-late",
    vectorName: "provider-late-vector",
    vectorSize: 2,
  },
  embed: (_text: string) => Effect.succeed([[0, 1]]),
  embedBatch: (texts: string[]) => Effect.succeed(texts.map(() => [[0, 1]])),
});

describe("embedding service resolve initialization", () => {
  it.effect("DenseEmbedding resolves descriptor from dense-only resolver", () =>
    Effect.gen(function* () {
      const modelName = "dense-model";
      const vectorName = "dense-vector";

      const client = makeMockClient({
        "/embed/dense/resolve": (body) => {
          expect(body).toEqual({
            method: "fastembed",
            model_name: modelName,
            vector_name: vectorName,
          });
          return {
            model_name: modelName,
            vector_name: vectorName,
            vector_size: 384,
          };
        },
      });

      const layer = DenseEmbedding.Fastembed({ modelName, vectorName }).pipe(
        Layer.provide(makePythonApiClientLayer(client)),
      );

      const dense = yield* DenseEmbedding.pipe(Effect.provide(layer));

      expect(dense.descriptor).toEqual({
        method: "fastembed",
        modelName,
        vectorName,
        vectorSize: 384,
      });
    }),
  );

  it.effect("SparseEmbedding resolves descriptor from sparse-only resolver", () =>
    Effect.gen(function* () {
      const modelName = "sparse-model";
      const vectorName = "sparse-vector";

      const client = makeMockClient({
        "/embed/sparse/resolve": (body) => {
          expect(body).toEqual({
            method: "fastembed",
            model_name: modelName,
            vector_name: vectorName,
          });
          return {
            model_name: modelName,
            vector_name: vectorName,
          };
        },
      });

      const layer = SparseEmbedding.Fastembed({ modelName, vectorName }).pipe(
        Layer.provide(makePythonApiClientLayer(client)),
      );

      const sparse = yield* SparseEmbedding.pipe(Effect.provide(layer));

      expect(sparse.descriptor).toEqual({
        method: "fastembed",
        modelName,
        vectorName,
      });
    }),
  );

  it.effect("SparseEmbedding fails when resolved descriptor mismatches config", () =>
    Effect.gen(function* () {
      const client = makeMockClient({
        "/embed/sparse/resolve": () => ({
          model_name: "unexpected-model",
          vector_name: "unexpected-vector",
        }),
      });

      const layer = SparseEmbedding.Fastembed({
        modelName: "expected-model",
        vectorName: "expected-vector",
      }).pipe(Layer.provide(makePythonApiClientLayer(client)));

      const error = yield* Effect.scoped(Effect.flip(Layer.build(layer)));

      expect(error._tag).toBe("SparseResolveDescriptorMismatchError");
      expect(error).toBeInstanceOf(SparseResolveDescriptorMismatchError);
      expect(error).toMatchObject({
        expectedModelName: "expected-model",
        expectedVectorName: "expected-vector",
        actualModelName: "unexpected-model",
        actualVectorName: "unexpected-vector",
      });
    }),
  );

  it.effect("LateEmbedding resolves descriptor from late-only resolver", () =>
    Effect.gen(function* () {
      const modelName = "late-model";
      const vectorName = "late-vector";

      const client = makeMockClient({
        "/embed/late/resolve": (body) => {
          expect(body).toEqual({
            method: "fastembed",
            model_name: modelName,
            vector_name: vectorName,
          });
          return {
            model_name: modelName,
            vector_name: vectorName,
            vector_size: 128,
          };
        },
      });

      const layer = LateEmbedding.Fastembed({ modelName, vectorName }).pipe(
        Layer.provide(makePythonApiClientLayer(client)),
      );

      const late = yield* LateEmbedding.pipe(Effect.provide(layer));

      expect(late.descriptor).toEqual({
        method: "fastembed",
        modelName,
        vectorName,
        vectorSize: 128,
      });
    }),
  );

  it.effect("EmbeddingProvider.make supports dense-only", () =>
    Effect.gen(function* () {
      const provider = yield* EmbeddingProvider.pipe(
        Effect.provide(EmbeddingProvider.make(denseTestService, undefined, undefined)),
      );

      expect(Option.isSome(provider.dense)).toBe(true);
      expect(Option.isNone(provider.sparse)).toBe(true);
      expect(Option.isNone(provider.late)).toBe(true);
    }),
  );

  it.effect("EmbeddingProvider.make supports sparse-only", () =>
    Effect.gen(function* () {
      const provider = yield* EmbeddingProvider.pipe(
        Effect.provide(EmbeddingProvider.make(undefined, sparseTestService, undefined)),
      );

      expect(Option.isNone(provider.dense)).toBe(true);
      expect(Option.isSome(provider.sparse)).toBe(true);
      expect(Option.isNone(provider.late)).toBe(true);
    }),
  );

  it.effect("EmbeddingProvider.make supports dense+sparse+late", () =>
    Effect.gen(function* () {
      const provider = yield* EmbeddingProvider.pipe(
        Effect.provide(
          EmbeddingProvider.make(denseTestService, sparseTestService, lateTestService),
        ),
      );

      expect(Option.isSome(provider.dense)).toBe(true);
      expect(Option.isSome(provider.sparse)).toBe(true);
      expect(Option.isSome(provider.late)).toBe(true);
    }),
  );

  it("EmbeddingProvider.make rejects late-only or empty configurations at runtime", () => {
    expect(() => (EmbeddingProvider.make as any)(undefined, undefined, lateTestService)).toThrow(
      /dense or sparse/,
    );
    expect(() => (EmbeddingProvider.make as any)(undefined, undefined, undefined)).toThrow(
      /dense or sparse/,
    );
  });
});
