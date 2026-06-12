import { Effect, Layer } from "effect";
import type { Client } from "../../src/generated/python-api/client";
import { PythonApiClient } from "../../src/services/PythonApiClient";
import { DenseEmbedding } from "../../src/services/embeddings";

export const makePythonApiClientLayer = (client: Client) =>
  Layer.succeed(PythonApiClient, new PythonApiClient({ client }));

export const makeDenseTestLayer = (suffix: string) => {
  const service = makeDenseTestService(suffix);
  return Layer.succeed(DenseEmbedding, service);
};

export const makeDenseTestService = (suffix: string) => {
  const vectorName = `dense-test-${suffix}`;
  return DenseEmbedding.of({
    descriptor: {
      method: "test",
      modelName: `dense-model-${suffix}`,
      vectorName,
      vectorSize: 4,
    },
    embed: (text: string) => Effect.succeed([text.length, 1, 2, 3]),
    embedBatch: (texts: string[]) => Effect.succeed(texts.map((text) => [text.length, 1, 2, 3])),
  });
};
