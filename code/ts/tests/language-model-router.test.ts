import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import { describe, expect, it } from "@effect/vitest";
import { Effect, Layer } from "effect";
import {
  ConfiguredLanguageModelMismatchError,
  generateText,
  LanguageModel,
  type RoutedLanguageModelRoute,
  RoutedLanguageModelLayerFromLayers,
} from "../src/services/LanguageModel";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const makeGenerateResult = (text: string): LanguageModelV3GenerateResult => ({
  content: [{ type: "text", text }],
  finishReason: {
    unified: "stop",
    raw: "stop",
  },
  usage: {
    inputTokens: {
      total: undefined,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: undefined,
      text: undefined,
      reasoning: undefined,
    },
  },
  warnings: [],
});

const makeFakeLayerRoute = ({
  label,
  servedModelId,
  maxConcurrency,
  responseText,
  delayMs,
  beforeCall,
  afterCall,
}: {
  readonly label: string;
  readonly servedModelId: string;
  readonly maxConcurrency: number;
  readonly responseText: string;
  readonly delayMs: number;
  readonly beforeCall?: () => void;
  readonly afterCall?: () => void;
}): RoutedLanguageModelRoute => {
  const model: LanguageModelV3 = {
    specificationVersion: "v3",
    provider: label,
    modelId: servedModelId,
    supportedUrls: {},
    doGenerate: async (
      _options: LanguageModelV3CallOptions,
    ): Promise<LanguageModelV3GenerateResult> => {
      beforeCall?.();
      try {
        await sleep(delayMs);
        return makeGenerateResult(responseText);
      } finally {
        afterCall?.();
      }
    },
    doStream: async (): Promise<LanguageModelV3StreamResult> => {
      throw new Error("Streaming is not used in this test");
    },
  };

  return {
    label,
    maxConcurrency,
    layer: Layer.succeed(LanguageModel, model),
  };
};

describe("RoutedLanguageModelLayerFromLayers", () => {
  it.effect("distributes queued requests to whichever endpoint becomes free first", () =>
    Effect.gen(function* () {
      const hits = new Map<string, number>();
      const increment = (label: string) => hits.set(label, (hits.get(label) ?? 0) + 1);

      const routes = [
        makeFakeLayerRoute({
          label: "a",
          servedModelId: "shared-model",
          maxConcurrency: 1,
          responseText: "a",
          delayMs: 50,
          beforeCall: () => increment("a"),
        }),
        makeFakeLayerRoute({
          label: "b",
          servedModelId: "shared-model",
          maxConcurrency: 1,
          responseText: "b",
          delayMs: 5,
          beforeCall: () => increment("b"),
        }),
      ] as const;

      const texts = yield* Effect.all(
        [
          generateText({ prompt: "one" }),
          generateText({ prompt: "two" }),
          generateText({ prompt: "three" }),
        ],
        { concurrency: "unbounded" },
      ).pipe(
        Effect.provide(RoutedLanguageModelLayerFromLayers("shared-model", routes)),
        Effect.scoped,
      );

      expect(texts.map((result) => result.text).sort()).toEqual(["a", "b", "b"]);
      expect(hits.get("a")).toBe(1);
      expect(hits.get("b")).toBe(2);
    }),
  );

  it.effect("respects the configured outstanding-request limit per endpoint", () =>
    Effect.gen(function* () {
      let inFlight = 0;
      let maxInFlight = 0;

      const routes = [
        makeFakeLayerRoute({
          label: "concurrency",
          servedModelId: "shared-model",
          maxConcurrency: 2,
          responseText: "ok",
          delayMs: 20,
          beforeCall: () => {
            inFlight += 1;
            maxInFlight = Math.max(maxInFlight, inFlight);
          },
          afterCall: () => {
            inFlight -= 1;
          },
        }),
      ] as const;

      yield* Effect.all(
        Array.from({ length: 5 }, (_, index) => generateText({ prompt: `prompt-${index}` })),
        { concurrency: "unbounded" },
      ).pipe(
        Effect.provide(RoutedLanguageModelLayerFromLayers("shared-model", routes)),
        Effect.scoped,
      );

      expect(maxInFlight).toBe(2);
    }),
  );

  it.effect("fails when configured endpoints do not all serve the requested model", () =>
    Effect.gen(function* () {
      const routes = [
        makeFakeLayerRoute({
          label: "a",
          servedModelId: "shared-model",
          maxConcurrency: 1,
          responseText: "a",
          delayMs: 1,
        }),
        makeFakeLayerRoute({
          label: "b",
          servedModelId: "different-model",
          maxConcurrency: 1,
          responseText: "b",
          delayMs: 1,
        }),
      ] as const;

      const result = yield* Effect.either(
        generateText({ prompt: "mismatch" }).pipe(
          Effect.provide(RoutedLanguageModelLayerFromLayers("shared-model", routes)),
          Effect.scoped,
        ),
      );

      expect(result._tag).toBe("Left");
      if (result._tag !== "Left") return;
      expect(result.left._tag).toBe("ConfiguredLanguageModelMismatchError");
      expect(result.left).toBeInstanceOf(ConfiguredLanguageModelMismatchError);
    }),
  );
});
