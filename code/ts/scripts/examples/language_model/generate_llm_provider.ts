import { generateText as aiGenerateText } from "ai";
import { Effect, Layer, Logger } from "effect";
import { LanguageModelError } from "../../../src/services/LanguageModel";
import { LLMProvider, LiteLLMModelId } from "../../../src/services/LLMProvider";

const modelId = LiteLLMModelId.GptOss120B;

const program = Effect.gen(function* () {
  const { provider } = yield* LLMProvider;

  const model = yield* Effect.try({
    try: () => provider.languageModel(modelId),
    catch: (error) =>
      LanguageModelError.make({
        reason: `Failed to resolve language model: ${modelId}`,
        cause: error,
      }),
  });

  const result = yield* Effect.tryPromise({
    try: () =>
      aiGenerateText({
        model,
        prompt: "Explain what a service layer is in Effect.ts, in one sentences.",
      }),
    catch: (error) =>
      LanguageModelError.make({
        reason: "Failed to generate text",
        cause: error,
      }),
  });

  yield* Effect.log(`LLMProvider: ${result.text}`);
});

const appLayer = Layer.mergeAll(LLMProvider.IdunLiteLLM, Logger.pretty);

await Effect.runPromise(program.pipe(Effect.provide(appLayer)));
