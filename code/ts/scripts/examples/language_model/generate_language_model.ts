import { generateText as aiGenerateText } from "ai";
import { Effect, Layer, Logger } from "effect";
import {
  LanguageModel,
  LanguageModelError,
  layer as languageModelLayer,
} from "../../../src/services/LanguageModel";
import { LLMProvider, LiteLLMModelId } from "../../../src/services/LLMProvider";

const modelId = LiteLLMModelId.GptOss120B;

const program = Effect.gen(function* () {
  const model = yield* LanguageModel;
  const result = yield* Effect.tryPromise({
    try: () =>
      aiGenerateText({
        model,
        prompt: "Explain how Effect services are provided, in one sentence.",
      }),
    catch: (error) =>
      LanguageModelError.make({
        reason: "Failed to generate text",
        cause: error,
      }),
  });

  yield* Effect.log(`LanguageModel: ${result.text}`);
});

const appLayer = Layer.mergeAll(
  languageModelLayer(modelId).pipe(Layer.provide(LLMProvider.IdunLiteLLM)),
  Logger.pretty,
);

await Effect.runPromise(program.pipe(Effect.provide(appLayer)));
