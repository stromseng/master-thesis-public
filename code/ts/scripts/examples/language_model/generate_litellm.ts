import { Effect, Layer, Logger } from "effect";
import { generateText, LiteLLMLanguageModelLayer } from "../../../src/services/LanguageModel";

const program = Effect.gen(function* () {
  const litellmResult = yield* generateText({
    prompt: "Give me a one-sentence summary of Bun the js runtime.",
  });

  yield* Effect.log(`LiteLLM: ${litellmResult.text}`);
});

const appLayer = Layer.mergeAll(LiteLLMLanguageModelLayer.gptOss120b, Logger.pretty);

await Effect.runPromise(program.pipe(Effect.provide(appLayer)));
