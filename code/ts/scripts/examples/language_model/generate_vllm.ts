import { Effect, Layer, Logger } from "effect";
import { z } from "zod";
import {
  generateText,
  generateObject,
  VLLMLanguageModelLayer,
} from "../../../src/services/LanguageModel";

const PersonSchema = z.object({
  name: z.string(),
  age: z.number(),
  occupation: z.string(),
});

const program = Effect.gen(function* () {
  // Test text generation
  const textResult = yield* generateText({
    prompt: "Give me a one-sentence summary of Bun the js runtime.",
  });
  yield* Effect.log(`Text generation: ${textResult.text}`);

  // Test structured output
  const objectResult = yield* generateObject({
    schema: PersonSchema,
    prompt: "Generate a fictional person with a name, age, and occupation.",
  });
  // @effect-diagnostics-next-line preferSchemaOverJson:off
  yield* Effect.log(`Structured output: ${JSON.stringify(objectResult.object, null, 2)}`);
});

const appLayer = Layer.mergeAll(VLLMLanguageModelLayer.qwen25_1_5BInstruct, Logger.pretty);

await Effect.runPromise(program.pipe(Effect.provide(appLayer)));
