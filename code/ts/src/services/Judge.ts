import * as ai from "ai";
import { jsonSchema } from "ai";
import { Context, Effect, Layer, Schema } from "effect";
import * as JSONSchema from "effect/JSONSchema";
import { JudgeLanguageModelLayer, LanguageModel } from "./LanguageModel";

export class JudgeError extends Schema.TaggedError<JudgeError>()("JudgeError", {
  reason: Schema.String,
  cause: Schema.Defect,
}) {}

const FactualityChoice = Schema.Literal("A", "B", "C", "D", "E");

export type FactualityChoice = Schema.Schema.Type<typeof FactualityChoice>;
export type FactualityScore = 0 | 0.5 | 1;

export type FactualityInput = {
  input: string;
  expected: string;
  output: string;
  incorrectAnswers: readonly string[];
};

export type FactualityResult = {
  reason: string;
  choice: FactualityChoice;
  score: FactualityScore;
};

const FactualityJudgeResponse = Schema.Struct({
  reason: Schema.String,
  choice: FactualityChoice,
});

type FactualityJudgeResponse = Schema.Schema.Type<typeof FactualityJudgeResponse>;

const factualityJudgeResponseJsonSchema = jsonSchema<FactualityJudgeResponse>(
  JSONSchema.make(FactualityJudgeResponse),
);

const CHOICE_SCORES = {
  A: 0.5,
  B: 1,
  C: 1,
  D: 0,
  E: 1,
} as const satisfies Record<FactualityChoice, FactualityScore>;

const buildFactualityPrompt = ({
  input,
  expected,
  output,
  incorrectAnswers,
}: FactualityInput): string => {
  const incorrectSection =
    incorrectAnswers && incorrectAnswers.length > 0
      ? `<known-incorrect-answers>${incorrectAnswers.join("; ")}</known-incorrect-answers>\n`
      : "";

  return `You are comparing a submitted answer to an expert answer on a given question. Here is the data:
<data>
<question>${input}</question>
<expert>${expected}</expert>
<submission>${output}</submission>
${incorrectSection}</data>

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies.${incorrectAnswers && incorrectAnswers.length > 0 ? " If the submission matches or is closer to a known incorrect answer than to the expert answer, it must be treated as a disagreement (D)." : ""} Answer the question by selecting one of the following options:
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.

Return JSON with:
- choice: one of "A", "B", "C", "D", or "E"
- reason: a concise explanation for the selected choice`;
};

const makeFactuality = (
  model: ai.LanguageModel,
): ((input: FactualityInput) => Effect.Effect<FactualityResult, JudgeError>) => {
  const impl = Effect.fn("Judge.factuality")(function* ({
    input,
    expected,
    output,
    incorrectAnswers,
  }: FactualityInput) {
    const response = yield* Effect.tryPromise({
      try: () =>
        ai.generateObject({
          model,
          prompt: buildFactualityPrompt({ input, expected, output, incorrectAnswers }),
          schema: factualityJudgeResponseJsonSchema,
          experimental_telemetry: { isEnabled: true },
        }),
      catch: (cause) =>
        new JudgeError({
          reason: "Failed to generate factuality judgment",
          cause,
        }),
    });

    const parsed = yield* Schema.decodeUnknown(FactualityJudgeResponse)(response.object).pipe(
      Effect.mapError(
        (cause) =>
          new JudgeError({
            reason: "Judge returned malformed factuality response",
            cause,
          }),
      ),
    );

    return {
      choice: parsed.choice,
      score: CHOICE_SCORES[parsed.choice],
      reason: parsed.reason,
    } satisfies FactualityResult;
  });

  return (input) => impl(input);
};

export class Judge extends Context.Tag("@app/Judge")<
  Judge,
  {
    readonly factuality: (input: FactualityInput) => Effect.Effect<FactualityResult, JudgeError>;
  }
>() {}

export const JudgeLayer = Layer.effect(
  Judge,
  Effect.gen(function* () {
    const model = yield* LanguageModel;
    yield* Effect.logInfo("Judge model: %s", typeof model === "string" ? model : model.modelId);
    return Judge.of({
      factuality: makeFactuality(model),
    });
  }),
);

export const LayerDefault = JudgeLayer.pipe(Layer.provide(JudgeLanguageModelLayer));
