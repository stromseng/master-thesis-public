import { jsonSchema } from "ai";
import { Effect, Runtime } from "effect";
import * as JSONSchema from "effect/JSONSchema";
import * as S from "effect/Schema";
import { asExperimentEvaluator, type EvaluatorParams } from "./experiment_setup";
import type { EvaluationResult } from "@arizeai/phoenix-client/types/experiments";
import { OpenEndedEvalQuestion } from "./open-ended-question-schema";
import { Judge } from "../src/services/Judge";

export const OpenEndedDatasetInputSchema = OpenEndedEvalQuestion.pipe(
  S.omit("referenceCorrectAnswers", "referenceIncorrectAnswers"),
);
export type OpenEndedDatasetInput = S.Schema.Type<typeof OpenEndedDatasetInputSchema>;

export const OpenEndedDatasetExpectedSchema = OpenEndedEvalQuestion.pipe(
  S.pick("referenceCorrectAnswers", "referenceIncorrectAnswers"),
);
export type OpenEndedDatasetExpected = S.Schema.Type<typeof OpenEndedDatasetExpectedSchema>;

export const openEndedTaskOutputSchema = S.Struct({
  reason: S.String,
  answer: S.String,
});

export const openEndedTaskOutputJsonSchema = jsonSchema<
  S.Schema.Type<typeof openEndedTaskOutputSchema>
>(JSONSchema.make(openEndedTaskOutputSchema));

export const buildOpenEndedSystemPrompt = ({ rag }: { rag: boolean }): string => {
  const outputSchemaText = JSON.stringify(JSONSchema.make(openEndedTaskOutputSchema), null, 2);
  const lines = ["You are an experienced maritime officer."];
  if (rag) {
    lines.push("Use provided context if relevant, otherwise rely on your knowledge.");
  }
  lines.push("Your task is to answer an open-ended maritime question.");
  lines.push("Return the answer text in answer and a concise reason in reason.");
  lines.push("Return raw JSON only.");
  lines.push("Do not wrap the JSON in markdown code fences, backticks, or surrounding text.");
  lines.push("Follow this JSON Schema exactly:");
  lines.push(outputSchemaText);
  return lines.join("\n");
};

export const buildOpenEndedQuestionPrompt = (question: OpenEndedDatasetInput) =>
  `Question: ${question.questionText}`;

const scoreLabel = (score: number): "perfect" | "none" | "partial" => {
  if (score === 1) return "perfect";
  if (score === 0) return "none";
  return "partial";
};

class MissingOutputError extends S.TaggedError<MissingOutputError>()("MissingOutputError", {
  reason: S.String,
}) {}

class InvalidAnswerError extends S.TaggedError<InvalidAnswerError>()("InvalidAnswerError", {
  reason: S.String,
}) {}

const evaluateOpenEndedCorrectness = Effect.fn("openEnded.evaluateCorrectness")(function* (
  params: EvaluatorParams,
) {
  const input = yield* S.decodeUnknown(OpenEndedDatasetInputSchema)(params.input);
  const expected = yield* S.decodeUnknown(OpenEndedDatasetExpectedSchema)(params.expected);

  if (!params.output) {
    return yield* new MissingOutputError({ reason: "Missing output." });
  }

  const output = yield* S.decodeUnknown(openEndedTaskOutputSchema)(params.output);
  if (output.answer.trim().length === 0) {
    return yield* new InvalidAnswerError({ reason: "Missing or invalid output.answer." });
  }

  const judge = yield* Judge;

  const judgment = yield* judge.factuality({
    input: input.questionText,
    expected: expected.referenceCorrectAnswers.join("; "),
    output: output.answer,
    incorrectAnswers: expected.referenceIncorrectAnswers,
  });

  return {
    score: judgment.score,
    label: scoreLabel(judgment.score),
    explanation: judgment.reason,
    metadata: {
      answer: output.answer,
      reason: output.reason,
      judgment,
    },
  };
});

export const makeOpenEndedCorrectnessEvaluator = (runtime: Runtime.Runtime<Judge>) => {
  const evaluate = evaluateOpenEndedCorrectness;

  return asExperimentEvaluator({
    name: "open-ended-correctness",
    kind: "CODE",
    evaluate: (params: EvaluatorParams) =>
      Runtime.runPromise(runtime)(
        evaluate(params) as Effect.Effect<EvaluationResult, unknown, Judge>,
      ),
  });
};
