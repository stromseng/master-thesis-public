import { jsonSchema } from "ai";
import * as JSONSchema from "effect/JSONSchema";
import * as S from "effect/Schema";
import { asExperimentEvaluator, type EvaluatorParams } from "./experiment_setup";
import type { EvalQuestion } from "./question-schema";

export type QuestionTaskOutput = {
  answerIds: readonly string[];
  reason: string;
};

type QuestionTaskOutputJson = {
  answerIds: string[];
  reason: string;
};

type SingleAnswerTaskOutputJson = {
  answerId: string;
  reason: string;
};

export type QuestionDatasetExpected = {
  correctOptionIds: readonly string[];
};

export type QuestionDatasetInput = Omit<EvalQuestion, "correctOptionIds">;

export type AnswerMode = "single" | "multi";

// --- Multi-answer schema (existing) ---

export const questionTaskOutputSchema = S.Struct({
  reason: S.String,
  answerIds: S.Array(S.String),
});

export const questionTaskOutputJsonSchema = jsonSchema<QuestionTaskOutputJson>(
  JSONSchema.make(questionTaskOutputSchema),
);

// --- Single-answer schema (new) ---

export const singleAnswerTaskOutputSchema = S.Struct({
  reason: S.String,
  answerId: S.String,
});

export const singleAnswerTaskOutputJsonSchema = jsonSchema<SingleAnswerTaskOutputJson>(
  JSONSchema.make(singleAnswerTaskOutputSchema),
);

// --- Answer mode helpers ---

export const getAnswerMode = (correctOptionIds: readonly string[]): AnswerMode =>
  correctOptionIds.length === 1 ? "single" : "multi";

export const getOutputEffectSchema = (mode: AnswerMode): S.Schema.AnyNoContext =>
  mode === "single" ? singleAnswerTaskOutputSchema : questionTaskOutputSchema;

export const getOutputJsonSchema = (mode: AnswerMode) =>
  mode === "single" ? singleAnswerTaskOutputJsonSchema : questionTaskOutputJsonSchema;

const getOutputSchemaPromptText = (mode: AnswerMode): string =>
  JSON.stringify(JSONSchema.make(getOutputEffectSchema(mode)), null, 2);

export const normalizeTaskOutput = (output: Record<string, unknown>): QuestionTaskOutput => {
  const reason = typeof output.reason === "string" ? output.reason : "";
  if ("answerId" in output && typeof output.answerId === "string") {
    return { answerIds: [output.answerId], reason };
  }
  const answerIds = Array.isArray(output.answerIds) ? (output.answerIds as string[]) : [];
  return { answerIds, reason };
};

// --- System prompt builder ---

export const buildSystemPrompt = ({
  answerMode,
  rag,
}: {
  answerMode: AnswerMode;
  rag: boolean;
}): string => {
  const outputSchemaText = getOutputSchemaPromptText(answerMode);
  const lines = ["You are an experienced maritime officer."];
  if (rag) {
    lines.push("Use provided context if relevant, otherwise rely on your knowledge.");
  }
  if (answerMode === "single") {
    lines.push("Your task is to answer a single answer multiple choice question.");
    lines.push("Return the selected option ID in answerId and a concise reason in reason.");
  } else {
    lines.push("Your task is to answer a multiple answer multiple choice question.");
    lines.push("Return the selected option IDs in answerIds and a concise reason in reason.");
  }
  lines.push(
    "Option IDs are dataset-defined strings. They are often simple labels like A, B, C, or D, but they may also use other letters, numbers, or mixed alphanumeric IDs.",
  );
  lines.push("Return raw JSON only.");
  lines.push("Do not wrap the JSON in markdown code fences, backticks, or any surrounding text.");
  lines.push("Follow this JSON Schema exactly:");
  lines.push(outputSchemaText);
  return lines.join("\n");
};

const normalizeOptionId = (id: string) => id.trim().toUpperCase();

export const toNormalizedUniqueIds = (ids: readonly string[]): readonly string[] => [
  ...new Set(ids.map(normalizeOptionId).filter((id) => id.length > 0)),
];

export const computeTruePositives = (
  predictedIds: readonly string[],
  expectedIds: readonly string[],
): number => {
  const expectedSet = new Set(expectedIds);
  return predictedIds.filter((id) => expectedSet.has(id)).length;
};

export const computePrecision = (tp: number, predictedCount: number): number =>
  predictedCount > 0 ? tp / predictedCount : 0;

export const computeRecall = (tp: number, expectedCount: number): number =>
  expectedCount > 0 ? tp / expectedCount : 0;

export const computeF1 = (precision: number, recall: number): number =>
  precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

const scoreLabel = (score: number): "perfect" | "none" | "partial" => {
  if (score === 1) return "perfect";
  if (score === 0) return "none";
  return "partial";
};

const isStringArray = (value: unknown): value is readonly string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const selectMetric = (
  metric: "precision" | "recall" | "f1",
  tp: number,
  predictedCount: number,
  expectedCount: number,
): number => {
  if (metric === "precision") {
    return computePrecision(tp, predictedCount);
  }
  if (metric === "recall") {
    return computeRecall(tp, expectedCount);
  }
  const precision = computePrecision(tp, predictedCount);
  const recall = computeRecall(tp, expectedCount);
  return computeF1(precision, recall);
};

const makeMetricEvaluator = (name: string, metric: "precision" | "recall" | "f1") =>
  asExperimentEvaluator({
    name,
    kind: "CODE",
    evaluate: async ({ output, expected }: EvaluatorParams) => {
      const expectedIds = (expected as QuestionDatasetExpected | undefined)?.correctOptionIds;

      if (!isStringArray(expectedIds)) {
        return {
          score: 0,
          label: "error",
          explanation: "Missing expected.correctOptionIds in dataset output.",
        };
      }

      const outputRecord = output as Record<string, unknown> | null | undefined;
      if (!outputRecord) {
        return {
          score: 0,
          label: "error",
          explanation: "Missing output.",
        };
      }

      const normalized = normalizeTaskOutput(outputRecord);
      const predictedIds = normalized.answerIds;

      if (!isStringArray(predictedIds) || predictedIds.length === 0) {
        return {
          score: 0,
          label: "error",
          explanation:
            "Missing or invalid output.answerIds/answerId (expected string or string array).",
        };
      }

      const normalizedPredictedIds = toNormalizedUniqueIds(predictedIds);
      const normalizedExpectedIds = toNormalizedUniqueIds(expectedIds);
      const tp = computeTruePositives(normalizedPredictedIds, normalizedExpectedIds);
      const predictedCount = normalizedPredictedIds.length;
      const expectedCount = normalizedExpectedIds.length;
      const score = selectMetric(metric, tp, predictedCount, expectedCount);

      return {
        score,
        label: scoreLabel(score),
        explanation: `pred=[${normalizedPredictedIds.join(", ")}], expected=[${normalizedExpectedIds.join(", ")}], tp=${tp}, ${metric}=${score.toFixed(4)}`,
        metadata: {
          tp,
          predictedCount,
          expectedCount,
          predictedIds: normalizedPredictedIds,
          expectedIds: normalizedExpectedIds,
        },
      };
    },
  });

export const questionPrecisionEvaluator = makeMetricEvaluator("question-precision", "precision");

export const questionRecallEvaluator = makeMetricEvaluator("question-recall", "recall");

export const questionF1Evaluator = makeMetricEvaluator("question-f1", "f1");

export const questionMetricEvaluators = [
  questionPrecisionEvaluator,
  questionRecallEvaluator,
  questionF1Evaluator,
] as const;
