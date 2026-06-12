import { describe, expect, it } from "@effect/vitest";
import {
  buildSystemPrompt,
  computeF1,
  computePrecision,
  computeRecall,
  computeTruePositives,
  getAnswerMode,
  normalizeTaskOutput,
  questionF1Evaluator,
  questionPrecisionEvaluator,
  questionRecallEvaluator,
  toNormalizedUniqueIds,
} from "../../evals/evaluator";

describe("question set metrics", () => {
  const deriveMetrics = (predictedIdsRaw: readonly string[], expectedIdsRaw: readonly string[]) => {
    const predictedIds = toNormalizedUniqueIds(predictedIdsRaw);
    const expectedIds = toNormalizedUniqueIds(expectedIdsRaw);
    const tp = computeTruePositives(predictedIds, expectedIds);
    const precision = computePrecision(tp, predictedIds.length);
    const recall = computeRecall(tp, expectedIds.length);
    const f1 = computeF1(precision, recall);

    return { predictedIds, expectedIds, tp, precision, recall, f1 };
  };

  it("computes exact match metrics", () => {
    const metrics = deriveMetrics(["B"], ["B"]);
    expect(metrics.precision).toBe(1);
    expect(metrics.recall).toBe(1);
    expect(metrics.f1).toBe(1);
    expect(metrics.tp).toBe(1);
  });

  it("computes partial overlap metrics", () => {
    const metrics = deriveMetrics(["A", "B"], ["B", "C"]);
    expect(metrics.precision).toBe(0.5);
    expect(metrics.recall).toBe(0.5);
    expect(metrics.f1).toBe(0.5);
    expect(metrics.tp).toBe(1);
  });

  it("returns zero metrics on no overlap", () => {
    const metrics = deriveMetrics(["A"], ["C"]);
    expect(metrics.precision).toBe(0);
    expect(metrics.recall).toBe(0);
    expect(metrics.f1).toBe(0);
    expect(metrics.tp).toBe(0);
  });

  it("returns zero metrics on empty prediction", () => {
    const metrics = deriveMetrics([], ["A"]);
    expect(metrics.precision).toBe(0);
    expect(metrics.recall).toBe(0);
    expect(metrics.f1).toBe(0);
  });

  it("normalizes case/whitespace and deduplicates IDs", () => {
    const metrics = deriveMetrics([" a ", "A", "b"], ["A", "B"]);
    expect(metrics.predictedIds).toEqual(["A", "B"]);
    expect(metrics.expectedIds).toEqual(["A", "B"]);
    expect(metrics.precision).toBe(1);
    expect(metrics.recall).toBe(1);
    expect(metrics.f1).toBe(1);
  });

  it("counts invalid IDs as false positives", () => {
    const metrics = deriveMetrics(["A", "Z"], ["A"]);
    expect(metrics.precision).toBe(0.5);
    expect(metrics.recall).toBe(1);
    expect(metrics.f1).toBe(2 / 3);
  });

  it("exposes separate precision/recall/f1 helper functions", () => {
    const tp = computeTruePositives(["A", "B"], ["B", "C"]);
    const precision = computePrecision(tp, 2);
    const recall = computeRecall(tp, 2);
    const f1 = computeF1(precision, recall);

    expect(tp).toBe(1);
    expect(precision).toBe(0.5);
    expect(recall).toBe(0.5);
    expect(f1).toBe(0.5);
  });
});

describe("question metric evaluators", () => {
  const baseArgs = {
    input: {},
    metadata: {},
    expected: { correctOptionIds: ["A", "B"] },
    output: { answerIds: ["A", "C"], reason: "test" },
  } as const;

  it("precision evaluator returns precision score", async () => {
    const result = await questionPrecisionEvaluator.evaluate(baseArgs);
    expect(result.score).toBe(0.5);
    expect(result.label).toBe("partial");
    expect(result.metadata).toBeDefined();
  });

  it("recall evaluator returns recall score", async () => {
    const result = await questionRecallEvaluator.evaluate(baseArgs);
    expect(result.score).toBe(0.5);
    expect(result.label).toBe("partial");
    expect(result.metadata).toBeDefined();
  });

  it("f1 evaluator returns f1 score", async () => {
    const result = await questionF1Evaluator.evaluate(baseArgs);
    expect(result.score).toBe(0.5);
    expect(result.label).toBe("partial");
    expect(result.metadata).toBeDefined();
  });

  it("returns error result when expected is missing", async () => {
    const result = await questionF1Evaluator.evaluate({
      input: {},
      output: { answerIds: ["A"], reason: "test" },
      expected: undefined,
      metadata: {},
    });
    expect(result.score).toBe(0);
    expect(result.label).toBe("error");
  });

  it("returns error result when output.answerIds is invalid", async () => {
    const result = await questionF1Evaluator.evaluate({
      input: {},
      output: { answerIds: "A", reason: "test" } as unknown as {
        answerIds: string[];
        reason: string;
      },
      expected: { correctOptionIds: ["A"] },
      metadata: {},
    });
    expect(result.score).toBe(0);
    expect(result.label).toBe("error");
  });

  it("handles single-answer format (answerId string)", async () => {
    const result = await questionF1Evaluator.evaluate({
      input: {},
      output: { answerId: "A", reason: "test" },
      expected: { correctOptionIds: ["A"] },
      metadata: {},
    });
    expect(result.score).toBe(1);
    expect(result.label).toBe("perfect");
  });

  it("handles single-answer format with wrong answer", async () => {
    const result = await questionPrecisionEvaluator.evaluate({
      input: {},
      output: { answerId: "B", reason: "test" },
      expected: { correctOptionIds: ["A"] },
      metadata: {},
    });
    expect(result.score).toBe(0);
    expect(result.label).toBe("none");
  });

  it("handles single-answer format with case normalization", async () => {
    const result = await questionRecallEvaluator.evaluate({
      input: {},
      output: { answerId: " a ", reason: "test" },
      expected: { correctOptionIds: ["A"] },
      metadata: {},
    });
    expect(result.score).toBe(1);
    expect(result.label).toBe("perfect");
  });

  it("returns error when output is null", async () => {
    const result = await questionF1Evaluator.evaluate({
      input: {},
      output: null,
      expected: { correctOptionIds: ["A"] },
      metadata: {},
    });
    expect(result.score).toBe(0);
    expect(result.label).toBe("error");
  });
});

describe("normalizeTaskOutput", () => {
  it("normalizes single-answer format", () => {
    const result = normalizeTaskOutput({ answerId: "A", reason: "test" });
    expect(result).toEqual({ answerIds: ["A"], reason: "test" });
  });

  it("normalizes multi-answer format", () => {
    const result = normalizeTaskOutput({ answerIds: ["A", "B"], reason: "test" });
    expect(result).toEqual({ answerIds: ["A", "B"], reason: "test" });
  });

  it("handles missing reason", () => {
    const result = normalizeTaskOutput({ answerId: "A" });
    expect(result).toEqual({ answerIds: ["A"], reason: "" });
  });

  it("handles missing answerIds", () => {
    const result = normalizeTaskOutput({ reason: "test" });
    expect(result).toEqual({ answerIds: [], reason: "test" });
  });
});

describe("getAnswerMode", () => {
  it("returns single for one correct option", () => {
    expect(getAnswerMode(["A"])).toBe("single");
  });

  it("returns multi for multiple correct options", () => {
    expect(getAnswerMode(["A", "B"])).toBe("multi");
  });

  it("returns multi for empty array", () => {
    expect(getAnswerMode([])).toBe("multi");
  });
});

describe("buildSystemPrompt", () => {
  it("builds single-answer prompt without RAG", () => {
    const prompt = buildSystemPrompt({ answerMode: "single", rag: false });
    expect(prompt).toContain("single answer");
    expect(prompt).toContain("answerId");
    expect(prompt).toContain("A, B, C, or D");
    expect(prompt).toContain("Return raw JSON only.");
    expect(prompt).toContain('"type": "object"');
    expect(prompt).toContain('"answerId"');
    expect(prompt).not.toContain("context");
  });

  it("builds multi-answer prompt without RAG", () => {
    const prompt = buildSystemPrompt({ answerMode: "multi", rag: false });
    expect(prompt).toContain("multiple answer");
    expect(prompt).toContain("answerIds");
    expect(prompt).toContain("other letters, numbers, or mixed alphanumeric IDs");
    expect(prompt).toContain("Return raw JSON only.");
    expect(prompt).toContain('"type": "object"');
    expect(prompt).toContain('"answerIds"');
  });

  it("builds single-answer prompt with RAG", () => {
    const prompt = buildSystemPrompt({ answerMode: "single", rag: true });
    expect(prompt).toContain("single answer");
    expect(prompt).toContain("context");
    expect(prompt).toContain("Follow this JSON Schema exactly:");
  });

  it("builds multi-answer prompt with RAG", () => {
    const prompt = buildSystemPrompt({ answerMode: "multi", rag: true });
    expect(prompt).toContain("multiple answer");
    expect(prompt).toContain("context");
    expect(prompt).toContain("markdown code fences");
  });
});
