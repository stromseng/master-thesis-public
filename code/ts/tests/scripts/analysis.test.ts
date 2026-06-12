import { describe, expect, it } from "@effect/vitest";
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import { Effect, Layer } from "effect";
import {
  chunkByTokenBudget,
  computeFailureModeStats,
  recursiveSummarizeMode,
  type AnalysisResult,
  validateResumeCheckpoint,
} from "../../scripts/analysis/analyze-failure-modes";
import { LanguageModel } from "../../src/services/LanguageModel";

const repeatedWords = (word: string, count: number) =>
  Array.from({ length: count }, () => word).join(" ");

const makeAnalysisResult = (index: number): AnalysisResult => ({
  sourceDataset: "pei2024-uk",
  exampleId: `example-${index}`,
  questionId: `question-${index}`,
  questionKey: `pei2024-uk::question-${index}`,
  modelId: "openai/gpt-oss-120b",
  questionText: `Question ${index} ${repeatedWords(`question${index}`, 20)}`,
  questionImages: [],
  options: [
    { id: "A", text: "Alpha", images: [] },
    { id: "B", text: "Bravo", images: [] },
  ],
  modelAnswer: ["A"],
  modelReason: "Reason",
  correctAnswer: ["B"],
  analysis: {
    failureMode: "knowledge_gap",
    explanation: repeatedWords(`explanation${index}`, 60),
    difficultyFactors: ["close distractors"],
    requiredKnowledge: ["rule recall"],
  },
});

const extractPromptText = (prompt: LanguageModelV3CallOptions["prompt"]) =>
  prompt
    .map((message) => {
      if (typeof message.content === "string") return message.content;
      return message.content
        .map((part) => (part.type === "text" || part.type === "reasoning" ? part.text : ""))
        .join("\n");
    })
    .join("\n");

const makeFakeLanguageModel = (callLog: string[]): LanguageModelV3 => {
  let chunkCallCount = 0;
  let mergeCallCount = 0;
  let compressCallCount = 0;

  return {
    specificationVersion: "v3",
    provider: "test",
    modelId: "fake-summary-model",
    supportedUrls: {},
    doGenerate: async ({ prompt }): Promise<LanguageModelV3GenerateResult> => {
      const promptText = extractPromptText(prompt);
      callLog.push(promptText);

      let text: string;
      if (promptText.includes("Compress this partial summary")) {
        compressCallCount += 1;
        text = `compressed-${compressCallCount} ${repeatedWords(`compact${compressCallCount}`, 12)}`;
      } else if (promptText.includes('partial summaries for "knowledge_gap"')) {
        mergeCallCount += 1;
        text = `merged-${mergeCallCount} ${repeatedWords(`merge${mergeCallCount}`, 70)}`;
      } else if (promptText.includes('failures for "knowledge_gap"')) {
        chunkCallCount += 1;
        text = `chunk-${chunkCallCount} ${repeatedWords(`detail${chunkCallCount}`, 70)}`;
      } else {
        throw new Error(`Unexpected prompt:\n${promptText}`);
      }

      const result: LanguageModelV3GenerateResult = {
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
      };
      return result;
    },
    doStream: async (): Promise<LanguageModelV3StreamResult> => {
      throw new Error("Streaming is not used in this test");
    },
  };
};

describe("failure mode recursive summarization", () => {
  it("splits analyses into multiple chunks under a small token budget", () => {
    const analyses = Array.from({ length: 4 }, (_, index) => makeAnalysisResult(index + 1));

    const chunks = chunkByTokenBudget("knowledge_gap", "Needs domain knowledge", analyses, 120);

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.flat().length).toBe(analyses.length);
  });

  it.effect("recursively compresses and merges summaries through the LanguageModel service", () =>
    Effect.gen(function* () {
      const analyses = Array.from({ length: 4 }, (_, index) => makeAnalysisResult(index + 1));
      const callLog: string[] = [];
      const fakeModel = makeFakeLanguageModel(callLog);
      const languageModelLayer = Layer.succeed(LanguageModel, fakeModel);

      const summary = yield* recursiveSummarizeMode(
        "knowledge_gap",
        "Needs domain knowledge",
        analyses,
        120,
      ).pipe(Effect.provide(languageModelLayer));

      const summarizeCalls = callLog.filter((prompt) =>
        prompt.includes('failures for "knowledge_gap"'),
      );
      const mergeCalls = callLog.filter((prompt) =>
        prompt.includes('partial summaries for "knowledge_gap"'),
      );
      const compressCalls = callLog.filter((prompt) =>
        prompt.includes("Compress this partial summary"),
      );

      expect(summarizeCalls.length).toBeGreaterThan(1);
      expect(mergeCalls.length).toBeGreaterThan(1);
      expect(compressCalls.length).toBeGreaterThan(0);
      expect(summary).toContain("merged-");
    }),
  );
});

describe("failure mode stats", () => {
  it("includes aggregates for each dataset/model pair", () => {
    const analyses: AnalysisResult[] = [
      makeAnalysisResult(1),
      makeAnalysisResult(2),
      {
        ...makeAnalysisResult(3),
        sourceDataset: "crewcn",
        modelId: "moonshotai/Kimi-K2.5",
        analysis: {
          failureMode: "interpretation_or_confusion",
          explanation: "Confused similar terms",
          difficultyFactors: ["near-duplicate options"],
          requiredKnowledge: ["signal distinction"],
        },
      },
      {
        ...makeAnalysisResult(4),
        sourceDataset: "pei2024-uk",
        modelId: "moonshotai/Kimi-K2.5",
        analysis: {
          failureMode: "reasoning_error",
          explanation: "Failed a comparison step",
          difficultyFactors: ["requires multi-step elimination"],
          requiredKnowledge: ["rule comparison"],
        },
      },
    ];

    const stats = computeFailureModeStats(analyses);

    expect(stats.overall.length).toBeGreaterThan(0);
    expect(stats.byDataset["pei2024-uk"]).toBeDefined();
    expect(stats.byModel["moonshotai/Kimi-K2.5"]).toBeDefined();
    expect(stats.byDatasetModel["pei2024-uk"]?.["openai/gpt-oss-120b"]).toEqual([
      { mode: "knowledge_gap", count: 2 },
    ]);
    expect(stats.byDatasetModel["pei2024-uk"]?.["moonshotai/Kimi-K2.5"]).toEqual([
      { mode: "reasoning_error", count: 1 },
    ]);
    expect(stats.byDatasetModel["crewcn"]?.["moonshotai/Kimi-K2.5"]).toEqual([
      { mode: "interpretation_or_confusion", count: 1 },
    ]);
  });
});

describe("resume checkpoint validation", () => {
  it.effect("accepts matching checkpoint metadata", () =>
    validateResumeCheckpoint(
      "checkpoint.json",
      {
        analysisModel: "openai/gpt-oss-120b",
        analysisMode: "text",
        sourceDatasets: ["crewcn", "pei2024-uk"],
        includedModelIds: ["moonshotai/Kimi-K2.5"],
        excludedMetadataKeywords: ["RAG"],
        includeRag: false,
        limitPerDataset: 10,
      },
      {
        analysisModelId: "openai/gpt-oss-120b",
        analysisMode: "text",
        sourceDatasets: ["pei2024-uk", "crewcn"],
        includedModelIds: ["moonshotai/Kimi-K2.5"],
        excludedMetadataKeywords: ["RAG"],
        includeRag: false,
        limitPerDataset: 10,
      },
    ),
  );

  it.effect("rejects mismatched checkpoint metadata", () =>
    Effect.gen(function* () {
      const result = yield* validateResumeCheckpoint(
        "checkpoint.json",
        {
          analysisModel: "openai/gpt-oss-120b",
          analysisMode: "text",
          sourceDatasets: ["pei2024-uk"],
          includedModelIds: ["moonshotai/Kimi-K2.5"],
          excludedMetadataKeywords: ["RAG"],
          includeRag: false,
          limitPerDataset: 10,
        },
        {
          analysisModelId: "openai/gpt-oss-120b",
          analysisMode: "text",
          sourceDatasets: ["crewcn"],
          includedModelIds: ["moonshotai/Kimi-K2.5"],
          excludedMetadataKeywords: ["RAG"],
          includeRag: false,
          limitPerDataset: 10,
        },
      ).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("rejects mismatched checkpoint analysis mode", () =>
    Effect.gen(function* () {
      const result = yield* validateResumeCheckpoint(
        "checkpoint.json",
        {
          analysisModel: "openai/gpt-oss-120b",
          analysisMode: "text",
          sourceDatasets: ["pei2024-uk"],
          includedModelIds: ["moonshotai/Kimi-K2.5"],
          excludedMetadataKeywords: [],
          includeRag: false,
          limitPerDataset: 10,
        },
        {
          analysisModelId: "openai/gpt-oss-120b",
          analysisMode: "multimodal",
          sourceDatasets: ["pei2024-uk"],
          includedModelIds: ["moonshotai/Kimi-K2.5"],
          excludedMetadataKeywords: [],
          includeRag: false,
          limitPerDataset: 10,
        },
      ).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );
});
