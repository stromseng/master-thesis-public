import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { Context, Effect, Layer } from "effect";
import { Judge, JudgeError, JudgeLayer, LayerDefault } from "../src/services/Judge";
import { LanguageModel, LiteLLMLanguageModelLayer } from "../src/services/LanguageModel";
import { LiteLLMModelId } from "../src/services/LLMProvider";

const extractPromptText = (prompt: LanguageModelV3CallOptions["prompt"]) =>
  prompt
    .map((message) => {
      if (typeof message.content === "string") return message.content;
      return message.content
        .map((part) => (part.type === "text" || part.type === "reasoning" ? part.text : ""))
        .join("\n");
    })
    .join("\n");

const makeFakeJudgeModel = (responseText: string, promptLog?: string[]): LanguageModelV3 => ({
  specificationVersion: "v3",
  provider: "test",
  modelId: "fake-judge-model",
  supportedUrls: {},
  doGenerate: async ({ prompt }): Promise<LanguageModelV3GenerateResult> => {
    promptLog?.push(extractPromptText(prompt));
    return {
      content: [{ type: "text", text: responseText }],
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
  },
  doStream: async (): Promise<LanguageModelV3StreamResult> => {
    throw new Error("Streaming is not used in this test");
  },
});

const makeJudgeLayer = (responseText: string, promptLog?: string[]) =>
  JudgeLayer.pipe(
    Layer.provide(
      Layer.succeed(
        LanguageModel,
        makeFakeJudgeModel(responseText, promptLog) as unknown as typeof LanguageModel.Service,
      ),
    ),
  );

const runFactuality = (responseText: string, promptLog?: string[]) =>
  Effect.gen(function* () {
    const judge = yield* Judge;
    return yield* judge.factuality({
      input: "What is the stand-on vessel in a crossing situation?",
      expected: "The vessel that has the other vessel on her port side.",
      output: "The vessel with the other vessel on her port side is the stand-on vessel.",
      incorrectAnswers: [],
    });
  }).pipe(Effect.provide(makeJudgeLayer(responseText, promptLog)));

describe("Judge.factuality", () => {
  it.each([
    ["A", 0.5],
    ["B", 1],
    ["C", 1],
    ["D", 0],
    ["E", 1],
  ] as const)("maps choice %s to score %s", async (choice, score) => {
    const result = await Effect.runPromise(
      runFactuality(JSON.stringify({ choice, reason: `Selected ${choice}` })),
    );

    expect(result).toEqual({
      choice,
      score,
      reason: `Selected ${choice}`,
    });
  });

  it("fails with JudgeError on malformed judge model output", async () => {
    const result = await Effect.runPromise(
      Effect.either(runFactuality(JSON.stringify({ choice: "Z", reason: "Invalid choice" }))),
    );

    expect(result._tag).toBe("Left");
    if (result._tag !== "Left") return;
    expect(result.left).toBeInstanceOf(JudgeError);
    if (result.left._tag !== "JudgeError") return;
    expect(result.left.reason).toContain("Judge returned malformed factuality response");
  });

  it("includes the question, expert answer, and submission in the judge prompt", async () => {
    const promptLog: string[] = [];

    await Effect.runPromise(
      runFactuality(JSON.stringify({ choice: "C", reason: "Exact match" }), promptLog),
    );

    expect(promptLog).toHaveLength(1);
    expect(promptLog[0]).toContain(
      "<question>What is the stand-on vessel in a crossing situation?</question>",
    );
    expect(promptLog[0]).toContain(
      "<expert>The vessel that has the other vessel on her port side.</expert>",
    );
    expect(promptLog[0]).toContain(
      "<submission>The vessel with the other vessel on her port side is the stand-on vessel.</submission>",
    );
  });
});

describe("LayerDefault", () => {
  const originalApiKey = process.env.LITE_LLM_API_KEY;
  const originalBaseUrl = process.env.LITE_LLM_BASE_URL;

  beforeEach(() => {
    process.env.LITE_LLM_API_KEY = "test-key";
    process.env.LITE_LLM_BASE_URL = "https://example.com/v1";
  });

  afterEach(() => {
    if (originalApiKey === undefined) {
      delete process.env.LITE_LLM_API_KEY;
    } else {
      process.env.LITE_LLM_API_KEY = originalApiKey;
    }

    if (originalBaseUrl === undefined) {
      delete process.env.LITE_LLM_BASE_URL;
    } else {
      process.env.LITE_LLM_BASE_URL = originalBaseUrl;
    }
  });

  it("provides Judge backed by the default LiteLLM Qwen language model", async () => {
    const modelLayer = Layer.effect(
      LanguageModel,
      Effect.gen(function* () {
        const context = yield* Effect.scoped(Layer.build(LiteLLMLanguageModelLayer.qwen35122ba10b));
        return Context.get(context, LanguageModel);
      }),
    );

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const context = yield* Effect.scoped(Layer.build(modelLayer));
        const model = Context.get(context, LanguageModel);
        return {
          modelId: (model as { modelId?: string }).modelId,
          provider: (model as { config?: { provider?: string } }).config?.provider,
        };
      }),
    );

    expect(result).toEqual({
      modelId: LiteLLMModelId.Qwen35122ba10b,
      provider: "litellm.chat",
    });
  });

  it("provides a working Judge service", async () => {
    const judge = await Effect.runPromise(Judge.pipe(Effect.provide(LayerDefault)));

    expect(judge.factuality).toBeDefined();
  });
});
