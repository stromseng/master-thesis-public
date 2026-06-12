import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { Provider as AiProvider } from "ai";
import { Config, Context, Effect, Layer, Redacted, Schema } from "effect";

export class CreateProviderError extends Schema.TaggedError<CreateProviderError>()(
  "CreateProviderError",
  {
    providerName: Schema.String,
    baseUrl: Schema.String,
    cause: Schema.Defect,
  },
) {}

export type LLMProviderType = Pick<AiProvider, "languageModel" | "embeddingModel">;

export const LiteLLMModelId = {
  MistralLarge3Instruct: "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",
  GptOss120B: "openai/gpt-oss-120b",
  Glm47Fp8: "zai-org/GLM-4.7-FP8",
  kimiK25: "moonshotai/Kimi-K2.5",
  Qwen35122ba10b: "Qwen/Qwen3.5-122B-A10B-FP8",
  NorwAIMagistral24BReasoning: "NorwAI/NorwAI-Magistral-24B-reasoning",
  Qwen3Embedding8B: "Qwen/Qwen3-Embedding-8B",
} as const;

export type LiteLLMModelId = (typeof LiteLLMModelId)[keyof typeof LiteLLMModelId];

export const VLLMModelId = {
  Qwen25_1_5BInstruct: "Qwen/Qwen2.5-1.5B-Instruct",
  Llamarine: "pentagoniac/llamarine",
  Qwen35_35BA3B: "Qwen/Qwen3.5-35B-A3B",
  Qwen3_4BInstruct: "Qwen/Qwen3-4B-Instruct-2507",
  Qwen35_27B: "Qwen/Qwen3.5-27B",
  DevstralSmall2_24B: "mistralai/Devstral-Small-2-24B-Instruct-2512",
  Qwen3_30BA3BThinking: "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
  Qwen3Next_80BA3BInstruct: "Qwen/Qwen3-Next-80B-A3B-Instruct",
} as const;

export type VLLMModelId = (typeof VLLMModelId)[keyof typeof VLLMModelId];

export const DEFAULT_LITE_LLM_BASE_URL = "https://llm.hpc.ntnu.no/v1";
export const DEFAULT_VLLM_PORT = "8000";
export const DEFAULT_VLLM_BASE_URL = `http://localhost:${DEFAULT_VLLM_PORT}/v1`;

export const getVLLMBaseUrlFromPort = (port: string | number) => `http://localhost:${port}/v1`;

/** Resolve the vLLM base URL: VLLM_BASE_URL > http://localhost:{VLLM_PORT}/v1 > http://localhost:8000/v1 */
export const VLLMBaseUrl = Config.string("VLLM_BASE_URL").pipe(
  Config.orElse(() =>
    Config.string("VLLM_PORT").pipe(
      Config.map(getVLLMBaseUrlFromPort),
      Config.orElse(() => Config.succeed(DEFAULT_VLLM_BASE_URL)),
    ),
  ),
);

const FETCH_TIMEOUT_MS = 4 * 60 * 60 * 1000; // 4 hours

/**
 * Custom fetch that extends the timeout to 4 hours for long-running LLM inference.
 * The AI SDK always passes its own abortSignal to fetch, so we combine it with our
 * timeout signal using AbortSignal.any() to ensure both can trigger cancellation.
 *
 * Also injects `chat_template_kwargs: { enable_thinking: true }` for Gemma models.
 * Gemma requires explicit client-side opt-in for thinking mode (unlike Qwen where it's
 * on by default). The AI SDK's providerOptions can't pass arbitrary nested objects to
 * the request body, so we inject it here at the HTTP level.
 */
export const fetchWithTimeout = Object.assign(
  (url: RequestInfo | URL, init?: RequestInit) => {
    const timeoutSignal = AbortSignal.timeout(FETCH_TIMEOUT_MS);
    const signal = init?.signal ? AbortSignal.any([init.signal, timeoutSignal]) : timeoutSignal;
    let body = init?.body;
    if (body && typeof body === "string") {
      try {
        const parsed = JSON.parse(body);
        if (typeof parsed.model === "string" && parsed.model.toLowerCase().includes("gemma")) {
          parsed.chat_template_kwargs = { enable_thinking: true };
        }
        if (parsed.response_format) {
          parsed.guided_decoding_backend = "outlines";
        }
        body = JSON.stringify(parsed);
      } catch {
        // not JSON, leave as-is
      }
    }
    // Bun's fetch has a socket-level idle timeout (default ~5 min / 240-300s) that fires
    // independently of AbortSignal.timeout() and kills long-running LLM inference requests.
    // Setting `timeout: false` disables it, leaving our AbortSignal as the sole timeout.

    // See https://github.com/oven-sh/bun/issues/16682
    return fetch(url, { ...init, body, signal, timeout: false } as RequestInit);
  },
  { preconnect: fetch.preconnect },
);

export const makeLiteLLMProvider = ({
  apiKey,
  baseUrl = DEFAULT_LITE_LLM_BASE_URL,
  providerName = "litellm",
}: {
  readonly apiKey: string;
  readonly baseUrl?: string;
  readonly providerName?: string;
}) =>
  Effect.try({
    try: () =>
      createOpenAICompatible({
        name: providerName,
        apiKey,
        baseURL: baseUrl,
        supportsStructuredOutputs: true,
        fetch: fetchWithTimeout,
      }),
    catch: (cause) => CreateProviderError.make({ providerName, baseUrl, cause }),
  });

export const makeVLLMProvider = ({
  baseUrl = DEFAULT_VLLM_BASE_URL,
  providerName = "vllm",
}: {
  readonly baseUrl?: string;
  readonly providerName?: string;
}) =>
  Effect.try({
    try: () =>
      createOpenAICompatible({
        name: providerName,
        apiKey: "EMPTY",
        baseURL: baseUrl,
        supportsStructuredOutputs: true,
        fetch: fetchWithTimeout,
      }),
    catch: (cause) => CreateProviderError.make({ providerName, baseUrl, cause }),
  });

export class LLMProvider extends Context.Tag("@app/LLMProvider")<
  LLMProvider,
  {
    readonly provider: LLMProviderType;
  }
>() {
  static readonly IdunLiteLLM = Layer.effect(
    LLMProvider,
    Effect.gen(function* () {
      const apiKey = yield* Config.redacted("LITE_LLM_API_KEY");
      const baseUrl = yield* Config.string("LITE_LLM_BASE_URL").pipe(
        Config.orElse(() => Config.succeed(DEFAULT_LITE_LLM_BASE_URL)),
      );

      const provider = yield* makeLiteLLMProvider({
        apiKey: Redacted.value(apiKey),
        baseUrl,
        providerName: "litellm",
      });

      return LLMProvider.of({ provider });
    }),
  );

  static readonly LocalVLLM = Layer.effect(
    LLMProvider,
    Effect.gen(function* () {
      const baseUrl = yield* VLLMBaseUrl;

      const provider = yield* makeVLLMProvider({ baseUrl, providerName: "vllm" });

      return LLMProvider.of({ provider });
    }),
  );
}
