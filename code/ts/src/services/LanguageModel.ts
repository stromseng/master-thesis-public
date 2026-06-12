import * as ai from "ai";
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import {
  Config,
  ConfigError,
  Context,
  Deferred,
  Effect,
  Layer,
  Option,
  Queue,
  Redacted,
  Runtime,
  Schema,
} from "effect";
import {
  DEFAULT_LITE_LLM_BASE_URL,
  DEFAULT_VLLM_PORT,
  CreateProviderError,
  type LLMProviderType,
  LLMProvider,
  LiteLLMModelId,
  makeLiteLLMProvider,
  makeVLLMProvider,
  getVLLMBaseUrlFromPort,
  VLLMBaseUrl,
  VLLMModelId,
} from "./LLMProvider";

export type GenerateTextOptions = Omit<Parameters<typeof ai.generateText>[0], "model">;
export type GenerateTextResult = Awaited<ReturnType<typeof ai.generateText>>;
export type GenerateObjectOptions<
  SCHEMA extends ai.FlexibleSchema<unknown> = ai.FlexibleSchema<ai.JSONValue>,
  OUTPUT extends "object" | "array" | "enum" | "no-schema" = ai.InferSchema<SCHEMA> extends string
    ? "enum"
    : "object",
  RESULT = OUTPUT extends "array" ? Array<ai.InferSchema<SCHEMA>> : ai.InferSchema<SCHEMA>,
> = Omit<Parameters<typeof ai.generateObject<SCHEMA, OUTPUT, RESULT>>[0], "model">;
export type GenerateObjectResult<
  SCHEMA extends ai.FlexibleSchema<unknown> = ai.FlexibleSchema<ai.JSONValue>,
  OUTPUT extends "object" | "array" | "enum" | "no-schema" = ai.InferSchema<SCHEMA> extends string
    ? "enum"
    : "object",
  RESULT = OUTPUT extends "array" ? Array<ai.InferSchema<SCHEMA>> : ai.InferSchema<SCHEMA>,
> = Awaited<ReturnType<typeof ai.generateObject<SCHEMA, OUTPUT, RESULT>>>;
export type StreamTextOptions = Omit<Parameters<typeof ai.streamText>[0], "model">;
export type StreamTextResult = Awaited<ReturnType<typeof ai.streamText>>;

export class LanguageModelError extends Schema.TaggedError<LanguageModelError>()(
  "LanguageModelError",
  {
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class ResolveLanguageModelError extends Schema.TaggedError<ResolveLanguageModelError>()(
  "ResolveLanguageModelError",
  {
    modelId: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class VLLMModelDiscoveryError extends Schema.TaggedError<VLLMModelDiscoveryError>()(
  "VLLMModelDiscoveryError",
  {
    baseUrl: Schema.String,
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class VLLMModelNotFoundError extends Schema.TaggedError<VLLMModelNotFoundError>()(
  "VLLMModelNotFoundError",
  {
    baseUrl: Schema.String,
    reason: Schema.String,
  },
) {}

export class LanguageModelEndpointsConfigError extends Schema.TaggedError<LanguageModelEndpointsConfigError>()(
  "LanguageModelEndpointsConfigError",
  {
    reason: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class LanguageModelEndpointsNotConfiguredError extends Schema.TaggedError<LanguageModelEndpointsNotConfiguredError>()(
  "LanguageModelEndpointsNotConfiguredError",
  {
    reason: Schema.String,
  },
) {}

export class ConfiguredLanguageModelNotFoundError extends Schema.TaggedError<ConfiguredLanguageModelNotFoundError>()(
  "ConfiguredLanguageModelNotFoundError",
  {
    modelId: Schema.String,
    availableModels: Schema.Array(Schema.String),
  },
) {}

export class ConfiguredLanguageModelSelectionError extends Schema.TaggedError<ConfiguredLanguageModelSelectionError>()(
  "ConfiguredLanguageModelSelectionError",
  {
    reason: Schema.String,
    availableModels: Schema.Array(Schema.String),
  },
) {}

export class ConfiguredLanguageModelMismatchError extends Schema.TaggedError<ConfiguredLanguageModelMismatchError>()(
  "ConfiguredLanguageModelMismatchError",
  {
    requestedModelId: Schema.String,
    availableModels: Schema.Array(Schema.String),
  },
) {}

export class LanguageModelEndpointRequestError extends Schema.TaggedError<LanguageModelEndpointRequestError>()(
  "LanguageModelEndpointRequestError",
  {
    modelId: Schema.String,
    baseUrl: Schema.String,
    operation: Schema.Literal("generate", "stream"),
    cause: Schema.Defect,
  },
) {}

export class LanguageModel extends Context.Tag("@app/LanguageModel")<
  LanguageModel,
  ai.LanguageModel
>() {}

export const make = (modelId: string) =>
  LLMProvider.pipe(
    Effect.flatMap(({ provider }) =>
      Effect.try({
        try: () => provider.languageModel(modelId),
        catch: (cause) =>
          new ResolveLanguageModelError({
            modelId,
            cause,
          }),
      }),
    ),
  );

export const layer = (modelId: string) => Layer.effect(LanguageModel, make(modelId));

const DEFAULT_ENDPOINT_CONCURRENCY = {
  litellm: 2,
  vllm: 25,
} as const;

const PositiveIntSchema = Schema.Number.pipe(Schema.int(), Schema.positive());
const NonEmptyStringArraySchema = Schema.NonEmptyArray(Schema.String);
const EndpointPortSchema = Schema.Union(PositiveIntSchema, Schema.String);

const LiteLLMEndpointConfigSchema = Schema.Struct({
  provider: Schema.Literal("litellm"),
  baseUrl: Schema.optional(Schema.String),
  apiKeyConfigKey: Schema.optional(Schema.String),
  models: Schema.optional(NonEmptyStringArraySchema),
  maxConcurrency: Schema.optional(PositiveIntSchema),
});

const VLLMEndpointConfigSchema = Schema.Struct({
  provider: Schema.Literal("vllm"),
  baseUrl: Schema.optional(Schema.String),
  port: Schema.optional(EndpointPortSchema),
  models: Schema.optional(NonEmptyStringArraySchema),
  maxConcurrency: Schema.optional(PositiveIntSchema),
});

const RouterEndpointConfigSchema = Schema.Union(
  LiteLLMEndpointConfigSchema,
  VLLMEndpointConfigSchema,
);
const LegacyLanguageModelEndpointsConfigSchema = Schema.NonEmptyArray(RouterEndpointConfigSchema);
const RouterLanguageModelEndpointsConfigSchema = Schema.Struct({
  model: Schema.String,
  endpoints: Schema.NonEmptyArray(RouterEndpointConfigSchema),
});
const LanguageModelEndpointsConfigSchema = Schema.Union(
  RouterLanguageModelEndpointsConfigSchema,
  LegacyLanguageModelEndpointsConfigSchema,
);

type LanguageModelEndpointConfig = typeof RouterEndpointConfigSchema.Type;
type RouterLanguageModelEndpointsConfig = typeof RouterLanguageModelEndpointsConfigSchema.Type;
type LanguageModelEndpointsConfig = typeof LanguageModelEndpointsConfigSchema.Type;
type NormalizedLanguageModelEndpointsConfig = {
  readonly model: Option.Option<string>;
  readonly endpoints: ReadonlyArray<LanguageModelEndpointConfig>;
};

type ResolvedLanguageModelEndpoint = {
  readonly label: string;
  readonly providerKind: "litellm" | "vllm";
  readonly baseUrl: string;
  readonly maxConcurrency: number;
  readonly servedModelId: string;
  readonly provider: LLMProviderType;
};

export type RoutedLanguageModelRoute<E = never, R = never> = {
  readonly label: string;
  readonly maxConcurrency: number;
  readonly layer: Layer.Layer<LanguageModel, E, R>;
};

type ResolvedLanguageModelWorkerEndpoint = ResolvedLanguageModelEndpoint & {
  readonly modelId: string;
  readonly model: LanguageModelV3;
};

type ResolvedLanguageModelWorkerRoute = {
  readonly label: string;
  readonly baseUrl: string;
  readonly maxConcurrency: number;
  readonly modelId: string;
  readonly model: LanguageModelV3;
};

type GenerateLanguageModelQueueRequest = {
  readonly _tag: "Generate";
  readonly options: LanguageModelV3CallOptions;
  readonly response: Deferred.Deferred<LanguageModelV3GenerateResult, unknown>;
};

type StreamLanguageModelQueueRequest = {
  readonly _tag: "Stream";
  readonly options: LanguageModelV3CallOptions;
  readonly response: Deferred.Deferred<LanguageModelV3StreamResult, unknown>;
};

type LanguageModelQueueRequest =
  | GenerateLanguageModelQueueRequest
  | StreamLanguageModelQueueRequest;

const normalizeEndpointPort = (port: string | number | undefined): string => {
  if (port === undefined) return DEFAULT_VLLM_PORT;
  if (typeof port === "number") return String(port);
  if (/^\d+$/.test(port)) return port;
  throw new Error(`Invalid port value: ${port}`);
};

const isRouterLanguageModelEndpointsConfig = (
  config: LanguageModelEndpointsConfig,
): config is RouterLanguageModelEndpointsConfig => !Array.isArray(config);

const loadLanguageModelEndpointsConfig = Effect.fn(
  "LanguageModel.loadLanguageModelEndpointsConfig",
)(function* (configKey = "LANGUAGE_MODEL_ENDPOINTS") {
  const rawConfig = yield* Config.option(Config.string(configKey));
  if (Option.isNone(rawConfig)) {
    return Option.none();
  }

  const decoded = yield* Schema.decodeUnknown(Schema.parseJson(LanguageModelEndpointsConfigSchema))(
    rawConfig.value,
  ).pipe(
    Effect.mapError(
      (cause) =>
        new LanguageModelEndpointsConfigError({
          reason: `${configKey} must be valid JSON with a valid endpoint shape`,
          cause,
        }),
    ),
    Effect.catchAllDefect((cause) =>
      Effect.fail(
        new LanguageModelEndpointsConfigError({
          reason: `${configKey} could not be decoded`,
          cause,
        }),
      ),
    ),
  );

  if (!isRouterLanguageModelEndpointsConfig(decoded)) {
    return Option.some({
      model: Option.none(),
      endpoints: decoded,
    } satisfies NormalizedLanguageModelEndpointsConfig);
  }

  return Option.some({
    model: Option.some(decoded.model),
    endpoints: decoded.endpoints,
  } satisfies NormalizedLanguageModelEndpointsConfig);
});

const failLanguageModelEndpointRequest = ({
  modelId,
  baseUrl,
  operation,
  cause,
}: {
  readonly modelId: string;
  readonly baseUrl: string;
  readonly operation: "generate" | "stream";
  readonly cause: unknown;
}) =>
  new LanguageModelEndpointRequestError({
    modelId,
    baseUrl,
    operation,
    cause,
  });

const resolveLanguageModelV3 = ({
  modelId,
  model,
  source,
}: {
  readonly modelId: string;
  readonly model: ai.LanguageModel;
  readonly source: string;
}) => {
  if (typeof model === "object" && model !== null && model.specificationVersion === "v3") {
    return Effect.succeed(model);
  }

  return Effect.fail(
    new ResolveLanguageModelError({
      modelId,
      cause: new Error(`${source} did not provide a LanguageModelV3 instance`),
    }),
  );
};

const resolveConfiguredEndpointModelId = ({
  configuredModelIds,
  endpointLabel,
  providerKind,
}: {
  readonly configuredModelIds: ReadonlyArray<string>;
  readonly endpointLabel: string;
  readonly providerKind: "litellm" | "vllm";
}) => {
  const uniqueModelIds = [...new Set(configuredModelIds)];
  if (uniqueModelIds.length === 1) {
    return Effect.succeed(uniqueModelIds[0]!);
  }

  return Effect.fail(
    new LanguageModelEndpointsConfigError({
      reason: `${endpointLabel} (${providerKind}) must resolve exactly one model in router mode`,
      cause: new Error(`Resolved models: ${uniqueModelIds.join(", ") || "none"}`),
    }),
  );
};

const ensureEndpointServesConfiguredModel = ({
  configuredModelIds,
  configuredModelId,
  endpointLabel,
  providerKind,
}: {
  readonly configuredModelIds: ReadonlyArray<string>;
  readonly configuredModelId: string;
  readonly endpointLabel: string;
  readonly providerKind: "litellm" | "vllm";
}) => {
  const uniqueModelIds = [...new Set(configuredModelIds)];
  if (uniqueModelIds.includes(configuredModelId)) {
    return Effect.succeed(configuredModelId);
  }

  return Effect.fail(
    new LanguageModelEndpointsConfigError({
      reason: `${endpointLabel} (${providerKind}) does not serve the configured router model`,
      cause: new Error(
        `Requested model: ${configuredModelId}; available models: ${uniqueModelIds.join(", ") || "none"}`,
      ),
    }),
  );
};

const fetchOpenAICompatibleModelIds = ({
  baseUrl,
  providerKind,
  apiKey,
}: {
  readonly baseUrl: string;
  readonly providerKind: "litellm" | "vllm";
  readonly apiKey?: string;
}) =>
  Effect.gen(function* () {
    const response = yield* Effect.tryPromise({
      try: () =>
        fetch(`${baseUrl}/models`, {
          headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
        }).then((r) => r.json()),
      catch: (cause) =>
        new LanguageModelEndpointsConfigError({
          reason: "LANGUAGE_MODEL_ENDPOINTS has an invalid shape",
          cause,
        }),
    }).pipe(
      Effect.catchTag("LanguageModelEndpointsConfigError", (error) =>
        Effect.fail(
          new VLLMModelDiscoveryError({
            baseUrl,
            reason: error.reason,
            cause: error.cause,
          }),
        ),
      ),
      Effect.catchAllDefect(
        (cause) =>
          new VLLMModelDiscoveryError({
            baseUrl,
            reason: `Failed to fetch models from the ${providerKind} server`,
            cause,
          }),
      ),
    );

    const modelIds =
      response && typeof response === "object" && "data" in response && Array.isArray(response.data)
        ? response.data
            .map((entry: unknown) =>
              entry && typeof entry === "object" && "id" in entry && typeof entry.id === "string"
                ? entry.id
                : undefined,
            )
            .filter((id: string | undefined): id is string => id !== undefined)
        : [];

    if (modelIds.length > 0) {
      return modelIds;
    }

    return yield* new LanguageModelEndpointsConfigError({
      reason: `${providerKind} /models returned no model IDs`,
      cause: new Error(`Base URL: ${baseUrl}`),
    });
  });

const fetchVLLMModelIds = (baseUrl: string) =>
  fetchOpenAICompatibleModelIds({
    baseUrl,
    providerKind: "vllm",
  }).pipe(
    Effect.catchTag("LanguageModelEndpointsConfigError", (error) =>
      Effect.fail(
        new VLLMModelDiscoveryError({
          baseUrl,
          reason: error.reason,
          cause: error.cause,
        }),
      ),
    ),
  );

const resolveLanguageModelEndpoint = ({
  endpoint,
  index,
  configuredModelId,
}: {
  readonly endpoint: LanguageModelEndpointConfig;
  readonly index: number;
  readonly configuredModelId: Option.Option<string>;
}) =>
  Effect.gen(function* () {
    if (endpoint.provider === "litellm") {
      const label = `litellm:${index + 1}`;
      const baseUrl = endpoint.baseUrl ?? DEFAULT_LITE_LLM_BASE_URL;
      const apiKeyConfigKey = endpoint.apiKeyConfigKey ?? "LITE_LLM_API_KEY";
      const apiKey = yield* Config.redacted(apiKeyConfigKey);
      const provider = yield* makeLiteLLMProvider({
        apiKey: Redacted.value(apiKey),
        baseUrl,
        providerName: `litellm-router-${index + 1}`,
      });
      const configuredModelIds =
        endpoint.models ??
        (yield* fetchOpenAICompatibleModelIds({
          baseUrl,
          providerKind: endpoint.provider,
          apiKey: Redacted.value(apiKey),
        }));
      const servedModelId = yield* Option.match(configuredModelId, {
        onNone: () =>
          resolveConfiguredEndpointModelId({
            configuredModelIds,
            endpointLabel: label,
            providerKind: endpoint.provider,
          }),
        onSome: (modelId) =>
          ensureEndpointServesConfiguredModel({
            configuredModelIds,
            configuredModelId: modelId,
            endpointLabel: label,
            providerKind: endpoint.provider,
          }),
      });

      return {
        label,
        providerKind: endpoint.provider,
        baseUrl,
        maxConcurrency: endpoint.maxConcurrency ?? DEFAULT_ENDPOINT_CONCURRENCY.litellm,
        servedModelId,
        provider,
      } satisfies ResolvedLanguageModelEndpoint;
    }

    const label = `vllm:${index + 1}`;
    const baseUrl =
      endpoint.baseUrl ?? getVLLMBaseUrlFromPort(normalizeEndpointPort(endpoint.port));
    const provider = yield* makeVLLMProvider({
      baseUrl,
      providerName: `vllm-router-${index + 1}`,
    });
    const configuredModelIds = endpoint.models ?? (yield* fetchVLLMModelIds(baseUrl));
    const servedModelId = yield* Option.match(configuredModelId, {
      onNone: () =>
        resolveConfiguredEndpointModelId({
          configuredModelIds,
          endpointLabel: label,
          providerKind: endpoint.provider,
        }),
      onSome: (modelId) =>
        ensureEndpointServesConfiguredModel({
          configuredModelIds,
          configuredModelId: modelId,
          endpointLabel: label,
          providerKind: endpoint.provider,
        }),
    });

    return {
      label,
      providerKind: endpoint.provider,
      baseUrl,
      maxConcurrency: endpoint.maxConcurrency ?? DEFAULT_ENDPOINT_CONCURRENCY.vllm,
      servedModelId,
      provider,
    } satisfies ResolvedLanguageModelEndpoint;
  }).pipe(
    Effect.catchTag("CreateProviderError", (error) =>
      Effect.fail(
        new LanguageModelEndpointsConfigError({
          reason: `Failed to create provider for ${endpoint.provider} endpoint`,
          cause: error,
        }),
      ),
    ),
    Effect.catchAllDefect((cause) =>
      Effect.fail(
        new LanguageModelEndpointsConfigError({
          reason: `Failed to normalize ${endpoint.provider} endpoint configuration`,
          cause,
        }),
      ),
    ),
  );

const loadResolvedLanguageModelEndpoints = Effect.fn(
  "LanguageModel.loadResolvedLanguageModelEndpoints",
)(function* (configKey = "LANGUAGE_MODEL_ENDPOINTS") {
  const configuredEndpoints = yield* loadLanguageModelEndpointsConfig(configKey);
  if (!Option.isSome(configuredEndpoints)) {
    return Option.none();
  }

  const resolved = yield* Effect.forEach(
    configuredEndpoints.value.endpoints,
    (endpoint, index) =>
      resolveLanguageModelEndpoint({
        endpoint,
        index,
        configuredModelId: configuredEndpoints.value.model,
      }),
    { concurrency: "unbounded" },
  );

  return Option.some(resolved);
});

const getConfiguredLanguageModelIds = Effect.fn("LanguageModel.getConfiguredLanguageModelIds")(
  function* (configKey = "LANGUAGE_MODEL_ENDPOINTS") {
    const configuredEndpoints = yield* loadResolvedLanguageModelEndpoints(configKey);
    if (!Option.isSome(configuredEndpoints)) {
      return Option.none();
    }

    const configuredRouter = yield* loadLanguageModelEndpointsConfig(configKey);
    if (Option.isSome(configuredRouter) && Option.isSome(configuredRouter.value.model)) {
      return Option.some([configuredRouter.value.model.value]);
    }

    const modelIds = [
      ...new Set(
        configuredEndpoints.value.map(
          (endpoint: ResolvedLanguageModelEndpoint) => endpoint.servedModelId,
        ),
      ),
    ];
    return Option.some(modelIds);
  },
);

const resolveRequestedLanguageModelEndpoints = (
  modelId: string,
  endpoints: ReadonlyArray<ResolvedLanguageModelEndpoint>,
) =>
  Effect.gen(function* () {
    const matchingEndpoints: ReadonlyArray<
      Effect.Effect<
        ResolvedLanguageModelWorkerEndpoint,
        ResolveLanguageModelError | ConfiguredLanguageModelMismatchError
      >
    > = endpoints.map((endpoint) =>
      endpoint.servedModelId !== modelId
        ? Effect.fail(
            new ConfiguredLanguageModelMismatchError({
              requestedModelId: modelId,
              availableModels: [...new Set(endpoints.map((item) => item.servedModelId))],
            }),
          )
        : Effect.try({
            try: () => endpoint.provider.languageModel(modelId),
            catch: (cause) =>
              new ResolveLanguageModelError({
                modelId,
                cause,
              }),
          }).pipe(
            Effect.flatMap((model) =>
              resolveLanguageModelV3({
                modelId,
                model,
                source: endpoint.label,
              }),
            ),
            Effect.map(
              (model) =>
                ({
                  ...endpoint,
                  modelId,
                  model,
                }) satisfies ResolvedLanguageModelWorkerEndpoint,
            ),
          ),
    );

    return yield* Effect.all(matchingEndpoints, { concurrency: "unbounded" });
  });

const resolveConfiguredLanguageModelEndpoints = (
  modelId: string,
  configKey = "LANGUAGE_MODEL_ENDPOINTS",
) =>
  Effect.gen(function* () {
    const configuredEndpoints = yield* loadResolvedLanguageModelEndpoints(configKey);
    if (!Option.isSome(configuredEndpoints)) {
      return yield* new LanguageModelEndpointsNotConfiguredError({
        reason: `${configKey} is not configured`,
      });
    }

    return yield* resolveRequestedLanguageModelEndpoints(modelId, configuredEndpoints.value);
  });

const resolveRequestedLanguageModelRoutes = <E, R>(
  modelId: string,
  routes: ReadonlyArray<RoutedLanguageModelRoute<E, R>>,
) =>
  Effect.gen(function* () {
    const resolvedRoutes = yield* Effect.forEach(
      routes,
      (route) =>
        Effect.gen(function* () {
          const context = yield* Layer.build(route.layer);
          const providedModel = yield* resolveLanguageModelV3({
            modelId,
            model: Context.get(context, LanguageModel),
            source: route.label,
          });
          return {
            label: route.label,
            baseUrl: route.label,
            maxConcurrency: route.maxConcurrency,
            modelId: providedModel.modelId,
            model: providedModel,
          } satisfies ResolvedLanguageModelWorkerRoute;
        }),
      { concurrency: "unbounded" },
    );

    const availableModels = [...new Set(resolvedRoutes.map((route) => route.modelId))];
    if (availableModels.length === 1 && availableModels[0] === modelId) {
      return resolvedRoutes;
    }

    return yield* new ConfiguredLanguageModelMismatchError({
      requestedModelId: modelId,
      availableModels,
    });
  });

const wrapQueuedStream = (
  stream: ReadableStream<LanguageModelV3StreamPart>,
  onFinalize: () => void,
): ReadableStream<LanguageModelV3StreamPart> => {
  let finalized = false;
  const finalize = () => {
    if (!finalized) {
      finalized = true;
      onFinalize();
    }
  };

  return new ReadableStream<LanguageModelV3StreamPart>({
    async start(controller) {
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            finalize();
            return;
          }
          controller.enqueue(value);
        }
      } catch (error) {
        controller.error(error);
        finalize();
      } finally {
        reader.releaseLock();
      }
    },
    async cancel(reason) {
      try {
        await stream.cancel(reason);
      } finally {
        finalize();
      }
    },
  });
};

const processLanguageModelQueueRequest = (
  request: LanguageModelQueueRequest,
  endpoint: ResolvedLanguageModelWorkerEndpoint | ResolvedLanguageModelWorkerRoute,
  runtime: Runtime.Runtime<never>,
) =>
  Effect.gen(function* () {
    if (request._tag === "Generate") {
      const result = yield* Effect.exit(
        Effect.tryPromise({
          try: () => endpoint.model.doGenerate(request.options),
          catch: (cause) =>
            failLanguageModelEndpointRequest({
              modelId: endpoint.modelId,
              baseUrl: endpoint.baseUrl,
              operation: "generate",
              cause,
            }),
        }),
      );
      yield* Deferred.done(request.response, result);
      return;
    }

    const streamResult = yield* Effect.exit(
      Effect.tryPromise({
        try: () => endpoint.model.doStream(request.options),
        catch: (cause) =>
          failLanguageModelEndpointRequest({
            modelId: endpoint.modelId,
            baseUrl: endpoint.baseUrl,
            operation: "stream",
            cause,
          }),
      }),
    );

    if (streamResult._tag === "Failure") {
      yield* Deferred.done(request.response, streamResult);
      return;
    }

    const released = yield* Deferred.make<void>();
    const wrapped = {
      ...streamResult.value,
      stream: wrapQueuedStream(streamResult.value.stream, () => {
        Runtime.runFork(runtime, Deferred.succeed(released, undefined));
      }),
    } satisfies LanguageModelV3StreamResult;

    yield* Deferred.succeed(request.response, wrapped);
    yield* Deferred.await(released);
  });

const startLanguageModelEndpointWorker = (
  queue: Queue.Queue<LanguageModelQueueRequest>,
  endpoint: ResolvedLanguageModelWorkerEndpoint | ResolvedLanguageModelWorkerRoute,
  workerIndex: number,
  runtime: Runtime.Runtime<never>,
) =>
  Effect.forever(
    Effect.gen(function* () {
      const request = yield* Queue.take(queue);
      yield* processLanguageModelQueueRequest(request, endpoint, runtime).pipe(
        Effect.catchAllCause((cause) =>
          Effect.logError(
            `Language model queue worker failed for ${endpoint.label} (${endpoint.baseUrl}) [worker ${workerIndex + 1}]`,
            { cause },
          ),
        ),
      );
    }),
  );

const makeQueuedLanguageModelFromResolvedRoutes = (
  modelId: string,
  routes: ReadonlyArray<ResolvedLanguageModelWorkerEndpoint | ResolvedLanguageModelWorkerRoute>,
) =>
  Effect.gen(function* () {
    const runtime = yield* Effect.runtime<never>();
    const queue = yield* Queue.unbounded<LanguageModelQueueRequest>();

    for (const route of routes) {
      for (let index = 0; index < route.maxConcurrency; index++) {
        yield* Effect.forkScoped(startLanguageModelEndpointWorker(queue, route, index, runtime));
      }
    }

    const submitGenerate = (options: LanguageModelV3CallOptions) =>
      Effect.gen(function* () {
        const response = yield* Deferred.make<LanguageModelV3GenerateResult, unknown>();
        yield* Queue.offer(queue, {
          _tag: "Generate",
          options,
          response,
        });
        return yield* Deferred.await(response);
      });

    const submitStream = (options: LanguageModelV3CallOptions) =>
      Effect.gen(function* () {
        const response = yield* Deferred.make<LanguageModelV3StreamResult, unknown>();
        yield* Queue.offer(queue, {
          _tag: "Stream",
          options,
          response,
        });
        return yield* Deferred.await(response);
      });

    return {
      specificationVersion: "v3",
      provider: "router.queue",
      modelId,
      supportedUrls: {},
      doGenerate: (options) => Runtime.runPromise(runtime, submitGenerate(options)),
      doStream: (options) => Runtime.runPromise(runtime, submitStream(options)),
    } satisfies LanguageModelV3;
  });

const makeQueuedLanguageModel = (modelId: string, configKey = "LANGUAGE_MODEL_ENDPOINTS") =>
  Effect.gen(function* () {
    const endpoints = yield* resolveConfiguredLanguageModelEndpoints(modelId, configKey);
    return yield* makeQueuedLanguageModelFromResolvedRoutes(modelId, endpoints);
  });

const makeQueuedLanguageModelFromLayers = <E, R>(
  modelId: string,
  routes: ReadonlyArray<RoutedLanguageModelRoute<E, R>>,
) =>
  Effect.gen(function* () {
    const resolvedRoutes = yield* resolveRequestedLanguageModelRoutes(modelId, routes);
    return yield* makeQueuedLanguageModelFromResolvedRoutes(modelId, resolvedRoutes);
  });

export const RoutedLanguageModelLayer = (modelId: string, configKey = "LANGUAGE_MODEL_ENDPOINTS") =>
  Layer.scoped(LanguageModel, makeQueuedLanguageModel(modelId, configKey));

export const RoutedLanguageModelLayerFromLayers = <E, R>(
  modelId: string,
  routes: ReadonlyArray<RoutedLanguageModelRoute<E, R>>,
) => Layer.scoped(LanguageModel, makeQueuedLanguageModelFromLayers(modelId, routes));

export const LiteLLMLanguageModelLayer = {
  mistralLarge3Instruct: layer(LiteLLMModelId.MistralLarge3Instruct).pipe(
    Layer.provide(LLMProvider.IdunLiteLLM),
  ),
  gptOss120b: layer(LiteLLMModelId.GptOss120B).pipe(Layer.provide(LLMProvider.IdunLiteLLM)),
  glm47Fp8: layer(LiteLLMModelId.Glm47Fp8).pipe(Layer.provide(LLMProvider.IdunLiteLLM)),
  kimiK25: layer(LiteLLMModelId.kimiK25).pipe(Layer.provide(LLMProvider.IdunLiteLLM)),
  qwen35122ba10b: layer(LiteLLMModelId.Qwen35122ba10b).pipe(Layer.provide(LLMProvider.IdunLiteLLM)),
  norwAIMagistral24BReasoning: layer(LiteLLMModelId.NorwAIMagistral24BReasoning).pipe(
    Layer.provide(LLMProvider.IdunLiteLLM),
  ),
  qwen3Embedding8B: layer(LiteLLMModelId.Qwen3Embedding8B).pipe(
    Layer.provide(LLMProvider.IdunLiteLLM),
  ),
} as const;

export const VLLMLanguageModelLayer = {
  qwen25_1_5BInstruct: layer(VLLMModelId.Qwen25_1_5BInstruct).pipe(
    Layer.provide(LLMProvider.LocalVLLM),
  ),
  llamarine: layer(VLLMModelId.Llamarine).pipe(Layer.provide(LLMProvider.LocalVLLM)),
  qwen35_35BA3B: layer(VLLMModelId.Qwen35_35BA3B).pipe(Layer.provide(LLMProvider.LocalVLLM)),
  qwen3_4BInstruct: layer(VLLMModelId.Qwen3_4BInstruct).pipe(Layer.provide(LLMProvider.LocalVLLM)),
  devstralSmall2_24B: layer(VLLMModelId.DevstralSmall2_24B).pipe(
    Layer.provide(LLMProvider.LocalVLLM),
  ),
  qwen35_27B: layer(VLLMModelId.Qwen35_27B).pipe(Layer.provide(LLMProvider.LocalVLLM)),
  qwen3_30BA3BThinking: layer(VLLMModelId.Qwen3_30BA3BThinking).pipe(
    Layer.provide(LLMProvider.LocalVLLM),
  ),
  qwen3Next_80BA3BInstruct: layer(VLLMModelId.Qwen3Next_80BA3BInstruct).pipe(
    Layer.provide(LLMProvider.LocalVLLM),
  ),
} as const;

// ---------------------------------------------------------------------------
// ENV-driven eval layers
// ---------------------------------------------------------------------------

const ProviderConfig = (configKey: "EVAL_PROVIDER" | "JUDGE_PROVIDER") =>
  Config.string(configKey).pipe(
    Config.orElse(() => Config.succeed("litellm")),
    Config.validate({
      message: `${configKey} must be 'litellm' or 'vllm'`,
      validation: (s): s is "litellm" | "vllm" => s === "litellm" || s === "vllm",
    }),
  );

const resolveProviderLayer = (provider: "litellm" | "vllm") =>
  provider === "litellm" ? LLMProvider.IdunLiteLLM : LLMProvider.LocalVLLM;

const knownModelIds: Record<"litellm" | "vllm", ReadonlySet<string>> = {
  litellm: new Set(Object.values(LiteLLMModelId)),
  vllm: new Set(Object.values(VLLMModelId)),
};

const warnUnknownModel = ({
  provider,
  modelId,
  modelConfigKey,
}: {
  readonly provider: "litellm" | "vllm";
  readonly modelId: string;
  readonly modelConfigKey: string;
}) =>
  knownModelIds[provider].has(modelId)
    ? Effect.void
    : Effect.logWarning(
        `Model "${modelId}" is not in the known ${provider} model list. Double-check ${modelConfigKey}.`,
      );

/**
 * Fetch the first model ID from a vLLM/sglang server's /v1/models endpoint.
 * Fails with a tagged error if the server is unreachable or returns no models.
 */
const fetchVLLMModelId = () =>
  Effect.gen(function* () {
    const baseUrl = yield* VLLMBaseUrl;
    const [id] = yield* fetchVLLMModelIds(baseUrl);
    if (id) {
      yield* Effect.log(`Auto-detected vLLM model: ${id}`);
      return id;
    }
    return yield* new VLLMModelNotFoundError({
      baseUrl,
      reason: "vLLM /models returned no model IDs",
    });
  });

const resolveConfiguredEvalModel = ({
  explicitConfigKey,
  endpointsConfigKey,
  defaultModelId,
}: {
  readonly explicitConfigKey: "EVAL_MODEL" | "EVAL_MULTIMODAL_MODEL" | "JUDGE_MODEL";
  readonly endpointsConfigKey: "LANGUAGE_MODEL_ENDPOINTS" | "JUDGE_LANGUAGE_MODEL_ENDPOINTS";
  readonly defaultModelId: string;
}) =>
  Effect.gen(function* () {
    const explicitModel = yield* Config.option(Config.string(explicitConfigKey));
    if (Option.isSome(explicitModel)) {
      return {
        modelId: explicitModel.value,
        source: explicitConfigKey,
      };
    }

    const configuredModelIds = yield* getConfiguredLanguageModelIds(endpointsConfigKey);
    if (Option.isNone(configuredModelIds)) {
      return {
        modelId: defaultModelId,
        source: "default",
      };
    }

    if (configuredModelIds.value.length === 1) {
      return {
        modelId: configuredModelIds.value[0]!,
        source: endpointsConfigKey,
      };
    }

    return yield* new ConfiguredLanguageModelSelectionError({
      reason: `Multiple configured models are available; set ${explicitConfigKey} explicitly`,
      availableModels: configuredModelIds.value,
    });
  });

type EvalLanguageModelLayerError =
  | ConfigError.ConfigError
  | CreateProviderError
  | ResolveLanguageModelError
  | VLLMModelDiscoveryError
  | VLLMModelNotFoundError
  | LanguageModelEndpointsConfigError
  | LanguageModelEndpointsNotConfiguredError
  | ConfiguredLanguageModelSelectionError
  | ConfiguredLanguageModelMismatchError;

const makeDirectEvalLanguageModelLayer = (
  provider: "litellm" | "vllm",
  modelId: string,
): Layer.Layer<LanguageModel, EvalLanguageModelLayerError> =>
  layer(modelId).pipe(Layer.provide(resolveProviderLayer(provider)));

const makeRoutedEvalLanguageModelLayer = (
  modelId: string,
  endpointsConfigKey: "LANGUAGE_MODEL_ENDPOINTS" | "JUDGE_LANGUAGE_MODEL_ENDPOINTS",
): Layer.Layer<LanguageModel, EvalLanguageModelLayerError> =>
  RoutedLanguageModelLayer(modelId, endpointsConfigKey);

const makeConfiguredLanguageModelLayer = ({
  label,
  endpointsConfigKey,
  providerConfigKey,
  modelConfigKey,
  defaultModelId,
}: {
  readonly label: string;
  readonly endpointsConfigKey: "LANGUAGE_MODEL_ENDPOINTS" | "JUDGE_LANGUAGE_MODEL_ENDPOINTS";
  readonly providerConfigKey: "EVAL_PROVIDER" | "JUDGE_PROVIDER";
  readonly modelConfigKey: "EVAL_MODEL" | "EVAL_MULTIMODAL_MODEL" | "JUDGE_MODEL";
  readonly defaultModelId: string;
}): Layer.Layer<LanguageModel, EvalLanguageModelLayerError> =>
  Layer.unwrapEffect(
    Effect.gen(function* () {
      const configuredEndpoints = yield* loadLanguageModelEndpointsConfig(endpointsConfigKey);
      if (Option.isSome(configuredEndpoints)) {
        const { modelId, source } = yield* resolveConfiguredEvalModel({
          explicitConfigKey: modelConfigKey,
          endpointsConfigKey,
          defaultModelId,
        });
        yield* Effect.log(
          `${label} model: provider=router.queue, model=${modelId}, source=${source}`,
        );
        return makeRoutedEvalLanguageModelLayer(modelId, endpointsConfigKey);
      }

      const provider = yield* ProviderConfig(providerConfigKey);
      const explicitModel = yield* Config.option(Config.string(modelConfigKey));
      const modelSource =
        explicitModel._tag === "Some"
          ? modelConfigKey
          : provider === "vllm"
            ? "vllm auto-detected"
            : "default";
      const modelId = yield* explicitModel._tag === "Some"
        ? Effect.succeed(explicitModel.value)
        : provider === "vllm"
          ? fetchVLLMModelId()
          : Effect.succeed(defaultModelId);
      yield* warnUnknownModel({ provider, modelId, modelConfigKey });
      yield* Effect.log(
        `${label} model: provider=${provider}, model=${modelId}, source=${modelSource}`,
      );
      return makeDirectEvalLanguageModelLayer(provider, modelId);
    }),
  );

/** Text eval layer — reads EVAL_PROVIDER + EVAL_MODEL + LANGUAGE_MODEL_ENDPOINTS */
export const EvalLanguageModelLayer: Layer.Layer<LanguageModel, EvalLanguageModelLayerError> =
  makeConfiguredLanguageModelLayer({
    label: "Text eval",
    endpointsConfigKey: "LANGUAGE_MODEL_ENDPOINTS",
    providerConfigKey: "EVAL_PROVIDER",
    modelConfigKey: "EVAL_MODEL",
    defaultModelId: LiteLLMModelId.GptOss120B,
  });

/** Multimodal eval layer — reads EVAL_PROVIDER + EVAL_MULTIMODAL_MODEL + LANGUAGE_MODEL_ENDPOINTS */
export const EvalMultimodalLanguageModelLayer: Layer.Layer<
  LanguageModel,
  EvalLanguageModelLayerError
> = makeConfiguredLanguageModelLayer({
  label: "Multimodal eval",
  endpointsConfigKey: "LANGUAGE_MODEL_ENDPOINTS",
  providerConfigKey: "EVAL_PROVIDER",
  modelConfigKey: "EVAL_MULTIMODAL_MODEL",
  defaultModelId: LiteLLMModelId.kimiK25,
});

/** Judge layer input model — reads JUDGE_PROVIDER + JUDGE_MODEL + JUDGE_LANGUAGE_MODEL_ENDPOINTS */
export const JudgeLanguageModelLayer: Layer.Layer<LanguageModel, EvalLanguageModelLayerError> =
  makeConfiguredLanguageModelLayer({
    label: "Judge",
    endpointsConfigKey: "JUDGE_LANGUAGE_MODEL_ENDPOINTS",
    providerConfigKey: "JUDGE_PROVIDER",
    modelConfigKey: "JUDGE_MODEL",
    defaultModelId: LiteLLMModelId.Qwen35122ba10b,
  });

export const generateText = (options: GenerateTextOptions) =>
  Effect.gen(function* () {
    const model = yield* LanguageModel;
    return yield* Effect.tryPromise({
      try: () => {
        const { messages, prompt, ...sharedOptions } = options;

        if (messages !== undefined) {
          return ai.generateText({
            ...sharedOptions,
            model,
            messages,
            experimental_telemetry: { isEnabled: true },
          });
        }

        if (prompt !== undefined) {
          return ai.generateText({
            ...sharedOptions,
            model,
            prompt,
            experimental_telemetry: { isEnabled: true },
          });
        }

        throw new Error("generateText requires either prompt or messages");
      },
      catch: (error) =>
        LanguageModelError.make({
          reason: "Failed to generate text",
          cause: error,
        }),
    });
  });

export const generateObject = <
  SCHEMA extends ai.FlexibleSchema<unknown> = ai.FlexibleSchema<ai.JSONValue>,
  OUTPUT extends "object" | "array" | "enum" | "no-schema" = ai.InferSchema<SCHEMA> extends string
    ? "enum"
    : "object",
  RESULT = OUTPUT extends "array" ? Array<ai.InferSchema<SCHEMA>> : ai.InferSchema<SCHEMA>,
>(
  options: GenerateObjectOptions<SCHEMA, OUTPUT, RESULT>,
) =>
  Effect.gen(function* () {
    const model = yield* LanguageModel;
    const res = yield* Effect.tryPromise({
      try: () => {
        const { messages, prompt, ...sharedOptions } = options;

        if (messages !== undefined) {
          return ai.generateObject<SCHEMA, OUTPUT, RESULT>({
            ...sharedOptions,
            model,
            messages,
            experimental_telemetry: { isEnabled: true },
          });
        }

        if (prompt !== undefined) {
          return ai.generateObject<SCHEMA, OUTPUT, RESULT>({
            ...sharedOptions,
            model,
            prompt,
            experimental_telemetry: { isEnabled: true },
          });
        }

        throw new Error("generateObject requires either prompt or messages");
      },
      catch: (error) =>
        LanguageModelError.make({
          reason: "Failed to generate object",
          cause: error,
        }),
    });
    yield* Effect.annotateCurrentSpan("ai.usage.reasoningTokens", res.reasoning);
    return res;
  });

export const streamText = (options: StreamTextOptions) =>
  Effect.gen(function* () {
    const model = yield* LanguageModel;
    return yield* Effect.try({
      try: () => {
        const { messages, prompt, ...sharedOptions } = options;

        if (messages !== undefined) {
          return ai.streamText({
            ...sharedOptions,
            model,
            messages,
            experimental_telemetry: { isEnabled: true },
          });
        }

        if (prompt !== undefined) {
          return ai.streamText({
            ...sharedOptions,
            model,
            prompt,
            experimental_telemetry: { isEnabled: true },
          });
        }

        throw new Error("streamText requires either prompt or messages");
      },
      catch: (error) =>
        LanguageModelError.make({
          reason: "Failed to stream text",
          cause: error,
        }),
    });
  });
