import { createClient, type PhoenixClient as PhoenixClientType } from "@arizeai/phoenix-client";
import { Context, Effect, Layer, Schema } from "effect";
import { wrapClientCall } from "./utils/wrapClientCall";

// ==========================================================================
// Types
// ==========================================================================

export type PhoenixClientOptions = Parameters<typeof createClient>[0];
export const PHOENIX_LOCALHOST_BASE_URL = "http://127.0.0.1:6006";
export const PHOENIX_SKYHIGH_BASE_URL = "http://example.com:6006";

export type PhoenixGraphqlRequest<A, I = A> = {
  operationName: string;
  query: string;
  schema: Schema.Schema<A, I>;
  variables?: Record<string, unknown>;
};

// ==========================================================================
// Errors
// ==========================================================================

export class PhoenixSyncError extends Schema.TaggedError<PhoenixSyncError>()("PhoenixSyncError", {
  cause: Schema.Defect,
}) {}

export class PhoenixAsyncError extends Schema.TaggedError<PhoenixAsyncError>()(
  "PhoenixAsyncError",
  { cause: Schema.Defect },
) {}

export class PhoenixGraphqlError extends Schema.TaggedError<PhoenixGraphqlError>()(
  "PhoenixGraphqlError",
  {
    operationName: Schema.String,
    reason: Schema.String,
    errors: Schema.Array(Schema.String),
  },
) {}

export class PhoenixGraphqlDecodeError extends Schema.TaggedError<PhoenixGraphqlDecodeError>()(
  "PhoenixGraphqlDecodeError",
  {
    operationName: Schema.String,
    cause: Schema.Defect,
  },
) {}

export type PhoenixError = PhoenixSyncError | PhoenixAsyncError;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const toHeaders = (headersLike: PhoenixClientType["config"]["headers"]): Headers => {
  const headers = new Headers();

  if (headersLike instanceof Headers) {
    headersLike.forEach((value, key) => headers.set(key, value));
    return headers;
  }

  if (Array.isArray(headersLike)) {
    for (const [key, value] of headersLike) {
      headers.set(key, value);
    }
    return headers;
  }

  if (headersLike) {
    for (const [key, value] of Object.entries(headersLike)) {
      if (value == null) continue;
      if (Array.isArray(value)) {
        headers.set(key, value.map((item) => String(item)).join(", "));
      } else {
        headers.set(key, String(value));
      }
    }
  }

  return headers;
};

// ==========================================================================
// Service Definition
// ==========================================================================

export interface PhoenixClientImpl {
  readonly use: <T>(
    fn: (client: PhoenixClientType) => T,
  ) => Effect.Effect<Awaited<T>, PhoenixSyncError | PhoenixAsyncError>;
  readonly graphql: <A, I = A>(
    request: PhoenixGraphqlRequest<A, I>,
  ) => Effect.Effect<A, PhoenixAsyncError | PhoenixGraphqlError | PhoenixGraphqlDecodeError>;
  readonly health: () => Effect.Effect<void, PhoenixSyncError | PhoenixAsyncError>;
}

export class PhoenixClient extends Context.Tag("@app/PhoenixClient")<
  PhoenixClient,
  PhoenixClientImpl
>() {
  static readonly layer = (options?: PhoenixClientOptions) =>
    Layer.effect(
      PhoenixClient,
      Effect.sync(() => {
        const client = createClient(options);

        const use = <T>(
          fn: (c: PhoenixClientType) => T,
        ): Effect.Effect<Awaited<T>, PhoenixSyncError | PhoenixAsyncError> =>
          wrapClientCall(
            () => fn(client),
            (cause) => new PhoenixSyncError({ cause }),
            (cause) => new PhoenixAsyncError({ cause }),
          ).pipe(Effect.withSpan("PhoenixClient.use"));

        const graphql = <A, I = A>({
          operationName,
          query,
          schema,
          variables,
        }: PhoenixGraphqlRequest<A, I>): Effect.Effect<
          A,
          PhoenixAsyncError | PhoenixGraphqlError | PhoenixGraphqlDecodeError
        > =>
          Effect.gen(function* () {
            const baseUrl = client.config.baseUrl;
            if (!baseUrl) {
              return yield* new PhoenixGraphqlError({
                operationName,
                reason: "Phoenix client baseUrl is not configured",
                errors: [],
              });
            }

            const headers = toHeaders(client.config.headers);
            headers.set("content-type", "application/json");

            const response = yield* Effect.tryPromise({
              try: () =>
                fetch(new URL("/graphql", baseUrl), {
                  method: "POST",
                  headers,
                  body: JSON.stringify({
                    operationName,
                    query,
                    variables: variables ?? {},
                  }),
                }),
              catch: (cause) => new PhoenixAsyncError({ cause }),
            });

            const payload = yield* Effect.tryPromise({
              try: () => response.json() as Promise<unknown>,
              catch: (cause) => new PhoenixAsyncError({ cause }),
            });

            if (!response.ok) {
              return yield* new PhoenixGraphqlError({
                operationName,
                reason: `${response.status} ${response.statusText}`,
                errors: [],
              });
            }

            if (!isRecord(payload)) {
              return yield* new PhoenixGraphqlDecodeError({
                operationName,
                cause: new Error("Phoenix GraphQL response was not an object"),
              });
            }

            const errors =
              Array.isArray(payload.errors) && payload.errors.length > 0
                ? payload.errors.map((error) => {
                    if (!isRecord(error)) return String(error);
                    const message = error.message;
                    return typeof message === "string" ? message : JSON.stringify(error);
                  })
                : [];

            if (errors.length > 0) {
              return yield* new PhoenixGraphqlError({
                operationName,
                reason: "Phoenix GraphQL returned errors",
                errors,
              });
            }

            return yield* Schema.decodeUnknown(schema)(payload.data).pipe(
              Effect.mapError((cause) => new PhoenixGraphqlDecodeError({ operationName, cause })),
            );
          }).pipe(Effect.withSpan("PhoenixClient.graphql"));

        const health = Effect.fn("PhoenixClient.health")(function* () {
          yield* use((c) => c.GET("/v1/datasets"));
        });

        return { use, graphql, health };
      }),
    );

  static readonly Default = PhoenixClient.layer();

  static readonly localhost = PhoenixClient.layer({
    options: { baseUrl: PHOENIX_LOCALHOST_BASE_URL },
  });

  static readonly skyhigh = PhoenixClient.layer({
    options: { baseUrl: PHOENIX_SKYHIGH_BASE_URL },
  });
}
