import { Effect, Schema, Logger } from "effect";

class InnerDatabaseError extends Schema.TaggedError<InnerDatabaseError>()("InnerDatabaseError", {
  innerContext: Schema.String,
  cause: Schema.optional(Schema.Defect),
}) {}

class DatabaseError extends Schema.TaggedError<DatabaseError>()("DatabaseError", {
  reason: Schema.String,
  cause: Schema.Union(InnerDatabaseError, Schema.Defect),
}) {}

const failingDbQuery = () => {
  const err = new Error("Connection refused to postgres:5432");
  (err as any).code = "ECONNREFUSED";
  throw err;
};

const innerFunction = Effect.fn("innerFunction")(function* () {
  return yield* Effect.tryPromise({
    try: async () => failingDbQuery(),
    catch: (error) =>
      new InnerDatabaseError({
        innerContext: "Failed to connect to the database",
        cause: error,
      }),
  });
});

const queryDatabase = Effect.fn("queryDatabase")(function* (query: string) {
  return yield* innerFunction().pipe(
    Effect.catchTag("InnerDatabaseError", (err) =>
      Effect.fail(
        new DatabaseError({
          reason: `Database query failed for query: ${query}`,
          cause: err,
        }),
      ),
    ),
  );
});

await queryDatabase("my query").pipe(
  Effect.tapError((err) => Effect.logDebug(`Error occurred: ${err.cause}`)),
  Effect.catchAllCause((cause) => Effect.logError(cause)),
  Effect.provide(Logger.pretty),
  Effect.runPromise,
);
