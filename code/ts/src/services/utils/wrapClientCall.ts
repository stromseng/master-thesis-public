import { Effect } from "effect";

export const wrapClientCall = <T, ESync, EAsync>(
  fn: () => T,
  onSyncError: (cause: unknown) => ESync,
  onAsyncError: (cause: unknown) => EAsync,
): Effect.Effect<Awaited<T>, ESync | EAsync> =>
  Effect.gen(function* () {
    const result = yield* Effect.try({
      try: fn,
      catch: onSyncError,
    });

    if (result instanceof Promise) {
      return yield* Effect.tryPromise({
        try: () => result,
        catch: onAsyncError,
      });
    }

    return result as Awaited<T>;
  });
