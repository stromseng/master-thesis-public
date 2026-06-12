import { describe, expect, it } from "@effect/vitest";
import { RetryError } from "ai";
import { Duration, Effect } from "effect";
import { waitAndRetryOnConnectivityFailure } from "../../evals/experiment_setup";
import { LanguageModelError } from "../../src/services/LanguageModel";

const makeConnectivityError = () =>
  LanguageModelError.make({
    reason: "Failed to generate object",
    cause: new RetryError({
      message:
        "Failed after 3 attempts. Last error: Cannot connect to API: Unable to connect. Is the computer able to access the url?",
      reason: "maxRetriesExceeded",
      errors: [
        new Error(
          "Cannot connect to API: Unable to connect. Is the computer able to access the url?",
        ),
      ],
    }),
  });

const makeNonConnectivityError = () =>
  LanguageModelError.make({
    reason: "Failed to generate object",
    cause: new RetryError({
      message: "Failed after 3 attempts. Last error: rate limit exceeded",
      reason: "maxRetriesExceeded",
      errors: [new Error("rate limit exceeded")],
    }),
  });

describe("waitAndRetryOnConnectivityFailure", () => {
  it.live("polls until connectivity failures recover", () =>
    Effect.gen(function* () {
      let attempts = 0;

      const effect = Effect.suspend(() => {
        attempts += 1;
        return attempts < 3 ? Effect.fail(makeConnectivityError()) : Effect.succeed("ok");
      });

      const result = yield* waitAndRetryOnConnectivityFailure(effect, {
        pollInterval: Duration.millis(1),
      });

      expect(result).toBe("ok");
      expect(attempts).toBe(3);
    }),
  );

  it.effect("does not swallow unrelated retry errors", () =>
    Effect.gen(function* () {
      let attempts = 0;

      const effect = Effect.suspend(() => {
        attempts += 1;
        return Effect.fail(makeNonConnectivityError());
      });

      const result = yield* waitAndRetryOnConnectivityFailure(effect, {
        pollInterval: Duration.millis(1),
      }).pipe(Effect.either);

      expect(result._tag).toBe("Left");
      expect(attempts).toBe(1);
    }),
  );
});
