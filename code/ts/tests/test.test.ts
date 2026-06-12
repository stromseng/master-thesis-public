import { describe, expect, it } from "@effect/vitest";
import { Effect } from "effect";

describe("Calculator", () => {
  it("creates instances", () => {
    const result = 1 + 1;
    expect(result).toBe(2);
  });

  it.effect("adds numbers with Effect", () =>
    Effect.gen(function* () {
      const result = yield* Effect.succeed(1 + 1);
      expect(result).toBe(2);
    }),
  );
});
