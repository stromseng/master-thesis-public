import { describe, expect, it } from "@effect/vitest";
import { Effect } from "effect";
import { PhoenixClient } from "../src/services/PhoenixClient";

describe("PhoenixClient skyhigh", () => {
  it.live("can list datasets", () =>
    Effect.gen(function* () {
      const phoenix = yield* PhoenixClient;
      yield* phoenix.health();
      const result = yield* phoenix.use((client) => client.GET("/v1/datasets"));

      expect(result).toBeDefined();
    }).pipe(Effect.provide(PhoenixClient.skyhigh)),
  );
});
