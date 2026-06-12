import { Config, Effect, Layer } from "effect";
import { createClient, createConfig, type Client } from "../generated/python-api/client";

const DEFAULT_PYTHON_API_URL = "http://localhost:8001";
const DEFAULT_PYTHON_API_SKYHIGH_URL = "http://example.com:8001";

export interface PythonApiClientService {
  readonly client: Client;
}

const makePythonApiClient = (baseUrl: string): PythonApiClientService => ({
  client: createClient(createConfig({ baseUrl })),
});

export class PythonApiClient extends Effect.Service<PythonApiClient>()("@app/PythonApiClient", {
  effect: Effect.gen(function* () {
    const baseUrl = yield* Config.string("PYTHON_API_URL").pipe(
      Config.orElse(() => Config.succeed(DEFAULT_PYTHON_API_URL)),
    );
    return makePythonApiClient(baseUrl);
  }),
}) {
  static readonly layer = (baseUrl: string) =>
    Layer.succeed(PythonApiClient, new PythonApiClient(makePythonApiClient(baseUrl)));

  static readonly skyhigh = Layer.effect(
    PythonApiClient,
    Effect.gen(function* () {
      const baseUrl = yield* Config.string("SKYHIGH_URL").pipe(
        Config.orElse(() => Config.succeed(DEFAULT_PYTHON_API_SKYHIGH_URL)),
      );
      return new PythonApiClient(makePythonApiClient(baseUrl));
    }),
  );
}
