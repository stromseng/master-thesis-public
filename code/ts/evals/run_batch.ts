#!/usr/bin/env bun
// Batch eval runner — runs every (model x script) combination with progress tracking.
//
// Note before running check out evals/batch.config.template.json5 for nessecary steps
//
// Usage:  bun evals/run_batch.ts
//
// Config resolution (highest priority wins):
//   1. BATCH_CONFIG env var -> path to a JSON5/JSON config file
//   2. BATCH_MODELS / BATCH_SCRIPTS / BATCH_PARALLEL env vars -> inline JSON overrides
//   3. Default file evals/batch.config.json5 (or .json)
//   4. Error -- no silent defaults, requires explicit config
//
// See evals/batch.config.template.json5 for the full list of available options.
//
// Flags:
//   --dry-run   Validate config and print the run plan without executing scripts

import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { Effect, Ref, Schedule, Duration } from "effect";
import * as Progress from "effective-progress";
import { type ModelEntry, type ScriptEntry, decodeBatchConfig } from "./batch-config";

// ── Constants ────────────────────────────────────────────────────────────────

const DRY_RUN = process.argv.includes("--dry-run");
const EVALS_DIR = import.meta.dir;
const DEFAULT_JSON5_PATH = resolve(EVALS_DIR, "batch.config.json5");
const DEFAULT_JSON_PATH = resolve(EVALS_DIR, "batch.config.json");

const MAX_WAIT_MS = 4 * 60 * 60 * 1000; // 4 hours
const POLL_INTERVAL_MS = 10_000; // 10 seconds

// ── Types ────────────────────────────────────────────────────────────────────

interface RunResult {
  provider: string;
  model: string;
  script: string;
  scriptKey: string;
  ok: boolean;
  durationMs: number;
  experimentId?: string;
}

type ResolvedEntry = { label: string; env: Record<string, string> };
type BatchMode = "normal" | "experiment-only" | "evaluate-only";
type RouterEndpointConfig = {
  readonly provider?: unknown;
  readonly port?: unknown;
  readonly baseUrl?: unknown;
};

// ── Pure helpers ─────────────────────────────────────────────────────────────

const fmtDuration = (ms: number) => {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
};

const fmtScript = (e: ScriptEntry) => {
  if (typeof e === "string") return e;
  const envStr = Object.entries(e.env)
    .map(([k, v]) => `${k}=${v}`)
    .join(" ");
  return `${e.script}  (${envStr})`;
};

const scriptName = (e: ScriptEntry) => (typeof e === "string" ? e : e.script);
const scriptMode = (e: ScriptEntry, defaultMode: BatchMode): BatchMode =>
  typeof e === "string" ? defaultMode : (e.mode ?? defaultMode);
const scriptExperimentId = (
  e: ScriptEntry,
  experimentIds: Readonly<Record<string, string>>,
): string | undefined =>
  typeof e === "string" ? experimentIds[e] : (e.experimentId ?? experimentIds[e.script]);

const scriptEnvSuffix = (e: ScriptEntry) => {
  if (typeof e === "string") return "";
  const entries = Object.entries(e.env);
  if (entries.length === 0) return "";
  return `  (${entries.map(([k, v]) => `${k}=${v}`).join(" ")})`;
};

const printBanner = (opts: {
  source: string;
  models: number;
  scripts: number;
  env: number;
  total: number;
  mode: BatchMode;
  parallel: boolean;
  concurrency?: number;
}) => {
  const title = DRY_RUN ? "Batch Eval Runner (dry run)" : "Batch Eval Runner";
  const bannerWidth = 62;
  const padding = bannerWidth - title.length;
  const leftPad = Math.floor(padding / 2);
  const rightPad = padding - leftPad;
  console.log(`\u2554${"=".repeat(bannerWidth)}\u2557`);
  console.log(`\u2551${" ".repeat(leftPad)}${title}${" ".repeat(rightPad)}\u2551`);
  console.log(`\u255a${"=".repeat(bannerWidth)}\u255d`);
  console.log(`  Config:  ${opts.source}`);
  console.log(`  Models:  ${opts.models}`);
  console.log(`  Scripts: ${opts.scripts}`);
  console.log(`  Env:     ${opts.env} global override(s)`);
  console.log(`  Mode:    ${opts.mode}`);
  console.log(`  Total:   ${opts.total} runs`);
  console.log(`  Parallel models: ${opts.parallel ? "yes" : "no"}`);
  if (opts.concurrency) console.log(`  Concurrency: ${opts.concurrency}`);
  console.log("");
};

const printDryRun = (
  resolvedModels: ResolvedEntry[],
  scripts: readonly ScriptEntry[],
  env: Readonly<Record<string, string>>,
) => {
  console.log("  Models:");
  for (const { label } of resolvedModels) {
    console.log(`    - ${label}`);
  }
  console.log("");
  const envEntries = Object.entries(env);
  if (envEntries.length > 0) {
    console.log("  Global env:");
    for (const [key, value] of envEntries) {
      console.log(`    - ${key}=${value}`);
    }
    console.log("");
  }
  console.log("  Scripts:");
  for (const s of scripts) {
    console.log(`    - ${fmtScript(s)}`);
  }
  console.log("");
  console.log("  Dry run complete -- config is valid.");
};

const printSummary = (results: RunResult[], totalDuration: number) => {
  const total = results.length;
  const passed = results.filter((r) => r.ok).length;
  const failed = results.filter((r) => !r.ok).length;

  console.log("\n");
  console.log("\u2554==============================================================\u2557");
  console.log("\u2551                         Summary                             \u2551");
  console.log("\u255a==============================================================\u255d");
  console.log(`  Total time: ${fmtDuration(totalDuration)}`);
  console.log(`  Passed: ${passed}/${total}    Failed: ${failed}/${total}`);

  if (failed > 0) {
    console.log("\n  Failed runs:");
    for (const r of results.filter((r) => !r.ok)) {
      console.log(`    \u2717  ${r.model}  x  ${r.script}`);
    }
  }

  console.log("");
};

// ── Effect-wrapped functions ─────────────────────────────────────────────────

const readConfigFile = (path: string) =>
  Effect.tryPromise(() => Bun.file(path).text()).pipe(Effect.map((text) => Bun.JSON5.parse(text)));

const loadConfig = Effect.gen(function* () {
  let raw: unknown;
  let source: string;

  const envConfigPath = process.env.BATCH_CONFIG;
  if (envConfigPath) {
    const absPath = resolve(envConfigPath);
    if (!existsSync(absPath)) {
      return yield* Effect.die(`Error: BATCH_CONFIG points to ${absPath} but it does not exist.`);
    }
    raw = yield* readConfigFile(absPath);
    source = `BATCH_CONFIG=${envConfigPath}`;
  } else if (existsSync(DEFAULT_JSON5_PATH)) {
    raw = yield* readConfigFile(DEFAULT_JSON5_PATH);
    source = "evals/batch.config.json5";
  } else if (existsSync(DEFAULT_JSON_PATH)) {
    raw = yield* readConfigFile(DEFAULT_JSON_PATH);
    source = "evals/batch.config.json";
  } else {
    return yield* Effect.die(
      [
        "Error: No batch config found.",
        "",
        "  Create a config file:",
        "    cp evals/batch.config.template.json5 evals/batch.config.json5",
        "",
        "  Or specify one via env:",
        "    BATCH_CONFIG=path/to/config.json5 bun evals/run_batch.ts",
      ].join("\n"),
    );
  }

  // Apply per-field env overrides
  const obj = raw as Record<string, unknown>;

  if (process.env.BATCH_MODELS) {
    // @effect-diagnostics-next-line preferSchemaOverJson:off
    obj.models = JSON.parse(process.env.BATCH_MODELS);
    source += " + BATCH_MODELS override";
  }
  if (process.env.BATCH_SCRIPTS) {
    // @effect-diagnostics-next-line preferSchemaOverJson:off
    obj.scripts = JSON.parse(process.env.BATCH_SCRIPTS);
    source += " + BATCH_SCRIPTS override";
  }
  if (process.env.BATCH_PARALLEL !== undefined) {
    obj.parallel = process.env.BATCH_PARALLEL === "true" || process.env.BATCH_PARALLEL === "1";
    source += " + BATCH_PARALLEL override";
  }

  const config = decodeBatchConfig(obj);
  return { config, source };
});

const detectVllmModel = (port: string) =>
  Effect.gen(function* () {
    const baseUrl = `http://localhost:${port}/v1`;
    const res = yield* Effect.tryPromise(() =>
      fetch(`${baseUrl}/models`).then((r) => r.json() as Promise<{ data?: { id: string }[] }>),
    ).pipe(
      Effect.catchAll(() =>
        Effect.die(`Could not reach vLLM at ${baseUrl}/models -- is it running?`),
      ),
    );

    const ids = Array.isArray(res?.data) ? res.data.map((m) => m.id) : [];
    if (ids.length === 0) {
      return yield* Effect.die(`vLLM on port ${port} has no models loaded.`);
    }
    if (ids.length > 1) {
      return yield* Effect.die(
        `vLLM on port ${port} serves ${ids.length} models -- specify "model" explicitly in config.\n  Available: ${ids.join(", ")}`,
      );
    }
    return ids[0]!;
  });

const resolveEntry = (entry: ModelEntry) =>
  Effect.gen(function* () {
    if (entry.provider === "litellm") {
      const short = entry.model.split("/").pop() ?? entry.model;
      return {
        label: `[litellm] ${short}`,
        env: { EVAL_MODEL: entry.model, EVAL_PROVIDER: "litellm" } as Record<string, string>,
      };
    }

    const model = entry.model ?? (yield* detectVllmModel(entry.port));
    const short = model.split("/").pop() ?? model;
    return {
      label: `[vllm :${entry.port}] ${short}`,
      env: { EVAL_MODEL: model, EVAL_PROVIDER: "vllm", VLLM_PORT: entry.port } as Record<
        string,
        string
      >,
    };
  });

const resolveEnvDrivenEntry = (env: Readonly<Record<string, string>>): ResolvedEntry => {
  const configuredModel = env.LANGUAGE_MODEL_ENDPOINTS ?? env.EVAL_MODEL;
  if (!configuredModel) {
    return {
      label: "[env] language model",
      env: {},
    };
  }

  try {
    const parsed = JSON.parse(configuredModel) as { model?: unknown };
    if (typeof parsed.model === "string") {
      const short = parsed.model.split("/").pop() ?? parsed.model;
      return {
        label: `[router] ${short}`,
        env: {},
      };
    }
  } catch {
    // Plain model env values are handled below.
  }

  const short = configuredModel.split("/").pop() ?? configuredModel;
  return {
    label: `[env] ${short}`,
    env: {},
  };
};

const portFromBaseUrl = (baseUrl: string) => {
  try {
    const url = new URL(baseUrl);
    return url.hostname === "localhost" || url.hostname === "127.0.0.1"
      ? url.port || undefined
      : undefined;
  } catch {
    return undefined;
  }
};

const isRouterEndpointConfig = (value: unknown): value is RouterEndpointConfig =>
  typeof value === "object" && value !== null;

const routerEndpointsFromConfig = (value: unknown): readonly RouterEndpointConfig[] | undefined => {
  if (Array.isArray(value)) return value.filter(isRouterEndpointConfig);
  if (typeof value !== "object" || value === null) return undefined;

  const endpoints = (value as { readonly endpoints?: unknown }).endpoints;
  return Array.isArray(endpoints) ? endpoints.filter(isRouterEndpointConfig) : undefined;
};

const extractRouterVllmPorts = (rawConfig: string | undefined): string[] => {
  if (!rawConfig) return [];

  try {
    const endpoints = routerEndpointsFromConfig(JSON.parse(rawConfig));
    if (!endpoints) return [];

    return endpoints.flatMap((endpoint) => {
      if (endpoint.provider !== "vllm") return [];
      if (typeof endpoint.port === "number") return [String(endpoint.port)];
      if (typeof endpoint.port === "string") return [endpoint.port];
      if (typeof endpoint.baseUrl === "string") {
        const port = portFromBaseUrl(endpoint.baseUrl);
        return port ? [port] : [];
      }
      return ["8000"];
    });
  } catch {
    return [];
  }
};

const vllmPortsFromRouterEnv = (env: Readonly<Record<string, string>>) => [
  ...extractRouterVllmPorts(env.LANGUAGE_MODEL_ENDPOINTS),
  ...extractRouterVllmPorts(env.JUDGE_LANGUAGE_MODEL_ENDPOINTS),
];

const waitForVllmHealth = (port: string) =>
  Effect.gen(function* () {
    const modelsUrl = `http://localhost:${port}/v1/models`;
    const maxAttempts = Math.floor(MAX_WAIT_MS / POLL_INTERVAL_MS);

    yield* Effect.logInfo(`Waiting for vLLM on port ${port} to become ready (up to 4 hours)...`);

    // Log at increasing intervals: ~17 checkpoints across 4 hours
    const logAtMs = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 180, 210, 240].map(
      (m) => m * 60_000,
    );
    let nextLogIdx = 0;
    const startTime = Date.now();

    const check = Effect.tryPromise(() =>
      fetch(modelsUrl, { signal: AbortSignal.timeout(5_000) }),
    ).pipe(
      Effect.filterOrFail(
        (res) => res.ok,
        (res) => `returned ${res.status}`,
      ),
      Effect.tapError(() => {
        const elapsed = Date.now() - startTime;
        if (nextLogIdx < logAtMs.length && elapsed >= logAtMs[nextLogIdx]!) {
          nextLogIdx++;
          return Effect.logInfo(
            `Still waiting for vLLM on port ${port}... (${fmtDuration(elapsed)} elapsed)`,
          );
        }
        return Effect.void;
      }),
    );

    const res = yield* check.pipe(
      Effect.retry(
        Schedule.intersect(
          Schedule.recurs(maxAttempts),
          Schedule.spaced(Duration.millis(POLL_INTERVAL_MS)),
        ),
      ),
      Effect.catchAll(() =>
        Effect.die(`vLLM on port ${port} did not become ready within 4 hours.`),
      ),
    );

    const body = yield* Effect.tryPromise(
      () => res.json() as Promise<{ data?: { id: string }[] }>,
    ).pipe(Effect.orElseSucceed(() => ({ data: [] as { id: string }[] })));

    const modelIds = Array.isArray(body?.data) ? body.data.map((m) => m.id) : [];

    yield* Effect.logInfo(`vLLM on port ${port} is ready`, { models: modelIds });
  });

// ── Script loading & execution ───────────────────────────────────────────────

/** Set env vars and return a restore function. */
const withEnv = (overrides: Record<string, string>) => {
  const prev: Record<string, string | undefined> = {};
  for (const [k, v] of Object.entries(overrides)) {
    prev[k] = process.env[k];
    process.env[k] = v;
  }
  return () => {
    for (const [k, v] of Object.entries(prev)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  };
};

type ScriptModule = {
  run?: Effect.Effect<void, unknown, never>;
  runExperimentOnly?: Effect.Effect<string, unknown, never>;
  runEvaluatorsOnly?: (experimentId: string) => Effect.Effect<void, unknown, never>;
};

/** Import a script module. */
const loadScriptModule = (scriptPath: string) =>
  Effect.tryPromise(async () => {
    // Resolve relative to the ts/ directory (parent of evals/)
    const absPath = resolve(EVALS_DIR, "..", scriptPath);
    // Cache-bust: import() caches by URL, so the same script with different env
    // vars would return stale module-level layer compositions (e.g. resolveDenseLayer).
    return (await import(`${absPath}?t=${Date.now()}`)) as ScriptModule;
  });

const loadScriptEffect = (scriptPath: string, mode: BatchMode, experimentId: string | undefined) =>
  loadScriptModule(scriptPath).pipe(
    Effect.map((mod) => {
      if (mode === "normal") {
        if (!mod.run) {
          throw new Error(
            `Script "${scriptPath}" does not export a \`run\` effect. ` +
              `Add \`export const run = Effect.scoped(program.pipe(Effect.provide(taskLayer)))\` to the script.`,
          );
        }
        return mod.run.pipe(Effect.as<void | string>(undefined));
      }

      if (mode === "experiment-only") {
        if (!mod.runExperimentOnly) {
          throw new Error(
            `Script "${scriptPath}" does not export \`runExperimentOnly\`; staged eval/judge runs require open-ended staged exports.`,
          );
        }
        return mod.runExperimentOnly;
      }

      if (!experimentId) {
        throw new Error(
          `Script "${scriptPath}" is in evaluate-only mode but no experiment ID was provided.`,
        );
      }
      if (!mod.runEvaluatorsOnly) {
        throw new Error(
          `Script "${scriptPath}" does not export \`runEvaluatorsOnly\`; staged eval/judge runs require open-ended staged exports.`,
        );
      }
      return mod.runEvaluatorsOnly(experimentId).pipe(Effect.as<void | string>(undefined));
    }),
  );

const runScript = (
  resolved: ResolvedEntry,
  scriptEntry: ScriptEntry,
  globalEnv: Readonly<Record<string, string>>,
  defaultMode: BatchMode,
  experimentIds: Readonly<Record<string, string>>,
  concurrency: number | undefined,
) =>
  Effect.gen(function* () {
    const script = scriptName(scriptEntry);
    const mode = scriptMode(scriptEntry, defaultMode);
    const experimentId = scriptExperimentId(scriptEntry, experimentIds);
    const scriptEnv = typeof scriptEntry === "string" ? {} : scriptEntry.env;
    const concurrencyEnv: Record<string, string> = concurrency
      ? { EVAL_CONCURRENCY: String(concurrency) }
      : {};

    // Set model + script + concurrency env vars for the imported module's layers
    const envOverrides = { ...concurrencyEnv, ...resolved.env, ...globalEnv, ...scriptEnv };
    const restoreEnv = withEnv(envOverrides);

    const start = Date.now();

    const exit = yield* Effect.exit(
      Effect.gen(function* () {
        const run = yield* loadScriptEffect(script, mode, experimentId);
        return yield* run;
      }).pipe(Effect.ensuring(Effect.sync(restoreEnv))),
    );
    const ok = exit._tag === "Success";
    const generatedExperimentId =
      ok && mode === "experiment-only" && typeof exit.value === "string" ? exit.value : undefined;

    if (!ok) {
      yield* Effect.logError(`Script "${script}" failed`, { exit });
    }

    const durationMs = Date.now() - start;

    return {
      provider:
        resolved.env.EVAL_PROVIDER ??
        globalEnv.EVAL_PROVIDER ??
        (globalEnv.LANGUAGE_MODEL_ENDPOINTS ? "router.queue" : "unknown"),
      model: resolved.label,
      scriptKey: script,
      script: `${script}${scriptEnvSuffix(scriptEntry)}`,
      ok,
      durationMs,
      ...(generatedExperimentId && { experimentId: generatedExperimentId }),
    } satisfies RunResult;
  });

// ── Main program ─────────────────────────────────────────────────────────────

const main = Effect.gen(function* () {
  const { config, source } = yield* loadConfig;
  const { models, scripts, env } = config;
  const modelCount = models.length > 0 ? models.length : 1;
  const total = modelCount * scripts.length;

  printBanner({
    source,
    models: modelCount,
    scripts: scripts.length,
    env: Object.keys(env).length,
    total,
    mode: config.mode,
    parallel: config.parallel,
    concurrency: config.concurrency,
  });

  // Wait for vLLM hosts to become healthy before resolving/importing layers.
  // Env-driven router configs hide vLLM endpoints from the `models` list, but
  // the LanguageModel layer still validates them via /models during import.
  const vllmPorts = [
    ...new Set([
      ...models.filter((m) => m.provider === "vllm").map((m) => m.port),
      ...vllmPortsFromRouterEnv(env),
    ]),
  ];
  if (vllmPorts.length > 0 && !DRY_RUN) {
    yield* Effect.all(vllmPorts.map(waitForVllmHealth), { concurrency: "unbounded" });
  }

  // Resolve all model entries (triggers auto-detect for vLLM without explicit model)
  const resolvedModels =
    models.length > 0 ? yield* Effect.all(models.map(resolveEntry)) : [resolveEnvDrivenEntry(env)];

  if (DRY_RUN) {
    printDryRun([...resolvedModels], scripts, env);
    return;
  }

  const resultsRef = yield* Ref.make<RunResult[]>([]);
  const batchStart = Date.now();

  // Run each model as a progress task, with child tasks per script
  const runModel = (resolved: ResolvedEntry) =>
    Progress.task(
      Effect.gen(function* () {
        const progress = yield* Progress.Progress;
        const modelTaskId = yield* Progress.Task;

        for (const scriptEntry of scripts) {
          const result = yield* Progress.task(
            runScript(
              resolved,
              scriptEntry,
              env,
              config.mode,
              config.experimentIds,
              config.concurrency,
            ),
            { description: scriptName(scriptEntry), parentId: modelTaskId },
          );

          if (result.ok) {
            yield* progress.incrementSucceeded(modelTaskId, 1);
          } else {
            yield* progress.incrementFailed(modelTaskId, 1);
          }

          yield* Ref.update(resultsRef, (rs) => [...rs, result]);
        }
      }),
      { description: resolved.label, total: scripts.length },
    );

  if (config.parallel) {
    yield* Effect.all(resolvedModels.map(runModel), { concurrency: "unbounded" });
  } else {
    for (const resolved of resolvedModels) {
      yield* runModel(resolved);
    }
  }

  const results = yield* Ref.get(resultsRef);
  const totalDuration = Date.now() - batchStart;
  printSummary(results, totalDuration);

  if (config.experimentIdsOutput) {
    const experimentIds = Object.fromEntries(
      results.flatMap((result) => {
        if (!result.experimentId) return [];
        return [[result.scriptKey, result.experimentId]];
      }),
    );
    yield* Effect.tryPromise(() =>
      Bun.write(config.experimentIdsOutput!, JSON.stringify(experimentIds, null, 2)),
    );
  }

  // Exit with non-zero if any run failed
  const failed = results.filter((r) => !r.ok).length;
  if (failed > 0) {
    return yield* Effect.die(`${failed}/${total} runs failed`);
  }
});

Effect.runPromise(main).catch((error) => {
  console.error(error);
  process.exit(1);
});
