#!/usr/bin/env bun
// Pipeline orchestrator — cycles through local IDUN-hosted eval/judge models,
// running eval batches and staged open-ended judging automatically.
//
// Usage:
//   bun evals/run_pipeline.ts
//   bun evals/run_pipeline.ts --dry-run
//   bun evals/run_pipeline.ts --config path/to/pipeline.config.json5

import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { Schema } from "effect";
import { type ScriptEntry, ScriptEntrySchema } from "./batch-config";

const DRY_RUN = process.argv.includes("--dry-run");
const EVALS_DIR = import.meta.dir;
const TS_DIR = resolve(EVALS_DIR, "..");
const REPO_ROOT = resolve(TS_DIR, "../..");
const PY_DIR = resolve(REPO_ROOT, "code/python");
const BATCH_CONFIG_PATH = resolve(EVALS_DIR, "batch.config.json5");
const IDUN_ENV = { ...process.env, PYTHONPATH: "src" };

type Backend = "vllm" | "sglang";
const BackendSchema = Schema.Literal("vllm", "sglang");

const PositiveIntSchema = Schema.Number.pipe(Schema.int(), Schema.positive());

const LocalModelSpecSchema = Schema.Struct({
  provider: Schema.optional(BackendSchema),
  model: Schema.String,
  jobId: Schema.optional(Schema.String),
  localPort: Schema.optional(PositiveIntSchema),
  serveArgs: Schema.optional(Schema.String),
  maxConcurrency: Schema.optional(PositiveIntSchema),
});

const LocalEndpointSpecSchema = Schema.Struct({
  provider: BackendSchema,
  jobId: Schema.String,
  localPort: PositiveIntSchema,
  serveArgs: Schema.optional(Schema.String),
  maxConcurrency: PositiveIntSchema,
});

const LiteLLMEndpointSpecSchema = Schema.Struct({
  provider: Schema.Literal("litellm"),
  baseUrl: Schema.optional(Schema.String),
  apiKeyConfigKey: Schema.optional(Schema.String),
  models: Schema.optional(Schema.NonEmptyArray(Schema.String)),
  maxConcurrency: PositiveIntSchema,
});

const LiteLLMJudgeSpecSchema = Schema.Struct({
  provider: Schema.Literal("litellm"),
  model: Schema.String,
  maxConcurrency: Schema.optional(PositiveIntSchema),
});

const RoutedModelSpecSchema = Schema.Struct({
  model: Schema.String,
  endpoints: Schema.NonEmptyArray(Schema.Union(LocalEndpointSpecSchema, LiteLLMEndpointSpecSchema)),
});

const EvalModelSpecSchema = Schema.Union(
  Schema.String,
  RoutedModelSpecSchema,
  LocalModelSpecSchema,
  LiteLLMJudgeSpecSchema,
);
const JudgeModelSpecSchema = EvalModelSpecSchema;

const PipelineRun = Schema.Struct({
  model: Schema.optional(Schema.String),
  evalModel: Schema.optional(EvalModelSpecSchema),
  judgeModel: Schema.optional(JudgeModelSpecSchema),
  scripts: Schema.NonEmptyArray(ScriptEntrySchema),
  backend: Schema.optional(BackendSchema),
  serveArgs: Schema.optional(Schema.String),
  concurrency: Schema.optional(PositiveIntSchema),
});

const PipelineConfig = Schema.Struct({
  jobId: Schema.optionalWith(Schema.String, { default: () => "auto" }),
  localPort: Schema.optionalWith(PositiveIntSchema, { default: () => 8000 }),
  backend: Schema.optionalWith(BackendSchema, { default: (): Backend => "vllm" }),
  runs: Schema.NonEmptyArray(PipelineRun),
});

type PipelineConfig = typeof PipelineConfig.Type;
type PipelineRun = typeof PipelineRun.Type;
type LocalModelSpec = typeof LocalModelSpecSchema.Type;
type LocalEndpointSpec = typeof LocalEndpointSpecSchema.Type;
type LiteLLMEndpointSpec = typeof LiteLLMEndpointSpecSchema.Type;
type RoutedModelSpec = typeof RoutedModelSpecSchema.Type;
type JudgeModelSpec = typeof JudgeModelSpecSchema.Type;

const decodePipelineConfig = Schema.decodeUnknownSync(PipelineConfig);

interface ExistingServerState {
  found: boolean;
  model?: string;
  backend?: string;
  running?: boolean;
  status?: string;
}

type LocalModelRequest = {
  kind: "local";
  model: string;
  backend: Backend;
  jobId: string;
  localPort: number;
  serveArgs?: string;
  maxConcurrency?: number;
};

type LiteLLMModelRequest = {
  kind: "litellm";
  model: string;
  baseUrl?: string;
  apiKeyConfigKey?: string;
  models?: readonly [string, ...string[]];
  maxConcurrency?: number;
};

type ModelEndpointRequest = LocalModelRequest | LiteLLMModelRequest;

type RoleModelRequest = {
  model: string;
  endpoints: readonly ModelEndpointRequest[];
};

type NormalizedRun = {
  label: string;
  evalModel: RoleModelRequest;
  judgeModel?: RoleModelRequest;
  scripts: readonly ScriptEntry[];
  concurrency?: number;
};

interface RunModelResult {
  model: string;
  ok: boolean;
  skipped: boolean;
  durationMs: number;
}

type TunnelProc = ReturnType<typeof Bun.spawn>;

const fmtDuration = (ms: number): string => {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
};

const scriptName = (e: ScriptEntry): string => (typeof e === "string" ? e : e.script);

const withScriptMode = (
  entry: ScriptEntry,
  mode: "normal" | "experiment-only" | "evaluate-only",
): ScriptEntry =>
  typeof entry === "string" ? { script: entry, env: {}, mode } : { ...entry, mode };

const isLiteLLMModel = (spec: unknown): spec is typeof LiteLLMJudgeSpecSchema.Type =>
  typeof spec === "object" && spec !== null && "provider" in spec && spec.provider === "litellm";

const isRoutedModel = (spec: unknown): spec is RoutedModelSpec =>
  typeof spec === "object" && spec !== null && "endpoints" in spec;

const isLiteLLMEndpoint = (spec: unknown): spec is LiteLLMEndpointSpec =>
  typeof spec === "object" && spec !== null && "provider" in spec && spec.provider === "litellm";

const localRequestKey = (request: LocalModelRequest): string =>
  JSON.stringify({
    model: request.model,
    backend: request.backend,
    serveArgs: request.serveArgs ?? "",
  });

const normalizeLocalEndpoint = ({
  model,
  spec,
  run,
  config,
  jobId,
}: {
  model: string;
  spec: LocalEndpointSpec;
  run: PipelineRun;
  config: PipelineConfig;
  jobId: string;
}): LocalModelRequest => ({
  kind: "local",
  model,
  backend: spec.provider ?? run.backend ?? config.backend,
  jobId: spec.jobId ?? jobId,
  localPort: spec.localPort ?? config.localPort,
  serveArgs: spec.serveArgs ?? run.serveArgs,
  maxConcurrency: spec.maxConcurrency,
});

const normalizeLiteLLMEndpoint = (
  model: string,
  spec: LiteLLMEndpointSpec,
): LiteLLMModelRequest => ({
  kind: "litellm",
  model,
  baseUrl: spec.baseUrl,
  apiKeyConfigKey: spec.apiKeyConfigKey,
  models: spec.models,
  maxConcurrency: spec.maxConcurrency,
});

const normalizeRoleModel = ({
  spec,
  run,
  config,
  jobId,
}: {
  spec: string | LocalModelSpec | JudgeModelSpec;
  run: PipelineRun;
  config: PipelineConfig;
  jobId: string;
}): RoleModelRequest => {
  if (typeof spec === "string") {
    return {
      model: spec,
      endpoints: [
        {
          kind: "local",
          model: spec,
          backend: run.backend ?? config.backend,
          jobId,
          localPort: config.localPort,
          serveArgs: run.serveArgs,
        },
      ],
    };
  }

  if (isRoutedModel(spec)) {
    return {
      model: spec.model,
      endpoints: spec.endpoints.map((endpoint) =>
        isLiteLLMEndpoint(endpoint)
          ? normalizeLiteLLMEndpoint(spec.model, endpoint)
          : normalizeLocalEndpoint({ model: spec.model, spec: endpoint, run, config, jobId }),
      ),
    };
  }

  if (isLiteLLMModel(spec)) {
    return {
      model: spec.model,
      endpoints: [
        {
          kind: "litellm",
          model: spec.model,
          maxConcurrency: spec.maxConcurrency,
        },
      ],
    };
  }

  return {
    model: spec.model,
    endpoints: [
      {
        kind: "local",
        model: spec.model,
        backend: spec.provider ?? run.backend ?? config.backend,
        jobId: spec.jobId ?? jobId,
        localPort: spec.localPort ?? config.localPort,
        serveArgs: spec.serveArgs ?? run.serveArgs,
        maxConcurrency: spec.maxConcurrency,
      },
    ],
  };
};

const normalizeRuns = (config: PipelineConfig, jobId: string): NormalizedRun[] =>
  config.runs.map((run, index) => {
    const evalSpec = run.evalModel ?? run.model;
    if (!evalSpec) {
      throw new Error(`Pipeline run ${index + 1} must set either "model" or "evalModel".`);
    }

    const evalModel = normalizeRoleModel({ spec: evalSpec, run, config, jobId });
    const judgeModel = run.judgeModel
      ? normalizeRoleModel({ spec: run.judgeModel, run, config, jobId })
      : undefined;

    return {
      label: judgeModel ? `${evalModel.model} -> judge ${judgeModel.model}` : evalModel.model,
      evalModel,
      judgeModel,
      scripts: run.scripts,
      concurrency: run.concurrency,
    };
  });

const endpointConfig = (request: ModelEndpointRequest): Record<string, unknown> => {
  if (request.kind === "litellm") {
    return {
      provider: "litellm",
      ...(request.baseUrl && { baseUrl: request.baseUrl }),
      ...(request.apiKeyConfigKey && { apiKeyConfigKey: request.apiKeyConfigKey }),
      ...(request.models && { models: request.models }),
      ...(request.maxConcurrency != null && { maxConcurrency: request.maxConcurrency }),
    };
  }

  return {
    provider: "vllm",
    port: request.localPort,
    ...(request.maxConcurrency != null && { maxConcurrency: request.maxConcurrency }),
  };
};

const modelEndpointEnv = (request: RoleModelRequest): string =>
  JSON.stringify({
    model: request.model,
    endpoints: request.endpoints.map(endpointConfig),
  });

const judgeEnv = (request: RoleModelRequest): Record<string, string> => ({
  JUDGE_LANGUAGE_MODEL_ENDPOINTS: modelEndpointEnv(request),
});

const evalEnv = (request: RoleModelRequest): Record<string, string> => ({
  LANGUAGE_MODEL_ENDPOINTS: modelEndpointEnv(request),
});

const localEndpoints = (request: RoleModelRequest): LocalModelRequest[] =>
  request.endpoints.filter((endpoint): endpoint is LocalModelRequest => endpoint.kind === "local");

const runsNeedAutoJobId = (runs: readonly NormalizedRun[]): boolean =>
  runs.some((run) =>
    [run.evalModel, run.judgeModel].some((model) =>
      model?.endpoints.some((endpoint) => endpoint.kind === "local" && endpoint.jobId === "auto"),
    ),
  );

const runsHaveLocalEndpoints = (runs: readonly NormalizedRun[]): boolean =>
  runs.some((run) =>
    [run.evalModel, run.judgeModel].some((model) =>
      model?.endpoints.some((endpoint) => endpoint.kind === "local"),
    ),
  );

const printBanner = (config: PipelineConfig, runs: readonly NormalizedRun[]): void => {
  const totalScripts = runs.reduce((n, r) => n + r.scripts.length, 0);
  const title = DRY_RUN ? "Pipeline Orchestrator (dry run)" : "Pipeline Orchestrator";
  const w = 62;
  const pad = w - title.length;
  const lp = Math.floor(pad / 2);
  const rp = pad - lp;

  console.log(`\u2554${"=".repeat(w)}\u2557`);
  console.log(`\u2551${" ".repeat(lp)}${title}${" ".repeat(rp)}\u2551`);
  console.log(`\u255a${"=".repeat(w)}\u255d`);
  if (runsHaveLocalEndpoints(runs)) {
    console.log(`  Job ID:   ${config.jobId}`);
    console.log(`  Port:     ${config.localPort}`);
    console.log(`  Backend:  ${config.backend} (default)`);
  } else {
    console.log("  IDUN:     not required (all endpoints are hosted externally)");
  }
  console.log(`  Runs:     ${runs.length}`);
  console.log(`  Scripts:  ${totalScripts} total`);
  console.log("");
  for (const [i, run] of runs.entries()) {
    console.log(`  [${i + 1}] ${run.label} (${run.scripts.length} scripts)`);
    for (const endpoint of run.evalModel.endpoints) {
      if (endpoint.kind === "litellm") {
        console.log(`      eval:  litellm ${run.evalModel.model}`);
      } else {
        console.log(
          `      eval:  ${endpoint.backend} ${run.evalModel.model} job=${endpoint.jobId} port=${endpoint.localPort}`,
        );
        if (endpoint.serveArgs) console.log(`      eval serve args: ${endpoint.serveArgs}`);
      }
    }
    if (run.judgeModel) {
      for (const endpoint of run.judgeModel.endpoints) {
        if (endpoint.kind === "litellm") {
          console.log(`      judge: litellm ${run.judgeModel.model}`);
        } else {
          console.log(
            `      judge: ${endpoint.backend} ${run.judgeModel.model} job=${endpoint.jobId} port=${endpoint.localPort}`,
          );
          if (endpoint.serveArgs) console.log(`      judge serve args: ${endpoint.serveArgs}`);
        }
      }
    }
    for (const s of run.scripts) console.log(`      - ${scriptName(s)}`);
  }
  console.log("");
};

const printSummary = (results: RunModelResult[], totalMs: number): void => {
  const passed = results.filter((r) => r.ok).length;
  const failed = results.filter((r) => !r.ok && !r.skipped).length;
  const skipped = results.filter((r) => r.skipped).length;

  console.log(`\n\u2554${"=".repeat(62)}\u2557`);
  console.log(`\u2551                       Pipeline Summary                       \u2551`);
  console.log(`\u255a${"=".repeat(62)}\u255d`);
  console.log(`  Total time: ${fmtDuration(totalMs)}`);
  console.log(
    `  Passed: ${passed}/${results.length}  Failed: ${failed}/${results.length}  Skipped: ${skipped}/${results.length}`,
  );
  console.log("");
  for (const r of results) {
    const icon = r.ok ? "\u2713" : r.skipped ? "SKIP" : "\u2717";
    console.log(`  [${icon}] ${r.model} (${fmtDuration(r.durationMs)})`);
  }
  console.log("");
};

const loadPipelineConfig = async (configPath?: string): Promise<PipelineConfig> => {
  const defaultPath = resolve(EVALS_DIR, "pipeline.config.json5");
  const path = configPath ? resolve(configPath) : defaultPath;

  if (!existsSync(path)) {
    console.error(`Pipeline config not found: ${path}`);
    console.error("");
    console.error("  Create a config file:");
    console.error("    cp evals/pipeline.config.template.json5 evals/pipeline.config.json5");
    console.error("");
    console.error("  Or specify one via flag:");
    console.error("    bun evals/run_pipeline.ts --config path/to/config.json5");
    process.exit(1);
  }

  const text = await Bun.file(path).text();
  const raw = Bun.JSON5.parse(text);
  return decodePipelineConfig(raw);
};

const exec = async (
  cmd: string[],
  opts?: { cwd?: string; env?: Record<string, string | undefined> },
): Promise<{ exitCode: number; stdout: string; stderr: string }> => {
  const proc = Bun.spawn(cmd, {
    cwd: opts?.cwd ?? REPO_ROOT,
    env: opts?.env,
    stdin: null,
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  const exitCode = await proc.exited;
  return { exitCode, stdout, stderr };
};

const idunCmd = (...args: string[]): string[] => ["uv", "run", "python", "-m", "idun", ...args];

const execIdun = async (
  args: string[],
): Promise<{ exitCode: number; stdout: string; stderr: string }> =>
  exec(idunCmd(...args), {
    cwd: PY_DIR,
    env: IDUN_ENV,
  });

const detectJobId = async (): Promise<string> => {
  console.log("  Auto-detecting SLURM job ID...");
  const result = await exec([
    "ssh",
    "idun",
    "squeue -u $USER -t RUNNING -p GPUQ -o '%i' --noheader",
  ]);
  if (result.exitCode !== 0) {
    console.error("  Failed to query SLURM:");
    if (result.stderr.trim()) console.error(`  ${result.stderr.trim()}`);
    process.exit(1);
  }

  const jobIds = result.stdout.trim().split("\n").filter(Boolean);

  if (jobIds.length === 0) {
    console.error(
      "  No RUNNING GPU jobs found. Submit a job first with `cd code/python && PYTHONPATH=src uv run python -m idun submit`.",
    );
    process.exit(1);
  }
  if (jobIds.length > 1) {
    console.error(`  Multiple RUNNING GPU jobs found: ${jobIds.join(", ")}`);
    console.error("  Set jobId explicitly in pipeline.config.json5");
    process.exit(1);
  }

  const jobId = jobIds[0]!;
  console.log(`  Detected job: ${jobId}`);
  return jobId;
};

const getExistingServerState = async (jobId: string): Promise<ExistingServerState | null> => {
  const result = await execIdun(["vllm", "inspect", "--job-id", jobId, "--json"]);

  if (result.exitCode !== 0) {
    const msg = result.stderr.trim() || result.stdout.trim();
    if (msg) console.warn(`  Could not inspect existing server state: ${msg}`);
    return null;
  }

  try {
    return JSON.parse(result.stdout) as ExistingServerState;
  } catch {
    const msg = result.stdout.trim();
    if (msg) console.warn(`  Could not parse existing server state: ${msg}`);
    return null;
  }
};

const stopServer = async (jobId: string): Promise<void> => {
  console.log(`  Stopping inference server on job ${jobId}...`);
  const result = await execIdun(["vllm", "stop", "--job-id", jobId]);
  const msg = result.stderr.trim() || result.stdout.trim();
  if (msg) console.log(`  ${msg}`);
};

const serveModel = async (request: LocalModelRequest): Promise<boolean> => {
  console.log(`  Serving model: ${request.model} [${request.backend}] (job ${request.jobId})...`);
  const cmd = [
    "vllm",
    "serve",
    request.model,
    "--job-id",
    request.jobId,
    "--backend",
    request.backend,
  ];
  if (request.serveArgs) cmd.push("--extra", request.serveArgs);
  const result = await execIdun(cmd);
  if (result.exitCode !== 0) {
    console.error(`  Failed to serve ${request.model}:`);
    const msg = result.stderr.trim() || result.stdout.trim();
    if (msg) console.error(`  ${msg}`);
    return false;
  }
  const msg = result.stdout.trim();
  if (msg) {
    for (const line of msg.split("\n").slice(-5)) console.log(`  ${line}`);
  }
  return true;
};

const spawnTunnel = (request: LocalModelRequest): TunnelProc => {
  console.log(`  Starting SSH tunnel for job ${request.jobId} (localhost:${request.localPort})...`);
  return Bun.spawn(
    idunCmd(
      "vllm",
      "connect",
      "--job-id",
      request.jobId,
      "--local-port",
      String(request.localPort),
    ),
    {
      cwd: PY_DIR,
      env: IDUN_ENV,
      stdin: null,
      stdout: "pipe",
      stderr: "pipe",
    },
  );
};

const killTunnel = async (proc: TunnelProc, localPort: number): Promise<void> => {
  console.log(`  Closing SSH tunnel on localhost:${localPort}...`);
  proc.kill();
  await Promise.race([proc.exited, Bun.sleep(2000)]);

  try {
    Bun.spawnSync(["bash", "-c", `lsof -ti:${localPort} | xargs kill 2>/dev/null`]);
  } catch {
    /* ignore */
  }
};

const writeBatchConfig = async (config: {
  env: Record<string, string>;
  scripts: readonly ScriptEntry[];
  concurrency?: number;
  mode?: "normal" | "experiment-only" | "evaluate-only";
  experimentIds?: Record<string, string>;
  experimentIdsOutput?: string;
}): Promise<void> => {
  const batchConfig = {
    env: config.env,
    scripts: config.scripts,
    parallel: false,
    mode: config.mode ?? "normal",
    ...(config.experimentIds && { experimentIds: config.experimentIds }),
    ...(config.experimentIdsOutput && { experimentIdsOutput: config.experimentIdsOutput }),
    ...(config.concurrency != null && { concurrency: config.concurrency }),
  };
  await Bun.write(BATCH_CONFIG_PATH, JSON.stringify(batchConfig, null, 2));
};

const runBatch = async (): Promise<boolean> => {
  console.log("  Running batch eval...\n");
  const proc = Bun.spawn(["bun", "evals/run_batch.ts"], {
    cwd: TS_DIR,
    stdin: null,
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await proc.exited;
  return exitCode === 0;
};

const readExperimentIds = async (path: string): Promise<Record<string, string>> => {
  const text = await Bun.file(path).text();
  return JSON.parse(text) as Record<string, string>;
};

const hasStagedExports = async (script: ScriptEntry): Promise<boolean> => {
  const path = resolve(TS_DIR, scriptName(script));
  if (!existsSync(path)) {
    throw new Error(`Script file does not exist: ${scriptName(script)}`);
  }
  const source = await Bun.file(path).text();
  return (
    /\bexport\s+const\s+runExperimentOnly\b/.test(source) &&
    /\bexport\s+const\s+runEvaluatorsOnly\b/.test(source)
  );
};

const splitScriptsByStagedExports = async (
  scripts: readonly ScriptEntry[],
): Promise<{ normal: ScriptEntry[]; staged: ScriptEntry[] }> => {
  const checks = await Promise.all(
    scripts.map(async (script) => [script, await hasStagedExports(script)] as const),
  );
  return {
    normal: checks.filter(([, staged]) => !staged).map(([script]) => script),
    staged: checks.filter(([, staged]) => staged).map(([script]) => script),
  };
};

const main = async (): Promise<void> => {
  const configFlag = process.argv.indexOf("--config");
  const configPath = configFlag !== -1 ? process.argv[configFlag + 1] : undefined;

  const config = await loadPipelineConfig(configPath);
  let jobId = config.jobId;
  let runs = normalizeRuns(config, jobId);
  if (config.jobId === "auto" && runsNeedAutoJobId(runs) && !DRY_RUN) {
    jobId = await detectJobId();
    runs = normalizeRuns(config, jobId);
  }
  printBanner({ ...config, jobId }, runs);

  if (DRY_RUN) {
    console.log("  Dry run complete -- config is valid.");
    return;
  }

  const servedByJob = new Map<string, string>();
  const tunnelsByJob = new Map<string, { proc: TunnelProc; port: number }>();

  const closeTunnelForJob = async (jobIdToClose: string): Promise<void> => {
    const tunnel = tunnelsByJob.get(jobIdToClose);
    if (!tunnel) return;
    await killTunnel(tunnel.proc, tunnel.port);
    tunnelsByJob.delete(jobIdToClose);
  };

  const ensureLocalModel = async (request: LocalModelRequest): Promise<boolean> => {
    const requestedKey = localRequestKey(request);
    const existingServer = await getExistingServerState(request.jobId);
    const existingMatches =
      existingServer?.found === true &&
      existingServer.running === true &&
      existingServer.model === request.model &&
      existingServer.backend === request.backend;

    if (!existingMatches) {
      if (existingServer?.found) {
        const actualBackend = existingServer.backend ?? "unknown";
        const actualModel = existingServer.model ?? "unknown";
        const status = existingServer.status ?? "unknown";
        console.log(
          `  Existing server on job ${request.jobId} does not match scheduled phase (${actualBackend} ${actualModel}, ${status}); restarting.`,
        );
      }

      await closeTunnelForJob(request.jobId);
      await stopServer(request.jobId);
      const served = await serveModel(request);
      if (!served) return false;
      servedByJob.set(request.jobId, requestedKey);
    } else {
      servedByJob.set(request.jobId, requestedKey);
      const status = existingServer.status ?? "OK";
      console.log(
        `  Reusing existing ${request.backend} server for ${request.model} on job ${request.jobId} (${status}).`,
      );
    }

    const activeTunnel = tunnelsByJob.get(request.jobId);
    if (activeTunnel?.port !== request.localPort) {
      if (activeTunnel) await closeTunnelForJob(request.jobId);
      tunnelsByJob.set(request.jobId, { proc: spawnTunnel(request), port: request.localPort });
      await Bun.sleep(2000);
    }

    return true;
  };

  const runBatchPhase = async ({
    phase,
    localModels,
    env,
    scripts,
    concurrency,
    mode,
    experimentIds,
    experimentIdsOutput,
  }: {
    phase: string;
    localModels?: readonly LocalModelRequest[];
    env: Record<string, string>;
    scripts: readonly ScriptEntry[];
    concurrency?: number;
    mode: "normal" | "experiment-only" | "evaluate-only";
    experimentIds?: Record<string, string>;
    experimentIdsOutput?: string;
  }): Promise<boolean> => {
    console.log(`\n  Phase: ${phase}`);
    for (const localModel of localModels ?? []) {
      const ready = await ensureLocalModel(localModel);
      if (!ready) return false;
    }

    try {
      await writeBatchConfig({
        env,
        scripts,
        concurrency,
        mode,
        experimentIds,
        experimentIdsOutput,
      });
      return await runBatch();
    } finally {
      await Promise.all(
        (localModels ?? []).map((localModel) => closeTunnelForJob(localModel.jobId)),
      );
    }
  };

  process.on("SIGINT", async () => {
    console.log("\nInterrupted. Cleaning up...");
    await Promise.all([...tunnelsByJob.keys()].map(closeTunnelForJob));
    process.exit(130);
  });

  const results: RunModelResult[] = [];
  const pipelineStart = Date.now();

  for (const [i, run] of runs.entries()) {
    console.log(`\n${"=".repeat(62)}`);
    console.log(`  Run ${i + 1}/${runs.length}: ${run.label}`);
    console.log(`  Scripts: ${run.scripts.length}`);
    console.log(`${"=".repeat(62)}\n`);

    const modelStart = Date.now();
    let ok = false;
    let skipped = false;

    if (!run.judgeModel) {
      ok = await runBatchPhase({
        phase: "normal",
        localModels: localEndpoints(run.evalModel),
        env: evalEnv(run.evalModel),
        scripts: run.scripts,
        concurrency: run.concurrency,
        mode: "normal",
      });
    } else {
      const { normal: normalScripts, staged: stagedScripts } = await splitScriptsByStagedExports(
        run.scripts,
      );
      const experimentIdsOutput = resolve(
        tmpdir(),
        `pipeline-experiment-ids-${process.pid}-${i}.json`,
      );
      const evalScripts = [
        ...normalScripts.map((script) => withScriptMode(script, "normal")),
        ...stagedScripts.map((script) => withScriptMode(script, "experiment-only")),
      ];

      const experimentOk =
        evalScripts.length === 0
          ? true
          : await runBatchPhase({
              phase: stagedScripts.length > 0 ? "eval + experiment-only" : "normal",
              localModels: localEndpoints(run.evalModel),
              env: evalEnv(run.evalModel),
              scripts: evalScripts,
              concurrency: run.concurrency,
              mode: "normal",
              ...(stagedScripts.length > 0 && { experimentIdsOutput }),
            });

      if (!experimentOk) {
        ok = false;
      } else if (stagedScripts.length === 0) {
        ok = true;
      } else {
        const experimentIds = await readExperimentIds(experimentIdsOutput);
        ok = await runBatchPhase({
          phase: "evaluate-only",
          localModels: localEndpoints(run.judgeModel),
          env: judgeEnv(run.judgeModel),
          scripts: stagedScripts.map((script) => withScriptMode(script, "evaluate-only")),
          concurrency: run.concurrency,
          mode: "evaluate-only",
          experimentIds,
        });
      }
    }

    if (!ok) {
      skipped = false;
      console.error(`  Run failed: ${run.label}\n`);
    }

    results.push({
      model: run.label,
      ok,
      skipped,
      durationMs: Date.now() - modelStart,
    });
  }

  await Promise.all([...tunnelsByJob.keys()].map(closeTunnelForJob));
  printSummary(results, Date.now() - pipelineStart);

  const failed = results.filter((r) => !r.ok).length;
  if (failed > 0) process.exit(1);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
