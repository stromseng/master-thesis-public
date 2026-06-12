#!/usr/bin/env bun
/**
 * Analyze throughput of every model on the IDUN LiteLLM provider.
 *
 * Phase 1: Streams a prompt to each model concurrently — measures TTFT, tokens/sec, total latency.
 * Phase 2: Fires concurrent burst requests per model to measure requests/min capacity.
 *
 * Usage:
 *   bun code/ts/scripts/analyze-litellm-throughput.ts
 *
 * Options (env vars):
 *   LITE_LLM_BASE_URL      — override base URL (default: https://llm.hpc.ntnu.no/v1)
 *   THROUGHPUT_PROMPT       — custom prompt (default: built-in)
 *   THROUGHPUT_MAX_TOKENS   — max output tokens per model (default: 256)
 *   BURST_DURATION_SEC      — how long to run burst test per model (default: 30)
 *   BURST_CONCURRENCY       — max concurrent requests during burst (default: 10)
 *   BURST_MAX_TOKENS        — max tokens per burst request (default: 32)
 *   SKIP_BURST              — set to "1" to skip the burst/RPM phase
 */

import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import * as ai from "ai";
import { Config, Effect, Logger } from "effect";
import { encode } from "gpt-tokenizer";

// ── Config ────────────────────────────────────────────────────────────

const AppConfig = Effect.all({
  baseUrl: Config.string("LITE_LLM_BASE_URL").pipe(
    Config.orElse(() => Config.succeed("https://llm.hpc.ntnu.no/v1")),
  ),
  apiKey: Config.string("LITE_LLM_API_KEY"),
  prompt: Config.string("THROUGHPUT_PROMPT").pipe(
    Config.orElse(() =>
      Config.succeed(
        "Explain the concept of deadweight tonnage in maritime shipping. Be detailed and thorough.",
      ),
    ),
  ),
  maxTokens: Config.number("THROUGHPUT_MAX_TOKENS").pipe(Config.orElse(() => Config.succeed(256))),
  burstDurationSec: Config.number("BURST_DURATION_SEC").pipe(
    Config.orElse(() => Config.succeed(30)),
  ),
  burstConcurrency: Config.number("BURST_CONCURRENCY").pipe(
    Config.orElse(() => Config.succeed(10)),
  ),
  burstMaxTokens: Config.number("BURST_MAX_TOKENS").pipe(Config.orElse(() => Config.succeed(32))),
  skipBurst: Config.string("SKIP_BURST").pipe(
    Config.orElse(() => Config.succeed("0")),
    Config.map((s) => s === "1"),
  ),
});

// ── Types ─────────────────────────────────────────────────────────────

interface ModelEntry {
  id: string;
  object: string;
  owned_by?: string;
}

interface ThroughputResult {
  model: string;
  ttftMs: number;
  totalMs: number;
  outputTokens: number;
  tokensPerSec: number;
  error?: string;
}

interface BurstResult {
  model: string;
  durationSec: number;
  completed: number;
  failed: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  requestsPerMin: number;
  inputTokensPerMin: number;
  outputTokensPerMin: number;
  avgLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  error?: string;
}

// ── Helpers ───────────────────────────────────────────────────────────

const percentile = (sorted: number[], p: number) => {
  if (sorted.length === 0) return 0;
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)]!;
};

const OUTPUT_DIR = "data/etc";

// ── Fetch models ──────────────────────────────────────────────────────

const fetchModels = (baseUrl: string, apiKey: string) =>
  Effect.tryPromise({
    try: async () => {
      const res = await fetch(`${baseUrl}/models`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = (await res.json()) as { data: ModelEntry[] };
      return json.data;
    },
    catch: (e) => new Error(`Failed to fetch models: ${e}`),
  });

// ── Phase 1: single-request streaming throughput ──────────────────────

const measureModel = (
  provider: ReturnType<typeof createOpenAICompatible>,
  modelId: string,
  prompt: string,
  maxOutputTokens: number,
): Effect.Effect<ThroughputResult> =>
  Effect.tryPromise({
    try: async () => {
      const model = provider.languageModel(modelId);
      const t0 = performance.now();
      let ttft = 0;
      let firstToken = true;

      const result = ai.streamText({ model, prompt, maxOutputTokens, maxRetries: 0 });

      let fullText = "";
      for await (const chunk of result.textStream) {
        if (firstToken && chunk.length > 0) {
          ttft = performance.now() - t0;
          firstToken = false;
        }
        fullText += chunk;
      }

      const totalMs = performance.now() - t0;
      const outputTokens = encode(fullText).length;
      const generationMs = totalMs - ttft;
      const tokensPerSec = generationMs > 0 ? (outputTokens / generationMs) * 1000 : 0;

      return {
        model: modelId,
        ttftMs: ttft,
        totalMs,
        outputTokens,
        tokensPerSec,
      } satisfies ThroughputResult;
    },
    catch: (err) => err,
  }).pipe(
    Effect.tap((r) =>
      Effect.log(
        `${r.model} — TTFT: ${r.ttftMs.toFixed(0)}ms | Tok/s: ${r.tokensPerSec.toFixed(1)} | Tokens: ${r.outputTokens} | Total: ${(r.totalMs / 1000).toFixed(1)}s`,
      ),
    ),
    Effect.catchAll((err) => {
      const message = err instanceof Error ? err.message.slice(0, 200) : String(err);
      return Effect.log(`${modelId} — ERROR: ${message}`).pipe(
        Effect.map(
          () =>
            ({
              model: modelId,
              ttftMs: 0,
              totalMs: 0,
              outputTokens: 0,
              tokensPerSec: 0,
              error: message,
            }) satisfies ThroughputResult,
        ),
      );
    }),
  );

// ── Phase 2: burst / RPM ─────────────────────────────────────────────

const BURST_PROMPTS = [
  "What is a bulkhead on a ship?",
  "Define the term 'draft' in maritime context.",
  "What is the purpose of a bilge pump?",
  "Explain port and starboard.",
  "What does SOLAS stand for?",
  "Describe the function of a rudder.",
  "What is a Plimsoll line?",
  "Define 'knot' as a unit of speed.",
  "What is the bridge of a ship?",
  "Explain maritime right of way rules.",
];

const measureBurst = (
  provider: ReturnType<typeof createOpenAICompatible>,
  modelId: string,
  durationSec: number,
  concurrency: number,
  maxOutputTokens: number,
): Effect.Effect<BurstResult> =>
  Effect.tryPromise({
    try: async () => {
      const model = provider.languageModel(modelId);
      const deadline = performance.now() + durationSec * 1000;

      let completed = 0;
      let failed = 0;
      let totalInputTokens = 0;
      let totalOutputTokens = 0;
      const latencies: number[] = [];

      const worker = async () => {
        let i = 0;
        while (performance.now() < deadline) {
          const prompt = BURST_PROMPTS[i % BURST_PROMPTS.length]!;
          const t0 = performance.now();
          try {
            const res = await ai.generateText({ model, prompt, maxOutputTokens, maxRetries: 0 });
            latencies.push(performance.now() - t0);
            totalInputTokens += res.usage?.inputTokens ?? encode(prompt).length;
            totalOutputTokens += res.usage?.outputTokens ?? encode(res.text).length;
            completed++;
          } catch {
            failed++;
          }
          i++;
        }
      };

      const t0 = performance.now();
      await Promise.all(Array.from({ length: concurrency }, () => worker()));
      const actualDurationSec = (performance.now() - t0) / 1000;
      const minuteFactor = 60 / actualDurationSec;

      latencies.sort((a, b) => a - b);

      return {
        model: modelId,
        durationSec: actualDurationSec,
        completed,
        failed,
        totalInputTokens,
        totalOutputTokens,
        requestsPerMin: completed * minuteFactor,
        inputTokensPerMin: totalInputTokens * minuteFactor,
        outputTokensPerMin: totalOutputTokens * minuteFactor,
        avgLatencyMs:
          latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0,
        p50LatencyMs: percentile(latencies, 50),
        p95LatencyMs: percentile(latencies, 95),
      } satisfies BurstResult;
    },
    catch: (e) => e,
  }).pipe(
    Effect.tap((r) =>
      Effect.log(
        `${r.model} — ${r.completed} reqs in ${r.durationSec.toFixed(0)}s → ` +
          `${r.requestsPerMin.toFixed(0)} RPM | ` +
          `${r.outputTokensPerMin.toFixed(0)} out.tok/min | ` +
          `avg ${r.avgLatencyMs.toFixed(0)}ms | p95 ${r.p95LatencyMs.toFixed(0)}ms` +
          (r.failed > 0 ? ` | ${r.failed} errors` : ""),
      ),
    ),
    Effect.catchAll((err) => {
      const message = err instanceof Error ? err.message.slice(0, 200) : String(err);
      return Effect.log(`${modelId} — BURST ERROR: ${message}`).pipe(
        Effect.map(
          () =>
            ({
              model: modelId,
              durationSec: 0,
              completed: 0,
              failed: 0,
              totalInputTokens: 0,
              totalOutputTokens: 0,
              requestsPerMin: 0,
              inputTokensPerMin: 0,
              outputTokensPerMin: 0,
              avgLatencyMs: 0,
              p50LatencyMs: 0,
              p95LatencyMs: 0,
              error: message,
            }) satisfies BurstResult,
        ),
      );
    }),
  );

// ── Printing ──────────────────────────────────────────────────────────

const printPhase1Summary = (results: ThroughputResult[]) =>
  Effect.sync(() => {
    const successful = results
      .filter((r) => !r.error)
      .sort((a, b) => b.tokensPerSec - a.tokensPerSec);
    const failed = results.filter((r) => r.error);

    console.log("\n" + "=".repeat(110));
    console.log("PHASE 1: SINGLE-REQUEST THROUGHPUT (sorted by tok/s)");
    console.log("=".repeat(110));

    if (successful.length > 0) {
      console.log(
        [
          "Model".padEnd(55),
          "TTFT(ms)".padStart(10),
          "Tok/s".padStart(8),
          "Tokens".padStart(8),
          "Total(s)".padStart(10),
        ].join(" | "),
      );
      console.log("-".repeat(110));
      for (const r of successful) {
        console.log(
          [
            r.model.padEnd(55),
            r.ttftMs.toFixed(0).padStart(10),
            r.tokensPerSec.toFixed(1).padStart(8),
            String(r.outputTokens).padStart(8),
            (r.totalMs / 1000).toFixed(1).padStart(10),
          ].join(" | "),
        );
      }
    }

    if (failed.length > 0) {
      console.log(`\nFailed models (${failed.length}):`);
      for (const r of failed) console.log(`  - ${r.model}: ${r.error}`);
    }

    return { successful, failed };
  });

const printPhase2Summary = (results: BurstResult[]) =>
  Effect.sync(() => {
    const sorted = results
      .filter((r) => !r.error && r.completed > 0)
      .sort((a, b) => b.requestsPerMin - a.requestsPerMin);
    const burstFailed = results.filter((r) => r.error);

    console.log("\n" + "=".repeat(120));
    console.log("PHASE 2: BURST / RPM SUMMARY (sorted by RPM)");
    console.log("=".repeat(120));

    if (sorted.length > 0) {
      console.log(
        [
          "Model".padEnd(50),
          "RPM".padStart(7),
          "Out.tok/min".padStart(14),
          "Avg(ms)".padStart(9),
          "P50(ms)".padStart(9),
          "P95(ms)".padStart(9),
          "OK/Fail".padStart(9),
        ].join(" | "),
      );
      console.log("-".repeat(120));
      for (const r of sorted) {
        console.log(
          [
            r.model.padEnd(50),
            r.requestsPerMin.toFixed(0).padStart(7),
            r.outputTokensPerMin.toFixed(0).padStart(14),
            r.avgLatencyMs.toFixed(0).padStart(9),
            r.p50LatencyMs.toFixed(0).padStart(9),
            r.p95LatencyMs.toFixed(0).padStart(9),
            `${r.completed}/${r.failed}`.padStart(9),
          ].join(" | "),
        );
      }
    }

    if (burstFailed.length > 0) {
      console.log(`\nFailed burst tests (${burstFailed.length}):`);
      for (const r of burstFailed) console.log(`  - ${r.model}: ${r.error}`);
    }

    console.log("\n" + "=".repeat(120));
  });

// ── Main program ──────────────────────────────────────────────────────

const program = Effect.gen(function* () {
  const cfg = yield* AppConfig;

  const litellm = createOpenAICompatible({
    name: "litellm",
    apiKey: cfg.apiKey,
    baseURL: cfg.baseUrl,
  });

  // Fetch models
  yield* Effect.log(`Fetching models from ${cfg.baseUrl}/models ...`);
  const allModels = yield* fetchModels(cfg.baseUrl, cfg.apiKey);

  const chatModels = allModels.filter(
    (m) => !m.id.toLowerCase().includes("embedding") && !m.id.toLowerCase().includes("embed"),
  );

  yield* Effect.log(
    `Found ${allModels.length} models total, ${chatModels.length} chat/generation models`,
  );
  for (const m of chatModels) yield* Effect.log(`  - ${m.id}`);

  // Phase 1: run all models concurrently
  yield* Effect.log(
    `\nPhase 1: streaming throughput (${cfg.prompt.length} char prompt, max ${cfg.maxTokens} output tokens) — all models concurrently`,
  );

  const phase1Results = yield* Effect.all(
    chatModels.map((m) => measureModel(litellm, m.id, cfg.prompt, cfg.maxTokens)),
    { concurrency: "unbounded" },
  );

  const { successful } = yield* printPhase1Summary(phase1Results);

  // Phase 2: burst RPM (all models concurrently)
  let phase2Results: BurstResult[] = [];
  if (cfg.skipBurst || successful.length === 0) {
    if (cfg.skipBurst) yield* Effect.log("Skipping burst/RPM phase (SKIP_BURST=1)");
  } else {
    yield* Effect.log(
      `\nPhase 2: burst test (${cfg.burstConcurrency} concurrent workers, ${cfg.burstDurationSec}s per model, ${cfg.burstMaxTokens} max tokens)`,
    );

    phase2Results = yield* Effect.all(
      successful.map((r) =>
        measureBurst(
          litellm,
          r.model,
          cfg.burstDurationSec,
          cfg.burstConcurrency,
          cfg.burstMaxTokens,
        ),
      ),
      { concurrency: "unbounded" },
    );

    yield* printPhase2Summary(phase2Results);
  }

  // ── Save results to file ────────────────────────────────────────────
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = `${OUTPUT_DIR}/litellm-throughput-${timestamp}.json`;
  const output = {
    timestamp: new Date().toISOString(),
    config: {
      baseUrl: cfg.baseUrl,
      promptLength: cfg.prompt.length,
      maxTokens: cfg.maxTokens,
      burstDurationSec: cfg.burstDurationSec,
      burstConcurrency: cfg.burstConcurrency,
      burstMaxTokens: cfg.burstMaxTokens,
    },
    phase1: phase1Results,
    phase2: phase2Results,
  };
  yield* Effect.tryPromise({
    try: () => Bun.write(outPath, JSON.stringify(output, null, 2)),
    catch: (e) => new Error(`Failed to write results: ${e}`),
  });
  yield* Effect.log(`Results saved to ${outPath}`);
});

// ── Run ───────────────────────────────────────────────────────────────

await Effect.runPromise(program.pipe(Effect.provide(Logger.pretty)));
