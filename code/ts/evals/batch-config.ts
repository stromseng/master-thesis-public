import { Schema } from "effect";

// ── Model entries — discriminated union on "provider" ────────────────────────

const LiteLLMModelEntry = Schema.Struct({
  provider: Schema.Literal("litellm"),
  model: Schema.String,
});

const VLLMModelEntry = Schema.Struct({
  provider: Schema.Literal("vllm"),
  model: Schema.optional(Schema.String),
  port: Schema.String,
});

const ModelEntry = Schema.Union(LiteLLMModelEntry, VLLMModelEntry);

// ── Script entries — string or object with env ───────────────────────────────

const ScriptWithEnv = Schema.Struct({
  script: Schema.String,
  env: Schema.optionalWith(Schema.Record({ key: Schema.String, value: Schema.String }), {
    default: () => ({}),
  }),
  mode: Schema.optional(Schema.Literal("normal", "experiment-only", "evaluate-only")),
  experimentId: Schema.optional(Schema.String),
});

const ScriptEntry = Schema.Union(Schema.String, ScriptWithEnv);
const BatchMode = Schema.Literal("normal", "experiment-only", "evaluate-only");

// ── Top-level config ─────────────────────────────────────────────────────────

const BatchConfig = Schema.Struct({
  models: Schema.optionalWith(Schema.Array(ModelEntry), { default: () => [] }),
  scripts: Schema.NonEmptyArray(ScriptEntry),
  env: Schema.optionalWith(Schema.Record({ key: Schema.String, value: Schema.String }), {
    default: () => ({}),
  }),
  mode: Schema.optionalWith(BatchMode, { default: () => "normal" as const }),
  experimentIds: Schema.optionalWith(Schema.Record({ key: Schema.String, value: Schema.String }), {
    default: () => ({}),
  }),
  experimentIdsOutput: Schema.optional(Schema.String),
  parallel: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  concurrency: Schema.optional(Schema.Number.pipe(Schema.int(), Schema.positive())),
});

// ── Exports ──────────────────────────────────────────────────────────────────

export type BatchConfig = typeof BatchConfig.Type;
export type ModelEntry = typeof ModelEntry.Type;
export type ScriptEntry = typeof ScriptEntry.Type;
export const decodeBatchConfig = Schema.decodeUnknownSync(BatchConfig);
export { ScriptEntry as ScriptEntrySchema, ScriptWithEnv as ScriptWithEnvSchema };
