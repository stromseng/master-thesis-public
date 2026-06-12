// Usage:
//   bun scripts/phoenix/patch-shititong-datasets.ts
//   bun scripts/phoenix/patch-shititong-datasets.ts --apply
//   bun scripts/phoenix/patch-shititong-datasets.ts --dataset shititong-en-vision --apply
//   bun scripts/phoenix/patch-shititong-datasets.ts --base-url http://127.0.0.1:6006
//
// Dry-run by default. The script compares the canonical local Shititong
// datasets against Phoenix, then:
// - patches changed examples in place via GraphQL
// - deletes stale examples via GraphQL
// - appends newly missing examples via REST
// - creates missing datasets when they do not exist yet
import { appendDatasetExamples, getDatasetExamples } from "@arizeai/phoenix-client/datasets";
import type { Example, ExampleWithId } from "@arizeai/phoenix-client/types/datasets";
import { Effect, Layer, Logger, Schema } from "effect";
import { createOrGetDataset, getDatasetByName } from "../../evals/experiment_setup";
import {
  flattenOpenEndedQuestionGroups,
  loadOpenEndedQuestionGroups,
} from "../../evals/open-ended-dataset";
import {
  flattenQuestionGroups as flattenShititongQuestionGroups,
  loadEnTextGroups,
  loadEnVisionGroups,
  loadZhTextGroups,
  loadZhVisionGroups,
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
  SHITITONG_ZH_VISION_DATASET_NAME,
  type QuestionDatasetMetadata,
} from "../../evals/shititong/shititong";
import { PHOENIX_SKYHIGH_BASE_URL, PhoenixClient } from "../../src/services/PhoenixClient";

type CliArgs = {
  apply: boolean;
  baseUrl: string;
  batchSize: number;
  datasetNames: string[];
  versionDescription: string;
};

type DatasetConfig = {
  dataset: string;
  name: string;
  description: string;
  loadExamples: () => Effect.Effect<LocalDatasetExample[], unknown>;
};

type LocalDatasetExample = Example & {
  metadata: QuestionDatasetMetadata | Record<string, unknown>;
};

type ExamplePatchInput = {
  exampleId: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

type PatchOperation = {
  key: string;
  changedFields: string[];
  patch: ExamplePatchInput;
};

type AppendOperation = {
  key: string;
  example: LocalDatasetExample;
};

type DeleteOperation = {
  key: string;
  exampleId: string;
};

type DatasetDiff = {
  patches: PatchOperation[];
  appends: AppendOperation[];
  deletes: DeleteOperation[];
  unchangedCount: number;
};

type MissingDatasetPlan = {
  kind: "missing";
  config: DatasetConfig;
  localExamples: LocalDatasetExample[];
};

type ExistingDatasetPlan = {
  kind: "existing";
  config: DatasetConfig;
  datasetId: string;
  remoteVersionId: string;
  localExamples: LocalDatasetExample[];
  remoteCount: number;
  diff: DatasetDiff;
};

type DatasetPlan = MissingDatasetPlan | ExistingDatasetPlan;

class CliArgsError extends Schema.TaggedError<CliArgsError>()("CliArgsError", {
  reason: Schema.String,
}) {}

class ExampleMetadataError extends Schema.TaggedError<ExampleMetadataError>()(
  "ExampleMetadataError",
  {
    datasetName: Schema.String,
    source: Schema.Literal("local", "remote"),
    reason: Schema.String,
    exampleId: Schema.optional(Schema.String),
  },
) {}

class DuplicateExampleKeyError extends Schema.TaggedError<DuplicateExampleKeyError>()(
  "DuplicateExampleKeyError",
  {
    datasetName: Schema.String,
    source: Schema.Literal("local", "remote"),
    key: Schema.String,
  },
) {}

class VerificationError extends Schema.TaggedError<VerificationError>()("VerificationError", {
  datasetName: Schema.String,
  reason: Schema.String,
}) {}

const PATCH_DATASET_EXAMPLES_MUTATION = `
  mutation PatchDatasetExamples($input: PatchDatasetExamplesInput!) {
    patchDatasetExamples(input: $input) {
      __typename
    }
  }
`;

const DELETE_DATASET_EXAMPLES_MUTATION = `
  mutation DeleteDatasetExamples($input: DeleteDatasetExamplesInput!) {
    deleteDatasetExamples(input: $input) {
      __typename
    }
  }
`;

const PATCH_DATASET_EXAMPLES_SCHEMA = Schema.Struct({
  patchDatasetExamples: Schema.Struct({
    __typename: Schema.String,
  }),
});

const DELETE_DATASET_EXAMPLES_SCHEMA = Schema.Struct({
  deleteDatasetExamples: Schema.Struct({
    __typename: Schema.String,
  }),
});

const DATASET_CONFIGS: readonly DatasetConfig[] = [
  {
    dataset: "shititong-en-text",
    name: SHITITONG_EN_TEXT_DATASET_NAME,
    description: "Shititong English deduped text-only maritime exam questions",
    loadExamples: () =>
      loadEnTextGroups().pipe(
        Effect.map(
          (groups) =>
            flattenShititongQuestionGroups(groups, "shititong-en-text") as LocalDatasetExample[],
        ),
      ),
  },
  {
    dataset: "shititong-en-vision",
    name: SHITITONG_EN_VISION_DATASET_NAME,
    description: "Shititong English deduped vision maritime exam questions (with images)",
    loadExamples: () =>
      loadEnVisionGroups().pipe(
        Effect.map(
          (groups) =>
            flattenShititongQuestionGroups(groups, "shititong-en-vision") as LocalDatasetExample[],
        ),
      ),
  },
  {
    dataset: "shititong-zh-text",
    name: SHITITONG_ZH_TEXT_DATASET_NAME,
    description: "Shititong Chinese deduped text-only maritime exam questions",
    loadExamples: () =>
      loadZhTextGroups().pipe(
        Effect.map(
          (groups) =>
            flattenShititongQuestionGroups(groups, "shititong-zh-text") as LocalDatasetExample[],
        ),
      ),
  },
  {
    dataset: "shititong-zh-vision",
    name: SHITITONG_ZH_VISION_DATASET_NAME,
    description: "Shititong Chinese deduped vision maritime exam questions (with images)",
    loadExamples: () =>
      loadZhVisionGroups().pipe(
        Effect.map(
          (groups) =>
            flattenShititongQuestionGroups(groups, "shititong-zh-vision") as LocalDatasetExample[],
        ),
      ),
  },
  {
    dataset: "pei2024-zh-open-ended",
    name: "pei2024-zh-open-ended",
    description: "PEI 2024 Chinese theory test rewritten as open-ended evals",
    loadExamples: () =>
      loadOpenEndedQuestionGroups([
        "open_ended",
        "pei2024-zh-2026-04-10T09-12-07Z.open-ended.eval.json",
      ]).pipe(
        Effect.map(
          (groups) =>
            flattenOpenEndedQuestionGroups(
              groups,
              "pei2024-zh-open-ended",
            ) as LocalDatasetExample[],
        ),
      ),
  },
] as const;

const DEFAULT_VERSION_DESCRIPTION = "Fix Shititong image URLs and drop invalid image refs";
const DEFAULT_BATCH_SIZE = 250;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const canonicalizeJson = (value: unknown): unknown => {
  if (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }

  if (typeof value === "bigint") {
    return value.toString();
  }

  if (Array.isArray(value)) {
    return value.map((item) => canonicalizeJson(item));
  }

  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalizeJson(item)]),
    );
  }

  return String(value);
};

const stableStringify = (value: unknown): string => JSON.stringify(canonicalizeJson(value));

const jsonEquals = (left: unknown, right: unknown): boolean =>
  stableStringify(left) === stableStringify(right);

const chunk = <T>(items: readonly T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
};

const formatSample = (items: readonly string[], limit = 5): string =>
  items.slice(0, limit).join(", ");

const parseArgs = Effect.fn("patchShititongDatasets.parseArgs")(function* (
  argv: readonly string[],
) {
  const datasetNames: string[] = [];
  let apply = false;
  let baseUrl = PHOENIX_SKYHIGH_BASE_URL;
  let batchSize = DEFAULT_BATCH_SIZE;
  let versionDescription = DEFAULT_VERSION_DESCRIPTION;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--apply") {
      apply = true;
      continue;
    }
    if (arg === "--dataset") {
      const value = argv[index + 1];
      if (!value) {
        return yield* new CliArgsError({ reason: "Missing value after --dataset" });
      }
      datasetNames.push(value);
      index += 1;
      continue;
    }
    if (arg === "--base-url") {
      const value = argv[index + 1];
      if (!value) {
        return yield* new CliArgsError({ reason: "Missing value after --base-url" });
      }
      baseUrl = value;
      index += 1;
      continue;
    }
    if (arg === "--batch-size") {
      const value = argv[index + 1];
      if (!value) {
        return yield* new CliArgsError({ reason: "Missing value after --batch-size" });
      }
      const parsed = Number.parseInt(value, 10);
      if (!Number.isFinite(parsed) || parsed < 1) {
        return yield* new CliArgsError({
          reason: `Invalid --batch-size '${value}', expected a positive integer`,
        });
      }
      batchSize = parsed;
      index += 1;
      continue;
    }
    if (arg === "--version-description") {
      const value = argv[index + 1];
      if (!value) {
        return yield* new CliArgsError({ reason: "Missing value after --version-description" });
      }
      versionDescription = value;
      index += 1;
      continue;
    }
    return yield* new CliArgsError({ reason: `Unknown argument '${arg}'` });
  }

  return {
    apply,
    baseUrl,
    batchSize,
    datasetNames,
    versionDescription,
  };
});

const selectDatasetConfigs = Effect.fn("patchShititongDatasets.selectDatasetConfigs")(function* (
  datasetNames: readonly string[],
) {
  if (datasetNames.length === 0) {
    return [...DATASET_CONFIGS];
  }

  const configs = datasetNames.map((name) =>
    DATASET_CONFIGS.find((config) => config.name === name),
  );
  const missing = datasetNames.filter((_name, index) => !configs[index]);
  if (missing.length > 0) {
    return yield* new CliArgsError({
      reason: `Unknown dataset name(s): ${missing.join(", ")}`,
    });
  }

  return configs as DatasetConfig[];
});

const toExampleKey = Effect.fn("patchShititongDatasets.toExampleKey")(function* (
  config: DatasetConfig,
  metadata: unknown,
  source: "local" | "remote",
  exampleId?: string,
) {
  if (!isRecord(metadata)) {
    return yield* new ExampleMetadataError({
      datasetName: config.name,
      source,
      reason: "Example metadata is not an object",
      exampleId,
    });
  }

  const dataset = metadata.dataset;
  const groupId = metadata.groupId;
  const questionId = metadata.questionId;

  if (dataset !== config.dataset || typeof groupId !== "string" || typeof questionId !== "string") {
    return yield* new ExampleMetadataError({
      datasetName: config.name,
      source,
      reason: `Example metadata is missing the expected dataset/groupId/questionId fields`,
      exampleId,
    });
  }

  return `${dataset}::${groupId}::${questionId}`;
});

const buildLocalExampleMap = Effect.fn("patchShititongDatasets.buildLocalExampleMap")(function* (
  config: DatasetConfig,
  examples: readonly LocalDatasetExample[],
) {
  const map = new Map<string, LocalDatasetExample>();

  for (const example of examples) {
    const key = yield* toExampleKey(config, example.metadata, "local");
    if (map.has(key)) {
      return yield* new DuplicateExampleKeyError({
        datasetName: config.name,
        source: "local",
        key,
      });
    }
    map.set(key, example);
  }

  return map;
});

const buildRemoteExampleMap = Effect.fn("patchShititongDatasets.buildRemoteExampleMap")(function* (
  config: DatasetConfig,
  examples: readonly ExampleWithId[],
) {
  const map = new Map<string, ExampleWithId>();

  for (const example of examples) {
    const key = yield* toExampleKey(config, example.metadata, "remote", example.id);
    if (map.has(key)) {
      return yield* new DuplicateExampleKeyError({
        datasetName: config.name,
        source: "remote",
        key,
      });
    }
    map.set(key, example);
  }

  return map;
});

const diffDatasetExamples = Effect.fn("patchShititongDatasets.diffDatasetExamples")(function* (
  config: DatasetConfig,
  localExamples: readonly LocalDatasetExample[],
  remoteExamples: readonly ExampleWithId[],
) {
  const localByKey = yield* buildLocalExampleMap(config, localExamples);
  const remoteByKey = yield* buildRemoteExampleMap(config, remoteExamples);

  const patches: PatchOperation[] = [];
  const appends: AppendOperation[] = [];
  const deletes: DeleteOperation[] = [];
  let unchangedCount = 0;

  for (const [key, localExample] of localByKey) {
    const remoteExample = remoteByKey.get(key);
    if (!remoteExample) {
      appends.push({ key, example: localExample });
      continue;
    }

    const patch: ExamplePatchInput = { exampleId: remoteExample.id };
    const changedFields: string[] = [];

    if (!jsonEquals(localExample.input, remoteExample.input)) {
      patch.input = localExample.input;
      changedFields.push("input");
    }

    if (!jsonEquals(localExample.output ?? {}, remoteExample.output ?? {})) {
      patch.output = (localExample.output ?? {}) as Record<string, unknown>;
      changedFields.push("output");
    }

    if (!jsonEquals(localExample.metadata ?? {}, remoteExample.metadata ?? {})) {
      patch.metadata = (localExample.metadata ?? {}) as Record<string, unknown>;
      changedFields.push("metadata");
    }

    if (changedFields.length === 0) {
      unchangedCount += 1;
      continue;
    }

    patches.push({ key, changedFields, patch });
  }

  for (const [key, remoteExample] of remoteByKey) {
    if (!localByKey.has(key)) {
      deletes.push({ key, exampleId: remoteExample.id });
    }
  }

  return { patches, appends, deletes, unchangedCount };
});

const loadLocalExamples = Effect.fn("patchShititongDatasets.loadLocalExamples")(function* (
  config: DatasetConfig,
) {
  return yield* config.loadExamples();
});

const listRemoteExamples = Effect.fn("patchShititongDatasets.listRemoteExamples")(function* (
  datasetId: string,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) => getDatasetExamples({ client, dataset: { datasetId } }));
});

const patchDatasetExamples = Effect.fn("patchShititongDatasets.patchDatasetExamples")(function* (
  patches: readonly ExamplePatchInput[],
  versionDescription: string,
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.graphql({
    operationName: "PatchDatasetExamples",
    query: PATCH_DATASET_EXAMPLES_MUTATION,
    variables: {
      input: {
        patches,
        versionDescription,
      },
    },
    schema: PATCH_DATASET_EXAMPLES_SCHEMA,
  });
});

const deleteDatasetExamples = Effect.fn("patchShititongDatasets.deleteDatasetExamples")(function* (
  exampleIds: readonly string[],
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.graphql({
    operationName: "DeleteDatasetExamples",
    query: DELETE_DATASET_EXAMPLES_MUTATION,
    variables: {
      input: {
        exampleIds,
      },
    },
    schema: DELETE_DATASET_EXAMPLES_SCHEMA,
  });
});

const appendExamples = Effect.fn("patchShititongDatasets.appendExamples")(function* (
  datasetId: string,
  examples: readonly LocalDatasetExample[],
) {
  const phoenix = yield* PhoenixClient;
  return yield* phoenix.use((client) =>
    appendDatasetExamples({
      client,
      dataset: { datasetId },
      examples: [...examples],
    }),
  );
});

const createDataset = Effect.fn("patchShititongDatasets.createDataset")(function* (
  config: DatasetConfig,
  examples: readonly LocalDatasetExample[],
) {
  return yield* createOrGetDataset({
    name: config.name,
    description: config.description,
    examples: [...examples],
  });
});

const buildDatasetPlan = Effect.fn("patchShititongDatasets.buildDatasetPlan")(function* (
  config: DatasetConfig,
) {
  const localExamples = yield* loadLocalExamples(config);
  const existing = yield* getDatasetByName(config.name);

  if (!existing) {
    return {
      kind: "missing" as const,
      config,
      localExamples,
    };
  }

  const remote = yield* listRemoteExamples(existing.id);
  const diff = yield* diffDatasetExamples(config, localExamples, remote.examples);

  return {
    kind: "existing" as const,
    config,
    datasetId: existing.id,
    remoteVersionId: remote.versionId,
    localExamples,
    remoteCount: remote.examples.length,
    diff,
  };
});

const verifyDataset = Effect.fn("patchShititongDatasets.verifyDataset")(function* (
  config: DatasetConfig,
  localExamples: readonly LocalDatasetExample[],
) {
  const existing = yield* getDatasetByName(config.name);
  if (!existing) {
    return yield* new VerificationError({
      datasetName: config.name,
      reason: "Dataset is missing after sync",
    });
  }

  const remote = yield* listRemoteExamples(existing.id);
  const diff = yield* diffDatasetExamples(config, localExamples, remote.examples);
  const driftCount = diff.patches.length + diff.appends.length + diff.deletes.length;

  if (driftCount > 0) {
    return yield* new VerificationError({
      datasetName: config.name,
      reason: `Dataset still differs after sync (${diff.patches.length} patches, ${diff.appends.length} appends, ${diff.deletes.length} deletes remaining)`,
    });
  }

  yield* Effect.logInfo(`Verified dataset ${config.name}`, {
    datasetName: config.name,
    versionId: remote.versionId,
    exampleCount: remote.examples.length,
  });
});

const logDatasetPlan = Effect.fn("patchShititongDatasets.logDatasetPlan")(function* (
  plan: DatasetPlan,
) {
  if (plan.kind === "missing") {
    yield* Effect.logInfo(`Dataset ${plan.config.name} is missing in Phoenix`, {
      datasetName: plan.config.name,
      localCount: plan.localExamples.length,
      action: "create",
    });
    return;
  }

  yield* Effect.logInfo(`Dataset ${plan.config.name} diff`, {
    datasetName: plan.config.name,
    datasetId: plan.datasetId,
    remoteVersionId: plan.remoteVersionId,
    localCount: plan.localExamples.length,
    remoteCount: plan.remoteCount,
    unchangedCount: plan.diff.unchangedCount,
    patchCount: plan.diff.patches.length,
    appendCount: plan.diff.appends.length,
    deleteCount: plan.diff.deletes.length,
  });

  if (plan.diff.patches.length > 0) {
    yield* Effect.logInfo(`Patch sample for ${plan.config.name}`, {
      datasetName: plan.config.name,
      sample: formatSample(
        plan.diff.patches.map((patch) => `${patch.key} [${patch.changedFields.join(",")}]`),
      ),
    });
  }

  if (plan.diff.appends.length > 0) {
    yield* Effect.logInfo(`Append sample for ${plan.config.name}`, {
      datasetName: plan.config.name,
      sample: formatSample(plan.diff.appends.map((append) => append.key)),
    });
  }

  if (plan.diff.deletes.length > 0) {
    yield* Effect.logInfo(`Delete sample for ${plan.config.name}`, {
      datasetName: plan.config.name,
      sample: formatSample(plan.diff.deletes.map((remove) => remove.key)),
    });
  }
});

const applyDatasetPlan = (args: CliArgs) =>
  Effect.fn("patchShititongDatasets.applyDatasetPlan")(function* (plan: DatasetPlan) {
    if (plan.kind === "missing") {
      yield* Effect.logInfo(`Creating dataset ${plan.config.name}`, {
        datasetName: plan.config.name,
        exampleCount: plan.localExamples.length,
      });
      yield* createDataset(plan.config, plan.localExamples);
      yield* verifyDataset(plan.config, plan.localExamples);
      return;
    }

    const deleteBatches = chunk(plan.diff.deletes, args.batchSize);
    for (let index = 0; index < deleteBatches.length; index += 1) {
      const batch = deleteBatches[index]!;
      yield* Effect.logInfo(`Deleting stale examples from ${plan.config.name}`, {
        datasetName: plan.config.name,
        batch: index + 1,
        batches: deleteBatches.length,
        count: batch.length,
      });
      yield* deleteDatasetExamples(batch.map((item) => item.exampleId));
    }

    const patchBatches = chunk(plan.diff.patches, args.batchSize);
    for (let index = 0; index < patchBatches.length; index += 1) {
      const batch = patchBatches[index]!;
      yield* Effect.logInfo(`Patching examples in ${plan.config.name}`, {
        datasetName: plan.config.name,
        batch: index + 1,
        batches: patchBatches.length,
        count: batch.length,
      });
      yield* patchDatasetExamples(
        batch.map((item) => item.patch),
        args.versionDescription,
      );
    }

    const appendBatches = chunk(plan.diff.appends, args.batchSize);
    for (let index = 0; index < appendBatches.length; index += 1) {
      const batch = appendBatches[index]!;
      yield* Effect.logInfo(`Appending examples to ${plan.config.name}`, {
        datasetName: plan.config.name,
        batch: index + 1,
        batches: appendBatches.length,
        count: batch.length,
      });
      yield* appendExamples(
        plan.datasetId,
        batch.map((item) => item.example),
      );
    }

    yield* verifyDataset(plan.config, plan.localExamples);
  });

const logOverallSummary = Effect.fn("patchShititongDatasets.logOverallSummary")(function* (
  plans: readonly DatasetPlan[],
  apply: boolean,
) {
  const summary = plans.reduce(
    (acc, plan) => {
      if (plan.kind === "missing") {
        acc.createCount += 1;
        acc.createExamples += plan.localExamples.length;
        return acc;
      }
      acc.patchCount += plan.diff.patches.length;
      acc.appendCount += plan.diff.appends.length;
      acc.deleteCount += plan.diff.deletes.length;
      acc.unchangedCount += plan.diff.unchangedCount;
      return acc;
    },
    {
      createCount: 0,
      createExamples: 0,
      patchCount: 0,
      appendCount: 0,
      deleteCount: 0,
      unchangedCount: 0,
    },
  );

  yield* Effect.logInfo(`Shititong dataset sync ${apply ? "apply" : "dry-run"} summary`, {
    mode: apply ? "apply" : "dry-run",
    datasets: plans.length,
    createCount: summary.createCount,
    createExamples: summary.createExamples,
    patchCount: summary.patchCount,
    appendCount: summary.appendCount,
    deleteCount: summary.deleteCount,
    unchangedCount: summary.unchangedCount,
  });
});

const syncShititongDatasets = (args: CliArgs) =>
  Effect.gen(function* () {
    const configs = yield* selectDatasetConfigs(args.datasetNames);

    yield* Effect.logInfo("Preparing Shititong Phoenix dataset sync", {
      mode: args.apply ? "apply" : "dry-run",
      baseUrl: args.baseUrl,
      batchSize: args.batchSize,
      versionDescription: args.versionDescription,
      datasets: configs.map((config) => config.name),
    });

    const plans = yield* Effect.forEach(configs, (config) => buildDatasetPlan(config), {
      concurrency: 1,
    });

    yield* Effect.forEach(plans, logDatasetPlan, { discard: true });
    yield* logOverallSummary(plans, args.apply);

    if (!args.apply) {
      return;
    }

    yield* Effect.forEach(plans, applyDatasetPlan(args), {
      concurrency: 1,
      discard: true,
    });
  });

const run = Effect.gen(function* () {
  const args = yield* parseArgs(process.argv.slice(2));
  return yield* syncShititongDatasets(args).pipe(
    Effect.provide(
      Layer.mergeAll(PhoenixClient.layer({ options: { baseUrl: args.baseUrl } }), Logger.pretty),
    ),
  );
});

if (import.meta.main) {
  Effect.runPromise(run);
}
