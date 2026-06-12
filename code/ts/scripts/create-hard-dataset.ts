// Usage:
//   bun scripts/create-hard-dataset.ts
//   bun scripts/create-hard-dataset.ts --force   # delete and recreate existing hard datasets
//   bun scripts/create-hard-dataset.ts --dry-run # collect and report only, never mutate datasets
//
// Queries all Phoenix evaluation datasets, fetches experiment results, and
// creates two combined hard datasets:
//   - "hard-text"       — text-only questions where a considered model scored F1 == 0
//   - "hard-multimodal" — multimodal questions where a considered model scored F1 == 0
import { Effect, Logger } from "effect";
import { CREWCN_DATASET_NAME } from "../evals/crew/crew";
import { navreasDatasets, toDatasetName } from "../evals/navreas/navreas";
import { PEI2024_UK_DATASET_NAME, PEI2024_ZH_DATASET_NAME } from "../evals/pei2024/pei2024";
import { RAYNOR_DATASET_NAME_V2, RAYNOR_DATASET_NAME_MULTIMODAL_V2 } from "../evals/raynor/raynor";
import {
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
} from "../evals/shititong/shititong";
import {
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
  US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
} from "../evals/us_coast_guard/us_coast_guard";
import { PhoenixClient, type PhoenixClientImpl } from "../src/services/PhoenixClient";

// Text-only datasets
const TEXT_DATASETS = [
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
  SHITITONG_EN_TEXT_DATASET_NAME,
  RAYNOR_DATASET_NAME_V2,
  CREWCN_DATASET_NAME,
  PEI2024_UK_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
  PEI2024_ZH_DATASET_NAME,
];

// Multimodal datasets
const MULTIMODAL_DATASETS = [
  US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
  RAYNOR_DATASET_NAME_MULTIMODAL_V2,
  SHITITONG_EN_VISION_DATASET_NAME,
  ...navreasDatasets.map(toDatasetName),
];

const HARD_TEXT_NAME = "hard-text";
const HARD_MULTIMODAL_NAME = "hard-multimodal";
const SCORE_ANNOTATION_NAME = "question-f1";

const FORCE = process.argv.includes("--force");
const DRY_RUN = process.argv.includes("--dry-run");

// Optional model filter:
// Leave empty to consider all models. Add model IDs to only include those models
// when deciding whether a question is hard (F1 == 0).
const INCLUDED_MODEL_IDS: readonly string[] = ["moonshotai/Kimi-K2.5"];
const INCLUDED_MODEL_ID_SET = new Set(INCLUDED_MODEL_IDS);

type DatasetInfo = { id: string; name: string; example_count: number };
type ExperimentInfo = { id: string; dataset_id: string; metadata: Record<string, unknown> };

type HardExample = {
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metadata: Record<string, unknown>;
};

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/** List all datasets, paginating through all results */
const listAllDatasets = Effect.fn("listAllDatasets")(function* (phoenix: PhoenixClientImpl) {
  const datasets: DatasetInfo[] = [];
  let cursor: string | undefined;

  while (true) {
    const response = yield* phoenix.use((client) =>
      client.GET("/v1/datasets", {
        params: { query: { limit: 100, ...(cursor ? { cursor } : {}) } },
      }),
    );

    const data = response.data?.data ?? [];
    for (const d of data) {
      datasets.push({ id: d.id, name: d.name, example_count: d.example_count });
    }

    const nextCursor = response.data?.next_cursor;
    if (!nextCursor || data.length === 0) break;
    cursor = nextCursor;
  }

  return datasets;
});

/** List all experiments for a dataset, paginating */
const listExperiments = Effect.fn("listExperiments")(function* (
  phoenix: PhoenixClientImpl,
  datasetId: string,
) {
  const experiments: ExperimentInfo[] = [];
  let cursor: string | undefined;

  while (true) {
    const response = yield* phoenix.use((client) =>
      client.GET("/v1/datasets/{dataset_id}/experiments", {
        params: {
          path: { dataset_id: datasetId },
          query: { limit: 50, ...(cursor ? { cursor } : {}) },
        },
      }),
    );

    const data = response.data?.data ?? [];
    for (const e of data) {
      experiments.push({
        id: e.id,
        dataset_id: e.dataset_id,
        metadata: (e.metadata ?? {}) as Record<string, unknown>,
      });
    }

    const nextCursor = response.data?.next_cursor;
    if (!nextCursor || data.length === 0) break;
    cursor = nextCursor;
  }

  return experiments;
});

// Shape from the /v1/experiments/{id}/json export
type ExperimentAnnotation = {
  name?: string;
  score?: number | null;
  error?: string | null;
};

type ExperimentJsonRun = {
  example_id: string;
  error: string | null;
  annotations?: ExperimentAnnotation[];
};

/** Read an evaluator score from run annotations */
const getRunAnnotationScore = (run: ExperimentJsonRun, annotationName: string): number | null => {
  if (run.error) return null;

  const annotations = Array.isArray(run.annotations) ? run.annotations : [];
  const annotation = annotations.find((a) => a.name === annotationName);
  if (!annotation || annotation.error) return null;
  if (typeof annotation.score !== "number") return null;

  return annotation.score;
};

const getErrorMessage = (error: { cause: unknown }): string =>
  error.cause instanceof Error ? error.cause.message : String(error.cause);

const isMissingExperimentExport = (error: { cause: unknown }): boolean => {
  const message = getErrorMessage(error);
  return (
    message.includes("/v1/experiments/") &&
    message.includes("/json") &&
    (message.includes("404") || message.includes("Not Found"))
  );
};

/** Download the full experiment JSON export */
const downloadExperimentJson = Effect.fn("downloadExperimentJson")(function* (
  phoenix: PhoenixClientImpl,
  experimentId: string,
) {
  const response = yield* phoenix.use((client) =>
    client.GET("/v1/experiments/{experiment_id}/json", {
      params: { path: { experiment_id: experimentId } },
    }),
  );

  // The client may return a parsed object or a raw string
  const raw = response.data as unknown;
  const data = typeof raw === "string" ? JSON.parse(raw) : raw;

  if (!Array.isArray(data)) {
    yield* Effect.logWarning(`Experiment ${experimentId}: unexpected format, skipping`);
    return [] as ExperimentJsonRun[];
  }

  return data as ExperimentJsonRun[];
});

type DatasetExample = {
  id: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metadata: Record<string, unknown>;
};

/** Fetch all examples for a dataset */
const getDatasetExamples = Effect.fn("getDatasetExamples")(function* (
  phoenix: PhoenixClientImpl,
  datasetId: string,
) {
  const response = yield* phoenix.use((client) =>
    client.GET("/v1/datasets/{id}/examples", {
      params: { path: { id: datasetId } },
    }),
  );

  return (response.data?.data?.examples ?? []) as DatasetExample[];
});

/** Upload a new dataset */
const uploadDataset = Effect.fn("uploadDataset")(function* (
  phoenix: PhoenixClientImpl,
  params: {
    name: string;
    description: string;
    examples: HardExample[];
  },
) {
  const response = yield* phoenix.use((client) =>
    client.POST("/v1/datasets/upload", {
      params: { query: { sync: true } },
      body: {
        action: "create",
        name: params.name,
        description: params.description,
        inputs: params.examples.map((e) => e.input),
        outputs: params.examples.map((e) => e.output),
        metadata: params.examples.map((e) => e.metadata),
      },
    }),
  );

  return response.data?.data;
});

/** Delete a dataset by ID */
const deleteDataset = Effect.fn("deleteDataset")(function* (
  phoenix: PhoenixClientImpl,
  datasetId: string,
) {
  yield* phoenix.use((client) =>
    client.DELETE("/v1/datasets/{id}", {
      params: { path: { id: datasetId } },
    }),
  );
});

/**
 * Collect hard examples from a single dataset: returns examples where any
 * experiment had F1 == 0.
 */
const collectHardExamples = Effect.fn("collectHardExamples")(function* (
  phoenix: PhoenixClientImpl,
  dataset: DatasetInfo,
) {
  const experiments = yield* listExperiments(phoenix, dataset.id);
  yield* Effect.log(`[${dataset.name}] ${experiments.length} experiments`);

  if (experiments.length === 0) return [] as HardExample[];

  // Collect example IDs that any model got completely wrong
  const failedExampleIds = new Set<string>();
  let skippedByModelFilter = 0;
  let skippedMissingExperimentExports = 0;
  let consideredExperimentCount = 0;

  for (const experiment of experiments) {
    const modelId = (experiment.metadata as { model?: { id?: string } })?.model?.id ?? "unknown";

    if (INCLUDED_MODEL_ID_SET.size > 0 && !INCLUDED_MODEL_ID_SET.has(modelId)) {
      skippedByModelFilter++;
      continue;
    }

    consideredExperimentCount++;
    const runs = yield* downloadExperimentJson(phoenix, experiment.id).pipe(
      Effect.catchTags({
        PhoenixAsyncError: (error) => {
          if (!isMissingExperimentExport(error)) return Effect.fail(error);
          skippedMissingExperimentExports++;
          return Effect.logWarning(
            `[${dataset.name}]   "${modelId}": experiment ${experiment.id} export missing (404), skipping`,
          ).pipe(Effect.as([] as ExperimentJsonRun[]));
        },
        PhoenixSyncError: (error) => {
          if (!isMissingExperimentExport(error)) return Effect.fail(error);
          skippedMissingExperimentExports++;
          return Effect.logWarning(
            `[${dataset.name}]   "${modelId}": experiment ${experiment.id} export missing (404), skipping`,
          ).pipe(Effect.as([] as ExperimentJsonRun[]));
        },
      }),
    );

    let zeroScoreCount = 0;
    let missingScoreCount = 0;
    for (const run of runs) {
      const score = getRunAnnotationScore(run, SCORE_ANNOTATION_NAME);
      if (score === null) {
        missingScoreCount++;
        continue;
      }
      if (score === 0) {
        failedExampleIds.add(run.example_id);
        zeroScoreCount++;
      }
    }

    yield* Effect.log(
      `[${dataset.name}]   "${modelId}": ${runs.length} runs, ${zeroScoreCount} ${SCORE_ANNOTATION_NAME}=0, ${missingScoreCount} missing/null score`,
    );
  }

  if (skippedByModelFilter > 0) {
    yield* Effect.log(
      `[${dataset.name}] Skipped ${skippedByModelFilter} experiments not in included model IDs`,
    );
  }
  if (skippedMissingExperimentExports > 0) {
    yield* Effect.logWarning(
      `[${dataset.name}] Skipped ${skippedMissingExperimentExports} experiments with missing exports`,
    );
  }

  if (INCLUDED_MODEL_ID_SET.size > 0 && consideredExperimentCount === 0) {
    yield* Effect.logWarning(
      `[${dataset.name}] No experiments found for included model IDs: ${INCLUDED_MODEL_IDS.join(", ")}`,
    );
    return [] as HardExample[];
  }

  if (failedExampleIds.size === 0) {
    yield* Effect.log(`[${dataset.name}] No hard questions found`);
    return [] as HardExample[];
  }

  // Fetch original examples and filter
  const examples = yield* getDatasetExamples(phoenix, dataset.id);
  const hard = examples
    .filter((e) => failedExampleIds.has(e.id))
    .map((e) => ({
      input: e.input,
      output: e.output,
      metadata: {
        ...e.metadata,
        source_dataset: dataset.name,
        source_example_id: e.id,
      },
    }));

  yield* Effect.log(`[${dataset.name}] ${hard.length} hard examples (of ${examples.length})`);
  return hard;
});

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

const program = Effect.gen(function* () {
  const phoenix = yield* PhoenixClient;

  if (FORCE) yield* Effect.log("--force flag set: will delete and recreate existing hard datasets");
  if (DRY_RUN) {
    yield* Effect.log(
      "--dry-run flag set: collecting data and printing summary only (no upload/delete)",
    );
  }
  if (FORCE && DRY_RUN) {
    yield* Effect.logWarning(
      "--force is ignored in --dry-run mode (existing datasets will not be deleted)",
    );
  }
  if (INCLUDED_MODEL_ID_SET.size > 0) {
    yield* Effect.log(`Included model ID filter enabled (${INCLUDED_MODEL_ID_SET.size} IDs):`);
    for (const modelId of INCLUDED_MODEL_IDS) {
      yield* Effect.log(`  - ${modelId}`);
    }
  }

  yield* Effect.log("Fetching all datasets from Phoenix...");
  const allDatasets = yield* listAllDatasets(phoenix);
  yield* Effect.log(`Found ${allDatasets.length} datasets total`);

  // Check if hard datasets already exist
  const existingText = allDatasets.find((d) => d.name === HARD_TEXT_NAME);
  const existingMultimodal = allDatasets.find((d) => d.name === HARD_MULTIMODAL_NAME);

  if (FORCE && !DRY_RUN) {
    if (existingText) {
      yield* Effect.log(
        `Deleting existing "${HARD_TEXT_NAME}" (${existingText.example_count} examples)...`,
      );
      yield* deleteDataset(phoenix, existingText.id);
    }
    if (existingMultimodal) {
      yield* Effect.log(
        `Deleting existing "${HARD_MULTIMODAL_NAME}" (${existingMultimodal.example_count} examples)...`,
      );
      yield* deleteDataset(phoenix, existingMultimodal.id);
    }
  } else if (!DRY_RUN) {
    if (existingText) {
      yield* Effect.log(
        `"${HARD_TEXT_NAME}" already exists (${existingText.example_count} examples). Use --force to recreate.`,
      );
    }
    if (existingMultimodal) {
      yield* Effect.log(
        `"${HARD_MULTIMODAL_NAME}" already exists (${existingMultimodal.example_count} examples). Use --force to recreate.`,
      );
    }
    if (existingText && existingMultimodal) return;
  }

  const shouldProcessText = DRY_RUN || FORCE || !existingText;
  const shouldProcessMultimodal = DRY_RUN || FORCE || !existingMultimodal;

  // Log datasets that exist on Phoenix but aren't used as sources
  const allSourceNames = new Set([...TEXT_DATASETS, ...MULTIMODAL_DATASETS]);
  const ignored = allDatasets.filter(
    (d) =>
      !allSourceNames.has(d.name) && d.name !== HARD_TEXT_NAME && d.name !== HARD_MULTIMODAL_NAME,
  );
  if (ignored.length > 0) {
    yield* Effect.log(`\nIgnoring ${ignored.length} datasets not in source lists:`);
    for (const d of ignored) {
      yield* Effect.log(`  - ${d.name} (${d.example_count} examples)`);
    }
  }

  // Collect hard examples across all source datasets
  const collectForGroup = Effect.fn("collectForGroup")(function* (
    sourceNames: string[],
    label: string,
  ) {
    const sources = allDatasets.filter((d) => sourceNames.includes(d.name));
    const missing = sourceNames.filter((n) => !sources.some((d) => d.name === n));
    yield* Effect.log(
      `\n=== ${label} === (${sources.length}/${sourceNames.length} source datasets found)`,
    );
    if (missing.length > 0) {
      yield* Effect.logWarning(`Missing datasets: ${missing.join(", ")}`);
    }

    const allHard: HardExample[] = [];
    const perDatasetCounts: { name: string; hard: number; total: number }[] = [];
    for (const ds of sources) {
      const hard = yield* collectHardExamples(phoenix, ds);
      allHard.push(...hard);
      perDatasetCounts.push({ name: ds.name, hard: hard.length, total: ds.example_count });
    }
    return { examples: allHard, perDatasetCounts };
  });

  // Text
  if (shouldProcessText) {
    const { examples: textHard, perDatasetCounts: textCounts } = yield* collectForGroup(
      TEXT_DATASETS,
      "TEXT",
    );
    yield* Effect.log(`\nTotal text hard examples: ${textHard.length}`);

    if (textHard.length > 0 && !DRY_RUN) {
      const result = yield* uploadDataset(phoenix, {
        name: HARD_TEXT_NAME,
        description:
          "Combined hard text-only questions across all evaluation datasets — questions where at least one model scored F1=0",
        examples: textHard,
      });
      yield* Effect.log(
        `Created "${HARD_TEXT_NAME}" with ${textHard.length} examples (dataset_id: ${result?.dataset_id})`,
      );
    }
    if (textHard.length > 0 && DRY_RUN) {
      yield* Effect.log(
        `[dry-run] Would upload "${HARD_TEXT_NAME}" with ${textHard.length} examples`,
      );
    }

    yield* Effect.log("\n--- Text Dataset Breakdown ---");
    const totalTextExamples = textCounts.reduce((sum, c) => sum + c.total, 0);
    for (const c of textCounts) {
      const pct = c.total > 0 ? ((c.hard / c.total) * 100).toFixed(1) : "0.0";
      yield* Effect.log(`  ${c.name}: ${c.hard}/${c.total} hard (${pct}%)`);
    }
    yield* Effect.log(
      `  TOTAL: ${textHard.length}/${totalTextExamples} hard (${totalTextExamples > 0 ? ((textHard.length / totalTextExamples) * 100).toFixed(1) : "0.0"}%)`,
    );
  }

  // Multimodal
  if (shouldProcessMultimodal) {
    const { examples: mmHard, perDatasetCounts: mmCounts } = yield* collectForGroup(
      MULTIMODAL_DATASETS,
      "MULTIMODAL",
    );
    yield* Effect.log(`\nTotal multimodal hard examples: ${mmHard.length}`);

    if (mmHard.length > 0 && !DRY_RUN) {
      const result = yield* uploadDataset(phoenix, {
        name: HARD_MULTIMODAL_NAME,
        description:
          "Combined hard multimodal questions across all evaluation datasets — questions where at least one model scored F1=0",
        examples: mmHard,
      });
      yield* Effect.log(
        `Created "${HARD_MULTIMODAL_NAME}" with ${mmHard.length} examples (dataset_id: ${result?.dataset_id})`,
      );
    }
    if (mmHard.length > 0 && DRY_RUN) {
      yield* Effect.log(
        `[dry-run] Would upload "${HARD_MULTIMODAL_NAME}" with ${mmHard.length} examples`,
      );
    }

    yield* Effect.log("\n--- Multimodal Dataset Breakdown ---");
    const totalMmExamples = mmCounts.reduce((sum, c) => sum + c.total, 0);
    for (const c of mmCounts) {
      const pct = c.total > 0 ? ((c.hard / c.total) * 100).toFixed(1) : "0.0";
      yield* Effect.log(`  ${c.name}: ${c.hard}/${c.total} hard (${pct}%)`);
    }
    yield* Effect.log(
      `  TOTAL: ${mmHard.length}/${totalMmExamples} hard (${totalMmExamples > 0 ? ((mmHard.length / totalMmExamples) * 100).toFixed(1) : "0.0"}%)`,
    );
  }

  yield* Effect.log("\nDone!");
});

const layer = PhoenixClient.skyhigh;

Effect.runPromise(program.pipe(Effect.provide(layer), Effect.provide(Logger.pretty))).catch(
  console.error,
);
