// Usage:
//   bun scripts/analysis/fetch-multimodal-experiment-scores.ts
//   bun scripts/analysis/fetch-multimodal-experiment-scores.ts --output multimodal-latest.json
//   bun scripts/analysis/fetch-multimodal-experiment-scores.ts --concurrency 12
//   bun scripts/analysis/fetch-multimodal-experiment-scores.ts --base-url http://127.0.0.1:6006
//
// Fetches the most recent Phoenix experiment scores for every model across all
// multimodal datasets, using the shared aggregate-experiment-scores.ts fetcher.
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { navreasDatasets, toDatasetName } from "../../evals/navreas/navreas";
import { RAYNOR_DATASET_NAME_MULTIMODAL_V2 } from "../../evals/raynor/raynor";
import {
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_VISION_DATASET_NAME,
} from "../../evals/shititong/shititong";
import { US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2 } from "../../evals/us_coast_guard/us_coast_guard";

const MULTIMODAL_DATASETS = [
  US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
  RAYNOR_DATASET_NAME_MULTIMODAL_V2,
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_VISION_DATASET_NAME,
  ...navreasDatasets.map(toDatasetName),
] as const;

const EXCLUDED_MODEL_IDS = ["openai/gpt-oss-120b"] as const;

const scriptDir = dirname(fileURLToPath(import.meta.url));
const aggregateScript = join(scriptDir, "aggregate-experiment-scores.ts");
const passthroughArgs = process.argv.slice(2);

const aggregateArgs = [
  aggregateScript,
  "--prefer-scored",
  "--exclude-metadata-keyword",
  "meta-llama/",
  "--exclude-metadata-keyword",
  "llamarine",
  ...EXCLUDED_MODEL_IDS.flatMap((modelId) => ["--exclude-metadata-keyword", modelId]),
  ...MULTIMODAL_DATASETS.flatMap((datasetName) => ["--dataset", datasetName]),
  ...passthroughArgs,
];

const subprocess = Bun.spawn(["bun", ...aggregateArgs], {
  stdin: "inherit",
  stdout: "inherit",
  stderr: "inherit",
});

const exitCode = await subprocess.exited;
process.exit(exitCode);
