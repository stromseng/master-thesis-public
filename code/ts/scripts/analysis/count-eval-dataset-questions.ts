// Usage:
//   bun scripts/analysis/count-eval-dataset-questions.ts
//
// Counts the local question files backing the eval datasets and writes:
//   data/analysis/question-counts.json
import { mkdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname } from "node:path";
import * as S from "effect/Schema";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../../evals/question-schema";
import {
  OpenEndedQuestionGroupsFromJson,
  type OpenEndedQuestionGroups,
} from "../../evals/open-ended-question-schema";
import { dataPath, evalDataPath } from "../../src/utils/repo";

type DatasetKind = "text-only" | "multimodal" | "open-ended";

type QuestionGroupLike = EvalQuestionGroups | OpenEndedQuestionGroups;

type DatasetDefinition = {
  name: string;
  kind: DatasetKind;
  format: "mcq" | "open-ended";
  fileSegments: readonly [string, ...string[]];
  notes?: string;
  derivedFrom?: string;
  limit?: number;
};

type DatasetCount = {
  name: string;
  kind: DatasetKind;
  questionCount: number;
  groupCount: number;
  file: string;
  notes?: string;
  derivedFrom?: string;
};

type AnalysisOutput = {
  generatedAt: string;
  outputFile: string;
  summary: {
    totalQuestions: number;
    textOnlyQuestions: number;
    multimodalQuestions: number;
    openEndedQuestions: number;
    datasetCount: number;
    textOnlyDatasetCount: number;
    multimodalDatasetCount: number;
    openEndedDatasetCount: number;
  };
  datasets: DatasetCount[];
};

const DATASETS: readonly DatasetDefinition[] = [
  {
    name: "crewcn",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["crewcn", "crewcn_exam_questions.json"],
  },
  {
    name: "pei2024-uk",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["pei2024application", "uk_theory_test.json"],
  },
  {
    name: "pei2024-zh",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["pei2024application", "zh_theory_test.json"],
  },
  {
    name: "raynor-v2",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["raynor", "text_only.json"],
  },
  {
    name: "shititong-en-text",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["shititong", "shititong_english_deduped_text.json"],
  },
  {
    name: "shititong-zh-text",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["shititong", "shititong_chinese_deduped_text.json"],
  },
  {
    name: "us-coast-guard-text-only-v2",
    kind: "text-only",
    format: "mcq",
    fileSegments: ["us_coast_guard", "all_questions_text_only.json"],
  },
  {
    name: "navreas-scene-understanding",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["navreas", "scene_understanding.eval.json"],
  },
  {
    name: "navreas-colreg-compliance",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["navreas", "colreg_compliance_and_good_seamanship.eval.json"],
  },
  {
    name: "navreas-spatial-relationship",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["navreas", "spatial_relationship_and_estimation_of_motion.eval.json"],
  },
  {
    name: "raynor-multimodal-v2",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["raynor", "multimodal.json"],
  },
  {
    name: "shititong-en-vision",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["shititong", "shititong_english_deduped_vision.json"],
  },
  {
    name: "shititong-zh-vision",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["shititong", "shititong_chinese_deduped_vision.json"],
  },
  {
    name: "us-coast-guard-multimodal-v2",
    kind: "multimodal",
    format: "mcq",
    fileSegments: ["us_coast_guard", "all_questions_multimodal.json"],
  },
  {
    name: "crewcn-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: ["open_ended", "crewcn-2026-04-10T09-12-07Z.open-ended.eval.json"],
    derivedFrom: "crewcn",
  },
  {
    name: "pei2024-uk-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: ["open_ended", "pei2024-uk-2026-04-09T11-52-35Z.open-ended.eval.json"],
    derivedFrom: "pei2024-uk",
  },
  {
    name: "pei2024-zh-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: ["open_ended", "pei2024-zh-2026-04-10T09-12-07Z.open-ended.eval.json"],
    derivedFrom: "pei2024-zh",
  },
  {
    name: "raynor-v2-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: ["open_ended", "raynor-v2-2026-04-10T09-12-07Z.open-ended.eval.json"],
    derivedFrom: "raynor-v2",
  },
  {
    name: "shititong-en-text-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: ["open_ended", "shititong-en-text-2026-04-10T09-12-07Z.open-ended.eval.json"],
    derivedFrom: "shititong-en-text",
  },
  {
    name: "shititong-zh-text-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: ["open_ended", "shititong-zh-text-2026-04-15T13-22-35Z.open-ended.eval.json"],
    derivedFrom: "shititong-zh-text",
  },
  {
    name: "us-coast-guard-text-only-v2-open-ended",
    kind: "open-ended",
    format: "open-ended",
    fileSegments: [
      "open_ended",
      "us-coast-guard-text-only-v2-2026-04-10T09-12-07Z.open-ended.eval.json",
    ],
    derivedFrom: "us-coast-guard-text-only-v2",
  },
] as const;

const countQuestions = (groups: QuestionGroupLike, limit?: number): number => {
  const count = groups.reduce((total, group) => total + group.questions.length, 0);
  return limit === undefined ? count : Math.min(count, limit);
};

const loadDataset = async (definition: DatasetDefinition): Promise<DatasetCount> => {
  const raw = await readFile(evalDataPath(...definition.fileSegments), "utf-8");
  const groups =
    definition.format === "mcq"
      ? S.decodeUnknownSync(QuestionGroupsFromJson)(raw)
      : S.decodeUnknownSync(OpenEndedQuestionGroupsFromJson)(raw);

  return {
    name: definition.name,
    kind: definition.kind,
    questionCount: countQuestions(groups, definition.limit),
    groupCount: groups.length,
    file: `data/evals/${definition.fileSegments.join("/")}`,
    ...(definition.notes ? { notes: definition.notes } : {}),
    ...(definition.derivedFrom ? { derivedFrom: definition.derivedFrom } : {}),
  };
};

const sumByKind = (datasets: readonly DatasetCount[], kind: DatasetKind): number =>
  datasets
    .filter((dataset) => dataset.kind === kind)
    .reduce((total, dataset) => total + dataset.questionCount, 0);

const countByKind = (datasets: readonly DatasetCount[], kind: DatasetKind): number =>
  datasets.filter((dataset) => dataset.kind === kind).length;

const main = async () => {
  const datasets = await Promise.all(DATASETS.map(loadDataset));
  const outputPath = dataPath("analysis", "question-counts.json");

  const output: AnalysisOutput = {
    generatedAt: new Date().toISOString(),
    outputFile: "data/analysis/question-counts.json",
    summary: {
      totalQuestions: datasets.reduce((total, dataset) => total + dataset.questionCount, 0),
      textOnlyQuestions: sumByKind(datasets, "text-only"),
      multimodalQuestions: sumByKind(datasets, "multimodal"),
      openEndedQuestions: sumByKind(datasets, "open-ended"),
      datasetCount: datasets.length,
      textOnlyDatasetCount: countByKind(datasets, "text-only"),
      multimodalDatasetCount: countByKind(datasets, "multimodal"),
      openEndedDatasetCount: countByKind(datasets, "open-ended"),
    },
    datasets,
  };

  mkdirSync(dirname(outputPath), { recursive: true });
  await Bun.write(outputPath, `${JSON.stringify(output, null, 2)}\n`);

  console.log(`Wrote ${output.outputFile}`);
  console.log(
    [
      `Text-only: ${output.summary.textOnlyQuestions}`,
      `Multimodal: ${output.summary.multimodalQuestions}`,
      `Open-ended: ${output.summary.openEndedQuestions}`,
      `Total: ${output.summary.totalQuestions}`,
    ].join("\n"),
  );
};

await main();
