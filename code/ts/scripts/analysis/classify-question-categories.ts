// Usage:
//   bun scripts/analysis/classify-question-categories.ts
//   bun scripts/analysis/classify-question-categories.ts --dataset pei2024-uk
//   bun scripts/analysis/classify-question-categories.ts --dataset crewcn --dataset raynor-v2
//   bun scripts/analysis/classify-question-categories.ts --limit 100 --batch-size 12
//   bun scripts/analysis/classify-question-categories.ts --sample-percent 2
//   bun scripts/analysis/classify-question-categories.ts --output data/analysis/question-category-analysis-run.json
//   bun scripts/analysis/classify-question-categories.ts --resume data/analysis/question-category-analysis-run.json
//   bun scripts/analysis/classify-question-categories.ts --dry-run
//
// Environment variables (same as eval scripts):
//   EVAL_PROVIDER  — "litellm" (default) or "vllm"
//   EVAL_MODEL     — explicit model ID (auto-detected for vllm if omitted)
//   VLLM_BASE_URL / VLLM_PORT — vLLM endpoint config
//
// Classifies questions in the evaluation datasets into topic and reasoning
// categories using an LLM and writes a JSON report to data/analysis/.
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path";
import { jsonSchema, type ImagePart, type ModelMessage, type TextPart } from "ai";
import { Command, Options } from "@effect/cli";
import { NodeContext, NodeRuntime } from "@effect/platform-node";
import * as Progress from "effective-progress";
import { Cause, Duration, Effect, Layer, Logger, LogLevel, Option, Schedule, Schema } from "effect";
import * as JSONSchema from "effect/JSONSchema";
import * as S from "effect/Schema";
import {
  loadQuestionGroups as loadCrewQuestionGroups,
  CREWCN_DATASET_NAME,
} from "../../evals/crew/crew";
import {
  loadQuestionGroups as loadNavreasQuestionGroups,
  loadImage as loadNavreasImage,
  navreasDatasets,
  toDatasetName as toNavreasDatasetName,
} from "../../evals/navreas/navreas";
import {
  loadUkQuestionGroups,
  loadZhQuestionGroups,
  PEI2024_UK_DATASET_NAME,
  PEI2024_ZH_DATASET_NAME,
} from "../../evals/pei2024/pei2024";
import { type EvalQuestion, type EvalQuestionGroups } from "../../evals/question-schema";
import {
  loadImage as loadRaynorImage,
  loadMultimodalQuestionGroups as loadRaynorMultimodalQuestionGroups,
  loadQuestionGroups as loadRaynorQuestionGroups,
  RAYNOR_DATASET_NAME_MULTIMODAL_V2,
  RAYNOR_DATASET_NAME_V2,
} from "../../evals/raynor/raynor";
import {
  loadEnTextGroups,
  loadEnVisionGroups,
  loadZhTextGroups,
  loadZhVisionGroups,
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_EN_VISION_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
  SHITITONG_ZH_VISION_DATASET_NAME,
} from "../../evals/shititong/shititong";
import {
  loadImage as loadUsCoastGuardImage,
  loadMultimodalQuestionGroups as loadUsCoastGuardMultimodalQuestionGroups,
  loadTextOnlyQuestionGroups as loadUsCoastGuardTextOnlyQuestionGroups,
  US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
} from "../../evals/us_coast_guard/us_coast_guard";
import {
  EvalLanguageModelLayer,
  EvalMultimodalLanguageModelLayer,
  generateObject,
  LanguageModelError,
} from "../../src/services/LanguageModel";
import { dataPath, repoPath } from "../../src/utils/repo";

// ---------------------------------------------------------------------------
// Configurable category definitions
// ---------------------------------------------------------------------------

export const TOPIC_CATEGORIES = [
  {
    id: "collision_avoidance_and_colregs",
    description:
      "Collision avoidance, COLREG encounter rules, right-of-way, restricted visibility conduct, and associated operational decisions",
  },
  {
    id: "navigation_positioning_and_passage_planning",
    description:
      "Navigation, position fixing, bearings, courses, passage planning, watchkeeping, and route execution outside collision-rule questions",
  },
  {
    id: "shiphandling_and_seamanship",
    description:
      "Shiphandling, anchoring, towing, mooring, berthing, ropes, deck work, and practical seamanship outside pure collision rules",
  },
  {
    id: "safety_emergency_and_survival",
    description:
      "Safety management, emergencies, firefighting, life-saving appliances, drills, and survival actions",
  },
  {
    id: "marine_engineering_and_propulsion",
    description:
      "Main propulsion, auxiliary machinery, pumps, fuel, lubrication, cooling, compressors, and machinery operation",
  },
  {
    id: "electrical_technical_and_bridge_systems",
    description:
      "Electrical systems, bridge equipment, communications, radar, ARPA, ECDIS, GNSS, gyro, sensors, and instrumentation as technical systems",
  },
  {
    id: "cargo_stability_and_ship_construction",
    description:
      "Cargo operations, loading, unloading, stability, trim, stress, structure, construction, and vessel geometry",
  },
  {
    id: "maritime_regulation_and_compliance",
    description:
      "Formal legal and regulatory obligations, conventions, certification, documentation, port and flag requirements, and compliance duties",
  },
  {
    id: "meteorology_oceanography_and_environment",
    description:
      "Weather, oceanography, sea state, currents, environmental conditions, and environmental effects on operations",
  },
  {
    id: "crew_management_and_communication",
    description:
      "Crew duties, bridge teamwork, SMCP, watch organization, training, leadership, and professional responsibilities",
  },
  {
    id: "non_maritime_off_domain",
    description:
      "Questions that are clearly outside maritime knowledge domains, such as aviation, railway, construction, general law, politics, or other unrelated subjects",
  },
  {
    id: "corrupted_or_prompt_leakage",
    description:
      "Questions whose text is corrupted, incomplete, OCR-garbled, or polluted with repair instructions, prompt text, or other non-question artifacts",
  },
  {
    id: "other_unclear",
    description: "Use only when none of the other topic categories fit well",
  },
] as const;

export const SECONDARY_FACETS = [
  {
    id: "charts_publications_and_notices",
    description:
      "Charts, sailing directions, notices, almanacs, publications, and reference tables",
  },
  {
    id: "tides_currents_and_hydrography",
    description:
      "Tides, currents, tidal streams, depths, soundings, and hydrographic interpretation",
  },
  {
    id: "lights_shapes_buoys_and_sound_signals",
    description: "Lights, day shapes, buoyage, marks, symbols, and sound-signal patterns",
  },
  {
    id: "radar_arpa_ecdis_gnss_and_instrumentation",
    description:
      "Radar, ARPA, ECDIS, GPS/GNSS, gyro, AIS, echo sounder, and related instrumentation",
  },
  {
    id: "mooring_anchoring_and_berthing",
    description: "Mooring, anchoring, berthing, line handling, and associated practical operations",
  },
  {
    id: "cargo_operations",
    description: "Cargo-handling workflows, stowage, loading, unloading, and cargo care",
  },
  {
    id: "stability_trim_stress_and_structure",
    description:
      "Stability, trim, stress, loading condition, hull form, and vessel structure details",
  },
  {
    id: "solas_marpol_stcw_and_formal_requirements",
    description:
      "SOLAS, MARPOL, STCW, certification, statutory forms, and formal convention requirements",
  },
  {
    id: "bridge_resource_management_and_smcp",
    description:
      "Bridge teamwork, BRM, closed-loop communication, SMCP, and coordination procedures",
  },
  {
    id: "pollution_prevention_and_environmental_protection",
    description:
      "Pollution prevention, discharge restrictions, and environmental protection controls",
  },
] as const;

export const TEXT_REASONING_CATEGORIES = [
  {
    id: "factual_recall",
    description:
      "Primarily requires recalling a fact, definition, threshold, rule text, or direct piece of knowledge",
  },
  {
    id: "rule_application",
    description:
      "Requires applying a known rule, convention, or procedure to the presented situation or options",
  },
  {
    id: "calculation",
    description:
      "Requires arithmetic, formula use, numeric conversion, or stepwise quantitative calculation",
  },
  {
    id: "scenario_judgment",
    description:
      "Requires choosing the best action or diagnosis in a practical scenario, but not detailed spatial geometry",
  },
  {
    id: "spatial_reasoning",
    description:
      "Requires geometric or relational reasoning about vessel positions, motion, crossing, overtaking, or scene layout",
  },
  {
    id: "procedural_sequence",
    description:
      "Requires understanding ordered steps, sequencing, or the correct operational workflow",
  },
] as const;

export const MULTIMODAL_REASONING_CATEGORIES = [
  ...TEXT_REASONING_CATEGORIES,
  {
    id: "visual_identification",
    description:
      "Requires identifying symbols, shapes, lights, buoys, charts, or diagrams from visual cues",
  },
] as const;

export const CLASSIFICATION_MODE_DEFINITIONS = [
  {
    id: "text_only",
    description: "Question can be classified from text alone with no image interpretation required",
  },
  {
    id: "multimodal",
    description: "Question classification depends on the presence of question or option images",
  },
] as const;

export const REASONING_CATEGORY_DEFINITIONS = [
  ...TEXT_REASONING_CATEGORIES,
  ...MULTIMODAL_REASONING_CATEGORIES.filter(
    (category) => !TEXT_REASONING_CATEGORIES.some((existing) => existing.id === category.id),
  ),
] as const;

export type TopicCategory = (typeof TOPIC_CATEGORIES)[number]["id"];
export type SecondaryFacet = (typeof SECONDARY_FACETS)[number]["id"];
export type TextReasoningCategory = (typeof TEXT_REASONING_CATEGORIES)[number]["id"];
export type MultimodalReasoningCategory = (typeof MULTIMODAL_REASONING_CATEGORIES)[number]["id"];
export type ReasoningCategory = TextReasoningCategory | MultimodalReasoningCategory;
export type ClassificationMode = "text_only" | "multimodal";

const TOPIC_CATEGORY_IDS = TOPIC_CATEGORIES.map((category) => category.id) as unknown as readonly [
  TopicCategory,
  ...TopicCategory[],
];
const SECONDARY_FACET_IDS = SECONDARY_FACETS.map((facet) => facet.id) as unknown as readonly [
  SecondaryFacet,
  ...SecondaryFacet[],
];
const TEXT_REASONING_CATEGORY_IDS = TEXT_REASONING_CATEGORIES.map(
  (category) => category.id,
) as unknown as readonly [TextReasoningCategory, ...TextReasoningCategory[]];
const MULTIMODAL_REASONING_CATEGORY_IDS = MULTIMODAL_REASONING_CATEGORIES.map(
  (category) => category.id,
) as unknown as readonly [MultimodalReasoningCategory, ...MultimodalReasoningCategory[]];
// ---------------------------------------------------------------------------
// Classification schema
// ---------------------------------------------------------------------------

const SecondaryFacetsSchema = S.Array(S.Literal(...SECONDARY_FACET_IDS))
  .annotations({
    description:
      "Optional supporting facet tags. Use zero to three items. Avoid duplicates and use only true supporting facets.",
  })
  .pipe(
    S.filter((facets): facets is ReadonlyArray<SecondaryFacet> => facets.length <= 3, {
      identifier: "AtMostThreeSecondaryFacets",
      description: "Use zero to three secondary facets",
    }),
  );

const BaseQuestionClassificationSchema = S.Struct({
  primaryTopic: S.Literal(...TOPIC_CATEGORY_IDS).annotations({
    description: "Best-fit primary domain for the question",
  }),
  secondaryFacets: SecondaryFacetsSchema,
});

const TextQuestionClassificationSchema = S.Struct({
  ...BaseQuestionClassificationSchema.fields,
  reasoningType: S.Literal(...TEXT_REASONING_CATEGORY_IDS).annotations({
    description: "Main reasoning pattern needed to answer the question",
  }),
});

const MultimodalQuestionClassificationSchema = S.Struct({
  ...BaseQuestionClassificationSchema.fields,
  reasoningType: S.Literal(...MULTIMODAL_REASONING_CATEGORY_IDS).annotations({
    description: "Main reasoning pattern needed to answer the question",
  }),
});

const TextBatchQuestionClassificationSchema = S.Struct({
  questionId: S.String.annotations({
    description: "The exact question ID from the prompt",
  }),
  ...TextQuestionClassificationSchema.fields,
});

const TextBatchClassificationResponseSchema = S.Struct({
  results: S.Array(TextBatchQuestionClassificationSchema),
});

const MultimodalQuestionClassificationResponseSchema = S.Struct({
  questionId: S.String.annotations({
    description: "The exact question ID from the prompt",
  }),
  ...MultimodalQuestionClassificationSchema.fields,
});

const textBatchClassificationJsonSchema = jsonSchema<{
  results: Array<{
    questionId: string;
    primaryTopic: TopicCategory;
    secondaryFacets: SecondaryFacet[];
    reasoningType: TextReasoningCategory;
  }>;
}>(JSONSchema.make(TextBatchClassificationResponseSchema));

const multimodalQuestionClassificationJsonSchema = jsonSchema<{
  questionId: string;
  primaryTopic: TopicCategory;
  secondaryFacets: SecondaryFacet[];
  reasoningType: MultimodalReasoningCategory;
}>(JSONSchema.make(MultimodalQuestionClassificationResponseSchema));

export type TextQuestionClassification = S.Schema.Type<typeof TextQuestionClassificationSchema>;
export type MultimodalQuestionClassification = S.Schema.Type<
  typeof MultimodalQuestionClassificationSchema
>;
export type QuestionClassification = TextQuestionClassification | MultimodalQuestionClassification;

export type QuestionClassificationResult = {
  sourceDataset: string;
  groupId: string;
  questionKey: string;
  questionId: string;
  questionText: string;
  optionCount: number;
  hasImages: boolean;
  questionMetadata: unknown;
  questionSource?: Record<string, string>;
  groupMetadata?: unknown;
  groupSource?: Record<string, string>;
  classificationMode: ClassificationMode;
  classification: QuestionClassification;
};

// ---------------------------------------------------------------------------
// Dataset registry
// ---------------------------------------------------------------------------

type DatasetDefinition = {
  id: string;
  family: string;
  language: "EN" | "ZH";
  modality: "text" | "multimodal" | "spatial";
  loadGroups: () => Effect.Effect<EvalQuestionGroups, unknown>;
  loadImage?: (uri: string) => Effect.Effect<Uint8Array, unknown>;
};

export const DATASET_DEFINITIONS: ReadonlyArray<DatasetDefinition> = [
  {
    id: PEI2024_UK_DATASET_NAME,
    family: "PEI2024",
    language: "EN",
    modality: "text",
    loadGroups: loadUkQuestionGroups,
  },
  {
    id: PEI2024_ZH_DATASET_NAME,
    family: "PEI2024",
    language: "ZH",
    modality: "text",
    loadGroups: loadZhQuestionGroups,
  },
  {
    id: US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
    family: "US Coast Guard",
    language: "EN",
    modality: "text",
    loadGroups: loadUsCoastGuardTextOnlyQuestionGroups,
  },
  {
    id: US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
    family: "US Coast Guard",
    language: "EN",
    modality: "multimodal",
    loadGroups: loadUsCoastGuardMultimodalQuestionGroups,
    loadImage: loadUsCoastGuardImage,
  },
  {
    id: RAYNOR_DATASET_NAME_V2,
    family: "Raynor",
    language: "EN",
    modality: "text",
    loadGroups: loadRaynorQuestionGroups,
  },
  {
    id: RAYNOR_DATASET_NAME_MULTIMODAL_V2,
    family: "Raynor",
    language: "EN",
    modality: "multimodal",
    loadGroups: loadRaynorMultimodalQuestionGroups,
    loadImage: loadRaynorImage,
  },
  {
    id: SHITITONG_EN_TEXT_DATASET_NAME,
    family: "Shititong",
    language: "EN",
    modality: "text",
    loadGroups: loadEnTextGroups,
  },
  {
    id: SHITITONG_EN_VISION_DATASET_NAME,
    family: "Shititong",
    language: "EN",
    modality: "multimodal",
    loadGroups: loadEnVisionGroups,
  },
  {
    id: SHITITONG_ZH_TEXT_DATASET_NAME,
    family: "Shititong",
    language: "ZH",
    modality: "text",
    loadGroups: loadZhTextGroups,
  },
  {
    id: SHITITONG_ZH_VISION_DATASET_NAME,
    family: "Shititong",
    language: "ZH",
    modality: "multimodal",
    loadGroups: loadZhVisionGroups,
  },
  {
    id: CREWCN_DATASET_NAME,
    family: "CrewCN",
    language: "ZH",
    modality: "text",
    loadGroups: loadCrewQuestionGroups,
  },
  ...navreasDatasets.map((dataset) => ({
    id: toNavreasDatasetName(dataset),
    family: "NavReas",
    language: "EN" as const,
    modality: "spatial" as const,
    loadGroups: () => loadNavreasQuestionGroups(dataset.filename),
    loadImage: loadNavreasImage,
  })),
] as const;

const DATASET_IDS = DATASET_DEFINITIONS.map((dataset) => dataset.id);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class UnknownDatasetError extends Schema.TaggedError<UnknownDatasetError>()(
  "UnknownDatasetError",
  {
    requestedDatasets: Schema.Array(Schema.String),
    availableDatasets: Schema.Array(Schema.String),
  },
) {}

export class InvalidBatchClassificationError extends Schema.TaggedError<InvalidBatchClassificationError>()(
  "InvalidBatchClassificationError",
  {
    sourceDataset: Schema.String,
    missingQuestionIds: Schema.Array(Schema.String),
    duplicateQuestionIds: Schema.Array(Schema.String),
    extraQuestionIds: Schema.Array(Schema.String),
  },
) {}

export class WriteOutputError extends Schema.TaggedError<WriteOutputError>()("WriteOutputError", {
  path: Schema.String,
  cause: Schema.Defect,
}) {}

export class ReadCheckpointError extends Schema.TaggedError<ReadCheckpointError>()(
  "ReadCheckpointError",
  {
    path: Schema.String,
    cause: Schema.Defect,
  },
) {}

export class ResumeCheckpointMismatchError extends Schema.TaggedError<ResumeCheckpointMismatchError>()(
  "ResumeCheckpointMismatchError",
  {
    path: Schema.String,
    mismatches: Schema.Array(Schema.String),
  },
) {}

export class InvalidSamplePercentError extends Schema.TaggedError<InvalidSamplePercentError>()(
  "InvalidSamplePercentError",
  {
    value: Schema.String,
  },
) {}

export class UnsupportedLocalImageError extends Schema.TaggedError<UnsupportedLocalImageError>()(
  "UnsupportedLocalImageError",
  {
    sourceDataset: Schema.String,
    uri: Schema.String,
  },
) {}

export class ClassificationTimeoutError extends Schema.TaggedError<ClassificationTimeoutError>()(
  "ClassificationTimeoutError",
  {
    questionId: Schema.String,
    duration: Schema.String,
  },
) {}

/** Returns true if the error is a deterministic failure that should not be retried. */
const isUnrecoverableError = (error: unknown): boolean =>
  error instanceof ClassificationTimeoutError ||
  (error instanceof LanguageModelError &&
    error.cause instanceof Error &&
    error.cause.name === "AI_APICallError");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type SourceQuestion = {
  sourceDataset: string;
  groupId: string;
  family: string;
  language: "EN" | "ZH";
  datasetModality: "text" | "multimodal" | "spatial";
  groupMetadata?: unknown;
  groupSource?: Record<string, string>;
  question: EvalQuestion;
  loadImage?: (uri: string) => Effect.Effect<Uint8Array, unknown>;
};

type PreparedImageReference = {
  label: string;
  caption?: string;
  part: ImagePart;
};

const SAVE_EVERY = 100;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const normalizeStringArray = (values: readonly string[] = []) => [...new Set(values)].sort();

const questionKeyOf = (sourceDataset: string, questionId: string) =>
  `${sourceDataset}::${questionId}`;

const roundPercentage = (count: number, total: number): number =>
  total === 0 ? 0 : Number(((count / total) * 100).toFixed(1));

const toSortedCountEntries = (
  counts: Map<string, number>,
  total: number,
): Array<{ category: string; count: number; percentage: number }> =>
  [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([category, count]) => ({
      category,
      count,
      percentage: roundPercentage(count, total),
    }));

const toDenseCountEntries = (
  categories: ReadonlyArray<string>,
  counts: Map<string, number>,
  total: number,
) =>
  categories.map((category) => {
    const count = counts.get(category) ?? 0;
    return {
      category,
      count,
      percentage: roundPercentage(count, total),
    };
  });

const countBy = <A>(items: ReadonlyArray<A>, keyOf: (item: A) => string) => {
  const counts = new Map<string, number>();
  for (const item of items) {
    const key = keyOf(item);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
};

const chunkArray = <A>(items: ReadonlyArray<A>, chunkSize: number): A[][] => {
  if (chunkSize <= 0) return [Array.from(items)];
  const chunks: A[][] = [];
  for (let index = 0; index < items.length; index += chunkSize) {
    chunks.push(items.slice(index, index + chunkSize));
  }
  return chunks;
};

const stableHash = (value: string): number => {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const questionHasAnyImages = (question: EvalQuestion): boolean =>
  question.images.length > 0 || question.options.some((option) => option.images.length > 0);

/** Check that an image URI is loadable for the given source question. */
const isValidImageUri = (sourceQuestion: SourceQuestion, rawUri: string): boolean => {
  const uri = rawUri.replace(/"+$/, "");
  if (!uri || uri.trim().length === 0) return false;
  if (uri.startsWith("http://") || uri.startsWith("https://")) {
    try {
      new URL(uri);
      return true;
    } catch {
      return false;
    }
  }
  // Local paths are only valid if the dataset provides an image loader
  return sourceQuestion.loadImage !== undefined;
};

const getClassificationMode = (sourceQuestion: SourceQuestion): ClassificationMode =>
  questionHasAnyImages(sourceQuestion.question) ? "multimodal" : "text_only";

const summarizeMetadata = (metadata: unknown): string | undefined => {
  if (!isRecord(metadata)) return undefined;

  const interestingKeys = ["questionType", "chapter", "section", "applicability", "examTitle"];

  const pairs: string[] = [];
  for (const key of interestingKeys) {
    const value = metadata[key];
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      pairs.push(`${key}: ${String(value)}`);
    }
  }

  if (pairs.length === 0) return undefined;
  return pairs.join("\n");
};

const formatImageReferences = (question: EvalQuestion): string => {
  if (question.images.length === 0) return "None";
  return question.images
    .map((image, index) => {
      const caption = image.caption ? ` | caption: ${image.caption}` : "";
      return `${index + 1}. Question image ${index + 1}${caption}`;
    })
    .join("\n");
};

const formatOptionsForTextPrompt = (question: EvalQuestion): string =>
  question.options
    .map((option) => {
      const imageNote =
        option.images.length === 0
          ? ""
          : ` [attached option images: ${option.images
              .map((_, index) => `Option ${option.id} image ${index + 1}`)
              .join(", ")}]`;
      const optionText = option.text.length > 0 ? option.text : "[see attached option image]";
      return `${option.id}) ${optionText}${imageNote}`;
    })
    .join("\n");

const formatQuestionForTextPrompt = (sourceQuestion: SourceQuestion): string => {
  const metadata = summarizeMetadata(sourceQuestion.question.metadata);
  return [
    `Question ID: ${sourceQuestion.question.id}`,
    `Dataset: ${sourceQuestion.sourceDataset}`,
    `Family: ${sourceQuestion.family}`,
    `Language: ${sourceQuestion.language}`,
    `Dataset modality: ${sourceQuestion.datasetModality}`,
    "Classification mode: text_only",
    "",
    "Question text:",
    sourceQuestion.question.questionText,
    "",
    "Options:",
    formatOptionsForTextPrompt(sourceQuestion.question),
    "",
    "Question-level images:",
    "None",
    ...(metadata ? ["", "Metadata hints:", metadata] : []),
  ].join("\n");
};

const formatOptionsForMultimodalPrompt = (question: EvalQuestion): string =>
  question.options
    .map((option) => {
      const imageNote =
        option.images.length === 0
          ? ""
          : ` [attached option images: ${option.images
              .map((_, index) => `Option ${option.id} image ${index + 1}`)
              .join(", ")}]`;
      const optionText = option.text.length > 0 ? option.text : "[see attached option image]";
      return `${option.id}) ${optionText}${imageNote}`;
    })
    .join("\n");

const buildSystemPrompt = (mode: ClassificationMode) => {
  const topicList = TOPIC_CATEGORIES.map(
    (category) => `- ${category.id}: ${category.description}`,
  ).join("\n");
  const facetList = SECONDARY_FACETS.map((facet) => `- ${facet.id}: ${facet.description}`).join(
    "\n",
  );
  const reasoningCategories =
    mode === "multimodal" ? MULTIMODAL_REASONING_CATEGORIES : TEXT_REASONING_CATEGORIES;
  const reasoningList = reasoningCategories
    .map((category) => `- ${category.id}: ${category.description}`)
    .join("\n");
  const visualRule =
    mode === "multimodal"
      ? '\n- Use "visual_identification" only when the answer depends on interpreting the actual image content, not merely because the question has an image attached.'
      : "";

  return `You classify maritime exam and benchmark questions into a fixed maritime taxonomy.

You must classify each question using exactly:
1. one primary topic
2. zero to three secondary facets
3. one reasoning type

Primary topic categories:
${topicList}

Secondary facet categories:
${facetList}

Reasoning categories:
${reasoningList}

Rules:
- Questions may be in English or Chinese. Classify by meaning, not by language.
- Choose the primary topic as the main knowledge domain being tested.
- Use secondary facets only for supporting cross-cutting themes, referenced systems, or cited materials.
- Use "non_maritime_off_domain" when the question is understandable but clearly not maritime.
- Use "corrupted_or_prompt_leakage" when the question text is broken, incomplete, OCR-garbled, or contains leaked repair/prompt instructions instead of a clean question.
- Use "other_unclear" only when the question still seems maritime-relevant but no specific topic category fits well.
- Secondary facets must be distinct.
- Use "calculation" only when the respondent must actually compute something.
- Use "factual_recall" only when the question is mostly direct memory or definition recall.
- Use "factual_recall" rather than "rule_application" when the question asks for a definition, threshold, or stated requirement without situational application.${visualRule}
- Base the classification on the question only, not on any presumed answer quality.
- Use metadata hints only as weak disambiguating evidence. Never let metadata override the question itself.

Tie-breakers:
- Use "collision_avoidance_and_colregs" when the governing knowledge is a collision rule or encounter obligation. Use "navigation_positioning_and_passage_planning" for position fixing, course/bearing work, passage execution, charts, or tides when no specific collision rule is central.
- Use "shiphandling_and_seamanship" for anchoring, mooring, berthing, towing, line handling, and practical handling tasks that are not mainly about COLREG obligations.
- Use "maritime_regulation_and_compliance" when the question is primarily about statutory duties, certificates, convention requirements, or formal compliance rather than operational execution.
- Use "electrical_technical_and_bridge_systems" when the focal knowledge is how a system or instrument works or should be operated. If the equipment is only context for a navigation or COLREG judgment, keep that operational topic as primary and use a secondary facet if helpful.
- Use "cargo_stability_and_ship_construction" when loading, unloading, stowage, stability, trim, stress, or structural knowledge is central.
- Use "non_maritime_off_domain" rather than forcing an unrelated question into the closest maritime bucket.
- Use "corrupted_or_prompt_leakage" rather than "non_maritime_off_domain" when the dominant problem is unusable or contaminated text rather than merely being off-topic.
- Return the exact question ID from the prompt.`;
};

const buildTextBatchPrompt = (questions: ReadonlyArray<SourceQuestion>) =>
  `Classify the following ${questions.length} text-only questions.\n\n${questions
    .map((question, index) => `## Question ${index + 1}\n${formatQuestionForTextPrompt(question)}`)
    .join("\n\n")}`;

const buildMultimodalQuestionPrompt = (sourceQuestion: SourceQuestion) => {
  const metadata = summarizeMetadata(sourceQuestion.question.metadata);
  return [
    "Classify this multimodal question.",
    "",
    `Question ID: ${sourceQuestion.question.id}`,
    `Dataset: ${sourceQuestion.sourceDataset}`,
    `Family: ${sourceQuestion.family}`,
    `Language: ${sourceQuestion.language}`,
    `Dataset modality: ${sourceQuestion.datasetModality}`,
    "Classification mode: multimodal",
    "",
    "Question text:",
    sourceQuestion.question.questionText,
    "",
    "Options:",
    formatOptionsForMultimodalPrompt(sourceQuestion.question),
    "",
    "Attached question-level images:",
    formatImageReferences(sourceQuestion.question),
    ...(metadata ? ["", "Metadata hints:", metadata] : []),
    "",
    "The labeled images follow below in the same order they are referenced.",
  ].join("\n");
};

export const normalizeClassification = (
  classification: QuestionClassification,
): QuestionClassification => {
  const secondaryFacets = [...new Set(classification.secondaryFacets)].slice(0, 3);

  return {
    ...classification,
    secondaryFacets,
  };
};

export const selectDatasets = (
  requestedDatasets: ReadonlyArray<string>,
): Effect.Effect<ReadonlyArray<DatasetDefinition>, UnknownDatasetError> => {
  if (requestedDatasets.length === 0) {
    return Effect.succeed(DATASET_DEFINITIONS);
  }

  const selected = DATASET_DEFINITIONS.filter((dataset) => requestedDatasets.includes(dataset.id));
  const unknownDatasets = requestedDatasets.filter((requested) => !DATASET_IDS.includes(requested));

  if (unknownDatasets.length > 0) {
    return new UnknownDatasetError({
      requestedDatasets: unknownDatasets,
      availableDatasets: [...DATASET_IDS].sort(),
    });
  }

  return Effect.succeed(selected);
};

export const parseSamplePercent = (
  rawValue: Option.Option<string>,
): Effect.Effect<number, InvalidSamplePercentError> =>
  Option.match(rawValue, {
    onNone: () => Effect.succeed(100),
    onSome: (value) => {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 100) {
        return new InvalidSamplePercentError({ value });
      }
      return Effect.succeed(parsed);
    },
  });

export const sampleQuestionsByPercentage = <
  A extends { sourceDataset: string; question: { id: string } },
>(
  items: ReadonlyArray<A>,
  samplePercent: number,
): ReadonlyArray<A> => {
  if (samplePercent <= 0) return items;
  if (samplePercent >= 100) return items;
  if (items.length === 0) return items;

  const targetCount = Math.max(1, Math.ceil((items.length * samplePercent) / 100));
  const selectedIds = new Set(
    [...items]
      .sort(
        (left, right) =>
          stableHash(`${left.sourceDataset}::${left.question.id}`) -
            stableHash(`${right.sourceDataset}::${right.question.id}`) ||
          left.question.id.localeCompare(right.question.id),
      )
      .slice(0, targetCount)
      .map((item) => item.question.id),
  );

  return items.filter((item) => selectedIds.has(item.question.id));
};

const flattenQuestionGroups = (
  dataset: DatasetDefinition,
  groups: EvalQuestionGroups,
  samplePercent: number,
  limit: number,
): ReadonlyArray<SourceQuestion> => {
  const questions = groups.flatMap((group) =>
    group.questions.map((question) => ({
      sourceDataset: dataset.id,
      groupId: group.id ?? "unknown-group",
      family: dataset.family,
      language: dataset.language,
      datasetModality: dataset.modality,
      groupMetadata: group.metadata,
      groupSource: group.source,
      question,
      loadImage: dataset.loadImage,
    })),
  );

  const sampledQuestions = sampleQuestionsByPercentage(questions, samplePercent);
  return limit > 0 ? sampledQuestions.slice(0, limit) : sampledQuestions;
};

const loadImagePart = Effect.fn("questionCategoryAnalysis.loadImagePart")(function* (
  sourceQuestion: SourceQuestion,
  rawUri: string,
) {
  // Some shititong URIs have a trailing " baked in from data ingestion
  const uri = rawUri.replace(/"+$/, "");
  if (uri.startsWith("http://") || uri.startsWith("https://")) {
    return { type: "image", image: new URL(uri) } satisfies ImagePart;
  }

  if (sourceQuestion.loadImage) {
    return { type: "image", image: yield* sourceQuestion.loadImage(uri) } satisfies ImagePart;
  }

  return yield* new UnsupportedLocalImageError({
    sourceDataset: sourceQuestion.sourceDataset,
    uri,
  });
});

const prepareImageReferences = Effect.fn("questionCategoryAnalysis.prepareImageReferences")(
  function* (sourceQuestion: SourceQuestion) {
    const references: Array<{ label: string; caption?: string; uri: string }> = [];

    for (const [index, image] of sourceQuestion.question.images.entries()) {
      references.push({
        label: `Question image ${index + 1}`,
        caption: image.caption,
        uri: image.uri,
      });
    }

    for (const option of sourceQuestion.question.options) {
      for (const [index, image] of option.images.entries()) {
        references.push({
          label: `Option ${option.id} image ${index + 1}`,
          caption: image.caption,
          uri: image.uri,
        });
      }
    }

    const validRefs = references.filter((ref) => isValidImageUri(sourceQuestion, ref.uri));
    const skippedCount = references.length - validRefs.length;
    if (skippedCount > 0) {
      const skippedUris = references
        .filter((ref) => !isValidImageUri(sourceQuestion, ref.uri))
        .map((ref) => ref.uri);
      yield* Effect.logWarning(
        `Skipping ${skippedCount} invalid image URI(s) for question ${sourceQuestion.question.id}`,
        { skippedUris },
      );
    }

    return yield* Effect.forEach(validRefs, (reference) =>
      loadImagePart(sourceQuestion, reference.uri).pipe(
        Effect.map(
          (part) =>
            ({
              label: reference.label,
              caption: reference.caption,
              part,
            }) satisfies PreparedImageReference,
        ),
      ),
    );
  },
);

const buildMultimodalMessages = Effect.fn("questionCategoryAnalysis.buildMultimodalMessages")(
  function* (sourceQuestion: SourceQuestion) {
    const promptText: TextPart = {
      type: "text",
      text: buildMultimodalQuestionPrompt(sourceQuestion),
    };
    const references = yield* prepareImageReferences(sourceQuestion);
    const content: Array<TextPart | ImagePart> = [promptText];

    for (const reference of references) {
      const captionSuffix = reference.caption ? ` | caption: ${reference.caption}` : "";
      content.push({
        type: "text",
        text: `${reference.label}${captionSuffix}`,
      });
      content.push(reference.part);
    }

    return {
      messages: [
        { role: "system", content: buildSystemPrompt("multimodal") },
        { role: "user", content },
      ] satisfies ModelMessage[],
    };
  },
);

const classifyTextBatch = Effect.fn("questionCategoryAnalysis.classifyTextBatch")(function* (
  batch: ReadonlyArray<SourceQuestion>,
) {
  const batchIds = batch.map((q) => q.question.id);
  yield* Effect.logDebug("Starting text batch classification", {
    batchSize: batch.length,
    dataset: batch[0]?.sourceDataset,
    questionIds: batchIds,
  });
  const { object: rawObject } = yield* generateObject({
    schema: textBatchClassificationJsonSchema,
    system: buildSystemPrompt("text_only"),
    prompt: buildTextBatchPrompt(batch),
    temperature: 0,
  });
  yield* Effect.logDebug("Text batch generateObject completed", {
    batchSize: batch.length,
    dataset: batch[0]?.sourceDataset,
  });
  const response = yield* S.decodeUnknown(TextBatchClassificationResponseSchema)(rawObject);

  const requestedIds = batch.map((question) => question.question.id);
  const resultCounts = new Map<string, number>();
  for (const result of response.results) {
    resultCounts.set(result.questionId, (resultCounts.get(result.questionId) ?? 0) + 1);
  }

  const duplicateQuestionIds = [...resultCounts.entries()]
    .filter(([, count]) => count > 1)
    .map(([questionId]) => questionId)
    .sort();
  const missingQuestionIds = requestedIds
    .filter((questionId) => !resultCounts.has(questionId))
    .sort();
  const extraQuestionIds = response.results
    .map((result) => result.questionId)
    .filter((questionId) => !requestedIds.includes(questionId))
    .sort();

  if (
    missingQuestionIds.length > 0 ||
    duplicateQuestionIds.length > 0 ||
    extraQuestionIds.length > 0
  ) {
    return yield* new InvalidBatchClassificationError({
      sourceDataset: batch[0]?.sourceDataset ?? "unknown-dataset",
      missingQuestionIds,
      duplicateQuestionIds,
      extraQuestionIds,
    });
  }

  const resultsById = new Map(
    response.results.map((result) => [
      result.questionId,
      normalizeClassification({
        primaryTopic: result.primaryTopic,
        secondaryFacets: result.secondaryFacets,
        reasoningType: result.reasoningType,
      }),
    ]),
  );

  return batch.map((sourceQuestion) => ({
    sourceDataset: sourceQuestion.sourceDataset,
    groupId: sourceQuestion.groupId,
    questionKey: `${sourceQuestion.sourceDataset}::${sourceQuestion.question.id}`,
    questionId: sourceQuestion.question.id,
    questionText: sourceQuestion.question.questionText,
    optionCount: sourceQuestion.question.options.length,
    hasImages: questionHasAnyImages(sourceQuestion.question),
    questionMetadata: sourceQuestion.question.metadata,
    questionSource: sourceQuestion.question.source,
    groupMetadata: sourceQuestion.groupMetadata,
    groupSource: sourceQuestion.groupSource,
    classificationMode: "text_only" as const,
    classification: resultsById.get(sourceQuestion.question.id)!,
  })) satisfies QuestionClassificationResult[];
});

const classifyMultimodalQuestion = Effect.fn("questionCategoryAnalysis.classifyMultimodalQuestion")(
  function* (sourceQuestion: SourceQuestion) {
    yield* Effect.logDebug("Starting multimodal classification", {
      questionId: sourceQuestion.question.id,
      dataset: sourceQuestion.sourceDataset,
    });
    const { messages } = yield* buildMultimodalMessages(sourceQuestion);
    yield* Effect.logDebug("Built multimodal messages, calling generateObject", {
      questionId: sourceQuestion.question.id,
    });
    const { object: rawObject } = yield* generateObject({
      messages,
      schema: multimodalQuestionClassificationJsonSchema,
      temperature: 0,
    });
    yield* Effect.logDebug("Multimodal generateObject completed", {
      questionId: sourceQuestion.question.id,
    });
    const response = yield* S.decodeUnknown(MultimodalQuestionClassificationResponseSchema)(
      rawObject,
    );

    if (response.questionId !== sourceQuestion.question.id) {
      return yield* new InvalidBatchClassificationError({
        sourceDataset: sourceQuestion.sourceDataset,
        missingQuestionIds:
          response.questionId === sourceQuestion.question.id ? [] : [sourceQuestion.question.id],
        duplicateQuestionIds: [],
        extraQuestionIds:
          response.questionId === sourceQuestion.question.id ? [] : [response.questionId],
      });
    }

    return {
      sourceDataset: sourceQuestion.sourceDataset,
      groupId: sourceQuestion.groupId,
      questionKey: `${sourceQuestion.sourceDataset}::${sourceQuestion.question.id}`,
      questionId: sourceQuestion.question.id,
      questionText: sourceQuestion.question.questionText,
      optionCount: sourceQuestion.question.options.length,
      hasImages: questionHasAnyImages(sourceQuestion.question),
      questionMetadata: sourceQuestion.question.metadata,
      questionSource: sourceQuestion.question.source,
      groupMetadata: sourceQuestion.groupMetadata,
      groupSource: sourceQuestion.groupSource,
      classificationMode: "multimodal" as const,
      classification: normalizeClassification({
        primaryTopic: response.primaryTopic,
        secondaryFacets: response.secondaryFacets,
        reasoningType: response.reasoningType,
      }),
    } satisfies QuestionClassificationResult;
  },
);

export const aggregateClassificationStats = (
  results: ReadonlyArray<QuestionClassificationResult>,
) => {
  const overall = {
    totalQuestions: results.length,
    byClassificationMode: toSortedCountEntries(
      countBy(results, (result) => result.classificationMode),
      results.length,
    ),
    byPrimaryTopic: toSortedCountEntries(
      countBy(results, (result) => result.classification.primaryTopic),
      results.length,
    ),
    byReasoningType: toSortedCountEntries(
      countBy(results, (result) => result.classification.reasoningType),
      results.length,
    ),
  };

  const byDataset = Object.fromEntries(
    [...new Set(results.map((result) => result.sourceDataset))].sort().map((datasetId) => {
      const datasetResults = results.filter((result) => result.sourceDataset === datasetId);
      return [
        datasetId,
        {
          totalQuestions: datasetResults.length,
          byClassificationMode: toSortedCountEntries(
            countBy(datasetResults, (result) => result.classificationMode),
            datasetResults.length,
          ),
          byPrimaryTopic: toSortedCountEntries(
            countBy(datasetResults, (result) => result.classification.primaryTopic),
            datasetResults.length,
          ),
          byReasoningType: toSortedCountEntries(
            countBy(datasetResults, (result) => result.classification.reasoningType),
            datasetResults.length,
          ),
        },
      ];
    }),
  );

  return { overall, byDataset };
};

export const buildClassificationAggregates = (
  results: ReadonlyArray<QuestionClassificationResult>,
) => {
  const datasetIds = [...new Set(results.map((result) => result.sourceDataset))].sort();
  const primaryTopicIds = TOPIC_CATEGORIES.map((category) => category.id);
  const reasoningCategoryIds = REASONING_CATEGORY_DEFINITIONS.map((category) => category.id);
  const classificationModeIds = CLASSIFICATION_MODE_DEFINITIONS.map((category) => category.id);
  const secondaryFacetIds = SECONDARY_FACETS.map((facet) => facet.id);

  const resultsByDataset = new Map(
    datasetIds.map((datasetId) => [
      datasetId,
      results.filter((result) => result.sourceDataset === datasetId),
    ]),
  );

  const datasetSummaries = datasetIds.map((datasetId) => {
    const datasetResults = resultsByDataset.get(datasetId) ?? [];
    return {
      datasetId,
      totalQuestions: datasetResults.length,
    };
  });

  const buildDistribution = (
    categoryIds: ReadonlyArray<string>,
    selector: (result: QuestionClassificationResult) => string,
  ) => ({
    overall: {
      totalQuestions: results.length,
      counts: toDenseCountEntries(categoryIds, countBy(results, selector), results.length),
    },
    byDataset: datasetIds.map((datasetId) => {
      const datasetResults = resultsByDataset.get(datasetId) ?? [];
      return {
        datasetId,
        totalQuestions: datasetResults.length,
        counts: toDenseCountEntries(
          categoryIds,
          countBy(datasetResults, selector),
          datasetResults.length,
        ),
      };
    }),
  });

  const countSecondaryFacetAssignments = (items: ReadonlyArray<QuestionClassificationResult>) => {
    const counts = new Map<string, number>();
    for (const item of items) {
      for (const facet of item.classification.secondaryFacets) {
        counts.set(facet, (counts.get(facet) ?? 0) + 1);
      }
    }
    return counts;
  };

  const buildSecondaryFacetUsage = (items: ReadonlyArray<QuestionClassificationResult>) => {
    const counts = countSecondaryFacetAssignments(items);
    const totalFacetAssignments = [...counts.values()].reduce((sum, count) => sum + count, 0);
    return {
      totalQuestions: items.length,
      totalFacetAssignments,
      counts: secondaryFacetIds.map((category) => {
        const count = counts.get(category) ?? 0;
        return {
          category,
          count,
          percentageOfQuestions: roundPercentage(count, items.length),
          percentageOfFacetAssignments: roundPercentage(count, totalFacetAssignments),
        };
      }),
    };
  };

  return {
    totals: {
      overallQuestions: results.length,
      byDataset: datasetSummaries,
    },
    distributions: {
      primaryTopic: buildDistribution(
        primaryTopicIds,
        (result) => result.classification.primaryTopic,
      ),
      reasoningType: buildDistribution(
        reasoningCategoryIds,
        (result) => result.classification.reasoningType,
      ),
      classificationMode: buildDistribution(
        classificationModeIds,
        (result) => result.classificationMode,
      ),
    },
    secondaryFacetUsage: {
      overall: buildSecondaryFacetUsage(results),
      byDataset: datasetIds.map((datasetId) => ({
        datasetId,
        ...buildSecondaryFacetUsage(resultsByDataset.get(datasetId) ?? []),
      })),
    },
  };
};

const buildOutput = (
  results: ReadonlyArray<QuestionClassificationResult>,
  stats: ReturnType<typeof aggregateClassificationStats>,
  aggregates: ReturnType<typeof buildClassificationAggregates>,
  config: {
    sourceDatasets: ReadonlyArray<string>;
    samplePercent: number;
    limitPerDataset: number;
    batchSize: number;
  },
) => ({
  metadata: {
    generatedAt: new Date().toISOString(),
    sourceDatasets: [...config.sourceDatasets],
    samplePercentPerDataset: config.samplePercent,
    limitPerDataset: config.limitPerDataset,
    batchSize: config.batchSize,
    topicCategories: TOPIC_CATEGORIES,
    secondaryFacets: SECONDARY_FACETS,
    textReasoningCategories: TEXT_REASONING_CATEGORIES,
    multimodalReasoningCategories: MULTIMODAL_REASONING_CATEGORIES,
    reasoningCategories: REASONING_CATEGORY_DEFINITIONS,
    classificationModes: CLASSIFICATION_MODE_DEFINITIONS,
  },
  stats,
  aggregates,
  classifications: [...results],
});

const buildTypstOutput = (output: ReturnType<typeof buildOutput>) => ({
  metadata: output.metadata,
  stats: output.stats,
  aggregates: output.aggregates,
});

const resolveOutputPath = (path: Option.Option<string>) =>
  Option.match(path, {
    onNone: () =>
      dataPath(
        "analysis",
        `question-category-analysis-${new Date()
          .toISOString()
          .replaceAll(":", "-")
          .replace(/\.\d+Z$/, "Z")}.json`,
      ),
    onSome: (providedPath) => (isAbsolute(providedPath) ? providedPath : repoPath(providedPath)),
  });

const typstOutputPath = (fullPath: string) =>
  fullPath.endsWith(".json")
    ? fullPath.replace(/\.json$/, "-typst.json")
    : `${fullPath}-typst.json`;

const saveOutput = (outputPath: string, output: ReturnType<typeof buildOutput>) =>
  writeJson(outputPath, output);

const saveCheckpoint = (outputPath: string, output: ReturnType<typeof buildOutput>) =>
  saveOutput(outputPath, output).pipe(
    Effect.zipRight(writeJson(typstOutputPath(outputPath), buildTypstOutput(output))),
  );

const loadCheckpoint = (path: string) =>
  Effect.try({
    try: () => JSON.parse(readFileSync(path, "utf8")) as ReturnType<typeof buildOutput>,
    catch: (cause) => new ReadCheckpointError({ path, cause }),
  });

export const validateResumeCheckpoint = (
  checkpointPath: string,
  checkpointMetadata: ReturnType<typeof buildOutput>["metadata"] | undefined,
  current: {
    sourceDatasets: readonly string[];
    samplePercentPerDataset: number;
    limitPerDataset: number;
    batchSize: number;
    topicCategoryIds: readonly string[];
    secondaryFacetIds: readonly string[];
    reasoningCategoryIds: readonly string[];
  },
) =>
  Effect.gen(function* () {
    if (!checkpointMetadata) {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} has no metadata; skipping resume compatibility checks`,
      );
      return;
    }

    const mismatches: string[] = [];

    if (Array.isArray(checkpointMetadata.sourceDatasets)) {
      const checkpointDatasets = normalizeStringArray(checkpointMetadata.sourceDatasets);
      const currentDatasets = normalizeStringArray(current.sourceDatasets);
      if (JSON.stringify(checkpointDatasets) !== JSON.stringify(currentDatasets)) {
        mismatches.push(
          `source datasets differ: checkpoint=${checkpointDatasets.join(", ")}, current=${currentDatasets.join(", ")}`,
        );
      }
    } else {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store sourceDatasets; skipping that compatibility check`,
      );
    }

    if (
      typeof checkpointMetadata.samplePercentPerDataset === "number" &&
      checkpointMetadata.samplePercentPerDataset !== current.samplePercentPerDataset
    ) {
      mismatches.push(
        `sample percent differs: checkpoint=${checkpointMetadata.samplePercentPerDataset}, current=${current.samplePercentPerDataset}`,
      );
    }

    if (
      typeof checkpointMetadata.limitPerDataset === "number" &&
      checkpointMetadata.limitPerDataset !== current.limitPerDataset
    ) {
      mismatches.push(
        `per-dataset limit differs: checkpoint=${checkpointMetadata.limitPerDataset}, current=${current.limitPerDataset}`,
      );
    }

    if (
      typeof checkpointMetadata.batchSize === "number" &&
      checkpointMetadata.batchSize !== current.batchSize
    ) {
      mismatches.push(
        `batch size differs: checkpoint=${checkpointMetadata.batchSize}, current=${current.batchSize}`,
      );
    }

    if (Array.isArray(checkpointMetadata.topicCategories)) {
      const checkpointTopicIds = checkpointMetadata.topicCategories.flatMap((category) =>
        typeof category?.id === "string" ? [category.id] : [],
      );
      if (JSON.stringify(checkpointTopicIds) !== JSON.stringify([...current.topicCategoryIds])) {
        mismatches.push("topic categories differ between checkpoint and current script");
      }
    }

    if (Array.isArray(checkpointMetadata.secondaryFacets)) {
      const checkpointFacetIds = checkpointMetadata.secondaryFacets.flatMap((facet) =>
        typeof facet?.id === "string" ? [facet.id] : [],
      );
      if (JSON.stringify(checkpointFacetIds) !== JSON.stringify([...current.secondaryFacetIds])) {
        mismatches.push("secondary facets differ between checkpoint and current script");
      }
    }

    if (Array.isArray(checkpointMetadata.reasoningCategories)) {
      const checkpointReasoningIds = checkpointMetadata.reasoningCategories.flatMap((category) =>
        typeof category?.id === "string" ? [category.id] : [],
      );
      if (
        JSON.stringify(checkpointReasoningIds) !== JSON.stringify([...current.reasoningCategoryIds])
      ) {
        mismatches.push("reasoning categories differ between checkpoint and current script");
      }
    } else {
      yield* Effect.logWarning(
        `Resume checkpoint ${checkpointPath} does not store reasoningCategories; skipping that compatibility check`,
      );
    }

    if (mismatches.length > 0) {
      return yield* new ResumeCheckpointMismatchError({
        path: checkpointPath,
        mismatches,
      });
    }
  });

const writeJson = Effect.fn("questionCategoryAnalysis.writeJson")(function* (
  path: string,
  data: unknown,
) {
  yield* Effect.tryPromise({
    try: () => mkdir(dirname(path), { recursive: true }),
    catch: (cause) =>
      new WriteOutputError({
        path,
        cause,
      }),
  });

  yield* Effect.tryPromise({
    try: () => writeFile(path, `${JSON.stringify(data, null, 2)}\n`, "utf-8"),
    catch: (cause) =>
      new WriteOutputError({
        path,
        cause,
      }),
  });
});

// ---------------------------------------------------------------------------
// CLI program
// ---------------------------------------------------------------------------

const datasetOption = Options.text("dataset").pipe(
  Options.repeated,
  Options.withDescription(
    `Dataset ID to analyze. Repeatable. Leave empty to analyze all datasets (${DATASET_IDS.join(", ")}).`,
  ),
);

const limitOption = Options.integer("limit").pipe(
  Options.withDefault(0),
  Options.withDescription("Maximum number of questions to classify per dataset. 0 means no limit."),
);

const samplePercentOption = Options.text("sample-percent").pipe(
  Options.optional,
  Options.withDescription(
    "Deterministically sample this percentage of each dataset before classification, e.g. 2 or 2.5. Defaults to 100 when omitted.",
  ),
);

const batchSizeOption = Options.integer("batch-size").pipe(
  Options.withDefault(1),
  Options.withDescription("Number of questions to send to the classifier in each LLM call."),
);

const concurrencyOption = Options.integer("concurrency").pipe(
  Options.withDefault(4),
  Options.withDescription("How many classification batches to run concurrently."),
);

const outputOption = Options.text("output").pipe(
  Options.optional,
  Options.withDescription(
    "Output path, relative to the repo root unless absolute. Defaults to data/analysis/...",
  ),
);

const resumeOption = Options.text("resume").pipe(
  Options.optional,
  Options.withDescription("Resume from an existing full analysis JSON file and continue in-place."),
);

const dryRunOption = Options.boolean("dry-run").pipe(
  Options.withDescription("Only resolve datasets and counts; do not call the classifier."),
);

const runClassification = Effect.fn("questionCategoryAnalysis.runClassification")(
  function* (config: {
    dataset: ReadonlyArray<string>;
    limit: number;
    samplePercent: Option.Option<string>;
    batchSize: number;
    concurrency: number;
    output: Option.Option<string>;
    resume: Option.Option<string>;
    dryRun: boolean;
  }) {
    const resolvedSamplePercent = yield* parseSamplePercent(config.samplePercent);
    const selectedDatasets = yield* selectDatasets(config.dataset);
    const loadedDatasets = yield* Effect.forEach(selectedDatasets, (dataset) =>
      dataset.loadGroups().pipe(
        Effect.map((groups) => ({
          dataset,
          groups,
        })),
      ),
    );

    const selectedQuestionsByDataset = loadedDatasets.map(({ dataset, groups }) => ({
      dataset,
      groups,
      selectedQuestions: flattenQuestionGroups(
        dataset,
        groups,
        resolvedSamplePercent,
        config.limit,
      ),
    }));

    const sourceQuestions = selectedQuestionsByDataset.flatMap(
      ({ selectedQuestions }) => selectedQuestions,
    );
    const sourceQuestionByKey = new Map(
      sourceQuestions.map((question) => [
        questionKeyOf(question.sourceDataset, question.question.id),
        question,
      ]),
    );

    yield* Effect.log(
      `Selected ${selectedDatasets.length} dataset(s): ${selectedDatasets.map((dataset) => dataset.id).join(", ")}`,
    );
    if (resolvedSamplePercent > 0) {
      yield* Effect.log(
        `Using deterministic per-dataset sampling at ${resolvedSamplePercent}% before any per-dataset limit`,
      );
    }
    for (const { dataset, groups, selectedQuestions } of selectedQuestionsByDataset) {
      const totalQuestions = groups.reduce((sum, group) => sum + group.questions.length, 0);
      yield* Effect.log(`  ${dataset.id}: ${selectedQuestions.length}/${totalQuestions} questions`);
    }

    if (config.dryRun) {
      if (Option.match(config.resume, { onNone: () => false, onSome: () => true })) {
        yield* Effect.log("Ignoring --resume in dry-run mode");
      }
      if (Option.match(config.output, { onNone: () => false, onSome: () => true })) {
        yield* Effect.log("Ignoring --output in dry-run mode");
      }
      yield* Effect.log(
        `Dry run complete. ${sourceQuestions.length} question(s) would be classified.`,
      );
      return;
    }

    const resumePath = Option.match(config.resume, {
      onNone: () => undefined,
      onSome: (path) => path,
    });
    const explicitOutputPath = Option.match(config.output, {
      onNone: () => undefined,
      onSome: (path) => path,
    });

    if (resumePath && explicitOutputPath && resumePath !== explicitOutputPath) {
      yield* Effect.log(
        `Ignoring --output because --resume was provided; resuming in-place at ${resumePath}`,
      );
    }
    if (resumePath && !existsSync(resumePath)) {
      return yield* new ReadCheckpointError({
        path: resumePath,
        cause: new Error(`Checkpoint file does not exist: ${resumePath}`),
      });
    }

    const outputPath = resumePath ?? resolveOutputPath(config.output);
    const checkpoint = resumePath ? yield* loadCheckpoint(resumePath) : null;

    if (resumePath) {
      yield* validateResumeCheckpoint(resumePath, checkpoint?.metadata, {
        sourceDatasets: selectedDatasets.map((dataset) => dataset.id),
        samplePercentPerDataset: resolvedSamplePercent,
        limitPerDataset: config.limit,
        batchSize: config.batchSize,
        topicCategoryIds: TOPIC_CATEGORIES.map((category) => category.id),
        secondaryFacetIds: SECONDARY_FACETS.map((facet) => facet.id),
        reasoningCategoryIds: REASONING_CATEGORY_DEFINITIONS.map((category) => category.id),
      });
    }

    const checkpointClassifications = Array.isArray(checkpoint?.classifications)
      ? checkpoint.classifications
      : [];
    const resumedResults: QuestionClassificationResult[] = checkpointClassifications.flatMap(
      (entry) => {
        if (!isRecord(entry)) return [];
        const sourceDataset =
          typeof entry.sourceDataset === "string" ? entry.sourceDataset : undefined;
        const questionId = typeof entry.questionId === "string" ? entry.questionId : undefined;
        if (!sourceDataset || !questionId) return [];
        const questionKey =
          typeof entry.questionKey === "string"
            ? entry.questionKey
            : questionKeyOf(sourceDataset, questionId);
        const sourceQuestion = sourceQuestionByKey.get(questionKey);
        return [
          {
            ...entry,
            sourceDataset,
            groupId:
              typeof entry.groupId === "string"
                ? entry.groupId
                : (sourceQuestion?.groupId ?? "unknown-group"),
            questionKey,
            questionId,
            questionText:
              typeof entry.questionText === "string"
                ? entry.questionText
                : (sourceQuestion?.question.questionText ?? ""),
            optionCount:
              typeof entry.optionCount === "number"
                ? entry.optionCount
                : (sourceQuestion?.question.options.length ?? 0),
            hasImages:
              typeof entry.hasImages === "boolean"
                ? entry.hasImages
                : sourceQuestion
                  ? questionHasAnyImages(sourceQuestion.question)
                  : false,
            questionMetadata:
              "questionMetadata" in entry
                ? entry.questionMetadata
                : sourceQuestion?.question.metadata,
            questionSource:
              isRecord(entry.questionSource) &&
              Object.values(entry.questionSource).every((v) => typeof v === "string")
                ? (entry.questionSource as Record<string, string>)
                : sourceQuestion?.question.source,
            groupMetadata:
              "groupMetadata" in entry ? entry.groupMetadata : sourceQuestion?.groupMetadata,
            groupSource:
              isRecord(entry.groupSource) &&
              Object.values(entry.groupSource).every((v) => typeof v === "string")
                ? (entry.groupSource as Record<string, string>)
                : sourceQuestion?.groupSource,
          } satisfies QuestionClassificationResult,
        ];
      },
    );

    const completedQuestionKeys = new Set(resumedResults.map((result) => result.questionKey));
    const pendingSourceQuestions = sourceQuestions.filter(
      (question) =>
        !completedQuestionKeys.has(questionKeyOf(question.sourceDataset, question.question.id)),
    );

    if (checkpoint) {
      yield* Effect.log(
        `Resuming from ${resumePath}: ${resumedResults.length} completed classifications restored`,
      );
    }

    if (sourceQuestions.length === 0) {
      yield* Effect.log("No questions selected. Nothing to do.");
      return;
    }

    if (config.dataset.length === 0 && config.limit === 0 && resolvedSamplePercent === 100) {
      yield* Effect.logWarning(
        "No dataset filter or per-dataset limit was provided. This will classify every question in the benchmark suite.",
      );
    }

    const resultOrder = new Map(
      sourceQuestions.map((question, index) => [
        questionKeyOf(question.sourceDataset, question.question.id),
        index,
      ]),
    );
    const sortResultsForOutput = (resultsToSort: ReadonlyArray<QuestionClassificationResult>) =>
      [...resultsToSort].sort(
        (left, right) =>
          (resultOrder.get(left.questionKey) ?? Number.MAX_SAFE_INTEGER) -
          (resultOrder.get(right.questionKey) ?? Number.MAX_SAFE_INTEGER),
      );
    const buildOutputFor = (resultsForOutput: ReadonlyArray<QuestionClassificationResult>) => {
      const sortedResults = sortResultsForOutput(resultsForOutput);
      return buildOutput(
        sortedResults,
        aggregateClassificationStats(sortedResults),
        buildClassificationAggregates(sortedResults),
        {
          sourceDatasets: selectedDatasets.map((dataset) => dataset.id),
          samplePercent: resolvedSamplePercent,
          limitPerDataset: config.limit,
          batchSize: config.batchSize,
        },
      );
    };

    const checkpointSaveSemaphore = yield* Effect.makeSemaphore(1);
    const collectedResults: QuestionClassificationResult[] = [...resumedResults];
    let completedCount = resumedResults.length;
    let lastSavedAt = resumedResults.length;
    const totalPlannedCount = sourceQuestions.length;
    const appendResults = (newResults: ReadonlyArray<QuestionClassificationResult>) =>
      checkpointSaveSemaphore.withPermits(1)(
        Effect.gen(function* () {
          collectedResults.push(...newResults);
          for (const result of newResults) {
            completedQuestionKeys.add(result.questionKey);
          }
          completedCount += newResults.length;
          if (completedCount - lastSavedAt >= SAVE_EVERY) {
            lastSavedAt = completedCount;
            const checkpointOutput = buildOutputFor(collectedResults);
            yield* saveCheckpoint(outputPath, checkpointOutput).pipe(
              Effect.tapError((error) =>
                Effect.logWarning(`Incremental save failed: ${String(error)}`),
              ),
              Effect.catchAll(() => Effect.void),
            );
            yield* Effect.log(`Incremental save (${completedCount}/${totalPlannedCount})`);
          }
        }),
      );

    const textOnlyQuestions = pendingSourceQuestions.filter(
      (question) => getClassificationMode(question) === "text_only",
    );
    const multimodalQuestions = pendingSourceQuestions.filter(
      (question) => getClassificationMode(question) === "multimodal",
    );
    const textBatches = chunkArray(textOnlyQuestions, config.batchSize);
    yield* Effect.log(
      `Classifying ${pendingSourceQuestions.length} remaining question(s) with concurrency=${config.concurrency}`,
    );
    if (resumedResults.length > 0) {
      yield* Effect.log(
        `  Resumed: ${resumedResults.length} previously completed classification(s)`,
      );
    }
    if (textOnlyQuestions.length > 0) {
      yield* Effect.log(
        `  Text-only: ${textOnlyQuestions.length} question(s) in ${textBatches.length} batch(es)`,
      );
    }
    if (multimodalQuestions.length > 0) {
      yield* Effect.log(
        `  Multimodal: ${multimodalQuestions.length} question(s) as single-question image calls`,
      );
    }
    const { textResults, multimodalResults } = yield* Progress.all(
      {
        textResults: Progress.task(
          Effect.gen(function* () {
            if (textBatches.length === 0) {
              return [] as QuestionClassificationResult[];
            }

            return yield* Progress.forEach(
              textBatches,
              (batch) =>
                classifyTextBatch(batch).pipe(
                  Effect.timeoutFail({
                    duration: Duration.minutes(10),
                    onTimeout: () =>
                      new ClassificationTimeoutError({
                        questionId: batch.map((q) => q.question.id).join(", "),
                        duration: "10 minutes",
                      }),
                  }),
                  Effect.tapError((error) =>
                    Effect.logDebug("Text batch classification failed, may retry", {
                      error,
                      dataset: batch[0]?.sourceDataset,
                      questionIds: batch.map((q) => q.question.id),
                    }),
                  ),
                  Effect.retry({
                    schedule: Schedule.exponential("1 second").pipe(
                      Schedule.delayed((delay) => Duration.min(delay, Duration.minutes(2))),
                      Schedule.intersect(Schedule.recurs(9)),
                    ),
                    while: (error) => !isUnrecoverableError(error),
                  }),
                  Effect.tap((batchResults) => appendResults(batchResults)),
                  Effect.catchAll((error) =>
                    Effect.logWarning(`Skipping text batch (${batch.length} questions)`, {
                      error,
                      dataset: batch[0]?.sourceDataset,
                    }).pipe(Effect.as([] as QuestionClassificationResult[])),
                  ),
                ),
              {
                description: "Classifying text-only batches",
                concurrency: config.concurrency,
              },
            ).pipe(Effect.map((nestedResults) => nestedResults.flat()));
          }).pipe(Effect.provide(EvalLanguageModelLayer)),
          {
            description: "Text-only classification",
            total: textOnlyQuestions.length,
            countDisplay: "processedOnly",
          },
        ),
        multimodalResults: Progress.task(
          Effect.gen(function* () {
            if (multimodalQuestions.length === 0) {
              return [] as QuestionClassificationResult[];
            }

            const rawResults = yield* Progress.forEach(
              multimodalQuestions,
              (question) =>
                classifyMultimodalQuestion(question).pipe(
                  Effect.timeoutFail({
                    duration: Duration.minutes(10),
                    onTimeout: () =>
                      new ClassificationTimeoutError({
                        questionId: question.question.id,
                        duration: "10 minutes",
                      }),
                  }),
                  Effect.tapError((error) =>
                    Effect.logDebug("Multimodal classification failed, may retry", {
                      error,
                      questionId: question.question.id,
                      dataset: question.sourceDataset,
                    }),
                  ),
                  Effect.retry({
                    schedule: Schedule.exponential("1 second").pipe(
                      Schedule.delayed((delay) => Duration.min(delay, Duration.minutes(2))),
                      Schedule.intersect(Schedule.recurs(9)),
                    ),
                    while: (error) => !isUnrecoverableError(error),
                  }),
                  Effect.tap((result) => appendResults([result])),
                  Effect.map(Option.some),
                  Effect.catchAll((error) =>
                    Effect.logWarning(`Skipping multimodal question ${question.question.id}`, {
                      error,
                      dataset: question.sourceDataset,
                    }).pipe(Effect.as(Option.none())),
                  ),
                ),
              {
                description: "Classifying multimodal questions",
                concurrency: config.concurrency,
              },
            );
            return rawResults.flatMap(Option.toArray);
          }).pipe(Effect.provide(EvalMultimodalLanguageModelLayer)),
          {
            description: "Multimodal classification",
            total: multimodalQuestions.length,
            countDisplay: "processedOnly",
          },
        ),
      },
      {
        description: "Classifying questions",
        concurrency: 1,
      },
    );

    const results = sortResultsForOutput([...resumedResults, ...textResults, ...multimodalResults]);
    const output = buildOutputFor(results);

    yield* saveCheckpoint(outputPath, output);
    yield* Effect.log(`Wrote classification report to ${outputPath}`);
    yield* Effect.log(`Wrote Typst export to ${typstOutputPath(outputPath)}`);
  },
);

export const command = Command.make(
  "classify-question-categories",
  {
    dataset: datasetOption,
    limit: limitOption,
    samplePercent: samplePercentOption,
    batchSize: batchSizeOption,
    concurrency: concurrencyOption,
    output: outputOption,
    resume: resumeOption,
    dryRun: dryRunOption,
  },
  runClassification,
).pipe(
  Command.withDescription(
    "Classify benchmark questions into configurable topic and reasoning categories.",
  ),
);

export const cli = Command.run(command, {
  name: "Question Category Analysis",
  version: "v0.1.0",
});

const runtimeLayer = Layer.mergeAll(NodeContext.layer, Logger.pretty);

if (import.meta.main) {
  cli(process.argv).pipe(
    Logger.withMinimumLogLevel(LogLevel.Debug),
    Effect.provide(runtimeLayer),
    Effect.catchAllCause((cause) => Effect.sync(() => console.error(Cause.pretty(cause)))),
    NodeRuntime.runMain,
  );
}
