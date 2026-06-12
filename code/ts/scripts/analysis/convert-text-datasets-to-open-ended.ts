// Usage:
//   bun scripts/analysis/convert-text-datasets-to-open-ended.ts
//   bun scripts/analysis/convert-text-datasets-to-open-ended.ts --dataset crewcn --dataset pei2024-uk
//   bun scripts/analysis/convert-text-datasets-to-open-ended.ts --limit 100 --concurrency 8
//   bun scripts/analysis/convert-text-datasets-to-open-ended.ts --resume
//   bun scripts/analysis/convert-text-datasets-to-open-ended.ts --dry-run
//   LANGUAGE_MODEL_ENDPOINTS='{"model":"Qwen/Qwen3.5-122B-A10B-FP8","endpoints":[{"provider":"litellm","maxConcurrency":2},{"provider":"vllm","port":8000,"maxConcurrency":25},{"provider":"vllm","port":8001,"maxConcurrency":25}]}' bun scripts/analysis/convert-text-datasets-to-open-ended.ts --resume --concurrency 52
//
// Environment variables (same as eval scripts):
//   LANGUAGE_MODEL_ENDPOINTS — JSON router config. When set, this script uses the queued model router via EvalLanguageModelLayer.
//   EVAL_PROVIDER  — "litellm" (default) or "vllm"
//   EVAL_MODEL     — explicit model ID. In router mode this is optional when LANGUAGE_MODEL_ENDPOINTS.model is set.
//   VLLM_BASE_URL / VLLM_PORT — vLLM endpoint config
//
// Rewrites local text-only MCQ eval datasets into open-ended datasets for later
// judge-based evaluation. The converter writes one dataset JSON and one resumable
// checkpoint/report JSON per source dataset.
import { existsSync, readFileSync } from "node:fs";
import { mkdir, readdir, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute } from "node:path";
import { jsonSchema } from "ai";
import { Command, Options } from "@effect/cli";
import { NodeContext, NodeRuntime } from "@effect/platform-node";
import * as Progress from "effective-progress";
import { Cause, Effect, Layer, Logger, Option, Schema } from "effect";
import * as JSONSchema from "effect/JSONSchema";
import * as S from "effect/Schema";
import {
  loadQuestionGroups as loadCrewQuestionGroups,
  CREWCN_DATASET_NAME,
} from "../../evals/crew/crew";
import { taskRetryPolicy } from "../../evals/experiment_setup";
import {
  OpenEndedQuestionGroups,
  type OpenEndedEvalQuestion,
  type OpenEndedQuestionGroup,
} from "../../evals/open-ended-question-schema";
import {
  loadUkQuestionGroups,
  loadZhQuestionGroups,
  PEI2024_UK_DATASET_NAME,
  PEI2024_ZH_DATASET_NAME,
} from "../../evals/pei2024/pei2024";
import { type EvalQuestion, type EvalQuestionGroups } from "../../evals/question-schema";
import {
  loadQuestionGroups as loadRaynorQuestionGroups,
  RAYNOR_DATASET_NAME_V2,
} from "../../evals/raynor/raynor";
import {
  loadEnTextGroups,
  loadZhTextGroups,
  SHITITONG_EN_TEXT_DATASET_NAME,
  SHITITONG_ZH_TEXT_DATASET_NAME,
} from "../../evals/shititong/shititong";
import {
  loadTextOnlyQuestionGroups as loadUsCoastGuardTextOnlyQuestionGroups,
  US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
} from "../../evals/us_coast_guard/us_coast_guard";
import {
  EvalLanguageModelLayer,
  generateObject,
  LanguageModel,
} from "../../src/services/LanguageModel";
import { repoPath } from "../../src/utils/repo";

type DatasetDefinition = {
  id: string;
  family: string;
  language: "EN" | "ZH";
  loadGroups: () => Effect.Effect<EvalQuestionGroups, unknown>;
};

export const TEXT_DATASET_DEFINITIONS: ReadonlyArray<DatasetDefinition> = [
  {
    id: PEI2024_UK_DATASET_NAME,
    family: "PEI2024",
    language: "EN",
    loadGroups: loadUkQuestionGroups,
  },
  {
    id: PEI2024_ZH_DATASET_NAME,
    family: "PEI2024",
    language: "ZH",
    loadGroups: loadZhQuestionGroups,
  },
  {
    id: US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
    family: "US Coast Guard",
    language: "EN",
    loadGroups: loadUsCoastGuardTextOnlyQuestionGroups,
  },
  {
    id: RAYNOR_DATASET_NAME_V2,
    family: "Raynor",
    language: "EN",
    loadGroups: loadRaynorQuestionGroups,
  },
  {
    id: CREWCN_DATASET_NAME,
    family: "CrewCN",
    language: "ZH",
    loadGroups: loadCrewQuestionGroups,
  },
  {
    id: SHITITONG_EN_TEXT_DATASET_NAME,
    family: "Shititong",
    language: "EN",
    loadGroups: loadEnTextGroups,
  },
  {
    id: SHITITONG_ZH_TEXT_DATASET_NAME,
    family: "Shititong",
    language: "ZH",
    loadGroups: loadZhTextGroups,
  },
] as const;

export const TEXT_DATASET_IDS = TEXT_DATASET_DEFINITIONS.map((dataset) => dataset.id);
export const SAVE_EVERY = 10;

type SelectedQuestionGroup = {
  id?: string;
  metadata?: unknown;
  source?: Record<string, string>;
  questions: ReadonlyArray<EvalQuestion>;
};

type SourceQuestion = {
  sourceDataset: string;
  groupId: string;
  groupMetadata?: unknown;
  groupSource?: Record<string, string>;
  question: EvalQuestion;
};

type SelectedDataset = {
  dataset: DatasetDefinition;
  groups: ReadonlyArray<SelectedQuestionGroup>;
  questions: ReadonlyArray<SourceQuestion>;
};

type ConversionCandidate = {
  sourceDataset: string;
  groupId: string;
  questionId: string;
  questionKey: string;
  sourceQuestionText: string;
  questionMetadata: unknown;
  questionSource?: Record<string, string>;
  groupMetadata?: unknown;
  images: EvalQuestion["images"];
  correctAnswers: readonly [string, ...string[]];
  incorrectAnswers: readonly string[];
};

type SourceExtractionFailureReason =
  | "missing_question_text"
  | "missing_options"
  | "missing_correct_option_ids"
  | "correct_option_id_not_found"
  | "unresolved_reference_option";

type SourceExtractionFailure = {
  sourceDataset: string;
  groupId: string;
  questionId: string;
  questionKey: string;
  sourceQuestionText: string;
  reason: SourceExtractionFailureReason;
  detail: string;
};

type SourceExtractionResult =
  | { ok: true; value: ConversionCandidate }
  | { ok: false; failure: SourceExtractionFailure };

type RewriteDecision = {
  convertible: boolean;
  reason: string;
  openEndedQuestion?: string;
};

const rewriteDecisionSchema = S.Struct({
  convertible: S.Boolean,
  reason: S.String,
  openEndedQuestion: S.optional(S.String),
});

const rewriteDecisionJsonSchema = jsonSchema<RewriteDecision>(
  JSONSchema.make(rewriteDecisionSchema),
);

export type SkipReason =
  | "invalid_source_question"
  | "non_convertible"
  | "missing_rewrite"
  | "answer_leakage"
  | "rewrite_error";

type EvalModelIdentity = {
  modelId: string;
  provider?: string;
};

export type ConvertedEntry = {
  status: "converted";
  questionKey: string;
  sourceDataset: string;
  groupId: string;
  questionId: string;
  sourceQuestionText: string;
  conversionReason: string;
  convertedQuestion: OpenEndedEvalQuestion;
};

export type SkippedEntry = {
  status: "skipped";
  questionKey: string;
  sourceDataset: string;
  groupId: string;
  questionId: string;
  sourceQuestionText: string;
  reason: SkipReason;
  detail: string;
};

export type ProcessedEntry = ConvertedEntry | SkippedEntry;

export type CheckpointMetadata = {
  generatedAt: string;
  runTimestamp: string;
  sourceDataset: string;
  limitPerDataset: number;
  concurrency: number;
  saveEvery: number;
  evalModel: EvalModelIdentity;
};

export type ConversionCheckpoint = {
  metadata: CheckpointMetadata;
  counts: {
    totalSelected: number;
    processed: number;
    converted: number;
    skipped: number;
  };
  skippedByReason: Record<string, number>;
  entries: ProcessedEntry[];
};

export type ConversionAnalysisSummary = {
  metadata: CheckpointMetadata;
  counts: {
    totalSelected: number;
    processed: number;
    converted: number;
    skipped: number;
  };
  skippedByReason: Record<string, number>;
};

type RunConfig = {
  dataset: ReadonlyArray<string>;
  limit: number;
  concurrency: number;
  outputDir: Option.Option<string>;
  reportDir: Option.Option<string>;
  resume: boolean;
  dryRun: boolean;
};

export type CliArgs = RunConfig;

export class UnknownDatasetError extends Schema.TaggedError<UnknownDatasetError>()(
  "UnknownDatasetError",
  {
    requestedDatasets: Schema.Array(Schema.String),
    availableDatasets: Schema.Array(Schema.String),
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

export class ResumeStateNotFoundError extends Schema.TaggedError<ResumeStateNotFoundError>()(
  "ResumeStateNotFoundError",
  {
    reportDir: Schema.String,
    sourceDataset: Schema.String,
  },
) {}

export class ReadReportDirectoryError extends Schema.TaggedError<ReadReportDirectoryError>()(
  "ReadReportDirectoryError",
  {
    path: Schema.String,
    cause: Schema.Defect,
  },
) {}

const REWRITE_SYSTEM_PROMPT = `You are rewriting maritime multiple-choice questions into open-ended questions for evaluation.

Critical goal:
- The rewritten question must test the model's inherent knowledge, not answer recognition from provided options.

Rules:
1) Rewrite into a single open-ended question with no options, no answer letters, and no candidate lists.
2) Do not reveal or closely paraphrase the original correct answer wording.
3) Keep enough concrete context so grading can be done reliably by an LLM judge.
4) If the original MCQ cannot be converted cleanly without ambiguity or answer leakage, set convertible=false.
5) Prefer minimal rewriting beyond what is needed to remove the multiple-choice framing.

Return a structured object with:
- convertible: boolean
- reason: string
- openEndedQuestion?: string`;

const normalizeOptionId = (id: string): string => id.trim().toUpperCase();

const romanNumerals = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ"] as const;

const asciiRomanToUnicode = (text: string): string =>
  text
    .replace(/\bVIII\b/g, "Ⅷ")
    .replace(/\bVII\b/g, "Ⅶ")
    .replace(/\bVI\b/g, "Ⅵ")
    .replace(/\bIV\b/g, "Ⅳ")
    .replace(/\bIII\b/g, "Ⅲ")
    .replace(/\bII\b/g, "Ⅱ")
    .replace(/\bIX\b/g, "Ⅸ")
    .replace(/\bX\b/g, "Ⅹ")
    .replace(/\bV\b/g, "Ⅴ")
    .replace(/\bI\b/g, "Ⅰ");

const normalizeRomanSelectorText = (text: string): string =>
  asciiRomanToUnicode(text)
    .replace(/[，,]/g, "、")
    .replace(/[～~－—-]/g, "-")
    .replace(/\s+/g, "")
    .trim();

const isRomanSelectorText = (text: string): boolean => {
  const normalized = normalizeRomanSelectorText(text);
  return normalized.length > 0 && /^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ、-]+$/u.test(normalized);
};

const expandRomanRange = (rangeText: string): readonly string[] | undefined => {
  const [start, end, ...rest] = rangeText.split("-");
  if (!start || !end || rest.length > 0) return undefined;

  const startIndex = romanNumerals.indexOf(start as (typeof romanNumerals)[number]);
  const endIndex = romanNumerals.indexOf(end as (typeof romanNumerals)[number]);
  if (startIndex < 0 || endIndex < 0 || startIndex > endIndex) return undefined;

  return romanNumerals.slice(startIndex, endIndex + 1);
};

const parseRomanSelectorLabels = (text: string): readonly string[] | undefined => {
  if (!isRomanSelectorText(text)) return undefined;

  const labels: string[] = [];
  for (const part of normalizeRomanSelectorText(text).split("、").filter(Boolean)) {
    if (part.includes("-")) {
      const rangeLabels = expandRomanRange(part);
      if (!rangeLabels) return undefined;
      labels.push(...rangeLabels);
    } else {
      labels.push(part);
    }
  }

  return labels.length > 0 ? labels : undefined;
};

const romanStatementMarker = /([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|VIII|VII|VI|IV|III|II|IX|X|V|I)\s*[、,，．.]?\s*/g;

const extractRomanStatementMap = (questionText: string): ReadonlyMap<string, string> => {
  const matches: Array<{ label: string; markerStart: number; contentStart: number }> = [];

  for (const match of questionText.matchAll(romanStatementMarker)) {
    const label = normalizeRomanSelectorText(match[1] ?? "");
    if (!romanNumerals.includes(label as (typeof romanNumerals)[number])) continue;

    matches.push({
      label,
      markerStart: match.index ?? 0,
      contentStart: (match.index ?? 0) + match[0].length,
    });
  }

  return new Map(
    matches.flatMap((match, index) => {
      const next = matches[index + 1];
      const statement = questionText
        .slice(match.contentStart, next ? next.markerStart : questionText.length)
        .trim()
        .replace(/[;；。,.，、\s]+$/u, "")
        .trim();

      return statement.length > 0 ? [[match.label, statement] as const] : [];
    }),
  );
};

const resolveReferenceOptionText = (
  optionText: string,
  romanStatements: ReadonlyMap<string, string>,
): string | undefined => {
  const selectorLabels = parseRomanSelectorLabels(optionText);
  if (!selectorLabels) return optionText;

  const statements = selectorLabels.map((label) => romanStatements.get(label));
  if (statements.some((statement) => !statement)) return undefined;

  return statements.join("; ");
};

const normalizeLeakText = (text: string): string =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

const toSignificantTokens = (text: string): string[] => {
  const stopwords = new Set([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
  ]);

  return normalizeLeakText(text)
    .split(" ")
    .filter((token) => token.length >= 3 && !stopwords.has(token));
};

export const hasAnswerLeakage = (openEndedQuestion: string, referenceAnswer: string): boolean => {
  const normalizedQuestion = normalizeLeakText(openEndedQuestion);
  const normalizedAnswer = normalizeLeakText(referenceAnswer);
  if (!normalizedQuestion || !normalizedAnswer) return false;

  if (normalizedAnswer.length >= 8 && normalizedQuestion.includes(normalizedAnswer)) {
    return true;
  }

  const answerTokens = toSignificantTokens(referenceAnswer);
  if (answerTokens.length < 3) return false;

  const questionTokens = new Set(toSignificantTokens(openEndedQuestion));
  const overlap = answerTokens.filter((token) => questionTokens.has(token)).length;
  return overlap / answerTokens.length >= 0.8;
};

export const questionKeyOf = (sourceDataset: string, questionId: string) =>
  `${sourceDataset}::${questionId}`;

const countSkippedByReason = (entries: ReadonlyArray<ProcessedEntry>): Record<string, number> => {
  const counts = new Map<string, number>();
  for (const entry of entries) {
    if (entry.status !== "skipped") continue;
    counts.set(entry.reason, (counts.get(entry.reason) ?? 0) + 1);
  }

  return Object.fromEntries(
    [...counts.entries()].sort((left, right) => left[0].localeCompare(right[0])),
  );
};

export const selectDatasets = (requestedDatasets: ReadonlyArray<string>) =>
  Effect.gen(function* () {
    if (requestedDatasets.length === 0) {
      return [...TEXT_DATASET_DEFINITIONS];
    }

    const requestedSet = new Set(requestedDatasets);
    const selected = TEXT_DATASET_DEFINITIONS.filter((dataset) => requestedSet.has(dataset.id));
    const missing = requestedDatasets.filter(
      (datasetId, index, allIds) =>
        allIds.indexOf(datasetId) === index &&
        !selected.some((dataset) => dataset.id === datasetId),
    );

    if (missing.length > 0) {
      return yield* new UnknownDatasetError({
        requestedDatasets: missing,
        availableDatasets: [...TEXT_DATASET_IDS],
      });
    }

    return selected;
  });

export const selectQuestionSubset = (
  dataset: DatasetDefinition,
  groups: EvalQuestionGroups,
  limitPerDataset: number,
): SelectedDataset => {
  const selectedGroups: SelectedQuestionGroup[] = [];
  const selectedQuestions: SourceQuestion[] = [];
  let selectedCount = 0;

  for (const group of groups) {
    if (limitPerDataset > 0 && selectedCount >= limitPerDataset) break;

    const remaining =
      limitPerDataset > 0 ? Math.max(limitPerDataset - selectedCount, 0) : group.questions.length;
    const chosenQuestions = group.questions.slice(0, remaining);
    if (chosenQuestions.length === 0) continue;

    const groupId = group.id ?? "unknown-group";
    selectedGroups.push({
      id: group.id,
      metadata: group.metadata,
      source: group.source,
      questions: chosenQuestions,
    });

    for (const question of chosenQuestions) {
      selectedQuestions.push({
        sourceDataset: dataset.id,
        groupId,
        groupMetadata: group.metadata,
        groupSource: group.source,
        question,
      });
      selectedCount++;
    }
  }

  return {
    dataset,
    groups: selectedGroups,
    questions: selectedQuestions,
  };
};

export const extractConversionCandidate = (
  sourceQuestion: SourceQuestion,
): SourceExtractionResult => {
  const questionText = sourceQuestion.question.questionText?.trim() ?? "";
  if (questionText.length === 0) {
    return {
      ok: false,
      failure: {
        sourceDataset: sourceQuestion.sourceDataset,
        groupId: sourceQuestion.groupId,
        questionId: sourceQuestion.question.id,
        questionKey: questionKeyOf(sourceQuestion.sourceDataset, sourceQuestion.question.id),
        sourceQuestionText: sourceQuestion.question.questionText ?? "",
        reason: "missing_question_text",
        detail: "questionText is missing or empty",
      },
    };
  }

  if (sourceQuestion.question.options.length === 0) {
    return {
      ok: false,
      failure: {
        sourceDataset: sourceQuestion.sourceDataset,
        groupId: sourceQuestion.groupId,
        questionId: sourceQuestion.question.id,
        questionKey: questionKeyOf(sourceQuestion.sourceDataset, sourceQuestion.question.id),
        sourceQuestionText: questionText,
        reason: "missing_options",
        detail: "question has no options",
      },
    };
  }

  if (sourceQuestion.question.correctOptionIds.length === 0) {
    return {
      ok: false,
      failure: {
        sourceDataset: sourceQuestion.sourceDataset,
        groupId: sourceQuestion.groupId,
        questionId: sourceQuestion.question.id,
        questionKey: questionKeyOf(sourceQuestion.sourceDataset, sourceQuestion.question.id),
        sourceQuestionText: questionText,
        reason: "missing_correct_option_ids",
        detail: "question has no correctOptionIds",
      },
    };
  }

  const correctIdSet = new Set(
    sourceQuestion.question.correctOptionIds.map((id) => normalizeOptionId(id)),
  );
  const romanStatements = extractRomanStatementMap(questionText);
  const correctAnswers: string[] = [];
  const incorrectAnswers: string[] = [];

  for (const option of sourceQuestion.question.options) {
    const resolvedOptionText = resolveReferenceOptionText(option.text, romanStatements);
    if (correctIdSet.has(normalizeOptionId(option.id))) {
      if (!resolvedOptionText) {
        return {
          ok: false,
          failure: {
            sourceDataset: sourceQuestion.sourceDataset,
            groupId: sourceQuestion.groupId,
            questionId: sourceQuestion.question.id,
            questionKey: questionKeyOf(sourceQuestion.sourceDataset, sourceQuestion.question.id),
            sourceQuestionText: questionText,
            reason: "unresolved_reference_option",
            detail: `correct option "${option.text}" references statements that could not be resolved from the question text`,
          },
        };
      }
      correctAnswers.push(resolvedOptionText);
    } else if (resolvedOptionText) {
      incorrectAnswers.push(resolvedOptionText);
    }
  }

  if (correctAnswers.length === 0) {
    return {
      ok: false,
      failure: {
        sourceDataset: sourceQuestion.sourceDataset,
        groupId: sourceQuestion.groupId,
        questionId: sourceQuestion.question.id,
        questionKey: questionKeyOf(sourceQuestion.sourceDataset, sourceQuestion.question.id),
        sourceQuestionText: questionText,
        reason: "correct_option_id_not_found",
        detail: "none of correctOptionIds matched any option id",
      },
    };
  }

  return {
    ok: true,
    value: {
      sourceDataset: sourceQuestion.sourceDataset,
      groupId: sourceQuestion.groupId,
      questionId: sourceQuestion.question.id,
      questionKey: questionKeyOf(sourceQuestion.sourceDataset, sourceQuestion.question.id),
      sourceQuestionText: questionText,
      questionMetadata: sourceQuestion.question.metadata,
      questionSource: sourceQuestion.question.source,
      groupMetadata: sourceQuestion.groupMetadata,
      images: sourceQuestion.question.images,
      correctAnswers: correctAnswers as [string, ...string[]],
      incorrectAnswers,
    },
  };
};

const buildRewritePrompt = (
  sourceQuestion: SourceQuestion,
  candidate: ConversionCandidate,
): string => {
  const formattedOptions = sourceQuestion.question.options
    .map((option) => `- ${option.id}: ${option.text}`)
    .join("\n");

  return `Original MCQ question:
${candidate.sourceQuestionText}

Original options:
${formattedOptions}

Original correct option text:
${candidate.correctAnswers.join(" | ")}

Rewrite this into a single open-ended question following the rules.`;
};

const rewriteQuestion = (sourceQuestion: SourceQuestion, candidate: ConversionCandidate) =>
  generateObject({
    schema: rewriteDecisionJsonSchema,
    system: REWRITE_SYSTEM_PROMPT,
    prompt: buildRewritePrompt(sourceQuestion, candidate),
    temperature: 0,
  }).pipe(Effect.map((result) => result.object as RewriteDecision));

const toConvertedQuestion = (
  candidate: ConversionCandidate,
  rewrittenQuestionText: string,
  conversionReason: string,
  evalModel: EvalModelIdentity,
): OpenEndedEvalQuestion => ({
  id: candidate.questionId,
  questionText: rewrittenQuestionText,
  referenceCorrectAnswers: candidate.correctAnswers,
  referenceIncorrectAnswers: [...candidate.incorrectAnswers],
  metadata: {
    sourceDataset: candidate.sourceDataset,
    sourceGroupId: candidate.groupId,
    sourceQuestionId: candidate.questionId,
    conversion: {
      reason: conversionReason,
      modelId: evalModel.modelId,
      ...(evalModel.provider ? { provider: evalModel.provider } : {}),
    },
  },
  images: [...candidate.images],
});

const timestampForFile = () =>
  new Date()
    .toISOString()
    .replaceAll(":", "-")
    .replace(/\.\d+Z$/, "Z");

const makeDatasetOutputFilename = (datasetId: string, runTimestamp: string) =>
  `${datasetId}-${runTimestamp}.open-ended.eval.json`;

const makeAnalysisFilename = (datasetId: string, runTimestamp: string) =>
  `${datasetId}-${runTimestamp}.conversion.json`;

const makeStateFilename = (datasetId: string, runTimestamp: string) =>
  `${datasetId}-${runTimestamp}.conversion.state.json`;

const resolveDatasetOutputPathForRun = (
  outputDir: string,
  datasetId: string,
  runTimestamp: string,
) => `${outputDir}/${makeDatasetOutputFilename(datasetId, runTimestamp)}`;

const resolveAnalysisPathForRun = (reportDir: string, datasetId: string, runTimestamp: string) =>
  `${reportDir}/${makeAnalysisFilename(datasetId, runTimestamp)}`;

const resolveStatePathForRun = (reportDir: string, datasetId: string, runTimestamp: string) =>
  `${reportDir}/${makeStateFilename(datasetId, runTimestamp)}`;

const stateFilenamePrefix = (datasetId: string) => `${datasetId}-`;
const stateFilenameSuffix = ".conversion.state.json";

const extractRunTimestampFromStatePath = (datasetId: string, statePath: string) => {
  const filename = basename(statePath);
  if (
    !filename.startsWith(stateFilenamePrefix(datasetId)) ||
    !filename.endsWith(stateFilenameSuffix)
  ) {
    return undefined;
  }

  return filename.slice(
    stateFilenamePrefix(datasetId).length,
    filename.length - stateFilenameSuffix.length,
  );
};

export const resolveLatestResumeStatePath = (reportDir: string, datasetId: string) =>
  Effect.gen(function* () {
    const filenames = yield* Effect.tryPromise({
      try: () => readdir(reportDir),
      catch: (cause) =>
        new ReadReportDirectoryError({
          path: reportDir,
          cause,
        }),
    }).pipe(Effect.catchTag("ReadReportDirectoryError", () => Effect.succeed([] as string[])));

    const latest = filenames
      .filter(
        (filename) =>
          filename.startsWith(stateFilenamePrefix(datasetId)) &&
          filename.endsWith(stateFilenameSuffix),
      )
      .sort()
      .at(-1);

    if (!latest) {
      return yield* new ResumeStateNotFoundError({
        reportDir,
        sourceDataset: datasetId,
      });
    }

    return `${reportDir}/${latest}`;
  });

type DatasetRunPaths = {
  outputPath: string;
  analysisPath: string;
  statePath: string;
  runTimestamp: string;
};

export const resolveDatasetRunPaths = (
  outputDir: string,
  reportDir: string,
  datasetId: string,
  resume: boolean,
  defaultRunTimestamp: string,
) =>
  Effect.gen(function* () {
    if (!resume) {
      return {
        outputPath: resolveDatasetOutputPathForRun(outputDir, datasetId, defaultRunTimestamp),
        analysisPath: resolveAnalysisPathForRun(reportDir, datasetId, defaultRunTimestamp),
        statePath: resolveStatePathForRun(reportDir, datasetId, defaultRunTimestamp),
        runTimestamp: defaultRunTimestamp,
      } satisfies DatasetRunPaths;
    }

    const statePath = yield* resolveLatestResumeStatePath(reportDir, datasetId).pipe(
      Effect.catchTag("ResumeStateNotFoundError", () => Effect.succeed(null)),
    );

    if (statePath === null) {
      return {
        outputPath: resolveDatasetOutputPathForRun(outputDir, datasetId, defaultRunTimestamp),
        analysisPath: resolveAnalysisPathForRun(reportDir, datasetId, defaultRunTimestamp),
        statePath: resolveStatePathForRun(reportDir, datasetId, defaultRunTimestamp),
        runTimestamp: defaultRunTimestamp,
      } satisfies DatasetRunPaths;
    }

    const runTimestamp =
      extractRunTimestampFromStatePath(datasetId, statePath) ?? defaultRunTimestamp;

    return {
      outputPath: resolveDatasetOutputPathForRun(outputDir, datasetId, runTimestamp),
      analysisPath: resolveAnalysisPathForRun(reportDir, datasetId, runTimestamp),
      statePath,
      runTimestamp,
    } satisfies DatasetRunPaths;
  });

const resolveDir = (path: Option.Option<string>, fallbackSegments: readonly string[]) =>
  Option.match(path, {
    onNone: () => repoPath(...fallbackSegments),
    onSome: (providedPath) => (isAbsolute(providedPath) ? providedPath : repoPath(providedPath)),
  });

const writeJson = Effect.fn("openEndedConversion.writeJson")(function* (
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

const loadCheckpoint = (path: string) =>
  Effect.try({
    try: () => JSON.parse(readFileSync(path, "utf8")) as ConversionCheckpoint,
    catch: (cause) => new ReadCheckpointError({ path, cause }),
  });

export const validateResumeCheckpoint = (
  checkpointPath: string,
  checkpointMetadata: CheckpointMetadata | undefined,
  current: {
    sourceDataset: string;
    limitPerDataset: number;
    concurrency: number;
    saveEvery: number;
    evalModel: EvalModelIdentity;
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

    if (checkpointMetadata.sourceDataset !== current.sourceDataset) {
      mismatches.push(
        `source dataset differs: checkpoint=${checkpointMetadata.sourceDataset}, current=${current.sourceDataset}`,
      );
    }

    const currentAllowsAtLeastCheckpointScope =
      current.limitPerDataset === 0 ||
      checkpointMetadata.limitPerDataset === current.limitPerDataset ||
      (checkpointMetadata.limitPerDataset > 0 &&
        current.limitPerDataset > checkpointMetadata.limitPerDataset);

    if (!currentAllowsAtLeastCheckpointScope) {
      mismatches.push(
        `per-dataset limit cannot shrink on resume: checkpoint=${checkpointMetadata.limitPerDataset}, current=${current.limitPerDataset}`,
      );
    }

    if (checkpointMetadata.saveEvery !== current.saveEvery) {
      mismatches.push(
        `save cadence differs: checkpoint=${checkpointMetadata.saveEvery}, current=${current.saveEvery}`,
      );
    }

    if (checkpointMetadata.evalModel.modelId !== current.evalModel.modelId) {
      mismatches.push(
        `eval model differs: checkpoint=${checkpointMetadata.evalModel.modelId}, current=${current.evalModel.modelId}`,
      );
    }

    if (mismatches.length > 0) {
      return yield* new ResumeCheckpointMismatchError({
        path: checkpointPath,
        mismatches,
      });
    }

    if (
      typeof checkpointMetadata.evalModel.provider === "string" &&
      typeof current.evalModel.provider === "string" &&
      checkpointMetadata.evalModel.provider !== current.evalModel.provider
    ) {
      yield* Effect.logWarning(
        `Eval provider changed: checkpoint=${checkpointMetadata.evalModel.provider}, current=${current.evalModel.provider}. Resuming anyway.`,
      );
    }
  });

export const buildOpenEndedDataset = (
  selectedGroups: ReadonlyArray<SelectedQuestionGroup>,
  entries: ReadonlyArray<ProcessedEntry>,
  sourceDataset: string,
): S.Schema.Type<typeof OpenEndedQuestionGroups> => {
  const convertedByKey = new Map(
    entries
      .filter((entry): entry is ConvertedEntry => entry.status === "converted")
      .map((entry) => [entry.questionKey, entry.convertedQuestion]),
  );

  return selectedGroups.map(
    (group): OpenEndedQuestionGroup => ({
      ...(group.id ? { id: group.id } : {}),
      ...(group.metadata !== undefined ? { metadata: group.metadata } : {}),
      ...(group.source ? { source: group.source } : {}),
      questions: group.questions.flatMap((question) => {
        const converted = convertedByKey.get(questionKeyOf(sourceDataset, question.id));
        return converted ? [converted] : [];
      }),
    }),
  );
};

export const buildCheckpoint = (
  dataset: SelectedDataset,
  entries: ReadonlyArray<ProcessedEntry>,
  config: {
    runTimestamp: string;
    limitPerDataset: number;
    concurrency: number;
    evalModel: EvalModelIdentity;
  },
): ConversionCheckpoint => {
  const convertedCount = entries.filter((entry) => entry.status === "converted").length;
  const skippedCount = entries.length - convertedCount;

  return {
    metadata: {
      generatedAt: new Date().toISOString(),
      runTimestamp: config.runTimestamp,
      sourceDataset: dataset.dataset.id,
      limitPerDataset: config.limitPerDataset,
      concurrency: config.concurrency,
      saveEvery: SAVE_EVERY,
      evalModel: config.evalModel,
    },
    counts: {
      totalSelected: dataset.questions.length,
      processed: entries.length,
      converted: convertedCount,
      skipped: skippedCount,
    },
    skippedByReason: countSkippedByReason(entries),
    entries: [...entries],
  };
};

export const buildAnalysisSummary = (
  dataset: SelectedDataset,
  entries: ReadonlyArray<ProcessedEntry>,
  config: {
    runTimestamp: string;
    limitPerDataset: number;
    concurrency: number;
    evalModel: EvalModelIdentity;
  },
): ConversionAnalysisSummary => {
  const checkpoint = buildCheckpoint(dataset, entries, config);
  return {
    metadata: checkpoint.metadata,
    counts: checkpoint.counts,
    skippedByReason: checkpoint.skippedByReason,
  };
};

const restoreEntries = (
  checkpoint: ConversionCheckpoint | null,
  questionOrder: ReadonlyMap<string, number>,
): ProcessedEntry[] =>
  checkpoint
    ? [...checkpoint.entries].sort(
        (left, right) =>
          (questionOrder.get(left.questionKey) ?? Number.MAX_SAFE_INTEGER) -
          (questionOrder.get(right.questionKey) ?? Number.MAX_SAFE_INTEGER),
      )
    : [];

const processSourceQuestion = (sourceQuestion: SourceQuestion, evalModel: EvalModelIdentity) =>
  Effect.gen(function* () {
    const extracted = extractConversionCandidate(sourceQuestion);
    if (!extracted.ok) {
      return {
        status: "skipped",
        questionKey: extracted.failure.questionKey,
        sourceDataset: extracted.failure.sourceDataset,
        groupId: extracted.failure.groupId,
        questionId: extracted.failure.questionId,
        sourceQuestionText: extracted.failure.sourceQuestionText,
        reason: "invalid_source_question",
        detail: `${extracted.failure.reason}: ${extracted.failure.detail}`,
      } satisfies SkippedEntry;
    }

    const decisionResult = yield* rewriteQuestion(sourceQuestion, extracted.value).pipe(
      taskRetryPolicy,
      Effect.either,
    );

    if (decisionResult._tag === "Left") {
      return {
        status: "skipped",
        questionKey: extracted.value.questionKey,
        sourceDataset: extracted.value.sourceDataset,
        groupId: extracted.value.groupId,
        questionId: extracted.value.questionId,
        sourceQuestionText: extracted.value.sourceQuestionText,
        reason: "rewrite_error",
        detail: String(decisionResult.left),
      } satisfies SkippedEntry;
    }

    const decision = decisionResult.right;
    if (!decision.convertible) {
      return {
        status: "skipped",
        questionKey: extracted.value.questionKey,
        sourceDataset: extracted.value.sourceDataset,
        groupId: extracted.value.groupId,
        questionId: extracted.value.questionId,
        sourceQuestionText: extracted.value.sourceQuestionText,
        reason: "non_convertible",
        detail: decision.reason || "Model marked question as non-convertible",
      } satisfies SkippedEntry;
    }

    const rewrittenQuestionText = decision.openEndedQuestion?.trim() ?? "";
    if (rewrittenQuestionText.length === 0) {
      return {
        status: "skipped",
        questionKey: extracted.value.questionKey,
        sourceDataset: extracted.value.sourceDataset,
        groupId: extracted.value.groupId,
        questionId: extracted.value.questionId,
        sourceQuestionText: extracted.value.sourceQuestionText,
        reason: "missing_rewrite",
        detail: decision.reason || "convertible=true but openEndedQuestion is empty",
      } satisfies SkippedEntry;
    }

    if (
      extracted.value.correctAnswers.some((answer) =>
        hasAnswerLeakage(rewrittenQuestionText, answer),
      )
    ) {
      return {
        status: "skipped",
        questionKey: extracted.value.questionKey,
        sourceDataset: extracted.value.sourceDataset,
        groupId: extracted.value.groupId,
        questionId: extracted.value.questionId,
        sourceQuestionText: extracted.value.sourceQuestionText,
        reason: "answer_leakage",
        detail: "Rewritten question appears to leak the reference answer",
      } satisfies SkippedEntry;
    }

    return {
      status: "converted",
      questionKey: extracted.value.questionKey,
      sourceDataset: extracted.value.sourceDataset,
      groupId: extracted.value.groupId,
      questionId: extracted.value.questionId,
      sourceQuestionText: extracted.value.sourceQuestionText,
      conversionReason: decision.reason,
      convertedQuestion: toConvertedQuestion(
        extracted.value,
        rewrittenQuestionText,
        decision.reason,
        evalModel,
      ),
    } satisfies ConvertedEntry;
  });

const runDatasetConversion = (
  selectedDataset: SelectedDataset,
  config: {
    defaultRunTimestamp: string;
    limitPerDataset: number;
    concurrency: number;
    outputDir: string;
    reportDir: string;
    resume: boolean;
    evalModel: EvalModelIdentity;
  },
) =>
  Effect.gen(function* () {
    const { outputPath, analysisPath, statePath, runTimestamp } = yield* resolveDatasetRunPaths(
      config.outputDir,
      config.reportDir,
      selectedDataset.dataset.id,
      config.resume,
      config.defaultRunTimestamp,
    );
    const questionOrder = new Map(
      selectedDataset.questions.map((question, index) => [
        questionKeyOf(question.sourceDataset, question.question.id),
        index,
      ]),
    );

    const checkpoint =
      config.resume && existsSync(statePath) ? yield* loadCheckpoint(statePath) : null;

    if (config.resume && checkpoint) {
      yield* validateResumeCheckpoint(statePath, checkpoint.metadata, {
        sourceDataset: selectedDataset.dataset.id,
        limitPerDataset: config.limitPerDataset,
        concurrency: config.concurrency,
        saveEvery: SAVE_EVERY,
        evalModel: config.evalModel,
      });
    } else if (config.resume) {
      yield* Effect.log(`No checkpoint found for ${selectedDataset.dataset.id}; starting fresh`);
    }

    const restoredEntries = restoreEntries(checkpoint, questionOrder);
    const completedQuestionKeys = new Set(restoredEntries.map((entry) => entry.questionKey));
    const pendingQuestions = selectedDataset.questions.filter(
      (question) =>
        !completedQuestionKeys.has(questionKeyOf(question.sourceDataset, question.question.id)),
    );

    const collectedEntries: ProcessedEntry[] = [...restoredEntries];
    let completedCount = restoredEntries.length;
    let lastSavedAt = restoredEntries.length;

    const sortEntriesForOutput = (entries: ReadonlyArray<ProcessedEntry>) =>
      [...entries].sort(
        (left, right) =>
          (questionOrder.get(left.questionKey) ?? Number.MAX_SAFE_INTEGER) -
          (questionOrder.get(right.questionKey) ?? Number.MAX_SAFE_INTEGER),
      );

    const saveState = (entries: ReadonlyArray<ProcessedEntry>) => {
      const sortedEntries = sortEntriesForOutput(entries);
      const datasetOutput = buildOpenEndedDataset(
        selectedDataset.groups,
        sortedEntries,
        selectedDataset.dataset.id,
      );
      const checkpointOutput = buildCheckpoint(selectedDataset, sortedEntries, {
        runTimestamp,
        limitPerDataset: config.limitPerDataset,
        concurrency: config.concurrency,
        evalModel: config.evalModel,
      });
      const analysisOutput = buildAnalysisSummary(selectedDataset, sortedEntries, {
        runTimestamp,
        limitPerDataset: config.limitPerDataset,
        concurrency: config.concurrency,
        evalModel: config.evalModel,
      });

      return writeJson(outputPath, datasetOutput).pipe(
        Effect.zipRight(writeJson(statePath, checkpointOutput)),
        Effect.zipRight(writeJson(analysisPath, analysisOutput)),
      );
    };

    const checkpointSaveSemaphore = yield* Effect.makeSemaphore(1);
    const appendEntry = (entry: ProcessedEntry) =>
      checkpointSaveSemaphore.withPermits(1)(
        Effect.gen(function* () {
          collectedEntries.push(entry);
          completedQuestionKeys.add(entry.questionKey);
          completedCount += 1;

          if (completedCount - lastSavedAt >= SAVE_EVERY) {
            lastSavedAt = completedCount;
            yield* saveState(collectedEntries).pipe(
              Effect.tapError((error) =>
                Effect.logWarning(
                  `Incremental save failed for ${selectedDataset.dataset.id}: ${String(error)}`,
                ),
              ),
              Effect.catchAll(() => Effect.void),
            );
            yield* Effect.log(
              `[${selectedDataset.dataset.id}] incremental save (${completedCount}/${selectedDataset.questions.length})`,
            );
          }
        }),
      );

    if (restoredEntries.length > 0) {
      yield* Effect.log(
        `[${selectedDataset.dataset.id}] restored ${restoredEntries.length} processed question(s) from checkpoint`,
      );
    }

    if (pendingQuestions.length > 0) {
      yield* Effect.log(
        `[${selectedDataset.dataset.id}] converting ${pendingQuestions.length} remaining question(s) with concurrency=${config.concurrency}`,
      );
      yield* Progress.forEach(
        pendingQuestions,
        (sourceQuestion) =>
          processSourceQuestion(sourceQuestion, config.evalModel).pipe(
            Effect.tap((entry) => appendEntry(entry)),
          ),
        {
          description: `Converting ${selectedDataset.dataset.id}`,
          concurrency: config.concurrency,
        },
      );
    } else {
      yield* Effect.log(`[${selectedDataset.dataset.id}] nothing left to convert`);
    }

    yield* saveState(collectedEntries);
    const finalCheckpoint = buildCheckpoint(
      selectedDataset,
      sortEntriesForOutput(collectedEntries),
      {
        runTimestamp,
        limitPerDataset: config.limitPerDataset,
        concurrency: config.concurrency,
        evalModel: config.evalModel,
      },
    );
    yield* Effect.log(
      [
        `[${selectedDataset.dataset.id}] wrote dataset ${outputPath}`,
        `[${selectedDataset.dataset.id}] wrote analysis ${analysisPath}`,
        `[${selectedDataset.dataset.id}] wrote state ${statePath}`,
        `[${selectedDataset.dataset.id}] selected=${finalCheckpoint.counts.totalSelected} processed=${finalCheckpoint.counts.processed} converted=${finalCheckpoint.counts.converted} skipped=${finalCheckpoint.counts.skipped}`,
      ].join("\n"),
    );
  });

const resolveEvalModelIdentity = Effect.fn("openEndedConversion.resolveEvalModelIdentity")(
  function* () {
    const model = (yield* LanguageModel) as { modelId?: string; provider?: string };
    return {
      modelId: model.modelId ?? "unknown",
      ...(typeof model.provider === "string" ? { provider: model.provider } : {}),
    } satisfies EvalModelIdentity;
  },
);

const datasetOption = Options.text("dataset").pipe(
  Options.repeated,
  Options.withDescription(
    `Dataset ID to convert. Repeatable. Leave empty to convert all text-only datasets (${TEXT_DATASET_IDS.join(", ")}).`,
  ),
);

const limitOption = Options.integer("limit").pipe(
  Options.withDefault(0),
  Options.withDescription("Maximum number of questions to convert per dataset. 0 means no limit."),
);

const concurrencyOption = Options.integer("concurrency").pipe(
  Options.withDefault(4),
  Options.withDescription("How many question conversions to run concurrently inside a dataset."),
);

const outputDirOption = Options.text("output-dir").pipe(
  Options.optional,
  Options.withDescription(
    "Directory for generated open-ended dataset JSON files. Relative paths resolve from the repo root.",
  ),
);

const reportDirOption = Options.text("report-dir").pipe(
  Options.optional,
  Options.withDescription(
    "Directory for per-dataset conversion checkpoints/reports. Relative paths resolve from the repo root.",
  ),
);

const resumeOption = Options.boolean("resume").pipe(
  Options.withDescription(
    "Resume from stable per-dataset checkpoint files in the report directory.",
  ),
);

const dryRunOption = Options.boolean("dry-run").pipe(
  Options.withDescription(
    "Only resolve datasets and counts; do not call the conversion model or write files.",
  ),
);

const runConversion = Effect.fn("openEndedConversion.runConversion")(function* (config: RunConfig) {
  const selectedDatasets = yield* selectDatasets(config.dataset);
  const resolvedOutputDir = resolveDir(config.outputDir, ["data", "evals", "open_ended"]);
  const resolvedReportDir = resolveDir(config.reportDir, ["data", "analysis", "open_ended"]);

  yield* Effect.log(
    `Selected ${selectedDatasets.length} dataset(s): ${selectedDatasets.map((dataset) => dataset.id).join(", ")}`,
  );

  const preparedDatasets = yield* Effect.forEach(selectedDatasets, (dataset) =>
    dataset
      .loadGroups()
      .pipe(Effect.map((groups) => selectQuestionSubset(dataset, groups, config.limit))),
  );

  for (const selectedDataset of preparedDatasets) {
    const totalQuestions = selectedDataset.questions.length;
    yield* Effect.log(
      `  ${selectedDataset.dataset.id}: ${totalQuestions} question(s) selected${config.limit > 0 ? ` (limit=${config.limit})` : ""}`,
    );
  }

  if (config.dryRun) {
    if (config.resume) {
      yield* Effect.log("Ignoring --resume in dry-run mode");
    }
    yield* Effect.log(
      `Dry run complete. ${preparedDatasets.reduce((sum, dataset) => sum + dataset.questions.length, 0)} question(s) would be processed.`,
    );
    return;
  }

  const evalModel = yield* resolveEvalModelIdentity();
  const defaultRunTimestamp = timestampForFile();
  yield* Effect.log(
    `Using eval model ${evalModel.modelId}${evalModel.provider ? ` via ${evalModel.provider}` : ""}`,
  );

  for (const selectedDataset of preparedDatasets) {
    yield* runDatasetConversion(selectedDataset, {
      defaultRunTimestamp,
      limitPerDataset: config.limit,
      concurrency: config.concurrency,
      outputDir: resolvedOutputDir,
      reportDir: resolvedReportDir,
      resume: config.resume,
      evalModel,
    });
  }
});

export const makeCommand = <A, E, R>(handler: (config: RunConfig) => Effect.Effect<A, E, R>) =>
  Command.make(
    "convert-text-datasets-to-open-ended",
    {
      dataset: datasetOption,
      limit: limitOption,
      concurrency: concurrencyOption,
      outputDir: outputDirOption,
      reportDir: reportDirOption,
      resume: resumeOption,
      dryRun: dryRunOption,
    },
    handler,
  ).pipe(
    Command.withDescription(
      "Rewrite local text-only MCQ eval datasets into open-ended datasets for judge-based evaluation.",
    ),
  );

export const command = makeCommand(runConversion);

export const cli = Command.run(command, {
  name: "Text Dataset Open-Ended Conversion",
  version: "v0.1.0",
});

const runtimeLayer = Layer.mergeAll(NodeContext.layer, Logger.pretty, EvalLanguageModelLayer);

if (import.meta.main) {
  cli(process.argv).pipe(
    Effect.provide(runtimeLayer),
    Effect.catchAllCause((cause) => Effect.sync(() => console.error(Cause.pretty(cause)))),
    NodeRuntime.runMain,
  );
}
