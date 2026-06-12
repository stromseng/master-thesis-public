import { Effect } from "effect";
import * as S from "effect/Schema";
import { readFile } from "node:fs/promises";
import { evalDataPath } from "../../src/utils/repo";
import { createOrGetDataset, getDatasetByName } from "../experiment_setup";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../question-schema";
import type { QuestionDatasetExpected, QuestionDatasetInput } from "../evaluator";

export type ShititongDataset =
  | "shititong-en-text"
  | "shititong-en-vision"
  | "shititong-zh-text"
  | "shititong-zh-vision";

export type QuestionDatasetMetadata = {
  dataset: ShititongDataset;
  groupId: string;
  questionId: string;
};

export const SHITITONG_EN_TEXT_DATASET_NAME = "shititong-en-text";
export const SHITITONG_EN_VISION_DATASET_NAME = "shititong-en-vision";
export const SHITITONG_ZH_TEXT_DATASET_NAME = "shititong-zh-text";
export const SHITITONG_ZH_VISION_DATASET_NAME = "shititong-zh-vision";

export class LoadShititongQuestionsError extends S.TaggedError<LoadShititongQuestionsError>()(
  "LoadShititongQuestionsError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

const loadJsonFile = (filename: string, label: string) =>
  Effect.gen(function* () {
    const raw = yield* Effect.tryPromise({
      try: () => readFile(evalDataPath("shititong", filename), "utf-8"),
      catch: (error) =>
        new LoadShititongQuestionsError({
          reason: `Failed to read Shititong ${label} file`,
          cause: error,
        }),
    });

    return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
      Effect.mapError(
        (error) =>
          new LoadShititongQuestionsError({
            reason: `Failed to decode Shititong ${label} question groups`,
            cause: error,
          }),
      ),
    );
  }).pipe(Effect.withSpan(`shititong.load.${label}`));

export const loadEnTextGroups = () =>
  loadJsonFile("shititong_english_deduped_text.json", "en-text");
export const loadEnVisionGroups = () =>
  loadJsonFile("shititong_english_deduped_vision.json", "en-vision");
export const loadZhTextGroups = () =>
  loadJsonFile("shititong_chinese_deduped_text.json", "zh-text");
export const loadZhVisionGroups = () =>
  loadJsonFile("shititong_chinese_deduped_vision.json", "zh-vision");

export const flattenQuestionGroups = (
  groups: EvalQuestionGroups,
  dataset: ShititongDataset,
): Array<{
  input: QuestionDatasetInput;
  output: QuestionDatasetExpected;
  metadata: QuestionDatasetMetadata;
}> =>
  groups.flatMap((group) =>
    group.questions.map((question) => {
      const { correctOptionIds, ...input } = question;
      return {
        input: input satisfies QuestionDatasetInput,
        output: { correctOptionIds } satisfies QuestionDatasetExpected,
        metadata: {
          dataset,
          groupId: group.id ?? "unknown-group",
          questionId: question.id,
        } satisfies QuestionDatasetMetadata,
      };
    }),
  );

const getOrCreateDataset = (
  name: string,
  description: string,
  dataset: ShititongDataset,
  loadGroups: () => Effect.Effect<EvalQuestionGroups, LoadShititongQuestionsError>,
) =>
  Effect.gen(function* () {
    const existing = yield* getDatasetByName(name);
    if (existing) {
      return { datasetId: existing.id, total: existing.example_count };
    }

    const groups = yield* loadGroups();
    const examples = flattenQuestionGroups(groups, dataset);

    const { datasetId } = yield* createOrGetDataset({
      name,
      description,
      examples,
    });

    return { datasetId, total: examples.length };
  }).pipe(Effect.withSpan(`shititong.getOrCreate.${dataset}`));

export const getOrCreateShititongEnTextDataset = () =>
  getOrCreateDataset(
    SHITITONG_EN_TEXT_DATASET_NAME,
    "Shititong English deduped text-only maritime exam questions",
    "shititong-en-text",
    loadEnTextGroups,
  );

export const getOrCreateShititongEnVisionDataset = () =>
  getOrCreateDataset(
    SHITITONG_EN_VISION_DATASET_NAME,
    "Shititong English deduped vision maritime exam questions (with images)",
    "shititong-en-vision",
    loadEnVisionGroups,
  );

export const getOrCreateShititongZhTextDataset = () =>
  getOrCreateDataset(
    SHITITONG_ZH_TEXT_DATASET_NAME,
    "Shititong Chinese deduped text-only maritime exam questions",
    "shititong-zh-text",
    loadZhTextGroups,
  );

export const getOrCreateShititongZhVisionDataset = () =>
  getOrCreateDataset(
    SHITITONG_ZH_VISION_DATASET_NAME,
    "Shititong Chinese deduped vision maritime exam questions (with images)",
    "shititong-zh-vision",
    loadZhVisionGroups,
  );

// Keep old export name for backwards compatibility with existing basic.ts
export const getOrCreateShititongDataset = getOrCreateShititongEnTextDataset;

export const buildQuestionPrompt = (question: QuestionDatasetInput) => {
  const choices = question.options
    .map((option) => {
      const hasImages = option.images && option.images.length > 0;
      const text = option.text || (hasImages ? "[See image]" : "");
      return `${option.id}) ${text}`;
    })
    .join("\n");
  return `Question: ${question.questionText}\nChoices:\n${choices}`;
};
