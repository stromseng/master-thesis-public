import { Effect, Logger } from "effect";
import * as S from "effect/Schema";
import { readFile } from "node:fs/promises";
import { evalDataPath } from "../../src/utils/repo";
import type { QuestionDatasetExpected, QuestionDatasetInput } from "../evaluator";
import { createOrGetDataset, getDatasetByName } from "../experiment_setup";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../question-schema";

export const baseLayer = Logger.pretty;

type QuestionDatasetMetadata = {
  dataset: "navreas";
  groupId: string;
  questionId: string;
  imageUris: readonly string[];
};

export const navreasDatasets = [
  {
    filename: "scene_understanding.eval.json",
    key: "scene-understanding",
    title: "Scene Understanding",
  },
  {
    filename: "colreg_compliance_and_good_seamanship.eval.json",
    key: "colreg-compliance",
    title: "COLREG Compliance",
  },
  {
    filename: "spatial_relationship_and_estimation_of_motion.eval.json",
    key: "spatial-relationship",
    title: "Spatial Relationship",
  },
] as const;

export type NavreasDataset = (typeof navreasDatasets)[number];

export const toDatasetName = (dataset: NavreasDataset): string => `navreas-${dataset.key}`;

export class LoadNavreasQuestionGroupsError extends S.TaggedError<LoadNavreasQuestionGroupsError>()(
  "LoadNavreasQuestionGroupsError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export class LoadNavreasImageError extends S.TaggedError<LoadNavreasImageError>()(
  "LoadNavreasImageError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export const loadQuestionGroups = Effect.fn("navreas.loadQuestionGroups")(function* (
  filename: string,
) {
  const raw = yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("navreas", filename), "utf-8"),
    catch: (error) =>
      new LoadNavreasQuestionGroupsError({
        reason: `Failed to read NavReas question groups file: ${filename}`,
        cause: error,
      }),
  });

  return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
    Effect.mapError(
      (error) =>
        new LoadNavreasQuestionGroupsError({
          reason: `Failed to decode NavReas question groups file: ${filename}`,
          cause: error,
        }),
    ),
  );
});

const flattenQuestionGroups = (
  groups: EvalQuestionGroups,
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
          dataset: "navreas",
          groupId: group.id ?? "unknown-group",
          questionId: question.id,
          imageUris: question.images.map((image) => image.uri),
        } satisfies QuestionDatasetMetadata,
      };
    }),
  );

export const getOrCreateNavreasDataset = Effect.fn("navreas.getOrCreateDataset")(function* (
  dataset: NavreasDataset,
) {
  const datasetName = toDatasetName(dataset);
  const existing = yield* getDatasetByName(datasetName);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadQuestionGroups(dataset.filename);
  const examples = flattenQuestionGroups(groups);

  const { datasetId } = yield* createOrGetDataset({
    name: datasetName,
    description: `NavReas ${dataset.title} in canonical question schema format`,
    examples,
  });

  return { datasetId, total: examples.length };
});

export const loadImage = Effect.fn("navreas.loadImage")(function* (filename: string) {
  return yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("navreas", "images", filename)),
    catch: (error) =>
      new LoadNavreasImageError({
        reason: `Failed to read NavReas image: ${filename}`,
        cause: error,
      }),
  });
});

export const buildQuestionPrompt = (question: QuestionDatasetInput) => {
  const choices = question.options.map((option) => `${option.id}) ${option.text}`).join("\n");
  return `Question: ${question.questionText}\nChoices:\n${choices}`;
};
