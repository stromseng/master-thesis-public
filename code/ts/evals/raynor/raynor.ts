import { Effect } from "effect";
import * as S from "effect/Schema";
import { readFile } from "node:fs/promises";
import { evalDataPath } from "../../src/utils/repo";
import { createOrGetDataset, getDatasetByName } from "../experiment_setup";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../question-schema";
import type { QuestionDatasetExpected, QuestionDatasetInput } from "../evaluator";

type QuestionDatasetMetadata = {
  dataset: "raynor";
  groupId: string;
  questionId: string;
};

export const RAYNOR_DATASET_NAME_V2 = "raynor-v2";
export const RAYNOR_DATASET_NAME_MULTIMODAL_V2 = "raynor-multimodal-v2";

export class LoadRaynorQuestionsError extends S.TaggedError<LoadRaynorQuestionsError>()(
  "LoadRaynorQuestionsError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export class LoadRaynorImageError extends S.TaggedError<LoadRaynorImageError>()(
  "LoadRaynorImageError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export const loadQuestionGroups = Effect.fn("raynor.loadQuestionGroups")(function* () {
  const raw = yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("raynor", "text_only.json"), "utf-8"),
    catch: (error) =>
      new LoadRaynorQuestionsError({
        reason: "Failed to read Raynor questions file",
        cause: error,
      }),
  });

  return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
    Effect.mapError(
      (error) =>
        new LoadRaynorQuestionsError({
          reason: "Failed to decode Raynor question groups",
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
          dataset: "raynor",
          groupId: group.id ?? "unknown-group",
          questionId: question.id,
        } satisfies QuestionDatasetMetadata,
      };
    }),
  );

export const getOrCreateRaynorDataset = Effect.fn("raynor.getOrCreateDataset")(function* () {
  const existing = yield* getDatasetByName(RAYNOR_DATASET_NAME_V2);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadQuestionGroups();
  const examples = flattenQuestionGroups(groups);

  const { datasetId } = yield* createOrGetDataset({
    name: RAYNOR_DATASET_NAME_V2,
    description: "Raynor Maritime navigation rules questions - text only (v2)",
    examples,
  });

  return { datasetId, total: examples.length };
});

export const loadMultimodalQuestionGroups = Effect.fn("raynor.loadMultimodalQuestionGroups")(
  function* () {
    const raw = yield* Effect.tryPromise({
      try: () => readFile(evalDataPath("raynor", "multimodal.json"), "utf-8"),
      catch: (error) =>
        new LoadRaynorQuestionsError({
          reason: "Failed to read Raynor multimodal questions file",
          cause: error,
        }),
    });

    return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
      Effect.mapError(
        (error) =>
          new LoadRaynorQuestionsError({
            reason: "Failed to decode Raynor multimodal question groups",
            cause: error,
          }),
      ),
    );
  },
);

export const getOrCreateRaynorMultimodalDataset = Effect.fn("raynor.getOrCreateMultimodalDataset")(
  function* () {
    const existing = yield* getDatasetByName(RAYNOR_DATASET_NAME_MULTIMODAL_V2);
    if (existing) {
      return { datasetId: existing.id, total: existing.example_count };
    }

    const groups = yield* loadMultimodalQuestionGroups();
    const examples = flattenQuestionGroups(groups);

    const { datasetId } = yield* createOrGetDataset({
      name: RAYNOR_DATASET_NAME_MULTIMODAL_V2,
      description:
        "Raynor Maritime navigation rules questions - multimodal only (questions with images) (v2)",
      examples,
    });

    return { datasetId, total: examples.length };
  },
);

export const loadImage = Effect.fn("raynor.loadImage")(function* (filename: string) {
  return yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("raynor", "images", filename)),
    catch: (error) =>
      new LoadRaynorImageError({
        reason: `Failed to read Raynor image: ${filename}`,
        cause: error,
      }),
  });
});

export const buildQuestionPrompt = (question: QuestionDatasetInput) => {
  const choices = question.options.map((option) => `${option.id}) ${option.text}`).join("\n");
  return `Question: ${question.questionText}\nChoices:\n${choices}`;
};
