import { Effect } from "effect";
import * as S from "effect/Schema";
import { readFile } from "node:fs/promises";
import { evalDataPath } from "../../src/utils/repo";
import { createOrGetDataset, getDatasetByName } from "../experiment_setup";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../question-schema";
import type { QuestionDatasetExpected, QuestionDatasetInput } from "../evaluator";

type QuestionDatasetMetadata = {
  dataset: "us-coast-guard";
  groupId: string;
  questionId: string;
};

export const US_COAST_GUARD_DATASET_NAME_V2 = "us-coast-guard-v2";
export const US_COAST_GUARD_DATASET_NAME_TAKE_10_V2 = "us-coast-guard-take-10-v2";
export const US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2 = "us-coast-guard-text-only-v2";
export const US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2 = "us-coast-guard-multimodal-v2";

export class LoadUsCoastGuardQuestionsError extends S.TaggedError<LoadUsCoastGuardQuestionsError>()(
  "LoadUsCoastGuardQuestionsError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export class LoadUsCoastGuardImageError extends S.TaggedError<LoadUsCoastGuardImageError>()(
  "LoadUsCoastGuardImageError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

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
          dataset: "us-coast-guard",
          groupId: group.id ?? "unknown-group",
          questionId: question.id,
        } satisfies QuestionDatasetMetadata,
      };
    }),
  );

export const getOrCreateUsCoastGuardDatasetTake10 = Effect.fn(
  "us_coast_guard.getOrCreateDatasetTake10",
)(function* () {
  const existing = yield* getDatasetByName(US_COAST_GUARD_DATASET_NAME_TAKE_10_V2);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadTextOnlyQuestionGroups();
  const examples = flattenQuestionGroups(groups).slice(0, 10);

  const { datasetId } = yield* createOrGetDataset({
    name: US_COAST_GUARD_DATASET_NAME_TAKE_10_V2,
    description: "US Coast Guard exam questions - text only, first 10 (smoke test) (v2)",
    examples,
  });

  return { datasetId, total: examples.length };
});

export const loadTextOnlyQuestionGroups = Effect.fn("us_coast_guard.loadTextOnlyQuestionGroups")(
  function* () {
    const raw = yield* Effect.tryPromise({
      try: () => readFile(evalDataPath("us_coast_guard", "all_questions_text_only.json"), "utf-8"),
      catch: (error) =>
        new LoadUsCoastGuardQuestionsError({
          reason: "Failed to read US Coast Guard text-only questions file",
          cause: error,
        }),
    });

    return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
      Effect.mapError(
        (error) =>
          new LoadUsCoastGuardQuestionsError({
            reason: "Failed to decode US Coast Guard text-only question groups",
            cause: error,
          }),
      ),
    );
  },
);

export const getOrCreateUsCoastGuardTextOnlyDataset = Effect.fn(
  "us_coast_guard.getOrCreateTextOnlyDataset",
)(function* () {
  const existing = yield* getDatasetByName(US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadTextOnlyQuestionGroups();
  const examples = flattenQuestionGroups(groups);

  const { datasetId } = yield* createOrGetDataset({
    name: US_COAST_GUARD_DATASET_NAME_TEXT_ONLY_V2,
    description: "US Coast Guard exam questions - text only (no images) (v2)",
    examples,
  });

  return { datasetId, total: examples.length };
});

export const loadMultimodalQuestionGroups = Effect.fn(
  "us_coast_guard.loadMultimodalQuestionGroups",
)(function* () {
  const raw = yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("us_coast_guard", "all_questions_multimodal.json"), "utf-8"),
    catch: (error) =>
      new LoadUsCoastGuardQuestionsError({
        reason: "Failed to read US Coast Guard multimodal questions file",
        cause: error,
      }),
  });

  return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
    Effect.mapError(
      (error) =>
        new LoadUsCoastGuardQuestionsError({
          reason: "Failed to decode US Coast Guard multimodal question groups",
          cause: error,
        }),
    ),
  );
});

export const getOrCreateUsCoastGuardMultimodalDataset = Effect.fn(
  "us_coast_guard.getOrCreateMultimodalDataset",
)(function* () {
  const existing = yield* getDatasetByName(US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadMultimodalQuestionGroups();
  const examples = flattenQuestionGroups(groups);

  const { datasetId } = yield* createOrGetDataset({
    name: US_COAST_GUARD_DATASET_NAME_MULTIMODAL_V2,
    description: "US Coast Guard exam questions - multimodal only (questions with images) (v2)",
    examples,
  });

  return { datasetId, total: examples.length };
});

export const loadImage = Effect.fn("us_coast_guard.loadImage")(function* (filename: string) {
  return yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("us_coast_guard", "images", filename)),
    catch: (error) =>
      new LoadUsCoastGuardImageError({
        reason: `Failed to read US Coast Guard image: ${filename}`,
        cause: error,
      }),
  });
});

export const buildQuestionPrompt = (question: QuestionDatasetInput) => {
  const choices = question.options.map((option) => `${option.id}) ${option.text}`).join("\n");
  return `Question: ${question.questionText}\nChoices:\n${choices}`;
};
