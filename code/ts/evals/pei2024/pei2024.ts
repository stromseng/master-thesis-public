import { Effect } from "effect";
import * as S from "effect/Schema";
import { readFile } from "node:fs/promises";
import { evalDataPath } from "../../src/utils/repo";
import { createOrGetDataset, getDatasetByName } from "../experiment_setup";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../question-schema";
import type { QuestionDatasetExpected, QuestionDatasetInput } from "../evaluator";

type Pei2024Dataset = "pei2024-uk" | "pei2024-zh";

type QuestionDatasetMetadata = {
  dataset: Pei2024Dataset;
  groupId: string;
  questionId: string;
};

export const PEI2024_UK_DATASET_NAME = "pei2024-uk";
export const PEI2024_ZH_DATASET_NAME = "pei2024-zh";

export class LoadPeiQuestionsError extends S.TaggedError<LoadPeiQuestionsError>()(
  "LoadPeiQuestionsError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

const loadJsonFile = (filename: string, label: string) =>
  Effect.gen(function* () {
    const raw = yield* Effect.tryPromise({
      try: () => readFile(evalDataPath("pei2024application", filename), "utf-8"),
      catch: (error) =>
        new LoadPeiQuestionsError({
          reason: `Failed to read PEI ${label} file`,
          cause: error,
        }),
    });

    return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
      Effect.mapError(
        (error) =>
          new LoadPeiQuestionsError({
            reason: `Failed to decode PEI ${label} question groups`,
            cause: error,
          }),
      ),
    );
  }).pipe(Effect.withSpan(`pei2024.load.${label}`));

export const loadUkQuestionGroups = () => loadJsonFile("uk_theory_test.json", "uk");
export const loadZhQuestionGroups = () => loadJsonFile("zh_theory_test.json", "zh");

const flattenQuestionGroups = (
  groups: EvalQuestionGroups,
  dataset: Pei2024Dataset,
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
  dataset: Pei2024Dataset,
  loadGroups: () => Effect.Effect<EvalQuestionGroups, LoadPeiQuestionsError>,
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
  }).pipe(Effect.withSpan(`pei2024.getOrCreate.${dataset}`));

export const getOrCreatePeiUkDataset = () =>
  getOrCreateDataset(
    PEI2024_UK_DATASET_NAME,
    "PEI 2024 UK theory test in canonical question schema format",
    "pei2024-uk",
    loadUkQuestionGroups,
  );

export const getOrCreatePeiZhDataset = () =>
  getOrCreateDataset(
    PEI2024_ZH_DATASET_NAME,
    "PEI 2024 Chinese theory test in canonical question schema format",
    "pei2024-zh",
    loadZhQuestionGroups,
  );

// Keep old export name for backwards compatibility with rag.eval.ts
export const getOrCreatePeiDataset = getOrCreatePeiUkDataset;

export const buildQuestionPrompt = (question: QuestionDatasetInput) => {
  const choices = question.options.map((option) => `${option.id}) ${option.text}`).join("\n");
  return `Question: ${question.questionText}\nChoices:\n${choices}`;
};
