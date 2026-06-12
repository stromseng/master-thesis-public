import { Effect } from "effect";
import * as S from "effect/Schema";
import { readFile } from "node:fs/promises";
import { evalDataPath } from "../../src/utils/repo";
import { createOrGetDataset, getDatasetByName } from "../experiment_setup";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../question-schema";
import type { QuestionDatasetExpected, QuestionDatasetInput } from "../evaluator";

type QuestionDatasetMetadata = {
  dataset: "crewcn";
  groupId: string;
  questionId: string;
};

export const CREWCN_DATASET_NAME = "crewcn";
export const CREWCN_SAMPLE_100_DATASET_NAME = "crewcn-sample-100";

/** Simple seeded PRNG (mulberry32) for reproducible sampling. */
const mulberry32 = (seed: number) => {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** Fisher-Yates shuffle with seeded PRNG, returns a new array. */
const seededShuffle = <T>(arr: readonly T[], seed: number): T[] => {
  const result = [...arr];
  const rng = mulberry32(seed);
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
};

export class LoadCrewQuestionsError extends S.TaggedError<LoadCrewQuestionsError>()(
  "LoadCrewQuestionsError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export const loadQuestionGroups = Effect.fn("crew.loadQuestionGroups")(function* () {
  const raw = yield* Effect.tryPromise({
    try: () => readFile(evalDataPath("crewcn", "crewcn_exam_questions.json"), "utf-8"),
    catch: (error) =>
      new LoadCrewQuestionsError({
        reason: "Failed to read CrewCN exam questions file",
        cause: error,
      }),
  });

  return yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
    Effect.mapError(
      (error) =>
        new LoadCrewQuestionsError({
          reason: "Failed to decode CrewCN question groups",
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
          dataset: "crewcn",
          groupId: group.id ?? "unknown-group",
          questionId: question.id,
        } satisfies QuestionDatasetMetadata,
      };
    }),
  );

export const getOrCreateCrewDataset = Effect.fn("crew.getOrCreateDataset")(function* () {
  const existing = yield* getDatasetByName(CREWCN_DATASET_NAME);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadQuestionGroups();
  const examples = flattenQuestionGroups(groups);

  const { datasetId } = yield* createOrGetDataset({
    name: CREWCN_DATASET_NAME,
    description: "CrewCN Chinese maritime exam questions in canonical question schema format",
    examples,
  });

  return { datasetId, total: examples.length };
});

export const getOrCreateCrewSample100Dataset = Effect.fn("crew.getOrCreateSample100Dataset")(
  function* () {
    const existing = yield* getDatasetByName(CREWCN_SAMPLE_100_DATASET_NAME);
    if (existing) {
      return { datasetId: existing.id, total: existing.example_count };
    }

    const groups = yield* loadQuestionGroups();
    const allExamples = flattenQuestionGroups(groups);
    const examples = seededShuffle(allExamples, 42).slice(0, 100);

    const { datasetId } = yield* createOrGetDataset({
      name: CREWCN_SAMPLE_100_DATASET_NAME,
      description: "CrewCN 100 sampled questions (seed=42) in canonical question schema format",
      examples,
    });

    return { datasetId, total: examples.length };
  },
);

export const buildQuestionPrompt = (question: QuestionDatasetInput) => {
  const choices = question.options.map((option) => `${option.id}) ${option.text}`).join("\n");
  return `Question: ${question.questionText}\nChoices:\n${choices}`;
};
