import { readFile } from "node:fs/promises";
import { Effect } from "effect";
import * as S from "effect/Schema";
import { evalDataPath } from "../src/utils/repo";
import { createOrGetDataset, getDatasetByName } from "./experiment_setup";
import {
  OpenEndedQuestionGroupsFromJson,
  type OpenEndedQuestionGroups,
} from "./open-ended-question-schema";
import type { OpenEndedDatasetExpected, OpenEndedDatasetInput } from "./open-ended-evaluator";

export class LoadOpenEndedDatasetError extends S.TaggedError<LoadOpenEndedDatasetError>()(
  "LoadOpenEndedDatasetError",
  {
    reason: S.String,
    cause: S.Defect,
  },
) {}

export type OpenEndedDatasetDefinition = {
  dataset: string;
  name: string;
  description: string;
  fileSegments: readonly [string, ...string[]];
};

type OpenEndedDatasetMetadata = {
  dataset: string;
  groupId: string;
  questionId: string;
  sourceDataset?: string;
};

const getMetadataSourceDataset = (metadata: unknown): string | undefined => {
  if (typeof metadata !== "object" || metadata === null) return undefined;
  const value = (metadata as { sourceDataset?: unknown }).sourceDataset;
  return typeof value === "string" ? value : undefined;
};

export const loadOpenEndedQuestionGroups = Effect.fn("openEnded.loadQuestionGroups")(function* (
  fileSegments: readonly [string, ...string[]],
) {
  const raw = yield* Effect.tryPromise({
    try: () => readFile(evalDataPath(...fileSegments), "utf-8"),
    catch: (error) =>
      new LoadOpenEndedDatasetError({
        reason: `Failed to read open-ended dataset file: ${fileSegments.join("/")}`,
        cause: error,
      }),
  });

  return yield* S.decodeUnknown(OpenEndedQuestionGroupsFromJson)(raw).pipe(
    Effect.mapError(
      (error) =>
        new LoadOpenEndedDatasetError({
          reason: `Failed to decode open-ended dataset file: ${fileSegments.join("/")}`,
          cause: error,
        }),
    ),
  );
});

export const flattenOpenEndedQuestionGroups = (
  groups: OpenEndedQuestionGroups,
  dataset: string,
): Array<{
  input: OpenEndedDatasetInput;
  output: OpenEndedDatasetExpected;
  metadata: OpenEndedDatasetMetadata;
}> =>
  groups.flatMap((group) =>
    group.questions.map((question) => {
      const { referenceCorrectAnswers, referenceIncorrectAnswers, ...input } = question;
      return {
        input: input satisfies OpenEndedDatasetInput,
        output: {
          referenceCorrectAnswers,
          referenceIncorrectAnswers,
        } satisfies OpenEndedDatasetExpected,
        metadata: {
          dataset,
          groupId: group.id ?? "unknown-group",
          questionId: question.id,
          sourceDataset: getMetadataSourceDataset(question.metadata),
        } satisfies OpenEndedDatasetMetadata,
      };
    }),
  );

export const getOrCreateOpenEndedDataset = Effect.fn("openEnded.getOrCreateDataset")(function* (
  definition: OpenEndedDatasetDefinition,
) {
  const existing = yield* getDatasetByName(definition.name);
  if (existing) {
    return { datasetId: existing.id, total: existing.example_count };
  }

  const groups = yield* loadOpenEndedQuestionGroups(definition.fileSegments);
  const examples = flattenOpenEndedQuestionGroups(groups, definition.dataset);

  const { datasetId } = yield* createOrGetDataset({
    name: definition.name,
    description: definition.description,
    examples,
  });

  return { datasetId, total: examples.length };
});

export type SlicedOpenEndedDatasetDefinition = OpenEndedDatasetDefinition & {
  maxQuestions: number;
};

export const getOrCreateSlicedOpenEndedDataset = Effect.fn("openEnded.getOrCreateSlicedDataset")(
  function* (definition: SlicedOpenEndedDatasetDefinition) {
    const existing = yield* getDatasetByName(definition.name);
    if (existing) {
      return { datasetId: existing.id, total: existing.example_count };
    }

    const groups = yield* loadOpenEndedQuestionGroups(definition.fileSegments);
    const examples = flattenOpenEndedQuestionGroups(groups, definition.dataset).slice(
      0,
      definition.maxQuestions,
    );

    const { datasetId } = yield* createOrGetDataset({
      name: definition.name,
      description: definition.description,
      examples,
    });

    return { datasetId, total: examples.length };
  },
);
