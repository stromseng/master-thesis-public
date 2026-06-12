import * as S from "effect/Schema";

export class QuestionImage extends S.Class<QuestionImage>("QuestionImage")({
  id: S.optional(S.String),
  uri: S.String,
  caption: S.optional(S.String),
}) {}

const DefaultImages = S.optionalWith(S.Array(QuestionImage), {
  default: () => [],
});

const SourceRecord = S.Record({ key: S.String, value: S.String });

export class QuestionOption extends S.Class<QuestionOption>("QuestionOption")({
  id: S.String,
  text: S.String,
  images: S.optionalWith(S.Array(QuestionImage), {
    default: () => [],
  }),
}) {}

const baseQuestionFields = {
  id: S.String.annotations({
    title: "question-id",
    description:
      "Unique question identifier. For Raynor questions, use the `raynor-<number>` format.",
    examples: ["raynor-4285"],
  }),
  questionText: S.String,
  metadata: S.Unknown,
  images: DefaultImages,
  source: S.optional(SourceRecord),
};

const NonEmptyOptionIds = S.Array(S.String).pipe(
  S.filter((ids): ids is readonly [string, ...string[]] => ids.length >= 1, {
    identifier: "NonEmptyOptionIds",
  }),
);

export class MultipleChoiceQuestion extends S.Class<MultipleChoiceQuestion>(
  "MultipleChoiceQuestion",
)({
  ...baseQuestionFields,
  options: S.NonEmptyArray(QuestionOption),
  correctOptionIds: NonEmptyOptionIds,
}) {}

export const EvalQuestion = MultipleChoiceQuestion;
export type EvalQuestion = S.Schema.Type<typeof EvalQuestion>;

export class QuestionGroup extends S.Class<QuestionGroup>("QuestionGroup")({
  id: S.optional(S.String),
  metadata: S.optional(S.Unknown),
  source: S.optional(SourceRecord),
  questions: S.Array(EvalQuestion),
}) {}

export const EvalQuestionGroups = S.Array(QuestionGroup);
export type EvalQuestionGroups = S.Schema.Type<typeof EvalQuestionGroups>;

export const EvalQuestionFromJson = S.parseJson(EvalQuestion);
export const QuestionGroupFromJson = S.parseJson(QuestionGroup);
export const QuestionGroupsFromJson = S.parseJson(EvalQuestionGroups);
