import * as S from "effect/Schema";
import { QuestionImage } from "./question-schema";

const DefaultImages = S.optionalWith(S.Array(QuestionImage), {
  default: () => [],
});

const SourceRecord = S.Record({ key: S.String, value: S.String });

const NonEmptyAnswers = S.Array(S.String).pipe(
  S.filter((answers): answers is readonly [string, ...string[]] => answers.length >= 1, {
    identifier: "NonEmptyAnswers",
  }),
);

export class OpenEndedEvalQuestion extends S.Class<OpenEndedEvalQuestion>("OpenEndedEvalQuestion")({
  id: S.String.annotations({
    title: "question-id",
    description: "Unique question identifier copied from the source MCQ dataset.",
  }),
  questionText: S.String,
  referenceCorrectAnswers: NonEmptyAnswers,
  referenceIncorrectAnswers: S.Array(S.String),
  metadata: S.Unknown,
  images: DefaultImages,
  source: S.optional(SourceRecord),
}) {}

export class OpenEndedQuestionGroup extends S.Class<OpenEndedQuestionGroup>(
  "OpenEndedQuestionGroup",
)({
  id: S.optional(S.String),
  metadata: S.optional(S.Unknown),
  source: S.optional(SourceRecord),
  questions: S.Array(OpenEndedEvalQuestion),
}) {}

export const OpenEndedQuestionGroups = S.Array(OpenEndedQuestionGroup);
export type OpenEndedQuestionGroups = S.Schema.Type<typeof OpenEndedQuestionGroups>;

export const OpenEndedQuestionGroupsFromJson = S.parseJson(OpenEndedQuestionGroups);
