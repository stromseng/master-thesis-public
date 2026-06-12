import { describe, expect, it } from "@effect/vitest";
import { Effect } from "effect";
import * as S from "effect/Schema";
import { EvalQuestion, QuestionGroup, QuestionGroupsFromJson } from "../../evals/question-schema";

describe("eval question schema", () => {
  it.effect("decodes a valid multiple_choice question with one correct option", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-1",
        questionText: "What is COLREG Rule 5 about?",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "Lookout" },
          { id: "B", text: "Anchoring" },
        ],
        correctOptionIds: ["A"],
      });

      expect(decoded.correctOptionIds).toEqual(["A"]);
      expect(decoded.images).toEqual([]);
    }),
  );

  it.effect("rejects multiple_choice questions with zero correctOptionIds", () =>
    Effect.gen(function* () {
      const result = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-2",
        questionText: "Choose one",
        metadata: { fixture: "test" },
        options: [{ id: "A", text: "Alpha" }],
        correctOptionIds: [],
      }).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("decodes a valid multiple_choice question with multiple correct options", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-3",
        questionText: "Select all that apply",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "Alpha" },
          { id: "B", text: "Beta" },
        ],
        correctOptionIds: ["A", "B"],
      });

      expect(decoded.correctOptionIds).toEqual(["A", "B"]);
    }),
  );

  it.effect("decodes another valid multiple_choice question", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-4",
        questionText: "Select all required watchkeeping checks",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "Radar" },
          { id: "B", text: "Visual lookout" },
          { id: "C", text: "Coffee break" },
        ],
        correctOptionIds: ["A", "B"],
      });

      expect(decoded.correctOptionIds).toEqual(["A", "B"]);
    }),
  );

  it.effect("decodes multiple_choice questions with one correctOptionId", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-5",
        questionText: "Select all that apply",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "A" },
          { id: "B", text: "B" },
        ],
        correctOptionIds: ["A"],
      });

      expect(decoded.correctOptionIds).toEqual(["A"]);
    }),
  );

  it.effect("decodes true/false as a regular two-option multiple choice question", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-6",
        questionText: "A vessel constrained by draft has priority over a vessel at anchor.",
        metadata: { fixture: "test" },
        options: [
          { id: "true", text: "True" },
          { id: "false", text: "False" },
        ],
        correctOptionIds: ["true"],
      });

      expect(decoded.correctOptionIds).toEqual(["true"]);
    }),
  );

  it.effect("accepts arbitrary option IDs for two-option multiple choice", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-7",
        questionText: "Binary but non-boolean labels",
        metadata: { fixture: "test" },
        options: [
          { id: "yes", text: "Yes" },
          { id: "no", text: "No" },
        ],
        correctOptionIds: ["yes"],
      });

      expect(decoded.correctOptionIds).toEqual(["yes"]);
    }),
  );

  it.effect("decodes question-level and option-level images", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-8",
        questionText: "Identify the mark in the image.",
        metadata: { fixture: "test" },
        images: [{ uri: "https://example.com/question.png", caption: "Question figure" }],
        options: [
          { id: "A", text: "Safe water mark", images: [{ uri: "https://example.com/a.png" }] },
          { id: "B", text: "Cardinal mark", images: [{ uri: "https://example.com/b.png" }] },
        ],
        correctOptionIds: ["B"],
      });

      expect(decoded.images.length).toBe(1);
      expect(decoded.options[0]?.images.length).toBe(1);
      expect(decoded.options[1]?.images.length).toBe(1);
    }),
  );

  it.effect("decodes source as a string-to-string record", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-9",
        questionText: "Source metadata test",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "Yes" },
          { id: "B", text: "No" },
        ],
        correctOptionIds: ["A"],
        source: {
          provider: "shititong",
          chapter: "12",
          language: "english",
        },
      });

      expect(decoded.source).toBeDefined();
    }),
  );

  it.effect("rejects non-object source values", () =>
    Effect.gen(function* () {
      const result = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-10",
        questionText: "Invalid source test",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "Yes" },
          { id: "B", text: "No" },
        ],
        correctOptionIds: ["A"],
        source: "not-an-object",
      }).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("rejects non-string source values", () =>
    Effect.gen(function* () {
      const result = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-10b",
        questionText: "Invalid source map value",
        metadata: { fixture: "test" },
        options: [
          { id: "A", text: "Yes" },
          { id: "B", text: "No" },
        ],
        correctOptionIds: ["A"],
        source: {
          provider: "shititong",
          chapter: 12,
        },
      }).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("ignores unknown question fields like messages", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(EvalQuestion)({
        id: "q-11",
        questionText: "Unexpected fields",
        metadata: { fixture: "test" },
        messages: [
          {
            role: "system",
            content: "Unexpected",
          },
        ],
        options: [
          { id: "A", text: "One" },
          { id: "B", text: "Two" },
        ],
        correctOptionIds: ["A"],
      });

      expect(decoded.images).toEqual([]);
      expect(Object.prototype.hasOwnProperty.call(decoded, "messages")).toBe(false);
    }),
  );
});

describe("question group schema", () => {
  it.effect("decodes a valid QuestionGroup with group-level source and questions", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(QuestionGroup)({
        id: "raynor-group",
        source: {
          provider: "raynormaritime",
          exam: "ALL",
        },
        questions: [
          {
            id: "raynor-4285",
            questionText: "Sample question",
            metadata: { fixture: "question-group-test" },
            options: [
              { id: "A", text: "Alpha" },
              { id: "B", text: "Beta" },
            ],
            correctOptionIds: ["A"],
          },
        ],
      });

      expect(decoded.id).toBe("raynor-group");
      expect(decoded.questions.length).toBe(1);
      expect(decoded.source?.provider).toBe("raynormaritime");
    }),
  );

  it.effect("decodes a root array with multiple groups", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(QuestionGroupsFromJson)(
        `[
          {
            "id": "raynor",
            "source": { "provider": "raynormaritime" },
            "questions": [
              {
                "id": "raynor-1",
                "questionText": "Q1",
                "metadata": { "fixture": "question-group-test" },
                "options": [{ "id": "A", "text": "Answer" }],
                "correctOptionIds": ["A"]
              }
            ]
          },
          {
            "id": "navreas",
            "source": { "provider": "navreas" },
            "questions": [
              {
                "id": "navreas-1",
                "questionText": "Q2",
                "metadata": { "fixture": "question-group-test" },
                "options": [{ "id": "A", "text": "Answer" }],
                "correctOptionIds": ["A"]
              }
            ]
          }
        ]`,
      );

      expect(decoded.length).toBe(2);
      expect(decoded[0]?.id).toBe("raynor");
      expect(decoded[1]?.id).toBe("navreas");
    }),
  );

  it.effect("accepts optional group id and metadata as absent", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(QuestionGroup)({
        source: {
          provider: "raynormaritime",
        },
        questions: [
          {
            id: "raynor-2",
            questionText: "Group optional field test",
            metadata: { fixture: "question-group-test" },
            options: [{ id: "A", text: "Answer" }],
            correctOptionIds: ["A"],
          },
        ],
      });

      expect(decoded.id).toBeUndefined();
      expect(decoded.metadata).toBeUndefined();
    }),
  );

  it.effect("accepts optional group metadata when present", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(QuestionGroup)({
        metadata: { split: "train", version: "v1" },
        source: {
          provider: "raynormaritime",
        },
        questions: [
          {
            id: "raynor-3",
            questionText: "Group metadata test",
            metadata: { fixture: "question-group-test" },
            options: [{ id: "A", text: "Answer" }],
            correctOptionIds: ["A"],
          },
        ],
      });

      expect(decoded.metadata).toEqual({ split: "train", version: "v1" });
    }),
  );

  it.effect("accepts optional question-level source override in a group", () =>
    Effect.gen(function* () {
      const decoded = yield* S.decodeUnknown(QuestionGroup)({
        source: { provider: "raynormaritime", exam: "ALL" },
        questions: [
          {
            id: "raynor-4",
            questionText: "Question source override",
            metadata: { fixture: "question-group-test" },
            source: { provider: "raynormaritime", questionPage: "42" },
            options: [{ id: "A", text: "Answer" }],
            correctOptionIds: ["A"],
          },
        ],
      });

      expect(decoded.questions[0]?.source?.questionPage).toBe("42");
    }),
  );

  it.effect("rejects old plain array-of-questions for QuestionGroupsFromJson", () =>
    Effect.gen(function* () {
      const result = yield* S.decodeUnknown(QuestionGroupsFromJson)(
        `[
          {
            "id": "q-legacy",
            "questionText": "Legacy root shape",
            "metadata": { "fixture": "question-group-test" },
            "options": [{ "id": "A", "text": "Answer" }],
            "correctOptionIds": ["A"]
          }
        ]`,
      ).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("rejects invalid non-object group source values", () =>
    Effect.gen(function* () {
      const result = yield* S.decodeUnknown(QuestionGroup)({
        source: "raynor",
        questions: [
          {
            id: "raynor-5",
            questionText: "Invalid source type",
            metadata: { fixture: "question-group-test" },
            options: [{ id: "A", text: "Answer" }],
            correctOptionIds: ["A"],
          },
        ],
      }).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("rejects invalid question payload inside group.questions", () =>
    Effect.gen(function* () {
      const result = yield* S.decodeUnknown(QuestionGroup)({
        source: { provider: "raynormaritime" },
        questions: [
          {
            id: "raynor-6",
            questionText: "Missing correctOptionIds should fail",
            metadata: { fixture: "question-group-test" },
            options: [{ id: "A", text: "Answer" }],
          },
        ],
      }).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );
});
