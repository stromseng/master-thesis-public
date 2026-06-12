import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Command } from "@effect/cli";
import { NodeContext } from "@effect/platform-node";
import { describe, expect, it } from "@effect/vitest";
import { Effect, Option } from "effect";
import * as S from "effect/Schema";
import type { EvalQuestion, EvalQuestionGroups } from "../../evals/question-schema";
import {
  OpenEndedQuestionGroupsFromJson,
  type OpenEndedQuestionGroups,
} from "../../evals/open-ended-question-schema";
import {
  SAVE_EVERY,
  TEXT_DATASET_IDS,
  buildAnalysisSummary,
  buildCheckpoint,
  buildOpenEndedDataset,
  extractConversionCandidate,
  hasAnswerLeakage,
  makeCommand,
  questionKeyOf,
  resolveDatasetRunPaths,
  resolveLatestResumeStatePath,
  selectDatasets,
  selectQuestionSubset,
  validateResumeCheckpoint,
  type CliArgs,
  type ConvertedEntry,
  type ProcessedEntry,
} from "../../scripts/analysis/convert-text-datasets-to-open-ended";

const makeQuestion = (overrides: Partial<EvalQuestion> = {}): EvalQuestion => ({
  id: "question-1",
  questionText: "Which vessel is the stand-on vessel in a crossing situation?",
  metadata: { topic: "colregs" },
  images: [],
  options: [
    { id: "A", text: "The vessel with the other on her starboard side", images: [] },
    { id: "B", text: "The vessel with the other on her port side", images: [] },
    { id: "C", text: "The faster vessel", images: [] },
  ],
  correctOptionIds: ["B"],
  ...overrides,
});

const makeGroup = (questions: ReadonlyArray<EvalQuestion>): EvalQuestionGroups[number] => ({
  id: "group-1",
  metadata: { source: "test" },
  source: { file: "test.json" },
  questions,
});

class TestSetupError extends S.TaggedError<TestSetupError>()("TestSetupError", {
  step: S.String,
  cause: S.Defect,
}) {}

const parseCliArgs = (args: ReadonlyArray<string>) =>
  Effect.gen(function* () {
    let parsedConfig: CliArgs | undefined;
    const cli = Command.run(
      makeCommand((config) =>
        Effect.sync(() => {
          parsedConfig = config;
        }),
      ),
      {
        name: "test",
        version: "0.0.0",
      },
    );

    yield* cli(args).pipe(Effect.provide(NodeContext.layer));

    expect(parsedConfig).toBeDefined();
    return parsedConfig!;
  });

const expectConvertedEntry = (entry: ProcessedEntry): ConvertedEntry => {
  expect(entry.status).toBe("converted");
  if (entry.status !== "converted") {
    throw new Error("Expected a converted entry");
  }
  return entry;
};

describe("Effect CLI parsing", () => {
  it.effect("parses defaults", () =>
    Effect.gen(function* () {
      const config = yield* parseCliArgs(["node", "test"]);

      expect(config.dataset).toEqual([]);
      expect(config.limit).toBe(0);
      expect(config.concurrency).toBe(4);
      expect(Option.isNone(config.outputDir)).toBe(true);
      expect(Option.isNone(config.reportDir)).toBe(true);
      expect(config.resume).toBe(false);
      expect(config.dryRun).toBe(false);
    }),
  );

  it.effect("parses overrides", () =>
    Effect.gen(function* () {
      const config = yield* parseCliArgs([
        "node",
        "test",
        "--dataset",
        "crewcn",
        "--dataset",
        "pei2024-uk",
        "--limit",
        "25",
        "--concurrency",
        "8",
        "--output-dir",
        "data/custom-output",
        "--report-dir",
        "data/custom-report",
        "--resume",
        "--dry-run",
      ]).pipe(Effect.provide(NodeContext.layer));

      expect(config.dataset).toEqual(["crewcn", "pei2024-uk"]);
      expect(config.limit).toBe(25);
      expect(config.concurrency).toBe(8);
      expect(Option.getOrUndefined(config.outputDir)).toBe("data/custom-output");
      expect(Option.getOrUndefined(config.reportDir)).toBe("data/custom-report");
      expect(config.resume).toBe(true);
      expect(config.dryRun).toBe(true);
    }),
  );
});

describe("dataset selection", () => {
  it.effect("defaults to the text-only registry only", () =>
    Effect.gen(function* () {
      const datasets = yield* selectDatasets([]);
      const datasetIds = datasets.map((dataset) => dataset.id);

      expect(datasetIds).toEqual(TEXT_DATASET_IDS);
      expect(datasetIds).not.toContain("navreas-scene-understanding");
      expect(datasetIds).not.toContain("raynor-multimodal-v2");
      expect(datasetIds).not.toContain("us-coast-guard-multimodal-v2");
    }),
  );
});

describe("candidate extraction", () => {
  it("maps single-answer MCQs into correct and incorrect answer sets", () => {
    const sourceQuestion: Parameters<typeof extractConversionCandidate>[0] = {
      sourceDataset: "pei2024-uk",
      groupId: "group-1",
      groupMetadata: {},
      question: makeQuestion(),
    };

    const result = extractConversionCandidate(sourceQuestion);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.correctAnswers).toEqual(["The vessel with the other on her port side"]);
    expect(result.value.incorrectAnswers).toEqual([
      "The vessel with the other on her starboard side",
      "The faster vessel",
    ]);
  });

  it("maps multi-answer MCQs into correct and incorrect answer sets", () => {
    const sourceQuestion: Parameters<typeof extractConversionCandidate>[0] = {
      sourceDataset: "crewcn",
      groupId: "group-1",
      groupMetadata: {},
      question: makeQuestion({
        id: "question-2",
        correctOptionIds: ["A", "C"],
      }),
    };

    const result = extractConversionCandidate(sourceQuestion);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.correctAnswers).toEqual([
      "The vessel with the other on her starboard side",
      "The faster vessel",
    ]);
    expect(result.value.incorrectAnswers).toEqual(["The vessel with the other on her port side"]);
  });

  it("expands Roman numeral selector options into source statement text", () => {
    const sourceQuestion: Parameters<typeof extractConversionCandidate>[0] = {
      sourceDataset: "pei2024-zh",
      groupId: "group-1",
      groupMetadata: {},
      question: makeQuestion({
        id: "pei2024application-zh-0990",
        questionText:
          "下列说法正确的是Ⅰ、船长上驾驶台，就说明船长开始对航行值班负责Ⅱ、船长直接发出操船口令，说明船长已经声明亲自指挥Ⅲ、船长上驾驶台说明已解除驾驶员的值班责任",
        options: [
          { id: "A", text: "Ⅰ", images: [] },
          { id: "B", text: "Ⅱ", images: [] },
          { id: "C", text: "Ⅰ、Ⅲ", images: [] },
        ],
        correctOptionIds: ["B"],
      }),
    };

    const result = extractConversionCandidate(sourceQuestion);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.correctAnswers).toEqual(["船长直接发出操船口令，说明船长已经声明亲自指挥"]);
    expect(result.value.incorrectAnswers).toEqual([
      "船长上驾驶台，就说明船长开始对航行值班负责",
      "船长上驾驶台，就说明船长开始对航行值班负责; 船长上驾驶台说明已解除驾驶员的值班责任",
    ]);
  });

  it("rejects correct selector options when the source statements are missing", () => {
    const sourceQuestion: Parameters<typeof extractConversionCandidate>[0] = {
      sourceDataset: "pei2024-zh",
      groupId: "group-1",
      groupMetadata: {},
      question: makeQuestion({
        id: "pei2024application-zh-truncated",
        questionText: "关于《国际海上避碰规则》适用的船舶、下列说法不正确的是 Ⅰ、指的是在航船",
        options: [
          { id: "A", text: "Ⅰ、Ⅱ", images: [] },
          { id: "B", text: "Ⅰ", images: [] },
        ],
        correctOptionIds: ["A"],
      }),
    };

    const result = extractConversionCandidate(sourceQuestion);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.failure.reason).toBe("unresolved_reference_option");
  });
});

describe("answer leakage detection", () => {
  it("rejects rewritten questions that leak the answer", () => {
    expect(
      hasAnswerLeakage(
        "What is the required vertical safe working load marking on the derrick boom?",
        "safe working load",
      ),
    ).toBe(true);
  });
});

describe("dataset and checkpoint output", () => {
  const dataset = selectQuestionSubset(
    {
      id: "pei2024-uk",
      family: "PEI2024",
      language: "EN",
      loadGroups: () => Effect.succeed([]),
    },
    [
      makeGroup([
        makeQuestion(),
        makeQuestion({
          id: "question-2",
          questionText: "What is the purpose of a spring line when berthing?",
          options: [
            { id: "A", text: "To control fore-and-aft movement", images: [] },
            { id: "B", text: "To increase draft", images: [] },
          ],
          correctOptionIds: ["A"],
        }),
      ]),
    ],
    0,
  );

  const processedEntries: ProcessedEntry[] = [
    {
      status: "converted",
      questionKey: questionKeyOf("pei2024-uk", "question-1"),
      sourceDataset: "pei2024-uk",
      groupId: "group-1",
      questionId: "question-1",
      sourceQuestionText: "Which vessel is the stand-on vessel in a crossing situation?",
      conversionReason: "Removed the options and kept the crossing context.",
      convertedQuestion: {
        id: "question-1",
        questionText:
          "In a crossing situation between two power-driven vessels, which vessel is the stand-on vessel?",
        referenceCorrectAnswers: ["The vessel with the other on her port side"],
        referenceIncorrectAnswers: [
          "The vessel with the other on her starboard side",
          "The faster vessel",
        ],
        metadata: {},
        images: [],
      },
    },
    {
      status: "skipped",
      questionKey: questionKeyOf("pei2024-uk", "question-2"),
      sourceDataset: "pei2024-uk",
      groupId: "group-1",
      questionId: "question-2",
      sourceQuestionText: "What is the purpose of a spring line when berthing?",
      reason: "non_convertible",
      detail: "The options define the framing too narrowly to rewrite safely.",
    },
  ];

  it("excludes skipped questions from the dataset and keeps aggregate counts in the analysis summary", () => {
    const openEndedDataset = buildOpenEndedDataset(
      dataset.groups,
      processedEntries,
      dataset.dataset.id,
    );
    const analysisSummary = buildAnalysisSummary(dataset, processedEntries, {
      runTimestamp: "2026-04-09T13-00-00Z",
      limitPerDataset: 0,
      concurrency: 4,
      evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
    });

    expect(openEndedDataset[0]?.questions).toHaveLength(1);
    expect(openEndedDataset[0]?.questions[0]?.id).toBe("question-1");
    expect(analysisSummary.counts.converted).toBe(1);
    expect(analysisSummary.counts.skipped).toBe(1);
    expect(analysisSummary.skippedByReason).toEqual({ non_convertible: 1 });
  });

  it("keeps detailed entries only in checkpoint state", () => {
    const checkpoint = buildCheckpoint(dataset, processedEntries, {
      runTimestamp: "2026-04-09T13-00-00Z",
      limitPerDataset: 0,
      concurrency: 4,
      evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
    });

    expect(checkpoint.entries).toHaveLength(2);
    expect(checkpoint.entries[0]?.questionId).toBe("question-1");
  });

  it("keeps only minimal source traceability metadata", () => {
    const convertedEntry = expectConvertedEntry(processedEntries[0]!);
    const tracedQuestion = {
      ...convertedEntry,
      convertedQuestion: {
        ...convertedEntry.convertedQuestion,
        metadata: {
          sourceDataset: "pei2024-uk",
          sourceGroupId: "group-1",
          sourceQuestionId: "question-1",
          conversion: {
            reason: "Removed the options and kept the crossing context.",
            modelId: "openai/gpt-oss-120b",
          },
        },
      },
    } satisfies ConvertedEntry;

    const openEndedDataset = buildOpenEndedDataset(
      dataset.groups,
      [tracedQuestion, processedEntries[1]!],
      dataset.dataset.id,
    );
    const metadata = openEndedDataset[0]?.questions[0]?.metadata as {
      sourceDataset?: string;
      sourceGroupId?: string;
      sourceQuestionId?: string;
      conversion?: { modelId: string };
      sourceMcq?: unknown;
      sourceQuestionText?: unknown;
    };

    expect(metadata.sourceDataset).toBe("pei2024-uk");
    expect(metadata.sourceGroupId).toBe("group-1");
    expect(metadata.sourceQuestionId).toBe("question-1");
    expect(metadata.conversion?.modelId).toBe("openai/gpt-oss-120b");
    expect(metadata.sourceMcq).toBeUndefined();
    expect(metadata.sourceQuestionText).toBeUndefined();
  });

  it("exports SAVE_EVERY as 10", () => {
    expect(SAVE_EVERY).toBe(10);
  });

  it.effect("generated open-ended JSON decodes with the shared schema", () =>
    Effect.gen(function* () {
      const openEndedDataset = buildOpenEndedDataset(
        dataset.groups,
        processedEntries,
        dataset.dataset.id,
      );
      const encoded = yield* S.encode(OpenEndedQuestionGroupsFromJson)(openEndedDataset);
      const decoded = yield* S.decodeUnknown(OpenEndedQuestionGroupsFromJson)(encoded);

      expect((decoded as OpenEndedQuestionGroups)[0]?.questions).toHaveLength(1);
      expect(
        (decoded as OpenEndedQuestionGroups)[0]?.questions[0]?.referenceCorrectAnswers,
      ).toEqual(["The vessel with the other on her port side"]);
    }),
  );
});

describe("resume checkpoint validation", () => {
  it.effect("accepts matching checkpoint metadata", () =>
    validateResumeCheckpoint(
      "/tmp/pei2024-uk.conversion.json",
      {
        generatedAt: "2026-04-09T00:00:00.000Z",
        runTimestamp: "2026-04-09T13-00-00Z",
        sourceDataset: "pei2024-uk",
        limitPerDataset: 100,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
      {
        sourceDataset: "pei2024-uk",
        limitPerDataset: 100,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
    ),
  );

  it.effect("accepts a higher limit on resume", () =>
    validateResumeCheckpoint(
      "/tmp/pei2024-uk.conversion.json",
      {
        generatedAt: "2026-04-09T00:00:00.000Z",
        runTimestamp: "2026-04-09T13-00-00Z",
        sourceDataset: "pei2024-uk",
        limitPerDataset: 100,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
      {
        sourceDataset: "pei2024-uk",
        limitPerDataset: 250,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
    ),
  );

  it.effect("accepts no-limit on resume", () =>
    validateResumeCheckpoint(
      "/tmp/pei2024-uk.conversion.json",
      {
        generatedAt: "2026-04-09T00:00:00.000Z",
        runTimestamp: "2026-04-09T13-00-00Z",
        sourceDataset: "pei2024-uk",
        limitPerDataset: 100,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
      {
        sourceDataset: "pei2024-uk",
        limitPerDataset: 0,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
    ),
  );

  it.effect("accepts different concurrency on resume", () =>
    validateResumeCheckpoint(
      "/tmp/pei2024-uk.conversion.json",
      {
        generatedAt: "2026-04-09T00:00:00.000Z",
        runTimestamp: "2026-04-09T13-00-00Z",
        sourceDataset: "pei2024-uk",
        limitPerDataset: 100,
        concurrency: 4,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
      {
        sourceDataset: "pei2024-uk",
        limitPerDataset: 100,
        concurrency: 2,
        saveEvery: 10,
        evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
      },
    ),
  );

  it.effect("rejects mismatched checkpoint metadata", () =>
    Effect.gen(function* () {
      const result = yield* validateResumeCheckpoint(
        "/tmp/pei2024-uk.conversion.json",
        {
          generatedAt: "2026-04-09T00:00:00.000Z",
          runTimestamp: "2026-04-09T13-00-00Z",
          sourceDataset: "pei2024-uk",
          limitPerDataset: 100,
          concurrency: 4,
          saveEvery: 10,
          evalModel: { modelId: "openai/gpt-oss-120b", provider: "test" },
        },
        {
          sourceDataset: "pei2024-uk",
          limitPerDataset: 50,
          concurrency: 8,
          saveEvery: 10,
          evalModel: { modelId: "moonshotai/Kimi-K2.5", provider: "test" },
        },
      ).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );
});

describe("resume state path resolution", () => {
  it.effect("fails when no matching timestamped state exists", () =>
    Effect.gen(function* () {
      const result = yield* resolveLatestResumeStatePath(
        "/tmp/does-not-exist-open-ended",
        "pei2024-uk",
      ).pipe(Effect.either);

      expect(result._tag).toBe("Left");
    }),
  );

  it.effect("starts a fresh timestamped run when resuming a dataset with no prior state", () =>
    Effect.gen(function* () {
      const reportDir = `${tmpdir()}/open-ended-no-state-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const outputDir = `${tmpdir()}/open-ended-output-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const runTimestamp = "2026-04-10T19-00-00Z";

      const resolved = yield* resolveDatasetRunPaths(
        outputDir,
        reportDir,
        "pei2024-zh",
        true,
        runTimestamp,
      );

      expect(resolved.runTimestamp).toBe(runTimestamp);
      expect(resolved.outputPath).toBe(
        `${outputDir}/pei2024-zh-${runTimestamp}.open-ended.eval.json`,
      );
      expect(resolved.analysisPath).toBe(`${reportDir}/pei2024-zh-${runTimestamp}.conversion.json`);
      expect(resolved.statePath).toBe(
        `${reportDir}/pei2024-zh-${runTimestamp}.conversion.state.json`,
      );
    }),
  );

  it.effect("reuses the existing run timestamp when resuming a dataset with prior state", () =>
    Effect.gen(function* () {
      const reportDir = `${tmpdir()}/open-ended-existing-state-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const outputDir = `${tmpdir()}/open-ended-output-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const previousRunTimestamp = "2026-04-09T11-52-35Z";

      yield* Effect.tryPromise({
        try: () =>
          mkdir(reportDir, { recursive: true }).then(() =>
            writeFile(
              join(reportDir, `pei2024-uk-${previousRunTimestamp}.conversion.state.json`),
              "{}",
              "utf8",
            ),
          ),
        catch: (cause) =>
          new TestSetupError({
            step: "create resume state fixture",
            cause,
          }),
      });

      const resolved = yield* resolveDatasetRunPaths(
        outputDir,
        reportDir,
        "pei2024-uk",
        true,
        "2026-04-10T19-00-00Z",
      );

      expect(resolved.runTimestamp).toBe(previousRunTimestamp);
      expect(resolved.outputPath).toBe(
        `${outputDir}/pei2024-uk-${previousRunTimestamp}.open-ended.eval.json`,
      );
      expect(resolved.analysisPath).toBe(
        `${reportDir}/pei2024-uk-${previousRunTimestamp}.conversion.json`,
      );
      expect(resolved.statePath).toBe(
        `${reportDir}/pei2024-uk-${previousRunTimestamp}.conversion.state.json`,
      );
    }),
  );
});
