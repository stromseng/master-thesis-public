// Usage:
//   bun scripts/analysis/analyze-multi-answer-questions.ts
//
// Loads all shititong dataset files (en/zh, text/vision) and flags questions
// that have multiple correct answers. Classifies each into:
//   - "garbled"         — duplicate IDs or IDs not matching any option
//   - "all_correct"     — every option is marked correct (likely broken single-answer)
//   - "plausible_multi" — a strict subset of options marked correct (may be legit)
//
// Writes a JSON report to data/analysis/.
import { mkdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { Effect } from "effect";
import * as S from "effect/Schema";
import { QuestionGroupsFromJson, type EvalQuestionGroups } from "../../evals/question-schema";
import { dataPath, evalDataPath } from "../../src/utils/repo";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Category = "garbled" | "all_correct" | "plausible_multi";

interface FlaggedQuestion {
  id: string;
  dataset: string;
  category: Category;
  categoryReason: string;
  groupId: string;
  questionType: string | null;
  questionText: string;
  options: Array<{ id: string; text: string }>;
  correctOptionIds: string[];
  /** De-duplicated correct IDs (for garbled questions) */
  uniqueCorrectIds: string[];
}

interface DatasetSummary {
  dataset: string;
  totalQuestions: number;
  singleAnswer: number;
  multiAnswer: number;
  garbled: number;
  allCorrect: number;
  plausibleMulti: number;
}

interface AnalysisOutput {
  generatedAt: string;
  summary: {
    totalQuestions: number;
    totalMultiAnswer: number;
    garbled: number;
    allCorrect: number;
    plausibleMulti: number;
  };
  datasets: DatasetSummary[];
  flaggedQuestions: FlaggedQuestion[];
}

// ---------------------------------------------------------------------------
// Dataset definitions
// ---------------------------------------------------------------------------

const DATASETS = [
  { name: "shititong-en-text", file: "shititong_english_deduped_text.json" },
  { name: "shititong-en-vision", file: "shititong_english_deduped_vision.json" },
  { name: "shititong-zh-text", file: "shititong_chinese_deduped_text.json" },
  { name: "shititong-zh-vision", file: "shititong_chinese_deduped_vision.json" },
] as const;

// ---------------------------------------------------------------------------
// Analysis logic
// ---------------------------------------------------------------------------

function classifyQuestion(
  correctOptionIds: string[],
  optionIds: string[],
): { category: Category; reason: string } {
  const optionSet = new Set(optionIds);
  const uniqueCorrect = [...new Set(correctOptionIds)];
  const hasDuplicates = uniqueCorrect.length < correctOptionIds.length;
  const hasInvalidIds = correctOptionIds.some((id) => !optionSet.has(id));

  if (hasDuplicates || hasInvalidIds) {
    const reasons: string[] = [];
    if (hasDuplicates)
      reasons.push(
        `duplicate IDs in correctOptionIds (${correctOptionIds.length} → ${uniqueCorrect.length} unique)`,
      );
    if (hasInvalidIds) {
      const invalid = correctOptionIds.filter((id) => !optionSet.has(id));
      reasons.push(`IDs not matching any option: [${invalid.join(", ")}]`);
    }
    return { category: "garbled", reason: reasons.join("; ") };
  }

  if (uniqueCorrect.length === optionIds.length) {
    return {
      category: "all_correct",
      reason: `All ${optionIds.length} options marked correct`,
    };
  }

  return {
    category: "plausible_multi",
    reason: `${uniqueCorrect.length} of ${optionIds.length} options marked correct: [${uniqueCorrect.join(", ")}]`,
  };
}

function analyzeDataset(
  datasetName: string,
  groups: EvalQuestionGroups,
): { flagged: FlaggedQuestion[]; summary: DatasetSummary } {
  const flagged: FlaggedQuestion[] = [];
  let totalQuestions = 0;
  let singleAnswer = 0;

  for (const group of groups) {
    for (const question of group.questions) {
      totalQuestions++;
      if (question.correctOptionIds.length <= 1) {
        singleAnswer++;
        continue;
      }

      const optionIds = question.options.map((o) => o.id);
      const { category, reason } = classifyQuestion([...question.correctOptionIds], optionIds);
      const metadata = question.metadata as Record<string, unknown> | undefined;

      flagged.push({
        id: question.id,
        dataset: datasetName,
        category,
        categoryReason: reason,
        groupId: group.id ?? "unknown",
        questionType: (metadata?.questionType as string) ?? null,
        questionText: question.questionText,
        options: question.options.map((o) => ({ id: o.id, text: o.text })),
        correctOptionIds: [...question.correctOptionIds],
        uniqueCorrectIds: [...new Set(question.correctOptionIds)],
      });
    }
  }

  const garbled = flagged.filter((q) => q.category === "garbled").length;
  const allCorrect = flagged.filter((q) => q.category === "all_correct").length;
  const plausibleMulti = flagged.filter((q) => q.category === "plausible_multi").length;

  return {
    flagged,
    summary: {
      dataset: datasetName,
      totalQuestions,
      singleAnswer,
      multiAnswer: flagged.length,
      garbled,
      allCorrect,
      plausibleMulti,
    },
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const program = Effect.gen(function* () {
  const allFlagged: FlaggedQuestion[] = [];
  const datasetSummaries: DatasetSummary[] = [];

  for (const { name, file } of DATASETS) {
    yield* Effect.log(`Loading ${name}...`);
    const raw = yield* Effect.tryPromise({
      try: () => readFile(evalDataPath("shititong", file), "utf-8"),
      catch: (error) => new Error(`Failed to read ${file}: ${error}`),
    });

    const groups = yield* S.decodeUnknown(QuestionGroupsFromJson)(raw).pipe(
      Effect.mapError((error) => new Error(`Failed to decode ${file}: ${error}`)),
    );

    const { flagged, summary } = analyzeDataset(name, groups);
    allFlagged.push(...flagged);
    datasetSummaries.push(summary);

    yield* Effect.log(
      `  ${summary.totalQuestions} questions, ${summary.multiAnswer} multi-answer ` +
        `(${summary.garbled} garbled, ${summary.allCorrect} all-correct, ${summary.plausibleMulti} plausible)`,
    );
  }

  const output: AnalysisOutput = {
    generatedAt: new Date().toISOString(),
    summary: {
      totalQuestions: datasetSummaries.reduce((s, d) => s + d.totalQuestions, 0),
      totalMultiAnswer: allFlagged.length,
      garbled: allFlagged.filter((q) => q.category === "garbled").length,
      allCorrect: allFlagged.filter((q) => q.category === "all_correct").length,
      plausibleMulti: allFlagged.filter((q) => q.category === "plausible_multi").length,
    },
    datasets: datasetSummaries,
    flaggedQuestions: allFlagged,
  };

  const ts = new Date()
    .toISOString()
    .replaceAll(":", "-")
    .replace(/\.\d+Z$/, "Z");
  const outputPath = dataPath("analysis", `multi-answer-analysis-${ts}.json`);
  mkdirSync(dirname(outputPath), { recursive: true });

  yield* Effect.tryPromise({
    try: () => Bun.write(outputPath, `${JSON.stringify(output, null, 2)}\n`),
    catch: (e) => new Error(`Failed to write output: ${e}`),
  });

  yield* Effect.log(`\nResults written to ${outputPath}`);
  yield* Effect.log(
    `\nOverall: ${output.summary.totalMultiAnswer} multi-answer questions across all datasets`,
  );
  yield* Effect.log(`  Garbled:         ${output.summary.garbled}`);
  yield* Effect.log(`  All correct:     ${output.summary.allCorrect}`);
  yield* Effect.log(`  Plausible multi: ${output.summary.plausibleMulti}`);
});

Effect.runPromise(program);
