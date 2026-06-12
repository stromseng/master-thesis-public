/**
 * Phase 4: Merge and Deduplicate Questions
 *
 * This script reads all question files from Phase 3, merges them,
 * removes duplicates, and generates summary statistics.
 *
 * Usage: bun run scrape:5-merge [--debug]
 */

import { readdirSync } from "node:fs";
import { join } from "node:path";
import { CONFIG, type Question, type QuestionsOutput } from "./config.ts";
import { createProgressBar, loadJson, log, logError, saveJson } from "./utils.ts";

const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

interface MergedOutput {
  exported_at: string;
  source: string;
  statistics: {
    total_files: number;
    total_questions_before_dedup: number;
    total_questions_after_dedup: number;
    duplicates_removed: number;
    empty_options_removed: number;
    by_type: Record<string, number>;
  };
  questions: Question[];
}

function hashQuestion(q: Question): string {
  // Use original_id if available, otherwise hash the question text
  if (q.original_id) {
    return `id:${q.original_id}`;
  }
  // Fallback to question text hash
  const text = (q.question || "").trim().toLowerCase();
  return `text:${text}`;
}

async function mergeQuestions(): Promise<void> {
  debugLog("Starting question merge and deduplication...");

  // Read all question files
  let files: string[];
  try {
    files = readdirSync(CONFIG.QUESTIONS_DIR).filter((f) => f.endsWith(".json"));
    console.log(`Found ${files.length} question files`);
  } catch (error) {
    logError(
      "Failed to read questions directory. Please run Phase 3 first (bun run scrape:questions)",
      error,
    );
    process.exit(1);
  }

  if (files.length === 0) {
    console.log("No question files found. Nothing to merge.");
    return;
  }

  const allQuestions: Question[] = [];
  const seenHashes = new Set<string>();
  let totalBeforeDedup = 0;
  let duplicatesRemoved = 0;
  let emptyOptionsRemoved = 0;
  const typeCount: Record<string, number> = {};
  const progress = createProgressBar(files.length, "unique");

  for (let i = 0; i < files.length; i++) {
    const file = files[i]!;
    const filePath = join(CONFIG.QUESTIONS_DIR, file);

    try {
      const data = loadJson<QuestionsOutput>(filePath);

      // Build source metadata for questions from this file
      const source: Question["source"] = {
        exam_url: data.exam_link,
        ...(data.exam_title && { exam_title: data.exam_title }),
        ...(data.found_by_keywords && { found_by_keywords: data.found_by_keywords }),
      };

      for (const question of data.questions) {
        totalBeforeDedup++;

        // Skip questions with no options
        if (!question.options || Object.keys(question.options).length === 0) {
          emptyOptionsRemoved++;
          continue;
        }

        const hash = hashQuestion(question);

        if (seenHashes.has(hash)) {
          duplicatesRemoved++;
          continue;
        }

        seenHashes.add(hash);

        // Add source metadata to the question
        allQuestions.push({
          ...question,
          source,
        });

        // Count by type
        const type = question.type || "unknown";
        typeCount[type] = (typeCount[type] || 0) + 1;
      }
    } catch (error) {
      logError(`Failed to read file: ${file}`, error);
    }

    progress.update(i + 1, allQuestions.length);
  }

  progress.finish();

  // Re-number questions sequentially
  allQuestions.forEach((q, idx) => {
    q.id = idx + 1;
  });

  // Build output
  const output: MergedOutput = {
    exported_at: new Date().toISOString(),
    source: "shititong.cn",
    statistics: {
      total_files: files.length,
      total_questions_before_dedup: totalBeforeDedup,
      total_questions_after_dedup: allQuestions.length,
      duplicates_removed: duplicatesRemoved,
      empty_options_removed: emptyOptionsRemoved,
      by_type: typeCount,
    },
    questions: allQuestions,
  };

  // Save merged file
  const outputPath = join(CONFIG.OUTPUT_DIR, "all-questions.json");
  saveJson(outputPath, output, DEBUG);

  // Print summary
  console.log(`\nMerged ${files.length} files: ${allQuestions.length} unique questions`);
  console.log(`Removed: ${duplicatesRemoved} duplicates, ${emptyOptionsRemoved} empty`);
  debugLog("\nBy question type:");
  for (const [type, count] of Object.entries(typeCount).sort((a, b) => b[1] - a[1])) {
    debugLog(`  ${type}: ${count}`);
  }
  debugLog(`\nSaved to: ${outputPath}`);
}

// Run the script
mergeQuestions()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
