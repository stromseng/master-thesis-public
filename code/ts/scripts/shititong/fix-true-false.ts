/**
 * Phase 3.5: Fix True/False Questions
 *
 * This script reads all question files from Phase 3 and adds default options
 * for 判断题 (True/False) questions that have empty options.
 *
 * The website doesn't store options for True/False questions, so we add:
 * - A: 正确 (Correct/True)
 * - B: 错误 (Wrong/False)
 *
 * Usage: bun run scrape:4-fix-tf [--debug]
 */

import { readdirSync } from "node:fs";
import { join } from "node:path";
import { CONFIG, type QuestionsOutput } from "./config.ts";
import { createProgressBar, loadJson, log, saveJson } from "./utils.ts";

const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

// Default options for question types that don't store them
const DEFAULT_OPTIONS: Record<string, { A: string; B: string }> = {
  判断题: { A: "true", B: "false" }, // True/False
};

async function fixTrueFalseQuestions(): Promise<void> {
  debugLog("Starting True/False question fix...");

  // Read all question files
  let files: string[];
  try {
    files = readdirSync(CONFIG.QUESTIONS_DIR).filter((f) => f.endsWith(".json"));
    console.log(`Found ${files.length} question files`);
  } catch {
    console.error(
      "Failed to read questions directory. Please run Phase 3 first (bun run scrape:4-questions)",
    );
    process.exit(1);
  }

  if (files.length === 0) {
    console.log("No question files found. Nothing to fix.");
    return;
  }

  let totalFixed = 0;
  let filesModified = 0;
  const typeStats: Record<string, number> = {};
  const progress = createProgressBar(files.length, "fixed");

  for (let i = 0; i < files.length; i++) {
    const file = files[i]!;
    const filePath = join(CONFIG.QUESTIONS_DIR, file);

    try {
      const data = loadJson<QuestionsOutput>(filePath);
      let fileModified = false;

      for (const question of data.questions) {
        // Check if this question type needs default options
        const questionType = question.type;
        if (!questionType || !DEFAULT_OPTIONS[questionType]) {
          continue;
        }

        // Check if options are empty
        const hasOptions = question.options && Object.keys(question.options).length > 0;
        if (hasOptions) {
          continue;
        }

        // Add default options
        question.options = { ...DEFAULT_OPTIONS[questionType] };
        totalFixed++;
        fileModified = true;

        // Track stats
        typeStats[questionType] = (typeStats[questionType] || 0) + 1;

        debugLog(`Fixed: ${file} - Q${question.id} (${questionType})`);
      }

      // Save file if modified
      if (fileModified) {
        saveJson(filePath, data, false);
        filesModified++;
      }
    } catch (error) {
      console.error(`Failed to process file: ${file}`, error);
    }

    progress.update(i + 1, totalFixed);
  }

  progress.finish();

  // Print summary
  console.log(`\nFixed ${totalFixed} questions in ${filesModified} files`);
  if (Object.keys(typeStats).length > 0) {
    console.log("\nBy question type:");
    for (const [type, count] of Object.entries(typeStats).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${type}: ${count}`);
    }
  }
}

// Run the script
fixTrueFalseQuestions()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
