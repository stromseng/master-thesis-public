/**
 * Phase 5: Split Questions by Language
 *
 * This script reads all-questions.json and splits questions into
 * Chinese and English based on character detection.
 *
 * Usage: bun run scrape:6-split [--debug]
 */

import { join } from "node:path";
import { CONFIG, type Question } from "./config.ts";
import { createProgressBar, loadJson, log, saveJson } from "./utils.ts";

const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

interface MergedInput {
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

interface SplitOutput {
  exported_at: string;
  source: string;
  language: "chinese" | "english" | "mixed";
  statistics: {
    total_questions: number;
    by_type: Record<string, number>;
  };
  questions: Question[];
}

/**
 * Check if a character is in the CJK Unicode ranges (Chinese characters)
 * Main ranges:
 * - CJK Unified Ideographs: U+4E00 to U+9FFF
 * - CJK Unified Ideographs Extension A: U+3400 to U+4DBF
 * - CJK Unified Ideographs Extension B-F: U+20000 to U+2FA1F
 * - CJK Compatibility Ideographs: U+F900 to U+FAFF
 */
function isChinese(char: string): boolean {
  const code = char.charCodeAt(0);
  return (
    (code >= 0x4e00 && code <= 0x9fff) || // CJK Unified Ideographs
    (code >= 0x3400 && code <= 0x4dbf) || // CJK Extension A
    (code >= 0xf900 && code <= 0xfaff) // CJK Compatibility
  );
}

/**
 * Check if text contains any Chinese characters
 */
function containsChinese(text: string | null): boolean {
  if (!text) return false;
  for (const char of text) {
    if (isChinese(char)) return true;
  }
  return false;
}

/**
 * Check if a character is an English letter (A-Z, a-z)
 */
function isEnglishLetter(char: string): boolean {
  const code = char.charCodeAt(0);
  return (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a);
}

/**
 * Strip HTML tags from text to get only the content
 * Only matches actual HTML tags (starting with letter or /), not comparisons like <70
 */
function stripHtmlTags(text: string): string {
  return text.replace(/<\/?[a-zA-Z][^>]*>/g, "");
}

/**
 * Calculate the percentage of Chinese characters vs English letters in options (0-1)
 * Only counts actual letters (Chinese or English), ignores numbers/punctuation/symbols
 * Also strips HTML tags before counting
 */
function getChineseRatioInOptions(question: Question): number {
  if (!question.options) return 0;

  let chineseChars = 0;
  let englishChars = 0;

  for (const value of Object.values(question.options)) {
    if (!value) continue;
    const cleanValue = stripHtmlTags(value);
    for (const char of cleanValue) {
      if (isChinese(char)) {
        chineseChars++;
      } else if (isEnglishLetter(char)) {
        englishChars++;
      }
      // Ignore numbers, punctuation, symbols, whitespace
    }
  }

  const totalLetters = chineseChars + englishChars;
  return totalLetters > 0 ? chineseChars / totalLetters : 0;
}

const LOWER_THRESHOLD = 0.4; // Below 30% Chinese = english
const UPPER_THRESHOLD = 0.6; // Above 70% Chinese = chinese

/**
 * Classify a question by language:
 * - "chinese": Question text contains Chinese OR options are >=70% Chinese
 * - "mixed": English question + options are 30-70% Chinese (both languages present)
 * - "english": English question + options are <30% Chinese (>=70% English)
 */
function classifyQuestion(question: Question): "chinese" | "mixed" | "english" {
  const questionHasChinese = containsChinese(question.question);

  if (questionHasChinese) {
    return "chinese";
  }

  const chineseRatio = getChineseRatioInOptions(question);

  if (chineseRatio >= UPPER_THRESHOLD) {
    return "chinese"; // Options are predominantly Chinese
  } else if (chineseRatio >= LOWER_THRESHOLD) {
    return "mixed"; // True mix: 30-70% Chinese
  }

  return "english"; // Options are predominantly English (<30% Chinese)
}

async function splitByLanguage(): Promise<void> {
  debugLog("Starting language split...");

  // Load all-questions.json
  const inputPath = join(CONFIG.OUTPUT_DIR, "all-questions.json");
  let data: MergedInput;

  try {
    data = loadJson<MergedInput>(inputPath);
    console.log(`Loaded ${data.questions.length} questions from all-questions.json`);
  } catch {
    console.error(
      "Failed to load all-questions.json. Please run Phase 4 first (bun run scrape:5-merge)",
    );
    process.exit(1);
  }

  const chineseQuestions: Question[] = [];
  const mixedQuestions: Question[] = [];
  const englishQuestions: Question[] = [];
  const chineseTypeCount: Record<string, number> = {};
  const mixedTypeCount: Record<string, number> = {};
  const englishTypeCount: Record<string, number> = {};

  const progress = createProgressBar(data.questions.length, "processed");

  for (let i = 0; i < data.questions.length; i++) {
    const question = data.questions[i]!;
    const classification = classifyQuestion(question);
    const type = question.type || "unknown";

    if (classification === "chinese") {
      chineseQuestions.push(question);
      chineseTypeCount[type] = (chineseTypeCount[type] || 0) + 1;
    } else if (classification === "mixed") {
      mixedQuestions.push(question);
      mixedTypeCount[type] = (mixedTypeCount[type] || 0) + 1;
    } else {
      englishQuestions.push(question);
      englishTypeCount[type] = (englishTypeCount[type] || 0) + 1;
    }

    progress.update(
      i + 1,
      chineseQuestions.length + mixedQuestions.length + englishQuestions.length,
    );
  }

  progress.finish();

  // Re-number questions sequentially in each output
  chineseQuestions.forEach((q, idx) => {
    q.id = idx + 1;
  });
  mixedQuestions.forEach((q, idx) => {
    q.id = idx + 1;
  });
  englishQuestions.forEach((q, idx) => {
    q.id = idx + 1;
  });

  // Build outputs
  const timestamp = new Date().toISOString();

  const chineseOutput: SplitOutput = {
    exported_at: timestamp,
    source: "shititong.cn",
    language: "chinese",
    statistics: {
      total_questions: chineseQuestions.length,
      by_type: chineseTypeCount,
    },
    questions: chineseQuestions,
  };

  const mixedOutput: SplitOutput = {
    exported_at: timestamp,
    source: "shititong.cn",
    language: "mixed",
    statistics: {
      total_questions: mixedQuestions.length,
      by_type: mixedTypeCount,
    },
    questions: mixedQuestions,
  };

  const englishOutput: SplitOutput = {
    exported_at: timestamp,
    source: "shititong.cn",
    language: "english",
    statistics: {
      total_questions: englishQuestions.length,
      by_type: englishTypeCount,
    },
    questions: englishQuestions,
  };

  // Save files
  const chinesePath = join(CONFIG.OUTPUT_DIR, "chinese-questions.json");
  const mixedPath = join(CONFIG.OUTPUT_DIR, "mixed-questions.json");
  const englishPath = join(CONFIG.OUTPUT_DIR, "english-questions.json");

  saveJson(chinesePath, chineseOutput, DEBUG);
  saveJson(mixedPath, mixedOutput, DEBUG);
  saveJson(englishPath, englishOutput, DEBUG);

  // Print summary
  const total = data.questions.length;
  const chinesePercent = ((chineseQuestions.length / total) * 100).toFixed(1);
  const mixedPercent = ((mixedQuestions.length / total) * 100).toFixed(1);
  const englishPercent = ((englishQuestions.length / total) * 100).toFixed(1);

  console.log(`\nSplit ${total} questions by language:`);
  console.log(`  Chinese: ${chineseQuestions.length} (${chinesePercent}%)`);
  console.log(
    `  Mixed:   ${mixedQuestions.length} (${mixedPercent}%) - English question, Chinese options`,
  );
  console.log(`  English: ${englishQuestions.length} (${englishPercent}%)`);

  if (DEBUG) {
    console.log("\nChinese questions by type:");
    for (const [type, count] of Object.entries(chineseTypeCount).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${type}: ${count}`);
    }
    console.log("\nMixed questions by type:");
    for (const [type, count] of Object.entries(mixedTypeCount).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${type}: ${count}`);
    }
    console.log("\nEnglish questions by type:");
    for (const [type, count] of Object.entries(englishTypeCount).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${type}: ${count}`);
    }
  }

  debugLog(`\nSaved to:\n  ${chinesePath}\n  ${mixedPath}\n  ${englishPath}`);
}

// Run the script
splitByLanguage()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
