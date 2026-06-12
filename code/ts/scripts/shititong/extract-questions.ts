/**
 * Phase 3: Extract Questions from Each Chapter
 *
 * This script reads the chapter links from Phase 2 and extracts
 * all questions from each chapter page.
 *
 * Usage: bun run scrape:4-questions [--debug]
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "playwright";
import {
  CONFIG,
  type ChapterLink,
  type ChapterLinksOutput,
  type Question,
  type QuestionsOutput,
} from "./config.ts";
import {
  createBrowserContext,
  createProgressBar,
  delay,
  handleCollectPopup,
  loadJson,
  log,
  logError,
  saveJson,
} from "./utils.ts";

const CONCURRENT_TABS = 1;
const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

interface RawQuestion {
  fid?: string;
  tigan?: string;
  a?: string;
  b?: string;
  c?: string;
  d?: string;
  e?: string;
  f?: string;
  daan?: string;
  tixing?: string;
  koujue?: string;
  jiexidata?: string;
  nanyi?: string;
  fenid?: string;
}

async function extractQuestions(): Promise<void> {
  debugLog("Starting question extraction...");

  // Load chapter links from Phase 2
  let chapterLinks: ChapterLinksOutput;
  try {
    chapterLinks = loadJson<ChapterLinksOutput>(CONFIG.CHAPTER_LINKS_FILE);
    debugLog(
      `Loaded ${chapterLinks.total_chapters} chapters from ${chapterLinks.total_exams} exams`,
    );
  } catch (error) {
    logError(
      "Failed to load chapter links. Please run Phase 2 first (bun run scrape:chapters)",
      error,
    );
    process.exit(1);
  }

  // Flatten all chapters with their exam info
  const allChapters: Array<{
    tikuid: string;
    examUrl: string;
    title?: string;
    foundByKeywords?: string[];
    chapter: ChapterLink;
  }> = [];

  let skippedCount = 0;
  for (const exam of chapterLinks.exams) {
    for (const chapter of exam.chapters) {
      // Skip if already processed
      const outputFile = getOutputFilePath(exam.tikuid, chapter.zjid);
      if (existsSync(outputFile)) {
        debugLog(`Skipping already processed: ${exam.tikuid}/${chapter.zjid}`);
        skippedCount++;
        continue;
      }

      allChapters.push({
        tikuid: exam.tikuid,
        examUrl: exam.exam_url,
        ...(exam.title && { title: exam.title }),
        ...(exam.found_by_keywords && { foundByKeywords: exam.found_by_keywords }),
        chapter,
      });
    }
  }

  console.log(`Found ${allChapters.length} chapters to process (${skippedCount} already done)`);

  if (allChapters.length === 0) {
    console.log("All chapters already processed!");
    return;
  }

  const context = await createBrowserContext();

  // Create multiple tabs
  const pages: Page[] = [];
  for (let i = 0; i < CONCURRENT_TABS; i++) {
    pages.push(await context.newPage());
  }

  let processedCount = 0;
  let totalQuestions = 0;
  const progress = createProgressBar(allChapters.length, "questions");

  try {
    // Process in batches
    for (let i = 0; i < allChapters.length; i += CONCURRENT_TABS) {
      const batch = allChapters.slice(i, i + CONCURRENT_TABS);

      debugLog(
        `\n--- Processing batch ${Math.floor(i / CONCURRENT_TABS) + 1} (items ${i + 1}-${Math.min(i + CONCURRENT_TABS, allChapters.length)} of ${allChapters.length}) ---`,
      );

      // Process batch concurrently
      const batchPromises = batch.map((item, idx) =>
        processChapter(
          pages[idx]!,
          item.tikuid,
          item.examUrl,
          item.title,
          item.foundByKeywords,
          item.chapter,
          i + idx + 1,
          allChapters.length,
        ),
      );

      const batchResults = await Promise.all(batchPromises);

      // Count results
      for (const result of batchResults) {
        if (result > 0) {
          totalQuestions += result;
          processedCount++;
        }
      }

      progress.update(i + batch.length, totalQuestions);
      debugLog(`Batch complete: ${totalQuestions} total questions extracted`);

      // Small delay between batches
      if (i + CONCURRENT_TABS < allChapters.length) {
        await delay(CONFIG.REQUEST_DELAY_MS);
      }
    }

    progress.finish();
    console.log(`Processed ${processedCount} chapters, extracted ${totalQuestions} questions`);
  } catch (error) {
    logError("Fatal error during extraction", error);
    throw error;
  } finally {
    await context.close();
  }
}

async function processChapter(
  page: Page,
  tikuid: string,
  examUrl: string,
  title: string | undefined,
  foundByKeywords: string[] | undefined,
  chapter: ChapterLink,
  index: number,
  total: number,
): Promise<number> {
  debugLog(`[${index}/${total}] Processing: ${title || tikuid} - ${chapter.zjid}`);

  try {
    const questions = await extractQuestionsFromChapter(page, chapter.url);

    if (questions.length === 0) {
      debugLog(`[${index}/${total}] No questions found`);
      return 0;
    }

    // Save to file
    const output: QuestionsOutput = {
      exported_at: new Date().toISOString(),
      tikuid,
      zjid: chapter.zjid,
      exam_link: examUrl,
      ...(title && { exam_title: title }),
      ...(foundByKeywords && { found_by_keywords: foundByKeywords }),
      chapter_link: chapter.url,
      total_questions: questions.length,
      source: "shititong.cn",
      questions,
    };

    const outputFile = getOutputFilePath(tikuid, chapter.zjid);
    saveJson(outputFile, output, DEBUG);

    debugLog(`[${index}/${total}] Extracted ${questions.length} questions`);
    return questions.length;
  } catch (error) {
    logError(`[${index}/${total}] Failed to process chapter: ${chapter.zjid}`, error);
    return 0;
  }
}

async function extractQuestionsFromChapter(page: Page, chapterUrl: string): Promise<Question[]> {
  // Navigate to page (page=1 is fine, window.list contains ALL questions)
  await page.goto(chapterUrl, {
    timeout: CONFIG.PAGE_LOAD_TIMEOUT_MS,
    waitUntil: "domcontentloaded",
  });

  // Handle collect popup if it appears
  await handleCollectPopup(page);

  // Wait for questions to load (check multiple locations)
  try {
    await page.waitForFunction(
      `
      (() => {
        const locations = [
          () => window.list,
          () => window.parent?.list,
          () => window.frames?.[0]?.list,
          () => window.parent?.frames?.[0]?.list,
          () => document.querySelector("iframe")?.contentWindow?.list,
        ];

        for (const loc of locations) {
          try {
            const list = loc();
            if (list && Array.isArray(list) && list.length > 0 && list[0].tigan) {
              return true;
            }
          } catch {}
        }
        return false;
      })()
    `,
      { timeout: 10000 },
    );
  } catch {
    debugLog("  No questions found");
    return [];
  }

  // Extract questions from wherever they are
  // Note: window.list contains ALL questions at once, no pagination needed
  const rawQuestions: RawQuestion[] = await page.evaluate(`
    (() => {
      const locations = [
        () => window.list,
        () => window.parent?.list,
        () => window.frames?.[0]?.list,
        () => window.parent?.frames?.[0]?.list,
        () => document.querySelector("iframe")?.contentWindow?.list,
      ];

      for (const loc of locations) {
        try {
          const list = loc();
          if (list && Array.isArray(list) && list.length > 0 && list[0].tigan) {
            return list;
          }
        } catch {}
      }
      return [];
    })()
  `);

  // Transform questions
  const questions = rawQuestions.map((q, idx) => transformQuestion(q, idx + 1));

  return questions;
}

function transformQuestion(raw: RawQuestion, id: number): Question {
  const options: Question["options"] = {};

  if (raw.a?.trim()) options.A = raw.a.trim();
  if (raw.b?.trim()) options.B = raw.b.trim();
  if (raw.c?.trim()) options.C = raw.c.trim();
  if (raw.d?.trim()) options.D = raw.d.trim();
  if (raw.e?.trim()) options.E = raw.e.trim();
  if (raw.f?.trim()) options.F = raw.f.trim();

  return {
    id,
    question: raw.tigan?.trim() || null,
    options,
    answer: raw.daan || null,
    type: raw.tixing || null,
    hint: raw.koujue || null,
    explanation: raw.jiexidata || null,
    difficulty: raw.nanyi || null,
    category_id: raw.fenid || null,
    original_id: raw.fid || null,
  };
}

function getOutputFilePath(tikuid: string, zjid: string): string {
  return join(CONFIG.QUESTIONS_DIR, `${tikuid}_${zjid}.json`);
}

// Run the script
extractQuestions()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
