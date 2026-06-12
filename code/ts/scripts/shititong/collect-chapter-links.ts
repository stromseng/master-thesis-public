/**
 * Phase 2: Collect Chapter Links from Exam Pages
 *
 * This script reads the exam links from Phase 1 and extracts
 * all chapter/section links from each exam page.
 *
 * Usage: bun run scrape:3-chapters [--debug]
 */

import { existsSync } from "node:fs";
import type { Page } from "playwright";
import {
  CONFIG,
  type AuthorAnalysis,
  type ChapterLink,
  type ChapterLinksOutput,
  type ExamLink,
  type ExamLinksOutput,
  type ExamWithChapters,
  type FilteredExamLinksOutput,
} from "./config.ts";
import {
  buildChapterUrl,
  createBrowserContext,
  createProgressBar,
  delay,
  loadJson,
  log,
  logError,
  saveJson,
} from "./utils.ts";

const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

async function collectChapterLinks(): Promise<void> {
  debugLog("Starting chapter links collection...");

  // Check for --filtered flag or use filtered file if available
  const useFiltered =
    process.argv.includes("--filtered") ||
    (existsSync(CONFIG.FILTERED_EXAM_LINKS_FILE) && !process.argv.includes("--all"));

  // Load exam links from Phase 1 (or filtered from Phase 1.5)
  let links: ExamLink[];
  try {
    if (useFiltered) {
      const filtered = loadJson<FilteredExamLinksOutput>(CONFIG.FILTERED_EXAM_LINKS_FILE);
      // Only use relevant links
      links = filtered.links.filter((l) => l.relevant);
      console.log(
        `Loaded ${links.length} relevant links (${filtered.total_irrelevant} filtered out)`,
      );
    } else {
      const examLinks = loadJson<ExamLinksOutput>(CONFIG.EXAM_LINKS_FILE);
      links = examLinks.links;
      console.log(`Loaded ${links.length} exam links`);
      if (existsSync(CONFIG.FILTERED_EXAM_LINKS_FILE)) {
        console.log(`Note: Use --filtered to use filtered file, or --all to use unfiltered.`);
      }
    }
  } catch (error) {
    logError("Failed to load exam links. Please run Phase 1 first (bun run scrape:1-exams)", error);
    process.exit(1);
  }

  const context = await createBrowserContext();

  // Create multiple tabs
  const pages: Page[] = [];
  for (let i = 0; i < CONFIG.CONCURRENT_TABS; i++) {
    pages.push(await context.newPage());
  }

  const results: ExamWithChapters[] = [];
  let totalChapters = 0;
  let processedCount = 0;
  const progress = createProgressBar(links.length, "chapters");

  try {
    // Process in batches of CONFIG.CONCURRENT_TABS
    for (let i = 0; i < links.length; i += CONFIG.CONCURRENT_TABS) {
      const batch = links.slice(i, i + CONFIG.CONCURRENT_TABS);

      debugLog(
        `\n--- Processing batch ${Math.floor(i / CONFIG.CONCURRENT_TABS) + 1} (items ${i + 1}-${Math.min(i + CONFIG.CONCURRENT_TABS, links.length)} of ${links.length}) ---`,
      );

      // Process batch concurrently
      const batchPromises = batch.map((exam, idx) =>
        processExam(pages[idx]!, exam, i + idx + 1, links.length),
      );

      const batchResults = await Promise.all(batchPromises);

      // Collect results
      for (const result of batchResults) {
        results.push(result);
        totalChapters += result.chapters.length;
        processedCount++;
      }

      progress.update(i + batch.length, totalChapters);
      debugLog(`Batch complete: ${totalChapters} total chapters found`);

      // Save progress incrementally every 20 exams
      if (processedCount % 20 === 0 || i + CONFIG.CONCURRENT_TABS >= links.length) {
        saveProgress(results, totalChapters, DEBUG);
      }

      // Small delay between batches to be nice to the server
      if (i + CONFIG.CONCURRENT_TABS < links.length) {
        await delay(CONFIG.REQUEST_DELAY_MS);
      }
    }

    // Save final results
    saveProgress(results, totalChapters, DEBUG);

    progress.finish();
    console.log(`Processed ${results.length} exams, found ${totalChapters} chapters`);

    // Print author analysis
    const authorAnalysis = generateAuthorAnalysis(results);
    const withAuthorCount = authorAnalysis.total_with_author;
    const uniqueAuthors = authorAnalysis.authors.length;
    console.log(
      `Authors: ${withAuthorCount} with author (${uniqueAuthors} unique), ${authorAnalysis.total_without_author} without`,
    );
    if (DEBUG) {
      printAuthorAnalysis(authorAnalysis);
    }
  } catch (error) {
    logError("Fatal error during collection", error);
    // Save whatever we have
    if (results.length > 0) {
      saveProgress(results, totalChapters, DEBUG);
      console.log("Partial results saved.");
    }
    throw error;
  } finally {
    await context.close();
  }
}

async function processExam(
  page: Page,
  exam: ExamLink,
  index: number,
  total: number,
): Promise<ExamWithChapters> {
  debugLog(`[${index}/${total}] Processing: ${exam.title || exam.tikuid}`);

  try {
    const result = await collectChaptersFromExam(page, exam.url, exam.tikuid);
    debugLog(
      `[${index}/${total}] Found ${result.chapters.length} chapters${result.author ? `, author: ${result.author}` : ""}`,
    );

    return {
      tikuid: exam.tikuid,
      exam_url: exam.url,
      title: exam.title,
      ...(result.author && { author: result.author }),
      found_by_keywords: exam.found_by_keywords,
      chapters: result.chapters,
    };
  } catch (error) {
    logError(`[${index}/${total}] Failed to process: ${exam.tikuid}`, error);
    return {
      tikuid: exam.tikuid,
      exam_url: exam.url,
      title: exam.title,
      found_by_keywords: exam.found_by_keywords,
      chapters: [],
    };
  }
}

type ChapterResult = {
  chapters: ChapterLink[];
  author?: string;
};

async function collectChaptersFromExam(
  page: Page,
  examUrl: string,
  tikuid: string,
): Promise<ChapterResult> {
  // Navigate to exam page
  await page.goto(examUrl, {
    timeout: CONFIG.PAGE_LOAD_TIMEOUT_MS,
    waitUntil: "networkidle",
  });

  await delay(1500); // Wait for dynamic content to load

  // Extract author (题库作者)
  let author: string | undefined;
  try {
    const authorValue = await page.$eval(
      'div:has(> div:text-is("题库作者")) > div.text-primary',
      (el) => el.textContent?.trim(),
    );
    // Treat "未知" (unknown) as no author
    if (authorValue && authorValue !== "未知") {
      author = authorValue;
    }
  } catch {
    // Author element not found, leave as undefined
  }

  const chapters: ChapterLink[] = [];

  try {
    // Wait for chapter buttons to be visible
    await page.waitForSelector(CONFIG.SELECTORS.CHAPTER_BUTTON, { timeout: 5000 });

    // Extract all zjid values from chapter buttons
    const zjids = await page.$$eval(CONFIG.SELECTORS.CHAPTER_BUTTON, (buttons) =>
      buttons
        .map((btn) => btn.getAttribute("zjid"))
        .filter((zjid): zjid is string => zjid !== null && zjid.length > 0),
    );

    debugLog(`  Found ${zjids.length} chapter buttons`);

    for (const zjid of zjids) {
      chapters.push({
        zjid: zjid,
        url: buildChapterUrl(tikuid, zjid),
      });
    }
  } catch {
    // No chapter buttons found - this exam might have questions directly
    debugLog("  No chapter buttons found, checking for direct questions...");

    // Some pages have questions directly without chapters
    // In this case, we create a single "chapter" entry with the exam's tikuid
    // The URL pattern for direct questions is different
    const hasDirectQuestions = await checkForDirectQuestions(page);
    if (hasDirectQuestions) {
      debugLog("  Found direct questions (no chapters)");
      chapters.push({
        zjid: tikuid, // Use tikuid as zjid for direct questions
        url: `${CONFIG.BASE_URL}/page/datiwk.html?tikuid=${tikuid}&type=gongkai&page=1`,
      });
    }
  }

  return { chapters, ...(author && { author }) };
}

async function checkForDirectQuestions(page: Page): Promise<boolean> {
  try {
    // Look for common elements that indicate direct questions
    // This could be a "开始答题" (start answering) button or similar
    const directAnswerButton = page.locator(
      'a:has-text("开始答题"), button:has-text("开始答题"), a:has-text("进入答题"), button:has-text("进入答题")',
    );
    return await directAnswerButton.isVisible({ timeout: 2000 });
  } catch {
    return false;
  }
}

function generateAuthorAnalysis(results: ExamWithChapters[]): AuthorAnalysis {
  const withAuthor = results.filter((r) => r.author);
  const withoutAuthor = results.filter((r) => !r.author);
  const total = results.length;

  // Count authors
  const authorCounts = new Map<string, number>();
  for (const result of withAuthor) {
    const count = authorCounts.get(result.author!) || 0;
    authorCounts.set(result.author!, count + 1);
  }

  // Sort authors by count descending
  const authors = Array.from(authorCounts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  return {
    total_with_author: withAuthor.length,
    total_without_author: withoutAuthor.length,
    with_author_percent: total > 0 ? Math.round((withAuthor.length / total) * 10000) / 100 : 0,
    without_author_percent:
      total > 0 ? Math.round((withoutAuthor.length / total) * 10000) / 100 : 0,
    authors,
  };
}

function printAuthorAnalysis(analysis: AuthorAnalysis): void {
  log(`\n--- Author Analysis ---`);
  log(`With author: ${analysis.total_with_author} (${analysis.with_author_percent}%)`);
  log(`Without author: ${analysis.total_without_author} (${analysis.without_author_percent}%)`);
  if (analysis.authors.length > 0) {
    log(`\nAuthors (${analysis.authors.length} unique):`);
    for (const { name, count } of analysis.authors.slice(0, 20)) {
      log(`  ${name}: ${count}`);
    }
    if (analysis.authors.length > 20) {
      log(`  ... and ${analysis.authors.length - 20} more`);
    }
  }
}

function saveProgress(results: ExamWithChapters[], totalChapters: number, verbose = true): void {
  const authorAnalysis = generateAuthorAnalysis(results);

  const output: ChapterLinksOutput = {
    collected_at: new Date().toISOString(),
    total_exams: results.length,
    total_chapters: totalChapters,
    author_analysis: authorAnalysis,
    exams: results,
  };

  saveJson(CONFIG.CHAPTER_LINKS_FILE, output, verbose);
}

// Run the script
collectChapterLinks()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
