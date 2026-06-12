/**
 * Phase 1.5: Filter Exam Links by Relevance (Optional)
 *
 * Uses LLM to filter out irrelevant exam datasets based on titles.
 * Only keeps datasets relevant to maritime/navigation English exams.
 *
 * Usage: bun run scrape:2-filter [--debug]
 *
 * Requires: LITE_LLM_API_KEY environment variable
 */

import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateObject } from "ai";
import { z } from "zod";
import {
  CONFIG,
  type ExamLink,
  type ExamLinksOutput,
  type FilteredExamLink,
  type FilteredExamLinksOutput,
} from "./config.ts";

// Analysis types
type KeywordAnalysis = {
  keyword: string;
  total_links: number;
  relevant: number;
  irrelevant: number;
  relevant_percent: number;
  irrelevant_percent: number;
};

type AnalysisOutput = {
  analyzed_at: string;
  source_file: string;
  summary: {
    total_links: number;
    total_relevant: number;
    total_irrelevant: number;
    relevant_percent: number;
    irrelevant_percent: number;
  };
  keyword_analysis: KeywordAnalysis[];
  overlap_analysis: {
    links_with_single_keyword: number;
    links_with_multiple_keywords: number;
    average_keywords_per_link: number;
  };
  top_irrelevant_patterns: Array<{
    pattern: string;
    count: number;
    example_titles: string[];
  }>;
};
import { createProgressBar, loadJson, log, logError, saveJson } from "./utils.ts";

// Configuration
const BATCH_SIZE = 15; // Number of titles to process per LLM call
const CONCURRENT_BATCHES = 10; // Number of batches to process concurrently
const MODEL_ID = "openai/gpt-oss-120b"; // More reliable for simple text output
const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

async function filterExamLinks(): Promise<void> {
  debugLog("Starting exam links filtering...");

  // Check for API key
  const apiKey = process.env.LITE_LLM_API_KEY;
  if (!apiKey) {
    logError("LITE_LLM_API_KEY environment variable is required");
    process.exit(1);
  }

  // Load exam links
  let examLinks: ExamLinksOutput;
  try {
    examLinks = loadJson<ExamLinksOutput>(CONFIG.EXAM_LINKS_FILE);
    console.log(`Loaded ${examLinks.links.length} exam links`);
  } catch (error) {
    logError("Failed to load exam links. Please run Phase 1 first (bun run scrape:1-exams)", error);
    process.exit(1);
  }

  // Create LiteLLM provider
  const provider = createOpenAICompatible({
    name: "litellm",
    apiKey,
    baseURL: "https://llm.hpc.ntnu.no/v1",
    supportsStructuredOutputs: true,
  });

  const filteredLinks: FilteredExamLink[] = [];
  const totalBatches = Math.ceil(examLinks.links.length / BATCH_SIZE);

  debugLog(
    `Processing ${examLinks.links.length} titles in ${totalBatches} batches (${CONCURRENT_BATCHES} concurrent)...`,
  );

  // Create all batch definitions
  const allBatches: Array<{ batch: ExamLink[]; batchNum: number }> = [];
  for (let i = 0; i < examLinks.links.length; i += BATCH_SIZE) {
    allBatches.push({
      batch: examLinks.links.slice(i, i + BATCH_SIZE),
      batchNum: Math.floor(i / BATCH_SIZE) + 1,
    });
  }

  const progress = createProgressBar(totalBatches, "relevant exams");
  let processedBatches = 0;

  // Process batches in groups of CONCURRENT_BATCHES
  for (let i = 0; i < allBatches.length; i += CONCURRENT_BATCHES) {
    const concurrentBatches = allBatches.slice(i, i + CONCURRENT_BATCHES);
    const batchNums = concurrentBatches.map((b) => b.batchNum).join(", ");
    debugLog(`\n--- Processing batches ${batchNums} / ${totalBatches} ---`);

    const results = await Promise.all(
      concurrentBatches.map(async ({ batch, batchNum }) => {
        try {
          const batchResults = await classifyBatch(provider, batch);
          const relevant = batchResults.filter((r) => r.relevant).length;
          debugLog(`  Batch ${batchNum}: ${relevant}/${batch.length} relevant`);
          return batchResults;
        } catch (error) {
          logError(`Failed to process batch ${batchNum}`, error);
          // On error, mark all as relevant to avoid losing data
          return batch.map(
            (link): FilteredExamLink => ({
              ...link,
              relevant: true,
              reason: "Error during classification - kept by default",
            }),
          );
        }
      }),
    );

    // Add all results from this concurrent group
    for (const batchResults of results) {
      filteredLinks.push(...batchResults);
    }

    processedBatches += concurrentBatches.length;
    const relevantSoFar = filteredLinks.filter((l) => l.relevant).length;
    progress.update(processedBatches, relevantSoFar);

    // Save progress after each concurrent group
    saveProgress(examLinks.links.length, filteredLinks);
  }

  progress.finish();

  // Calculate statistics
  const relevantCount = filteredLinks.filter((l) => l.relevant).length;
  const irrelevantCount = filteredLinks.filter((l) => !l.relevant).length;

  // Save final results
  const output: FilteredExamLinksOutput = {
    filtered_at: new Date().toISOString(),
    source_file: CONFIG.EXAM_LINKS_FILE,
    total_input: examLinks.links.length,
    total_relevant: relevantCount,
    total_irrelevant: irrelevantCount,
    model_used: MODEL_ID,
    links: filteredLinks,
  };

  saveJson(CONFIG.FILTERED_EXAM_LINKS_FILE, output, DEBUG);

  console.log(`Filtered: ${relevantCount} relevant, ${irrelevantCount} irrelevant`);
  debugLog(`Output: ${CONFIG.FILTERED_EXAM_LINKS_FILE}`);

  // Run analysis
  const analysis = analyzeResults(filteredLinks, examLinks.search_keywords);
  if (DEBUG) printAnalysis(analysis);
  saveJson(CONFIG.FILTER_ANALYSIS_FILE, analysis, DEBUG);
  debugLog(`\nAnalysis saved to: ${CONFIG.FILTER_ANALYSIS_FILE}`);
}

function saveProgress(totalInput: number, filteredLinks: FilteredExamLink[]): void {
  const relevantCount = filteredLinks.filter((l) => l.relevant).length;
  const irrelevantCount = filteredLinks.filter((l) => !l.relevant).length;

  const output: FilteredExamLinksOutput = {
    filtered_at: new Date().toISOString(),
    source_file: CONFIG.EXAM_LINKS_FILE,
    total_input: totalInput,
    total_relevant: relevantCount,
    total_irrelevant: irrelevantCount,
    model_used: MODEL_ID,
    links: filteredLinks,
  };

  saveJson(CONFIG.FILTERED_EXAM_LINKS_FILE, output, false);
}

// Schema for structured classification output
const ClassificationSchema = z.object({
  classifications: z.array(
    z.object({
      index: z.number().describe("The index of the title in the input list"),
      relevant: z.boolean().describe("Whether the exam is relevant to maritime/shipping"),
    }),
  ),
});

async function classifyBatch(
  provider: ReturnType<typeof createOpenAICompatible>,
  batch: ExamLink[],
): Promise<FilteredExamLink[]> {
  // Prepare titles list
  const titlesText = batch.map((link, idx) => `${idx}. ${link.title}`).join("\n");

  const systemPrompt = `You classify Chinese exam titles for relevance to maritime/shipping industry.

RELEVANT (true):
- 航海 (Navigation) - anything about ship navigation
- 海事 (Maritime) - maritime industry exams
- 船舶 (Ship/Vessel) - exams about ships
- 轮机 (Marine Engineering) - ship engine/machinery exams
- 海员 (Seafarer) - seafarer certification exams
- 船长/大副/二副 (Captain/Officers) - ship officer exams
- 水手/机工 (Sailor/Engine crew) - crew certification
- Any exam about ships, ports, shipping, or maritime operations

IRRELEVANT (false):
- 航空/飞机/南航 (Aviation/Aircraft/Airlines) - NOT ships
- 驾驶证/驾照/科目 (Driver's license) - car driving, not ships
- Medical, IT, construction exams unrelated to ships
- Gibberish/test titles (嘤嘤嘤, 库库库, random characters)

Key rule: If it's about ships, maritime, or seafaring → relevant=true. If it's about planes, cars, or unrelated fields → relevant=false.`;

  const userPrompt = `Classify each of these titles for maritime relevance:

${titlesText}`;

  const result = await generateObject({
    model: provider.languageModel(MODEL_ID),
    system: systemPrompt,
    prompt: userPrompt,
    schema: ClassificationSchema,
    temperature: 0,
  });

  // Build a map from index to relevance
  const classifications = new Map<number, boolean>();
  for (const item of result.object.classifications) {
    classifications.set(item.index, item.relevant);
  }

  // Log if we got unexpected results
  if (classifications.size !== batch.length) {
    debugLog(`  Warning: Expected ${batch.length} classifications, got ${classifications.size}`);
  }

  return batch.map((link, idx): FilteredExamLink => {
    const isRelevant = classifications.get(idx) ?? true; // Default to relevant if missing
    return {
      ...link,
      relevant: isRelevant,
    };
  });
}

function analyzeResults(
  filteredLinks: FilteredExamLink[],
  searchKeywords: string[],
): AnalysisOutput {
  const totalLinks = filteredLinks.length;
  const relevantLinks = filteredLinks.filter((l) => l.relevant);
  const irrelevantLinks = filteredLinks.filter((l) => !l.relevant);

  // Keyword analysis
  const keywordAnalysis: KeywordAnalysis[] = searchKeywords.map((keyword) => {
    const linksWithKeyword = filteredLinks.filter((l) => l.found_by_keywords.includes(keyword));
    const relevantWithKeyword = linksWithKeyword.filter((l) => l.relevant);
    const irrelevantWithKeyword = linksWithKeyword.filter((l) => !l.relevant);
    const total = linksWithKeyword.length;

    return {
      keyword,
      total_links: total,
      relevant: relevantWithKeyword.length,
      irrelevant: irrelevantWithKeyword.length,
      relevant_percent:
        total > 0 ? Math.round((relevantWithKeyword.length / total) * 10000) / 100 : 0,
      irrelevant_percent:
        total > 0 ? Math.round((irrelevantWithKeyword.length / total) * 10000) / 100 : 0,
    };
  });

  // Sort by relevance percentage (descending)
  keywordAnalysis.sort((a, b) => b.relevant_percent - a.relevant_percent);

  // Overlap analysis
  const linksWithSingleKeyword = filteredLinks.filter(
    (l) => l.found_by_keywords.length === 1,
  ).length;
  const linksWithMultipleKeywords = filteredLinks.filter(
    (l) => l.found_by_keywords.length > 1,
  ).length;
  const totalKeywordsFound = filteredLinks.reduce((sum, l) => sum + l.found_by_keywords.length, 0);

  // Find common patterns in irrelevant titles
  const irrelevantPatterns = findIrrelevantPatterns(irrelevantLinks);

  return {
    analyzed_at: new Date().toISOString(),
    source_file: CONFIG.FILTERED_EXAM_LINKS_FILE,
    summary: {
      total_links: totalLinks,
      total_relevant: relevantLinks.length,
      total_irrelevant: irrelevantLinks.length,
      relevant_percent: Math.round((relevantLinks.length / totalLinks) * 10000) / 100,
      irrelevant_percent: Math.round((irrelevantLinks.length / totalLinks) * 10000) / 100,
    },
    keyword_analysis: keywordAnalysis,
    overlap_analysis: {
      links_with_single_keyword: linksWithSingleKeyword,
      links_with_multiple_keywords: linksWithMultipleKeywords,
      average_keywords_per_link: Math.round((totalKeywordsFound / totalLinks) * 100) / 100,
    },
    top_irrelevant_patterns: irrelevantPatterns,
  };
}

function findIrrelevantPatterns(
  irrelevantLinks: FilteredExamLink[],
): Array<{ pattern: string; count: number; example_titles: string[] }> {
  // Common patterns that indicate irrelevant content
  const patternKeywords = [
    { pattern: "航空/飞机/南航 (Aviation)", keywords: ["航空", "飞机", "南航", "机长", "空乘"] },
    {
      pattern: "驾驶证/驾照 (Driver's License)",
      keywords: ["驾驶证", "驾照", "科目", "交规", "机动车"],
    },
    { pattern: "医学/医疗 (Medical)", keywords: ["医学", "医疗", "护士", "医师", "药师"] },
    { pattern: "建筑/工程 (Construction)", keywords: ["建筑", "建造", "工程师", "施工"] },
    { pattern: "IT/计算机 (IT)", keywords: ["计算机", "软件", "程序", "网络工程"] },
    { pattern: "教育/教师 (Education)", keywords: ["教师", "教育", "教资"] },
    { pattern: "乱码/测试 (Gibberish/Test)", keywords: ["嘤嘤嘤", "库库库", "测试", "test"] },
  ];

  const results: Array<{ pattern: string; count: number; example_titles: string[] }> = [];

  for (const { pattern, keywords } of patternKeywords) {
    const matching = irrelevantLinks.filter((link) =>
      keywords.some((kw) => link.title.includes(kw)),
    );

    if (matching.length > 0) {
      results.push({
        pattern,
        count: matching.length,
        example_titles: matching.slice(0, 3).map((l) => l.title),
      });
    }
  }

  // Sort by count descending
  results.sort((a, b) => b.count - a.count);

  return results;
}

// Helper to calculate visual width of string (Chinese chars = 2, others = 1)
function getVisualWidth(str: string): number {
  let width = 0;
  for (const char of str) {
    // Chinese characters and other wide characters
    width += /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]/.test(char) ? 2 : 1;
  }
  return width;
}

// Pad string to target visual width
function padEndVisual(str: string, targetWidth: number): string {
  const currentWidth = getVisualWidth(str);
  const padding = Math.max(0, targetWidth - currentWidth);
  return str + " ".repeat(padding);
}

function printAnalysis(analysis: AnalysisOutput): void {
  log(`\n========== Analysis Results ==========`);

  // Summary
  log(`\n--- Summary ---`);
  log(`Total links: ${analysis.summary.total_links}`);
  log(`Relevant: ${analysis.summary.total_relevant} (${analysis.summary.relevant_percent}%)`);
  log(`Irrelevant: ${analysis.summary.total_irrelevant} (${analysis.summary.irrelevant_percent}%)`);

  // Keyword analysis
  log(`\n--- Keyword Effectiveness ---`);
  log(
    `${"Keyword".padEnd(12)} | ${"Total".padStart(6)} | ${"Rel".padStart(5)} | ${"Irrel".padStart(5)} | ${"Rel%".padStart(7)} | ${"Irrel%".padStart(7)}`,
  );
  log("-".repeat(62));
  for (const kw of analysis.keyword_analysis) {
    log(
      `${padEndVisual(kw.keyword, 12)} | ${String(kw.total_links).padStart(6)} | ${String(kw.relevant).padStart(5)} | ${String(kw.irrelevant).padStart(5)} | ${String(kw.relevant_percent + "%").padStart(7)} | ${String(kw.irrelevant_percent + "%").padStart(7)}`,
    );
  }

  // Overlap analysis
  log(`\n--- Overlap Analysis ---`);
  log(`Links found by single keyword: ${analysis.overlap_analysis.links_with_single_keyword}`);
  log(
    `Links found by multiple keywords: ${analysis.overlap_analysis.links_with_multiple_keywords}`,
  );
  log(`Average keywords per link: ${analysis.overlap_analysis.average_keywords_per_link}`);

  // Irrelevant patterns
  if (analysis.top_irrelevant_patterns.length > 0) {
    log(`\n--- Top Irrelevant Patterns ---`);
    for (const pattern of analysis.top_irrelevant_patterns) {
      log(`${pattern.pattern}: ${pattern.count} links`);
      for (const title of pattern.example_titles) {
        log(`    - ${title}`);
      }
    }
  }
}

// Run the script
filterExamLinks()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
