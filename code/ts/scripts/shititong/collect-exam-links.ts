/**
 * Phase 1: Collect Exam Links
 *
 * This script searches for maritime exam packages on shititong.cn
 * and collects all exam links to JSON.
 *
 * Usage: bun run scrape:1-exams [--debug]
 */

import { CONFIG, type ExamLink, type ExamLinksOutput } from "./config.ts";
import {
  createBrowserContext,
  createProgressBar,
  delay,
  extractTikuid,
  log,
  logError,
  saveJson,
} from "./utils.ts";

const DEBUG = process.argv.includes("--debug");

function debugLog(message: string): void {
  if (DEBUG) log(message);
}

async function collectExamLinks(): Promise<void> {
  debugLog("Starting exam links collection...");

  const context = await createBrowserContext();
  const page = await context.newPage();

  const allLinks = new Map<string, ExamLink>();
  const keywords = CONFIG.SEARCH_KEYWORDS;
  const progress = createProgressBar(keywords.length, "exams");

  try {
    // Navigate to search page
    debugLog(`Navigating to: ${CONFIG.SEARCH_URL}`);
    await page.goto(CONFIG.SEARCH_URL, {
      timeout: CONFIG.PAGE_LOAD_TIMEOUT_MS,
      waitUntil: "networkidle",
    });

    await delay(2000);

    // Search with each keyword
    for (let i = 0; i < keywords.length; i++) {
      const keyword = keywords[i]!;
      debugLog(`\n========== Searching for keyword: ${keyword} ==========`);

      try {
        // Get current first link to detect when results change
        const firstLinkBefore = await page
          .locator(CONFIG.SELECTORS.EXAM_LINK)
          .first()
          .getAttribute("href")
          .catch(() => null);

        // Clear and fill search input
        const searchInput = page.locator(CONFIG.SELECTORS.SEARCH_INPUT);
        await searchInput.fill("");
        await searchInput.fill(keyword);

        // Click search button
        const searchButton = page.locator(CONFIG.SELECTORS.SEARCH_BUTTON);
        await searchButton.click();

        // Wait for results to actually change (or timeout after 5s)
        debugLog("Waiting for search results to update...");
        const waitStart = Date.now();
        while (Date.now() - waitStart < 5000) {
          await delay(300);
          const firstLinkAfter = await page
            .locator(CONFIG.SELECTORS.EXAM_LINK)
            .first()
            .getAttribute("href")
            .catch(() => null);
          if (firstLinkAfter && firstLinkAfter !== firstLinkBefore) {
            debugLog("Results updated!");
            break;
          }
        }

        await delay(300); // Small buffer for rendering

        // Collect links from search results
        await collectLinksFromPage(page, allLinks, keyword);

        // Handle pagination if present
        await collectFromPagination(page, allLinks, keyword);
      } catch (error) {
        logError(`Failed to search for keyword: ${keyword}`, error);
      }

      progress.update(i + 1, allLinks.size);
      await delay(CONFIG.REQUEST_DELAY_MS);
    }

    progress.finish();

    // Save results
    const output: ExamLinksOutput = {
      collected_at: new Date().toISOString(),
      search_url: CONFIG.SEARCH_URL,
      search_keywords: [...CONFIG.SEARCH_KEYWORDS],
      total_links: allLinks.size,
      links: Array.from(allLinks.values()),
    };

    saveJson(CONFIG.EXAM_LINKS_FILE, output, DEBUG);

    console.log(`Found ${allLinks.size} unique exam packages`);
  } catch (error) {
    logError("Fatal error during collection", error);
    throw error;
  } finally {
    await context.close();
  }
}

async function collectLinksFromPage(
  page: Awaited<ReturnType<Awaited<ReturnType<typeof createBrowserContext>>["newPage"]>>,
  linkMap: Map<string, ExamLink>,
  keyword: string,
): Promise<number> {
  debugLog("Extracting links from current page...");

  try {
    // Wait for links to be visible
    await page.waitForSelector(CONFIG.SELECTORS.EXAM_LINK, { timeout: 5000 });

    // Extract all links in one browser call - much faster!
    const extractedLinks = await page.$$eval('a.view-btn[href*="cha-kan/tikuheji"]', (links) =>
      links.map((link) => {
        const href = link.getAttribute("href") || "";
        // Navigate up to the flex container parent and find h4 title
        // Structure: div[flex] > div(icon) + div(content with h4) + div(link)
        let title = "";
        const parent = link.parentElement?.parentElement;
        if (parent) {
          const h4 = parent.querySelector("h4");
          title = h4?.textContent?.trim() || "";
        }
        if (!title) {
          title = link.textContent?.trim() || "";
        }
        return { href, title };
      }),
    );

    debugLog(`Found ${extractedLinks.length} links on page`);

    let newLinks = 0;
    for (const { href, title } of extractedLinks) {
      const tikuid = extractTikuid(href);
      if (!tikuid) continue;

      const existing = linkMap.get(tikuid);
      if (existing) {
        // Add keyword if not already present
        if (!existing.found_by_keywords.includes(keyword)) {
          existing.found_by_keywords.push(keyword);
        }
        continue;
      }

      linkMap.set(tikuid, {
        title: title,
        url: href,
        tikuid: tikuid,
        found_by_keywords: [keyword],
      });

      newLinks++;
      debugLog(`  + [${keyword}] ${title.substring(0, 50) || "(no title)"}`);
    }

    debugLog(`Added ${newLinks} new links (${linkMap.size} total)`);
    return newLinks;
  } catch {
    debugLog("No exam links found on current page");
    return 0;
  }
}

async function collectFromPagination(
  page: Awaited<ReturnType<Awaited<ReturnType<typeof createBrowserContext>>["newPage"]>>,
  linkMap: Map<string, ExamLink>,
  keyword: string,
): Promise<void> {
  let pageNum = 1;
  const maxPages = 20; // Safety limit

  while (pageNum < maxPages) {
    try {
      // Look for next page button
      const nextButton = page.locator(
        'a.next, button.next, [aria-label="下一页"], .pagination a:has-text("下一页")',
      );

      if (!(await nextButton.isVisible({ timeout: 2000 }))) {
        break;
      }

      // Check if next button is disabled
      const isDisabled = await nextButton.getAttribute("disabled");
      const className = await nextButton.getAttribute("class");
      if (isDisabled || className?.includes("disabled")) {
        break;
      }

      await nextButton.click();
      await delay(1500);
      await page.waitForLoadState("networkidle");

      const newLinks = await collectLinksFromPage(page, linkMap, keyword);
      if (newLinks === 0) {
        // No new links found, probably reached the end
        break;
      }

      pageNum++;
    } catch {
      // No pagination or reached end
      break;
    }
  }
}

// Run the script
collectExamLinks()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Script failed:", error);
    process.exit(1);
  });
