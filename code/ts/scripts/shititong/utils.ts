import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { chromium, type BrowserContext, type Page } from "playwright";
import { CONFIG, type Cookie } from "./config.ts";

export async function loadCookies(): Promise<Cookie[]> {
  if (!existsSync(CONFIG.COOKIES_PATH)) {
    throw new Error(
      `Cookies file not found at ${CONFIG.COOKIES_PATH}\n` +
        "Please create cookies.json with your authentication cookies.\n" +
        "Required cookies: userkeyhead, userkeyid, userkeyname, usermapper, usertoken",
    );
  }

  const cookies = JSON.parse(readFileSync(CONFIG.COOKIES_PATH, "utf-8"));
  return cookies;
}

export async function createBrowserContext(): Promise<BrowserContext> {
  const browser = await chromium.launch({
    headless: CONFIG.HEADLESS,
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
  });

  // Load cookies
  const cookies = await loadCookies();
  await context.addCookies(cookies);

  return context;
}

export async function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function ensureDir(filePath: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

export function saveJson<T>(filePath: string, data: T, verbose = true): void {
  ensureDir(filePath);
  writeFileSync(filePath, JSON.stringify(data, null, 2));
  if (verbose) console.log(`Saved: ${filePath}`);
}

export function loadJson<T>(filePath: string): T {
  if (!existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return JSON.parse(readFileSync(filePath, "utf-8"));
}

export function extractTikuid(url: string): string | null {
  const match = url.match(CONFIG.PATTERNS.TIKUID);
  if (match?.[1]) return match[1];

  const fallback = url.match(CONFIG.PATTERNS.TIKUID_FROM_URL);
  return fallback?.[1] ?? null;
}

export async function handleCollectPopup(page: Page): Promise<boolean> {
  try {
    const confirmButton = page.locator(CONFIG.SELECTORS.COLLECT_POPUP_CONFIRM);
    if (await confirmButton.isVisible({ timeout: 2000 })) {
      await confirmButton.click();
      await delay(500);
      return true;
    }
  } catch {
    // No popup present
  }
  return false;
}

export function buildChapterUrl(tikuid: string, zjid: string): string {
  return `${CONFIG.BASE_URL}/page/datiwk.html?tikuid=${tikuid}&type=gongkai&page=1&gongneng=zhangjie&zhangjieid=${zjid}`;
}

export function log(message: string): void {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
}

export function logError(message: string, error?: unknown): void {
  const timestamp = new Date().toISOString();
  console.error(`[${timestamp}] ERROR: ${message}`);
  if (error) {
    console.error(error);
  }
}

export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

export function createProgressBar(total: number, itemLabel = "items", barWidth = 30) {
  let current = 0;
  let itemCount = 0;
  const startTime = Date.now();

  return {
    update(completed: number, count: number) {
      current = completed;
      itemCount = count;
      this.render();
    },
    render() {
      const percent = current / total;
      const filled = Math.round(barWidth * percent);
      const empty = barWidth - filled;
      const bar = "█".repeat(filled) + "░".repeat(empty);

      const elapsed = Date.now() - startTime;
      const avgTimePerItem = current > 0 ? elapsed / current : 0;
      const remaining = avgTimePerItem * (total - current);
      const eta = current > 0 ? formatDuration(remaining) : "calculating...";

      const line = `\r[${bar}] ${current}/${total} (${(percent * 100).toFixed(1)}%) | ${itemCount} ${itemLabel} | ETA: ${eta}  `;
      process.stdout.write(line);
    },
    finish() {
      const elapsed = Date.now() - startTime;
      process.stdout.write("\n");
      console.log(`Completed in ${formatDuration(elapsed)}`);
    },
  };
}
