#!/usr/bin/env python3
"""Scrape USCG sample examination PDFs via browser navigation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse, unquote

from bs4 import BeautifulSoup
from playwright.sync_api import BrowserContext, Page, sync_playwright
from utils.repo import REPO_ROOT

ROOT_URL = "https://www.dco.uscg.mil/nmc/examinations/"
SEED_URLS = [
    ROOT_URL,
    "https://www.dco.uscg.mil/nmc/exams/deck_officer/",
    "https://www.dco.uscg.mil/nmc/exams/deck_ratings/",
    "https://www.dco.uscg.mil/nmc/exams/engine_officer/",
    "https://www.dco.uscg.mil/nmc/exams/engine_ratings/",
]
ALLOWED_DOMAINS = {"www.dco.uscg.mil", "dco.uscg.mil"}
SCOPE_HINTS = (
    "/nmc/examinations",
    "/nmc/exams/",
    "/national-maritime-center-nmc/examinations/",
    "/national-maritime-center-cg-nmc/",
    "/dev_nmc/",
    "/redesign-template-one/",
    "/dev-",
)
SKIP_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".xml",
    ".zip",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evals_to_parse/scraped_coast_guard"
FAKE_CHROME_USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.7499.4 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.7499.4 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
]


@dataclass
class CrawlResult:
    visited_pages: set[str] = field(default_factory=set)
    failed_pages: dict[str, str] = field(default_factory=dict)
    pdf_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    failed_downloads: dict[str, str] = field(default_factory=dict)
    downloaded: dict[str, str] = field(default_factory=dict)


class USCGExamScraper:
    def __init__(
        self,
        output_dir: Path,
        max_pages: int,
        delay_ms: int,
        timeout_ms: int,
        headless: bool,
        user_agent: str | None,
        random_user_agent: bool,
        stealth: bool,
        retry_403: int,
    ) -> None:
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay_ms = delay_ms
        self.timeout_ms = timeout_ms
        self.headless = headless
        self.user_agent = user_agent
        self.random_user_agent = random_user_agent
        self.stealth = stealth
        self.retry_403 = retry_403
        self.result = CrawlResult()

    def _select_user_agent(self) -> str:
        if self.user_agent:
            return self.user_agent
        if self.random_user_agent:
            return random.choice(FAKE_CHROME_USER_AGENTS)
        return FAKE_CHROME_USER_AGENTS[0]

    def _install_stealth(self, context: BrowserContext) -> None:
        if not self.stealth:
            return
        context.add_init_script(
            """
Object.defineProperty(navigator, "webdriver", { get: () => undefined });
Object.defineProperty(navigator, "languages", { get: () => ["en-US", "en"] });
Object.defineProperty(navigator, "plugins", { get: () => [1, 2, 3, 4] });
window.chrome = window.chrome || { runtime: {} };
"""
        )

    @staticmethod
    def _rewrite_legacy_url(url: str) -> str:
        replacements = (
            ("dev-deck-officer", "/nmc/exams/deck_officer/"),
            ("dev-deck-rating", "/nmc/exams/deck_ratings/"),
            ("dev-deck-ratings", "/nmc/exams/deck_ratings/"),
            ("dev-engine-officer", "/nmc/exams/engine_officer/"),
            ("dev-engine-rating", "/nmc/exams/engine_ratings/"),
            ("dev-engine-ratings", "/nmc/exams/engine_ratings/"),
        )
        lower = url.lower()
        for needle, replacement in replacements:
            if needle in lower:
                return f"https://www.dco.uscg.mil{replacement}"
        return url

    @staticmethod
    def _normalize_url(url: str, base_url: str) -> str | None:
        if not url:
            return None
        absolute = urljoin(base_url, url.strip())
        clean, _ = urldefrag(absolute)
        clean = USCGExamScraper._rewrite_legacy_url(clean)
        parsed = urlparse(clean)
        if parsed.scheme not in {"http", "https"}:
            return None
        if parsed.netloc.lower() not in ALLOWED_DOMAINS:
            return None
        return clean

    @staticmethod
    def _is_exam_pdf_url(url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.path.lower().endswith(".pdf")
            and "/nmc/pdfs/examinations/" in parsed.path.lower()
        )

    @staticmethod
    def _is_scoped_page(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc.lower() not in ALLOWED_DOMAINS:
            return False
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
            return False
        return any(hint in path for hint in SCOPE_HINTS)

    @staticmethod
    def _extract_links(html: str, base_url: str) -> set[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: set[str] = set()
        for anchor in soup.select("a[href]"):
            href_value = anchor.get("href")
            href = href_value if isinstance(href_value, str) else ""
            normalized = USCGExamScraper._normalize_url(href, base_url)
            if normalized:
                links.add(normalized)
        return links

    @staticmethod
    def _pdf_name_from_url(url: str) -> str:
        parsed = urlparse(url)
        raw_name = unquote(Path(parsed.path).name)
        if not raw_name.lower().endswith(".pdf"):
            raw_name = f"{raw_name or 'exam'}.pdf"

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        if not safe_name:
            safe_name = "exam.pdf"

        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        return f"{stem}_{digest}{suffix}"

    def _goto_with_403_retry(self, page: Page, url: str):
        last_response = None
        for attempt in range(self.retry_403 + 1):
            response = page.goto(
                url, wait_until="domcontentloaded", timeout=self.timeout_ms
            )
            last_response = response
            if response is None:
                continue
            if response.status != 403:
                return response
            if attempt < self.retry_403:
                page.wait_for_timeout(min(2000, 700 * (attempt + 1)))
        return last_response

    def crawl(self, page: Page) -> None:
        queue = deque(SEED_URLS)
        queued: set[str] = set(SEED_URLS)

        while queue and len(self.result.visited_pages) < self.max_pages:
            url = queue.popleft()
            if url in self.result.visited_pages:
                continue
            if not self._is_scoped_page(url):
                continue

            try:
                response = self._goto_with_403_retry(page, url)
                if response is None:
                    self.result.failed_pages[url] = "No HTTP response"
                    continue
                if response.status >= 400:
                    self.result.failed_pages[url] = f"HTTP {response.status}"
                    continue
                if self.delay_ms > 0:
                    page.wait_for_timeout(self.delay_ms)
                html = page.content()
            except Exception as exc:  # noqa: BLE001
                self.result.failed_pages[url] = str(exc)
                continue

            self.result.visited_pages.add(url)
            links = self._extract_links(html, base_url=page.url)

            for link in links:
                if self._is_exam_pdf_url(link):
                    self.result.pdf_sources[link].add(url)
                    continue

                if (
                    self._is_scoped_page(link)
                    and link not in self.result.visited_pages
                    and link not in queued
                ):
                    queue.append(link)
                    queued.add(link)

    def download_pdfs(self, context: BrowserContext) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        page = context.new_page()
        for index, pdf_url in enumerate(sorted(self.result.pdf_sources), start=1):
            filename = self._pdf_name_from_url(pdf_url)
            destination = self.output_dir / filename
            if destination.exists():
                self.result.downloaded[pdf_url] = str(destination)
                continue

            referer = sorted(self.result.pdf_sources[pdf_url])[0]
            try:
                response = context.request.get(
                    pdf_url,
                    headers={"Referer": referer, "Origin": "https://www.dco.uscg.mil"},
                    timeout=self.timeout_ms,
                )
                if response.ok:
                    body = response.body()
                else:
                    nav_response = page.goto(
                        pdf_url,
                        referer=referer,
                        wait_until="domcontentloaded",
                        timeout=self.timeout_ms,
                    )
                    if nav_response is None or nav_response.status >= 400:
                        self.result.failed_downloads[pdf_url] = (
                            f"HTTP {nav_response.status}"
                            if nav_response
                            else f"HTTP {response.status}"
                        )
                        continue
                    body = nav_response.body()
                if not body:
                    self.result.failed_downloads[pdf_url] = "Empty response body"
                    continue

                destination.write_bytes(body)
                self.result.downloaded[pdf_url] = str(destination)
            except Exception as exc:  # noqa: BLE001
                self.result.failed_downloads[pdf_url] = str(exc)

            if index % 25 == 0:
                time.sleep(0.3)
        page.close()

    def write_manifest(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        payload = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "root_url": ROOT_URL,
            "stats": {
                "visited_pages": len(self.result.visited_pages),
                "failed_pages": len(self.result.failed_pages),
                "pdf_links_found": len(self.result.pdf_sources),
                "pdf_downloaded": len(self.result.downloaded),
                "pdf_failed_downloads": len(self.result.failed_downloads),
            },
            "pdfs": [
                {
                    "url": url,
                    "file_path": self.result.downloaded.get(url),
                    "source_pages": sorted(self.result.pdf_sources[url]),
                }
                for url in sorted(self.result.pdf_sources)
            ],
            "failed_pages": self.result.failed_pages,
            "failed_downloads": self.result.failed_downloads,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    def run(self, download: bool) -> Path:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            user_agent = self._select_user_agent()
            context = browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1600, "height": 1000},
                locale="en-US",
                timezone_id="America/New_York",
            )
            self._install_stealth(context)
            page = context.new_page()
            self.crawl(page)
            if download:
                self.download_pdfs(context)
            manifest_path = self.write_manifest()
            browser.close()
        return manifest_path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl USCG sample exam pages and download linked PDFs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for PDFs and manifest (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-pages", type=int, default=250, help="Maximum number of pages to crawl."
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=900,
        help="Delay after each page load in milliseconds.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=45000,
        help="Per-request timeout in milliseconds.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Custom user-agent string. Overrides built-in fake browser agents.",
    )
    parser.add_argument(
        "--random-user-agent",
        dest="random_user_agent",
        action="store_true",
        help="Enable random fake user-agent rotation.",
    )
    parser.add_argument(
        "--no-random-user-agent",
        dest="random_user_agent",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-stealth",
        action="store_true",
        help="Disable navigator/chrome stealth patches.",
    )
    parser.add_argument(
        "--retry-403",
        type=int,
        default=3,
        help="Retry count for HTTP 403 page responses during crawl.",
    )
    parser.add_argument(
        "--headful",
        dest="headful",
        action="store_true",
        help="Run Chromium in headed mode.",
    )
    parser.add_argument(
        "--headless",
        dest="headful",
        action="store_false",
        help="Run Chromium in headless mode.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover PDF links and write manifest; do not download files.",
    )
    parser.set_defaults(headful=True, random_user_agent=False)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    scraper = USCGExamScraper(
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        delay_ms=args.delay_ms,
        timeout_ms=args.timeout_ms,
        headless=not args.headful,
        user_agent=args.user_agent,
        random_user_agent=args.random_user_agent,
        stealth=not args.no_stealth,
        retry_403=args.retry_403,
    )

    try:
        manifest_path = scraper.run(download=not args.discover_only)
    except Exception as exc:  # noqa: BLE001
        print(f"Scrape failed: {exc}", file=sys.stderr)
        print(
            "If Chromium is missing, install it with: uv run playwright install chromium",
            file=sys.stderr,
        )
        return 1

    print(f"Visited pages: {len(scraper.result.visited_pages)}")
    print(f"PDF links found: {len(scraper.result.pdf_sources)}")
    print(f"PDFs downloaded: {len(scraper.result.downloaded)}")
    print(f"Failed pages: {len(scraper.result.failed_pages)}")
    print(f"Failed downloads: {len(scraper.result.failed_downloads)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
