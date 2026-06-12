#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false
"""
Raynor Maritime Navigation Rules Question Scraper.

This script deterministically scrapes question IDs from Raynor Maritime using
`FilterQuest` requests (0000-9999 by default), classifies questions by scope
text, downloads missing images, and writes grouped canonical JSON output.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from requests import Session
from tqdm import tqdm

from generated.eval_question_models import (
    Model as EvalQuestionGroupsModel,
    NonEmptyOptionIds,
    QuestionImage,
)
from utils.repo import REPO_ROOT
from generated.eval_question_models import (
    Option as EvalOption,
)
from generated.eval_question_models import (
    Question as EvalQuestionModel,
)
from generated.eval_question_models import (
    QuestionGroup as EvalQuestionGroupModel,
)

ID_NAMESPACE = "raynor"
ZONE_SLUGS = ("international-only", "inland", "both")
SLUG_TO_FILTER_ZONE: dict[str, str] = {
    "international-only": "INTERNATIONAL_ONLY",
    "inland": "INLAND",
    "both": "BOTH",
}
SLUG_TO_QUESTION_SCOPE_LABEL: dict[str, str] = {
    "international-only": "INTERNATIONAL ONLY",
    "inland": "US INLAND ONLY",
    "both": "BOTH INTERNATIONAL & US INLAND",
}

RAYNOR_SOURCE_BASE: dict[str, str] = {
    "provider": "raynormaritime",
    "baseUrl": "https://www.raynormaritime.com/NavRules/Default.asp",
}


@dataclass(frozen=True)
class ParsedQuestion:
    original_question_id: str
    question_text: str
    zone_slug: str
    answers: dict[str, str]
    correct_option_id: str
    image_urls: list[str]


@dataclass(frozen=True)
class CliArgs:
    start_id: int
    end_id: int
    max_accepted: int | None
    output: Path | None
    rescrape: Path | None


def _attribute_to_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in cast(list[object], value):
            if isinstance(item, str):
                return item
    return None


def _is_remote_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in {"http", "https"}


def _url_to_filename(url: str) -> str:
    path = urlparse(url).path
    name = unquote(Path(path).name)
    return name.replace(" ", "_")


def _url_to_local_path(url: str) -> str:
    return _url_to_filename(url)


def _resolve_local_path(uri: str, images_dir: Path) -> Path:
    rel = Path(uri)
    if rel.parts and rel.parts[0] == "images":
        rel = Path(*rel.parts[1:])
    return images_dir / rel


def _question_is_multimodal(question: EvalQuestionModel) -> bool:
    if question.images:
        return True
    for option in question.options:
        if option.images:
            return True
    return False


def _clean_option_text(text: str) -> str:
    return re.sub(r"^(?:[A-D]\s*[.)]\s*)+", "", text).strip()


def _normalize_question_id(raw_question_id: str, zone_slug: str) -> str:
    return f"{ID_NAMESPACE}-{zone_slug}-{raw_question_id}"


class RaynorMaritimeScraper:
    """Scraper for Raynor Maritime Navigation Rules Practice Exam."""

    BASE_URL: str = "https://www.raynormaritime.com/NavRules/Default.asp"
    DEFAULT_DATA_DIR: Path = REPO_ROOT / "data" / "evals" / "raynor"
    DEFAULT_OUTPUT_FILE: Path = DEFAULT_DATA_DIR / "all_questions.json"
    LEGACY_RESCRAPE_FILE: Path = DEFAULT_DATA_DIR / "raynor_questions.json"
    DEFAULT_TEXT_ONLY_FILE: Path = DEFAULT_DATA_DIR / "text_only.json"
    DEFAULT_MULTIMODAL_FILE: Path = DEFAULT_DATA_DIR / "multimodal.json"
    DEFAULT_IMAGES_DIR: Path = DEFAULT_DATA_DIR / "images"

    def __init__(
        self,
        output_file: str | Path | None = None,
        images_dir: Path | None = None,
    ):
        self.session: Session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        self.image_session: Session = requests.Session()
        self.image_session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
        )

        self.output_file: Path = (
            Path(output_file) if output_file is not None else self.DEFAULT_OUTPUT_FILE
        )
        self.images_dir: Path = images_dir or self._infer_images_dir(self.output_file)

        self.questions_by_zone: dict[str, list[EvalQuestionModel]] = {
            zone: [] for zone in ZONE_SLUGS
        }
        self.question_ids_seen_by_zone: dict[str, set[str]] = {
            zone: set() for zone in ZONE_SLUGS
        }

        self.scanned_ids: int = 0
        self.skipped_invalid: int = 0
        self.unknown_marker_pages: int = 0
        self.international_only_count: int = 0

        self._warned_messages: set[str] = set()
        self._image_download_successes: set[str] = set()
        self._image_download_failures: set[str] = set()

    @classmethod
    def _infer_images_dir(cls, output_file: Path) -> Path:
        parent_images = output_file.parent / "images"
        if output_file.parent != Path(".") and parent_images.exists():
            return parent_images
        return cls.DEFAULT_IMAGES_DIR

    @classmethod
    def resolve_rescrape_source(cls, source_file: Path) -> Path:
        """Resolve the input file for --rescrape, with a fallback for the old default."""
        if source_file.exists():
            return source_file

        if source_file == cls.LEGACY_RESCRAPE_FILE and cls.DEFAULT_OUTPUT_FILE.exists():
            print(
                "Rescrape source "
                f"{source_file} not found; using {cls.DEFAULT_OUTPUT_FILE} instead"
            )
            return cls.DEFAULT_OUTPUT_FILE

        raise FileNotFoundError(f"Rescrape source file not found: {source_file}")

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        print(message)

    def _download_remote_image(self, uri: str, destination: Path) -> bool:
        if uri in self._image_download_successes:
            return True
        if uri in self._image_download_failures:
            return False

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self.image_session.get(uri, timeout=30)
            response.raise_for_status()
            _ = destination.write_bytes(response.content)
            self._image_download_successes.add(uri)
            time.sleep(0.2)
            return True
        except Exception as exc:
            self._image_download_failures.add(uri)
            self._warn_once(
                f"WARNING: failed to download image {uri} to {destination}: {exc}"
            )
            return False

    def _group_source(self, zone_slug: str) -> dict[str, str]:
        return {**RAYNOR_SOURCE_BASE, "exam": SLUG_TO_FILTER_ZONE[zone_slug]}

    def _group_id(self, zone_slug: str) -> str:
        return f"raynor-{zone_slug}"

    def _scope_prefixed_question_text(self, question_text: str, zone_slug: str) -> str:
        scope_label = SLUG_TO_QUESTION_SCOPE_LABEL[zone_slug]
        return f"{scope_label} {question_text}"

    def _resolve_image_uri(self, uri: str, *, warn_missing: bool) -> str:
        if _is_remote_uri(uri):
            local_rel = _url_to_local_path(uri)
            local_abs = _resolve_local_path(local_rel, self.images_dir)
            if local_abs.exists():
                return local_rel
            if warn_missing and self._download_remote_image(uri, local_abs):
                return local_rel
            if warn_missing:
                self._warn_once(
                    f"WARNING: missing local image {local_abs} for {uri}; using remote URI"
                )
            return uri

        local_abs = _resolve_local_path(uri, self.images_dir)
        if local_abs.exists():
            if uri.startswith("images/"):
                return uri.removeprefix("images/")
            return local_abs.name
        return uri

    def _question_images_from_uris(
        self, image_uris: list[str], *, warn_missing: bool
    ) -> list[QuestionImage]:
        return [
            QuestionImage(uri=self._resolve_image_uri(uri, warn_missing=warn_missing))
            for uri in image_uris
        ]

    def fetch_question_by_id(self, question_id: str) -> str:
        data = {
            "Action": "Begin",
            "FilterZone": "ALL",
            "FilterText": "",
            "FilterQuest": question_id,
            "DefaultButton": "Begin",
        }

        try:
            response = self.session.post(self.BASE_URL, data=data, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            self._warn_once(f"WARNING: failed to fetch question {question_id}: {exc}")
            return ""

    def _parse_zone_slug(self, soup: BeautifulSoup) -> tuple[str | None, bool]:
        h3_text = " ".join(
            tag.get_text(" ", strip=True).upper() for tag in soup.find_all("h3")
        )
        page_text = h3_text or soup.get_text(" ", strip=True).upper()

        if "BOTH INTERNATIONAL AND INLAND" in page_text:
            return "both", False
        if "INLAND ONLY" in page_text:
            return "inland", False
        if "INTERNATIONAL ONLY" in page_text:
            return "international-only", True

        return None, False

    def _extract_question_text(self, soup: BeautifulSoup) -> str | None:
        for td in soup.find_all("td", {"colspan": "3"}):
            b_tag = td.find("b")
            if not isinstance(b_tag, Tag):
                continue
            text = b_tag.get_text(" ", strip=True)
            if len(text) > 10:
                return text

        for text_tag in soup.find_all(["p", "div", "td", "b"]):
            text = text_tag.get_text(" ", strip=True)
            if (
                len(text) > 20
                and "Score:" not in text
                and "out of" not in text
                and re.match(r"^[A-D]\.", text) is None
            ):
                return text

        return None

    def _extract_answers(
        self, soup: BeautifulSoup
    ) -> tuple[dict[str, str], str | None]:
        answers: dict[str, str] = {}
        correct_option_id: str | None = None

        for radio in soup.find_all("input", {"name": "CheckAns"}):
            option_id = _attribute_to_string(radio.get("value"))
            if option_id is None:
                continue

            option_id = option_id.strip().upper()
            if option_id not in {"A", "B", "C", "D"}:
                continue

            label = radio.find_parent("label")
            if not isinstance(label, Tag):
                continue

            option_text = _clean_option_text(label.get_text(" ", strip=True))
            if not option_text:
                continue

            answers[option_id] = option_text
            if radio.has_attr("checked"):
                correct_option_id = option_id

        sorted_answers = {key: answers[key] for key in sorted(answers)}
        return sorted_answers, correct_option_id

    def parse_question(
        self, html: str, requested_question_id: str
    ) -> ParsedQuestion | None:
        soup = BeautifulSoup(html, "html.parser")

        zone_slug, is_international_only = self._parse_zone_slug(soup)
        if zone_slug is None:
            self.unknown_marker_pages += 1
            return None

        if is_international_only:
            self.international_only_count += 1

        question_text = self._extract_question_text(soup)
        answers, correct_option_id = self._extract_answers(soup)

        if question_text is None or not answers or correct_option_id is None:
            return None

        if correct_option_id not in answers:
            return None

        question_text = self._scope_prefixed_question_text(question_text, zone_slug)

        image_urls: list[str] = []
        for img in soup.find_all("img"):
            src = _attribute_to_string(img.get("src"))
            if src is None:
                continue
            if "logo" in src.lower():
                continue
            image_urls.append(urljoin(self.BASE_URL, src))

        return ParsedQuestion(
            original_question_id=requested_question_id,
            question_text=question_text,
            zone_slug=zone_slug,
            answers=answers,
            correct_option_id=correct_option_id,
            image_urls=image_urls,
        )

    def _to_eval_question(self, parsed: ParsedQuestion) -> EvalQuestionModel:
        question_images = self._question_images_from_uris(
            parsed.image_urls, warn_missing=True
        )

        options = [
            EvalOption(id=option_id, text=text, images=[])
            for option_id, text in sorted(parsed.answers.items())
        ]

        return EvalQuestionModel(
            id=_normalize_question_id(parsed.original_question_id, parsed.zone_slug),
            questionText=parsed.question_text,
            metadata={},
            images=question_images,
            source={"originalQuestionId": parsed.original_question_id},
            options=options,
            correctOptionIds=NonEmptyOptionIds(root=[parsed.correct_option_id]),
        )

    def scrape_ids(
        self, start_id: int = 0, end_id: int = 9999, max_accepted: int | None = None
    ) -> int:
        if start_id < 0 or end_id > 9999 or start_id > end_id:
            raise ValueError("ID range must satisfy 0 <= start_id <= end_id <= 9999")

        accepted = 0
        total = end_id - start_id + 1
        pbar = tqdm(total=total, desc="Scraping IDs", unit="id")

        try:
            for raw_id in range(start_id, end_id + 1):
                if max_accepted is not None and accepted >= max_accepted:
                    pbar.write(f"Reached max accepted questions: {max_accepted}")
                    break

                requested_question_id = f"{raw_id:04d}"
                html = self.fetch_question_by_id(requested_question_id)
                self.scanned_ids += 1

                if not html:
                    self.skipped_invalid += 1
                    _ = pbar.update(1)
                    continue

                parsed = self.parse_question(html, requested_question_id)
                if parsed is None:
                    self.skipped_invalid += 1
                    _ = pbar.update(1)
                    continue

                question = self._to_eval_question(parsed)
                question_ids_seen = self.question_ids_seen_by_zone[parsed.zone_slug]
                if question.id in question_ids_seen:
                    self.skipped_invalid += 1
                    _ = pbar.update(1)
                    continue

                question_ids_seen.add(question.id)
                self.questions_by_zone[parsed.zone_slug].append(question)
                accepted += 1

                _ = pbar.update(1)

                if accepted % 20 == 0:
                    self.save_to_json(self.output_file)
                    pbar.write(f"Auto-saved progress at {accepted} accepted questions")
        finally:
            pbar.close()

        return accepted

    @staticmethod
    def _extract_question_ids_from_file(path: Path) -> list[str]:
        """Read an existing JSON file and return sorted unique originalQuestionId values."""
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        raw_ids: set[str] = set()
        groups = data if isinstance(data, list) else [data]
        for group in groups:
            for question in group.get("questions", []):
                source = question.get("source", {})
                original_id = source.get("originalQuestionId")
                if isinstance(original_id, str):
                    raw_ids.add(original_id)
        return sorted(raw_ids)

    def scrape_from_existing(self, source_file: Path) -> int:
        """Re-scrape only the question IDs found in an existing JSON file."""
        question_ids = self._extract_question_ids_from_file(source_file)
        if not question_ids:
            print(f"No question IDs found in {source_file}")
            return 0

        print(f"Re-scraping {len(question_ids)} question IDs from {source_file}")

        accepted = 0
        pbar = tqdm(total=len(question_ids), desc="Re-scraping", unit="id")

        try:
            for requested_question_id in question_ids:
                html = self.fetch_question_by_id(requested_question_id)
                self.scanned_ids += 1

                if not html:
                    self.skipped_invalid += 1
                    _ = pbar.update(1)
                    continue

                parsed = self.parse_question(html, requested_question_id)
                if parsed is None:
                    self.skipped_invalid += 1
                    _ = pbar.update(1)
                    continue

                question = self._to_eval_question(parsed)
                question_ids_seen = self.question_ids_seen_by_zone[parsed.zone_slug]
                if question.id in question_ids_seen:
                    self.skipped_invalid += 1
                    _ = pbar.update(1)
                    continue

                question_ids_seen.add(question.id)
                self.questions_by_zone[parsed.zone_slug].append(question)
                accepted += 1

                _ = pbar.update(1)

                if accepted % 20 == 0:
                    self.save_to_json(self.output_file)
                    pbar.write(f"Auto-saved progress at {accepted} accepted questions")
        finally:
            pbar.close()

        return accepted

    def total_question_count(self) -> int:
        return sum(len(self.questions_by_zone[zone]) for zone in ZONE_SLUGS)

    def accepted_by_zone(self) -> dict[str, int]:
        return {zone: len(self.questions_by_zone[zone]) for zone in ZONE_SLUGS}

    def all_questions(self) -> list[EvalQuestionModel]:
        return [
            question for zone in ZONE_SLUGS for question in self.questions_by_zone[zone]
        ]

    def download_missing_images(self) -> None:
        """Download remote images referenced by all question groups."""
        urls: set[str] = set()
        for question in self.all_questions():
            for image in question.images or []:
                if _is_remote_uri(image.uri):
                    urls.add(image.uri)

        remote_urls = sorted(urls)
        if not remote_urls:
            print("No remote images to download")
            return

        skipped = 0
        downloaded = 0
        failed = 0

        for url in tqdm(remote_urls, desc="Downloading images", unit="img"):
            filename = _url_to_filename(url)
            dest = self.images_dir / filename

            if dest.exists():
                skipped += 1
                continue

            if self._download_remote_image(url, dest):
                downloaded += 1
            else:
                failed += 1

        print(
            f"Done: {downloaded} downloaded, {skipped} skipped (already exist), {failed} failed"
        )

        self._localize_question_images()

    def _localize_question_images(self) -> None:
        """Rewrite image URIs to local images/<name> paths when files are present."""
        for zone_slug in ZONE_SLUGS:
            zone_questions = self.questions_by_zone[zone_slug]
            for index, question in enumerate(zone_questions):
                normalized_images = [
                    QuestionImage(
                        id=image.id,
                        caption=image.caption,
                        uri=self._resolve_image_uri(image.uri, warn_missing=False),
                    )
                    for image in question.images or []
                ]
                zone_questions[index] = question.model_copy(
                    update={"images": normalized_images}
                )

    def _group_payload(self, zone_slug: str) -> EvalQuestionGroupModel:
        return EvalQuestionGroupModel(
            id=self._group_id(zone_slug),
            metadata={"filterZone": SLUG_TO_FILTER_ZONE[zone_slug]},
            source=self._group_source(zone_slug),
            questions=self.questions_by_zone[zone_slug],
        )

    def save_to_json(self, filename: str | Path | None = None) -> None:
        """Save scraped question groups to a JSON file."""
        target = Path(filename) if filename is not None else self.output_file
        target.parent.mkdir(parents=True, exist_ok=True)

        self._localize_question_images()

        payload = [
            self._group_payload(zone_slug).model_dump(mode="json", exclude_none=True)
            for zone_slug in ZONE_SLUGS
        ]
        with target.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        print(f"Saved {self.total_question_count()} questions to {target}")

    def save_split_outputs(self) -> dict[str, int]:
        """Save all_questions, text-only, and multimodal JSON files."""
        self._localize_question_images()

        all_questions = self.all_questions()
        multimodal_questions = [q for q in all_questions if _question_is_multimodal(q)]
        text_only_questions = [
            q for q in all_questions if not _question_is_multimodal(q)
        ]

        data_dir = self.output_file.parent
        data_dir.mkdir(parents=True, exist_ok=True)

        def _write_group(
            path: Path, group_id: str, questions: list[EvalQuestionModel]
        ) -> None:
            group = EvalQuestionGroupModel(
                id=group_id,
                metadata={
                    "provider": ID_NAMESPACE,
                    "questionCount": len(questions),
                },
                source=RAYNOR_SOURCE_BASE,
                questions=questions,
            )
            payload = EvalQuestionGroupsModel(root=[group]).model_dump(
                mode="json",
                exclude_none=True,
            )
            with path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)

        _write_group(
            data_dir / "all_questions.json",
            "raynor-all",
            all_questions,
        )
        _write_group(
            data_dir / "text_only.json",
            "raynor-all-text-only",
            text_only_questions,
        )
        _write_group(
            data_dir / "multimodal.json",
            "raynor-all-multimodal",
            multimodal_questions,
        )

        stats = {
            "all_questions_total": len(all_questions),
            "all_questions_text_only": len(text_only_questions),
            "all_questions_multimodal": len(multimodal_questions),
        }
        print(f"\nSplit output to {data_dir}:")
        for key, count in stats.items():
            print(f"  {key}: {count}")
        return stats


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Scrape Raynor questions via deterministic 4-digit ID sweep"
    )
    _ = parser.add_argument("--start-id", type=int, default=0, help="Start ID (0-9999)")
    _ = parser.add_argument("--end-id", type=int, default=9999, help="End ID (0-9999)")
    _ = parser.add_argument(
        "--max-accepted",
        type=int,
        default=None,
        help="Stop after this many accepted questions",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path override",
    )
    _ = parser.add_argument(
        "--rescrape",
        type=Path,
        nargs="?",
        const=RaynorMaritimeScraper.DEFAULT_OUTPUT_FILE,
        default=None,
        help="Re-scrape question IDs from an existing JSON file "
        "(defaults to data/evals/raynor/all_questions.json)",
    )
    parsed = cast(dict[str, object], vars(parser.parse_args()))
    start_id = parsed.get("start_id")
    end_id = parsed.get("end_id")
    max_accepted = parsed.get("max_accepted")
    output = parsed.get("output")
    rescrape = parsed.get("rescrape")

    if not isinstance(start_id, int):
        raise ValueError("--start-id must be an int")
    if not isinstance(end_id, int):
        raise ValueError("--end-id must be an int")
    if max_accepted is not None and not isinstance(max_accepted, int):
        raise ValueError("--max-accepted must be an int")
    if output is not None and not isinstance(output, Path):
        raise ValueError("--output must be a path")
    if rescrape is not None and not isinstance(rescrape, Path):
        raise ValueError("--rescrape must be a path")

    return CliArgs(
        start_id=start_id,
        end_id=end_id,
        max_accepted=max_accepted,
        output=output,
        rescrape=rescrape,
    )


def main() -> None:
    """Run the scraper."""
    args = parse_args()

    output_file = args.output or RaynorMaritimeScraper.DEFAULT_OUTPUT_FILE
    rescrape_source = (
        RaynorMaritimeScraper.resolve_rescrape_source(args.rescrape)
        if args.rescrape is not None
        else None
    )
    scraper = RaynorMaritimeScraper(output_file=output_file)

    try:
        if rescrape_source is not None:
            accepted = scraper.scrape_from_existing(rescrape_source)
        else:
            accepted = scraper.scrape_ids(
                start_id=args.start_id,
                end_id=args.end_id,
                max_accepted=args.max_accepted,
            )
        scraper.download_missing_images()
        scraper.save_to_json(output_file)
        scraper.save_split_outputs()
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
        print("Saving progress before exiting...")
        scraper.save_to_json(output_file)
        scraper.save_split_outputs()
        return
    except Exception:
        print("\n\nError during scraping. Saving progress before exiting...")
        scraper.save_to_json(output_file)
        scraper.save_split_outputs()
        raise

    accepted_by_zone = scraper.accepted_by_zone()

    print("\n" + "=" * 50)
    print(f"Scanned IDs: {scraper.scanned_ids}")
    print(f"Accepted questions: {accepted}")
    print(f"Skipped invalid pages: {scraper.skipped_invalid}")
    print(f"Unknown marker pages: {scraper.unknown_marker_pages}")
    print(f"INTERNATIONAL ONLY encountered: {scraper.international_only_count}")
    print(f"International-only questions: {accepted_by_zone['international-only']}")
    print(f"Inland questions: {accepted_by_zone['inland']}")
    print(f"Both questions: {accepted_by_zone['both']}")

    all_questions = scraper.all_questions()
    if all_questions:
        sample = all_questions[0]
        option_summary = {option.id: option.text for option in sample.options}
        print("\nSample question:")
        print(f"ID: {sample.id}")
        print(f"Question: {sample.questionText}")
        print(f"Answers: {option_summary}")
        print(f"Correct: {sample.correctOptionIds.root}")


if __name__ == "__main__":
    main()
