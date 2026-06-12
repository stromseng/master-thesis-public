#!/usr/bin/env python3
"""Convert USCG exam PDFs to single markdown files plus extracted images."""

from __future__ import annotations

import argparse
import importlib.util
import importlib.machinery
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import types
from typing import Any, cast
import urllib.error
import urllib.request

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from examscrapers.us_coast_guard.io_utils import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MARKDOWN_ROOT_DIR,
    now_utc_iso,
    write_json,
)

console = Console()
logger = logging.getLogger("uscg_pdf_to_markdown")
PAGE_SUFFIX_RE = re.compile(r"_(\d+)\.md$")
_shims_ready = False
_pipeline_cache: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Convert USCG PDFs to markdown using PaddleOCR-VL doc-parser and write "
            "one merged markdown file per exam."
        ),
    )
    parser.add_argument("pdfs", nargs="*", help="Optional explicit PDF files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MARKDOWN_ROOT_DIR)
    parser.add_argument("--server-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.5")
    parser.add_argument("--limit-pdfs", type=int, default=None)
    parser.add_argument(
        "--exam-stem",
        action="append",
        default=None,
        help="Exam stem(s) to process. May be supplied multiple times.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Checkpoint path (default: <output-dir>/intermediate/pdf_to_markdown_progress.json)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Report path (default: <output-dir>/intermediate/pdf_to_markdown_report.json)",
    )
    parser.add_argument(
        "--disable-model-check",
        action="store_true",
        help="Skip GET /v1/models validation before running.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def api_get(url: str, timeout: float = 8.0) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "entries": {}}
    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    return payload


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_utc_iso()
    write_json(path, payload)


def _module_missing(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is None
    except (ModuleNotFoundError, ValueError):
        # ValueError happens when a partially loaded module has __spec__ = None.
        return True


def ensure_legacy_langchain_shims() -> None:
    """Provide legacy LangChain module paths expected by paddlex."""
    global _shims_ready
    if _shims_ready:
        return

    needs_docstore = _module_missing("langchain.docstore.document")
    needs_splitter = _module_missing("langchain.text_splitter")
    if not needs_docstore and not needs_splitter:
        _shims_ready = True
        return

    if needs_docstore:
        from langchain_core.documents import Document

        docstore_module: Any = sys.modules.get("langchain.docstore")
        if docstore_module is None:
            docstore_module = types.ModuleType("langchain.docstore")
            docstore_module.__spec__ = importlib.machinery.ModuleSpec(
                "langchain.docstore", loader=None
            )
            sys.modules["langchain.docstore"] = docstore_module

        document_module: Any = sys.modules.get("langchain.docstore.document")
        if document_module is None:
            document_module = types.ModuleType("langchain.docstore.document")
            document_module.__spec__ = importlib.machinery.ModuleSpec(
                "langchain.docstore.document", loader=None
            )
            sys.modules["langchain.docstore.document"] = document_module

        setattr(document_module, "Document", Document)
        setattr(docstore_module, "document", document_module)

    if needs_splitter:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter_module: Any = sys.modules.get("langchain.text_splitter")
        if splitter_module is None:
            splitter_module = types.ModuleType("langchain.text_splitter")
            splitter_module.__spec__ = importlib.machinery.ModuleSpec(
                "langchain.text_splitter", loader=None
            )
            sys.modules["langchain.text_splitter"] = splitter_module
        setattr(
            splitter_module,
            "RecursiveCharacterTextSplitter",
            RecursiveCharacterTextSplitter,
        )

    _shims_ready = True


def get_pipeline(server_url: str, model: str) -> Any:
    global _pipeline_cache
    if (
        isinstance(_pipeline_cache, dict)
        and _pipeline_cache.get("server_url") == server_url
        and _pipeline_cache.get("model") == model
    ):
        return _pipeline_cache["pipeline"]

    ensure_legacy_langchain_shims()
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCRVL  # pylint: disable=import-outside-toplevel

    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=server_url,
        vl_rec_api_model_name=model,
    )
    _pipeline_cache = {
        "server_url": server_url,
        "model": model,
        "pipeline": pipeline,
    }
    return pipeline


def resolve_pdf_paths(args: argparse.Namespace) -> list[Path]:
    if args.pdfs:
        pdf_paths = [Path(path).expanduser().resolve() for path in args.pdfs]
    else:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise RuntimeError(f"Input directory does not exist: {input_dir}")
        pdf_paths = sorted(input_dir.glob("q*.pdf"))

    if args.exam_stem:
        requested = set(args.exam_stem)
        pdf_paths = [path for path in pdf_paths if path.stem in requested]

    deduped: list[Path] = []
    seen_codes: dict[str, Path] = {}
    skipped_by_code: dict[str, list[Path]] = {}
    for path in pdf_paths:
        exam_code = path.stem.split("_", maxsplit=1)[0].lower()
        first = seen_codes.get(exam_code)
        if first is None:
            seen_codes[exam_code] = path
            deduped.append(path)
            continue
        skipped_by_code.setdefault(exam_code, []).append(path)

    if skipped_by_code:
        skipped_total = sum(len(paths) for paths in skipped_by_code.values())
        logger.info(
            "Deduplicated %s PDF(s) by exam code (qxxx). Keeping first file per code.",
            skipped_total,
        )
        for exam_code in sorted(skipped_by_code):
            kept = seen_codes[exam_code].name
            skipped = ", ".join(path.name for path in skipped_by_code[exam_code])
            logger.debug("Exam code %s: kept=%s skipped=[%s]", exam_code, kept, skipped)

    if args.limit_pdfs is not None:
        deduped = deduped[: args.limit_pdfs]

    return deduped


def _page_sort_key(path: Path) -> tuple[int, str]:
    match = PAGE_SUFFIX_RE.search(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10_000_000, path.name)


def _ensure_entries(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries_obj = state.get("entries")
    if not isinstance(entries_obj, dict):
        entries_obj = {}
        state["entries"] = entries_obj
    return cast(dict[str, dict[str, Any]], entries_obj)


def _copy_tree_contents(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if not path.is_file():
            continue
        shutil.copy2(path, dst / path.name)
        copied += 1
    return copied


def _merge_markdown_pages(exam_stem: str, tmp_dir: Path) -> tuple[str, int]:
    page_files = sorted(tmp_dir.glob(f"{exam_stem}_*.md"), key=_page_sort_key)
    if not page_files:
        single = tmp_dir / f"{exam_stem}.md"
        if single.exists():
            page_files = [single]
    if not page_files:
        raise RuntimeError(f"No markdown pages were generated for {exam_stem}")

    merged_parts = [
        path.read_text(encoding="utf-8", errors="ignore") for path in page_files
    ]
    merged = "\n\n".join(merged_parts)
    # Normalize relative image refs to the final `images/` folder in each exam directory.
    merged = merged.replace('src="imgs/', 'src="images/')
    merged = merged.replace("(imgs/", "(images/")
    return merged, len(page_files)


def process_pdf(
    *,
    pdf_path: Path,
    output_dir: Path,
    server_url: str,
    model: str,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    exam_stem = pdf_path.stem
    exam_dir = output_dir / "exams" / exam_stem
    markdown_path = exam_dir / f"{exam_stem}.md"
    images_dir = exam_dir / "images"

    if dry_run:
        return {
            "exam_stem": exam_stem,
            "status": "dry_run",
            "pdf_file": pdf_path.as_posix(),
            "markdown_file": markdown_path.as_posix(),
            "images_dir": images_dir.as_posix(),
        }

    if markdown_path.exists() and not force:
        return {
            "exam_stem": exam_stem,
            "status": "cached",
            "pdf_file": pdf_path.as_posix(),
            "markdown_file": markdown_path.as_posix(),
            "images_dir": images_dir.as_posix(),
        }

    pipeline = get_pipeline(server_url=server_url, model=model)

    tmp_dir = exam_dir / "_tmp_docparser"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = pipeline.predict(str(pdf_path))
        result_count = 0
        for result in results:
            result.save_to_markdown(str(tmp_dir))
            result_count += 1

        merged_markdown, page_count = _merge_markdown_pages(exam_stem, tmp_dir)

        if force and exam_dir.exists():
            shutil.rmtree(exam_dir)
            exam_dir.mkdir(parents=True, exist_ok=True)
        else:
            exam_dir.mkdir(parents=True, exist_ok=True)

        markdown_path.write_text(merged_markdown, encoding="utf-8")

        copied_images = _copy_tree_contents(tmp_dir / "imgs", images_dir)
        manifest = {
            "exam_stem": exam_stem,
            "pdf_file": pdf_path.as_posix(),
            "result_items": result_count,
            "page_files": page_count,
            "images_extracted": copied_images,
            "generated_at": now_utc_iso(),
        }
        write_json(exam_dir / "manifest.json", manifest)

        return {
            "exam_stem": exam_stem,
            "status": "ok",
            "pdf_file": pdf_path.as_posix(),
            "markdown_file": markdown_path.as_posix(),
            "images_dir": images_dir.as_posix(),
            "result_items": result_count,
            "page_files": page_count,
            "images_extracted": copied_images,
        }
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    progress_path = (
        args.progress_path.expanduser().resolve()
        if args.progress_path is not None
        else (intermediate_dir / "pdf_to_markdown_progress.json")
    )
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else (intermediate_dir / "pdf_to_markdown_report.json")
    )

    pdf_paths = resolve_pdf_paths(args)
    if not pdf_paths:
        logger.info("No PDF files selected.")
        return

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise RuntimeError(f"Input PDF does not exist: {pdf_path}")

    if not args.disable_model_check:
        try:
            models = api_get(f"{args.server_url.rstrip('/')}/models")
            model_ids: list[str] = []
            data = models.get("data")
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    model_ids.append(item["id"])
            if model_ids and args.model not in model_ids:
                raise RuntimeError(
                    f"Model `{args.model}` not found on server. Available: {model_ids}"
                )
            logger.info("Server model check OK: %s", args.model)
        except urllib.error.URLError as error:
            logger.warning("Could not reach model endpoint: %s", error)

    # Initialize compatibility shims once before conversion starts.
    ensure_legacy_langchain_shims()

    progress_state: dict[str, Any] = (
        {"version": 1, "entries": {}}
        if args.reset_progress
        else load_progress(progress_path)
    )
    progress_state["version"] = 1
    progress_state.setdefault("started_at", now_utc_iso())
    progress_state["mode"] = "dry_run" if args.dry_run else "write"
    progress_state["server_url"] = args.server_url
    progress_state["model"] = args.model
    progress_state["output_dir"] = output_dir.as_posix()

    if args.resume and not args.force:
        entries = _ensure_entries(progress_state)
        done_ok = {
            stem for stem, entry in entries.items() if entry.get("status") == "ok"
        }
        before = len(pdf_paths)
        pdf_paths = [path for path in pdf_paths if path.stem not in done_ok]
        logger.info(
            "Resume enabled: skipping %s completed exam(s).", before - len(pdf_paths)
        )

    if not pdf_paths:
        logger.info("Nothing left to process after resume filter.")
        return

    logger.info(
        "Selected %s PDF(s). dry_run=%s force=%s (sequential mode)",
        len(pdf_paths),
        args.dry_run,
        args.force,
    )
    logger.info("Progress path: %s", progress_path)

    report_entries: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting PDFs", total=len(pdf_paths))

        def _record_entry(entry: dict[str, Any]) -> None:
            exam_stem = str(entry.get("exam_stem", "unknown"))
            report_entries.append(entry)

            entries = _ensure_entries(progress_state)
            prev = entries.get(exam_stem, {})
            attempts = int(prev.get("attempts", 0)) + 1
            entries[exam_stem] = {
                **entry,
                "attempts": attempts,
                "updated_at": now_utc_iso(),
            }
            save_progress(progress_path, progress_state)

            if entry.get("status") in {"ok", "cached"}:
                logger.info(
                    "Completed %s: status=%s pages=%s images=%s",
                    exam_stem,
                    entry.get("status"),
                    entry.get("page_files", 0),
                    entry.get("images_extracted", 0),
                )
            elif entry.get("status") == "error":
                logger.error("Failed %s: %s", exam_stem, entry.get("error"))

        def _run_one(pdf_path: Path) -> dict[str, Any]:
            return process_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                server_url=args.server_url,
                model=args.model,
                force=args.force,
                dry_run=args.dry_run,
            )

        for pdf_path in pdf_paths:
            progress.update(task, description=f"Processing {pdf_path.stem}")
            try:
                entry = _run_one(pdf_path)
            except Exception as error:  # noqa: BLE001
                logger.exception("Failed processing %s", pdf_path.stem)
                entry = {
                    "exam_stem": pdf_path.stem,
                    "status": "error",
                    "error": str(error),
                    "pdf_file": pdf_path.as_posix(),
                }
            _record_entry(entry)
            progress.advance(task)

    summary = {
        "mode": "dry_run" if args.dry_run else "write",
        "files_selected": len(pdf_paths),
        "ok": sum(1 for entry in report_entries if entry.get("status") == "ok"),
        "cached": sum(1 for entry in report_entries if entry.get("status") == "cached"),
        "errors": sum(1 for entry in report_entries if entry.get("status") == "error"),
        "dry_run": sum(
            1 for entry in report_entries if entry.get("status") == "dry_run"
        ),
        "entries": report_entries,
    }
    write_json(report_path, summary)
    logger.info(
        "Done. files=%s ok=%s cached=%s errors=%s dry_run=%s",
        summary["files_selected"],
        summary["ok"],
        summary["cached"],
        summary["errors"],
        summary["dry_run"],
    )
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
