#!/usr/bin/env python3
"""Download and convert NavReas dataset into eval question-group format.

Uses langextract to LLM-extract structured MCQ data from each entry.

Source: https://github.com/MO-RISE/navreas-dataset
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any
from urllib.request import urlopen

from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from generated.eval_question_models import (
    Model as EvalQuestionGroupsModel,
    NonEmptyOptionIds,
    Option as EvalOption,
    Question as EvalQuestionModel,
    QuestionGroup as EvalQuestionGroupModel,
    QuestionImage,
)
from utils.repo import REPO_ROOT

REPO = "MO-RISE/navreas-dataset"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/questions"
CATEGORIES = [
    "scene_understanding",
    "colreg_compliance_and_good_seamanship",
    "spatial_relationship_and_estimation_of_motion",
]

DATA_DIR = REPO_ROOT / "data" / "evals" / "navreas"
IMAGE_DIR = DATA_DIR / "images"

DEFAULT_MODEL_ID = "openai/gpt-oss-120b"
DEFAULT_LITELLM_BASE_URL = "https://llm.hpc.ntnu.no/v1"

CHOICE_LETTERS = tuple("ABCDEFGHIJ")

# ---------------------------------------------------------------------------
# Prompt & examples for langextract
# ---------------------------------------------------------------------------

PROMPT_DESCRIPTION = """\
Extract structured MCQ data from NavReas maritime navigation questions.

Each entry contains a marine traffic situation description, a question, multiple choice options labeled (A)-(J), and the correct answer.

Requirements:
- Return one extraction per entry using extraction_class = "navreas_question"
- Extract attributes:
  - situation: the marine traffic situation description (text before "Question:")
  - question: the question being asked (text after "Question:" before the options)
  - choice_A through choice_J: text of each option (only include options that exist)
  - correct_answer: single letter A-J identifying the correct option
- The correct answer may be given as text (e.g. "yes") or as a letter (e.g. "(A)"). Map text answers to the matching option letter.
- Handle typos in option text (e.g. "approacing" matches answer "approaching")
"""

EXAMPLE_SPECS: list[dict[str, Any]] = [
    # 1. scene_understanding — 2 options, text answer
    {
        "text": (
            "Marine traffic situation: The own ship, is a 122.0 meters long Passenger/Ro-Ro Cargo Ship moving at a speed of 10.0 knots on a course of 0.0 degrees. Around the own ship there are 3 target ships. Target ship 1, a General Cargo Ship of 50.0 meters, making 9.0 knots on a course of 83.2 degrees. Target ship 1 lies 4.2 nautical miles off, bearing 315.0 degrees relative. Question: Is there any risk of collision between the target ship 1 and the own ship?\n        Please select the appropriate option:\n        (A) yes\n        (B) no."
            "\n---\nCorrect answer: yes"
        ),
        "extraction_text": "Is there any risk of collision between the target ship 1 and the own ship?",
        "attributes": {
            "situation": "Marine traffic situation: The own ship, is a 122.0 meters long Passenger/Ro-Ro Cargo Ship moving at a speed of 10.0 knots on a course of 0.0 degrees. Around the own ship there are 3 target ships. Target ship 1, a General Cargo Ship of 50.0 meters, making 9.0 knots on a course of 83.2 degrees. Target ship 1 lies 4.2 nautical miles off, bearing 315.0 degrees relative.",
            "question": "Is there any risk of collision between the target ship 1 and the own ship?",
            "choice_A": "yes",
            "choice_B": "no",
            "correct_answer": "A",
        },
    },
    # 2. colreg_compliance — 10 options, letter answer
    {
        "text": (
            "Marine traffic situation: \nThere a 3 ships in this marine traffic situation:\nShip 1 has s course of 90 degrees and a speed of 20 knots.\nShip 2 has a course of 90 degrees and a speed of 10 knots.\nShip 3 has a course of 315 degrees and a speed of 10 knots.\n\n Question: Which of the following possible solutions is the safest? \n \n(A) solution 1: Ship 1 reduces her speed matching the speed of ship 2.\n(B) solution 2: Ship 1 changes her course.\n(C) solution 3: Ship 1 changes her course to port.\n(D) solution 4: Ship 1 changes her course to starboard.\n(E) solution 5: Ship 1 changes her course to port to overtake ship 2.\n(F) solution 6: Ship 2 changes her course to port.\n(G) solution 7: Ship 2 reduces her speed.\n(H) solution 8: Ship 3 changes her course to starboard.\n(I) solution 9: Ship 3 changes her course to starboard.\n(J) solution 10: Ship 3 changes her course to port."
            "\n---\nCorrect answer: (A)"
        ),
        "extraction_text": "Which of the following possible solutions is the safest?",
        "attributes": {
            "situation": "Marine traffic situation: \nThere a 3 ships in this marine traffic situation:\nShip 1 has s course of 90 degrees and a speed of 20 knots.\nShip 2 has a course of 90 degrees and a speed of 10 knots.\nShip 3 has a course of 315 degrees and a speed of 10 knots.",
            "question": "Which of the following possible solutions is the safest?",
            "choice_A": "solution 1: Ship 1 reduces her speed matching the speed of ship 2.",
            "choice_B": "solution 2: Ship 1 changes her course.",
            "choice_C": "solution 3: Ship 1 changes her course to port.",
            "choice_D": "solution 4: Ship 1 changes her course to starboard.",
            "choice_E": "solution 5: Ship 1 changes her course to port to overtake ship 2.",
            "choice_F": "solution 6: Ship 2 changes her course to port.",
            "choice_G": "solution 7: Ship 2 reduces her speed.",
            "choice_H": "solution 8: Ship 3 changes her course to starboard.",
            "choice_I": "solution 9: Ship 3 changes her course to starboard.",
            "choice_J": "solution 10: Ship 3 changes her course to port.",
            "correct_answer": "A",
        },
    },
    # 3. spatial_relationship — inline options with typo, text answer
    {
        "text": (
            "Marine traffic situation: The own ship, is a 122.0 meters long Passenger/Ro-Ro Cargo Ship moving at a speed of 10.0 knots on a course of 0.0 degrees. Around the own ship there is 1 target ship. Target ship 1, a Passenger/Ro-Ro Cargo Ship of 178.0 meters, making 18.0 knots on a course of 133.2 degrees. Target ship 1 lies 6.7 nautical miles off, bearing 280.0 degrees relative. Question: Is the target ship 1 approaching or receding the own ship? \n        Please select the appropriate option:\n        (A) approacing or (B) receding."
            "\n---\nCorrect answer: approaching"
        ),
        "extraction_text": "Is the target ship 1 approaching or receding the own ship?",
        "attributes": {
            "situation": "Marine traffic situation: The own ship, is a 122.0 meters long Passenger/Ro-Ro Cargo Ship moving at a speed of 10.0 knots on a course of 0.0 degrees. Around the own ship there is 1 target ship. Target ship 1, a Passenger/Ro-Ro Cargo Ship of 178.0 meters, making 18.0 knots on a course of 133.2 degrees. Target ship 1 lies 6.7 nautical miles off, bearing 280.0 degrees relative.",
            "question": "Is the target ship 1 approaching or receding the own ship?",
            "choice_A": "approacing",
            "choice_B": "receding",
            "correct_answer": "A",
        },
    },
]


# ---------------------------------------------------------------------------
# langextract helpers
# ---------------------------------------------------------------------------


def load_langextract_module() -> Any:
    """Import langextract lazily with a clear error."""
    try:
        import langextract as lx  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError(
            "langextract is not installed. Add it to the python environment "
            "before running this script (e.g. `uv add langextract`)."
        ) from error
    return lx


def register_idun_litellm_provider(lx: Any) -> str:
    """Register custom Idun provider and return provider class name."""
    from langextract.core import base_model, exceptions  # pylint: disable=import-outside-toplevel
    from langextract.core import types as core_types  # pylint: disable=import-outside-toplevel

    @lx.providers.router.register(
        r"^IdunLiteLLMLanguageModel$",
        r"^openai/",
        r"^mistralai/",
        r"^zai-org/",
        r"^moonshotai/",
        r"^Qwen/",
        r"^NorwAI/",
        priority=30,
    )
    class IdunLiteLLMLanguageModel(base_model.BaseLanguageModel):
        model_id: str
        api_key: str | None
        base_url: str
        temperature: float | None
        max_workers: int
        _client: Any

        def __init__(
            self,
            model_id: str = DEFAULT_MODEL_ID,
            api_key: str | None = None,
            base_url: str = DEFAULT_LITELLM_BASE_URL,
            temperature: float | None = 0.0,
            max_workers: int = 8,
            **kwargs: Any,
        ) -> None:
            try:
                import openai  # pylint: disable=import-outside-toplevel
            except ImportError as error:
                raise exceptions.InferenceConfigError(
                    "openai package is required for Idun provider."
                ) from error

            self.model_id = model_id
            self.api_key = api_key
            self.base_url = base_url
            self.temperature = temperature
            self.max_workers = max_workers
            self._extra_kwargs = kwargs or {}
            if not self.api_key:
                raise exceptions.InferenceConfigError(
                    "LITE_LLM_API_KEY is required for Idun provider."
                )
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            super().__init__()

        @property
        def requires_fence_output(self) -> bool:
            return False

        def infer(self, batch_prompts: list[str], **kwargs: Any):  # type: ignore[override]
            merged_kwargs = self.merge_kwargs(kwargs)
            temperature = merged_kwargs.get("temperature", self.temperature)
            top_p = merged_kwargs.get("top_p")
            max_output_tokens = merged_kwargs.get("max_output_tokens")

            def _call(prompt: str) -> str:
                request: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": "Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                }
                if temperature is not None:
                    request["temperature"] = temperature
                if top_p is not None:
                    request["top_p"] = top_p
                if max_output_tokens is not None:
                    request["max_tokens"] = max_output_tokens
                response = self._client.chat.completions.create(**request)
                return response.choices[0].message.content or ""

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(_call, p) for p in batch_prompts]
                for future in futures:
                    try:
                        output_text = future.result()
                        yield [core_types.ScoredOutput(score=1.0, output=output_text)]
                    except Exception as error:
                        raise exceptions.InferenceRuntimeError(
                            f"Idun LiteLLM API error: {error}", original=error
                        ) from error

    return "IdunLiteLLMLanguageModel"


def build_langextract_examples(lx: Any) -> list[Any]:
    """Build few-shot examples for langextract."""
    examples: list[Any] = []
    for spec in EXAMPLE_SPECS:
        examples.append(
            lx.data.ExampleData(
                text=spec["text"],
                extractions=[
                    lx.data.Extraction(
                        extraction_class="navreas_question",
                        extraction_text=spec["extraction_text"],
                        attributes=spec["attributes"],
                    )
                ],
            )
        )
    return examples


def run_langextract_single(
    text: str,
    lx: Any,
    config: Any,
    examples: list[Any],
) -> dict[str, Any] | None:
    """Run langextract on a single entry's text. Returns extraction dict or None."""
    annotated = lx.extract(
        text_or_documents=text,
        prompt_description=PROMPT_DESCRIPTION,
        examples=examples,
        config=config,
        use_schema_constraints=False,
        fence_output=False,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        max_char_buffer=8000,
        show_progress=False,
    )
    doc_dict = lx.data_lib.annotated_document_to_dict(annotated)
    extractions = doc_dict.get("extractions", [])
    for ext in extractions:
        if ext.get("extraction_class") == "navreas_question":
            return ext
    return None


def parse_extraction(
    extraction: dict[str, Any],
) -> tuple[str, str, list[EvalOption], list[str]] | None:
    """Parse langextract extraction into (situation, question, options, correct_ids).

    Returns None if extraction is invalid.
    """
    attrs = extraction.get("attributes") or {}

    situation = str(attrs.get("situation", "")).strip()
    question = str(attrs.get("question", "")).strip()
    if not situation or not question:
        return None

    options: list[EvalOption] = []
    for letter in CHOICE_LETTERS:
        val = attrs.get(f"choice_{letter}")
        if val is not None:
            text = str(val).strip()
            if text:
                options.append(EvalOption(id=letter, text=text))

    if not options:
        return None

    correct_raw = str(attrs.get("correct_answer", "")).strip().upper()
    # Extract just the letter
    m = re.search(r"[A-J]", correct_raw)
    if not m:
        return None
    correct_letter = m.group(0)

    if not any(o.id == correct_letter for o in options):
        return None

    return situation, question, options, [correct_letter]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_category(category: str) -> list[dict[str, object]]:
    """Download a category JSON file from GitHub."""
    url = f"{RAW_BASE}/{category}.json"
    print(f"Downloading {url}")
    with urlopen(url, timeout=60) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array for {category}")
    return payload


def _download_image(filename: str) -> None:
    """Download a single image from GitHub if not already present."""
    dest = IMAGE_DIR / filename
    if dest.exists():
        return
    url = f"{RAW_BASE}/{filename}"
    print(f"Downloading image {filename}")
    with urlopen(url, timeout=60) as response:  # noqa: S310
        dest.write_bytes(response.read())


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def _process_entry(
    index: int,
    entry: dict[str, object],
    category: str,
    lx: Any,
    config: Any,
    examples: list[Any],
) -> EvalQuestionModel | None:
    """Process a single NavReas entry using langextract."""
    prompt = entry.get("prompt", [])
    answers = entry.get("answers", [])
    image_name = entry.get("image", "")

    if not isinstance(prompt, list) or not isinstance(answers, list):
        return None

    # Find user content
    user_content = ""
    for msg in prompt:
        if isinstance(msg, dict) and str(msg.get("role", "")) == "user":
            user_content = str(msg.get("content", ""))

    if not user_content:
        return None

    # Build langextract input
    answer_text = str(answers[0]) if answers else ""
    lx_input = f"{user_content}\n---\nCorrect answer: {answer_text}"

    extraction = run_langextract_single(lx_input, lx, config, examples)
    if extraction is None:
        print(f"  Warning: no extraction for {category} Q{index}")
        return None

    parsed = parse_extraction(extraction)
    if parsed is None:
        print(f"  Warning: invalid extraction for {category} Q{index}")
        return None

    situation, question, options, correct_ids = parsed

    # Build images list
    images: list[QuestionImage] = []
    if isinstance(image_name, str) and image_name:
        images.append(QuestionImage(uri=image_name))

    question_id = f"navreas-{category}-{index:04d}"
    question_text = f"{situation}\n\nQuestion: {question}"

    return EvalQuestionModel(
        id=question_id,
        questionText=question_text,
        metadata={
            "category": category,
            "originalIndex": index,
        },
        images=images if images else None,
        source={
            "provider": "github",
            "repository": REPO,
            "category": category,
        },
        options=options,
        correctOptionIds=NonEmptyOptionIds(root=correct_ids),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NavReas dataset into structured MCQ JSON using langextract."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load repository root .env so IDUN credentials are picked up automatically.
    load_dotenv(REPO_ROOT / ".env", override=False)

    api_key = args.api_key or os.environ.get("LITE_LLM_API_KEY")
    base_url = args.base_url or os.environ.get(
        "LITE_LLM_BASE_URL", DEFAULT_LITELLM_BASE_URL
    )
    if not api_key:
        raise RuntimeError(
            "LITE_LLM_API_KEY is required (set in .env or pass --api-key)."
        )
    max_workers: int = args.max_workers

    # Initialize langextract
    lx = load_langextract_module()
    provider_name = register_idun_litellm_provider(lx)
    examples = build_langextract_examples(lx)

    config = lx.factory.ModelConfig(
        model_id=args.model_id,
        provider=provider_name,
        provider_kwargs={
            "api_key": api_key,
            "base_url": base_url,
            "temperature": 0.0,
            "max_workers": max_workers,
        },
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        # Download or read locally
        local_path = DATA_DIR / f"{category}.json"
        if local_path.exists():
            print(f"Reading local {local_path.name}")
            entries = json.loads(local_path.read_text(encoding="utf-8"))
        else:
            entries = _download_category(category)
            # Save downloaded JSON locally for future runs
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"  Saved {local_path.name}")

        # Collect and download images
        image_names: set[str] = set()
        for entry in entries:
            img = entry.get("image", "")
            if isinstance(img, str) and img:
                image_names.add(img)

        for img_name in sorted(image_names):
            _download_image(img_name)

        # Process entries in parallel with progress bar
        questions: list[EvalQuestionModel] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_entry, index, entry, category, lx, config, examples
                ): index
                for index, entry in enumerate(entries, start=1)
            }
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(category, total=len(futures))
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            questions.append(result)
                    except Exception as exc:
                        progress.console.print(
                            f"  Error processing {category} Q{idx}: {exc}"
                        )
                    progress.advance(task)

        # Sort by index to maintain order
        questions.sort(key=lambda q: q.id)

        print(f"  {category}: {len(questions)} questions converted")

        # Write output
        group = EvalQuestionGroupModel(
            id=f"navreas-{category}",
            metadata={
                "dataset": "navreas",
                "category": category,
                "questionCount": len(questions),
                "convertedAtUtc": datetime.now(UTC).isoformat(),
            },
            source={
                "provider": "github",
                "repository": REPO,
                "sourceUrl": f"https://github.com/{REPO}",
                "branch": BRANCH,
            },
            questions=questions,
        )

        output_path = DATA_DIR / f"{category}.eval.json"
        payload = EvalQuestionGroupsModel(root=[group]).model_dump(
            mode="json", exclude_none=True
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
