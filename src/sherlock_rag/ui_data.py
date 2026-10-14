"""Read verified UI content and check local RAG inputs before model loading."""

from __future__ import annotations

import json
from pathlib import Path

from sherlock_rag.ingest import PROJECT_ROOT, sha256


class UiSetupError(ValueError):
    """A local UI input is missing or differs from its frozen provenance."""

    def __init__(self, message: str, command: str):
        super().__init__(message)
        self.command = command


def load_reviewed_bank(
    root: Path = PROJECT_ROOT,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Use only the five-entry v2 bank, checked against locked raw and review data."""
    from eval.question_bank_review import BankReviewError, compile_bank

    directory = root / "questions/generated_v1"
    restore = (
        "git restore questions/generated_v1/results.json "
        "questions/generated_v1/results.sha256 questions/generated_v1/reviews.json "
        "questions/generated_v1/reviews_v2.json questions/generated_v1/summary_v2.json "
        "questions/generated_v1/bank_v2.jsonl"
    )
    try:
        compiled = compile_bank(root, version=2)
        summary = json.loads(
            (directory / "summary_v2.json").read_text(encoding="utf-8")
        )
        bank_bytes = (directory / "bank_v2.jsonl").read_bytes()
        if summary != compiled or sha256(bank_bytes) != compiled["bank_sha256"]:
            raise ValueError("V2 summary or bank hash differs from the locked review")
        bank = [json.loads(line) for line in bank_bytes.splitlines()]
        if (
            len(bank) != 5
            or compiled["counts"].get("accepted") != 5
            or compiled["stories_with_accepted_question"] != 5
            or any(
                item.get("raw_results_sha256") != compiled["raw_results_sha256"]
                for item in bank
            )
        ):
            raise ValueError("V2 bank coverage or provenance differs")
        raw = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        stories = list(
            dict.fromkeys(
                (item["story_id"], item["story_title"]) for item in raw["results"]
            )
        )
        if len(stories) != 12:
            raise ValueError("Frozen run does not contain 12 stories")
        return bank, stories
    except (OSError, ValueError, KeyError, BankReviewError) as exc:
        raise UiSetupError(
            f"Pregledana v2 banka nije dostupna ili nije provjerena: {exc}", restore
        ) from exc


def preflight_rag(root: Path = PROJECT_ROOT) -> None:
    """Validate files and frozen settings without importing or loading ML models."""
    from sherlock_rag.chunking import load_verified_source
    from sherlock_rag.rag import load_settings
    from sherlock_rag.retrieval import load_verified_chunks
    from sherlock_rag.semantic import load_index

    setup = "conda run -n sherlock-rag python -m sherlock_rag.ingest"
    chunk = (
        "conda run -n sherlock-rag python -m sherlock_rag.chunking "
        "--chunk-size-words 250 --overlap-words 50"
    )
    index = "conda run -n sherlock-rag python -m sherlock_rag.semantic build --config w250_o50"
    for path in (
        "data/raw/sherlock_holmes.txt",
        "data/processed/book.txt",
        "data/source_manifest.json",
    ):
        if not (root / path).is_file():
            raise UiSetupError(f"Nedostaje provjereni izvor: {path}", setup)
    try:
        load_verified_source(root)
    except (OSError, ValueError, KeyError) as exc:
        raise UiSetupError(f"Izvor knjige nije provjeren: {exc}", setup) from exc
    if not (root / "data/processed/chunks_w250_o50.jsonl").is_file():
        raise UiSetupError("Nedostaju A odlomci (250/50).", chunk)
    try:
        load_verified_chunks("w250_o50", root)
    except (OSError, ValueError, KeyError) as exc:
        raise UiSetupError(f"A odlomci nisu provjereni: {exc}", chunk) from exc
    for path in (
        "config/retrieval.json",
        "config/generation.json",
        "prompts/local_answer_v1.txt",
    ):
        if not (root / path).is_file():
            raise UiSetupError(
                f"Nedostaje konfiguracija: {path}", f"git restore {path}"
            )
    try:
        retrieval, generation, _ = load_settings(root, "v1", "default")
        if (
            retrieval["method"] != "semantic"
            or retrieval["configuration"] != "w250_o50"
            or generation["prompt_version"] != "local-answer-v1"
        ):
            raise ValueError("UI requires semantic A retrieval and generation v1")
    except (OSError, ValueError, KeyError) as exc:
        raise UiSetupError(
            f"Podrazumijevane konfiguracije nisu ispravne: {exc}",
            "git restore config/retrieval.json config/generation.json prompts/local_answer_v1.txt",
        ) from exc
    for path in ("indexes/w250_o50.npy", "indexes/w250_o50.json"):
        if not (root / path).is_file():
            raise UiSetupError(f"Nedostaje semantički indeks: {path}", index)
    try:
        load_index("w250_o50", root)
    except (OSError, ValueError, KeyError) as exc:
        raise UiSetupError(f"Semantički indeks nije provjeren: {exc}", index) from exc
