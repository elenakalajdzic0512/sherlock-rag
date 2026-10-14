"""Validate separate content ratings and build the reviewed question bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256

FIELDS = (
    "premise_clear_and_true",
    "asks_available_fact",
    "answer_correct",
    "quote_supports_answer",
    "quote_sufficient",
    "quote_precise",
)
RUN_DIR = Path("questions/generated_v1")


class BankReviewError(ValueError):
    """The review does not match locked raw generation or its citations."""


def compile_bank(
    root: Path = PROJECT_ROOT, *, version: int = 2, write: bool = False
) -> dict:
    if version not in (1, 2):
        raise BankReviewError("Unknown review version")
    if version == 1 and write:
        raise BankReviewError("Historical v1 review and bank are immutable")
    directory = root / RUN_DIR
    raw_bytes = (directory / "results.json").read_bytes()
    raw_sha = sha256(raw_bytes)
    if raw_sha != (directory / "results.sha256").read_text().strip():
        raise BankReviewError("Raw results hash differs")
    raw = json.loads(raw_bytes)
    if raw.get("state") != "complete" or len(raw.get("results", [])) != 36:
        raise BankReviewError("Raw run is incomplete")
    suffix = "" if version == 1 else "_v2"
    review_path = directory / f"reviews{suffix}.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("results_sha256") != raw_sha
        or review.get("reviewer") != "implementation-agent"
    ):
        raise BankReviewError("Review provenance differs")
    if version == 2:
        historical_sha = sha256((directory / "reviews.json").read_bytes())
        if (
            review.get("rating_schema") != "question-content-v2"
            or review.get("supersedes_reviews_sha256") != historical_sha
        ):
            raise BankReviewError("V2 review lineage differs")
    ratings = review.get("ratings")
    if not isinstance(ratings, list) or len(ratings) != 36:
        raise BankReviewError("Review must rate all 36 selections")
    bank = []
    counts = Counter()
    by_story: dict[str, Counter] = {}
    for case, rating in zip(raw["results"], ratings, strict=True):
        if (
            rating.get("id") != case["id"]
            or not isinstance(rating.get("rationale"), str)
            or not rating["rationale"].strip()
        ):
            raise BankReviewError("Ordered review ID or rationale differs")
        story = case["story_id"]
        story_counts = by_story.setdefault(story, Counter())
        status = case["status"]
        counts[status] += 1
        story_counts[status] += 1
        if status == "candidate":
            if any(type(rating.get(field)) is not bool for field in FIELDS):
                raise BankReviewError(f"Candidate rating missing: {case['id']}")
            citation = case["citation"]
            if (
                citation["source_id"] != "C1"
                or citation["chunk_id"] != case["chunk_id"]
            ):
                raise BankReviewError(
                    f"Candidate citation source differs: {case['id']}"
                )
            if case["duplicate_of"]:
                counts["duplicate"] += 1
                story_counts["duplicate"] += 1
            if all(rating[field] for field in FIELDS) and case["duplicate_of"] is None:
                counts["accepted"] += 1
                story_counts["accepted"] += 1
                bank.append(
                    {
                        "id": case["id"],
                        "story_id": case["story_id"],
                        "story_title": case["story_title"],
                        "chunk_id": case["chunk_id"],
                        "question": case["question"],
                        "expected_answer": case["expected_answer"],
                        "citation": citation,
                        "source_sha256": case["source_sha256"],
                        "prompt_sha256": case["prompt_sha256"],
                        "config_sha256": case["config_sha256"],
                        "runtime_fingerprint_sha256": case[
                            "runtime_fingerprint_sha256"
                        ],
                        "raw_results_sha256": raw_sha,
                    }
                )
        elif any(rating.get(field) is not None for field in FIELDS):
            raise BankReviewError(f"Noncandidate has content rating: {case['id']}")
    bank_bytes = b"".join(
        (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for item in bank
    )
    output = {
        "raw_results_sha256": raw_sha,
        "reviews_sha256": sha256(review_path.read_bytes()),
        "bank_sha256": sha256(bank_bytes),
        "total": len(raw["results"]),
        "counts": dict(sorted(counts.items())),
        "by_story": {
            story: dict(sorted(values.items()))
            for story, values in sorted(by_story.items())
        },
        "stories_with_accepted_question": sum(
            values["accepted"] > 0 for values in by_story.values()
        ),
    }
    if version == 2:
        output["review_version"] = 2
        output["supersedes_reviews_sha256"] = review["supersedes_reviews_sha256"]
    if write:
        bank_path = directory / "bank_v2.jsonl"
        if bank_path.exists() and bank_path.read_bytes() != bank_bytes:
            raise BankReviewError("Existing reviewed bank differs; refusing overwrite")
        atomic_write(bank_path, bank_bytes)
        atomic_write(
            directory / "summary_v2.json",
            (
                json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode(),
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            compile_bank(version=args.version, write=args.write),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
