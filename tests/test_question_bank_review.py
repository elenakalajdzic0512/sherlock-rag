"""Offline checks for the separately reviewed question bank."""

from __future__ import annotations

import json
import shutil

import pytest

from eval.question_bank_review import BankReviewError, compile_bank
from sherlock_rag.ingest import PROJECT_ROOT


def copy_locked_run(tmp_path):
    run = tmp_path / "questions/generated_v1"
    run.mkdir(parents=True)
    source = PROJECT_ROOT / "questions/generated_v1"
    for name in ("results.json", "results.sha256", "reviews.json", "reviews_v2.json"):
        shutil.copyfile(source / name, run / name)
    return run


def test_v2_bank_excludes_unclear_question_and_preserves_historical_review(
    tmp_path,
) -> None:
    run = copy_locked_run(tmp_path)
    summary = compile_bank(tmp_path, write=True)
    assert summary["counts"] == {
        "accepted": 5,
        "candidate": 28,
        "error": 3,
        "insufficient_evidence": 5,
    }
    bank = [
        json.loads(line) for line in (run / "bank_v2.jsonl").read_text().splitlines()
    ]
    assert len(bank) == 5
    assert summary["stories_with_accepted_question"] == 5
    assert {item["id"] for item in bank} == {
        "qg-v1-01-2",
        "qg-v1-04-1",
        "qg-v1-05-1",
        "qg-v1-10-1",
        "qg-v1-12-1",
    }
    review = json.loads((run / "reviews_v2.json").read_text())
    rating = next(item for item in review["ratings"] if item["id"] == "qg-v1-10-3")
    assert rating["premise_clear_and_true"] is False
    assert "first-person speaker" in rating["rationale"]
    assert compile_bank(tmp_path, version=1)["counts"]["accepted"] == 6
    with pytest.raises(BankReviewError, match="immutable"):
        compile_bank(tmp_path, version=1, write=True)
    assert not (run / "bank.jsonl").exists()


def test_review_rejects_changed_raw_hash_and_reordered_ratings(tmp_path) -> None:
    run = copy_locked_run(tmp_path)
    review = json.loads((run / "reviews_v2.json").read_text())
    review["ratings"][0], review["ratings"][1] = (
        review["ratings"][1],
        review["ratings"][0],
    )
    (run / "reviews_v2.json").write_text(json.dumps(review))
    with pytest.raises(BankReviewError, match="Ordered review ID"):
        compile_bank(tmp_path)
    shutil.copyfile(
        PROJECT_ROOT / "questions/generated_v1/reviews_v2.json", run / "reviews_v2.json"
    )
    (run / "results.json").write_bytes((run / "results.json").read_bytes() + b" ")
    with pytest.raises(BankReviewError, match="Raw results hash"):
        compile_bank(tmp_path)


def test_v2_review_rejects_broken_historical_lineage(tmp_path) -> None:
    run = copy_locked_run(tmp_path)
    review = json.loads((run / "reviews_v2.json").read_text())
    review["supersedes_reviews_sha256"] = "wrong"
    (run / "reviews_v2.json").write_text(json.dumps(review))
    with pytest.raises(BankReviewError, match="lineage"):
        compile_bank(tmp_path)
