"""Offline metric and review checks with controlled frozen results."""

from __future__ import annotations

import json

import pytest

from sherlock_rag.evaluate_rag import encode_json
from sherlock_rag.ingest import sha256
from sherlock_rag.review_rag import (
    BOOLEAN_FIELDS,
    ReviewError,
    compile_metrics,
    validate_reviews,
)


def _case(case_id: str, dataset: str, category: str, status: str) -> dict:
    reason = "answered" if status == "answered" else "insufficient_evidence"
    if status == "error":
        reason = "invalid_model_output"
    return {
        "id": case_id,
        "dataset": dataset,
        "category": category,
        "query": case_id,
        "result": {
            "status": status,
            "reason_code": reason,
            "attempts": 2 if status == "error" else 1,
            "citations_resolved": status != "error",
            "submitted_chunks": [{"source_id": "S1", "text": "Book evidence."}],
            "claims": (
                [
                    {
                        "text": "Book evidence.",
                        "citations": [{"quote_raw": "Book evidence."}],
                    }
                ]
                if status == "answered"
                else []
            ),
            "generation_seconds": 1.0,
            "output_tokens": 10,
            "input_tokens": 100,
        },
    }


def _review(case: dict) -> dict:
    return {
        "id": case["id"],
        **{field: None for field in BOOLEAN_FIELDS},
        "support_source_id": None,
        "support_quote": None,
        "failure_class": (
            "invalid_output" if case["result"]["status"] == "error" else None
        ),
        "rationale": "Reviewed answer and sent passage; no rating applies here.",
    }


def test_complete_conditional_metrics_errors_and_corrections(tmp_path) -> None:
    cases = [_case(f"q{i:02}", "original", "direct", "answered") for i in range(1, 25)]
    cases += [
        _case(f"q{i:02}", "original", "unsupported", "abstained") for i in range(25, 31)
    ]
    cases += [
        _case(f"c{i:02}", "challenge", "unmentioned_detail", "abstained")
        for i in range(1, 7)
    ]
    cases += [
        _case(f"c{i:02}", "challenge", "false_premise", "abstained")
        for i in range(7, 11)
    ]
    cases += [
        _case(f"c{i:02}", "challenge", "outside_corpus", "abstained")
        for i in range(11, 13)
    ]
    cases[0]["result"]["status"] = "abstained"
    cases[0]["result"]["reason_code"] = "insufficient_evidence"
    cases[0]["result"]["claims"] = []
    cases[1]["result"]["status"] = "error"
    cases[1]["result"]["reason_code"] = "invalid_model_output"
    cases[1]["result"]["citations_resolved"] = False
    cases[1]["result"]["claims"] = []
    cases[1]["result"]["attempts"] = 2
    cases[6]["result"]["status"] = "abstained"
    cases[6]["result"]["reason_code"] = "insufficient_evidence"
    cases[6]["result"]["claims"] = []
    cases[30]["result"]["status"] = "error"
    cases[30]["result"]["reason_code"] = "invalid_model_output"
    cases[30]["result"]["citations_resolved"] = False
    cases[30]["result"]["attempts"] = 2
    cases[36]["result"]["status"] = "answered"
    cases[36]["result"]["reason_code"] = "answered"
    cases[37]["result"]["status"] = "answered"
    cases[37]["result"]["reason_code"] = "answered"

    reviews = [_review(case) for case in cases]
    for i in range(24):
        reviews[i].update(answer_correct=True, all_claims_supported=True)
    for i in (0, 1, 2, 6):
        reviews[i].update(
            context_sufficient=True,
            support_source_id="S1",
            support_quote="Book evidence.",
        )
    reviews[0]["abstention_justified"] = False
    reviews[2]["all_claims_supported"] = False
    reviews[6]["abstention_justified"] = False
    reviews[3]["answer_correct"] = None
    reviews[36]["supported_premise_correction"] = True
    reviews[36]["all_claims_supported"] = True
    reviews[37]["all_claims_supported"] = False

    result = {"run_id": "fake", "state": "complete", "total": 42, "results": cases}
    run_dir = tmp_path / "fake"
    run_dir.mkdir()
    frozen = encode_json(result)
    (run_dir / "results.json").write_bytes(frozen)
    (run_dir / "results.sha256").write_text(sha256(frozen) + "\n")
    (run_dir / "reviews.json").write_bytes(
        encode_json(
            {
                "results_sha256": sha256(frozen),
                "reviewer": "implementation-agent",
                "reviews": reviews,
            }
        )
    )
    metrics = compile_metrics(run_dir)
    assert (run_dir / "results.json").read_bytes() == frozen
    assert metrics["automatic"]["all"]["status_counts"] == {
        "abstained": 17,
        "answered": 23,
        "error": 2,
    }
    assert metrics["automatic"]["all"]["final_invalid_model_output"] == 2
    assert metrics["automatic"]["all"]["first_pass_mechanical_success"] == {
        "numerator": 40,
        "denominator": 42,
        "rate": 40 / 42,
    }
    assert metrics["automatic"]["all"]["repair_cases"] == 2
    assert metrics["automatic"]["original_unsupported"]["status_counts"] == {
        "abstained": 6
    }
    assert metrics["automatic"]["challenge_unmentioned_detail"]["status_counts"] == {
        "abstained": 5,
        "error": 1,
    }
    assert metrics["content_review"]["original_answerable"][
        "supported_correct_over_all"
    ] == {
        "numerator": 19,
        "denominator": 24,
        "rate": 19 / 24,
    }
    assert (
        metrics["content_review"]["original_answerable"][
            "generation_success_given_sufficient_context"
        ]["denominator"]
        == 4
    )
    assert (
        metrics["content_review"]["original_answerable"][
            "unjustified_abstention_given_sufficient_context"
        ]["numerator"]
        == 2
    )
    assert (
        metrics["content_review"]["original_answerable"][
            "null_or_not_applicable_ratings"
        ]["answer_correct"]
        == 1
    )
    false_premise = metrics["content_review"]["unsupported_and_challenges"][
        "false_premise"
    ]
    assert false_premise["strict_refusal"]["numerator"] == 2
    assert false_premise["supported_premise_correction"] == 1
    assert false_premise["displayed_unsupported_claim"] == 1
    assert false_premise["invalid_model_output"] == 0
    assert false_premise["other_technical_errors"] == 0


def test_review_requires_explicit_nulls_and_real_sent_quote() -> None:
    result = {"results": [_case("q01", "original", "direct", "answered")], "total": 1}
    review = _review(result["results"][0])
    review["context_sufficient"] = True
    payload = {
        "results_sha256": "frozen",
        "reviewer": "implementation-agent",
        "reviews": [review],
    }
    with pytest.raises(ReviewError, match="real sent quote"):
        validate_reviews(result, "frozen", payload)
    review["support_source_id"] = "S1"
    review["support_quote"] = "Book evidence."
    assert validate_reviews(result, "frozen", payload)["q01"] == review
    del review["answer_correct"]
    with pytest.raises(ReviewError, match="must be present"):
        validate_reviews(result, "frozen", payload)


def test_changed_frozen_result_refused(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps({"state": "complete", "total": 0, "results": []})
    )
    (run_dir / "results.sha256").write_text("0" * 64)
    with pytest.raises(ReviewError, match="hash differs"):
        compile_metrics(run_dir)
