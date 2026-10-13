"""Compute explicit-denominator answer metrics from frozen results and agent reviews."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256


class ReviewError(ValueError):
    """Frozen results or their separate content reviews are inconsistent."""


BOOLEAN_FIELDS = (
    "answer_correct",
    "all_claims_supported",
    "context_sufficient",
    "abstention_justified",
    "supported_premise_correction",
    "atomic_claims",
    "quote_length_ok",
)


def load_frozen_results(run_dir: Path) -> tuple[dict, str]:
    data = (run_dir / "results.json").read_bytes()
    expected = (run_dir / "results.sha256").read_text().strip()
    actual = sha256(data)
    if actual != expected:
        raise ReviewError("Frozen results hash differs")
    result = json.loads(data)
    if result.get("state") != "complete" or len(
        result.get("results", [])
    ) != result.get("total"):
        raise ReviewError("Results are incomplete")
    return result, actual


def validate_reviews(result: dict, result_sha: str, reviews: dict) -> dict[str, dict]:
    if (
        reviews.get("results_sha256") != result_sha
        or reviews.get("reviewer") != "implementation-agent"
    ):
        raise ReviewError("Review provenance differs from frozen results")
    rows = reviews.get("reviews")
    if not isinstance(rows, list) or len(rows) != len(result["results"]):
        raise ReviewError("Every result needs exactly one review")
    by_id = {}
    for case, review in zip(result["results"], rows, strict=True):
        if review.get("id") != case["id"] or review["id"] in by_id:
            raise ReviewError("Review IDs or order differ")
        if any(field not in review for field in BOOLEAN_FIELDS):
            raise ReviewError("Every content rating must be present, including nulls")
        if any(
            review[field] is not None and not isinstance(review[field], bool)
            for field in BOOLEAN_FIELDS
        ):
            raise ReviewError("Content ratings must be true, false, or null")
        expected_failure = None
        if case["result"]["status"] == "error":
            expected_failure = (
                "invalid_output"
                if case["result"].get("reason_code") == "invalid_model_output"
                else "technical_error"
            )
        if "failure_class" not in review or review["failure_class"] != expected_failure:
            raise ReviewError(
                f"Failure class differs from runtime result: {case['id']}"
            )
        if (
            not isinstance(review.get("rationale"), str)
            or not review["rationale"].strip()
        ):
            raise ReviewError("Every review needs a rationale")
        support_id = review.get("support_source_id")
        support_quote = review.get("support_quote")
        if review.get("context_sufficient") is True:
            sources = {
                p["source_id"]: p["text"]
                for p in case["result"].get("submitted_chunks", [])
            }
            if (
                support_id not in sources
                or not isinstance(support_quote, str)
                or not support_quote.strip()
                or support_quote not in sources[support_id]
            ):
                raise ReviewError(
                    f"Sufficient context needs a real sent quote: {case['id']}"
                )
        elif support_id is not None or support_quote is not None:
            raise ReviewError("Only sufficient context may carry a support quote")
        by_id[review["id"]] = review
    return by_id


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _known_measurements(
    cases: list[dict], field: str, *, integer: bool = False
) -> list[int | float]:
    values = []
    for case in cases:
        value = case["result"].get(field)
        valid_type = type(value) is int if integer else type(value) in (int, float)
        if valid_type and math.isfinite(value) and value >= 0:
            values.append(value)
    return values


def automatic_metrics(cases: list[dict]) -> dict:
    statuses = Counter(case["result"]["status"] for case in cases)
    reasons = Counter(case["result"].get("reason_code") for case in cases)
    generated = [
        case
        for case in cases
        if type(case["result"].get("attempts")) is int
        and case["result"]["attempts"] > 0
    ]
    first_pass = sum(
        type(case["result"].get("attempts")) is int
        and case["result"]["attempts"] == 1
        and case["result"]["status"] in ("answered", "abstained")
        and case["result"].get("citations_resolved") is True
        for case in cases
    )
    repairs = sum(
        type(case["result"].get("attempts")) is int and case["result"]["attempts"] == 2
        for case in cases
    )
    attempts = _known_measurements(cases, "attempts", integer=True)
    input_tokens = _known_measurements(cases, "input_tokens", integer=True)
    output_tokens = _known_measurements(cases, "output_tokens", integer=True)
    times = _known_measurements(cases, "generation_seconds")
    generated_input_tokens = _known_measurements(
        generated, "input_tokens", integer=True
    )
    quotes = [
        len(citation["quote_raw"])
        for case in cases
        for claim in case["result"].get("claims", [])
        for citation in claim.get("citations", [])
    ]
    return {
        "count": len(cases),
        "status_counts": dict(sorted(statuses.items())),
        "reason_code_counts": dict(sorted(reasons.items())),
        "measurement_coverage": {
            field: {"recorded": len(values), "missing": len(cases) - len(values)}
            for field, values in {
                "attempts": attempts,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_seconds": times,
            }.items()
        },
        "first_pass_mechanical_success": _ratio(first_pass, len(generated)),
        "generated_cases": len(generated),
        "repair_cases": repairs,
        "repair_frequency": _ratio(repairs, len(generated)),
        "final_invalid_model_output": reasons["invalid_model_output"],
        "other_technical_errors": sum(
            case["result"]["status"] == "error"
            and case["result"].get("reason_code") != "invalid_model_output"
            for case in cases
        ),
        "runner_error_cases": reasons["runner_error"],
        "generation_seconds_total": sum(times) if times else None,
        "generation_seconds_mean_per_case": statistics.mean(times) if times else None,
        "output_tokens_total": sum(output_tokens) if output_tokens else None,
        "output_tokens_mean_per_case": statistics.mean(output_tokens)
        if output_tokens
        else None,
        "final_attempt_input_tokens": {
            "minimum": min(generated_input_tokens, default=None),
            "maximum": max(generated_input_tokens, default=None),
            "recorded": len(generated_input_tokens),
            "missing": len(generated) - len(generated_input_tokens),
        },
        "resolved_quote_characters": {
            "count": len(quotes),
            "minimum": min(quotes, default=None),
            "maximum": max(quotes, default=None),
            "mean": statistics.mean(quotes) if quotes else None,
        },
    }


def content_metrics(cases: list[dict], reviews: dict[str, dict]) -> dict:
    original_answerable = [
        c
        for c in cases
        if c["dataset"] == "original" and c["category"] != "unsupported"
    ]
    if len(original_answerable) != 24:
        raise ReviewError("Expected 24 original answerable questions")
    answered = [c for c in original_answerable if c["result"]["status"] == "answered"]
    sufficient = [
        c for c in original_answerable if reviews[c["id"]]["context_sufficient"] is True
    ]

    def supported_correct(case: dict) -> bool:
        review = reviews[case["id"]]
        return (
            case["result"]["status"] == "answered"
            and review["answer_correct"] is True
            and review["all_claims_supported"] is True
        )

    scored = {
        "supported_correct_over_all": _ratio(
            sum(supported_correct(c) for c in original_answerable), 24
        ),
        "answer_accuracy_among_displayed": _ratio(
            sum(reviews[c["id"]]["answer_correct"] is True for c in answered),
            len(answered),
        ),
        "unsupported_claim_among_displayed": _ratio(
            sum(reviews[c["id"]]["all_claims_supported"] is False for c in answered),
            len(answered),
        ),
        "generation_success_given_sufficient_context": _ratio(
            sum(supported_correct(c) for c in sufficient), len(sufficient)
        ),
        "unjustified_abstention_given_sufficient_context": _ratio(
            sum(
                c["result"]["status"] == "abstained"
                and reviews[c["id"]]["abstention_justified"] is False
                for c in sufficient
            ),
            len(sufficient),
        ),
        "sufficient_context_count": len(sufficient),
        "null_or_not_applicable_ratings": {
            field: sum(reviews[c["id"]][field] is None for c in original_answerable)
            for field in BOOLEAN_FIELDS
        },
    }
    groups = {
        "original_unsupported": [
            c
            for c in cases
            if c["dataset"] == "original" and c["category"] == "unsupported"
        ],
        **{
            category: [
                c
                for c in cases
                if c["dataset"] == "challenge" and c["category"] == category
            ]
            for category in ("unmentioned_detail", "false_premise", "outside_corpus")
        },
    }
    unsupported = {}
    for name, group in groups.items():
        unsupported[name] = {
            "count": len(group),
            "strict_refusal": _ratio(
                sum(c["result"]["status"] == "abstained" for c in group), len(group)
            ),
            "invalid_model_output": sum(
                c["result"].get("reason_code") == "invalid_model_output" for c in group
            ),
            "other_technical_errors": sum(
                c["result"]["status"] == "error"
                and c["result"].get("reason_code") != "invalid_model_output"
                for c in group
            ),
            "displayed_unsupported_claim": sum(
                c["result"]["status"] == "answered"
                and reviews[c["id"]]["all_claims_supported"] is False
                for c in group
            ),
            "supported_premise_correction": sum(
                reviews[c["id"]]["supported_premise_correction"] is True for c in group
            ),
            "unresolved_content_reviews": sum(
                c["result"]["status"] == "answered"
                and reviews[c["id"]]["all_claims_supported"] is None
                for c in group
            ),
        }
    return {"original_answerable": scored, "unsupported_and_challenges": unsupported}


def compile_metrics(run_dir: Path) -> dict:
    result, result_sha = load_frozen_results(run_dir)
    reviews_bytes = (run_dir / "reviews.json").read_bytes()
    reviews = json.loads(reviews_bytes)
    by_id = validate_reviews(result, result_sha, reviews)
    cases = result["results"]
    groups = {
        "original_answerable": [
            c
            for c in cases
            if c["dataset"] == "original" and c["category"] != "unsupported"
        ],
        "original_unsupported": [
            c
            for c in cases
            if c["dataset"] == "original" and c["category"] == "unsupported"
        ],
        "challenge_unmentioned_detail": [
            c
            for c in cases
            if c["dataset"] == "challenge" and c["category"] == "unmentioned_detail"
        ],
        "challenge_false_premise": [
            c
            for c in cases
            if c["dataset"] == "challenge" and c["category"] == "false_premise"
        ],
        "challenge_outside_corpus": [
            c
            for c in cases
            if c["dataset"] == "challenge" and c["category"] == "outside_corpus"
        ],
    }
    output = {
        "run_id": result["run_id"],
        "results_sha256": result_sha,
        "reviews_sha256": sha256(reviews_bytes),
        "reviewer": reviews["reviewer"],
        "automatic": {
            name: automatic_metrics(group)
            for name, group in {"all": cases, **groups}.items()
        },
        "content_review": content_metrics(cases, by_id),
    }
    atomic_write(
        run_dir / "metrics.json",
        (
            json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        print("Invalid run_id", file=sys.stderr)
        return 1
    try:
        metrics = compile_metrics(PROJECT_ROOT / "eval/rag_runs" / args.run_id)
    except (ReviewError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Review metrics failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
