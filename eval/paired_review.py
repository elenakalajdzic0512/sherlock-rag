"""Validate and summarize a versioned paired human/agent content review offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("baseline_v1", "answer_v2_candidate")
FIELDS = ("supported", "quote_sufficient", "quote_precise", "atomic")


class PairedReviewError(ValueError):
    """The paired review disagrees with frozen results or itself."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator}


def load_and_validate(root: Path, paired_path: Path) -> tuple[dict, dict, dict]:
    paired = read_json(paired_path)
    if paired.get("rubric_version") != "paired-content-v1":
        raise PairedReviewError("Unsupported paired rubric version")
    rubric_path = root / paired["rubric_path"]
    if digest(rubric_path.read_bytes()) != paired.get("rubric_sha256"):
        raise PairedReviewError("Paired rubric hash differs")
    runs = {}
    for name in RUNS:
        run_dir = root / "eval/rag_runs" / name
        result_bytes = (run_dir / "results.json").read_bytes()
        if (
            digest(result_bytes) != paired["runs"][name]["results_sha256"]
            or digest(result_bytes) != (run_dir / "results.sha256").read_text().strip()
        ):
            raise PairedReviewError(f"Frozen results changed: {name}")
        if (
            digest((run_dir / "reviews.json").read_bytes())
            != paired["runs"][name]["historical_reviews_sha256"]
        ):
            raise PairedReviewError(f"Historical review changed: {name}")
        runs[name] = json.loads(result_bytes)["results"]
        if len(runs[name]) != 42:
            raise PairedReviewError(f"Expected 42 cases: {name}")
    baseline, candidate = (runs[name] for name in RUNS)
    for a, b in zip(baseline, candidate, strict=True):
        if a["id"] != b["id"] or a["query"] != b["query"]:
            raise PairedReviewError("Case order or questions differ")
        if a["result"].get("submitted_chunks") != b["result"].get("submitted_chunks"):
            raise PairedReviewError(f"Sent passages differ: {a['id']}")
    for a, b in zip(
        paired["runs"][RUNS[0]]["ratings"],
        paired["runs"][RUNS[1]]["ratings"],
        strict=True,
    ):
        if a["id"] != b["id"] or a["context_sufficient"] != b["context_sufficient"]:
            raise PairedReviewError(
                f"Identical sent context rated differently: {a['id']}"
            )
    seen = defaultdict(list)
    for name in RUNS:
        entries = paired["runs"][name]["ratings"]
        if len(entries) != 42:
            raise PairedReviewError(f"Expected 42 ratings: {name}")
        for case, rating in zip(runs[name], entries, strict=True):
            case_id = case["id"]
            if rating.get("id") != case_id or not rating.get("rationale"):
                raise PairedReviewError(
                    f"Missing ordered case rationale: {name}/{case_id}"
                )
            answer = case["result"]
            status = answer["status"]
            claims = answer.get("claims", []) if status == "answered" else []
            if len(rating.get("claims", [])) != len(claims):
                raise PairedReviewError(f"Claim rating count differs: {name}/{case_id}")
            if status == "answered":
                if type(rating.get("answer_correct")) is not bool:
                    raise PairedReviewError(
                        f"Answer correctness missing: {name}/{case_id}"
                    )
            elif rating.get("answer_correct") is not None:
                raise PairedReviewError(f"Nonanswer rated as correct: {name}/{case_id}")
            if status == "abstained":
                for field in ("reason_adds_unsupported_facts", "refusal_appropriate"):
                    if type(rating.get(field)) is not bool:
                        raise PairedReviewError(
                            f"Refusal rating missing: {name}/{case_id}"
                        )
            elif any(
                rating.get(field) is not None
                for field in ("reason_adds_unsupported_facts", "refusal_appropriate")
            ):
                raise PairedReviewError(
                    f"Nonrefusal has refusal rating: {name}/{case_id}"
                )
            if case["dataset"] == "original" and case["category"] != "unsupported":
                if type(rating.get("context_sufficient")) is not bool:
                    raise PairedReviewError(f"Context rating missing: {name}/{case_id}")
            elif rating.get("context_sufficient") is not None:
                raise PairedReviewError(
                    f"Nonanswerable context rated: {name}/{case_id}"
                )
            if case["category"] == "false_premise":
                for field in (
                    "supported_correction_possible",
                    "formal_cited_correction",
                ):
                    if type(rating.get(field)) is not bool:
                        raise PairedReviewError(
                            f"Premise rating missing: {name}/{case_id}"
                        )
                if rating["formal_cited_correction"] and status != "answered":
                    raise PairedReviewError(
                        f"Unanswered correction marked formal: {name}/{case_id}"
                    )
            elif any(
                rating.get(field) is not None
                for field in (
                    "supported_correction_possible",
                    "formal_cited_correction",
                )
            ):
                raise PairedReviewError(
                    f"Non-premise correction rated: {name}/{case_id}"
                )
            sources = {
                p["source_id"]: p["text"] for p in answer.get("submitted_chunks", [])
            }
            for claim, judgment in zip(claims, rating["claims"], strict=True):
                if any(type(judgment.get(field)) is not bool for field in FIELDS):
                    raise PairedReviewError(f"Claim rating missing: {name}/{case_id}")
                evidence = []
                for citation in claim["citations"]:
                    source_id = citation["source_id"]
                    quote = citation["quote"]
                    if source_id not in sources or quote not in sources[source_id]:
                        raise PairedReviewError(f"Citation differs: {name}/{case_id}")
                    evidence.append(
                        (source_id, citation["quote_raw"], quote, sources[source_id])
                    )
                key = (case_id, claim["text"], tuple(evidence))
                seen[key].append((name, {field: judgment[field] for field in FIELDS}))
    for (case_id, _, _), ratings in seen.items():
        if len(ratings) > 1 and any(value != ratings[0][1] for _, value in ratings[1:]):
            raise PairedReviewError(
                f"Identical claim/evidence rated differently: {case_id}"
            )
    for index, (a, b) in enumerate(zip(baseline, candidate, strict=True)):
        if (
            a["result"]["status"] == b["result"]["status"] == "answered"
            and a["result"]["claims"] == b["result"]["claims"]
            and paired["runs"][RUNS[0]]["ratings"][index]["answer_correct"]
            != paired["runs"][RUNS[1]]["ratings"][index]["answer_correct"]
        ):
            raise PairedReviewError(f"Identical answer rated differently: {a['id']}")
    return (
        paired,
        runs,
        {
            "identical_claim_groups": sum(len(v) > 1 for v in seen.values()),
            "identical_claim_disagreements": 0,
        },
    )


def summarize(cases: list[dict], ratings: list[dict]) -> dict:
    displayed = [
        (case, review)
        for case, review in zip(cases, ratings, strict=True)
        if case["result"]["status"] == "answered"
    ]
    refusals = [
        (case, review)
        for case, review in zip(cases, ratings, strict=True)
        if case["result"]["status"] == "abstained"
    ]
    answerable = [
        (case, review)
        for case, review in zip(cases, ratings, strict=True)
        if case["dataset"] == "original" and case["category"] != "unsupported"
    ]
    sufficient = [
        (case, review) for case, review in answerable if review["context_sufficient"]
    ]

    def supported_correct(case: dict, review: dict) -> bool:
        return (
            case["result"]["status"] == "answered"
            and review["answer_correct"]
            and all(claim["supported"] for claim in review["claims"])
        )

    all_claims = [claim for _, review in displayed for claim in review["claims"]]
    premise = [
        (case, review)
        for case, review in zip(cases, ratings, strict=True)
        if case["category"] == "false_premise"
    ]
    return {
        "cases": len(cases),
        "displayed": len(displayed),
        "refusals": len(refusals),
        "errors": sum(case["result"]["status"] == "error" for case in cases),
        "supported_correct_answerable": ratio(
            sum(supported_correct(*item) for item in answerable), len(answerable)
        ),
        "supported_correct_given_context": ratio(
            sum(supported_correct(*item) for item in sufficient), len(sufficient)
        ),
        "answer_correct_displayed": ratio(
            sum(review["answer_correct"] for _, review in displayed), len(displayed)
        ),
        "unsupported_displayed": ratio(
            sum(
                not all(claim["supported"] for claim in review["claims"])
                for _, review in displayed
            ),
            len(displayed),
        ),
        "quote_sufficient": ratio(
            sum(claim["quote_sufficient"] for claim in all_claims), len(all_claims)
        ),
        "quote_precise": ratio(
            sum(claim["quote_precise"] for claim in all_claims), len(all_claims)
        ),
        "reason_safe": ratio(
            sum(not review["reason_adds_unsupported_facts"] for _, review in refusals),
            len(refusals),
        ),
        "refusal_appropriate": ratio(
            sum(review["refusal_appropriate"] for _, review in refusals), len(refusals)
        ),
        "correction_possible": ratio(
            sum(review["supported_correction_possible"] for _, review in premise),
            len(premise),
        ),
        "formal_cited_correction": ratio(
            sum(review["formal_cited_correction"] for _, review in premise),
            len(premise),
        ),
    }


def compare(root: Path, paired_path: Path) -> dict:
    paired, runs, consistency = load_and_validate(root, paired_path)
    split = {
        row["question_id"]: row["split"]
        for row in map(
            json.loads, (root / "eval/questions.jsonl").read_text().splitlines()
        )
    }
    output = {
        "rubric_version": paired["rubric_version"],
        "paired_review_sha256": digest(paired_path.read_bytes()),
        "consistency": consistency,
        "runs": {},
    }
    for name in RUNS:
        cases = runs[name]
        ratings = paired["runs"][name]["ratings"]
        groups = {
            "all": list(range(len(cases))),
            "dev": [
                i
                for i, c in enumerate(cases)
                if c["dataset"] == "original" and split[c["id"]] == "dev"
            ],
            "test": [
                i
                for i, c in enumerate(cases)
                if c["dataset"] == "original" and split[c["id"]] == "test"
            ],
            **{
                category: [
                    i
                    for i, c in enumerate(cases)
                    if c["dataset"] == "challenge" and c["category"] == category
                ]
                for category in (
                    "unmentioned_detail",
                    "false_premise",
                    "outside_corpus",
                )
            },
        }
        output["runs"][name] = {
            group: summarize([cases[i] for i in indices], [ratings[i] for i in indices])
            for group, indices in groups.items()
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--paired-review", type=Path)
    args = parser.parse_args()
    path = (
        args.paired_review
        or args.root / "eval/paired_reviews/answer_v1_v2_rubric_v1.json"
    )
    print(
        json.dumps(
            compare(args.root, path), ensure_ascii=False, sort_keys=True, indent=2
        )
    )


if __name__ == "__main__":
    main()
