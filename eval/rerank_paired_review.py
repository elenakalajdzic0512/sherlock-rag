"""Check frozen rerank-vs-baseline ratings under paired content rubric v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.paired_review import digest, read_json, summarize

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PAIRED = "eval/paired_reviews/answer_v1_v2_rubric_v1.json"
CANDIDATE_REVIEW = "eval/paired_reviews/rerank_v1_rubric_v1.json"


class RerankReviewError(ValueError):
    """The separate review does not match frozen runs or the common rubric."""


def compare(root: Path = ROOT, review_path: Path | None = None) -> dict:
    review_path = review_path or root / CANDIDATE_REVIEW
    review = read_json(review_path)
    old_path = root / BASELINE_PAIRED
    if review["baseline_paired_sha256"] != digest(old_path.read_bytes()):
        raise RerankReviewError("Baseline paired ratings changed")
    old = read_json(old_path)
    if (
        review["rubric_version"] != old["rubric_version"]
        or review["rubric_sha256"] != old["rubric_sha256"]
    ):
        raise RerankReviewError("Common paired rubric changed")
    runs = {}
    for name in ("baseline_v1", "rerank_v1_candidate"):
        path = root / "eval/rag_runs" / name
        data = (path / "results.json").read_bytes()
        if digest(data) != (path / "results.sha256").read_text().strip():
            raise RerankReviewError(f"Frozen results changed: {name}")
        if digest(data) != review["results_sha256"][name]:
            raise RerankReviewError(f"Review result hash differs: {name}")
        runs[name] = json.loads(data)["results"]
    baseline = old["runs"]["baseline_v1"]["ratings"]
    candidate = review["ratings"]
    if len(baseline) != len(candidate) or len(candidate) != 42:
        raise RerankReviewError("Review must cover all 42 cases")
    matching_claims = 0
    changed_contexts = 0
    for a, b, x, y in zip(
        runs["baseline_v1"],
        runs["rerank_v1_candidate"],
        baseline,
        candidate,
        strict=True,
    ):
        if a["id"] != b["id"] or x["id"] != y["id"] or a["query"] != b["query"]:
            raise RerankReviewError("Case IDs, order, or queries differ")
        ar, br = a["result"], b["result"]
        if ar.get("submitted_chunks") != br.get("submitted_chunks"):
            changed_contexts += 1
        if not y.get("rationale") or len(y.get("claims", [])) != len(
            br.get("claims", [])
        ):
            raise RerankReviewError(f"Missing rationale or claim ratings: {b['id']}")
        if br["status"] == "answered" and type(y.get("answer_correct")) is not bool:
            raise RerankReviewError(f"Answer rating missing: {b['id']}")
        if br["status"] == "abstained" and any(
            type(y.get(f)) is not bool
            for f in ("reason_adds_unsupported_facts", "refusal_appropriate")
        ):
            raise RerankReviewError(f"Refusal rating missing: {b['id']}")
        if (
            b["dataset"] == "original"
            and b["category"] != "unsupported"
            and type(y.get("context_sufficient")) is not bool
        ):
            raise RerankReviewError(f"Context rating missing: {b['id']}")
        if b["category"] == "false_premise" and any(
            type(y.get(f)) is not bool
            for f in ("supported_correction_possible", "formal_cited_correction")
        ):
            raise RerankReviewError(f"Premise rating missing: {b['id']}")
        sources = {p["source_id"]: p["text"] for p in br.get("submitted_chunks", [])}
        for claim, rating in zip(br.get("claims", []), y["claims"], strict=True):
            if any(
                type(rating.get(f)) is not bool
                for f in ("supported", "quote_sufficient", "quote_precise", "atomic")
            ):
                raise RerankReviewError(f"Claim rating missing: {b['id']}")
            if any(
                c["source_id"] not in sources
                or c["quote"] not in sources[c["source_id"]]
                for c in claim["citations"]
            ):
                raise RerankReviewError(f"Citation differs from sent text: {b['id']}")
        if ar["status"] == br["status"] == "answered":
            for ca, cb, ra, rb in zip(
                ar["claims"], br["claims"], x["claims"], y["claims"]
            ):
                context_a = {p["source_id"]: p["text"] for p in ar["submitted_chunks"]}
                evidence_a = [
                    (c["quote"], context_a[c["source_id"]]) for c in ca["citations"]
                ]
                evidence_b = [
                    (c["quote"], sources[c["source_id"]]) for c in cb["citations"]
                ]
                if ca["text"] == cb["text"] and evidence_a == evidence_b:
                    matching_claims += 1
                    if ra != rb:
                        raise RerankReviewError(
                            f"Identical claim/evidence rated differently: {b['id']}"
                        )
    split = {
        q["question_id"]: q["split"]
        for q in map(
            json.loads, (root / "eval/questions.jsonl").read_text().splitlines()
        )
    }
    metrics = {}
    for name, ratings in (
        ("baseline_v1", baseline),
        ("rerank_v1_candidate", candidate),
    ):
        cases = runs[name]
        groups = {
            "all": list(range(42)),
            **{
                part: [
                    i
                    for i, c in enumerate(cases)
                    if c["dataset"] == "original" and split[c["id"]] == part
                ]
                for part in ("dev", "test")
            },
            **{
                cat: [
                    i
                    for i, c in enumerate(cases)
                    if c["category"] == cat and c["dataset"] == "challenge"
                ]
                for cat in ("unmentioned_detail", "false_premise", "outside_corpus")
            },
        }
        metrics[name] = {
            group: summarize([cases[i] for i in ids], [ratings[i] for i in ids])
            for group, ids in groups.items()
        }
    return {
        "rubric_version": review["rubric_version"],
        "review_sha256": digest(review_path.read_bytes()),
        "changed_contexts": changed_contexts,
        "identical_claim_evidence_groups": matching_claims,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(compare(args.root), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
