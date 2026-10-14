"""Offline checks for paired frozen-run review invariants and refusal metrics."""

import json
from copy import deepcopy

import pytest

from eval.paired_review import ROOT, PairedReviewError, compare, summarize


def test_identical_q06_claim_cannot_receive_different_support(tmp_path) -> None:
    source = ROOT / "eval/paired_reviews/answer_v1_v2_rubric_v1.json"
    paired = json.loads(source.read_text(encoding="utf-8"))
    by_run = {}
    for name, run in paired["runs"].items():
        by_run[name] = next(row for row in run["ratings"] if row["id"] == "q06")
    assert by_run["baseline_v1"]["claims"] == by_run["answer_v2_candidate"]["claims"]
    assert by_run["baseline_v1"]["claims"][0]["supported"] is True
    assert by_run["baseline_v1"]["claims"][0]["quote_sufficient"] is False
    assert compare(ROOT, source)["consistency"]["identical_claim_groups"] >= 1

    changed = deepcopy(paired)
    row = next(
        row
        for row in changed["runs"]["answer_v2_candidate"]["ratings"]
        if row["id"] == "q06"
    )
    row["claims"][0]["supported"] = False
    path = tmp_path / "different-ratings.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(PairedReviewError, match="Identical claim/evidence"):
        compare(ROOT, path)

    changed = deepcopy(paired)
    row = next(
        row
        for row in changed["runs"]["answer_v2_candidate"]["ratings"]
        if row["id"] == "q06"
    )
    row["answer_correct"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(PairedReviewError, match="Identical answer"):
        compare(ROOT, path)


def test_missed_cited_correction_is_separate_from_safe_refusal() -> None:
    case = {
        "dataset": "challenge",
        "category": "false_premise",
        "result": {"status": "abstained"},
    }
    review = {
        "claims": [],
        "answer_correct": None,
        "context_sufficient": None,
        "reason_adds_unsupported_facts": False,
        "refusal_appropriate": True,
        "supported_correction_possible": True,
        "formal_cited_correction": False,
    }
    metrics = summarize([case], [review])
    assert metrics["reason_safe"] == {"numerator": 1, "denominator": 1}
    assert metrics["refusal_appropriate"] == {"numerator": 1, "denominator": 1}
    assert metrics["correction_possible"] == {"numerator": 1, "denominator": 1}
    assert metrics["formal_cited_correction"] == {"numerator": 0, "denominator": 1}
