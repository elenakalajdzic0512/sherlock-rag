"""Offline answer-evaluation checkpoint tests with controlled fake pipelines."""

from __future__ import annotations

import json

import pytest

from sherlock_rag.evaluate_rag import RunError, run_cases

CASES = [
    {"id": "q1", "dataset": "original", "category": "direct", "query": "first"},
    {"id": "q2", "dataset": "original", "category": "unsupported", "query": "second"},
    {"id": "c1", "dataset": "challenge", "category": "false_premise", "query": "third"},
]
FINGERPRINT = {"device": "cpu", "dtype": "float32", "hash": "same"}


class FakePipeline:
    def __init__(self, calls: list[str], fail_at: str | None = None):
        self.calls = calls
        self.fail_at = fail_at
        self.generator = type(
            "Generator", (), {"device": "cpu", "dtype": "float32", "load_seconds": 0.1}
        )()

    def answer(self, query: str) -> dict:
        if query == self.fail_at:
            raise KeyboardInterrupt("controlled interruption")
        self.calls.append(query)
        return {
            "status": {"first": "answered", "second": "abstained", "third": "error"}[
                query
            ],
            "reason_code": query,
        }


def test_query_only_and_all_statuses_checkpointed(tmp_path) -> None:
    calls: list[str] = []
    result = run_cases(
        tmp_path / "run", "run", FINGERPRINT, CASES, lambda: FakePipeline(calls)
    )
    assert calls == ["first", "second", "third"]
    assert [x["result"]["status"] for x in result["results"]] == [
        "answered",
        "abstained",
        "error",
    ]
    assert result["state"] == "complete"
    assert (tmp_path / "run/results.sha256").exists()
    assert (
        json.loads((tmp_path / "run/results.json").read_text())["results"]
        == result["results"]
    )


def test_resume_skips_saved_answers_and_rejects_fingerprint_change(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_cases(
            path, "run", FINGERPRINT, CASES, lambda: FakePipeline(calls, "second")
        )
    assert calls == ["first"]
    saved = json.loads((path / "results.json").read_text())
    assert len(saved["results"]) == 1
    with pytest.raises(RunError, match="fingerprint"):
        run_cases(
            path,
            "run",
            {**FINGERPRINT, "hash": "changed"},
            CASES,
            lambda: FakePipeline(calls),
        )
    assert calls == ["first"]
    completed = run_cases(path, "run", FINGERPRINT, CASES, lambda: FakePipeline(calls))
    assert calls == ["first", "second", "third"]
    assert len(completed["results"]) == 3
    again = run_cases(
        path,
        "run",
        FINGERPRINT,
        CASES,
        lambda: (_ for _ in ()).throw(AssertionError("reload")),
    )
    assert again == completed


def test_resume_rejects_changed_saved_case_or_frozen_hash(tmp_path) -> None:
    path = tmp_path / "run"
    run_cases(path, "run", FINGERPRINT, CASES, lambda: FakePipeline([]))
    changed = [{**CASES[0], "query": "different"}, *CASES[1:]]
    with pytest.raises(RunError, match="prefix"):
        run_cases(path, "run", FINGERPRINT, changed, lambda: FakePipeline([]))
    (path / "results.json").write_text((path / "results.json").read_text() + " ")
    with pytest.raises(RunError, match="hash differs"):
        run_cases(path, "run", FINGERPRINT, CASES, lambda: FakePipeline([]))
