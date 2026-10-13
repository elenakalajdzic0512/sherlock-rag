"""Offline answer-evaluation checkpoint tests with controlled fake pipelines."""

from __future__ import annotations

import json

import pytest

from sherlock_rag import evaluate_rag
from sherlock_rag.evaluate_rag import RunError, build_fingerprint, run_cases
from sherlock_rag.review_rag import automatic_metrics

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


def _fingerprint_tree(tmp_path, monkeypatch) -> None:
    package = tmp_path / "src/sherlock_rag"
    package.mkdir(parents=True)
    for name in evaluate_rag.REQUIRED_RUNTIME_MODULES:
        (package / name).write_text(f"# {name}\n")
    for relative in (
        "config/retrieval.json",
        "config/generation.json",
        "data/source_manifest.json",
        "indexes/w250_o50.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    monkeypatch.setattr(
        evaluate_rag,
        "load_cases",
        lambda root: ([CASES[0]], {"questions": "q", "challenges": "c"}),
    )
    monkeypatch.setattr(
        evaluate_rag,
        "load_settings",
        lambda root: (
            {"configuration": "w250_o50", "method": "semantic", "top_k": 5},
            {
                "prompt_sha256": "prompt",
                "model_id": "model",
                "model_revision": "revision",
                "prompt_version": "v1",
                "preferred_device": "cpu",
                "dtype": "float32",
                "do_sample": False,
                "max_new_tokens": 20,
                "max_input_tokens": 100,
            },
            "prompt",
        ),
    )
    monkeypatch.setattr(
        evaluate_rag,
        "load_verified_source",
        lambda root: (
            "book",
            None,
            {"raw": {"sha256": "raw"}, "processed": {"sha256": "book"}},
        ),
    )
    monkeypatch.setattr(
        evaluate_rag,
        "load_index",
        lambda config, root: (
            [{"chunk_id": "one"}],
            None,
            {"chunks_sha256": "chunks", "vectors_sha256": "vectors"},
        ),
    )
    monkeypatch.setattr(evaluate_rag.metadata, "version", lambda name: "test-version")


def test_fingerprint_v2_tracks_runtime_source_and_excludes_cache(
    tmp_path, monkeypatch
) -> None:
    _fingerprint_tree(tmp_path, monkeypatch)
    original = build_fingerprint(tmp_path)
    assert original == build_fingerprint(tmp_path)
    assert original["fingerprint_schema_version"] == 2
    files = original["project_code_files_sha256"]
    assert list(files) == sorted(files)
    assert "src/sherlock_rag/semantic.py" in files
    cache = tmp_path / "src/sherlock_rag/__pycache__"
    cache.mkdir()
    (cache / "generated.py").write_text("# cache\n")
    generated = tmp_path / "src/sherlock_rag/generated"
    generated.mkdir()
    (generated / "artifact.py").write_text("# generated file\n")
    assert build_fingerprint(tmp_path) == original
    (tmp_path / "src/sherlock_rag/semantic.py").write_text("# changed retrieval\n")
    changed = build_fingerprint(tmp_path)
    assert changed != original
    assert (
        changed["project_code_files_sha256"]["src/sherlock_rag/semantic.py"]
        != files["src/sherlock_rag/semantic.py"]
    )


def test_changed_source_fingerprint_blocks_resume_before_model_or_write(
    tmp_path, monkeypatch
) -> None:
    _fingerprint_tree(tmp_path, monkeypatch)
    fingerprint = build_fingerprint(tmp_path)
    run_dir = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_cases(
            run_dir, "run", fingerprint, CASES, lambda: FakePipeline([], "second")
        )
    before = (run_dir / "results.json").read_bytes()
    (tmp_path / "src/sherlock_rag/semantic.py").write_text("# changed retrieval\n")
    changed = build_fingerprint(tmp_path)
    loaded = []
    with pytest.raises(RunError, match="fingerprint"):
        run_cases(run_dir, "run", changed, CASES, lambda: loaded.append("loaded"))
    assert loaded == []
    assert (run_dir / "results.json").read_bytes() == before


def test_runner_error_is_checkpointed_then_metrics_remain_available(tmp_path) -> None:
    calls = []

    class CrashingPipeline(FakePipeline):
        def answer(self, query: str) -> dict:
            calls.append(query)
            if query == "second":
                raise RuntimeError("controlled failure")
            return super().answer(query)

    result = run_cases(
        tmp_path / "run",
        "run",
        FINGERPRINT,
        CASES,
        lambda: CrashingPipeline([]),
    )
    assert calls == ["first", "second", "third"]
    failed = result["results"][1]["result"]
    assert failed == {
        "status": "error",
        "reason_code": "runner_error",
        "reason": "controlled failure",
    }
    assert (
        json.loads((tmp_path / "run/results.json").read_text())["results"][1]["result"]
        == failed
    )
    metrics = automatic_metrics(result["results"])
    assert metrics["status_counts"] == {"answered": 1, "error": 2}
    assert metrics["reason_code_counts"]["runner_error"] == 1
    assert metrics["other_technical_errors"] == 2
    assert metrics["runner_error_cases"] == 1
    assert metrics["first_pass_mechanical_success"]["numerator"] == 0
    assert metrics["measurement_coverage"]["attempts"] == {"recorded": 0, "missing": 3}
    assert metrics["measurement_coverage"]["generation_seconds"] == {
        "recorded": 0,
        "missing": 3,
    }
    assert metrics["generation_seconds_total"] is None
    assert metrics["output_tokens_total"] is None
