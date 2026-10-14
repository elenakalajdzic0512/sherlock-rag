"""Offline selection, citation, validation, duplicate, and checkpoint tests."""

from __future__ import annotations

import json
import shutil

import pytest

from sherlock_rag import question_generation as qg
from sherlock_rag.ingest import PROJECT_ROOT


def synthetic_chunks(count: int = 5) -> list[dict]:
    return [
        {
            "story_index": story,
            "story_id": f"s{story}",
            "story_title": f"Story {story}",
            "chunk_index": index,
            "chunk_id": f"c{story}-{index}",
        }
        for story in range(1, 13)
        for index in range(count)
    ]


def test_fraction_selection_is_story_ordered_and_independent_of_quality() -> None:
    picked = qg.select_chunks(synthetic_chunks())
    assert len(picked) == 36
    assert [item["selected_position_zero_based"] for item in picked[:3]] == [1, 2, 3]
    assert [item["chunk_id"] for item in picked[3:6]] == ["c2-1", "c2-2", "c2-3"]
    assert picked[-1]["id"] == "qg-v1-12-3"


BOOK = "Holmes  found\na clue."
CHUNK = {
    "chunk_id": "c1",
    "story_id": "s1",
    "story_title": "Story",
    "char_start": 0,
    "char_end": len(BOOK),
    "text": BOOK,
    "source_line_start": 5,
    "source_line_end": 6,
}


def candidate(**changes) -> str:
    data = {
        "status": "candidate",
        "question": "What did Holmes find?",
        "expected_answer": "A clue.",
        "source_id": "C1",
        "quote": "Holmes found a clue.",
        "reason": "",
    }
    data.update(changes)
    return json.dumps(data)


def test_quote_maps_display_to_original_offsets_and_lines() -> None:
    result = qg.validate_output(candidate(), CHUNK, BOOK, 5)
    citation = result["citation"]
    assert citation["quote"] == "Holmes found a clue."
    assert citation["quote_raw"] == BOOK
    assert (citation["char_start"], citation["char_end"]) == (0, len(BOOK))
    assert (citation["source_line_start"], citation["source_line_end"]) == (5, 6)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        candidate(source_id="C2"),
        candidate(quote="holmes found a clue."),
        candidate(reason="extra"),
        candidate(question=""),
        candidate(status="unknown"),
        candidate(extra="field"),
    ],
)
def test_invalid_candidate_is_rejected_without_silent_repair(raw) -> None:
    with pytest.raises(qg.QuestionGenerationError):
        qg.validate_output(raw, CHUNK, BOOK, 5)


def test_insufficient_evidence_contract() -> None:
    raw = candidate(
        status="insufficient_evidence",
        question="",
        expected_answer="",
        source_id="",
        quote="",
        reason="No concrete fact.",
    )
    assert qg.validate_output(raw, CHUNK, BOOK, 5) == {
        "status": "insufficient_evidence",
        "reason": "No concrete fact.",
    }
    with pytest.raises(qg.QuestionGenerationError):
        qg.validate_output(
            candidate(status="insufficient_evidence", reason="No fact."), CHUNK, BOOK, 5
        )


class FakeGenerator:
    device = "cpu"
    dtype = "float32"
    load_seconds = 0.1

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def count_tokens(self, messages):
        return sum(len(message["content"]) for message in messages)

    def generate(self, messages):
        output = self.outputs[self.calls]
        self.calls += 1
        return output, self.count_tokens(messages), 8, 0.2


def test_repair_raw_attempts_and_normalized_duplicates() -> None:
    first = FakeGenerator(["bad", candidate()])
    prior = {}
    selected = {"id": "qg-v1-01-1", "chunk_id": "c1"}
    fingerprint = {
        "processed_book_sha256": "source",
        "prompt_sha256": "prompt",
        "config_sha256": "config",
    }
    output = qg.generate_case(
        CHUNK,
        selected,
        first,
        "prompt",
        {"max_input_tokens": 6144},
        BOOK,
        5,
        fingerprint,
        prior,
    )
    assert output["status"] == "candidate" and output["attempts"] == 2
    assert output["raw_attempts"][0]["validation_error"] == "incomplete or invalid JSON"
    assert output["duplicate_of"] is None
    second = FakeGenerator([candidate(question="  what DID holmes find?  ")])
    selected2 = {"id": "qg-v1-01-2", "chunk_id": "c1"}
    duplicate = qg.generate_case(
        CHUNK,
        selected2,
        second,
        "prompt",
        {"max_input_tokens": 6144},
        BOOK,
        5,
        fingerprint,
        prior,
    )
    assert duplicate["duplicate_of"] == selected["id"]


def test_full_prompt_budget_prevents_generation() -> None:
    generator = FakeGenerator([candidate()])
    fingerprint = {
        "processed_book_sha256": "source",
        "prompt_sha256": "prompt",
        "config_sha256": "config",
    }
    result = qg.generate_case(
        CHUNK,
        {"id": "one", "chunk_id": "c1"},
        generator,
        "prompt",
        {"max_input_tokens": 1},
        BOOK,
        5,
        fingerprint,
        {},
    )
    assert result["reason_code"] == "context_budget_exceeded"
    assert result["attempts"] == 0 and generator.calls == 0


def test_resume_rejects_changed_fingerprint_before_model_load_or_write(
    tmp_path, monkeypatch
) -> None:
    selected = [{"id": f"item-{i}", "chunk_id": f"chunk-{i}"} for i in range(36)]
    chunks = [{"chunk_id": f"chunk-{i}"} for i in range(36)]
    fingerprint = {"value": "old"}
    monkeypatch.setattr(qg, "prepare", lambda root: {"selected": selected})
    monkeypatch.setattr(qg, "build_fingerprint", lambda root: fingerprint.copy())
    monkeypatch.setattr(qg, "load_config", lambda root: ({}, "prompt"))
    monkeypatch.setattr(qg, "load_verified_chunks", lambda config, root: chunks)
    monkeypatch.setattr(
        qg,
        "load_verified_source",
        lambda root: (
            "book",
            None,
            {"processed": {"source_lines_inclusive": {"start": 1}}},
        ),
    )
    calls = []

    def fake_generate(chunk, selected_item, *args):
        if len(calls) == 1:
            raise KeyboardInterrupt("pause")
        calls.append(selected_item["id"])
        return {
            "id": selected_item["id"],
            "chunk_id": selected_item["chunk_id"],
            "status": "candidate",
            "reason_code": "candidate",
            "question": "What?",
        }

    monkeypatch.setattr(qg, "generate_case", fake_generate)
    with pytest.raises(KeyboardInterrupt):
        qg.run(tmp_path, lambda config: FakeGenerator([]))
    saved = (tmp_path / qg.RUN_DIR / "results.json").read_bytes()
    fingerprint["value"] = "changed"
    with pytest.raises(qg.QuestionGenerationError, match="fingerprint"):
        qg.run(
            tmp_path,
            lambda config: (_ for _ in ()).throw(AssertionError("model loaded")),
        )
    assert (tmp_path / qg.RUN_DIR / "results.json").read_bytes() == saved


def config_files(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "prompts").mkdir()
    for name in ("generation.json", "question_generation_v1.json"):
        shutil.copyfile(PROJECT_ROOT / "config" / name, tmp_path / "config" / name)
    shutil.copyfile(
        PROJECT_ROOT / "prompts/question_generation_v1.txt",
        tmp_path / "prompts/question_generation_v1.txt",
    )
    return tmp_path / "config/question_generation_v1.json"


@pytest.mark.parametrize("field", ["max_input_tokens", "max_new_tokens"])
@pytest.mark.parametrize("value", [True, "6144", 6144.0, None, 0, -1])
def test_config_rejects_nonpositive_or_noninteger_budgets_before_model(
    tmp_path, monkeypatch, field, value
) -> None:
    config_path = config_files(tmp_path)
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(
        qg.QuestionGenerationError, match=f"{field} must be a positive integer"
    ):
        qg.load_config(tmp_path)
    monkeypatch.setattr(qg, "prepare", lambda root: {"selected": []})
    monkeypatch.setattr(qg, "build_fingerprint", lambda root: {})
    with pytest.raises(
        qg.QuestionGenerationError, match=f"{field} must be a positive integer"
    ):
        qg.run(
            tmp_path,
            lambda config: (_ for _ in ()).throw(AssertionError("model loaded")),
        )


def test_config_accepts_exact_valid_budgets(tmp_path) -> None:
    config_files(tmp_path)
    config, prompt = qg.load_config(tmp_path)
    assert config["max_input_tokens"] == 6144
    assert config["max_new_tokens"] == 512
    assert prompt


def test_frozen_historical_run_is_read_only_after_runtime_code_change(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / qg.RUN_DIR
    run_dir.mkdir(parents=True)
    source = PROJECT_ROOT / qg.RUN_DIR
    for name in ("results.json", "results.sha256"):
        shutil.copyfile(source / name, run_dir / name)
    raw_before = (run_dir / "results.json").read_bytes()
    hash_before = (run_dir / "results.sha256").read_bytes()
    saved = json.loads(raw_before)
    selection = json.loads((source / "selection.json").read_text())
    current = saved["fingerprint"].copy()
    current["runtime_code_sha256"] = {"changed.py": "new"}
    monkeypatch.setattr(qg, "prepare", lambda root: selection)
    monkeypatch.setattr(qg, "build_fingerprint", lambda root: current)
    monkeypatch.setattr(qg, "load_config", lambda root: ({}, "prompt"))
    output = qg.run(
        tmp_path,
        lambda config: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    assert output["fingerprint"] == saved["fingerprint"]
    assert (run_dir / "results.json").read_bytes() == raw_before
    assert (run_dir / "results.sha256").read_bytes() == hash_before
