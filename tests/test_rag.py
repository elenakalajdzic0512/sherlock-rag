"""Offline citation, budget, and repair tests; no real model or network."""

from __future__ import annotations

import copy
import json

import pytest

from sherlock_rag.ingest import sha256
from sherlock_rag.rag import (
    RagPipeline,
    ResponseError,
    load_settings,
    make_passages,
    normalize_with_spans,
    resolve_response,
)

BOOK = "Holmes’s  curious\nclue appeared. Holmes’s  curious\nclue appeared."


def chunk(start: int = 0, end: int = len(BOOK), number: int = 1) -> dict:
    return {
        "chunk_id": f"id-{number}",
        "story_id": "story-1",
        "story_title": "Test story",
        "char_start": start,
        "char_end": end,
        "text": BOOK[start:end],
        "source_line_start": 10,
        "source_line_end": 11,
    }


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.queries = []

    def search(self, query: str, top_k: int) -> list[dict]:
        self.queries.append(query)
        return self.chunks[:top_k]


class Generator:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = []

    def count_tokens(self, messages: list[dict]) -> int:
        # A deterministic fake tokenizer over the full system + user chat prompt.
        return sum(len(message["content"]) for message in messages)

    def generate(self, messages: list[dict]) -> tuple[str, int, int, float]:
        self.calls.append(copy.deepcopy(messages))
        return self.outputs[len(self.calls) - 1], self.count_tokens(messages), 20, 0.1


def answered(quote: str = "Holmes’s curious clue", source: str = "C1") -> str:
    return json.dumps(
        {
            "status": "answered",
            "claims": [
                {
                    "text": "A clue appeared.",
                    "citations": [{"source_id": source, "quote": quote}],
                }
            ],
            "reason": "",
        }
    )


def pipeline(outputs: list[str], chunks: list[dict] | None = None, limit: int = 6144):
    r = Retriever([chunk()] if chunks is None else chunks)
    g = Generator(outputs)
    settings = {"method": "semantic", "configuration": "w250_o50", "top_k": 5}
    model = {
        "model_id": "fake",
        "model_revision": "rev",
        "prompt_version": "v1",
        "prompt_sha256": "hash",
        "max_input_tokens": limit,
        "max_new_tokens": 768,
    }
    return RagPipeline(r, g, BOOK, 10, settings, model, "Use evidence."), r, g


def test_valid_answer_resolves_unicode_and_whitespace_to_raw_book() -> None:
    p, r, g = pipeline([answered()])
    out = p.answer("What appeared?")
    assert out["status"] == "answered" and out["citations_resolved"] is True
    assert out["semantic_support_verified"] is None
    cite = out["claims"][0]["citations"][0]
    assert cite["quote_raw"] == "Holmes’s  curious\nclue"
    assert (cite["char_start"], cite["char_end"]) == (0, len(cite["quote_raw"]))
    assert cite["source_line_start"] == 10 and cite["source_line_end"] == 11
    assert out["answer"] == "A clue appeared. [C1]"
    assert r.queries == ["What appeared?"] and len(g.calls) == 1


def test_first_of_multiple_occurrences_is_used() -> None:
    resolved = resolve_response(answered(), make_passages([chunk()], BOOK), BOOK, 10)
    assert resolved["claims"][0]["citations"][0]["char_start"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":',
        "not json",
        json.dumps(
            {
                "status": "answered",
                "claims": [{"text": "x", "citations": []}],
                "reason": "",
            }
        ),
        answered(source="C9"),
        answered(quote="imaginary quote"),
        json.dumps(
            {
                "status": "abstained",
                "claims": [{"text": "x", "citations": []}],
                "reason": "no evidence",
            }
        ),
    ],
)
def test_invalid_output_rejected(raw: str) -> None:
    with pytest.raises(ResponseError):
        resolve_response(raw, make_passages([chunk()], BOOK), BOOK, 10)


def test_valid_abstention_and_empty_retrieval_without_model_call() -> None:
    abstained = json.dumps(
        {"status": "abstained", "claims": [], "reason": "No relevant evidence."}
    )
    p, _, _ = pipeline([abstained])
    assert p.answer("unknown")["status"] == "abstained"
    p, _, g = pipeline([], [])
    out = p.answer("unknown")
    assert (
        out["status"] == "abstained" and out["reason_code"] == "insufficient_evidence"
    )
    assert out["attempts"] == 0 and not g.calls


def test_one_repair_only_and_failure_is_error_not_abstention() -> None:
    p, _, g = pipeline(["bad", answered()])
    out = p.answer("question")
    assert out["status"] == "answered" and out["attempts"] == 2
    assert "mechanical validation" in g.calls[1][-1]["content"]
    p, _, g = pipeline(["bad", "still bad"])
    out = p.answer("question")
    assert out["status"] == "error" and out["reason_code"] == "invalid_model_output"
    assert out["claims"] == [] and out["answer"] == "" and len(g.calls) == 2


def test_budget_counts_complete_prompt_and_drops_whole_last_chunk() -> None:
    first = chunk(0, 32, 1)
    second = chunk(33, len(BOOK), 2)
    p, _, g = pipeline([answered()], [first, second], limit=180)
    out = p.answer("question")
    assert out["dropped_chunk_ids"] == ["id-2"]
    assert [c["chunk_id"] for c in out["submitted_chunks"]] == ["id-1"]
    assert len(g.calls) == 1
    p, _, g = pipeline([], [first], limit=20)
    out = p.answer("question")
    assert out["reason_code"] == "invalid_input" and out["status"] == "error"
    assert not g.calls


def test_invalid_input_and_model_error_are_distinct() -> None:
    p, r, _ = pipeline([])
    assert p.answer(" ")["reason_code"] == "invalid_input"
    assert not r.queries
    p, _, g = pipeline([])

    def broken(messages):
        raise RuntimeError("MPS unavailable")

    g.generate = broken
    out = p.answer("question")
    assert out["status"] == "error" and out["reason_code"] == "resource_or_model_error"


def test_normalization_preserves_non_whitespace_unicode() -> None:
    normalized, spans = normalize_with_spans("é ’  x\ny")
    assert normalized == "é ’ x y"
    assert spans[3] == (3, 5)


def test_load_settings_reads_selected_retrieval_without_mutation(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "prompts").mkdir()
    prompt = b"Only evidence."
    (tmp_path / "prompts/p.txt").write_bytes(prompt)
    (tmp_path / "config/retrieval.json").write_text(
        json.dumps({"method": "semantic", "configuration": "w250_o50", "top_k": 5})
    )
    (tmp_path / "config/generation.json").write_text(
        json.dumps(
            {
                "prompt_file": "prompts/p.txt",
                "prompt_sha256": sha256(prompt),
                "max_input_tokens": 6144,
                "max_new_tokens": 768,
            }
        )
    )
    retrieval, _, text = load_settings(tmp_path)
    assert retrieval == {"method": "semantic", "configuration": "w250_o50", "top_k": 5}
    assert text == "Only evidence."
