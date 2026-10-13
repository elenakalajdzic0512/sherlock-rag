"""Offline semantic ranking, token checks, and artifact integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from test_chunking import fixture_root
from test_retrieval import chunk

from sherlock_rag.chunking import run_chunking
from sherlock_rag.semantic import (
    DIMENSION,
    QUERY_PREFIX,
    QwenEncoder,
    SemanticError,
    SemanticIndex,
    build_index,
    check_token_lengths,
    index_paths,
    load_index,
    normalize_vectors,
)


def vector(x: float, y: float = 0) -> np.ndarray:
    row = np.zeros(DIMENSION, dtype=np.float32)
    row[:2] = (x, y)
    return row


class FakeTokenizer:
    def __init__(self) -> None:
        self.inputs = []

    def encode(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> list[int]:
        self.inputs.append((text, add_special_tokens, truncation))
        return [1] * (len(text.split()) + 2)


class FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.stack([vector(1) for _ in texts])


class FakeEncoder:
    device = "cpu"

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def measure_documents(self, texts: list[str]) -> int:
        return check_token_lengths(self.tokenizer, texts, 4096, "document")

    def measure_queries(self, queries: list[str]) -> int:
        return check_token_lengths(
            self.tokenizer, [QUERY_PREFIX + q for q in queries], 4096, "query"
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([vector(1) for _ in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return vector(1)


def test_normalization_and_known_cosine_order() -> None:
    chunks = [chunk("first", 2, 2), chunk("second", 1, 1), chunk("third", 3, 1)]
    rows = np.stack([vector(0, 5), vector(2, 0), vector(1, 0)])
    normalized = normalize_vectors(rows, expected_rows=3)
    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized, axis=1) == pytest.approx([1, 1, 1])
    results = SemanticIndex(chunks, normalized, FakeEncoder()).search("question", 3)
    assert [r["chunk_id"] for r in results] == ["id-1", "id-3", "id-2"]
    assert [r["score"] for r in results] == pytest.approx([1, 1, 0])
    assert results[0]["source_line_start"] == chunks[1]["source_line_start"]


def test_empty_query_and_invalid_top_k() -> None:
    index = SemanticIndex([chunk("x", 1)], np.stack([vector(1)]), FakeEncoder())
    assert index.search(" \n", 5) == []
    for value in (0, -1, 1.5, True, None):
        with pytest.raises(SemanticError):
            index.search("q", value)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1, 2)),
        np.zeros((1, DIMENSION)),
        np.full((1, DIMENSION), np.nan),
        np.full((1, DIMENSION), np.inf),
    ],
)
def test_bad_vector_shape_finiteness_or_norm(bad) -> None:
    with pytest.raises(SemanticError):
        normalize_vectors(bad, expected_rows=1)


def test_query_prefix_only_and_no_truncation() -> None:
    encoder = object.__new__(QwenEncoder)
    encoder.model = FakeModel()
    encoder.tokenizer = FakeTokenizer()
    encoder._query_cache = {}
    encoder.encode_query("Who was Irene Adler?")
    assert encoder.model.calls[0][0] == ["Who was Irene Adler?"]
    assert encoder.model.calls[0][1]["prompt"] == QUERY_PREFIX
    assert encoder.tokenizer.inputs[0] == (
        QUERY_PREFIX + "Who was Irene Adler?",
        True,
        False,
    )
    encoder.encode_documents(["Exact chunk text"])
    assert encoder.model.calls[1][1]["prompt"] == ""
    assert encoder.model.calls[1][0] == ["Exact chunk text"]
    assert encoder.tokenizer.inputs == [
        (QUERY_PREFIX + "Who was Irene Adler?", True, False)
    ]


def test_token_limit_rejected_without_truncation() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(SemanticError, match="exceeding 3"):
        check_token_lengths(tokenizer, ["one two three"], 3, "document")
    assert tokenizer.inputs == [("one two three", True, False)]


def synthetic_index(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    fixture_root(tmp_path)
    run_chunking(250, 50, tmp_path)
    other = run_chunking(500, 100, tmp_path)
    (tmp_path / other["output_path"]).unlink()
    monkeypatch.setattr("sherlock_rag.semantic.package_versions", lambda: {"fake": "1"})
    build_index("w250_o50", FakeEncoder(), tmp_path)
    return index_paths("w250_o50", tmp_path)


def test_saved_index_and_selected_config_without_other(
    tmp_path: Path, monkeypatch
) -> None:
    vector_path, meta_path = synthetic_index(tmp_path, monkeypatch)
    chunks, vectors, meta = load_index("w250_o50", tmp_path)
    assert len(chunks) == vectors.shape[0] == meta["shape"][0]
    assert vector_path.exists() and meta_path.exists()
    # The manifest retains B, but its local chunk JSONL and index are absent.
    assert (
        "w500_o100"
        in json.loads((tmp_path / "data/chunk_manifest.json").read_text())[
            "configurations"
        ]
    )
    assert not (tmp_path / "data/processed/chunks_w500_o100.jsonl").exists()


def test_saved_index_hash_order_and_staleness(tmp_path: Path, monkeypatch) -> None:
    vector_path, meta_path = synthetic_index(tmp_path, monkeypatch)
    original = vector_path.read_bytes()
    vector_path.write_bytes(original + b"x")
    with pytest.raises(SemanticError, match="hash differs"):
        load_index("w250_o50", tmp_path)
    vector_path.write_bytes(original)
    meta = json.loads(meta_path.read_text())
    meta["ordered_chunk_ids"].reverse()
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SemanticError, match="ordered_chunk_ids"):
        load_index("w250_o50", tmp_path)
    meta["ordered_chunk_ids"].reverse()
    meta["model_revision"] = "stale"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SemanticError, match="model_revision"):
        load_index("w250_o50", tmp_path)


def test_search_cli_uses_saved_index_and_controlled_encoder(
    monkeypatch, capsys
) -> None:
    from sherlock_rag.semantic import main

    chunks = [chunk("evidence text", 1)]
    vectors = np.stack([vector(1)])
    monkeypatch.setattr(
        "sherlock_rag.semantic.load_index",
        lambda config: (chunks, vectors, {"model_revision": "test-revision"}),
    )
    monkeypatch.setattr(
        "sherlock_rag.semantic.load_encoder", lambda: (FakeEncoder(), None)
    )
    assert (
        main(["search", "--config", "w250_o50", "--query", "question", "--top-k", "1"])
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["results"][0]["chunk_id"] == "id-1"
    assert output["model_revision"] == "test-revision"
