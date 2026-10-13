"""Offline BM25 and verified-loading tests using synthetic documents and corpus."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from test_chunking import fixture_root

from sherlock_rag.chunking import run_chunking
from sherlock_rag.ingest import sha256
from sherlock_rag.retrieval import (
    BM25Index,
    RetrievalError,
    load_verified_chunks,
    tokenize,
)


def chunk(text: str, index: int, story: int = 1) -> dict:
    return {
        "text": text,
        "chunk_id": f"id-{index}",
        "story_id": f"story-{story}",
        "story_title": f"Story {story}",
        "story_index": story,
        "chunk_index": index,
        "char_start": index * 10,
        "char_end": index * 10 + len(text),
        "processed_line_start": index,
        "processed_line_end": index,
        "source_line_start": index + 5,
        "source_line_end": index + 5,
        "word_count": len(text.split()),
    }


def test_unicode_tokenization_and_same_rules_for_query_and_document() -> None:
    sample = "ÉCOLE Holmes’s Holmes'car _foo_bar 42—red-headed café!"
    assert tokenize(sample) == [
        "école",
        "holmes's",
        "holmes'car",
        "foo",
        "bar",
        "42",
        "red",
        "headed",
        "café",
    ]
    index = BM25Index([chunk("Holmes’s clue", 1)])
    assert index.search("HOLMES'S", 1)[0]["matched_query_terms"] == ["holmes's"]


def test_document_frequency_and_hand_checked_numeric_score() -> None:
    index = BM25Index(
        [
            chunk("apple apple banana", 1),
            chunk("banana carrot", 2),
            chunk("apple carrot carrot", 3),
        ]
    )
    assert index.df == {"apple": 2, "banana": 2, "carrot": 2}
    assert index.avgdl == pytest.approx(8 / 3)
    # N=3, df(apple)=2, dl=3, avgdl=8/3, tf=2.
    expected = math.log(1.6) * 4.4 / 3.3125
    result = index.search("apple apple", 3)
    assert result[0]["chunk_id"] == "id-1"
    assert result[0]["score"] == pytest.approx(expected)
    assert result == index.search("apple", 3)


def test_stable_ties_and_no_zero_score_fillers() -> None:
    index = BM25Index(
        [chunk("x", 4, 2), chunk("x", 2, 1), chunk("x", 1, 1), chunk("z", 3, 3)]
    )
    assert [r["chunk_id"] for r in index.search("x", 9)] == ["id-1", "id-2", "id-4"]
    assert [r["story_index"] for r in index.search("x", 9)] == [1, 1, 2]
    assert index.search("unseen", 5) == []
    title_only = chunk("ordinary text", 5)
    title_only["story_title"] = "Secret Keyword"
    assert BM25Index([title_only]).search("secret", 5) == []
    assert index.search("..._--", 5) == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, None])
def test_invalid_top_k(top_k) -> None:
    with pytest.raises(RetrievalError):
        BM25Index([chunk("x", 1)]).search("x", top_k)


@pytest.mark.parametrize(
    "k1,b",
    [
        (0, 0.75),
        (-1, 0.75),
        (math.nan, 0.75),
        (math.inf, 0.75),
        (1.2, -0.1),
        (1.2, 1.1),
        (1.2, math.nan),
        (1.2, math.inf),
    ],
)
def test_invalid_bm25_parameters(k1, b) -> None:
    with pytest.raises(RetrievalError):
        BM25Index([chunk("x", 1)], k1, b)


def test_selected_config_without_other_output_and_missing_command(
    tmp_path: Path,
) -> None:
    fixture_root(tmp_path)
    a = run_chunking(250, 50, tmp_path)
    b = run_chunking(500, 100, tmp_path)
    (tmp_path / b["output_path"]).unlink()
    chunks = load_verified_chunks("w250_o50", tmp_path)
    assert len(chunks) == a["chunk_count"]
    with pytest.raises(
        RetrievalError,
        match="python -m sherlock_rag.chunking --chunk-size-words 500 --overlap-words 100",
    ):
        load_verified_chunks("w500_o100", tmp_path)


def test_tampered_and_stale_selected_jsonl_rejected(tmp_path: Path) -> None:
    fixture_root(tmp_path)
    a = run_chunking(250, 50, tmp_path)
    path = tmp_path / a["output_path"]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(RetrievalError, match="differs from manifest"):
        load_verified_chunks("w250_o50", tmp_path)
    data = json.loads(original.splitlines()[0])
    data["text"] = "tampered"
    changed = (
        (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
        + b"\n".join(original.splitlines()[1:])
        + b"\n"
    )
    path.write_bytes(changed)
    manifest_path = tmp_path / "data/chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["configurations"]["w250_o50"]["sha256"] = sha256(changed)
    manifest["configurations"]["w250_o50"]["bytes"] = len(changed)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RetrievalError, match="failed audit"):
        load_verified_chunks("w250_o50", tmp_path)
    path.write_bytes(original)
    manifest["source"]["processed_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RetrievalError, match="source or algorithm"):
        load_verified_chunks("w250_o50", tmp_path)
