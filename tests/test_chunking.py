"""Offline tests of word windows, provenance, and artifact preservation."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from sherlock_rag.chunking import (
    ChunkingError,
    audit_chunks,
    build_chunks,
    jsonl_bytes,
    load_verified_source,
    run_chunking,
)
from sherlock_rag.ingest import (
    END_MARKER,
    HEADINGS,
    START_MARKER,
    extract_book,
    sha256,
)

RAW_SHA = "a" * 64
PROCESSED_SHA = "b" * 64


def small_book(
    word_counts: tuple[int, ...],
) -> tuple[str, tuple[tuple[str, int, int], ...]]:
    text = ""
    story_lines = []
    for index, total_words in enumerate(word_counts, 1):
        assert total_words >= 3
        heading = f"{('I', 'II', 'III')[index - 1]}. STORY {index}"
        story_lines.append((heading, len(text.splitlines()) + 1, 1))
        text += (
            heading + "\n" + " ".join(f"word{n}" for n in range(total_words - 3)) + "\n"
        )
    return text, tuple(story_lines)


def chunks_for(
    counts: tuple[int, ...], size: int, overlap: int
) -> tuple[str, tuple, list[dict]]:
    text, stories = small_book(counts)
    return (
        text,
        stories,
        build_chunks(text, stories, 55, RAW_SHA, PROCESSED_SHA, size, overlap),
    )


@pytest.mark.parametrize("count", [6, 10])
def test_short_and_exact_window_have_one_chunk(count: int) -> None:
    text, _, chunks = chunks_for((count,), 10, 2)
    assert len(chunks) == 1
    assert chunks[0]["word_start"] == 0
    assert chunks[0]["word_end"] == count
    assert chunks[0]["text"] == text[chunks[0]["char_start"] : chunks[0]["char_end"]]


def test_multiple_windows_and_short_last_window() -> None:
    _, _, chunks = chunks_for((12,), 5, 2)
    assert [(c["word_start"], c["word_end"]) for c in chunks] == [
        (0, 5),
        (3, 8),
        (6, 11),
        (9, 12),
    ]
    assert chunks[-1]["word_count"] == 3
    assert [c["chunk_index"] for c in chunks] == [1, 2, 3, 4]


def test_exact_overlap_and_no_contained_extra_window() -> None:
    _, _, chunks = chunks_for((11,), 5, 2)
    assert [(c["word_start"], c["word_end"]) for c in chunks] == [
        (0, 5),
        (3, 8),
        (6, 11),
    ]
    assert all(a["word_end"] - b["word_start"] == 2 for a, b in pairwise(chunks))


def test_no_overlap_and_invalid_configuration() -> None:
    _, _, chunks = chunks_for((11,), 5, 0)
    assert [(c["word_start"], c["word_end"]) for c in chunks] == [
        (0, 5),
        (5, 10),
        (10, 11),
    ]
    text, stories = small_book((11,))
    for size, overlap in ((0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)):
        with pytest.raises(ChunkingError):
            build_chunks(text, stories, 55, RAW_SHA, PROCESSED_SHA, size, overlap)


def test_story_boundary_coverage_and_exact_substrings() -> None:
    text, stories, chunks = chunks_for((17, 13), 5, 2)
    assert [c["story_index"] for c in chunks] == sorted(
        c["story_index"] for c in chunks
    )
    for index, total in enumerate((17, 13), 1):
        items = [c for c in chunks if c["story_index"] == index]
        assert set().union(
            *(set(range(c["word_start"], c["word_end"])) for c in items)
        ) == set(range(total))
        assert all(c["story_id"].endswith(f"{index:02d}") for c in items)
        assert all(c["text"] == text[c["char_start"] : c["char_end"]] for c in items)
    audit_chunks(chunks, text, stories, 55, RAW_SHA, PROCESSED_SHA, 5, 2)


def test_unicode_character_offsets_and_line_mapping() -> None:
    text = "I. ÀLPHA\nnaïve 😀 café\n"
    stories = (("I. ÀLPHA", 1, 55),)
    chunks = build_chunks(text, stories, 137, RAW_SHA, PROCESSED_SHA, 3, 1)
    assert len(chunks) == 2
    assert chunks[1]["char_start"] == text.index("naïve")
    assert (
        len(text[: chunks[1]["char_start"]].encode("utf-8")) > chunks[1]["char_start"]
    )
    assert chunks[1]["processed_line_start"] == 2
    assert chunks[1]["source_line_start"] == 138
    assert chunks[1]["text"] == text[chunks[1]["char_start"] : chunks[1]["char_end"]]


def test_stable_ids_and_jsonl_and_audit_rejects_tampering() -> None:
    text, stories, first = chunks_for((14, 14), 6, 2)
    second = build_chunks(text, stories, 55, RAW_SHA, PROCESSED_SHA, 6, 2)
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert len({c["chunk_id"] for c in first}) == len(first)
    assert jsonl_bytes(first) == jsonl_bytes(second)
    changed = [dict(c) for c in first]
    changed[0]["text"] = "wrong"
    with pytest.raises(ChunkingError, match="exact substring"):
        audit_chunks(changed, text, stories, 55, RAW_SHA, PROCESSED_SHA, 6, 2)


def fixture_root(root: Path) -> None:
    lines = [
        "Title: The Adventures of Sherlock Holmes",
        "Author: Arthur Conan Doyle",
        "Release date: March 1, 1999 [eBook #1661]",
        START_MARKER,
        "The Adventures of Sherlock Holmes",
        "Contents",
    ]
    for heading in HEADINGS:
        roman, title = heading.split(". ", 1)
        lines.append(f"   {roman}. {title.title()}")
    lines.append("")
    for heading in HEADINGS:
        lines += [
            heading,
            "",
            "A story has several words, _italics_, and punctuation.",
            "",
        ]
    lines += [END_MARKER, "Gutenberg license"]
    raw = "\n".join(lines).encode("utf-8")
    book = extract_book(raw)
    processed = book.text.encode("utf-8")
    (root / "data/raw").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "data/raw/sherlock_holmes.txt").write_bytes(raw)
    (root / "data/processed/book.txt").write_bytes(processed)
    manifest = {
        "book": {"gutenberg_id": 1661},
        "raw": {"sha256": sha256(raw)},
        "processed": {
            "sha256": sha256(processed),
            "source_lines_inclusive": {
                "start": book.source_line_start,
                "end": book.source_line_end,
            },
        },
    }
    (root / "data/source_manifest.json").write_text(json.dumps(manifest))


def test_both_configurations_and_repeat_preserve_other_output(tmp_path: Path) -> None:
    fixture_root(tmp_path)
    a = run_chunking(8, 2, tmp_path)
    a_path = tmp_path / a["output_path"]
    a_bytes = a_path.read_bytes()
    b = run_chunking(10, 0, tmp_path)
    b_path = tmp_path / b["output_path"]
    b_bytes = b_path.read_bytes()
    manifest_path = tmp_path / "data/chunk_manifest.json"
    before = manifest_path.read_bytes()
    assert set(json.loads(before)["configurations"]) == {"w8_o2", "w10_o0"}
    assert run_chunking(8, 2, tmp_path) == a
    assert (a_path.read_bytes(), b_path.read_bytes(), manifest_path.read_bytes()) == (
        a_bytes,
        b_bytes,
        before,
    )
    assert a["sha256"] == sha256(a_bytes)
    assert b["sha256"] == sha256(b_bytes)


def test_modified_source_or_stale_manifest_is_rejected(tmp_path: Path) -> None:
    fixture_root(tmp_path)
    raw_path = tmp_path / "data/raw/sherlock_holmes.txt"
    processed_path = tmp_path / "data/processed/book.txt"
    original_raw = raw_path.read_bytes()
    original_processed = processed_path.read_bytes()
    processed_path.unlink()
    with pytest.raises(ChunkingError, match="Required source file is missing"):
        run_chunking(8, 2, tmp_path)
    processed_path.write_bytes(original_processed)
    source_manifest_path = tmp_path / "data/source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["book"]["gutenberg_id"] = 1662
    source_manifest_path.write_text(json.dumps(source_manifest))
    with pytest.raises(ChunkingError, match="Gutenberg #1661"):
        run_chunking(8, 2, tmp_path)
    source_manifest["book"]["gutenberg_id"] = 1661
    source_manifest_path.write_text(json.dumps(source_manifest))
    raw_path.write_bytes(original_raw + b"tamper")
    with pytest.raises(ChunkingError, match="Raw source SHA-256"):
        load_verified_source(tmp_path)
    raw_path.write_bytes(original_raw)
    processed_path.write_bytes(original_processed + b"tamper")
    with pytest.raises(ChunkingError, match="Processed source SHA-256"):
        run_chunking(8, 2, tmp_path)
    processed_path.write_bytes(original_processed)
    run_chunking(8, 2, tmp_path)
    manifest_path = tmp_path / "data/chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["raw_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ChunkingError, match="different source"):
        run_chunking(10, 0, tmp_path)


@pytest.mark.parametrize("order", [((8, 2), (10, 0)), ((10, 0), (8, 2))])
def test_missing_both_outputs_regenerate_in_either_order(
    tmp_path: Path, order: tuple[tuple[int, int], tuple[int, int]]
) -> None:
    fixture_root(tmp_path)
    expected = {}
    original_files = {}
    for size, overlap in ((8, 2), (10, 0)):
        result = run_chunking(size, overlap, tmp_path)
        expected[(size, overlap)] = result
        output = tmp_path / result["output_path"]
        original_files[(size, overlap)] = output.read_bytes()
    manifest_path = tmp_path / "data/chunk_manifest.json"
    original_manifest = manifest_path.read_bytes()
    for result in expected.values():
        (tmp_path / result["output_path"]).unlink()
    for size, overlap in order:
        assert run_chunking(size, overlap, tmp_path) == expected[(size, overlap)]
        assert manifest_path.read_bytes() == original_manifest
        assert set(json.loads(manifest_path.read_text())["configurations"]) == {
            "w8_o2",
            "w10_o0",
        }
    for config, result in expected.items():
        assert (tmp_path / result["output_path"]).read_bytes() == original_files[config]
    for size, overlap in order:
        assert run_chunking(size, overlap, tmp_path) == expected[(size, overlap)]
    assert manifest_path.read_bytes() == original_manifest
    for config, result in expected.items():
        assert (tmp_path / result["output_path"]).read_bytes() == original_files[config]


@pytest.mark.parametrize("requested,missing", [((8, 2), (10, 0)), ((10, 0), (8, 2))])
def test_missing_other_output_keeps_both_manifest_entries(
    tmp_path: Path, requested: tuple[int, int], missing: tuple[int, int]
) -> None:
    fixture_root(tmp_path)
    results = {config: run_chunking(*config, tmp_path) for config in ((8, 2), (10, 0))}
    manifest_path = tmp_path / "data/chunk_manifest.json"
    original_manifest = manifest_path.read_bytes()
    requested_path = tmp_path / results[requested]["output_path"]
    requested_bytes = requested_path.read_bytes()
    missing_path = tmp_path / results[missing]["output_path"]
    missing_path.unlink()
    assert run_chunking(*requested, tmp_path) == results[requested]
    assert requested_path.read_bytes() == requested_bytes
    assert not missing_path.exists()
    assert manifest_path.read_bytes() == original_manifest
    assert set(json.loads(manifest_path.read_text())["configurations"]) == {
        "w8_o2",
        "w10_o0",
    }


def test_changed_existing_other_output_blocks_without_modifications(
    tmp_path: Path,
) -> None:
    fixture_root(tmp_path)
    a = run_chunking(8, 2, tmp_path)
    b = run_chunking(10, 0, tmp_path)
    a_path = tmp_path / a["output_path"]
    b_path = tmp_path / b["output_path"]
    manifest_path = tmp_path / "data/chunk_manifest.json"
    b_path.write_bytes(b"changed output")
    before = (a_path.read_bytes(), b_path.read_bytes(), manifest_path.read_bytes())
    with pytest.raises(ChunkingError, match="Existing output for w10_o0 has changed"):
        run_chunking(8, 2, tmp_path)
    assert (
        a_path.read_bytes(),
        b_path.read_bytes(),
        manifest_path.read_bytes(),
    ) == before
