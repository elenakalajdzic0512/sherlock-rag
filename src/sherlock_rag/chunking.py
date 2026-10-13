"""Create deterministic, source-traceable word-window chunks of the verified book."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, extract_book, sha256

ALGORITHM_VERSION = "sliding-word-v1"
WORD_PATTERN = re.compile(r"\S+")


class ChunkingError(ValueError):
    """The source, configuration, or existing chunk artifacts are inconsistent."""


def validate_config(chunk_size_words: int, overlap_words: int) -> None:
    if chunk_size_words <= 0 or not 0 <= overlap_words < chunk_size_words:
        raise ChunkingError(
            "Require chunk_size_words > 0 and 0 <= overlap_words < chunk_size_words"
        )


def line_starts(text: str) -> list[int]:
    return [0, *(match.end() for match in re.finditer("\n", text))]


def line_number(starts: list[int], char_offset: int) -> int:
    return bisect_right(starts, char_offset)


def story_spans(
    book_text: str, story_lines: tuple[tuple[str, int, int], ...]
) -> list[tuple[int, str, int, int]]:
    starts = line_starts(book_text)
    boundaries: list[int] = []
    for heading, processed_line, _ in story_lines:
        if not 1 <= processed_line <= len(starts):
            raise ChunkingError("Story line is outside the processed book")
        offset = starts[processed_line - 1]
        if not book_text.startswith(heading, offset):
            raise ChunkingError("Story heading does not match the processed book")
        boundaries.append(offset)
    if not boundaries or boundaries[0] != 0 or boundaries != sorted(set(boundaries)):
        raise ChunkingError("Story boundaries are missing or out of order")
    return [
        (index, story_lines[index - 1][0].split(". ", 1)[1], start, end)
        for index, (start, end) in enumerate(
            zip(boundaries, boundaries[1:] + [len(book_text)], strict=True), 1
        )
    ]


def chunk_id(
    raw_sha256: str,
    processed_sha256: str,
    chunk_size_words: int,
    overlap_words: int,
    story_id: str,
    word_start: int,
    word_end: int,
    char_start: int,
    char_end: int,
) -> str:
    identity = "|".join(
        map(
            str,
            (
                raw_sha256,
                processed_sha256,
                ALGORITHM_VERSION,
                chunk_size_words,
                overlap_words,
                story_id,
                word_start,
                word_end,
                char_start,
                char_end,
            ),
        )
    )
    return "sh1661-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_chunks(
    book_text: str,
    story_lines: tuple[tuple[str, int, int], ...],
    source_line_start: int,
    raw_sha256: str,
    processed_sha256: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[dict]:
    """Window over regex words while slicing text from the original string."""
    validate_config(chunk_size_words, overlap_words)
    if source_line_start < 1:
        raise ChunkingError("Source line start must be positive")
    starts = line_starts(book_text)
    chunks: list[dict] = []
    stride = chunk_size_words - overlap_words
    for story_index, title, story_start, story_end in story_spans(
        book_text, story_lines
    ):
        words = list(WORD_PATTERN.finditer(book_text, story_start, story_end))
        if not words:
            raise ChunkingError(f"Story {story_index} contains no words")
        story_id = f"gutenberg-1661-story-{story_index:02d}"
        word_start = 0
        chunk_index = 1
        while word_start < len(words):
            word_end = min(word_start + chunk_size_words, len(words))
            char_start = words[word_start].start()
            char_end = words[word_end - 1].end()
            processed_line_start = line_number(starts, char_start)
            processed_line_end = line_number(starts, char_end - 1)
            chunks.append(
                {
                    "chunk_id": chunk_id(
                        raw_sha256,
                        processed_sha256,
                        chunk_size_words,
                        overlap_words,
                        story_id,
                        word_start,
                        word_end,
                        char_start,
                        char_end,
                    ),
                    "story_id": story_id,
                    "story_title": title,
                    "story_index": story_index,
                    "chunk_index": chunk_index,
                    "text": book_text[char_start:char_end],
                    "word_count": word_end - word_start,
                    "chunk_size_words": chunk_size_words,
                    "overlap_words": overlap_words,
                    "word_start": word_start,
                    "word_end": word_end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "processed_line_start": processed_line_start,
                    "processed_line_end": processed_line_end,
                    "source_line_start": source_line_start + processed_line_start - 1,
                    "source_line_end": source_line_start + processed_line_end - 1,
                    "raw_sha256": raw_sha256,
                    "processed_sha256": processed_sha256,
                    "algorithm_version": ALGORITHM_VERSION,
                }
            )
            if word_end == len(words):
                break
            word_start += stride
            chunk_index += 1
    audit_chunks(
        chunks,
        book_text,
        story_lines,
        source_line_start,
        raw_sha256,
        processed_sha256,
        chunk_size_words,
        overlap_words,
    )
    return chunks


def audit_chunks(
    chunks: list[dict],
    book_text: str,
    story_lines: tuple[tuple[str, int, int], ...],
    source_line_start: int,
    raw_sha256: str,
    processed_sha256: str,
    chunk_size_words: int,
    overlap_words: int,
) -> None:
    """Check every interval, story boundary, overlap, line mapping, and word."""
    spans = story_spans(book_text, story_lines)
    starts = line_starts(book_text)
    by_story: dict[int, list[dict]] = defaultdict(list)
    ids: set[str] = set()
    if [chunk["story_index"] for chunk in chunks] != sorted(
        chunk["story_index"] for chunk in chunks
    ):
        raise ChunkingError("Chunks are not ordered by story")
    for chunk in chunks:
        index = chunk["story_index"]
        if not 1 <= index <= len(spans):
            raise ChunkingError("Chunk has an invalid story index")
        _, title, story_start, story_end = spans[index - 1]
        first, end = chunk["char_start"], chunk["char_end"]
        if not story_start <= first < end <= story_end:
            raise ChunkingError("Chunk crosses a story boundary")
        if chunk["text"] != book_text[first:end]:
            raise ChunkingError("Chunk is not an exact substring")
        if (
            chunk["story_title"] != title
            or chunk["story_id"] != f"gutenberg-1661-story-{index:02d}"
        ):
            raise ChunkingError("Story metadata does not match its boundary")
        if chunk["word_count"] != len(WORD_PATTERN.findall(chunk["text"])):
            raise ChunkingError("Chunk word count does not match its text")
        if not 0 < chunk["word_count"] <= chunk_size_words:
            raise ChunkingError("Chunk size is outside the configured limit")
        if chunk["word_count"] != chunk["word_end"] - chunk["word_start"]:
            raise ChunkingError("Word interval does not match chunk size")
        processed_start = line_number(starts, first)
        processed_end = line_number(starts, end - 1)
        if (chunk["processed_line_start"], chunk["processed_line_end"]) != (
            processed_start,
            processed_end,
        ):
            raise ChunkingError("Processed line mapping is incorrect")
        if (chunk["source_line_start"], chunk["source_line_end"]) != (
            source_line_start + processed_start - 1,
            source_line_start + processed_end - 1,
        ):
            raise ChunkingError("Source line mapping is incorrect")
        if (chunk["raw_sha256"], chunk["processed_sha256"]) != (
            raw_sha256,
            processed_sha256,
        ):
            raise ChunkingError("Chunk source hashes do not match")
        if (
            chunk["chunk_size_words"],
            chunk["overlap_words"],
            chunk["algorithm_version"],
        ) != (
            chunk_size_words,
            overlap_words,
            ALGORITHM_VERSION,
        ):
            raise ChunkingError("Chunk configuration does not match")
        expected_id = chunk_id(
            raw_sha256,
            processed_sha256,
            chunk_size_words,
            overlap_words,
            chunk["story_id"],
            chunk["word_start"],
            chunk["word_end"],
            first,
            end,
        )
        if chunk["chunk_id"] != expected_id or expected_id in ids:
            raise ChunkingError("Chunk ID is incorrect or duplicated")
        ids.add(expected_id)
        by_story[index].append(chunk)
    if len(by_story) != len(spans):
        raise ChunkingError("At least one story has no chunks")
    for index, _, story_start, story_end in spans:
        words = list(WORD_PATTERN.finditer(book_text, story_start, story_end))
        items = by_story[index]
        if [item["chunk_index"] for item in items] != list(range(1, len(items) + 1)):
            raise ChunkingError("Chunk indices are not consecutive")
        if items[0]["word_start"] != 0 or items[-1]["word_end"] != len(words):
            raise ChunkingError("Story word coverage is incomplete")
        for item in items:
            a, b = item["word_start"], item["word_end"]
            if not 0 <= a < b <= len(words):
                raise ChunkingError("Word interval is outside its story")
            if (item["char_start"], item["char_end"]) != (
                words[a].start(),
                words[b - 1].end(),
            ):
                raise ChunkingError("Character interval does not match word boundaries")
        for previous, current in pairwise(items):
            if (
                current["word_start"]
                != previous["word_start"] + chunk_size_words - overlap_words
            ):
                raise ChunkingError("Chunk stride or overlap is incorrect")
            if previous["word_end"] != previous["word_start"] + chunk_size_words:
                raise ChunkingError("Extra window was created after a final chunk")
            if current["word_start"] > previous["word_end"]:
                raise ChunkingError("Story has a gap in word coverage")


def load_verified_source(
    root: Path,
) -> tuple[str, tuple[tuple[str, int, int], ...], dict]:
    """Verify both stored hashes and a fresh extraction; never download or alter source."""
    data = root / "data"
    raw_path = data / "raw/sherlock_holmes.txt"
    processed_path = data / "processed/book.txt"
    manifest_path = data / "source_manifest.json"
    for path in (raw_path, processed_path, manifest_path):
        if not path.is_file():
            raise ChunkingError(f"Required source file is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("book", {}).get("gutenberg_id") != 1661:
        raise ChunkingError("Source manifest is not for Gutenberg #1661")
    raw = raw_path.read_bytes()
    processed = processed_path.read_bytes()
    if sha256(raw) != manifest["raw"]["sha256"]:
        raise ChunkingError("Raw source SHA-256 differs from data/source_manifest.json")
    if sha256(processed) != manifest["processed"]["sha256"]:
        raise ChunkingError(
            "Processed source SHA-256 differs from data/source_manifest.json"
        )
    extracted = extract_book(raw)
    if extracted.text.encode("utf-8") != processed:
        raise ChunkingError(
            "Processed book differs from a fresh extraction of the raw source"
        )
    expected_lines = manifest["processed"]["source_lines_inclusive"]
    if (extracted.source_line_start, extracted.source_line_end) != (
        expected_lines["start"],
        expected_lines["end"],
    ):
        raise ChunkingError(
            "Source line interval differs from data/source_manifest.json"
        )
    return extracted.text, extracted.story_lines, manifest


def jsonl_bytes(chunks: list[dict]) -> bytes:
    return (
        "".join(
            json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for chunk in chunks
        )
    ).encode("utf-8")


def summary(
    chunks: list[dict], output_path: str, output: bytes, size: int, overlap: int
) -> dict:
    counts = [
        sum(chunk["story_index"] == index for chunk in chunks) for index in range(1, 13)
    ]
    lengths = [chunk["word_count"] for chunk in chunks]
    return {
        "output_path": output_path,
        "sha256": sha256(output),
        "bytes": len(output),
        "chunk_size_words": size,
        "overlap_words": overlap,
        "chunk_count": len(chunks),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "average_words": round(sum(lengths) / len(lengths), 2),
        "chunks_per_story": counts,
    }


def run_chunking(
    chunk_size_words: int, overlap_words: int, root: Path = PROJECT_ROOT
) -> dict:
    """Write one JSONL configuration and merge its manifest entry without losing others."""
    validate_config(chunk_size_words, overlap_words)
    text, story_lines, source_manifest = load_verified_source(root)
    raw_sha = source_manifest["raw"]["sha256"]
    processed_sha = source_manifest["processed"]["sha256"]
    source_start = source_manifest["processed"]["source_lines_inclusive"]["start"]
    source_identity = {
        "gutenberg_id": source_manifest["book"]["gutenberg_id"],
        "raw_sha256": raw_sha,
        "processed_sha256": processed_sha,
        "source_line_start": source_start,
    }
    chunks = build_chunks(
        text,
        story_lines,
        source_start,
        raw_sha,
        processed_sha,
        chunk_size_words,
        overlap_words,
    )
    key = f"w{chunk_size_words}_o{overlap_words}"
    relative_output = f"data/processed/chunks_{key}.jsonl"
    output_path = root / relative_output
    output = jsonl_bytes(chunks)
    manifest_path = root / "data/chunk_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("source") != source_identity
            or existing.get("algorithm_version") != ALGORITHM_VERSION
        ):
            raise ChunkingError(
                "Existing chunk manifest refers to a different source or algorithm"
            )
        for other_key, entry in existing["configurations"].items():
            if other_key == key:
                continue
            other_path = root / entry["output_path"]
            if other_path.exists() and (
                not other_path.is_file()
                or sha256(other_path.read_bytes()) != entry["sha256"]
            ):
                raise ChunkingError(f"Existing output for {other_key} has changed")
    else:
        existing = {
            "algorithm_version": ALGORITHM_VERSION,
            "source": source_identity,
            "configurations": {},
        }
    result = summary(chunks, relative_output, output, chunk_size_words, overlap_words)
    existing["configurations"][key] = result
    manifest_bytes = (
        json.dumps(existing, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if not output_path.exists() or output_path.read_bytes() != output:
        atomic_write(output_path, output)
    if not manifest_path.exists() or manifest_path.read_bytes() != manifest_bytes:
        atomic_write(manifest_path, manifest_bytes)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size-words", type=int, required=True)
    parser.add_argument("--overlap-words", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_chunking(args.chunk_size_words, args.overlap_words)
    except (ChunkingError, OSError, ValueError, KeyError) as exc:
        print(f"Chunking failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
