"""Verified in-memory BM25 search over one chunk configuration."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from sherlock_rag.chunking import (
    ALGORITHM_VERSION,
    ChunkingError,
    audit_chunks,
    load_verified_source,
)
from sherlock_rag.ingest import PROJECT_ROOT, sha256

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
CONFIGURATIONS = {"w250_o50": (250, 50), "w500_o100": (500, 100)}
K1 = 1.2
B = 0.75


class RetrievalError(ValueError):
    """An input or verified retrieval artifact is invalid."""


def tokenize(text: str) -> list[str]:
    """Casefold Unicode words/numbers; preserve internal apostrophes only."""
    return TOKEN_PATTERN.findall(text.replace("’", "'").casefold())


def load_verified_chunks(config: str, root: Path = PROJECT_ROOT) -> list[dict]:
    """Verify the selected JSONL, its provenance, and every source interval."""
    if config not in CONFIGURATIONS:
        raise RetrievalError(f"Unknown configuration: {config}")
    book, story_lines, source = load_verified_source(root)
    manifest_path = root / "data/chunk_manifest.json"
    if not manifest_path.is_file():
        raise RetrievalError(f"Chunk manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_line_start = source["processed"]["source_lines_inclusive"]["start"]
    identity = {
        "gutenberg_id": 1661,
        "raw_sha256": source["raw"]["sha256"],
        "processed_sha256": source["processed"]["sha256"],
        "source_line_start": source_line_start,
    }
    if (
        manifest.get("source") != identity
        or manifest.get("algorithm_version") != ALGORITHM_VERSION
    ):
        raise RetrievalError("Chunk manifest source or algorithm does not match")
    entry = manifest.get("configurations", {}).get(config)
    if entry is None:
        raise RetrievalError(f"Chunk manifest has no {config} entry")
    size, overlap = CONFIGURATIONS[config]
    expected_path = f"data/processed/chunks_{config}.jsonl"
    if (
        entry.get("chunk_size_words"),
        entry.get("overlap_words"),
        entry.get("output_path"),
    ) != (size, overlap, expected_path):
        raise RetrievalError(f"Chunk manifest configuration {config} is invalid")
    path = root / expected_path
    if not path.is_file():
        raise RetrievalError(
            f"Selected chunk output is missing: {path}. Regenerate with: "
            f"python -m sherlock_rag.chunking --chunk-size-words {size} "
            f"--overlap-words {overlap}"
        )
    data = path.read_bytes()
    if sha256(data) != entry.get("sha256") or len(data) != entry.get("bytes"):
        raise RetrievalError(f"Selected chunk output {config} differs from manifest")
    try:
        chunks = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        if len(chunks) != entry.get("chunk_count"):
            raise RetrievalError(f"Selected chunk count {config} differs from manifest")
        audit_chunks(
            chunks,
            book,
            story_lines,
            source_line_start,
            identity["raw_sha256"],
            identity["processed_sha256"],
            size,
            overlap,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        ChunkingError,
        KeyError,
        TypeError,
        IndexError,
    ) as exc:
        raise RetrievalError(
            f"Selected chunk output {config} failed audit: {exc}"
        ) from exc
    return chunks


class BM25Index:
    """Robertson-style positive IDF BM25, indexing chunk text only."""

    def __init__(self, chunks: list[dict], k1: float = K1, b: float = B) -> None:
        if (
            isinstance(k1, bool)
            or not isinstance(k1, (int, float))
            or not math.isfinite(k1)
            or k1 <= 0
        ):
            raise RetrievalError("k1 must be a finite positive number")
        if (
            isinstance(b, bool)
            or not isinstance(b, (int, float))
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise RetrievalError("b must be a finite number between 0 and 1")
        if not chunks:
            raise RetrievalError("Cannot index an empty chunk list")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.avgdl = sum(self.lengths) / len(chunks)
        if self.avgdl == 0:
            raise RetrievalError("Cannot index chunks without retrieval tokens")
        self.df = Counter(term for counts in self.counts for term in counts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalError("top_k must be a positive integer")
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        n = len(self.chunks)
        ranked = []
        for chunk, counts, length in zip(
            self.chunks, self.counts, self.lengths, strict=True
        ):
            score = 0.0
            found = []
            for term in terms:
                tf = counts[term]
                if tf == 0:
                    continue
                found.append(term)
                df = self.df[term]
                idf = math.log1p((n - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
                )
            if score > 0:
                ranked.append((score, chunk, found))
        ranked.sort(
            key=lambda item: (-item[0], item[1]["story_index"], item[1]["chunk_index"])
        )
        return [
            {
                "rank": rank,
                "score": score,
                "chunk_id": chunk["chunk_id"],
                "story_id": chunk["story_id"],
                "story_title": chunk["story_title"],
                "story_index": chunk["story_index"],
                "chunk_index": chunk["chunk_index"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "processed_line_start": chunk["processed_line_start"],
                "processed_line_end": chunk["processed_line_end"],
                "source_line_start": chunk["source_line_start"],
                "source_line_end": chunk["source_line_end"],
                "text": chunk["text"],
                "matched_query_terms": found,
                "context_words": chunk["word_count"],
            }
            for rank, (score, chunk, found) in enumerate(ranked[:top_k], 1)
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        results = BM25Index(load_verified_chunks(args.config)).search(
            args.query, args.top_k
        )
    except (RetrievalError, ChunkingError, OSError, ValueError, KeyError) as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"config": args.config, "query": args.query, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
