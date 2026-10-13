"""Pinned local Qwen embedding indexes and exact cosine retrieval.

Heavy semantic dependencies are imported only when this module's functions run.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256
from sherlock_rag.retrieval import CONFIGURATIONS, load_verified_chunks

if TYPE_CHECKING:
    import numpy as np

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TASK = (
    "Given a question about a book, retrieve passages containing evidence to answer it."
)
QUERY_PREFIX = f"Instruct: {TASK}\nQuery:"
MAX_SEQ_LENGTH = 4096
DIMENSION = 1024
BATCH_SIZE = 2
INDEX_VERSION = "qwen3-embedding-0.6b-cosine-v1"


class SemanticError(ValueError):
    """A model, token length, vector, or saved index fails validation."""


def rebuild_command(config: str) -> str:
    return f"python -m sherlock_rag.semantic build --config {config}"


def normalize_vectors(
    values: object, *, expected_rows: int | None = None
) -> np.ndarray:
    """Convert to full-dimension finite float32 and enforce nonzero L2 norms."""
    import numpy as np

    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION:
        raise SemanticError(f"Expected a 2D matrix with {DIMENSION} columns")
    if expected_rows is not None and vectors.shape[0] != expected_rows:
        raise SemanticError(f"Expected {expected_rows} vectors, got {vectors.shape[0]}")
    if not np.isfinite(vectors).all():
        raise SemanticError("Vectors contain NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise SemanticError("Vectors contain a zero or invalid norm")
    vectors = vectors / norms[:, None]
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0):
        raise SemanticError("Vectors are not unit length after normalization")
    return vectors.astype(np.float32, copy=False)


def check_token_lengths(
    tokenizer: object, texts: list[str], limit: int, label: str
) -> int:
    """Measure complete tokenized inputs, including special tokens, without truncation."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise SemanticError("Token limit must be a positive integer")
    longest = 0
    for index, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        length = len(ids)
        longest = max(longest, length)
        if length > limit:
            raise SemanticError(
                f"{label} {index} has {length} tokens, exceeding {limit}; "
                "no input was truncated or skipped"
            )
    return longest


def package_versions() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in ("numpy", "sentence-transformers", "torch", "transformers")
    }


class QwenEncoder:
    """Pinned SentenceTransformer adapter with a fixed query-only instruction."""

    def __init__(self, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(
            MODEL_ID, revision=MODEL_REVISION, device=device
        )
        model_limit = self.model.max_seq_length
        tokenizer_limit = self.model.tokenizer.model_max_length
        if model_limit < MAX_SEQ_LENGTH or tokenizer_limit < MAX_SEQ_LENGTH:
            raise SemanticError(
                f"Model/tokenizer limit {model_limit}/{tokenizer_limit} is below {MAX_SEQ_LENGTH}"
            )
        self.model.max_seq_length = MAX_SEQ_LENGTH
        self.tokenizer = self.model.tokenizer
        self._query_cache = {}
        # This real, tiny encode is also the MPS smoke check when MPS is chosen.
        probe = self.model.encode(
            ["A short Sherlock Holmes passage."],
            prompt="",
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        normalize_vectors(probe, expected_rows=1)

    def measure_documents(self, texts: list[str]) -> int:
        return check_token_lengths(self.tokenizer, texts, MAX_SEQ_LENGTH, "document")

    def measure_queries(self, queries: list[str]) -> int:
        return check_token_lengths(
            self.tokenizer,
            [QUERY_PREFIX + query for query in queries],
            MAX_SEQ_LENGTH,
            "query",
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            prompt="",
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return normalize_vectors(vectors, expected_rows=len(texts))

    def encode_query(self, query: str) -> np.ndarray:
        if query in self._query_cache:
            return self._query_cache[query]
        self.measure_queries([query])
        vector = self.model.encode(
            [query],
            prompt=QUERY_PREFIX,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = normalize_vectors(vector, expected_rows=1)[0]
        if len(self._query_cache) >= 256:
            self._query_cache.clear()
        self._query_cache[query] = result
        return result


def load_encoder() -> tuple[QwenEncoder, str | None]:
    """Prefer MPS after a real encode; fall back to CPU only on an MPS failure."""
    import torch

    if torch.backends.mps.is_available():
        try:
            return QwenEncoder("mps"), None
        except Exception as exc:  # noqa: BLE001 - any MPS smoke failure warrants CPU fallback
            reason = f"MPS smoke encode failed: {type(exc).__name__}: {exc}"
            return QwenEncoder("cpu"), reason
    return QwenEncoder("cpu"), "MPS unavailable"


def index_paths(config: str, root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    if config not in CONFIGURATIONS:
        raise SemanticError(f"Unknown configuration: {config}")
    return root / f"indexes/{config}.npy", root / f"indexes/{config}.json"


def build_index(
    config: str,
    encoder: object,
    root: Path = PROJECT_ROOT,
    device_note: str | None = None,
) -> dict:
    """Audit source and lengths, then atomically persist one ordered float32 matrix."""
    import numpy as np

    chunks = load_verified_chunks(config, root)
    texts = [chunk["text"] for chunk in chunks]
    started = time.perf_counter()
    max_document_tokens = encoder.measure_documents(texts)
    vectors = normalize_vectors(
        encoder.encode_documents(texts), expected_rows=len(chunks)
    )
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    vector_bytes = buffer.getvalue()
    elapsed = time.perf_counter() - started
    vector_path, meta_path = index_paths(config, root)
    source_manifest = (root / "data/source_manifest.json").read_bytes()
    chunk_manifest = json.loads(
        (root / "data/chunk_manifest.json").read_text(encoding="utf-8")
    )
    chunk_bytes = (
        root / chunk_manifest["configurations"][config]["output_path"]
    ).read_bytes()
    meta = {
        "index_version": INDEX_VERSION,
        "config": config,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "query_prefix": QUERY_PREFIX,
        "pooling": "sentence-transformers Pooling lasttoken, include_prompt=True; Normalize module",
        "normalization": "L2 unit vectors; exact float32 dot product",
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_document_tokens": max_document_tokens,
        "zero_truncated_inputs": True,
        "batch_size": BATCH_SIZE,
        "versions": package_versions(),
        "device": encoder.device,
        "device_note": device_note,
        "dtype": "float32",
        "shape": list(vectors.shape),
        "ordered_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "source_manifest_sha256": sha256(source_manifest),
        "raw_sha256": chunk_manifest["source"]["raw_sha256"],
        "processed_sha256": chunk_manifest["source"]["processed_sha256"],
        "chunks_sha256": sha256(chunk_bytes),
        "vectors_sha256": sha256(vector_bytes),
        "build_seconds": elapsed,
    }
    atomic_write(vector_path, vector_bytes)
    atomic_write(
        meta_path,
        (
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return meta


def load_index(
    config: str, root: Path = PROJECT_ROOT
) -> tuple[list[dict], np.ndarray, dict]:
    import numpy as np

    chunks = load_verified_chunks(config, root)
    vector_path, meta_path = index_paths(config, root)
    if not vector_path.is_file() or not meta_path.is_file():
        raise SemanticError(
            f"Semantic index for {config} is missing. Rebuild with: {rebuild_command(config)}"
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {
            "index_version": INDEX_VERSION,
            "config": config,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "query_prefix": QUERY_PREFIX,
            "pooling": "sentence-transformers Pooling lasttoken, include_prompt=True; Normalize module",
            "normalization": "L2 unit vectors; exact float32 dot product",
            "versions": package_versions(),
            "max_seq_length": MAX_SEQ_LENGTH,
            "dtype": "float32",
            "shape": [len(chunks), DIMENSION],
            "ordered_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            "source_manifest_sha256": sha256(
                (root / "data/source_manifest.json").read_bytes()
            ),
            "raw_sha256": chunks[0]["raw_sha256"],
            "processed_sha256": chunks[0]["processed_sha256"],
            "chunks_sha256": sha256(
                (root / f"data/processed/chunks_{config}.jsonl").read_bytes()
            ),
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise SemanticError(f"Semantic index {key} is stale or mismatched")
        if (
            meta.get("zero_truncated_inputs") is not True
            or not isinstance(meta.get("max_document_tokens"), int)
            or not 0 < meta["max_document_tokens"] <= MAX_SEQ_LENGTH
            or meta.get("device") not in ("mps", "cpu")
        ):
            raise SemanticError(
                "Semantic index token audit or device metadata is invalid"
            )
        data = vector_path.read_bytes()
        if sha256(data) != meta.get("vectors_sha256"):
            raise SemanticError("Semantic vector file hash differs from metadata")
        vectors = np.load(io.BytesIO(data), allow_pickle=False)
        if vectors.dtype != np.float32 or vectors.shape != (len(chunks), DIMENSION):
            raise SemanticError(
                "Semantic vector matrix dtype or shape differs from metadata"
            )
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1, atol=1e-3, rtol=0
        ):
            raise SemanticError("Semantic vector matrix has invalid values or norms")
        return chunks, vectors, meta
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SemanticError(
            f"Semantic index for {config} is invalid: {exc}. Rebuild with: {rebuild_command(config)}"
        ) from exc


class SemanticIndex:
    """Exact cosine search over a validated matrix; no answerability threshold."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray, encoder: object):
        import numpy as np

        self.chunks = chunks
        self.vectors = normalize_vectors(vectors, expected_rows=len(chunks))
        self.encoder = encoder
        if not np.allclose(self.vectors, vectors, atol=1e-3, rtol=0):
            raise SemanticError("Document vectors are not normalized")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        import numpy as np

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise SemanticError("top_k must be a positive integer")
        if not query.strip():
            return []
        query_vector = normalize_vectors(
            np.asarray(self.encoder.encode_query(query)).reshape(1, -1), expected_rows=1
        )[0]
        scores = self.vectors @ query_vector
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: (
                -float(scores[i]),
                self.chunks[i]["story_index"],
                self.chunks[i]["chunk_index"],
            ),
        )
        return [
            {
                "rank": rank,
                "score": float(scores[i]),
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
                "context_words": chunk["word_count"],
            }
            for rank, i in enumerate(ranked[:top_k], 1)
            for chunk in [self.chunks[i]]
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "search"):
        command = commands.add_parser(name)
        command.add_argument("--config", choices=CONFIGURATIONS, required=True)
        if name == "search":
            command.add_argument("--query", required=True)
            command.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            encoder, note = load_encoder()
            print(
                json.dumps(
                    build_index(args.config, encoder, device_note=note),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            chunks, vectors, meta = load_index(args.config)
            encoder, _ = load_encoder()
            results = SemanticIndex(chunks, vectors, encoder).search(
                args.query, args.top_k
            )
            print(
                json.dumps(
                    {
                        "config": args.config,
                        "model_revision": meta["model_revision"],
                        "query": args.query,
                        "results": results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except (SemanticError, OSError, ValueError, ImportError) as exc:
        print(f"Semantic retrieval failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
