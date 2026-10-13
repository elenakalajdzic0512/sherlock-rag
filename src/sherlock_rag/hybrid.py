"""Fuse verified BM25 and semantic rankings with reciprocal rank fusion."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from sherlock_rag.ingest import PROJECT_ROOT
from sherlock_rag.retrieval import CONFIGURATIONS, BM25Index

DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 5
METHOD = "equal-weight-rrf-v1"


class HybridError(ValueError):
    """Hybrid inputs, candidate rankings, or parameters are inconsistent."""


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HybridError(f"{name} must be a positive integer")
    return value


class HybridIndex:
    """RRF over two rankings of the same ordered, source-verified chunks."""

    def __init__(
        self,
        bm25: object,
        semantic: object,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.candidate_k = positive_int(candidate_k, "candidate_k")
        self.rrf_k = positive_int(rrf_k, "rrf_k")
        self.bm25 = bm25
        self.semantic = semantic
        left = bm25.chunks
        right = semantic.chunks
        if not left or len(left) != len(right):
            raise HybridError("BM25 and semantic chunk sets differ")
        ids = [chunk["chunk_id"] for chunk in left]
        if len(ids) != len(set(ids)):
            raise HybridError("Chunk IDs are duplicated")
        identity_fields = (
            "chunk_id",
            "story_id",
            "story_index",
            "chunk_index",
            "char_start",
            "char_end",
            "raw_sha256",
            "processed_sha256",
            "chunk_size_words",
            "overlap_words",
            "text",
        )
        for a, b in zip(left, right, strict=True):
            if any(a.get(key) != b.get(key) for key in identity_fields):
                raise HybridError(
                    "BM25 and semantic chunk IDs, source, or order differ"
                )
        self.chunks = left
        self.by_id = {chunk["chunk_id"]: chunk for chunk in left}

    def _validated_candidates(self, method: str, results: list[dict]) -> list[dict]:
        if len(results) > self.candidate_k:
            raise HybridError(f"{method} returned too many candidates")
        seen: set[str] = set()
        for position, result in enumerate(results, 1):
            chunk_id = result.get("chunk_id")
            if chunk_id in seen or chunk_id not in self.by_id:
                raise HybridError(f"{method} returned a duplicate or unknown chunk ID")
            seen.add(chunk_id)
            chunk = self.by_id[chunk_id]
            if result.get("rank") != position:
                raise HybridError(f"{method} returned an invalid rank")
            if any(
                result.get(key) != chunk.get(key)
                for key in (
                    "story_id",
                    "story_index",
                    "chunk_index",
                    "char_start",
                    "char_end",
                    "source_line_start",
                    "source_line_end",
                    "text",
                )
            ):
                raise HybridError(f"{method} candidate source or text differs")
            score = result.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise HybridError(f"{method} returned an invalid score")
        return results

    def search_with_candidates(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> tuple[list[dict], list[dict]]:
        top_k = positive_int(top_k, "top_k")
        if self.candidate_k < top_k:
            raise HybridError("candidate_k must be >= top_k")
        if not query.strip():
            return [], []
        bm25 = self._validated_candidates(
            "BM25", self.bm25.search(query, self.candidate_k)
        )
        semantic = self._validated_candidates(
            "semantic", self.semantic.search(query, self.candidate_k)
        )
        union: dict[str, dict] = {}
        for method, ranking in (("bm25", bm25), ("semantic", semantic)):
            for result in ranking:
                slot = union.setdefault(result["chunk_id"], {})
                slot[f"{method}_rank"] = result["rank"]
                slot[f"{method}_score"] = result["score"]
        ranked = []
        for chunk_id, ranks in union.items():
            chunk = self.by_id[chunk_id]
            bm_rank = ranks.get("bm25_rank")
            sem_rank = ranks.get("semantic_rank")
            bm_part = 1 / (self.rrf_k + bm_rank) if bm_rank is not None else 0.0
            sem_part = 1 / (self.rrf_k + sem_rank) if sem_rank is not None else 0.0
            ranked.append(
                {
                    "score_type": METHOD,
                    "score": bm_part + sem_part,
                    "chunk_id": chunk_id,
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
                    "bm25_rank": bm_rank,
                    "bm25_score": ranks.get("bm25_score"),
                    "semantic_rank": sem_rank,
                    "semantic_score": ranks.get("semantic_score"),
                    "bm25_rrf_contribution": bm_part,
                    "semantic_rrf_contribution": sem_part,
                }
            )
        ranked.sort(
            key=lambda item: (-item["score"], item["story_index"], item["chunk_index"])
        )
        for rank, result in enumerate(ranked[:top_k], 1):
            result["rank"] = rank
        candidates = [
            self.by_id[chunk_id]
            for chunk_id in sorted(
                union,
                key=lambda cid: (
                    self.by_id[cid]["story_index"],
                    self.by_id[cid]["chunk_index"],
                ),
            )
        ]
        return ranked[:top_k], candidates

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        return self.search_with_candidates(query, top_k)[0]


def load_hybrid(
    config: str,
    encoder: object,
    root: Path = PROJECT_ROOT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[HybridIndex, dict]:
    from sherlock_rag.semantic import SemanticIndex, load_index

    chunks, vectors, meta = load_index(config, root)
    return HybridIndex(
        BM25Index(chunks), SemanticIndex(chunks, vectors, encoder), candidate_k, rrf_k
    ), meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    args = parser.parse_args(argv)
    try:
        from sherlock_rag.semantic import SemanticIndex, load_encoder, load_index

        positive_int(args.top_k, "top_k")
        positive_int(args.candidate_k, "candidate_k")
        positive_int(args.rrf_k, "rrf_k")
        if args.candidate_k < args.top_k:
            raise HybridError("candidate_k must be >= top_k")
        # Verify the selected local matrix before loading the model; never rebuild it here.
        chunks, vectors, meta = load_index(args.config)
        encoder, _ = load_encoder()
        index = HybridIndex(
            BM25Index(chunks),
            SemanticIndex(chunks, vectors, encoder),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
        results = index.search(args.query, args.top_k)
    except (HybridError, OSError, ValueError, ImportError) as exc:
        print(f"Hybrid retrieval failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "method": METHOD,
                "config": args.config,
                "model_revision": meta["model_revision"],
                "query": args.query,
                "parameters": {
                    "top_k": args.top_k,
                    "candidate_k": args.candidate_k,
                    "rrf_k": args.rrf_k,
                    "weights": {"bm25": 1, "semantic": 1},
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
