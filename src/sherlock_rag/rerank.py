"""Explicit, pinned cross-encoder reranking of verified A chunk candidates."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from sherlock_rag.ingest import PROJECT_ROOT
from sherlock_rag.retrieval import BM25Index
from sherlock_rag.semantic import SemanticIndex, load_encoder, load_index

CONFIG_PATH = "config/retrieval_rerank.json"


class RerankError(ValueError):
    """A candidate, pair length, or pinned reranker is invalid."""


def load_config(root: Path = PROJECT_ROOT) -> dict:
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if (
        config.get("method") != "rerank"
        or config.get("configuration") != "w250_o50"
        or config.get("top_k") != 5
        or config.get("bm25_candidates") != 20
        or config.get("semantic_candidates") != 20
        or config.get("model_id") != "cross-encoder/ms-marco-MiniLM-L6-v2"
        or config.get("model_revision") != "233902d25c440f23af6f7d6e94d2946bac0bee0a"
        or config.get("tokenizer_revision") != config.get("model_revision")
        or config.get("max_pair_tokens") != 512
        or config.get("batch_size") != 8
    ):
        raise RerankError("Experimental reranker config differs from frozen candidate")
    return config


def merge_candidates(
    bm25: list[dict], semantic: list[dict], chunks: list[dict]
) -> list[dict]:
    """Merge by verified ID, retaining independent source ranks and raw scores."""
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise RerankError("Duplicate chunk IDs in corpus")
    union: dict[str, dict] = {}
    for method, results in (("bm25", bm25), ("semantic", semantic)):
        if len(results) > 20:
            raise RerankError(f"{method} returned over 20 candidates")
        seen: set[str] = set()
        for rank, item in enumerate(results, 1):
            cid = item.get("chunk_id")
            if cid in seen or cid not in by_id or item.get("rank") != rank:
                raise RerankError(f"{method} candidate ID or rank is invalid")
            seen.add(cid)
            chunk = by_id[cid]
            if any(
                item.get(key) != chunk.get(key)
                for key in (
                    "story_id",
                    "story_index",
                    "chunk_index",
                    "char_start",
                    "char_end",
                    "text",
                )
            ):
                raise RerankError(f"{method} candidate differs from verified source")
            score = item.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RerankError(f"{method} score is invalid")
            entry = union.setdefault(
                cid,
                {
                    **chunk,
                    "context_words": chunk["word_count"],
                    "bm25_rank": None,
                    "bm25_score": None,
                    "semantic_rank": None,
                    "semantic_score": None,
                },
            )
            entry[f"{method}_rank"] = rank
            entry[f"{method}_score"] = float(score)
    if len(union) > 40:
        raise RerankError("Candidate union exceeds 40")
    return sorted(
        union.values(),
        key=lambda x: (x["story_index"], x["chunk_index"], x["chunk_id"]),
    )


def check_pair_lengths(
    tokenizer: object, query: str, candidates: list[dict], limit: int = 512
) -> int:
    """Count full query/text pairs, including special tokens; never truncate."""
    longest = 0
    for item in candidates:
        encoded = tokenizer(
            query, item["text"], add_special_tokens=True, truncation=False
        )
        length = len(encoded["input_ids"])
        longest = max(longest, length)
        if length > limit:
            raise RerankError(
                f"Pair for {item['chunk_id']} has {length} tokens, exceeding {limit}; no input truncated"
            )
    return longest


def rank_candidates(
    candidates: list[dict], scores: list[float], top_k: int = 5
) -> list[dict]:
    if len(scores) != len(candidates) or top_k != 5:
        raise RerankError("Reranker score count or top_k is invalid")
    ranked = []
    for item, score in zip(candidates, scores, strict=True):
        if not math.isfinite(score):
            raise RerankError("Reranker score is not finite")
        ranked.append({**item, "reranker_score": float(score), "score": float(score)})
    ranked.sort(
        key=lambda x: (
            -x["reranker_score"],
            x["story_index"],
            x["chunk_index"],
            x["chunk_id"],
        )
    )
    for rank, item in enumerate(ranked, 1):
        item["reranker_rank"] = rank
        item["rank"] = rank
    return ranked[:top_k]


class CrossEncoderScorer:
    """Load one pinned model and tokenizer; prefer MPS after a real forward pass."""

    def __init__(self, config: dict):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        started = time.perf_counter()
        model_id, revision = config["model_id"], config["model_revision"]
        model_config = AutoConfig.from_pretrained(model_id, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=config["tokenizer_revision"]
        )
        self.limit = min(
            model_config.max_position_embeddings, self.tokenizer.model_max_length
        )
        if self.limit != config["max_pair_tokens"]:
            raise RerankError(
                f"Pinned effective token limit is {self.limit}, expected 512"
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision
        )
        self.model.eval()
        self.device_note = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "mps":
            try:
                self.model.to("mps")
                probe = self.tokenizer(
                    "probe", "passage", return_tensors="pt", truncation=False
                ).to("mps")
                with torch.inference_mode():
                    self.model(**probe)
            except Exception as exc:  # noqa: BLE001 - hardware fallback boundary
                self.device_note = (
                    f"MPS forward pass failed: {type(exc).__name__}: {exc}"
                )
                self.device = "cpu"
                self.model.to("cpu")
        else:
            self.device_note = "MPS unavailable"
        self.load_seconds = time.perf_counter() - started

    def score(
        self, query: str, candidates: list[dict], batch_size: int = 8
    ) -> list[float]:
        import torch

        scores = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                [item["text"] for item in batch],
                add_special_tokens=True,
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits.reshape(-1)
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores


class RerankedIndex:
    def __init__(self, bm25: object, semantic: object, scorer: object, config: dict):
        self.bm25, self.semantic, self.scorer, self.config = (
            bm25,
            semantic,
            scorer,
            config,
        )
        self.chunks = bm25.chunks
        if len(self.chunks) != len(semantic.chunks) or any(
            left != right
            for left, right in zip(self.chunks, semantic.chunks, strict=True)
        ):
            raise RerankError("BM25 and semantic chunks differ")
        self.last_audit: dict = {}

    def search_with_candidates(
        self, query: str, top_k: int = 5
    ) -> tuple[list[dict], list[dict]]:
        if top_k != 5:
            raise RerankError("Frozen reranker returns exactly top five")
        started = time.perf_counter()
        bm25 = self.bm25.search(query, self.config["bm25_candidates"])
        bm25_seconds = time.perf_counter() - started
        started = time.perf_counter()
        semantic = self.semantic.search(query, self.config["semantic_candidates"])
        semantic_seconds = time.perf_counter() - started
        union = merge_candidates(bm25, semantic, self.chunks)
        max_tokens = check_pair_lengths(
            self.scorer.tokenizer, query, union, self.scorer.limit
        )
        started = time.perf_counter()
        scores = self.scorer.score(query, union, self.config["batch_size"])
        rerank_seconds = time.perf_counter() - started
        final = rank_candidates(union, scores)
        full_order = sorted(
            zip(union, scores, strict=True),
            key=lambda pair: (
                -pair[1],
                pair[0]["story_index"],
                pair[0]["chunk_index"],
                pair[0]["chunk_id"],
            ),
        )
        all_ranks = {
            item["chunk_id"]: rank for rank, (item, _) in enumerate(full_order, 1)
        }
        # Full union is ordered by source; scores are attached independently of final rank.
        candidates = [
            {
                **item,
                "reranker_score": float(score),
                "reranker_rank": all_ranks[item["chunk_id"]],
            }
            for item, score in zip(union, scores, strict=True)
        ]
        self.last_audit = {
            "bm25_seconds": bm25_seconds,
            "semantic_seconds": semantic_seconds,
            "rerank_seconds": rerank_seconds,
            "max_pair_tokens": max_tokens,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        return final, candidates

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.search_with_candidates(query, top_k)[0]


def load_reranked(root: Path = PROJECT_ROOT) -> tuple[RerankedIndex, dict]:
    config = load_config(root)
    chunks, vectors, meta = load_index("w250_o50", root)
    encoder, encoder_note = load_encoder()
    scorer = CrossEncoderScorer(config)
    index = RerankedIndex(
        BM25Index(chunks), SemanticIndex(chunks, vectors, encoder), scorer, config
    )
    return index, {
        "index_meta": meta,
        "encoder_device": encoder.device,
        "encoder_note": encoder_note,
        "reranker_device": scorer.device,
        "reranker_device_note": scorer.device_note,
        "reranker_load_seconds": scorer.load_seconds,
    }
