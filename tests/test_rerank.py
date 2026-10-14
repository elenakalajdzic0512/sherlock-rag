"""Offline reranking controls with fake rankings, scorer, and tokenizer."""

from __future__ import annotations

import json

import pytest

from sherlock_rag import evaluate_rag
from sherlock_rag.ingest import PROJECT_ROOT
from sherlock_rag.rag import RagError, load_settings
from sherlock_rag.rerank import (
    RerankedIndex,
    RerankError,
    check_pair_lengths,
    load_config,
    merge_candidates,
    rank_candidates,
)


def chunks() -> list[dict]:
    return [
        {
            "chunk_id": f"c{i}",
            "story_id": "s",
            "story_index": 1,
            "chunk_index": i,
            "char_start": i * 10,
            "char_end": i * 10 + 5,
            "text": f"text {i}",
            "word_count": 2,
        }
        for i in range(3)
    ]


def result(chunk: dict, rank: int, score: float) -> dict:
    return {**chunk, "rank": rank, "score": score}


def test_union_retains_independent_scores_and_deduplicates() -> None:
    c = chunks()
    union = merge_candidates(
        [result(c[0], 1, 2.0), result(c[1], 2, 1.0)],
        [result(c[1], 1, 0.8), result(c[2], 2, 0.7)],
        c,
    )
    assert len(union) == 3
    assert union[1]["bm25_rank"] == 2 and union[1]["semantic_rank"] == 1
    assert union[1]["bm25_score"] == 1.0 and union[1]["semantic_score"] == 0.8


def test_rank_uses_reranker_score_then_source_order_for_ties() -> None:
    c = chunks()
    ranked = rank_candidates([c[2], c[1], c[0]], [0.1, 0.9, 0.9])
    assert [item["chunk_id"] for item in ranked] == ["c0", "c1", "c2"]
    assert [item["rank"] for item in ranked] == [1, 2, 3]


class FakeTokenizer:
    def __call__(self, query, text, **kwargs):
        assert kwargs["add_special_tokens"] is True
        assert kwargs["truncation"] is False
        return {"input_ids": list(range(len(query) + len(text) + 3))}


def test_pair_limit_counts_complete_pair_and_reports_chunk() -> None:
    items = [{"chunk_id": "first", "text": "ab"}, {"chunk_id": "long", "text": "abcde"}]
    assert check_pair_lengths(FakeTokenizer(), "q", items[:1], 6) == 6
    with pytest.raises(RerankError, match="long.*9 tokens.*6"):
        check_pair_lengths(FakeTokenizer(), "q", items, 6)


class Search:
    def __init__(self, chunks_, order):
        self.chunks = chunks_
        self.order = order

    def search(self, query, top_k):
        return [
            result(self.chunks[i], rank, float(top_k - rank))
            for rank, i in enumerate(self.order[:top_k], 1)
        ]


class Scorer:
    tokenizer = FakeTokenizer()
    limit = 512

    def score(self, query, candidates, batch_size):
        return [float(item["chunk_index"]) for item in candidates]


def test_index_keeps_all_candidates_and_final_ranks() -> None:
    c = chunks()
    config = {"bm25_candidates": 20, "semantic_candidates": 20, "batch_size": 8}
    index = RerankedIndex(Search(c, [0, 1]), Search(c, [1, 2]), Scorer(), config)
    final, union = index.search_with_candidates("q")
    assert [x["chunk_id"] for x in final] == ["c2", "c1", "c0"]
    assert len(union) == 3 and [x["reranker_rank"] for x in union] == [3, 2, 1]


def test_explicit_selection_preserves_default(tmp_path) -> None:
    config = tmp_path / "config"
    prompt = tmp_path / "prompts/p.txt"
    config.mkdir()
    prompt.parent.mkdir()
    prompt.write_text("prompt")
    from sherlock_rag.ingest import sha256

    generation = {
        "prompt_file": "prompts/p.txt",
        "prompt_sha256": sha256(b"prompt"),
        "max_input_tokens": 10,
        "max_new_tokens": 10,
    }
    (config / "generation.json").write_text(json.dumps(generation))
    (config / "retrieval.json").write_text(
        json.dumps({"method": "semantic", "configuration": "w250_o50", "top_k": 5})
    )
    assert load_settings(tmp_path)[0]["method"] == "semantic"
    assert (
        load_settings(PROJECT_ROOT, retrieval_variant="rerank")[0]["method"] == "rerank"
    )
    with pytest.raises(RagError, match="Unknown retrieval variant"):
        load_settings(tmp_path, retrieval_variant="unknown")


def test_frozen_candidate_config_rejects_parameter_change(tmp_path) -> None:
    source = PROJECT_ROOT / "config/retrieval_rerank.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "retrieval_rerank.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_config(tmp_path) == config
    config["bm25_candidates"] = 19
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RerankError, match="frozen candidate"):
        load_config(tmp_path)


def test_rerank_fingerprint_includes_variant_and_model(monkeypatch, tmp_path) -> None:
    from test_evaluate_rag import _fingerprint_tree

    _fingerprint_tree(tmp_path, monkeypatch)
    (tmp_path / "config/retrieval_rerank.json").write_text('{"method":"rerank"}')
    original_settings = evaluate_rag.load_settings

    def settings(root, generation_variant="v1", retrieval_variant="default"):
        retrieval, generation, prompt = original_settings(root, generation_variant)
        if retrieval_variant == "rerank":
            retrieval = {
                **retrieval,
                "method": "rerank",
                "model_id": "model",
                "model_revision": "rev",
                "tokenizer_revision": "rev",
                "bm25_candidates": 20,
                "semantic_candidates": 20,
            }
        return retrieval, generation, prompt

    monkeypatch.setattr(evaluate_rag, "load_settings", settings)
    one = evaluate_rag.build_fingerprint(tmp_path, retrieval_variant="rerank")
    assert (
        one["retrieval_variant"] == "rerank" and one["reranker_model_revision"] == "rev"
    )
    assert one == evaluate_rag.build_fingerprint(tmp_path, retrieval_variant="rerank")
    (tmp_path / "config/retrieval_rerank.json").write_text("changed")
    assert one != evaluate_rag.build_fingerprint(tmp_path, retrieval_variant="rerank")


def test_rerank_review_rejects_inconsistent_identical_evidence(tmp_path) -> None:
    from eval.rerank_paired_review import RerankReviewError, compare

    source = PROJECT_ROOT / "eval/paired_reviews/rerank_v1_rubric_v1.json"
    assert compare(PROJECT_ROOT, source)["identical_claim_evidence_groups"] >= 1
    review = json.loads(source.read_text(encoding="utf-8"))
    row = next(row for row in review["ratings"] if row["id"] == "q02")
    row["claims"][0]["supported"] = False
    changed = tmp_path / "inconsistent.json"
    changed.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(RerankReviewError, match="Identical claim/evidence"):
        compare(PROJECT_ROOT, changed)
