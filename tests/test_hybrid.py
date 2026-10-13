"""Offline RRF and evaluation tests with synthetic chunks and fake rankers."""

from __future__ import annotations

import copy

import pytest
from test_retrieval import chunk

from sherlock_rag.evaluate import evaluate_questions
from sherlock_rag.hybrid import HybridError, HybridIndex


def corpus(n: int = 8) -> list[dict]:
    chunks = [chunk(f"text {i}", i) for i in range(1, n + 1)]
    for item in chunks:
        item.update(
            raw_sha256="raw",
            processed_sha256="processed",
            chunk_size_words=250,
            overlap_words=50,
        )
    return chunks


class FakeRanker:
    def __init__(self, chunks: list[dict], ids: list[int]):
        self.chunks = chunks
        self.ids = ids

    def search(self, query: str, top_k: int) -> list[dict]:
        if not query.strip():
            return []
        by_id = {c["chunk_id"]: c for c in self.chunks}
        return [
            {
                **by_id[f"id-{i}"],
                "rank": rank,
                "score": float(10 - rank),
                "context_words": by_id[f"id-{i}"]["word_count"],
            }
            for rank, i in enumerate(self.ids[:top_k], 1)
        ]


def make_index(bm: list[int], sem: list[int], n: int = 8, **kwargs) -> HybridIndex:
    chunks = corpus(n)
    return HybridIndex(
        FakeRanker(chunks, bm), FakeRanker(copy.deepcopy(chunks), sem), **kwargs
    )


def test_hand_calculated_rrf_union_and_contributions() -> None:
    index = make_index([1, 2], [2, 3], candidate_k=2, rrf_k=60)
    results, candidates = index.search_with_candidates("clue", 2)
    assert [r["chunk_id"] for r in results] == ["id-2", "id-1"]
    assert {c["chunk_id"] for c in candidates} == {"id-1", "id-2", "id-3"}
    both, single = results
    assert both["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert both["bm25_rank"] == 2 and both["semantic_rank"] == 1
    assert both["bm25_score"] == 8.0 and both["semantic_score"] == 9.0
    assert both["bm25_rrf_contribution"] == pytest.approx(1 / 62)
    assert both["semantic_rrf_contribution"] == pytest.approx(1 / 61)
    assert single["semantic_rank"] is None and single["semantic_score"] is None
    assert single["semantic_rrf_contribution"] == 0
    assert both["rank"] == 1 and single["rank"] == 2
    assert both["score_type"] == "equal-weight-rrf-v1"


def test_empty_lexical_both_empty_and_blank_query() -> None:
    assert [r["chunk_id"] for r in make_index([], [2]).search("q", 1)] == ["id-2"]
    assert make_index([], []).search("q", 1) == []
    assert make_index([1], [2]).search("   ", 1) == []


def test_stable_ties_follow_story_then_chunk() -> None:
    index = make_index([3], [1], candidate_k=2)
    assert [r["chunk_id"] for r in index.search("q", 2)] == ["id-1", "id-3"]


@pytest.mark.parametrize(
    "candidate_k,rrf_k,top_k",
    [
        (0, 60, 1),
        (True, 60, 1),
        (2.0, 60, 1),
        (2, 0, 1),
        (2, False, 1),
        (2, 60, 0),
        (2, 60, True),
        (1, 60, 2),
    ],
)
def test_invalid_parameters(candidate_k, rrf_k, top_k) -> None:
    with pytest.raises(HybridError):
        make_index([1], [1], candidate_k=candidate_k, rrf_k=rrf_k).search("q", top_k)


def test_source_and_candidate_mismatch_rejected() -> None:
    chunks = corpus()
    other = copy.deepcopy(chunks)
    other[0]["processed_sha256"] = "wrong"
    with pytest.raises(HybridError, match="source"):
        HybridIndex(FakeRanker(chunks, [1]), FakeRanker(other, [1]))
    index = make_index([1, 1], [2], candidate_k=2)
    with pytest.raises(HybridError, match="duplicate"):
        index.search("q", 1)
    index = make_index([1], [2])
    original_search = index.bm25.search

    def unknown(query: str, top_k: int) -> list[dict]:
        result = original_search(query, top_k)
        result[0]["chunk_id"] = "alien"
        return result

    index.bm25.search = unknown
    with pytest.raises(HybridError, match="unknown"):
        index.search("q", 1)


def test_union_hit_can_be_fusion_top_five_miss_and_unsupported_excluded() -> None:
    # id-7 is rank 7 in one candidate list; five shared IDs outrank it.
    index = make_index([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 8, 6], candidate_k=7)
    questions = []
    for split in ("dev", "test"):
        for kind in ("direct", "paraphrase"):
            questions.append(
                {
                    "question_id": f"{split}-{kind}",
                    "split": split,
                    "kind": kind,
                    "query": "clue",
                    "answerable": True,
                    "story_id": "story-1",
                    "char_start": 70,
                    "char_end": 72,
                }
            )
        questions.append(
            {
                "question_id": f"{split}-unsupported",
                "split": split,
                "kind": "unsupported",
                "query": "clue",
                "answerable": False,
            }
        )
    report = evaluate_questions(index, questions)
    assert report["metrics"]["dev"]["all"]["candidate_union_hit_rate"] == 1
    assert report["metrics"]["dev"]["all"]["hit_at_5"] == 0
    assert report["metrics"]["dev"]["all"]["count"] == 2
    assert report["queries"][0]["candidate_union_hit"] is True
    assert report["queries"][0]["first_relevant_rank"] is None
    assert "candidate_union_hit" not in report["queries"][2]
    assert "first_relevant_rank" not in report["queries"][2]
