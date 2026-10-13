"""Offline gold relevance and metric tests."""

from __future__ import annotations

import pytest
from test_retrieval import chunk

from sherlock_rag.evaluate import evaluate_questions, is_relevant, ranking_metrics
from sherlock_rag.retrieval import BM25Index


def test_complete_interval_required_even_with_correct_story() -> None:
    gold = {"story_id": "story-1", "char_start": 12, "char_end": 18}
    assert is_relevant({"story_id": "story-1", "char_start": 10, "char_end": 20}, gold)
    assert not is_relevant(
        {"story_id": "story-1", "char_start": 10, "char_end": 17}, gold
    )
    assert not is_relevant(
        {"story_id": "story-2", "char_start": 10, "char_end": 20}, gold
    )


def test_hit_at_k_and_mrr_at_five_known_ranks() -> None:
    metrics = ranking_metrics([1, 3, 5, None])
    assert metrics["hit_at_1"] == 0.25
    assert metrics["hit_at_3"] == 0.5
    assert metrics["hit_at_5"] == 0.75
    assert metrics["mrr_at_5"] == pytest.approx((1 + 1 / 3 + 1 / 5) / 4)


def test_unsupported_excluded_from_metrics() -> None:
    index = BM25Index([chunk("alpha", 1)])
    questions = [
        {
            "question_id": "a",
            "split": "dev",
            "kind": "direct",
            "query": "alpha",
            "answerable": True,
            "story_id": "story-1",
            "char_start": 10,
            "char_end": 15,
        },
        {
            "question_id": "b",
            "split": "dev",
            "kind": "paraphrase",
            "query": "unknown",
            "answerable": True,
            "story_id": "story-1",
            "char_start": 10,
            "char_end": 15,
        },
        {
            "question_id": "c",
            "split": "test",
            "kind": "direct",
            "query": "alpha",
            "answerable": True,
            "story_id": "story-1",
            "char_start": 10,
            "char_end": 15,
        },
        {
            "question_id": "d",
            "split": "test",
            "kind": "paraphrase",
            "query": "unknown",
            "answerable": True,
            "story_id": "story-1",
            "char_start": 10,
            "char_end": 15,
        },
        {
            "question_id": "e",
            "split": "dev",
            "kind": "unsupported",
            "query": "alpha",
            "answerable": False,
        },
    ]
    result = evaluate_questions(index, questions)
    assert result["metrics"]["dev"]["all"]["count"] == 2
    assert result["metrics"]["dev"]["all"]["hit_at_1"] == 0.5
    assert result["queries"][-1]["candidate_count_top_5"] == 1
    assert "first_relevant_rank" not in result["queries"][-1]
