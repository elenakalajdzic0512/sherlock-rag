"""Validate fixed source-grounded questions and evaluate both BM25 chunk sets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sherlock_rag.chunking import (
    line_number,
    line_starts,
    load_verified_source,
    story_spans,
)
from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256
from sherlock_rag.retrieval import (
    CONFIGURATIONS,
    K1,
    B,
    BM25Index,
    load_verified_chunks,
)


class EvaluationError(ValueError):
    """The gold set or its provenance is invalid."""


def load_questions(root: Path = PROJECT_ROOT) -> tuple[list[dict], str]:
    book, story_lines, source = load_verified_source(root)
    spans = story_spans(book, story_lines)
    starts = line_starts(book)
    path = root / "eval/questions.jsonl"
    data = path.read_bytes()
    questions = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    if len(questions) != 30 or len({q.get("question_id") for q in questions}) != 30:
        raise EvaluationError("Require 30 uniquely identified questions")
    counts = Counter((q.get("split"), q.get("kind")) for q in questions)
    if counts != Counter(
        {
            ("dev", "direct"): 6,
            ("dev", "paraphrase"): 6,
            ("test", "direct"): 6,
            ("test", "paraphrase"): 6,
            ("dev", "unsupported"): 3,
            ("test", "unsupported"): 3,
        }
    ):
        raise EvaluationError("Question split/kind counts are incorrect")
    for split in ("dev", "test"):
        stories = [
            q.get("story_id")
            for q in questions
            if q.get("split") == split and q.get("answerable") is True
        ]
        if (
            set(stories) != {f"gutenberg-1661-story-{i:02d}" for i in range(1, 13)}
            or len(stories) != 12
        ):
            raise EvaluationError(f"{split} must cover each story exactly once")
    source_start = source["processed"]["source_lines_inclusive"]["start"]
    for q in questions:
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise EvaluationError("Every query must be nonempty")
        if q["kind"] == "unsupported":
            if q.get("answerable") is not False or not q.get("reason"):
                raise EvaluationError(
                    "Unsupported question needs a reason and answerable=false"
                )
            if any(
                field in q
                for field in (
                    "expected_answer",
                    "story_id",
                    "evidence_quote",
                    "char_start",
                    "char_end",
                )
            ):
                raise EvaluationError(
                    "Unsupported question must not have gold evidence"
                )
            continue
        if q.get("answerable") is not True or not q.get("expected_answer"):
            raise EvaluationError("Answerable question lacks an expected answer")
        story_index = int(q["story_id"].rsplit("-", 1)[1])
        _, _, story_a, story_b = spans[story_index - 1]
        a, b = q["char_start"], q["char_end"]
        if not story_a <= a < b <= story_b or book[a:b] != q["evidence_quote"]:
            raise EvaluationError(f"Gold evidence mismatch for {q['question_id']}")
        p_start, p_end = line_number(starts, a), line_number(starts, b - 1)
        if (
            q["processed_line_start"],
            q["processed_line_end"],
            q["source_line_start"],
            q["source_line_end"],
            q["processed_sha256"],
        ) != (
            p_start,
            p_end,
            source_start + p_start - 1,
            source_start + p_end - 1,
            source["processed"]["sha256"],
        ):
            raise EvaluationError(f"Gold provenance mismatch for {q['question_id']}")
    return questions, sha256(data)


def is_relevant(result: dict, question: dict) -> bool:
    return (
        result["story_id"] == question["story_id"]
        and result["char_start"] <= question["char_start"]
        and result["char_end"] >= question["char_end"]
    )


def ranking_metrics(ranks: list[int | None]) -> dict:
    if not ranks:
        raise EvaluationError("Cannot compute metrics for an empty set")
    return {
        "count": len(ranks),
        "hit_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None and rank <= 5)
        / len(ranks),
    }


def evaluate_questions(index: BM25Index, questions: list[dict]) -> dict:
    """Pass only each query to search; gold never modifies the index or ranking."""
    details = []
    for q in questions:
        results = index.search(q["query"], 5)
        detail = {
            "question_id": q["question_id"],
            "split": q["split"],
            "kind": q["kind"],
            "candidate_count_top_5": len(results),
            "top_5_context_words": sum(r["context_words"] for r in results),
            "top_5": [
                {
                    "rank": r["rank"],
                    "score": r["score"],
                    "chunk_id": r["chunk_id"],
                    "story_id": r["story_id"],
                    "char_start": r["char_start"],
                    "char_end": r["char_end"],
                    **({"relevant": is_relevant(r, q)} if q["answerable"] else {}),
                }
                for r in results
            ],
        }
        if q["answerable"]:
            detail["first_relevant_rank"] = next(
                (r["rank"] for r in results if is_relevant(r, q)), None
            )
            detail["gold_interval"] = [q["char_start"], q["char_end"]]
            detail["gold_story_id"] = q["story_id"]
        details.append(detail)
    metrics = {}
    for split in ("dev", "test"):
        items = [
            d for d in details if d["split"] == split and d["kind"] != "unsupported"
        ]
        metrics[split] = {
            "all": ranking_metrics([d["first_relevant_rank"] for d in items]),
            "direct": ranking_metrics(
                [d["first_relevant_rank"] for d in items if d["kind"] == "direct"]
            ),
            "paraphrase": ranking_metrics(
                [d["first_relevant_rank"] for d in items if d["kind"] == "paraphrase"]
            ),
            "average_top_5_context_words": sum(d["top_5_context_words"] for d in items)
            / len(items),
        }
    return {"metrics": metrics, "queries": details}


def run_evaluation(root: Path = PROJECT_ROOT) -> dict:
    questions, question_sha = load_questions(root)
    source = json.loads(
        (root / "data/source_manifest.json").read_text(encoding="utf-8")
    )
    chunk_manifest_bytes = (root / "data/chunk_manifest.json").read_bytes()
    chunk_manifest = json.loads(chunk_manifest_bytes)
    result = {
        "method": "bm25-positive-idf-v1",
        "parameters": {"k1": K1, "b": B, "top_k": 5, "query_terms": "unique"},
        "input_sha256": {
            "questions_jsonl": question_sha,
            "source_manifest_json": sha256(
                (root / "data/source_manifest.json").read_bytes()
            ),
            "processed_book": source["processed"]["sha256"],
            "chunk_manifest_json": sha256(chunk_manifest_bytes),
        },
        "configurations": {},
    }
    for config in CONFIGURATIONS:
        chunks = load_verified_chunks(config, root)
        result["configurations"][config] = {
            "chunk_count": len(chunks),
            "chunks_sha256": chunk_manifest["configurations"][config]["sha256"],
            **evaluate_questions(BM25Index(chunks), questions),
        }
    output = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    atomic_write(root / "eval/bm25_results.json", output)
    return result


def run_semantic_evaluation(
    root: Path = PROJECT_ROOT, encoder: object | None = None
) -> dict:
    """Evaluate saved semantic document matrices with unchanged gold rules."""
    from sherlock_rag.semantic import (
        DIMENSION,
        INDEX_VERSION,
        MAX_SEQ_LENGTH,
        MODEL_ID,
        MODEL_REVISION,
        QUERY_PREFIX,
        SemanticIndex,
        load_encoder,
        load_index,
    )

    questions, question_sha = load_questions(root)
    if encoder is None:
        encoder, device_note = load_encoder()
    else:
        device_note = None
    max_query_tokens = encoder.measure_queries([q["query"] for q in questions])
    source_bytes = (root / "data/source_manifest.json").read_bytes()
    source = json.loads(source_bytes)
    chunk_manifest_bytes = (root / "data/chunk_manifest.json").read_bytes()
    result = {
        "method": INDEX_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": {
            "top_k": 5,
            "score": "exact cosine dot product",
            "query_prefix": QUERY_PREFIX,
            "dimension": DIMENSION,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
        "evaluation_device": encoder.device,
        "evaluation_device_note": device_note,
        "max_query_tokens": max_query_tokens,
        "zero_truncated_queries": True,
        "input_sha256": {
            "questions_jsonl": question_sha,
            "source_manifest_json": sha256(source_bytes),
            "processed_book": source["processed"]["sha256"],
            "chunk_manifest_json": sha256(chunk_manifest_bytes),
        },
        "configurations": {},
    }
    for config in CONFIGURATIONS:
        chunks, vectors, meta = load_index(config, root)
        index = SemanticIndex(chunks, vectors, encoder)
        vector_path = root / f"indexes/{config}.npy"
        result["configurations"][config] = {
            "chunk_count": len(chunks),
            "chunks_sha256": meta["chunks_sha256"],
            "vectors_sha256": sha256(vector_path.read_bytes()),
            "max_document_tokens": meta["max_document_tokens"],
            "build_seconds": meta["build_seconds"],
            **evaluate_questions(index, questions),
        }
    output = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    atomic_write(root / "eval/semantic_results.json", output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bm25", "semantic"), default="bm25")
    args = parser.parse_args(argv)
    try:
        result = (
            run_evaluation() if args.method == "bm25" else run_semantic_evaluation()
        )
    except (EvaluationError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "input_sha256": result["input_sha256"],
                "metrics": {
                    k: v["metrics"] for k, v in result["configurations"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
