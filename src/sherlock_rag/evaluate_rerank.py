"""Measure the one frozen reranker candidate on the original reference set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sherlock_rag.evaluate import is_relevant, load_questions, ranking_metrics
from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256
from sherlock_rag.rerank import CONFIG_PATH, load_reranked


def summarize(details: list[dict], split: str) -> dict:
    items = [d for d in details if d["split"] == split and d["answerable"]]
    return {
        **ranking_metrics([d["first_relevant_rank"] for d in items]),
        "union_hit": sum(d["union_hit"] for d in items),
        "final_hit": sum(d["first_relevant_rank"] is not None for d in items),
        "mean_context_words": sum(d["context_words"] for d in items) / len(items),
        "mean_bm25_seconds": sum(d["timing"]["bm25_seconds"] for d in items)
        / len(items),
        "mean_semantic_seconds": sum(d["timing"]["semantic_seconds"] for d in items)
        / len(items),
        "mean_rerank_seconds": sum(d["timing"]["rerank_seconds"] for d in items)
        / len(items),
    }


def run(root: Path = PROJECT_ROOT) -> dict:
    questions, question_sha = load_questions(root)
    index, runtime = load_reranked(root)
    details = []
    for question in questions:
        final, candidates = index.search_with_candidates(question["query"])
        audit = index.last_audit
        detail = {
            "question_id": question["question_id"],
            "split": question["split"],
            "kind": question["kind"],
            "answerable": question["answerable"],
            "query": question["query"],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "top_5": final,
            "context_words": sum(item["context_words"] for item in final),
            "timing": {
                key: audit[key]
                for key in ("bm25_seconds", "semantic_seconds", "rerank_seconds")
            },
            "max_pair_tokens": audit["max_pair_tokens"],
        }
        if question["answerable"]:
            detail.update(
                {
                    "gold_story_id": question["story_id"],
                    "gold_interval": [question["char_start"], question["char_end"]],
                    "union_hit": any(
                        is_relevant(item, question) for item in candidates
                    ),
                    "first_relevant_rank": next(
                        (item["rank"] for item in final if is_relevant(item, question)),
                        None,
                    ),
                }
            )
        details.append(detail)
    baseline = json.loads(
        (root / "eval/semantic_results.json").read_text(encoding="utf-8")
    )
    prior = baseline["configurations"]["w250_o50"]["metrics"]
    metrics = {split: summarize(details, split) for split in ("dev", "test")}
    dev = metrics["dev"]
    old = prior["dev"]["all"]
    passed = dev["hit_at_5"] > old["hit_at_5"] or (
        dev["hit_at_5"] == old["hit_at_5"] and dev["mrr_at_5"] > old["mrr_at_5"]
    )
    output = {
        "protocol": "docs/rerank-protocol.md",
        "config_sha256": sha256((root / CONFIG_PATH).read_bytes()),
        "questions_sha256": question_sha,
        "semantic_results_sha256": sha256(
            (root / "eval/semantic_results.json").read_bytes()
        ),
        "runtime": runtime,
        "metrics": metrics,
        "baseline_metrics": {split: prior[split]["all"] for split in ("dev", "test")},
        "dev_gate_passed": passed,
        "queries": details,
    }
    atomic_write(
        root / "eval/rerank_results.json",
        (
            json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return output


def main() -> int:
    try:
        output = run()
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Rerank evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "dev_gate_passed": output["dev_gate_passed"],
                "runtime": {
                    k: v for k, v in output["runtime"].items() if k != "index_meta"
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
