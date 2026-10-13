"""Run the fixed Phase 6a engineering smoke set with one local model load."""

from __future__ import annotations

import json

from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256
from sherlock_rag.rag import load_pipeline

QUESTION_IDS = ("q01", "q09", "q13", "q15", "q25", "q26", "q27", "q28")


def main() -> None:
    question_bytes = (PROJECT_ROOT / "eval/questions.jsonl").read_bytes()
    # Only IDs and query strings enter the runtime. Gold fields are never passed.
    queries = {
        row["question_id"]: row["query"]
        for row in (json.loads(line) for line in question_bytes.splitlines())
        if row["question_id"] in QUESTION_IDS
    }
    if set(queries) != set(QUESTION_IDS):
        raise ValueError("The fixed smoke IDs are missing")
    pipeline = load_pipeline()
    result = {
        "question_set_sha256": sha256(question_bytes),
        "question_ids": list(QUESTION_IDS),
        "model_load_seconds": pipeline.generator.load_seconds,
        "device": pipeline.generator.device,
        "dtype": str(pipeline.generator.dtype),
        "results": [],
    }
    for question_id in QUESTION_IDS:
        item = {
            "question_id": question_id,
            "query": queries[question_id],
            "result": pipeline.answer(queries[question_id]),
        }
        result["results"].append(item)
        output = (
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        atomic_write(PROJECT_ROOT / "eval/rag_smoke_results.json", output)
        print(
            question_id,
            item["result"]["status"],
            item["result"]["reason_code"],
            flush=True,
        )


if __name__ == "__main__":
    main()
