"""Generate one source-traceable question from each frozen A passage."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from sherlock_rag.chunking import load_verified_source
from sherlock_rag.evaluate_rag import encode_json, project_code_hashes, utc_now
from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256
from sherlock_rag.rag import (
    LocalGenerator,
    ResponseError,
    make_passages,
    resolve_response,
)
from sherlock_rag.retrieval import load_verified_chunks

CONFIG_PATH = "config/question_generation_v1.json"
SELECTION_PATH = "questions/generated_v1/selection.json"
RUN_DIR = "questions/generated_v1"
RUN_ID = "book_questions_v1"
FRACTIONS = (1, 2, 3)


class QuestionGenerationError(ValueError):
    """The selection, model output, source, or checkpoint is invalid."""


def select_chunks(chunks: list[dict]) -> list[dict]:
    """Choose floor((n-1)*f) for f=.25,.50,.75 in each story."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk["story_index"]].append(chunk)
    if set(groups) != set(range(1, 13)):
        raise QuestionGenerationError("Expected A chunks for all 12 stories")
    selected = []
    for story in range(1, 13):
        group = groups[story]
        if len(group) < 4:
            raise QuestionGenerationError(f"Story {story} has too few chunks")
        indices = [(len(group) - 1) * fraction // 4 for fraction in FRACTIONS]
        if len(set(indices)) != 3:
            raise QuestionGenerationError(f"Story {story} selection is not unique")
        for slot, position in enumerate(indices, 1):
            chunk = group[position]
            selected.append(
                {
                    "id": f"qg-v1-{story:02d}-{slot}",
                    "story_id": chunk["story_id"],
                    "story_title": chunk["story_title"],
                    "story_index": story,
                    "selection_slot": slot,
                    "story_chunk_count": len(group),
                    "selected_position_zero_based": position,
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                }
            )
    return selected


def selection_document(chunks: list[dict], root: Path = PROJECT_ROOT) -> dict:
    return {
        "version": "story-fractions-v1",
        "configuration": "w250_o50",
        "rule": "floor((n-1)*f), f in (0.25, 0.50, 0.75), zero-based within each story",
        "chunks_sha256": sha256(
            (root / "data/processed/chunks_w250_o50.jsonl").read_bytes()
        ),
        "selected": select_chunks(chunks),
    }


def prepare(root: Path = PROJECT_ROOT) -> dict:
    chunks = load_verified_chunks("w250_o50", root)
    document = selection_document(chunks, root)
    path = root / SELECTION_PATH
    data = encode_json(document)
    if path.exists():
        if path.read_bytes() != data:
            raise QuestionGenerationError("Frozen passage selection differs")
    else:
        atomic_write(path, data)
    return document


def load_config(root: Path = PROJECT_ROOT) -> tuple[dict, str]:
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    baseline = json.loads((root / "config/generation.json").read_text(encoding="utf-8"))
    for field in (
        "model_id",
        "model_revision",
        "dtype",
        "preferred_device",
        "max_input_tokens",
        "do_sample",
    ):
        if config.get(field) != baseline.get(field):
            raise QuestionGenerationError(f"Pinned generation field differs: {field}")
    if (
        config.get("version") != "question-generation-v1"
        or config.get("max_new_tokens") != 512
        or config.get("max_repair_attempts") != 1
        or config.get("prompt_file") != "prompts/question_generation_v1.txt"
    ):
        raise QuestionGenerationError(
            "Question generation configuration differs from v1"
        )
    prompt_bytes = (root / config["prompt_file"]).read_bytes()
    if sha256(prompt_bytes) != config.get("prompt_sha256"):
        raise QuestionGenerationError("Question generation prompt hash differs")
    return config, prompt_bytes.decode("utf-8")


def build_fingerprint(root: Path = PROJECT_ROOT) -> dict:
    chunks = load_verified_chunks("w250_o50", root)
    selection = selection_document(chunks, root)
    frozen = (root / SELECTION_PATH).read_bytes()
    if frozen != encode_json(selection):
        raise QuestionGenerationError("Frozen passage selection differs")
    config, _ = load_config(root)
    book, _, source = load_verified_source(root)
    if not book:
        raise QuestionGenerationError("Verified book is empty")
    return {
        "schema_version": 1,
        "selection_sha256": sha256(frozen),
        "selected_chunk_ids": [x["chunk_id"] for x in selection["selected"]],
        "source_manifest_sha256": sha256(
            (root / "data/source_manifest.json").read_bytes()
        ),
        "raw_source_sha256": source["raw"]["sha256"],
        "processed_book_sha256": source["processed"]["sha256"],
        "chunks_sha256": selection["chunks_sha256"],
        "prompt_sha256": config["prompt_sha256"],
        "config_sha256": sha256((root / CONFIG_PATH).read_bytes()),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "max_input_tokens": config["max_input_tokens"],
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": config["do_sample"],
        "runtime_code_sha256": project_code_hashes(root),
        "versions": {
            name: metadata.version(name) for name in ("torch", "transformers")
        },
    }


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_messages(
    prompt: str, chunk: dict, book: str, repair: str = ""
) -> list[dict]:
    passage = make_passages([chunk], book)[0]
    content = f"Source [C1] | story: {chunk['story_title']} | chunk_id: {chunk['chunk_id']}\n{passage.text}"
    if repair:
        content += f"\n\nYour previous JSON failed mechanical validation: {repair}. Return one corrected full JSON object from the same passage."
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


def validate_output(raw: str, chunk: dict, book: str, source_start: int) -> dict:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise QuestionGenerationError("incomplete or invalid JSON") from exc
    fields = {"status", "question", "expected_answer", "source_id", "quote", "reason"}
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or any(not isinstance(value, str) for value in payload.values())
    ):
        raise QuestionGenerationError("JSON must contain exactly six string fields")
    status = payload["status"]
    if status == "insufficient_evidence":
        if (
            any(
                payload[key]
                for key in ("question", "expected_answer", "source_id", "quote")
            )
            or not payload["reason"].strip()
        ):
            raise QuestionGenerationError(
                "insufficient_evidence requires empty candidate fields and a reason"
            )
        return {"status": status, "reason": payload["reason"].strip()}
    if status != "candidate":
        raise QuestionGenerationError("Unknown question generation status")
    if (
        not payload["question"].strip()
        or not payload["expected_answer"].strip()
        or payload["source_id"] != "C1"
        or not payload["quote"].strip()
        or payload["reason"] != ""
    ):
        raise QuestionGenerationError(
            "Candidate requires question, answer, C1 quote and empty reason"
        )
    try:
        synthetic = json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": payload["expected_answer"],
                        "citations": [
                            {
                                "source_id": payload["source_id"],
                                "quote": payload["quote"],
                            }
                        ],
                    }
                ],
                "reason": "",
            }
        )
        resolved = resolve_response(
            synthetic, make_passages([chunk], book), book, source_start
        )
    except ResponseError as exc:
        raise QuestionGenerationError(str(exc)) from exc
    return {
        "status": "candidate",
        "question": payload["question"].strip(),
        "expected_answer": payload["expected_answer"].strip(),
        "citation": resolved["claims"][0]["citations"][0],
        "reason": "",
    }


def generate_case(
    chunk: dict,
    selected: dict,
    generator: object,
    prompt: str,
    config: dict,
    book: str,
    source_start: int,
    fingerprint: dict,
    prior_questions: dict[str, str],
) -> dict:
    item = {
        **selected,
        "status": "error",
        "reason_code": None,
        "reason": "",
        "question": "",
        "expected_answer": "",
        "citation": None,
        "duplicate_of": None,
        "attempts": 0,
        "raw_attempts": [],
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "generation_seconds_total": 0.0,
        "source_sha256": fingerprint["processed_book_sha256"],
        "prompt_sha256": fingerprint["prompt_sha256"],
        "config_sha256": fingerprint["config_sha256"],
        "runtime_fingerprint_sha256": sha256(encode_json(fingerprint)),
    }
    failure = ""
    for attempt in range(2):
        messages = prompt_messages(prompt, chunk, book, failure)
        count = generator.count_tokens(messages)
        if count > config["max_input_tokens"]:
            item.update(
                reason_code="context_budget_exceeded",
                reason=f"Full prompt has {count} tokens; limit {config['max_input_tokens']}",
            )
            return item
        raw, input_tokens, output_tokens, seconds = generator.generate(messages)
        item["attempts"] += 1
        item["input_tokens_total"] += input_tokens
        item["output_tokens_total"] += output_tokens
        item["generation_seconds_total"] += seconds
        attempt_record = {
            "raw": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": seconds,
            "validation_error": None,
        }
        item["raw_attempts"].append(attempt_record)
        try:
            parsed = validate_output(raw, chunk, book, source_start)
        except QuestionGenerationError as exc:
            failure = str(exc)
            attempt_record["validation_error"] = failure
            continue
        item.update(parsed)
        item["reason_code"] = parsed["status"]
        if parsed["status"] == "candidate":
            normalized = normalize_question(parsed["question"])
            item["duplicate_of"] = prior_questions.get(normalized)
            prior_questions.setdefault(normalized, selected["id"])
        return item
    item.update(reason_code="invalid_model_output", reason=failure)
    return item


def run(
    root: Path = PROJECT_ROOT,
    generator_factory: Callable[[dict], object] = LocalGenerator,
) -> dict:
    selection = prepare(root)
    fingerprint = build_fingerprint(root)
    config, prompt = load_config(root)
    run_dir = root / RUN_DIR
    result_path, hash_path = run_dir / "results.json", run_dir / "results.sha256"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("run_id") != RUN_ID or result.get("fingerprint") != fingerprint:
            raise QuestionGenerationError(
                "Checkpoint fingerprint differs; refusing resume"
            )
        saved = result.get("results")
        if (
            not isinstance(saved, list)
            or len(saved) > 36
            or any(
                saved[i].get("id") != selection["selected"][i]["id"]
                or saved[i].get("chunk_id") != selection["selected"][i]["chunk_id"]
                for i in range(len(saved))
            )
        ):
            raise QuestionGenerationError("Checkpoint case prefix differs")
    else:
        if hash_path.exists():
            raise QuestionGenerationError("Frozen hash exists without results")
        result = {
            "run_id": RUN_ID,
            "fingerprint": fingerprint,
            "fingerprint_sha256": sha256(encode_json(fingerprint)),
            "created_at_utc": utc_now(),
            "state": "running",
            "total": 36,
            "results": [],
            "runtime_sessions": [],
        }
    if hash_path.exists():
        if (
            sha256(result_path.read_bytes()) != hash_path.read_text().strip()
            or result["state"] != "complete"
            or len(result["results"]) != 36
        ):
            raise QuestionGenerationError("Frozen results changed or incomplete")
        return result
    if len(result["results"]) == 36:
        result["state"] = "complete"
    else:
        chunks = load_verified_chunks("w250_o50", root)
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        book, _, source = load_verified_source(root)
        source_start = source["processed"]["source_lines_inclusive"]["start"]
        prior = {
            normalize_question(item["question"]): item["id"]
            for item in result["results"]
            if item["status"] == "candidate"
        }
        generator = generator_factory(config)
        result["runtime_sessions"].append(
            {
                "started_at_utc": utc_now(),
                "model_load_seconds": generator.load_seconds,
                "device": generator.device,
                "dtype": str(generator.dtype),
            }
        )
        for selected in selection["selected"][len(result["results"]) :]:
            item = generate_case(
                by_id[selected["chunk_id"]],
                selected,
                generator,
                prompt,
                config,
                book,
                source_start,
                fingerprint,
                prior,
            )
            result["results"].append(item)
            result["updated_at_utc"] = utc_now()
            atomic_write(result_path, encode_json(result))
            print(selected["id"], item["status"], item["reason_code"], flush=True)
        result["state"] = "complete"
    result["updated_at_utc"] = utc_now()
    data = encode_json(result)
    atomic_write(result_path, data)
    atomic_write(hash_path, (sha256(data) + "\n").encode())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare()
            print(
                json.dumps(
                    {
                        "selected": len(output["selected"]),
                        "selection_path": SELECTION_PATH,
                    }
                )
            )
        else:
            output = run()
            print(
                json.dumps(
                    {
                        "state": output["state"],
                        "completed": len(output["results"]),
                        "total": output["total"],
                    }
                )
            )
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Question generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
