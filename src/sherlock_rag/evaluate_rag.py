"""Checkpoint and fingerprint full local answer evaluation without passing gold to RAG."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from sherlock_rag.chunking import load_verified_source
from sherlock_rag.ingest import PROJECT_ROOT, atomic_write, sha256
from sherlock_rag.rag import GENERATION_CONFIGS, load_pipeline, load_settings
from sherlock_rag.semantic import load_index

QUESTIONS_SHA256 = "fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b"
CHALLENGES_SHA256 = "cdf59802eeeddde1b3d0430651d999abcc51f5f1186a36b44a8a8bd0fdafe006"
FINGERPRINT_SCHEMA_VERSION = 2
REQUIRED_RUNTIME_MODULES = frozenset(
    {
        "__init__.py",
        "chunking.py",
        "evaluate_rag.py",
        "hybrid.py",
        "ingest.py",
        "rag.py",
        "retrieval.py",
        "semantic.py",
    }
)
EXCLUDED_CODE_DIRECTORIES = frozenset(
    {"__pycache__", ".cache", "build", "dist", "generated", ".generated"}
)


class RunError(ValueError):
    """An evaluation input, checkpoint, or fingerprint is invalid."""


def encode_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_cases(root: Path = PROJECT_ROOT) -> tuple[list[dict], dict[str, str]]:
    paths = {
        "questions": root / "eval/questions.jsonl",
        "challenges": root / "eval/answer_challenges.jsonl",
    }
    data = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: sha256(blob) for name, blob in data.items()}
    if hashes != {"questions": QUESTIONS_SHA256, "challenges": CHALLENGES_SHA256}:
        raise RunError("Frozen question or challenge hash differs")
    questions = [json.loads(line) for line in data["questions"].splitlines()]
    challenges = [json.loads(line) for line in data["challenges"].splitlines()]
    if len(questions) != 30 or len(challenges) != 12:
        raise RunError("Expected 30 original and 12 challenge questions")
    cases = [
        {
            "id": q["question_id"],
            "dataset": "original",
            "category": q["kind"],
            "query": q["query"],
        }
        for q in questions
    ] + [
        {
            "id": q["challenge_id"],
            "dataset": "challenge",
            "category": q["category"],
            "query": q["query"],
        }
        for q in challenges
    ]
    if len({case["id"] for case in cases}) != 42 or any(
        not case["query"].strip() for case in cases
    ):
        raise RunError("Question IDs or query text are invalid")
    return cases, hashes


def project_code_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Hash package source, including future modules, but never generated files."""
    package = root / "src/sherlock_rag"
    paths = sorted(
        path
        for path in package.rglob("*.py")
        if path.is_file()
        and not path.is_symlink()
        and not EXCLUDED_CODE_DIRECTORIES.intersection(path.relative_to(package).parts)
    )
    present = {path.name for path in paths if path.parent == package}
    if not REQUIRED_RUNTIME_MODULES <= present:
        missing = sorted(REQUIRED_RUNTIME_MODULES - present)
        raise RunError(f"Runtime source modules are missing: {missing}")
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()) for path in paths
    }


def build_fingerprint(
    root: Path = PROJECT_ROOT, generation_variant: str = "v1"
) -> dict:
    cases, hashes = load_cases(root)
    retrieval, generation, _ = load_settings(root, generation_variant)
    book, _, source = load_verified_source(root)
    config = retrieval["configuration"]
    if retrieval["method"] != "semantic":
        raise RunError("This baseline expects the frozen semantic retrieval selection")
    chunks, vectors, meta = load_index(config, root)
    del vectors
    if not cases or not chunks or not book:
        raise RunError("Empty evaluated corpus or question set")
    versions = {
        name: metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "sentence-transformers",
            "numpy",
            "accelerate",
            "safetensors",
        )
    }
    fingerprint = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "generation_variant": generation_variant,
        "generation_config_file": GENERATION_CONFIGS[generation_variant],
        "project_code_files_sha256": project_code_hashes(root),
        "dataset_sha256": hashes,
        "retrieval_config_sha256": sha256(
            (root / "config/retrieval.json").read_bytes()
        ),
        "generation_config_sha256": sha256(
            (root / GENERATION_CONFIGS[generation_variant]).read_bytes()
        ),
        "prompt_sha256": generation["prompt_sha256"],
        "source_manifest_sha256": sha256(
            (root / "data/source_manifest.json").read_bytes()
        ),
        "raw_source_sha256": source["raw"]["sha256"],
        "processed_book_sha256": source["processed"]["sha256"],
        "chunk_count": len(chunks),
        "chunks_sha256": meta["chunks_sha256"],
        "index_meta_sha256": sha256((root / f"indexes/{config}.json").read_bytes()),
        "vectors_sha256": meta["vectors_sha256"],
        "method": retrieval["method"],
        "configuration": config,
        "top_k": retrieval["top_k"],
        "model_id": generation["model_id"],
        "model_revision": generation["model_revision"],
        "prompt_version": generation["prompt_version"],
        "device": generation["preferred_device"],
        "dtype": generation["dtype"],
        "decoding": {
            "do_sample": generation["do_sample"],
            "max_new_tokens": generation["max_new_tokens"],
        },
        "max_input_tokens": generation["max_input_tokens"],
        "versions": versions,
    }
    return fingerprint


def run_cases(
    run_dir: Path,
    run_id: str,
    fingerprint: dict,
    cases: list[dict],
    pipeline_factory: Callable[[], object],
) -> dict:
    """Resume only a matching prefix; never rerun or select among saved answers."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise RunError("run_id must be a short alphanumeric slug")
    result_path = run_dir / "results.json"
    hash_path = run_dir / "results.sha256"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("run_id") != run_id or result.get("fingerprint") != fingerprint:
            raise RunError("Checkpoint fingerprint or run_id differs; refusing resume")
        saved = result.get("results")
        if not isinstance(saved, list) or len(saved) > len(cases):
            raise RunError("Checkpoint result count is invalid")
        for index, item in enumerate(saved):
            if any(
                item.get(key) != cases[index][key]
                for key in ("id", "dataset", "category", "query")
            ):
                raise RunError("Checkpoint cases are not the expected prefix")
    else:
        if hash_path.exists():
            raise RunError("Result hash exists without results.json")
        result = {
            "run_id": run_id,
            "fingerprint": fingerprint,
            "fingerprint_sha256": sha256(encode_json(fingerprint)),
            "created_at_utc": utc_now(),
            "state": "running",
            "total": len(cases),
            "results": [],
            "runtime_sessions": [],
        }
    if hash_path.exists():
        if sha256(result_path.read_bytes()) != hash_path.read_text().strip():
            raise RunError("Frozen result hash differs")
        if len(result["results"]) != len(cases) or result["state"] != "complete":
            raise RunError("Frozen result is incomplete")
        return result
    if len(result["results"]) < len(cases):
        pipeline = pipeline_factory()
        generator = pipeline.generator
        if (
            generator.device != fingerprint["device"]
            or str(generator.dtype).split(".")[-1] != fingerprint["dtype"]
        ):
            raise RunError("Runtime device or dtype differs from fingerprint")
        result["runtime_sessions"].append(
            {
                "started_at_utc": utc_now(),
                "model_load_seconds": generator.load_seconds,
                "device": generator.device,
                "dtype": str(generator.dtype),
            }
        )
        for case in cases[len(result["results"]) :]:
            # The RAG call receives query only; gold and category are never arguments.
            try:
                answer = pipeline.answer(case["query"])
            except Exception as exc:  # noqa: BLE001 - record unexpected technical errors
                answer = {
                    "status": "error",
                    "reason_code": "runner_error",
                    "reason": str(exc),
                }
            result["results"].append({**case, "result": answer})
            result["updated_at_utc"] = utc_now()
            atomic_write(result_path, encode_json(result))
            print(case["id"], answer["status"], answer["reason_code"], flush=True)
    result["state"] = "complete"
    result["updated_at_utc"] = utc_now()
    output = encode_json(result)
    atomic_write(result_path, output)
    atomic_write(hash_path, (sha256(output) + "\n").encode())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--generation-variant", choices=tuple(GENERATION_CONFIGS), default="v1"
    )
    args = parser.parse_args(argv)
    try:
        cases, _ = load_cases()
        fingerprint = build_fingerprint(generation_variant=args.generation_variant)
        run_dir = PROJECT_ROOT / "eval/rag_runs" / args.run_id
        result = run_cases(
            run_dir,
            args.run_id,
            fingerprint,
            cases,
            lambda: load_pipeline(generation_variant=args.generation_variant),
        )
    except (RunError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"RAG evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "state": result["state"],
                "completed": len(result["results"]),
                "total": result["total"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
