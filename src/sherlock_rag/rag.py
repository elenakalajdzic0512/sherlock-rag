"""Local evidence-grounded answers with mechanically resolved source citations."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sherlock_rag.chunking import line_number, line_starts, load_verified_source
from sherlock_rag.ingest import PROJECT_ROOT, sha256


class RagError(ValueError):
    """Invalid configuration, user input, or model response."""


class ResponseError(RagError):
    """Generated JSON or a cited quote failed mechanical validation."""


class RetrievalSetupError(RagError):
    """Selected verified source, chunks, or index could not be loaded."""


@dataclass
class Passage:
    source_id: str
    chunk: dict
    text: str
    raw_spans: list[tuple[int, int]]


def normalize_with_spans(raw: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse only whitespace, retaining each displayed character's raw span."""
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\s+|\S+", raw):
        if match.group().isspace():
            chars.append(" ")
            spans.append(match.span())
        else:
            for offset, char in enumerate(match.group(), match.start()):
                chars.append(char)
                spans.append((offset, offset + 1))
    return "".join(chars), spans


def make_passages(results: list[dict], book: str) -> list[Passage]:
    passages = []
    for number, chunk in enumerate(results, 1):
        start, end = chunk["char_start"], chunk["char_end"]
        if not 0 <= start < end <= len(book) or book[start:end] != chunk["text"]:
            raise RagError(f"Retrieved chunk {chunk['chunk_id']} differs from book.txt")
        normalized, spans = normalize_with_spans(chunk["text"])
        passages.append(Passage(f"C{number}", chunk, normalized, spans))
    return passages


def prompt_messages(
    prompt: str, query: str, passages: list[Passage], repair: str = ""
) -> list[dict]:
    blocks = [
        f"[{p.source_id}] chunk_id={p.chunk['chunk_id']} story={p.chunk['story_title']} "
        f"source_lines={p.chunk['source_line_start']}-{p.chunk['source_line_end']}\n{p.text}"
        for p in passages
    ]
    evidence = "\n\n".join(blocks) if blocks else "(no passages)"
    content = f"Question: {query}\n\nSource passages:\n{evidence}"
    if repair:
        content += f"\n\nYour previous response failed mechanical validation: {repair}. Return a corrected complete JSON object using these same passages."
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


def resolve_response(
    raw: str, passages: list[Passage], book: str, source_start: int
) -> dict:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResponseError("incomplete or invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"status", "claims", "reason"}:
        raise ResponseError("response must have status, claims, and reason only")
    status, claims, reason = payload["status"], payload["claims"], payload["reason"]
    if (
        status not in ("answered", "abstained")
        or not isinstance(claims, list)
        or not isinstance(reason, str)
    ):
        raise ResponseError("invalid status, claims, or reason type")
    if status == "abstained":
        if claims or not reason.strip():
            raise ResponseError("abstained requires no claims and a reason")
        return {
            "status": status,
            "answer": "",
            "claims": [],
            "reason": reason.strip(),
            "citations_resolved": True,
        }
    if not claims or reason.strip():
        raise ResponseError("answered requires claims and an empty reason")
    by_id = {p.source_id: p for p in passages}
    starts = line_starts(book)
    resolved = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"text", "citations"}:
            raise ResponseError("invalid claim structure")
        if (
            not isinstance(claim["text"], str)
            or not claim["text"].strip()
            or not isinstance(claim["citations"], list)
            or not claim["citations"]
        ):
            raise ResponseError("every claim needs text and citations")
        citations = []
        for citation in claim["citations"]:
            if not isinstance(citation, dict) or set(citation) != {
                "source_id",
                "quote",
            }:
                raise ResponseError("invalid citation structure")
            source_id, quote = citation["source_id"], citation["quote"]
            if not isinstance(source_id, str) or source_id not in by_id:
                raise ResponseError("unknown source_id")
            if not isinstance(quote, str) or not quote.strip():
                raise ResponseError("empty quote")
            passage = by_id[source_id]
            local = passage.text.find(quote)  # First occurrence if repeated.
            if local < 0:
                raise ResponseError("quote is not in the displayed passage")
            raw_start = passage.chunk["char_start"] + passage.raw_spans[local][0]
            raw_end = (
                passage.chunk["char_start"]
                + passage.raw_spans[local + len(quote) - 1][1]
            )
            quote_raw = book[raw_start:raw_end]
            if normalize_with_spans(quote_raw)[0] != quote:
                raise ResponseError("quote cannot be mapped to original text")
            first_line = line_number(starts, raw_start)
            last_line = line_number(starts, raw_end - 1)
            citations.append(
                {
                    "source_id": source_id,
                    "chunk_id": passage.chunk["chunk_id"],
                    "quote": quote,
                    "quote_raw": quote_raw,
                    "char_start": raw_start,
                    "char_end": raw_end,
                    "source_line_start": source_start + first_line - 1,
                    "source_line_end": source_start + last_line - 1,
                    "story_id": passage.chunk["story_id"],
                    "story_title": passage.chunk["story_title"],
                }
            )
        resolved.append({"text": claim["text"].strip(), "citations": citations})
    rendered = " ".join(
        f"{claim['text']} {' '.join('[' + c['source_id'] + ']' for c in claim['citations'])}"
        for claim in resolved
    )
    return {
        "status": status,
        "answer": rendered,
        "claims": resolved,
        "reason": "",
        "citations_resolved": True,
    }


def load_settings(root: Path = PROJECT_ROOT) -> tuple[dict, dict, str]:
    retrieval = json.loads((root / "config/retrieval.json").read_text(encoding="utf-8"))
    generation = json.loads(
        (root / "config/generation.json").read_text(encoding="utf-8")
    )
    if retrieval.get("method") not in ("bm25", "semantic", "hybrid") or retrieval.get(
        "configuration"
    ) not in ("w250_o50", "w500_o100"):
        raise RagError("Unsupported retrieval selection")
    if (
        not isinstance(retrieval.get("top_k"), int)
        or isinstance(retrieval["top_k"], bool)
        or retrieval["top_k"] <= 0
    ):
        raise RagError("Invalid retrieval top_k")
    prompt_path = root / generation["prompt_file"]
    prompt_bytes = prompt_path.read_bytes()
    if sha256(prompt_bytes) != generation["prompt_sha256"]:
        raise RagError("Prompt hash differs from generation config")
    if generation["max_input_tokens"] <= 0 or generation["max_new_tokens"] <= 0:
        raise RagError("Invalid token budget")
    return retrieval, generation, prompt_bytes.decode("utf-8")


class LocalGenerator:
    """One pinned model/tokenizer load for all queries in a process."""

    def __init__(self, settings: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start = time.monotonic()
        self.device = settings["preferred_device"]
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RagError("MPS unavailable; local model was not loaded")
        self.dtype = (
            torch.bfloat16 if settings["dtype"] == "bfloat16" else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings["model_id"], revision=settings["model_revision"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings["model_id"],
            revision=settings["model_revision"],
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = settings["max_new_tokens"]
        self.load_seconds = time.monotonic() - start
        self.eos_token_id = self.model.generation_config.eos_token_id
        self.pad_token_id = self.model.generation_config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

    def count_tokens(self, messages: list[dict]) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return int(ids.shape[-1])

    def generate(self, messages: list[dict]) -> tuple[str, int, int, float]:
        import torch

        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                tokens,
                attention_mask=torch.ones_like(tokens),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        elapsed = time.monotonic() - start
        new = output[0][tokens.shape[-1] :]
        return (
            self.tokenizer.decode(new, skip_special_tokens=True),
            int(tokens.shape[-1]),
            len(new),
            elapsed,
        )


class RagPipeline:
    def __init__(
        self,
        retriever: object,
        generator: object,
        book: str,
        source_start: int,
        retrieval: dict,
        generation: dict,
        prompt: str,
    ):
        self.retriever = retriever
        self.generator = generator
        self.book = book
        self.source_start = source_start
        self.retrieval = retrieval
        self.generation = generation
        self.prompt = prompt

    def answer(self, query: str) -> dict:
        base = {
            "status": "error",
            "answer": "",
            "claims": [],
            "reason_code": None,
            "reason": "",
            "retrieval_method": self.retrieval["method"],
            "configuration": self.retrieval["configuration"],
            "model_id": self.generation["model_id"],
            "model_revision": self.generation["model_revision"],
            "prompt_version": self.generation["prompt_version"],
            "prompt_sha256": self.generation["prompt_sha256"],
            "max_input_tokens": self.generation["max_input_tokens"],
            "max_new_tokens": self.generation["max_new_tokens"],
            "input_tokens": 0,
            "output_tokens": 0,
            "dropped_chunk_ids": [],
            "attempts": 0,
            "raw_model_outputs": [],
            "submitted_chunks": [],
            "citations_resolved": False,
            "semantic_support_verified": None,
            "generation_seconds": 0.0,
        }
        if not isinstance(query, str) or not query.strip():
            return {
                **base,
                "reason_code": "invalid_input",
                "reason": "Question must be nonempty",
            }
        try:
            bare = prompt_messages(self.prompt, query, [])
            bare_tokens = self.generator.count_tokens(bare)
        except Exception as exc:  # noqa: BLE001 - model tokenizer boundary
            return {
                **base,
                "reason_code": "resource_or_model_error",
                "reason": str(exc),
            }
        if bare_tokens > self.generation["max_input_tokens"]:
            return {
                **base,
                "reason_code": "invalid_input",
                "reason": "Question and instructions exceed token budget",
            }
        try:
            results = self.retriever.search(query, self.retrieval["top_k"])
            passages = make_passages(results, self.book)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            return {**base, "reason_code": "retrieval_error", "reason": str(exc)}
        if not passages:
            return {
                **base,
                "status": "abstained",
                "reason_code": "insufficient_evidence",
                "reason": "No passages were retrieved",
            }
        dropped = []
        messages = prompt_messages(self.prompt, query, passages)
        try:
            while (
                self.generator.count_tokens(messages)
                > self.generation["max_input_tokens"]
            ):
                dropped.append(passages.pop().chunk["chunk_id"])
                messages = prompt_messages(self.prompt, query, passages)
        except Exception as exc:  # noqa: BLE001 - model tokenizer boundary
            return {
                **base,
                "reason_code": "resource_or_model_error",
                "reason": str(exc),
            }
        base["dropped_chunk_ids"] = dropped
        base["submitted_chunks"] = [
            {
                "source_id": p.source_id,
                "chunk_id": p.chunk["chunk_id"],
                "story_title": p.chunk["story_title"],
                "char_start": p.chunk["char_start"],
                "char_end": p.chunk["char_end"],
                "source_line_start": p.chunk["source_line_start"],
                "source_line_end": p.chunk["source_line_end"],
                "text": p.text,
            }
            for p in passages
        ]
        if not passages:
            return {
                **base,
                "status": "abstained",
                "reason_code": "insufficient_evidence",
                "reason": "No complete passage fits the token budget",
            }
        failure = "invalid response"
        for attempt in range(2):
            if attempt:
                messages = prompt_messages(self.prompt, query, passages, failure)
                try:
                    repair_tokens = self.generator.count_tokens(messages)
                except Exception as exc:  # noqa: BLE001 - model tokenizer boundary
                    return {
                        **base,
                        "reason_code": "resource_or_model_error",
                        "reason": str(exc),
                    }
                if repair_tokens > self.generation["max_input_tokens"]:
                    break
            try:
                raw, input_tokens, output_tokens, elapsed = self.generator.generate(
                    messages
                )
            except Exception as exc:  # noqa: BLE001 - local model boundary
                return {
                    **base,
                    "reason_code": "resource_or_model_error",
                    "reason": str(exc),
                }
            base["attempts"] += 1
            base["input_tokens"] = input_tokens
            base["output_tokens"] += output_tokens
            base["generation_seconds"] += elapsed
            base["raw_model_outputs"].append(raw)
            try:
                resolved = resolve_response(raw, passages, self.book, self.source_start)
            except ResponseError as exc:
                failure = str(exc)
                continue
            return {
                **base,
                **resolved,
                "reason_code": "answered"
                if resolved["status"] == "answered"
                else "insufficient_evidence",
            }
        return {**base, "reason_code": "invalid_model_output", "reason": failure}


def load_pipeline(root: Path = PROJECT_ROOT) -> RagPipeline:
    retrieval, generation, prompt = load_settings(root)
    try:
        book, _, source = load_verified_source(root)
        method, config = retrieval["method"], retrieval["configuration"]
        if method == "bm25":
            from sherlock_rag.retrieval import BM25Index, load_verified_chunks

            index = BM25Index(load_verified_chunks(config, root))
        elif method == "semantic":
            from sherlock_rag.semantic import SemanticIndex, load_encoder, load_index

            chunks, vectors, _ = load_index(config, root)
            encoder, _ = load_encoder()
            index = SemanticIndex(chunks, vectors, encoder)
        else:
            from sherlock_rag.hybrid import load_hybrid
            from sherlock_rag.semantic import load_encoder

            encoder, _ = load_encoder()
            index, _ = load_hybrid(config, encoder, root)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise RetrievalSetupError(str(exc)) from exc
    generator = LocalGenerator(generation)
    source_start = source["processed"]["source_lines_inclusive"]["start"]
    return RagPipeline(
        index, generator, book, source_start, retrieval, generation, prompt
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    args = parser.parse_args(argv)
    if not args.query.strip():
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason_code": "invalid_input",
                    "reason": "Question must be nonempty",
                }
            )
        )
        return 1
    try:
        pipeline = load_pipeline()
        result = pipeline.answer(args.query)
        result["model_load_seconds"] = pipeline.generator.load_seconds
        result["device"] = pipeline.generator.device
        result["dtype"] = str(pipeline.generator.dtype)
    except RetrievalSetupError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason_code": "retrieval_error",
                    "reason": str(exc),
                }
            )
        )
        return 1
    except RagError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason_code": "configuration_error",
                    "reason": str(exc),
                }
            )
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI reports model/load failures
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason_code": "resource_or_model_error",
                    "reason": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
