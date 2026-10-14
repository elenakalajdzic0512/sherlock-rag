# sherlock-rag

A book-grounded retrieval-augmented generation (RAG) project based on Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, [Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661).

**Status:** Phase 7a provides a frozen, source-traceable question-generation batch and a separately reviewed question bank. The current v2 review accepts five questions from five stories; the original six-question review remains historical. The interface remains planned. Semantic A retrieval and generation v1 remain the defaults.

## Planned features

- Search the book and answer questions grounded in its text.
- Cite real passages from the source.
- Decline to answer when the evidence is insufficient.
- Generate questions from the book.
- Evaluate retrieval and answer quality.
- Provide a simple user interface.

See [the roadmap](docs/roadmap.md), [progress](docs/progress.md), [corpus audit](docs/corpus-audit.md), [chunking audit](docs/chunking-audit.md), [retrieval baseline](docs/retrieval-baseline.md), [semantic baseline](docs/semantic-baseline.md), and [hybrid baseline](docs/hybrid-baseline.md).

## Development setup

Use a dedicated Python 3.11 conda environment:

```sh
conda create -n sherlock-rag python=3.11 pip
conda run -n sherlock-rag python -m pip install -e '.[dev]'
```

On this machine, PyPI's file host was unavailable during Phase 1. The working installation used conda-forge for development tools and the local build backend for the editable package:

```sh
conda install -n sherlock-rag pytest ruff
conda run -n sherlock-rag python -m pip install --no-index --no-build-isolation --no-deps -e '.[dev]'
```

## Acquire and prepare the book

Run from any directory with the project environment:

```sh
conda run -n sherlock-rag python -m sherlock_rag.ingest
```

The command downloads and validates the Gutenberg TXT, preserves its original bytes in `data/raw/sherlock_holmes.txt`, and extracts the narrative into `data/processed/book.txt`. Both text files are ignored by Git. It writes a tracked [source manifest](data/source_manifest.json) and [corpus audit](docs/corpus-audit.md). A different candidate source never silently replaces an existing original. If a network connection fails, it uses the validated existing original or checks only `~/Downloads/pg1661.txt` and `~/Downloads/sherlock_holmes.txt`.

To use a manually downloaded TXT instead:

```sh
conda run -n sherlock-rag python -m sherlock_rag.ingest --input-file /absolute/path/to/pg1661.txt
```

## Traceable word chunks

Verify the existing raw and processed files against `data/source_manifest.json`, then create either configuration:

```sh
conda run -n sherlock-rag python -m sherlock_rag.chunking --chunk-size-words 250 --overlap-words 50
conda run -n sherlock-rag python -m sherlock_rag.chunking --chunk-size-words 500 --overlap-words 100
```

The commands write separate ignored `data/processed/chunks_w250_o50.jsonl` and `data/processed/chunks_w500_o100.jsonl` files. The tracked [chunk manifest](data/chunk_manifest.json) retains both configurations, their output hashes, and summary statistics. Chunk sizes and overlaps count `\S+` whitespace-delimited words, not model tokens. Each chunk contains an exact substring of `book.txt` and carries story, word, character, and source-line positions. The command refuses missing or changed source files and never downloads the book.

After a fresh clone, restore the exact `data/raw/sherlock_holmes.txt` and `data/processed/book.txt` files recorded by `data/source_manifest.json`, then run the two commands above in either order. The tracked chunk manifest describes expected JSONL artifacts; missing local JSONL files are regenerated without dropping the other configuration entry. An existing output for the other configuration with a mismatched hash is rejected before any files are changed.

## BM25 retrieval and evaluation

Search either verified chunk set from the terminal:

```sh
conda run -n sherlock-rag python -m sherlock_rag.retrieval --config w250_o50 --query "How did Holmes recognize that Watson was practising medicine again?" --top-k 5
conda run -n sherlock-rag python -m sherlock_rag.retrieval --config w500_o100 --query "Where was the photograph hidden?" --top-k 5
```

The JSON results include full BM25 scores, exact chunk text, story and source lines, and matched query terms. A score is lexical relevance, not a confidence percentage or an answerability decision. Search checks the chosen JSONL against the tracked manifest and source; if it is missing after cloning, the error gives its chunk regeneration command. The other configuration may be absent locally.

Re-run the fixed question-set evaluation and write the tracked machine-readable result:

```sh
conda run -n sherlock-rag python -m sherlock_rag.evaluate
```

See [the baseline report](docs/retrieval-baseline.md) for tokenization, formulas, split metrics, context size, and concrete successes and misses. The evaluator validates the source-grounded questions before searching.

## Local semantic retrieval (Phase 5a)

Install the optional semantic extra only in the project environment:

```sh
conda run -n sherlock-rag python -m pip install -e '.[semantic]'
```

The verified [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) model is pinned to revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Build a local index for either configuration after restoring the verified source and chunk JSONL. The model download and matrices stay outside Git:

```sh
conda run -n sherlock-rag python -m sherlock_rag.semantic build --config w250_o50
conda run -n sherlock-rag python -m sherlock_rag.semantic build --config w500_o100
conda run -n sherlock-rag python -m sherlock_rag.semantic search --config w250_o50 --query "Who was the beggar?" --top-k 5
conda run -n sherlock-rag python -m sherlock_rag.evaluate --method semantic
```

Each command verifies the selected source, chunks, and index. A missing index reports its rebuild command. BM25 remains available without installing this extra. The [semantic report](docs/semantic-baseline.md) records the actual versions, device, token-length audit, index hashes, and comparison on the unchanged reference set. Similarity scores are not confidence percentages or decisions about answerability. Phase 5a was merged in PR #5.

## Hybrid retrieval (Phase 5b)

Search one verified configuration with equal-weight reciprocal rank fusion of the top 20 BM25 and semantic candidates:

```sh
conda run -n sherlock-rag python -m sherlock_rag.hybrid --config w250_o50 --query "Where was Irene Adler’s photograph hidden?" --top-k 5 --candidate-k 20 --rrf-k 60
conda run -n sherlock-rag python -m sherlock_rag.evaluate --method hybrid
```

The JSON output records ranks, raw method scores, RRF contributions, exact chunk text and source locations, and all parameters. Missing or invalid saved semantic indexes fail with a rebuild command; the chosen configuration works even if the other one is missing locally. See the [hybrid report](docs/hybrid-baseline.md) for dev/test results and candidate-union misses. [`config/retrieval.json`](config/retrieval.json) selects semantic A for the next phase by a predeclared dev-only rule. Search results alone are not evidence of answerability.

## Local grounded answers (Phase 6a)

Install the optional generation dependencies in the project environment, then run the pinned local model:

```sh
conda run -n sherlock-rag python -m pip install -e '.[generation]'
conda run -n sherlock-rag python -m sherlock_rag.rag --query "What kind of creature caused Roylott’s death?"
```

The selected retrieval method and chunk configuration come from [`config/retrieval.json`](config/retrieval.json). [`config/generation.json`](config/generation.json) pins the Qwen/Qwen3-4B-Instruct-2507 revision, bfloat16 MPS runtime, 6144 input-token budget, and 768 greedy output tokens. The full chat prompt is measured without truncation. When it exceeds the input budget, entire lowest-ranked chunks are removed and listed in `dropped_chunk_ids`; if the question and instructions alone are too long, the command returns `invalid_input`.

Each result contains a status, reason code, the exact passages supplied to the model, raw model outputs, token counts and timing, and validated claims with resolved source citations. Whitespace-only display normalization is mapped back to original character offsets and source lines. Repeated quotes resolve to their first occurrence within the cited passage. A malformed JSON response or citation gets at most one repair attempt with the same passages, then returns `error`. A citation's existence is mechanically checked; whether the cited words support the claim still needs review. The model and cache are outside Git.

Reason codes distinguish outcomes: empty retrieval or a valid model refusal returns `abstained` with `insufficient_evidence`; an oversized question and instructions returns `error` with `invalid_input`; retrieved passages that all exceed the context budget return `error` with `context_budget_exceeded` and their IDs in `dropped_chunk_ids`; invalid JSON or citations after the repair limit return `error` with `invalid_model_output`. A context-budget error does not call the generator and is not an evidence-based refusal.

The fixed eight-question engineering audit is in [the local smoke report](docs/local-rag-smoke.md). Phase 6a was merged in PR #7.

## Answer evaluation (Phase 6b)

The original `baseline_v1` results are frozen and remain reviewable without loading a model:

```sh
conda run -n sherlock-rag python -m sherlock_rag.review_rag --run-id baseline_v1
```

The original run used a legacy fingerprint that hashed only `rag.py` as pipeline code. It could miss changes in retrieval modules such as `semantic.py`. New runs use fingerprint schema v2, with sorted relative paths and SHA-256 hashes for all Python source files in `src/sherlock_rag`, including ingestion, retrieval, generation, and runner code; cache and generated directories are excluded. A changed source file rejects checkpoint resume before model loading or writing. The old `baseline_v1` result and its fingerprint are not migrated or regenerated. To run the current code after restoring the verified source, 250/50 chunks, and semantic index, use a **new run ID**:

```sh
conda run -n sherlock-rag python -m sherlock_rag.evaluate_rag --run-id followup_v2
```

The runner sends only the query from each of the 30 unchanged reference questions and 12 source-reviewed challenges, checkpoints after every case, and resumes only with an identical fingerprint. Once all 42 cases finish, it freezes `results.json` with `results.sha256`; rerunning a completed matching run never regenerates answers. The review command requires separate implementation-agent content ratings in `reviews.json`, checks the frozen result hash and every cited support quote, and writes `metrics.json`. It never changes raw answers. The challenge set was written after previous findings, so the [answer baseline report](docs/answer-baseline.md) treats it as an internal stress test rather than an independent benchmark. If a runner-level exception lacks attempt, token, or timing fields, metrics report those measurements as missing, with explicit recorded/missing counts, and count the result as a technical error.

## Answer reliability candidate (Phase 6c1)

The default answer and evaluation commands still use prompt v1. Explicitly select the frozen v2 candidate with `--generation-variant answer-v2` and a new run ID:

```sh
conda run -n sherlock-rag python -m sherlock_rag.rag --generation-variant answer-v2 --query "Which company employed James Windibank?"
conda run -n sherlock-rag python -m sherlock_rag.evaluate_rag --run-id answer_v2_candidate --generation-variant answer-v2
conda run -n sherlock-rag python -m sherlock_rag.review_rag --run-id answer_v2_candidate
```

The recorded run is frozen, so the evaluation command only checks and returns its existing result when the full fingerprint matches. See the [pre-run protocol](docs/answer-reliability-protocol.md) and [comparison report](docs/answer-reliability.md). V2 is a research candidate, not the default. Its mechanical citation checks do not prove semantic support.

The primary content comparison applies one [paired rubric](docs/paired-content-rubric-v1.md) to both frozen runs. The original per-run reviews remain historical. Validate the paired ratings and reproduce their metrics offline without loading a model:

```sh
conda run -n sherlock-rag python -m eval.paired_review
```

## Generate source-traceable questions (Phase 7a)

Restore and verify the original book, processed book, and A (250/50) chunks as above. Question generation uses only those passages and the existing pinned local Qwen generation model; it does not load retrieval or embedding models. The first batch is in English. First freeze the deterministic three-per-story selection, then run the 36-passage batch:

```sh
conda run -n sherlock-rag python -m sherlock_rag.question_generation prepare
conda run -n sherlock-rag python -m sherlock_rag.question_generation run
conda run -n sherlock-rag python eval/question_bank_review.py
```

The first two commands verify the source and exact frozen inputs. `run` checkpoints every passage and resumes an incomplete batch only with the same full fingerprint. A completed, hash-locked batch is read without model loading or regeneration when its source, selection, prompt, and config still match; its original runtime fingerprint remains unchanged. It saves every raw attempt in `results.json`. The last command validates the current v2 content ratings and reports [`bank_v2.jsonl`](questions/generated_v1/bank_v2.jsonl) without loading the model. The original `reviews.json`, `bank.jsonl`, and `summary.json` remain historical. See the [frozen protocol](docs/question-generation-protocol.md) and [audit and examples](docs/question-generation-audit.md). The v2 bank is the reviewed product content for a future interface, not an independent evaluation benchmark. It currently covers five of twelve stories; rejected outputs remain visible in the raw run and reviews.

## Local interface (Phase 7b)

Install the separate UI extra in the existing conda environment, then start the server from this repository on localhost only:

```sh
conda run -n sherlock-rag python -m pip install -e '.[ui]'
conda run --no-capture-output -n sherlock-rag python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. The UI extra pins Streamlit 1.65.0. The app offers **Pitaj knjigu** for the existing semantic A retrieval and generation v1 pipeline, and **Pitanja iz knjige** for the five source-reviewed questions in `bank_v2.jsonl`. No model loads while browsing the bank; the shared local pipeline is cached after the first submitted question. A form prevents typing, filtering, or revealing answers from triggering generation. Results stay attached to the submitted question across reruns. The reviewed expected answer in the bank is separate from any newly generated RAG answer.

Restore the verified source and processed book, regenerate A chunks, and rebuild its semantic index using the commands above if those ignored local files are missing. The app checks the source, chunks, index, prompt/config and v2 bank provenance before using them and reports the relevant preparation command; it does not download data or models on page load. See the [UI verification report](docs/local-ui.md) for the real-model smoke checks and limitations. The app is local only and does not authenticate users or deploy publicly.

## Checks

```sh
conda run -n sherlock-rag python -m pytest
conda run -n sherlock-rag ruff check .
conda run -n sherlock-rag ruff format --check .
conda run -n sherlock-rag python -m pip check
git diff --check
```

The editable installation makes `sherlock_rag` importable from outside this repository without setting `PYTHONPATH`.
