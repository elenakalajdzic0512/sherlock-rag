# Progress

## Implemented

- **Phase 0:** Private `elenakalajdzic0512/sherlock-rag` repository and initial `main` commit `8250000d31dfa5a828d3e5102645bf745c80c49e`.
- **Phase 1:** Minimal setuptools `src/` package and dedicated Python 3.11 environment. PR #1 merged as `da4eb37c38673a4a46c7e408dcb1ccc2ba432105`.
- **Phase 2:** Verified Gutenberg #1661 acquisition, exact narrative extraction, manifest, audit, and offline tests. PR #2 merged as `49f8dd6adee7c0244d7988ca61170898225ca7c9`.
- **Phase 3:** Deterministic, story-bounded sliding-word chunks with exact text slices and word, character, processed-line, and source-line provenance. Both requested configurations and their manifest were generated from the verified local book.

## Phase 3 verification

- Before chunking, raw SHA-256 `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` and processed SHA-256 `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a` matched `data/source_manifest.json`; fresh `extract_book` output matched `book.txt` exactly.
- Configuration A (250 words, 50 overlap) produced 525 chunks, min/max/mean 86/250/247.76 words; JSONL SHA-256 `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd`.
- Configuration B (500 words, 100 overlap) produced 265 chunks, min/max/mean 109/500/489.52 words; JSONL SHA-256 `e8e886a889f8e1d7916eb236a800d694abb1f5c496ea4114c9e16a5a26eee87e`.
- Every saved JSONL record was audited for exact substring, one-story boundary, size, word coverage, Unicode character and line mapping, and unique ID. Both configurations were repeated with byte-identical JSONL and chunk manifest output. See [chunking audit](chunking-audit.md).
- `conda run -n sherlock-rag python -m pytest -q --tb=short`: 28 passed, offline.
- `conda run -n sherlock-rag ruff check .`: passed.
- `conda run -n sherlock-rag ruff format --check .`: passed.
- `conda run -n sherlock-rag python -m pip check`: no broken requirements.
- `git diff --check`: passed. Generated JSONL files remain ignored by Git.

## Planned

Phase 4 adds and evaluates a BM25 retrieval baseline. Semantic and hybrid retrieval, grounded answers, answer evaluation, question generation, and the interface remain planned. No model tokenizer has been selected.

## Blocked

None currently known.
