# Progress

## Implemented

- **Phase 0:** Private `elenakalajdzic0512/sherlock-rag` repository and initial `main` commit `8250000d31dfa5a828d3e5102645bf745c80c49e`.
- **Phase 1:** Minimal setuptools `src/` package and dedicated Python 3.11 environment. PR #1 merged as `da4eb37c38673a4a46c7e408dcb1ccc2ba432105`.
- **Phase 2:** Verified Gutenberg #1661 acquisition, exact narrative extraction, manifest, audit, and offline tests. PR #2 merged as `49f8dd6adee7c0244d7988ca61170898225ca7c9`.
- **Phase 3:** Deterministic, story-bounded sliding-word chunks with exact text slices and word, character, processed-line, and source-line provenance. Both requested configurations and their manifest were generated from the verified local book. PR #3 merged as `7f2e5be6b8959bf27dd543caca7f8adc5372af0d`.
- **Phase 4:** Verified loading of either selected JSONL, Unicode lexical tokenization, in-memory BM25, JSON search CLI, fixed 30-question source-grounded benchmark, and machine-readable evaluation for both chunk configurations. PR #4 merged as `6465decce4f2b40f2d547072fea5fbdf46d1bb77`. See [retrieval baseline](retrieval-baseline.md).
- **Phase 5a:** Pinned local Qwen embedding model, checked MPS encoding, verified float32 indexes for both configurations, exact cosine search, and comparison with the unchanged BM25 reference set. PR #5 merged as `7965d0c100bad64610313a579aaca00ab5153fde`. See [semantic baseline](semantic-baseline.md).
- **Phase 5b:** Verified equal-weight RRF across existing BM25 and semantic rankings, candidate-union diagnostics, unchanged reference-set evaluation, and a dev-only choice for Phase 6. PR #6 merged as `f2f718797128c557627363bd2d7aa476a9e1e45b`. See [hybrid baseline](hybrid-baseline.md).
- **Phase 6a (ready for review):** Local pinned answer model, selected semantic A retrieval, complete-prompt token budgeting, mechanically resolved book citations, basic abstention, one repair attempt, and fixed eight-question engineering smoke audit. See [local smoke report](local-rag-smoke.md).

## Phase 3 verification

- Before chunking, raw SHA-256 `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` and processed SHA-256 `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a` matched `data/source_manifest.json`; fresh `extract_book` output matched `book.txt` exactly.
- Configuration A (250 words, 50 overlap) produced 525 chunks, min/max/mean 86/250/247.76 words; JSONL SHA-256 `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd`.
- Configuration B (500 words, 100 overlap) produced 265 chunks, min/max/mean 109/500/489.52 words; JSONL SHA-256 `e8e886a889f8e1d7916eb236a800d694abb1f5c496ea4114c9e16a5a26eee87e`.
- Fresh-clone simulation copied only verified raw/book files and the source/chunk manifests into a temporary folder. A then B regenerated 525 and 265 chunks with the original JSONL hashes, kept both manifest entries byte-identical, and repeated both runs without byte changes. The original project inputs were unchanged. Missing other-configuration outputs no longer block regeneration; a present one with a wrong hash still blocks before writes.
- Every saved JSONL record was audited for exact substring, one-story boundary, size, word coverage, Unicode character and line mapping, and unique ID. Both configurations were repeated with byte-identical JSONL and chunk manifest output. See [chunking audit](chunking-audit.md).
- `conda run -n sherlock-rag python -m pytest -q --tb=short`: 32 passed, offline.
- `conda run -n sherlock-rag ruff check .`: passed.
- `conda run -n sherlock-rag ruff format --check .`: passed.
- `conda run -n sherlock-rag python -m pip check`: no broken requirements.
- `git diff --check`: passed. Generated JSONL files remain ignored by Git.

## Phase 4 verification

- The 24 answerable questions (two distinct facts per story) and six unsupported questions were written against the real book before the first BM25 measurement. Each dev/test split has 12 answerable questions (six direct, six paraphrase) and three unsupported. The frozen question JSONL SHA-256 is `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`.
- A (250/50): dev Hit@5 0.500, MRR@5 0.319, mean top-five context 1,216.6 whitespace words; test Hit@5 0.583, MRR@5 0.312, mean context 1,225.0 words.
- B (500/100): dev Hit@5 0.583, MRR@5 0.424, mean top-five context 2,379.1 words; test Hit@5 0.833, MRR@5 0.586, mean context 2,421.3 words. Full Hit@1/3/5 and kind-specific metrics appear in the baseline report and tracked JSON.
- Unsupported queries returned five lexical candidates each in both configurations. They are excluded from answerable metrics; this phase makes no answerability decision.
- Full offline pytest suite: 53 passed. Ruff check and format check passed; pip check found no broken requirements; git diff --check passed.

## Phase 5a verification

- Model revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; sentence-transformers 5.7.0, NumPy 2.4.6, PyTorch 2.14.1, transformers 4.57.6. MPS smoke encode passed. All embeddings are full 1024-dimensional, L2-normalized float32.
- With the fixed query prefix and special tokens, maxima were 384 (A documents), 725 (B documents), and 35 (all 30 queries), below the 4096 limit. Zero truncations. A: 525 vectors in 39.41 s; B: 265 vectors in 41.74 s. Local indexes are ignored by Git.
- Semantic A dev/test Hit@5 0.583/0.583 and MRR@5 0.500/0.375; semantic B dev/test Hit@5 0.583/0.583 and MRR@5 0.444/0.378. The baseline report shows all metrics, context amounts, improvements, and regressions.
- Repeating evaluation from the saved indexes produced identical rankings, scores, and JSON bytes on this MPS run. Source, chunk JSONL, questions, and BM25 results retain their original SHA-256 values.
- Full offline pytest suite: 64 passed, 0 skipped. Ruff check/format, pip check, and git diff --check passed.

## Phase 5b verification

- Both existing indexes were loaded and checked; no document embeddings were rebuilt. RRF uses 20 candidates per method, `rrf_k=60`, equal weights, and top five results. The same 30 questions and previous BM25/semantic JSON bytes were preserved.
- Hybrid A dev/test Hit@5 0.500/0.500, MRR@5 0.444/0.361, and candidate-union hit 0.917/1.000. Hybrid B dev/test Hit@5 0.583/0.833, MRR@5 0.500/0.494, and candidate-union hit 0.917/1.000. The [report](hybrid-baseline.md) gives full metrics, context, by-kind results, and fusion misses.
- A second evaluation from the same saved indexes produced byte-identical JSON, SHA-256 `226d999f964a0cf21322d42fa0e19b088c1aa1e3864adfd18b0d25ea6e77126e`. No floating-point difference was observed on this MPS run.
- Semantic A was selected for Phase 6 using only dev Hit@5, MRR@5, and context, following the rule recorded before measurement. Test results were excluded from the choice.
- Full offline pytest suite: 77 passed. Ruff check and format check passed; pip check found no broken requirements; git diff --check passed.

## Phase 6a verification

- The local generation model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`; it loaded once for the eight-question batch on MPS as bfloat16. Model load took 4.34 seconds; measured generation totaled 69.47 seconds. Complete chat input lengths in the batch were 2205–2341 tokens, below the 6144 input budget, and no chunk was dropped.
- The fixed smoke set produced three answered cases with mechanically resolved citations (`q09`, `q13`, `q15`), four appropriate refusals (`q25`–`q28`), and one `invalid_model_output` error (`q01`). The gold passage for `q01` was absent from retrieved top five. One `q09` citation is overly long despite being mechanically valid. See [the audit](local-rag-smoke.md), which separates structural checks from implementation-agent content review.
- The public `python -m sherlock_rag.rag --query "What kind of creature caused Roylott’s death?"` command independently returned an answered result with a resolved C1 citation on the same local model.
- Full offline pytest suite: 91 passed. Ruff check, Ruff format check, pip check, and git diff --check passed.

## Planned

Phase 6b performs full answer, citation-support, and abstention evaluation. Question generation and the interface remain planned.

## Blocked

None currently known.
