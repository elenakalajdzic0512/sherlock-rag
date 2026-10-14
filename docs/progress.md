# Progress

## Implemented

- **Phase 0:** Private `elenakalajdzic0512/sherlock-rag` repository and initial `main` commit `8250000d31dfa5a828d3e5102645bf745c80c49e`.
- **Phase 1:** Minimal setuptools `src/` package and dedicated Python 3.11 environment. PR #1 merged as `da4eb37c38673a4a46c7e408dcb1ccc2ba432105`.
- **Phase 2:** Verified Gutenberg #1661 acquisition, exact narrative extraction, manifest, audit, and offline tests. PR #2 merged as `49f8dd6adee7c0244d7988ca61170898225ca7c9`.
- **Phase 3:** Deterministic, story-bounded sliding-word chunks with exact text slices and word, character, processed-line, and source-line provenance. Both requested configurations and their manifest were generated from the verified local book. PR #3 merged as `7f2e5be6b8959bf27dd543caca7f8adc5372af0d`.
- **Phase 4:** Verified loading of either selected JSONL, Unicode lexical tokenization, in-memory BM25, JSON search CLI, fixed 30-question source-grounded benchmark, and machine-readable evaluation for both chunk configurations. PR #4 merged as `6465decce4f2b40f2d547072fea5fbdf46d1bb77`. See [retrieval baseline](retrieval-baseline.md).
- **Phase 5a:** Pinned local Qwen embedding model, checked MPS encoding, verified float32 indexes for both configurations, exact cosine search, and comparison with the unchanged BM25 reference set. PR #5 merged as `7965d0c100bad64610313a579aaca00ab5153fde`. See [semantic baseline](semantic-baseline.md).
- **Phase 5b:** Verified equal-weight RRF across existing BM25 and semantic rankings, candidate-union diagnostics, unchanged reference-set evaluation, and a dev-only choice for Phase 6. PR #6 merged as `f2f718797128c557627363bd2d7aa476a9e1e45b`. See [hybrid baseline](hybrid-baseline.md).
- **Phase 6a:** Local pinned answer model, selected semantic A retrieval, complete-prompt token budgeting, mechanically resolved book citations, basic abstention, one repair attempt, and fixed eight-question engineering smoke audit. PR #7 merged as `da435faf301054de4e5aad1b99b489d93387a67c`. See [local smoke report](local-rag-smoke.md).
- **Phase 6b:** Frozen 30 original and 12 source-reviewed challenge questions, checkpointed 42-case local run, separate implementation-agent content review, explicit-denominator metrics, and [answer baseline report](answer-baseline.md). PR #8 merged as `37ea75dfa67cb522684c39b7bc5417aa947b2b1c`.
- **Phase 6c1:** One pre-registered [prompt v2 candidate](answer-reliability-protocol.md), unchanged selected retrieval/model settings, frozen 42-case results, separate implementation-agent review, and [baseline comparison](answer-reliability.md). PR #9 merged as `bf239c83b3197ae08032aeda9b9423289be8a2e1`; v1 remains the default.
- **Phase 6c2 (merged in PR #10):** One pinned [reranking candidate](rerank-protocol.md), source-verified BM25/semantic A union, full pair-length audit, 30-question retrieval comparison, dev-gated frozen 42-case generation-v1 run, and separate [paired content comparison](rerank-experiment.md). Semantic A remains the default.
- **Phase 7a (ready for review):** A frozen 36-passage local [question-generation run](question-generation-audit.md), separate implementation-agent content ratings, and a six-question reviewed product bank. Five of twelve stories are covered; the shortfall is reported without replacement questions.

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
- **Phase 6a review fix:** When retrieval succeeds but no whole passage fits the input budget, the pipeline now returns `error` / `context_budget_exceeded`, retains every dropped chunk ID, and makes zero generation attempts. Empty retrieval and valid model refusal remain `abstained` / `insufficient_evidence`. Both generation token budgets are validated as positive integers before model loading. Offline budget reproductions and type/value tests cover the distinction; the eight recorded smoke outputs were not regenerated. Full offline suite: 107 passed; Ruff check/format, pip check, and git diff --check passed.

## Phase 6b verification

- Original question SHA-256 `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`; pre-run challenge SHA-256 `cdf59802eeeddde1b3d0430651d999abcc51f5f1186a36b44a8a8bd0fdafe006`. Full runtime fingerprint SHA-256 `115287d5caac7917599e324dcd759836aeb0596d69542e6090d2ed21d83ed589`.
- All 42 local results were frozen at SHA-256 `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721` before content review. Separate review SHA-256 `4507af793178a88b23e7139bf698dd8437b00ebee61e5feffe99e4504395cea5`.
- Automatic statuses: 15 answered, 20 abstained, seven invalid-model-output errors. First-pass mechanical success 20/42; 22/42 repair attempts. Generation took 345.39 seconds and produced 5,910 output tokens.
- Implementation-agent review: supported correct answer 10/24 original answerable; displayed-answer correctness 11/15; unsupported claims 5/15 displayed; sufficient sent context 17/24, with supported correct generation 10/17 and unjustified abstention 0/17. Original unsupported refusals 6/6; unmentioned-detail 6/6; false-premise 3/4 with one invalid output and one supported correction in a refusal reason; other-book 2/2. This is not independent human adjudication.
- Full offline pytest suite: 113 passed. Ruff check and format check, pip check, and git diff --check passed.
- **Phase 6b PR #8 review fix:** New fingerprint schema v2 covers sorted relative paths and SHA-256 contents for all package Python source, including `semantic.py` and the runner. A changed module blocks checkpoint resume before model loading and file writes. `baseline_v1` keeps its old, narrower fingerprint and frozen result; review remains available without the model. Metrics now record missing attempt/token/time fields and classify `runner_error` as a technical error. Recomputed baseline metrics retain the original outcome counts and content scores. GitHub visibility was restored to `PRIVATE` and verified. Full offline suite: 116 passed; Ruff check/format, pip check, and git diff --check passed.

## Phase 6c1 verification

- The only candidate was frozen before measurement: prompt SHA-256 `9f7e4d13b5d08782b4274899861d5766c5705bf38665fc246c6a947894c50108`, experimental generation config SHA-256 `98e68afb0298dc3da2553a3e48c9160fc50517d2f5bf5d8100e5f26574c87877`, and runtime fingerprint SHA-256 `56b872ab5bc37c4e7967a7c98caba4d2dc72368970c406c7336f794b943ef2df`.
- All 42 candidate cases completed and froze at results SHA-256 `50b74c80d81607dfdb3b40324f840133949432cb170c0ee376a54b23f141e782`. Each sent passage list and dropped-chunk list exactly matches `baseline_v1`. Baseline result SHA-256 remains `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721`.
- First-pass structural success rose from 20/42 to 40/42; final invalid outputs fell from seven to one. Reviewed supported-correct original answers rose from 10/24 to 14/24, while displayed answers with unsupported claims remained five. Four formerly invalid cases became supported answers; q01 became an unsupported displayed answer and c09 accepted a false premise. V2 remains experimental. See [the detailed comparison](answer-reliability.md).
- Full offline pytest suite: 118 passed. Ruff check and Ruff format check passed; pip check found no broken requirements; git diff --check passed.
- **PR #9 paired-review correction:** Recorded a common [paired content rubric](paired-content-rubric-v1.md), re-rated every displayed claim and refusal in both frozen runs, and stored separate versioned ratings and metrics under `eval/paired_reviews`. q06 now has identical passage-support and selected-quote ratings in both runs. Candidate q10’s extra crew claim lacks an identifiable Calhoun referent in its own cited chunk. The paired checker found seven identical claim/evidence groups with zero rating disagreements. Safe refusal reasons and missed correction opportunities are separate: c07/c08 candidate refusals are safe and appropriate, although a supported correction was possible. Frozen results and historical reviews remain unchanged.
- Paired-review correction checks: 120 offline pytest tests passed; Ruff check and format check passed; pip check found no broken requirements; git diff --check passed. Recomputed paired metrics are byte-identical to their saved artifact. Both raw result hashes, both historical review hashes, and v1/v2 prompt and generation-config hashes were verified unchanged.

## Planned

A separate task can address the seven uncovered stories and the interface. Neither is part of Phase 7a. Existing answer-quality limitations remain as recorded in prior reports.

## Phase 6c2 verification

- The config SHA-256 `038981273a4b39183e5c65abdec520025ec538191311af9139da43938e2cc058` and pinned model/tokenizer revision were recorded before retrieval scoring. MPS passed a real reranker forward pass; the 512-position model and tokenizer saw a maximum complete pair length of 432 with zero truncations.
- BM25/semantic candidate union gold hit was 11/12 dev and 12/12 test. Final reranked Hit@5 was 8/12 dev and 7/12 test, compared with semantic A 7/12 and 7/12. Dev met the frozen gate; no test tuning occurred.
- The one 42-case generation-v1 run froze at SHA-256 `4f6f71cccd3c7dba6d16e159373323816f6c8c374e6aee5fb87d68f96628dfef` with expanded runtime fingerprint `1a17f7b8e49c1c96c6a0b3952222cd6f2e82e828350abff2966fe12f5db79fea`. All cases retained candidate unions and sent passages; every sent context differs from baseline.
- Under paired content rubric v1, supported-correct original answers stayed 10/24, unsupported displayed claims moved from 5/15 to 4/14, and safe refusal reasons from 19/20 to 17/22. The implementation-agent review is separate from retrieval metrics; see [full report](rerank-experiment.md).
- Full offline pytest suite: 128 passed. Ruff check and format check passed; pip check found no broken requirements; staged `git diff --check` passed. The frozen result SHA, current rerank fingerprint, and derived paired metrics were revalidated without loading models.

## Phase 7a verification

- Frozen selection: 36 A passages, three quarter positions per each of 12 stories; selection SHA-256 `6d10a4bf2d0ce1ab9047f6982f1b6efd31be165ca1e103f2c161d95213d383c0`. Existing book and A-chunk hashes matched the source and chunk manifests.
- One pinned local Qwen run loaded once on MPS/bfloat16. The largest full input prompt measured 719 tokens versus the 6,144 limit; 39 attempts yielded 4,073 output tokens in 182.03 generation seconds. Raw results were frozen before review at SHA-256 `1fed79cb75bd77d90269dce030b01a97dd9ad999a72986abb64aa8335976046d`.
- Outcomes: 28 mechanically valid candidates, five `insufficient_evidence`, three mechanical errors, zero duplicate normalized questions. Separate implementation-agent review accepted six questions from five stories. Stories 02, 03, 06, 07, 08, 09 and 11 remain uncovered. See [audit](question-generation-audit.md) for per-story results and examples.
- The prior question set, challenges and all three frozen 42-case result files retain their recorded SHA-256 values; source manifests, prompt v1/v2, generation configs, and retrieval settings were not edited. The reviewed bank is product content, not an independent benchmark.
- Full offline pytest suite: 143 passed. Ruff check and Ruff format check passed; pip check found no broken requirements; staged `git diff --check` passed. The separate bank compiler reproduced the recorded hashes without loading a model.

## Blocked

None currently known.
