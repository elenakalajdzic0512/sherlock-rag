# Phase 8 fresh-clone audit

## Scope and identity

- Audited remote `main` at merge commit `a2cf51a27c91357ee72a58dd5319056a766cb94a` in a separate clone at `/private/tmp/sherlock-rag-audit-20261014`. The original working project and its environment were not used as input data.
- The GitHub repository was confirmed `PRIVATE`. The fresh clone began with no ignored raw/book files, chunk JSONL outputs or indexes. A new `sherlock-rag-audit` conda environment was created with Python 3.11.17 at `/Users/ladylazarus/miniforge3/envs/sherlock-rag-audit`.
- Existing Hugging Face cache entries for the pinned embedding and generation revisions were reused. This audit does **not** verify downloading either model from an empty cache. The Gutenberg TXT was downloaded anew from the documented URL and matched the frozen raw hash.

## Commands exercised in the audit clone

The audit used the following commands from the clone root, with `sherlock-rag-audit` substituted for the README environment name:

```sh
git clone --branch main --single-branch https://github.com/elenakalajdzic0512/sherlock-rag.git /private/tmp/sherlock-rag-audit-20261014
conda create -y -n sherlock-rag-audit python=3.11 pip
conda run -n sherlock-rag-audit python -m pip install -e '.[dev,semantic,generation,ui]'
conda run -n sherlock-rag-audit python -m sherlock_rag.ingest
conda run -n sherlock-rag-audit python -m sherlock_rag.chunking --chunk-size-words 250 --overlap-words 50
conda run -n sherlock-rag-audit python -m sherlock_rag.chunking --chunk-size-words 500 --overlap-words 100
conda run --no-capture-output -n sherlock-rag-audit python -m sherlock_rag.semantic build --config w250_o50
conda run -n sherlock-rag-audit python eval/question_bank_review.py
conda run -n sherlock-rag-audit python -m eval.paired_review
conda run -n sherlock-rag-audit python -m eval.rerank_paired_review
conda run -n sherlock-rag-audit python -m pytest -q --tb=short
conda run -n sherlock-rag-audit ruff check .
conda run -n sherlock-rag-audit ruff format --check .
conda run -n sherlock-rag-audit python -m pip check
git diff --check
printf '\n' | conda run --no-capture-output -n sherlock-rag-audit python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

The Streamlit command used port 8502 only to avoid a collision with the normal README port 8501. `printf '\n'` accepted its optional first-run email prompt without supplying an address. No `--no-deps` shortcut or conda dependency fallback was needed; pip installed and resolved all four extras from the standard package index and caches. This demonstrates a complete dependency installation in the fresh environment, although wheel cache reuse means it is not a cold package download test.

Installed versions: Python 3.11.17, sherlock-rag 0.1.0, pytest 9.1.1, Ruff 0.17.0, NumPy 2.4.6, sentence-transformers 5.7.0, transformers 4.57.6, PyTorch 2.14.1, accelerate 1.15.0, Streamlit 1.65.0. `pip check` reported no broken requirements.

## Audit-found ingestion fix

On unmodified `a2cf51a`, the first successful ingestion in a fresh clone created the correct raw and processed bytes, but changed `acquired_at_utc` in the tracked `data/source_manifest.json` and `docs/corpus-audit.md`. The tracked chunk manifest still referred to the original source-manifest hash, so the documented path was not reproducible as written. Its original and processed hashes already matched; changing expected hashes would have hidden this defect.

The Phase 8 fix preserves the tracked acquisition record whenever a newly acquired original matches its raw hash and rejects a differing source or extracted metadata before writes. Two offline regressions cover reacquisition with absent ignored files and rejection of a changed TXT. In the audit clone the fix was applied, only the two audit-generated tracked timestamp changes were restored, and the same documented ingestion command was rerun. It left tracked files unchanged. A and B chunk generation then matched their manifests. The full test and formatting checks were repeated on the fixed code.

## Frozen inputs and derived checks

| Artifact | Fresh-clone SHA-256 / outcome |
| --- | --- |
| Gutenberg original | `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` |
| Processed book | `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a` |
| Tracked source manifest | `9968d6e285e6af29d135fa74400092123a2af75c1bc18a6ea743688b1777cc54`; unchanged after fixed ingestion |
| A chunks (250/50) | 525; `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd` |
| B chunks (500/100) | 265; `e8e886a889f8e1d7916eb236a800d694abb1f5c496ea4114c9e16a5a26eee87e` |
| Current v2 bank | 5 questions, 5/12 stories; `5bfde8a2c2f156dd90aa9cf250ce455de068bdb03c43c1c951e98caab1e453c8` |
| Question-generation raw run | `1fed79cb75bd77d90269dce030b01a97dd9ad999a72986abb64aa8335976046d` |
| Baseline answer raw run | `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721` |
| Prompt-v2 answer raw run | `50b74c80d81607dfdb3b40324f840133949432cb170c0ee376a54b23f141e782` |
| Reranked answer raw run | `4f6f71cccd3c7dba6d16e159373323816f6c8c374e6aee5fb87d68f96628dfef` |

The A semantic index was newly built from 525 verified chunks on MPS with the pinned embedding model. Its fresh vector SHA-256 was `9b41ccdcdbd791dc24efb7ec70795c00bdcdf70b3e92566e8403964d12ebfd1a`; build time was 38.01 seconds. Its metadata points to the unchanged source-manifest, raw, processed and A-chunk hashes. Index matrices are local generated files and are not expected to have a universal byte hash across environments.

The v2 bank compiler validated the locked raw result, historical-to-v2 review lineage, summary and bank hash without loading a model. Both paired review programs recomputed metrics equal to their tracked JSON values. Baseline and prompt-v2 run metrics were recomputed on temporary copies and compared equal to the saved metrics; the tracked run directories were not written. No full evaluation or question-generation run was restarted.

## Offline checks and local browser

- Fresh environment: **171 tests passed** on the fixed code. Ruff check passed; Ruff format check passed (55 files); `pip check` found no broken requirements; `git diff --check` passed. The two new ingestion tests are included in the 171.
- The browser opened the actual audit-clone app on `127.0.0.1:8502`. It showed exactly five v2 cards; `THE RED-HEADED LEAGUE` showed the uncovered-story message. Revealing the first card showed its reviewed expected answer, C1 quote, story title and original lines 613–615. Transferring that question filled the RAG form and did not generate an answer.
- With the cached pinned local model, `What kind of creature caused Roylott's death?` returned **answered**: “A swamp adder caused Roylott's death [C1]”, with the original *THE ADVENTURE OF THE SPECKLED BAND* quote and original lines 7844–7845. Switching to the bank and back retained that result under the submitted question, with a notice when the current form text differed.
- In the same server process, `In what year did Sherlock Holmes use a smartphone?` returned **abstained**. The UI said the retrieved passages were insufficient without asserting that an answer is absent from the book. Server output showed one checkpoint-shard load for both submissions and no second load event; the second question reused the cached pipeline. These are functional smoke checks, not a new answer-quality evaluation.

## Limits

Semantic A and generation v1 remain the defaults. The baseline has **10/24 supported-correct answers** under paired rubric v1. A valid quote string and source mapping do not establish semantic support for an answer. The fixed evaluation set has already been seen, and content reviews are implementation-agent ratings rather than independent human adjudication. The current reviewed bank covers only five of twelve stories. The audit reused a model cache and tested one local machine; it does not prove first-time model download or portability to other hardware.
