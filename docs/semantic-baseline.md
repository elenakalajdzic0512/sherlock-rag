# Local semantic retrieval baseline

## Model, inputs, and safeguards

- Model: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), pinned to commit `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` for every load. The actual local SentenceTransformer has a Qwen3 transformer, last-token pooling (`include_prompt=True`), and a Normalize module. Full output dimension: 1024.
- Installed in the `sherlock-rag` conda environment: `numpy 2.4.6`, `sentence-transformers 5.7.0`, `torch 2.14.1`, `transformers 4.57.6`. No CUDA or Flash Attention package was installed. MPS was available and passed a real small encode, so both indexes and evaluation used MPS.
- Documents are the exact `chunk.text` values, with no title or gold fields. The same fixed task instruction is used for every query, irrespective of story, answer, or split. The exact prefix is:

```text
Instruct: Given a question about a book, retrieve passages containing evidence to answer it.
Query:
```

The prefix ends immediately after `Query:`; the query text follows with no added space or newline.

- Saved vectors are L2-normalized, full-dimension float32. Search uses an exact dot product (cosine similarity) over all rows, with story/chunk order breaking full-score ties. Cosine similarity is neither an answer probability nor an answerability decision.
- `max_seq_length=4096` was set after checking the model and tokenizer limits. With special tokens, `truncation=False` measured maxima of **384** document tokens (A), **725** (B), and **35** for all 30 queries with their real prefix. **Zero inputs were truncated or skipped.** An overlong input raises an error before encoding.
- Each ignored `indexes/*.npy` and `indexes/*.json` pair is configuration-specific. Loading verifies the selected chunk JSONL, source and chunk hashes, pinned model and code version, library versions, prefix, pooling, normalization, matrix shape/dtype/hash, L2 norms, and exact ordered chunk IDs. Missing other-configuration files are irrelevant.

### Local index audit (tracked summary; vectors remain ignored)

| Config | Vectors × dimensions | MPS build time (s) | Max document tokens | Chunk JSONL SHA-256 | NPY SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| w250_o50 | 525 × 1024 | 39.41 | 384 | `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd` | `9b41ccdcdbd791dc24efb7ec70795c00bdcdf70b3e92566e8403964d12ebfd1a` |
| w500_o100 | 265 × 1024 | 41.74 | 725 | `e8e886a889f8e1d7916eb236a800d694abb1f5c496ea4114c9e16a5a26eee87e` | `1e318f71f1fc1519a9dc98ef3cc80263ec1bc71b168194324a7d67497caae4e7` |

Build time includes the token-length pass, document encoding, float32 validation, and NPY serialization; it excludes model download and load. The processed book SHA-256 remains `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a`. The frozen question-set SHA-256 remains `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`.

## BM25 versus semantic results

Gold relevance is unchanged: a returned chunk must belong to the right story **and fully contain one marked evidence interval**. Partial overlap is a miss. Other valid passages may exist but be uncredited. Unsupported questions are excluded. Context words sum the chunking whitespace word counts of the returned top five, including repeated overlap. A and B therefore represent different quantities of text.

| Config | Method | Split | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Mean top-5 words |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w250_o50 | BM25 | dev | 12 | 0.167 | 0.500 | 0.500 | 0.319 | 1216.6 |
| w250_o50 | BM25 | test | 12 | 0.167 | 0.500 | 0.583 | 0.312 | 1225.0 |
| w250_o50 | Semantic | dev | 12 | 0.417 | 0.583 | 0.583 | 0.500 | 1229.1 |
| w250_o50 | Semantic | test | 12 | 0.250 | 0.417 | 0.583 | 0.375 | 1208.2 |
| w500_o100 | BM25 | dev | 12 | 0.333 | 0.500 | 0.583 | 0.424 | 2379.1 |
| w500_o100 | BM25 | test | 12 | 0.417 | 0.750 | 0.833 | 0.586 | 2421.3 |
| w500_o100 | Semantic | dev | 12 | 0.333 | 0.583 | 0.583 | 0.444 | 2412.8 |
| w500_o100 | Semantic | test | 12 | 0.250 | 0.500 | 0.583 | 0.378 | 2405.2 |

### Semantic results by query kind

| Config | Split | Kind | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| w250_o50 | dev | direct | 6 | 0.167 | 0.333 | 0.333 | 0.250 |
| w250_o50 | dev | paraphrase | 6 | 0.667 | 0.833 | 0.833 | 0.750 |
| w250_o50 | test | direct | 6 | 0.000 | 0.167 | 0.333 | 0.125 |
| w250_o50 | test | paraphrase | 6 | 0.500 | 0.667 | 0.833 | 0.625 |
| w500_o100 | dev | direct | 6 | 0.167 | 0.333 | 0.333 | 0.250 |
| w500_o100 | dev | paraphrase | 6 | 0.500 | 0.833 | 0.833 | 0.639 |
| w500_o100 | test | direct | 6 | 0.000 | 0.167 | 0.333 | 0.117 |
| w500_o100 | test | paraphrase | 6 | 0.500 | 0.833 | 0.833 | 0.639 |

On **dev**, semantic A and B tie at Hit@5=0.583. Semantic A has higher MRR@5 (0.500 versus 0.444) and returns about half the text of B, so A is the better semantic candidate on this dev set. Semantic A also improves on BM25 A dev MRR@5 (0.319 to 0.500). Semantic B barely improves BM25 B dev MRR@5 (0.424 to 0.444). The B test result is worse than BM25 B, but test was already seen in Phase 4 and is **not a new independent final test**. No parameters or gold questions were adjusted after measurement.

## Concrete improvements and regressions

Ranks are the first top-five chunk fully containing the gold evidence; `miss` means none of the first five qualifies.

| Question and gold interval | Config | BM25 rank | Semantic rank | Observation |
| --- | --- | ---: | ---: | --- |
| q08: What about the wound led Holmes to infer a left-handed killer? — 04 [175108, 175242) | w250_o50 | miss | 2 | Semantic retrieves the left-handedness wound explanation; BM25 misses. |
| q08: What about the wound led Holmes to infer a left-handed killer? — 04 [175108, 175242) | w500_o100 | miss | 2 | Semantic retrieves the left-handedness wound explanation; BM25 misses. |
| q15: What kind of creature caused Roylott’s death? — 08 [364278, 364345) | w250_o50 | miss | 1 | Semantic puts the swamp-adder explanation first; BM25 misses. |
| q15: What kind of creature caused Roylott’s death? — 08 [364278, 364345) | w500_o100 | miss | 1 | Semantic puts the swamp-adder explanation first; BM25 misses. |
| q13: To which noblewoman did the jewel found in the goose belong? — 07 [285643, 285689) | w250_o50 | miss | 2 | Semantic retrieves the Countess quote in A and improves its rank in B. |
| q13: To which noblewoman did the jewel found in the goose belong? — 07 [285643, 285689) | w500_o100 | 2 | 1 | Semantic retrieves the Countess quote in A and improves its rank in B. |
| q07: What word had the dying McCarthy tried to say? — 04 [173664, 173675) | w250_o50 | 3 | miss | Semantic ranks related Boscombe passages but misses the exact “BALLARAT” quote. |
| q07: What word had the dying McCarthy tried to say? — 04 [173664, 173675) | w500_o100 | 4 | miss | Semantic ranks related Boscombe passages but misses the exact “BALLARAT” quote. |
| q24: What yearly salary did Rucastle first offer Violet Hunter? — 12 [517078, 517137) | w250_o50 | 2 | miss | Semantic returns Copper Beeches passages but misses the exact £100 salary offer. |
| q24: What yearly salary did Rucastle first offer Violet Hunter? — 12 [517078, 517137) | w500_o100 | 2 | miss | Semantic returns Copper Beeches passages but misses the exact £100 salary offer. |
| q20: Whom did Hatty Doran see at the altar rails? — 10 [448729, 448806) | w250_o50 | 4 | miss | Semantic misses the altar-rails passage that BM25 finds. |
| q20: Whom did Hatty Doran see at the altar rails? — 10 [448729, 448806) | w500_o100 | 2 | miss | Semantic misses the altar-rails passage that BM25 finds. |

## Unsupported questions and repeatability

All six unsupported questions returned five candidates per configuration; they remain outside Hit@k and MRR. For q25 (Holmes and a smartphone), the top cosine scores were 0.565 (A) and 0.532 (B). For q28 (Watson’s email password), they were 0.671 (A) and 0.600 (B). These scores do not indicate that the book answers either question.

The real evaluation was repeated from the saved matrices. All 60 configuration-question top-five rankings and scores were identical in this MPS run (maximum observed score difference 0); `eval/semantic_results.json` was byte-identical. Re-encoding documents or queries on another device may introduce small floating-point differences; validation allows L2 norm error up to 0.001, and near ties can change rank. Byte-identical cross-device outputs are not promised.

This is a small, internally composed reference set with one annotated evidence interval per answerable question. It measures retrieval of that interval, not the complete set of legitimate answers or broad quality.

## Commands

```sh
conda run -n sherlock-rag python -m pip install -e ".[semantic]"
conda run -n sherlock-rag python -m sherlock_rag.semantic build --config w250_o50
conda run -n sherlock-rag python -m sherlock_rag.semantic build --config w500_o100
conda run -n sherlock-rag python -m sherlock_rag.semantic search --config w250_o50 --query "Who was the beggar?" --top-k 5
conda run -n sherlock-rag python -m sherlock_rag.evaluate --method semantic
```

After cloning, restore the raw and processed files recorded by the source manifest, regenerate either missing chunk JSONL using the README commands, then build its corresponding semantic index. The tracked `eval/semantic_results.json` contains parameters, input hashes, rankings, and all split/kind metrics; model weights and local matrices remain outside Git. Semantic and hybrid search are separate phases: hybrid is planned for Phase 5b.
