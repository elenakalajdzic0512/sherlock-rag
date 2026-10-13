# Hybrid retrieval baseline (Phase 5b)

The fixed 30-question source-grounded set was evaluated without changing questions, gold intervals, BM25 results, semantic results, chunks, source text, model revision, or saved vectors. The question file SHA-256 is `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`.

## Method

For each selected chunk configuration independently, BM25 and exact cosine search each return at most 20 candidates. Equal-weight reciprocal rank fusion assigns each unique `chunk_id` `1/(60 + bm25_rank) + 1/(60 + semantic_rank)` with one-based ranks and zero for an absent method. The top five are sorted by descending RRF score, then story and chunk index. Raw scores are retained for inspection only. Both methods must use the identical verified source and ordered chunks. Hybrid search fails with a rebuild command if the selected semantic index is absent or invalid. It never silently substitutes BM25.

Gold relevance means that the chunk belongs to the gold story and completely contains the annotated evidence character interval. Only query text reaches either search method. The table uses 12 answerable questions per split. `Words` is mean whitespace-delimited words across the returned top five. Candidate-union hit is measured before fusion, and only applies to hybrid.

| Method | Config | Split | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Words | Union hit |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | A 250/50 | dev | .167 | .500 | .500 | .319 | 1216.6 | — |
| BM25 | A 250/50 | test | .167 | .500 | .583 | .313 | 1225.0 | — |
| BM25 | B 500/100 | dev | .333 | .500 | .583 | .424 | 2379.1 | — |
| BM25 | B 500/100 | test | .417 | .750 | .833 | .586 | 2421.3 | — |
| Semantic | A 250/50 | dev | .417 | .583 | .583 | .500 | 1229.1 | — |
| Semantic | A 250/50 | test | .250 | .417 | .583 | .375 | 1208.2 | — |
| Semantic | B 500/100 | dev | .333 | .583 | .583 | .444 | 2412.8 | — |
| Semantic | B 500/100 | test | .250 | .500 | .583 | .378 | 2405.2 | — |
| Hybrid | A 250/50 | dev | .417 | .500 | .500 | .444 | 1222.4 | .917 |
| Hybrid | A 250/50 | test | .250 | .500 | .500 | .361 | 1225.0 | 1.000 |
| Hybrid | B 500/100 | dev | .417 | .583 | .583 | .500 | 2422.4 | .917 |
| Hybrid | B 500/100 | test | .333 | .583 | .833 | .494 | 2421.3 | 1.000 |

By question kind, hybrid A Hit@5 is dev direct 2/6, dev paraphrase 4/6, test direct 1/6, test paraphrase 5/6. Hybrid B is 2/6, 5/6, 4/6, and 6/6 respectively. The machine-readable result records Hit@1/3/5, MRR@5, context, and union hit by kind for both splits.

Hybrid B improves the test paraphrase Hit@5 to 6/6 from 5/6 for each individual method. On `q08`, BM25 B misses, while semantic and hybrid B place the evidence at rank 2. On `q16`, both individual B methods have weaker or missing results (BM25 rank 5; semantic misses), while hybrid B places evidence at rank 3. Hybrid A regresses on dev `q15`: semantic A finds the evidence at rank 1, while hybrid A loses it from the top five. Hybrid B regresses on test `q18`: semantic B has rank 5, while hybrid B misses. Hybrid B also moves test `q14` from BM25 rank 1 to hybrid rank 5. These differences show why RRF is not assumed to improve retrieval.

Candidate-union hit is 11/12 on both dev configurations and 12/12 on both test configurations. The only absent-union cases are dev `q23` for A and dev `q11` for B. The union contains gold evidence but fusion excludes it from top five for A: dev `q01`, `q03`, `q07`, `q11`, `q15`; test `q02`, `q16`, `q18`, `q20`, `q22`, `q24`. For B: dev `q01`, `q03`, `q07`, `q23`; test `q18`, `q24`. For example, A `q15` has 32 unique candidates, including the gold chunk, but its fused top five miss it. A candidate-union hit is therefore not a retrieval success at top five.

All six unsupported questions return five hybrid candidates in each configuration; they are excluded from answerable retrieval metrics. Candidate rankings and RRF scores do not establish that an answer exists. This phase adds no answerability threshold or generated answer.

## Choice for Phase 6

Before measuring hybrid performance, [the decision log](decisions.md) fixed the six-way choice rule: highest dev Hit@5, then highest dev MRR@5, then lowest average top-five context words, then a stable order. Semantic A and hybrid B both reach dev Hit@5 7/12 and MRR@5 .500. Semantic A wins the context tie-break, 1229.1 versus 2422.4 words, and is recorded in [`config/retrieval.json`](../config/retrieval.json). The test split is reported separately and was not used for selection; earlier phases had already exposed it.

This is a small, manually annotated set with one marked evidence interval per answerable question. Another passage may support an answer without matching the marked interval. The 250/50 and 500/100 configurations also return different amounts of text, so hit rates and context cost should be read together. Grounded answer quality, citations, abstention, and practical context packing remain unmeasured.

## Reproducibility

The evaluator reused verified local semantic matrices and one pinned query encoder for both configurations. Repeating `python -m sherlock_rag.evaluate --method hybrid` from the same indexes produced byte-identical `eval/hybrid_results.json` on this MPS run (SHA-256 `226d999f964a0cf21322d42fa0e19b088c1aa1e3864adfd18b0d25ea6e77126e`). The recorded floating-point values did not differ between these two runs. Hardware or library differences can change the final floating-point bits or tied rankings, so index metadata pins the library versions and model revision.
