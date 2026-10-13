# BM25 retrieval baseline

## Fixed inputs and method

- Source: Project Gutenberg #1661, verified against `data/source_manifest.json`; the processed book SHA-256 is `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a`.
- The 30-question set was curated by reading the actual book **before the first ranking run** and was frozen at SHA-256 `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`. It has 24 answerable questions (two different facts per story) and six deliberately unsupported questions. Each split has 12 answerable questions, one from each story, with six direct and six paraphrased; each split also has three unsupported questions.
- Only `chunk.text` is indexed. The query alone is passed to search. Story IDs, answers, evidence intervals, and split labels never enter ranking or enrich queries.
- Tokenization: Unicode `casefold`; normalize `’` to ASCII apostrophe; extract words and numbers with optional internal apostrophes. Underscores, hyphens, and other punctuation separate tokens. For example, `Holmes’s _red-headed_ 42` becomes `holmes's`, `red`, `headed`, `42`. No stemming, lemmatization, or stopword removal. This differs from chunking’s whitespace word count.
- Parameters: `k1=1.2`, `b=0.75`; `N` is the number of chunks in the selected configuration, `df` counts chunks containing a term, and `dl`/`avgdl` count retrieval tokens. Repeated query terms count once. `idf(t)=ln(1+(N-df(t)+0.5)/(df(t)+0.5))`; each matching term contributes `idf(t) * tf(t,d) * (k1+1) / (tf(t,d)+k1*(1-b+b*dl(d)/avgdl))`. Results use the full score and ties sort by story and chunk index. Zero-score chunks are omitted.
- Gold relevance requires the correct story **and full containment** of the exact evidence character interval. Merely overlapping it is a miss. The quote, source hashes, offsets, line numbers, and split distribution are validated before measurement.
- The returned context-word count below sums the chunking whitespace word counts across the returned top five, including repeated overlap. It is not a unique-word count.

## Answerable-question results

| Configuration | Split | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Mean top-5 context words |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| w250_o50 | dev | 12 | 0.167 | 0.500 | 0.500 | 0.319 | 1216.6 |
| w250_o50 | test | 12 | 0.167 | 0.500 | 0.583 | 0.312 | 1225.0 |
| w500_o100 | dev | 12 | 0.333 | 0.500 | 0.583 | 0.424 | 2379.1 |
| w500_o100 | test | 12 | 0.417 | 0.750 | 0.833 | 0.586 | 2421.3 |

### By query kind

| Configuration | Split | Kind | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| w250_o50 | dev | direct | 6 | 0.167 | 0.500 | 0.500 | 0.306 |
| w250_o50 | dev | paraphrase | 6 | 0.167 | 0.500 | 0.500 | 0.333 |
| w250_o50 | test | direct | 6 | 0.167 | 0.333 | 0.500 | 0.292 |
| w250_o50 | test | paraphrase | 6 | 0.167 | 0.667 | 0.667 | 0.333 |
| w500_o100 | dev | direct | 6 | 0.167 | 0.333 | 0.500 | 0.264 |
| w500_o100 | dev | paraphrase | 6 | 0.500 | 0.667 | 0.667 | 0.583 |
| w500_o100 | test | direct | 6 | 0.333 | 0.667 | 0.833 | 0.533 |
| w500_o100 | test | paraphrase | 6 | 0.500 | 0.833 | 0.833 | 0.639 |

B has higher dev Hit@5 (0.583 versus 0.500) and MRR@5 (0.424 versus 0.319), so it is the better retrieval candidate on this **dev** set. Its top five contain about 2,379 whitespace words on average versus 1,217 for A; the configurations return different amounts of text. The test metrics are reported as held-out observations, not tuning targets. These 24 internally curated questions are too small to establish general retrieval quality.

## Concrete rankings and errors

Each cell lists the top five as `story number / chunk number`; `*` marks a chunk fully covering the gold evidence. An unmarked chunk from the correct story can still miss the precise passage or cover only part of its interval.

| Question | Gold story and interval | A top five | B top five | Finding |
| --- | --- | --- | --- | --- |
| q09: What society did the initials K. K. K. signify? | 05 [214650, 214693) | 05/28*, 05/14, 05/13, 05/23, 05/9 | 05/14*, 05/7, 05/5, 05/9, 05/12 | Both hit at rank 1; the K. K. K. expansion is in the returned passage. |
| q17: What material did the engineer’s client claim his hydraulic press compressed? | 09 [385602, 385824) | 09/15, 09/16*, 09/25, 09/26, 09/12 | 09/8*, 10/15, 09/13, 10/6, 10/5 | A hits at rank 2; B hits at rank 1 for the fuller’s-earth claim. |
| q13: To which noblewoman did the jewel found in the goose belong? | 07 [285643, 285689) | 07/14, 07/33, 07/34, 07/13, 07/4 | 07/7, 07/6*, 07/17, 07/2, 07/16 | A returns story 7 passages but misses the exact Countess quote; B covers it at rank 2. |
| q01: Where was Irene Adler’s photograph hidden? | 01 [38918, 39002) | 01/43, 01/40, 01/39, 01/41, 01/16 | 01/22, 01/21, 01/8, 01/19, 01/20 | Both return story 1 passages, but none of the first five fully contains the sliding-panel quote. |
| q08: What about the wound led Holmes to infer a left-handed killer? | 04 [175108, 175242) | 11/28, 04/35, 04/36, 09/35, 09/34 | 09/16, 09/17, 11/14, 04/18, 09/3 | Both miss the left-handedness evidence in the first five; several high-ranked passages are from other stories. |
| q23: Which family member was Violet Hunter hired to impersonate? | 12 [551728, 551935) | 12/50, 12/5, 12/4, 11/11, 12/6 | 12/3, 12/25, 12/2, 11/6, 02/4 | Both return nearby story 12 passages but miss the full explanation identifying Alice. |

## Unsupported queries

All three dev and all three test unsupported queries returned five positive-score candidates in each configuration. They are **excluded** from Hit@k and MRR. For example, q25 asks when Holmes first used a smartphone and q28 asks for Watson’s email password; both return five lexically overlapping passages in A and B, although the book supplies no answer. A positive BM25 score measures shared terms, not answerability or confidence. An answer abstention mechanism is planned for a later phase.

## Reproduction and artifacts

```sh
conda run -n sherlock-rag python -m sherlock_rag.evaluate
conda run -n sherlock-rag python -m sherlock_rag.retrieval --config w250_o50 --query "How did Holmes recognize that Watson was practising medicine again?" --top-k 5
```

The tracked `eval/bm25_results.json` stores full per-question top-five rankings, metrics, BM25 parameters, and SHA-256 values for the question set, processed book, source manifest, chunk manifest, and both selected chunk JSONL files. The local JSONL files are ignored and must be regenerated from the verified source after cloning. No persistent retrieval index, semantic retrieval, answer generation, or UI is included.
