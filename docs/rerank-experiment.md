# Phase 6c2: one local reranking experiment

This is an internal regression comparison on previously seen questions and challenges. The [protocol](rerank-protocol.md) froze one A/250-50 candidate, the model revision, tie rule, token limit and dev gate before scoring. No alternate model or parameter was tried after results were viewed. The default remains semantic A with generation v1.

## Retrieval and runtime

The pinned [MS MARCO MiniLM-L6-v2 cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, scored full `(question, original chunk.text)` pairs. BM25 and semantic each supplied 20 candidates; the union had at most 40. Their raw scores remained separate. The cross-encoder logit ordered candidates with source-order tie breaks. It is a ranking signal, not evidence of answerability or semantic support.

The pinned tokenizer and model both reported 512 positions. Every pair was measured with special tokens and no truncation; the maximum was **432 tokens** across both the 30-question retrieval evaluation and the 42-case answer run. No pair exceeded the limit. Both embedding and reranker inference used MPS; the reranker passed a real MPS forward pass, with no CPU fallback. Reranker load took **11.71 s**. Mean per-answerable-question BM25/semantic/rerank times were **0.0008/0.0261/0.1620 s** on dev and **0.0007/0.0250/0.1050 s** on test. These wall times are local and hardware specific.

| Split | Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Union gold hit | Final gold hit | Mean top-five words |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev (12 answerable) | Semantic A | 5/12 | 7/12 | 7/12 | 0.500 | — | 7/12 | 1229.1 |
| Dev (12 answerable) | Reranked A | 4/12 | 8/12 | 8/12 | 0.486 | 11/12 | 8/12 | 1230.3 |
| Test (12 answerable) | Semantic A | 3/12 | 5/12 | 7/12 | 0.375 | — | 7/12 | 1208.2 |
| Test (12 answerable) | Reranked A | 6/12 | 7/12 | 7/12 | 0.542 | 12/12 | 7/12 | 1225.0 |

Gold remains the original right-story, whole-evidence-interval criterion. It was never edited after results. Reranking recovered a final gold hit for **q07, q11 and q20**, and lost it for **q13 and q22**. The union contained gold for 23/24 answerable questions, but only 15/24 reached the final five. q13 illustrates the distinction: its marked interval was missed, while separate sent passages still supported the Countess as owner. Such alternative evidence is discussed separately and never credited retroactively as a gold hit. The dev Hit@5 increase from 7/12 to 8/12 met the declared gate, so one generation run followed. The test figures were not used to tune or select.

## One generation-v1 run

`rerank_v1_candidate` completed all 42 cases and was locked at SHA-256 `4f6f71cccd3c7dba6d16e159373323816f6c8c374e6aee5fb87d68f96628dfef`, under fingerprint `1a17f7b8e49c1c96c6a0b3952222cd6f2e82e828350abff2966fe12f5db79fea`. The fingerprint records the explicit retrieval variant, pinned reranker model/tokenizer revision and config, BM25 chunk source, semantic vector index, generation v1, questions, corpus and runtime source hashes. The local generative model loaded once on MPS/bfloat16; its revision, prompt, greedy decoding, 6144/768 token budgets and one repair attempt are unchanged. Every result retains source ranks/scores, full reranker candidate union, final submitted passages, raw attempts, token counts and generation time. **All 42 sent contexts differ from baseline**, so this is a retrieval-plus-generation comparison, not a clean prompt comparison.

| Measure, all 42 | Baseline v1 | Reranked v1 |
| --- | ---: | ---: |
| First-pass mechanical success | 20/42 | 22/42 |
| Repair attempts | 22 | 20 |
| Displayed answers / refusals / errors | 15 / 20 / 7 | 14 / 22 / 6 |
| Supported correct answers, original answerable | 10/24 | 10/24 |
| Supported correct when sent context sufficient | 10/17 | 10/19 |
| Displayed answers with unsupported claims | 5/15 | 4/14 |
| Safe refusal reasons | 19/20 | 17/22 |
| Appropriate refusals | 20/20 | 19/22 |
| Selected quote sufficiency / precision | 6/15 / 11/15 | 5/15 / 11/15 |
| Output tokens / generation time | 5,910 / 345.39 s | 5,499 / 308.38 s |
| Mean selected quote length | 183.6 characters | 131.9 characters |

These content ratings follow [paired rubric v1](paired-content-rubric-v1.md) through separate [candidate ratings](../eval/paired_reviews/rerank_v1_rubric_v1.json) and an offline [comparison checker](../eval/rerank_paired_review.py). Historical baseline ratings were reused exactly; every reranked displayed answer and refusal reason was inspected against its *own* sent context. The checker verified 42 aligned cases, both result hashes, the common rubric, three identical claim/evidence groups with no differing ratings, and all citation text. It found 42 changed sent contexts. It checks internal consistency, not semantic truth. Ratings are implementation-agent judgments, not independent adjudication.

By original split, supported-correct answers remain **4/12 dev** and **6/12 test**. Dev displayed unsupported claims are **2/6** for each run; test drops from **3/9** to **1/7**, while test refusals rise from four to six. The newly available gold evidence for q07 produces a supported answer, but q11 still ends in invalid JSON and q20 incorrectly refuses despite seeing Frank at the altar rails. q01 also incorrectly refuses while the sent C1 passage places Adler’s photograph behind the small sliding shutter. q24 answers with a later £120 offer as if it were the *first* £100 offer. c04 gives “Mrs. Stoner” as a given name. q13’s answer is factually correct but its cited C4 passage alone does not link the jewel to the goose; that link is in uncited C1. These distinctions prevent a retrieval gain from being reported as an answer-quality gain.

Challenge categories: original unsupported questions remain six refusals; unmentioned-detail challenges show five safe refusals plus the unsupported c04 answer; all four false-premise challenges refuse safely, with two opportunities for a cited correction (c07 snake, c08 goose) and zero formal cited corrections; both outside-corpus challenges refuse. A missed correction opportunity is separate from an unsafe refusal reason. Some original-answerable refusal reasons contain unsupported descriptions of the sent evidence (notably q01 and q20), which lowers refusal safety despite the retrieval improvement.

## Recommendation and limits

**Keep semantic A as the default.** The fixed reranker improved dev gold Hit@5 by one question and test MRR@5, but did not improve supported-correct answers (10/24), added local retrieval latency and worsened the reviewed safety of some refusal reasons. Retain the reranker behind its explicit experiment flag for further diagnosis. The small, previously seen benchmark has one marked gold interval per answerable question; alternate legitimate evidence and content judgments require separate human review. No new prompt, generative model, automatic semantic verifier, question generation or interface was introduced.

The full data are [retrieval rankings](../eval/rerank_results.json), [locked answers](../eval/rag_runs/rerank_v1_candidate/results.json), and [paired ratings](../eval/paired_reviews/rerank_v1_rubric_v1.json).
