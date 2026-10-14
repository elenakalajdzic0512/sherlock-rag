# Phase 6c2: frozen local reranking protocol

Recorded before the first reranker score or retrieval evaluation. This is one fixed candidate and an internal regression comparison: the question and challenge sets have already been seen.

## Fixed candidate

- Configuration A (`w250_o50`) only. Retrieve the top 20 from BM25 and the top 20 from the existing pinned semantic A index; merge by `chunk_id` (at most 40). Neither source score is added to the other.
- Score each `(question, original chunk.text)` pair with [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), pinned model **and tokenizer** to Hugging Face commit `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The [Sentence Transformers CrossEncoder documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html) describes query/passage scoring and sorting, not answerability verification.
- Model configuration reports `max_position_embeddings=512`; the pinned tokenizer reports `model_max_length=512`. Measure every complete pair with special tokens and `truncation=False`; stop evaluation on any pair over 512. Inference also uses `truncation=False`.
- Rank by descending raw sequence-classification logit, then ascending `story_index`, `chunk_index`, `chunk_id`. Return five. Preserve both source ranks/scores, reranker score, final rank and all candidates. Batch size 8; one model load per measurement. Prefer MPS after a real forward-pass smoke check; record CPU fallback if needed. Measure model load and reranking time separately.
- Experimental config: `config/retrieval_rerank.json`; it is selected only with `--retrieval-variant rerank`. Omission preserves semantic A and generation v1. Generation settings are unchanged.
- Frozen experimental config SHA-256: `038981273a4b39183e5c65abdec520025ec538191311af9139da43938e2cc058`.

## Retrieval measurement and decision

Evaluate the original 30 questions once. Compare with frozen semantic A using the *same* gold rule: a chunk in the right story must fully contain one annotated evidence interval. Unsupported questions are excluded from Hit@k/MRR. Report dev and test separately: Hit@1/3/5, MRR@5, union gold hit, final-five gold hit, context words, BM25/semantic/rerank times, and question-level gains/losses. Alternative valid evidence may be described separately and does not modify gold.

Continue to one 42-case generation-v1 run only if dev Hit@5 is higher than semantic A, or if dev Hit@5 ties and MRR@5 is higher. Test is not used for selection. If the gate fails or any pair exceeds 512 tokens, stop without a generative run. A better retrieval metric alone is not evidence of better answers.

If the gate passes, use one new run ID, keep all raw attempts, candidate union, submitted context and costs, freeze results before review, then compare against `baseline_v1` with paired rubric v1. Assess context sufficiency for both changed contexts. No prompt, model, or retrieval tuning after results are viewed.

## Gate decision recorded before generation

The one retrieval measurement passed the predeclared dev rule: Hit@5 increased from 7/12 to 8/12. The reranked generation run is therefore `rerank_v1_candidate`, with generation v1 unchanged. Its runtime fingerprint SHA-256 before model load is `1a17f7b8e49c1c96c6a0b3952222cd6f2e82e828350abff2966fe12f5db79fea`; this includes both source index hashes, the selected retrieval config and revision, generation config/prompt, cases, corpus and package source. No content ratings were inspected to make this gate decision.
