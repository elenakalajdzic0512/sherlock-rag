# Phase 6c1: one frozen prompt candidate

This is an internal regression comparison, not an independent held-out evaluation. The original test set and all challenge cases were already visible in earlier work. The [protocol](answer-reliability-protocol.md), prompt SHA-256, generation config SHA-256, and runtime fingerprint were recorded before the first candidate generation. Prompt v2 was measured once and was not edited after inspecting outputs. The implementation-agent ratings below are separate from the frozen raw results and are not automatic proof of semantic support.

## Reproducibility and preservation

The 42-case `answer_v2_candidate` run completed with one local model load and per-case checkpoints. Its frozen `results.json` SHA-256 is `50b74c80d81607dfdb3b40324f840133949432cb170c0ee376a54b23f141e782`, and the recorded runtime fingerprint SHA-256 is `56b872ab5bc37c4e7967a7c98caba4d2dc72368970c406c7336f794b943ef2df`. Every candidate `submitted_chunks` list equals its baseline counterpart, including source IDs, ranking, full text, and metadata; all `dropped_chunk_ids` also match. This is a clean prompt/config comparison for sent evidence. The model/revision, MPS bfloat16, greedy decoding, token budgets, one-repair limit, questions, and selected semantic A retrieval stayed fixed. Prompt v1, the v1 generation config, and baseline results/review were not modified.

The candidate prompt SHA-256 is `9f7e4d13b5d08782b4274899861d5766c5705bf38665fc246c6a947894c50108`; its config SHA-256 is `98e68afb0298dc3da2553a3e48c9160fc50517d2f5bf5d8100e5f26574c87877`. The baseline result SHA-256 remains `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721`. The candidate review is in `eval/rag_runs/answer_v2_candidate/reviews.json`; computed metrics are in the adjacent `metrics.json`.

## Mechanical and content comparison

`All` includes the 30 original questions and 12 challenges. `Dev` and `test` refer only to the original set (15 each). A supported correct answer requires both answer correctness and full support. An unsupported claim is counted among displayed answers; an answer can be wrong without making an unsupported factual claim, as in q07. Sufficient-context denominators count original answerable cases whose sent passages contain an answer. Quote lengths are characters of resolved cited substrings.

| Metric | All baseline → v2 | Dev baseline → v2 | Test baseline → v2 |
| --- | ---: | ---: | ---: |
| First-pass mechanical success | 20/42 → 40/42 | 5/15 → 14/15 | 4/15 → 14/15 |
| Repair attempts | 22/42 → 2/42 | 10/15 → 1/15 | 11/15 → 1/15 |
| Final invalid model output | 7/42 → 1/42 | 4/15 → 1/15 | 2/15 → 0/15 |
| Supported correct / original answerable | 10/24 → 14/24 | 4/12 → 6/12 | 6/12 → 8/12 |
| Unsupported claim / displayed answer | 5/15 → 5/20 | 2/6 → 2/9 | 3/9 → 2/10 |
| Supported correct / sufficient sent context | 10/17 → 14/17 | 4/7 → 6/7 | 6/10 → 8/10 |
| Displayed answers / refusals / errors | 15 / 20 / 7 → 20 / 21 / 1 | 6 / 5 / 4 → 9 / 5 / 1 | 9 / 4 / 2 → 10 / 5 / 0 |
| Mean quote length, characters | 183.6 → 132.9 | 267.2 → 143.1 | 127.9 → 117.1 |
| Maximum quote length, characters | 922 → 540 | 922 → 540 | 335 → 335 |
| Final-attempt input tokens, total | 95,365 → 100,378 | 34,079 → 35,821 | 33,912 → 35,626 |
| Output tokens, total | 5,910 → 2,584 | 2,832 → 1,060 | 2,299 → 975 |
| Generation seconds, total | 345.39 → 169.91 | 160.79 → 67.61 | 134.66 → 63.18 |

Across all 42, baseline refusals were 20/42 and candidate refusals 21/42. The candidate’s refusal reasons are evidence-safe in 19/21 implementation-agent judgments; c07 and c08 contain context sufficient for a cited correction, yet return uncited refusals. The baseline review marked 20/20 refusals justified and counted one correction in c07’s refusal reason, not a formally cited `answered` correction. Under the candidate’s explicit answered-form correction requirement, neither run produced a valid cited false-premise correction. Candidate c09 is worse: it accepts the false Bible-copying premise and cites only a salary advertisement. This makes raw refusal or correction counts sensitive to the stated rubric; inspect the cases rather than reading them as a single success rate.

## Former errors and remaining failures

Of seven baseline errors, q14, q19, q21, and q22 are now displayed with correct, supported answers. q01 is now displayed but **unsupported**: its quote is explicitly conditional and says only that a photograph *probably* went to Norton, while the answer asserts a location at the Inner Temple. c08 becomes a refusal despite a sent goose passage that could support a correction. q17 still fails exact-quote validation after the repair limit. Thus six former errors become displayed outputs (five answers and one refusal), and each was reviewed for content. q18 remains a displayed but wrong actor assignment: the passage says investigators found a thumb, while the answer says Hatherley discovered it. q03 substitutes a vacancy advertisement for Wilson’s assigned work; q07 repeats a witness account without answering the intended word. q06 gives the correct employer but its chosen quote omits the employer name. The candidate still contains a 540-character quote and unnecessary extra claims in q09 and q10.

## Recommendation

Keep v1 as the default and retain v2 as an explicitly selectable research candidate. V2 improves first-pass structure, shortens output, and raises reviewed supported-correct answers from 10/24 to 14/24 with identical sent passages. It does not reduce the absolute count of displayed answers with unsupported claims (five), and c09 exposes a false-premise regression. The remaining errors require a separately reviewed task. Neither this run nor the implementation-agent review establishes general semantic reliability.
