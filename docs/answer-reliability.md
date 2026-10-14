# Phase 6c1: one frozen prompt candidate

This is an internal regression comparison, not an independent held-out evaluation. The original test set and all challenge cases were already visible in earlier work. The [protocol](answer-reliability-protocol.md), prompt SHA-256, generation config SHA-256, and runtime fingerprint were recorded before the first candidate generation. Prompt v2 was measured once and was not edited after inspecting outputs. The **primary content comparison below uses the separate [paired rubric v1](paired-content-rubric-v1.md)** on both frozen runs. These implementation-agent ratings are not automatic proof of semantic support.

## Reproducibility and preservation

The 42-case `answer_v2_candidate` run completed with one local model load and per-case checkpoints. Its frozen `results.json` SHA-256 is `50b74c80d81607dfdb3b40324f840133949432cb170c0ee376a54b23f141e782`, and the recorded runtime fingerprint SHA-256 is `56b872ab5bc37c4e7967a7c98caba4d2dc72368970c406c7336f794b943ef2df`. Every candidate `submitted_chunks` list equals its baseline counterpart, including source IDs, ranking, full text, and metadata; all `dropped_chunk_ids` also match. This is a clean prompt/config comparison for sent evidence. The model/revision, MPS bfloat16, greedy decoding, token budgets, one-repair limit, questions, and selected semantic A retrieval stayed fixed. Prompt v1, the v1 generation config, and baseline results/review were not modified.

The candidate prompt SHA-256 is `9f7e4d13b5d08782b4274899861d5766c5705bf38665fc246c6a947894c50108`; its config SHA-256 is `98e68afb0298dc3da2553a3e48c9160fc50517d2f5bf5d8100e5f26574c87877`. The baseline result SHA-256 remains `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721`. The original `reviews.json` and `metrics.json` under both run directories are **historical reviews and metrics**, preserved byte-for-byte. The new paired ratings are in `eval/paired_reviews/answer_v1_v2_rubric_v1.json`; their derived metrics are in the adjacent `answer_v1_v2_rubric_v1_metrics.json`. Run `python -m eval.paired_review` in the project environment to validate source hashes, equal sent contexts, complete ratings, and identical-claim consistency before recalculating metrics.

## Mechanical and content comparison

`All` includes the 30 original questions and 12 challenges. `Dev` and `test` refer only to the original set (15 each). A supported correct answer requires both answer correctness and support from each claim's **own cited passage**. Immediate context in that passage may resolve a reference; another uncited passage cannot. Selected-quote sufficiency and precision are separate. An unsupported claim is counted among displayed answers; an answer can be wrong without making an unsupported factual claim, as in candidate q07. Sufficient-context denominators count original answerable cases whose sent passages contain an answer. Quote lengths are characters of resolved cited substrings. Content ratings in this table are from paired rubric v1; mechanical measures come directly from unchanged raw results.

| Metric | All baseline → v2 | Dev baseline → v2 | Test baseline → v2 |
| --- | ---: | ---: | ---: |
| First-pass mechanical success | 20/42 → 40/42 | 5/15 → 14/15 | 4/15 → 14/15 |
| Repair attempts | 22/42 → 2/42 | 10/15 → 1/15 | 11/15 → 1/15 |
| Final invalid model output | 7/42 → 1/42 | 4/15 → 1/15 | 2/15 → 0/15 |
| Supported correct / original answerable | 10/24 → 14/24 | 4/12 → 6/12 | 6/12 → 8/12 |
| Unsupported claim / displayed answer | 5/15 → 5/20 | 2/6 → 2/9 | 3/9 → 2/10 |
| Supported correct / sufficient sent context | 10/17 → 14/17 | 4/7 → 6/7 | 6/10 → 8/10 |
| Selected quote sufficient / claims | 6/15 → 11/21 | 4/6 → 6/9 | 2/9 → 5/11 |
| Selected quote precise / claims | 11/15 → 15/21 | 4/6 → 6/9 | 7/9 → 9/11 |
| Factually safe refusal reason / refusals | 19/20 → 21/21 | 4/5 → 5/5 | 4/4 → 5/5 |
| Appropriate refusal / refusals | 20/20 → 21/21 | 5/5 → 5/5 | 4/4 → 5/5 |
| Displayed answers / refusals / errors | 15 / 20 / 7 → 20 / 21 / 1 | 6 / 5 / 4 → 9 / 5 / 1 | 9 / 4 / 2 → 10 / 5 / 0 |
| Mean quote length, characters | 183.6 → 132.9 | 267.2 → 143.1 | 127.9 → 117.1 |
| Maximum quote length, characters | 922 → 540 | 922 → 540 | 335 → 335 |
| Final-attempt input tokens, total | 95,365 → 100,378 | 34,079 → 35,821 | 33,912 → 35,626 |
| Output tokens, total | 5,910 → 2,584 | 2,832 → 1,060 | 2,299 → 975 |
| Generation seconds, total | 345.39 → 169.91 | 160.79 → 67.61 | 134.66 → 63.18 |

The paired review separates refusal safety, refusal appropriateness, opportunity to correct, and a formally cited `answered` correction. Across the four false-premise challenges, **both runs had 2/4 correction opportunities** (c07 and c08) and **0/4 formal cited corrections**. Baseline had three displayed refusals, all factually safe and appropriate, plus one error (c08). Candidate had three factually safe and appropriate refusals, including c07 and c08, plus one unsupported displayed answer (c09). Missing a citable correction does **not** make a refusal reason unsafe. Baseline c07 mentions the swamp adder in its refusal reason but offers no formal citation. Candidate c09 accepts the false Bible-copying premise and cites only a salary advertisement.

| Challenge category | Baseline | Candidate v2 |
| --- | --- | --- |
| Unmentioned detail (6) | 6/6 safe, appropriate refusals | 6/6 safe, appropriate refusals |
| False premise (4) | 3/3 safe, appropriate refusals; 1 error; 2/4 correction opportunities; 0/4 formal corrections | 3/3 safe, appropriate refusals; 1 unsupported answer; 2/4 correction opportunities; 0/4 formal corrections |
| Outside corpus (2) | 2/2 safe, appropriate refusals | 2/2 safe, appropriate refusals |

The paired check finds **seven identical claim/quote/cited-passage groups and zero rating differences**. In q06, both runs contain the identical claim, selected quote, and C1 passage. Both are correct and passage-supported; both selected quotes omit the company name and are therefore insufficient *as quotes alone*. Candidate q10's extra crew claim is not passage-supported under this rubric: its own C2 says “He and the two mates” without identifying Calhoun, and C1 cited for another claim cannot supply the missing referent. This exchanges one historical candidate unsupported-claim judgment (q06) for q10; the aggregate count remains five. The consistency check is a comparison invariant, not a semantic verifier.

The historical individual reviews previously marked baseline q06 supported and candidate q06 unsupported; they also marked candidate c07/c08 refusals as unjustified solely because a correction was available. Those earlier judgments are retained as historical results, not used in the primary table. The paired review additionally flags baseline q23's refusal reason as factually unsafe because it asserts that Hunter was mistaken for someone else, a fact absent from the sent passages. Candidate q23's extra school detail is present in its sent context.

## Former errors and remaining failures

Of seven baseline errors, q14, q19, q21, and q22 are now displayed with correct, supported answers. q01 is now displayed but **unsupported**: its quote is explicitly conditional and says only that a photograph *probably* went to Norton, while the answer asserts a location at the Inner Temple. c08 becomes a safe refusal despite a sent goose passage that could support a correction. q17 still fails exact-quote validation after the repair limit. Six former errors become displayed outputs (five answers and one refusal), and each was reviewed for content. q18 remains a displayed but wrong actor assignment: the passage says firemen found a thumb, while the answer says Hatherley discovered it. q03 substitutes a vacancy advertisement for Wilson’s assigned work; candidate q07 repeats a witness account without answering the intended word. The candidate still contains a 540-character quote and unnecessary extra claims in q09 and q10.

## Recommendation

Keep v1 as the default and retain v2 as an explicitly selectable research candidate. V2 improves first-pass structure, shortens output, and raises reviewed supported-correct answers from 10/24 to 14/24 with identical sent passages. It does not reduce the absolute count of displayed answers with unsupported claims (five), and c09 exposes a false-premise regression. The remaining errors require a separately reviewed task. Neither this run nor the implementation-agent review establishes general semantic reliability.
