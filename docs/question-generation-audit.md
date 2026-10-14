# Phase 7a: question generation audit

This is one local production-content batch, not a new evaluation benchmark. The [protocol](question-generation-protocol.md) and exact 36-passage selection were recorded before generation. The original evaluation questions, challenges, answers and gold spans were never supplied to the model. The existing RAG defaults and earlier frozen results were not changed.

## Frozen run and provenance

The selected A chunks have SHA-256 `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd`; selection SHA-256 is `6d10a4bf2d0ce1ab9047f6982f1b6efd31be165ca1e103f2c161d95213d383c0`. Prompt SHA-256 is `27d70824e32b508226d61480d5095a86416826b685f2109e29cc07fbb49db23f`, config SHA-256 is `fe346dbf189c3a09aa9b16e795be3f418824a0ead5ba4362c41897aed2b45e6a`, and full runtime fingerprint SHA-256 is `27d12172fa3605184cf8f3381493986737006e7534b1898ba91407773f4d284c`. The locked raw `questions/generated_v1/results.json` SHA-256 is **`1fed79cb75bd77d90269dce030b01a97dd9ad999a72986abb64aa8335976046d`**. The separate implementation-agent `reviews.json` SHA-256 is `1fb9127edff54fabb39d7f916300374fc09662957b8612a0d24c419050181d3f`; reviewed bank SHA-256 is `dbb0b98ff48f9ba0e8ed2e9d5bd3be774e63734a12e42f1d80403ebe4675f99d`.

The pinned Qwen model loaded **once** on MPS as bfloat16 in 4.49 seconds. Thirty-three passages required one generation attempt and three required two. Across 39 attempts, measured generation totaled 182.03 seconds and 4,073 output tokens. The largest measured complete input prompt was 719 tokens, below the 6,144-token budget. No prompt was silently truncated. Output was greedy and capped at 512 new tokens per attempt. No embedding or retrieval model was used.

## Outcomes

| Story | Mechanically valid candidates | Insufficient evidence | Mechanical errors | Duplicates | Content accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 A Scandal in Bohemia | 3 | 0 | 0 | 0 | 1 |
| 02 The Red-Headed League | 2 | 1 | 0 | 0 | 0 |
| 03 A Case of Identity | 3 | 0 | 0 | 0 | 0 |
| 04 The Boscombe Valley Mystery | 2 | 0 | 1 | 0 | 1 |
| 05 The Five Orange Pips | 2 | 0 | 1 | 0 | 1 |
| 06 The Man with the Twisted Lip | 2 | 1 | 0 | 0 | 0 |
| 07 The Adventure of the Blue Carbuncle | 3 | 0 | 0 | 0 | 0 |
| 08 The Adventure of the Speckled Band | 1 | 1 | 1 | 0 | 0 |
| 09 The Adventure of the Engineer's Thumb | 2 | 1 | 0 | 0 | 0 |
| 10 The Adventure of the Noble Bachelor | 2 | 1 | 0 | 0 | 2 |
| 11 The Adventure of the Beryl Coronet | 3 | 0 | 0 | 0 | 0 |
| 12 The Adventure of the Copper Beeches | 3 | 0 | 0 | 0 | 1 |
| **Total** | **28** | **5** | **3** | **0** | **6** |

All 28 mechanically valid candidates received six separate content ratings with rationales in `questions/generated_v1/reviews.json`. Only six passed premise clarity/truth, direct availability, answer correctness, whole-answer citation support, quote sufficiency, and quote precision. The bank covers **5/12 stories**. Stories 02, 03, 06, 07, 08, 09 and 11 have no accepted question. The three mechanical errors and five insufficient-evidence outcomes remain in the locked raw run. No rejected question was silently repaired or replaced after review.

Examples of rejections: one question treated a hypothetical account of Boone as established fact; another treated an alias as a person's real name; another confused Mrs. Toller with Mrs. Rucastle; another lost the qualifier “no more than.” Exact quote containment alone does not certify the premise, answer or support. The review is an implementation-agent judgment, not independent human validation.

## Reviewed examples

| ID | Question | Expected answer | Selected source quote |
| --- | --- | --- | --- |
| `qg-v1-04-1` | What was the signal that Mr. James McCarthy recognized when he heard it near the Boscombe Pool? | Cooee | When about a hundred yards from the pool I heard a cry of “Cooee!” which was a usual signal between my father and myself. |
| `qg-v1-05-1` | What symbol was printed on the lid of the brass box? | the treble K | upon the lid was printed the treble K |
| `qg-v1-12-1` | What is the amount offered to Miss Hunter per year to compensate for any inconvenience at The Copper Beeches? | £120 a year | We are willing to give £ 30 a quarter, or £ 120 a year, so as to recompense you for any little inconvenience which our fads may cause you. |

The exact UTF-8 quotes, original substrings, character offsets and line numbers are in the [reviewed bank](../questions/generated_v1/bank.jsonl). The model provided the displayed quote only; the source offsets and lines were resolved from the verified book. The full rejected and accepted attempts, token counts and times remain in the locked raw result.
