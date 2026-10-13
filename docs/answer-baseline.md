# Phase 6b: local answer baseline

## Protocol and provenance

This is a measurement of the existing Phase 6a pipeline, not a tuning run. The 30 original questions remain byte-identical (SHA-256 `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`). Before generation, twelve challenge questions were written and source-reviewed: six details not stated in the book, four false premises, and two requests about other Holmes books. Their frozen JSONL SHA-256 is `cdf59802eeeddde1b3d0430651d999abcc51f5f1186a36b44a8a8bd0fdafe006`. The challenge set was authored after earlier project findings. It is an internal stress test, not an independent final benchmark.

The runner sent **only each query string** to the unchanged local pipeline, in the frozen order, and saved each result before advancing. It loaded the pipeline once. Its checkpoint resumes only when the input/runtime fingerprint matches and its saved case prefix is intact. It never reruns saved cases or selects among attempts. After all 42 calls, `results.json` was frozen at SHA-256 `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721`. The separate implementation-agent review was written only after that freeze; `reviews.json` SHA-256 is `4507af793178a88b23e7139bf698dd8437b00ebee61e5feffe99e4504395cea5`. The results file still contains the raw model attempts, sent passages, dropped IDs, token counts, and timings. No smoke result was overwritten.

The recorded fingerprint SHA-256 is `115287d5caac7917599e324dcd759836aeb0596d69542e6090d2ed21d83ed589`. Its full fields are in `results.json`: hashes of both question files, retrieval and generation configuration, `local-answer-v1` prompt (`5c7d37c1fe34b4e816c4ec5e332efac37049316a5f56e9d7ca140db0fc356267`), pipeline code (`a7fb59ec66557c598c2253786518b48ff5c24a4a4ec6f2975f9f2381b1a12946`), source manifest and raw/processed source, selected 525-chunk JSONL and index metadata/vectors; semantic A, top five; Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`; MPS bfloat16; greedy decoding, 6144 input and 768 new-token budgets; and installed torch, transformers, sentence-transformers, NumPy, accelerate, and safetensors versions. Model load took 5.57 seconds in this run.

**Historical fingerprint limit:** `baseline_v1` predates fingerprint schema v2. Its `pipeline_code_sha256` covers `rag.py` only, so a change to `semantic.py` or another imported project module would not have invalidated a partial legacy checkpoint. The frozen raw result and its original fingerprint have not been changed or migrated. The post-run review command still validates and scores them without loading the model. New run IDs use schema v2, which records deterministic relative paths and content hashes for all package Python source, including the runner and retrieval dependencies; missing runtime modules fail before generation. Cache and generated directories are excluded. A current-code run cannot silently resume this legacy run because the fingerprints differ.

## Automatic results

These numbers are pipeline outputs and structural checks, not semantic entailment. `error` is never counted as a refusal. `answered` means citations mechanically resolved, not that the claim is true.

| Set | Questions | Answered | Abstained | Error | Final invalid output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original answerable | 24 | 15 | 3 | 6 | 6 |
| Original unsupported | 6 | 0 | 6 | 0 | 0 |
| Challenge: unmentioned detail | 6 | 0 | 6 | 0 | 0 |
| Challenge: false premise | 4 | 0 | 3 | 1 | 1 |
| Challenge: other book | 2 | 0 | 2 | 0 | 0 |
| **All** | **42** | **15** | **20** | **7** | **7** |

Reason codes across all 42: `answered` 15, `insufficient_evidence` 20, and `invalid_model_output` 7. No other technical error or context-budget error occurred. All 42 reached generation. First-pass mechanical success was **20/42 (47.6%)**; **22/42 (52.4%)** used the one allowed repair attempt, and **7/42 (16.7%)** remained invalid. In the 24 answerable cases, only 3/24 passed on the first attempt; 21/24 required repair and 6/24 still failed. The run used 345.39 seconds of model generation (8.22 seconds/case), plus 5.57 seconds to load the model, and generated 5,910 output tokens (140.7/case, including both attempts). Final-attempt input lengths were 2,091–2,370 tokens, below the 6,144-token input budget. Fifteen displayed citations had raw quote lengths 34–922 characters, mean 183.6. These are generation timings, not complete end-to-end retrieval times.

Recomputing `metrics.json` from the frozen results and reviews under the corrected metric code retained every baseline status, reason, success numerator/denominator, timing, token total, and content rating above. All 42 baseline cases have recorded attempt, token, and generation-time fields; none is a `runner_error`. The metric output now makes measurement coverage explicit for each field. For future runner-level exceptions, missing values stay missing, totals/means use only recorded values, and the exception counts as a technical error rather than a refusal or successful answer.

## Implementation-agent content review

The implementation agent inspected every question, displayed result, cited words, full sent passages, and any raw failure, then compared the original cases to the unchanged source-grounded gold and challenges to their pre-run source review. Ratings in `reviews.json` are `true`, `false`, or `null`, with a rationale per case. For every `context_sufficient=true` rating, the review names a sent `source_id` and exact words from that passage. A missing gold interval alone was not treated as proof of insufficient context: for example, `q02`, `q16`, and `q18` were checked for other usable evidence. These ratings are not independent human adjudication or an automated entailment certificate; runtime `semantic_support_verified` remains `null`.

| Original answerable metric | Numerator / denominator | Rate |
| --- | ---: | ---: |
| Correct **and** all displayed claims supported, all questions | 10/24 | 41.7% |
| Correct among displayed answers | 11/15 | 73.3% |
| Displayed answer with at least one unsupported claim | 5/15 | 33.3% |
| Supported correct answer when sent context was sufficient | 10/17 | 58.8% |
| Unjustified abstention when sent context was sufficient | 0/17 | 0% |

For the remaining seven of 24, the sent context did not establish the requested answer. The three original answerable refusals (`q11`, `q20`, `q23`) were justified by their sent passages, though the book contains their answers. No answerable refusal had sufficient sent context under this review. Of the 24, nine correctness/support ratings are `null` because three refused and six ended in error; they were **not** silently omitted from the 24-question success denominator. Eleven of fifteen displayed answers were judged atomic; twelve of fifteen had acceptably short quotes. The other `null` field counts, including inapplicable refusal and premise-correction ratings, are explicit in `metrics.json`.

| Unsupported/challenge set | Strict refusals | Invalid output | Other technical errors | Unsupported displayed claim | Supported premise correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original unsupported | 6/6 | 0 | 0 | 0 | 0 |
| Unmentioned detail | 6/6 | 0 | 0 | 0 | 0 |
| False premise | 3/4 | 1 | 0 | 0 | 1 |
| Other book | 2/2 | 0 | 0 | 0 | 0 |

The one supported premise correction is `c07`: the result is formally `abstained`, but its reason correctly says a swamp adder, not a dog, killed Roylott, and sent passage C2 supports it. This positive correction appears in the refusal reason without a formal resolved citation. `c08` attempted a goose correction in raw output but remained mechanically invalid; it is counted as an error, not a refusal or displayed correction. A future valid `answered` correction would also count as good behavior if its claim and citation were supported.

## Failure analysis with examples

| Case and question | Status and displayed answer | Cited text or sent evidence | Review |
| --- | --- | --- | --- |
| `q01` Where was Irene Adler's photograph hidden? | `error / invalid_model_output`; nothing displayed. | Raw attempts cited C2: “If the former, she had probably transferred the photograph to his keeping.” | The sliding-panel passage was not retrieved. The model promoted Holmes's conditional speculation about Norton into a categorical hiding-place claim. The validator rejected `answered` with a nonempty `reason`; it did **not** detect the semantic error. An exact quote from this speculative passage could pass the mechanical check with a false claim. |
| `q03` What work was Jabez Wilson assigned? | `answered`; a vacancy and £4 weekly nominal services. | C2 advertisement for “purely nominal services.” | The copying assignment was absent from sent passages. The output answered a related employment question, not the work asked for. |
| `q04` Why arrange Wilson's League job? | `answered`; move him out of the way **so they could commit a robbery**. | C1: “get this not over-bright pawnbroker out of the way”. | The requested core answer is correct. Its extra robbery clause has no separate support in the cited words, so all-claims-supported is false. |
| `q07` What word did dying McCarthy try to say? | `answered`; “rat.” | C4 reports the son caught only “a rat”. | The complete word was *Ballarat* in a different passage. Retrieval missed it and the model treated a partial hearing as the intended word. |
| `q09` What did K. K. K. signify? | `answered`; Ku Klux Klan plus formation detail. | C1 contains the name and history in a 922-character raw citation. | Factually supported, but the citation is far longer than needed and the claim joins two facts. |
| `q14` What was the hotel attendant's real name? | `error / invalid_model_output`; nothing displayed. | Sent C2: “my real name is James Ryder.” | Retrieval supplied enough evidence; malformed raw JSON/nonempty reason prevented a valid display after repair. |
| `q18` What injury did Victor Hatherley discover? | `answered`; he “discovered a severed human thumb.” | C2 says investigators discovered a thumb on a window-sill; C3 identifies Hatherley's thumb. | The answer misattributes the later discovery to Hatherley and does not state his injury plainly. The available context was sufficient by combining the sent evidence. |
| `q24` What salary did Rucastle **first** offer Violet Hunter? | `answered`; £120. | C3 discusses the increased £120 salary. | The book's first offer was £100 and was absent from the sent context. The cited later amount does not support “first.” |
| `c07` What breed of dog killed Roylott? | `abstained`; reason corrects dog to swamp adder. | Sent C2: “It is a swamp adder!” | Supported correction inside a refusal; counted separately from strict-refusal rate. |
| `c08` Which turkey held the blue carbuncle? | `error / invalid_model_output`; nothing displayed. | Sent C2 calls it a goose; raw attempts tried to correct it. | The second attempt still used a quote not found exactly in C4. No correction was displayed. |
| `c11` What happened at Reichenbach Falls? | `abstained`; no answer displayed. | No passage from *The Final Problem* exists in this corpus. | Appropriate boundary against importing knowledge from another Holmes book. |

Across the original answerable questions, the five displayed answers with unsupported claims were `q03`, `q04`, `q07`, `q18`, and `q24`. Four displayed answers were factually wrong (`q03`, `q07`, `q18`, `q24`). The six answerable invalid outputs were `q01`, `q14`, `q17`, `q19`, `q21`, and `q22`; `c08` adds one challenge invalid output. There were no other technical errors. This separates retrieval misses, sufficient-context generation failures, invalid structure/citations, and unsupported additions. A mechanically resolved citation never establishes that its claim follows from the quoted words.

## Next decisions, not applied in this baseline

1. Improve retrieval recall for answer-bearing passages and inspect ranking for `q01`, `q03`, `q07`, `q11`, `q20`, `q23`, and `q24`; preserve the frozen set for comparison.
2. Test general claim-by-claim support checks that distinguish speculation and qualifiers from categorical assertions and reject added facts without their own evidence. Avoid name- or question-specific rules.
3. Improve structured output reliability and citation brevity with a separate, versioned experiment; retain raw attempts and report its own costs and quality trade-offs.

Question generation and the interface are outside this phase. A quality-improvement task should be reviewed separately before changing the existing baseline.
