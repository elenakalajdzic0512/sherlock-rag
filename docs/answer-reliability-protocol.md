# Phase 6c1: frozen single-candidate protocol

Recorded before the first candidate generation on 2026-10-14. This is an internal regression comparison: the original test questions and the challenge questions have already been seen during earlier development. No prompt iteration follows inspection of this run.

## Candidate and immutable inputs

- Run ID: `answer_v2_candidate`; explicit CLI selection: `--generation-variant answer-v2`. Omission retains v1.
- Prompt: `prompts/local_answer_v2.txt`, SHA-256 `9f7e4d13b5d08782b4274899861d5766c5705bf38665fc246c6a947894c50108`.
- Generation config: `config/generation_answer_v2.json`, SHA-256 `98e68afb0298dc3da2553a3e48c9160fc50517d2f5bf5d8100e5f26574c87877`.
- Runtime fingerprint schema v2 SHA-256: `56b872ab5bc37c4e7967a7c98caba4d2dc72368970c406c7336f794b943ef2df`. It includes the selected prompt and config, runtime source code, questions, corpus, chunks, index, package versions, and retrieval selection. The tracked source files that contribute to this hash are frozen for this experiment.
- Retrieval config SHA-256 `b71e6e620d09ffe409f636e578faed43bad57cd28599f7be07b2bf94ef919ded`; semantic A, top five, unchanged.
- Original questions SHA-256 `fb2dfa8a302055b9993f06933d4fa15190041ca411b582ad5263336f3192de0b`; challenges SHA-256 `cdf59802eeeddde1b3d0430651d999abcc51f5f1186a36b44a8a8bd0fdafe006`.
- Frozen baseline result SHA-256 `18019b31adec06a9ea061e5794e4ffa87317cb6c7160c70a4f7bacc4621d7721`; baseline review SHA-256 `4507af793178a88b23e7139bf698dd8437b00ebee61e5feffe99e4504395cea5`. Neither is regenerated.
- Same pinned local Qwen model/revision, MPS bfloat16, greedy decoding, 6144 input tokens, 768 new-token limit, and at most one repair. No paid service.

## Measurement and review

Run all 42 cases in frozen order, load retrieval and generation models once, checkpoint each raw result, then freeze results with `results.sha256` before content review. Compare `submitted_chunks` in each candidate result with baseline by source ID, order, text, and all saved metadata. Any mismatch is reported and excluded from a clean prompt-only comparison. Failed generation with no submitted chunks cannot establish equality from saved result alone; inspect its raw attempt/context fields and mark comparison coverage accordingly.

After freezing, separately review every displayed answer and abstention reason for factual accuracy, full support from its cited words or submitted context, qualifiers, premise handling, and excess quote length. Review all outputs that were errors in baseline but become displayed. The review is an implementation-agent judgment, not automatic proof or independent human adjudication.

Report overall and original dev/test separately, with explicit denominators: first-pass mechanical success, repair attempts, terminal errors; supported-correct answers over all original answerable cases; unsupported claims among displayed answers; supported-correct answers when sent context suffices; justified refusals and cited corrections for false premises; quote lengths, input/output tokens, and generation time. Keep challenge categories separate, including false-premise cases. A lower invalid-output count alone is not an improvement unless newly displayed answers are supported. Make a recommendation without switching the default.
