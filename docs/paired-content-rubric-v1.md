# Paired content rubric v1

Recorded before the paired re-review of the frozen `baseline_v1` and `answer_v2_candidate` runs. This rubric supersedes their historical `reviews.json` files **for comparison only**. Those files and both raw `results.json` files remain unchanged. The same rules apply to both runs, including false-premise challenges.

## Displayed answers

- `answer_correct`: Does the displayed answer directly and accurately address the question? Evaluate actors, event order, qualifiers, and false premises. This is separate from whether the answer is supported by its citation.
- `all_claims_supported`: Does each complete displayed claim follow from its **own cited passage or passages**? Read the full text of those cited chunks. Immediate context within the same cited chunk may resolve a pronoun, speaker, or incomplete reference in the selected quote. A fact in another, uncited chunk cannot repair an unsupported cited claim. One unsupported clause makes this field false. A true but irrelevant claim may be supported while the answer remains incorrect.
- `citation_quote_sufficient`: Do the selected quote words themselves convey enough evidence for the complete cited claim, including names and referents? This is intentionally stricter than passage-level `all_claims_supported`.
- `citation_quote_precise`: Are the selected quote words focused on the claim without unnecessary surrounding material? A quote can be sufficient but imprecise, or brief but insufficient.
- `atomic_claims`: Does each claim contain one material fact rather than bundling unrelated facts? Record this separately from correctness and support.
- `context_sufficient`: For an original answerable question, do the **sent** passages contain the requested answer, whether or not the model cited it? This measures retrieval opportunity, not citation validity. For a false-premise challenge, use `supported_correction_possible` instead.

## Refusals and false premises

- `reason_adds_unsupported_facts`: Does a displayed refusal reason assert a fact that the sent passages do not support? Merely failing to cite a supported correction does not make the reason factually unsafe. An explanation limited to the absence of requested evidence is safe unless it falsely describes the sent passages.
- `refusal_appropriate`: Is declining the requested answer acceptable given the sent evidence and question? For the frozen false-premise challenges, both a grounded correction and a safe refusal are acceptable; a missed opportunity to correct is recorded separately. An unsupported answer is never an appropriate refusal.
- `supported_correction_possible`: For false-premise challenges, is a specific correction available from the sent passages with a direct source citation? This is an opportunity measure and does not penalize a safe refusal by itself.
- `formal_cited_correction`: Did the displayed **answered** form correct the premise and cite passage evidence? An uncited correction in a refusal reason is not formal.

For errors, there is no displayed answer or refusal to rate; applicable fields are null. Every non-null judgment has a case-specific rationale in the paired review. These are implementation-agent judgments, not automated semantic verification or independent human adjudication. A mechanical consistency check detects divergent ratings for identical claims, selected quotes, and cited passage text; it cannot establish whether either rating is semantically right.
