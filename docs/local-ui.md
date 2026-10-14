# Phase 7b: local Streamlit interface

The app is a local presentation layer over the existing semantic A retrieval and generation v1 pipeline. It does not change model settings, prompts, source data, retrieval rankings, evaluation results, or the question-generation run. The current bank is only `questions/generated_v1/bank_v2.jsonl` (five questions, five of twelve stories). Its bytes and summary are checked against the locked raw results and v2 implementation-agent review before display. The historical six-question bank is not a fallback.

## Runtime and behavior

The installed UI version is **Streamlit 1.65.0**, pinned in `pyproject.toml` as an optional `ui` extra. PyTorch 2.14.1, transformers 4.57.6, sentence-transformers 5.7.0, NumPy 2.4.6 and accelerate 1.15.0 remained unchanged after installation. The implementation follows the official Streamlit documentation for [installation](https://docs.streamlit.io/get-started/installation/command-line), [forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form_submit_button), [resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource), [session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state), and [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

`Pitaj knjigu` calls the existing pipeline only after a nonempty form submission. The reviewed v2 bank, source, A chunks, semantic index and default prompt/config are checked before model loading. `st.cache_resource` retains the pipeline across reruns; a lock rejects a second simultaneous generation request. The result is stored with the exact submitted query. Answered results show the model's C-label claims, original quote, story title, original line numbers and expandable sent passages. Abstentions say only that the retrieved passages are insufficient. Errors use localized messages and never display raw invalid model output as an answer. A reminder asks readers to check the answer in the cited passages; no semantic verification is claimed.

`Pitanja iz knjige` verifies the v2 bank without loading either ML model. It displays five cards with story filtering, reveals the reviewed expected answer and source proof on request, and transfers a question to the RAG form without submitting it. The selector includes all twelve stories; an uncovered story shows an explicit absence message and no rejected replacement.

## Verification

- Offline Streamlit AppTest used fake pipelines to cover bank browsing without model load, empty questions, answered/abstained/error rendering, citation binding, reruns, filtering, question transfer and concurrency. The bank loader test rejects tampered v2 bytes.
- The app was started on `127.0.0.1:8501` with the real local environment. The browser showed all five v2 cards and their English questions, including absence of historical `qg-v1-10-3`. The Serbian layout was inspected at laptop width and a narrower viewport.
- Real-model question `What kind of creature caused Roylott's death?` returned **answered**: “A swamp adder caused Roylott's death [C1]”. The UI showed *THE ADVENTURE OF THE SPECKLED BAND*, original source lines 7844–7845, and the original quote beginning “It is a swamp adder!”; the full C1–C5 submitted passages were expandable.
- In the same server process, `In what year did Sherlock Holmes use a smartphone?` returned **abstained**. The UI said the retrieved passages were insufficient, without claiming that the entire book lacks an answer. Server output showed one model checkpoint load for the two submissions; the second response reused the cached pipeline. No new questions or evaluation artifacts were generated.

These are functional and visual smoke checks, not a new answer-quality evaluation. Existing answer-support limitations remain those documented in earlier phases. The app was not deployed publicly and has no authentication.
