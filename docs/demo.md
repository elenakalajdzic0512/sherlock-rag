# Local demonstration script (about four minutes)

## Before presenting

Start from a prepared local clone with the verified original, book, A chunks and semantic index. In the project conda environment, run:

```sh
conda run --no-capture-output -n sherlock-rag python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. The first submitted question loads the pinned local models and may take longer; for a timed presentation, submit the Roylott question once before the audience arrives, then leave the same server process running. Do not present cached-model reuse as a cold-start timing result. Do not expose the local server publicly.

## 0:00–0:45 — Architecture

Explain the path: verified Project Gutenberg #1661 TXT → exact extracted narrative → story-bounded 250/50 chunks with source lines → local semantic A index → top-five passages → pinned Qwen generation v1 → mechanically checked JSON claims and citations. The app is a local presentation layer and does not change retrieval, prompt, model or frozen evaluations. The source and result manifests are tracked; original text, chunks, index and model cache are locally recreated.

## 0:45–1:45 — Answer with source citation

In **Pitaj knjigu**, submit `What kind of creature caused Roylott's death?`. Show the generated answer, `[C1]`, the story title *THE ADVENTURE OF THE SPECKLED BAND*, original lines 7844–7845, and the exact quote beginning “It is a swamp adder!” Expand C1 to show the passage actually supplied to the model. Ask the audience to check the answer against the cited text. The citation has verified provenance; semantic support is not automatically proven.

## 1:45–2:25 — Evidence-limited refusal

Submit `In what year did Sherlock Holmes use a smartphone?`. Show the refusal and its wording: the retrieved passages are insufficient for a reliable answer. It does not claim that an answer is absent from the entire book. Distinguish this from a technical error or malformed model output.

## 2:25–3:20 — Reviewed questions

Switch to **Pitanja iz knjige**. Point out the current coverage: five reviewed questions from five of twelve stories. Reveal one expected answer and its source quote. Filter to `THE RED-HEADED LEAGUE` to show the explicit uncovered-story message; no rejected question is substituted. Transfer a covered question to **Pitaj knjigu** and show that the form fills without automatically calling the model. The bank's reviewed expected answer and a new RAG answer are separate outputs.

## 3:20–4:15 — Evaluation and limits

Use [the baseline report](answer-baseline.md) and [the paired comparison](answer-reliability.md) to state the current default baseline: **10/24 supported-correct answers** on original answerable questions under paired rubric v1. The prompt-v2 and reranked runs are historical experiments; semantic A plus generation v1 remain defaults. The fixed test set was already seen during development, and the content ratings were made by an implementation agent, not independent human reviewers. The five-question bank is product content, not an independent benchmark. Close with the [fresh-clone audit](final-audit.md): reproducible source/chunk hashes and local operation were checked on one Mac, reusing the pinned Hugging Face model cache.
