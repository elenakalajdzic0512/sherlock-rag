# Project decisions

- **Corpus:** Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, Project Gutenberg #1661. The planned source format is Gutenberg's UTF-8 TXT. The original was downloaded and validated in Phase 2; its bytes are kept unchanged in ignored local data.
- **Python:** Use Python 3.11 in a dedicated `sherlock-rag` conda environment.
- **Packaging:** Use a `src/` layout with setuptools. Install the project in editable mode for development.
- **Dependencies:** Add them only when a phase needs them. Phase 1 has no runtime dependencies; its development extra contains pytest and Ruff.
- **Retrieval:** Phase 4 uses an in-memory, standard-library BM25 lexical baseline over verified `chunk.text` only, with k1=1.2, b=0.75, positive Robertson IDF, and Unicode casefold tokenization. Phase 5a adds a pinned local Qwen3-Embedding-0.6B exact-cosine baseline over the same text and frozen gold set. Queries receive one fixed instruction; documents do not. Semantic A has the best semantic dev MRR@5 and a smaller context than B, but no final retrieval method is selected. Hybrid comparison is Phase 5b.
- **Models and services:** Phase 5a pins Qwen3-Embedding-0.6B for retrieval; the LLM provider and answer model remain undecided.
- **Chunking:** Use deterministic sliding windows within each story. Sizes of 250/50 and 500/100 mean `\S+` whitespace-delimited words and overlap, not model tokens. Preserve exact book substrings and source-line provenance. The embedding model, LLM, and their tokenizer remain undecided.

- **Phase 5b selection rule (fixed before hybrid measurement):** Compare all six method/configuration pairs on answerable dev questions only. Select the highest Hit@5; break ties by highest MRR@5, then lowest mean returned top-five whitespace words. If still tied, use this fixed order: BM25 A, BM25 B, semantic A, semantic B, hybrid A, hybrid B. Report test separately; do not tune the 20/60 RRF parameters or the gold set from test results.
- **Phase 5b result:** Semantic A and hybrid B tie at dev Hit@5 7/12 and MRR@5 .500. Semantic A wins the predeclared context tie-break (1229.1 versus 2422.4 returned words) and is selected for Phase 6 in `config/retrieval.json`. Hybrid remains available for comparison.
