# Project decisions

- **Corpus:** Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, Project Gutenberg #1661. The planned source format is Gutenberg's UTF-8 TXT. The original was downloaded and validated in Phase 2; its bytes are kept unchanged in ignored local data.
- **Python:** Use Python 3.11 in a dedicated `sherlock-rag` conda environment.
- **Packaging:** Use a `src/` layout with setuptools. Install the project in editable mode for development.
- **Dependencies:** Add them only when a phase needs them. Phase 1 has no runtime dependencies; its development extra contains pytest and Ruff.
- **Retrieval:** Phase 4 uses an in-memory, standard-library BM25 lexical baseline over verified `chunk.text` only, with k1=1.2, b=0.75, positive Robertson IDF, and Unicode casefold tokenization. The gold question set was frozen before measurement. B leads on dev retrieval metrics but returns nearly twice the top-five text. Semantic and hybrid methods are Phase 5 experiments; no final method is selected.
- **Models and services:** The LLM provider and embedding model are undecided.
- **Chunking:** Use deterministic sliding windows within each story. Sizes of 250/50 and 500/100 mean `\S+` whitespace-delimited words and overlap, not model tokens. Preserve exact book substrings and source-line provenance. The embedding model, LLM, and their tokenizer remain undecided.
