# Project decisions

- **Corpus:** Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, Project Gutenberg #1661. The planned source format is Gutenberg's UTF-8 TXT. The book has not been downloaded yet.
- **Python:** Use Python 3.11 in a dedicated `sherlock-rag` conda environment.
- **Packaging:** Use a `src/` layout with setuptools. Install the project in editable mode for development.
- **Dependencies:** Add them only when a phase needs them. Phase 1 has no runtime dependencies; its development extra contains pytest and Ruff.
- **Retrieval:** BM25, semantic retrieval, and hybrid retrieval are planned experiments. No method is assumed to be best in advance.
- **Models and services:** The LLM provider and embedding model are undecided.
