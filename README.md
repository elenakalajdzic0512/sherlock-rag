# sherlock-rag

A book-grounded retrieval-augmented generation (RAG) project based on Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, [Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661).

**Status:** Phase 1 Python project foundation is ready for review. Book ingestion and RAG are not implemented yet.

## Planned features

- Search the book and answer questions grounded in its text.
- Cite real passages from the source.
- Decline to answer when the evidence is insufficient.
- Generate questions from the book.
- Evaluate retrieval and answer quality.
- Provide a simple user interface.

See [the roadmap](docs/roadmap.md) for phase criteria and [progress](docs/progress.md) for verified results.

## Development setup

Use a dedicated Python 3.11 conda environment:

```sh
conda create -n sherlock-rag python=3.11 pip
conda run -n sherlock-rag python -m pip install -e '.[dev]'
```

On this machine, PyPI's file host was unavailable during Phase 1. The working installation used conda-forge for development tools and the local build backend for the editable package:

```sh
conda install -n sherlock-rag pytest ruff
conda run -n sherlock-rag python -m pip install --no-index --no-build-isolation --no-deps -e '.[dev]'
```

Run checks with the project environment:

```sh
conda run -n sherlock-rag python --version
conda run -n sherlock-rag python -m pip check
conda run -n sherlock-rag ruff check .
conda run -n sherlock-rag ruff format --check .
git diff --check
```

The editable installation makes `sherlock_rag` importable from outside this repository without setting `PYTHONPATH`.
