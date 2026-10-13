# sherlock-rag

A book-grounded retrieval-augmented generation (RAG) project based on Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, [Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661).

**Status:** Phase 2 source ingestion is ready for review. Chunking, retrieval, answers, evaluation, question generation, and the interface are not implemented yet.

## Planned features

- Search the book and answer questions grounded in its text.
- Cite real passages from the source.
- Decline to answer when the evidence is insufficient.
- Generate questions from the book.
- Evaluate retrieval and answer quality.
- Provide a simple user interface.

See [the roadmap](docs/roadmap.md), [progress](docs/progress.md), and [corpus audit](docs/corpus-audit.md).

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

## Acquire and prepare the book

Run from any directory with the project environment:

```sh
conda run -n sherlock-rag python -m sherlock_rag.ingest
```

The command downloads and validates the Gutenberg TXT, preserves its original bytes in `data/raw/sherlock_holmes.txt`, and extracts the narrative into `data/processed/book.txt`. Both text files are ignored by Git. It writes a tracked [source manifest](data/source_manifest.json) and [corpus audit](docs/corpus-audit.md). A different candidate source never silently replaces an existing original. If a network connection fails, it uses the validated existing original or checks only `~/Downloads/pg1661.txt` and `~/Downloads/sherlock_holmes.txt`.

To use a manually downloaded TXT instead:

```sh
conda run -n sherlock-rag python -m sherlock_rag.ingest --input-file /absolute/path/to/pg1661.txt
```

## Checks

```sh
conda run -n sherlock-rag python -m pytest
conda run -n sherlock-rag ruff check .
conda run -n sherlock-rag ruff format --check .
conda run -n sherlock-rag python -m pip check
git diff --check
```

The editable installation makes `sherlock_rag` importable from outside this repository without setting `PYTHONPATH`.
