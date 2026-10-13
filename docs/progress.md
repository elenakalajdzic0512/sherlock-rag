# Progress

## Implemented

- **Phase 0:** Created the private `elenakalajdzic0512/sherlock-rag` repository, initialized `main`, and added the initial README and ignore rules. Initial commit: `8250000d31dfa5a828d3e5102645bf745c80c49e`.
- **Phase 1:** Added a minimal setuptools package using the `src/` layout, a Python 3.11 conda environment at `/Users/ladylazarus/miniforge3/envs/sherlock-rag`, an editable installation, development tools, and phase documentation on `chore/project-foundation`.

## Verification

- Python: 3.11.17 at `/Users/ladylazarus/miniforge3/envs/sherlock-rag/bin/python` on macOS arm64.
- Editable install: successful with `conda run -n sherlock-rag python -m pip install --no-index --no-build-isolation --no-deps -e '.[dev]'` after installing pytest 9.1.1 and Ruff 0.17.0 through conda.
- Import from `/private/tmp` without `PYTHONPATH`: resolved to `src/sherlock_rag/__init__.py`.
- `conda run -n sherlock-rag python -m pip check`: no broken requirements.
- `conda run -n sherlock-rag ruff check .`: passed.
- `conda run -n sherlock-rag ruff format --check .`: seven files already formatted.
- `git diff --check`: passed.
- Ignore rules cover the local virtual environments, caches, local secrets, downloaded data, and generated indexes. `.env.example` remains trackable.
- No test suite was added or run in Phase 1; the package has no behavior to test yet.

The initial online `pip install -e '.[dev]'` failed because `files.pythonhosted.org` could not be resolved. The conda plus offline editable installation above completed the setup; this is no longer a blocker.

## Planned

Phases 2–7 cover source preparation, chunking, retrieval, grounded answers and evaluation, question generation, and an interface. See [roadmap.md](roadmap.md).

## Blocked

None currently known.
