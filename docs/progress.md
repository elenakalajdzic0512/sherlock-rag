# Progress

## Implemented

- **Phase 0:** Private `elenakalajdzic0512/sherlock-rag` repository and initial `main` commit `8250000d31dfa5a828d3e5102645bf745c80c49e`.
- **Phase 1:** Minimal setuptools `src/` package, dedicated Python 3.11 conda environment, editable installation, development tools, and documentation. PR #1 was merged into `main` as `da4eb37c38673a4a46c7e408dcb1ccc2ba432105`.
- **Phase 2:** Standard-library ingestion command, strict Gutenberg #1661 validation, preservation of original bytes, exact narrative extraction, tracked manifest and audit, and offline behavior tests.

## Phase 2 source and verification

- Acquired by download from `https://www.gutenberg.org/cache/epub/1661/pg1661.txt` on `2026-10-13T18:30:26+00:00`.
- Raw original: 607606 bytes, 12310 lines, 107558 whitespace-delimited tokens; SHA-256 `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0`.
- Processed narrative: 575243 bytes, 11899 lines, 104423 whitespace-delimited tokens; SHA-256 `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a`.
- Exact inclusive original interval: lines 55–11953. Independent byte comparison after CRLF/CR-to-LF normalization passed. Twelve story headings were found in order, each with nonempty content; see [corpus audit](corpus-audit.md).
- A repeated run left raw, processed, and manifest SHA-256 values unchanged, including the recorded acquisition date.
- `conda run -n sherlock-rag python -m pytest -q`: 17 passed, offline.
- `conda run -n sherlock-rag ruff check .`: passed.
- `conda run -n sherlock-rag ruff format --check .`: passed.
- `conda run -n sherlock-rag python -m pip check`: no broken requirements.
- `git diff --check`: passed. The raw and processed files are ignored by Git.

## Planned

Phases 3–7 cover chunking, retrieval, grounded answers and evaluation, question generation, and an interface. See [roadmap.md](roadmap.md).

## Blocked

None currently known.
