# Roadmap

Each phase is scoped and reviewed separately. Later criteria describe planned work, not current functionality.

| Phase | Goal | Verifiable completion criteria | Status |
| --- | --- | --- | --- |
| 0 | Establish the repository. | Private GitHub repository, `main`, initial README and ignore rules, clean push. | Complete |
| 1 | Establish the Python project foundation. | Dedicated Python 3.11 environment, editable install, import from outside the repository, dependency and Ruff checks, documentation, and a reviewable pull request. | Complete |
| 2 | Acquire and prepare the source text. | Record the Gutenberg #1661 UTF-8 TXT source, retain an unchanged raw copy, produce a separate processed artifact, and verify extraction against the source. | Complete |
| 3 | Build traceable chunks. | Produce stable chunks with story and source-location metadata; verify coverage, boundaries, and reproducibility. | Complete |
| 4 | Add a lexical retrieval baseline. | Implement BM25 retrieval and measure it on documented book-grounded queries. | Complete |
| 5a | Add local semantic retrieval. | Build pinned, source-verified embedding indexes and compare exact cosine retrieval with BM25 on the fixed question set. | Complete; PR #5 merged |
| 5b | Compare hybrid retrieval. | Combine candidate retrieval methods and compare them with the same evaluation set and metrics. | Complete; PR #6 merged |
| 6a | Add local grounded answers. | Run a pinned local model on selected retrieval passages; resolve citation offsets and audit a fixed smoke set. | Complete; PR #7 merged |
| 6b | Evaluate answers and abstention. | Measure content support, citation quality, and unsupported-question behavior on a documented full set. | Complete; PR #8 merged |
| 6c1 | Test one prompt-only answer reliability candidate. | Freeze a separate prompt/config and run all 42 cases with unchanged retrieval; review content separately and compare dev/test. | Complete; PR #9 merged |
| 6c2 | Test one local reranking candidate. | Freeze pinned top-20 BM25 and semantic A union with cross-encoder top five; compare retrieval and, only if dev gate passes, review one generation-v1 run. | Complete; PR under review |
| 7 | Add question generation and a simple interface. | Generate verifiable questions from source text and expose supported search and answer flows through a usable interface. | Planned; depends on answer quality review |
