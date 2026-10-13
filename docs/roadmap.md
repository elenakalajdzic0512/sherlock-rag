# Roadmap

Each phase is scoped and reviewed separately. Later criteria describe planned work, not current functionality.

| Phase | Goal | Verifiable completion criteria | Status |
| --- | --- | --- | --- |
| 0 | Establish the repository. | Private GitHub repository, `main`, initial README and ignore rules, clean push. | Complete |
| 1 | Establish the Python project foundation. | Dedicated Python 3.11 environment, editable install, import from outside the repository, dependency and Ruff checks, documentation, and a reviewable pull request. | Complete |
| 2 | Acquire and prepare the source text. | Record the Gutenberg #1661 UTF-8 TXT source, retain an unchanged raw copy, produce a separate processed artifact, and verify extraction against the source. | Ready for review |
| 3 | Build traceable chunks. | Produce stable chunks with story and source-location metadata; verify coverage, boundaries, and reproducibility. | Planned |
| 4 | Add a lexical retrieval baseline. | Implement BM25 retrieval and measure it on documented book-grounded queries. | Planned |
| 5 | Compare semantic and hybrid retrieval. | Implement candidate methods and compare them with the same evaluation set and metrics; document the selected approach. | Planned |
| 6 | Add grounded answers and evaluation. | Answers cite real source passages, abstain when evidence is insufficient, and pass documented retrieval and answer evaluations. | Planned |
| 7 | Add question generation and a simple interface. | Generate verifiable questions from source text and expose supported search and answer flows through a usable interface. | Planned |
