# Corpus audit: Project Gutenberg #1661

The acquired original is preserved byte for byte in `data/raw/sherlock_holmes.txt` (ignored by Git). The processed file is `data/processed/book.txt` (also ignored).

## Extraction and traceability

- Source: https://www.gutenberg.org/ebooks/1661 (TXT: https://www.gutenberg.org/cache/epub/1661/pg1661.txt).
- Acquisition method: download; recorded at: 2026-10-13T18:30:26+00:00.
- Raw: 607606 bytes, 12310 lines, 107558 whitespace-delimited tokens, SHA-256 `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0`.
- Processed: 575243 bytes, 11899 lines, 104423 whitespace-delimited tokens, SHA-256 `d46abd0f05ef0a55f321bf3bdb7fac0302e1843afcf0e4840b46a806518ffb6a`.
- Exact source interval: original lines 55–11953, inclusive, numbered from 1.
- Rule: decode UTF-8 strictly, normalize CRLF and bare CR to LF, select that complete interval, and make no other text edits. A final newline is included only if present on the selected last source line.
- Proof: re-extracting the interval from the original with this rule yields the exact processed bytes and SHA-256 above.

The removed prefix contains Gutenberg's preface, the book title page, and the table of contents. The removed suffix contains blank separators, the Gutenberg end marker, and the license. Narrative paragraphs, punctuation, case, italic markers, and internal headings remain unchanged.

## Story starts

| # | Story heading | Processed line | Original line |
| --- | --- | ---: | ---: |
| 1 | I. A SCANDAL IN BOHEMIA | 1 | 55 |
| 2 | II. THE RED-HEADED LEAGUE | 1081 | 1135 |
| 3 | III. A CASE OF IDENTITY | 2117 | 2171 |
| 4 | IV. THE BOSCOMBE VALLEY MYSTERY | 2868 | 2922 |
| 5 | V. THE FIVE ORANGE PIPS | 3922 | 3976 |
| 6 | VI. THE MAN WITH THE TWISTED LIP | 4759 | 4813 |
| 7 | VII. THE ADVENTURE OF THE BLUE CARBUNCLE | 5801 | 5855 |
| 8 | VIII. THE ADVENTURE OF THE SPECKLED BAND | 6729 | 6783 |
| 9 | IX. THE ADVENTURE OF THE ENGINEER’S THUMB | 7868 | 7922 |
| 10 | X. THE ADVENTURE OF THE NOBLE BACHELOR | 8777 | 8831 |
| 11 | XI. THE ADVENTURE OF THE BERYL CORONET | 9754 | 9808 |
| 12 | XII. THE ADVENTURE OF THE COPPER BEECHES | 10820 | 10874 |

The first story's opening narrative line includes: “To Sherlock Holmes she is always _the_ woman. I have seldom heard him” The last story's final narrative line is: “Walsall, where I believe that she has met with considerable success.”

## Known limitations

This phase preserves line wrapping and paragraphs as supplied by Gutenberg. It does not correct possible transcription errors, assign paragraph identifiers, or perform chunking. Word counts use `str.split()` and are not linguistic token counts. Source line numbers refer to the captured original and can change if Gutenberg updates its TXT.
