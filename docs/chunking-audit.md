# Chunking audit: Project Gutenberg #1661

Phase 3 uses `sliding-word-v1`. A word is a `\S+` match (a non-whitespace run), not a language-model token. Each window stays inside one of the twelve story boundaries obtained by `extract_book`. Character offsets count Python Unicode characters; word and character intervals are zero-based with exclusive ends, while line numbers start at 1 with inclusive ends.

## Real-book results

| Configuration | Chunk size / overlap | Chunks | Min words | Max words | Mean words | JSONL bytes | JSONL SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 250 / 50 | 525 | 86 | 250 | 247.76 | 1,072,037 | `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd` |
| B | 500 / 100 | 265 | 109 | 500 | 489.52 | 901,264 | `e8e886a889f8e1d7916eb236a800d694abb1f5c496ea4114c9e16a5a26eee87e` |

| # | Story | A chunks | B chunks |
| ---: | --- | ---: | ---: |
| 1 | A Scandal in Bohemia | 43 | 22 |
| 2 | The Red-Headed League | 46 | 23 |
| 3 | A Case of Identity | 35 | 18 |
| 4 | The Boscombe Valley Mystery | 48 | 24 |
| 5 | The Five Orange Pips | 37 | 19 |
| 6 | The Man with the Twisted Lip | 46 | 23 |
| 7 | The Adventure of the Blue Carbuncle | 39 | 20 |
| 8 | The Adventure of the Speckled Band | 49 | 25 |
| 9 | The Adventure of the Engineer’s Thumb | 42 | 21 |
| 10 | The Adventure of the Noble Bachelor | 41 | 21 |
| 11 | The Adventure of the Beryl Coronet | 49 | 24 |
| 12 | The Adventure of the Copper Beeches | 50 | 25 |

## Concrete overlap examples

The first two chunks of **A**, both in *A Scandal in Bohemia*, cover story word intervals `[0, 250)` and `[200, 450)`. They overlap in exactly 50 words. Their book character intervals are `[0, 1392)` and `[1105, 2565)`, processed lines 1–27 and 21–45, and original source lines 55–81 and 75–99.

The first two chunks of **B** cover story word intervals `[0, 500)` and `[400, 900)`, overlapping in exactly 100 words. Their book character intervals are `[0, 2833)` and `[2287, 4919)`, processed lines 1–49 and 40–91, and original source lines 55–103 and 94–145.

## Complete verification

Both saved JSONL files were read back and every record was checked for exact `book_text[char_start:char_end]` equality, a single-story character interval, word boundaries and counts, configured window size and stride, processed and original line mapping, source hashes, and deterministic unique IDs. For every story, the union of word intervals covers every `\S+` match without a gap, including the title. All twelve stories occur in source order. The saved JSONL hashes match `data/chunk_manifest.json`.

Both commands were run a second time. The two JSONL files and the chunk manifest remained byte-identical, including their SHA-256 values. The source manifest and book text were verified before generation and were not changed.

## Fresh-clone regeneration check

An isolated temporary folder received only the verified original, `book.txt`, `source_manifest.json`, and the tracked `chunk_manifest.json`; neither JSONL file was copied. Generating A then B succeeded: A produced 525 chunks with SHA-256 `2e652bf929a39136726f9f78102f5ca5fcf11bb7b73e155382dd0b88dff979cd`, and B produced 265 chunks with SHA-256 `e8e886a889f8e1d7916eb236a800d694abb1f5c496ea4114c9e16a5a26eee87e`. Both manifest entries remained byte-identical to the tracked copy. Repeating both runs left both JSONL files and the manifest byte-identical. The original project files were unchanged.

Offline tests also rebuilt both missing outputs in A-then-B and B-then-A order, kept both manifest entries when only the other output was missing, and confirmed that a present other output with a wrong hash stops the operation without modifying either output or the manifest.

These statistics describe chunk construction only. Retrieval quality has not been measured; BM25 is planned for Phase 4. No embedding model, LLM, or model tokenizer has been selected.
