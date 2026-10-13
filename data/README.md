# Source data

- Book: *The Adventures of Sherlock Holmes*, Arthur Conan Doyle.
- Project Gutenberg entry: https://www.gutenberg.org/ebooks/1661
- UTF-8 TXT source: https://www.gutenberg.org/cache/epub/1661/pg1661.txt

Phase 2 downloaded and validated the TXT on 2026-10-13. The byte-for-byte original is `data/raw/sherlock_holmes.txt`; the separately extracted narrative is `data/processed/book.txt`. Both are ignored by Git. The tracked [source manifest](source_manifest.json) records hashes, sizes, counts, acquisition method, and original line interval. The [corpus audit](../docs/corpus-audit.md) records all twelve actual story starts and the extraction proof.

Only CRLF and bare CR line endings are normalized to LF in the processed file. The original is never edited. Re-running ingestion with the same original gives the same processed bytes and content hashes; a different candidate original is reported and rejected.
