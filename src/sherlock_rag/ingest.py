"""Acquire and extract the unmodified narrative of Gutenberg eBook #1661."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BOOK_URL = "https://www.gutenberg.org/ebooks/1661"
TXT_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
BOOK_TITLE = "The Adventures of Sherlock Holmes"
AUTHOR = "Arthur Conan Doyle"
START_MARKER = (
    "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
)
END_MARKER = (
    "*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
)
HEADINGS = (
    "I. A SCANDAL IN BOHEMIA",
    "II. THE RED-HEADED LEAGUE",
    "III. A CASE OF IDENTITY",
    "IV. THE BOSCOMBE VALLEY MYSTERY",
    "V. THE FIVE ORANGE PIPS",
    "VI. THE MAN WITH THE TWISTED LIP",
    "VII. THE ADVENTURE OF THE BLUE CARBUNCLE",
    "VIII. THE ADVENTURE OF THE SPECKLED BAND",
    "IX. THE ADVENTURE OF THE ENGINEER’S THUMB",
    "X. THE ADVENTURE OF THE NOBLE BACHELOR",
    "XI. THE ADVENTURE OF THE BERYL CORONET",
    "XII. THE ADVENTURE OF THE COPPER BEECHES",
)
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
MAX_BYTES = 10_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IngestionError(ValueError):
    """The source could not be acquired or verified without changing existing data."""


@dataclass(frozen=True)
class ExtractedBook:
    text: str
    raw_text: str
    source_line_start: int
    source_line_end: int
    story_lines: tuple[tuple[str, int, int], ...]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_count(text: str) -> int:
    """Count logical lines; a final line separator does not add a line."""
    return len(text.splitlines())


def word_count(text: str) -> int:
    """Count Unicode whitespace-delimited tokens, including standalone punctuation."""
    return len(text.split())


def extract_book(raw: bytes) -> ExtractedBook:
    """Return an exact line interval after normalizing CRLF and CR to LF."""
    if len(raw) > MAX_BYTES:
        raise IngestionError("Source exceeds the 10 MB limit")
    if raw.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise IngestionError("Source is an HTML page, not the Gutenberg TXT")
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise IngestionError("Source is not valid UTF-8") from exc
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines(keepends=True)
    plain = [line.removesuffix("\n") for line in lines]
    starts = [i for i, line in enumerate(plain) if line == START_MARKER]
    ends = [i for i, line in enumerate(plain) if line == END_MARKER]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise IngestionError("Missing, repeated, or misordered Gutenberg markers")
    start_marker, end_marker = starts[0], ends[0]
    header = "\n".join(plain[:start_marker])
    for required in (f"Title: {BOOK_TITLE}", f"Author: {AUTHOR}", "[eBook #1661]"):
        if required not in header:
            raise IngestionError(f"Source identity mismatch: missing {required}")
    contents = [
        i for i in range(start_marker + 1, end_marker) if plain[i] == "Contents"
    ]
    if len(contents) != 1:
        raise IngestionError("Expected one contents heading before the stories")
    toc_start = contents[0] + 1
    toc_rows: list[int] = []
    toc_pattern = re.compile(r"^\s*([IVX]+)\.\s+(.+?)\s*$")
    for i in range(toc_start, end_marker):
        match = toc_pattern.fullmatch(plain[i])
        if match is None:
            if toc_rows and not plain[i].strip():
                continue
            if len(toc_rows) == 12:
                break
            continue
        expected_index = len(toc_rows)
        if expected_index == 12:
            break
        expected_title = HEADINGS[expected_index].split(". ", 1)[1]
        if (
            match.group(1) != ROMAN[expected_index]
            or match.group(2).casefold() != expected_title.casefold()
        ):
            raise IngestionError(
                "Contents entries do not match the twelve expected stories"
            )
        toc_rows.append(i)
    if len(toc_rows) != 12:
        raise IngestionError("Contents is incomplete")
    heading_positions = [
        (i, plain[i])
        for i in range(toc_rows[-1] + 1, end_marker)
        if plain[i] in HEADINGS
    ]
    if [heading for _, heading in heading_positions] != list(HEADINGS):
        raise IngestionError(
            "Narrative headings are missing, duplicated, or out of order"
        )
    first = heading_positions[0][0]
    if any(plain[i].strip() for i in range(toc_rows[-1] + 1, first)):
        raise IngestionError("Unexpected text between contents and first story")
    last = end_marker - 1
    while last > first and not plain[last].strip():
        last -= 1
    if last <= heading_positions[-1][0]:
        raise IngestionError("Last story has no narrative text")
    for (position, _), (next_position, _) in zip(
        heading_positions, heading_positions[1:] + [(last + 1, "")], strict=True
    ):
        if not any(
            len(plain[i].split()) >= 3 for i in range(position + 1, next_position)
        ):
            raise IngestionError("A story has no content")
    text = "".join(lines[first : last + 1])
    story_lines = tuple(
        (heading, position - first + 1, position + 1)
        for position, heading in heading_positions
    )
    return ExtractedBook(text, normalized, first + 1, last + 1, story_lines)


def download_bytes(url: str = TXT_URL, timeout: float = 30.0) -> bytes:
    """Fetch TXT with an HTTP check, timeout, and bounded response size."""
    request = Request(url, headers={"User-Agent": "sherlock-rag/0.1 source-ingestion"})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise IngestionError(f"HTTP {response.status} while downloading source")
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise IngestionError("Downloaded source exceeds the 10 MB limit")
    return data


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def audit_text(book: ExtractedBook, manifest: dict) -> str:
    raw = manifest["raw"]
    processed = manifest["processed"]
    rows = "\n".join(
        f"| {number} | {heading} | {processed_line} | {source_line} |"
        for number, (heading, processed_line, source_line) in enumerate(
            book.story_lines, 1
        )
    )
    lines = book.text.splitlines()
    first_excerpt = next(
        line
        for line in lines[1:]
        if line.strip() and not re.fullmatch(r"[IVX]+\.", line.strip())
    )
    last_excerpt = lines[-1]
    return f"""# Corpus audit: Project Gutenberg #1661

The acquired original is preserved byte for byte in `data/raw/sherlock_holmes.txt` (ignored by Git). The processed file is `data/processed/book.txt` (also ignored).

## Extraction and traceability

- Source: {BOOK_URL} (TXT: {TXT_URL}).
- Acquisition method: {manifest["acquisition"]["method"]}; recorded at: {manifest["acquisition"]["acquired_at_utc"]}.
- Raw: {raw["bytes"]} bytes, {raw["lines"]} lines, {raw["words"]} whitespace-delimited tokens, SHA-256 `{raw["sha256"]}`.
- Processed: {processed["bytes"]} bytes, {processed["lines"]} lines, {processed["words"]} whitespace-delimited tokens, SHA-256 `{processed["sha256"]}`.
- Exact source interval: original lines {book.source_line_start}–{book.source_line_end}, inclusive, numbered from 1.
- Rule: decode UTF-8 strictly, normalize CRLF and bare CR to LF, select that complete interval, and make no other text edits. A final newline is included only if present on the selected last source line.
- Proof: re-extracting the interval from the original with this rule yields the exact processed bytes and SHA-256 above.

The removed prefix contains Gutenberg's preface, the book title page, and the table of contents. The removed suffix contains blank separators, the Gutenberg end marker, and the license. Narrative paragraphs, punctuation, case, italic markers, and internal headings remain unchanged.

## Story starts

| # | Story heading | Processed line | Original line |
| --- | --- | ---: | ---: |
{rows}

The first story's opening narrative line includes: “{first_excerpt.strip()}” The last story's final narrative line is: “{last_excerpt.strip()}”

## Known limitations

This phase preserves line wrapping and paragraphs as supplied by Gutenberg. It does not correct possible transcription errors, assign paragraph identifiers, or perform chunking. Word counts use `str.split()` and are not linguistic token counts. Source line numbers refer to the captured original and can change if Gutenberg updates its TXT.
"""


def ingest(
    raw_path: Path,
    processed_path: Path,
    manifest_path: Path,
    audit_path: Path,
    *,
    input_file: Path | None = None,
    fetcher=download_bytes,
    downloads_dir: Path | None = None,
) -> dict:
    """Validate before any replacement, preserving an existing different original."""
    existing = raw_path.read_bytes() if raw_path.exists() else None
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    if input_file is not None:
        candidate = input_file.read_bytes()
        method = "local-file"
    else:
        try:
            candidate = fetcher(TXT_URL, 30.0)
            method = "download"
        except HTTPError as exc:
            raise IngestionError(f"HTTP {exc.code} while downloading source") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if existing is not None:
                candidate = existing
                method = "local-file"
            else:
                directory = downloads_dir or Path.home() / "Downloads"
                fallback = next(
                    (
                        path
                        for path in (
                            directory / "pg1661.txt",
                            directory / "sherlock_holmes.txt",
                        )
                        if path.is_file()
                    ),
                    None,
                )
                if fallback is None:
                    raise IngestionError(
                        "Download failed and no original was found in data/raw/ or the two expected Downloads filenames"
                    ) from exc
                candidate = fallback.read_bytes()
                method = "local-file"
    book = extract_book(candidate)
    if existing is not None and existing != candidate:
        raise IngestionError(
            f"Source changed: existing SHA-256 {sha256(existing)}, candidate SHA-256 {sha256(candidate)}. Existing original was preserved."
        )
    candidate_hash = sha256(candidate)
    if (
        existing is not None
        and previous
        and previous.get("raw", {}).get("sha256") == candidate_hash
    ):
        acquisition = previous["acquisition"]
    elif existing is not None:
        acquisition = {"method": "local-file", "acquired_at_utc": None}
    else:
        acquisition = {
            "method": method,
            "acquired_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    processed_bytes = book.text.encode("utf-8")
    manifest = {
        "book": {
            "title": BOOK_TITLE,
            "author": AUTHOR,
            "gutenberg_id": 1661,
            "page_url": BOOK_URL,
            "txt_url": TXT_URL,
        },
        "acquisition": acquisition,
        "raw": {
            "path": "data/raw/sherlock_holmes.txt",
            "sha256": candidate_hash,
            "bytes": len(candidate),
            "lines": line_count(book.raw_text),
            "words": word_count(book.raw_text),
        },
        "processed": {
            "path": "data/processed/book.txt",
            "sha256": sha256(processed_bytes),
            "bytes": len(processed_bytes),
            "lines": line_count(book.text),
            "words": word_count(book.text),
            "source_lines_inclusive": {
                "start": book.source_line_start,
                "end": book.source_line_end,
            },
            "processing": "extract-v1: strict UTF-8; CRLF and CR to LF; exact inclusive narrative line interval",
        },
        "counting": "Lines are str.splitlines() logical lines; a trailing newline adds no line. Words are str.split() whitespace-delimited tokens.",
    }
    audit = audit_text(book, manifest).encode("utf-8")
    if existing is None:
        atomic_write(raw_path, candidate)
    for path, data in (
        (processed_path, processed_bytes),
        (
            manifest_path,
            (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        ),
        (audit_path, audit),
    ):
        if not path.exists() or path.read_bytes() != data:
            atomic_write(path, data)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=Path, help="Use a locally downloaded Gutenberg #1661 TXT"
    )
    args = parser.parse_args(argv)
    try:
        manifest = ingest(
            PROJECT_ROOT / "data/raw/sherlock_holmes.txt",
            PROJECT_ROOT / "data/processed/book.txt",
            PROJECT_ROOT / "data/source_manifest.json",
            PROJECT_ROOT / "docs/corpus-audit.md",
            input_file=args.input_file,
        )
    except (IngestionError, OSError, ValueError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
