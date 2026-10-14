"""Offline behavior tests for the Gutenberg #1661 ingestion boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from sherlock_rag.ingest import (
    END_MARKER,
    HEADINGS,
    START_MARKER,
    IngestionError,
    extract_book,
    ingest,
)


def source(*, endings: str = "\n", toc_uppercase: bool = False) -> bytes:
    lines = [
        "The Project Gutenberg eBook of The Adventures of Sherlock Holmes",
        "Title: The Adventures of Sherlock Holmes",
        "Author: Arthur Conan Doyle",
        "Release date: March 1, 1999 [eBook #1661]",
        START_MARKER,
        "The Adventures of Sherlock Holmes",
        "by Arthur Conan Doyle",
        "Contents",
    ]
    for heading in HEADINGS:
        roman, title = heading.split(". ", 1)
        lines.append(heading if toc_uppercase else f"   {roman}. {title.title()}")
    lines += ["", ""]
    for number, heading in enumerate(HEADINGS, 1):
        lines += [
            heading,
            "",
            "I." if number == 1 else "",
            f"To Sherlock Holmes: story {number} keeps _italics_ and punctuation!",
            "",
        ]
    lines += ["The last narrative line.", "", END_MARKER, "Gutenberg license text"]
    return endings.join(lines).encode("utf-8")


def paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return (
        tmp_path / "data/raw/sherlock_holmes.txt",
        tmp_path / "data/processed/book.txt",
        tmp_path / "data/source_manifest.json",
        tmp_path / "docs/corpus-audit.md",
    )


def test_extract_removes_wrapper_and_preserves_narrative() -> None:
    book = extract_book(source())
    assert book.text.startswith("I. A SCANDAL IN BOHEMIA\n\nI.\n")
    assert "To Sherlock Holmes: story 1 keeps _italics_ and punctuation!" in book.text
    assert book.text.endswith("The last narrative line.\n")
    assert "Contents" not in book.text
    assert "Gutenberg license text" not in book.text
    assert [heading for heading, _, _ in book.story_lines] == list(HEADINGS)


def test_heading_after_contents_is_the_real_start() -> None:
    raw = source(toc_uppercase=True)
    book = extract_book(raw)
    normalized_lines = raw.decode().splitlines()
    assert normalized_lines.index(HEADINGS[0]) < book.source_line_start - 1
    assert normalized_lines[book.source_line_start - 1] == HEADINGS[0]
    assert book.story_lines[0][1] == 1
    assert book.story_lines[-1][1] > book.story_lines[0][1]


@pytest.mark.parametrize("endings", ["\n", "\r\n", "\r"])
def test_line_endings_and_exact_source_interval(endings: str) -> None:
    raw = source(endings=endings)
    book = extract_book(raw)
    normalized = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines(keepends=True)
    expected = "".join(lines[book.source_line_start - 1 : book.source_line_end])
    assert book.text == expected
    assert "\r" not in book.text
    assert book.story_lines[-1][2] <= book.source_line_end


@pytest.mark.parametrize(
    "bad",
    [
        lambda raw: raw.replace(START_MARKER.encode(), b"NO START MARKER"),
        lambda raw: raw.replace(END_MARKER.encode(), b"NO END MARKER"),
        lambda raw: raw.replace(b"[eBook #1661]", b"[eBook #9999]"),
        lambda raw: raw.replace(b"Author: Arthur Conan Doyle", b"Author: Someone Else"),
        lambda raw: raw.replace(HEADINGS[6].encode(), b"VII. WRONG STORY", 1),
        lambda raw: raw.replace(
            b"To Sherlock Holmes: story 7 keeps _italics_ and punctuation!", b""
        ),
        lambda raw: b"<!doctype html><html>Bad Gateway</html>",
    ],
)
def test_invalid_or_incomplete_source_is_rejected(bad) -> None:
    with pytest.raises(IngestionError):
        extract_book(bad(source()))


def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(IngestionError, match="UTF-8"):
        extract_book(source() + b"\xff")


def test_repeat_ingestion_is_byte_identical_and_preserves_date(tmp_path: Path) -> None:
    input_file = tmp_path / "pg1661.txt"
    input_file.write_bytes(source(endings="\r\n"))
    raw, processed, manifest_file, audit = paths(tmp_path)
    first = ingest(raw, processed, manifest_file, audit, input_file=input_file)
    before = (
        raw.read_bytes(),
        processed.read_bytes(),
        manifest_file.read_bytes(),
        audit.read_bytes(),
    )
    second = ingest(raw, processed, manifest_file, audit, input_file=input_file)
    after = (
        raw.read_bytes(),
        processed.read_bytes(),
        manifest_file.read_bytes(),
        audit.read_bytes(),
    )
    assert first == second == json.loads(manifest_file.read_text())
    assert before == after
    assert first["acquisition"]["method"] == "local-file"
    assert hashlib.sha256(before[0]).hexdigest() == first["raw"]["sha256"]
    assert hashlib.sha256(before[1]).hexdigest() == first["processed"]["sha256"]


def test_fresh_clone_reacquisition_preserves_tracked_manifest_and_audit(
    tmp_path: Path,
) -> None:
    raw, processed, manifest_file, audit = paths(tmp_path)
    ingest(raw, processed, manifest_file, audit, fetcher=lambda *_: source())
    tracked = (manifest_file.read_bytes(), audit.read_bytes())
    raw.unlink()
    processed.unlink()

    result = ingest(raw, processed, manifest_file, audit, fetcher=lambda *_: source())

    assert (manifest_file.read_bytes(), audit.read_bytes()) == tracked
    assert raw.read_bytes() == source()
    assert result == json.loads(tracked[0])


def test_fresh_clone_rejects_changed_source_without_rewriting_manifest(
    tmp_path: Path,
) -> None:
    raw, processed, manifest_file, audit = paths(tmp_path)
    ingest(raw, processed, manifest_file, audit, fetcher=lambda *_: source())
    tracked = (manifest_file.read_bytes(), audit.read_bytes())
    raw.unlink()
    processed.unlink()

    with pytest.raises(
        IngestionError, match="differs from the existing source manifest"
    ):
        ingest(
            raw,
            processed,
            manifest_file,
            audit,
            fetcher=lambda *_: source(endings="\r\n"),
        )
    assert not raw.exists()
    assert not processed.exists()
    assert (manifest_file.read_bytes(), audit.read_bytes()) == tracked


def test_existing_original_survives_bad_candidate_or_download(tmp_path: Path) -> None:
    raw, processed, manifest_file, audit = paths(tmp_path)
    ingest(raw, processed, manifest_file, audit, fetcher=lambda *_: source())
    original = (raw.read_bytes(), processed.read_bytes(), manifest_file.read_bytes())
    with pytest.raises(IngestionError, match="Source changed"):
        ingest(
            raw,
            processed,
            manifest_file,
            audit,
            fetcher=lambda *_: source(endings="\r\n"),
        )
    with pytest.raises(IngestionError, match="HTML"):
        ingest(
            raw,
            processed,
            manifest_file,
            audit,
            fetcher=lambda *_: b"<html>failure</html>",
        )
    with pytest.raises(IngestionError, match="HTTP 503"):
        ingest(
            raw,
            processed,
            manifest_file,
            audit,
            fetcher=lambda *_: (_ for _ in ()).throw(
                HTTPError("url", 503, "Unavailable", {}, None)
            ),
        )
    assert original == (
        raw.read_bytes(),
        processed.read_bytes(),
        manifest_file.read_bytes(),
    )


def test_network_failure_uses_existing_raw_without_changing_acquisition(
    tmp_path: Path,
) -> None:
    raw, processed, manifest_file, audit = paths(tmp_path)
    first = ingest(raw, processed, manifest_file, audit, fetcher=lambda *_: source())
    second = ingest(
        raw,
        processed,
        manifest_file,
        audit,
        fetcher=lambda *_: (_ for _ in ()).throw(URLError("offline")),
    )
    assert second == first


def test_network_failure_checks_only_expected_downloads_names(tmp_path: Path) -> None:
    raw, processed, manifest_file, audit = paths(tmp_path)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "unrelated.txt").write_bytes(source())
    offline = lambda *_: (_ for _ in ()).throw(URLError("offline"))
    with pytest.raises(IngestionError, match="two expected Downloads filenames"):
        ingest(
            raw,
            processed,
            manifest_file,
            audit,
            fetcher=offline,
            downloads_dir=downloads,
        )
    (downloads / "pg1661.txt").write_bytes(source())
    result = ingest(
        raw, processed, manifest_file, audit, fetcher=offline, downloads_dir=downloads
    )
    assert result["acquisition"]["method"] == "local-file"
    assert raw.read_bytes() == (downloads / "pg1661.txt").read_bytes()
