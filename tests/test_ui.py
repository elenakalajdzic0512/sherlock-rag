"""Offline Streamlit interactions with a fake RAG pipeline."""

from __future__ import annotations

import json
import shutil
from threading import Lock

import pytest
from streamlit.testing.v1 import AppTest

from sherlock_rag import ui
from sherlock_rag.ingest import PROJECT_ROOT
from sherlock_rag.ui_data import UiSetupError, load_reviewed_bank, preflight_rag


def app() -> AppTest:
    result = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
    assert not result.exception
    return result


class FakePipeline:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[str] = []

    def answer(self, query: str) -> dict:
        self.calls.append(query)
        return self.result


def install_fake(monkeypatch, result: dict) -> FakePipeline:
    pipeline = FakePipeline(result)
    monkeypatch.setattr(ui, "preflight_rag", lambda: None)
    monkeypatch.setattr(ui, "load_shared_pipeline", lambda: ui.SharedPipeline(pipeline))
    return pipeline


def test_bank_uses_only_v2_and_never_loads_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        ui,
        "load_shared_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    at = app()
    at.radio(key="view").set_value(ui.BANK_VIEW).run()
    assert not at.exception
    transfers = [
        button.key for button in at.button if button.key.startswith("transfer-")
    ]
    assert len(transfers) == 5
    assert "transfer-qg-v1-10-3" not in transfers
    at.button(key="reveal-qg-v1-05-1").click().run()
    assert not at.exception
    assert "qg-v1-05-1" in at.session_state["revealed_ids"]
    at.button(key="transfer-qg-v1-05-1").click().run()
    assert at.radio(key="view").value == ui.ASK_VIEW
    assert (
        at.text_input(key="draft_query").value
        == "What symbol was printed on the lid of the brass box?"
    )
    assert not at.exception


def test_uncovered_story_has_no_fallback_card() -> None:
    at = app()
    at.radio(key="view").set_value(ui.BANK_VIEW).run()
    at.selectbox(key="story_filter").set_value("THE RED-HEADED LEAGUE").run()
    assert not at.exception
    assert any("nema prihvaćenog pitanja" in item.value for item in at.info)
    assert not [button for button in at.button if button.key.startswith("transfer-")]


def test_empty_question_does_not_load_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        ui,
        "preflight_rag",
        lambda: (_ for _ in ()).throw(AssertionError("preflight called")),
    )
    at = app()
    at.button[0].click().run()
    assert not at.exception
    assert any("Unesite pitanje" in item.value for item in at.warning)


@pytest.mark.parametrize(
    ("status", "expected_collection", "expected_text"),
    [
        ("answered", "success", "Odgovor generativnog modela"),
        ("abstained", "info", "nisu dovoljni"),
        ("error", "error", "nije vratio ispravan odgovor"),
    ],
)
def test_result_statuses_and_citations(
    monkeypatch, status, expected_collection, expected_text
) -> None:
    citation = {
        "source_id": "C1",
        "story_title": "THE FIVE ORANGE PIPS",
        "source_line_start": 4191,
        "source_line_end": 4191,
        "quote_raw": "upon the lid was printed the treble K",
    }
    result = {
        "status": status,
        "reason_code": "invalid_model_output" if status == "error" else status,
        "claims": [{"text": "The treble K.", "citations": [citation]}]
        if status == "answered"
        else [],
        "submitted_chunks": [
            {
                "source_id": "C1",
                "story_title": citation["story_title"],
                "source_line_start": 4180,
                "source_line_end": 4200,
                "text": "A full source passage.",
            }
        ],
        "raw_model_outputs": ["SECRET INVALID RAW MODEL OUTPUT"],
    }
    pipeline = install_fake(monkeypatch, result)
    at = app()
    at.text_input(key="draft_query").set_value(
        "What symbol was printed on the brass box?"
    )
    at.button[0].click().run()
    assert not at.exception
    assert pipeline.calls == ["What symbol was printed on the brass box?"]
    assert any(expected_text in item.value for item in getattr(at, expected_collection))
    content = " ".join(item.value for item in at.markdown)
    assert "SECRET INVALID RAW MODEL OUTPUT" not in content
    if status == "answered":
        assert "The treble K. [C1]" in content
        assert any("redovi originala 4191" in item.value for item in at.caption)
        assert any("Korišćeni odlomci" in item.value for item in at.markdown)
    elif status == "abstained":
        assert any("cijeloj knjizi" in item.value for item in at.info)


def test_rerun_filter_and_draft_change_do_not_regenerate(monkeypatch) -> None:
    pipeline = install_fake(
        monkeypatch,
        {
            "status": "answered",
            "claims": [
                {
                    "text": "A swamp adder.",
                    "citations": [
                        {
                            "source_id": "C1",
                            "story_title": "THE ADVENTURE OF THE SPECKLED BAND",
                            "source_line_start": 7844,
                            "source_line_end": 7845,
                            "quote_raw": "It is a swamp adder!",
                        }
                    ],
                }
            ],
            "submitted_chunks": [],
        },
    )
    at = app()
    at.text_input(key="draft_query").set_value("First question?")
    at.button[0].click().run()
    assert pipeline.calls == ["First question?"]
    at.run()
    at.radio(key="view").set_value(ui.BANK_VIEW).run()
    at.selectbox(key="story_filter").set_value("A SCANDAL IN BOHEMIA").run()
    at.radio(key="view").set_value(ui.ASK_VIEW).run()
    at.text_input(key="draft_query").set_value("Different question?").run()
    assert pipeline.calls == ["First question?"]
    assert at.session_state["last_result"]["query"] == "First question?"
    assert any("Rezultat za: First question?" in item.value for item in at.markdown)
    assert any("prethodno poslatog pitanja" in item.value for item in at.caption)
    assert any("A swamp adder. [C1]" in item.value for item in at.markdown)
    assert any("redovi originala 7844" in item.value for item in at.caption)


def test_shared_pipeline_lock_prevents_parallel_generation(monkeypatch) -> None:
    pipeline = FakePipeline({"status": "abstained"})
    shared = ui.SharedPipeline(pipeline, Lock())
    shared.lock.acquire()
    monkeypatch.setattr(ui, "preflight_rag", lambda: None)
    monkeypatch.setattr(ui, "load_shared_pipeline", lambda: shared)
    try:
        at = app()
        at.text_input(key="draft_query").set_value("A question?")
        at.button[0].click().run()
        assert any("već obrađuje" in item.value for item in at.warning)
        assert pipeline.calls == []
    finally:
        shared.lock.release()


def test_bank_hash_tampering_is_rejected(tmp_path) -> None:
    folder = tmp_path / "questions/generated_v1"
    folder.mkdir(parents=True)
    source = PROJECT_ROOT / "questions/generated_v1"
    for name in (
        "results.json",
        "results.sha256",
        "reviews.json",
        "reviews_v2.json",
        "summary_v2.json",
        "bank_v2.jsonl",
    ):
        shutil.copyfile(source / name, folder / name)
    bank = [
        json.loads(line) for line in (folder / "bank_v2.jsonl").read_text().splitlines()
    ]
    bank[0]["expected_answer"] = "invented"
    (folder / "bank_v2.jsonl").write_text(
        "\n".join(json.dumps(item) for item in bank) + "\n"
    )
    with pytest.raises(UiSetupError, match="v2 banka"):
        load_reviewed_bank(tmp_path)


def test_missing_bank_reports_restore_command(tmp_path) -> None:
    with pytest.raises(UiSetupError, match="v2 banka") as caught:
        load_reviewed_bank(tmp_path)
    assert "git restore questions/generated_v1" in caught.value.command


def test_missing_source_reports_existing_ingest_command(tmp_path) -> None:
    with pytest.raises(UiSetupError, match="Nedostaje provjereni izvor") as caught:
        preflight_rag(tmp_path)
    assert (
        caught.value.command
        == "conda run -n sherlock-rag python -m sherlock_rag.ingest"
    )
