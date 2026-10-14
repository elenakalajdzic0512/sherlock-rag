"""Local Serbian Streamlit interface for existing RAG and the reviewed v2 bank."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import streamlit as st

from sherlock_rag.ingest import PROJECT_ROOT
from sherlock_rag.ui_data import UiSetupError, load_reviewed_bank, preflight_rag

ASK_VIEW = "Pitaj knjigu"
BANK_VIEW = "Pitanja iz knjige"


@dataclass
class SharedPipeline:
    pipeline: object
    lock: Lock = field(default_factory=Lock)


@st.cache_resource(show_spinner=False)
def load_shared_pipeline() -> SharedPipeline:
    """Load semantic A and generation v1 once, on the first valid submission."""
    from sherlock_rag.rag import load_pipeline

    return SharedPipeline(load_pipeline(PROJECT_ROOT, "v1", "default"))


def show_setup_error(exc: UiSetupError) -> None:
    st.error(str(exc))
    st.caption("Pripremite lokalne podatke i zatim pokušajte ponovo:")
    st.code(exc.command, language="bash")


def show_submitted_chunks(result: dict) -> None:
    chunks = result.get("submitted_chunks") or []
    if not chunks:
        return
    st.markdown("#### Korišćeni odlomci")
    for chunk in chunks:
        label = f"[{chunk['source_id']}] {chunk['story_title']} · redovi {chunk['source_line_start']}–{chunk['source_line_end']}"
        with st.expander(label):
            st.write(chunk["text"])


def show_result(query: str, result: dict) -> None:
    st.markdown(f"### Rezultat za: {query}")
    status = result.get("status")
    if status == "answered":
        st.success("Odgovor generativnog modela")
        for claim in result.get("claims", []):
            labels = " ".join(f"[{cite['source_id']}]" for cite in claim["citations"])
            st.markdown(f"{claim['text']} {labels}")
            for cite in claim["citations"]:
                with st.container(border=True):
                    st.caption(
                        f"[{cite['source_id']}] {cite['story_title']} · redovi originala "
                        f"{cite['source_line_start']}–{cite['source_line_end']}"
                    )
                    st.write(cite["quote_raw"])
        show_submitted_chunks(result)
        st.caption(
            "Provjerite generisani odgovor u citiranim odlomcima. Sadržajna podrška nije automatski potvrđena."
        )
    elif status == "abstained":
        st.info(
            "Pronađeni odlomci nisu dovoljni za pouzdan odgovor na ovo pitanje. To ne znači da odgovor ne postoji u cijeloj knjizi."
        )
        show_submitted_chunks(result)
    else:
        messages = {
            "invalid_input": "Pitanje je prazno ili predugačko. Skratite ga i pokušajte ponovo.",
            "context_budget_exceeded": "Nijedan kompletan pronađeni odlomak ne staje u ulazni budžet modela.",
            "retrieval_error": "Pretraga nije uspjela. Provjerite lokalne podatke i indeks.",
            "invalid_model_output": "Model nije vratio ispravan odgovor sa provjerljivim citatima.",
            "resource_or_model_error": "Lokalni model trenutno nije mogao da obradi pitanje.",
        }
        st.error(
            result.get("ui_message")
            or messages.get(result.get("reason_code"), "Obrada pitanja nije uspjela.")
        )
        if result.get("setup_command"):
            st.code(result["setup_command"], language="bash")
        show_submitted_chunks(result)


def ask_page() -> None:
    st.markdown("## Pitaj knjigu")
    st.write(
        "Postavite pitanje o *The Adventures of Sherlock Holmes*. Odgovor se zasniva na pronađenim odlomcima knjige."
    )
    with st.form("ask_form", clear_on_submit=False):
        st.text_input(
            "Vaše pitanje",
            key="draft_query",
            placeholder="Na primjer: What was on the brass box?",
        )
        submitted = st.form_submit_button("Pošalji pitanje", type="primary")
    if submitted:
        query = st.session_state.get("draft_query", "").strip()
        if not query:
            st.warning("Unesite pitanje prije slanja.")
        else:
            try:
                load_reviewed_bank()
                preflight_rag()
                with st.spinner("Učitavam lokalni model i pripremam odgovor…"):
                    shared = load_shared_pipeline()
                    if not shared.lock.acquire(blocking=False):
                        st.warning(
                            "Drugo pitanje se već obrađuje. Pokušajte ponovo kada se završi."
                        )
                        return
                    try:
                        result = shared.pipeline.answer(query)
                    finally:
                        shared.lock.release()
            except UiSetupError as exc:
                result = {
                    "status": "error",
                    "reason_code": "setup_error",
                    "ui_message": str(exc),
                    "setup_command": exc.command,
                }
            except Exception:  # noqa: BLE001 - user-facing boundary, no raw model output
                result = {"status": "error", "reason_code": "resource_or_model_error"}
            st.session_state["last_result"] = {"query": query, "result": result}
    saved = st.session_state.get("last_result")
    if saved:
        if st.session_state.get("draft_query", "").strip() != saved["query"]:
            st.caption(
                "Prikazan je rezultat prethodno poslatog pitanja. Novo pitanje se obrađuje tek nakon slanja."
            )
        show_result(saved["query"], saved["result"])


def bank_page() -> None:
    st.markdown("## Pitanja iz knjige")
    try:
        bank, stories = load_reviewed_bank()
    except UiSetupError as exc:
        show_setup_error(exc)
        return
    st.write(
        "Pet pregledanih pitanja iz pet od dvanaest priča. Očekivani odgovori potiču iz pregledane v2 banke; novi RAG odgovori se generišu zasebno."
    )
    options = ["Sve priče", *(title for _, title in stories)]
    selected_title = st.selectbox("Filtriraj po priči", options, key="story_filter")
    filtered = (
        bank
        if selected_title == "Sve priče"
        else [item for item in bank if item["story_title"] == selected_title]
    )
    if not filtered:
        st.info(
            "Za ovu priču nema prihvaćenog pitanja u pregledanoj v2 banci. Odbačeni kandidati se ne prikazuju kao zamjena."
        )
        return
    for item in filtered:
        with st.container(border=True):
            st.caption(item["story_title"])
            st.markdown(f"### {item['question']}")
            if st.button(
                "Otkrij očekivani odgovor i dokaz", key=f"reveal-{item['id']}"
            ):
                revealed = set(st.session_state.get("revealed_ids", []))
                revealed.add(item["id"])
                st.session_state["revealed_ids"] = sorted(revealed)
            if item["id"] in st.session_state.get("revealed_ids", []):
                st.markdown("**Pregledani očekivani odgovor iz banke**")
                st.write(item["expected_answer"])
                cite = item["citation"]
                st.caption(
                    f"Izvor [{cite['source_id']}] · {cite['story_title']} · redovi {cite['source_line_start']}–{cite['source_line_end']}"
                )
                st.write(cite["quote_raw"])
            st.button(
                "Postavi ovo pitanje RAG-u",
                key=f"transfer-{item['id']}",
                on_click=transfer_question,
                args=(item["question"],),
            )


def transfer_question(question: str) -> None:
    st.session_state["draft_query"] = question
    st.session_state["view"] = ASK_VIEW


def main() -> None:
    st.set_page_config(
        page_title="Sherlock RAG · lokalna biblioteka", page_icon="📚", layout="wide"
    )
    st.markdown(
        """<style>
        .stApp {background: #f6f2e8; color: #20372b;}
        .block-container {max-width: 940px; padding-top: 2rem; padding-bottom: 4rem;}
        h1, h2, h3 {color: #214c3a; letter-spacing: -.02em;}
        div[data-testid="stVerticalBlockBorderWrapper"] {background: #fffdf7; border-color: #d9dfd4; border-radius: 14px;}
        .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {background: #245a43; border-color: #245a43;}
        @media (max-width: 700px) {.block-container {padding: 1rem 1rem 3rem;}}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Sherlock RAG")
    st.caption("Lokalna biblioteka · Arthur Conan Doyle · Project Gutenberg #1661")
    if "view" not in st.session_state:
        st.session_state["view"] = ASK_VIEW
    st.radio(
        "Prikaz",
        (ASK_VIEW, BANK_VIEW),
        horizontal=True,
        key="view",
        label_visibility="collapsed",
    )
    if st.session_state["view"] == ASK_VIEW:
        ask_page()
    else:
        bank_page()


if __name__ == "__main__":
    main()
