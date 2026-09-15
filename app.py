import os
import tempfile
from pathlib import Path
import streamlit as st

from src.ingest_printed import process_printed_pdf
from src.ingest_handwritten import process_handwritten_pdf
from src.rag_engine import add_documents_to_db, ask_study_assistant

st.set_page_config(
    page_title="Mentora — Study Assistant",
    page_icon="🕮",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Visual identity
#
# Mentora is a study companion built around one idea: your own notes and
# textbooks, indexed and answerable. The look borrows from the reading room
# rather than the SaaS dashboard — ink, paper, and a gold marginal note —
# because that's the world this product actually lives in.
#
# Styling lives in static/style.css so design tweaks don't require touching
# the app logic below.
# ---------------------------------------------------------------------------
def load_css(path: str) -> None:
    css_file = Path(path)
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Stylesheet not found at {path} — using Streamlit defaults.")

load_css("static/style.css")

# 1. State Management
if "subjects" not in st.session_state:
    st.session_state.subjects = ["General"]
if "current_subject" not in st.session_state:
    st.session_state.current_subject = "General"
if "messages" not in st.session_state:
    st.session_state.messages = {}

current_sub = st.session_state.current_subject
if current_sub not in st.session_state.messages:
    st.session_state.messages[current_sub] = []

# 2. Sidebar: Subject Manager
with st.sidebar:
    st.markdown('<div class="mentora-wordmark">Mentora</div>', unsafe_allow_html=True)
    st.markdown('<hr class="mentora-rule">', unsafe_allow_html=True)

    with st.form(key="new_subject_form", clear_on_submit=True):
        new_sub_input = st.text_input("New subject", placeholder="e.g. Thermodynamics", label_visibility="collapsed")
        submit_button = st.form_submit_button("Add subject")

        if submit_button and new_sub_input.strip():
            clean_name = new_sub_input.strip()
            if clean_name not in st.session_state.subjects:
                st.session_state.subjects.append(clean_name)
                st.session_state.messages[clean_name] = []
                st.session_state.current_subject = clean_name
                st.rerun()

    st.markdown('<div class="mentora-eyebrow">Your subjects</div>', unsafe_allow_html=True)
    selected = st.radio(
        "Select active workspace",
        options=st.session_state.subjects,
        index=st.session_state.subjects.index(st.session_state.current_subject),
        label_visibility="collapsed",
    )
    if selected != st.session_state.current_subject:
        st.session_state.current_subject = selected
        st.rerun()

# 3. Main Workspace Area
st.markdown(f'<div class="mentora-header">{current_sub}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="mentora-caption">Answers here are grounded only in what you\'ve added to {current_sub} — nothing else leaks in.</div>',
    unsafe_allow_html=True,
)

# File Ingestion Widget
with st.expander("Add material to this subject", expanded=False):
    col1, col2 = st.columns([1, 2])
    with col1:
        doc_type = st.radio(
            "Material type",
            ["Printed", "Handwritten"],
            help="Printed: clean digital PDFs, parsed via PyMuPDF4LLM. Handwritten: scanned notes, parsed via Gemini Vision.",
            label_visibility="collapsed",
        )
    with col2:
        uploaded_file = st.file_uploader("Select a PDF", type=["pdf"], label_visibility="collapsed")

    if st.button("Index this document") and uploaded_file is not None:
        with st.spinner("Reading, chunking, and indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                if doc_type == "Printed":
                    chunks = process_printed_pdf(tmp_path, current_sub)
                else:
                    chunks = process_handwritten_pdf(tmp_path, current_sub)

                add_documents_to_db(chunks)
                st.success(f"Indexed {len(chunks)} passages into {current_sub}.")
            except Exception as e:
                st.error(f"Couldn't index this document: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# 4. Chat History Rendering
for msg in st.session_state.messages[current_sub]:
    avatar = "🙂" if msg["role"] == "user" else "🕮"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 5. Query Handling & Generation
user_input = st.chat_input(f"Ask something about {current_sub}...")
if user_input:
    st.session_state.messages[current_sub].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🕮"):
        with st.spinner("Checking your materials..."):
            try:
                response_text, sources = ask_study_assistant(
                    query=user_input,
                    subject_id=current_sub,
                    chat_history=st.session_state.messages[current_sub]
                )
                st.markdown(response_text)

                if sources:
                    with st.expander("Sourced from"):
                        for i, doc in enumerate(sources, start=1):
                            meta = doc.metadata
                            st.markdown(
                                f"**{i:02d} · {meta.get('source_file', 'Doc')}** — "
                                f"p.{meta.get('page_number', 'N/A')}, {meta.get('section_path', 'General')}"
                            )
                            st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

                st.session_state.messages[current_sub].append({"role": "assistant", "content": response_text})
            except Exception as err:
                st.error(f"Couldn't generate a response: {err}")