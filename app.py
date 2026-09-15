import os
import tempfile
from pathlib import Path
import streamlit as st

from src.ingest_printed import process_printed_pdf
from src.ingest_handwritten import process_handwritten_pdf
from src.rag_engine import add_documents_to_db, ask_study_assistant

st.set_page_config(
    page_title="Mentora AI - Study Assistant",
    page_icon="🎓",
    layout="wide"
)

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
    st.title("📚 Subjects")
    
    with st.form(key="new_subject_form", clear_on_submit=True):
        new_sub_input = st.text_input("Create New Subject:")
        submit_button = st.form_submit_button("Add Subject")
        
        if submit_button and new_sub_input.strip():
            clean_name = new_sub_input.strip()
            if clean_name not in st.session_state.subjects:
                st.session_state.subjects.append(clean_name)
                st.session_state.messages[clean_name] = []
                st.session_state.current_subject = clean_name
                st.rerun()

    st.markdown("---")
    selected = st.radio(
        "Select Active Workspace:",
        options=st.session_state.subjects,
        index=st.session_state.subjects.index(st.session_state.current_subject)
    )
    if selected != st.session_state.current_subject:
        st.session_state.current_subject = selected
        st.rerun()

# 3. Main Workspace Area
st.title(f"🎓 Mentora AI: {current_sub}")
st.caption(f"Currently scoped to '{current_sub}'. Documents uploaded here remain strictly isolated.")

# File Ingestion Widget
with st.expander("📥 Upload & Ingest Study Material", expanded=False):
    col1, col2 = st.columns([1, 2])
    with col1:
        doc_type = st.radio(
            "Material Type:",
            ["Printed Textbook / Slides", "Handwritten Notes"],
            help="Choose 'Printed' for clean digital PDFs (parsed via PyMuPDF4LLM), or 'Handwritten' for scans (parsed via Gemini Vision)."
        )
    with col2:
        uploaded_file = st.file_uploader("Select PDF file to index", type=["pdf"])

    if st.button("Index Document into Subject") and uploaded_file is not None:
        with st.spinner("Extracting content, chunking, and updating vector index..."):
            # Save temporary file for local readers
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                if doc_type == "Printed Textbook / Slides":
                    chunks = process_printed_pdf(tmp_path, current_sub)
                else:
                    chunks = process_handwritten_pdf(tmp_path, current_sub)

                add_documents_to_db(chunks)
                st.success(f"Successfully processed and indexed {len(chunks)} chunks into '{current_sub}'!")
            except Exception as e:
                st.error(f"Error indexing document: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# 4. Chat History Rendering
for msg in st.session_state.messages[current_sub]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Query Handling & Generation
user_input = st.chat_input(f"Ask any question about {current_sub}...")
if user_input:
    # Append & display user prompt
    st.session_state.messages[current_sub].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your study materials..."):
            try:
                response_text, sources = ask_study_assistant(
                    query=user_input,
                    subject_id=current_sub,
                    chat_history=st.session_state.messages[current_sub]
                )
                st.markdown(response_text)

                if sources:
                    with st.expander("🔍 Cited Material Context"):
                        for i, doc in enumerate(sources, start=1):
                            meta = doc.metadata
                            st.markdown(
                                f"**Chunk {i}** — `{meta.get('source_file', 'Doc')}` "
                                f"(Page: {meta.get('page_number', 'N/A')}, Section: *{meta.get('section_path', 'General')}*)"
                            )
                            st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

                # Append assistant response to current subject's history
                st.session_state.messages[current_sub].append({"role": "assistant", "content": response_text})
            except Exception as err:
                st.error(f"Failed to generate response: {err}")