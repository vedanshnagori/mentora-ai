# Mentora AI - Personalized Agentic Study Assistant

Mentora AI is a syllabus-grounded, pedagogical study companion designed to eliminate the friction of manual, page-by-page document ingestion[cite: 3]. It ingests complete digital textbooks and scanned handwritten notes into partitioned workspaces, delivering high-speed tutoring grounded with exact document and page citations[cite: 1, 2].

---

## Architecture Overview

```
[ User Interface (Streamlit) ]
       │
       ├── Subject Isolation Workspace (`subject_id` partition)
       └── Dual-Track Ingestion Pipeline
             │
             ├── Digital PDFs ──────► PyMuPDF4LLM (Layout & Markdown Parsing)
             │
             └── Handwritten Notes ─► pdf2image + Gemini 2.5 Flash VLM
                                             │
                                             ▼
                                [ Structural Heading Chunking ]
                                (Hierarchical Breadcrumbs & MD5 Hashes)
                                             │
                                             ▼
                                [ ChromaDB Vector Database ]
                                (Local CPU BAAI/bge-small-en-v1.5 / Gemini)
                                             │
                                             ▼
                                [ Pedagogical Tutor Engine ]
                                (Groq Llama 3.3 / Gemini 2.5 Flash + Citations)
```

---

## Key Features

- **Dual-Track Document Ingestion:**
  - **Digital PDFs:** Preserves tables, headers, and multi-column layouts as structured Markdown using `pymupdf4llm`[cite: 2].
  - **Handwritten Notes:** Converts pages to high-DPI images with `pdf2image` and transcribes them via Gemini Vision, preserving LaTeX formulas ($...$) and diagram descriptions[cite: 2].
- **Structural Heading-Based Chunking:** Splits text along Markdown header hierarchies (`#`, `##`, `###`), attaching exact page numbers, content-derived MD5 hashes (`doc_id`), and section breadcrumbs to metadata[cite: 2].
- **Subject-Isolated Vector Store:** Partitions ChromaDB storage using strict metadata filters (`where={"subject_id": active_subject}`) to eliminate cross-subject retrieval contamination[cite: 2].
- **Pedagogical Grounded Tutoring:** Generates conversational explanations with verified source citations (`[filename, p. X]`) and closes explanations with comprehension checks[cite: 2].
- **Agentic CRAG Ready:** Supports Corrective RAG routing to self-evaluate retrieval relevance and eliminate hallucinations[cite: 1, 6].

---

## Tech Stack

- **Orchestration & LLM Framework:** LangChain, Google GenAI SDK[cite: 2]
- **Inference Models:** 
  - Vision & OCR: Gemini 2.5 Flash / Groq Vision[cite: 2]
  - Tutoring & Synthesis: Groq `llama-3.3-70b-versatile` / Gemini 2.5 Flash[cite: 2]
- **Embeddings:** `BAAI/bge-small-en-v1.5` (via `sentence-transformers`) / `gemini-embedding-001`[cite: 2]
- **Vector Database:** ChromaDB[cite: 2]
- **Document Processing:** PyMuPDF4LLM, pdf2image, Pillow[cite: 2]
- **Frontend / Application Server:** Streamlit[cite: 2]
- **Package Manager:** `uv`[cite: 2]

---

## Project Structure

```text
mentora-ai/
├── data/
│   └── chroma_db/            # Local vector persistence directory
├── src/
│   ├── __init__.py
│   ├── ingest_printed.py     # Markdown extraction for clean PDFs
│   ├── ingest_handwritten.py # VLM transcription for handwritten notes
│   └── rag_engine.py         # Embedding generation, vector storage, and QA
├── app.py                    # Multi-workspace Streamlit UI
├── pyproject.toml            # Project dependencies and configs
├── .env                      # API keys (GEMINI_API_KEY, GROQ_API_KEY)
└── .gitignore
```

---

## Getting Started

### 1. System Prerequisites

Install `poppler-utils` (required by `pdf2image` for rendering PDF pages):

```bash
# Fedora / RHEL
sudo dnf install -y poppler-utils

# Ubuntu / Debian
sudo apt-get install -y poppler-utils
```

### 2. Environment Setup

Clone the repository and sync dependencies using `uv` with Python 3.12:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/mentora-ai.git
cd mentora-ai

# Pin Python 3.12 and install dependencies
uv python pin 3.12
uv sync
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
```

### 4. Running the Application

Launch the Streamlit interface:

```bash
uv run streamlit run app.py
```

---

## Usage

1. Open `http://localhost:8501` in your browser.
2. Create or select a subject workspace in the sidebar (e.g., `Physics 101`).
3. Expand **Upload Material** and select either **Printed Textbook / Slides** or **Handwritten Notes**.
4. Click **Index Document into Subject**.
5. Query your notes in the chat bar—every answer will cite specific pages and document names.
