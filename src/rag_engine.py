import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

load_dotenv()

class GeminiEmbeddings(Embeddings):
    """Embeddings powered by Google gemini-embedding-001."""
    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return response.embeddings[0].values

CHROMA_PERSIST_DIR = "data/chroma_db"

def get_vector_store() -> Chroma:
    embeddings = GeminiEmbeddings()
    return Chroma(
        collection_name="mentora_study_notes",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

def add_documents_to_db(documents: list[Document]):
    if not documents:
        return
    db = get_vector_store()
    db.add_documents(documents)


def ask_study_assistant(query: str, subject_id: str, chat_history: list[dict]) -> tuple[str, list[Document]]:
    db = get_vector_store()
    
    # Isolate retrieval strictly to the active subject workspace
    retriever = db.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"subject_id": subject_id}
        }
    )
    relevant_docs = retriever.invoke(query)
    
    # Format context with citation headers and section breadcrumbs
    context_blocks = []
    for d in relevant_docs:
        meta = d.metadata
        source = meta.get("source_file", "Unknown")
        page = meta.get("page_number", "N/A")
        section = meta.get("section_path", "")
        
        header = f"[Source: {source}, Page: {page}]"
        if section and section != "General Content":
            header += f"\n[Section: {section}]"
            
        context_blocks.append(f"{header}\n{d.page_content}")
        
    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""
You are Mentora AI, an intelligent, warm, and clear study tutor for '{subject_id}'.
Base your explanation on the student's study materials below whenever relevant.

Study Material Context:
{context_str}

Guidelines:
1. Synthesize concepts clearly and pedagogically.
2. If the user asks about something not present in their notes, answer using general knowledge, but explicitly state that it was not found in their uploaded notes.
3. CITATION RULE: Whenever referencing a concept, theorem, or equation from the notes, cite the exact source file and page (e.g., [Lecture1.pdf, p. 5]).
4. Use standard LaTeX for equations ($...$ for inline, $$...$$ for standalone).
5. Conclude with a single brief question to verify comprehension.
"""

    # Build contents for the Gemini SDK
    contents = [system_prompt]
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Tutor"
        contents.append(f"{role}: {msg['content']}")
    contents.append(f"User: {query}")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="\n\n".join(contents),
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
        ),
    )
    
    return response.text or "", relevant_docs