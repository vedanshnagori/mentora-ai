import hashlib
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

VLM_PROMPT = """
You are an expert academic transcriber. Transcribe this handwritten study material into clean, structured Markdown:
1. Accurately transcribe all handwritten text.
2. Render mathematical equations, formulas, and symbols in valid LaTeX ($...$ for inline, $$...$$ for display).
3. If there is a diagram, chart, or drawing, summarize it in brackets: [Diagram: describe contents and labels].
4. If a word or symbol is completely illegible, write [unclear] instead of guessing.
5. Use standard Markdown headers (#, ##, ###) if you detect section titles or headings.
Output only the transcribed Markdown without conversational filler.
"""

def calculate_file_hash(file_path: Path) -> str:
    """Generates an MD5 hash of the file to uniquely identify document versions."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def transcribe_page_image(image_bytes: bytes) -> str:
    """Sends a single image to Gemini Vision for transcription."""
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            VLM_PROMPT
        ]
    )
    return response.text or ""

def process_handwritten_pdf(file_path: str | Path, subject_id: str) -> list[Document]:
    """
    Renders PDF pages as images, transcribes them via Gemini Vision, 
    splits the Markdown structurally, and attaches sanitized metadata.
    """
    file_path = Path(file_path)
    doc_id = calculate_file_hash(file_path)
    
    # Convert PDF pages to PIL images at 200 DPI
    images = convert_from_path(str(file_path), dpi=200)
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", " "]
    )

    all_chunks: list[Document] = []
    global_chunk_idx = 0

    for page_num, img in enumerate(images, start=1):
        # Convert PIL Image to bytes for transmission
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        
        raw_text = transcribe_page_image(img_buffer.getvalue())
        
        if not raw_text.strip():
            continue

        # Split logically by headers first, then length
        header_docs = markdown_splitter.split_text(raw_text)
        page_docs = text_splitter.split_documents(header_docs)

        # Attach metadata matching the printed pipeline
        for doc in page_docs:
            headers = [
                doc.metadata.get("Header 1"),
                doc.metadata.get("Header 2"),
                doc.metadata.get("Header 3"),
            ]
            section_path = " > ".join([h for h in headers if h]) or "General Content"

            doc.metadata = {
                "subject_id": str(subject_id),
                "doc_id": doc_id,
                "source_file": file_path.name,
                "source_type": "handwritten_notes",
                "page_number": page_num,
                "section_path": section_path,
                "chunk_index": global_chunk_idx,
            }
            all_chunks.append(doc)
            global_chunk_idx += 1
            
    return all_chunks