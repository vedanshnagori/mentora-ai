import hashlib
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

def calculate_file_hash(file_path: Path) -> str:
    """Generates an MD5 hash of the file to uniquely identify document versions."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_printed_pdf(file_path: str | Path, subject_id: str) -> list[Document]:
    """
    Extracts page-aware Markdown, splits hierarchically by header and size,
    and attaches comprehensive, ChromaDB-safe citation metadata.
    """
    file_path = Path(file_path)
    doc_id = calculate_file_hash(file_path)
    
    # Extract page chunks with 1-based page metadata
    pages_data = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

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
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )

    all_chunks: list[Document] = []
    global_chunk_idx = 0

    for page in pages_data:
        raw_text = page.get("text", "")
        if not raw_text.strip():
            continue

        page_meta = page.get("metadata", {})
        page_number = int(page_meta.get("page_number", 1))

        # Split on headers first, then text length
        header_docs = markdown_splitter.split_text(raw_text)
        page_docs = text_splitter.split_documents(header_docs)

        for doc in page_docs:
            # Reconstruct header hierarchy into a readable breadcrumb
            headers = [
                doc.metadata.get("Header 1"),
                doc.metadata.get("Header 2"),
                doc.metadata.get("Header 3"),
            ]
            section_path = " > ".join([h for h in headers if h]) or "General Content"

            # Assign strict, sanitized metadata
            doc.metadata = {
                "subject_id": str(subject_id),
                "doc_id": doc_id,
                "source_file": file_path.name,
                "source_type": "printed_pdf",
                "page_number": page_number,
                "section_path": section_path,
                "chunk_index": global_chunk_idx,
            }
            all_chunks.append(doc)
            global_chunk_idx += 1

    return all_chunks