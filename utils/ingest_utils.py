
import os
from typing import List, Dict


# Lightweight extractors
def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def extract_text_from_pdf(path: str) -> str:
    # Try PyPDF2
    try:
        import PyPDF2
        text_chunks = []
        with open(path, "rb") as fp:
            reader = PyPDF2.PdfReader(fp)
            for page in reader.pages:
                text_chunks.append(page.extract_text() or "")
        return "\n\n".join(text_chunks)
    except Exception:
        # Fallback to pdfplumber if installed
        try:
            import pdfplumber
            text_chunks = []
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    text_chunks.append(p.extract_text() or "")
            return "\n\n".join(text_chunks)
        except Exception as e:
            raise RuntimeError("PDF extraction failed: install PyPDF2 or pdfplumber") from e

# Chunking utility
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Splits text into overlapping chunks. Returns list of dicts: {'text': ..., 'chunk_index': i, 'source': None}
    """
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    start = 0
    chunks = []
    idx = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "chunk_index": idx})
        idx += 1
        start = end - chunk_overlap
    return chunks
