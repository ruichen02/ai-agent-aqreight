import os
import re
import hashlib
from typing import List, Dict, Tuple
from .settings import settings

ALLOWED_EXTENSIONS = (".md", ".txt")


def _read_text_file(path: str) -> str:
    """Read file content safely."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Failed to read {path}: {e}")
        return ""


def _md_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split Markdown or text into (section_title, body) tuples.
    Preserves hierarchical headings: # Title / ## Subsection
    """
    title = None
    sections = []
    current_section = None
    buffer = []

    lines = text.splitlines()
    for line in lines:
        if line.startswith("# "):
            if title is None:
                title = line.lstrip("# ").strip()
            else:
                # Flush last section
                if current_section and buffer:
                    sections.append((f"{title} / {current_section}", "\n".join(buffer).strip()))
                    buffer = []
                current_section = None
                title = line.lstrip("# ").strip()
        elif line.startswith("## "):
            if current_section and buffer:
                sections.append((f"{title} / {current_section}", "\n".join(buffer).strip()))
                buffer = []
            current_section = line.lstrip("#").strip()
        else:
            buffer.append(line)

    # Add last buffered section
    if title and current_section and buffer:
        sections.append((f"{title} / {current_section}", "\n".join(buffer).strip()))
    elif title and buffer:
        sections.append((title, "\n".join(buffer).strip()))

    return sections or [("Body", text.strip())]

def _txt_sections(fname: str, text: str) -> List[Tuple[str, str]]:
    """
    Wrap plain text into a single (title, body) section using filename as title.
    """
    title = os.path.splitext(fname)[0]  # filename without extension
    return [(title, text.strip())]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word-based chunks."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks


def doc_hash(text: str) -> str:
    """Compute SHA256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents(data_dir: str) -> List[Dict]:
    """Load all allowed files from data_dir, return structured docs with hashes."""
    docs = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(ALLOWED_EXTENSIONS):
            continue

        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        if not text.strip():
            continue

        # Use the right section parser
        if fname.lower().endswith(".md"):
            sections = _md_sections(text)
        else:
            sections = _txt_sections(fname, text)

        # Build doc entries
        for section_title, body in sections:
            h = doc_hash(body)
            docs.append({
                "title": section_title,
                "section": section_title.split(" / ")[-1] if " / " in section_title else section_title,
                "text": body,
                "file": fname,
                "hash": h
            })

    return docs


