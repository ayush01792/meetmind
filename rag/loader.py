# rag/loader.py
import os
from pypdf import PdfReader

def load_file(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:  # ← must use file_path, not a hardcoded path
            text = f.read()

    elif extension == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

    else:
        raise ValueError(f"Unsupported file type: {extension}. Use .txt or .pdf")

    return text.strip()