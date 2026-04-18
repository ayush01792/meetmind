# rag/chunker.py
# This file splits raw meeting text into overlapping chunks
# so that no important context is lost at chunk boundaries

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split raw text into overlapping chunks.

    Args:
        text: the full meeting notes text
        chunk_size: max characters per chunk (500 is a good default)
        chunk_overlap: how many characters overlap between chunks (avoids cutting sentences)
    Returns:
        list of text chunks
    """

    # RecursiveCharacterTextSplitter is smart — it tries to split at
    # paragraph breaks first, then sentences, then words, then characters.
    # This means chunks are more natural and don't cut mid-sentence.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]  # tries these in order
    )

    chunks = splitter.split_text(text)
    return chunks