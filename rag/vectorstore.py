# rag/vectorstore.py
# This file handles:
# 1. Converting text chunks into vectors (embeddings) using Gemini
# 2. Storing those vectors in a FAISS index
# 3. Searching the index for the most relevant chunks given a query

import os
import faiss
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini client for embeddings
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    """
    Convert a single piece of text into a vector using Gemini embeddings.

    Args:
        text: any string — a chunk or a question
    Returns:
        list of floats (the vector)
    """
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def build_vectorstore(chunks: list[str]) -> tuple:
    """
    Take a list of text chunks, embed each one, and store in FAISS.

    Args:
        chunks: list of text chunks from the chunker
    Returns:
        (faiss_index, chunks) — the index and original chunks
        We return both because FAISS only stores vectors, not the original text.
        We need the original chunks to display the answer source later.
    """
    print(f"Embedding {len(chunks)} chunks...")

    # Embed every chunk and collect into a list
    embeddings = []
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        embeddings.append(vector)
        print(f"  Embedded chunk {i+1}/{len(chunks)}")

    # Convert to numpy array — FAISS requires numpy float32 format
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Get the vector dimension (3072 for gemini-embedding-001)
    dimension = embeddings_np.shape[1]

    # Create a FAISS index using L2 distance (Euclidean)
    # IndexFlatL2 = simple exact search, great for small datasets like meeting notes
    index = faiss.IndexFlatL2(dimension)

    # Add all vectors to the index
    index.add(embeddings_np)

    print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")
    return index, chunks


def search(query: str, index, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    Given a user question, find the top_k most relevant chunks.

    Args:
        query: the user's question
        index: the FAISS index
        chunks: the original text chunks (parallel to the index)
        top_k: how many chunks to return (3 is usually enough)
    Returns:
        list of the most relevant text chunks
    """
    # Embed the query using the same model as the chunks
    # This is crucial — query and chunks must use the same embedding model
    query_vector = get_embedding(query)

    # Convert to numpy float32 array, reshape to (1, dimension) for FAISS
    query_np = np.array([query_vector], dtype=np.float32)

    # Search the index — returns distances and indices of nearest neighbors
    distances, indices = index.search(query_np, top_k)

    # Retrieve the actual text chunks using the returned indices
    results = []
    for idx in indices[0]:
        if idx < len(chunks):  # safety check
            results.append(chunks[idx])

    return results