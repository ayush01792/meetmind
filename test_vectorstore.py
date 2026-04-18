# test_vectorstore.py
from rag.loader import load_file
from rag.chunker import chunk_text
from rag.vectorstore import build_vectorstore, search

# Step 1: Load and chunk the meeting notes
text = load_file("sample_notes/meeting1.txt")
chunks = chunk_text(text)

# Step 2: Build the FAISS vector store
index, chunks = build_vectorstore(chunks)
print()

# Step 3: Search with different questions
questions = [
    "What did Ayush get assigned to do?",
    "What are the action items?",
    "What is the Q2 deadline?",
]

for question in questions:
    print(f"Question: {question}")
    results = search(question, index, chunks, top_k=1)
    print(f"Most relevant chunk:\n{results[0]}")
    print()