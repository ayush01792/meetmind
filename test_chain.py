# test_chain.py
from rag.loader import load_file
from rag.chunker import chunk_text
from rag.vectorstore import build_vectorstore
from rag.chain import ask

# Step 1: Load, chunk, and index the meeting notes
text = load_file("sample_notes/meeting1.txt")
chunks = chunk_text(text)
index, chunks = build_vectorstore(chunks)
print()

# Step 2: Ask questions
questions = [
    "What did Ayush get assigned to do?",
    "What are all the action items from this meeting?",
    "What is the Q2 deadline?",
    "Who is handling the frontend migration?"
]

for question in questions:
    print(f"Q: {question}")
    result = ask(question, index, chunks)
    print(f"A: {result['answer']}")
    print()