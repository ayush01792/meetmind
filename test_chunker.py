# test_chunker.py
from rag.loader import load_file
from rag.chunker import chunk_text

# Load the sample meeting notes
text = load_file("sample_notes/meeting1.txt")
print(f"=== Raw text loaded ===")
print(f"Total characters: {len(text)}")
print()

# Chunk it
chunks = chunk_text(text)
print(f"=== Chunks created ===")
print(f"Total chunks: {len(chunks)}")
print()

# Print each chunk with its index
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()