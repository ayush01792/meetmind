import os 
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# connect to gemini 



print("=== TEST 1: Embedding ===")

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Ayush was assigned to own the backend API changes."
)

vector = result.embeddings[0].values
print(f"Vector length: {len(vector)}")    # should print 3072
print(f"First 5 numbers: {vector[:5]}")  # actual numbers
print()

print("=== TEST 2: LLM (Groq) ===")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",   # free model on Groq
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)
print(response.choices[0].message.content)