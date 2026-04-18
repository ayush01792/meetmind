# rag/chain.py
# This is the brain of MeetMind.
# It takes a user question, finds relevant chunks using FAISS,
# and sends them to Groq LLM to generate a grounded answer.

import os
from groq import Groq
from dotenv import load_dotenv
from rag.vectorstore import search

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt — this tells the LLM how to behave
# "Only answer from the context" prevents hallucination
SYSTEM_PROMPT = """You are MeetMind, an intelligent meeting assistant.
You help users recall information from their meeting notes.

Rules:
- Answer ONLY using the context provided below. Do not use outside knowledge.
- If the answer is not in the context, say "I couldn't find this in the meeting notes."
- Be concise and direct.
- If asked for action items or summaries, format them as bullet points.
"""

def ask(query: str, index, chunks: list[str]) -> dict:
    """
    Full RAG pipeline — retrieve relevant chunks and generate an answer.

    Args:
        query: the user's question
        index: FAISS index
        chunks: original text chunks
    Returns:
        dict with 'answer' and 'sources' (the chunks used to answer)
    """
    # Step 1: Retrieve the top 3 most relevant chunks
    relevant_chunks = search(query, index, chunks, top_k=3)

    # Step 2: Join chunks into a single context string
    context = "\n\n---\n\n".join(relevant_chunks)

    # Step 3: Build the prompt — context + question
    user_message = f"""Context from meeting notes:
{context}

Question: {query}"""

    # Step 4: Send to Groq LLM and get the answer
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.2,  # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content

    # Return both the answer and the source chunks for transparency
    return {
        "answer": answer,
        "sources": relevant_chunks
    }