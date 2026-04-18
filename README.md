# 🧠 MeetMind — AI Meeting Notes Assistant

Ask questions about your meeting notes using natural language.

## Tech Stack
- **Embeddings**: Google Gemini (gemini-embedding-001)
- **Vector Store**: FAISS
- **LLM**: Groq (Llama 3.1)
- **Framework**: LangChain + Streamlit

## How it works
1. Upload your meeting notes (.txt or .pdf)
2. Notes are split into chunks and embedded using Gemini
3. FAISS stores the vectors locally
4. When you ask a question, the top 3 relevant chunks are retrieved and sent to Groq LLM
5. LLM answers strictly from the meeting context — no hallucination

## Run locally
Clone the repo, create a .env file with your API keys, then run:
  pip install -r requirements.txt
  streamlit run app.py

## Environment variables
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
