# app.py
# The main Streamlit UI for MeetMind
# Run with: streamlit run app.py

import streamlit as st
from rag.loader import load_file
from rag.chunker import chunk_text
from rag.vectorstore import build_vectorstore
from rag.chain import ask
import tempfile
import os

# --- Page config ---
st.set_page_config(
    page_title="MeetMind",
    page_icon="🧠",
    layout="centered"
)

# --- Header ---
st.title("🧠 MeetMind")
st.caption("Upload your meeting notes and ask anything about them.")
st.divider()

# --- Session state ---
# Streamlit reruns the whole script on every interaction.
# We use session_state to persist the index and chunks across reruns.
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File upload ---
st.subheader("📄 Upload Meeting Notes")
uploaded_file = st.file_uploader(
    "Upload a .txt or .pdf file",
    type=["txt", "pdf"]
)

if uploaded_file is not None:
    # Check if this is a new file (avoid re-processing same file)
    if st.session_state.get("uploaded_filename") != uploaded_file.name:

        with st.spinner("Processing your meeting notes..."):
            # Save uploaded file to a temp location so loader can read it
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Run the RAG pipeline
            text = load_file(tmp_path)
            chunks = chunk_text(text)
            index, chunks = build_vectorstore(chunks)

            # Store in session state so it persists
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.chat_history = []  # reset chat for new file

            os.unlink(tmp_path)  # clean up temp file

        st.success(f"✅ Processed **{uploaded_file.name}** — {len(chunks)} chunks indexed. Ask away!")

# --- Quick action buttons ---
if st.session_state.index is not None:
    st.divider()
    st.subheader("⚡ Quick Questions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Summarize this meeting"):
            st.session_state.quick_query = "Give me a summary of this meeting."
        if st.button("✅ List all action items"):
            st.session_state.quick_query = "What are all the action items from this meeting?"
    with col2:
        if st.button("📅 What deadlines were mentioned?"):
            st.session_state.quick_query = "What deadlines were mentioned in this meeting?"
        if st.button("❓ What problems were discussed?"):
            st.session_state.quick_query = "What problems or concerns were discussed?"

# --- Chat interface ---
if st.session_state.index is not None:
    st.divider()
    st.subheader("💬 Ask Anything")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle quick action button queries
    default_query = st.session_state.pop("quick_query", "")

    # Chat input
    user_input = st.chat_input("Ask about your meeting...") or default_query

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(user_input, st.session_state.index, st.session_state.chunks)
            st.write(result["answer"])

            # Show source chunks in an expander
            with st.expander("📎 View source chunks used"):
                for i, source in enumerate(result["sources"]):
                    st.caption(f"Source {i+1}")
                    st.text(source)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"]
        })

# --- Empty state ---
if st.session_state.index is None:
    st.info("👆 Upload a meeting notes file above to get started.")