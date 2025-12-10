import streamlit as st
import numpy as np
import faiss_cpu as faiss
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------
# STREAMLIT PAGE SETUP
# ----------------------------------------------------
st.set_page_config(page_title="Local RAG Chatbot", layout="centered")

st.title("📘 Local RAG Chatbot (FAISS + MiniLM)")
st.write("Ask questions based on your uploaded document.")

# ----------------------------------------------------
# LOAD LOCAL EMBEDDING MODEL (CACHED)
# ----------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ----------------------------------------------------
# LOAD FAISS INDEX (CACHED)
# ----------------------------------------------------
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    return index

index = load_faiss_index()

# ----------------------------------------------------
# LOAD CHUNKS (CACHED)
# ----------------------------------------------------
@st.cache_resource
def load_chunks():
    return np.load("chunks.npy", allow_pickle=True)

chunks = load_chunks()

# ----------------------------------------------------
# RAG QUERY FUNCTION WITH THRESHOLD
# ----------------------------------------------------
def query_rag(user_query, top_k=3, threshold=0.25):

    # Embed the query
    vec = embedder.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(vec)

    # FAISS search
    D, I = index.search(vec, top_k)

    best_score = float(D[0][0])

    # Debug similarity score (optional)
    st.write(f"🔍 Similarity Score: {best_score}")

    # Out-of-domain detection
    if best_score < threshold:
        return "I don’t know the answer. It is outside my knowledge."

    # Retrieve best chunks
    retrieved = [chunks[i] for i in I[0]]
    answer = "\n\n".join(retrieved)

    return answer

# ----------------------------------------------------
# CHAT HISTORY
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------------------------------
# CHAT INPUT (THIS FIXES THE REPEATING TEXT ISSUE)
# ----------------------------------------------------
user_input = st.chat_input("Ask anything from your document...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get RAG response
    bot_response = query_rag(user_input)

    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Immediately display bot message
    with st.chat_message("assistant"):
        st.write(bot_response)
