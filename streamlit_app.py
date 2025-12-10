import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------
# Load Embedding Model (LOCAL MODEL)
# ---------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------------------
# Load FAISS Index
# ---------------------------------------------
@st.cache_resource
def load_faiss():
    index = faiss.read_index("faiss_index.index")
    return index

index = load_faiss()

# ---------------------------------------------
# Load Chunks
# ---------------------------------------------
@st.cache_resource
def load_chunks():
    return np.load("chunks.npy", allow_pickle=True)

chunks = load_chunks()

# ---------------------------------------------
# Query Function with Threshold
# ---------------------------------------------
def get_response(query, top_k=3, threshold=0.15):
    # Embed query
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)

    # Search FAISS
    D, I = index.search(q_vec, top_k)

    best_score = float(D[0][0])

    # Debug: Show similarity score
    st.write("🔍 Similarity Score:", best_score)

    # Unknown-question detection
    if best_score < threshold:
        return "I don’t know the answer. It is outside my knowledge."

    # Retrieve matching chunks
    results = [chunks[idx] for idx in I[0]]
    return "\n\n".join(results)

# ---------------------------------------------
# Streamlit Chat UI
# ---------------------------------------------
st.title("📘 Local RAG Chatbot (FAISS + MiniLM)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat history display
for role, message in st.session_state.chat_history:
    if role == "You":
        st.write(f"🧑 **You:** {message}")
    else:
        st.write(f"🤖 **Bot:** {message}")

# User input
user_input = st.text_input("Ask your question:")

# Handle query
if user_input:
    answer = get_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))
    st.rerun()
