import streamlit as st
import numpy as np
import faiss_cpu as faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Local RAG Chatbot", layout="centered")
st.title("📘 Local RAG Chatbot")
st.write("Ask questions based on your uploaded document.")

# -----------------------------
# Sidebar: Threshold setting
# -----------------------------
RAG_THRESHOLD = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,  # default 0.25
    step=0.01
)

st.sidebar.write(f"Current similarity threshold: {RAG_THRESHOLD}")

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "vector" not in st.session_state:
    st.session_state.vector = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

# -----------------------------
# Load embeddings model (cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")  # smaller model for Cloud

embedder = load_embedder()

# -----------------------------
# Load FAISS index and chunks (cached)
# -----------------------------
@st.cache_resource
def load_faiss_and_chunks():
    try:
        index = faiss.read_index("faiss_index.index")
        chunks = np.load("chunks.npy", allow_pickle=True)
        return index, chunks
    except Exception as e:
        st.warning("FAISS index or chunks not found. Upload files first.")
        return None, None

if st.session_state.vector is None or st.session_state.chunks is None:
    st.session_state.vector, st.session_state.chunks = load_faiss_and_chunks()

index = st.session_state.vector
chunks = st.session_state.chunks

# -----------------------------
# RAG query function
# -----------------------------
def query_rag(user_query, top_k=3, threshold=RAG_THRESHOLD):
    # Embed the query
    vec = embedder.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(vec)

    # FAISS search
    D, I = index.search(vec, top_k)
    best_score = float(D[0][0])

    # Debug similarity score (optional)
    st.write(f"🔍 Similarity Score: {best_score:.3f}")

    # If similarity is below threshold, return default message
    if best_score < threshold:
        return "I don’t know the answer. It is outside my knowledge."

    # Retrieve best chunks
    retrieved = [chunks[i] for i in I[0]]
    answer = "\n\n".join(retrieved)
    return answer

# -----------------------------
# Display chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask anything from your document...")

if user_input:
    # Prevent re-executing the same query
    if user_input != st.session_state.last_query:
        st.session_state.last_query = user_input

        if index is None or chunks is None:
            st.warning("FAISS index or chunks not loaded. Upload files first.")
        else:
            # Save user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Run RAG query
            answer = query_rag(user_input)

            # Save bot message
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display bot message immediately
            with st.chat_message("assistant"):
                st.write(answer)
    else:
        st.info("You already asked this question.")
