import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

st.title("PDF Chatbot (Local Embeddings)")

# ------------------------
# Load FAISS index and chunks
# ------------------------
@st.cache_data(show_spinner=True)
def load_faiss_index():
    index = faiss.read_index("faiss_index.index")
    chunks = np.load("chunks.npy", allow_pickle=True)
    return index, chunks

index, chunks = load_faiss_index()
st.success(f"FAISS index loaded with {index.ntotal} vectors, {len(chunks)} chunks.")

# ------------------------
# Load local embedding model
# ------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()
st.success("Embedding model loaded")

# ------------------------
# Initialize chat history
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# User input
# ------------------------
user_input = st.text_input("Ask me something about the PDF:")

def get_response(query, top_k=3):
    # Embed query using local model
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    
    # Search FAISS index
    D, I = index.search(q_vec, top_k)
    
    # Retrieve top chunks
    responses = [chunks[idx] for idx in I[0]]
    return "\n\n".join(responses)

# ------------------------
# On submit
# ------------------------
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# ------------------------
# Display chat history
# ------------------------
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")



