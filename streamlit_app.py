import streamlit as st
import faiss_cpu as faiss
import numpy as np
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="Local RAG Chatbot", layout="wide")

# -----------------------------------
# INITIALIZE SESSION STATE VARIABLES
# -----------------------------------
if "vector" not in st.session_state:
    st.session_state.vector = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# SHOW Chat History Like ChatGPT
# -----------------------------------
st.title("📘 Local PDF RAG Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------------
# PDF Upload and Vector Creation (Runs Once Only)
# -----------------------------------
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

if uploaded_pdf and st.session_state.vector is None:
    loader = PyPDFLoader(uploaded_pdf)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    st.session_state.vector = FAISS.from_documents(chunks, embedder)

    st.success("PDF successfully processed and Vector DB is ready!")


# -----------------------------------
# CHAT INPUT — THIS IS THE PART THAT FIXES EVERYTHING
# -----------------------------------
user_input = st.chat_input("Ask something about the PDF...")

if user_input and st.session_state.vector:

    # 1️⃣ Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2️⃣ Search in FAISS
    results = st.session_state.vector.similarity_search(user_input, k=3)

    if len(results) == 0:
        bot_reply = "I don’t know the answer. It is outside my knowledge."
    else:
        context = "\n".join([d.page_content for d in results])

        # Since you're using local mode (no OpenAI), the answer must be simple
        # Here we return the extracted context directly
        bot_reply = f"Here is what I found:\n\n{context}"

    # 3️⃣ Save bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # 4️⃣ Display bot answer
    with st.chat_message("assistant"):
        st.write(bot_reply)


elif user_input and st.session_state.vector is None:
    st.warning("Please upload a PDF first!")
