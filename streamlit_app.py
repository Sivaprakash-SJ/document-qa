import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="RAG App", layout="wide")

st.title("📘 PDF based RAG Chatbot")

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# ---------------------------
# PDF UPLOAD + VECTOR DB BUILD
# ---------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(uploaded_file)
    doc_pages = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(doc_pages)

    # Embedding model
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Build vector DB
    st.session_state.vector_db = FAISS.from_documents(docs, embed)

    st.success("PDF processed & vector database created!")


# ---------------------------
# CHAT UI
# ---------------------------
st.subheader("💬 Ask your question")

query = st.text_input("Type your question here:")

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please type a question.")
    elif st.session_state.vector_db is None:
        st.error("Please upload a PDF first.")
    else:
        # Retrieve from vector DB
        docs = st.session_state.vector_db.similarity_search(query, k=3)

        context = "\n\n".join([d.page_content for d in docs])

        # LLM Call (example OpenAI)
        llm = OpenAI(model="gpt-4o-mini", temperature=0)

        prompt = f"""
        You are a PDF-based assistant.

        Context:
        {context}

        Question:
        {query}

        Answer in simple terms:"""

        answer = llm(prompt)

        # Store conversation
        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(("assistant", answer))

        # Display the answer once
        st.write("### 🧠 Answer:")
        st.write(answer)


# ---------------------------
# SHOW CHAT HISTORY
# ---------------------------
st.markdown("---")
st.write("### 📝 Previous Q&A")

for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg}")

