import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("📘 PDF RAG Chatbot")

# --------------------------
# SESSION STATE FIX (VERY IMPORTANT)
# --------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "history" not in st.session_state:
    st.session_state.history = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# --------------------------
# PDF Upload — RUNS ONLY ONCE
# --------------------------
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

if uploaded_pdf and st.session_state.vector_db is None:
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(uploaded_pdf)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(pages)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    st.session_state.vector_db = FAISS.from_documents(chunks, embedder)
    st.success("Vector DB created successfully!")


# --------------------------
# CHAT INPUT BOX (Fixed, Stable, One Location)
# --------------------------
question = st.text_input("Ask your question:", value="", key="question_box")


# ---------------------------------------------------
# PROCESS QUERY ONLY WHEN BUTTON CLICKED (NOT TYPING)
# ---------------------------------------------------
if st.button("Submit"):
    if question.strip() == "":
        st.warning("Type a question.")
    elif st.session_state.vector_db is None:
        st.error("Upload a PDF first.")
    else:
        # Prevent repeated execution
        if question != st.session_state.last_question:

            st.session_state.last_question = question

            # Retrieve relevant context
            docs = st.session_state.vector_db.similarity_search(question, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            # LLM call
            llm = OpenAI(model="gpt-4o-mini", temperature=0)
            prompt = f"""
            Use this context to answer:

            Context:
            {context}

            Question:
            {question}

            Give a simple answer.
            """
            answer = llm(prompt)

            # Add to history
            st.session_state.history.append(("You", question))
            st.session_state.history.append(("Bot", answer))

        else:
            st.info("Already answered this question.")


# --------------------------
# SHOW ANSWER AREA
# --------------------------
st.markdown("### 🧠 Latest Answer")

if st.session_state.history:
    last_role, last_msg = st.session_state.history[-1]
    if last_role == "Bot":
        st.write(last_msg)


# --------------------------
# CHAT HISTORY
# --------------------------
st.markdown("---")
st.markdown("### 📝 Chat History")

for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑‍💼 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")
