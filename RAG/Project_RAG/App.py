import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="📚 RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

PERSIST_DIR = "B:\\AI\\RAG\\Project_RAG\\DB"

# ---------------------------------------------------------------------
# Cached resources (so we don't reload models on every rerun)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return MistralAIEmbeddings(model="mistral-embed")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small")


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def get_retriever(k: int = 3, fetch_k: int = 10, lambda_mult: float = 0.5):
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
        Only use the information provided in the context to answer the question.

        If the answer is not in the context, respond with:
        "I don't know based on the given context."
        Do not invent or assume information.
        """,
    ),
    (
        "human",
        """Context:
        {context}

        Question:
        {question}
        """,
    ),
])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_file_to_documents(uploaded_file):
    """Save uploaded file to a temp path and load with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        docs = loader.load()
        # Attach source name so we can show/filter later
        for d in docs:
            d.metadata["source_name"] = uploaded_file.name
        return docs
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def ingest_files(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Load, split and add uploaded files into the Chroma vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    total_chunks = 0
    vs = get_vectorstore()

    progress = st.progress(0.0, text="Preparing…")
    n = len(uploaded_files)

    for i, uf in enumerate(uploaded_files, start=1):
        progress.progress((i - 1) / n, text=f"Loading {uf.name}…")
        docs = load_file_to_documents(uf)

        progress.progress((i - 0.5) / n, text=f"Splitting & embedding {uf.name}…")
        chunks = splitter.split_documents(docs)
        if chunks:
            vs.add_documents(chunks)
            total_chunks += len(chunks)

        progress.progress(i / n, text=f"Done {uf.name}")

    progress.empty()
    return total_chunks


def answer_question(question: str, k: int, fetch_k: int, lambda_mult: float):
    retriever = get_retriever(k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    final_prompt = prompt_template.format(
        context=context,
        question=question,
    )
    result = get_llm().invoke(final_prompt)
    return result.content, docs


# ---------------------------------------------------------------------
# Sidebar — Upload & settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Knowledge Base")

    st.subheader("Upload books / documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    with st.expander("⚙️ Chunking settings", expanded=False):
        chunk_size = st.slider("Chunk size", 200, 4000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 800, 150, step=50)

    if st.button("📥 Ingest uploaded files", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Ingesting into vector store…"):
                try:
                    added = ingest_files(uploaded_files, chunk_size, chunk_overlap)
                    st.success(f"✅ Added {added} chunks from {len(uploaded_files)} file(s).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    st.subheader("🔎 Retrieval settings")
    k = st.slider("k (returned)", 1, 10, 3)
    fetch_k = st.slider("fetch_k (pool)", 1, 30, 10)
    lambda_mult = st.slider("MMR λ (diversity ↔ relevance)", 0.0, 1.0, 0.5, 0.05)

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------
# Main — Chat
# ---------------------------------------------------------------------
st.title("📚 Chat with your books")
st.caption("A RAG app powered by Mistral + Chroma + LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources used"):
                for i, s in enumerate(msg["sources"], start=1):
                    src = s.get("source_name") or s.get("source") or "unknown"
                    page = s.get("page")
                    header = f"**{i}. {src}**" + (f" — page {page}" if page is not None else "")
                    st.markdown(header)
                    st.write(s["preview"])

# Chat input
question = st.chat_input("Ask a question about your uploaded books…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, docs = answer_question(question, k, fetch_k, lambda_mult)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                docs = []

        st.markdown(answer)

        sources = []
        if docs:
            with st.expander("📄 Sources used"):
                for i, d in enumerate(docs, start=1):
                    meta = d.metadata or {}
                    src = meta.get("source_name") or meta.get("source") or "unknown"
                    page = meta.get("page")
                    header = f"**{i}. {src}**" + (f" — page {page}" if page is not None else "")
                    st.markdown(header)
                    preview = d.page_content[:500] + ("…" if len(d.page_content) > 500 else "")
                    st.write(preview)
                    sources.append({
                        "source_name": src,
                        "page": page,
                        "preview": preview,
                    })

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })