"""Web UI:  streamlit run streamlit_app.py"""
import streamlit as st
from dotenv import load_dotenv
from rag import load_documents, chunk_documents, Retriever, generate_answer

load_dotenv()
st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚")
st.title("📚 RAG AI Knowledge Assistant")


@st.cache_resource
def get_retriever(folder: str):
    return Retriever(chunk_documents(load_documents(folder)))


folder = st.sidebar.text_input("Knowledge folder", "data")
top_k = st.sidebar.slider("Chunks to retrieve", 1, 6, 3)
try:
    retriever = get_retriever(folder)
    st.sidebar.success(f"{len(retriever.chunks)} chunks indexed")
except Exception as e:
    st.error(str(e))
    st.stop()

query = st.text_input("Ask a question about your documents")
if query:
    hits = retriever.search(query, top_k)
    answer, mode = generate_answer(query, hits)
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Mode: {mode}")
    if hits:
        st.subheader("Sources")
        for h in hits:
            with st.expander(f"{h.chunk.source} #{h.chunk.index} - score {h.score:.2f}"):
                st.write(h.chunk.text)
