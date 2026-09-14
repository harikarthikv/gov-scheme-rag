"""Streamlit interface for the government-scheme PDF assistant."""

import streamlit as st

from src.bootstrap import create_rag_pipeline

st.set_page_config(page_title="Government scheme assistant", layout="wide")
st.session_state.setdefault("rag", None)
st.session_state.setdefault("chat_history", [])


@st.cache_resource(show_spinner="Loading the embedding model and index...")
def load_rag_pipeline():
    """Create one reusable pipeline for this Streamlit process."""
    return create_rag_pipeline()


def show_sources(sources: list[dict], scores: list[float]) -> None:
    """Show retrieval metadata without exposing internal Chroma objects."""
    if not sources:
        return
    with st.expander("Retrieved PDF pages"):
        for source, score in zip(sources, scores):
            name = source.get("scheme_name", source.get("filename", "Unknown"))
            st.caption(f"{name} — page {source.get('page', '?')} (relevance: {score:.4f})")


with st.sidebar:
    st.header("Government scheme assistant")
    st.caption("Answers are generated from the PDFs in `data/gov_myscheme`.")
    if st.session_state.rag is None:
        if st.button("Load assistant", type="primary", icon=":material/play_arrow:"):
            try:
                st.session_state.rag = load_rag_pipeline()
            except Exception as error:
                st.error(f"Could not load the assistant: {error}")
    elif st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.chat_history = []
        st.rerun()

st.title("Government scheme PDF assistant")
st.write("Ask about information contained in the locally indexed scheme PDFs.")

if st.session_state.rag is None:
    st.info("Select **Load assistant** in the sidebar before asking a question.")
    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []), message.get("scores", []))

if question := st.chat_input("Ask a question about a scheme"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDFs and generating an answer..."):
            try:
                result = st.session_state.rag.query(question)
                st.markdown(result["answer"])
                show_sources(result["sources"], result["similarity_scores"])
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "scores": result["similarity_scores"],
                    }
                )
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
