import streamlit as st

from app.rag import MedicalRAG


# Page configuration
st.set_page_config(
    page_title="Medical Research RAG",
    page_icon="🧬",
    layout="wide",
)


# Load RAG system
@st.cache_resource
def load_rag():
    return MedicalRAG(
        index_path="data/medical_rag.index",
        documents_path="data/documents.pkl",
        model_path="models/all-MiniLM-L6-v2",
    )


rag = load_rag()


# Title
st.title("🧬 Medical Research RAG")

st.write(
    "Ask questions about the biomedical research corpus."
)


# Question input
question = st.text_input(
    "Ask a research question:"
)


# Ask button
if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching the research..."):

            result = rag.ask(question)

        st.subheader("Answer")

        st.write(result["answer"])

        st.subheader("Sources")

        if result["sources"]:

            for source in result["sources"]:
                st.write(
                    f"- Abstract ID: {source}"
                )

        else:
            st.write("No relevant sources found.")