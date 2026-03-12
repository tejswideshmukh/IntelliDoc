"""
IntelliDoc — AI-powered document Q&A
"""

import streamlit as st
from rag import SimpleRAG
import PyPDF2
import io
import groq


# Initialize RAG system (cached to avoid reloading)
@st.cache_resource
def load_rag_system():
    return SimpleRAG()


def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_txt(file):
    """Extract text from TXT file"""
    return file.read().decode('utf-8')


def format_answer(relevant_chunks):
    """Format relevant chunks as answer"""
    if not relevant_chunks:
        return "I couldn't find any relevant information. Please upload a document first."
    
    # Combine relevant chunks with clear formatting
    answer = "**Relevant information from the document:**\n\n"
    for i, chunk in enumerate(relevant_chunks, 1):
        answer += f"**{i}.** {chunk}\n\n---\n\n"
    
    return answer


def main():
    st.set_page_config(page_title="IntelliDoc", page_icon="📖", layout="centered")

    # ── Library / parchment theme ──────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

    /* Main background — warm parchment */
    .stApp {
        background-color: #f5efe0;
        font-family: 'Lora', Georgia, serif;
    }

    /* Sidebar — dark library wood */
    [data-testid="stSidebar"] {
        background-color: #2c1f0e;
    }
    [data-testid="stSidebar"] * {
        color: #e8d5b0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Playfair Display', serif;
        color: #f0c97f !important;
        border-bottom: 1px solid #5a3e1b;
        padding-bottom: 6px;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #5a3e1b;
        color: #f5efe0 !important;
        border: 1px solid #8b6134;
        border-radius: 6px;
        font-family: 'Lora', serif;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #7a5530;
    }

    /* Main title */
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #3b1f0a !important;
        font-size: 2.6rem !important;
        letter-spacing: 1px;
    }

    /* Section headers */
    h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #5a3e1b !important;
    }

    /* Tagline text */
    .tagline {
        color: #7a5530;
        font-family: 'Lora', serif;
        font-style: italic;
        font-size: 1.05rem;
        margin-top: -12px;
        margin-bottom: 20px;
    }

    /* Divider */
    hr {
        border-color: #c9a87c;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #fdf6e3;
        border: 1px solid #d4b896;
        border-radius: 10px;
        padding: 4px 8px;
        margin-bottom: 8px;
    }

    /* Chat input box */
    [data-testid="stChatInput"] textarea {
        background-color: #fdf6e3 !important;
        border: 1px solid #c9a87c !important;
        font-family: 'Lora', serif !important;
        color: #3b1f0a !important;
        border-radius: 8px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #fdf6e3;
        border: 1px solid #c9a87c;
        border-radius: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #3d2b10;
        border: 1px dashed #8b6134;
        border-radius: 8px;
        padding: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("# 📖 IntelliDoc")
    st.markdown('<p class="tagline">Your documents, answered intelligently.</p>', unsafe_allow_html=True)
    st.divider()

    # Initialize RAG system
    rag = load_rag_system()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Upload a Document")

        uploaded_file = st.file_uploader(
            "PDF or TXT files",
            type=['pdf', 'txt'],
            help="Upload a PDF or plain text file to query"
        )

        if uploaded_file is not None:
            doc_id = uploaded_file.name

            if st.session_state.get("loaded_doc") != doc_id:
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text = extract_text_from_txt(uploaded_file)
                else:
                    text = ""

                if text.strip():
                    num_chunks = rag.add_document(text, doc_id)
                    st.session_state["loaded_doc"] = doc_id
                    st.success(f"Loaded — {num_chunks} passages indexed.")
                else:
                    st.error("Could not extract text from file.")

            st.info(f"📖 {uploaded_file.name}")

        st.divider()

        if st.button("Clear Library"):
            rag.reset()
            st.session_state.pop("loaded_doc", None)
            st.success("Library cleared.")
            st.rerun()

        st.markdown("---")
        st.markdown('<p style="font-size:0.75rem; color:#8b6134; font-style:italic;">IntelliDoc uses AI to retrieve and answer from your uploaded documents.</p>', unsafe_allow_html=True)

    # ── Chat area ──────────────────────────────────────────────────────────
    st.markdown("### Ask IntelliDoc")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Reading through the pages..."):
            relevant_chunks = rag.search(question, top_k=3)

        if not relevant_chunks:
            answer = "No relevant passages found. Please upload a document first."
        else:
            try:
                answer = rag.generate_answer(question, relevant_chunks)
            except groq.AuthenticationError:
                answer = "Groq API key is invalid or expired. Please update GROQ_API_KEY in your Streamlit secrets."
            except groq.APIError as e:
                answer = f"Groq API error: {type(e).__name__}: {e}"
            else:
                with st.expander("View source passages"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Passage {i}:** {chunk}")
                        st.divider()
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
