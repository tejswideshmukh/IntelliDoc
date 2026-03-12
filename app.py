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

    # ── Dark & sleek theme — amber gold accents ────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

    /* Base */
    html, body, [class*="css"], .stApp {
        background-color: #0d0d0d !important;
        color: #e8e8e8 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #1f1f1f;
    }
    [data-testid="stSidebar"] * {
        color: #cccccc !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #f0a500 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid #222 !important;
        padding-bottom: 8px !important;
    }

    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: #888 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        width: 100% !important;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #f0a500 !important;
        color: #f0a500 !important;
    }

    /* Hide default streamlit header/footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* Main title */
    .intellidoc-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
        margin-bottom: 0;
    }
    .intellidoc-title span {
        color: #f0a500;
    }
    .intellidoc-tagline {
        color: #555;
        font-size: 0.9rem;
        font-weight: 300;
        letter-spacing: 0.05em;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    /* Gold divider */
    .gold-divider {
        border: none;
        border-top: 1px solid #f0a500;
        opacity: 0.3;
        margin: 0 0 28px 0;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #f0a500;
        margin-bottom: 12px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #141414 !important;
        border: 1px solid #1f1f1f !important;
        border-radius: 10px !important;
        color: #e8e8e8 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #e8e8e8 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: #111 !important;
        border-top: 1px solid #1f1f1f !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #1a1a1a !important;
        color: #e8e8e8 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #f0a500 !important;
        box-shadow: 0 0 0 1px #f0a50033 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #111 !important;
        border: 1px solid #1f1f1f !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #888 !important;
        font-size: 0.82rem !important;
    }

    /* Divider */
    hr { border-color: #1f1f1f !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #141414 !important;
        border: 1px dashed #2a2a2a !important;
        border-radius: 8px !important;
    }

    /* Success / info / error */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="intellidoc-title">Intelli<span>Doc</span></p>'
        '<p class="intellidoc-tagline">Your documents, answered intelligently.</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

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

        if st.button("Clear Document"):
            rag.reset()
            st.session_state.pop("loaded_doc", None)
            st.success("Cleared.")
            st.rerun()

        st.markdown("---")
        st.markdown('<p style="font-size:0.72rem;color:#333;font-weight:300;">Powered by Groq · llama-3.1-8b</p>', unsafe_allow_html=True)

    # ── Chat area ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Ask IntelliDoc</p>', unsafe_allow_html=True)

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
