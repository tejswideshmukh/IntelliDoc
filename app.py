"""
Simple RAG Document Q&A System
Streamlit web interface for asking questions about uploaded documents
"""

import streamlit as st
from rag import SimpleRAG
import PyPDF2
import io


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
    st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")
    
    st.title("📚 Simple RAG Document Q&A System")
    st.markdown("Upload documents and ask questions about them!")
    
    # Initialize RAG system
    rag = load_rag_system()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files"
        )
        
        if uploaded_file is not None:
            doc_id = uploaded_file.name

            # Only process if this file hasn't been loaded yet this session
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
                    st.success(f"✅ Document loaded! ({num_chunks} chunks created)")
                else:
                    st.error("Could not extract text from file")

            st.info(f"📄 {uploaded_file.name}")
        
        st.divider()
        
        if st.button("🗑️ Clear All Documents"):
            rag.reset()
            st.session_state.pop("loaded_doc", None)
            st.success("Documents cleared!")
            st.rerun()
    
    # Main area for Q&A
    st.header("💬 Ask a Question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Add user question to chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.spinner("Searching and generating answer..."):
            relevant_chunks = rag.search(question, top_k=3)

        if not relevant_chunks:
            answer = "I couldn't find any relevant information. Please upload a document first."
        else:
            answer = rag.generate_answer(question, relevant_chunks)
            with st.expander("View source chunks"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**{i}.** {chunk}")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
