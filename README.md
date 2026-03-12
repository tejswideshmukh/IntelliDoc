# Simple RAG Document Q&A System

A simple semantic search system for asking questions about uploaded documents. Uses vector embeddings and semantic search to find relevant document chunks. Built with open-source models and designed to run on CPU.

## Demo 
![IntelliDoc Demo](Intellidoc_demo.mp4)
## Features

- 📄 Upload PDF and TXT documents
- 🔍 Semantic search using sentence transformers
- 💬 Ask questions and get relevant document excerpts
- 📚 View top matching chunks from your documents
- 💻 Runs on CPU (no GPU required)
- ⚡ Fast and lightweight (no LLM needed)

## Tech Stack

- **UI**: Streamlit
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Upload a document (PDF or TXT) using the sidebar
2. Ask questions about the document in the chat interface
3. View the most relevant document chunks matching your query

## How It Works

1. Documents are split into chunks
2. Each chunk is converted to a vector embedding
3. When you ask a question, it's also converted to an embedding
4. The system finds the most similar chunks using cosine similarity
5. Top matching chunks are displayed as results

## Project Structure

```
rag-qa-system/
├── app.py              # Streamlit UI
├── rag.py              # Core RAG logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Notes

- First run will download the embedding model (~80MB)
- Documents are stored in memory (cleared on restart)
- Uses semantic search only (no LLM generation)


