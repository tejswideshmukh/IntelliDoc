"""
Simple RAG (Retrieval-Augmented Generation) System
Handles document processing, embeddings, and question answering
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os


class SimpleRAG:
    def __init__(self):
        # Load embedding model (CPU-friendly)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB (in-memory for simplicity)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})
        
    def chunk_document(self, text, chunk_size=1000, overlap=200):
        """
        Split document into chunks respecting paragraph boundaries.
        Falls back to character-based splitting for oversized paragraphs.
        chunk_size: max characters per chunk
        overlap: characters to overlap when splitting large paragraphs
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > chunk_size:
                    # Paragraph too large — split by characters with overlap
                    start = 0
                    while start < len(para):
                        chunk = para[start:start + chunk_size].strip()
                        if chunk:
                            chunks.append(chunk)
                        start += chunk_size - overlap
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    def add_document(self, text, doc_id):
        """Process and store a document"""
        # Split into chunks
        chunks = self.chunk_document(text)
        
        if not chunks:
            return 0
        
        # Generate embeddings for chunks
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Create IDs for chunks
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=chunk_ids
        )
        
        return len(chunks)
    

    def search(self, query, top_k=3, distance_threshold=0.9):
        """
        Search for relevant document chunks.
        distance_threshold: cosine similarity ,
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        # Search in ChromaDB, also fetch distances
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )

        if not results['documents']:
            return []

        docs = results['documents'][0]
        dists = results['distances'][0]

        # Always include the best match; apply threshold to the rest
        relevant_chunks = [docs[0]]
        for doc, dist in zip(docs[1:], dists[1:]):
            if dist <= distance_threshold:
                relevant_chunks.append(doc)

        return relevant_chunks
    
    def reset(self):
        """Clear all stored documents"""
        try:
            self.client.delete_collection(name="documents")
            self.collection = self.client.create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            )
