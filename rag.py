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
        self.collection = self.client.get_or_create_collection(name="documents")
        
    def chunk_document(self, text, chunk_size=1000, overlap=200):
        """
        Split document into overlapping chunks
        chunk_size: number of characters per chunk
        overlap: number of characters to overlap between chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
            
            start = end - overlap  # Overlap for context
            
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
    
    def search(self, query, top_k=3):
        """Search for relevant document chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Extract relevant chunks
        relevant_chunks = results['documents'][0] if results['documents'] else []
        
        return relevant_chunks
    
    def reset(self):
        """Clear all stored documents"""
        try:
            self.client.delete_collection(name="documents")
            self.collection = self.client.create_collection(name="documents")
        except:
            self.collection = self.client.create_collection(name="documents")
