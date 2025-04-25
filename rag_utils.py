"""
RAG (Retrieval Augmented Generation) utilities for financial document processing.
"""
import os
import io
import PyPDF2
import docx
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
import re

# Import Chroma
import chromadb
from chromadb.utils import embedding_functions

# Create directories for document cache
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

# Document processing functions
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + f"\n\n[Page {page_num + 1}]\n\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_word(docx_file) -> str:
    """Extract text from a Word document."""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from Word document: {e}")
        return ""

def process_document(uploaded_file) -> Tuple[str, Optional[str]]:
    """Process an uploaded document (PDF or Word)."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Make a copy of the file in memory
        file_bytes = uploaded_file.getvalue()
        
        if file_extension == 'pdf':
            # Process PDF file
            pdf_file = io.BytesIO(file_bytes)
            text = extract_text_from_pdf(pdf_file)
            return text, None
        elif file_extension in ['doc', 'docx']:
            # Process Word file
            docx_file = io.BytesIO(file_bytes)
            text = extract_text_from_word(docx_file)
            return text, None
        else:
            return "", f"Unsupported file format: {file_extension}. Please upload PDF or Word documents."
    except Exception as e:
        return "", f"Error processing document: {e}"

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    # First clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only keep chunks with substantial content
            chunks.append(chunk)
    
    # If we have no chunks (e.g., document was too small), return the whole text
    if not chunks and text:
        chunks = [text]
        
    return chunks

class ChromaDocumentDatabase:
    """Chroma vector database for document chunks."""
    
    def __init__(self, collection_name="financial_documents"):
        # Setup embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Setup Chroma client - persistent mode
        persistent_dir = "data/chroma_db"
        os.makedirs(persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persistent_dir)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
    
    def add_document(self, text: str, source: str, chunk_size: int = 1000, overlap: int = 200) -> int:
        """Add a document to the database."""
        chunks = split_text_into_chunks(text, chunk_size, overlap)
        
        if not chunks:
            return 0
        
        # Create IDs for chunks
        ids = [f"{source.replace(' ', '_').replace('.', '_')}_{i}" for i in range(len(chunks))]
        
        # Add chunks to collection
        metadatas = [{"source": source} for _ in chunks]
        
        try:
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            return len(chunks)
        except Exception as e:
            print(f"Error adding document to Chroma DB: {e}")
            return 0
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        if self.collection.count() == 0:
            return []
        
        # Search the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format the results
        formatted_results = []
        
        if not results["documents"]:
            return []
            
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                'chunk': doc,
                'source': results["metadatas"][0][i]["source"],
                'score': float(results["distances"][0][i]) if "distances" in results else 0.0
            })
        
        return formatted_results

# Create or get document database
@st.cache_resource
def get_document_database():
    return ChromaDocumentDatabase()

# Process a document and add it to the vector database
def process_and_add_document(uploaded_file):
    """Process a document and add it to the vector database."""
    try:
        # Extract text from document
        doc_text, error = process_document(uploaded_file)
        
        if error:
            return False, error
        
        if not doc_text:
            return False, "No text could be extracted from the document"
        
        # Add to vector database
        doc_db = get_document_database()
        chunks_added = doc_db.add_document(doc_text, uploaded_file.name)
        
        return True, f"Successfully processed {uploaded_file.name} and added {chunks_added} chunks to the database"
    except Exception as e:
        return False, f"Error processing document: {e}"