"""
Simple utilities for financial document processing.
"""
import os
import io
import re
import json
import pandas as pd
import streamlit as st
from typing import Tuple, Dict, List, Any, Optional
from datetime import datetime

# Create directories for document storage
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/document_index", exist_ok=True)

def process_and_add_document(uploaded_file) -> Tuple[bool, str]:
    """
    Process an uploaded document and extract its text content.
    
    Args:
        uploaded_file: The uploaded file from st.file_uploader
        
    Returns:
        Tuple of (success_boolean, message_string)
    """
    try:
        # Get file extension
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # Read file content
        file_content = uploaded_file.getvalue()
        
        # Process based on file type
        if file_extension == '.pdf':
            # Process PDF file
            try:
                import PyPDF2
                pdf_file = io.BytesIO(file_content)
                
                # Extract text from PDF
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + f"\n\n[Page {page_num + 1}]\n\n"
                
                if not text.strip():
                    return False, "Could not extract text from PDF (possibly scanned document)"
                
            except ImportError:
                return False, "PyPDF2 not installed. Run: pip install PyPDF2"
            except Exception as e:
                return False, f"Error processing PDF: {str(e)}"
            
        elif file_extension == '.docx':
            # Process DOCX file
            try:
                from docx import Document
                docx_file = io.BytesIO(file_content)
                
                # Extract text from DOCX
                doc = Document(docx_file)
                text = "\n".join([para.text for para in doc.paragraphs if para.text])
                
                if not text.strip():
                    return False, "Could not extract text from DOCX file"
                
            except ImportError:
                return False, "python-docx not installed. Run: pip install python-docx"
            except Exception as e:
                return False, f"Error processing DOCX: {str(e)}"
                
        elif file_extension == '.doc':
            # Sorry, .doc files need additional libraries
            return False, "DOC format not supported. Please convert to DOCX or PDF."
            
        else:
            return False, f"Unsupported file format: {file_extension}. Please upload PDF or DOCX files."
        
        # Save extracted text to file
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file_name)
        text_file_path = os.path.join("data/documents", f"text_{safe_filename}.txt")
        
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save metadata
        metadata = {
            "filename": file_name,
            "processed_time": datetime.now().isoformat(),
            "file_size": len(file_content),
            "text_file_path": text_file_path
        }
        
        metadata_path = os.path.join("data/document_index", f"meta_{safe_filename}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in session state for easy access
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = {}
            
        st.session_state.processed_documents[file_name] = {
            'text': text,
            'path': text_file_path,
            'metadata': metadata
        }
        
        return True, f"Successfully processed {file_name}"
        
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

def get_document_content(filename: str) -> Optional[str]:
    """
    Get the content of a processed document by filename.
    
    Args:
        filename: The original filename
        
    Returns:
        The document text or None if not found
    """
    # Check in session state first
    if 'processed_documents' in st.session_state and filename in st.session_state.processed_documents:
        return st.session_state.processed_documents[filename]['text']
    
    # Not in session state, try to find in files
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    metadata_path = os.path.join("data/document_index", f"meta_{safe_filename}.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            text_path = metadata.get('text_file_path')
            
            if text_path and os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Store in session state for future access
                if 'processed_documents' not in st.session_state:
                    st.session_state.processed_documents = {}
                
                st.session_state.processed_documents[filename] = {
                    'text': text,
                    'path': text_path,
                    'metadata': metadata
                }
                
                return text
        except Exception as e:
            st.error(f"Error retrieving document: {e}")
    
    return None

def get_all_documents() -> List[Dict[str, Any]]:
    """
    Get all processed documents.
    
    Returns:
        List of document metadata
    """
    documents = []
    
    # Check document index directory
    index_dir = "data/document_index"
    if os.path.exists(index_dir):
        for filename in os.listdir(index_dir):
            if filename.startswith("meta_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(index_dir, filename), 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        documents.append(metadata)
                except Exception:
                    pass
    
    return documents

def search_document_content(query: str) -> List[Dict[str, Any]]:
    """
    Simple text search across all documents.
    
    Args:
        query: Search query
        
    Returns:
        List of search results with document info and snippets
    """
    results = []
    query_lower = query.lower()
    
    # Get all documents
    documents = get_all_documents()
    
    for doc_meta in documents:
        filename = doc_meta.get("filename", "")
        text_path = doc_meta.get("text_file_path", "")
        
        try:
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple search
                if query_lower in content.lower():
                    # Find context around the match
                    content_lower = content.lower()
                    pos = content_lower.find(query_lower)
                    
                    # Extract a snippet showing the match in context
                    start = max(0, pos - 100)
                    end = min(len(content), pos + len(query) + 100)
                    
                    # Find the start of a complete word
                    while start > 0 and content[start] != ' ' and content[start] != '\n':
                        start -= 1
                    
                    # Find the end of a complete word
                    while end < len(content) - 1 and content[end] != ' ' and content[end] != '\n':
                        end += 1
                    
                    snippet = content[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    
                    results.append({
                        "filename": filename,
                        "snippet": snippet,
                        "text_path": text_path
                    })
        except Exception:
            pass
    
    return results