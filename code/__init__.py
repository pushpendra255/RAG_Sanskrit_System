# Sanskrit RAG Package
"""
Retrieval-Augmented Generation System for Sanskrit Documents
"""

__version__ = "1.0"
__author__ = "Pushpendra Singh"
__description__ = "CPU-based RAG system for Sanskrit document understanding"

from .main import SanskritRAG
from .ingest import DocumentLoader, DocumentPreprocessor
from .chunker import TextChunker
from .embeddings import EmbeddingGenerator
from .retriever import FAISSRetriever
from .generator import LLMGenerator

__all__ = [
    'SanskritRAG',
    'DocumentLoader',
    'DocumentPreprocessor',
    'TextChunker',
    'EmbeddingGenerator',
    'FAISSRetriever',
    'LLMGenerator'
]
