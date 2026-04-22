"""
Retriever module for vector database management and similarity-based retrieval.
Uses FAISS for efficient similarity search.
"""

import os
import json
from typing import List, Tuple
import numpy as np
try:
    from utils import ensure_dir_exists, print_progress
except ImportError:
    from .utils import ensure_dir_exists, print_progress


class FAISSRetriever:
    """
    Vector database retriever using FAISS for efficient similarity search.
    Handles storage and retrieval of embeddings.
    """
    
    def __init__(self, embedding_dim: int, index_path: str = None):
        """
        Initialize FAISS retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to store/load FAISS index
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.chunks_metadata = []
        self.index_path = index_path or os.path.join(
            os.path.dirname(__file__), '..', 'models'
        )
        ensure_dir_exists(self.index_path)
        
        try:
            import faiss
            self.faiss = faiss
            self._create_index()
        except ImportError:
            print("Error: faiss-cpu not installed.")
            print("Install with: pip install faiss-cpu")
            raise
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        # Use IndexFlatL2 for L2 (cosine) distance
        # For cosine similarity, normalize embeddings
        self.index = self.faiss.IndexFlatL2(self.embedding_dim)
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of embeddings (N x D)
            chunks: List of chunk dictionaries with metadata
        """
        if embeddings.size == 0:
            print("Warning: No embeddings to add!")
            return
        
        # Normalize embeddings for cosine similarity
        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype('float32')
        
        # Normalize: divide by L2 norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)  # Epsilon to avoid division by zero
        
        # Add to index
        self.index.add(embeddings)
        self.chunks = chunks
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for top-k similar chunks.
        
        Args:
            query_embedding: Query embedding (1D array)
            k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            print("Warning: Index is empty!")
            return []
        
        if k > self.index.ntotal:
            k = self.index.ntotal
        
        # Prepare query embedding
        query_emb = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (query_norm + 1e-8)
        
        # Search
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (0-1)
                # For normalized vectors with IndexFlatL2:
                # faiss distance = ||q-d||^2 = 2 - 2*cosine_similarity
                similarity = 1 - (distance / 2.0)
                similarity = max(0, min(1, similarity))  # Clamp to [0, 1]
                
                results.append((chunk, similarity))

        # Ensure strict descending score order.
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save_index(self, name: str = "faiss_index") -> Tuple[str, str]:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            name: Base name for index files
            
        Returns:
            Tuple of (index_path, metadata_path)
        """
        index_file = os.path.join(self.index_path, f"{name}.index")
        metadata_file = os.path.join(self.index_path, f"{name}_metadata.json")
        
        # Save FAISS index
        self.faiss.write_index(self.index, index_file)
        
        # Save full chunks with content (CRITICAL for retrieval)
        metadata = {
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'chunks': self.chunks  # Save FULL chunk dictionaries with content
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return index_file, metadata_file
    
    def load_index(self, name: str = "faiss_index") -> bool:
        """
        Load FAISS index and chunks from disk.
        
        Args:
            name: Base name for index files
            
        Returns:
            True if loading successful, False otherwise
        """
        index_file = os.path.join(self.index_path, f"{name}.index")
        metadata_file = os.path.join(self.index_path, f"{name}_metadata.json")
        
        if not os.path.exists(index_file):
            print(f"Index file not found: {index_file}")
            return False
        
        try:
            # Load FAISS index
            self.index = self.faiss.read_index(index_file)
            
            # Load chunks from metadata (CRITICAL for retrieval)
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.chunks = metadata.get('chunks', [])
            else:
                print(f"Warning: Metadata file not found: {metadata_file}")
                self.chunks = []
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'total_chunks': len(self.chunks),
            'index_type': 'FAISS IndexFlatL2'
        }


def main():
    """Test retriever functionality."""
    from ingest import DocumentLoader, DocumentPreprocessor
    from chunker import TextChunker
    from embeddings import EmbeddingGenerator
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    print("="*60)
    print("Vector Database Retriever Pipeline")
    print("="*60)
    
    # Load, preprocess, and chunk
    loader = DocumentLoader(data_dir)
    documents = loader.load_documents()
    
    if not documents:
        print("No documents found!")
        return
    
    preprocessor = DocumentPreprocessor()
    processed_docs = preprocessor.preprocess(documents)
    
    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(processed_docs)
    
    if not chunks:
        print("No chunks created!")
        return
    
    # Generate embeddings
    emb_gen = EmbeddingGenerator()
    embeddings, chunks = emb_gen.encode_chunks(chunks, batch_size=16)
    
    # Create retriever and add embeddings
    retriever = FAISSRetriever(embedding_dim=emb_gen.embedding_dim)
    retriever.add_embeddings(embeddings, chunks)
    
    # Save index
    retriever.save_index()
    
    # Print stats
    stats = retriever.get_index_stats()
    print("\n" + "="*60)
    print("Index Statistics:")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test search
    test_query = "यह एक परीक्षण है"
    print(f"\n" + "="*60)
    print(f"Test Query: {test_query}")
    print("="*60)
    
    query_emb = emb_gen.encode_query(test_query)
    results = retriever.search(query_emb, k=3)
    
    print(f"\nTop {len(results)} results:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Relevance Score: {score:.4f}")
        print(f"   Source: {chunk['source']}")
        print(f"   Content: {chunk['content'][:150]}...")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
