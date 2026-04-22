"""
Embeddings module for generating vector representations of text chunks.
Uses lightweight multilingual sentence-transformers models optimized for CPU.
"""

import os
import json
import re
from typing import List, Tuple
import numpy as np
try:
    from utils import print_progress, ensure_dir_exists, normalize_sanskrit_text
except ImportError:
    from .utils import print_progress, ensure_dir_exists, normalize_sanskrit_text


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using lightweight transformer models.
    Uses sentence-transformers for efficient multilingual embeddings.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", cache_dir: str = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model
                       'intfloat/multilingual-e5-small' - optimized for non-Latin scripts (376 dims)
                       'all-MiniLM-L6-v2' - lightweight fallback (22MB), 384 dimensions
            cache_dir: Directory to cache model
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', 'models'
        )
        ensure_dir_exists(self.cache_dir)
        
        try:
            from sentence_transformers import SentenceTransformer
            # Set cache directory for model downloads
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = self.cache_dir
            
            self.model = SentenceTransformer(
                model_name,
                cache_folder=self.cache_dir,
                device='cpu'  # Force CPU usage
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
        except ImportError:
            print("Error: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32, prefix: str = None) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for processing
            prefix: Optional prefix for intfloat/multilingual-e5-small model
                   Use "passage: " for documents, "query: " for queries
            
        Returns:
            Numpy array of embeddings (N x embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Strip noise before embedding.
        texts = [self._clean_for_embedding(text) for text in texts]

        # Add prefix if specified (required for multilingual-e5-small)
        if prefix:
            texts = [f"{prefix}{text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings

    def _clean_for_embedding(self, text: str) -> str:
        """Normalize and denoise text before embedding."""
        text = normalize_sanskrit_text(text or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def encode_chunks(self, chunks: List[dict], batch_size: int = 32) -> Tuple[np.ndarray, List[dict]]:
        """
        Encode text chunks with passage prefix for multilingual-e5-small.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings array, chunks with embedding info)
        """
        if not chunks:
            return np.array([]), []
        
        # Extract text content
        texts = [self._clean_for_embedding(chunk.get('content', '')) for chunk in chunks]
        
        # Encode with "passage: " prefix (required for multilingual-e5-small)
        embeddings = self.encode(texts, batch_size=batch_size, prefix="passage: ")
        
        # Add embedding info to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding_id'] = i
            chunk['embedding_dim'] = self.embedding_dim
        
        return embeddings, chunks
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string with query prefix for multilingual-e5-small.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector (1 x embedding_dim)
        """
        cleaned_query = self._clean_for_embedding(query)
        # Use "query: " prefix (required for multilingual-e5-small)
        embedding = self.model.encode([f"query: {cleaned_query}"], convert_to_numpy=True)
        return embedding[0]  # Return 1D array
    
    def similarity(self, embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query embedding and chunk embeddings.
        
        Args:
            embedding: Query embedding (1D array)
            embeddings: Array of chunk embeddings (N x D)
            
        Returns:
            Similarity scores (1D array of length N)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape query embedding if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        similarities = cosine_similarity(embedding, embeddings)[0]
        return similarities
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'model_type': 'sentence-transformers',
            'device': 'cpu'
        }


def save_embeddings(embeddings: np.ndarray, save_path: str) -> None:
    """
    Save embeddings to binary file for quick loading.
    
    Args:
        embeddings: Numpy array of embeddings
        save_path: Path to save file
    """
    np.save(save_path, embeddings)
    print(f"Embeddings saved to: {save_path}")


def load_embeddings(load_path: str) -> np.ndarray:
    """
    Load embeddings from binary file.
    
    Args:
        load_path: Path to embeddings file
        
    Returns:
        Numpy array of embeddings
    """
    embeddings = np.load(load_path)
    return embeddings


def main():
    """Test embedding generation."""
    from ingest import DocumentLoader, DocumentPreprocessor
    from chunker import TextChunker
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    print("="*60)
    print("Embeddings Generation Pipeline")
    print("="*60)
    
    # Load, preprocess, and chunk documents
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
    embeddings, chunks_with_emb = emb_gen.encode_chunks(chunks, batch_size=16)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Model info: {emb_gen.get_model_info()}")
    
    # Test query embedding
    test_query = "यह एक परीक्षण है"  # "This is a test" in Sanskrit/Hindi
    print(f"\nTest query: {test_query}")
    query_emb = emb_gen.encode_query(test_query)
    print(f"Query embedding shape: {query_emb.shape}")
    
    # Test similarity
    similarities = emb_gen.similarity(query_emb, embeddings)
    top_indices = np.argsort(-similarities)[:3]
    print("\nTop 3 similar chunks:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. Similarity: {similarities[idx]:.4f}")
        print(f"   Content: {chunks[idx]['content'][:100]}...")


if __name__ == '__main__':
    try:
        import numpy as np
        main()
    except ImportError:
        print("Error: numpy not installed. Install with: pip install numpy")
