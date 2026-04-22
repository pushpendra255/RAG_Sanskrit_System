"""Text chunking helpers for Sanskrit document retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

try:
    from utils import print_progress, split_into_sentences
except ImportError:
    from .utils import print_progress, split_into_sentences


class TextChunker:
    """Chunk Sanskrit text while preserving story and sentence boundaries."""

    def __init__(self, chunk_size: int = 140, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict] = []

    def chunk_text(self, text: str, source: str = "unknown", section_title: str | None = None) -> List[Dict]:
        """Chunk a single section of text into sentence-aware windows."""
        if not text or not text.strip():
            return []

        sentences = [sentence.strip() for sentence in split_into_sentences(text) if sentence.strip()]
        if not sentences:
            sentences = [text.strip()]

        chunks: List[Dict] = []
        buffer: List[str] = []
        buffer_words = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            if sentence_length == 0:
                continue

            if buffer and buffer_words + sentence_length > self.chunk_size:
                chunks.append(self._build_chunk(chunks, buffer, source, section_title))
                buffer, buffer_words = self._tail_overlap(buffer)

            buffer.append(sentence)
            buffer_words += sentence_length

        if buffer:
            chunks.append(self._build_chunk(chunks, buffer, source, section_title))

        return chunks

    def chunk_documents(self, processed_docs: List[Dict]) -> List[Dict]:
        """Chunk every processed document and propagate section metadata."""
        all_chunks: List[Dict] = []

        for index, document in enumerate(processed_docs, 1):
            section_map: Dict[str, List[str]] = defaultdict(list)
            for paragraph in document.get("paragraphs", []):
                section_map[paragraph["section_title"]].append(paragraph["text"])

            if not section_map and document.get("cleaned_content"):
                section_map[document["source"]].append(document["cleaned_content"])

            for section_title, paragraphs in section_map.items():
                section_text = "\n".join(paragraphs)
                section_chunks = self.chunk_text(section_text, source=document["source"], section_title=section_title)
                for chunk in section_chunks:
                    chunk["global_chunk_id"] = len(all_chunks)
                    all_chunks.append(chunk)

            print_progress(index, len(processed_docs), f"Chunking documents ({len(all_chunks)} chunks)")

        self.chunks = all_chunks
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks

    def _tail_overlap(self, sentences: List[str]) -> tuple[List[str], int]:
        """Keep a short tail of the previous chunk for overlap."""
        if self.overlap <= 0 or not sentences:
            return [], 0

        tail: List[str] = []
        tail_words = 0
        for sentence in reversed(sentences):
            sentence_length = len(sentence.split())
            if tail_words + sentence_length > self.overlap and tail:
                break
            tail.insert(0, sentence)
            tail_words += sentence_length

        return tail, tail_words

    def _build_chunk(
        self,
        existing_chunks: List[Dict],
        sentences: List[str],
        source: str,
        section_title: str | None,
    ) -> Dict:
        """Create a chunk dictionary with consistent metadata."""
        content = " । ".join(sentence.strip(" ।") for sentence in sentences if sentence.strip())
        if not content.endswith("।"):
            content += "।"

        romanized_content = self._romanize(content)
        romanized_section = self._romanize(section_title or source)

        return {
            "content": content,
            "source": source,
            "section_title": section_title or source,
            "romanized_content": romanized_content,
            "romanized_section_title": romanized_section,
            "chunk_id": len(existing_chunks),
            "char_count": len(content),
            "token_count": len(content.split()),
        }

    @staticmethod
    def _romanize(text: str) -> str:
        """Romanize Devanagari text for transliterated lexical fallback."""
        try:
            from indic_transliteration import sanscript
        except ImportError:
            return text

        try:
            return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.HK)
        except Exception:
            return text

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict[str, float]:
        """Return aggregate chunking statistics."""
        if not chunks:
            return {}

        char_counts = [chunk["char_count"] for chunk in chunks]
        token_counts = [chunk.get("token_count", 0) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_chars": sum(char_counts) / len(chunks),
            "avg_chunk_tokens": sum(token_counts) / len(chunks),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "total_chars": sum(char_counts),
        }


def main() -> None:
    """Run a standalone chunking smoke test."""
    import os

    try:
        from ingest import DocumentLoader, DocumentPreprocessor
    except ImportError:
        from .ingest import DocumentLoader, DocumentPreprocessor

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")

    print("=" * 60)
    print("Text Chunking Pipeline")
    print("=" * 60)

    loader = DocumentLoader(data_dir)
    documents = loader.load_documents()
    if not documents:
        raise SystemExit("No documents found.")

    preprocessor = DocumentPreprocessor()
    processed_docs = preprocessor.preprocess(documents)

    chunker = TextChunker()
    chunks = chunker.chunk_documents(processed_docs)
    stats = chunker.get_chunk_statistics(chunks)

    print("\n" + "=" * 60)
    print("Chunking Statistics")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("Sample Chunks")
    print("=" * 60)
    for chunk in chunks[:3]:
        print(f"{chunk['source']} | {chunk['section_title']} | {chunk['char_count']} chars")
        print(chunk["content"][:220] + "...")
        print()


if __name__ == "__main__":
    main()
