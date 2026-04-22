"""CLI entrypoint for the Sanskrit Retrieval-Augmented Generation system."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

if __package__ in {None, ""}:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from chunker import TextChunker
    from embeddings import EmbeddingGenerator
    from generator import LLMGenerator, format_context
    from ingest import DocumentLoader, DocumentPreprocessor
    from retriever import FAISSRetriever
    from utils import (
        FALLBACK_ANSWER,
        contains_devanagari,
        ensure_dir_exists,
        get_focus_tokens,
        get_query_hint_tokens,
        prepare_query_variants,
        romanized_anchor_score,
        romanized_overlap_score,
        token_anchor_score,
        text_overlap_score,
    )
else:
    from .chunker import TextChunker
    from .embeddings import EmbeddingGenerator
    from .generator import LLMGenerator, format_context
    from .ingest import DocumentLoader, DocumentPreprocessor
    from .retriever import FAISSRetriever
    from .utils import (
        FALLBACK_ANSWER,
        contains_devanagari,
        ensure_dir_exists,
        get_focus_tokens,
        get_query_hint_tokens,
        prepare_query_variants,
        romanized_anchor_score,
        romanized_overlap_score,
        token_anchor_score,
        text_overlap_score,
    )


class SanskritRAG:
    """End-to-end Sanskrit RAG pipeline with CPU-only retrieval and generation."""

    def __init__(
        self,
        data_dir: str | None = None,
        models_dir: str | None = None,
        embedding_model: str = "intfloat/multilingual-e5-small",
        generator_backend: str = "extractive",
        generator_model: str = "google/flan-t5-small",
        chunk_size: int = 140,
        overlap: int = 30,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir or os.path.join(base_dir, "..", "data")
        self.models_dir = models_dir or os.path.join(base_dir, "..", "models")
        self.embedding_model = embedding_model
        self.generator_backend = generator_backend
        self.generator_model = generator_model
        self.chunk_size = chunk_size
        self.overlap = overlap

        ensure_dir_exists(self.models_dir)

        self.loader = DocumentLoader(self.data_dir)
        self.preprocessor = DocumentPreprocessor()
        self.chunker = TextChunker(chunk_size=self.chunk_size, overlap=self.overlap)
        self.embedding_generator = EmbeddingGenerator(model_name=self.embedding_model, cache_dir=self.models_dir)
        self.retriever = FAISSRetriever(self.embedding_generator.embedding_dim, index_path=self.models_dir)
        self.generator: LLMGenerator | None = None

    def _get_generator(self) -> LLMGenerator:
        """Lazily initialize the generator so `--build` stays lightweight."""
        if self.generator is None:
            self.generator = LLMGenerator(
                backend=self.generator_backend,
                model_name=self.generator_model,
                cache_dir=self.models_dir,
            )
        return self.generator

    def build_index(self) -> Dict:
        """Load documents, preprocess, chunk, embed, and persist the FAISS index."""
        start_time = time.perf_counter()
        documents = self.loader.load_documents()
        if not documents:
            raise RuntimeError("No supported documents found in the data directory.")

        processed_docs = self.preprocessor.preprocess(documents)
        chunks = self.chunker.chunk_documents(processed_docs)
        if not chunks:
            raise RuntimeError("No chunks were created from the documents.")

        embeddings, chunks_with_embeddings = self.embedding_generator.encode_chunks(chunks, batch_size=16)
        self.retriever.add_embeddings(embeddings, chunks_with_embeddings)
        self.retriever.save_index()

        stats = {
            "documents": len(processed_docs),
            "sections": sum(len(doc.get("section_titles", [])) for doc in processed_docs),
            "chunks": len(chunks_with_embeddings),
            "embedding_dimension": self.embedding_generator.embedding_dim,
            "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        }
        self._save_build_metadata(stats)
        return stats

    def load_index(self) -> bool:
        """Load a previously built FAISS index."""
        return self.retriever.load_index()

    def ensure_index(self) -> None:
        """Load the index if it exists, otherwise build it."""
        if not self.load_index():
            self.build_index()

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[dict, float]]:
        """Retrieve chunks using vector search plus lightweight lexical reranking."""
        query_variants = prepare_query_variants(query)
        if not query_variants:
            query_variants = [query]

        merged_scores: Dict[Tuple[str, int, str], Dict] = {}
        search_width = max(top_k * 3, 6)
        latin_query = not contains_devanagari(query)
        focus_tokens = get_focus_tokens(query)
        hint_tokens = get_query_hint_tokens(query)
        focus_tokens = list(dict.fromkeys(focus_tokens + hint_tokens))
        primary_focus_tokens = focus_tokens[:1]

        for variant in query_variants:
            query_embedding = self.embedding_generator.encode_query(variant)
            results = self.retriever.search(query_embedding, k=search_width)
            for chunk, vector_score in results:
                key = (
                    chunk.get("source", "document"),
                    chunk.get("chunk_id", -1),
                    chunk.get("section_title", ""),
                )
                content_overlap = text_overlap_score(variant, chunk.get("content", ""))
                title_overlap = text_overlap_score(variant, chunk.get("section_title", ""))
                lexical_score = max(content_overlap, title_overlap)
                anchor_score = max(
                    token_anchor_score(focus_tokens, chunk.get("content", "")),
                    token_anchor_score(focus_tokens, chunk.get("section_title", "")),
                )
                primary_anchor_score = max(
                    token_anchor_score(primary_focus_tokens, chunk.get("content", "")),
                    token_anchor_score(primary_focus_tokens, chunk.get("section_title", "")),
                )
                if latin_query:
                    lexical_score = max(
                        lexical_score,
                        romanized_overlap_score(variant, chunk.get("romanized_content", "")),
                        romanized_overlap_score(variant, chunk.get("romanized_section_title", "")),
                    )
                    anchor_score = max(
                        anchor_score,
                        romanized_anchor_score(focus_tokens, chunk.get("romanized_content", "")),
                        romanized_anchor_score(focus_tokens, chunk.get("romanized_section_title", "")),
                    )
                    primary_anchor_score = max(
                        primary_anchor_score,
                        romanized_anchor_score(primary_focus_tokens, chunk.get("romanized_content", "")),
                        romanized_anchor_score(primary_focus_tokens, chunk.get("romanized_section_title", "")),
                    )

                combined_score = (
                    (0.48 * vector_score)
                    + (0.16 * lexical_score)
                    + (0.18 * anchor_score)
                    + (0.18 * primary_anchor_score)
                )
                if title_overlap >= 0.20:
                    combined_score += 0.10
                if anchor_score >= 0.50:
                    combined_score += 0.12
                if primary_anchor_score >= 0.99:
                    combined_score += 0.18

                existing = merged_scores.get(key)
                if existing is None or combined_score > existing["score"]:
                    merged_scores[key] = {"chunk": chunk, "score": combined_score}

        if latin_query:
            for chunk in self.retriever.chunks:
                key = (
                    chunk.get("source", "document"),
                    chunk.get("chunk_id", -1),
                    chunk.get("section_title", ""),
                )
                romanized_score = max(
                    romanized_overlap_score(query, chunk.get("romanized_content", "")),
                    romanized_overlap_score(query, chunk.get("romanized_section_title", "")),
                )
                anchor_score = max(
                    romanized_anchor_score(focus_tokens, chunk.get("romanized_content", "")),
                    romanized_anchor_score(focus_tokens, chunk.get("romanized_section_title", "")),
                )
                primary_anchor_score = max(
                    romanized_anchor_score(primary_focus_tokens, chunk.get("romanized_content", "")),
                    romanized_anchor_score(primary_focus_tokens, chunk.get("romanized_section_title", "")),
                )
                if romanized_score <= 0:
                    continue

                combined_score = (0.12 + (0.33 * romanized_score) + (0.25 * anchor_score) + (0.30 * primary_anchor_score))
                if anchor_score >= 0.50:
                    combined_score += 0.15
                if primary_anchor_score >= 0.99:
                    combined_score += 0.20
                existing = merged_scores.get(key)
                if existing is None or combined_score > existing["score"]:
                    merged_scores[key] = {"chunk": chunk, "score": combined_score}

        ranked = sorted(
            ((payload["chunk"], payload["score"]) for payload in merged_scores.values()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def answer_query(self, query: str, top_k: int = 4) -> Dict:
        """Retrieve grounded context and generate an answer."""
        self.ensure_index()

        query_variants = prepare_query_variants(query)
        start_time = time.perf_counter()
        results = self.retrieve(query, top_k=top_k)
        context = format_context(results, max_context_length=1800, max_chunks=top_k)
        answer = self._get_generator().generate(context=context, query=query, query_variants=query_variants)
        latency = round(time.perf_counter() - start_time, 2)

        sources = []
        for chunk, score in results:
            label = chunk.get("section_title", chunk.get("source", "document"))
            sources.append(
                {
                    "source": chunk.get("source", "document"),
                    "section": label,
                    "score": float(round(float(score), 4)),
                }
            )

        return {
            "query": query,
            "query_variants": query_variants,
            "answer": answer or FALLBACK_ANSWER,
            "context": context,
            "sources": sources,
            "latency_seconds": latency,
            "generator_backend": self._get_generator().runtime_backend,
        }

    def benchmark(self) -> Dict:
        """Run a small evaluation set for the provided corpus."""
        benchmark_queries = [
            {"query": "शंखनादः किम् अकरोत्?", "keywords": ["शर्कराम्", "सन्चिकायाम्", "दुग्धम्"]},
            {"query": "मूर्खभृत्यस्य कथायाः सन्देशः कः?", "keywords": ["मूर्खभृत्यस्य", "विनश्यति", "वरम्"]},
            {"query": "घण्टाकर्णः कथं नष्टः?", "keywords": ["वृद्धा", "घण्टा", "वानर"]},
            {"query": "देवः भक्तं किम् उक्तवान्?", "keywords": ["त्रिवारम्", "प्रयत्न", "साहाय्यम्"]},
            {"query": "कालिदासः कविं कथं सहाय्यं कृतवान्?", "keywords": ["नवकविं", "सुभाषितं", "दत्तवान्"]},
        ]

        runs = []
        for item in benchmark_queries:
            result = self.answer_query(item["query"], top_k=4)
            answer_text = result["answer"]
            matched = any(keyword in answer_text for keyword in item["keywords"])
            runs.append(
                {
                    "query": item["query"],
                    "answer": answer_text,
                    "latency_seconds": result["latency_seconds"],
                    "matched_keyword": matched,
                }
            )

        average_latency = round(sum(run["latency_seconds"] for run in runs) / max(1, len(runs)), 2)
        keyword_accuracy = round(sum(1 for run in runs if run["matched_keyword"]) / max(1, len(runs)), 2)
        summary = {
            "runs": runs,
            "average_latency_seconds": average_latency,
            "keyword_accuracy": keyword_accuracy,
        }

        benchmark_path = os.path.join(self.models_dir, "benchmark_results.json")
        with open(benchmark_path, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, ensure_ascii=False, indent=2)
        return summary

    def interactive_loop(self, top_k: int = 4) -> None:
        """Start an interactive terminal session."""
        self.ensure_index()
        print("\nInteractive Sanskrit RAG")
        print("Type a Sanskrit or transliterated question. Use 'exit' to quit.\n")

        while True:
            query = input("Query> ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break

            result = self.answer_query(query, top_k=top_k)
            self._print_query_result(result)

    def _save_build_metadata(self, stats: Dict) -> None:
        metadata_path = os.path.join(self.models_dir, "build_info.json")
        with open(metadata_path, "w", encoding="utf-8") as file_handle:
            json.dump(stats, file_handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _print_query_result(result: Dict) -> None:
        print(f"Answer: {result['answer']}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CPU-only Sanskrit RAG system")
    parser.add_argument("--build", action="store_true", help="Build the FAISS index from documents")
    parser.add_argument("--query", type=str, help="Run a single question against the corpus")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive QA session")
    parser.add_argument("--benchmark", action="store_true", help="Run the bundled benchmark queries")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the document directory")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to store embeddings and indexes")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks to use")
    parser.add_argument(
        "--generator-backend",
        choices=["auto", "transformers", "extractive"],
        default="extractive",
        help="Answer generation backend",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="google/flan-t5-small",
        help="Transformers generator model name",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    rag = SanskritRAG(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        generator_backend=args.generator_backend,
        generator_model=args.generator_model,
    )

    if args.build:
        stats = rag.build_index()
        print("\nIndex build completed.")
        for key, value in stats.items():
            print(f"- {key}: {value}")

    if args.benchmark:
        benchmark = rag.benchmark()
        print("\nBenchmark completed.")
        print(f"Keyword accuracy: {benchmark['keyword_accuracy']}")
        for run in benchmark["runs"]:
            print(f"- {run['query']} -> {run['answer']}")

    if args.query:
        result = rag.answer_query(args.query, top_k=args.top_k)
        SanskritRAG._print_query_result(result)

    if args.interactive:
        rag.interactive_loop(top_k=args.top_k)

    if not any([args.build, args.query, args.interactive, args.benchmark]):
        print("No action specified. Use --build, --query, --interactive, or --benchmark.")


if __name__ == "__main__":
    main()
