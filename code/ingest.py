"""Document ingestion and preprocessing for Sanskrit source files."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from utils import (
        clean_text,
        extract_paragraphs,
        get_file_extension,
        is_probably_heading,
        is_probably_noise,
        normalize_sanskrit_text,
        print_progress,
    )
except ImportError:
    from .utils import (
        clean_text,
        extract_paragraphs,
        get_file_extension,
        is_probably_heading,
        is_probably_noise,
        normalize_sanskrit_text,
        print_progress,
    )


class DocumentLoader:
    """Load Sanskrit documents from TXT, DOCX, and PDF files."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents: List[Tuple[str, str]] = []

    def load_documents(self) -> List[Tuple[str, str]]:
        """Load all supported documents from the configured data directory."""
        documents: List[Tuple[str, str]] = []

        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory '{self.data_dir}' not found.")
            return documents

        files = sorted(os.listdir(self.data_dir))
        total_files = len(files)

        for index, filename in enumerate(files, 1):
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.isfile(file_path):
                continue

            extension = get_file_extension(filename)
            content = None

            try:
                if extension == ".txt":
                    content = self._load_txt(file_path)
                elif extension == ".docx":
                    content = self._load_docx(file_path)
                elif extension == ".pdf":
                    content = self._load_pdf(file_path)
                else:
                    print(f"Skipping unsupported file format: {filename}")
                    continue

                if content and content.strip():
                    documents.append((filename, content))
                else:
                    print(f"Empty content: {filename}")
            except Exception as exc:
                print(f"Error loading {filename}: {exc}")

            print_progress(index, total_files, "Loading documents")

        self.documents = documents
        return documents

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read()

    def _load_docx(self, file_path: str) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError("python-docx is required for DOCX ingestion.") from exc

        document = Document(file_path)
        paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)

    def _load_pdf(self, file_path: str) -> str:
        try:
            import PyPDF2
        except ImportError as exc:
            raise ImportError("PyPDF2 is required for PDF ingestion.") from exc

        pages: List[str] = []
        with open(file_path, "rb") as file_handle:
            reader = PyPDF2.PdfReader(file_handle)
            for page in reader.pages:
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages.append(extracted.strip())

        return "\n\n".join(pages)


class DocumentPreprocessor:
    """Normalize Sanskrit documents and preserve story-level structure."""

    def __init__(self):
        self.processed_docs: List[Dict] = []

    def preprocess(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """Clean documents, detect headings, and keep paragraph metadata."""
        processed: List[Dict] = []

        for index, (filename, content) in enumerate(documents, 1):
            cleaned = clean_text(content)
            paragraphs = extract_paragraphs(cleaned)
            section_entries = self._build_section_entries(filename, paragraphs)
            cleaned_content = "\n\n".join(entry["text"] for entry in section_entries)

            char_count = len(cleaned_content)
            word_count = len(cleaned_content.split())

            processed.append(
                {
                    "source": filename,
                    "original_content": content,
                    "cleaned_content": cleaned_content,
                    "paragraphs": section_entries,
                    "section_titles": list(dict.fromkeys(entry["section_title"] for entry in section_entries)),
                    "char_count": char_count,
                    "word_count": word_count,
                    "language": "Sanskrit",
                }
            )
            print_progress(index, len(documents), "Preprocessing documents")

        self.processed_docs = processed
        print(f"\nPreprocessed {len(processed)} document(s)")
        return processed

    def _build_section_entries(self, filename: str, paragraphs: List[str]) -> List[Dict[str, str]]:
        """Attach each paragraph to a detected story heading."""
        section_entries: List[Dict[str, str]] = []
        current_section = Path(filename).stem

        for paragraph in paragraphs:
            normalized = normalize_sanskrit_text(paragraph)
            if not normalized or is_probably_noise(normalized):
                continue

            heading_candidate = self._extract_heading_candidate(normalized)
            if heading_candidate and is_probably_heading(heading_candidate):
                current_section = heading_candidate.strip('" ')
                continue

            body_text = self._clean_paragraph_body(normalized)
            if not body_text:
                continue

            section_entries.append(
                {
                    "section_title": current_section,
                    "text": body_text,
                }
            )

        if section_entries:
            return section_entries

        fallback_text = normalize_sanskrit_text(clean_text("\n\n".join(paragraphs)))
        if fallback_text:
            return [{"section_title": Path(filename).stem, "text": fallback_text}]
        return []

    @staticmethod
    def _extract_heading_candidate(text: str) -> str:
        """Extract a Devanagari title prefix from mixed heading lines."""
        if not text:
            return ""

        prefix = re.split(r"[A-Za-z]", text, maxsplit=1)[0]
        prefix = prefix.strip(" \"'|:-")
        return prefix or text

    @staticmethod
    def _clean_paragraph_body(text: str) -> str:
        """Remove English glosses and metadata from mixed-language paragraphs."""
        text = re.sub(r"\([^)]*[A-Za-z][^)]*\)", " ", text)
        text = re.sub(r"[A-Za-z][A-Za-z0-9 ,;:'\"-]{12,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        segments = re.split(r"[|]", text)
        kept_segments = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            devanagari_count = len(re.findall(r"[\u0900-\u097F]", segment))
            latin_count = len(re.findall(r"[A-Za-z]", segment))
            if devanagari_count == 0 and latin_count > 0:
                continue
            if devanagari_count and devanagari_count >= latin_count:
                kept_segments.append(segment)

        cleaned = " ".join(kept_segments) if kept_segments else text
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def get_statistics(self) -> Dict[str, float]:
        """Return aggregate preprocessing statistics."""
        if not self.processed_docs:
            return {}

        total_chars = sum(doc["char_count"] for doc in self.processed_docs)
        total_words = sum(doc["word_count"] for doc in self.processed_docs)
        total_sections = sum(len(doc.get("section_titles", [])) for doc in self.processed_docs)

        return {
            "total_documents": len(self.processed_docs),
            "total_sections": total_sections,
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_doc": total_chars / len(self.processed_docs),
            "avg_words_per_doc": total_words / len(self.processed_docs),
        }


def main() -> None:
    """Run a standalone ingestion smoke test."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")

    print("=" * 60)
    print("Document Ingestion Pipeline")
    print("=" * 60)

    loader = DocumentLoader(data_dir)
    documents = loader.load_documents()
    if not documents:
        raise SystemExit("No documents found to process.")

    preprocessor = DocumentPreprocessor()
    processed_docs = preprocessor.preprocess(documents)
    stats = preprocessor.get_statistics()

    print("\n" + "=" * 60)
    print("Preprocessing Statistics")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    if processed_docs:
        sample = processed_docs[0]
        print("\n" + "=" * 60)
        print("Sample Processed Content")
        print("=" * 60)
        print(f"Document: {sample['source']}")
        print(sample["cleaned_content"][:400] + "...")


if __name__ == "__main__":
    main()
