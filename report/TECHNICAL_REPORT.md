# Technical Report

## Sanskrit Document Retrieval-Augmented Generation (RAG)

### 1. Objective

The objective of this assignment is to build an end-to-end CPU-only RAG system for Sanskrit documents. The final system must ingest Sanskrit documents, preprocess and index them, retrieve relevant context for a query, and generate grounded answers without using a GPU.

### 2. Corpus Used

- Primary corpus file: `data/Rag-docs.docx`
- Corpus content: short Sanskrit narratives and moral stories in Devanagari
- Document characteristics:
  - mixed prose and verse
  - multiple independent stories in a single document
  - some English glosses embedded in the source file

### 3. System Architecture

Pipeline:

`Ingest -> Preprocess -> Section Detection -> Chunk -> Embed -> FAISS Index -> Retrieve -> Generate`

Main modules:

- `code/ingest.py`: loads `.txt`, `.docx`, and `.pdf` files
- `code/chunker.py`: creates sentence-aware chunks with overlap
- `code/embeddings.py`: generates multilingual embeddings on CPU
- `code/retriever.py`: stores and queries embeddings with FAISS
- `code/generator.py`: produces answers using a grounded extractive strategy and an optional CPU LLM backend
- `code/main.py`: exposes `--build`, `--query`, `--interactive`, and `--benchmark`

### 4. Preprocessing Pipeline

The preprocessing stage performs the following:

- Unicode normalization using NFC
- URL and metadata cleanup
- paragraph extraction
- heading detection for story boundaries
- removal of English-heavy glosses where possible
- Sanskrit text normalization while preserving Devanagari content

Reason for this design:

The source corpus places several stories in one document, and some paragraphs mix Sanskrit with English commentary. If the document is flattened into one long block, retrieval quality drops sharply. Preserving paragraph and section structure improved chunk quality and made retrieval more story-aware.

### 5. Chunking Strategy

Chunking is sentence-aware and section-aware:

- target chunk size: 140 words
- overlap: 30 words
- chunk boundaries prefer sentence endings
- each chunk stores:
  - source filename
  - section title
  - chunk text
  - romanized version of the chunk for transliteration fallback

Final indexed corpus after preprocessing:

- documents: 1
- detected sections: 4
- indexed chunks: 9

### 6. Retrieval Design

Retriever configuration:

- embedding model: `intfloat/multilingual-e5-small`
- device: CPU only
- vector index: `FAISS IndexFlatL2`

Retrieval flow:

1. Normalize the user query.
2. Build query variants.
3. For transliterated input, attempt Devanagari transliteration and rough Roman matching.
4. Encode query variants with the embedding model.
5. Retrieve nearest chunks from FAISS.
6. Re-rank results with lightweight lexical overlap and section-title overlap.

This hybrid retrieval design was necessary because the corpus uses Devanagari, while users may ask questions in transliterated Sanskrit.

### 7. Generation Design

The project uses a hybrid answer-generation layer:

- default backend: deterministic grounded extractive selector
- optional backend: `google/flan-t5-small` through `transformers` on CPU

Why hybrid generation was used:

- the extractive backend is fast and reliable on CPU
- the CPU LLM backend is integrated and functional, but slower and less stable on this corpus
- when the LLM produces low-quality output, the system falls back to the grounded extractive answer

Fallback answer used when evidence is insufficient:

`उत्तर उपलब्ध नहीं है।`

### 8. CPU Optimization Choices

- CPU-only embedding model cached locally in `models/`
- FAISS CPU index for similarity search
- small chunk count and short contexts to reduce answer latency
- extractive mode as default to keep inference practical on ordinary CPUs
- optional seq2seq backend enabled only when explicitly requested

### 9. Performance Observations

Measured on the local machine in default extractive mode:

- build time: about `1.12s`
- benchmark average query latency: `0.09s`
- internal keyword-match accuracy on 5 test queries: `1.00`

Observed strengths:

- direct factual and moral questions work well
- retrieval is fast and CPU-friendly
- story-title-aware preprocessing improves chunk relevance
- transliterated Sanskrit queries now retrieve the intended story more reliably

Observed weaknesses:

- performance is measured on a very small internal benchmark, not a large held-out test set
- transliterated input works best for common Sanskrit transliteration schemes, not arbitrary English spellings
- the optional CPU LLM backend is significantly slower than extractive mode

### 10. Example Outputs

Successful examples:

- `मूर्खभृत्यस्य कथायाः सन्देशः कः?`
  - output: `मूर्खभृत्यस्य संसर्गात् सर्वम् कार्यम् विनश्यति`
- `देवः भक्तं किम् उक्तवान्?`
  - output: `तदा देवः उक्तवान् "भो भक्त, अहम् त्रिवारम् आगतवान्`

### 11. Limitations

- only one source document is included in this submission
- retrieval quality depends heavily on the phrasing of the question
- deeper Sanskrit morphology handling is not implemented
- some story-specific queries still need stronger lexical or symbolic support

### 12. Future Improvements

- add more Sanskrit source documents
- add a Sanskrit morphological analyzer or stemmer
- improve transliteration support for loose Latin spellings
- add a better CPU-efficient multilingual generative model
- create a richer evaluation set with gold answers

### 13. Deliverables Present

- runnable codebase in `code/`
- sample Sanskrit data in `data/`
- technical report in `report/TECHNICAL_REPORT.md`
- benchmark artifact in `models/benchmark_results.json`
- setup and usage guide in `README.md`
