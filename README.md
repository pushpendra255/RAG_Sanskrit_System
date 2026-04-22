# Sanskrit Document Retrieval-Augmented Generation (RAG)

This project implements a CPU-only Retrieval-Augmented Generation pipeline for Sanskrit documents. It ingests Sanskrit source files, preprocesses and chunks them, builds a FAISS vector index, retrieves the most relevant context, and returns grounded answers. The default mode is a fast extractive answerer, and an optional CPU LLM backend is also available through `transformers`.

## What Is Included

- `code/ingest.py`: document loading and preprocessing for `.txt`, `.docx`, and `.pdf`
- `code/chunker.py`: section-aware sentence chunking with overlap
- `code/embeddings.py`: CPU embeddings with `intfloat/multilingual-e5-small`
- `code/retriever.py`: FAISS-based retrieval
- `code/generator.py`: grounded answer generation with extractive fallback and optional CPU LLM
- `code/main.py`: end-to-end CLI entrypoint
- `data/Rag-docs.docx`: Sanskrit source corpus used in this submission
- `report/TECHNICAL_REPORT.md`: technical report
- `models/`: cached embedding model, FAISS index, and benchmark outputs

## CPU-Only Design

- Embeddings run on CPU through `sentence-transformers`
- Retrieval uses `faiss-cpu`
- Default answering uses a deterministic grounded selector for low latency
- Optional CPU LLM answering uses `google/flan-t5-small` via `transformers`

## Query Support

- Sanskrit in Devanagari is supported directly
- Transliteration support is included for common Sanskrit schemes such as `HK`, `ITRANS`, and `IAST`
- Rough Latin-script matching is also used as a fallback during retrieval

## Setup

Create and activate the virtual environment:

```powershell
python -m venv rag_venv
rag_venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Build The Index

```powershell
python code/main.py --build
```

This reads the document corpus from `data/`, preprocesses it, creates chunks, encodes them, and stores the FAISS index in `models/`.

## Run Queries

Single query:

```powershell
python code/main.py --query "मूर्खभृत्यस्य कथायाः सन्देशः कः?"
```

Interactive mode:

```powershell
python code/main.py --interactive
```

Optional CPU LLM mode:

```powershell
python code/main.py --query "शंखनादः किम् अकरोत्?" --generator-backend auto
```

The optional LLM path is slower on CPU and may download `google/flan-t5-small` on first use.

## Example Queries

- `मूर्खभृत्यस्य कथायाः सन्देशः कः?`
- `देवः भक्तं किम् उक्तवान्?`
- `ghaNTAkarNaH kathaM naSTaH`
- `शीतं बहु बाधति इति कः अवदत्?`

## Benchmark

Run the bundled benchmark:

```powershell
python code/main.py --benchmark
```

Current measured benchmark on this corpus in default extractive mode:

- Build time: `1.12s`
- Average latency: `0.09s`
- Keyword-match accuracy on 5 internal test queries: `1.00`

Benchmark results are saved to `models/benchmark_results.json`.

## Notes

- The corpus contains some mixed Sanskrit-English material, so preprocessing removes English glosses where possible.
- The project is optimized for reproducibility and grounded answers, not free-form creative generation.
- The extractive mode is the recommended default for evaluation because it is fast, CPU-friendly, and stable.
- The optional CPU LLM mode remains available, but the extractive default currently gives better consistency on this corpus.

## Submission Structure

- `code/`
- `data/`
- `models/`
- `report/`
- `README.md`
