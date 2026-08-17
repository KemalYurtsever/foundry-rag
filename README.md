# Foundry Local RAG 

This is a local, keyless RAG project with structure-aware ingestion, recursive chunking, Unicode-aware BM25 retrieval, optional Foundry Local embeddings, reciprocal-rank fusion, semantic and intent reranking, confidence-based abstention, grounded citations, and regression evaluation. Extracted sections, file hashes, model metadata, and embeddings are persisted transactionally in SQLite.

Supported document types are `.txt`, `.md`, `.pdf`, `.docx`, and `.pptx`. PowerPoint presentations are extracted and cited slide by slide. Documents are split recursively at paragraph, line, sentence, word, and character boundaries. Chunk size and overlap adapt to each document, retrieval count adapts to the model context window, and response length uses the selected model's advertised output limit. `--chunk-size`, `--overlap`, and `--top-k` are available only when you want explicit overrides. Uploaded files are limited to 50 MB, validated by content, and stored with restrictive permissions in the private `.rag_cache/uploads` directory. Office packages are also checked for decompression abuse; packages containing macros, ActiveX, or embedded OLE content are rejected. The app never launches uploaded documents in Office, a browser, or another associated application. Legacy `.doc` and `.ppt` are intentionally unsupported because safely extracting them would require launching Microsoft Office on untrusted files.

## Project presentation

[Watch the Foundry Local RAG presentation on YouTube](https://www.youtube.com/watch?v=NHJ2ffBocgg).

## Install and test

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe rag.py --self-test
.\.venv\Scripts\python.exe rag.py "What does RAG do?"
```

The self-test verifies strict document grounding with tiny in-memory examples. The last command tests retrieval with extractive output. Add supported files under `data/documents`, including nested folders. Use `--verbose` for retrieval diagnostics. A JSON object passed through `--config settings.json` can supply CLI defaults; explicit command-line values take precedence.

## Run with Foundry Local

Pass a model alias from your Foundry Local catalog. `--download` downloads it only when it is not already cached:

```powershell
.\.venv\Scripts\python.exe rag.py --list-models
.\.venv\Scripts\python.exe rag.py --model MODEL_ALIAS --download "What does RAG do?"
```

After the first download, omit `--download`. The SDK loads the model in-process and creates its chat client directly, there are no API keys or environment variables.

## Hybrid retrieval

The app automatically uses `qwen3-embedding-0.6b` when it is cached, and otherwise falls back to BM25. Enable semantic retrieval by downloading it once:

```powershell
.\.venv\Scripts\python.exe rag.py --download-embedding "test question"
```

Use `--no-embeddings` to force BM25-only retrieval. The local database is `.rag_cache/rag.sqlite3` by default; change its parent directory with `--cache-dir`. SQLite WAL mode supports safe concurrent readers and writers. The same model is always used for document and query vectors, and stored dimensions are validated before reuse.

## Retrieval, languages, and OCR

Filenames and inferred headings contribute to ranking. `--neighbor-window 1` includes immediately adjacent chunks when an answer crosses a chunk boundary.

Tokenization is Unicode-aware and contains no domain-specific synonym table. Use multilingual embedding and generation models for semantic retrieval and answers across languages.

Scanned PDF OCR is opt-in:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ocr]"
.\.venv\Scripts\python.exe rag.py --ocr --ocr-language eng "question"
```

Install Tesseract separately and select an installed language such as `eng`, `ron`, or `tur`. Only pages with almost no extractable text are sent through OCR.

An optional real-model integration check verifies that document and query vectors have matching dimensions:

```powershell
$env:RAG_INTEGRATION_MODEL="EMBEDDING_MODEL_ALIAS"
.\.venv\Scripts\python.exe -m unittest tests.test_rag.RAGTests.test_real_embedding_model_uses_matching_dimensions -v
```

## Structure

```text
.rag_cache/uploads/   private managed knowledge files
src/foundry_rag/      ingestion, retrieval, Foundry generation, pipeline, CLI
tests/                unit and end-to-end smoke tests
```

## Desktop application

Launch the graphical application:

```powershell
.\.venv\Scripts\python.exe app.py
```

Use **Import files** to add TXT, Markdown, PDF, DOCX, or PPTX files. The application validates and copies them into the private `.rag_cache/uploads` store, rebuilds the index in a background thread, and displays verified answers and source filenames/pages in the chat.

The desktop app also provides Precise, Balanced, Thorough, and Custom retrieval profiles; a retrieval inspector; direct evidence views; incremental cached indexing with progress and cancellation; searchable/sortable multi-select file management; persistent settings; and dark/light themes. Uploaded documents are never handed to an external application.

Keyboard shortcuts:

- `Ctrl+O`: import files
- `Ctrl+L`: focus the question box
- `Ctrl+K`: show or hide knowledge files
- `Ctrl+Shift+I`: inspect the last retrieval
- `F5`: update the index
- `Escape`: cancel an active index update

