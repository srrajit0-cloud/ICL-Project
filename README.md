# ICL-Project: RAG learning guide

A beginner-friendly guide to building a Retrieval-Augmented Generation (RAG) system with Python, ChromaDB, sentence-transformer embeddings, and an optional LLM provider. The material is organized around document ingestion, chunking, vector storage, retrieval, grounded generation, and vector-store CRUD operations.

## Guide

- [End-to-end RAG pipeline](docs/RAG_PIPELINE.md) — concepts, setup, and implementation steps from source documents to generated answers.
- [ChromaDB CRUD operations](docs/CHROMA_CRUD.md) — create, read, update, upsert, delete, and search records.

## Quick start

Use Python 3.10 or later. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

The guide examples use local sample text and a persistent ChromaDB collection. An LLM API key is only needed for the answer-generation stage. Copy `.env.example` to `.env` and add a provider key if you choose to run that stage. Never commit real API keys.

## RAG in brief

1. Load trusted source documents.
2. Split documents into manageable, overlapping chunks while retaining source metadata.
3. Embed each chunk and persist it in ChromaDB.
4. Embed a user question and retrieve the closest chunks, optionally filtering by metadata.
5. Pass the retrieved evidence and question to an LLM; instruct it to abstain when the evidence is insufficient.
6. Return the answer with source references and evaluate retrieval and answer quality.

See the linked guides for runnable Python examples, operational notes, and CRUD details.
