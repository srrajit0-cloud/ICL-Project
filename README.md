# ICL-Project: RAG learning materials

A step-by-step Retrieval-Augmented Generation (RAG) learning project using Python, ChromaDB, sentence-transformer embeddings, and optional Gemini/Groq generation. The notebooks follow the learning sequence of the reference ICL-Project, with reproducible examples and persistent local storage.

## Run the notebooks

Use Python 3.10 or later. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook
```

Open and run notebooks in order. The first three RAG steps work without an LLM API key. For Step 4, copy `.env.example` to `.env` and set `GROQ_API_KEY` or `GEMINI_API_KEY`. Keep `.env` private and never commit API keys.

## Step-by-step notebooks

1. [RAG Step 1: Load documents and chunk text](RAG_Step1_Document_Loading_and_Chunkin.ipynb)
2. [RAG Step 2: Store chunks in ChromaDB](RAG_Step2_Storing_Chunks_in_ChromaDB.ipynb)
3. [RAG Step 3: Retrieve relevant context](RAG_Step3_Retrieving_Relevant_Context.ipynb)
4. [RAG Step 4: Pass context to Gemini/Groq](RAG_Step4_Passing_Context_to_Gemini_Groq_LLM.ipynb)

## Supporting learning files

- [ChromaDB CRUD operations notebook](CRUD_Operations.ipynb)
- [Word embeddings notebook](Word_Embeddings.ipynb)
- [Cosine similarity notebook](Calculating_Cosine_Similarity_Mathematically.ipynb)
- [RAG pipeline guide](docs/RAG_PIPELINE.md)
- [ChromaDB CRUD guide](docs/CHROMA_CRUD.md)
- [Reference chatbot assessment PDF](https://github.com/karanbala007/ICL-Project/blob/main/mini_assessment_chatbot.pdf)

## Pipeline overview

`load documents → chunk → embed → store vectors → retrieve top-k context → generate a grounded answer → evaluate`

The notebooks provide small teaching examples. For real deployments, use documents you are authorized to process, preserve source metadata, evaluate retrieval/answer quality, and configure data retention and access controls.
