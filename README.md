# Advanced-RAG-VDA-5050-Fleet-Communication-Oracle

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant built for integration engineers, robotics technicians, and fleet managers working with Autonomous Mobile Robots (AMRs).

The system acts as a specialized copilot for the **VDA-5050 communication standard**, allowing users to query complex MQTT protocol rules, JSON schema payloads, and technical diagrams in any language, as well as upload custom field manuals dynamically.

---

## Key Features

* **Advanced RAG Engine:** Implements multi-query expansion, hybrid search (BM25 + vector similarity), and cross-encoder re-ranking for maximum retrieval accuracy on technical protocol specifications.
* **Multilingual Capability:** Query the system in English, German, French, Arabic, or other languages; the system retrieves relevant English protocol documentation and answers in the user's language.
* **Multimodal Support:**
    * Ingests technical sequence diagrams, network topologies, and state flowcharts via Vision LLMs.
    * Allows users to upload screenshots of robot UI error logs alongside text queries.
* **Dynamic Document Lifecycle (CRUD):**
    * Upload custom PDF documents via the frontend to expand knowledge on the fly (`POST /upload`).
    * Delete specific document vectors using metadata filtering (`DELETE /delete/{doc_id}`).
* **Modular Stack:** Decoupled FastAPI REST backend and Streamlit interactive frontend.

---

## Repository Structure

```text
Advanced-RAG-VDA-5050-Fleet-Oracle/
├── backend/
│   ├── core/
│   │   └── rag_pipeline.py     # LangChain, chunking, embeddings & retriever logic
│   ├── routers/
│   │   └── api.py              # FastAPI endpoints (/query, /upload, /delete)
│   └── main.py                 # FastAPI server entry point
├── frontend/
│   └── app.py                  # Streamlit chat & document management UI
├── data/
│   ├── raw_docs/               # Seed VDA-5050 PDFs, Markdown docs, JSON schemas & assets
│   └── user_uploads/           # Storage directory for runtime user uploads
├── evaluation/
│   ├── test_dataset.json       # Ground-truth Q&A evaluation benchmark
│   └── evaluate_rag.py         # RAGAS / Retrieval evaluation pipeline
├── .env.example                # Template for required environment variables
├── .gitignore                  # Git suppression rules
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview and setup guide

```

---

## Tech Stack

* **API Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.google.com/search?q=https://www.uvicorn.org/)
* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Store:** [Qdrant](https://qdrant.tech/) (Local disk mode for development / Production-ready)
* **LLM Model:** Google Gemini 2.5 Flash (via OpenRouter API - Multimodal Text & Vision)
* **Embedding Model:** BAAI/bge-m3 (Local via Sentence-Transformers - Multilingual support)
* **Evaluation Framework:** [RAGAS](https://www.google.com/search?q=https://github.com/explodinggradients/ragas)

---

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed or an active Conda environment:

```bash
conda create -n vda_rag python=3.10 -y
conda activate vda_rag

```

### 2. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/MarioAouad/Advanced-RAG-VDA-5050-Fleet-Communication-Oracle
cd Advanced-RAG-VDA-5050-Fleet-Communication-Oracle
pip install -r requirements.txt

```

### 3. Environment Variables

Create a `.env` file in the root directory based on `.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key_here
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

```

---

## How to Run

### 1. Start the FastAPI Backend

Run the backend server from the project root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

```

The API documentation will be accessible at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`.

### 2. Start the Streamlit Frontend

In a separate terminal window, run:

```bash
streamlit run frontend/app.py

```

The chatbot interface will open automatically in your browser at `http://localhost:8501`.

---

## API Summary

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check endpoint returning backend status |
| `POST` | `/query` | Submits text/image queries and returns grounded RAG answers with citations |
| `POST` | `/upload` | Processes an uploaded PDF, chunks it, and updates the vector database |
| `DELETE` | `/delete/{doc_id}` | Removes all vector chunks associated with a specific document ID |

---

## Technical Report & Benchmarks

For a detailed analysis of architectural choices, chunking experiments, re-ranking benchmarks, and evaluation metrics (Precision, Recall, Faithfulness), please refer to the detailed **Technical Report** in the project documentation directory.