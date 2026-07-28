# Advanced-RAG-VDA-5050-Fleet-Communication-Oracle

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant built for integration engineers, robotics technicians, and fleet managers working with Autonomous Mobile Robots (AMRs).

The system acts as a specialized copilot for the **VDA-5050 communication standard**, allowing users to query complex MQTT protocol rules, JSON schema payloads, and technical diagrams in any language, as well as upload custom field manuals dynamically.

---

## Key Features

* **Advanced RAG Engine:** Implements multi-query expansion, hybrid search (BM25 + vector similarity), and cross-encoder re-ranking for maximum retrieval accuracy on technical protocol specifications.
* **Multilingual Capability:** Query the system in English, German, French, Arabic, or other languages; the system retrieves relevant English protocol documentation and answers in the user's language.
* **Multimodal Architecture (Experimental):**
    * Pipeline engineered to process technical sequence diagrams and flowcharts via Vision LLMs (Rate-limited in free-tier; architecture preserved in `vision_parse.py`).
* **Dynamic Document Lifecycle:**
    * Upload custom manuals or schemas via the API to expand knowledge on the fly (`POST /ingest`).
* **API Backend:** Exposes FastAPI REST endpoints (`/ingest`, `/query`) for integration.

---

## Repository Structure

```text
Advanced-RAG-VDA-5050-Fleet-Oracle/
├── backend/
│   ├── core/
│   │   ├── config.py           # Centralized .env configuration & path resolution
│   │   ├── ingestion.py        # Document loading (.md, .schema) & text chunking
│   │   ├── vectorstore.py      # Qdrant client, embedding model, collection management
│   │   ├── retriever.py        # Dense similarity retriever (top-k)
│   │   ├── generator.py        # LLM chain (Groq — LLaMA 3 70B)
│   │   ├── rag_pipeline.py     # Facade re-exporting all pipeline functions
│   │   ├── vision_parse.py     # Experimental Multimodal Vision architecture
│   │   ├── run_ingestion.py    # CLI: load, chunk, embed, upsert into Qdrant
│   │   └── run_query.py        # CLI: interactive RAG query loop
│   └── main.py                 # FastAPI REST application (`/ingest`, `/query`)
├── data/
│   ├── raw_docs/               # Seed VDA-5050 Markdown docs, JSON schemas & diagrams
│   │   ├── VDA5050_EN.md       # Full VDA 5050 V3.0.0 specification
│   │   ├── json_schemas/       # 8 official JSON schemas (order, state, connection, etc.)
│   │   └── assets/             # Technical diagrams (.png) for multimodal processing
│   ├── user_uploads/           # Storage directory for runtime user uploads
│   ├── qdrant_db/              # Local Qdrant vector database storage
│   └── chunks.pkl              # Pickled chunks for BM25 sparse retrieval
├── evaluation/
│   ├── test_dataset.json       # 20 ground-truth Q&A pairs with expected_keywords & expected_sources
│   ├── evaluate_rag.py         # RAGAS evaluation pipeline (LLM-as-Judge, legacy)
│   ├── evaluate_fact_checklist.py  # Deterministic Fact-Checklist evaluation (zero-token)
│   └── results/                # Persisted evaluation run outputs
├── .env                        # Environment variables (API keys, paths)
├── .gitignore                  # Git suppression rules
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview and setup guide

```

---

## Tech Stack

* **API Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Store:** [Qdrant](https://qdrant.tech/) (Local disk mode for development / Production-ready)
* **LLM Model (Generator):** LLaMA 3.3 70B Versatile (via Groq)
* **Multimodal Vision Parser:** LLaMA 3.2 90B Vision (via Groq)
* **Embedding Model:** BAAI/bge-m3 (Local via Sentence-Transformers — Multilingual support)
* **Evaluation:** Deterministic Fact-Checklist (zero-token, no external LLM required)

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

Create a `.env` file in the root directory:

```env
# Generator Key (Groq): https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here

BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
QDRANT_PATH=./data/qdrant_db
EMBEDDING_MODEL_NAME=BAAI/bge-m3

```

---

## How to Run

### Phase 1 — RAG Pipeline (CLI)

**Step 1: Ingest documents into the vector store:**

```bash
python backend/core/run_ingestion.py
```

This loads all `.md` and `.schema` files, chunks them, and embeds into Qdrant.

**Step 2: Query the RAG pipeline (interactive mode):**

```bash
# Launch the interactive query loop
python backend/core/run_query.py

# Or pass a single question directly
python backend/core/run_query.py "What MQTT QoS levels does VDA 5050 specify?"
```

**Step 3: Evaluate the pipeline (Fact-Checklist):**

```bash
# Run the deterministic evaluation (no LLM API calls needed)
python evaluation/evaluate_fact_checklist.py --tag baseline

# Limit to first 5 questions for quick testing
python evaluation/evaluate_fact_checklist.py --tag quick_test --limit 5
```


### Phase 2 — FastAPI Backend

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API documentation will be accessible at `http://127.0.0.1:8000/docs`.

---

## API Summary

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/ingest` | Uploads a document (Markdown, Schema, PDF), chunks it, embeds it, and upserts to Qdrant |
| `POST` | `/query` | Submits text queries and returns grounded RAG answers with source citations |

---

## Evaluation Strategy

### Active: Fact-Checklist Evaluation (Deterministic)
A zero-token, fully automated method that scores the RAG pipeline against ground-truth technical keywords and source documents. No external LLM API calls are required.

| Metric | Description |
| --- | --- |
| **Fact Accuracy Score (%)** | Percentage of `expected_keywords` (technical terms, field names, enum values) found in the generated answer |
| **Retrieval Hit Rate (%)** | Percentage of `expected_sources` (target filenames like `order.schema`) found in the retrieved context chunks |

### Deprecated: RAGAS (LLM-as-Judge) — Why It Did Not Work

We initially attempted to use [RAGAS](https://github.com/explodinggradients/ragas) for evaluation, which uses an LLM to "judge" answer quality across four metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy). The implementation is preserved in `evaluation/evaluate_rag.py` for reference, but it was abandoned due to three fundamental blockers:

1. **Google Gemini Free Tier (15 RPM):** RAGAS fires dozens of LLM requests per question. Google's strict 15 Requests Per Minute anti-spam limit caused exponential backoff loops and eventual `EOFError` crashes — evaluation never completed.
2. **Groq Free Tier (100K TPD):** Switching to Groq's `llama-3.3-70b-versatile` hit the 100,000 Tokens Per Day hard cap after evaluating only ~8 questions. Additionally, Groq does not support the `n>1` parameter that RAGAS requires for its Answer Relevancy metric (`BadRequestError: 'n' must be at most 1`).
3. **Local Inference:** Running a 70B+ parameter model locally for accurate judging requires enterprise-grade hardware (48GB+ VRAM), which was not available.

The Fact-Checklist approach was adopted as a superior alternative for this technical domain: it is deterministic, reproducible, cost-free, and directly measures whether critical VDA-5050 protocol terms appear in generated answers.

---

## Technical Report & Benchmarks

For a detailed analysis of architectural choices, chunking experiments, re-ranking benchmarks, and evaluation metrics, please refer to the detailed **Technical Report** in the project documentation directory.