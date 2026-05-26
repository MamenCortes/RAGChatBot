# Oncology Support RAG Chatbot

## Index
1. Introduction
2. Project Goals
3. Main Features
4. System Architecture
5. Project Structure
6. Retrieval Methods
7. Evaluation Mode (`/eval`)
8. Requirements
9. Installation and Setup
10. Database Setup
11. Ingesting Documents
12. Running the Chatbot
13. How the RAG Pipeline Works
14. Future Improvements
14. Author

---

# 1. Introduction

This project is a prototype Retrieval-Augmented Generation (RAG) chatbot designed to provide supportive and informational assistance to oncology patients, especially breast cancer patients.

The system combines:
- A Large Language Model (LLM)
- A retrieval system over curated oncology documents
- A Telegram chatbot interface
- An evaluation workflow for clinicians

The chatbot is not intended to replace healthcare professionals or provide medical diagnoses. Its purpose is to provide empathetic, safe, and grounded informational support using curated medical content.

---

# 2. Project Goals

The project was created to:

- Explore the use of RAG systems in supportive oncology care
- Reduce hallucinations by grounding answers in curated documents
- Compare different retrieval strategies
- Evaluate response quality with clinician feedback
- Support multilingual retrieval and responses

---

# 3. Main Features

- Telegram chatbot interface
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- Hybrid retrieval (semantic + keyword search)
- Language-aware hybrid retrieval
- Evaluation mode for clinicians (`/eval`)
- PostgreSQL + pgvector vector database
- Multilingual support
- Chunk-based document ingestion pipeline

---

# 4. System Architecture

The system follows a standard RAG pipeline:

```text
User Question
      ↓
Telegram Bot
      ↓
Retrieval System
      ↓
Relevant Document Chunks
      ↓
LLM Prompt Construction
      ↓
Generated Response
      ↓
Telegram Reply
```

Main components:

| Component | Purpose |
|---|---|
| Telegram Bot | User interaction interface |
| Retrieval Engine | Finds relevant document chunks |
| Embedding Model | Converts text into vectors |
| PostgreSQL + pgvector | Stores vectors and metadata |
| LLM | Generates grounded responses |
| Evaluation System | Collects clinician preferences |

---

# 5. Project Structure

```text
project/
│
├── telegram_bot.py      # Main chatbot entry point
├── rag.py               # RAG pipeline and prompt generation
├── retrieval.py         # Retrieval methods
├── ingest.py            # Chunk insertion into database
├── ingest_docs.py       # Document ingestion pipeline
├── embeddings.py        # Embedding generation
├── chunking.py          # Text chunking logic
├── db.py                # PostgreSQL connection utilities
├── config.py            # Environment variables and settings
│
├── eval/
│   └── eval_results.csv # Clinician evaluation results
│
├── .env                 # Environment variables
└── README.md
```

---

# 6. Retrieval Methods

The project implements multiple retrieval approaches.

## 6.1 Semantic Search

Semantic retrieval uses text embeddings.

### How it works

1. Documents are converted into vector embeddings
2. The user query is also embedded
3. Vector similarity search retrieves the most semantically similar chunks

Implementation:
- Embeddings generated with Sentence Transformers
- Stored in PostgreSQL using `pgvector`
- Cosine similarity used for ranking

### Advantages

- Understands meaning, not only keywords
- Handles paraphrases well
- Better contextual matching

### Disadvantages

- Can retrieve conceptually related but imprecise chunks
- May miss exact terminology
- Embedding quality depends on the model

---

## 6.2 Hybrid Search

Hybrid retrieval combines:
- Semantic search
- Keyword/full-text search

The project uses Reciprocal Rank Fusion (RRF) to combine rankings.

### How it works

The system performs:

1. Semantic vector search
2. PostgreSQL full-text keyword search
3. Rank fusion using RRF

The final ranking rewards chunks that perform well in both retrieval methods.

### Why hybrid retrieval?

Semantic retrieval is strong for meaning.
Keyword retrieval is strong for exact terminology.
Combining both often improves reliability.

### Advantages

- Better precision than semantic-only retrieval
- More robust to medical terminology
- Reduces irrelevant semantic matches

### Disadvantages

- More computationally expensive
- Requires tuning
- Full-text search can behave differently across languages

---

## 6.3 Language-Aware Hybrid Search

This is an extension of hybrid retrieval.

### How it works

1. The system detects the language of the user query
2. Retrieval is restricted to chunks in the same language
3. Language-specific PostgreSQL text search configurations are used
4. Semantic and keyword retrieval are fused with RRF

Supported language configurations include:
- English
- Spanish
- French
- German

### Why use language-aware retrieval?

Standard keyword search performs poorly across languages.
Filtering by language improves retrieval relevance and response quality.

### Advantages

- Better multilingual retrieval
- More accurate keyword matching
- Reduces cross-language noise

### Disadvantages

- Depends on accurate language detection
- Requires language metadata in chunks
- More complex pipeline

---

# 7. Evaluation Mode (`/eval`)

The chatbot includes a special evaluation workflow for clinicians.

## Purpose

The goal is to compare different response-generation strategies and collect human preferences.

This allows clinicians or evaluators to:
- Compare answer quality
- Evaluate safety and clarity
- Measure the impact of retrieval
- Select preferred answers

---

## Current Evaluation Modes

The system compares 3 response types:

| Mode | Description |
|---|---|
| `system_prompt_only` | Only the system prompt is used |
| `no_retrieval` | LLM answers using general knowledge only |
| `rag_retrieval` | Full RAG pipeline with hybrid retrieval |

---

## How to Use Evaluation Mode

Inside Telegram:

```text
/eval
```

Then ask a question normally.

The bot will generate:
- Multiple anonymous responses
- Randomized ordering
- Voting buttons

The evaluator selects the preferred answer.

---

## What Gets Saved

Evaluation results are stored in:

```text
eval/eval_results.csv
```

The CSV stores information such as:
- Timestamp
- User question
- Generated answers
- Preferred answer
- Selected mode

This enables later quantitative and qualitative analysis.

---

# 8. Requirements

## Python

Recommended:

```text
Python 3.10+
```

---

## Main Python Dependencies

Core libraries used:

```text
python-telegram-bot
huggingface_hub
sentence-transformers
psycopg2
pgvector
python-dotenv
langdetect
```

Additional dependencies may be required depending on your environment.

---

## External Requirements

### Telegram Bot API Key

Create a Telegram bot using:

```text
@BotFather
```

Save the token in `.env`.

---

### Hugging Face API Key

Required for LLM inference.

Create one at:

```text
https://huggingface.co/
```

---

### PostgreSQL + pgvector

The system requires:
- PostgreSQL
- pgvector extension enabled

---

# 9. Installation and Setup

## Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

---

## Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file:

```env
TELEGRAM_TOKEN=your_telegram_token
HF_API_KEY=your_huggingface_api_key
LLM_MODEL_NAME=your_model_name

PGHOST=localhost
PGPORT=5432
PGDB=your_database
PGUSER=your_user
PGPASSWORD=your_password

EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
```

---

# 10. Database Setup

Enable the `pgvector` extension:

```sql
CREATE EXTENSION vector;
```

Create the chunk table before ingestion.

The table stores:
- Chunk content
- Embeddings
- Metadata
- Language labels
- Source information

---

# 11. Ingesting Documents

Documents must be processed before the chatbot can answer questions.

The ingestion pipeline:

1. Reads documents
2. Splits them into chunks
3. Generates embeddings
4. Stores chunks in PostgreSQL

Run:

```bash
python ingest_docs.py
```

---

# 12. Running the Chatbot

Main entry point:

```bash
python telegram_bot.py
```

The Telegram bot will start listening for messages.

---

# 13. How the RAG Pipeline Works

The chatbot follows these steps:

1. User sends a question
2. Query embedding is generated
3. Retrieval system searches relevant chunks
4. Retrieved context is inserted into the prompt
5. The LLM generates a grounded answer
6. The response is returned through Telegram

The prompts include safety constraints such as:
- Avoiding unsupported medical claims
- Avoiding diagnosis or treatment decisions
- Encouraging professional medical consultation when necessary

---

# 14. Future Improvements

Possible future extensions:

- Better multilingual support
- Reranking models
- Citation generation
- Conversation memory
- Safer medical guardrails
- Web interface
- Retrieval evaluation metrics
- Automatic hallucination detection

---

# 15. Author

Author: [María del Carmen Cortés Navarro](https://github.com/MamenCortes) in collaboration with [Abraham Otero](https://github.com/AbrahamOtero).

Prototype RAG chatbot for supportive oncology assistance and retrieval evaluation research.

