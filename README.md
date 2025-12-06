# Multimodal-RAG-System

> End-to-end multimodal RAG system that ingests DeepLearning.AI’s The Batch news articles and lets you
> ask questions using text + images, with a simple Streamlit UI

## Description

The multimodal RAG app:

-   Fetches and preprocesses The Batch newsletter articles.
-   Indexes both text and images into ChromaDB using:
-   Text embeddings: intfloat/multilingual-e5-base (by default)
-   Image embeddings: openai/clip-vit-base-patch32 (by default)

-   Exposes an agent that:
-   Chooses between text search, image search, or a hybrid strategy.
-   Returns an answer plus ranked sources (with preview images & snippets).
-   Provides a Streamlit chat UI with optional image upload.
-   Includes evaluation scripts (classic metrics + DeepEval).

The goal was to show how to build a practical, multimodal news search assistant - from ingestion and indexing, through retrieval and agents, all the way to a small demo UI and Docker deployment.

### Main Components

-   app.py – Streamlit app (chat + settings).
-   src/the_batch/ – Fetching & parsing The Batch articles.
-   src/embeddings/ – Text and image embedder wrappers.
-   src/vector_db/ – ChromaDB indexer and retrieval logic.
-   src/agent/ – agent and tools
-   src/scripts/chat.py - Simple CLI for chatting with agent
-   src/scripts/retrieve_data.py - Downloads and serializes The Batch articles.
-   src/scripts/ingest_chroma_db.py - Reads JSONL, builds embeddings, writes to Chroma.
-   data/ – Local data directory (JSONL, Chroma index, uploads).
-   data_test/ – Small test sets for RAG evaluation (classic metrics + DeepEval).
-   src/scripts/evaluate_rag.py - Evaluate data_test using classical RAG metrics
-   src/scripts/evaluate_rag_deepeval.py - Evaluate data_test using deepeval framework

### Tech Stack

-   Language: Python
-   UI: Streamlit
-   Vector DB: Chroma
-   LLM: Gemini (e.g. gemini-2.5-flash) via API
-   Text embeddings model: intfloat/multilingual-e5-base
-   Images embeddings model: openai/clip-vit-base-patch32 (CLIP)
-   Evaluation (optional): DeepEval + custom metrics scripts
-   Containerization (optional): Docker + docker-compose

## Getting Started

1. Prerequisites: Python 3.11
2. pip
3. A Gemini API key (for the chat agent), from Google AI Studio.
4. Internet access (to fetch The Batch articles and HF models).

### Environment Variables:

#optional

-   `GEMINI_MODEL`=gemini-2.5-flash
-   `TEXT_EMBED_MODEL`=intfloat/multilingual-e5-base
-   `CLIP_MODEL`=openai/clip-vit-base-patch32
-   `CHROMA_PATH`=data/chroma
-   `DATA_DIR`=data
-   `STREAMLIT_PORT`=8501
-   `OPENAI_API_KEY`=your_api_key_here

#necessary

-   `GEMINI_API_KEY`=your_api_key_here

> [!NOTE]
> All articles and the Chroma db are stored locally under `CHROMA_PATH`

Clone the Repository:

```shell
git clone https://github.com/Fenix125/Multimodal-RAG-System.git
cd Multimodal-RAG-System
```

Create & Activate Virtual Environment

```shell
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or:
# .venv\Scripts\activate
```

Install dependencies:

```shell
pip install --upgrade pip
pip install -r requirements.txt
```

> [!IMPORTANT]
> Make sure GEMINI_API_KEY is set – without it the agent won't work

### Running the App:

Option A – Let the UI handle ingestion

Start Streamlit:

```shell
streamlit run app.py
```

-   Open the app in your browser (usually http://localhost:8501).
-   Go to the Settings tab in the sidebar.
-   Click “Fetch articles” then “Yes, fetch & ingest”.

This will:

-   Download The Batch articles for several topics.
-   Build text + image embeddings.
-   Store everything in Chroma under data/chroma.

Switch back to Chat and start asking questions! This is the simplest way for demoing.

Option B – Ingest via CLI scripts:

```shell
# 1. Fetch raw articles into data/processed/the_batch_articles.jsonl
python -m src.scripts.retrieve_data

# 2. Build text + image embeddings and write to Chroma
python -m src.scripts.ingest_chroma_db

# 3. Run the Streamlit app
streamlit run app.py
```

Optional: Docker Deployment

Run with Docker:

```shell
docker compose up --build
```

Useful commands:

```shell
docker compose stop      # stop containers
docker compose start     # start them again
docker compose down      # stop + remove containers
```

> [!NOTE]
> The app-data volume keeps your Chroma DB between restarts,
> so you don’t repeatedly re-download and re-embed everything.

## Evaluation (Optional)

If you want to reproduce the evaluation:

1. src/scripts/evaluate_rag.py:
   Runs the agent over a dataset in data_test/metrics/
   Computes Recall@K, Precision@K, MRR, Hit@K
   Writes \*\_with_results.json and prints a small table
2. src/scripts/evaluate_rag_deepeval.py
   Uses DeepEval to compute:
    - ContextualRecall
    - ContextualRelevancy
    - AnswerRelevancy
    - Faithfulness

Works on datasets in data_test/deepeval/

> [!NOTE]
> You will need to setup `OPENAI_API_KEY` to use DeepEval evaluation

```shell
python -m src.scripts.evaluate_rag --help
python -m src.scripts.evaluate_rag_deepeval --help
```

(See each script’s --help for exact arguments.)

## License

This project is open-sourced.
See the LICENSE file in the repository for full details.
