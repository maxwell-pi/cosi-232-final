# Eratosthenes: NLP Paper Retrieval + Summarization

A research assistant that takes a natural language query and a set of seed papers and returns an annotated bibliography of relevant papers. Combines semantic retrieval with citation graph expansion. Interfaces include a Flask API, a Streamlit front-end, and a persistent SQL database. Data is sourced from OpenAlex.

# Running

## Set OpenAI API Key in .env

Make sure you set your OpenAI API key in a .env file:

```
OPENAI_API_KEY=sk-proj-EPM...
```

## Docker Compose

Run the system fully containerized. Persists the vector index and database across runs.

```
docker-compose up --build
```

Then, start Streamlit with:
```
streamlit run interface.py
```

Note: The RAG/network crawl process is pretty slow. Live RAG progress logs are only printed to the Flask console. They don’t stream into the Streamlit UI. When using Docker, you can hopefully view these logs in the terminal you ran the above command in, but if not, try:

```
docker-compose logs -f flask
```

## Local (Dev Mode)

Set up your environment with `uv` or `venv`, then:

```
# Setup step (downloads OpenAlex NLP papers and builds vector index)
python setup.py

# Start Flask backend
./run_flask.sh

# In a separate terminal, start Streamlit
uv run streamlit run interface.py
```

## Direct Pipeline

Make sure `setup.py` has been run once, so that the starter data set is downloaded and embedded.

To run the bibliography pipeline directly from Python:

```
python pipeline.py
```

Edit the bottom of `pipeline.py` to set the query and seed papers.

# What You Can Do

* Enter a research question and a set of seed papers (or select them by keyword)
* Get an LLM-generated summary and annotated bibliography
* Expand the seed set with citation neighbors
* Use an OpenAlex-derived FAISS vector index for filtering
* Track job status, logs, and results through the Flask API or Streamlit
* Visualize citation graphs of selected papers

# Repo Structure

* `pipeline.py` — main bibliography generation routine
* `setup.py` — first-time fetch NLP papers + build FAISS index
* `interface.py` — Streamlit UI
* `run_flask.sh` — launches Flask server
* `app/` — Flask app, DB models, routes, background tasks
* `data/` — OpenAlex NLP papers (JSONL)
* `index/` — FAISS vector index
* `rag_logs/` — stored output logs (which papers chosen)
* `summaries/` - stored output logs (generated output)
* `tests/` — pytest tests for core components
* `open_alex_library.py` — citation graph logic
* `bibliography.py` — wrapper around final output

