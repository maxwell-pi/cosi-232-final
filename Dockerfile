FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl build-essential

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --upgrade pip && pip install .

COPY . .

RUN chmod +x run_flask.sh

CMD bash -c '\
  if [ ! -f data/openalex_nlp.jsonl ] || [ ! -f index/faiss_index.idx ]; then \
    echo "Running setup.py to download NLP papers and build index..."; \
    python setup.py; \
  else \
    echo "Data and index already present."; \
  fi && \
  ./run_flask.sh'
