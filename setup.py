import os, jsonlines
from open_alex_library import fetch_topic
from vector_base import build_vector_base, save_vector_base

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for item in data:
            writer.write(item)

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

# download nlp data

SAVE_PATH = "data/openalex_nlp.jsonl"
TOPIC_ID_NLP = "3At10181"  # nlp

papers = fetch_topic(pages=1000)  # 1 page ~ 25 papers
save_jsonl(papers, SAVE_PATH)
print(f"Saved {len(papers)} papers to {SAVE_PATH}")

# create nlp data vector database

# papers = load_jsonl('data/openalex_nlp.jsonl')

INDEX_DIR = 'index/'

index, metadata = build_vector_base(papers)
save_vector_base(index, metadata, INDEX_DIR)
