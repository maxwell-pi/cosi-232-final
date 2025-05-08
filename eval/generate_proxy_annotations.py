import os
import json
import re
import requests
from tqdm import tqdm

LOG_DIR = "rag_logs"
HEADERS = {"User-Agent": "maxwell-pi (pickering@brandeis.edu)"}

def load_logs(log_dir=LOG_DIR):
    logs = []
    for f in os.listdir(log_dir):
        if f.endswith(".json"):
            with open(os.path.join(log_dir, f)) as file:
                logs.append(json.load(file))
    return logs

def fetch_citing_papers(openalex_id, limit=10):
    work_id = openalex_id.split("/")[-1]
    url = f"https://api.openalex.org/works?filter=cites:{work_id}&per-page={limit}"
    r = requests.get(url, headers=HEADERS)
    if not r.ok:
        return []
    return r.json().get("results", [])

def extract_mentioning_sentences(text, title_keywords, author_lastname):
    if not text:
        return []
    sentences = re.split(r'(?<=[.?!])\s+', text)
    matched = []
    for sent in sentences:
        if any(kw.lower() in sent.lower() for kw in title_keywords) or author_lastname.lower() in sent.lower():
            matched.append(sent.strip())
    return matched

def build_proxy_annotations(logs):
    proxy = {}

    for log in tqdm(logs):
        topic = log["topic"]
        proxy[topic] = {}

        for paper in log["bibliography"]:
            openalex_id = paper["id"]
            title = paper["title"]
            authors = paper["authors"]
            author_lastname = authors.split(",")[0].split()[-1] if authors else ""

            title_keywords = [w for w in re.findall(r'\w+', title) if len(w) > 3]

            citing = fetch_citing_papers(openalex_id)

            matched_sentences = []
            for cp in citing:
                abs_text = cp.get("abstract")
                if not abs_text:
                    inv = cp.get("abstract_inverted_index")
                    if inv:
                        abs_text = reconstruct_abstract(inv)
                if abs_text:
                    matches = extract_mentioning_sentences(abs_text, title_keywords, author_lastname)
                    matched_sentences.extend(matches)

            if matched_sentences:
                proxy[topic][openalex_id] = " ".join(matched_sentences)

    return proxy

def reconstruct_abstract(inverted_index):
    if not isinstance(inverted_index, dict):
        return None
    positions = {}
    for word, locs in inverted_index.items():
        for pos in locs:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))

def save_annotations(annotations, path="eval/proxy_annotations_from_citations.json"):
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nSaved proxy annotations to {path}")

if __name__ == "__main__":
    logs = load_logs()
    proxy_annos = build_proxy_annotations(logs)
    save_annotations(proxy_annos)
