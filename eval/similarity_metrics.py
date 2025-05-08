import os
import json
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests

client = OpenAI()
LOG_DIR = "rag_logs"
HEADERS = {"User-Agent": "maxwell-pi (pickering@brandeis.com)"}

def fetch_abstract(openalex_id):
    short_id = openalex_id.split("/")[-1]
    url = f"https://api.openalex.org/works/{short_id}"
    r = requests.get(url, headers=HEADERS)
    if r.ok:
        data = r.json()
        return data.get("abstract", None) or data.get("abstract_inverted_index", None)
    return None

def reconstruct_abstract(inverted_index):
    if not isinstance(inverted_index, dict):
        return None
    positions = {}
    for word, locs in inverted_index.items():
        for pos in locs:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def load_logs(log_dir=LOG_DIR):
    logs = []
    for f in os.listdir(log_dir):
        if f.endswith(".json"):
            with open(os.path.join(log_dir, f)) as file:
                logs.append(json.load(file))
    return logs

def compute_query_similarity(query, paper_abstracts):
    query_emb = get_embedding(query)
    abstract_embeddings = [get_embedding(ab) for ab in paper_abstracts]
    sims = cosine_similarity([query_emb], abstract_embeddings)[0]
    return {
        "avg_query_similarity": round(np.mean(sims), 4),
        "min_query_similarity": round(np.min(sims), 4),
        "max_query_similarity": round(np.max(sims), 4),
        "all_similarities": sims.tolist()
    }

def compute_keyword_diversity(abstracts, top_k=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf = vectorizer.fit_transform(abstracts)
    terms = vectorizer.get_feature_names_out()

    term_counts = {}
    for row in tfidf.toarray():
        for i, val in enumerate(row):
            if val > 0:
                term_counts[terms[i]] = term_counts.get(terms[i], 0) + 1

    most_common = sorted(term_counts.items(), key=lambda x: -x[1])[:top_k]
    total = sum(c for _, c in most_common)
    entropy = -sum((c / total) * np.log2(c / total) for _, c in most_common if c > 0)

    return {
        "top_keywords": [kw for kw, _ in most_common],
        "keyword_entropy": round(entropy, 4)
    }

def evaluate_logs(logs):
    results = []

    for log in tqdm(logs):
        query = log["topic"]
        bib = log["bibliography"]

        if not bib:
            continue

        paper_abstracts = []
        failed_ids = []

        for paper in bib:
            oid = paper["id"]
            abs_data = fetch_abstract(oid)
            if abs_data is None:
                failed_ids.append(oid)
                continue
            if isinstance(abs_data, dict):
                abstract = reconstruct_abstract(abs_data)
            else:
                abstract = abs_data
            if abstract:
                paper_abstracts.append(abstract)

        if not paper_abstracts:
            print(f"Skipping: {query} (no abstracts)")
            continue

        try:
            sim_metrics = compute_query_similarity(query, paper_abstracts)
            keyword_metrics = compute_keyword_diversity(paper_abstracts)
        except Exception as e:
            print(f"Failed on query '{query}': {e}")
            continue

        result = {
            "query": query,
            "n_selected": len(paper_abstracts),
            "avg_query_similarity": sim_metrics["avg_query_similarity"],
            "min_query_similarity": sim_metrics["min_query_similarity"],
            "max_query_similarity": sim_metrics["max_query_similarity"],
            "keyword_entropy": keyword_metrics["keyword_entropy"],
            "top_keywords": keyword_metrics["top_keywords"],
            "missing_abstracts": failed_ids
        }

        results.append(result)

    return results

def print_summary(results):
    print("\n===== SEMANTIC + DIVERSITY METRICS =====\n")
    for r in results:
        print(f" Query: {r['query']}")
        print(f"  Papers Evaluated: {r['n_selected']}")
        print(f"  Avg Similarity: {r['avg_query_similarity']}")
        print(f"  Min / Max Similarity: {r['min_query_similarity']} / {r['max_query_similarity']}")
        print(f"  Keyword Entropy: {r['keyword_entropy']}")
        print(f"  Top Keywords: {', '.join(r['top_keywords'])}")
        if r["missing_abstracts"]:
            print(f"  Missing Abstracts: {len(r['missing_abstracts'])}")
        print()

if __name__ == "__main__":
    logs = load_logs()
    results = evaluate_logs(logs)
    print_summary(results)
