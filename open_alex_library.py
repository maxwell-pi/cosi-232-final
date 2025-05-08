import requests
from tqdm import tqdm


HEADERS = {
    "User-Agent": "maxwell-pi (pickering@brandeis.edu)"
}

def extract_paper_data(result):
    if "abstract_inverted_index" in result and result["abstract_inverted_index"]:
        abstract_tokens = sorted(result["abstract_inverted_index"].items(), key=lambda kv: kv[1][0])
        abstract_text = " ".join([word for word, _ in abstract_tokens])
        return {
            "id": result["id"],
            "title": result["title"],
            "abstract": abstract_text,
            "year": result.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in result.get("authorships", [])],
            "referenced_works": result.get("referenced_works", []),
            "cited_by_count": result.get("cited_by_count", 0)
        }

def fetch_topic(concept_id, per_page=25, pages=20):
    results = []
    base_url = "https://api.openalex.org/works"
    
    for page in tqdm(range(1, pages + 1), desc="Downloading papers"):
        params = {
            "filter": f"primary_topic.id:{concept_id},has_abstract:true",
            "per-page": per_page,
            "page": page
        }
        resp = requests.get(base_url, params=params, headers=HEADERS)
        if resp.status_code != 200:
            raise RuntimeError(f'Failed: {resp}')
        for d in resp.json()["results"]:
            paper_data = extract_paper_data(d)
            if paper_data:
                results.append(paper_data)
    return results

# will return None if no abstract.
def fetch_paper(openalex_id: str) -> dict:
    id = openalex_id.split("/")[-1]
    r = requests.get(f"https://api.openalex.org/works/{id}", headers=HEADERS)
    data = r.json()
    return extract_paper_data(data)

def get_upstream(paper: dict, limit=20) -> list[dict]:
    papers = []
    for pid in paper["referenced_works"][:limit]:
        try:
            p = fetch_paper(pid)
            if p:
                papers.append(p)
        except:
            continue
    return papers

def get_downstream(paper_id: str, max_papers=20) -> list[dict]:
    id = paper_id.split("/")[-1]
    results = []
    url = f"https://api.openalex.org/works?filter=cites:{id}&per-page=25"
    r = requests.get(url, headers=HEADERS)
    for d in r.json().get("results", [])[:max_papers]:
        paper_data = extract_paper_data(d)
        if paper_data:
            results.append(paper_data)
    return results

def collect_paper_neighbors(top_semantic, seed_paper_ids, citation_depth):
    semantic_pool = top_semantic.copy()
    seen_ids = {p["id"] for p in semantic_pool}

    for pid in tqdm(seed_paper_ids + list(seen_ids)):
        print(f"\nFetching paper: {pid}")
        main = fetch_paper(pid)
        if main["id"] not in seen_ids:
            semantic_pool.append(main)
            seen_ids.add(main["id"])

        frontier = [main]
        visited = set()
        for _ in range(citation_depth):
            next_frontier = []
            for paper in frontier:
                if paper["id"] in visited:
                    continue
                visited.add(paper["id"])

                try:
                    upstream = get_upstream(paper)
                    downstream = get_downstream(paper["id"])
                    for p in upstream + downstream:
                        if p["id"] not in seen_ids:
                            semantic_pool.append(p)
                            seen_ids.add(p["id"])
                            next_frontier.append(p)
                except Exception as e:
                    print(f"Skipping citation expansion for {paper['id']}: {e}")
            frontier = next_frontier
    return semantic_pool
