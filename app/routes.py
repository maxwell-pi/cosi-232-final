from flask import request, jsonify
from . import db
from .models import QueryJob
from .tasks import process_query
from flask import current_app as app
import json, os
from open_alex_library import fetch_paper, collect_paper_neighbors

@app.route('/log/<int:job_id>')
def log(job_id):
    job = QueryJob.query.get_or_404(job_id)
    return jsonify({'log': job.log})

@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    query = data.get("research_query")
    seeds = data.get("seed_ids", [])
    retrieve_k = data.get("retrieve_k", 10)
    suggest_k = data.get("suggest_k", 5)
    citation_depth = data.get("citation_depth", 1)
    should_optimize_query = data.get("should_optimize_query", False)

    if not query or not seeds:
        return jsonify({"error": "Missing research_query or seed_ids"}), 400

    job = QueryJob(
        research_query=query,
        seed_ids=",".join(seeds),
        status="pending"
    )
    db.session.add(job)
    db.session.commit()

    process_query(
        job.id,
        retrieve_k=retrieve_k,
        suggest_k=suggest_k,
        citation_depth=citation_depth,
        should_optimize_query=should_optimize_query
    )

    return jsonify({"job_id": job.id})

@app.route("/status/<int:job_id>")
def status(job_id):
    job = QueryJob.query.get_or_404(job_id)
    return jsonify(job.as_dict())

@app.route("/result/<int:job_id>")
def result(job_id):
    job = QueryJob.query.get_or_404(job_id)
    if job.status != "complete":
        return jsonify({"error": "Job not finished"}), 400
    with open(job.result_path) as f:
        return jsonify(json.load(f))

@app.route("/jobs")
def jobs():
    recent = QueryJob.query.order_by(QueryJob.created_at.desc()).limit(20).all()
    return jsonify({"jobs": [j.as_dict() for j in recent]})


NLP_PAPERS_PATH = os.path.join("data", "openalex_nlp.jsonl")

@app.route("/search_nlp_papers")
def search_nlp_papers():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    results = []
    with open("data/openalex_nlp.jsonl") as f:
        for line in f:
            paper = json.loads(line)

            title = str(paper.get("title") or "")
            abstract = str(paper.get("abstract") or "")

            if query in title.lower() or query in abstract.lower():
                results.append({
                    "id": paper["id"],
                    "title": title,
                    "year": paper.get("year"),
                    "authors": ", ".join(paper.get("authors", []))
                })
                if len(results) >= 20:
                    break

    return jsonify({"results": results})


@app.route("/citation_graph/<openalex_id>")
def citation_graph(openalex_id):
    try:
        paper_id = openalex_id.split("/")[-1]
        center = fetch_paper(paper_id)
        if not center:
            return jsonify({"nodes": [], "edges": []})

        neighbors = collect_paper_neighbors([center], [], citation_depth=1)
        nodes = []
        edges = []
        seen_ids = set()

        for p in neighbors:
            pid = p["id"].split("/")[-1]
            if pid not in seen_ids:
                nodes.append({
                    "id": pid,
                    "label": p.get("title", "[No Title]")[:60],
                    "title": p.get("title", "[No Title]"),
                    "group": "paper"
                })
                seen_ids.add(pid)

        for p in neighbors:
            pid = p["id"].split("/")[-1]
            for ref in p.get("referenced_works", []):
                ref_id = ref.split("/")[-1]
                if ref_id in seen_ids:
                    edges.append({"from": pid, "to": ref_id})

        return jsonify({"nodes": nodes, "edges": edges})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
