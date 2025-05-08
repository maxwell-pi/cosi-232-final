import os
from .models import QueryJob, db
from pipeline import from_query_and_papers
from datetime import datetime

LOG_DIR = "summaries"

def process_query(job_id, retrieve_k=10, suggest_k=5, citation_depth=1, should_optimize_query=False):
    job = QueryJob.query.get(job_id)
    if not job:
        return

    def db_logger(msg):
        job.append_log(f"[{datetime.utcnow().isoformat()}] {msg}")
        db.session.commit()

    job.status = "running"
    db.session.commit()

    try:
        from_query_and_papers(
            query=job.research_query,
            seed_paper_ids=job.seed_ids.split(","),
            retrieve_k=retrieve_k,
            suggest_k=suggest_k,
            citation_depth=citation_depth,
            should_optimize_query=should_optimize_query,
            log=db_logger
        )

        log_files = sorted(
            [f for f in os.listdir(LOG_DIR) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)),
            reverse=True
        )
        latest_log = os.path.join(LOG_DIR, log_files[0])
        job.result_path = latest_log
        job.status = "complete"

    except Exception as e:
        job.status = "failed"
        job.result_path = f"Error: {type(e).__name__}: {str(e)}"

    db.session.commit()