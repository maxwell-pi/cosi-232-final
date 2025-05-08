from app.tasks import process_query
from app.models import QueryJob
import os

def test_process_query(db):
    job = QueryJob(research_query="What is a giraffe?", seed_ids="W2893912")
    db.session.add(job)
    db.session.commit()

    process_query(job.id)
    db.session.refresh(job)

    assert job.status in ["complete", "failed"]
    if job.status == "complete":
        assert os.path.exists(job.result_path)
