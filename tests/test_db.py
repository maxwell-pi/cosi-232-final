from app.models import QueryJob
from datetime import datetime
import pytest

def test_create_job(db):
    job = QueryJob(research_query="Some question", seed_ids="W123,W456")
    db.session.add(job)
    db.session.commit()
    assert job.id is not None
    assert isinstance(job.created_at, datetime)

def test_append_log(db):
    job = QueryJob(research_query="Test", seed_ids="W1")
    db.session.add(job)
    db.session.commit()
    job.append_log("Step 1 complete")
    db.session.commit()
    assert "Step 1" in job.log


def test_as_dict_structure(db):
    job = QueryJob(
        research_query="Dict output test",
        seed_ids="W123,W999"
    )
    db.session.add(job)
    db.session.commit()

    result = job.as_dict()
    assert isinstance(result, dict)
    assert result["research_query"] == "Dict output test"
    assert result["seed_ids"] == ["W123", "W999"]
    assert result["status"] == "pending"

def test_null_seed_ids_rejected(db):
    with pytest.raises(Exception):
        job = QueryJob(research_query="Missing seeds", seed_ids=None)
        db.session.add(job)
        db.session.commit()

def test_null_query_rejected(db):
    with pytest.raises(Exception):
        job = QueryJob(research_query=None, seed_ids="W1,W2")
        db.session.add(job)
        db.session.commit()

def test_status_update_and_persistence(db):
    job = QueryJob(
        research_query="Test status update",
        seed_ids="W321,W654"
    )
    db.session.add(job)
    db.session.commit()

    job.status = "running"
    db.session.commit()

    fetched = QueryJob.query.get(job.id)
    assert fetched.status == "running"
