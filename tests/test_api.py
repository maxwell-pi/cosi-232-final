import pytest
from unittest.mock import patch

PATCH_TARGET = "app.routes.process_query"

@pytest.mark.usefixtures("client", "db")
@patch(PATCH_TARGET)
def test_submit_and_status(mock_process_query, client, db):
    """
    Test submitting a job and checking its status. Mock out process_query to avoid long-running pipeline.
    """
    mock_process_query.return_value = None  

    payload = {
        "research_query": "Test query on NLP",
        "seed_ids": ["W123", "W456"]
    }

    response = client.post("/submit", json=payload)
    print("Submit response JSON:", response.json)
    assert response.status_code == 200
    assert "job_id" in response.json

    job_id = response.json["job_id"]

    status_response = client.get(f"/status/{job_id}")
    print("Status response JSON:", status_response.json)
    assert status_response.status_code == 200

    data = status_response.json
    assert data["research_query"] == payload["research_query"]
    assert data["seed_ids"] == payload["seed_ids"]
    assert data["status"] in ["pending", "running", "complete", "failed"]

@pytest.mark.usefixtures("client", "db")
def test_submit_missing_fields(client):
    """
    Test that submitting without required fields returns a 400 error.
    """
    response = client.post("/submit", json={"research_query": "Only the query"})
    assert response.status_code == 400

    response = client.post("/submit", json={"seed_ids": ["W1", "W2"]})
    assert response.status_code == 400

    response = client.post("/submit", json={})
    assert response.status_code == 400



@pytest.mark.usefixtures("client", "db")
@patch(PATCH_TARGET)
def test_result_returns_error_for_incomplete_job(mock_process_query, client, db):
    mock_process_query.return_value = None

    resp = client.post("/submit", json={
        "research_query": "Test query for incomplete job",
        "seed_ids": ["W1"]
    })
    job_id = resp.json["job_id"]

    result_resp = client.get(f"/result/{job_id}")
    assert result_resp.status_code == 400
    assert result_resp.json["error"] == "Job not finished"

@pytest.mark.usefixtures("client", "db")
@patch(PATCH_TARGET)
def test_log_endpoint_returns_log(mock_process_query, client, db):
    mock_process_query.return_value = None

    resp = client.post("/submit", json={
        "research_query": "Test for logs",
        "seed_ids": ["W42"]
    })
    job_id = resp.json["job_id"]

    from app.models import QueryJob
    from app import db as real_db

    job = QueryJob.query.get(job_id)
    job.append_log("Test step 1\n")
    real_db.session.commit()

    log_resp = client.get(f"/log/{job_id}")
    assert log_resp.status_code == 200
    assert "Test step 1" in log_resp.json["log"]

@pytest.mark.usefixtures("client", "db")
@patch(PATCH_TARGET)
def test_status_contains_all_fields(mock_process_query, client, db):
    mock_process_query.return_value = None

    resp = client.post("/submit", json={
        "research_query": "Test query full fields",
        "seed_ids": ["W123"]
    })
    job_id = resp.json["job_id"]

    status_resp = client.get(f"/status/{job_id}")
    data = status_resp.json

    assert "job_id" not in data  
    assert "research_query" in data
    assert "seed_ids" in data
    assert "status" in data
    assert "created_at" in data
    assert "updated_at" in data

@pytest.mark.usefixtures("client", "db")
def test_invalid_job_id_returns_404(client):
    bad_id = 9999

    for route in ["status", "result", "log"]:
        resp = client.get(f"/{route}/{bad_id}")
        assert resp.status_code == 404
