import json
import uuid
import os
from datetime import datetime
import re

from llm_calls import generate_summary

def sanitize_filename(text):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")[:50]

class Bibliography:

    def __init__(self, annotated, query, seed_ids):
        self.annotated = annotated
        self.query = query
        self.seed_ids = seed_ids

    def save_summary(self, output_dir='summaries'):
        summary = generate_summary(self.query, self.annotated)
        os.makedirs(output_dir, exist_ok=True)

        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat().replace(":", "-") + "Z"

        filename_base = sanitize_filename(self.query or "untitled_topic")
        filename = f"{timestamp}__{filename_base}__{run_id[:8]}.json"
        filepath = os.path.join(output_dir, filename)

        output_entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "topic": self.query,
            "seed_ids": self.seed_ids,
            "bibliography": summary
        }

        with open(filepath, "w") as f:
            json.dump(output_entry, f, indent=2)

        print(f"\nSaved log to: {filepath}")

    def save(self, output_dir="rag_logs"):

        os.makedirs(output_dir, exist_ok=True)

        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat().replace(":", "-") + "Z"

        filename_base = sanitize_filename(self.query or "untitled_topic")
        filename = f"{timestamp}__{filename_base}__{run_id[:8]}.json"
        filepath = os.path.join(output_dir, filename)

        output_entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "topic": self.query,
            "seed_ids": self.seed_ids,
            "bibliography": self.annotated
        }

        with open(filepath, "w") as f:
            json.dump(output_entry, f, indent=2)

        print(f"\nSaved log to: {filepath}")


    def print_report(self):
        print("\n===== ERATOSTHENES FINAL OUTPUT =====\n")

        print("RESEARCH QUESTION")
        print(f"→ {self.query}")

        print("\nSEED PAPERS")
        for i, pid in enumerate(self.seed_ids, 1):
            print(f"{i}. {pid}")

        print("\nANNOTATED BIBLIOGRAPHY")
        for i, entry in enumerate(self.annotated, 1):
            title = entry.get("title", "[No Title]")
            authors = entry.get("authors", "[Unknown Authors]")
            year = entry.get("year", "[Unknown Year]")
            annotation = entry.get("annotation", "[No Annotation Provided]")
            print(f"\n[{i}] {title} ({year})")
            print(f"    Authors: {authors}")
            print(f"    Summary: {annotation}")

        print("\n===== END =====\n")


