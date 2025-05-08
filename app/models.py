from . import db
from datetime import datetime

class QueryJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    research_query = db.Column(db.Text, nullable=False)
    seed_ids = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(32), default='pending')
    result_path = db.Column(db.Text, nullable=True)
    log = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def append_log(self, message):
        self.log = (self.log or "") + message + '\n'

    def as_dict(self):
        return {
            "id": self.id,
            "research_query": self.research_query,
            "seed_ids": self.seed_ids.split(","),
            "status": self.status,
            "result_path": self.result_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
