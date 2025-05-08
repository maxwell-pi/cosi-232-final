
# build vectors

import json
import os
import faiss
from sentence_transformers import SentenceTransformer


def build_vector_base(papers, batch=True, embedder=faiss.IndexFlatIP):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    abstracts = [p["abstract"] for p in papers]
    embeddings = model.encode(abstracts, convert_to_numpy=True, batch_size=64 if batch else 1, show_progress_bar=True)
    print('Abstract embedding produced.')

    index = embedder(embeddings.shape[1])
    index.add(embeddings)

    return index, papers


# save vectors

def save_vector_base(index, metadata, dirname):
    faiss.write_index(index, os.path.join(dirname, "faiss_index.idx"))

    with open(os.path.join(dirname, "metadata_store.json"), "w") as f:
        json.dump(metadata, f)


# retrieve from vectors

class Retriever:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    @classmethod
    def from_dir(cls, index_dirname):
        index = faiss.read_index(os.path.join(index_dirname, "faiss_index.idx"))
        with open(os.path.join(index_dirname, "metadata_store.json")) as f:
            metadata = json.load(f)
        return cls(index, metadata)

    def retrieve(self, query, k=5):
        q_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_vec, k)
        return [self.metadata[i] for i in I[0]]

# build and retrieve

def retrieve_from_throwaway_vectors(papers, query, suggest_k):

    index, metadata = build_vector_base(papers, batch=False, embedder=faiss.IndexFlatL2)
    retriever = Retriever(index, metadata)
    selected_papers = retriever.retrieve(query, suggest_k)
    
    return selected_papers
