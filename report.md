# Retrieval-Augmented Generation for Scientific Paper Exploration
## Maxwell Pickerings

This project implements a retrieval-augmented generation (RAG) pipeline for scientific paper exploration, using a seed corpus of NLP papers. Given a user query, it retrieves relevant papers from a dense vector index and generates an annotated bibliography using an LLM. The core idea is to help researchers survey a topic quickly by surfacing high-signal papers and summarizing their relevance, without requiring them to manually assemble and review search results. Paper selection is handled via FAISS-based vector similarity using MiniLM embeddings over OpenAlex abstracts. The process is iterative; the citation graph associated with a seed paper is traversed, and new vector database subsets are build up to encourage sampling from the discourse around a discovered paper. The generation step uses prompt engineering with GPT-4 to summarize the current state of the literature and justify paper recommendations.

Every run logs full parameters, citations, and LLM outputs to JSON. Evaluation includes both automatic metrics using proxy annotations and qualitative review of system outputs. Streamlit and Flask components provide a minimal interface and backend, though the main emphasis remains on the underlying retrieval and generation behavior. The system is self-contained and easy to run, with prebuilt Docker support for full environment setup.

Unfortunately, I had to shift away from the previous RLHF-based Codenames agent, which despite lots of effort never achieved convergence. Although dissapointing to give up, the RAG project is more tractable in the remaining time. 

## Project Background and Pivot

The project began as an attempt to build a reinforcement learning agent for the board game Codenames. The early idea was to model communication using asymmetrical agents—spymaster and guesser—and tune policies with reinforcement learning and human feedback. Unfortunately, the learning dynamics were too fragile. The feedback signals were too noisy, the convergence was unstable, and it was necessary to bootstrap plausible models in order to evaluate by self-play. I want to continue working on the project after the semester, but I realized that I had run out of time.

This RAG project is something I'd had some prototype code for since the original project proposal, as it was an avenue I was considering. I made a choice late in the game that I should pivot to finalizing that project instead, as I don't have the time to get the Codenames project to the state I'd like it to be in (or that you might like it to be in!).


## OpenAlex and Citation Graph Traversal

The project relies heavily on OpenAlex as a metadata source. OpenAlex provides structured access to academic papers, including titles, abstracts, authors, publication years, and citation relationships. Each paper is assigned a persistent ID (e.g., `https://openalex.org/WXXXXXX`), and metadata includes a list of works it cites (`referenced_works`) and a citation count. This makes it usable as a citation graph.

For this project, OpenAlex serves two purposes. First, it provides the raw dataset used to construct the vector index. A filtered subset of OpenAlex papers was downloaded; those tagged under the “Natural Language Processing” concept. This subset was extracted using OpenAlex’s metadata dumps and APIs, filtered for English-language content with usable abstracts. These papers were encoded into dense vectors and indexed with FAISS.

Second, OpenAlex is used at runtime to expand the initial set of retrieved papers. Given a seed set of OpenAlex IDs, the system can recursively fetch their citations and references up to a fixed depth (usually 1 or 2). This provides a simple way to pull in semantically related but structurally distant papers. The function `collect_paper_neighbors` wraps this traversal, caching responses to avoid redundant API calls.

The traversal is implemented as a directed graph walk. For each paper, it fetches the `referenced_works`, then pulls metadata for each referenced node. To keep search bounded, the expansion is limited by both depth and total number of nodes (e.g. 15 per layer). In practice, this brings in older papers that are foundational but no longer sit near the centroid of recent embeddings. It also helps break local maxima where the embedding model over-focuses on a particular subdomain (e.g., word vectors instead of sentence embeddings).


## RAG Pipeline

The core pipeline begins with a user-supplied query. That query can optionally be rewritten by an LLM to make it more information-rich and retrieval-friendly. This optimization step is useful when the original input is vague, too broad, or not phrased in a way that aligns well with the embedding space. The (rewritten or original) query is encoded using all-MiniLM-L6-v2, a small sentence-transformer. That embedding is used to query a FAISS index built over a curated set of NLP-tagged papers from OpenAlex.

The top-k most similar papers are returned. Optionally, the system expands this initial set by following inbound and outbound citation edges from the seed papers, pulling metadata for their neighbors via OpenAlex’s API. This step is useful for recovering slightly older or foundational papers that might not be nearest neighbors in vector space but still highly relevant.

The next stage constructs a prompt that combines the user’s query and a selection of paper abstracts. These abstracts are formatted consistently, each including title, authors, and abstract text. This prompt is passed to an LLM with a fixed instruction. The output is parsed and saved as structured JSON, including topic metadata, paper selections, and their generated annotations. Everything is written to disk.


## Embedding and Indexing

We use `all-MiniLM-L6-v2` from sentence-transformers. Abstracts are encoded and stored in FAISS with inner product similarity. The vector index is backed by metadata in a JSON file.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode([p["abstract"] for p in papers])
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
```

The global FAISS index uses IndexFlatIP, which performs inner product similarity. This is appropriate because all sentence embeddings are normalized, so inner product behaves like cosine similarity. Cosine is well-suited to the retrieval stage, where we want to find papers that are directionally aligned with the query regardless of magnitude. It gives fast, scalable dense retrieval across the full corpus.

For local sets, such as the top-k results or papers from citation expansion, we switch to IndexFlatL2. At this stage, the semantic space is already narrowed. The vectors are directionally similar, so cosine becomes less informative. L2 distance adds sensitivity by considering magnitude differences between vectors. This helps in re-ranking, clustering, and filtering for diversity. Also, inner product search is faster over the global index, where L2 would be infeasible.

## Retrieval Logic

Retrieval starts by embedding the user’s query using the same sentence-transformer model that was used to index the papers. This ensures that the embedding space remains consistent. Once the query is encoded, it is passed to a FAISS index which performs an inner product similarity search to return the top-k closest paper embeddings. Each index entry is linked to a paper record containing title, abstract, authors, and metadata. There is no reranking applied after retrieval—the output order is based directly on raw FAISS similarity scores.

```python
def retrieve(query, k=10):
    qvec = model.encode([query])
    scores, indices = index.search(qvec, k)
    return [papers[i] for i in indices[0]]
```

This basic retrieval process can be expanded by adding citation-based traversal. For each of the top-k seed papers, we optionally expand the set by following citation edges in OpenAlex. These can be either incoming citations (papers that cite the seed) or outgoing references (papers that the seed cites). We control expansion depth and fan-out, so a citation traversal of depth 1 with a limit of 5 expands each seed paper with up to 5 additional neighbors. These neighbor papers are fetched from OpenAlex and optionally added to the retrieval pool before generation. This improves recall, especially for sparsely embedded concepts.

To refine the pool, a lightweight filtering LLM can optionally remove low-relevance papers before generation. This helps reduce noise introduced by over-expansion. Filtering prompts contain the query and a candidate abstract, asking whether the abstract is relevant. Only those judged relevant continue into the prompt-building stage.

## Prompt Design and Generation

The generation step begins by constructing a prompt designed to elicit a concise and structured literature overview from an LLM. The prompt includes the original user query, followed by a list of relevant paper abstracts. Instructions are clearly laid out in natural language, asking the model to summarize the state of the literature and recommend key papers with justifications.


This prompt is simple but effective. It avoids any special formatting or mark-up that might confuse the LLM, relying instead on example structure and semantic instructions. The prompt is truncated as needed to fit token limits, prioritizing the top-k most relevant abstracts.

The model’s response is parsed line by line and converted into structured JSON. Each entry in the final bibliography includes the OpenAlex ID of the paper (linked from the FAISS metadata), its title, year, list of authors, and the LLM’s annotation justifying its inclusion. This output format is consistent across runs and easily stored, evaluated, or rendered in the interface. It makes downstream use straightforward whether in logs, UI, or export.

The modular prompt-generation-parse cycle makes it easy to test different prompt styles or switch models without disrupting the overall flow.

To support analysis, a separate scoring LLM may also be used. It computes relevance or similarity between the user query and each abstract, allowing post-hoc evaluation or the derivation of aggregate metrics. The use of a dedicated scoring model decouples retrieval from evaluation, giving more flexibility in pipeline tuning.

In addition, a summarizing LLM can be applied to produce a standalone overview of the literature, separate from the five-paper recommendation list. This can be especially useful for display in interfaces or for exporting concise topic overviews.


## Evaluation: Proxy Annotations

There is no gold-standard annotated bibliography dataset, so we construct a proxy using citation contexts. For each paper in the bibliography, we search for citing sentences from other papers. These citation contexts are used as approximate reference annotations. This allows us to compare generated annotations against human-written citing statements, despite their limitations.

### ROUGE and BLEU

We compute ROUGE-1, ROUGE-L, and BLEU scores using standard n-gram overlap metrics. These metrics capture lexical similarity, but are limited in measuring semantic accuracy or rhetorical framing. As expected, scores are low, since human-written citation lines often differ in wording even when they convey the same point.

```text
Average ROUGE-1 F: 0.1841
Average ROUGE-L F: 0.1122
Average BLEU:      0.0020
Total Samples:     57
```

These results reflect the challenges of matching the freeform text of citation lines. Generated annotations are often informative but lexically distant from reference snippets.

### Semantic and Diversity Metrics

We compute cosine similarity between each seed paper and the recommended bibliography entries using sentence embeddings. Higher similarity indicates semantic closeness to the query seed. We also compute keyword entropy on the output using TF-IDF to measure lexical diversity. Higher entropy implies a broader topical spread.

Sample result:

```text
Query: What are the theoretical foundations of compositionality in NLP?
Papers Evaluated: 3
Avg Similarity: 0.703
Min / Max Similarity: 0.6869 / 0.7177
Keyword Entropy: 3.2955
Top Keywords: compositionality, expression, meaning
```

More queries:

```text
Query: What strategies exist for low-resource languages?
Papers Evaluated: 8
Avg Similarity: 0.5187
Min / Max Similarity: 0.4591 / 0.5691
Keyword Entropy: 3.2796
Top Keywords: languages, resource, models, speakers, data

Query: Evaluation without humans.
Papers Evaluated: 4
Avg Similarity: 0.4928
Min / Max Similarity: 0.4132 / 0.5725
Keyword Entropy: 3.2766
Top Keywords: human, automatic, evaluation, quality, machine

Query: Cross-lingual embeddings
Papers Evaluated: 5
Avg Similarity: 0.641
Min / Max Similarity: 0.5519 / 0.6815
Keyword Entropy: 3.2827
Top Keywords: cross, embeddings, lingual, languages, projection
```

These results suggest that even when average similarity is modest, the keyword entropy stays relatively high, indicating topic coverage is not overly narrow.

## Qualitative Results

The qualitative behavior of the system varies by topic. For abstract queries with established theoretical traditions, the results show coherent aggregation around key papers. For example, for queries like "What are embeddings?", the retrieved bibliography emphasized canonical models like Word2Vec, interpretive analyses of linear structure, and cross-modal embeddings from vision-language systems. The generated summary framed embeddings correctly as real-valued representations derived from statistical interactions, and emphasized analogical structure as a salient property. These responses show the pipeline’s ability to trace both empirical and conceptual trajectories.

When asked about "how gender relates to language," the system surfaced a much broader slice of literature, ranging from foundational sociolinguistics (e.g. Lakoff’s work) to corpus-based empirical surveys, and bias-focused work in machine translation. This variety indicates that the RAG process captured distinct methodological traditions: theoretical linguistics, computational social science, and modern NLP. The summary highlighted descriptive variation in language use, structural gender bias, and system performance gaps. These responses demonstrate the pipeline’s fluency in moving between humanistic and technical perspectives when both are relevant.

On more technical queries like "Cross-lingual embeddings," the bibliography clustered tightly around alignment methods and projection techniques. Most papers tackled the problem of learning mappings between monolingual spaces, using either linear, non-linear, or self-supervised approaches. The summary emphasized the limits of assuming isomorphism and the importance of data quality. There was little conceptual drift, suggesting that the retrieval + expansion + filtering steps remained on topic and converged effectively.

In general, when the seed paper had a clearly scoped research focus, the retrieved bibliography maintained coherence. When the query was more open-ended, the diversity in recommendations increased. Across the board in my experience, the annotations avoided hallucination and tended to summarize relevant information rather than fabricate.

## Bibliography Output Example

```json
{
  "run_id": "579dcb8d-a576-425e-bab2-d6a7acaf8d22",
  "topic": "What are embeddings?",
  "seed_ids": ["https://openalex.org/W658020064"],
  "bibliography": [
    {
      "id": "https://openalex.org/W2807140737",
      "title": "What the Vec? Towards Probabilistically Grounded Embeddings",
      "authors": "Carl Allen, Ivana Balažević, Timothy M. Hospedales",
      "year": 2018,
      "annotation": "This paper provides a theoretical understanding of word embedding models like Word2Vec and GloVe. It explores how their parameters relate to semantic relationships such as paraphrasing, offering a probabilistically grounded interpretation."
    },
    {
      "id": "https://openalex.org/W2913433659",
      "title": "Analogies Explained: Towards Understanding Word Embeddings",
      "authors": "Carl Allen, Timothy M. Hospedales",
      "year": 2019,
      "annotation": "This work explains the analogical properties of word embeddings. It derives mathematical justifications for observed behaviors like parallelogram analogies, providing clarity on why embeddings capture semantic regularities."
    },
    {
      "id": "https://openalex.org/W4390873448",
      "title": "Linear Spaces of Meanings: Compositional Structures in Vision-Language Models",
      "authors": "Matthew Trager, Pramuditha Perera, Luca Zancato et al.",
      "year": 2023,
      "annotation": "Although focused on vision-language models, this paper analyzes embedding space geometry and compositionality. It presents methods for generating concepts using idealized vectors and provides insights into debiasing and classification."
    },
    {
      "id": "https://openalex.org/W2787481916",
      "title": "A Survey of Word Embeddings Evaluation Methods",
      "authors": "Amir Bakarov",
      "year": 2018,
      "annotation": "This paper surveys 16 intrinsic and 12 extrinsic evaluation methods for word embeddings. It is valuable for understanding how embeddings are tested and what challenges remain in standardizing evaluation."
    }
  ]
}
{
  "run_id": "579dcb8d-a576-425e-bab2-d6a7acaf8d22",
  "timestamp": "2025-05-08T02-55-27.516682Z",
  "topic": "What are embeddings?",
  "seed_ids": [
    "https://openalex.org/W658020064"
  ],
  "bibliography": "The current literature on embeddings, specifically word embeddings, reveals a deep exploration into their nature and applications. Word embeddings are real-valued word representations that effectively capture lexical semantics and are trained on natural language corpora. These embeddings are known for their seemingly linear behavior and analogical properties, as observed in models like word2vec (W2V), despite not being explicitly trained to do so. This behavior is associated with the parallelogram analogy of words, for instance, \"woman is to queen as man is to king\". \n\nIn terms of understanding why these embeddings work, research indicates that the properties of these embeddings arise from the interactions between PMI vectors that reflect semantic relationships, such as similarity and paraphrasing. These relationships are encoded in a low dimensional space under suitable projection. Moreover, embeddings from vision-language models (VLMs) have been found to have compositional structures, estimated from a smaller set of vectors in the embedding space, which can be used to generate concepts directly.\n\nRecommended Papers:\n\n1. \"Linear Spaces of Meanings: Compositional Structures in Vision-Language Models\" - This paper is important as it provides an understanding of compositional structures in VLM embeddings, and their applicability in different tasks such as classification, debiasing, and retrieval.\n\n2. \"What the Vec? Towards Probabilistically Grounded Embeddings\" - This paper provides valuable insight into the theoretical understanding of word embeddings and explains why their properties are useful in downstream tasks.\n\n3. \"Analogies Explained: Towards Understanding Word Embeddings\" - This paper provides explanations about the linear behavior and analogical properties of word embeddings, making it a key resource for understanding the complexities of these embeddings.\n\n4. \"A Survey of Word Embeddings Evaluation Methods\" - This paper presents an extensive overview of the field of word embeddings evaluation, highlighting significant challenges and approaches. It is crucial for understanding the evaluation methods and the datasets used for word embeddings."
}
```


## Running the Pipeline

The easiest way to run the project is from the command line. The `pipeline.py` script takes arguments for query, seed IDs, retrieval count, citation depth, and query rewriting. It outputs structured logs.

```bash
uv run python pipeline.py \
  --query "What are embeddings?" \
  --seed_ids https://openalex.org/W658020064
```

Alternatively, you can run via container. Start Flask with `run_flask.sh`, then run `streamlit run interface.py` locally. Jobs are tracked, and logs are shown in the container terminal.

```bash
docker-compose up --build
```

## Repository Organization

`pipeline.py` is the primary entry point. It runs the full RAG pipeline from CLI with arguments for query, retrieval size, citation depth, and more.
`setup.py` initializes the vector index and downloads data from OpenAlex. It only runs if the index or data is missing.
`run_flask.sh` starts the Flask server. Streamlit is launched separately.
The `app/` directory contains the backend:

`routes.py` defines Flask endpoints for job submission, status, and result retrieval.
`models.py` defines the SQLAlchemy schema for persistent job tracking.
`tasks.py` manages threaded RAG execution with logging to the database.
`openalex_routes.py` serves routes for NLP paper search and citation graph construction.
Retrieval logic is kept in `vector_base.py`. It wraps FAISS and provides helper methods for dense query lookup and document embedding. The index lives under index/.

LLM prompt construction and calls are handled in `llm_calls.py`. Bibliographic structuring is in `bibliography.py`.

Data lives in `data/`, including `openalex_nlp.jsonl`. RAG outputs are saved in `rag_logs/`, with one JSON file per run. These include metadata, topic, and the full generated bibliography.

Tests live under `tests/`. They use pytest and include mock jobs, log verification, and CLI checks.

The repository includes a `Dockerfile` and `docker-compose.yml`. The container persistently mounts `data/`, `index/`, and `rag_logs/` to preserve state between runs.


## Future Directions

The biggest win would be interactive visualization and log display. Users should be able to see why papers were selected, and how generation proceeded. This means adding a live-updating frontend and capturing intermediate model steps.

There's a lot of interesting work to be done with other citation graph strategies and interactions with the throwaway subset vectorizations. It would be interesting to allow the user to dynamically change the strategy parameters as they develop the bibliography.

