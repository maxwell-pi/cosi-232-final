from bibliography import Bibliography
from open_alex_library import collect_paper_neighbors
from vector_base import Retriever, retrieve_from_throwaway_vectors
from llm_calls import score_papers, annotate_relevance, optimize_query



def from_query_and_papers(query, seed_paper_ids, should_optimize_query=False, retrieve_k=7, suggest_k=3, citation_depth=1, log=None):

    def log_msg(msg):
        print(msg)
        if log:
            log(msg)

    retriever = Retriever.from_dir('index/')

    log_msg("ERATOSTHENES HYBRID MODE")

    log_msg(f"\nResearch Question: {query}")
    log_msg(f"Starting from {len(seed_paper_ids)} seed papers...")

    if should_optimize_query:
        optimized = optimize_query(query)
        log_msg(f"Optimized Query: {optimized}")
    else:
        optimized = query

    log_msg("Performing semantic retrieval...")
    retrieved = retriever.retrieve(optimized, k=retrieve_k)

    log_msg("Scoring retrievals...")
    if score_papers:
        scored = score_papers(optimized, retrieved)
        scored.sort(key=lambda pair: -pair[0])
        top_semantic = [p for s, p in scored[:suggest_k]]
    else:
        top_semantic = retrieved[:suggest_k]

    log_msg(f"Collected {len(top_semantic)} semantically relevant papers.")

    semantic_pool = collect_paper_neighbors(top_semantic, seed_paper_ids, citation_depth)

    log_msg(f"\nCollected {len(semantic_pool)} candidates before semantic filtering.")

    log_msg("Encoding abstracts for FAISS filtering...")
    
    selected_papers = retrieve_from_throwaway_vectors(semantic_pool, query, suggest_k)
    log_msg(f"Selected top {suggest_k} papers from semantic pool using FAISS.")

    log_msg("Running LLM relevance filter...")
    annotated = annotate_relevance(selected_papers, query)

    bib = Bibliography(annotated, query, seed_paper_ids)

    bib.save()
    bib.print_report()

    bib.save_summary()




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG pipeline on a query.")
    parser.add_argument("--query", required=True, help="Research question to answer")
    parser.add_argument("--seed_ids", nargs="+", required=True, help="Seed OpenAlex IDs")
    parser.add_argument("--retrieve_k", type=int, default=10, help="Number of papers to retrieve")
    parser.add_argument("--suggest_k", type=int, default=5, help="Number of key papers to suggest")
    parser.add_argument("--citation_depth", type=int, default=1, help="Citation traversal depth")
    parser.add_argument("--optimize", action="store_true", help="Use LLM to rewrite query")

    args = parser.parse_args()

    from_query_and_papers(
        query=args.query,
        seed_paper_ids=args.seed_ids,
        retrieve_k=args.retrieve_k,
        suggest_k=args.suggest_k,
        citation_depth=args.citation_depth,
        should_optimize_query=args.optimize,
        log=print,
    )
