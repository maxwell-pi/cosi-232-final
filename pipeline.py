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
    print(bib.summary)




if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--seed_ids", nargs='+', required=True, help="List of OpenAlex IDs (space-separated)")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--suggest_k", type=int, default=5)
    parser.add_argument("--citation_depth", type=int, default=1)
    parser.add_argument("--should_optimize_query", action="store_true")

    args = parser.parse_args()

    from_query_and_papers(
        query=args.query,
        seed_paper_ids=args.seed_ids,
        retrieve_k=args.retrieve_k,
        suggest_k=args.suggest_k,
        citation_depth=args.citation_depth,
        should_optimize_query=args.should_optimize_query,
        log=print,
    )

    # or just run from these:

    # from_query_and_papers('What are the theoretical foundations of compositionality in NLP?', ['w2153579005', 'w2962813108'])
    # from_query_and_papers('I want to find parallel corpus resources.', ['w22168010', 'w2047295649'])
    # from_query_and_papers('Evaluation without humans.', ['W2101105183'], retrieve_k=10, suggest_k=5)
    # from_query_and_papers('Early methods of machine translation.', ['W2006969979'])
    # from_query_and_papers('Can we model how languages evolve computationlly?', ['W2581563496'])
    # from_query_and_papers('What is a giraffe?', ['W2893912'])
    # from_query_and_papers('What strategies exist for low-resource langauages?', ['W2963088995', 'w3098341425'], retrieve_k=15, suggest_k=8)
