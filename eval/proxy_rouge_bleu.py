import os
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PROXY_PATH = "eval/proxy_annotations_from_citations.json"
LOG_DIR = "rag_logs"

def load_logs(log_dir=LOG_DIR):
    logs = []
    for f in os.listdir(log_dir):
        if f.endswith(".json"):
            with open(os.path.join(log_dir, f)) as file:
                logs.append(json.load(file))
    return logs

def load_proxy_annotations(path=PROXY_PATH):
    with open(path) as f:
        return json.load(f)

def evaluate_rouge_bleu(logs, proxy):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results = []

    for log in logs:
        query = log["topic"]
        bib = log["bibliography"]
        if query not in proxy:
            continue

        proxy_refs = proxy[query]
        for paper in bib:
            pid = paper["id"]
            sys_anno = paper["annotation"].strip().replace("\n", " ")
            ref_anno = proxy_refs.get(pid)
            if not ref_anno:
                continue

            scores = scorer.score(ref_anno, sys_anno)
            bleu = sentence_bleu(
                [ref_anno.split()],
                sys_anno.split(),
                smoothing_function=SmoothingFunction().method1
            )

            results.append({
                "query": query,
                "paper_id": pid,
                "rouge1_f": round(scores["rouge1"].fmeasure, 4),
                "rougeL_f": round(scores["rougeL"].fmeasure, 4),
                "bleu": round(bleu, 4)
            })

    return results

def print_score_summary(results):
    from statistics import mean

    if not results:
        print("No matched annotations found.")
        return

    rouge1 = mean(r["rouge1_f"] for r in results)
    rougel = mean(r["rougeL_f"] for r in results)
    bleu = mean(r["bleu"] for r in results)

    print("\n===== ROUGE & BLEU EVAL (Proxy Annotations) =====\n")
    print(f"Average ROUGE-1 F: {rouge1:.4f}")
    print(f"Average ROUGE-L F: {rougel:.4f}")
    print(f"Average BLEU:      {bleu:.4f}")
    print(f"Total Samples:     {len(results)}\n")

if __name__ == "__main__":
    logs = load_logs()
    proxy = load_proxy_annotations()
    results = evaluate_rouge_bleu(logs, proxy)
    print_score_summary(results)
