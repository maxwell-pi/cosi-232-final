from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

load_dotenv()

client = OpenAI()

def optimize_query(raw_query):

    prompt = f"""Rewrite this research idea into a specific, information-rich query a retrieval system could use:

Input: {raw_query}

Optimized:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


def generate_summary(query, papers, model="gpt-4"):

    def build_prompt(query, papers):
        prompt = (
            "You are a helpful research assistant. Given a research question and a set of paper abstracts, do the following:\n"
            "1. Summarize what the current literature says.\n"
            "2. Recommend the included papers and briefly explain why each is important.\n\n"
            f"Research Question: {query}\n\nRelevant Papers:\n"
        )
        for i, p in enumerate(papers):
            prompt += f"\n[Paper {i+1}]\nTitle: {p['title']}\nAbstract: {p['abstract']}\n"
        prompt += "\nOutput:\nLiterature Summary:"
        return prompt
    
    prompt = build_prompt(query, papers)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scholarly research assistant that writes clear, accurate summaries and bibliographies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


def score_papers(query, candidate_papers):
    relevance_scores = []
    for p in tqdm(candidate_papers):

        decision_prompt = f"""You are a research assistant. A user is exploring the topic: '{query}'.
Should this paper be included in the bibliography? Rate from 1-9, and explain.

Title: {p['title']}
Abstract: {p['abstract']}

Decision:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": decision_prompt}],
            temperature=0.3,
            max_tokens=100
        )

        try:
            score_text = response.choices[0].message.content.strip()
            score = int(score_text.split()[0])  # First token = numeric score
            relevance_scores.append((score, p))
        except (ValueError, IndexError):
            pass

    return relevance_scores


def relevance_and_annotation(target_paper: dict, candidate_paper: dict) -> str:

    prompt = f"""
You are a research assistant analyzing citation relevance.

TARGET PAPER:
Title: {target_paper['title']}
Abstract: {target_paper['abstract']}

CANDIDATE CITATION:
Title: {candidate_paper['title']}
Abstract: {candidate_paper['abstract']}

Decide whether the candidate paper is relevant to the intellectual context of the target paper. If it is, explain why. If not, say "Not relevant."
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def relevance_and_annotation2(topic: str, candidate_paper: dict) -> str:

    prompt = f"""
You are a research assistant analyzing citation relevance.

TOPIC: {topic}

CANDIDATE CITATION:
Title: {candidate_paper['title']}
Abstract: {candidate_paper['abstract']}

Decide whether the candidate paper is relevant to the intellectual context of the topic. If it is, explain why. If not, say "Not relevant."
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def annotate_relevance(selected_papers, query):
    annotated = []
    for candidate in selected_papers:
        try:
            explanation = relevance_and_annotation2(query, candidate)
            if not explanation.lower().startswith("not relevant"):
                annotated.append({
                    "id": candidate["id"],
                    "title": candidate["title"],
                    "authors": ", ".join(candidate["authors"][:3]) + (" et al." if len(candidate["authors"]) > 3 else ""),
                    "year": candidate["year"],
                    "annotation": explanation,
                    'abstract': candidate['abstract']
                })
        except Exception as e:
            print(f"Skipping paper due to error: {e}")
    return annotated