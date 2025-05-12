from typing import List
import os

def normalize(path):
    return os.path.basename(path).lower()

def evaluate_retrieval(gold_sources: List[str], retrieved_sources: List[str]):
    gold_set = set(normalize(src) for src in gold_sources)
    retrieved_set = set(normalize(src) for src in retrieved_sources)

    relevant_retrieved = len(gold_set & retrieved_set)
    total_retrieved = len(retrieved_set)
    total_relevant = len(gold_set)

    def precision(a, b): return a / b if b else 0
    def recall(a, b): return a / b if b else 0
    def f1(p, r): return 2 * p * r / (p + r) if p + r else 0

    p = precision(relevant_retrieved, total_retrieved)
    r = recall(relevant_retrieved, total_relevant)
    f = f1(p, r)

    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
