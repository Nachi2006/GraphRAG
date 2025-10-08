# utils/helpers.py
import math
import heapq

def cosine_similarity(a, b):
    # a and b are lists or tuples of floats
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def top_k_similar(query_emb, candidates, k=5):
    """
    candidates: list of dicts with 'embedding' and other metadata
    returns top-k with 'score' in descending order
    """
    heap = []
    for c in candidates:
        emb = c.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_emb, emb)
        if len(heap) < k:
            heapq.heappush(heap, (score, c))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, c))
    results = []
    while heap:
        score, c = heapq.heappop(heap)
        c2 = c.copy()
        c2["score"] = score
        results.append(c2)
    results.reverse()  # highest first
    return results
