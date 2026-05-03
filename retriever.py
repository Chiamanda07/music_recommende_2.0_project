"""
Retrieval system for the RAG music recommender.

Given a natural-language mood query, this module finds the top-N songs
from the database whose mood descriptions are most semantically similar,
using cosine similarity between sentence embeddings.
"""

from typing import List, Dict, Any

import numpy as np

from embedder import Embedder
from songs import SONGS


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and every row in matrix.

    Returns a 1-D array of similarity scores, one per row.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normalized_matrix = matrix / row_norms
    return normalized_matrix @ query_norm


class Retriever:
    """Finds the most mood-relevant songs for a free-text query."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def retrieve(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Return the top_n songs that best match the mood described in query.

        Each result is the original song dict from songs.py with an extra
        'similarity' key (float, 0–1) indicating how close the match is.
        """
        query_vec = self._embedder.embed(query)
        scores = _cosine_similarity(query_vec, self._embedder.song_embeddings)

        # Grab indices of the top_n highest scores.
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            song = dict(SONGS[idx])          # copy so we don't mutate the source
            song["similarity"] = float(scores[idx])
            results.append(song)

        return results


# ------------------------------------------------------------------
# Smoke test — run a few sample queries and print results
# ------------------------------------------------------------------
if __name__ == "__main__":
    embedder = Embedder()
    retriever = Retriever(embedder)

    test_queries = [
        "I feel really sad and heartbroken after a breakup",
        "I want to work out and feel unstoppable",
        "I am anxious and stressed about everything",
        "I feel nostalgic and want to think about old memories",
        "I just want to relax and do nothing",
    ]

    for query in test_queries:
        print(f"\nQuery : {query!r}")
        print("-" * 60)
        results = retriever.retrieve(query, top_n=3)
        for rank, song in enumerate(results, start=1):
            print(
                f"  {rank}. {song['title']} — {song['artist']}"
                f"  [{song['genre']}]  (similarity: {song['similarity']:.3f})"
            )
