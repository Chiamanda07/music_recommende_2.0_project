"""
Embedding layer for the RAG music recommender.

Responsibilities:
  - Load a pre-trained sentence-transformer model
  - Embed any text string into a fixed-size vector
  - Pre-compute and cache song embeddings to disk so they are only
    generated once; subsequent runs load from the cache instantly
"""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from songs import SONGS

# Cache file stores song embeddings so the model only runs once.
CACHE_PATH = Path(__file__).parent / "song_embeddings_cache.npz"

# "all-MiniLM-L6-v2" is fast, lightweight, and strong at semantic similarity.
MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps a SentenceTransformer model and manages the song embedding cache."""

    def __init__(self) -> None:
        print(f"Loading embedding model '{MODEL_NAME}'...")
        self._model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")

        self.song_embeddings: np.ndarray
        self._load_or_build_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D numpy array (embedding) for a single text string."""
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Return a 2-D numpy array of embeddings for a list of texts."""
        return self._model.encode(texts, convert_to_numpy=True)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _load_or_build_cache(self) -> None:
        """
        Load pre-computed song embeddings from disk if the cache exists
        and covers all current songs; otherwise rebuild and save it.
        """
        if self._cache_is_valid():
            data = np.load(CACHE_PATH, allow_pickle=False)
            self.song_embeddings = data["embeddings"]
            print(f"Loaded {len(self.song_embeddings)} song embeddings from cache.")
        else:
            self._build_cache()

    def _cache_is_valid(self) -> bool:
        """Return True only if the cache file exists and has the right number of songs."""
        if not CACHE_PATH.exists():
            return False
        try:
            data = np.load(CACHE_PATH, allow_pickle=False)
            return data["embeddings"].shape[0] == len(SONGS)
        except Exception:
            return False

    def _build_cache(self) -> None:
        """Embed all songs and write the result to disk."""
        print(f"Building embeddings for {len(SONGS)} songs (this runs once)...")
        descriptions = [song["mood_description"] for song in SONGS]
        self.song_embeddings = self.embed_batch(descriptions)
        np.savez(CACHE_PATH, embeddings=self.song_embeddings)
        print(f"Cache saved to '{CACHE_PATH}'.")


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    embedder = Embedder()

    sample = "I feel melancholy and nostalgic, thinking about old memories"
    vec = embedder.embed(sample)

    print(f"\nSample query : {sample!r}")
    print(f"Embedding shape : {vec.shape}")
    print(f"First 5 values  : {vec[:5].round(4)}")
    print(f"Song matrix shape: {embedder.song_embeddings.shape}")
    print("\nEmbedder is working correctly.")
