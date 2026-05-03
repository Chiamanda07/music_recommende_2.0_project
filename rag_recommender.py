"""
RAG Music Recommender — Phase 2.4 + 2.5

Flow:
  1. User types how they are feeling.
  2. Retriever finds the top-5 most mood-relevant songs via cosine similarity.
  3. Gemini receives the query + retrieved songs as context and picks the
     best match, explaining in 2-3 sentences why it fits the mood.
  4. User can type 'another' to get the next best pick or 'quit' to exit.
"""

import os
from typing import List, Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

from embedder import Embedder
from retriever import Retriever


# ── Setup ──────────────────────────────────────────────────────────────────────

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key or _api_key == "your_api_key_here":
    raise EnvironmentError(
        "Gemini API key not found. "
        "Open the .env file and replace 'your_api_key_here' with your real key.\n"
        "Get one free at: https://aistudio.google.com/app/apikey"
    )

genai.configure(api_key=_api_key)
_gemini = genai.GenerativeModel("gemini-1.5-flash")


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(mood_query: str, candidates: List[Dict[str, Any]], already_suggested: List[str]) -> str:
    """Build the prompt sent to Gemini."""
    songs_block = "\n".join(
        f"{i+1}. \"{s['title']}\" by {s['artist']} [{s['genre']}]\n"
        f"   Vibe: {s['mood_description']}"
        for i, s in enumerate(candidates)
        if s["title"] not in already_suggested
    )

    skip_note = (
        f"\nAlready suggested: {', '.join(already_suggested)}. Do NOT recommend these again.\n"
        if already_suggested else ""
    )

    return f"""You are a music recommender. A user described their mood as:
"{mood_query}"
{skip_note}
Here are the top candidate songs retrieved by semantic search:
{songs_block}

Pick the single best song for this mood. Reply in this exact format:
SONG: <title> by <artist>
REASON: <2-3 sentences explaining why this song fits the mood perfectly>

Do not add anything else."""


# ── Gemini call ────────────────────────────────────────────────────────────────

def get_recommendation(
    mood_query: str,
    candidates: List[Dict[str, Any]],
    already_suggested: List[str],
) -> str:
    """Send the prompt to Gemini and return its raw text response."""
    prompt = _build_prompt(mood_query, candidates, already_suggested)
    response = _gemini.generate_content(prompt)
    return response.text.strip()


# ── CLI ────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 60)
    print("         Welcome to the RAG Music Recommender")
    print("=" * 60)
    print("Describe how you're feeling and get a song that matches.")
    print("Commands: 'another' for a new pick  |  'quit' to exit\n")

    embedder = Embedder()
    retriever = Retriever(embedder)

    while True:
        mood_query = input("How are you feeling? ").strip()
        if not mood_query or mood_query.lower() == "quit":
            print("\nHope the music helps. Goodbye!")
            break

        print("\nSearching for the right song...\n")
        candidates = retriever.retrieve(mood_query, top_n=5)
        already_suggested: List[str] = []

        while True:
            reply = get_recommendation(mood_query, candidates, already_suggested)
            print("-" * 60)
            print(reply)
            print("-" * 60)

            # Parse the suggested title so we can skip it on 'another'
            for line in reply.splitlines():
                if line.startswith("SONG:"):
                    suggested_title = line.replace("SONG:", "").split(" by ")[0].strip().strip('"')
                    already_suggested.append(suggested_title)
                    break

            print("\nType 'another' for a different pick, a new mood to start over, or 'quit' to exit.")
            follow_up = input("> ").strip().lower()

            if follow_up == "quit":
                print("\nHope the music helps. Goodbye!")
                return
            elif follow_up == "another":
                if len(already_suggested) >= len(candidates):
                    print("\nNo more candidates — try describing your mood differently.\n")
                    break
                print("\nFinding another pick...\n")
            else:
                # Treat anything else as a new mood query
                mood_query = follow_up
                print(f"\nSearching for: {mood_query!r}\n")
                candidates = retriever.retrieve(mood_query, top_n=5)
                already_suggested = []


if __name__ == "__main__":
    run()
