# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

---

## Repo Structure

```plaintext
├── dataset.py         # Starter word lists and example posts (you will expand these)
├── mood_analyzer.py   # Rule based classifier with TODOs to improve
├── main.py            # Runs the rule based model and interactive demo
├── ml_experiments.py  # (New) A tiny ML classifier using scikit-learn
├── model_card.md      # Template to fill out after experimenting
└── requirements.txt   # Dependencies for optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based starter:

    ```bash
    python main.py
    ```

If pieces of the analyzer are not implemented yet, you will see helpful errors that guide you to the TODOs.

To try the ML model later, run:

```bash
python ml_experiments.py
```

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Observe unusual or incorrect predictions and think about why they happen.
- Train a tiny machine learning model and compare its behavior to your rule based system.
- Complete the model card with your findings about data, behavior, limitations, and improvements.
- The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules.
- When debugging, print tokens, scores, or intermediate choices.
- Ask an AI assistant to help create edge case posts or unusual wording.
- Try examples that mislead or confuse your model. Failure cases teach you the most.

---

## Phase 2: RAG Music Recommender Upgrade

The next version of this project adds a **Retrieval-Augmented Generation (RAG)** system. Instead of just labeling mood, the system will accept a natural language mood description and recommend a song that fits.

### How it works

1. The user types how they are feeling in plain English (e.g. "I feel nostalgic and a little melancholy").
2. The system converts that description into a vector embedding.
3. It searches a song database for the songs whose mood descriptions are most similar.
4. An LLM uses the retrieved songs as context and generates a personalized recommendation with an explanation.

### Implementation Checklist

#### Phase 2.1 — Song Dataset
- [x] Create `songs.py` with a list of songs, each containing: title, artist, genre, and a short mood description
- [x] Cover a wide range of moods: happy, sad, energetic, calm, angry, nostalgic, romantic, anxious, focused, etc.
- [x] Aim for at least 50 songs across different genres

#### Phase 2.2 — Embedding Layer
- [x] Add `sentence-transformers` to `requirements.txt`
- [x] Create `embedder.py` that converts any text string into a vector embedding using a pre-trained model
- [x] Pre-compute embeddings for all songs and cache them so they are not recomputed every run

#### Phase 2.3 — Retrieval System
- [x] Create `retriever.py` that accepts a query string and returns the top-N most similar songs using cosine similarity
- [x] Test retrieval with sample mood queries and verify the results make sense

#### Phase 2.4 — LLM Integration
- [x] Add the `google-generativeai` SDK to `requirements.txt`
- [x] Store the Gemini API key in a `.env` file and add `.env` to `.gitignore`
- [x] Create `rag_recommender.py` that sends the user's mood query plus the retrieved songs to Gemini
- [x] Prompt Gemini to pick the best match and explain in 2–3 sentences why it fits the mood

#### Phase 2.5 — Interactive CLI
- [ ] Add an interactive loop to `rag_recommender.py` where the user types their mood and receives a song recommendation
- [ ] Display the song title, artist, genre, and the LLM's explanation
- [ ] Let the user type `another` to get the next best match or `quit` to exit

#### Phase 2.6 — Testing and Evaluation
- [ ] Test with at least 10 diverse mood inputs and record whether the recommendations feel relevant
- [ ] Identify at least 3 edge cases where the system struggles (vague moods, conflicting feelings, slang)
- [ ] Document findings in `model_card.md`

#### Phase 2.7 — Documentation
- [ ] Update `requirements.txt` with all new dependencies (`sentence-transformers`, `google-generativeai`, `numpy`)
- [ ] Update repo structure section of this README to include the new files
- [ ] Add a usage example showing a sample mood input and recommendation output
