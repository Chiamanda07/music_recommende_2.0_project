# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).



## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.

**Examples of correct predictions:**  
Provide 2 or 3 examples and explain why they were correct.

**Examples of incorrect predictions:**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only

---

## Phase 2: RAG Music Recommender

### System Overview

**How it works:**  
The user types a free-text description of their mood. A sentence-transformer model (`all-MiniLM-L6-v2`) converts both the query and all song mood descriptions into vector embeddings. Cosine similarity is used to retrieve the top 5 most relevant songs. A Groq-hosted LLM (`llama-3.1-8b-instant`) then receives the query and candidates as context and picks the single best match, explaining why it fits.

**Components:**
- `songs.py` — 62 songs with mood descriptions across 12 mood categories
- `embedder.py` — sentence-transformer embedding + disk cache
- `retriever.py` — cosine similarity search
- `rag_recommender.py` — LLM integration and interactive CLI

---

### Test Results

Run `python rag_recommender.py`, enter each query below, and fill in what song was recommended and whether it felt like a good match (Yes / Somewhat / No).

| # | Mood Query | Song Recommended | Good Match? | Notes |
|---|-----------|-----------------|-------------|-------|
| 1 | I feel really happy and want to dance |"Can't Stop the Feeling!" by Justin Timberlake | Yes | No notes |
| 2 | I just went through a breakup and I'm heartbroken |"Someone Like You" by Adele | Yes | That's a good classic |
| 3 | I need to focus and get work done | "Lose Yourself" by Eminem |Maybe | I guess that's ok |
| 4 | I feel nostalgic and miss my childhood |Summer of '69" by Bryan Adams |Not bad | Different people might have different songs that resonate with teir own childhood |
| 5 | I'm stressed and anxious about everything | "Weightless" by Marconi Union | Good | No notes |


**Overall relevance:** Out of 5 queries, how many recommendations felt like a good match? 4/5

---

### Edge Case Testing

Test these three inputs and record what happens.

**Edge Case 1 — Vague mood:**  
Query: `I feel weird`  
Result: *"Liability" by Lorde*  
Why it's tricky: No clear emotional signal — "weird" could mean anxious, confused, excited, or anything.

**Edge Case 2 — Slang and mixed signals:**  
Query: `lowkey vibing but also kinda sad ngl`  
Result: *"Someone Like You" by Adele*  
Why it's tricky: Internet slang may not match the formal mood descriptions in the song database.

**Edge Case 3 — Conflicting emotions:**  
Query: `I'm happy for my friend but also jealous and kind of sad`  
Result: *"Someone Like You" by Adele*  
Why it's tricky: Three competing emotions make it hard to settle on one best-fit song.

---

### Findings and Limitations

**What worked well:**  
When you don't mix different emotions in a sentence, it gives you a good pick

**What struggled:**  
When there where different emotions in a sentence it picks only one. For example, everytime "sad" was mentioned, Adele's song "someone like you" came up. Even if the person was happy sad.

**Limitations of the RAG approach:**
- The song database only has 62 songs, so rare or niche moods may not have a strong match
- The LLM always picks a winner even when no candidate is a great fit
- Slang and informal language in queries may not align well with the formal mood descriptions written for each song
- The system has no memory — it cannot learn from feedback over time

**Ideas for improvement:**
- Expand the song database to 200+ songs with more niche moods
- Write mood descriptions using more casual, natural language to better match user queries
- Add user feedback ("did this recommendation feel right?") to improve future picks
- Use a reranking step where the LLM scores all 5 candidates before picking one
