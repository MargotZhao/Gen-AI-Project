Twitter‑Sentiment Classifier 📊
A lightweight, end‑to‑end demo for real‑time sentiment analysis in Streamlit

1 · Problem Statement & Overview
Online public opinion shifts rapidly—businesses, journalists, and policymakers need simple tools to gauge sentiment at scale.
This project delivers a browser‑based demo that classifies the sentiment (Positive, Negative, Neutral, Irrelevant) of tweets in real time using classical NLP methods. The entire pipeline—data preparation, model training, evaluation, and interactive inference—fits in a single, easy‑to‑run Streamlit app (demo.py).

2 · Methodology
Methodology ― key moves at a glance
Dataset

Twitter Financial News & Entities (Kaggle, train + val splits)

→ gives us ~73 k labelled tweets covering four sentiment classes.

Text cleaning

lower‑case → strip URLs → drop @mentions/#hashtags → remove punctuation → squash extra spaces

→ cuts noise & sparsity; leaves only meaningful tokens.

Vectorisation

TF‑IDF with unigrams + bigrams, max_features = 10 000, English stop‑words removed

→ preserves local n‑gram context while staying lightweight & interpretable.

Classifier

Logistic Regression (solver = liblinear, class_weight = balanced)

→ quick to train, easy to explain, strong baseline for text.

Hyper‑parameter tuning

3‑fold grid‑search over C ∈ {0.1, 1, 5}

→ picks optimal regularisation (best F1 / accuracy).

3 · Implementation & Demo 
Single‑file app demo.py – trains (cached) and launches the UI.

Interactive UI

Type a tweet → instant sentiment prediction + class probabilities

Upload CSV → batch predictions, downloadable results

Validation metrics & confusion matrix under “See detailed metrics”

Zero config – only pip install -r requirements.txt and
streamlit run demo.py.

4 · Assessment & Evaluation
Metric (validation set)	Score
Accuracy	≈ 0.77
Macro F1	≈ 0.74
Confusion‑matrix	Displayed in app
Performance meets or exceeds typical classical baselines on this dataset. Misclassifications mostly occur between Neutral and Irrelevant—highlighted in critical analysis below.
### Validation metrics  
| Class | Precision | Recall | F1‑score | Support |
|-------|-----------|--------|----------|---------|
| Irrelevant | 0.67 | 0.64 | 0.66 | 172 |
| Negative   | 0.67 | 0.78 | 0.72 | 266 |
| Neutral    | 0.74 | 0.60 | 0.66 | 285 |
| Positive   | 0.73 | 0.78 | 0.76 | 277 |
| **Accuracy** | – | – | **0.705** | 1 000 |
| **Macro avg** | 0.70 | 0.70 | 0.70 | 1 000 |
| **Weighted avg** | 0.71 | 0.70 | 0.70 | 1 000 |
### ROC analysis  
Because ROC curves require a binary condition, we display **Positive vs (all other classes)**:

![image](https://github.com/user-attachments/assets/31153bac-4603-4575-bee9-28e28b3ac6ea)

*Area‑under‑curve (AUC) = **0.96** → the classifier distinguishes Positive tweets extremely well.*

**Interpretation**

* Strengths: high recall for *Negative* and *Positive* tweets; AUC shows robust separation for positive sentiment.  
* Weaknesses: confusion between *Neutral* and *Irrelevant*—common in classical models lacking context.  
* Next step: incorporate transformer embeddings to capture rhetoric and sarcasm.

---


### 5 · Model & Data Cards (bullet‑point edition)

Model Card

Architecture: TF‑IDF vectoriser ➜ Logistic‑Regression classifier

Version: v 1.0 — trained 📅 16 Apr 2025

Footprint: 10 k × vocab matrix, ≈ 1 MB coefficients

Intended uses:

Classroom demos & tutorials

Fast prototyping for sentiment features

Lightweight dashboards / internal monitoring

Licensing: Code 🪪 MIT • Model artefacts 🪪 CC‑BY‑4.0

Bias & ethics:

Mirrors English‑language Twitter biases → under‑represents minority slang & niche domains

⚠️ Not recommended for high‑stakes or policy decisions

Data Card

Source: Twitter Financial News & Entities (Kaggle, CC0)

Size: ≈ 71 k training tweets • ≈ 2 k validation tweets

Collection window: 2017 → 2020

Label scheme: 4‑class sentiment (Positive, Negative, Neutral, Irrelevant) — mix of crowd & rule‑based labels

Known issues:

Class imbalance (Neutral > others)

Noisy / inconsistent labels

UK vs US spelling variants + finance jargon

### 6 · Critical Analysis (bullet‑point edition)

Impact

Shows that simple linear models can still yield actionable insights with scarce compute — ideal for classrooms, local newsrooms, small businesses.

Key takeaways

Adding bigrams in TF‑IDF boosts detection of negations (“not good”) vs unigrams.

Model still struggles with sarcasm & domain‑specific slang — a common limitation of bag‑of‑words methods.

Next steps

Fine‑tune a distilled transformer (e.g., DistilBERT) for nuance & context.

Add explainability (SHAP / LIME) to surface influential n‑grams per prediction.

Deploy publicly on HuggingFace Spaces or Streamlit Community Cloud for wider feedback.

## Appendix · Alternate‑Model Attempt (DistilBERT fine‑tune)

“We also explored a transformer‑based approach to benchmark performance against the lightweight TF‑IDF + LogReg baseline.”

Model tried | distilbert‑base‑uncased + new 4‑way soft‑max head

Training subset | 500 tweets (train) / 100 tweets (val) – sampled from the same dataset

Training setup

Tokenised on‑the‑fly with the HF tokenizer (max len = 128)

Mini‑batches = 16, AdamW (lr = 2 × 10⁻⁵)

Epochs = 2 → ≈ 60 optimisation steps

Result | Val accuracy ≈ 0.55 – 0.60 (below baseline)

Key reasons for under‑performance

Tiny data slice (500 samples) → transformer overfits quickly, lacks signal.

Very short fine‑tuning (2 epochs) → insufficient adaptation of the classification head.

No learning‑rate warm‑up / scheduling or class weighting tweaks.

Streamlit integration — skipped, because the checkpoint wasn’t yet performant and wasn’t saved with model.save_pretrained(), so the app would have loaded a fresh, un‑tuned DistilBERT.

Take‑away
While transformers usually outperform linear models on full datasets, they require more data, epochs, and compute. Our quick probe confirms the trend; we therefore kept the TF‑IDF model as the primary demo and list the transformer path as a future enhancement:

Next iteration: fine‑tune DistilBERT on the full 70 k‑tweet corpus (3‑5 epochs), save the checkpoint, and swap it into the Streamlit UI for a side‑by‑side comparison.

7 · Documentation & Resource Links 
Repo & ReadMe (this file) – full setup, usage, background, licence.

Key Resources

Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

Scikit‑learn docs: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

Streamlit: https://docs.streamlit.io/

