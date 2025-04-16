Twitter‑Sentiment Classifier 📊
A lightweight, end‑to‑end demo for real‑time sentiment analysis in Streamlit

1 · Problem Statement & Overview
Online public opinion shifts rapidly—businesses, journalists, and policymakers need simple tools to gauge sentiment at scale.
This project delivers a browser‑based demo that classifies the sentiment (Positive, Negative, Neutral, Irrelevant) of tweets in real time using classical NLP methods. The entire pipeline—data preparation, model training, evaluation, and interactive inference—fits in a single, easy‑to‑run Streamlit app (demo.py).

2 · Methodology
Step	Technique	Why it matters
Dataset	Twitter Financial News & Entities (training + validation splits from Kaggle)	Provides labelled tweets with diverse sentiment.
Text cleaning	Lower‑casing, URL/user/hashtag removal, punctuation stripping, whitespace normalization	Reduces noise and sparsity.
Vectorisation	TF‑IDF (unigrams + bigrams, max_features=10 000, English stop‑words)	Captures local n‑gram context while staying lightweight.
Classifier	Logistic Regression (liblinear solver, class‑balanced loss)	Fast, interpretable baseline suitable for small demos.
Hyper‑parameter search	Grid‑search (C ∊ {0.1, 1, 5}) with 3‑fold CV	Selects regularisation strength for best F1/accuracy.
Techniques draw directly on course topics: text pre‑processing, sparse vector space models, linear classifiers, cross‑validation.

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
![image](https://github.com/user-attachments/assets/77783726-4a5b-4c40-9f9f-a8c5c3f15f9d)
### ROC analysis  
Because ROC curves require a binary condition, we display **Positive vs (all other classes)**:

![ROC curve](assets/roc_positive_vs_others.png)

*Area‑under‑curve (AUC) = **0.96** → the classifier distinguishes Positive tweets extremely well.*

**Interpretation**

* Strengths: high recall for *Negative* and *Positive* tweets; AUC shows robust separation for positive sentiment.  
* Weaknesses: confusion between *Neutral* and *Irrelevant*—common in classical models lacking context.  
* Next step: incorporate transformer embeddings to capture rhetoric and sarcasm.

---


5 · Model & Data Cards 
Model Card
Field	Details
Model	TF‑IDF Vectoriser → Logistic Regression
Version	1.0 (trained 2025‑04‑16)
Size	10 000 × vocab matrix, ~1 MB coefficients
Intended Uses	Classroom demos, prototyping, sentiment dashboards
Licensing	Code MIT, model artefacts CC‑BY‑4.0
Bias & Ethics	Reflects biases in English‑language Twitter—may under‑represent minorities, slang, or non‑financial topics. Not suitable for high‑stakes decisions.
Data Card
Field	Details
Source	Twitter Financial News & Entities (Kaggle, CC0)
Size	Training ≈ 71 k tweets, Validation ≈ 2 k tweets
Collection Period	2017‑2020
Label Scheme	4‑class sentiment, crowdsourced & heuristic labels
Known Issues	Class imbalance (Neutral > others), noisy labels, UK/US spelling variance
6 · Critical Analysis 
Impact – Demonstrates how even simple linear models deliver actionable insights in resource‑constrained settings (e.g., journalism classrooms, small businesses).

Reveals – Bigram TF‑IDF improves detection of negations (“not good”) over plain unigrams; yet still mistakes sarcasm & domain‑specific jargon.

Next Steps

Fine‑tune a distilled transformer (DistilBERT) for better nuance.

Add explainability (e.g., SHAP for feature importance).

Deploy on HuggingFace Spaces for public access.

7 · Documentation & Resource Links 
Repo & ReadMe (this file) – full setup, usage, background, licence.

Key Resources

Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

Scikit‑learn docs: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

Streamlit: https://docs.streamlit.io/

