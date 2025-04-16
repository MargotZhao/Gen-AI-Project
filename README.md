### Twitter‑Sentiment Classifier 📊

A lightweight, end‑to‑end demo for real‑time sentiment analysis in Streamlit


### 1 · Problem Statement & Goal
---
- Need: Fast, no‑frills way to see how Twitter feels about a topic.

- Solution: A one‑page Streamlit app (demo.py) that tags each tweet as Positive · Negative · Neutral · Irrelevant in real time.
  

### 2 · Methodology – at a glance
---
Step	What we did	Why it matters
- Data	73 k tweets (Kaggle “Twitter Financial News & Entities”)	Four clear sentiment labels to learn from.
- Clean‑up	lower‑case → drop URLs, @ /@#, punctuation → trim spaces	Removes noise and keeps tokens meaningful.
- Features	TF‑IDF (unigrams + bigrams, top 10 k features, stop‑words off)	Captures local context, stays lightweight.
- Model	Logistic Regression (liblinear, class‑balanced)	Trains fast, easy to explain, strong baseline.
- Tuning	3‑fold grid‑search on C = 0.1 / 1 / 5	Finds the best regularisation for F1 / accuracy.

### 3 · Implementation & Demo 
---
- Single‑file app demo.py – trains (cached) and launches the UI.

- Interactive UI

- Type a tweet → instant sentiment prediction + class probabilities

- Upload CSV → batch predictions, downloadable results

- Validation metrics & confusion matrix under “See detailed metrics”

- Zero config – only pip install -r requirements.txt and
- streamlit run demo.py.

### 4 · Assessment & Evaluation
---
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

- Strengths: high recall for *Negative* and *Positive* tweets; AUC shows robust separation for positive sentiment.  
- Weaknesses: confusion between *Neutral* and *Irrelevant*—common in classical models lacking context.  
- Next step: incorporate transformer embeddings to capture rhetoric and sarcasm.


### 5 · Model Card & Data Card
---

Model Card (v 1.0, 16 Apr 2025)

- Architecture: TF‑IDF → Logistic‑Regression

- Size: ≈ 1 MB (10 k‑term matrix)

Intended use:

- Classroom demos & tutorials

- Quick sentiment prototypes

- Lightweight dashboards

License: Code MIT · Weights CC‑BY‑4.0

Bias / limits:

- Mirrors English‑Twitter bias; minority slang under‑represented

⚠️ Not for high‑stakes decisions

Data Card

- Source: Twitter Financial News & Entities (Kaggle, CC0)

- Split: 71 k train · 2 k val (2017–2020)

- Labels: 4‑class sentiment (Pos / Neg / Neu / Irr)

Known issues:

- Neutral class dominates

- Noisy / inconsistent labels


### 6 · Critical Analysis
---
Impact: Linear baseline gives actionable insights with almost no compute—great for classrooms, newsrooms, small businesses.

Key takeaways:

- Bigrams help catch negations (“not good”).

- Still weak on sarcasm & niche jargon.

Next steps:

- Fine‑tune DistilBERT for richer context.

- Add SHAP / LIME for explainability.

- Publish a public demo on HuggingFace Spaces or Streamlit Cloud.


### 7. Additional Attempts - DistilBERT Quick Probe
---
Aspect	Summary
- Model	distilbert‑base‑uncased with a new 4‑class soft‑max layer
- Data slice	500 tweets for training · 100 tweets for validation
- Training recipe	HF tokenizer (max_len = 128), batch 16, AdamW (lr 2 × 10⁻⁵), 2 epochs (~60 updates)
- Outcome	Val‑accuracy ≈ 0.57 → below the TF‑IDF + LogReg baseline
- Why under‑performed?
  - Tiny dataset → over‑fits
- Only 2 epochs → head barely adapts
- No LR warm‑up / scheduler · no class weighting
Streamlit status	Not shipped – checkpoint wasn’t worth saving / loading

Key takeaway	
- Transformers need more data and epochs to shine. Until we fine‑tune on the full ~70 k tweets, the TF‑IDF + LogReg model stays as the primary demo.

Next step (future work): train DistilBERT on the entire dataset for 3‑5 epochs, save the checkpoint, and plug it into the Streamlit UI for a side‑by‑side comparison.
Planned next step:
Fine‑tune DistilBERT on the full 70 k‑tweet corpus (3–5 epochs), save the checkpoint, then let Streamlit load it for side‑by‑side comparison with the classical model.


### 8 · Documentation & Resource Links 
---
Repo & ReadMe (this file) – full setup, usage, background, licence.

Key Resources

Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

Scikit‑learn docs: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

Streamlit: https://docs.streamlit.io/

