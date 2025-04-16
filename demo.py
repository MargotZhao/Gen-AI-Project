# demo.py
# --------------------------------------------------
# Streamlit demo for TF‑IDF + Logistic‑Regression
# --------------------------------------------------
import streamlit as st
from pathlib import Path
import zipfile, re, joblib, io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------------
# 1. Helper – text cleaning (same as in your training script)
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)        # URLs
    text = re.sub(r"@\w+|#\w+", "", text)                      # mentions / hashtags
    text = re.sub(r"[^\w\s]", "", text)                        # punctuation
    text = re.sub(r"\s+", " ", text).strip()                   # extra spaces
    return text

# ------------------------------------------------------------------
# 2. Cached training step – runs once, then re‑uses results
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_model(archive_path: Path):
    extract_dir = archive_path.with_suffix("")                 # same directory name sans ".zip"
    if not extract_dir.exists():
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)

    train_df = pd.read_csv(extract_dir / "twitter_training.csv",
                           header=None,
                           names=["Tweet ID", "Entity", "Sentiment", "Content"])
    val_df   = pd.read_csv(extract_dir / "twitter_validation.csv",
                           header=None,
                           names=["Tweet ID", "Entity", "Sentiment", "Content"])

    # Sentiment normalization
    valid_labels = ["Positive", "Negative", "Neutral", "Irrelevant"]
    train_df["Sentiment"] = train_df["Sentiment"].where(
        train_df["Sentiment"].isin(valid_labels), "Neutral"
    )
    val_df["Sentiment"] = val_df["Sentiment"].where(
        val_df["Sentiment"].isin(valid_labels), "Neutral"
    )

    # Drop NA & clean
    train_df.dropna(subset=["Content", "Sentiment"], inplace=True)
    val_df.dropna(subset=["Content", "Sentiment"], inplace=True)
    train_df["Content"] = train_df["Content"].apply(clean_text)
    val_df["Content"]   = val_df["Content"].apply(clean_text)

    # Vectorise & fit
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 max_features=10_000,
                                 stop_words="english")
    X_train = vectorizer.fit_transform(train_df["Content"])
    X_val   = vectorizer.transform(val_df["Content"])

    model = LogisticRegression(max_iter=1_000,
                               solver="liblinear",
                               class_weight="balanced",
                               C=1.0)                       # best C from your grid‑search
    model.fit(X_train, train_df["Sentiment"])

    # -------------------  Metrics -------------------
    y_val_pred = model.predict(X_val)
    acc        = accuracy_score(val_df["Sentiment"], y_val_pred)
    report_str = classification_report(val_df["Sentiment"], y_val_pred)

    # Confusion‑matrix figure
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_val, val_df["Sentiment"], cmap="Blues", colorbar=False
    )
    disp.figure_.set_size_inches(4, 3)
    cm_fig = disp.figure_

    # Persist so you can skip training next run
    joblib.dump((vectorizer, model), "sentiment_model.joblib")

    return vectorizer, model, acc, report_str, cm_fig


# ------------------------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Twitter‑Sentiment Demo", layout="centered")
st.title("📊 Twitter‑Sentiment Classification Demo")

# Path input – you can keep the default or browse for another archive
default_zip = Path("C:/Users/zhaos/Downloads/archive (1).zip")
archive_path = st.text_input("Path to the zipped dataset:",
                             value=str(default_zip),
                             placeholder="C:/path/to/archive.zip")
archive_path = Path(archive_path)

# Train / load
if not archive_path.exists():
    st.error("❌ The archive path you provided does not exist.")
    st.stop()

with st.spinner("Training / loading model..."):
    vect, model, val_acc, val_report, cm_fig = train_model(archive_path)

st.success(f"✅ Model ready  •  Validation accuracy: **{val_acc:.3f}**")

# Show classification report & confusion matrix
with st.expander("See detailed validation metrics"):
    st.code(val_report)
    st.pyplot(cm_fig, clear_figure=True)

# ------------------------------------------------------------------
# 4. Live single‑tweet prediction
# ------------------------------------------------------------------
st.header("Try it out!")
tweet_text = st.text_area("Enter a tweet:", height=100,
                          placeholder="Type or paste a tweet here…")
if st.button("Predict sentiment"):
    if not tweet_text.strip():
        st.warning("Please enter some text.")
    else:
        X_new = vect.transform([clean_text(tweet_text)])
        pred  = model.predict(X_new)[0]
        probs = model.predict_proba(X_new)[0]
        st.write(f"**Prediction:** {pred}")
        st.write("**Class probabilities:**")
        prob_df = pd.DataFrame({"Sentiment": model.classes_, "Probability": probs})
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

# ------------------------------------------------------------------
# 5. Batch prediction on uploaded CSV
# ------------------------------------------------------------------
st.header("Batch prediction (optional)")
csv_file = st.file_uploader("Upload a CSV with a 'Content' column:", type="csv")
if csv_file:
    df_pred = pd.read_csv(csv_file)
    if "Content" not in df_pred.columns:
        st.error("CSV must contain a 'Content' column.")
    else:
        df_pred["Clean"]   = df_pred["Content"].apply(clean_text)
        df_pred["Sentiment_Pred"] = model.predict(vect.transform(df_pred["Clean"]))
        # Download link
        buf = io.BytesIO()
        df_pred.to_csv(buf, index=False)
        st.download_button(
            label="📥 Download predictions",
            data=buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv"
        )
        st.success(f"Predicted sentiment for {len(df_pred)} rows! ✅")
