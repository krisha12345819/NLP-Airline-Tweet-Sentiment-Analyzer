"""
SkyPulse - Airline Tweet Sentiment Analyzer  (PR 1, Task 8)
Run with:  streamlit run app.py
"""
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

from nlp_utils import MAX_CHARS, clean_text, heuristic_score, heuristic_sentiment

st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"

APPROACHES = [
    "Heuristic (rules)",
    "VADER (ready-made)",
    "ML model (TF-IDF + Logistic Regression)",
]

EXAMPLES = {
    "😊 Positive example": "@united thanks for the great service, the crew was awesome!",
    "😠 Negative example": "@delta my flight was cancelled and I am not happy with the delay. Worst airline!!!",
    "😐 Neutral example": "@SouthwestAir what is the baggage allowance for international flights?",
}

LOOK = {
    "positive": {"emoji": "😊", "color": "#17a673", "grad": "linear-gradient(135deg,#0f9b6e,#34d399)"},
    "neutral":  {"emoji": "😐", "color": "#d98e04", "grad": "linear-gradient(135deg,#d98e04,#fbbf24)"},
    "negative": {"emoji": "😠", "color": "#d6455d", "grad": "linear-gradient(135deg,#c81e4a,#fb7185)"},
}
CLASS_ORDER = ["negative", "neutral", "positive"]

POS_MEANING = {
    "NN": "Noun", "NNS": "Noun (plural)", "NNP": "Proper noun", "VB": "Verb (base)",
    "VBD": "Verb (past)", "VBG": "Verb (-ing)", "VBN": "Verb (past part.)",
    "VBZ": "Verb (3rd person)", "VBP": "Verb (present)", "JJ": "Adjective",
    "JJR": "Adjective (compar.)", "JJS": "Adjective (superl.)", "RB": "Adverb",
    "PRP": "Pronoun", "PRP$": "Possessive pronoun", "IN": "Preposition",
    "DT": "Determiner", "CC": "Conjunction", "TO": "'to'", "MD": "Modal verb",
    "CD": "Number", "UH": "Interjection", "WRB": "Wh-adverb", "WP": "Wh-pronoun",
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; max-width: 860px; }

    .hero {
        position: relative; overflow: hidden; text-align: center; color: #fff;
        padding: 2.4rem 1.5rem 2.1rem; border-radius: 24px; margin-bottom: 1.3rem;
        background: linear-gradient(135deg, #0b1f4d 0%, #1565c0 55%, #00b4d8 100%);
        box-shadow: 0 14px 36px rgba(21,101,192,.32);
    }
    .hero::before {
        content: ""; position: absolute; inset: 0; opacity: .18;
        background:
          radial-gradient(circle at 15% 25%, #fff 0 2px, transparent 3px),
          radial-gradient(circle at 80% 18%, #fff 0 2px, transparent 3px),
          radial-gradient(circle at 62% 78%, #fff 0 2px, transparent 3px),
          radial-gradient(circle at 30% 85%, #fff 0 1.5px, transparent 3px),
          radial-gradient(circle at 92% 70%, #fff 0 1.5px, transparent 3px);
    }
    .hero h1 { margin: 0; font-size: 2.45rem; font-weight: 700; letter-spacing: .4px; position: relative; }
    .hero p  { margin: .55rem auto 0; font-size: 1.02rem; opacity: .95; max-width: 560px; position: relative; }
    .pill {
        display: inline-block; margin: .9rem .2rem 0; padding: .24rem .85rem; position: relative;
        background: rgba(255,255,255,.18); border-radius: 999px; font-size: .76rem;
        border: 1px solid rgba(255,255,255,.28);
    }

    .step { font-weight: 600; font-size: 1.05rem; margin: 1.1rem 0 .3rem; }
    .step span {
        display: inline-flex; align-items: center; justify-content: center; width: 1.6rem; height: 1.6rem;
        border-radius: 50%; background: #1565c0; color: #fff; font-size: .8rem; margin-right: .5rem;
    }

    .result-card {
        border-radius: 20px; padding: 1.4rem 1.2rem; text-align: center; color: #fff;
        margin: .4rem 0 .6rem; animation: pop .4s ease-out;
        box-shadow: 0 10px 26px rgba(0,0,0,.18);
    }
    .result-card .emoji { font-size: 3.2rem; line-height: 1; }
    .result-card .label { font-size: 1.9rem; font-weight: 700; margin-top: .25rem; letter-spacing: 1px; }
    .result-card .sub   { font-size: .98rem; opacity: .95; }
    @keyframes pop { from {transform: scale(.94); opacity: 0;} to {transform: scale(1); opacity: 1;} }

    .mini {
        border-radius: 16px; padding: .9rem .8rem; text-align: center; color: #fff;
        box-shadow: 0 6px 16px rgba(0,0,0,.14);
    }
    .mini .t { font-size: .74rem; opacity: .92; text-transform: uppercase; letter-spacing: .8px; }
    .mini .v { font-size: 1.2rem; font-weight: 700; margin-top: .15rem; }
    .mini .d { font-size: .72rem; opacity: .9; }
    .mini.off { background: rgba(120,120,140,.25); color: inherit; box-shadow: none; }

    .stButton > button[kind="primary"] {
        width: 100%; border: 0; border-radius: 14px; padding: .7rem 1rem; font-weight: 600; font-size: 1.05rem;
        background: linear-gradient(90deg, #1565c0, #00b4d8); color: #fff;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(21,101,192,.4); }
    .stButton > button[kind="secondary"] { border-radius: 12px; }
    .stTextArea textarea { border-radius: 14px; font-size: 1rem; }
    div[data-testid="stMetric"] {
        background: rgba(21,101,192,.08); border: 1px solid rgba(21,101,192,.2);
        padding: .6rem .9rem; border-radius: 14px;
    }
    .foot { text-align: center; opacity: .6; font-size: .8rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Start-up helpers (cached, so they run only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    """Download the NLTK data the app needs; never crash if the download fails."""
    for package in ("vader_lexicon", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    return True


@st.cache_resource
def load_model():
    """Load the saved scikit-learn pipeline (vectoriser + classifier) only once."""
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()      # raises LookupError if the lexicon is missing


def vader_label(compound):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def set_example(text):
    st.session_state["tweet"] = text


setup_nltk()


# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def predict_heuristic(raw):
    score = heuristic_score(raw)
    return heuristic_sentiment(raw), f"Lexicon score = {score:+d} (positive words minus negative words, negation handled)", None


def predict_vader(raw):
    scores = load_vader().polarity_scores(raw)          # VADER works best on the raw text
    detail = (f"VADER compound = {scores['compound']:+.3f}  "
              f"(neg {scores['neg']}, neu {scores['neu']}, pos {scores['pos']})")
    return vader_label(scores["compound"]), detail, None


def predict_ml(cleaned):
    model = load_model()
    proba = model.predict_proba([cleaned])[0]
    label = model.classes_[proba.argmax()]
    probs = pd.Series(proba, index=model.classes_, name="probability")
    probs = probs.reindex([c for c in CLASS_ORDER if c in probs.index])
    return label, f"Confidence = {proba.max():.1%}", probs


def top_words(model, cleaned, label, k=5):
    """Words (or word pairs) that pushed the ML prediction towards the predicted class the most."""
    vec, clf = model.steps[0][1], model.steps[-1][1]
    if not hasattr(clf, "coef_") or not hasattr(vec, "get_feature_names_out"):
        return None
    row = vec.transform([cleaned])
    idx = row.nonzero()[1]
    if len(idx) == 0:
        return None
    classes = list(clf.classes_)
    coef = clf.coef_[classes.index(label)] if len(classes) > 2 else clf.coef_[0]
    contrib = row[0, idx].toarray()[0] * coef[idx]
    names = np.asarray(vec.get_feature_names_out())[idx]
    order = np.argsort(contrib)[::-1][:k]
    out = pd.Series(contrib[order], index=names[order], name="influence")
    out = out[out > 0]
    return out if len(out) else None


def result_card(label, sub):
    look = LOOK[label]
    st.markdown(
        f"""<div class="result-card" style="background:{look['grad']}">
              <div class="emoji">{look['emoji']}</div>
              <div class="label">{label.upper()}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    message = f"Predicted sentiment: {label.upper()} {look['emoji']}"
    {"positive": st.success, "neutral": st.warning, "negative": st.error}[label](message)


def mini_card(title, label, detail):
    if label is None:
        st.markdown(f'<div class="mini off"><div class="t">{title}</div><div class="v">n/a</div>'
                    f'<div class="d">{detail}</div></div>', unsafe_allow_html=True)
        return
    look = LOOK[label]
    st.markdown(f'<div class="mini" style="background:{look["grad"]}"><div class="t">{title}</div>'
                f'<div class="v">{look["emoji"]} {label.title()}</div><div class="d">{detail}</div></div>',
                unsafe_allow_html=True)


def step(n, text):
    st.markdown(f'<div class="step"><span>{n}</span>{text}</div>', unsafe_allow_html=True)


def run_prediction(raw, approach):
    # 1. empty input
    if not raw or not raw.strip():
        st.warning("Please type a tweet or click one of the examples first.")
        return

    # 2. very long input
    if len(raw) > MAX_CHARS:
        raw = raw[:MAX_CHARS]
        st.info(f"The text was longer than {MAX_CHARS} characters, so only the first {MAX_CHARS} are analysed.")

    # 3. cleaned text (same function as in the notebook)
    cleaned = clean_text(raw)
    step(1, "Cleaned text")
    if not cleaned:
        st.warning(
            "Nothing is left after cleaning. The text contains only emojis, symbols, numbers or a link. "
            "Please write some words."
        )
        return
    st.code(cleaned, language=None)

    # 4. prediction with the chosen approach
    step(2, "Prediction")
    probs = None
    if approach == APPROACHES[0]:
        label, detail, _ = predict_heuristic(raw)
    elif approach == APPROACHES[1]:
        try:
            label, detail, _ = predict_vader(raw)
        except LookupError:
            st.error("The VADER lexicon could not be downloaded. Please choose another approach.")
            return
    else:
        try:
            label, detail, probs = predict_ml(cleaned)
        except Exception as error:
            st.error(f"The model file could not be loaded ({error}). "
                     "Check that sentiment_model.joblib is next to app.py and that the scikit-learn "
                     "version in requirements.txt matches the one used for training.")
            return

    result_card(label, detail)

    # 5. probability chart (ML model only)
    if probs is not None:
        step(3, "Class probabilities")
        st.bar_chart(probs, color="#1565c0", height=240)
        words = None
        try:
            words = top_words(load_model(), cleaned, label)
        except Exception:
            pass
        if words is not None:
            with st.expander("🔑 Which words pushed the prediction?"):
                st.bar_chart(words, color=LOOK[label]["color"], height=220)
                st.caption("Influence = TF-IDF weight x model coefficient for the predicted class.")

    # 6. compare all three approaches
    with st.expander("⚖️ Compare all three approaches on this tweet"):
        c1, c2, c3 = st.columns(3)
        h_label, h_detail, _ = predict_heuristic(raw)
        with c1:
            mini_card("Heuristic", h_label, h_detail.split(" (")[0])
        with c2:
            try:
                v_label, v_detail, _ = predict_vader(raw)
                mini_card("VADER", v_label, v_detail.split("  (")[0])
            except LookupError:
                mini_card("VADER", None, "lexicon unavailable")
        with c3:
            try:
                m_label, m_detail, _ = predict_ml(cleaned)
                mini_card("ML model", m_label, m_detail)
            except Exception:
                mini_card("ML model", None, "model unavailable")
        st.caption("When the three disagree, the trained ML model is usually the most reliable one (see the sidebar).")

    # 7. POS tags
    with st.expander("🧩 Show the POS tags of this tweet"):
        try:
            tags = nltk.pos_tag(cleaned.split())
            df = pd.DataFrame(tags, columns=["Word", "POS tag"])
            df["Meaning"] = df["POS tag"].map(POS_MEANING).fillna("Other")
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption("Tagged with NLTK's averaged-perceptron tagger. It was trained on news text, "
                       "so informal tweets may contain mistakes.")
        except LookupError:
            st.info("The POS tagger data is not available in this environment.")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ About this project")
    st.write(
        "The final step of an NLP practical: raw tweets are cleaned with RegEx, "
        "converted into features and classified with three different approaches."
    )
    st.markdown(
        "**Dataset:** [Twitter US Airline Sentiment (Kaggle)]"
        "(https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) "
        "- about 14,600 real tweets about six US airlines."
    )
    st.markdown("### 🧠 The three approaches")
    st.markdown(
        "- **Heuristic** - my own airline word lists + a negation rule\n"
        "- **VADER** - ready-made sentiment engine, no training\n"
        "- **ML model** - TF-IDF (1-2 grams) + Logistic Regression trained on the tweets"
    )
    st.markdown("### 📊 Test-set results")
    st.dataframe(
        pd.DataFrame(
            {"Accuracy": ["0.559", "0.549", "0.800"], "Macro-F1": ["0.572", "0.515", "0.753"]},
            index=["Heuristic", "VADER", "ML model"],
        ),
        width="stretch",
    )
    st.caption(
        "The data is imbalanced (about 63% negative), so macro-F1 is the main metric "
        "and the neutral class is the hardest to predict."
    )

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>✈️ Airline Tweet Sentiment Analyzer</h1>
        <p>Type a tweet about an airline and see whether it is <b>positive</b>, <b>neutral</b> or <b>negative</b>.</p>
        <span class="pill">RegEx cleaning</span><span class="pill">VADER</span>
        <span class="pill">TF-IDF + Logistic Regression</span><span class="pill">POS tagging</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "tweet" not in st.session_state:
    st.session_state["tweet"] = ""

st.markdown("**Try an example**")
cols = st.columns(3)
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(text,), width="stretch")

st.text_area("Tweet text", key="tweet", height=130, placeholder="Type or paste a tweet here ...")
st.caption(f"{len(st.session_state['tweet'])} / {MAX_CHARS} characters")

approach = st.radio("Choose an approach", APPROACHES, horizontal=True)

if st.button("Predict", type="primary"):
    run_prediction(st.session_state["tweet"], approach)

st.markdown(
    '<div class="foot">Made with ❤️ using Streamlit &amp; scikit-learn - NLP Practical Report 1 - Red &amp; White Skill Education</div>',
    unsafe_allow_html=True,
)
