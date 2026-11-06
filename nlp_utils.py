"""Shared helpers for the Airline Tweet Sentiment Analyzer.

The notebook (NLP_PR1.ipynb) and the Streamlit app (app.py) both import this file,
so cleaning and the heuristic classifier behave exactly the same in both places.
"""
import html
import re

MAX_CHARS = 500   # longer input is truncated in the app


def clean_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Decode HTML entities (&amp; -> &) and normalise curly apostrophes
    text = html.unescape(text)
    text = text.replace("’", "'").replace("‘", "'")

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove @mentions
    text = re.sub(r'@\w+', ' ', text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Expand negative contractions so the negation survives punctuation removal
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)

    # Remove other contractions ('s, 'll, 're, 've, 'd, 'm)
    text = re.sub(r"'(?:s|ll|re|ve|d|m)\b", "", text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation, emoji and symbols
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Airline-specific sentiment lexicon
positive_words = {
    "thanks", "thank", "thx", "great", "love", "loved", "good", "best",
    "awesome", "amazing", "appreciate", "appreciated", "helpful", "nice",
    "excellent", "fantastic", "wonderful", "perfect", "friendly", "kind",
    "smooth", "comfortable", "easy", "happy", "glad", "enjoy", "enjoyed",
    "impressed", "outstanding", "superb", "professional", "pleasant",
    "brilliant", "favorite", "fabulous", "grateful"
}

negative_words = {
    "cancelled", "canceled", "cancel", "flightled", "delayed", "delay",
    "delays", "late", "lost", "worst", "rude", "terrible", "awful",
    "horrible", "bad", "worse", "hate", "hold", "waiting", "stuck",
    "broken", "disappointed", "disappointing", "unacceptable", "ridiculous",
    "mess", "nightmare", "fail", "failed", "refund", "missed", "damaged",
    "stranded", "unhappy", "angry", "frustrated", "frustrating", "useless",
    "hours", "hrs"
}

negation_words = {"not", "no", "never"}


def heuristic_score(text):

    words = clean_text(text).split()

    score = 0

    for i, word in enumerate(words):

        if word in positive_words:
            polarity = 1
        elif word in negative_words:
            polarity = -1
        else:
            continue

        # flip the polarity if one of the previous two words is a negation
        if any(w in negation_words for w in words[max(0, i-2):i]):
            polarity = -polarity

        score += polarity

    return score


def heuristic_sentiment(text):

    score = heuristic_score(text)

    if score > 0:
        return "positive"

    elif score < 0:
        return "negative"

    else:
        return "neutral"