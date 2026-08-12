"""
Customer Review Cleaning System
--------------------------------
Unit I | Skill Task 1: E-commerce review preprocessing
Dataset: IMDb 50K Reviews (works with any review-style text dataset)
Libraries: re, string, pandas, nltk

This script builds a reusable text-cleaning pipeline for raw customer
reviews (e.g. scraped from an e-commerce site or the IMDb dataset), so the
cleaned text can be fed into further NLP / ML steps (sentiment analysis,
topic modeling, etc.)

Pipeline steps:
1. Lowercase text
2. Remove HTML tags (common in scraped reviews, e.g. IMDb <br /> tags)
3. Remove URLs
4. Remove punctuation
5. Remove digits/numbers
6. Remove extra whitespace
7. Tokenize
8. Remove stopwords
9. Lemmatize
10. Rejoin into a clean string
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------
# 0. One-time NLTK downloads (safe to re-run; will skip if already present)
# ---------------------------------------------------------------------
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else
                        f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------
# 1. Core cleaning function
# ---------------------------------------------------------------------
def clean_review(text: str, remove_stopwords: bool = True,
                  lemmatize: bool = True) -> str:
    """
    Clean a single raw review string and return the processed text.

    Parameters
    ----------
    text : str
        Raw review text.
    remove_stopwords : bool
        Whether to strip common English stopwords.
    lemmatize : bool
        Whether to lemmatize tokens (e.g. "running" -> "run").

    Returns
    -------
    str
        Cleaned, normalized review text.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags (IMDb reviews often contain <br /><br />)
    text = re.sub(r"<.*?>", " ", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove email addresses (common in scraped e-commerce reviews)
    text = re.sub(r"\S+@\S+", " ", text)

    # 5. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 6. Remove digits/numbers
    text = re.sub(r"\d+", " ", text)

    # 7. Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Tokenize
    tokens = word_tokenize(text)

    # 9. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    # 10. Remove any leftover single-character tokens (noise)
    tokens = [t for t in tokens if len(t) > 1]

    # 11. Lemmatize
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------
# 2. Batch cleaning for a DataFrame column
# ---------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame, text_col: str = "review",
                     new_col: str = "cleaned_review") -> pd.DataFrame:
    """
    Apply clean_review() to an entire DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw review text.
    text_col : str
        Name of the column with raw review text.
    new_col : str
        Name of the new column to store cleaned text.

    Returns
    -------
    pd.DataFrame
        DataFrame with the added cleaned-text column.
    """
    df = df.copy()
    df[new_col] = df[text_col].astype(str).apply(clean_review)
    return df


# ---------------------------------------------------------------------
# 3. Demo / usage example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # --- Option A: quick single-string test ---
    sample_review = (
        "This product was AMAZING!!! <br /><br /> I bought it on "
        "2023-05-01 for $19.99 and it changed my life. "
        "Check it out at https://example.com or email us at test@shop.com. "
        "10/10 would recommend :)"
    )
    print("RAW:\n", sample_review)
    print("\nCLEANED:\n", clean_review(sample_review))

    # --- Option B: batch clean a small mock dataset ---
    # Replace this with: pd.read_csv("IMDB Dataset.csv")
    # (download from https://ai.stanford.edu/~amaas/data/sentiment/
    #  or the Kaggle "IMDb 50K Movie Reviews" dataset)
    mock_data = pd.DataFrame({
        "review": [
            "Great product, fast shipping!! <br />Would buy again 5/5",
            "Terrible experience... item arrived broken. Contact: help@shop.com",
            "It's okay, nothing special. Visit www.moreinfo.com for details.",
        ],
        "sentiment": ["positive", "negative", "neutral"]
    })

    cleaned_df = clean_dataframe(mock_data, text_col="review")
    print("\n\nBATCH CLEANING RESULT:")
    print(cleaned_df[["review", "cleaned_review"]].to_string(index=False))

    # Save cleaned output
    cleaned_df.to_csv("cleaned_reviews.csv", index=False)
    print("\nSaved cleaned data to cleaned_reviews.csv")
