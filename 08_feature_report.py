"""
Feature feasibility report for three representation families.

Documents (no actual model loading needed here):
  1. Character n-grams     - vocabulary size computed from dataset
  2. Static embeddings     - metadata report (requires manual OOV check)
  3. Transformer embeddings- metadata report (requires manual extraction)

Produces:
  - outputs/tables/feature_report.csv
  - outputs/tables/feature_report.md

NOTE: Computing actual n-gram vocabulary requires only sklearn and the words
in the dataset - no internet access needed. Static / transformer reports are
template-based documentation tables that you fill with actual values once
models are available.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, MODELS

np.random.seed(RANDOM_SEED)


def compute_ngram_vocabulary(words: list, ngram_range: tuple,
                              min_df: int, max_features: int) -> dict:
    """
    Fit a character-level TF-IDF vectorizer on the word list to determine
    vocabulary size and feature dimensionality.
    Uses analyzer='char_wb' (within-word character n-grams).
    """
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
        strip_accents=None,   # preserve Dutch diacritics
    )
    vec.fit(words)
    vocab_size = len(vec.vocabulary_)
    return {
        "analyzer":           "char_wb (within-word character n-grams)",
        "ngram_range":        f"{ngram_range[0]}–{ngram_range[1]}",
        "min_df":             min_df,
        "max_features_cap":   max_features,
        "actual_vocab_size":  vocab_size,
        "feature_dim":        vocab_size,
        "preprocessing":      "lowercase; Dutch diacritics preserved; no stemming",
        "note":               "Vocabulary fit on full word list in exploratory phase. "
                              "In final benchmark, fit inside each training fold only.",
    }


def build_feature_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a methods-oriented feature documentation table.
    The n-gram section is computed; static and transformer sections are
    template rows that you update once models are available.
    """
    word_col = COLS["word"]
    n_total  = len(df)

    #  1. Character n-grams 
    if word_col in df.columns:
        words = df[word_col].dropna().astype(str).tolist()
        ngram_info = compute_ngram_vocabulary(
            words,
            ngram_range  = MODELS["ngram_range"],
            min_df       = MODELS["ngram_min_df"],
            max_features = MODELS["ngram_max_features"],
        )
        ngram_row = {
            "Representation_family": "Character n-grams (TF-IDF)",
            "Model_source":          "scikit-learn TfidfVectorizer (char_wb)",
            "Dimensionality":        ngram_info["actual_vocab_size"],
            "Coverage":              f"{n_total}/{n_total} words (100%)",
            "OOV_handling":          "N/A - representation derived from character substrings",
            "Preprocessing":         ngram_info["preprocessing"],
            "Key_parameters":        (f"ngram_range={ngram_info['ngram_range']}, "
                                      f"min_df={ngram_info['min_df']}, "
                                      f"max_features≤{ngram_info['max_features_cap']}"),
            "Leakage_note":          ngram_info["note"],
            "Runtime_estimate":      "Fast (<60s for full dataset on modern CPU)",
            "Status":                "IMPLEMENTED - computed from dataset",
        }
    else:
        ngram_row = {"Representation_family": "Character n-grams",
                     "Status": f"SKIPPED - '{word_col}' column missing"}

    #  2. Static embeddings 
    static_row = {
        "Representation_family": "Static word embeddings",
        "Model_source":          "[TO FILL] e.g. fastText Dutch (cc.nl.300.vec) or "
                                 "COW-derived Dutch word2vec",
        "Dimensionality":        "[TO FILL] e.g. 300",
        "Coverage":              "[TO FILL] - compute as: n_words_with_embedding / n_total",
        "OOV_handling":          "[TO FILL] Recommended: zero vector or subword fallback "
                                 "(fastText supports subword). Document choice explicitly.",
        "Preprocessing":         "[TO FILL] lowercase matching; punctuation handling",
        "Key_parameters":        "None at inference time (pre-trained)",
        "Leakage_note":          "Static embeddings are trained on external corpora; "
                                 "no fitting on thesis dataset required. No leakage risk.",
        "Runtime_estimate":      "[TO FILL] e.g. ~5min to load + extract for 24k words",
        "Status":                "TEMPLATE - update when model is available",
    }

    #  3. Transformer embeddings 
    transformer_row = {
        "Representation_family": "Transformer-derived embeddings",
        "Model_source":          "[TO FILL] e.g. BERTje (GroNLP/bert-base-dutch-cased) "
                                 "or RobBERT (pdelobelle/robbert-v2-dutch-base)",
        "Dimensionality":        "[TO FILL] e.g. 768 (base models)",
        "Coverage":              "[TO FILL] - tokenizer may split unknown words into subtokens; "
                                 "report subtoken coverage separately",
        "OOV_handling":          "Handled via subword tokenization (WordPiece/BPE); "
                                 "truly unknown tokens use [UNK] embedding.",
        "Preprocessing":         "No manual lowercasing for cased models; "
                                 "use [CLS] token embedding or mean-pool over token embeddings.",
        "Key_parameters":        ("[TO FILL] pooling_strategy = mean/CLS; "
                                  "layer_choice = last hidden layer or layer average"),
        "Leakage_note":          "Pre-trained on external corpora; no fine-tuning in this "
                                 "project. Embedding extraction is inference-only - no leakage.",
        "Runtime_estimate":      "[TO FILL] e.g. ~20–40 min on CPU for 24k words; "
                                 "~3–5 min on GPU. Recommend caching embeddings to disk.",
        "Status":                "TEMPLATE - update when model is selected",
    }

    report_df = pd.DataFrame([ngram_row, static_row, transformer_row])
    return report_df, ngram_info if word_col in df.columns else {}


def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS["tables"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[08] Loaded processed data: {df.shape[0]:,} rows")

    report_df, ngram_info = build_feature_report(df)

    # Save CSV
    report_csv = os.path.join(PATHS["tables"], "feature_report.csv")
    report_df.to_csv(report_csv, index=False)
    print(f"[08] Feature report saved → {report_csv}")

    # Save Markdown with section headers
    report_md = os.path.join(PATHS["tables"], "feature_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Feature Feasibility Report\n\n")
        f.write("This document summarises the three representation families used in the "
                "thesis benchmark. Sections marked **[TO FILL]** require updating once "
                "the static/transformer models are selected.\n\n")
        for i, (_, row) in enumerate(report_df.iterrows(), 1):
            f.write(f"## {i}. {row['Representation_family']}\n\n")
            for col, val in row.items():
                if col != "Representation_family":
                    f.write(f"- **{col}**: {val}\n")
            f.write("\n")

    print(f"[08] Feature report (MD)   → {report_md}")

    if ngram_info:
        print("\n[08] Character n-gram vocabulary statistics:")
        for k, v in ngram_info.items():
            print(f"  {k:<30}: {v}")


if __name__ == "__main__":
    run()
