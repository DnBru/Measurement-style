"""
Descriptive statistics for all affective variables.

Produces:
  - outputs/tables/descriptive_stats.csv
  - outputs/tables/descriptive_stats.md

Variables covered:
  Valence, Arousal, Happiness, Anger, Fear, Sadness, Disgust, Surprise,
  PositiveAffect, NegativeAffect, Ambivalence_product, Ambivalence_min

Statistics reported:
  N, mean, SD, min, max, median, IQR, skewness, kurtosis (excess)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED

np.random.seed(RANDOM_SEED)

STAT_VARIABLES = [
    COLS["valence"],          COLS["arousal"],
    COLS["happiness"],        COLS["anger"],
    COLS["fear"],             COLS["sadness"],
    COLS["disgust"],          COLS["surprise"],
    COLS["positive_affect"],  COLS["negative_affect"],
    COLS["ambivalence_prod"], COLS["ambivalence_min"],
]


def compute_descriptives(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Computes a full descriptive statistics table for a list of columns.
    Skips columns that are not in the DataFrame with a printed warning.
    """
    rows = []
    for col in variables:
        if col not in df.columns:
            print(f"  [WARNING] Column '{col}' not found - skipping descriptives for this variable.")
            continue
        s = df[col].dropna()
        n = len(s)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        row = {
            "Variable":  col,
            "N":         n,
            "Mean":      round(s.mean(), 4),
            "SD":        round(s.std(ddof=1), 4),
            "Min":       round(s.min(), 4),
            "Max":       round(s.max(), 4),
            "Median":    round(s.median(), 4),
            "Q1":        round(q1, 4),
            "Q3":        round(q3, 4),
            "IQR":       round(q3 - q1, 4),
            "Skewness":  round(scipy_stats.skew(s), 4),
            # Fisher's excess kurtosis (normal distribution = 0)
            "Kurtosis":  round(scipy_stats.kurtosis(s, fisher=True), 4),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convert a DataFrame to a Markdown table string."""
    lines = []
    if title:
        lines.append(f"# {title}\n")
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join(["::" if c not in ("Variable",) else ":"
                                 for c in df.columns]) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    """Main entry point."""
    os.makedirs(PATHS["tables"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(
                f"Processed dataset not found: {processed_path}\n"
                "Run 02_target_construction.py first."
            )
        df = pd.read_csv(processed_path)
        print(f"[03] Loaded processed data: {df.shape[0]:,} rows")

    desc_df = compute_descriptives(df, STAT_VARIABLES)

    # Save CSV
    csv_path = os.path.join(PATHS["tables"], "descriptive_stats.csv")
    desc_df.to_csv(csv_path, index=False)
    print(f"[03] Descriptive stats saved → {csv_path}")

    # Save Markdown
    md_path = os.path.join(PATHS["tables"], "descriptive_stats.md")
    md_content = df_to_markdown(
        desc_df,
        title="Descriptive Statistics - Affective Variables (Speed & Brysbaert, 2024)"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[03] Descriptive stats (MD) → {md_path}")

    # Print to console
    print("\n[03] Descriptive Statistics Table:")
    print(desc_df.to_string(index=False))

    return df


if __name__ == "__main__":
    run()
