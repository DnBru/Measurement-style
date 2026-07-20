"""
Correlation matrices and construct sanity checks.

Produces:
  - outputs/tables/pearson_correlations.csv
  - outputs/tables/spearman_correlations.csv
  - outputs/figures/heatmap_pearson.png
  - outputs/figures/heatmap_spearman.png
  - outputs/tables/construct_sanity_checks.csv
  - outputs/tables/top20_ambivalence_product.csv
  - outputs/tables/top20_ambivalence_min.csv
  - outputs/tables/word_examples.csv
    (near-neutral low-amb, near-neutral high-amb, high-PA/low-NA, low-val/high-NA)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, FIGURE, NEAR_NEUTRAL

np.random.seed(RANDOM_SEED)

CORR_VARIABLES = [
    COLS["valence"],          COLS["arousal"],
    COLS["happiness"],        COLS["anger"],
    COLS["fear"],             COLS["sadness"],
    COLS["disgust"],          COLS["surprise"],
    COLS["positive_affect"],  COLS["negative_affect"],
    COLS["ambivalence_prod"], COLS["ambivalence_min"],
]

DPI = FIGURE["dpi"]
FMT = FIGURE["fig_format"]


def compute_correlation_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compute Pearson or Spearman correlation matrix for available CORR_VARIABLES."""
    avail = [c for c in CORR_VARIABLES if c in df.columns]
    if method == "pearson":
        return df[avail].corr(method="pearson").round(4)
    elif method == "spearman":
        return df[avail].corr(method="spearman").round(4)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_heatmap(corr_matrix: pd.DataFrame, title: str, filename: str) -> None:
    """Save a styled correlation heatmap."""
    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.85), max(6, n * 0.75)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix, ax=ax,
        mask=mask,            # lower triangle only (reduce visual clutter)
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        vmin=-1.0, vmax=1.0,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Correlation coefficient"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=8)
    plt.tight_layout()
    path = os.path.join(PATHS["figures"], filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[05] Heatmap saved → {path}")


def build_sanity_check_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Answer the four key construct sanity questions with Pearson r and interpretation.
    """
    def safe_r(col_a: str, col_b: str) -> tuple[float, float]:
        if col_a not in df.columns or col_b not in df.columns:
            return np.nan, np.nan
        sub = df[[col_a, col_b]].dropna()
        r, p = scipy_stats.pearsonr(sub[col_a], sub[col_b])
        return round(r, 4), round(p, 6)

    def interpret_r(r: float) -> str:
        if np.isnan(r):      return "N/A (column missing)"
        if abs(r) >= 0.7:    return "Strong"
        elif abs(r) >= 0.4:  return "Moderate"
        elif abs(r) >= 0.2:  return "Weak"
        else:                return "Negligible"

    rows = []
    checks = [
        {
            "Question": "How strongly does PositiveAffect track Valence?",
            "Col_A": COLS["positive_affect"],
            "Col_B": COLS["valence"],
            "Expected": "Positive; if r ≈ 1, PA adds no new info over Valence.",
            "Implication": ("High r suggests PA largely tracks bipolar Valence. "
                            "Some divergence expected because PA is a discrete emotion, "
                            "not a bipolar continuum."),
        },
        {
            "Question": "How strongly does NegativeAffect track Valence?",
            "Col_A": COLS["negative_affect"],
            "Col_B": COLS["valence"],
            "Expected": "Negative; stronger negative r = more redundancy with Valence.",
            "Implication": ("High negative r means NA mirrors bipolar Valence. "
                            "Lower r suggests NA captures variance beyond Valence."),
        },
        {
            "Question": "Are the two ambivalence formulas highly similar?",
            "Col_A": COLS["ambivalence_prod"],
            "Col_B": COLS["ambivalence_min"],
            "Expected": "High r expected (both measure mixed affect); min is more conservative.",
            "Implication": ("If r ≥ 0.90, choice of formula matters little for ranking; "
                            "both can be reported as converging. If lower, Ambivalence_min "
                            "is systematically more conservative."),
        },
        {
            "Question": "Does unipolar-style representation provide info beyond bipolar Valence?",
            "Col_A": COLS["ambivalence_prod"],
            "Col_B": COLS["valence"],
            "Expected": "Should be low or near-zero: ambivalence peaks at midpoint of Valence.",
            "Implication": ("Low r confirms that Ambivalence is not simply a re-coding of "
                            "Valence - it captures a distinct construct (mixed affect). "
                            "This is the key empirical justification for the unipolar-style "
                            "decomposition in the thesis."),
        },
    ]

    for chk in checks:
        r, p = safe_r(chk["Col_A"], chk["Col_B"])
        rows.append({
            "Question":         chk["Question"],
            "Variable_A":       chk["Col_A"],
            "Variable_B":       chk["Col_B"],
            "Pearson_r":        r,
            "p_value":          p,
            "Strength":         interpret_r(r),
            "Direction":        "positive" if r > 0 else "negative" if r < 0 else "N/A",
            "Expected_pattern": chk["Expected"],
            "Thesis_implication": chk["Implication"],
        })

    return pd.DataFrame(rows)


def get_top_ambivalence_words(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    """Return top N words by an ambivalence column."""
    word_col = COLS["word"]
    keep_cols = [c for c in [word_col, COLS["valence"], COLS["positive_affect"],
                              COLS["negative_affect"], col] if c in df.columns]
    if col not in df.columns:
        print(f"  [WARNING] Column '{col}' not found - cannot compute top-{n} words.")
        return pd.DataFrame()
    return (df[keep_cols]
            .dropna(subset=[col])
            .nlargest(n, col)
            .reset_index(drop=True))


def get_word_examples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect illustrative word examples for four interpretable categories.
    Returns a combined DataFrame with a 'Category' column.
    """
    lo, hi   = NEAR_NEUTRAL["primary"]
    val_col  = COLS["valence"]
    pa_col   = COLS["positive_affect"]
    na_col   = COLS["negative_affect"]
    amb_col  = COLS["ambivalence_prod"]
    word_col = COLS["word"]
    n_examples = 10

    all_cols = [c for c in [word_col, val_col, pa_col, na_col, amb_col]
                if c in df.columns]

    results = []

    # 1. Near-neutral, low ambivalence
    if val_col in df.columns and amb_col in df.columns:
        near_neutral = df[(df[val_col] >= lo) & (df[val_col] <= hi)].copy()
        amb_median   = near_neutral[amb_col].median()
        low_amb_nn   = (near_neutral[near_neutral[amb_col] <= amb_median]
                        .nsmallest(n_examples, amb_col)[all_cols].copy())
        low_amb_nn["Category"] = "Near-neutral, Low Ambivalence"
        results.append(low_amb_nn)

        # 2. Near-neutral, high ambivalence
        high_amb_nn  = (near_neutral[near_neutral[amb_col] > amb_median]
                         .nlargest(n_examples, amb_col)[all_cols].copy())
        high_amb_nn["Category"] = "Near-neutral, High Ambivalence"
        results.append(high_amb_nn)

    # 3. High PA / Low NA
    if pa_col in df.columns and na_col in df.columns:
        pa_q75 = df[pa_col].quantile(0.75)
        na_q25 = df[na_col].quantile(0.25)
        high_pa_low_na = (df[(df[pa_col] >= pa_q75) & (df[na_col] <= na_q25)]
                          .nlargest(n_examples, pa_col)[all_cols].copy())
        high_pa_low_na["Category"] = "High PositiveAffect, Low NegativeAffect"
        results.append(high_pa_low_na)

    # 4. Low Valence / High NA
    if val_col in df.columns and na_col in df.columns:
        val_q25 = df[val_col].quantile(0.25)
        na_q75  = df[na_col].quantile(0.75)
        low_val_high_na = (df[(df[val_col] <= val_q25) & (df[na_col] >= na_q75)]
                           .nlargest(n_examples, na_col)[all_cols].copy())
        low_val_high_na["Category"] = "Low Valence, High NegativeAffect"
        results.append(low_val_high_na)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS["tables"],  exist_ok=True)
    os.makedirs(PATHS["figures"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[05] Loaded processed data: {df.shape[0]:,} rows")

    #  Correlation matrices 
    for method in ("pearson", "spearman"):
        corr  = compute_correlation_matrix(df, method)
        csv_p = os.path.join(PATHS["tables"], f"{method}_correlations.csv")
        corr.to_csv(csv_p)
        print(f"[05] {method.capitalize()} correlations → {csv_p}")
        plot_heatmap(
            corr,
            title=f"{method.capitalize()} Correlation Matrix - Affective Variables",
            filename=f"heatmap_{method}.{FMT}",
        )

    #  Sanity checks 
    sanity_df = build_sanity_check_table(df)
    sanity_path = os.path.join(PATHS["tables"], "construct_sanity_checks.csv")
    sanity_df.to_csv(sanity_path, index=False)
    print(f"[05] Sanity checks saved → {sanity_path}")
    print("\n[05] Construct Sanity Checks:")
    for _, row in sanity_df.iterrows():
        print(f"  Q: {row['Question']}")
        print(f"     r={row['Pearson_r']}, {row['Strength']}, p={row['p_value']}")
        print(f"     Implication: {row['Thesis_implication'][:80]}...")

    #  Top ambivalence words 
    for col_key, label in [("ambivalence_prod", "product"), ("ambivalence_min", "min")]:
        top_df = get_top_ambivalence_words(df, COLS[col_key], n=20)
        if not top_df.empty:
            p = os.path.join(PATHS["tables"], f"top20_ambivalence_{label}.csv")
            top_df.to_csv(p, index=False)
            print(f"[05] Top-20 ambivalence ({label}) → {p}")

    #  Word examples 
    examples_df = get_word_examples(df)
    if not examples_df.empty:
        examples_path = os.path.join(PATHS["tables"], "word_examples.csv")
        examples_df.to_csv(examples_path, index=False)
        print(f"[05] Word examples → {examples_path}")


if __name__ == "__main__":
    run()
