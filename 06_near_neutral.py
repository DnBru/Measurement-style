"""
Near-neutral word analysis and threshold sensitivity.

Produces:
  - outputs/tables/near_neutral_summary.csv
  - outputs/tables/near_neutral_summary.md
  - outputs/tables/threshold_sensitivity.csv
  - outputs/figures/scatter_valence_ambivalence_annotated.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, NEAR_NEUTRAL, FIGURE, SUBGROUPS

np.random.seed(RANDOM_SEED)
DPI = FIGURE["dpi"]
FMT = FIGURE["fig_format"]


def analyze_near_neutral(df: pd.DataFrame, lo: float, hi: float,
                          label: str) -> dict:
    """
    Compute summary statistics for words within a given Valence band.
    """
    val_col  = COLS["valence"]
    pa_col   = COLS["positive_affect"]
    na_col   = COLS["negative_affect"]
    ap_col   = COLS["ambivalence_prod"]
    am_col   = COLS["ambivalence_min"]

    if val_col not in df.columns:
        return {"threshold_label": label, "error": f"'{val_col}' column missing"}

    mask = (df[val_col] >= lo) & (df[val_col] <= hi)
    sub  = df[mask]
    n_total = len(df)
    n_sub   = len(sub)

    # High-ambivalence thresholds (Q3 of the full dataset)
    ap_q3 = df[ap_col].quantile(SUBGROUPS["high_ambivalence_quantile"]) if ap_col in df.columns else np.nan
    am_q3 = df[am_col].quantile(SUBGROUPS["high_ambivalence_quantile"]) if am_col in df.columns else np.nan

    def mean_col(col):
        return round(sub[col].mean(), 4) if col in sub.columns else np.nan

    def pct_above(col, threshold):
        if col not in sub.columns or np.isnan(threshold):
            return np.nan
        return round((sub[col] > threshold).mean() * 100, 2)

    return {
        "threshold_label":              label,
        "valence_range":                f"[{lo}, {hi}]",
        "n_near_neutral":               n_sub,
        "pct_of_total":                 round(n_sub / n_total * 100, 2),
        "mean_PositiveAffect":          mean_col(pa_col),
        "mean_NegativeAffect":          mean_col(na_col),
        "mean_Ambivalence_product":     mean_col(ap_col),
        "mean_Ambivalence_min":         mean_col(am_col),
        "pct_high_ambivalence_product": pct_above(ap_col, ap_q3),
        "pct_high_ambivalence_min":     pct_above(am_col, am_q3),
        "high_amb_product_threshold_Q3": round(ap_q3, 4) if not np.isnan(ap_q3) else np.nan,
        "high_amb_min_threshold_Q3":    round(am_q3, 4) if not np.isnan(am_q3) else np.nan,
    }


def run_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare near-neutral analysis across three threshold definitions.
    Returns a DataFrame summarising each threshold condition.
    """
    thresholds = [
        ("Primary [2.75, 3.25]", *NEAR_NEUTRAL["primary"]),
        ("Narrow  [2.90, 3.10]", *NEAR_NEUTRAL["narrow"]),
        ("Wide    [2.50, 3.50]", *NEAR_NEUTRAL["wide"]),
    ]
    rows = []
    for label, lo, hi in thresholds:
        row = analyze_near_neutral(df, lo, hi, label)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_valence_ambivalence(df: pd.DataFrame) -> None:
    """
    Scatter plot: Valence (x) vs Ambivalence_product (y).
    Highlights the near-neutral band as a vertical shaded region.
    """
    val_col = COLS["valence"]
    amb_col = COLS["ambivalence_prod"]
    if val_col not in df.columns or amb_col not in df.columns:
        print(f"  [WARNING] Cannot create scatter - column missing.")
        return

    lo, hi = NEAR_NEUTRAL["primary"]
    rng    = np.random.default_rng(RANDOM_SEED)
    idx    = rng.choice(len(df), size=min(6000, len(df)), replace=False)
    sub    = df.iloc[idx][[val_col, amb_col]].dropna()

    near_mask = (sub[val_col] >= lo) & (sub[val_col] <= hi)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub.loc[~near_mask, val_col], sub.loc[~near_mask, amb_col],
               alpha=0.2, s=8, color="#4C72B0", label="Other words")
    ax.scatter(sub.loc[near_mask, val_col], sub.loc[near_mask, amb_col],
               alpha=0.5, s=14, color=FIGURE["neutral_highlight_color"],
               label=f"Near-neutral [{lo},{hi}]")
    ax.axvspan(lo, hi, alpha=FIGURE["neutral_highlight_alpha"],
               color=FIGURE["neutral_highlight_color"], zorder=0)

    ax.set_xlabel("Valence", fontsize=12)
    ax.set_ylabel("Ambivalence (product)", fontsize=12)
    ax.set_title("Valence vs Ambivalence - Near-neutral Band Highlighted",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(PATHS["figures"],
                        f"scatter_valence_ambivalence_annotated.{FMT}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[06] Annotated scatter saved → {path}")


def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS["tables"],  exist_ok=True)
    os.makedirs(PATHS["figures"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[06] Loaded processed data: {df.shape[0]:,} rows")

    # Primary analysis
    lo_p, hi_p = NEAR_NEUTRAL["primary"]
    primary    = analyze_near_neutral(df, lo_p, hi_p, "Primary [2.75, 3.25]")
    print("\n[06] Near-Neutral Analysis (Primary Threshold):")
    for k, v in primary.items():
        print(f"  {k:<40}: {v}")

    # Threshold sensitivity
    sensitivity_df = run_threshold_sensitivity(df)
    sens_csv = os.path.join(PATHS["tables"], "threshold_sensitivity.csv")
    sensitivity_df.to_csv(sens_csv, index=False)
    print(f"\n[06] Threshold sensitivity table → {sens_csv}")
    print(sensitivity_df.to_string(index=False))

    # Check sensitivity: do conclusions change substantially?
    ns   = sensitivity_df["n_near_neutral"].values
    pcts = sensitivity_df["pct_of_total"].values
    pct_high_amb = sensitivity_df["pct_high_ambivalence_product"].values
    print("\n[06] Sensitivity summary:")
    print(f"  N near-neutral across thresholds: {ns[0]} / {ns[1]} / {ns[2]} "
          f"(narrow→primary→wide)")
    print(f"  % high-ambivalence (product) across thresholds: "
          f"{pct_high_amb[0]}% / {pct_high_amb[1]}% / {pct_high_amb[2]}%")
    max_diff = abs(pct_high_amb.max() - pct_high_amb.min())
    if max_diff <= 5.0:
        print(f"  → Conclusions are ROBUST to threshold choice (max diff = {max_diff:.1f}%)")
    else:
        print(f"  → THRESHOLD MATTERS: diff = {max_diff:.1f}%; discuss in thesis limitations.")

    # Save primary analysis as single-row CSV
    summary_df  = pd.DataFrame([primary])
    summary_csv = os.path.join(PATHS["tables"], "near_neutral_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Markdown version
    summary_md  = os.path.join(PATHS["tables"], "near_neutral_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Near-Neutral Analysis - Primary Threshold\n\n")
        for k, v in primary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n\n## Threshold Sensitivity\n\n")
        header = "| " + " | ".join(sensitivity_df.columns) + " |"
        sep    = "| " + " | ".join([""] * len(sensitivity_df.columns)) + " |"
        rows   = ["| " + " | ".join(str(v) for v in row) + " |"
                  for _, row in sensitivity_df.iterrows()]
        f.write("\n".join([header, sep] + rows))
    print(f"[06] Near-neutral summary (MD) → {summary_md}")

    # Figure
    plot_valence_ambivalence(df)


if __name__ == "__main__":
    run()
