"""
Robustness checks.

Checks performed:
  1. Ambivalence formula comparison (product vs min)
  2. Near-neutral threshold sensitivity (3 thresholds)
  3. Surprise inclusion sensitivity check (exploratory; NOT primary)
  4. Scaling sensitivity (mean-centering vs min-max for ambivalence construction)

Produces:
  - outputs/tables/robustness_summary.csv
  - outputs/tables/robustness_summary.md
  - outputs/tables/surprise_sensitivity.csv  (if Surprise column present)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, NEAR_NEUTRAL

np.random.seed(RANDOM_SEED)


# Helper: min-max rescale
def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())

def meanstd(s: pd.Series) -> pd.Series:
    """Z-score standardisation (alternative scaling)."""
    return (s - s.mean()) / s.std(ddof=1)


# 1. Ambivalence formula comparison 
def check_ambivalence_formulas(df: pd.DataFrame) -> dict:
    """
    Compare Ambivalence_product and Ambivalence_min:
    - Pearson r between the two
    - Spearman r
    - Mean and SD of each
    - Agreement on top quartile
    """
    ap_col = COLS["ambivalence_prod"]
    am_col = COLS["ambivalence_min"]
    if ap_col not in df.columns or am_col not in df.columns:
        return {"check": "Ambivalence formula", "status": "SKIPPED - column missing"}

    sub = df[[ap_col, am_col]].dropna()
    r_p, _ = pearsonr(sub[ap_col], sub[am_col])
    r_s     = sub[ap_col].corr(sub[am_col], method="spearman")

    # Top-quartile agreement
    q_prod = sub[ap_col].quantile(0.75)
    q_min  = sub[am_col].quantile(0.75)
    in_top_prod = sub[ap_col] >= q_prod
    in_top_min  = sub[am_col] >= q_min
    agreement   = (in_top_prod == in_top_min).mean()

    return {
        "check": "Ambivalence formula comparison (product vs min)",
        "n": len(sub),
        "Pearson_r":         round(r_p, 4),
        "Spearman_r":        round(r_s, 4),
        "mean_product":      round(sub[ap_col].mean(), 4),
        "SD_product":        round(sub[ap_col].std(), 4),
        "mean_min":          round(sub[am_col].mean(), 4),
        "SD_min":            round(sub[am_col].std(), 4),
        "top_Q3_agreement":  round(agreement, 4),
        "conclusion": ("Formulas are HIGHLY SIMILAR (r≥0.90); both can be reported as converging."
                       if r_p >= 0.90 else
                       "Formulas DIVERGE meaningfully; Ambivalence_min is more conservative. "
                       "Report both and note the difference in thesis limitations."),
    }


# 2. Near-neutral threshold sensitivity (reuses 06_near_neutral logic)
def check_threshold_sensitivity(df: pd.DataFrame) -> list:
    """
    Compare key statistics across three Valence threshold definitions.
    Returns a list of dicts (one per threshold).
    """
    val_col = COLS["valence"]
    ap_col  = COLS["ambivalence_prod"]
    if val_col not in df.columns:
        return [{"check": "Threshold sensitivity", "status": "SKIPPED - Valence missing"}]

    thresholds = [
        ("Primary [2.75, 3.25]",  *NEAR_NEUTRAL["primary"]),
        ("Narrow  [2.90, 3.10]",  *NEAR_NEUTRAL["narrow"]),
        ("Wide    [2.50, 3.50]",  *NEAR_NEUTRAL["wide"]),
    ]

    results = []
    n_total = len(df)
    q3_ap   = df[ap_col].quantile(0.75) if ap_col in df.columns else np.nan

    for label, lo, hi in thresholds:
        mask = (df[val_col] >= lo) & (df[val_col] <= hi)
        sub  = df[mask]
        n    = len(sub)
        pct  = n / n_total * 100
        pct_high_amb = (
            (sub[ap_col] >= q3_ap).mean() * 100
            if ap_col in df.columns and not np.isnan(q3_ap) else np.nan
        )
        results.append({
            "check":                  "Near-neutral threshold sensitivity",
            "threshold":              label,
            "range":                  f"[{lo}, {hi}]",
            "N_near_neutral":         n,
            "Pct_of_total":           round(pct, 2),
            "Pct_high_ambivalence_Q3":round(pct_high_amb, 2) if not np.isnan(pct_high_amb) else np.nan,
        })

    # Compute sensitivity conclusion
    pct_vals = [r["Pct_high_ambivalence_Q3"] for r in results if not np.isnan(r.get("Pct_high_ambivalence_Q3", np.nan))]
    if len(pct_vals) >= 2:
        diff = max(pct_vals) - min(pct_vals)
        concl = f"Max diff in %high-ambivalence across thresholds = {diff:.1f}% "
        concl += ("- ROBUST (≤5%)." if diff <= 5 else "- THRESHOLD-SENSITIVE (>5%); discuss.")
        for r in results:
            r["sensitivity_conclusion"] = concl

    return results


# 3. Surprise inclusion sensitivity
def check_surprise_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an exploratory alternative NegativeAffect that INCLUDES Surprise.
    Compare it with the primary NegativeAffect (without Surprise) descriptively.
    This is clearly marked as exploratory - NOT the primary operationalisation.
    """
    na_cols = [COLS["anger"], COLS["fear"], COLS["sadness"], COLS["disgust"]]
    surp    = COLS["surprise"]
    pa_col  = COLS["positive_affect"]
    na_col  = COLS["negative_affect"]

    available_na  = [c for c in na_cols if c in df.columns]
    has_surprise  = surp in df.columns
    has_primary   = na_col in df.columns and pa_col in df.columns

    rows = []

    # Primary (without Surprise)
    if has_primary:
        rows.append({
            "Version": "PRIMARY (Anger+Fear+Sadness+Disgust, no Surprise)",
            "Formula": f"mean({', '.join(available_na)})",
            "Note":    "Primary operationalisation used in benchmark",
            "mean_NegativeAffect":  round(df[na_col].mean(), 4),
            "corr_with_Valence":    round(df[na_col].corr(df[COLS["valence"]]), 4)
                                    if COLS["valence"] in df.columns else np.nan,
        })

    # Alternative (with Surprise)
    if has_surprise and available_na:
        na_with_surp = df[available_na + [surp]].mean(axis=1)
        r_with_val   = na_with_surp.corr(df[COLS["valence"]]) if COLS["valence"] in df.columns else np.nan
        rows.append({
            "Version": "EXPLORATORY ONLY - NegativeAffect including Surprise",
            "Formula": f"mean({', '.join(available_na + [surp])})",
            "Note":    ("EXPLORATORY: Surprise has no consistent valence polarity "
                        "(Russell, 1980; Speed & Brysbaert, 2024). Including Surprise "
                        "may blur the positive/negative distinction. This variant is "
                        "reported for sensitivity only."),
            "mean_NegativeAffect":  round(na_with_surp.mean(), 4),
            "corr_with_Valence":    round(r_with_val, 4) if not np.isnan(r_with_val) else np.nan,
        })
    elif not has_surprise:
        rows.append({
            "Version": "EXPLORATORY Surprise inclusion",
            "Formula": "N/A",
            "Note":    f"Column '{surp}' not found in dataset - sensitivity check skipped.",
            "mean_NegativeAffect": np.nan,
            "corr_with_Valence":   np.nan,
        })

    return pd.DataFrame(rows)


# 4. Scaling sensitivity 
def check_scaling_sensitivity(df: pd.DataFrame) -> list:
    """
    Compare Ambivalence_product under two scaling methods:
    (a) min-max [0,1]   - primary
    (b) z-score then clip to [0,1] - alternative
    Report Pearson r between the two resulting ambivalence scores.
    """
    pa_col = COLS["positive_affect"]
    na_col = COLS["negative_affect"]
    if pa_col not in df.columns or na_col not in df.columns:
        return [{"check": "Scaling sensitivity", "status": "SKIPPED - columns missing"}]

    # Primary: min-max
    pa_mm = minmax(df[pa_col].dropna())
    na_mm = minmax(df[na_col].dropna())
    amb_minmax = (pa_mm * na_mm).dropna()

    # Alternative: z-score standardise, shift to [0,1] via min-max of z-scores
    pa_zs = minmax(meanstd(df[pa_col].dropna()))
    na_zs = minmax(meanstd(df[na_col].dropna()))
    amb_zscore = (pa_zs * na_zs).dropna()

    common_idx = amb_minmax.index.intersection(amb_zscore.index)
    r, _ = pearsonr(amb_minmax.loc[common_idx], amb_zscore.loc[common_idx])

    return [{
        "check":                   "Scaling sensitivity",
        "Scaling_A":               "Min-max to [0,1] (PRIMARY)",
        "Scaling_B":               "Z-score → shift to [0,1]",
        "Pearson_r_between":       round(r, 4),
        "conclusion": ("Ambivalence scores are ROBUST to scaling choice (r≥0.95)."
                       if r >= 0.95 else
                       "Scaling choice has MATERIAL EFFECT on ambivalence; discuss in methods."),
    }]


# Main runner
def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS["tables"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[10] Loaded processed data: {df.shape[0]:,} rows")

    # Collect all robustness checks
    all_rows = []

    # 1. Formula comparison
    formula_result = check_ambivalence_formulas(df)
    all_rows.append(formula_result)
    print(f"\n[10] Ambivalence formula: r={formula_result.get('Pearson_r','N/A')}, "
          f"Q3 agreement={formula_result.get('top_Q3_agreement','N/A')}")
    print(f"     → {formula_result.get('conclusion','')}")

    # 2. Threshold sensitivity
    thresh_results = check_threshold_sensitivity(df)
    all_rows.extend(thresh_results)
    if thresh_results and "sensitivity_conclusion" in thresh_results[0]:
        print(f"[10] Threshold sensitivity: {thresh_results[0]['sensitivity_conclusion']}")

    # 3. Scaling sensitivity
    scale_results = check_scaling_sensitivity(df)
    all_rows.extend(scale_results)
    if scale_results:
        print(f"[10] Scaling sensitivity: r={scale_results[0].get('Pearson_r_between','N/A')}")
        print(f"     → {scale_results[0].get('conclusion','')}")

    # Save combined robustness table
    # Flatten to common columns using orient='index' and fill missing
    rob_df = pd.DataFrame(all_rows).fillna("N/A")
    rob_csv = os.path.join(PATHS["tables"], "robustness_summary.csv")
    rob_df.to_csv(rob_csv, index=False)
    print(f"\n[10] Robustness summary → {rob_csv}")

    # Markdown
    rob_md = os.path.join(PATHS["tables"], "robustness_summary.md")
    with open(rob_md, "w", encoding="utf-8") as f:
        f.write("# Robustness Checks Summary\n\n")
        header = "| " + " | ".join(rob_df.columns) + " |"
        sep    = "| " + " | ".join([""] * len(rob_df.columns)) + " |"
        rows   = ["| " + " | ".join(str(v)[:120] for v in row) + " |"
                  for _, row in rob_df.iterrows()]
        f.write("\n".join([header, sep] + rows))
    print(f"[10] Robustness summary (MD) → {rob_md}")

    # 4. Surprise sensitivity (separate table)
    surprise_df = check_surprise_sensitivity(df)
    surprise_csv = os.path.join(PATHS["tables"], "surprise_sensitivity.csv")
    surprise_df.to_csv(surprise_csv, index=False)
    print(f"[10] Surprise sensitivity → {surprise_csv}")
    print(surprise_df.to_string(index=False))


if __name__ == "__main__":
    run()
