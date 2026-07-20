"""
Lexical subgroup coverage tables.

Subgroups defined:
  - Frequency:      high vs low (median split of Lg10WF)
  - Concreteness:   high vs low (median split)
  - Imageability:   high vs low (median split)
  - Part of speech: counts per POS category
  - Near-neutral vs non-neutral (Valence threshold)
  - High vs low ambivalence (Q3 split of Ambivalence_product)

For each subgroup:
  - definition / thresholding rule (exact cutoff values documented)
  - N and % of full dataset
  - warns if any subgroup < min_reliable_n (from config)

Produces:
  - outputs/tables/subgroup_coverage.csv
  - outputs/tables/subgroup_coverage.md
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, NEAR_NEUTRAL, SUBGROUPS

np.random.seed(RANDOM_SEED)

MIN_N = SUBGROUPS["min_reliable_n"]


def make_binary_subgroup(df: pd.DataFrame, col: str, method: str = "median",
                          label_high: str = "High", label_low: str = "Low") -> pd.Series:
    """
    Split a continuous column into two groups.
    method='median' uses the median as the cutoff.
    Returns a Series with labels 'High' / 'Low' / NaN.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    if method == "median":
        cutoff = df[col].median()
    else:
        raise ValueError(f"Unsupported split method: {method}")
    result = pd.Series(np.where(df[col] > cutoff, label_high,
                        np.where(df[col] <= cutoff, label_low, np.nan)),
                       index=df.index)
    return result, cutoff


def build_subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full subgroup coverage table.
    Each row is one subgroup cell (e.g. Frequency=High).
    """
    n_total = len(df)
    rows    = []

    def add_rows(var_name: str, series: pd.Series, cutoff=None,
                 cutoff_note: str = ""):
        counts = series.value_counts(dropna=True).sort_index()
        n_nan  = series.isna().sum()
        for label, count in counts.items():
            rows.append({
                "Variable":       var_name,
                "Subgroup":       label,
                "N":              int(count),
                "Pct_of_total":   round(count / n_total * 100, 2),
                "Cutoff":         round(cutoff, 4) if cutoff is not None else "N/A",
                "Cutoff_note":    cutoff_note,
                "N_missing_in_var": int(n_nan),
                "Reliable":       "YES" if count >= MIN_N else f"WARN: N<{MIN_N}",
            })

    #  Frequency 
    freq_col = COLS["frequency"]
    if freq_col in df.columns:
        freq_grp, freq_cut = make_binary_subgroup(df, freq_col)
        add_rows("Frequency", freq_grp, freq_cut,
                 f"Median split; cutoff = {freq_cut:.4f} (log10 word freq)")
    else:
        rows.append({"Variable": "Frequency", "Subgroup": "SKIPPED",
                     "N": 0, "Pct_of_total": 0,
                     "Cutoff": "N/A",
                     "Cutoff_note": f"Column '{freq_col}' not found in dataset",
                     "N_missing_in_var": n_total, "Reliable": "N/A"})

    #  Concreteness 
    conc_col = COLS["concreteness"]
    if conc_col in df.columns:
        conc_grp, conc_cut = make_binary_subgroup(df, conc_col)
        add_rows("Concreteness", conc_grp, conc_cut,
                 f"Median split; cutoff = {conc_cut:.4f}")
    else:
        rows.append({"Variable": "Concreteness", "Subgroup": "SKIPPED",
                     "N": 0, "Pct_of_total": 0,
                     "Cutoff": "N/A",
                     "Cutoff_note": f"Column '{conc_col}' not found",
                     "N_missing_in_var": n_total, "Reliable": "N/A"})

    #  Imageability 
    imag_col = COLS["imageability"]
    if imag_col in df.columns:
        imag_grp, imag_cut = make_binary_subgroup(df, imag_col)
        add_rows("Imageability", imag_grp, imag_cut,
                 f"Median split; cutoff = {imag_cut:.4f}")
    else:
        rows.append({"Variable": "Imageability", "Subgroup": "SKIPPED",
                     "N": 0, "Pct_of_total": 0,
                     "Cutoff": "N/A",
                     "Cutoff_note": f"Column '{imag_col}' not found",
                     "N_missing_in_var": n_total, "Reliable": "N/A"})

    #  Part of speech 
    pos_col = COLS["pos"]
    if pos_col in df.columns:
        pos_series = df[pos_col].fillna("Unknown").astype(str)
        add_rows("Part of Speech (POS)", pos_series,
                 cutoff_note="Nominal categories; no split applied")
    else:
        rows.append({"Variable": "POS", "Subgroup": "SKIPPED",
                     "N": 0, "Pct_of_total": 0,
                     "Cutoff": "N/A",
                     "Cutoff_note": f"Column '{pos_col}' not found",
                     "N_missing_in_var": n_total, "Reliable": "N/A"})

    #  Near-neutral vs non-neutral 
    val_col = COLS["valence"]
    lo_p, hi_p = NEAR_NEUTRAL["primary"]
    if val_col in df.columns:
        nn_series = df[val_col].apply(
            lambda v: f"Near-neutral [{lo_p},{hi_p}]"
            if lo_p <= v <= hi_p else "Non-neutral"
        )
        add_rows("Near-neutral (Valence)", nn_series,
                 cutoff_note=f"Near-neutral defined as Valence ∈ [{lo_p}, {hi_p}]")
    else:
        rows.append({"Variable": "Near-neutral", "Subgroup": "SKIPPED",
                     "N": 0, "Pct_of_total": 0,
                     "Cutoff": "N/A",
                     "Cutoff_note": f"Column '{val_col}' not found",
                     "N_missing_in_var": n_total, "Reliable": "N/A"})

    #  High vs low ambivalence 
    amb_col = COLS["ambivalence_prod"]
    q_thresh = SUBGROUPS["high_ambivalence_quantile"]
    if amb_col in df.columns:
        amb_q = df[amb_col].quantile(q_thresh)
        amb_series = df[amb_col].apply(
            lambda v: f"High ambivalence (>{q_thresh:.0%} quantile)"
            if v > amb_q else "Low/moderate ambivalence"
        )
        add_rows("Ambivalence_product", amb_series, amb_q,
                 f"High = above Q{q_thresh*100:.0f} = {amb_q:.4f}")
    else:
        rows.append({"Variable": "Ambivalence_product", "Subgroup": "SKIPPED",
                     "N": 0, "Pct_of_total": 0,
                     "Cutoff": "N/A",
                     "Cutoff_note": f"Column '{amb_col}' not found",
                     "N_missing_in_var": n_total, "Reliable": "N/A"})

    return pd.DataFrame(rows)


def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS["tables"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[07] Loaded processed data: {df.shape[0]:,} rows")

    sg_df = build_subgroup_table(df)

    # Warn about small subgroups
    small = sg_df[sg_df["Reliable"].str.startswith("WARN", na=False)]
    if not small.empty:
        print(f"\n  [WARNING] Small subgroups detected (N < {MIN_N}):")
        for _, row in small.iterrows():
            print(f"    {row['Variable']} / {row['Subgroup']}: N={row['N']}")
    else:
        print(f"[07] All subgroups have N >= {MIN_N} ✓")

    # Save CSV
    sg_csv = os.path.join(PATHS["tables"], "subgroup_coverage.csv")
    sg_df.to_csv(sg_csv, index=False)
    print(f"[07] Subgroup coverage saved → {sg_csv}")

    # Save Markdown
    sg_md = os.path.join(PATHS["tables"], "subgroup_coverage.md")
    header = "| " + " | ".join(sg_df.columns) + " |"
    sep    = "| " + " | ".join([""] * len(sg_df.columns)) + " |"
    md_rows = ["| " + " | ".join(str(v) for v in row) + " |"
               for _, row in sg_df.iterrows()]
    with open(sg_md, "w", encoding="utf-8") as f:
        f.write("# Lexical Subgroup Coverage\n\n")
        f.write("\n".join([header, sep] + md_rows))
    print(f"[07] Subgroup coverage (MD) → {sg_md}")

    print("\n[07] Subgroup Coverage Table:")
    print(sg_df.to_string(index=False))


if __name__ == "__main__":
    run()
