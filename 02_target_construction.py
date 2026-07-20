"""
Construct all affective targets from the raw dataset.

Produces:
  - data/processed/emotion_norms_processed.csv    (full dataset + derived targets)
  - outputs/tables/target_formulas.csv            (formula documentation table)
  - outputs/tables/target_formulas.md

Target definitions

Bipolar (original):
  Valence   - from dataset directly
  Arousal   - from dataset directly

Unipolar-style (derived):
  PositiveAffect     = Happiness
  NegativeAffect     = mean(Anger, Fear, Sadness, Disgust)
                       [Surprise excluded; see config.py and notes below]

Rescaling to [0, 1] (for Ambivalence):
  PA_01 = (PositiveAffect - min(PositiveAffect)) / (max - min)
  NA_01 = (NegativeAffect - min(NegativeAffect)) / (max - min)

Ambivalence:
  Ambivalence_product = PA_01 × NA_01
  Ambivalence_min     = min(PA_01, NA_01)   [robustness alternative]

"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, TARGET_CONSTRUCTION

np.random.seed(RANDOM_SEED)


def minmax_scale(series: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Min-max scale a Series to [0, 1].
    Returns (scaled_series, min_val, max_val).
    min_val and max_val should be stored for documentation and
    re-use in the pilot modeling pipeline.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        raise ValueError(f"Cannot rescale column with zero range: min=max={min_val}")
    scaled = (series - min_val) / (max_val - min_val)
    return scaled, float(min_val), float(max_val)


def construct_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Add all derived target columns to the DataFrame.
    Returns (modified_df, rescaling_params).
    rescaling_params documents the exact min/max used for rescaling,
    enabling exact replication.
    """
    rescaling_params = {}

    # 1. PositiveAffect = Happiness
    pa_sources = TARGET_CONSTRUCTION["positive_affect_source"]
    pa_cols = [COLS[c.lower()] for c in ["happiness"]]  # single source
    missing_pa = [c for c in pa_cols if c not in df.columns]
    if missing_pa:
        raise KeyError(f"[02] Columns for PositiveAffect missing: {missing_pa}")

    df[COLS["positive_affect"]] = df[pa_cols].mean(axis=1)
    print(f"[02] PositiveAffect = mean of {pa_cols}  (currently: Happiness only)")

    # 2. NegativeAffect = mean(Anger, Fear, Sadness, Disgust)
    na_source_keys = ["anger", "fear", "sadness", "disgust"]
    na_cols = [COLS[k] for k in na_source_keys]
    missing_na = [c for c in na_cols if c not in df.columns]
    if missing_na:
        raise KeyError(f"[02] Columns for NegativeAffect missing: {missing_na}")

    df[COLS["negative_affect"]] = df[na_cols].mean(axis=1)
    print(f"[02] NegativeAffect = mean of {na_cols}")
    print(f"     NOTE: Surprise excluded by design - no consistent valence direction.")

    # 3. Rescale PositiveAffect and NegativeAffect to [0, 1]
    pa_01, pa_min, pa_max = minmax_scale(df[COLS["positive_affect"]])
    na_01, na_min, na_max = minmax_scale(df[COLS["negative_affect"]])

    rescaling_params["PositiveAffect"] = {"min": pa_min, "max": pa_max,
                                           "note": "fit on full dataset (exploratory phase only)"}
    rescaling_params["NegativeAffect"] = {"min": na_min, "max": na_max,
                                           "note": "fit on full dataset (exploratory phase only)"}

    # Store as intermediate columns for transparency
    df["PositiveAffect_01"] = pa_01.round(6)
    df["NegativeAffect_01"] = na_01.round(6)
    print(f"[02] PositiveAffect rescaled to [0,1]: min={pa_min:.4f}, max={pa_max:.4f}")
    print(f"[02] NegativeAffect rescaled to [0,1]: min={na_min:.4f}, max={na_max:.4f}")

    # 4. Ambivalence_product = PA_01 * NA_01
    df[COLS["ambivalence_prod"]] = (pa_01 * na_01).round(6)
    print(f"[02] Ambivalence_product = PositiveAffect_01 × NegativeAffect_01")
    print(f"     Conceptual: high only when BOTH positive and negative affect are elevated.")

    # 5. Ambivalence_min = min(PA_01, NA_01)  [robustness check]
    df[COLS["ambivalence_min"]] = df[["PositiveAffect_01", "NegativeAffect_01"]].min(axis=1).round(6)
    print(f"[02] Ambivalence_min = min(PositiveAffect_01, NegativeAffect_01)")
    print(f"     Conceptual: bounded by the lower component; alternative ambivalence formula.")

    # 6. Word length (computed if column not present in dataset)
    length_col = COLS["length"]
    word_col   = COLS["word"]
    if length_col not in df.columns and word_col in df.columns:
        df[length_col] = df[word_col].astype(str).str.len()
        print(f"[02] '{length_col}' column not found in dataset; computed as character length of '{word_col}'.")

    return df, rescaling_params


def build_formula_table() -> pd.DataFrame:
    """
    Builds a documentation table of all target formulas in plain language
    and code-like notation.
    """
    rows = [
        {
            "Target":       "Valence",
            "Type":         "Bipolar (original)",
            "Formula":      "Valence  [directly from dataset]",
            "Scale":        "1–5 (continuous; 1 = most negative, 5 = most positive)",
            "Notes":        "Core circumplex dimension (Russell, 1980). Bipolar: negative and "
                            "positive are opposite ends of a single scale.",
            "Reference":    "Speed & Brysbaert (2024)",
        },
        {
            "Target":       "Arousal",
            "Type":         "Bipolar (original)",
            "Formula":      "Arousal  [directly from dataset]",
            "Scale":        "1–5 (continuous; 1 = calm, 5 = aroused)",
            "Notes":        "Second circumplex dimension (Russell, 1980).",
            "Reference":    "Speed & Brysbaert (2024)",
        },
        {
            "Target":       "PositiveAffect",
            "Type":         "Unipolar-style (derived)",
            "Formula":      "PositiveAffect = Happiness",
            "Scale":        "1–5 (inherited from Happiness rating)",
            "Notes":        "CONCEPTUAL FLAG: Using Happiness as sole positive affect proxy is "
                            "a pragmatic simplification. No other positive discrete emotions are "
                            "available in the Speed & Brysbaert (2024) dataset. "
                            "Positive and negative affect are modelled as SEPARATE, "
                            "non-mutually-exclusive targets (unipolar-style; Souders & Yu, 2025).",
            "Reference":    "Bruinhart (2025) operationalisation; Speed & Brysbaert (2024) data",
        },
        {
            "Target":       "NegativeAffect",
            "Type":         "Unipolar-style (derived)",
            "Formula":      "NegativeAffect = mean(Anger, Fear, Sadness, Disgust)",
            "Scale":        "1–5 (mean of four 1–5 columns)",
            "Notes":        "Surprise excluded: no consistent valence polarity (Russell, 1980; "
                            "Speed & Brysbaert, 2024). Mean aggregation treats four negative "
                            "emotions as equally weighted; this is a design assumption.",
            "Reference":    "Bruinhart (2025) operationalisation; Speed & Brysbaert (2024) data",
        },
        {
            "Target":       "PositiveAffect_01",
            "Type":         "Intermediate (rescaled)",
            "Formula":      "PositiveAffect_01 = (PositiveAffect - min_PA) / (max_PA - min_PA)",
            "Scale":        "[0, 1]",
            "Notes":        "LEAKAGE WARNING: min/max fit on full dataset in exploratory phase. "
                            "Must be fit inside training folds in final benchmark.",
            "Reference":    "Standard min-max normalisation",
        },
        {
            "Target":       "NegativeAffect_01",
            "Type":         "Intermediate (rescaled)",
            "Formula":      "NegativeAffect_01 = (NegativeAffect - min_NA) / (max_NA - min_NA)",
            "Scale":        "[0, 1]",
            "Notes":        "Same leakage caveat as PositiveAffect_01.",
            "Reference":    "Standard min-max normalisation",
        },
        {
            "Target":       "Ambivalence_product",
            "Type":         "Unipolar-style (derived) - PRIMARY",
            "Formula":      "Ambivalence_product = PositiveAffect_01 × NegativeAffect_01",
            "Scale":        "[0, 1]; high only when BOTH components are high",
            "Notes":        "Product formula: sensitive to both components simultaneously. "
                            "Maximum is 0.25 (when both components = 0.5); "
                            "maximum achievable is 1.0 (both = 1.0). "
                            "Conceptually: captures 'mixed affect' (Vaccaro et al., 2020).",
            "Reference":    "Bruinhart (2025); Vaccaro et al. (2020)",
        },
        {
            "Target":       "Ambivalence_min",
            "Type":         "Unipolar-style (derived) - ROBUSTNESS CHECK",
            "Formula":      "Ambivalence_min = min(PositiveAffect_01, NegativeAffect_01)",
            "Scale":        "[0, 1]; bounded by the lower component",
            "Notes":        "Alternative ambivalence formula: a word must have BOTH components "
                            "elevated to score high. More conservative than product formula. "
                            "Used as a robustness check, NOT the primary operationalisation.",
            "Reference":    "Bruinhart (2025) robustness check",
        },
    ]
    return pd.DataFrame(rows)


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    """Main entry point. Returns the processed DataFrame."""
    os.makedirs(os.path.dirname(PATHS["processed_data"]), exist_ok=True)
    os.makedirs(PATHS["tables"], exist_ok=True)

    if df is None:
        if not os.path.exists(PATHS["raw_data"]):
            raise FileNotFoundError(f"Raw data not found: {PATHS['raw_data']}")
        df = pd.read_excel(PATHS["raw_data"], engine="openpyxl")
        print(f"[02] Loaded raw data: {df.shape[0]:,} rows")

    df, rescaling_params = construct_targets(df)

    #Save processed dataset
    df.to_csv(PATHS["processed_data"], index=False, encoding="utf-8")
    print(f"[02] Processed dataset saved → {PATHS['processed_data']}")

    #Save rescaling parameters
    rp_rows = []
    for col, params in rescaling_params.items():
        rp_rows.append({"Column": col, **params})
    rp_df = pd.DataFrame(rp_rows)
    rp_path = os.path.join(PATHS["tables"], "rescaling_parameters.csv")
    rp_df.to_csv(rp_path, index=False)
    print(f"[02] Rescaling parameters saved → {rp_path}")

    #Save formula table
    formula_df = build_formula_table()
    formula_csv = os.path.join(PATHS["tables"], "target_formulas.csv")
    formula_md  = os.path.join(PATHS["tables"], "target_formulas.md")
    formula_df.to_csv(formula_csv, index=False)

    #Markdown version
    header = "| " + " | ".join(formula_df.columns) + " |"
    sep    = "| " + " | ".join([""] * len(formula_df.columns)) + " |"
    md_rows = ["| " + " | ".join(str(v).replace("\n","") for v in row) + " |"
               for _, row in formula_df.iterrows()]
    with open(formula_md, "w", encoding="utf-8") as f:
        f.write("# Target Formula Documentation\n\n")
        f.write("\n".join([header, sep] + md_rows))
    print(f"[02] Formula table saved → {formula_csv}")
    print(f"[02] Formula table (MD)  → {formula_md}")

    #Print summary statistics for derived targets
    derived_targets = [
        COLS["positive_affect"], COLS["negative_affect"],
        COLS["ambivalence_prod"], COLS["ambivalence_min"],
    ]
    print("\n[02] Quick summary of derived targets:")
    print(df[derived_targets].describe().round(4).to_string())

    return df


if __name__ == "__main__":
    run()
