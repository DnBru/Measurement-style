"""

Produces:
  - outputs/tables/dataset_audit.csv       (per-column statistics)
  - outputs/tables/dataset_audit.md        (formatted Markdown)
  - outputs/tables/columns_used_thesis.csv (which columns are used and why)
  - A printed plain-text summary of the dataset

Flags:
  - Missing values (count + %)
  - Duplicate rows / duplicate word entries
  - Min / max / mean / SD for all affective variables
  - Any obvious anomalies (values outside expected scale range)
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, NEAR_NEUTRAL

np.random.seed(RANDOM_SEED)

# Affective columns to audit in detail (1–5 Likert scale expected)
AFFECTIVE_COLS = [
    COLS["valence"], COLS["arousal"],
    COLS["happiness"], COLS["anger"], COLS["fear"],
    COLS["sadness"], COLS["disgust"], COLS["surprise"],
]

EXPECTED_SCALE = (1.0, 5.0)   # Speed & Brysbaert (2024) use 1–5 ratings

# Columns used in the thesis, with justification
THESIS_COLUMNS_JUSTIFICATION = [
    ("Word",        "Primary identifier / word form for all analyses"),
    ("Valence",     "Bipolar affect target (RQ1); core dimension of circumplex model (Russell, 1980)"),
    ("Arousal",     "Bipolar affect target (RQ1); second circumplex dimension"),
    ("Happiness",   "Source for PositiveAffect (RQ1, RQ2); sole positive discrete emotion in dataset"),
    ("Anger",       "Component of NegativeAffect (RQ1, RQ2)"),
    ("Fear",        "Component of NegativeAffect (RQ1, RQ2)"),
    ("Sadness",     "Component of NegativeAffect (RQ1, RQ2)"),
    ("Disgust",     "Component of NegativeAffect (RQ1, RQ2)"),
    ("Surprise",    "Excluded from primary targets (no consistent valence); kept for sensitivity check"),
    ("Frequency_Zipf", "Lexical frequency control (RQ3, Section M); Zipf-scaled word frequency"),
    ("Concreteness","Subgroup variable for error analysis (RQ3)"),
    ("Imageability","Subgroup variable for error analysis (RQ3)"),
    ("AoA",         "Lexical control for generalizability analysis (Section M)"),
    ("PoS",         "Subgroup variable for error analysis (RQ3)"),
    ("Length",      "Derived lexical control (character count); proxy for morphological complexity"),
    ("DLP_RT",      ": lexical decision RT for external validity pilot (Section M)"),
]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from Excel. Raises FileNotFoundError with a clear message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at:\n  {path}\n"
            "Please place the Speed & Brysbaert (2024) Excel file at the path "
            "specified in configs/config.py → PATHS['raw_data']."
        )
    print(f"[01] Loading dataset from: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    print(f"     Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


def check_column_presence(df: pd.DataFrame) -> dict:
    """
    Check which expected columns are present vs. absent.
    Returns a dict: {col_name: True/False}.
    Prints a warning for each absent column.
    """
    presence = {}
    for alias, col in COLS.items():
        present = col in df.columns
        presence[alias] = present
        if not present:
            print(f"  [WARNING] Expected column '{col}' (alias: '{alias}') NOT FOUND in dataset.")
            print(f"            Analyses depending on this column will be skipped.")
    return presence


def build_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column audit table with dtype, missing counts, and basic stats.
    """
    rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100
        dtype = str(df[col].dtype)
        row = {
            "Column":         col,
            "dtype":          dtype,
            "N_missing":      int(n_missing),
            "Pct_missing":    round(pct_missing, 2),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row.update({
                "min":  round(df[col].min(), 4),
                "max":  round(df[col].max(), 4),
                "mean": round(df[col].mean(), 4),
                "SD":   round(df[col].std(), 4),
            })
            # Flag anomalies: values outside expected scale for affective columns
            if col in AFFECTIVE_COLS:
                n_out_of_range = ((df[col] < EXPECTED_SCALE[0]) |
                                  (df[col] > EXPECTED_SCALE[1])).sum()
                row["N_out_of_range_1_5"] = int(n_out_of_range)
        else:
            row.update({"min": "", "max": "", "mean": "", "SD": "",
                        "N_out_of_range_1_5": ""})
        rows.append(row)
    return pd.DataFrame(rows)


def duplicate_analysis(df: pd.DataFrame) -> dict:
    """Check for duplicate rows and duplicate word entries."""
    n_dup_rows  = df.duplicated().sum()
    word_col    = COLS["word"]
    n_dup_words = df[word_col].duplicated().sum() if word_col in df.columns else "N/A (column missing)"
    return {"duplicate_rows": int(n_dup_rows), "duplicate_words": n_dup_words}


def build_text_summary(df: pd.DataFrame, dupes: dict, presence: dict) -> str:
    """Return a plain-text summary paragraph for the dataset."""
    n_rows, n_cols = df.shape
    missing_aff = {c: int(df[c].isna().sum())
                   for c in AFFECTIVE_COLS if c in df.columns}
    any_missing = any(v > 0 for v in missing_aff.values())

    summary = (
        f"Dataset Summary (Speed & Brysbaert, 2024 - Dutch Emotion Norms)\n"
        f"{'-'*60}\n"
        f"Total words (rows)       : {n_rows:,}\n"
        f"Total variables (columns): {n_cols}\n"
        f"Duplicate rows           : {dupes['duplicate_rows']}\n"
        f"Duplicate word entries   : {dupes['duplicate_words']}\n"
        f"\nMissing values in affective columns:\n"
    )
    for col, n in missing_aff.items():
        pct = n / n_rows * 100
        flag = "  ← ATTENTION" if n > 0 else ""
        summary += f"  {col:<15}: {n:>5} ({pct:.2f}%){flag}\n"
    if not any_missing:
        summary += "  → No missing values in affective columns. ✓\n"

    # Scale range check
    summary += f"\nExpected scale range for affective ratings: {EXPECTED_SCALE}\n"
    for col in AFFECTIVE_COLS:
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            flag = ""
            if actual_min < EXPECTED_SCALE[0] or actual_max > EXPECTED_SCALE[1]:
                flag = "  ← OUT OF EXPECTED RANGE - INSPECT"
            summary += f"  {col:<15}: [{actual_min:.2f}, {actual_max:.2f}]{flag}\n"

    summary += (
        f"\nNear-neutral words (Valence ∈ {NEAR_NEUTRAL['primary']}): "
        f"will be computed in 06_near_neutral.py\n"
    )
    return summary


def audit_to_markdown(audit_df: pd.DataFrame) -> str:
    """Convert the audit DataFrame to a clean Markdown table string."""
    header = "| " + " | ".join(audit_df.columns) + " |"
    sep    = "| " + " | ".join([""] * len(audit_df.columns)) + " |"
    rows   = []
    for _, row in audit_df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows)


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    """Main entry point for this script. Returns loaded DataFrame for reuse."""
    os.makedirs(PATHS["tables"], exist_ok=True)
    os.makedirs(PATHS["logs"],   exist_ok=True)

    if df is None:
        df = load_data(PATHS["raw_data"])

    presence = check_column_presence(df)
    dupes    = duplicate_analysis(df)
    summary  = build_text_summary(df, dupes, presence)

    print("\n" + summary)

    # Save summary to log
    summary_path = os.path.join(PATHS["logs"], "dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[01] Dataset summary saved → {summary_path}")

    # Build and save audit table
    audit_df = build_audit_table(df)
    audit_csv = os.path.join(PATHS["tables"], "dataset_audit.csv")
    audit_md  = os.path.join(PATHS["tables"], "dataset_audit.md")
    audit_df.to_csv(audit_csv, index=False)
    with open(audit_md, "w", encoding="utf-8") as f:
        f.write("# Dataset Audit - Speed & Brysbaert (2024) Dutch Emotion Norms\n\n")
        f.write(audit_to_markdown(audit_df))
    print(f"[01] Audit table saved   → {audit_csv}")
    print(f"[01] Audit table (MD)    → {audit_md}")

    # Build and save thesis columns table
    thesis_cols_df = pd.DataFrame(
        THESIS_COLUMNS_JUSTIFICATION,
        columns=["Column", "Justification_for_thesis"]
    )
    thesis_cols_csv = os.path.join(PATHS["tables"], "thesis_columns_used.csv")
    thesis_cols_df.to_csv(thesis_cols_csv, index=False)
    print(f"[01] Thesis columns used → {thesis_cols_csv}")

    return df


if __name__ == "__main__":
    run()
