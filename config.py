"""
configs/config.py
==
Central configuration file for the thesis affect analysis pipeline.
ALL paths, column names, formula parameters, CV settings, and the random seed
are defined here. Edit this file to adapt the pipeline to any naming changes
in the dataset or to update thresholds.

Author : Delaney Bruinhart (2044910)
Dataset: Speed & Brysbaert (2024) Dutch Emotion Norms
"""

import os

# ==
# 1. RANDOM SEED - fixed for full reproducibility
# ==
RANDOM_SEED = 42

# ==
# 2. FILE PATHS
# ==
# Absolute path to the raw dataset Excel file (update if moved)
RAW_DATA_PATH = r"SpeedBrysbaertEmotionNorms.xlsx"

# Derive project root relative to this config file so paths work
# when the pipeline is run from any working directory.
# Absolute path to the raw dataset Excel file (update if moved)
RAW_DATA_PATH = r"SpeedBrysbaertEmotionNorms.xlsx"

# Derive project root relative to this config file so paths work
# when the pipeline is run from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "raw_data":        RAW_DATA_PATH,
    "processed_data":  os.path.join(PROJECT_ROOT, "data", "processed", "emotion_norms_processed.csv"),
    "tables":          os.path.join(PROJECT_ROOT, "outputs", "tables"),
    "figures":         os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "logs":            os.path.join(PROJECT_ROOT, "outputs", "logs"),
}

# ==
# 3. DATASET COLUMN NAMES
# Adjust these if the actual Excel column headers differ from the assumed names.
# All downstream scripts use these aliases, so one change here propagates everywhere.
# ==
COLS = {
    # Primary identifier
    "word":          "Word",

    # Bipolar targets (original dataset columns)
    "valence":       "Valence",
    "arousal":       "Arousal",

    # Discrete emotion columns (original dataset columns)
    "happiness":     "Happiness",
    "anger":         "Anger",
    "fear":          "Fear",
    "sadness":       "Sadness",
    "disgust":       "Disgust",
    "surprise":      "Surprise",   # NOTE: excluded from NegativeAffect by design; see below

    # Derived target columns (constructed in 02_target_construction.py)
    "positive_affect":    "PositiveAffect",
    "negative_affect":    "NegativeAffect",
    "ambivalence_prod":   "Ambivalence_product",
    "ambivalence_min":    "Ambivalence_min",

    # Lexical / psycholinguistic control columns
    # ASSUMPTION: these are the column names in the Speed & Brysbaert (2024) file.
    # If absent, relevant scripts will skip gracefully and log a warning.
    "frequency":     "Frequency_Zipf", #Lexical frequency control (RQ3, Section M); Zipf-scaled word frequency"),
    "concreteness":  "Concreteness",
    "imageability":  "Imageability",
    "aoa":           "AoA",            # age of acquisition
    "pos":           "PoS",            # part of speech
    "length":        "Length",         # character length (will be computed if missing)
    "rt":            "DLP_RT",         # lexical decision RT
    "accuracy":      "DLP_Acc",        # lexical decision accuracy
}

# ==
# 4. TARGET CONSTRUCTION PARAMETERS
# ==
TARGET_CONSTRUCTION = {
    # PositiveAffect operationalisation
    # CONCEPTUAL NOTE: Using Happiness as the sole proxy for positive affect
    # is a deliberate simplification following the proposal (Bruinhart, 2025).
    # It is acknowledged as a limitation: happiness is one of several positive
    # emotion categories, and enthusiasm/love/etc. are not included in this
    # dataset. This choice follows the available columns.
    "positive_affect_source": ["Happiness"],  # single column; mean taken if list > 1

    # NegativeAffect operationalisation
    # NOTE: Surprise is EXCLUDED because it does not have a consistently
    # positive or negative valence (Speed & Brysbaert, 2024; Russell, 1980).
    # Including Surprise would weaken the separation between positive and
    # negative affect. A sensitivity check including Surprise is run in
    # 10_robustness.py but is clearly marked as exploratory.
    "negative_affect_sources": ["Anger", "Fear", "Sadness", "Disgust"],

    # Rescaling: min-max to [0, 1] for ambivalence computation
    # IMPORTANT LEAKAGE NOTE: In this exploratory phase, rescaling is fit
    # on the full dataset. In the final benchmark, rescaling parameters
    # (min, max) must be computed on training folds only and applied to
    # validation/test folds. This is documented in 09_pilot_modeling.py.
    "rescale_range": (0.0, 1.0),
}

# ==
# 5. NEAR-NEUTRAL THRESHOLD
# ==
NEAR_NEUTRAL = {
    # Primary threshold (Valence scale: 1–5, midpoint = 3.0)
    "primary":    (2.75, 3.25),

    # Sensitivity check alternatives
    "narrow":     (2.90, 3.10),
    "wide":       (2.50, 3.50),

    # Valence scale bounds (for documentation)
    "scale_min":  1.0,
    "scale_max":  5.0,
    "midpoint":   3.0,
}

# ==
# 6. CROSS-VALIDATION SETTINGS
# ==
CV = {
    "n_folds":     5,
    "shuffle":     True,
    "random_seed": RANDOM_SEED,
    # NOTE: The same folds are reused across all targets and representations
    # to ensure that performance comparisons are made on identical splits.
    # Hyperparameter tuning (alpha for Ridge) uses nested CV (inner 3-fold)
    # only within each outer training fold, preventing leakage.
    "inner_folds": 3,     # for nested CV / alpha selection
    "save_folds":  True,  # save fold indices to outputs/tables/cv_fold_indices.csv
}

# ==
# 7. MODEL SETTINGS
# ==
MODELS = {
    "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    # Character n-gram settings (simple baseline representation)
    "ngram_range":  (2, 4),    # character n-gram range
    "ngram_min_df": 3,         # minimum document frequency for n-gram features
    "ngram_max_features": 50000,
}

# ==
# 8. SUBGROUP THRESHOLDING RULES
# ==
SUBGROUPS = {
    # All continuous variables split at the median unless noted
    "split_method": "median",
    # Minimum subgroup size to flag as reliable (warn if below)
    "min_reliable_n": 50,
    # High ambivalence: top quartile (Q3+) of Ambivalence_product
    "high_ambivalence_quantile": 0.75,
}

# ==
# 9. FIGURE SETTINGS
# ==
FIGURE = {
    "dpi":        300,
    "style":      "seaborn-v0_8-whitegrid",
    "palette":    "Set2",
    "fig_format": "png",
    # Near-neutral highlight colour on scatter/density plots
    "neutral_highlight_color": "#FFC857",
    "neutral_highlight_alpha": 0.25,
}

# ==
# 10. EVALUATION METRICS
# ==
METRICS = ["MAE", "RMSE", "Pearson_r"]

# ==
# 11. METHODOLOGICAL CITATION KEYS
# (For reference when writing thesis - not used computationally)
# ==
CITATIONS = {
    "dataset":        "Speed & Brysbaert (2024)",
    "benchmark_prec": "Plisiecki & Sobieszek (2024)",
    "bipolar_unipolar":"Briesemeister et al.",
    "lexical_controls":"Kuperman et al. (2014)",
    "ambivalence":    "Vaccaro et al. (2020)",
    "measurement":    "Flake & Fried (2020)",
    "reproducibility":"Nosek et al. (2015)",
}
