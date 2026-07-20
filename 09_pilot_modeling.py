"""
Pilot benchmark that runs Ridge regression with nested CV on multiple
representations:
 - character n-grams (TF-IDF char_wb)  -- built inside training folds (leak-free)
 - fastText static embeddings           -- loaded from NPZ cache (precomputed)
 - BERTje transformer embeddings       -- loaded from NPZ cache (precomputed)

This script expects the processed dataset at PATHS['processed_data'] (see config.py)
and cached embeddings at:
  data/processed/fasttext_emb.npz
  data/processed/bertje_emb.npz

Outputs:
 - outputs/tables/pilot_results.csv
 - outputs/tables/pilot_results.md
 - outputs/tables/cv_fold_indices.csv
 - outputs/logs/cv_documentation.md
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Add project root to path and import config
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, CV as CV_CFG, MODELS

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# Silence scipy ConstantInputWarning if present (we handle Pearson r=0 fallback)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except Exception:
    pass


# Pilot targets

PILOT_TARGETS = [
    COLS["valence"],
    COLS["positive_affect"],
    COLS["negative_affect"],
    COLS["ambivalence_prod"],
]


# cached embedding paths (NPZ)
# NPZ must contain arrays: 'words' (object) and 'emb' (float)

FASTTEXT_NPZ = os.path.join('data', 'processed', 'fasttext_emb.npz')
BERTJE_NPZ   = os.path.join('data', 'processed', 'bertje_emb.npz')

# Helpers

def load_npz_embeddings(npz_path: str):
    if not os.path.exists(npz_path):
        return None, None
    data = np.load(npz_path, allow_pickle=True)
    words = data['words'].tolist()
    emb = data['emb'].astype(np.float32)
    return words, emb


def pearson_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    try:
        r, _ = pearsonr(y, y_pred)
        if np.isnan(r):
            r = 0.0
    except Exception:
        r = 0.0
    return r


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r, _ = pearsonr(y_true, y_pred)
        if np.isnan(r):
            r = 0.0
    except Exception:
        r = 0.0
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "Pearson_r": round(r, 4)}

# Make char n-gram pipeline (used inside CV)

def make_ngram_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=MODELS["ngram_range"],
            min_df=MODELS["ngram_min_df"],
            max_features=MODELS["ngram_max_features"],
            lowercase=True,
            strip_accents=None,
        )),
        ("ridge", Ridge(alpha=alpha)),
    ])

# Run CV helper (outer folds list of (train_idx, val_idx))
# Precondition: Pipeline expects to be fit on X_train (list or array) and predict on X_val

def run_cv(X, y: np.ndarray, pipeline, folds: list, use_inner_cv: bool = True):
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Build train/val splits (works for list of strings or arrays)
        if hasattr(X, "__len__") and not isinstance(X, (np.ndarray,)):
            X_train = [X[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        if use_inner_cv:
            inner_cv = KFold(n_splits=CV_CFG["inner_folds"], shuffle=True, random_state=RANDOM_SEED)
            param_grid = {"ridge__alpha": MODELS["ridge_alphas"]}
            search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=pearson_scorer, refit=True, n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_alpha = search.best_params_.get("ridge__alpha", None)
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_alpha = getattr(pipeline.named_steps.get('ridge'), 'alpha', None)

        y_pred = best_model.predict(X_val)
        m = compute_metrics(y_val, y_pred)
        m['fold'] = fold_idx + 1
        m['best_alpha'] = best_alpha
        fold_results.append(m)

    # Aggregate
    metric_cols = ["MAE", "RMSE", "Pearson_r"]
    means = {f"{m}_mean": round(np.mean([f[m] for f in fold_results]), 4) for m in metric_cols}
    sds = {f"{m}_SD": round(np.std([f[m] for f in fold_results]), 4) for m in metric_cols}
    return {**means, **sds}, fold_results

# Dummy baseline

def run_dummy(y: np.ndarray, folds: list):
    fold_results = []
    for train_idx, val_idx in folds:
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_pred = np.full(len(y_val), y_train.mean())
        m = compute_metrics(y_val, y_pred)
        fold_results.append(m)
    metric_cols = ["MAE", "RMSE", "Pearson_r"]
    means = {f"{m}_mean": round(np.mean([f[m] for f in fold_results]), 4) for m in metric_cols}
    sds = {f"{m}_SD": round(np.std([f[m] for f in fold_results]), 4) for m in metric_cols}
    return {**means, **sds}

# Main runner

def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS['tables'], exist_ok=True)
    os.makedirs(PATHS['logs'], exist_ok=True)

    if df is None:
        processed_path = PATHS['processed_data']
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)

    word_col = COLS['word']
    if word_col not in df.columns:
        raise KeyError(f"Word column '{word_col}' not found.")

    # Create outer folds and save indices
    n_samples = len(df)
    outer_cv = KFold(n_splits=CV_CFG['n_folds'], shuffle=CV_CFG['shuffle'], random_state=CV_CFG['random_seed'])
    folds = list(outer_cv.split(np.arange(n_samples)))

    if CV_CFG.get('save_folds', True):
        fold_assignments = np.zeros(n_samples, dtype=int)
        for fold_idx, (_, val_idx) in enumerate(folds):
            fold_assignments[val_idx] = fold_idx + 1
        fold_df = df[[word_col]].copy()
        fold_df['fold'] = fold_assignments
        fold_df.to_csv(os.path.join(PATHS['tables'], 'cv_fold_indices.csv'), index=False)

    # Prepare n-gram representation (words list used by TF-IDF inside pipeline)
    words = df[word_col].fillna('').astype(str).tolist()

    # Load cached embeddings
    ft_words, ft_emb = load_npz_embeddings(FASTTEXT_NPZ)
    bj_words, bj_emb = load_npz_embeddings(BERTJE_NPZ)

    # Sanity checks: cached embeddings must align to processed CSV word ordering
    if ft_emb is not None:
        if ft_words != words:
            raise ValueError('fastText NPZ word order does not match processed CSV Word column. Rebuild fastText NPZ from the processed CSV.')
    if bj_emb is not None:
        if bj_words != words:
            raise ValueError('BERTje NPZ word order does not match processed CSV Word column. Rebuild BERTje NPZ from the processed CSV.')

    all_results = []

    for target_col in PILOT_TARGETS:
        if target_col not in df.columns:
            print(f"  [WARNING] Target '{target_col}' not found - skipping.")
            continue

        y = df[target_col].values.astype(float)
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 100:
            print(f"  [WARNING] Too few valid rows for '{target_col}' - skipping.")
            continue

        y_valid = y[valid_mask]
        # words_valid is used only for n-gram pipeline which handles text input
        words_valid = [words[i] for i in np.where(valid_mask)[0]]

        # Rebuild folds on valid indices
        valid_indices = np.where(valid_mask)[0]
        valid_cv = KFold(n_splits=CV_CFG['n_folds'], shuffle=CV_CFG['shuffle'], random_state=CV_CFG['random_seed'])
        valid_folds = list(valid_cv.split(np.arange(len(y_valid))))

        print(f"\n[09] Target: {target_col}  (N={len(y_valid):,})")

        # Dummy baseline
        dummy_metrics = run_dummy(y_valid, valid_folds)
        all_results.append({
            "Target": target_col,
            "Representation": "Dummy (mean predictor)",
            "Model": "DummyRegressor",
            **dummy_metrics,
            "Note": "Lower bound baseline",
        })
        print(f"  Dummy  - MAE={dummy_metrics['MAE_mean']:.4f}, r={dummy_metrics['Pearson_r_mean']:.4f}")

        # N-gram Ridge (pipeline fits TF-IDF inside CV)
        ngram_pipeline = make_ngram_pipeline()
        ngram_metrics, _ = run_cv(words_valid, y_valid, ngram_pipeline, valid_folds, use_inner_cv=True)
        all_results.append({
            "Target": target_col,
            "Representation": "Character n-grams (TF-IDF, char_wb 2-4)",
            "Model": "Ridge (alpha selected by inner 3-fold CV)",
            **ngram_metrics,
            "Note": "Pilot: TF-IDF fit inside training folds",
        })
        print(f"  N-gram - MAE={ngram_metrics['MAE_mean']:.4f}, r={ngram_metrics['Pearson_r_mean']:.4f}")

        # fastText Ridge (if cached)
        if ft_emb is not None:
            X_ft = ft_emb[valid_mask]
            ft_pipeline = Pipeline([('ridge', Ridge())])
            ft_metrics, _ = run_cv(X_ft, y_valid, ft_pipeline, valid_folds, use_inner_cv=True)
            all_results.append({
                "Target": target_col,
                "Representation": "fastText (static embeddings)",
                "Model": "Ridge (alpha selected by inner 3-fold CV)",
                **ft_metrics,
                "Note": "Pretrained static embeddings; loaded from NPZ cache",
            })
            print(f"  fastText - MAE={ft_metrics['MAE_mean']:.4f}, r={ft_metrics['Pearson_r_mean']:.4f}")

        # BERTje Ridge (if cached)
        if bj_emb is not None:
            X_bj = bj_emb[valid_mask]
            bj_pipeline = Pipeline([('ridge', Ridge())])
            bj_metrics, _ = run_cv(X_bj, y_valid, bj_pipeline, valid_folds, use_inner_cv=True)
            all_results.append({
                "Target": target_col,
                "Representation": "BERTje (mean pooled, final layer)",
                "Model": "Ridge (alpha selected by inner 3-fold CV)",
                **bj_metrics,
                "Note": "Transformer embeddings; loaded from NPZ cache",
            })
            print(f"  BERTje  - MAE={bj_metrics['MAE_mean']:.4f}, r={bj_metrics['Pearson_r_mean']:.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(PATHS['tables'], 'pilot_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n[09] Pilot results saved → {results_csv}")

    # Markdown
    results_md = os.path.join(PATHS['tables'], 'pilot_results.md')
    if not results_df.empty:
        header = "| " + " | ".join(results_df.columns) + " |"
        sep = "| " + " | ".join([""] * len(results_df.columns)) + " |"
        md_rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in results_df.iterrows()]
        with open(results_md, 'w', encoding='utf-8') as f:
            f.write('# Pilot Benchmark Results\n\n')
            f.write('\n'.join([header, sep] + md_rows))
        print(f"[09] Pilot results (MD)    → {results_md}")

    # CV documentation
    cv_doc_path = os.path.join(PATHS['logs'], 'cv_documentation.md')
    with open(cv_doc_path, 'w', encoding='utf-8') as f:
        f.write(f"# Cross-Validation Documentation\n\nSee config.py for CV settings. Outer folds: {CV_CFG['n_folds']}, inner folds for alpha selection: {CV_CFG['inner_folds']}.\n")
    print(f"[09] CV documentation saved → {cv_doc_path}")


if __name__ == '__main__':
    run()
