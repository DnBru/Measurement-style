## Thesis Affect Analysis Pipeline
### Delaney Bruinhart 
### Supervisor: dr. M. Paula Roncaglia
### 2nd Reader: ir. Martijn van Leeuwen


## Overview

This repository contains a reproducible exploratory and benchmarking pipeline for the thesis:

Investigating the unipolar versus bipolar measurement style in word classification

The pipeline is built around the Dutch word emotion norms dataset of Speed & Brysbaert (2024) and supports:

- dataset auditing
- target construction for bipolar and unipolar-style affect variables
- descriptive statistics and exploratory visualizations
- near-neutral and ambivalence analyses
- robustness checks
- predictive benchmarking across three lexical representation families:
  - character n-grams
  - fastText static embeddings
  - BERTje transformer embeddings

The benchmark is designed to compare representations under a common supervised regression setup using Ridge regression and a dummy baseline, so that differences can be interpreted primarily as differences in representation and target operationalization rather than differences in predictive architecture.



## Dataset

Expected raw dataset file:

/SpeedBrysbaertEmotionNorms.xlsx

Source: Speed, L. J., & Brysbaert, M. (2024). Ratings of valence, arousal, happiness, anger, fear, sadness, disgust, and surprise for 24,000 Dutch words. *Behavior Research Methods, 56*, 5023–5039.  
DOI: [10.3758/s13428-023-02239-6](https://doi.org/10.3758/s13428-023-02239-6)

OSF repository: [https://osf.io/9htuv/overview](https://osf.io/9htuv/overview)



## Project Structure

Thesis project/
├── config.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── src_representation_utils.py
├── make_fasttext_embeddings.py
├── make_bertje_embeddings.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── emotion_norms_processed.csv
│       ├── fasttext_emb.npz
│       ├── fasttext_emb.csv              (optional)
│       ├── bertje_emb.npz
│       └── bertje_emb.csv                (optional)
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── notebooks/
├── 00_environment_log.py
├── 01_dataset_audit.py
├── 02_target_construction.py
├── 03_descriptive_stats.py
├── 04_distributions.py
├── 05_correlations.py
├── 06_near_neutral.py
├── 07_subgroups.py
├── 08_feature_report.py
├── 09_pilot_modeling.py
├── 10_robustness.py


## Important Path Configuration

All internal file saving locations are controlled by `config.py`.
Make sure `PROJECT_ROOT` is set correctly:
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
This ensures that processed data and outputs are saved inside the root
rather than one directory above it.



## Installation

Install dependencies:

pip install -r requirements.txt



## Core Pipeline: What Each Script Does

### 00_environment_log.py
Logs:
- Python version
- OS/platform
- random seed
- package versions

Output:
- `outputs/logs/environment_log.txt`



### 01_dataset_audit.py
Audits the raw dataset and documents:
- total rows and columns
- column names and types
- missing values
- duplicates
- min/max/mean/SD of affect variables
- dataset summary
- thesis-relevant columns and their justification

Outputs:
- `outputs/tables/dataset_audit.csv`
- `outputs/tables/dataset_audit.md`
- `outputs/tables/thesis_columns_used.csv`
- `outputs/logs/dataset_summary.txt`



### 02_target_construction.py
Creates the derived targets:

- PositiveAffect = `Happiness`
- NegativeAffect = mean(`Anger`, `Fear`, `Sadness`, `Disgust`)
- PositiveAffect_01 = min-max rescaled PositiveAffect
- NegativeAffect_01 = min-max rescaled NegativeAffect
- Ambivalence_product = `PositiveAffect_01 × NegativeAffect_01`
- Ambivalence_min = `min(PositiveAffect_01, NegativeAffect_01)`

Also documents:
- formulas
- scaling parameters
- processing decisions

Outputs:
- `data/processed/emotion_norms_processed.csv`
- `outputs/tables/target_formulas.csv`
- `outputs/tables/target_formulas.md`
- `outputs/tables/rescaling_parameters.csv`



### 03_descriptive_stats.py
Computes descriptive statistics for original and derived affect variables.

Statistics include:
- N
- mean
- SD
- min
- max
- median
- IQR
- skewness
- kurtosis

Outputs:
- `outputs/tables/descriptive_stats.csv`
- `outputs/tables/descriptive_stats.md`



### 04_distributions.py
Creates publication-ready exploratory figures for:
- target distributions
- side-by-side target comparisons
- scatterplots between affect variables
- ambivalence comparisons
- valence–ambivalence visualization

Outputs:
- multiple `.png` files in `outputs/figures/`



### 05_correlations.py
Computes:
- Pearson correlations
- Spearman correlations
- construct sanity checks
- top ambivalence words
- example word groups

Outputs:
- `outputs/tables/pearson_correlations.csv`
- `outputs/tables/spearman_correlations.csv`
- `outputs/tables/construct_sanity_checks.csv`
- `outputs/tables/top20_ambivalence_product.csv`
- `outputs/tables/top20_ambivalence_min.csv`
- `outputs/tables/word_examples.csv`
- `outputs/figures/heatmap_pearson.png`
- `outputs/figures/heatmap_spearman.png`



### 06_near_neutral.py
Runs the near-neutral and threshold sensitivity analyses.

Primary threshold:
- `Valence ∈ [2.75, 3.25]`

Sensitivity thresholds:
- `[2.90, 3.10]`
- `[2.50, 3.50]`

Outputs:
- `outputs/tables/near_neutral_summary.csv`
- `outputs/tables/near_neutral_summary.md`
- `outputs/tables/threshold_sensitivity.csv`
- `outputs/figures/scatter_valence_ambivalence_annotated.png`



### 07_subgroups.py
Creates lexical subgroup coverage tables for:
- frequency
- concreteness
- imageability
- part of speech
- near-neutrality
- ambivalence level

Outputs:
- `outputs/tables/subgroup_coverage.csv`
- `outputs/tables/subgroup_coverage.md`



### 08_feature_report.py
Documents the representation families and their feasibility:
- character n-grams
- static embeddings
- transformer embeddings

Outputs:
- `outputs/tables/feature_report.csv`
- `outputs/tables/feature_report.md`



### 09_pilot_modeling.py
Runs the predictive benchmark.

It evaluates:
- Dummy baseline
- Ridge regression

Targets:
- `Valence`
- `PositiveAffect`
- `NegativeAffect`
- `Ambivalence_product`

Representations:
- Character n-grams (built inside CV folds)
- fastText static embeddings (loaded from NPZ cache if available)
- BERTje transformer embeddings (loaded from NPZ cache if available)

Cross-validation:
- 5-fold outer CV
- 3-fold inner CV for `alpha` selection
- fixed random seed = 42
- same folds reused across targets and representations

Outputs:
- `outputs/tables/pilot_results.csv`
- `outputs/tables/pilot_results.md`
- `outputs/tables/cv_fold_indices.csv`
- `outputs/logs/cv_documentation.md`



### 10_robustness.py
Runs compact robustness checks for:
- ambivalence formula comparison
- near-neutral threshold sensitivity
- surprise inclusion
- scaling sensitivity

Outputs:
- `outputs/tables/robustness_summary.csv`
- `outputs/tables/robustness_summary.md`
- `outputs/tables/surprise_sensitivity.csv`






## Representation Extraction Workflow

The pipeline separates:
1. embedding extraction
2. benchmarking

This avoids recomputing expensive embeddings every time you run the benchmark.



## Representation 1: Character n-grams

You do not need a separate extraction script for character n-grams.

Character n-grams are already handled inside `09_pilot_modeling.py` using:

- `TfidfVectorizer`
- `analyzer="char_wb"`
- `ngram_range=(2, 4)`

This representation is fit inside the training folds only, which avoids leakage.


## Representation 2: fastText static embeddings

### Required input
Place the Dutch fastText vectors here, data/models/cc.nl.300.vec

### Run
NPZ only(faster processing):
python make_fasttext_embeddings.py --fasttext_path data/models/cc.nl.300.vec

NPZ + CSV(can check by hand):
python make_fasttext_embeddings.py --fasttext_path data/models/cc.nl.300.vec --out_csv data/processed/fasttext_emb.csv

### Outputs
- `data/processed/fasttext_emb.npz`
- optional `data/processed/fasttext_emb.csv`

The NPZ file contains:
- `words`
- `emb`

This is the format expected by `09_pilot_modeling.py`.


## Representation 3: BERTje transformer embeddings

### Model
Default:
GroNLP/bert-base-dutch-cased

### Pooling
Mean pooling over final-layer token embeddings.

### Run
NPZ only:
python make_bertje_embeddings.py

NPZ + CSV:
python make_bertje_embeddings.py --out_csv data/processed/bertje_emb.csv

### Outputs
- `data/processed/bertje_emb.npz`
- optional `data/processed/bertje_emb.csv`

The NPZ file contains:
- `words`
- `emb`

This format is required for compatibility with `09_pilot_modeling.py`.



## Benchmark Order

### 1. Run target construction if needed
python run_pipeline.py --step 2

### 2. Create fastText embeddings
python make_fasttext_embeddings.py --fasttext_path data/models/cc.nl.300.vec

### 3. Create BERTje embeddings
python make_bertje_embeddings.py

### 4. Run the benchmark
python run_pipeline.py --step 9

This will produce benchmark results for:
- dummy baseline
- character n-grams
- fastText
- BERTje

as long as the cached NPZ files exist.

## Full Pipeline Usage

Run everything from the start:

python run_pipeline.py

Run from a specific step:

python run_pipeline.py --step 9

Examples:
- Step 2 → target construction
- Step 9 → predictive benchmark
- Step 10 → robustness

## Key Modeling Decisions

### Targets
Core predictive targets:
- `Valence`
- `PositiveAffect`
- `NegativeAffect`
- `Ambivalence_product`

### Main ambivalence operationalization
Primary:
- `Ambivalence_product`

Robustness:
- `Ambivalence_min`

### Near-neutral threshold
Primary:
- `[2.75, 3.25]`

Sensitivity:
- `[2.90, 3.10]`
- `[2.50, 3.50]`

### Representation families
- character n-grams
- fastText static embeddings
- BERTje transformer embeddings

### Regressor
- Ridge regression

### Baseline
- Dummy mean predictor

### Metrics
- MAE
- RMSE
- Pearson correlation

### Random seed
- `42`

# Reproducibility Notes

The pipeline is designed to support explicit reproducibility by:

- fixing the random seed
- logging package versions and Python version
- saving fold indices
- saving processed data
- saving all tables and figures to disk
- separating raw data, processed data, outputs, and logs
- separating expensive embedding extraction from the benchmark itself

Character n-gram TF–IDF is fit inside CV folds to avoid leakage.  
fastText and BERTje embeddings are pretrained and cached to disk before evaluation.





> All preprocessing, modeling, and evaluation were implemented in Python. The project used NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, statsmodels, openpyxl, transformers, gensim and PyTorch. Exact package versions are logged in `outputs/logs/environment_log.txt`.