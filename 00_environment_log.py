"""

Logs Python version, key package versions, OS platform, and random seed
to outputs/logs/environment_log.txt.

Run this FIRST before any other script to create a reproducibility record.
"""

import sys
import platform
import datetime
import os

# Path setup: add project root to sys.path so config can be imported
# regardless of working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PATHS, RANDOM_SEED


def get_package_version(package_name: str) -> str:
    """Return installed version of a package, or 'not installed'."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return "not installed / version unknown"


def log_environment(output_path: str) -> None:
    """
    Write environment details to a plain-text log file.
    Includes Python version, OS, key library versions, and the random seed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    packages = [
        "numpy", "pandas", "scipy", "scikit-learn",
        "matplotlib", "seaborn", "openpyxl",
        "statsmodels", "transformers", "torch",
    ]

    lines = [
        "=" * 60,
        "ENVIRONMENT LOG — Thesis Affect Analysis Pipeline",
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Python version : {sys.version}",
        f"Platform       : {platform.platform()}",
        f"Processor      : {platform.processor()}",
        f"Random seed    : {RANDOM_SEED}  (used for all CV splits, Ridge, numpy, random)",
        "",
        "--- Key package versions ---",
    ]

    for pkg in packages:
        lines.append(f"  {pkg:<20} {get_package_version(pkg)}")

    lines += [
        "",
        "--- Reproducibility notes ---",
        "  - All scripts import RANDOM_SEED from config.py.",
        "  - numpy random state is set at the top of each script.",
        "  - scikit-learn KFold uses shuffle=True with RANDOM_SEED.",
        "  - Cross-validation fold indices are saved to outputs/tables/cv_fold_indices.csv.",
        "  - Min-max rescaling parameters (for ambivalence) are computed on full data",
        "    in this exploratory phase; must be moved inside CV folds for final benchmark.",
        "",
        "=" * 60,
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[00] Environment log saved → {output_path}")
    # Also print to console so it is visible in terminal
    for line in lines:
        print(line)


if __name__ == "__main__":
    log_path = os.path.join(PATHS["logs"], "environment_log.txt")
    log_environment(log_path)
