"""
Publication-ready exploratory figures.

Figures produced (saved to outputs/figures/):
  dist_all_targets.png          - density/histogram grid for all main targets
  dist_bipolar_vs_unipolar.png  - side-by-side: Valence / PA / NA / Ambivalence
  scatter_valence_vs_PA.png     - Valence vs PositiveAffect
  scatter_valence_vs_NA.png     - Valence vs NegativeAffect
  scatter_PA_vs_NA.png          - PositiveAffect vs NegativeAffect
  scatter_amb_product_vs_min.png- Ambivalence_product vs Ambivalence_min
  All scatter plots highlight near-neutral words (Valence ∈ [2.75, 3.25]).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import PATHS, COLS, RANDOM_SEED, FIGURE, NEAR_NEUTRAL

np.random.seed(RANDOM_SEED)

# Style setup
try:
    plt.style.use(FIGURE["style"])
except OSError:
    plt.style.use("seaborn-whitegrid")   # fallback for older matplotlib

PALETTE    = sns.color_palette(FIGURE["palette"])
DPI        = FIGURE["dpi"]
FMT        = FIGURE["fig_format"]
N_COLOR    = FIGURE["neutral_highlight_color"]
N_ALPHA    = FIGURE["neutral_highlight_alpha"]

MAIN_TARGETS = [
    COLS["valence"],         COLS["arousal"],
    COLS["happiness"],       COLS["anger"],
    COLS["fear"],            COLS["sadness"],
    COLS["disgust"],         COLS["surprise"],
    COLS["positive_affect"], COLS["negative_affect"],
    COLS["ambivalence_prod"],COLS["ambivalence_min"],
]

SIDE_BY_SIDE = [
    COLS["valence"],         COLS["positive_affect"],
    COLS["negative_affect"], COLS["ambivalence_prod"],
    COLS["ambivalence_min"],
]


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PATHS["figures"], filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[04] Figure saved → {path}")


def plot_all_distributions(df: pd.DataFrame) -> None:
    """3×4 grid of density + rug plots for all 12 main targets."""
    cols   = [c for c in MAIN_TARGETS if c in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=50, density=True, alpha=0.45,
                color=PALETTE[i % len(PALETTE)], edgecolor="none")
        #KDE overlay
        try:
            kde = gaussian_kde(data)
            x_grid = np.linspace(data.min(), data.max(), 300)
            ax.plot(x_grid, kde(x_grid), color=PALETTE[i % len(PALETTE)],
                    linewidth=2.0)
        except Exception:
            pass
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Rating", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)

    #Remove unused subplots
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distributions of Affective Variables - Speed & Brysbaert (2024)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save_fig(fig, f"dist_all_targets.{FMT}")


def plot_side_by_side(df: pd.DataFrame) -> None:
    """Side-by-side density plots for key targets, with near-neutral band on Valence."""
    cols  = [c for c in SIDE_BY_SIDE if c in df.columns]
    n     = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    lo, hi = NEAR_NEUTRAL["primary"]

    for i, col in enumerate(cols):
        ax   = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=40, density=True, alpha=0.5,
                color=PALETTE[i % len(PALETTE)], edgecolor="none")
        try:
            kde = gaussian_kde(data)
            x_grid = np.linspace(data.min(), data.max(), 300)
            ax.plot(x_grid, kde(x_grid), color=PALETTE[i % len(PALETTE)],
                    linewidth=2.2)
        except Exception:
            pass

        # Highlight near-neutral band on Valence
        if col == COLS["valence"]:
            ax.axvspan(lo, hi, color=N_COLOR, alpha=N_ALPHA, label=f"Near-neutral [{lo},{hi}]")
            ax.legend(fontsize=8, framealpha=0.7)

        mean_val = data.mean()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean={mean_val:.2f}")
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("Rating", fontsize=10)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=10)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Key Affective Variables: Bipolar vs Unipolar-style Targets",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, f"dist_bipolar_vs_unipolar.{FMT}")


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                 filename: str, highlight_neutral: bool = True,
                 sample_n: int = 5000) -> None:
    """
    Scatter plot of x_col vs y_col.
    Samples up to sample_n points for legibility (random state fixed).
    Highlights near-neutral words in a distinct colour if requested.
    """
    if x_col not in df.columns or y_col not in df.columns:
        print(f"  [WARNING] Cannot plot {x_col} vs {y_col} - one or both columns missing.")
        return

    lo, hi      = NEAR_NEUTRAL["primary"]
    val_col     = COLS["valence"]
    plot_df     = df[[x_col, y_col]].dropna()
    if val_col in df.columns:
        near_mask   = (df[val_col] >= lo) & (df[val_col] <= hi)
        near_mask   = near_mask.reindex(plot_df.index, fill_value=False)
    else:
        near_mask   = pd.Series(False, index=plot_df.index)

    # Sample for speed
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(plot_df), size=min(sample_n, len(plot_df)), replace=False)
    plot_df   = plot_df.iloc[idx]
    near_mask = near_mask.iloc[idx]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Non-neutral background points
    ax.scatter(
        plot_df.loc[~near_mask, x_col],
        plot_df.loc[~near_mask, y_col],
        alpha=0.25, s=8, color=PALETTE[0], linewidths=0, label="Other"
    )
    # Near-neutral highlighted points
    if highlight_neutral and near_mask.sum() > 0:
        ax.scatter(
            plot_df.loc[near_mask, x_col],
            plot_df.loc[near_mask, y_col],
            alpha=0.6, s=12, color=N_COLOR, linewidths=0,
            label=f"Near-neutral Valence [{lo},{hi}]"
        )

    # Pearson r annotation
    r_val = plot_df[x_col].corr(plot_df[y_col])
    ax.text(0.05, 0.95, f"r = {r_val:.3f}", transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(f"{x_col} vs {y_col}", fontsize=12, fontweight="bold")
    if highlight_neutral and near_mask.sum() > 0:
        ax.legend(fontsize=9, framealpha=0.8)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    save_fig(fig, filename)


def run(df: pd.DataFrame = None) -> None:
    os.makedirs(PATHS["figures"], exist_ok=True)

    if df is None:
        processed_path = PATHS["processed_data"]
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"[04] Loaded processed data: {df.shape[0]:,} rows")

    print("[04] Generating distribution figures...")
    plot_all_distributions(df)
    plot_side_by_side(df)

    print("[04] Generating scatter plots...")
    plot_scatter(df, COLS["valence"],         COLS["positive_affect"],
                 f"scatter_valence_vs_PA.{FMT}")
    plot_scatter(df, COLS["valence"],         COLS["negative_affect"],
                 f"scatter_valence_vs_NA.{FMT}")
    plot_scatter(df, COLS["positive_affect"], COLS["negative_affect"],
                 f"scatter_PA_vs_NA.{FMT}", highlight_neutral=False)
    plot_scatter(df, COLS["ambivalence_prod"],COLS["ambivalence_min"],
                 f"scatter_amb_product_vs_min.{FMT}", highlight_neutral=False)

    print("[04] All figures complete.")


if __name__ == "__main__":
    run()
