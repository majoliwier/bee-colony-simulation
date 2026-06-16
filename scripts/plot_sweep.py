"""
Plot results from the parameter sweep CSV.

Usage (from repo root):
    python -m scripts.plot_sweep                          # auto-finds latest sweep CSV
    python -m scripts.plot_sweep --csv results/sweep_20260616.csv
    python -m scripts.plot_sweep --show                   # display instead of saving
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUT_DIR = "results"
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]  # colour-blind friendly

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(csv_path: str | None) -> pd.DataFrame:
    if csv_path is None:
        files = sorted(glob.glob(os.path.join(OUT_DIR, "sweep_*.csv")))
        if not files:
            raise FileNotFoundError(
                "No sweep CSV found in results/. Run: python -m scripts.sweep"
            )
        csv_path = files[-1]
    print(f"Loading {csv_path}")
    return pd.read_csv(csv_path)


def _save_or_show(fig: plt.Figure, name: str, show: bool) -> None:
    if show:
        plt.show()
    else:
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── individual plots ──────────────────────────────────────────────────────────

def plot_boxplot_nurse_age(df: pd.DataFrame, show: bool) -> None:
    """Box-plot: final nectar by nurse_to_forager_age (marginalised over other params)."""
    ages = sorted(df["nurse_to_forager_age"].unique())
    data = [df.loc[df["nurse_to_forager_age"] == a, "final_nectar"].values for a in ages]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels([str(a) for a in ages])
    ax.set_xlabel("nurse_to_forager_age (steps)")
    ax.set_ylabel("Final nectar")
    ax.set_title("Effect of nurse→forager transition age on colony nectar\n"
                 "(all trail & patch combinations, 3 seeds each)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _save_or_show(fig, "boxplot_nurse_age.png", show)


def plot_boxplot_patches(df: pd.DataFrame, show: bool) -> None:
    """Box-plot: final nectar by num_patches."""
    counts = sorted(df["num_patches"].unique())
    data   = [df.loc[df["num_patches"] == c, "final_nectar"].values for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Number of flower patches")
    ax.set_ylabel("Final nectar")
    ax.set_title("Effect of flower patch count on colony nectar\n"
                 "(all age & trail combinations, 3 seeds each)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _save_or_show(fig, "boxplot_patches.png", show)


def plot_boxplot_trail(df: pd.DataFrame, show: bool) -> None:
    """Box-plot: final nectar by trail_deposit_strength."""
    strengths = sorted(df["trail_deposit_strength"].unique())
    data      = [df.loc[df["trail_deposit_strength"] == s, "final_nectar"].values
                 for s in strengths]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", linewidth=2))
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels([str(s) for s in strengths])
    ax.set_xlabel("Trail pheromone deposit strength")
    ax.set_ylabel("Final nectar")
    ax.set_title("Effect of trail pheromone deposit strength on colony nectar\n"
                 "(all age & patch combinations, 3 seeds each)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _save_or_show(fig, "boxplot_trail.png", show)


def plot_heatmap_trail_patches(df: pd.DataFrame, show: bool) -> None:
    """Heat-map: mean final nectar — trail_deposit_strength × num_patches
       (averaged over all nurse ages and seeds)."""
    pivot = (df.groupby(["trail_deposit_strength", "num_patches"])["final_nectar"]
               .mean()
               .unstack("num_patches"))

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="YlGn", vmin=df["final_nectar"].min() * 0.95)
    cbar = fig.colorbar(im, ax=ax, label="Mean final nectar")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("Number of flower patches")
    ax.set_ylabel("Trail deposit strength")
    ax.set_title("Mean final nectar — trail strength × patch count\n"
                 "(averaged over all nurse ages, 3 seeds each)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color="black" if val < 750 else "white")

    fig.tight_layout()
    _save_or_show(fig, "heatmap_trail_patches.png", show)


def plot_lines_age_vs_patches(df: pd.DataFrame, show: bool) -> None:
    """Line plot: mean nectar vs num_patches, one line per nurse age
       (trail averaged out) — shows the interaction clearly."""
    ages   = sorted(df["nurse_to_forager_age"].unique())
    counts = sorted(df["num_patches"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for age, color in zip(ages, PALETTE):
        means = [
            df.loc[(df["nurse_to_forager_age"] == age) &
                   (df["num_patches"] == c), "final_nectar"].mean()
            for c in counts
        ]
        ax.plot(counts, means, marker="o", color=color, linewidth=2,
                markersize=7, label=f"age={age}")

    ax.set_xlabel("Number of flower patches")
    ax.set_ylabel("Mean final nectar")
    ax.set_title("Interaction: nurse age × patch count\n"
                 "(trail deposit strength averaged, 3 seeds each)")
    ax.legend(title="Nurse→forager age", framealpha=0.9)
    ax.set_xticks(counts)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _save_or_show(fig, "lines_age_patches.png", show)


def plot_peak_foragers(df: pd.DataFrame, show: bool) -> None:
    """Box-plot: peak forager count by nurse_to_forager_age — shows workforce
       dynamics independent of nectar availability."""
    ages = sorted(df["nurse_to_forager_age"].unique())
    data = [df.loc[df["nurse_to_forager_age"] == a, "peak_foragers"].values for a in ages]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels([str(a) for a in ages])
    ax.set_xlabel("nurse_to_forager_age (steps)")
    ax.set_ylabel("Peak forager count")
    ax.set_title("Peak forager workforce by nurse transition age\n"
                 "(early transition = more foragers earlier)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    _save_or_show(fig, "boxplot_peak_foragers.png", show)


def print_top_combos(df: pd.DataFrame, n: int = 5) -> None:
    """Print the N best and worst parameter combinations by mean nectar."""
    summary = (df.groupby(["nurse_to_forager_age", "trail_deposit_strength", "num_patches"])
                 ["final_nectar"].agg(["mean", "std", "count"])
                 .reset_index()
                 .sort_values("mean", ascending=False))

    print(f"\n-- Top {n} parameter combinations --")
    print(f"{'age':>4}  {'trail':>5}  {'patches':>7}  {'mean':>8}  {'std':>7}")
    print("-" * 42)
    for _, row in summary.head(n).iterrows():
        print(f"{int(row['nurse_to_forager_age']):>4}  "
              f"{row['trail_deposit_strength']:>5.1f}  "
              f"{int(row['num_patches']):>7}  "
              f"{row['mean']:>8.1f}  "
              f"{row['std']:>7.1f}")

    print(f"\n-- Bottom {n} parameter combinations --")
    print(f"{'age':>4}  {'trail':>5}  {'patches':>7}  {'mean':>8}  {'std':>7}")
    print("-" * 42)
    for _, row in summary.tail(n).iterrows():
        print(f"{int(row['nurse_to_forager_age']):>4}  "
              f"{row['trail_deposit_strength']:>5.1f}  "
              f"{int(row['num_patches']):>7}  "
              f"{row['mean']:>8.1f}  "
              f"{row['std']:>7.1f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bee colony parameter sweep results")
    parser.add_argument("--csv",  default=None, help="Path to sweep CSV (default: latest in results/)")
    parser.add_argument("--show", action="store_true", help="Display plots instead of saving to PNG")
    args = parser.parse_args()

    df = _load(args.csv)
    print(f"  Rows: {len(df)}  |  Combos: {df.groupby(['nurse_to_forager_age','trail_deposit_strength','num_patches']).ngroups}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nGenerating plots...")
    plot_boxplot_nurse_age(df, args.show)
    plot_boxplot_patches(df, args.show)
    plot_boxplot_trail(df, args.show)
    plot_heatmap_trail_patches(df, args.show)
    plot_lines_age_vs_patches(df, args.show)
    plot_peak_foragers(df, args.show)

    print_top_combos(df)

    if not args.show:
        print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
