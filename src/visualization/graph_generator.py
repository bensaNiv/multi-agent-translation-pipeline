"""Generate graphs and visualizations for semantic drift analysis."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_error_vs_distance(
    df: pd.DataFrame,
    metric: str = "cosine",
    output_path: Optional[str] = None,
):
    """Plot error rate vs semantic distance.

    Args:
        df: DataFrame with error_level and distance columns
        metric: Distance metric to plot ('cosine' or 'euclidean')
        output_path: Path to save figure (optional)

    Raises:
        ValueError: If metric not 'cosine' or 'euclidean'

    Example:
        >>> df = pd.DataFrame({
        ...     "error_level": [0, 25, 50],
        ...     "cosine_distance": [0.1, 0.3, 0.5]
        ... })
        >>> plot_error_vs_distance(df, metric="cosine")
    """
    if metric not in ["cosine", "euclidean"]:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    distance_col = f"{metric}_distance"
    if distance_col not in df.columns:
        raise ValueError(f"Column {distance_col} not found in DataFrame")

    plt.figure(figsize=(10, 6))

    grouped = df.groupby("error_level")[distance_col].agg(["mean", "std"])

    plt.errorbar(
        grouped.index,
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        capsize=5,
    )

    plt.xlabel("Spelling Error Rate (%)", fontsize=12)
    plt.ylabel(f"{metric.capitalize()} Distance", fontsize=12)
    plt.title(
        f"Semantic Drift vs. Spelling Error Rate\n({metric.capitalize()} Distance)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot to {output_path}")

    plt.close()


def plot_both_metrics(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot both cosine and Euclidean distances side by side.

    Args:
        df: DataFrame with error_level and distance columns
        output_path: Path to save figure (optional)

    Example:
        >>> df = pd.DataFrame({
        ...     "error_level": [0, 25, 50],
        ...     "cosine_distance": [0.1, 0.3, 0.5],
        ...     "euclidean_distance": [1.0, 2.0, 3.0]
        ... })
        >>> plot_both_metrics(df)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric in zip([ax1, ax2], ["cosine", "euclidean"]):
        distance_col = f"{metric}_distance"
        grouped = df.groupby("error_level")[distance_col].agg(["mean", "std"])

        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            capsize=5,
        )

        ax.set_xlabel("Spelling Error Rate (%)", fontsize=12)
        ax.set_ylabel(f"{metric.capitalize()} Distance", fontsize=12)
        ax.set_title(f"{metric.capitalize()} Distance", fontsize=13)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Semantic Drift vs. Spelling Error Rate\nComparison of Distance Metrics",
        fontsize=14,
        fontweight="bold",
    )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot to {output_path}")

    plt.close()


def generate_all_graphs(df: pd.DataFrame, output_dir: str = "results/graphs"):
    """Generate all standard graphs for the analysis.

    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save graphs

    Example:
        >>> df = pd.DataFrame({
        ...     "error_level": [0, 25],
        ...     "cosine_distance": [0.1, 0.3],
        ...     "euclidean_distance": [1.0, 2.0]
        ... })
        >>> generate_all_graphs(df, "results/graphs")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating graphs...")

    plot_error_vs_distance(
        df, metric="cosine", output_path=str(output_path / "cosine_distance.png")
    )

    plot_error_vs_distance(
        df,
        metric="euclidean",
        output_path=str(output_path / "euclidean_distance.png"),
    )

    plot_both_metrics(df, output_path=str(output_path / "both_metrics.png"))

    logger.info(f"✓ Generated 3 graphs in {output_dir}")
