#!/usr/bin/env python3
"""
Generate publication-quality visualizations with statistical analysis for semantic drift experiment.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_data(csv_path: str) -> pd.DataFrame:
    """Load semantic drift analysis data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points")
    return df

def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate correlation statistics and other metrics."""
    # Group by error level
    grouped = df.groupby('error_level').agg({
        'cosine_distance': ['mean', 'std', 'count'],
        'euclidean_distance': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Calculate Pearson correlation
    pearson_cosine = stats.pearsonr(df['error_level'], df['cosine_distance'])
    pearson_euclidean = stats.pearsonr(df['error_level'], df['euclidean_distance'])

    # Calculate Spearman correlation (non-parametric)
    spearman_cosine = stats.spearmanr(df['error_level'], df['cosine_distance'])
    spearman_euclidean = stats.spearmanr(df['error_level'], df['euclidean_distance'])

    stats_summary = {
        'grouped': grouped,
        'pearson_cosine': {'r': pearson_cosine[0], 'p': pearson_cosine[1]},
        'pearson_euclidean': {'r': pearson_euclidean[0], 'p': pearson_euclidean[1]},
        'spearman_cosine': {'rho': spearman_cosine[0], 'p': spearman_cosine[1]},
        'spearman_euclidean': {'rho': spearman_euclidean[0], 'p': spearman_euclidean[1]}
    }

    return stats_summary

def plot_cosine_distance(df: pd.DataFrame, stats_summary: dict, output_dir: Path):
    """Generate cosine distance plot with error bars and statistics."""
    fig, ax = plt.subplots(figsize=(12, 8))

    grouped = stats_summary['grouped']

    # Plot with error bars
    ax.errorbar(
        grouped['error_level'],
        grouped['cosine_distance_mean'],
        yerr=grouped['cosine_distance_std'],
        marker='o',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        label='Mean ± Std Dev'
    )

    # Add individual data points
    ax.scatter(
        df['error_level'],
        df['cosine_distance'],
        alpha=0.3,
        s=50,
        label='Individual measurements'
    )

    ax.set_xlabel('Spelling Error Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_title('Semantic Drift vs Spelling Error Rate\n(Cosine Distance)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add statistics annotation
    pearson = stats_summary['pearson_cosine']
    spearman = stats_summary['spearman_cosine']
    stats_text = f"Pearson r = {pearson['r']:.3f}, p = {pearson['p']:.4f}\n"
    stats_text += f"Spearman ρ = {spearman['rho']:.3f}, p = {spearman['p']:.4f}"

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11)

    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'cosine_distance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_euclidean_distance(df: pd.DataFrame, stats_summary: dict, output_dir: Path):
    """Generate Euclidean distance plot with error bars and statistics."""
    fig, ax = plt.subplots(figsize=(12, 8))

    grouped = stats_summary['grouped']

    # Plot with error bars
    ax.errorbar(
        grouped['error_level'],
        grouped['euclidean_distance_mean'],
        yerr=grouped['euclidean_distance_std'],
        marker='s',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color='orange',
        label='Mean ± Std Dev'
    )

    # Add individual data points
    ax.scatter(
        df['error_level'],
        df['euclidean_distance'],
        alpha=0.3,
        s=50,
        color='orange',
        label='Individual measurements'
    )

    ax.set_xlabel('Spelling Error Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Euclidean Distance', fontsize=14, fontweight='bold')
    ax.set_title('Semantic Drift vs Spelling Error Rate\n(Euclidean Distance)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add statistics annotation
    pearson = stats_summary['pearson_euclidean']
    spearman = stats_summary['spearman_euclidean']
    stats_text = f"Pearson r = {pearson['r']:.3f}, p = {pearson['p']:.4f}\n"
    stats_text += f"Spearman ρ = {spearman['rho']:.3f}, p = {spearman['p']:.4f}"

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=11)

    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'euclidean_distance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_both_metrics(df: pd.DataFrame, stats_summary: dict, output_dir: Path):
    """Generate combined plot with both metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    grouped = stats_summary['grouped']

    # Cosine distance subplot
    ax1.errorbar(
        grouped['error_level'],
        grouped['cosine_distance_mean'],
        yerr=grouped['cosine_distance_std'],
        marker='o',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color='steelblue'
    )
    ax1.scatter(
        df['error_level'],
        df['cosine_distance'],
        alpha=0.2,
        s=30,
        color='steelblue'
    )
    ax1.set_xlabel('Spelling Error Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Cosine Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    pearson_c = stats_summary['pearson_cosine']
    ax1.text(0.02, 0.98, f"r = {pearson_c['r']:.3f}\np = {pearson_c['p']:.4f}",
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Euclidean distance subplot
    ax2.errorbar(
        grouped['error_level'],
        grouped['euclidean_distance_mean'],
        yerr=grouped['euclidean_distance_std'],
        marker='s',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        color='darkorange'
    )
    ax2.scatter(
        df['error_level'],
        df['euclidean_distance'],
        alpha=0.2,
        s=30,
        color='darkorange'
    )
    ax2.set_xlabel('Spelling Error Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Euclidean Distance', fontsize=12, fontweight='bold')
    ax2.set_title('Euclidean Distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    pearson_e = stats_summary['pearson_euclidean']
    ax2.text(0.02, 0.98, f"r = {pearson_e['r']:.3f}\np = {pearson_e['p']:.4f}",
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.suptitle('Semantic Drift Analysis: Both Metrics',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = output_dir / 'both_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def generate_statistical_summary(stats_summary: dict, output_dir: Path):
    """Generate a text summary of statistical findings."""
    summary_text = "# Statistical Analysis Summary\n\n"
    summary_text += "## Correlation Analysis\n\n"

    summary_text += "### Cosine Distance\n"
    pc = stats_summary['pearson_cosine']
    sc = stats_summary['spearman_cosine']
    summary_text += f"- Pearson correlation: r = {pc['r']:.4f}, p-value = {pc['p']:.6f}\n"
    summary_text += f"- Spearman correlation: ρ = {sc['rho']:.4f}, p-value = {sc['p']:.6f}\n"

    if pc['p'] < 0.001:
        summary_text += "- **Highly significant** positive correlation (p < 0.001)\n"
    elif pc['p'] < 0.05:
        summary_text += "- **Significant** positive correlation (p < 0.05)\n"
    else:
        summary_text += "- No significant correlation (p ≥ 0.05)\n"
    summary_text += "\n"

    summary_text += "### Euclidean Distance\n"
    pe = stats_summary['pearson_euclidean']
    se = stats_summary['spearman_euclidean']
    summary_text += f"- Pearson correlation: r = {pe['r']:.4f}, p-value = {pe['p']:.6f}\n"
    summary_text += f"- Spearman correlation: ρ = {se['rho']:.4f}, p-value = {se['p']:.6f}\n"

    if pe['p'] < 0.001:
        summary_text += "- **Highly significant** positive correlation (p < 0.001)\n"
    elif pe['p'] < 0.05:
        summary_text += "- **Significant** positive correlation (p < 0.05)\n"
    else:
        summary_text += "- No significant correlation (p ≥ 0.05)\n"
    summary_text += "\n"

    summary_text += "## Mean Distances by Error Level\n\n"
    grouped = stats_summary['grouped']
    summary_text += "| Error % | Cosine Distance (mean ± std) | Euclidean Distance (mean ± std) | N |\n"
    summary_text += "|---------|------------------------------|----------------------------------|---|\n"

    for _, row in grouped.iterrows():
        summary_text += f"| {row['error_level']:.0f}% | "
        summary_text += f"{row['cosine_distance_mean']:.4f} ± {row['cosine_distance_std']:.4f} | "
        summary_text += f"{row['euclidean_distance_mean']:.4f} ± {row['euclidean_distance_std']:.4f} | "
        summary_text += f"{row['cosine_distance_count']:.0f} |\n"

    summary_text += "\n## Interpretation\n\n"
    summary_text += "Based on the statistical analysis:\n\n"

    if pc['r'] > 0 and pc['p'] < 0.05:
        summary_text += f"1. **Confirmed positive correlation**: As spelling error rate increases, "
        summary_text += f"semantic drift (cosine distance) increases with correlation r = {pc['r']:.3f}.\n"
    else:
        summary_text += "1. **Weak/no correlation**: The relationship between error rate and semantic drift is not statistically significant.\n"

    # Check for non-monotonic pattern
    means = grouped['cosine_distance_mean'].values
    if len(means) > 2:
        if not all(means[i] <= means[i+1] for i in range(len(means)-1)):
            summary_text += "\n2. **Non-monotonic pattern observed**: Semantic distance does not increase uniformly. "
            summary_text += "This may indicate:\n"
            summary_text += "   - Small sample size effects (n=5 per error level)\n"
            summary_text += "   - Mock translation artifacts\n"
            summary_text += "   - Need for real BERT embeddings for accurate measurement\n"

    output_path = output_dir / 'statistical_analysis.txt'
    with open(output_path, 'w') as f:
        f.write(summary_text)
    print(f"Saved: {output_path}")
    print("\n" + summary_text)

def main():
    """Main execution function."""
    # Paths
    csv_path = 'results/analysis/semantic_drift.csv'
    output_dir = Path('results/graphs')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication-Quality Visualizations")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    print()

    # Calculate statistics
    print("Calculating statistics...")
    stats_summary = calculate_statistics(df)
    print("Done.")
    print()

    # Generate plots
    print("Generating visualizations...")
    plot_cosine_distance(df, stats_summary, output_dir)
    plot_euclidean_distance(df, stats_summary, output_dir)
    plot_both_metrics(df, stats_summary, output_dir)
    print()

    # Generate statistical summary
    print("Generating statistical summary...")
    generate_statistical_summary(stats_summary, output_dir)
    print()

    print("=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == '__main__':
    main()
