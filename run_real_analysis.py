#!/usr/bin/env python3
"""
Run semantic drift analysis on REAL pipeline results from Claude Task agents.

This script uses Sentence-BERT embeddings to compute semantic distances
between original and final translated texts from the real experiment.
"""

import json
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean
import numpy as np


def load_real_results(json_path: str) -> list:
    """Load real pipeline results from JSON file."""
    print(f"📖 Loading real results from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    print(f"✓ Loaded {len(results)} real translation pipeline results")
    return results


def compute_semantic_distances(results: list, model_name: str = 'all-MiniLM-L6-v2') -> list:
    """
    Compute semantic distances using Sentence-BERT embeddings.

    Args:
        results: List of translation pipeline results
        model_name: Sentence-BERT model to use (default: all-MiniLM-L6-v2)

    Returns:
        List of analysis results with distance metrics
    """
    print(f"\n🤖 Loading Sentence-BERT model: {model_name}")
    print("   (Running on CPU - this may take 2-3 minutes for 35 sentences)")
    model = SentenceTransformer(model_name)
    print("✓ Model loaded successfully")

    analysis_results = []
    total = len(results)

    print(f"\n📊 Computing semantic distances for {total} results...")

    for i, result in enumerate(results, 1):
        sentence_id = result['sentence_id']
        error_level = result['error_level']
        original_text = result['original_text']
        final_text = result['final_english_text']

        # Generate embeddings
        embeddings = model.encode([original_text, final_text])
        original_embedding = embeddings[0]
        final_embedding = embeddings[1]

        # Compute distances
        cos_distance = float(cosine(original_embedding, final_embedding))
        euc_distance = float(euclidean(original_embedding, final_embedding))

        analysis_results.append({
            'sentence_id': sentence_id,
            'error_level': error_level,
            'cosine_distance': cos_distance,
            'euclidean_distance': euc_distance,
            'original_text': original_text,
            'final_text': final_text
        })

        # Progress indicator
        if i % 5 == 0 or i == total:
            print(f"  Progress: {i}/{total} ({i*100//total}%)")

    print("✓ Distance computation complete")
    return analysis_results


def save_to_csv(analysis_results: list, output_path: str):
    """Save analysis results to CSV file."""
    print(f"\n💾 Saving results to {output_path}...")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = ['sentence_id', 'error_level', 'cosine_distance',
                  'euclidean_distance', 'original_text', 'final_text']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_results)

    print(f"✓ Saved {len(analysis_results)} rows to {output_path}")


def print_summary(analysis_results: list):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    # Group by error level
    by_error = {}
    for result in analysis_results:
        error = result['error_level']
        if error not in by_error:
            by_error[error] = {'cosine': [], 'euclidean': []}
        by_error[error]['cosine'].append(result['cosine_distance'])
        by_error[error]['euclidean'].append(result['euclidean_distance'])

    print("\nSemantic Drift by Error Level:")
    print("-" * 60)
    print(f"{'Error %':<10} {'Cosine Distance':<30} {'Euclidean Distance':<30}")
    print(f"{'':10} {'(mean ± std)':<30} {'(mean ± std)':<30}")
    print("-" * 60)

    for error_level in sorted(by_error.keys()):
        cos_vals = by_error[error_level]['cosine']
        euc_vals = by_error[error_level]['euclidean']

        cos_mean = np.mean(cos_vals)
        cos_std = np.std(cos_vals)
        euc_mean = np.mean(euc_vals)
        euc_std = np.std(euc_vals)

        print(f"{error_level}%{'':<7} {cos_mean:.4f} ± {cos_std:.4f}{'':<15} "
              f"{euc_mean:.4f} ± {euc_std:.4f}")

    print("=" * 60)


def main():
    """Run complete analysis pipeline."""
    print("=" * 60)
    print("REAL SEMANTIC DRIFT ANALYSIS")
    print("Using Sentence-BERT Embeddings with Real Claude Agent Data")
    print("=" * 60)
    print()

    # Paths
    input_path = 'results/experiments/real_pipeline_results.json'
    output_path = 'results/analysis/semantic_drift.csv'

    # Step 1: Load real results
    results = load_real_results(input_path)

    # Step 2: Compute semantic distances
    analysis_results = compute_semantic_distances(results)

    # Step 3: Save to CSV
    save_to_csv(analysis_results, output_path)

    # Step 4: Print summary
    print_summary(analysis_results)

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review: results/analysis/semantic_drift.csv")
    print("  2. Generate graphs: python3 generate_real_graphs.py")
    print("  3. View statistical analysis: results/graphs/statistical_analysis.txt")
    print()


if __name__ == "__main__":
    main()
