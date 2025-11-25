"""Run semantic drift analysis on pipeline results.

This script computes semantic distances between original and final texts.
For demonstration, it uses a simplified embedding approach based on word overlap.
"""

import json
import csv
from pathlib import Path
from collections import Counter
import math


def simple_word_embedding(text):
    """Create simple word-based embedding (bag of words).

    This is a lightweight alternative to sentence-transformers for demonstration.
    In production, use actual sentence embeddings.
    """
    words = text.lower().split()
    word_counts = Counter(words)
    return word_counts


def cosine_distance_simple(embedding1, embedding2):
    """Calculate cosine distance between two word-based embeddings."""
    # Get all unique words
    all_words = set(embedding1.keys()) | set(embedding2.keys())

    # Create vectors
    vec1 = [embedding1.get(word, 0) for word in all_words]
    vec2 = [embedding2.get(word, 0) for word in all_words]

    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 1.0  # Maximum distance if either vector is zero

    similarity = dot_product / (mag1 * mag2)
    distance = 1 - similarity
    return distance


def euclidean_distance_simple(embedding1, embedding2):
    """Calculate Euclidean distance between two word-based embeddings."""
    all_words = set(embedding1.keys()) | set(embedding2.keys())

    vec1 = [embedding1.get(word, 0) for word in all_words]
    vec2 = [embedding2.get(word, 0) for word in all_words]

    squared_diffs = [(a - b) ** 2 for a, b in zip(vec1, vec2)]
    distance = math.sqrt(sum(squared_diffs))
    return distance


def main():
    """Run the semantic drift analysis."""
    print("=" * 60)
    print("Semantic Drift Analysis")
    print("=" * 60)
    print()

    # Load pipeline results
    print("📖 Loading pipeline results...")
    results_path = 'results/experiments/pipeline_results.json'

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"✓ Loaded {len(results)} results")
    print()

    # Analyze each result
    print("🔍 Computing semantic distances...")
    analysis_data = []

    for i, result in enumerate(results, 1):
        original_text = result['original_text']
        final_text = result['final_english_text']

        # Generate embeddings (simplified)
        orig_emb = simple_word_embedding(original_text)
        final_emb = simple_word_embedding(final_text)

        # Calculate distances
        cos_dist = cosine_distance_simple(orig_emb, final_emb)
        euc_dist = euclidean_distance_simple(orig_emb, final_emb)

        analysis_data.append({
            'sentence_id': result['sentence_id'],
            'error_level': result['error_level'],
            'cosine_distance': cos_dist,
            'euclidean_distance': euc_dist,
            'original_text': original_text,
            'final_text': final_text
        })

        if i % 7 == 0:
            print(f"  Processed {i}/{len(results)} results...")

    print(f"✓ Analyzed all {len(results)} results")
    print()

    # Save analysis results
    print("💾 Saving analysis results...")
    output_path = Path('results/analysis/semantic_drift.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sentence_id', 'error_level',
            'cosine_distance', 'euclidean_distance',
            'original_text', 'final_text'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_data)

    print(f"✓ Saved analysis to {output_path}")
    print()

    # Display statistics
    print("📊 Analysis Statistics:")
    print("-" * 60)

    # Group by error level
    error_levels = {}
    for item in analysis_data:
        level = item['error_level']
        if level not in error_levels:
            error_levels[level] = {'cosine': [], 'euclidean': []}
        error_levels[level]['cosine'].append(item['cosine_distance'])
        error_levels[level]['euclidean'].append(item['euclidean_distance'])

    print(f"{'Error %':<10} {'Avg Cosine':<15} {'Avg Euclidean':<15}")
    print("-" * 60)
    for level in sorted(error_levels.keys()):
        avg_cos = sum(error_levels[level]['cosine']) / len(error_levels[level]['cosine'])
        avg_euc = sum(error_levels[level]['euclidean']) / len(error_levels[level]['euclidean'])
        print(f"{level:<10} {avg_cos:<15.4f} {avg_euc:<15.4f}")

    print()
    print("=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    print()
    print("Next step:")
    print("  Run visualization: python3 run_visualization.py")


if __name__ == "__main__":
    main()
