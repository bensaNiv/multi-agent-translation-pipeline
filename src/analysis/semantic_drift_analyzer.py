"""Analyze semantic drift in translation pipeline results."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .embedding_generator import EmbeddingGenerator
from .distance_calculator import calculate_both_distances

logger = logging.getLogger(__name__)


class SemanticDriftAnalyzer:
    """Analyze semantic drift between original and translated texts.

    Uses sentence embeddings to measure how much meaning is lost
    or changed through the translation pipeline.

    Attributes:
        embedding_generator: Generator for creating embeddings

    Example:
        >>> analyzer = SemanticDriftAnalyzer()
        >>> df = analyzer.analyze_results(
        ...     "results/experiments/pipeline_results.json"
        ... )
        >>> "cosine_distance" in df.columns
        True
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize analyzer with embedding model.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.embedding_generator = EmbeddingGenerator(model_name=model_name)

    def analyze_results(self, results_path: str) -> pd.DataFrame:
        """Analyze pipeline results and compute semantic distances.

        Args:
            results_path: Path to pipeline results JSON file

        Returns:
            DataFrame with columns: sentence_id, error_level,
                                   cosine_distance, euclidean_distance

        Raises:
            FileNotFoundError: If results file doesn't exist
            ValueError: If results format is invalid

        Example:
            >>> analyzer = SemanticDriftAnalyzer()
            >>> df = analyzer.analyze_results("results/test.json")
            >>> "error_level" in df.columns
            True
        """
        results_file = Path(results_path)
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        analysis_data = []

        logger.info(f"Analyzing {len(results)} results...")

        for result in results:
            original_text = result.get("original_text")
            final_text = result.get("final_english_text")

            if not original_text or not final_text:
                logger.warning(
                    f"Skipping result {result.get('sentence_id')}: "
                    "missing text"
                )
                continue

            original_emb = self.embedding_generator.generate_embedding(
                original_text
            )
            final_emb = self.embedding_generator.generate_embedding(final_text)

            cosine_dist, euclidean_dist = calculate_both_distances(
                original_emb, final_emb
            )

            analysis_data.append(
                {
                    "sentence_id": result.get("sentence_id"),
                    "error_level": result.get("error_level"),
                    "cosine_distance": cosine_dist,
                    "euclidean_distance": euclidean_dist,
                    "original_text": original_text,
                    "final_text": final_text,
                }
            )

        df = pd.DataFrame(analysis_data)
        logger.info(f"✓ Analyzed {len(df)} results")

        return df

    def save_analysis(self, df: pd.DataFrame, output_path: str):
        """Save analysis results to CSV file.

        Args:
            df: DataFrame with analysis results
            output_path: Path to save CSV file

        Example:
            >>> analyzer = SemanticDriftAnalyzer()
            >>> df = pd.DataFrame({"error_level": [0, 25]})
            >>> analyzer.save_analysis(df, "results/analysis.csv")
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"✓ Saved analysis to {output_path}")
