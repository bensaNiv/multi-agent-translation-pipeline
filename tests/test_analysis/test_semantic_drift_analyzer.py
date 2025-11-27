"""Tests for semantic_drift_analyzer module."""

import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from src.analysis.semantic_drift_analyzer import SemanticDriftAnalyzer


class TestSemanticDriftAnalyzer:
    """Test suite for SemanticDriftAnalyzer class."""

    def test_initialization_default_model(self):
        """Test initialization with default model."""
        # Arrange & Act
        analyzer = SemanticDriftAnalyzer()

        # Assert
        assert analyzer.embedding_generator is not None
        assert analyzer.embedding_generator.model_name == "all-MiniLM-L6-v2"

    def test_initialization_custom_model(self):
        """Test initialization with custom model name."""
        # Arrange
        model_name = "all-MiniLM-L6-v2"

        # Act
        analyzer = SemanticDriftAnalyzer(model_name=model_name)

        # Assert
        assert analyzer.embedding_generator.model_name == model_name

    def test_analyze_results_with_valid_data(self):
        """Test analyzing results with valid JSON data."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_data = [
            {
                "sentence_id": "test_1",
                "error_level": 0,
                "original_text": "Hello world",
                "final_english_text": "Hello world",
            },
            {
                "sentence_id": "test_2",
                "error_level": 25,
                "original_text": "This is a test",
                "final_english_text": "This is a trial",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Act
            df = analyzer.analyze_results(temp_path)

            # Assert
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "sentence_id" in df.columns
            assert "error_level" in df.columns
            assert "cosine_distance" in df.columns
            assert "euclidean_distance" in df.columns
            assert "original_text" in df.columns
            assert "final_text" in df.columns

        finally:
            Path(temp_path).unlink()

    def test_analyze_results_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        nonexistent_path = "/tmp/nonexistent_file_12345.json"

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            analyzer.analyze_results(nonexistent_path)

    def test_analyze_results_distance_values(self):
        """Test that distance values are computed correctly."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_data = [
            {
                "sentence_id": "identical",
                "error_level": 0,
                "original_text": "Hello world",
                "final_english_text": "Hello world",
            },
            {
                "sentence_id": "different",
                "error_level": 50,
                "original_text": "Hello world",
                "final_english_text": "Goodbye universe",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Act
            df = analyzer.analyze_results(temp_path)

            # Assert
            identical_row = df[df["sentence_id"] == "identical"].iloc[0]
            different_row = df[df["sentence_id"] == "different"].iloc[0]

            # Identical texts should have very low distance
            assert identical_row["cosine_distance"] < 0.01
            assert identical_row["euclidean_distance"] < 0.1

            # Different texts should have higher distance
            assert different_row["cosine_distance"] > identical_row["cosine_distance"]
            assert (
                different_row["euclidean_distance"]
                > identical_row["euclidean_distance"]
            )

        finally:
            Path(temp_path).unlink()

    def test_analyze_results_skips_missing_text(self):
        """Test that results with missing text are skipped."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_data = [
            {
                "sentence_id": "valid",
                "error_level": 0,
                "original_text": "Hello world",
                "final_english_text": "Hello world",
            },
            {
                "sentence_id": "missing_original",
                "error_level": 25,
                "original_text": "",
                "final_english_text": "Test",
            },
            {
                "sentence_id": "missing_final",
                "error_level": 25,
                "original_text": "Test",
                "final_english_text": "",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Act
            df = analyzer.analyze_results(temp_path)

            # Assert
            assert len(df) == 1
            assert df.iloc[0]["sentence_id"] == "valid"

        finally:
            Path(temp_path).unlink()

    def test_analyze_results_all_error_levels(self):
        """Test analyzing results with multiple error levels."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        error_levels = [0, 10, 20, 25, 30, 40, 50]
        test_data = [
            {
                "sentence_id": f"test_{level}",
                "error_level": level,
                "original_text": f"Test sentence at error level {level}",
                "final_english_text": f"Test sentence at error level {level}",
            }
            for level in error_levels
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Act
            df = analyzer.analyze_results(temp_path)

            # Assert
            assert len(df) == len(error_levels)
            assert set(df["error_level"].unique()) == set(error_levels)

        finally:
            Path(temp_path).unlink()

    def test_save_analysis_creates_file(self):
        """Test that save_analysis creates CSV file."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_df = pd.DataFrame(
            {
                "sentence_id": ["test_1", "test_2"],
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
                "euclidean_distance": [1.0, 2.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "analysis.csv"

            # Act
            analyzer.save_analysis(test_df, str(output_path))

            # Assert
            assert output_path.exists()
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == 2
            assert "sentence_id" in loaded_df.columns

    def test_save_analysis_creates_parent_directory(self):
        """Test that save_analysis creates parent directories."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_df = pd.DataFrame({"error_level": [0, 25]})

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir) / "nested" / "directories" / "analysis.csv"
            )

            # Act
            analyzer.save_analysis(test_df, str(output_path))

            # Assert
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_analysis_preserves_data(self):
        """Test that save_analysis preserves all data correctly."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_df = pd.DataFrame(
            {
                "sentence_id": ["test_1", "test_2"],
                "error_level": [0, 25],
                "cosine_distance": [0.123456, 0.789012],
                "euclidean_distance": [1.234567, 2.345678],
                "original_text": ["Original 1", "Original 2"],
                "final_text": ["Final 1", "Final 2"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "analysis.csv"

            # Act
            analyzer.save_analysis(test_df, str(output_path))
            loaded_df = pd.read_csv(output_path)

            # Assert
            pd.testing.assert_frame_equal(
                test_df, loaded_df, check_dtype=False
            )

    def test_end_to_end_workflow(self):
        """Test complete workflow from analyze to save."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_data = [
            {
                "sentence_id": "test_1",
                "error_level": 0,
                "original_text": "The quick brown fox",
                "final_english_text": "The quick brown fox",
            },
            {
                "sentence_id": "test_2",
                "error_level": 25,
                "original_text": "The quick brown fox",
                "final_english_text": "The fast brown dog",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_path = Path(temp_dir) / "results.json"
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            output_path = Path(temp_dir) / "analysis.csv"

            # Act
            df = analyzer.analyze_results(str(input_path))
            analyzer.save_analysis(df, str(output_path))

            # Assert
            assert output_path.exists()
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == 2
            assert loaded_df["error_level"].tolist() == [0, 25]

    def test_analyze_results_unicode_handling(self):
        """Test that analyzer handles unicode text correctly."""
        # Arrange
        analyzer = SemanticDriftAnalyzer()
        test_data = [
            {
                "sentence_id": "hebrew",
                "error_level": 0,
                "original_text": "שלום עולם",  # Hebrew text
                "final_english_text": "Hello world",
            }
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            # Act
            df = analyzer.analyze_results(temp_path)

            # Assert
            assert len(df) == 1
            assert df.iloc[0]["original_text"] == "שלום עולם"

        finally:
            Path(temp_path).unlink()
