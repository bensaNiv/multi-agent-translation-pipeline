"""Integration tests for the complete translation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from src.controller.pipeline_controller import TranslationPipelineController
from src.input_generator.sentence_generator import generate_baseline_sentences
from src.input_generator.error_injector import inject_errors
from src.analysis.semantic_drift_analyzer import SemanticDriftAnalyzer
from src.visualization.graph_generator import generate_all_graphs


class TestPipelineIntegration:
    """Integration tests for complete end-to-end workflows."""

    def test_single_sentence_pipeline_execution(self):
        """Test executing pipeline for single sentence."""
        # Arrange
        controller = TranslationPipelineController()
        text = "This is a test sentence for the translation pipeline"

        # Act
        result = controller.execute_pipeline(
            original_text=text, error_level=0.0, sentence_id=1
        )

        # Assert
        assert result["sentence_id"] == 1
        assert result["original_text"] == text
        assert result["error_level"] == 0.0
        assert "final_english_text" in result
        assert "intermediate_translations" in result
        assert len(result["intermediate_translations"]) == 3

    def test_pipeline_with_multiple_error_levels(self):
        """Test pipeline with different error levels."""
        # Arrange
        controller = TranslationPipelineController()
        text = "The quick brown fox jumps over the lazy dog every single day"
        error_levels = [0, 10, 25, 50]

        # Act
        results = []
        for idx, level in enumerate(error_levels):
            errored_text = inject_errors(text, level, seed=42 + idx)
            result = controller.execute_pipeline(
                original_text=errored_text,
                error_level=level,
                sentence_id=idx,
            )
            results.append(result)

        # Assert
        assert len(results) == len(error_levels)
        for result, expected_level in zip(results, error_levels):
            assert result["error_level"] == expected_level
            assert result["final_english_text"] is not None

    def test_batch_execution(self):
        """Test executing batch of inputs."""
        # Arrange
        controller = TranslationPipelineController()
        inputs = [
            {
                "text": "First test sentence with enough words to meet minimum",
                "error_level": 0,
                "sentence_id": 1,
            },
            {
                "text": "Second test sentence with enough words to meet minimum",
                "error_level": 25,
                "sentence_id": 2,
            },
            {
                "text": "Third test sentence with enough words to meet minimum",
                "error_level": 50,
                "sentence_id": 3,
            },
        ]

        # Act
        results = controller.execute_batch(inputs)

        # Assert
        assert len(results) == 3
        assert results[0]["error_level"] == 0
        assert results[1]["error_level"] == 25
        assert results[2]["error_level"] == 50

    def test_end_to_end_input_to_results(self):
        """Test complete workflow from input generation to results."""
        # Arrange
        controller = TranslationPipelineController()

        # Act - Generate input
        base_sentences = generate_baseline_sentences()
        base_sentence = base_sentences[0]
        assert len(base_sentence.split()) >= 15

        # Act - Inject errors
        errored_text = inject_errors(base_sentence, 25.0, seed=42)

        # Act - Run pipeline
        result = controller.execute_pipeline(
            original_text=errored_text, error_level=25.0, sentence_id=1
        )

        # Assert
        assert result["original_text"] == errored_text
        assert result["error_level"] == 25.0
        assert result["final_english_text"] is not None
        assert len(result["metadata"]["agents_executed"]) == 3

    def test_save_and_load_results(self):
        """Test saving results and loading them back."""
        # Arrange
        controller = TranslationPipelineController()
        text = "Test sentence with enough words to meet minimum requirements here"

        # Act - Execute and save
        controller.execute_pipeline(text, 0.0, sentence_id=1)
        controller.execute_pipeline(text, 25.0, sentence_id=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.json"
            controller.save_results(str(output_path))

            # Assert - File exists
            assert output_path.exists()

            # Assert - Can load back
            with open(output_path, "r", encoding="utf-8") as f:
                loaded_results = json.load(f)

            assert len(loaded_results) == 2
            assert loaded_results[0]["sentence_id"] == 1
            assert loaded_results[1]["sentence_id"] == 2

    def test_end_to_end_pipeline_to_analysis(self):
        """Test complete workflow from pipeline to analysis."""
        # Arrange
        controller = TranslationPipelineController()
        text = "This is a comprehensive test sentence with many words included"

        # Act - Execute pipeline
        controller.execute_pipeline(text, 0.0, sentence_id=1)
        controller.execute_pipeline(text, 25.0, sentence_id=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save results
            results_path = Path(temp_dir) / "results.json"
            controller.save_results(str(results_path))

            # Analyze results
            analyzer = SemanticDriftAnalyzer()
            df = analyzer.analyze_results(str(results_path))

            # Assert
            assert len(df) == 2
            assert "cosine_distance" in df.columns
            assert "euclidean_distance" in df.columns
            assert df["error_level"].tolist() == [0.0, 25.0]

    def test_end_to_end_pipeline_to_visualization(self):
        """Test complete workflow from pipeline to graph generation."""
        # Arrange
        controller = TranslationPipelineController()
        text = "Complete test sentence for visualization with enough words included"

        # Act - Execute pipeline for multiple error levels
        error_levels = [0, 25, 50]
        for idx, level in enumerate(error_levels):
            errored = inject_errors(text, level, seed=42 + idx)
            controller.execute_pipeline(errored, level, sentence_id=idx)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save results
            results_path = Path(temp_dir) / "results.json"
            controller.save_results(str(results_path))

            # Analyze
            analyzer = SemanticDriftAnalyzer()
            df = analyzer.analyze_results(str(results_path))

            # Generate graphs
            graphs_dir = Path(temp_dir) / "graphs"
            generate_all_graphs(df, output_dir=str(graphs_dir))

            # Assert
            assert (graphs_dir / "cosine_distance.png").exists()
            assert (graphs_dir / "euclidean_distance.png").exists()
            assert (graphs_dir / "both_metrics.png").exists()

    def test_complete_experiment_workflow(self):
        """Test the complete experiment from start to finish."""
        # Arrange
        controller = TranslationPipelineController()
        error_levels = [0, 10, 20, 25, 30, 40, 50]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Act - Generate sentences and run experiments
            base_sentences = generate_baseline_sentences()
            base_sentence = base_sentences[0]

            for level in error_levels:
                errored_text = inject_errors(base_sentence, level, seed=42)
                controller.execute_pipeline(
                    original_text=errored_text,
                    error_level=level,
                    sentence_id=level,
                )

            # Save results
            results_path = Path(temp_dir) / "results.json"
            controller.save_results(str(results_path))

            # Analyze
            analyzer = SemanticDriftAnalyzer()
            df = analyzer.analyze_results(str(results_path))

            # Save analysis
            analysis_path = Path(temp_dir) / "analysis.csv"
            analyzer.save_analysis(df, str(analysis_path))

            # Generate graphs
            graphs_dir = Path(temp_dir) / "graphs"
            generate_all_graphs(df, output_dir=str(graphs_dir))

            # Assert - All outputs exist
            assert results_path.exists()
            assert analysis_path.exists()
            assert (graphs_dir / "cosine_distance.png").exists()

            # Assert - Results have correct structure
            assert len(df) == len(error_levels)
            assert set(df["error_level"].unique()) == set(error_levels)

            # Assert - Analysis has expected columns
            loaded_analysis = pd.read_csv(analysis_path)
            assert "cosine_distance" in loaded_analysis.columns
            assert "euclidean_distance" in loaded_analysis.columns

    def test_error_handling_in_pipeline(self):
        """Test that pipeline handles errors gracefully."""
        # Arrange
        controller = TranslationPipelineController()

        # Act & Assert - Empty text should raise error
        with pytest.raises(Exception):
            controller.execute_pipeline("", 0.0)

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        # Arrange
        controller = TranslationPipelineController()
        text = "Test sentence with enough words for minimum requirements"

        # Act
        result = controller.execute_pipeline(text, 25.0, sentence_id=1)

        # Assert
        assert "metadata" in result
        assert "timestamp" in result["metadata"]
        assert "agents_executed" in result["metadata"]
        assert len(result["metadata"]["agents_executed"]) == 3
        assert result["metadata"]["agents_executed"] == [
            "en_to_fr",
            "fr_to_he",
            "he_to_en",
        ]

    def test_batch_partial_failure_handling(self):
        """Test that batch execution continues after individual failures."""
        # Arrange
        controller = TranslationPipelineController()
        inputs = [
            {"text": "Valid sentence with enough words here", "error_level": 0},
            {"text": "", "error_level": 25},  # This should fail
            {"text": "Another valid sentence with words", "error_level": 50},
        ]

        # Act
        results = controller.execute_batch(inputs)

        # Assert - Should have 2 successful results despite one failure
        assert len(results) == 2

    def test_results_accumulation(self):
        """Test that results accumulate in controller."""
        # Arrange
        controller = TranslationPipelineController()
        text = "Test sentence for results accumulation with enough words"

        # Act
        assert len(controller.results) == 0

        controller.execute_pipeline(text, 0.0, sentence_id=1)
        assert len(controller.results) == 1

        controller.execute_pipeline(text, 25.0, sentence_id=2)
        assert len(controller.results) == 2

        controller.execute_pipeline(text, 50.0, sentence_id=3)
        assert len(controller.results) == 3

        # Assert
        assert controller.results[0]["sentence_id"] == 1
        assert controller.results[1]["sentence_id"] == 2
        assert controller.results[2]["sentence_id"] == 3
