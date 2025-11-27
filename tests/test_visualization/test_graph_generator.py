"""Tests for graph_generator module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.graph_generator import (
    plot_error_vs_distance,
    plot_both_metrics,
    generate_all_graphs,
)


class TestGraphGenerator:
    """Test suite for graph_generator module."""

    def test_plot_error_vs_distance_cosine(self):
        """Test plotting cosine distance."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25, 50],
                "cosine_distance": [0.1, 0.3, 0.5],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cosine.png"

            # Act
            plot_error_vs_distance(df, metric="cosine", output_path=str(output_path))

            # Assert
            assert output_path.exists()

    def test_plot_error_vs_distance_euclidean(self):
        """Test plotting Euclidean distance."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25, 50],
                "euclidean_distance": [1.0, 2.0, 3.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "euclidean.png"

            # Act
            plot_error_vs_distance(
                df, metric="euclidean", output_path=str(output_path)
            )

            # Assert
            assert output_path.exists()

    def test_plot_error_vs_distance_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        # Arrange
        df = pd.DataFrame({"error_level": [0], "cosine_distance": [0.1]})

        # Act & Assert
        with pytest.raises(ValueError, match="metric must be 'cosine' or 'euclidean'"):
            plot_error_vs_distance(df, metric="invalid")

    def test_plot_error_vs_distance_missing_column(self):
        """Test that missing column raises ValueError."""
        # Arrange
        df = pd.DataFrame({"error_level": [0, 25]})

        # Act & Assert
        with pytest.raises(ValueError, match="Column .* not found"):
            plot_error_vs_distance(df, metric="cosine")

    def test_plot_error_vs_distance_without_save(self):
        """Test plotting without saving (display only)."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25, 50],
                "cosine_distance": [0.1, 0.3, 0.5],
            }
        )

        # Act - should not raise error
        plot_error_vs_distance(df, metric="cosine", output_path=None)

        # Assert - just check it didn't crash
        # (matplotlib will close the figure)

    def test_plot_error_vs_distance_creates_parent_dir(self):
        """Test that parent directories are created."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "dir" / "plot.png"

            # Act
            plot_error_vs_distance(df, metric="cosine", output_path=str(output_path))

            # Assert
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_plot_error_vs_distance_with_std_dev(self):
        """Test plotting with standard deviation."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 0, 25, 25, 50, 50],
                "cosine_distance": [0.1, 0.12, 0.3, 0.32, 0.5, 0.52],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "plot.png"

            # Act
            plot_error_vs_distance(df, metric="cosine", output_path=str(output_path))

            # Assert
            assert output_path.exists()

    def test_plot_both_metrics(self):
        """Test plotting both metrics side by side."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25, 50],
                "cosine_distance": [0.1, 0.3, 0.5],
                "euclidean_distance": [1.0, 2.0, 3.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "both.png"

            # Act
            plot_both_metrics(df, output_path=str(output_path))

            # Assert
            assert output_path.exists()

    def test_plot_both_metrics_without_save(self):
        """Test plot_both_metrics without saving."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
                "euclidean_distance": [1.0, 2.0],
            }
        )

        # Act - should not raise error
        plot_both_metrics(df, output_path=None)

        # Assert - just check it didn't crash

    def test_plot_both_metrics_creates_parent_dir(self):
        """Test that plot_both_metrics creates parent directories."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
                "euclidean_distance": [1.0, 2.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "a" / "b" / "c" / "plot.png"

            # Act
            plot_both_metrics(df, output_path=str(output_path))

            # Assert
            assert output_path.exists()

    def test_generate_all_graphs(self):
        """Test generating all graphs at once."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25, 50],
                "cosine_distance": [0.1, 0.3, 0.5],
                "euclidean_distance": [1.0, 2.0, 3.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Act
            generate_all_graphs(df, output_dir=temp_dir)

            # Assert
            assert (Path(temp_dir) / "cosine_distance.png").exists()
            assert (Path(temp_dir) / "euclidean_distance.png").exists()
            assert (Path(temp_dir) / "both_metrics.png").exists()

    def test_generate_all_graphs_creates_output_dir(self):
        """Test that generate_all_graphs creates output directory."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
                "euclidean_distance": [1.0, 2.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "graphs" / "output"

            # Act
            generate_all_graphs(df, output_dir=str(output_dir))

            # Assert
            assert output_dir.exists()
            assert (output_dir / "cosine_distance.png").exists()

    def test_plot_with_single_error_level(self):
        """Test plotting with single error level."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [25],
                "cosine_distance": [0.3],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "single.png"

            # Act
            plot_error_vs_distance(df, metric="cosine", output_path=str(output_path))

            # Assert
            assert output_path.exists()

    def test_plot_with_many_error_levels(self):
        """Test plotting with many error levels."""
        # Arrange
        error_levels = list(range(0, 51, 5))
        df = pd.DataFrame(
            {
                "error_level": error_levels,
                "cosine_distance": [i * 0.01 for i in error_levels],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "many.png"

            # Act
            plot_error_vs_distance(df, metric="cosine", output_path=str(output_path))

            # Assert
            assert output_path.exists()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_error_vs_distance_calls_savefig(self, mock_savefig):
        """Test that savefig is called with correct parameters."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
            }
        )
        output_path = "/tmp/test_plot.png"

        # Act
        plot_error_vs_distance(df, metric="cosine", output_path=output_path)

        # Assert
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == Path(output_path)
        assert kwargs.get("dpi") == 300
        assert kwargs.get("bbox_inches") == "tight"

    @patch("matplotlib.pyplot.savefig")
    def test_plot_both_metrics_calls_savefig(self, mock_savefig):
        """Test that plot_both_metrics calls savefig correctly."""
        # Arrange
        df = pd.DataFrame(
            {
                "error_level": [0, 25],
                "cosine_distance": [0.1, 0.3],
                "euclidean_distance": [1.0, 2.0],
            }
        )
        output_path = "/tmp/test_both.png"

        # Act
        plot_both_metrics(df, output_path=output_path)

        # Assert
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == Path(output_path)
        assert kwargs.get("dpi") == 300
