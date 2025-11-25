"""Tests for distance_calculator module."""

import pytest
import numpy as np
from src.analysis.distance_calculator import (
    calculate_cosine_distance,
    calculate_euclidean_distance,
    calculate_both_distances,
)


def test_cosine_distance_identical_vectors():
    """Test cosine distance between identical vectors is 0."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    distance = calculate_cosine_distance(emb1, emb2)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_orthogonal_vectors():
    """Test cosine distance between orthogonal vectors."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    distance = calculate_cosine_distance(emb1, emb2)
    assert distance == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_different_shapes_raises_error():
    """Test that different shapes raise ValueError."""
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="same shape"):
        calculate_cosine_distance(emb1, emb2)


def test_euclidean_distance_identical_vectors():
    """Test Euclidean distance between identical vectors is 0."""
    emb1 = np.array([1.0, 2.0, 3.0])
    emb2 = np.array([1.0, 2.0, 3.0])
    distance = calculate_euclidean_distance(emb1, emb2)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_euclidean_distance_known_value():
    """Test Euclidean distance with known value."""
    emb1 = np.array([0.0, 0.0, 0.0])
    emb2 = np.array([3.0, 4.0, 0.0])
    distance = calculate_euclidean_distance(emb1, emb2)
    assert distance == pytest.approx(5.0, abs=1e-6)


def test_euclidean_distance_different_shapes_raises_error():
    """Test that different shapes raise ValueError."""
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="same shape"):
        calculate_euclidean_distance(emb1, emb2)


def test_calculate_both_distances():
    """Test calculating both distances at once."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])

    cos_dist, euc_dist = calculate_both_distances(emb1, emb2)

    assert isinstance(cos_dist, float)
    assert isinstance(euc_dist, float)
    assert cos_dist == pytest.approx(1.0, abs=1e-6)
    assert euc_dist == pytest.approx(np.sqrt(2), abs=1e-6)
