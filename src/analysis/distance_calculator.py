"""Calculate semantic distances between sentence embeddings."""

from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_distance(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> float:
    """Calculate cosine distance between two embeddings.

    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where 0 = identical, 2 = opposite

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine distance (0 = most similar)

    Raises:
        ValueError: If embeddings have different dimensions

    Example:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([1, 0, 0])
        >>> calculate_cosine_distance(emb1, emb2)
        0.0

    References:
        Best practice for sentence-transformers:
        https://datascience.stackexchange.com/questions/27726
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embeddings must have same shape: "
            f"{embedding1.shape} vs {embedding2.shape}"
        )

    emb1_2d = embedding1.reshape(1, -1)
    emb2_2d = embedding2.reshape(1, -1)

    similarity = cosine_similarity(emb1_2d, emb2_2d)[0, 0]
    distance = 1 - similarity

    return float(distance)


def calculate_euclidean_distance(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> float:
    """Calculate Euclidean distance between two embeddings.

    Range: [0, ∞) where 0 = identical

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Euclidean distance (0 = most similar)

    Raises:
        ValueError: If embeddings have different dimensions

    Example:
        >>> emb1 = np.array([0, 0, 0])
        >>> emb2 = np.array([3, 4, 0])
        >>> calculate_euclidean_distance(emb1, emb2)
        5.0
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embeddings must have same shape: "
            f"{embedding1.shape} vs {embedding2.shape}"
        )

    distance = np.linalg.norm(embedding1 - embedding2)
    return float(distance)


def calculate_both_distances(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> Tuple[float, float]:
    """Calculate both cosine and Euclidean distances.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Tuple of (cosine_distance, euclidean_distance)

    Raises:
        ValueError: If embeddings have different dimensions

    Example:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([0, 1, 0])
        >>> cos_dist, euc_dist = calculate_both_distances(emb1, emb2)
        >>> isinstance(cos_dist, float)
        True
        >>> isinstance(euc_dist, float)
        True
    """
    cosine_dist = calculate_cosine_distance(embedding1, embedding2)
    euclidean_dist = calculate_euclidean_distance(embedding1, embedding2)
    return cosine_dist, euclidean_dist
