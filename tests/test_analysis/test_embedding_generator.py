"""Tests for embedding_generator module."""

import pytest
import numpy as np

from src.analysis.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    def test_initialization_default_model(self):
        """Test initialization with default model."""
        # Arrange & Act
        generator = EmbeddingGenerator()

        # Assert
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.model is not None
        assert generator.device is not None

    def test_initialization_custom_model(self):
        """Test initialization with custom model name."""
        # Arrange
        model_name = "all-MiniLM-L6-v2"

        # Act
        generator = EmbeddingGenerator(model_name=model_name)

        # Assert
        assert generator.model_name == model_name
        assert generator.model is not None

    def test_generate_embedding_simple_text(self):
        """Test generating embedding for simple text."""
        # Arrange
        generator = EmbeddingGenerator()
        text = "Hello world"

        # Act
        embedding = generator.generate_embedding(text)

        # Assert
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 dimension

    def test_generate_embedding_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        # Arrange
        generator = EmbeddingGenerator()

        # Act & Assert
        with pytest.raises(ValueError, match="Text cannot be empty"):
            generator.generate_embedding("")

    def test_generate_embedding_whitespace_only_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        # Arrange
        generator = EmbeddingGenerator()

        # Act & Assert
        with pytest.raises(ValueError, match="Text cannot be empty"):
            generator.generate_embedding("   ")

    def test_generate_embedding_reproducibility(self):
        """Test that same text produces same embedding."""
        # Arrange
        generator = EmbeddingGenerator()
        text = "This is a test sentence"

        # Act
        embedding1 = generator.generate_embedding(text)
        embedding2 = generator.generate_embedding(text)

        # Assert
        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_generate_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        # Arrange
        generator = EmbeddingGenerator()
        text1 = "Hello world"
        text2 = "Goodbye universe"

        # Act
        embedding1 = generator.generate_embedding(text1)
        embedding2 = generator.generate_embedding(text2)

        # Assert
        assert not np.array_equal(embedding1, embedding2)
        # Embeddings should be different but similar in magnitude
        assert embedding1.shape == embedding2.shape

    def test_generate_embeddings_batch_simple(self):
        """Test batch embedding generation."""
        # Arrange
        generator = EmbeddingGenerator()
        texts = ["Hello world", "Test sentence", "Another example"]

        # Act
        embeddings = generator.generate_embeddings_batch(
            texts, show_progress=False
        )

        # Assert
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_generate_embeddings_batch_single_text(self):
        """Test batch generation with single text."""
        # Arrange
        generator = EmbeddingGenerator()
        texts = ["Single text"]

        # Act
        embeddings = generator.generate_embeddings_batch(
            texts, show_progress=False
        )

        # Assert
        assert embeddings.shape == (1, 384)

    def test_generate_embeddings_batch_custom_batch_size(self):
        """Test batch generation with custom batch size."""
        # Arrange
        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        # Act
        embeddings = generator.generate_embeddings_batch(
            texts, batch_size=2, show_progress=False
        )

        # Assert
        assert embeddings.shape == (5, 384)

    def test_generate_embeddings_batch_matches_individual(self):
        """Test that batch generation matches individual generation."""
        # Arrange
        generator = EmbeddingGenerator()
        texts = ["Test 1", "Test 2"]

        # Act
        batch_embeddings = generator.generate_embeddings_batch(
            texts, show_progress=False
        )
        individual1 = generator.generate_embedding(texts[0])
        individual2 = generator.generate_embedding(texts[1])

        # Assert
        np.testing.assert_array_almost_equal(
            batch_embeddings[0], individual1, decimal=5
        )
        np.testing.assert_array_almost_equal(
            batch_embeddings[1], individual2, decimal=5
        )

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        # Arrange
        generator = EmbeddingGenerator()

        # Act
        dimension = generator.get_embedding_dimension()

        # Assert
        assert isinstance(dimension, int)
        assert dimension == 384  # all-MiniLM-L6-v2

    def test_embedding_normalized(self):
        """Test that embeddings are normalized (unit vectors)."""
        # Arrange
        generator = EmbeddingGenerator()
        text = "Test normalization"

        # Act
        embedding = generator.generate_embedding(text)
        norm = np.linalg.norm(embedding)

        # Assert
        # sentence-transformers embeddings are L2-normalized
        assert abs(norm - 1.0) < 0.01

    def test_long_text_embedding(self):
        """Test embedding generation for long text."""
        # Arrange
        generator = EmbeddingGenerator()
        long_text = " ".join(["word"] * 100)

        # Act
        embedding = generator.generate_embedding(long_text)

        # Assert
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_special_characters_in_text(self):
        """Test embedding with special characters."""
        # Arrange
        generator = EmbeddingGenerator()
        text = "Hello! How are you? I'm fine, thanks."

        # Act
        embedding = generator.generate_embedding(text)

        # Assert
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_multilingual_text(self):
        """Test embedding with non-English text."""
        # Arrange
        generator = EmbeddingGenerator()
        # French text
        text = "Bonjour le monde"

        # Act
        embedding = generator.generate_embedding(text)

        # Assert
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
