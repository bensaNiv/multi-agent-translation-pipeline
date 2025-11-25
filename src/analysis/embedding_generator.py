"""Generate sentence embeddings for semantic similarity analysis."""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Sentence-BERT models.

    Uses pre-trained sentence-transformers models to generate
    high-quality sentence embeddings for semantic similarity analysis.

    Attributes:
        model: Loaded SentenceTransformer model
        model_name: Name of the model being used
        device: Device for computation ('cpu' or 'cuda')

    Example:
        >>> generator = EmbeddingGenerator()
        >>> embedding = generator.generate_embedding("Hello world")
        >>> embedding.shape
        (384,)
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """Initialize embedding generator with specified model.

        Args:
            model_name: Name of sentence-transformers model to use
            device: Device to use ('cpu' or 'cuda'). Auto-detects if None.

        Recommended Models:
            - 'all-MiniLM-L6-v2': Fast, 384-dim (RECOMMENDED)
            - 'all-mpnet-base-v2': High quality, 768-dim
            - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual

        References:
            https://sbert.net/docs/pretrained_models.html
        """
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = self.model.device
        logger.info(f"✓ Model loaded on device: {self.device}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding vector

        Raises:
            ValueError: If text is empty

        Example:
            >>> gen = EmbeddingGenerator()
            >>> emb = gen.generate_embedding("Test")
            >>> isinstance(emb, np.ndarray)
            True
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (num_texts, embedding_dim)

        Example:
            >>> gen = EmbeddingGenerator()
            >>> texts = ["Test 1", "Test 2", "Test 3"]
            >>> embs = gen.generate_embeddings_batch(texts)
            >>> embs.shape[0] == 3
            True
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)

        Example:
            >>> gen = EmbeddingGenerator()
            >>> dim = gen.get_embedding_dimension()
            >>> dim == 384
            True
        """
        return self.model.get_sentence_embedding_dimension()
