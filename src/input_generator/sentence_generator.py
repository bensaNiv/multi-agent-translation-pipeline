"""Generate baseline sentences for translation experiments."""

from typing import List


def generate_baseline_sentences() -> List[str]:
    """Generate baseline English sentences for experiments.

    Returns:
        List of sentences, each with ≥15 words

    Raises:
        ValueError: If any sentence has fewer than 15 words

    Example:
        >>> sentences = generate_baseline_sentences()
        >>> all(len(s.split()) >= 15 for s in sentences)
        True
    """
    sentences = [
        (
            "The quick brown fox jumps over the lazy dog while "
            "the sun shines brightly in the clear blue sky above"
        ),
        (
            "Scientists have recently discovered that machine "
            "learning algorithms can significantly improve "
            "translation quality when properly trained on diverse "
            "datasets"
        ),
        (
            "In the modern world of technology and innovation, "
            "artificial intelligence continues to transform how "
            "we communicate across different languages and cultures"
        ),
        (
            "The ancient library contained thousands of manuscripts "
            "written in various languages that scholars spent decades "
            "attempting to translate accurately"
        ),
        (
            "Climate change poses significant challenges for future "
            "generations, requiring immediate action from governments "
            "and citizens around the world today"
        ),
    ]

    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count < 15:
            raise ValueError(
                f"Sentence must have ≥15 words, got {word_count}"
            )

    return sentences


def validate_sentence(sentence: str, min_words: int = 15) -> bool:
    """Validate sentence meets minimum word requirement.

    Args:
        sentence: Sentence to validate
        min_words: Minimum required word count

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_sentence("This is a test sentence", 5)
        True
        >>> validate_sentence("Short", 10)
        False
    """
    return len(sentence.split()) >= min_words
