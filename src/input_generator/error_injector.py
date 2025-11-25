"""Inject spelling errors into text at specified rates."""

import random
from typing import List, Tuple


def inject_errors(text: str, error_rate: float, seed: int = 42) -> str:
    """Inject spelling errors into text at specified rate.

    Introduces realistic spelling errors by:
    - Character substitution (e.g., 'e' → 'a')
    - Character omission (e.g., 'hello' → 'helo')
    - Character duplication (e.g., 'hello' → 'helllo')

    Args:
        text: Original text
        error_rate: Percentage of words to modify (0-50)
        seed: Random seed for reproducibility

    Returns:
        Text with injected errors

    Raises:
        ValueError: If error_rate not in valid range [0, 50]

    Example:
        >>> inject_errors("hello world", 50.0, seed=42)
        'helo world'
    """
    if not 0 <= error_rate <= 50:
        raise ValueError(
            f"error_rate must be in [0, 50], got {error_rate}"
        )

    if error_rate == 0:
        return text

    random.seed(seed)
    words = text.split()
    num_errors = int(len(words) * (error_rate / 100))

    error_indices = random.sample(range(len(words)), num_errors)

    for idx in error_indices:
        words[idx] = _corrupt_word(words[idx])

    return " ".join(words)


def _corrupt_word(word: str) -> str:
    """Corrupt a single word with a realistic spelling error.

    Args:
        word: Word to corrupt

    Returns:
        Corrupted word
    """
    if len(word) <= 2:
        return word

    error_type = random.choice(["substitute", "omit", "duplicate"])
    pos = random.randint(1, len(word) - 2)

    if error_type == "substitute":
        replacements = {
            "a": "sqa",
            "e": "rwd",
            "i": "uko",
            "o": "pil",
            "t": "rfy",
            "n": "bhm",
            "s": "adw",
        }
        char = word[pos].lower()
        replacement = random.choice(replacements.get(char, "x"))
        word = word[:pos] + replacement + word[pos + 1 :]

    elif error_type == "omit":
        word = word[:pos] + word[pos + 1 :]

    elif error_type == "duplicate":
        word = word[:pos] + word[pos] + word[pos:]

    return word


def generate_error_variants(
    text: str, error_levels: List[float]
) -> List[Tuple[str, float]]:
    """Generate multiple versions of text with different error rates.

    Args:
        text: Original text
        error_levels: List of error percentages to generate

    Returns:
        List of (corrupted_text, error_level) tuples

    Example:
        >>> variants = generate_error_variants(
        ...     "hello world",
        ...     [0, 25, 50]
        ... )
        >>> len(variants)
        3
        >>> variants[0][1]
        0.0
    """
    variants = []
    for level in error_levels:
        corrupted = inject_errors(text, level, seed=int(level * 100))
        variants.append((corrupted, level))
    return variants
