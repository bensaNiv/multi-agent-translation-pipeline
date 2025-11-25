"""Tests for error_injector module."""

import pytest
from src.input_generator.error_injector import (
    inject_errors,
    generate_error_variants,
)


def test_inject_errors_zero_rate():
    """Test that 0% error rate returns original text."""
    text = "hello world"
    result = inject_errors(text, 0.0)
    assert result == text


def test_inject_errors_invalid_rate():
    """Test that invalid error rate raises ValueError."""
    with pytest.raises(ValueError, match="error_rate must be in"):
        inject_errors("hello world", 60.0)


def test_inject_errors_changes_text():
    """Test that non-zero error rate modifies text."""
    text = "hello world testing example sentence"
    result = inject_errors(text, 25.0, seed=42)
    assert result != text, "25% error rate should change text"


def test_inject_errors_word_count_preserved():
    """Test that error injection preserves word count."""
    text = "hello world testing example sentence"
    result = inject_errors(text, 50.0, seed=42)
    assert len(result.split()) == len(text.split())


def test_inject_errors_reproducibility():
    """Test that same seed produces same result."""
    text = "hello world testing example sentence"
    result1 = inject_errors(text, 25.0, seed=42)
    result2 = inject_errors(text, 25.0, seed=42)
    assert result1 == result2, "Same seed should give same result"


def test_generate_error_variants():
    """Test generation of multiple error variants."""
    text = "hello world"
    levels = [0, 10, 25, 50]
    variants = generate_error_variants(text, levels)

    assert len(variants) == 4
    assert variants[0][1] == 0.0
    assert variants[3][1] == 50.0
    assert variants[0][0] == text


def test_generate_error_variants_all_different():
    """Test that different error levels produce different results."""
    text = "hello world testing example sentence here today"
    levels = [10, 25, 50]
    variants = generate_error_variants(text, levels)

    texts = [v[0] for v in variants]
    assert len(set(texts)) > 1, "Different levels should produce variations"
