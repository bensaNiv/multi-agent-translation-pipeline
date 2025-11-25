"""Tests for sentence_generator module."""

import pytest
from src.input_generator.sentence_generator import (
    generate_baseline_sentences,
    validate_sentence,
)


def test_generate_baseline_sentences_count():
    """Test that baseline sentences are generated."""
    sentences = generate_baseline_sentences()
    assert len(sentences) >= 5, "Should generate at least 5 sentences"


def test_generate_baseline_sentences_word_count():
    """Test that all sentences have ≥15 words."""
    sentences = generate_baseline_sentences()
    for sentence in sentences:
        word_count = len(sentence.split())
        assert word_count >= 15, f"Sentence has only {word_count} words"


def test_validate_sentence_valid():
    """Test validation of valid sentence."""
    sentence = (
        "This is a test sentence with more than fifteen "
        "words in total here"
    )
    assert validate_sentence(sentence) is True


def test_validate_sentence_too_short():
    """Test validation rejects short sentences."""
    sentence = "Too short"
    assert validate_sentence(sentence) is False


def test_validate_sentence_custom_min_words():
    """Test validation with custom minimum word count."""
    sentence = "This has five words total"
    assert validate_sentence(sentence, min_words=5) is True
    assert validate_sentence(sentence, min_words=6) is False
