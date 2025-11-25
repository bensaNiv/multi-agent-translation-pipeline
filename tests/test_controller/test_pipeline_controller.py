"""Tests for pipeline_controller module."""

import pytest
from src.controller.pipeline_controller import TranslationPipelineController


def test_controller_initialization():
    """Test controller initializes with correct agents."""
    controller = TranslationPipelineController()
    assert "en_to_fr" in controller.agents
    assert "fr_to_he" in controller.agents
    assert "he_to_en" in controller.agents
    assert len(controller.results) == 0


def test_execute_pipeline_structure():
    """Test pipeline execution returns correct structure."""
    controller = TranslationPipelineController()
    result = controller.execute_pipeline("Test sentence", 0.0, sentence_id=1)

    assert "original_text" in result
    assert "error_level" in result
    assert "intermediate_translations" in result
    assert "final_english_text" in result
    assert "metadata" in result
    assert result["sentence_id"] == 1


def test_execute_pipeline_creates_intermediate_translations():
    """Test that pipeline captures intermediate translations."""
    controller = TranslationPipelineController()
    result = controller.execute_pipeline("Test sentence", 25.0)

    assert "en_to_fr" in result["intermediate_translations"]
    assert "fr_to_he" in result["intermediate_translations"]
    assert result["final_english_text"] is not None


def test_execute_pipeline_stores_metadata():
    """Test that pipeline stores execution metadata."""
    controller = TranslationPipelineController()
    result = controller.execute_pipeline("Test", 0.0)

    assert "timestamp" in result["metadata"]
    assert "agents_executed" in result["metadata"]
    assert len(result["metadata"]["agents_executed"]) == 3
    assert "en_to_fr" in result["metadata"]["agents_executed"]


def test_execute_batch():
    """Test batch execution."""
    controller = TranslationPipelineController()
    inputs = [
        {"text": "Test 1", "error_level": 0.0},
        {"text": "Test 2", "error_level": 25.0},
    ]
    results = controller.execute_batch(inputs)
    assert len(results) == 2
    assert results[0]["error_level"] == 0.0
    assert results[1]["error_level"] == 25.0


def test_execute_batch_with_sentence_ids():
    """Test batch execution preserves sentence IDs."""
    controller = TranslationPipelineController()
    inputs = [
        {"text": "Test 1", "error_level": 0.0, "sentence_id": 10},
        {"text": "Test 2", "error_level": 25.0, "sentence_id": 20},
    ]
    results = controller.execute_batch(inputs)
    assert results[0]["sentence_id"] == 10
    assert results[1]["sentence_id"] == 20


def test_controller_accumulates_results():
    """Test that controller accumulates results across executions."""
    controller = TranslationPipelineController()

    controller.execute_pipeline("Test 1", 0.0)
    controller.execute_pipeline("Test 2", 25.0)

    assert len(controller.results) == 2
