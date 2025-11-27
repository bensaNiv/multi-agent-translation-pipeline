"""Result management module for pipeline execution."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResultManager:
    """Manages storage and persistence of pipeline results.

    Handles result accumulation, formatting, and file I/O operations.

    Attributes:
        results: List of all experiment results

    Example:
        >>> manager = ResultManager()
        >>> manager.add_result({"test": "data"})
        >>> len(manager.results)
        1
    """

    def __init__(self):
        """Initialize result manager with empty results list."""
        self.results: List[Dict[str, Any]] = []

    def add_result(self, result: Dict[str, Any]):
        """Add a result to the collection.

        Args:
            result: Result dictionary to add

        Example:
            >>> manager = ResultManager()
            >>> manager.add_result({"id": 1})
            >>> manager.results[0]["id"]
            1
        """
        self.results.append(result)

    def save_results(
        self, output_path: str = "results/experiments/pipeline_results.json"
    ):
        """Save all experiment results to JSON file.

        Args:
            output_path: Path to save results

        Example:
            >>> manager = ResultManager()
            >>> manager.add_result({"test": "data"})
            >>> manager.save_results("test.json")
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(self.results)} results to {output_path}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all stored results.

        Returns:
            List of all results

        Example:
            >>> manager = ResultManager()
            >>> manager.add_result({"id": 1})
            >>> len(manager.get_results())
            1
        """
        return self.results

    def clear_results(self):
        """Clear all stored results.

        Example:
            >>> manager = ResultManager()
            >>> manager.add_result({"id": 1})
            >>> manager.clear_results()
            >>> len(manager.results)
            0
        """
        self.results = []
        logger.info("Results cleared")

    def get_result_count(self) -> int:
        """Get count of stored results.

        Returns:
            Number of results

        Example:
            >>> manager = ResultManager()
            >>> manager.add_result({"id": 1})
            >>> manager.get_result_count()
            1
        """
        return len(self.results)
