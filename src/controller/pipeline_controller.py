"""Translation pipeline controller for agent orchestration."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .agent_invoker import AgentInvoker
from .result_manager import ResultManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationPipelineController:
    """Controls execution of the multi-agent translation pipeline.

    Orchestrates three translation agents (EN→FR→HE→EN) sequentially,
    collects all intermediate translations, and stores results.

    Attributes:
        agent_invoker: Handler for agent invocations
        result_manager: Handler for result storage
        results: List of all experiment results (for compatibility)

    Example:
        >>> controller = TranslationPipelineController()
        >>> result = controller.execute_pipeline(
        ...     "Hello world test",
        ...     error_level=25.0
        ... )
        >>> assert "final_english_text" in result
    """

    def __init__(self):
        """Initialize controller with agent invoker and result manager."""
        self.agent_invoker = AgentInvoker()
        self.result_manager = ResultManager()

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Get results for backward compatibility.

        Returns:
            List of all experiment results
        """
        return self.result_manager.get_results()

    def execute_pipeline(
        self,
        original_text: str,
        error_level: float,
        sentence_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the full translation pipeline.

        Args:
            original_text: Original English text with spelling errors
            error_level: Percentage of spelling errors (0-50)
            sentence_id: Optional sentence identifier

        Returns:
            Dictionary containing complete experiment result

        Raises:
            ValueError: If any agent fails to execute

        Example:
            >>> controller = TranslationPipelineController()
            >>> result = controller.execute_pipeline("Test", 0.0)
            >>> "final_english_text" in result
            True
        """
        logger.info(
            f"Starting pipeline for sentence_id={sentence_id}, "
            f"error_level={error_level}%"
        )

        result = self._create_result_structure(
            original_text, error_level, sentence_id
        )

        try:
            self._execute_stage_1(result, original_text)
            self._execute_stage_2(result)
            self._execute_stage_3(result)

            logger.info("✓ Pipeline completed successfully")

        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}")
            result["metadata"]["error"] = str(e)
            raise

        self.result_manager.add_result(result)
        return result

    def _create_result_structure(
        self,
        original_text: str,
        error_level: float,
        sentence_id: Optional[int],
    ) -> Dict[str, Any]:
        """Create initial result structure.

        Args:
            original_text: Original text
            error_level: Error percentage
            sentence_id: Sentence identifier

        Returns:
            Result dictionary with initial structure
        """
        return {
            "sentence_id": sentence_id,
            "original_text": original_text,
            "error_level": error_level,
            "intermediate_translations": {},
            "final_english_text": None,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "agents_executed": [],
            },
        }

    def _execute_stage_1(self, result: Dict[str, Any], input_text: str):
        """Execute stage 1: English to French translation.

        Args:
            result: Result dictionary to update
            input_text: Text to translate
        """
        agent_name = self.agent_invoker.get_agent_name("en_to_fr")
        output = self.agent_invoker.invoke(agent_name, input_text)
        result["intermediate_translations"]["en_to_fr"] = output
        result["metadata"]["agents_executed"].append("en_to_fr")

    def _execute_stage_2(self, result: Dict[str, Any]):
        """Execute stage 2: French to Hebrew translation.

        Args:
            result: Result dictionary to update
        """
        input_text = result["intermediate_translations"]["en_to_fr"]
        agent_name = self.agent_invoker.get_agent_name("fr_to_he")
        output = self.agent_invoker.invoke(agent_name, input_text)
        result["intermediate_translations"]["fr_to_he"] = output
        result["metadata"]["agents_executed"].append("fr_to_he")

    def _execute_stage_3(self, result: Dict[str, Any]):
        """Execute stage 3: Hebrew to English translation.

        Args:
            result: Result dictionary to update
        """
        input_text = result["intermediate_translations"]["fr_to_he"]
        agent_name = self.agent_invoker.get_agent_name("he_to_en")
        output = self.agent_invoker.invoke(agent_name, input_text)
        result["final_english_text"] = output
        result["metadata"]["agents_executed"].append("he_to_en")

    def execute_batch(
        self, input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute pipeline for multiple input sentences.

        Args:
            input_data: List of dicts with 'text' and 'error_level' keys

        Returns:
            List of all results

        Example:
            >>> controller = TranslationPipelineController()
            >>> inputs = [
            ...     {"text": "Test 1", "error_level": 0},
            ...     {"text": "Test 2", "error_level": 25}
            ... ]
            >>> results = controller.execute_batch(inputs)
            >>> len(results) == 2
            True
        """
        results = []
        total = len(input_data)

        for idx, item in enumerate(input_data):
            logger.info(f"Processing {idx + 1}/{total}")
            try:
                result = self.execute_pipeline(
                    original_text=item["text"],
                    error_level=item["error_level"],
                    sentence_id=item.get("sentence_id", idx),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on item {idx}: {e}")
                continue

        logger.info(f"✓ Batch complete: {len(results)}/{total} successful")
        return results

    def save_results(
        self, output_path: str = "results/experiments/pipeline_results.json"
    ):
        """Save all experiment results to JSON file.

        Args:
            output_path: Path to save results

        Example:
            >>> controller = TranslationPipelineController()
            >>> controller.save_results("results/test.json")
        """
        self.result_manager.save_results(output_path)
