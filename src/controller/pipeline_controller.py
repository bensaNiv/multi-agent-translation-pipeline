"""Translation pipeline controller for agent orchestration."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationPipelineController:
    """Controls execution of the multi-agent translation pipeline.

    Orchestrates three translation agents (EN→FR→HE→EN) sequentially,
    collects all intermediate translations, and stores results.

    Attributes:
        agents: Dictionary mapping stage names to agent names
        results: List of all experiment results

    Example:
        >>> controller = TranslationPipelineController()
        >>> result = controller.execute_pipeline(
        ...     "Hello world test",
        ...     error_level=25.0
        ... )
        >>> assert "final_english_text" in result
    """

    def __init__(self):
        """Initialize controller with agent mappings."""
        self.agents = {
            "en_to_fr": "English_to_French_Translator",
            "fr_to_he": "French_to_Hebrew_Translator",
            "he_to_en": "Hebrew_to_English_Translator",
        }
        self.results: List[Dict[str, Any]] = []

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

        result = {
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

        try:
            stage_1_output = self._invoke_agent(
                agent_name=self.agents["en_to_fr"],
                input_text=original_text,
            )
            result["intermediate_translations"]["en_to_fr"] = stage_1_output
            result["metadata"]["agents_executed"].append("en_to_fr")

            stage_2_output = self._invoke_agent(
                agent_name=self.agents["fr_to_he"],
                input_text=stage_1_output,
            )
            result["intermediate_translations"]["fr_to_he"] = stage_2_output
            result["metadata"]["agents_executed"].append("fr_to_he")

            stage_3_output = self._invoke_agent(
                agent_name=self.agents["he_to_en"],
                input_text=stage_2_output,
            )
            result["final_english_text"] = stage_3_output
            result["metadata"]["agents_executed"].append("he_to_en")

            logger.info("✓ Pipeline completed successfully")

        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}")
            result["metadata"]["error"] = str(e)
            raise

        self.results.append(result)
        return result

    def _invoke_agent(self, agent_name: str, input_text: str) -> str:
        """Invoke a single translation agent.

        NOTE: This is a stub implementation. In actual Claude Code,
        this would use the Claude Code agent invocation mechanism.

        Args:
            agent_name: Name of agent to invoke
            input_text: Text to translate

        Returns:
            Translated text

        Raises:
            ValueError: If agent invocation fails
        """
        logger.warning(
            f"STUB: Invoking {agent_name} (placeholder translation)"
        )
        return f"[{agent_name}_output: {input_text[:30]}...]"

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
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(self.results)} results to {output_path}")
