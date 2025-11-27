"""Run real translation pipeline using Claude Code Task agents.

This script helps organize the manual execution of Task-based translation agents.
Results will be stored in a structured format for later analysis.
"""

import json
from pathlib import Path
from datetime import datetime


class RealExperimentController:
    """Controller for managing real Task-based agent translations."""

    def __init__(self):
        """Initialize the controller."""
        self.results = []
        self.results_file = Path("results/experiments/real_pipeline_results.json")
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

    def add_result(
        self,
        sentence_id: int,
        error_level: int,
        original_text: str,
        english_to_french: str,
        french_to_hebrew: str,
        hebrew_to_english: str,
    ):
        """Add a complete pipeline result.

        Args:
            sentence_id: ID of the source sentence
            error_level: Percentage of spelling errors (0-50)
            original_text: Original English text with errors
            english_to_french: Output from EN→FR agent
            french_to_hebrew: Output from FR→HE agent
            hebrew_to_english: Final output from HE→EN agent
        """
        result = {
            "sentence_id": sentence_id,
            "error_level": error_level,
            "timestamp": datetime.now().isoformat(),
            "original_text": original_text,
            "translations": {
                "en_to_fr": english_to_french,
                "fr_to_he": french_to_hebrew,
                "he_to_en": hebrew_to_english,
            },
            "final_english_text": hebrew_to_english,
        }
        self.results.append(result)
        print(f"✓ Added result for sentence {sentence_id}, error {error_level}%")

    def save_results(self):
        """Save all results to JSON file."""
        data = {
            "metadata": {
                "experiment_type": "real_task_based_agents",
                "total_runs": len(self.results),
                "timestamp": datetime.now().isoformat(),
            },
            "results": self.results,
        }

        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved {len(self.results)} results to {self.results_file}")

    def load_existing(self):
        """Load existing results if available."""
        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.results = data.get("results", [])
                print(f"✓ Loaded {len(self.results)} existing results")
        else:
            print("No existing results found, starting fresh")

    def print_summary(self):
        """Print a summary of current progress."""
        if not self.results:
            print("No results yet")
            return

        by_sentence = {}
        by_error = {}

        for r in self.results:
            sid = r["sentence_id"]
            err = r["error_level"]

            if sid not in by_sentence:
                by_sentence[sid] = []
            by_sentence[sid].append(err)

            if err not in by_error:
                by_error[err] = 0
            by_error[err] += 1

        print("\n" + "=" * 60)
        print("EXPERIMENT PROGRESS SUMMARY")
        print("=" * 60)
        print(f"Total pipeline runs completed: {len(self.results)}")
        print(f"\nBy Sentence:")
        for sid, errors in sorted(by_sentence.items()):
            print(f"  Sentence {sid}: {len(errors)} error levels - {sorted(errors)}")
        print(f"\nBy Error Level:")
        for err, count in sorted(by_error.items()):
            print(f"  {err}%: {count} sentences")
        print("=" * 60)


if __name__ == "__main__":
    controller = RealExperimentController()
    controller.load_existing()
    controller.print_summary()

    print("\n" + "=" * 60)
    print("MANUAL EXECUTION GUIDE")
    print("=" * 60)
    print(
        """
To run the experiment:

1. Load input data from data/input/sentences.json
2. For each sentence variant, run three Task agents:
   - Agent 1: English → French
   - Agent 2: French → Hebrew
   - Agent 3: Hebrew → English
3. Record results using controller.add_result(...)
4. Save with controller.save_results()

Example workflow in Python:
    from run_real_experiment import RealExperimentController
    controller = RealExperimentController()
    controller.load_existing()

    # After running Task agents manually and getting outputs:
    controller.add_result(
        sentence_id=0,
        error_level=0,
        original_text="...",
        english_to_french="...",
        french_to_hebrew="...",
        hebrew_to_english="..."
    )

    controller.save_results()
"""
    )
