"""Generate complete input dataset for experiments."""

import json
from pathlib import Path
from typing import List, Dict, Any

from .sentence_generator import generate_baseline_sentences
from .error_injector import generate_error_variants


def generate_input_dataset(
    output_path: str = "data/input/sentences.json",
    error_levels: List[float] = [0, 10, 20, 25, 30, 40, 50],
) -> Dict[str, Any]:
    """Generate complete input dataset with all error variants.

    Args:
        output_path: Path to save JSON output
        error_levels: List of error percentages to generate

    Returns:
        Dictionary containing all sentences and variants

    Example:
        >>> dataset = generate_input_dataset()
        >>> len(dataset['sentences'])
        5
        >>> len(dataset['sentences'][0]['variants'])
        7
    """
    baseline_sentences = generate_baseline_sentences()

    dataset = {
        "metadata": {
            "num_sentences": len(baseline_sentences),
            "error_levels": error_levels,
            "min_words": 15,
        },
        "sentences": [],
    }

    for idx, sentence in enumerate(baseline_sentences):
        word_count = len(sentence.split())
        variants = generate_error_variants(sentence, error_levels)

        sentence_data = {
            "id": idx,
            "original": sentence,
            "word_count": word_count,
            "variants": [
                {"text": text, "error_level": level}
                for text, level in variants
            ],
        }

        dataset["sentences"].append(sentence_data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated {len(baseline_sentences)} sentences")
    print(f"✓ Created {len(error_levels)} error variants per sentence")
    print(
        f"✓ Total variants: "
        f"{len(baseline_sentences) * len(error_levels)}"
    )
    print(f"✓ Saved to: {output_path}")

    return dataset


if __name__ == "__main__":
    generate_input_dataset()
