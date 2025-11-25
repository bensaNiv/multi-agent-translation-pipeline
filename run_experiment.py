"""Run the complete translation pipeline experiment with simulated translations."""

import json
import random
from pathlib import Path
from src.controller.pipeline_controller import TranslationPipelineController


class MockTranslationController(TranslationPipelineController):
    """Enhanced controller with simulated realistic translations."""

    def _invoke_agent(self, agent_name: str, input_text: str) -> str:
        """Simulate translation with semantic drift based on input errors.

        This mock simulates how translations degrade with spelling errors:
        - Clean text → accurate translation
        - Text with errors → increasingly divergent translation
        """
        # Extract error indicators from input
        words = input_text.split()

        # Simulate translation by slightly modifying the text
        # This creates synthetic semantic drift
        if "English_to_French" in agent_name:
            # Simulate EN→FR: add French-like words
            translated = self._simulate_french(words)
        elif "French_to_Hebrew" in agent_name:
            # Simulate FR→HE: transform structure
            translated = self._simulate_hebrew(words)
        else:  # Hebrew to English
            # Simulate HE→EN: return to English with drift
            translated = self._simulate_back_to_english(words)

        return translated

    def _simulate_french(self, words):
        """Simulate French translation."""
        # Simple simulation: keep structure, modify some words
        french_words = []
        replacements = {
            'the': 'le', 'quick': 'rapide', 'brown': 'brun',
            'fox': 'renard', 'jumps': 'saute', 'over': 'sur',
            'lazy': 'paresseux', 'dog': 'chien', 'while': 'tandis',
            'sun': 'soleil', 'shines': 'brille', 'in': 'dans',
            'sky': 'ciel', 'scientists': 'scientifiques',
            'discovered': 'découvert', 'that': 'que',
            'machine': 'machine', 'learning': 'apprentissage',
            'algorithms': 'algorithmes', 'can': 'peuvent',
            'improve': 'améliorer', 'quality': 'qualité',
            'modern': 'moderne', 'world': 'monde',
            'technology': 'technologie', 'artificial': 'artificielle',
            'intelligence': 'intelligence', 'continues': 'continue',
            'transform': 'transformer', 'communicate': 'communiquer',
            'across': 'à travers', 'different': 'différentes',
            'languages': 'langues', 'cultures': 'cultures',
        }

        for word in words:
            clean_word = word.lower().strip('.,!?')
            french_word = replacements.get(clean_word, word)
            french_words.append(french_word)

        return ' '.join(french_words)

    def _simulate_hebrew(self, words):
        """Simulate Hebrew translation (using transliteration)."""
        # Use Latin transliteration of Hebrew for simulation
        hebrew_sim = []
        replacements = {
            'le': 'ha', 'rapide': 'mahir', 'brun': 'chum',
            'renard': 'shual', 'saute': 'kofetz', 'sur': 'al',
            'paresseux': 'atzel', 'chien': 'kelev',
            'soleil': 'shemesh', 'ciel': 'shamayim',
            'scientifiques': 'madaim', 'machine': 'mechona',
            'apprentissage': 'limud', 'algorithmes': 'algoritmim',
            'qualité': 'aikhut', 'monde': 'olam',
            'technologie': 'technologia', 'intelligence': 'binah',
        }

        for word in words:
            clean_word = word.lower().strip('.,!?')
            hebrew_word = replacements.get(clean_word, f"heb_{word}")
            hebrew_sim.append(hebrew_word)

        return ' '.join(hebrew_sim)

    def _simulate_back_to_english(self, words):
        """Simulate translation back to English with semantic drift."""
        # Back-translation introduces errors and changes
        english_words = []
        back_translations = {
            'ha': 'the', 'mahir': 'fast', 'chum': 'brown',
            'shual': 'fox', 'kofetz': 'leaps', 'al': 'upon',
            'atzel': 'idle', 'kelev': 'dog', 'shemesh': 'sunshine',
            'shamayim': 'heavens', 'madaim': 'researchers',
            'mechona': 'device', 'limud': 'study',
            'algoritmim': 'procedures', 'aikhut': 'standard',
            'olam': 'earth', 'technologia': 'tech',
            'binah': 'understanding',
        }

        for word in words:
            clean_word = word.lower().strip('.,!?')
            # Remove heb_ prefix if present
            if clean_word.startswith('heb_'):
                clean_word = clean_word[4:]

            english_word = back_translations.get(clean_word, word)
            english_words.append(english_word)

        return ' '.join(english_words)


def main():
    """Run the complete experiment."""
    print("=" * 60)
    print("Multi-Agent Translation Pipeline Experiment")
    print("=" * 60)
    print()

    # Step 1: Load input data
    print("📖 Step 1: Loading input data...")
    with open('data/input/sentences.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data['sentences'])} sentences")
    print(f"✓ Error levels: {data['metadata']['error_levels']}")
    print()

    # Step 2: Run translation pipeline
    print("🔄 Step 2: Running translation pipeline...")
    controller = MockTranslationController()

    processed = 0
    total_variants = sum(len(s['variants']) for s in data['sentences'])

    for sentence in data['sentences']:
        for variant in sentence['variants']:
            processed += 1
            print(f"  Processing variant {processed}/{total_variants} "
                  f"(sentence {sentence['id']}, error {variant['error_level']}%)")

            controller.execute_pipeline(
                original_text=variant['text'],
                error_level=variant['error_level'],
                sentence_id=sentence['id']
            )

    # Save results
    results_path = 'results/experiments/pipeline_results.json'
    controller.save_results(results_path)
    print(f"\n✓ Saved {len(controller.results)} results to {results_path}")
    print()

    # Display sample result
    print("📋 Sample Result:")
    print("-" * 60)
    sample = controller.results[0]
    print(f"Sentence ID: {sample['sentence_id']}")
    print(f"Error Level: {sample['error_level']}%")
    print(f"Original: {sample['original_text'][:60]}...")
    print(f"Final: {sample['final_english_text'][:60]}...")
    print()

    print("=" * 60)
    print("✅ Experiment Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install dependencies: pip install sentence-transformers pandas matplotlib")
    print("2. Run analysis: python3 run_analysis.py")
    print("3. Generate graphs: python3 run_visualization.py")


if __name__ == "__main__":
    main()
