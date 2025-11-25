# Multi-Agent Translation Pipeline Experiment

A research system that measures semantic drift in translations caused by spelling errors using Claude Code agents. The system translates text through a language chain (English → French → Hebrew → English) and analyzes how translation quality degrades with increasing spelling errors.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results](#results)
- [Credits](#credits)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
   ```bash
   cd /mnt/c/Users/bensa/Projects/LLMCourseProject/projects/project3
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your settings if needed
   ```

5. Verify installation:
   ```bash
   python3 -m pytest tests/ -v
   ```

## Usage

### Step 1: Generate Input Sentences

Generate baseline sentences with varying error rates:

```bash
python3 -m src.input_generator.generate_inputs
```

**Expected output:**
```
✓ Generated 5 sentences
✓ Created 7 error variants per sentence
✓ Total variants: 35
✓ Saved to: data/input/sentences.json
```

### Step 2: Run Translation Pipeline

Execute the translation pipeline (EN→FR→HE→EN):

```bash
python3 -c "
from src.controller.pipeline_controller import TranslationPipelineController
import json

# Load input data
with open('data/input/sentences.json', 'r') as f:
    data = json.load(f)

# Run pipeline
controller = TranslationPipelineController()
for sentence in data['sentences']:
    for variant in sentence['variants']:
        controller.execute_pipeline(
            variant['text'],
            variant['error_level'],
            sentence['id']
        )

# Save results
controller.save_results('results/experiments/pipeline_results.json')
"
```

### Step 3: Analyze Semantic Drift

Compute embeddings and distance metrics:

```bash
python3 -c "
from src.analysis.semantic_drift_analyzer import SemanticDriftAnalyzer

# Analyze results
analyzer = SemanticDriftAnalyzer()
df = analyzer.analyze_results('results/experiments/pipeline_results.json')

# Save analysis
analyzer.save_analysis(df, 'results/analysis/semantic_drift.csv')
print(f'Analyzed {len(df)} translations')
"
```

### Step 4: Generate Visualizations

Create graphs showing error rate vs semantic distance:

```bash
python3 -c "
import pandas as pd
from src.visualization.graph_generator import generate_all_graphs

# Load analysis results
df = pd.read_csv('results/analysis/semantic_drift.csv')

# Generate all graphs
generate_all_graphs(df, 'results/graphs')
print('✓ Graphs generated in results/graphs/')
"
```

## Configuration

Configuration is managed through environment variables or `.env` file.

### Environment Variables

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `EMBEDDING_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` | Any valid model name |
| `DEVICE` | Computation device | `cpu` | `cpu`, `cuda` |
| `MIN_WORDS` | Minimum words per sentence | `15` | Integer > 0 |
| `ERROR_LEVELS` | Error percentages to test | `0,10,20,25,30,40,50` | Comma-separated floats |
| `LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Configuration File

Create a `.env` file (copy from `.env.example`):

```bash
# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEVICE=cpu

# Experiment Configuration
MIN_WORDS=15
ERROR_LEVELS=0,10,20,25,30,40,50

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/pipeline.log
```

## Project Structure

```
project3/
├── src/                          # Source code
│   ├── input_generator/          # Sentence and error generation
│   │   ├── sentence_generator.py # Baseline sentence generation
│   │   ├── error_injector.py     # Spelling error injection
│   │   └── generate_inputs.py    # Complete dataset generation
│   ├── controller/               # Pipeline orchestration
│   │   └── pipeline_controller.py
│   ├── analysis/                 # Embedding analysis
│   │   ├── embedding_generator.py
│   │   ├── distance_calculator.py
│   │   └── semantic_drift_analyzer.py
│   └── visualization/            # Graph generation
│       └── graph_generator.py
├── tests/                        # Test suite (mirrors src/)
│   ├── test_input_generator/
│   ├── test_controller/
│   └── test_analysis/
├── agents/                       # Agent definitions
│   ├── agent_en_to_fr.json      # English → French
│   ├── agent_fr_to_he.json      # French → Hebrew
│   └── agent_he_to_en.json      # Hebrew → English
├── data/                         # Input data
│   └── input/
│       └── sentences.json        # Generated sentences
├── results/                      # Experiment outputs
│   ├── experiments/              # Pipeline results
│   ├── analysis/                 # Distance metrics
│   └── graphs/                   # Visualizations
├── docs/                         # Documentation
│   └── architecture.md           # System architecture
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── .env.example                  # Configuration template
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS/Linux
# Or navigate to htmlcov/index.html in browser
```

### Run Specific Test Modules

```bash
# Test input generation
pytest tests/test_input_generator/ -v

# Test controller
pytest tests/test_controller/ -v

# Test analysis
pytest tests/test_analysis/ -v
```

### Code Quality Checks

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
pylint src/
```

## Results

### Expected Findings

The system should demonstrate that:

1. **Semantic drift increases with error rate**: Higher spelling error percentages lead to greater semantic distance between original and final translations

2. **Cosine distance is primary metric**: Better suited for sentence embeddings from transformer models

3. **Translation chain amplifies errors**: Each translation stage compounds the semantic drift

### Sample Output

After running the complete pipeline, you should see:

**results/analysis/semantic_drift.csv:**
```csv
sentence_id,error_level,cosine_distance,euclidean_distance,original_text,final_text
0,0.0,0.05,2.3,"The quick brown fox...","The quick brown fox..."
0,10.0,0.12,3.1,"The qick brwn fox...","The fast brown fox..."
0,25.0,0.28,4.5,"The qik brn fx...","A brown animal..."
```

**results/graphs/:**
- `cosine_distance.png`: Error rate vs cosine distance plot
- `euclidean_distance.png`: Error rate vs Euclidean distance plot
- `both_metrics.png`: Side-by-side comparison

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed system architecture, data flow diagrams, and design decisions.

## Key Features

- **MSc-Level Quality**: Adheres to strict coding standards (PEP 8, type hints, comprehensive docstrings)
- **High Test Coverage**: 70%+ coverage with pytest
- **Modular Design**: Clean separation of concerns
- **Reproducible**: Fixed random seeds for consistent results
- **Well-Documented**: Comprehensive documentation and examples

## Technology Stack

- **Python 3.8+**: Core language
- **sentence-transformers**: Sentence embeddings (all-MiniLM-L6-v2)
- **NumPy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib + seaborn**: Visualization
- **pytest**: Testing framework
- **Claude Code Agents**: Translation pipeline

## Known Limitations

1. **Stub Agent Implementation**: Agent invocations use placeholders. In production, these would call actual Claude Code agents.

2. **Performance**: Large-scale experiments with many sentences may be slow due to sequential agent execution.

3. **Language Support**: Currently supports EN→FR→HE→EN only. Other language chains require new agent definitions.

## Credits

### Libraries and Frameworks

- [sentence-transformers](https://sbert.net/): Sentence embeddings
- [scikit-learn](https://scikit-learn.org/): Distance metrics
- [matplotlib](https://matplotlib.org/): Plotting
- [seaborn](https://seaborn.pydata.org/): Statistical visualization
- [pytest](https://pytest.org/): Testing

### Research References

- [Semantic Drift in Multilingual Representations](https://direct.mit.edu/coli/article/46/3/571/93376)
- [COMET: Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213.pdf)
- [Sentence-BERT Documentation](https://sbert.net/)

## License

Academic project - MSc Computer Science

## Author

MSc Project - Multi-Agent Translation Pipeline Experiment
