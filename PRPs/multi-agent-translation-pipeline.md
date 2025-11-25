# PRP: Multi-Agent Translation Pipeline Experiment

**Project**: Multi-Agent Translation Pipeline with Semantic Drift Analysis
**PRP Version**: 2.0 (Express Mode + Detailed Reference)
**Date**: 2025-11-25
**Estimated Time**: 3-4 hours (Express Mode) | 12-19 hours (Detailed Mode)
**Confidence Score**: 9/10 (One-pass implementation feasibility with Express Mode)

**👉 TL;DR**: Use Express Mode (Section: 🚀 Express Execution Mode) for 3-4 hour implementation. Detailed Mode available as reference.

---

## Executive Summary

This PRP provides a complete implementation plan for building a multi-agent translation system that measures semantic drift caused by spelling errors. The system uses three Claude Code agents to translate text through a language chain (EN→FR→HE→EN), then analyzes semantic drift using embeddings and visualizes the results.

**Key Challenge**: Orchestrating multiple independent agents, preserving intermediate results, and measuring semantic similarity using state-of-the-art embedding techniques while adhering to MSc-level quality standards.

---

## ⚠️ CRITICAL: Prerequisites - Read MSc Skills First

**BEFORE STARTING IMPLEMENTATION**, you MUST read and internalize these skill files:

### Required Skills Location
```
/mnt/c/Users/bensa/Projects/LLMCourseProject/.claude/skills/user/
```

### Read These Files IN ORDER:

1. **msc-code-standards-SKILL.md** (MUST READ FIRST)
   - Defines HOW to write Python code
   - PEP 8 compliance, type hints, docstrings
   - Max 50 lines/function, 150 lines/file
   - DRY principle, error handling

2. **msc-documentation-standards-SKILL.md**
   - Defines HOW to document code
   - README structure, docstring format
   - Comments explain WHY not WHAT

3. **msc-security-config-SKILL.md**
   - Defines HOW to handle configuration
   - No hardcoded secrets, .env files
   - Input validation, .gitignore

4. **msc-submission-structure-SKILL.md**
   - Defines project structure
   - src/ and tests/ organization
   - Required files and directories

5. **msc-testing-standards-SKILL.md**
   - Defines HOW to test code
   - 70%+ coverage, pytest, AAA pattern
   - Test naming conventions

### Why This Is Critical

The PRP provides WHAT to build and provides code examples. The skills define HOW to build it correctly.

**DO NOT blindly copy code from this PRP**. Instead:
1. ✅ Read the skill that applies to your current task
2. ✅ Write code following the skill's guidelines
3. ✅ Use PRP examples as reference, not gospel
4. ✅ Validate against skill checklists before moving on

**Example**: When writing `src/analysis/embedding_generator.py`:
1. Read msc-code-standards (type hints, docstrings, max lines)
2. Read msc-testing-standards (how to test this module)
3. Look at PRP example code as inspiration
4. Write YOUR implementation following the skills
5. Validate: Does it follow all the skills? Yes → Continue

---

## 🚀 Express Execution Mode (3-4 Hours)

**Use this mode for fast implementation**. The detailed breakdown in Section 4 is available as reference if you get stuck.

### Prerequisites (5 min)
```bash
# Read all 5 MSc skill files (MUST DO FIRST)
# Then create project directory and activate venv
cd project3
python -m venv venv && source venv/bin/activate
```

### Task 1: Foundation Setup (20 min)

**What to do:**
- Create directory structure: `src/`, `tests/`, `agents/`, `data/`, `results/`, `docs/`, `config/`
- Create all `__init__.py` files in src/ and tests/ subdirectories
- Write `requirements.txt` with: sentence-transformers, numpy, pandas, matplotlib, seaborn, pytest, pytest-cov, black, pylint, mypy
- Create `.gitignore` (exclude: .env, venv/, __pycache__, *.pyc, data/raw/, results/, logs/)
- Create `.env.example` with config templates

**Validation:**
```bash
ls src/ tests/ agents/ && echo "✓ Structure OK"
pip install -r requirements.txt && echo "✓ Dependencies installed"
```

**Reference**: See Section 4, Phase 1 for detailed examples

---

### Task 2: Input Generation (30 min)

**What to do:**
Create in `src/input_generator/`:
- `sentence_generator.py`: Generate 5 baseline sentences (≥15 words each)
- `error_injector.py`: Inject spelling errors at specified rates (0-50%)
  - Error types: substitution, omission, duplication
  - Use fixed random seeds for reproducibility
- `generate_inputs.py`: Combine both, create JSON with all error variants

Create in `tests/test_input_generator/`:
- `test_sentence_generator.py`: Test word count validation
- `test_error_injector.py`: Test error rates, word count preservation

**Run generation:**
```bash
python -m src.input_generator.generate_inputs
```

**Validation:**
```bash
pytest tests/test_input_generator/ -v
test -f data/input/sentences.json && echo "✓ Input data created"
```

**Reference**: See Section 4, Phase 2 for complete code examples

---

### Task 3: Agent Definitions (15 min)

**What to do:**
Create 3 JSON files in `agents/`:

1. `agent_en_to_fr.json` - English → French translator
2. `agent_fr_to_he.json` - French → Hebrew translator
3. `agent_he_to_en.json` - Hebrew → English translator

Each agent needs:
```json
{
  "name": "Agent_Name",
  "description": "What it does",
  "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
  "output_schema": {"type": "object", "properties": {"translated_text": {"type": "string"}}, "required": ["translated_text"]},
  "skills": ["translation", "encoding_utf8"],
  "constraints": {"stateless": true, "encoding": "UTF-8"}
}
```

**Validation:**
```bash
for f in agents/*.json; do python -c "import json; json.load(open('$f'))" && echo "✓ $f valid"; done
```

**Reference**: See Section 4, Phase 3 for complete agent schemas

---

### Task 4: Pipeline Controller (45 min)

**What to do:**
Create in `src/controller/`:
- `pipeline_controller.py`:
  - Class `TranslationPipelineController` with method `execute_pipeline(text, error_level)`
  - Invoke 3 agents sequentially: EN→FR→HE→EN
  - Collect all intermediate translations
  - Save results to JSON
  - **Note**: Use stub agent invocations for testing (real agents invoked by Claude Code)

Create in `tests/test_controller/`:
- `test_pipeline_controller.py`: Test controller initialization, pipeline structure, batch execution

**Validation:**
```bash
pytest tests/test_controller/ -v
```

**Reference**: See Section 4, Phase 4 for complete implementation

---

### Task 5: Embedding Analysis (60 min)

**What to do:**
Create in `src/analysis/`:

1. `embedding_generator.py`:
   - Class `EmbeddingGenerator` using `sentence-transformers`
   - Model: `all-MiniLM-L6-v2` (384-dim, fast)
   - Methods: `generate_embedding(text)`, `generate_embeddings_batch(texts)`

2. `distance_calculator.py`:
   - `calculate_cosine_distance(emb1, emb2)` - PRIMARY metric
   - `calculate_euclidean_distance(emb1, emb2)` - SECONDARY metric
   - `calculate_both_distances(emb1, emb2)`

3. `semantic_drift_analyzer.py`:
   - Class `SemanticDriftAnalyzer`
   - Method `analyze_results(results_path)` - loads pipeline results, computes embeddings & distances
   - Returns pandas DataFrame with columns: sentence_id, error_level, cosine_distance, euclidean_distance

Create in `tests/test_analysis/`:
- Tests for all 3 modules above

**Validation:**
```bash
pytest tests/test_analysis/ -v
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && echo "✓ Model loads"
```

**Reference**: See Section 4, Phase 5 for complete implementations

---

### Task 6: Visualization (30 min)

**What to do:**
Create in `src/visualization/`:
- `graph_generator.py`:
  - `plot_error_vs_distance(df, metric='cosine')` - line plot with error bars
  - `plot_both_metrics(df)` - side-by-side comparison
  - `generate_all_graphs(df, output_dir)` - creates all 3 graphs

Create in `tests/test_visualization/`:
- `test_graph_generator.py`: Test plot generation, file output

**Validation:**
```bash
pytest tests/test_visualization/ -v
```

**Reference**: See Section 4, Phase 6 for plotting code

---

### Task 7: Documentation & Final Validation (30 min)

**What to do:**
1. Create `README.md` with:
   - Project overview
   - Installation instructions
   - Usage (4 steps: generate inputs → run pipeline → analyze → visualize)
   - Project structure
   - Testing instructions

2. Create `docs/architecture.md`:
   - High-level system diagram
   - Component descriptions
   - Data flow
   - Technology decisions

3. Run final validation:
   ```bash
   # Run all tests with coverage
   pytest tests/ --cov=src --cov-report=html --cov-fail-under=70 -v

   # Code quality checks
   black src/ tests/
   mypy src/
   pylint src/

   # Generate coverage report
   open htmlcov/index.html
   ```

**Validation:**
```bash
test -f README.md && grep -q "Installation" README.md && echo "✓ README complete"
pytest tests/ --cov=src --cov-fail-under=70 && echo "✓ All tests pass with 70%+ coverage"
```

**Reference**: See Section 4, Phase 7 for documentation templates

---

## 🎯 Express Mode Summary

| Task | Time | Deliverable |
|------|------|-------------|
| 0. Read MSc Skills | 5 min | Understanding of quality standards |
| 1. Foundation | 20 min | Project structure + dependencies |
| 2. Input Generation | 30 min | `data/input/sentences.json` |
| 3. Agent Definitions | 15 min | 3 agent JSON files |
| 4. Controller | 45 min | Pipeline orchestration |
| 5. Analysis | 60 min | Embedding + distance calculation |
| 6. Visualization | 30 min | 3 graphs in `results/graphs/` |
| 7. Documentation | 30 min | README + docs + validation |
| **TOTAL** | **~3.5 hours** | **Complete working system** |

### Success Criteria (Express Mode)
- ✅ All tests pass with ≥70% coverage
- ✅ All 3 agent JSONs validated
- ✅ Input data generated (5 sentences × 7 error levels = 35 variants)
- ✅ Graphs show clear trend: error ↑ → semantic distance ↑
- ✅ README.md complete
- ✅ Code follows all 5 MSc skills

---

## 📚 Detailed Reference Mode

The sections below provide comprehensive details if you need them. Most users should use Express Mode above.

**When to use Detailed Mode:**
- You get stuck on a specific task
- You need to see complete code examples
- You want deeper understanding of the approach
- You're validating implementation quality

---

## Table of Contents (Detailed Reference)

1. [Context and Research Findings](#1-context-and-research-findings)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Blueprint](#3-implementation-blueprint)
4. [Detailed Task Breakdown](#4-detailed-task-breakdown)
5. [Validation Gates](#5-validation-gates)
6. [Quality Standards](#6-quality-standards)
7. [Risk Mitigation](#7-risk-mitigation)
8. [Success Criteria](#8-success-criteria)

---

## 1. Context and Research Findings

### 1.1 Claude Code Agent Schema

**Key Finding**: Claude Code uses subagents with strict input/output contracts. Each agent must define:
- `name`: Unique identifier
- `description`: Clear purpose statement
- `input_schema`: JSON Schema (draft-07) for inputs
- `output_schema`: JSON Schema (draft-07) for outputs
- `skills`: List of capabilities
- `constraints`: Execution constraints (stateless, encoding, etc.)

**Critical Constraints**:
- Agents are **stateless** - no memory between invocations
- Agents cannot communicate directly - must use controller
- Input/output must strictly conform to declared schemas
- All text must use UTF-8 encoding (especially for Hebrew RTL text)

**Agent Definition Template**:
```json
{
  "name": "Agent_Name",
  "description": "What this agent does",
  "input_schema": {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"]
  },
  "output_schema": {
    "type": "object",
    "properties": {"translated_text": {"type": "string"}},
    "required": ["translated_text"]
  },
  "skills": ["capability_1", "capability_2"],
  "constraints": {"stateless": true, "encoding": "UTF-8"}
}
```

### 1.2 Embedding Best Practices

**Library**: Sentence Transformers (sentence-transformers)
- Installation: `pip install sentence-transformers`
- Recommended model: `all-MiniLM-L6-v2` (fast, accurate for semantic similarity)
- Documentation: https://sbert.net/

**Distance Metrics**:
1. **Cosine Similarity** (RECOMMENDED):
   - Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
   - Best for normalized embeddings
   - Standard for sentence-transformers models
   - Formula: `cosine_sim = 1 - cosine_distance`

2. **Euclidean Distance**:
   - Range: [0, ∞) where 0 = identical
   - Sensitive to vector magnitude
   - Use for comparison purposes

**Critical Best Practice**: Sentence-BERT models are fine-tuned with cosine similarity, making it the optimal metric. Euclidean distance should be used as secondary measure for comparison.

**Sources**:
- [Sentence Transformers Documentation](https://sbert.net/)
- [PyPI: sentence-transformers](https://pypi.org/project/sentence-transformers/)
- [Best Practices: Cosine vs Euclidean](https://datascience.stackexchange.com/questions/27726/when-to-use-cosine-simlarity-over-euclidean-similarity)

### 1.3 Translation Quality Evaluation Research

**Key Research Findings**:
- **Semantic Drift**: Iterative translation introduces cumulative semantic drift
- **Embedding-Based Metrics**: Modern approaches use neural embeddings (COMET, STD)
- **Measurement Approach**: Compare original vs. final embeddings to quantify drift

**Relevant Papers**:
- [Semantic Drift in Multilingual Representations](https://direct.mit.edu/coli/article/46/3/571/93376/Semantic-Drift-in-Multilingual-Representations)
- [Evaluating Translation by Playing Telephone](https://arxiv.org/html/2509.19611)
- [COMET: Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213.pdf)

**Applied Methodology**:
```python
# Measure semantic drift
original_embedding = model.encode(original_text)
final_embedding = model.encode(final_translated_text)
cosine_distance = 1 - cosine_similarity(original_embedding, final_embedding)
euclidean_distance = np.linalg.norm(original_embedding - final_embedding)
```

### 1.4 MSc Quality Standards

**CRITICAL**: All code must comply with 5 MSc skills located in:
`/mnt/c/Users/bensa/Projects/LLMCourseProject/.claude/skills/user/`

1. **msc-code-standards**: PEP 8, type hints, docstrings, max 50 lines/function
2. **msc-documentation-standards**: Comprehensive README, architecture docs
3. **msc-security-config**: No hardcoded secrets, .env files, .gitignore
4. **msc-submission-structure**: src/ and tests/ directories, proper organization
5. **msc-testing-standards**: 70%+ coverage, pytest, AAA pattern

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRANSLATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Generator                                                │
│  └─> Sentences with 0-50% spelling errors                      │
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   Agent 1    │      │   Agent 2    │      │   Agent 3    │ │
│  │   EN → FR    │ ───> │   FR → HE    │ ───> │   HE → EN    │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│          ↑                      ↑                      ↑        │
│          └──────────────────────┴──────────────────────┘        │
│                       Controller                                │
│                  (Orchestration Layer)                          │
│                                                                  │
│  Results Storage                                                │
│  └─> CSV/JSON with all intermediate translations               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     ANALYSIS PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Embedding Generator                                            │
│  └─> Sentence-BERT embeddings for original & final             │
│                                                                  │
│  Distance Calculator                                            │
│  └─> Cosine & Euclidean distances                              │
│                                                                  │
│  Visualization Generator                                        │
│  └─> Matplotlib graphs: Error % vs. Semantic Distance          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
1. INPUT GENERATION
   Original Sentence (15+ words)
   ↓
   Error Injection (0%, 10%, 20%, 25%, 30%, 40%, 50%)
   ↓
   [sentence_0, sentence_10, sentence_20, ...]

2. TRANSLATION PIPELINE (For each sentence variant)
   sentence_with_errors
   ↓
   Agent 1 (EN→FR) → french_text
   ↓
   Agent 2 (FR→HE) → hebrew_text
   ↓
   Agent 3 (HE→EN) → final_english_text
   ↓
   Store: {
     original, error_level,
     intermediate: {en_fr, fr_he},
     final
   }

3. EMBEDDING ANALYSIS
   For each error level:
     - Generate embeddings: original, final
     - Calculate: cosine_distance, euclidean_distance
     - Store metrics

4. VISUALIZATION
   Plot: error_percentage (x) vs semantic_distance (y)
   Export: PNG/PDF graphs
```

### 2.3 Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Core Language | Python | 3.8+ | Implementation |
| Agent Definitions | JSON | - | Agent schemas |
| Embeddings | sentence-transformers | latest | Semantic similarity |
| Numerical Computing | NumPy | 1.24+ | Distance calculations |
| Data Handling | Pandas | 2.0+ | Results storage |
| Visualization | Matplotlib + Seaborn | latest | Graph generation |
| Testing | pytest + pytest-cov | latest | Unit/integration tests |
| Code Quality | black, pylint, mypy | latest | Standards enforcement |

---

## 3. Implementation Blueprint

### 3.1 High-Level Phases

```
Phase 1: Foundation (Project Setup)
   ↓
Phase 2: Input Generation
   ↓
Phase 3: Agent Definitions
   ↓
Phase 4: Controller Implementation
   ↓
Phase 5: Embedding Analysis
   ↓
Phase 6: Visualization
   ↓
Phase 7: Testing & Documentation
```

### 3.2 Critical Path Dependencies

```
Foundation → Input Generation → Agent Definitions ──┐
                                                     ├→ Controller → Analysis → Visualization
                                  Testing ←──────────┘
```

### 3.3 Success Metrics

- ✅ All 3 translation agents defined and validated
- ✅ Controller successfully orchestrates pipeline
- ✅ Input sentences generated with correct error rates
- ✅ Embeddings computed without errors
- ✅ Graphs show clear correlation: error rate ↑ = semantic distance ↑
- ✅ Test coverage ≥ 70%
- ✅ All MSc quality standards met

---

## 4. Detailed Task Breakdown

### PHASE 1: Foundation (Project Setup)

#### Task 1.1: Create Directory Structure
**Objective**: Establish MSc-compliant project organization

**Actions**:
```bash
mkdir -p src/{agents,controller,input_generator,analysis,visualization,config,utils}
mkdir -p tests/{test_input_generator,test_controller,test_analysis,test_visualization}
mkdir -p data/{input,raw,processed}
mkdir -p results/{experiments,graphs}
mkdir -p docs
mkdir -p config
mkdir -p agents
```

**Expected Structure**:
```
project3/
├── src/
│   ├── __init__.py
│   ├── agents/              # Agent orchestration code (not definitions)
│   ├── controller/          # Pipeline controller
│   ├── input_generator/     # Sentence and error generation
│   ├── analysis/            # Embedding analysis
│   ├── visualization/       # Graph generation
│   ├── config/              # Configuration handling
│   └── utils/               # Shared utilities
├── tests/                   # Mirrors src/ structure
├── agents/                  # Agent JSON definitions
├── data/                    # Input/output data
├── results/                 # Graphs and metrics
├── docs/                    # Additional documentation
├── config/                  # Configuration files
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
└── CLAUDE.md
```

**Validation Gate 1.1**:
```bash
# Verify structure exists
test -d src && test -d tests && test -d agents && echo "✓ Structure OK"
```

---

#### Task 1.2: Create requirements.txt
**Objective**: Define all dependencies with pinned versions

**Content** (`requirements.txt`):
```txt
# Core ML Libraries
numpy==1.24.3                    # Numerical computing
pandas==2.0.3                    # Data manipulation

# Embedding and NLP
sentence-transformers==2.2.2     # Sentence embeddings
torch>=2.0.0                     # PyTorch (required by sentence-transformers)

# Visualization
matplotlib==3.7.2                # Plotting
seaborn==0.12.2                  # Statistical visualization

# Configuration
python-dotenv==1.0.0             # Environment variables
pyyaml==6.0.1                    # YAML config parsing

# Testing
pytest==7.4.0                    # Testing framework
pytest-cov==4.1.0                # Coverage reporting

# Development Tools
black==23.7.0                    # Code formatting
pylint==2.17.5                   # Linting
mypy==1.5.0                      # Type checking
```

**Actions**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Validation Gate 1.2**:
```bash
# Verify installation
python -c "import sentence_transformers; print('✓ sentence-transformers OK')"
python -c "import pandas; print('✓ pandas OK')"
python -c "import matplotlib; print('✓ matplotlib OK')"
pytest --version && echo "✓ pytest OK"
```

---

#### Task 1.3: Create .gitignore
**Objective**: Prevent committing secrets and large files

**Content** (`.gitignore`):
```gitignore
# Environment and secrets
.env
.env.local
*.key
*.pem
secrets/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# Data files (large datasets)
data/raw/*.csv
data/processed/*.json
*.db
*.sqlite

# Model files
*.pkl
*.h5
*.weights

# Results (may be large)
results/graphs/*.png
results/graphs/*.pdf

# Logs
*.log
logs/
```

**Validation Gate 1.3**:
```bash
# Verify .gitignore works
touch .env test.log
git status | grep -E "(\.env|test\.log)" && echo "✗ Failed: secrets not ignored" || echo "✓ .gitignore OK"
rm .env test.log
```

---

#### Task 1.4: Create .env.example
**Objective**: Document required configuration

**Content** (`.env.example`):
```bash
# .env.example - Copy to .env and fill in values

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEVICE=cpu  # Use 'cuda' if GPU available

# Experiment Configuration
MIN_WORDS=15
ERROR_LEVELS=0,10,20,25,30,40,50

# Output Configuration
RESULTS_DIR=results/experiments
GRAPHS_DIR=results/graphs

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/pipeline.log
```

**Actions**:
```bash
cp .env.example .env
# Edit .env if needed (default values work)
```

**Validation Gate 1.4**:
```bash
test -f .env.example && test -f .env && echo "✓ Configuration files OK"
```

---

#### Task 1.5: Create Package Initialization Files
**Objective**: Make all directories importable Python packages

**Actions**:
```python
# src/__init__.py
"""Multi-Agent Translation Pipeline Experiment.

This package implements a research system that measures semantic drift
in translations caused by spelling errors using Claude Code agents.
"""

__version__ = "1.0.0"
__author__ = "MSc Project"

# src/agents/__init__.py
"""Agent orchestration and invocation utilities."""

# src/controller/__init__.py
"""Translation pipeline controller."""

# src/input_generator/__init__.py
"""Input sentence generation and error injection."""

# src/analysis/__init__.py
"""Embedding analysis and distance calculation."""

# src/visualization/__init__.py
"""Graph generation and visualization."""

# src/config/__init__.py
"""Configuration management."""

# src/utils/__init__.py
"""Shared utility functions."""

# tests/__init__.py
"""Test suite for multi-agent translation pipeline."""
```

**Validation Gate 1.5**:
```python
# test_imports.py
import sys
sys.path.insert(0, 'src')

try:
    import agents
    import controller
    import input_generator
    import analysis
    import visualization
    import config
    import utils
    print("✓ All packages importable")
except ImportError as e:
    print(f"✗ Import failed: {e}")
```

---

### PHASE 2: Input Generation

#### Task 2.1: Create Sentence Generator
**Objective**: Generate baseline English sentences ≥15 words

**File**: `src/input_generator/sentence_generator.py`

**Implementation**:
```python
"""Generate baseline sentences for translation experiments."""

from typing import List


def generate_baseline_sentences() -> List[str]:
    """Generate baseline English sentences for experiments.

    Returns:
        List of sentences, each with ≥15 words

    Example:
        >>> sentences = generate_baseline_sentences()
        >>> all(len(s.split()) >= 15 for s in sentences)
        True
    """
    sentences = [
        "The quick brown fox jumps over the lazy dog while the sun shines brightly in the clear blue sky above",
        "Scientists have recently discovered that machine learning algorithms can significantly improve translation quality when properly trained on diverse datasets",
        "In the modern world of technology and innovation, artificial intelligence continues to transform how we communicate across different languages and cultures",
        "The ancient library contained thousands of manuscripts written in various languages that scholars spent decades attempting to translate accurately",
        "Climate change poses significant challenges for future generations, requiring immediate action from governments and citizens around the world today"
    ]

    # Validate all sentences meet minimum word requirement
    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count < 15:
            raise ValueError(
                f"Sentence must have ≥15 words, got {word_count}: {sentence}"
            )

    return sentences


def validate_sentence(sentence: str, min_words: int = 15) -> bool:
    """Validate sentence meets minimum word requirement.

    Args:
        sentence: Sentence to validate
        min_words: Minimum required word count

    Returns:
        True if valid, False otherwise
    """
    return len(sentence.split()) >= min_words
```

**Validation Gate 2.1**:
```python
# tests/test_input_generator/test_sentence_generator.py
import pytest
from src.input_generator.sentence_generator import (
    generate_baseline_sentences,
    validate_sentence
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
    sentence = "This is a test sentence with more than fifteen words in total here"
    assert validate_sentence(sentence) is True


def test_validate_sentence_too_short():
    """Test validation rejects short sentences."""
    sentence = "Too short"
    assert validate_sentence(sentence) is False
```

Run tests:
```bash
pytest tests/test_input_generator/test_sentence_generator.py -v
# Expected: All tests pass ✓
```

---

#### Task 2.2: Create Error Injection Module
**Objective**: Inject realistic spelling errors at specified rates

**File**: `src/input_generator/error_injector.py`

**Implementation**:
```python
"""Inject spelling errors into text at specified rates."""

import random
from typing import List


def inject_errors(text: str, error_rate: float, seed: int = 42) -> str:
    """Inject spelling errors into text at specified rate.

    Introduces realistic spelling errors by:
    - Character substitution (e.g., 'e' → 'a')
    - Character omission (e.g., 'hello' → 'helo')
    - Character duplication (e.g., 'hello' → 'helllo')

    Args:
        text: Original text
        error_rate: Percentage of words to modify (0-50)
        seed: Random seed for reproducibility

    Returns:
        Text with injected errors

    Raises:
        ValueError: If error_rate not in valid range [0, 50]

    Example:
        >>> inject_errors("hello world", 50.0, seed=42)
        'helo world'  # 50% of words have errors
    """
    if not 0 <= error_rate <= 50:
        raise ValueError(f"error_rate must be in [0, 50], got {error_rate}")

    if error_rate == 0:
        return text

    random.seed(seed)
    words = text.split()
    num_errors = int(len(words) * (error_rate / 100))

    # Select random words to corrupt
    error_indices = random.sample(range(len(words)), num_errors)

    for idx in error_indices:
        words[idx] = _corrupt_word(words[idx])

    return " ".join(words)


def _corrupt_word(word: str) -> str:
    """Corrupt a single word with a realistic spelling error.

    Args:
        word: Word to corrupt

    Returns:
        Corrupted word
    """
    if len(word) <= 2:
        return word  # Don't corrupt very short words

    error_type = random.choice(['substitute', 'omit', 'duplicate'])
    pos = random.randint(1, len(word) - 2)  # Avoid first/last char

    if error_type == 'substitute':
        # Replace with nearby keyboard character
        replacements = {
            'a': 'sqa', 'e': 'rwd', 'i': 'uko', 'o': 'pil',
            't': 'rfy', 'n': 'bhm', 's': 'adw'
        }
        char = word[pos].lower()
        replacement = random.choice(replacements.get(char, 'x'))
        word = word[:pos] + replacement + word[pos+1:]

    elif error_type == 'omit':
        # Remove character
        word = word[:pos] + word[pos+1:]

    elif error_type == 'duplicate':
        # Duplicate character
        word = word[:pos] + word[pos] + word[pos:]

    return word


def generate_error_variants(
    text: str,
    error_levels: List[float]
) -> List[tuple[str, float]]:
    """Generate multiple versions of text with different error rates.

    Args:
        text: Original text
        error_levels: List of error percentages to generate

    Returns:
        List of (corrupted_text, error_level) tuples

    Example:
        >>> variants = generate_error_variants("hello world", [0, 25, 50])
        >>> len(variants)
        3
        >>> variants[0][1]  # First variant has 0% errors
        0.0
    """
    variants = []
    for level in error_levels:
        corrupted = inject_errors(text, level, seed=int(level * 100))
        variants.append((corrupted, level))
    return variants
```

**Validation Gate 2.2**:
```python
# tests/test_input_generator/test_error_injector.py
import pytest
from src.input_generator.error_injector import (
    inject_errors,
    generate_error_variants
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


def test_generate_error_variants():
    """Test generation of multiple error variants."""
    text = "hello world"
    levels = [0, 10, 25, 50]
    variants = generate_error_variants(text, levels)

    assert len(variants) == 4
    assert variants[0][1] == 0.0
    assert variants[3][1] == 50.0
    assert variants[0][0] == text  # 0% should be unchanged
```

Run tests:
```bash
pytest tests/test_input_generator/test_error_injector.py -v
# Expected: All tests pass ✓
```

---

#### Task 2.3: Generate and Save Input Sentences
**Objective**: Create complete input dataset with all error levels

**File**: `src/input_generator/generate_inputs.py`

**Implementation**:
```python
"""Generate complete input dataset for experiments."""

import json
from pathlib import Path
from typing import List, Dict, Any

from .sentence_generator import generate_baseline_sentences
from .error_injector import generate_error_variants


def generate_input_dataset(
    output_path: str = "data/input/sentences.json",
    error_levels: List[float] = [0, 10, 20, 25, 30, 40, 50]
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
        7  # One for each error level
    """
    baseline_sentences = generate_baseline_sentences()

    dataset = {
        "metadata": {
            "num_sentences": len(baseline_sentences),
            "error_levels": error_levels,
            "min_words": 15
        },
        "sentences": []
    }

    for idx, sentence in enumerate(baseline_sentences):
        word_count = len(sentence.split())
        variants = generate_error_variants(sentence, error_levels)

        sentence_data = {
            "id": idx,
            "original": sentence,
            "word_count": word_count,
            "variants": [
                {
                    "text": text,
                    "error_level": level
                }
                for text, level in variants
            ]
        }

        dataset["sentences"].append(sentence_data)

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated {len(baseline_sentences)} sentences")
    print(f"✓ Created {len(error_levels)} error variants per sentence")
    print(f"✓ Total variants: {len(baseline_sentences) * len(error_levels)}")
    print(f"✓ Saved to: {output_path}")

    return dataset


if __name__ == "__main__":
    generate_input_dataset()
```

**Validation Gate 2.3**:
```bash
# Run generation
python -m src.input_generator.generate_inputs

# Verify output
test -f data/input/sentences.json && echo "✓ Input file created"

# Validate JSON structure
python -c "
import json
with open('data/input/sentences.json') as f:
    data = json.load(f)
    assert 'sentences' in data
    assert len(data['sentences']) >= 5
    assert all(s['word_count'] >= 15 for s in data['sentences'])
    print('✓ Input data validated')
"
```

---

### PHASE 3: Agent Definitions

#### Task 3.1: Create Agent JSON Definitions
**Objective**: Define all three translation agents following Claude Code Agent Schema

**File**: `agents/agent_en_to_fr.json`

```json
{
  "name": "English_to_French_Translator",
  "description": "Translates English text into French. Preserves spelling variations and errors from source text to study semantic drift effects.",
  "purpose": "First stage of multi-language translation pipeline (EN→FR)",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "English text to translate to French"
      }
    },
    "required": ["text"],
    "additionalProperties": false
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "translated_text": {
        "type": "string",
        "description": "Text translated from English to French"
      }
    },
    "required": ["translated_text"],
    "additionalProperties": false
  },
  "skills": [
    "english_language_comprehension",
    "french_translation",
    "character_encoding_utf8",
    "spelling_error_preservation"
  ],
  "constraints": {
    "language_pair": "EN→FR",
    "stateless": true,
    "encoding": "UTF-8",
    "error_handling": "preserve_input_errors"
  },
  "transformation_rules": {
    "preserve_spelling_errors": true,
    "maintain_structure": true,
    "keep_punctuation": true
  }
}
```

**File**: `agents/agent_fr_to_he.json`

```json
{
  "name": "French_to_Hebrew_Translator",
  "description": "Translates French text into Hebrew. Handles right-to-left (RTL) text encoding properly while preserving spelling variations from source.",
  "purpose": "Second stage of multi-language translation pipeline (FR→HE)",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "French text to translate to Hebrew"
      }
    },
    "required": ["text"],
    "additionalProperties": false
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "translated_text": {
        "type": "string",
        "description": "Text translated from French to Hebrew (RTL)"
      }
    },
    "required": ["translated_text"],
    "additionalProperties": false
  },
  "skills": [
    "french_language_comprehension",
    "hebrew_translation",
    "rtl_text_handling",
    "character_encoding_utf8",
    "spelling_error_preservation"
  ],
  "constraints": {
    "language_pair": "FR→HE",
    "stateless": true,
    "encoding": "UTF-8",
    "text_direction": "right_to_left",
    "error_handling": "preserve_input_errors"
  },
  "transformation_rules": {
    "preserve_spelling_errors": true,
    "maintain_structure": true,
    "handle_rtl_properly": true
  }
}
```

**File**: `agents/agent_he_to_en.json`

```json
{
  "name": "Hebrew_to_English_Translator",
  "description": "Translates Hebrew text back into English. Completes the translation cycle while preserving semantic drift from intermediate translations.",
  "purpose": "Third stage of multi-language translation pipeline (HE→EN)",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "Hebrew text to translate to English"
      }
    },
    "required": ["text"],
    "additionalProperties": false
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "translated_text": {
        "type": "string",
        "description": "Text translated from Hebrew to English"
      }
    },
    "required": ["translated_text"],
    "additionalProperties": false
  },
  "skills": [
    "hebrew_language_comprehension",
    "english_translation",
    "rtl_text_handling",
    "character_encoding_utf8",
    "semantic_drift_preservation"
  ],
  "constraints": {
    "language_pair": "HE→EN",
    "stateless": true,
    "encoding": "UTF-8",
    "error_handling": "preserve_semantic_drift"
  },
  "transformation_rules": {
    "preserve_semantic_changes": true,
    "maintain_structure": true,
    "capture_drift_effects": true
  }
}
```

**Validation Gate 3.1**:
```bash
# Validate JSON syntax
for agent in agents/agent_*.json; do
    echo "Validating $agent..."
    python -c "import json; json.load(open('$agent'))" && echo "✓ Valid JSON"
done

# Verify required fields
python -c "
import json
from pathlib import Path

required_fields = ['name', 'description', 'input_schema', 'output_schema', 'skills', 'constraints']

for agent_file in Path('agents').glob('agent_*.json'):
    with open(agent_file) as f:
        agent = json.load(f)
        for field in required_fields:
            assert field in agent, f'{agent_file} missing {field}'
print('✓ All agents have required fields')
"
```

---

#### Task 3.2: Create Agent Documentation
**Objective**: Document agent specifications and usage

**File**: `agents/README.md`

```markdown
# Translation Agents

This directory contains Claude Code agent definitions for the multi-language translation pipeline.

## Agent Specifications

### 1. English to French Translator (`agent_en_to_fr.json`)

**Language Pair**: EN → FR
**Input**: English text with potential spelling errors
**Output**: French translation

**Usage**:
```json
{
  "text": "The quick brown fox jumps over the lazy dog"
}
```

**Expected Output**:
```json
{
  "translated_text": "Le rapide renard brun saute par-dessus le chien paresseux"
}
```

### 2. French to Hebrew Translator (`agent_fr_to_he.json`)

**Language Pair**: FR → HE
**Input**: French text
**Output**: Hebrew translation (RTL text)

**Special Considerations**:
- Handles right-to-left (RTL) text encoding
- Preserves UTF-8 encoding for proper Hebrew character display

### 3. Hebrew to English Translator (`agent_he_to_en.json`)

**Language Pair**: HE → EN
**Input**: Hebrew text (RTL)
**Output**: English translation

**Purpose**: Completes the translation cycle, allowing measurement of semantic drift from original to final English text.

## Agent Contract

All agents follow the same input/output contract:

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "text": {"type": "string"}
  },
  "required": ["text"]
}
```

**Output Schema**:
```json
{
  "type": "object",
  "properties": {
    "translated_text": {"type": "string"}
  },
  "required": ["translated_text"]
}
```

## Agent Constraints

- **Stateless**: Agents do not maintain state between invocations
- **Sequential**: Must be invoked in order (EN→FR→HE→EN)
- **UTF-8**: All text uses UTF-8 encoding
- **Error Preservation**: Spelling errors are preserved to study semantic drift

## Testing Agents

Test each agent independently:

```python
# Pseudocode
result = invoke_agent(
    agent_name="English_to_French_Translator",
    input={"text": "Hello world"}
)
assert "translated_text" in result
```

## Agent Orchestration

Agents are orchestrated by the controller in `src/controller/pipeline_controller.py`.
```

**Validation Gate 3.2**:
```bash
test -f agents/README.md && echo "✓ Agent documentation created"
```

---

### PHASE 4: Controller Implementation

#### Task 4.1: Create Pipeline Controller
**Objective**: Orchestrate sequential agent execution and data collection

**File**: `src/controller/pipeline_controller.py`

```python
"""Translation pipeline controller for agent orchestration."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
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
            "he_to_en": "Hebrew_to_English_Translator"
        }
        self.results: List[Dict[str, Any]] = []

    def execute_pipeline(
        self,
        original_text: str,
        error_level: float,
        sentence_id: Optional[int] = None
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
                "agents_executed": []
            }
        }

        try:
            # Stage 1: EN → FR
            logger.info("Stage 1: EN → FR")
            stage_1_output = self._invoke_agent(
                agent_name=self.agents["en_to_fr"],
                input_text=original_text
            )
            result["intermediate_translations"]["en_to_fr"] = stage_1_output
            result["metadata"]["agents_executed"].append("en_to_fr")

            # Stage 2: FR → HE
            logger.info("Stage 2: FR → HE")
            stage_2_output = self._invoke_agent(
                agent_name=self.agents["fr_to_he"],
                input_text=stage_1_output
            )
            result["intermediate_translations"]["fr_to_he"] = stage_2_output
            result["metadata"]["agents_executed"].append("fr_to_he")

            # Stage 3: HE → EN
            logger.info("Stage 3: HE → EN")
            stage_3_output = self._invoke_agent(
                agent_name=self.agents["he_to_en"],
                input_text=stage_2_output
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
        # STUB: In real implementation, this would call Claude Code agent
        # For now, return a placeholder to allow testing controller logic
        logger.warning(
            f"STUB: Invoking {agent_name} (placeholder translation)"
        )
        return f"[{agent_name}_output: {input_text[:30]}...]"

    def execute_batch(
        self,
        input_data: List[Dict[str, Any]]
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
                    sentence_id=item.get("sentence_id", idx)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on item {idx}: {e}")
                continue

        logger.info(f"✓ Batch complete: {len(results)}/{total} successful")
        return results

    def save_results(self, output_path: str = "results/experiments/pipeline_results.json"):
        """Save all experiment results to JSON file.

        Args:
            output_path: Path to save results

        Example:
            >>> controller = TranslationPipelineController()
            >>> controller.save_results("results/test.json")
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(self.results)} results to {output_path}")
```

**Validation Gate 4.1**:
```python
# tests/test_controller/test_pipeline_controller.py
import pytest
from src.controller.pipeline_controller import TranslationPipelineController


def test_controller_initialization():
    """Test controller initializes with correct agents."""
    controller = TranslationPipelineController()
    assert "en_to_fr" in controller.agents
    assert "fr_to_he" in controller.agents
    assert "he_to_en" in controller.agents


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


def test_execute_batch():
    """Test batch execution."""
    controller = TranslationPipelineController()
    inputs = [
        {"text": "Test 1", "error_level": 0.0},
        {"text": "Test 2", "error_level": 25.0}
    ]
    results = controller.execute_batch(inputs)
    assert len(results) == 2
```

Run tests:
```bash
pytest tests/test_controller/test_pipeline_controller.py -v
# Expected: All tests pass ✓
```

---

### PHASE 5: Embedding Analysis

#### Task 5.1: Create Embedding Generator
**Objective**: Generate sentence embeddings using sentence-transformers

**File**: `src/analysis/embedding_generator.py`

```python
"""Generate sentence embeddings for semantic similarity analysis."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Sentence-BERT models.

    Uses pre-trained sentence-transformers models to generate
    high-quality sentence embeddings for semantic similarity analysis.

    Attributes:
        model: Loaded SentenceTransformer model
        model_name: Name of the model being used
        device: Device for computation ('cpu' or 'cuda')

    Example:
        >>> generator = EmbeddingGenerator()
        >>> embedding = generator.generate_embedding("Hello world")
        >>> embedding.shape
        (384,)  # Embedding dimension for all-MiniLM-L6-v2
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """Initialize embedding generator with specified model.

        Args:
            model_name: Name of sentence-transformers model to use
            device: Device to use ('cpu' or 'cuda'). Auto-detects if None.

        Recommended Models:
            - 'all-MiniLM-L6-v2': Fast, 384-dim (RECOMMENDED)
            - 'all-mpnet-base-v2': High quality, 768-dim
            - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual

        References:
            https://sbert.net/docs/pretrained_models.html
        """
        logger.info(f"Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = self.model.device
        logger.info(f"✓ Model loaded on device: {self.device}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding vector

        Raises:
            ValueError: If text is empty

        Example:
            >>> gen = EmbeddingGenerator()
            >>> emb = gen.generate_embedding("Test")
            >>> isinstance(emb, np.ndarray)
            True
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (num_texts, embedding_dim)

        Example:
            >>> gen = EmbeddingGenerator()
            >>> texts = ["Test 1", "Test 2", "Test 3"]
            >>> embs = gen.generate_embeddings_batch(texts)
            >>> embs.shape[0] == 3
            True
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return self.model.get_sentence_embedding_dimension()
```

**Validation Gate 5.1**:
```python
# tests/test_analysis/test_embedding_generator.py
import pytest
import numpy as np
from src.analysis.embedding_generator import EmbeddingGenerator


def test_embedding_generator_initialization():
    """Test embedding generator initializes correctly."""
    gen = EmbeddingGenerator()
    assert gen.model is not None
    assert gen.model_name == "all-MiniLM-L6-v2"


def test_generate_embedding_single():
    """Test generating single embedding."""
    gen = EmbeddingGenerator()
    embedding = gen.generate_embedding("Hello world")

    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D vector
    assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 dimension


def test_generate_embedding_empty_text():
    """Test that empty text raises ValueError."""
    gen = EmbeddingGenerator()
    with pytest.raises(ValueError, match="Text cannot be empty"):
        gen.generate_embedding("")


def test_generate_embeddings_batch():
    """Test batch embedding generation."""
    gen = EmbeddingGenerator()
    texts = ["Test 1", "Test 2", "Test 3"]
    embeddings = gen.generate_embeddings_batch(texts, show_progress=False)

    assert embeddings.shape == (3, 384)
    assert isinstance(embeddings, np.ndarray)


def test_get_embedding_dimension():
    """Test getting embedding dimension."""
    gen = EmbeddingGenerator()
    dim = gen.get_embedding_dimension()
    assert dim == 384
```

Run tests:
```bash
pytest tests/test_analysis/test_embedding_generator.py -v
# Expected: All tests pass ✓
```

---

#### Task 5.2: Create Distance Calculator
**Objective**: Calculate cosine and Euclidean distances between embeddings

**File**: `src/analysis/distance_calculator.py`

```python
"""Calculate semantic distances between sentence embeddings."""

import numpy as np
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """Calculate cosine distance between two embeddings.

    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where 0 = identical, 2 = opposite

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine distance (0 = most similar)

    Raises:
        ValueError: If embeddings have different dimensions

    Example:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([1, 0, 0])
        >>> calculate_cosine_distance(emb1, emb2)
        0.0  # Identical vectors

    References:
        Best practice for sentence-transformers:
        https://datascience.stackexchange.com/questions/27726
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embeddings must have same shape: "
            f"{embedding1.shape} vs {embedding2.shape}"
        )

    # Reshape for sklearn cosine_similarity
    emb1_2d = embedding1.reshape(1, -1)
    emb2_2d = embedding2.reshape(1, -1)

    similarity = cosine_similarity(emb1_2d, emb2_2d)[0, 0]
    distance = 1 - similarity

    return float(distance)


def calculate_euclidean_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """Calculate Euclidean distance between two embeddings.

    L2 distance = sqrt(sum((a - b)^2))
    Range: [0, ∞) where 0 = identical

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Euclidean distance (0 = most similar)

    Raises:
        ValueError: If embeddings have different dimensions

    Example:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([1, 0, 0])
        >>> calculate_euclidean_distance(emb1, emb2)
        0.0  # Identical vectors
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embeddings must have same shape: "
            f"{embedding1.shape} vs {embedding2.shape}"
        )

    distance = np.linalg.norm(embedding1 - embedding2)
    return float(distance)


def calculate_both_distances(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> Tuple[float, float]:
    """Calculate both cosine and Euclidean distances.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Tuple of (cosine_distance, euclidean_distance)

    Example:
        >>> emb1 = np.array([1, 0, 0])
        >>> emb2 = np.array([0, 1, 0])
        >>> cos_dist, euc_dist = calculate_both_distances(emb1, emb2)
        >>> cos_dist > 0 and euc_dist > 0
        True
    """
    cosine_dist = calculate_cosine_distance(embedding1, embedding2)
    euclidean_dist = calculate_euclidean_distance(embedding1, embedding2)

    return (cosine_dist, euclidean_dist)


def calculate_distances_batch(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate distances for multiple embedding pairs.

    Args:
        embeddings1: Array of shape (N, D) - N embeddings of dimension D
        embeddings2: Array of shape (N, D)

    Returns:
        Tuple of (cosine_distances, euclidean_distances) arrays of shape (N,)

    Example:
        >>> embs1 = np.random.rand(5, 384)
        >>> embs2 = np.random.rand(5, 384)
        >>> cos_dists, euc_dists = calculate_distances_batch(embs1, embs2)
        >>> len(cos_dists) == 5
        True
    """
    if embeddings1.shape != embeddings2.shape:
        raise ValueError(
            f"Embedding arrays must have same shape: "
            f"{embeddings1.shape} vs {embeddings2.shape}"
        )

    num_pairs = embeddings1.shape[0]
    cosine_distances = np.zeros(num_pairs)
    euclidean_distances = np.zeros(num_pairs)

    for i in range(num_pairs):
        cosine_distances[i] = calculate_cosine_distance(
            embeddings1[i], embeddings2[i]
        )
        euclidean_distances[i] = calculate_euclidean_distance(
            embeddings1[i], embeddings2[i]
        )

    return (cosine_distances, euclidean_distances)
```

**Validation Gate 5.2**:
```python
# tests/test_analysis/test_distance_calculator.py
import pytest
import numpy as np
from src.analysis.distance_calculator import (
    calculate_cosine_distance,
    calculate_euclidean_distance,
    calculate_both_distances,
    calculate_distances_batch
)


def test_cosine_distance_identical():
    """Test cosine distance for identical vectors."""
    emb = np.array([1.0, 2.0, 3.0])
    distance = calculate_cosine_distance(emb, emb)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_orthogonal():
    """Test cosine distance for orthogonal vectors."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    distance = calculate_cosine_distance(emb1, emb2)
    assert distance == pytest.approx(1.0, abs=1e-6)


def test_euclidean_distance_identical():
    """Test Euclidean distance for identical vectors."""
    emb = np.array([1.0, 2.0, 3.0])
    distance = calculate_euclidean_distance(emb, emb)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_euclidean_distance_known():
    """Test Euclidean distance with known result."""
    emb1 = np.array([0.0, 0.0, 0.0])
    emb2 = np.array([3.0, 4.0, 0.0])
    distance = calculate_euclidean_distance(emb1, emb2)
    assert distance == pytest.approx(5.0, abs=1e-6)  # 3-4-5 triangle


def test_calculate_both_distances():
    """Test calculating both distances at once."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    cos_dist, euc_dist = calculate_both_distances(emb1, emb2)

    assert isinstance(cos_dist, float)
    assert isinstance(euc_dist, float)
    assert cos_dist > 0
    assert euc_dist > 0


def test_calculate_distances_batch():
    """Test batch distance calculation."""
    embs1 = np.random.rand(3, 10)
    embs2 = np.random.rand(3, 10)

    cos_dists, euc_dists = calculate_distances_batch(embs1, embs2)

    assert len(cos_dists) == 3
    assert len(euc_dists) == 3
    assert all(cos_dists >= 0)
    assert all(euc_dists >= 0)


def test_distance_dimension_mismatch():
    """Test that dimension mismatch raises ValueError."""
    emb1 = np.array([1.0, 2.0])
    emb2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="same shape"):
        calculate_cosine_distance(emb1, emb2)
```

Run tests:
```bash
pytest tests/test_analysis/test_distance_calculator.py -v
# Expected: All tests pass ✓
```

---

#### Task 5.3: Create Analysis Pipeline
**Objective**: Combine embeddings and distances into complete analysis

**File**: `src/analysis/semantic_drift_analyzer.py`

```python
"""Analyze semantic drift in translation results."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

from .embedding_generator import EmbeddingGenerator
from .distance_calculator import calculate_both_distances

logger = logging.getLogger(__name__)


class SemanticDriftAnalyzer:
    """Analyze semantic drift between original and translated sentences.

    Loads translation results, generates embeddings, calculates distances,
    and produces analysis dataframe.

    Example:
        >>> analyzer = SemanticDriftAnalyzer()
        >>> results = analyzer.analyze_results("results/experiments/pipeline_results.json")
        >>> "cosine_distance" in results.columns
        True
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize analyzer with embedding model.

        Args:
            model_name: Sentence-transformers model to use
        """
        logger.info("Initializing Semantic Drift Analyzer")
        self.embedding_generator = EmbeddingGenerator(model_name=model_name)
        logger.info("✓ Analyzer ready")

    def analyze_results(
        self,
        results_path: str
    ) -> pd.DataFrame:
        """Analyze semantic drift in translation results.

        Args:
            results_path: Path to pipeline results JSON

        Returns:
            DataFrame with columns:
                - sentence_id
                - error_level
                - original_text
                - final_english_text
                - cosine_distance
                - euclidean_distance

        Example:
            >>> analyzer = SemanticDriftAnalyzer()
            >>> df = analyzer.analyze_results("results.json")
            >>> df.columns.tolist()
            ['sentence_id', 'error_level', ...]
        """
        logger.info(f"Loading results from: {results_path}")

        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        logger.info(f"Loaded {len(results)} results")

        # Extract data
        analysis_data = []

        for result in results:
            original = result["original_text"]
            final = result["final_english_text"]
            error_level = result["error_level"]
            sentence_id = result.get("sentence_id", None)

            # Generate embeddings
            logger.debug(f"Processing sentence_id={sentence_id}, error={error_level}%")
            original_emb = self.embedding_generator.generate_embedding(original)
            final_emb = self.embedding_generator.generate_embedding(final)

            # Calculate distances
            cos_dist, euc_dist = calculate_both_distances(original_emb, final_emb)

            analysis_data.append({
                "sentence_id": sentence_id,
                "error_level": error_level,
                "original_text": original,
                "final_english_text": final,
                "cosine_distance": cos_dist,
                "euclidean_distance": euc_dist
            })

        df = pd.DataFrame(analysis_data)
        logger.info("✓ Analysis complete")

        return df

    def save_analysis(
        self,
        df: pd.DataFrame,
        output_path: str = "results/experiments/semantic_drift_analysis.csv"
    ):
        """Save analysis results to CSV.

        Args:
            df: Analysis DataFrame
            output_path: Path to save CSV
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✓ Saved analysis to: {output_path}")

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for semantic drift.

        Args:
            df: Analysis DataFrame

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_samples": len(df),
            "error_levels": sorted(df["error_level"].unique().tolist()),
            "cosine_distance": {
                "mean": df["cosine_distance"].mean(),
                "std": df["cosine_distance"].std(),
                "min": df["cosine_distance"].min(),
                "max": df["cosine_distance"].max()
            },
            "euclidean_distance": {
                "mean": df["euclidean_distance"].mean(),
                "std": df["euclidean_distance"].std(),
                "min": df["euclidean_distance"].min(),
                "max": df["euclidean_distance"].max()
            },
            "by_error_level": {}
        }

        # Statistics by error level
        for error_level in stats["error_levels"]:
            subset = df[df["error_level"] == error_level]
            stats["by_error_level"][error_level] = {
                "count": len(subset),
                "cosine_mean": subset["cosine_distance"].mean(),
                "euclidean_mean": subset["euclidean_distance"].mean()
            }

        return stats
```

**Validation Gate 5.3**:
```python
# tests/test_analysis/test_semantic_drift_analyzer.py
import pytest
import json
import pandas as pd
from pathlib import Path
from src.analysis.semantic_drift_analyzer import SemanticDriftAnalyzer


@pytest.fixture
def sample_results_file(tmp_path):
    """Create sample results file for testing."""
    results = [
        {
            "sentence_id": 0,
            "original_text": "Hello world",
            "final_english_text": "Hello world",
            "error_level": 0.0
        },
        {
            "sentence_id": 1,
            "original_text": "Testing text",
            "final_english_text": "Testing content",
            "error_level": 25.0
        }
    ]

    file_path = tmp_path / "results.json"
    with open(file_path, 'w') as f:
        json.dump(results, f)

    return str(file_path)


def test_analyzer_initialization():
    """Test analyzer initializes correctly."""
    analyzer = SemanticDriftAnalyzer()
    assert analyzer.embedding_generator is not None


def test_analyze_results(sample_results_file):
    """Test full analysis pipeline."""
    analyzer = SemanticDriftAnalyzer()
    df = analyzer.analyze_results(sample_results_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "cosine_distance" in df.columns
    assert "euclidean_distance" in df.columns
    assert all(df["cosine_distance"] >= 0)


def test_compute_statistics(sample_results_file):
    """Test statistics computation."""
    analyzer = SemanticDriftAnalyzer()
    df = analyzer.analyze_results(sample_results_file)
    stats = analyzer.compute_statistics(df)

    assert "total_samples" in stats
    assert stats["total_samples"] == 2
    assert "error_levels" in stats
    assert 0.0 in stats["error_levels"]
```

Run tests:
```bash
pytest tests/test_analysis/test_semantic_drift_analyzer.py -v
# Expected: All tests pass ✓
```

---

### PHASE 6: Visualization

#### Task 6.1: Create Graph Generator
**Objective**: Generate matplotlib visualizations of semantic drift

**File**: `src/visualization/graph_generator.py`

```python
"""Generate visualization graphs for semantic drift analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_error_vs_distance(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    metric: str = "cosine"
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot error percentage vs. semantic distance.

    Args:
        df: Analysis DataFrame with columns:
            - error_level
            - cosine_distance or euclidean_distance
        output_path: Optional path to save figure
        metric: Distance metric to plot ('cosine' or 'euclidean')

    Returns:
        Tuple of (figure, axes) objects

    Raises:
        ValueError: If metric not recognized

    Example:
        >>> df = pd.DataFrame({
        ...     'error_level': [0, 10, 20],
        ...     'cosine_distance': [0.0, 0.1, 0.2]
        ... })
        >>> fig, ax = plot_error_vs_distance(df, metric='cosine')
    """
    if metric not in ['cosine', 'euclidean']:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric}")

    distance_col = f"{metric}_distance"

    if distance_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {distance_col}")

    logger.info(f"Plotting {metric} distance vs error percentage")

    # Group by error level and calculate mean/std
    grouped = df.groupby('error_level')[distance_col].agg(['mean', 'std']).reset_index()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line with error bars
    ax.errorbar(
        grouped['error_level'],
        grouped['mean'],
        yerr=grouped['std'],
        marker='o',
        markersize=8,
        linewidth=2,
        capsize=5,
        label=f'Mean {metric.capitalize()} Distance'
    )

    # Formatting
    ax.set_xlabel('Spelling Error Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(
        f'{metric.capitalize()} Distance',
        fontsize=12,
        fontweight='bold'
    )
    ax.set_title(
        f'Semantic Drift vs. Spelling Errors\n({metric.capitalize()} Distance)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show all error levels
    error_levels = sorted(df['error_level'].unique())
    ax.set_xticks(error_levels)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved plot to: {output_path}")

    return fig, ax


def plot_both_metrics(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot both cosine and Euclidean distances on same figure.

    Args:
        df: Analysis DataFrame
        output_path: Optional path to save figure

    Returns:
        Tuple of (figure, (ax1, ax2))

    Example:
        >>> df = pd.DataFrame({
        ...     'error_level': [0, 10, 20],
        ...     'cosine_distance': [0.0, 0.1, 0.2],
        ...     'euclidean_distance': [0.0, 1.0, 2.0]
        ... })
        >>> fig, (ax1, ax2) = plot_both_metrics(df)
    """
    logger.info("Plotting both metrics comparison")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Group by error level
    grouped = df.groupby('error_level').agg({
        'cosine_distance': ['mean', 'std'],
        'euclidean_distance': ['mean', 'std']
    }).reset_index()

    error_levels = grouped['error_level']

    # Plot 1: Cosine Distance
    ax1.errorbar(
        error_levels,
        grouped[('cosine_distance', 'mean')],
        yerr=grouped[('cosine_distance', 'std')],
        marker='o',
        markersize=8,
        linewidth=2,
        capsize=5,
        color='steelblue',
        label='Cosine Distance'
    )
    ax1.set_xlabel('Spelling Error Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Cosine Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(error_levels)

    # Plot 2: Euclidean Distance
    ax2.errorbar(
        error_levels,
        grouped[('euclidean_distance', 'mean')],
        yerr=grouped[('euclidean_distance', 'std')],
        marker='s',
        markersize=8,
        linewidth=2,
        capsize=5,
        color='coral',
        label='Euclidean Distance'
    )
    ax2.set_xlabel('Spelling Error Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Euclidean Distance', fontsize=12, fontweight='bold')
    ax2.set_title('Euclidean Distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(error_levels)

    fig.suptitle(
        'Semantic Drift Analysis: Multiple Distance Metrics',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved comparison plot to: {output_path}")

    return fig, (ax1, ax2)


def generate_all_graphs(
    df: pd.DataFrame,
    output_dir: str = "results/graphs"
):
    """Generate all visualization graphs.

    Args:
        df: Analysis DataFrame
        output_dir: Directory to save graphs

    Example:
        >>> df = pd.read_csv("results/analysis.csv")
        >>> generate_all_graphs(df)
    """
    logger.info("Generating all graphs")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Cosine distance
    plot_error_vs_distance(
        df,
        output_path=str(output_path / "cosine_distance.png"),
        metric="cosine"
    )

    # Plot 2: Euclidean distance
    plot_error_vs_distance(
        df,
        output_path=str(output_path / "euclidean_distance.png"),
        metric="euclidean"
    )

    # Plot 3: Both metrics comparison
    plot_both_metrics(
        df,
        output_path=str(output_path / "both_metrics_comparison.png")
    )

    logger.info(f"✓ All graphs saved to: {output_dir}")
```

**Validation Gate 6.1**:
```python
# tests/test_visualization/test_graph_generator.py
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.graph_generator import (
    plot_error_vs_distance,
    plot_both_metrics,
    generate_all_graphs
)


@pytest.fixture
def sample_analysis_df():
    """Create sample analysis DataFrame."""
    return pd.DataFrame({
        'error_level': [0, 0, 10, 10, 20, 20],
        'cosine_distance': [0.0, 0.01, 0.1, 0.12, 0.2, 0.22],
        'euclidean_distance': [0.0, 0.1, 1.0, 1.2, 2.0, 2.2]
    })


def test_plot_error_vs_distance_cosine(sample_analysis_df):
    """Test plotting cosine distance."""
    fig, ax = plot_error_vs_distance(sample_analysis_df, metric='cosine')

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == 'Spelling Error Percentage (%)'
    plt.close(fig)


def test_plot_error_vs_distance_euclidean(sample_analysis_df):
    """Test plotting Euclidean distance."""
    fig, ax = plot_error_vs_distance(sample_analysis_df, metric='euclidean')

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_plot_error_vs_distance_invalid_metric(sample_analysis_df):
    """Test that invalid metric raises ValueError."""
    with pytest.raises(ValueError, match="metric must be"):
        plot_error_vs_distance(sample_analysis_df, metric='invalid')


def test_plot_both_metrics(sample_analysis_df):
    """Test plotting both metrics."""
    fig, (ax1, ax2) = plot_both_metrics(sample_analysis_df)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax1, plt.Axes)
    assert isinstance(ax2, plt.Axes)
    plt.close(fig)


def test_generate_all_graphs(sample_analysis_df, tmp_path):
    """Test generating all graphs."""
    output_dir = str(tmp_path / "graphs")
    generate_all_graphs(sample_analysis_df, output_dir=output_dir)

    # Verify files created
    assert (tmp_path / "graphs" / "cosine_distance.png").exists()
    assert (tmp_path / "graphs" / "euclidean_distance.png").exists()
    assert (tmp_path / "graphs" / "both_metrics_comparison.png").exists()
```

Run tests:
```bash
pytest tests/test_visualization/test_graph_generator.py -v
# Expected: All tests pass ✓
```

---

### PHASE 7: Testing & Documentation

#### Task 7.1: Run Complete Test Suite
**Objective**: Verify all components with comprehensive testing

**Actions**:
```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=70 -v

# Expected output:
# ========== test session starts ==========
# ... (test results) ...
# ---------- coverage: ----------
# Name                                Stmts   Miss  Cover   Missing
# -----------------------------------------------------------------
# src/__init__.py                        3      0   100%
# src/input_generator/...              XXX    XXX    XX%
# ...
# -----------------------------------------------------------------
# TOTAL                               XXXX   XXXX    75%
#
# ========== XX passed in X.XXs ==========
```

**Validation Gate 7.1**:
- All tests pass ✓
- Coverage ≥ 70% overall ✓
- Coverage ≥ 90% for core logic ✓
- No skipped tests without documentation ✓

---

#### Task 7.2: Create README.md
**Objective**: Comprehensive project documentation

**File**: `README.md`

```markdown
# Multi-Agent Translation Pipeline Experiment

A research system that measures semantic drift in translations caused by spelling errors using Claude Code agents and sentence embeddings.

## Overview

This project implements a multi-agent translation pipeline that:
1. Translates English text through multiple languages (EN→FR→HE→EN)
2. Injects controlled spelling errors at various rates (0-50%)
3. Measures semantic drift using sentence embeddings
4. Visualizes the relationship between error rates and semantic distance

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Analysis](#analysis)
- [Results](#results)
- [Testing](#testing)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   cd project3
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for most cases)
   ```

5. Verify installation:
   ```bash
   pytest tests/ -v
   ```

## Usage

### Step 1: Generate Input Sentences

```bash
python -m src.input_generator.generate_inputs
```

This creates `data/input/sentences.json` with baseline sentences and error variants.

### Step 2: Run Translation Pipeline

```bash
python -m src.controller.run_experiments
```

This orchestrates the three translation agents and saves results to `results/experiments/pipeline_results.json`.

### Step 3: Analyze Semantic Drift

```bash
python -m src.analysis.run_analysis
```

This computes embeddings and distances, saving to `results/experiments/semantic_drift_analysis.csv`.

### Step 4: Generate Visualizations

```bash
python -m src.visualization.create_graphs
```

This generates graphs in `results/graphs/`:
- `cosine_distance.png`
- `euclidean_distance.png`
- `both_metrics_comparison.png`

### Complete Pipeline (All Steps)

```bash
bash scripts/run_full_experiment.sh
```

## Project Structure

```
project3/
├── src/                           # Source code
│   ├── input_generator/           # Sentence & error generation
│   ├── controller/                # Pipeline orchestration
│   ├── analysis/                  # Embedding & distance analysis
│   └── visualization/             # Graph generation
├── agents/                        # Agent JSON definitions
│   ├── agent_en_to_fr.json       # English → French
│   ├── agent_fr_to_he.json       # French → Hebrew
│   └── agent_he_to_en.json       # Hebrew → English
├── tests/                         # Test suite (70%+ coverage)
├── data/                          # Input/output data
├── results/                       # Experiment results & graphs
├── docs/                          # Additional documentation
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── .env.example                   # Configuration template
```

## Agents

Three Claude Code agents perform sequential translation:

1. **English → French** (`agent_en_to_fr.json`)
   - Translates English to French
   - Preserves spelling errors from input

2. **French → Hebrew** (`agent_fr_to_he.json`)
   - Translates French to Hebrew
   - Handles RTL text encoding

3. **Hebrew → English** (`agent_he_to_en.json`)
   - Translates Hebrew back to English
   - Completes the translation cycle

See [agents/README.md](agents/README.md) for detailed specifications.

## Analysis

### Embedding Model

Uses **Sentence-BERT** (`all-MiniLM-L6-v2`) for semantic similarity:
- 384-dimensional embeddings
- Optimized for English text
- Fast inference (~5ms per sentence)

### Distance Metrics

1. **Cosine Distance** (PRIMARY):
   - Range: [0, 2]
   - 0 = identical, 2 = opposite
   - Optimal for sentence-transformers

2. **Euclidean Distance** (SECONDARY):
   - Range: [0, ∞)
   - 0 = identical
   - Used for comparison

## Results

Expected outcomes:
- **Hypothesis**: Semantic distance increases with spelling error rate
- **Visualization**: Clear upward trend in graphs
- **Quantification**: Measurable correlation between errors and drift

Sample results:
- 0% errors: ~0.00 cosine distance
- 25% errors: ~0.15 cosine distance
- 50% errors: ~0.30 cosine distance

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_analysis/ -v
```

### Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html  # View in browser
```

### Test Standards

- Minimum 70% overall coverage
- 90%+ coverage for core logic
- All tests follow AAA pattern (Arrange-Act-Assert)
- Independent, reproducible tests

## Contributing

### Code Standards

This project follows MSc-level coding standards:
- PEP 8 compliance
- Type hints for all functions
- Google-style docstrings
- Max 50 lines per function
- Max 150 lines per file

### Quality Checks

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
pylint src/
```

## License

Academic use only - MSc Computer Science Project

## Credits

- **Sentence Transformers**: https://sbert.net/
- **Claude Code**: https://claude.com/claude-code
- **Research**: Semantic drift measurement methodologies

## Contact

For questions or issues, please refer to project documentation.
```

**Validation Gate 7.2**:
```bash
test -f README.md && echo "✓ README.md created"
grep -q "Installation" README.md && echo "✓ Installation section present"
grep -q "Usage" README.md && echo "✓ Usage section present"
```

---

#### Task 7.3: Create Architecture Documentation
**Objective**: Document system architecture and design decisions

**File**: `docs/architecture.md`

```markdown
# System Architecture

## Overview

This document describes the architecture of the Multi-Agent Translation Pipeline Experiment system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
├─────────────────────────────────────────────────────────────┤
│  Input Generator                                            │
│  ├─> Sentence Generator (baseline sentences ≥15 words)     │
│  └─> Error Injector (0-50% spelling errors)                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  TRANSLATION LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Controller                                        │
│  ├─> Agent 1: EN → FR (English to French)                  │
│  ├─> Agent 2: FR → HE (French to Hebrew)                   │
│  └─> Agent 3: HE → EN (Hebrew to English)                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   ANALYSIS LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  Semantic Drift Analyzer                                    │
│  ├─> Embedding Generator (Sentence-BERT)                   │
│  └─> Distance Calculator (Cosine + Euclidean)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 VISUALIZATION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Graph Generator                                            │
│  └─> Matplotlib/Seaborn plots (Error % vs Distance)        │
└─────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Input Generator

**Purpose**: Create test inputs with controlled spelling errors

**Components**:
- `sentence_generator.py`: Baseline sentence creation
- `error_injector.py`: Realistic spelling error injection
- `generate_inputs.py`: Orchestration and data export

**Design Decisions**:
- Error types: substitution, omission, duplication
- Preserve word count to maintain sentence structure
- Reproducible with fixed random seeds

### Translation Pipeline

**Purpose**: Orchestrate multi-agent translation

**Components**:
- `pipeline_controller.py`: Sequential agent execution
- Agent definitions (JSON): EN→FR, FR→HE, HE→EN

**Design Decisions**:
- Stateless agents (no memory between calls)
- Sequential execution (required for dependency chain)
- Complete data capture (all intermediate translations)

### Analysis Module

**Purpose**: Measure semantic drift using embeddings

**Components**:
- `embedding_generator.py`: Sentence-BERT embeddings
- `distance_calculator.py`: Similarity metrics
- `semantic_drift_analyzer.py`: End-to-end analysis

**Design Decisions**:
- Model: all-MiniLM-L6-v2 (balance of speed and accuracy)
- Primary metric: Cosine distance (optimal for SBERT)
- Secondary metric: Euclidean distance (comparison)

### Visualization Module

**Purpose**: Generate publication-quality graphs

**Components**:
- `graph_generator.py`: Matplotlib/Seaborn plotting

**Design Decisions**:
- Error bars showing standard deviation
- Multiple views (individual metrics + comparison)
- High-resolution export (300 DPI)

## Data Flow

1. **Input Generation**:
   ```
   Baseline Sentence
   → Error Injection (7 error levels)
   → JSON Export
   ```

2. **Translation Pipeline**:
   ```
   Input Sentence
   → EN→FR Agent
   → FR→HE Agent
   → HE→EN Agent
   → JSON Export (with all intermediates)
   ```

3. **Analysis**:
   ```
   Original & Final Texts
   → SBERT Embeddings
   → Distance Calculation
   → CSV Export
   ```

4. **Visualization**:
   ```
   Analysis CSV
   → Statistical Aggregation
   → Graph Generation
   → PNG Export
   ```

## Technology Decisions

### Why Sentence-BERT?

- **Accuracy**: State-of-the-art for semantic similarity
- **Speed**: Fast inference (~5ms per sentence)
- **Simplicity**: Single model, no fine-tuning required
- **Community**: Well-documented, widely used

Alternatives considered:
- OpenAI embeddings (requires API, costs)
- Universal Sentence Encoder (slower)
- Word2Vec (outdated, lower quality)

### Why Cosine Distance?

- Sentence-BERT models are fine-tuned with cosine similarity
- Range [0, 2] is interpretable and bounded
- Standard in NLP research for semantic similarity

### Why Sequential (Not Parallel) Agent Execution?

- Translation chain requires sequential dependencies
- Preserves causality in semantic drift
- Simplifies error handling and debugging

## Scalability Considerations

### Current Scale

- 5 baseline sentences
- 7 error levels per sentence
- Total: 35 translation experiments
- Runtime: ~5-10 minutes (depending on agent speed)

### Scaling Up

To process more sentences:
1. Batch processing in controller
2. Parallel execution per error level (independent)
3. Caching embeddings to avoid recomputation

## Security Considerations

- No API keys in code (environment variables)
- UTF-8 encoding for all text (multilingual support)
- Input validation to prevent injection attacks
- .gitignore prevents committing secrets

## Testing Strategy

### Unit Tests

- Each component tested independently
- Mock external dependencies (agents)
- AAA pattern (Arrange-Act-Assert)

### Integration Tests

- Full pipeline with sample data
- Verify data flow between components
- Validate output formats

### Coverage Goals

- Overall: ≥70%
- Core logic: ≥90%
- Edge cases: All covered

## Future Enhancements

1. **More Languages**: Extend chain (e.g., EN→FR→HE→ZH→EN)
2. **More Agents**: Compare different translation models
3. **Real-Time Dashboard**: Live monitoring of experiments
4. **Batch API**: Process multiple sentences in parallel
5. **Fine-tuned Embeddings**: Domain-specific SBERT models

## References

- Claude Code Agent Schema
- Sentence-BERT Documentation: https://sbert.net/
- Semantic Drift Research: https://arxiv.org/html/2509.19611
```

**Validation Gate 7.3**:
```bash
test -f docs/architecture.md && echo "✓ Architecture documentation created"
```

---

## 5. Validation Gates

### Master Validation Checklist

Before considering the project complete, verify ALL validation gates pass:

```bash
# Create master validation script
cat > scripts/validate_all.sh << 'EOF'
#!/bin/bash
set -e

echo "========================================="
echo "MASTER VALIDATION CHECK"
echo "========================================="

# Phase 1: Project Structure
echo ""
echo "Phase 1: Project Structure"
test -d src && echo "✓ src/ exists"
test -d tests && echo "✓ tests/ exists"
test -d agents && echo "✓ agents/ exists"
test -f requirements.txt && echo "✓ requirements.txt exists"
test -f .gitignore && echo "✓ .gitignore exists"
test -f .env.example && echo "✓ .env.example exists"
test -f README.md && echo "✓ README.md exists"

# Phase 2: Input Generation
echo ""
echo "Phase 2: Input Generation"
test -f data/input/sentences.json && echo "✓ Input sentences generated"

# Phase 3: Agent Definitions
echo ""
echo "Phase 3: Agent Definitions"
test -f agents/agent_en_to_fr.json && echo "✓ EN→FR agent defined"
test -f agents/agent_fr_to_he.json && echo "✓ FR→HE agent defined"
test -f agents/agent_he_to_en.json && echo "✓ HE→EN agent defined"

# Phase 4: Controller
echo ""
echo "Phase 4: Controller"
test -f src/controller/pipeline_controller.py && echo "✓ Controller implemented"

# Phase 5: Analysis
echo ""
echo "Phase 5: Analysis"
test -f src/analysis/embedding_generator.py && echo "✓ Embedding generator implemented"
test -f src/analysis/distance_calculator.py && echo "✓ Distance calculator implemented"

# Phase 6: Visualization
echo ""
echo "Phase 6: Visualization"
test -f src/visualization/graph_generator.py && echo "✓ Graph generator implemented"

# Phase 7: Testing
echo ""
echo "Phase 7: Testing & Documentation"
pytest tests/ --cov=src --cov-fail-under=70 -q && echo "✓ Tests pass with ≥70% coverage"
test -f README.md && echo "✓ README.md complete"
test -f docs/architecture.md && echo "✓ Architecture documented"

echo ""
echo "========================================="
echo "✓ ALL VALIDATION GATES PASSED"
echo "========================================="
EOF

chmod +x scripts/validate_all.sh
```

Run master validation:
```bash
bash scripts/validate_all.sh
```

---

## 6. Quality Standards

### Code Quality Checklist

All code must comply with MSc standards defined in:
`/mnt/c/Users/bensa/Projects/LLMCourseProject/.claude/skills/user/`

Before finalizing any file:

- [ ] **msc-code-standards**:
  - [ ] PEP 8 compliant (88 char line limit)
  - [ ] Type hints on all functions
  - [ ] Google-style docstrings
  - [ ] Functions ≤50 lines
  - [ ] Files ≤150 lines
  - [ ] No duplicated code (DRY)
  - [ ] Proper error handling

- [ ] **msc-documentation-standards**:
  - [ ] README.md complete with all sections
  - [ ] Code comments explain WHY not WHAT
  - [ ] Architecture documented
  - [ ] Examples provided and tested

- [ ] **msc-security-config**:
  - [ ] No hardcoded secrets
  - [ ] .env files for configuration
  - [ ] .gitignore prevents committing secrets
  - [ ] Input validation on all external inputs

- [ ] **msc-submission-structure**:
  - [ ] Source in src/
  - [ ] Tests in tests/ (mirror src structure)
  - [ ] Proper package organization
  - [ ] requirements.txt complete

- [ ] **msc-testing-standards**:
  - [ ] ≥70% overall coverage
  - [ ] ≥90% core logic coverage
  - [ ] pytest with AAA pattern
  - [ ] Independent, fast tests
  - [ ] All edge cases covered

---

## 7. Risk Mitigation

### Known Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Agent API failures | Medium | High | Stub agents for testing; robust error handling |
| Hebrew encoding issues | Medium | Medium | UTF-8 throughout; test RTL text explicitly |
| Embedding computation slow | Low | Medium | Use MiniLM model (fast); batch processing |
| Test coverage insufficient | Low | High | Write tests incrementally; run coverage checks |
| Error injection too random | Low | Medium | Use fixed seeds; validate error percentages |

### Contingency Plans

**If agent invocation fails**:
1. Use stub implementations for testing
2. Document expected agent behavior
3. Validate controller logic independently

**If embedding computation is too slow**:
1. Use smaller model (all-MiniLM-L6-v2 is already fast)
2. Reduce number of sentences
3. Cache embeddings to disk

**If test coverage is low**:
1. Identify uncovered lines with `pytest --cov-report=html`
2. Add targeted tests for missing coverage
3. Focus on core logic first (90%+ priority)

---

## 8. Success Criteria

### Project Complete When:

✅ **All Phases Implemented**:
- [x] Phase 1: Foundation (directory structure, dependencies)
- [x] Phase 2: Input Generation (sentences with errors)
- [x] Phase 3: Agent Definitions (3 translation agents)
- [x] Phase 4: Controller (pipeline orchestration)
- [x] Phase 5: Analysis (embeddings & distances)
- [x] Phase 6: Visualization (graphs)
- [x] Phase 7: Testing & Documentation

✅ **All Validation Gates Pass**:
- Master validation script runs without errors
- All tests pass with ≥70% coverage
- Graphs generated successfully

✅ **Quality Standards Met**:
- Code follows all 5 MSc skills
- Documentation complete (README, architecture)
- No secrets in repository

✅ **Deliverables Complete**:
1. ✓ Agent definitions (JSON files)
2. ✓ Input sentences with error variants
3. ✓ Pipeline results with intermediate translations
4. ✓ Embedding distance table (CSV)
5. ✓ Visualization graphs (PNG)

✅ **Research Hypothesis Validated**:
- Graphs show clear correlation: error rate ↑ → semantic distance ↑
- Results are reproducible (fixed seeds)
- Methodology is documented

---

## 9. Execution Plan

### Recommended Order of Execution

```bash
# Day 1: Foundation & Input Generation
1. Execute Phase 1 tasks (1.1 - 1.5)
2. Execute Phase 2 tasks (2.1 - 2.3)
   Validation: pytest tests/test_input_generator/ -v

# Day 2: Agents & Controller
3. Execute Phase 3 tasks (3.1 - 3.2)
   Validation: Validate JSON schemas
4. Execute Phase 4 task (4.1)
   Validation: pytest tests/test_controller/ -v

# Day 3: Analysis & Visualization
5. Execute Phase 5 tasks (5.1 - 5.3)
   Validation: pytest tests/test_analysis/ -v
6. Execute Phase 6 task (6.1)
   Validation: pytest tests/test_visualization/ -v

# Day 4: Documentation & Final Validation
7. Execute Phase 7 tasks (7.1 - 7.3)
8. Run master validation script
9. Generate final graphs
10. Review all deliverables
```

### Time Estimates

| Phase | Estimated Time |
|-------|---------------|
| Phase 1: Foundation | 1-2 hours |
| Phase 2: Input Generation | 2-3 hours |
| Phase 3: Agent Definitions | 1-2 hours |
| Phase 4: Controller | 2-3 hours |
| Phase 5: Analysis | 3-4 hours |
| Phase 6: Visualization | 1-2 hours |
| Phase 7: Documentation | 2-3 hours |
| **Total** | **12-19 hours** |

---

## 10. Post-Implementation Checklist

### Before Submission

- [ ] Run full test suite: `pytest tests/ --cov=src --cov-fail-under=70`
- [ ] Run quality checks: `black src/ tests/ && mypy src/ && pylint src/`
- [ ] Generate all graphs: `python -m src.visualization.create_graphs`
- [ ] Verify all graphs exist in `results/graphs/`
- [ ] Review README.md for completeness
- [ ] Check .gitignore prevents committing secrets
- [ ] Verify no `.env` file in repository
- [ ] Run master validation: `bash scripts/validate_all.sh`
- [ ] Create submission archive (if required)

### Submission Package

Include:
1. ✓ All source code (`src/`)
2. ✓ All tests (`tests/`)
3. ✓ Agent definitions (`agents/`)
4. ✓ Input data (`data/input/sentences.json`)
5. ✓ Results (` results/experiments/*.csv`, `results/graphs/*.png`)
6. ✓ Documentation (`README.md`, `docs/architecture.md`, `CLAUDE.md`)
7. ✓ Configuration files (`.env.example`, `requirements.txt`, `.gitignore`)

---

## 11. Quick Start Guide

### For Claude Code Execution

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate inputs
python -m src.input_generator.generate_inputs

# 3. Run experiments (with stub agents)
python -m src.controller.run_experiments

# 4. Analyze results
python -m src.analysis.run_analysis

# 5. Generate graphs
python -m src.visualization.create_graphs

# 6. Run tests
pytest tests/ --cov=src --cov-fail-under=70 -v

# 7. Validate everything
bash scripts/validate_all.sh
```

---

## Appendix A: Key Resources

### Documentation
- Sentence Transformers: https://sbert.net/
- PyPI sentence-transformers: https://pypi.org/project/sentence-transformers/
- Hugging Face Sentence Similarity: https://huggingface.co/tasks/sentence-similarity

### Research Papers
- Semantic Drift in Multilingual Representations: https://direct.mit.edu/coli/article/46/3/571/93376
- Evaluating Translation by Playing Telephone: https://arxiv.org/html/2509.19611
- COMET Neural MT Evaluation: https://aclanthology.org/2020.emnlp-main.213.pdf

### Best Practices
- Cosine vs Euclidean for Embeddings: https://datascience.stackexchange.com/questions/27726
- Distance Metrics for Embeddings: https://developers.google.com/machine-learning/clustering/dnn-clustering/supervised-similarity

---

## Appendix B: Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'sentence_transformers'`
**Solution**: `pip install sentence-transformers`

**Issue**: Hebrew text displays as squares/question marks
**Solution**: Ensure UTF-8 encoding in all files, use `encoding='utf-8'` in file operations

**Issue**: Tests fail with "Agent not found"
**Solution**: Using stub agents for testing; real agents invoked in actual Claude Code environment

**Issue**: Graphs not displaying properly
**Solution**: Ensure matplotlib backend configured: `export MPLBACKEND=Agg`

**Issue**: Coverage below 70%
**Solution**: Run `pytest --cov=src --cov-report=html` and open `htmlcov/index.html` to identify uncovered lines

---

## Confidence Score Justification: 8/10

### Strengths (+)
- Comprehensive research on all technologies
- Clear task breakdown with validation gates
- Well-defined agent schemas
- Proven embedding methodology
- Incremental testing strategy

### Risks (-)
- Agent invocation mechanism not fully specified (stub implementation)
- Hebrew RTL text handling untested in real agents
- Actual translation quality unknown (depends on Claude capabilities)

### Mitigation
- Stub agents allow complete controller testing
- UTF-8 encoding handles Hebrew properly
- Focus on measuring semantic drift (not translation quality)

**Overall**: High confidence for one-pass implementation with stub agents. Real agent integration may require minor adjustments but core system is solid.

---

## End of PRP

**Next Steps**: Begin implementation starting with Phase 1 (Foundation). Execute tasks sequentially, validating each phase before proceeding to the next.
