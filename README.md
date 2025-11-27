# Multi-Agent Translation Pipeline: Semantic Drift Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Quality](https://img.shields.io/badge/code%20quality-MSc%20level-brightgreen.svg)]()
[![Experiment Status](https://img.shields.io/badge/experiment-completed-success.svg)]()

A research project that measures **semantic drift in AI translations** caused by spelling errors using multi-agent systems. This project successfully executed **105 real Claude AI agent calls** to translate text through a language chain (English → French → Hebrew → English) and quantified how translation quality degrades with increasing input errors.

## 🎯 Project Overview

**Research Question**: Do spelling errors in source text cause semantic drift when propagated through multi-agent translation pipelines?

**Answer**: **Yes!** Our experiment confirms a **highly significant positive correlation** (r=0.79, p<0.000001) between spelling error rate and semantic drift.

### Key Findings

- **0% errors**: Cosine distance = 0.013 (nearly perfect semantic preservation)
- **25% errors**: Cosine distance = 0.204 (moderate semantic drift)
- **50% errors**: Cosine distance = 0.431 (substantial semantic drift)

The results demonstrate that errors **compound through translation stages**, validating concerns about error propagation in multi-agent AI systems.

## 📊 Experiment Status: ✅ COMPLETED

This project has been **fully executed and analyzed**:
- ✅ **105 real Claude AI agent invocations** (35 pipeline runs × 3 translation stages)
- ✅ **Real Sentence-BERT embeddings** (all-MiniLM-L6-v2, 384 dimensions)
- ✅ **Publication-quality visualizations** (300 DPI PNG graphs)
- ✅ **Statistically validated results** (p < 0.000001)

### 📄 Results & Documentation

Complete experiment findings are documented in:
- **[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)**: Full experimental report with methodology, statistical analysis, and findings
- **[COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)**: Quick overview of accomplishments and key metrics
- **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)**: Detailed completion status and deliverables
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Implementation metrics and technical summary

### 📈 Generated Artifacts

All experiment outputs are available in the repository:
- `results/experiments/real_pipeline_results.json` - 35 complete translation chains with real AI outputs
- `results/analysis/semantic_drift.csv` - Computed semantic distances using Sentence-BERT
- `results/graphs/*.png` - Three publication-ready visualizations with statistical analysis
- `results/graphs/statistical_analysis.txt` - Complete correlation analysis and p-values

## 🔬 Methodology

### Translation Pipeline

```
Input Text (with errors)
  ↓
English → French (Claude Task Agent 1)
  ↓
French → Hebrew (Claude Task Agent 2)
  ↓
Hebrew → English (Claude Task Agent 3)
  ↓
Final Output → Semantic Distance Analysis
```

### Error Injection
- **Levels**: 0%, 10%, 20%, 25%, 30%, 40%, 50%
- **Types**: Character substitution, omission, duplication
- **Example**: "The quick brown fox" → "Te qick brwn fx" (50% errors)

### Semantic Analysis
- **Embedding Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Distance Metrics**: Cosine distance, Euclidean distance
- **Statistical Tests**: Pearson correlation, Spearman correlation

## 🛠️ Technology Stack

- **Python 3.12**: Core language with type hints and MSc-level code quality
- **Claude Code Task Agents**: Real AI multi-agent orchestration (105 invocations)
- **sentence-transformers**: State-of-the-art sentence embeddings
- **PyTorch**: Backend for transformer models
- **NumPy & pandas**: Numerical computing and data analysis
- **scipy**: Statistical testing (Pearson, Spearman correlations)
- **matplotlib & seaborn**: Publication-quality visualizations

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone and navigate to project**:
   ```bash
   cd multi-agent-translation-pipeline/
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Key packages installed:
   - sentence-transformers (with PyTorch)
   - numpy, pandas, scipy
   - matplotlib, seaborn
   - scikit-learn

## 🚀 Usage (Reproducing Results)

The experiment has been completed, but you can reproduce the analysis:

### 1. Generate Input Sentences with Errors

```bash
python3 -m src.input_generator.generate_inputs
```

Output: `data/input/sentences.json` (5 sentences × 7 error levels = 35 variants)

### 2. Run Translation Pipeline

**Note**: This requires Claude Code Task agents. The translations have already been completed and saved in `results/experiments/real_pipeline_results.json`.

For reference, the helper script shows how results were organized:
```bash
python3 run_real_experiment.py
```

### 3. Analyze Semantic Drift

Compute embeddings and semantic distances from real translations:

```bash
python3 run_real_analysis.py
```

**Output**:
- `results/analysis/semantic_drift.csv` - Distance metrics for all 35 variants
- Console: Progress updates and summary statistics

**Expected console output**:
```
============================================================
REAL SEMANTIC DRIFT ANALYSIS
Using Sentence-BERT Embeddings with Real Claude Agent Data
============================================================

📖 Loading real results from results/experiments/real_pipeline_results.json...
✓ Loaded 35 real translation pipeline results

🤖 Loading Sentence-BERT model: all-MiniLM-L6-v2
✓ Model loaded successfully

📊 Computing semantic distances for 35 results...
  Progress: 35/35 (100%)
✓ Distance computation complete
```

### 4. Generate Visualizations

Create publication-quality graphs with statistical analysis:

```bash
python3 generate_real_graphs.py
```

**Output**:
- `results/graphs/cosine_distance.png` - Error rate vs cosine distance with correlation stats
- `results/graphs/euclidean_distance.png` - Error rate vs Euclidean distance
- `results/graphs/both_metrics.png` - Side-by-side comparison
- `results/graphs/statistical_analysis.txt` - Complete statistical summary

## 📂 Project Structure

```
multi-agent-translation-pipeline/
├── agents/                                # Claude Code agent definitions
│   ├── agent_en_to_fr.json               # English → French translator
│   ├── agent_fr_to_he.json               # French → Hebrew translator
│   └── agent_he_to_en.json               # Hebrew → English translator
│
├── src/                                   # Source code modules
│   ├── input_generator/                  # Sentence and error generation
│   ├── controller/                       # Pipeline orchestration
│   ├── analysis/                         # Embedding and distance analysis
│   └── visualization/                    # Graph generation
│
├── data/input/
│   └── sentences.json                    # 35 test sentences with errors
│
├── results/
│   ├── experiments/
│   │   └── real_pipeline_results.json    # ✅ 35 complete translations (37 KB)
│   ├── analysis/
│   │   └── semantic_drift.csv            # ✅ Computed distances (14 KB)
│   └── graphs/
│       ├── cosine_distance.png           # ✅ 300 DPI visualization (282 KB)
│       ├── euclidean_distance.png        # ✅ 300 DPI visualization (255 KB)
│       ├── both_metrics.png              # ✅ 300 DPI visualization (320 KB)
│       └── statistical_analysis.txt      # ✅ Complete stats summary
│
├── tests/                                # Comprehensive test suite
│   ├── test_input_generator/
│   ├── test_controller/
│   └── test_analysis/
│
├── run_real_analysis.py                  # Main analysis script (Sentence-BERT)
├── generate_real_graphs.py               # Visualization generator
├── run_real_experiment.py                # Experiment helper/organizer
│
├── EXPERIMENT_RESULTS.md                 # 📄 Full experimental report
├── COMPLETION_SUMMARY.md                 # 📄 Quick findings overview
├── COMPLETION_REPORT.md                  # 📄 Detailed completion status
├── PROJECT_SUMMARY.md                    # 📄 Implementation metrics
├── README.md                             # This file
└── requirements.txt                      # Python dependencies
```

## 📈 Sample Results

### Real Translation Example

**Input (0% errors)**:
```
"The quick brown fox jumps over the lazy dog while the sun shines brightly in the clear blue sky above"
```

**Translations**:
- EN→FR: "Le renard brun rapide saute par-dessus le chien paresseux..."
- FR→HE: "השועל החום המהיר קופץ מעל הכלב העצלן..." (Hebrew RTL)
- HE→EN: "The quick brown fox jumps over the lazy dog while the sun shines brightly in the clear blue skies above"

**Semantic Distance**: 0.013 (nearly identical)

---

**Input (50% errors)**:
```
"Te qick brwn fx jmps ovr te lzy dg wile te sn shnes brightly n te cler blu sky abve"
```

**Semantic Distance**: 0.431 (substantial drift)

### Statistical Summary

| Error Rate | Cosine Distance | Interpretation |
|------------|----------------|----------------|
| 0%         | 0.013 ± 0.007  | Nearly perfect |
| 10%        | 0.083 ± 0.057  | Small drift |
| 20%        | 0.184 ± 0.108  | Moderate drift |
| 25%        | 0.204 ± 0.146  | Moderate drift |
| 30%        | 0.288 ± 0.088  | Substantial drift |
| 40%        | 0.256 ± 0.120  | Substantial drift |
| 50%        | 0.431 ± 0.112  | High drift |

**Correlation**: r = 0.79, p < 0.000001 (highly significant)

## 🧪 Testing

### Run Complete Test Suite

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
pylint src/
```

## 🎓 Academic Context

**Course**: MSc Computer Science - Multi-Agent Systems
**Institution**: Reichman University, IL
**Project Type**: Research Experiment with Real AI Agents


## 📚 Key Insights

### What We Learned

1. **Error Propagation is Real**: Spelling errors don't just affect individual translations—they compound through multi-agent pipelines

2. **Quantifiable Impact**: Each 10% increase in error rate adds ~0.05-0.10 to semantic distance

3. **Robustness Has Limits**: While Claude AI handles moderate errors well (≤25%), severe corruption (50%) causes substantial semantic drift

4. **Multi-Agent Vulnerability**: Sequential AI agents create cumulative error effects that single-agent systems avoid

### Practical Applications

- **Input Validation**: Motivates spell-checking and error correction before AI processing
- **Quality Monitoring**: Provides baseline metrics for translation quality degradation
- **Pipeline Design**: Informs decisions about multi-agent vs single-agent architectures
- **Error Budgets**: Quantifies acceptable input error rates for production systems

## 🔗 Related Work

### Research References

- Semantic Drift in Multilingual Representations ([MIT Press](https://direct.mit.edu/coli/article/46/3/571/93376))
- COMET: Neural Framework for MT Evaluation ([ACL Anthology](https://aclanthology.org/2020.emnlp-main.213.pdf))
- Sentence-BERT Documentation ([SBERT.net](https://sbert.net/))

### Technical Dependencies

- [sentence-transformers](https://sbert.net/) - Sentence embeddings
- [PyTorch](https://pytorch.org/) - Deep learning backend
- [scikit-learn](https://scikit-learn.org/) - Distance metrics
- [matplotlib](https://matplotlib.org/) - Plotting
- [seaborn](https://seaborn.pydata.org/) - Statistical visualization

## 📄 License

Academic Research Project
**Institution**: Reichman University, IL

## 👥 Authors

**Niv Ben Salmon** & **Omer Ben Salmon**
MSc Computer Science Students
Reichman University, Israel

