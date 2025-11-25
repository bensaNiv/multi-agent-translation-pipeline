# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Project 3: Multi-Agent Translation Pipeline Experiment** - a research project designed to measure semantic drift in translations caused by spelling errors. The system uses Claude Code agents to translate text through a language chain (English → French → Hebrew → English) and analyzes how translation quality degrades with increasing spelling errors.

## MSc Standards and Quality Requirements

**CRITICAL**: This project must adhere to MSc-level coding and documentation standards. The following skills define HOW to build quality software:

### Available Skills (Located in parent directory)
These skills are available in `/mnt/c/Users/bensa/Projects/LLMCourseProject/.claude/skills/user/`:

1. **msc-code-standards**: Python coding standards including:
   - PEP 8 compliance with 88-character line limit
   - Mandatory type hints for all functions
   - Google-style docstrings (required for all public functions/classes)
   - Maximum 50 lines per function, 150 lines per file
   - DRY principle and separation of concerns
   - Proper error handling with informative exceptions

2. **msc-documentation-standards**: Documentation requirements including:
   - Comprehensive README.md with installation, usage, examples, and testing sections
   - Architecture documentation with diagrams
   - API documentation for all public interfaces
   - Code comments that explain WHY, not WHAT
   - Research documentation for experiments

3. **msc-security-config**: Security and configuration standards including:
   - NEVER hardcode secrets or configuration in code
   - Use environment variables and .env files
   - Proper .gitignore to exclude secrets
   - Input validation and sanitization
   - Secure file operations

4. **msc-submission-structure**: Required project structure including:
   - Mandatory files: README.md, requirements.txt, .gitignore
   - Source code in `src/` directory
   - Tests in `tests/` directory (mirroring src structure)
   - Proper package organization with `__init__.py`
   - Data and results organization

5. **msc-testing-standards**: Testing requirements including:
   - Minimum 70% test coverage (90%+ for core logic)
   - pytest framework with AAA pattern (Arrange-Act-Assert)
   - Tests for normal cases, edge cases, and error conditions
   - Independent, fast tests (< 1 second each)
   - Fixtures and parameterized tests

### How to Apply These Standards

When implementing ANY part of this project:
1. **Before writing code**: Review the relevant skill (e.g., msc-code-standards)
2. **During development**: Follow the patterns and conventions defined in the skills
3. **Before committing**: Verify compliance using the checklists in each skill
4. **Quality gates**: Code must pass linting (black, pylint), type checking (mypy), and testing (pytest with coverage)

**Example Quality Workflow**:
```bash
# Format code
black src/ tests/

# Check types
mypy src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70

# Lint code
pylint src/
```

## Project Architecture

### High-Level System Design

The project consists of four main components that work together in a pipeline:

1. **Three Translation Agents** (Claude Code Agent Schema-compliant):
   - `agent_en_to_fr.json`: English → French translator
   - `agent_fr_to_he.json`: French → Hebrew translator
   - `agent_he_to_en.json`: Hebrew → English translator
   - Each agent has defined input/output schemas and operates independently

2. **Experiment Controller** (orchestration layer):
   - Can be implemented as either a Python script or a 4th Claude Code agent
   - Coordinates the sequential execution of all three translation agents
   - Manages data flow between agents and collects intermediate results
   - Saves all outputs to structured format (CSV/JSON) for analysis

3. **Input Generation Module**:
   - Creates test sentences (≥15 words each)
   - Generates versions with varying spelling error rates (0%, 10%, 20%, 25%, 30%, 40%, 50%)
   - Ensures at least 25% error rate for main test cases

4. **Analysis Module** (Python):
   - Computes sentence embeddings using standard NLP libraries
   - Calculates semantic distance (cosine/Euclidean) between original and final translations
   - Generates visualization graphs (error rate vs. semantic distance)

### Data Flow

```
Input Sentence (with errors)
  → Agent 1 (EN→FR)
  → Agent 2 (FR→HE)
  → Agent 3 (HE→EN)
  → Output Sentence
  → Embedding Analysis
  → Visualization
```

All intermediate translations are captured and stored for analysis.

## Claude Code Agent Schema Requirements

When creating translation agents, each must include:
- `name`: Descriptive agent identifier
- `description`: What the agent does
- `skills`: List of capabilities
- `input_schema`: JSON schema defining expected input format
- `output_schema`: JSON schema defining output format
- Any transformation rules specific to that language pair

The agents should be designed to work independently - they receive text input and return translated text output, nothing more.

## Project Structure

```
project3/
├── task-descriptions/           # Project requirements and specifications
│   └── multi-agent-translation-description.md
├── agents/                      # Translation agent definitions (to be created)
│   ├── agent_en_to_fr.json
│   ├── agent_fr_to_he.json
│   └── agent_he_to_en.json
├── controller/                  # Orchestration logic (to be created)
│   └── experiment_controller.py
├── data/                        # Input sentences and results (to be created)
│   ├── input_sentences.json
│   └── experiment_results.csv
├── analysis/                    # Embedding analysis and visualization (to be created)
│   ├── embedding_analysis.py
│   └── generate_graphs.py
└── outputs/                     # Generated graphs and reports (to be created)
    └── graphs/
```

## Key Implementation Constraints

### Input Sentence Requirements
- Minimum 15 words per sentence
- Must generate versions at error rates: 0%, 10%, 20%, 25%, 30%, 40%, 50%
- At least 25% spelling errors required for main test cases
- Preserve word count when introducing errors (replace characters, not words)

### Agent Requirements
- Must follow Claude Code Agent Schema strictly
- Each agent is single-purpose (one language direction only)
- Input/output schemas must be consistent: `{"text": "string"}` → `{"translated_text": "string"}`
- No state sharing between agents - pure functional transformation

### Controller Requirements
- Must execute agents sequentially (no parallel execution)
- Capture and store all intermediate outputs
- Record metadata: original text, error rate, timestamps
- Export results in format compatible with Python analysis tools

### Analysis Requirements
- Use standard embedding libraries (sentence-transformers, OpenAI embeddings, or similar)
- Calculate both cosine and Euclidean distances for comparison
- Generate clear matplotlib/seaborn visualizations
- X-axis: spelling error percentage (0-50%)
- Y-axis: embedding distance from original

## Development Workflow

### Phase 1: Agent Creation
1. Create three translation agent JSON/YAML files following Claude Code Agent Schema
2. Validate schema compliance
3. Test each agent independently with sample inputs

### Phase 2: Input Generation
1. Create baseline sentences (≥15 words)
2. Implement error injection algorithm to generate variants
3. Validate error percentages match targets
4. Save all inputs for reproducibility

### Phase 3: Pipeline Implementation
1. Build controller to orchestrate agent calls
2. Implement data collection and storage
3. Test end-to-end pipeline with sample inputs
4. Verify all intermediate translations are captured

### Phase 4: Analysis
1. Load experiment results
2. Generate embeddings for original and final sentences
3. Calculate distance metrics
4. Create visualizations
5. Generate final report with graphs

## Submission Checklist

The final submission must include:
- [ ] All agent definition files (3 translation agents + optional controller agent)
- [ ] Input sentences at all error levels (0-50%)
- [ ] Complete pipeline outputs (all intermediate translations)
- [ ] Embedding distance table
- [ ] Graphs showing error rate vs. semantic drift
- [ ] (Optional) Python scripts for embedding analysis

## Important Notes

### Language Considerations
- **Hebrew text handling**: Ensure proper RTL (right-to-left) text encoding in agents and data files
- **Character encoding**: Use UTF-8 throughout the pipeline to support all three languages
- **Spelling errors**: Should be realistic (character substitutions, omissions) not random noise

### Agent Communication
- Agents do not communicate directly with each other
- Controller handles all inter-agent data passing
- Use JSON for all structured data exchange

### Error Handling
- Translation agents should handle malformed input gracefully
- Controller should log failures but continue processing remaining inputs
- Analysis module should skip incomplete experiments rather than failing

### Performance Expectations
- Each translation may take several seconds (LLM-based)
- Full experiment (7 error levels × N sentences × 3 translations) will take time
- Consider batching or parallelization for large-scale experiments

## Testing Strategy

1. **Unit Testing**: Test each agent independently with known translations
2. **Integration Testing**: Run full pipeline with one sentence at one error level
3. **End-to-End Testing**: Complete experiment with all error levels
4. **Validation**: Manually verify sample translations for quality

## Common Pitfalls to Avoid

- Don't hard-code language models or API keys in agent definitions
- Don't assume agents maintain state between calls
- Don't skip intermediate data storage (needed for analysis)
- Don't generate graphs without validating embedding calculations first
- Don't forget to handle Unicode properly for Hebrew text
