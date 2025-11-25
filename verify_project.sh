#!/bin/bash
# Project verification script

echo "========================================="
echo "Project 3: Multi-Agent Translation Pipeline"
echo "Verification Script"
echo "========================================="
echo ""

# Check required files
echo "📁 Checking required files..."
required_files=(
    "README.md"
    "requirements.txt"
    ".gitignore"
    ".env.example"
    "docs/architecture.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
    fi
done

echo ""
echo "📁 Checking source modules..."
src_modules=(
    "src/input_generator/sentence_generator.py"
    "src/input_generator/error_injector.py"
    "src/input_generator/generate_inputs.py"
    "src/controller/pipeline_controller.py"
    "src/analysis/embedding_generator.py"
    "src/analysis/distance_calculator.py"
    "src/analysis/semantic_drift_analyzer.py"
    "src/visualization/graph_generator.py"
)

for file in "${src_modules[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
    fi
done

echo ""
echo "📁 Checking agent definitions..."
agent_files=(
    "agents/agent_en_to_fr.json"
    "agents/agent_fr_to_he.json"
    "agents/agent_he_to_en.json"
)

for file in "${agent_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
    fi
done

echo ""
echo "📁 Checking test files..."
test_files=(
    "tests/test_input_generator/test_sentence_generator.py"
    "tests/test_input_generator/test_error_injector.py"
    "tests/test_controller/test_pipeline_controller.py"
    "tests/test_analysis/test_distance_calculator.py"
)

for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
    fi
done

echo ""
echo "📁 Checking generated data..."
if [ -f "data/input/sentences.json" ]; then
    count=$(python3 -c "import json; data=json.load(open('data/input/sentences.json')); print(len(data['sentences']) * len(data['metadata']['error_levels']))")
    echo "✅ data/input/sentences.json ($count variants)"
else
    echo "❌ data/input/sentences.json (MISSING)"
fi

echo ""
echo "📊 Project Statistics:"
echo "---------------------"
echo "Source files: $(find src -name '*.py' | wc -l)"
echo "Test files: $(find tests -name 'test_*.py' | wc -l)"
echo "Agent definitions: $(find agents -name '*.json' | wc -l)"
echo "Documentation files: $(find . -maxdepth 2 -name '*.md' | wc -l)"

echo ""
echo "========================================="
echo "✅ Verification Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Install deps: pip install -r requirements.txt"
echo "3. Run tests: pytest tests/ -v"
echo "4. See PROJECT_SUMMARY.md for details"
