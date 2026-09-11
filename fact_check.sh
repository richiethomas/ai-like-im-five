#!/bin/bash
# Fact-check a blog article using multi-LLM orchestration
# Usage: ./fact_check.sh <path_to_article_mdx>

set -e

ARTICLE_PATH="${1:-.}"

if [ ! -f "$ARTICLE_PATH" ]; then
    echo "❌ Error: File not found: $ARTICLE_PATH"
    exit 1
fi

if [ ! -f "fact_check.py" ]; then
    echo "❌ Error: fact_check.py not found. Run from project root."
    exit 1
fi

# Create venv if needed
VENV_DIR=".venv-fact-check"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv and install deps
source "$VENV_DIR/bin/activate"
if ! python3 -c "import anthropic, openai" 2>/dev/null; then
    echo "📦 Installing dependencies into venv..."
    pip install -q -r requirements_fact_check.txt
fi

# Check API keys
if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Missing API keys:"
    [ -z "$ANTHROPIC_API_KEY" ] && echo "   - ANTHROPIC_API_KEY"
    [ -z "$OPENAI_API_KEY" ] && echo "   - OPENAI_API_KEY"
    echo ""
    echo "Set them in your shell or .env file, then try again."
    exit 1
fi

echo ""
echo "🔍 Fact-checking: $ARTICLE_PATH"
echo "============================================"
echo ""

python3 fact_check.py "$ARTICLE_PATH"

# Show report location
REPORT_PATH="${ARTICLE_PATH%.*}_fact_check.json"
if [ -f "$REPORT_PATH" ]; then
    echo ""
    echo "📄 Full report: $REPORT_PATH"
    echo ""

    # Quick status summary
    STATUS=$(jq -r '.status' "$REPORT_PATH" 2>/dev/null || echo "unknown")
    ACTION_COUNT=$(jq '.agreed_upon_changes | length' "$REPORT_PATH" 2>/dev/null || echo "?")
    BLOCKER_COUNT=$(jq '.consensus_blockers | length' "$REPORT_PATH" 2>/dev/null || echo "?")

    echo "Status: $STATUS"
    echo "Action items (severity > 2): $ACTION_COUNT"
    echo "Consensus blockers: $BLOCKER_COUNT"
    echo ""
fi
