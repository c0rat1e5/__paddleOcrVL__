#!/bin/bash
# OCR Result Merger Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

# Run the merge app
echo "🔗 Starting OCR Result Merger..."
echo "📍 URL: http://localhost:7862"
echo ""
python app_merge_json.py
