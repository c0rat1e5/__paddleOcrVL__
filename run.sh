#!/bin/bash
# PaddleOCR-VL Gradio Application Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

# Run the app
echo "🚀 Starting PaddleOCR-VL Demo..."
echo "📍 URL: http://localhost:7860"
echo ""
python app.py
