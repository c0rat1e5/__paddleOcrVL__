#!/bin/bash
# PaddleOCR-VL Batch Folder Processing Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

# Run the batch processing app
echo "🚀 Starting PaddleOCR-VL Batch Folder Processing..."
echo "📍 URL: http://localhost:7861"
echo ""
echo "📁 フォルダ内の全画像を一括OCR処理できます"
echo ""
python app_batch_folder.py
