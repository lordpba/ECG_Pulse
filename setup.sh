#!/bin/bash

# PULSE-7B ECG Analysis Interface - Quick Setup Script
# This script sets up the environment and installs all dependencies

echo "🩺 PULSE-7B ECG Analysis Interface Setup"
echo "========================================"

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -Po "(?<=Python )\d+\.\d+")
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Found Python $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: python app_gradio.py"
echo "3. Open http://127.0.0.1:7860 in your browser"
echo ""
echo "⚠️  Note: First run will download the PULSE-7B model (~14GB)"
echo "    Make sure you have sufficient disk space and internet connection."