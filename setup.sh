#!/bin/bash

# MCA Stakeholder Analysis System - Quick Setup Script
# This script helps set up the environment for the hackathon demo

echo "🏛️  MCA Stakeholder Analysis System Setup"
echo "========================================"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip found: $(pip --version)"

# Install requirements
echo "📦 Installing Python packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Failed to install some packages. Please check the error messages above."
    exit 1
fi

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('✅ NLTK data downloaded successfully!')
except Exception as e:
    print(f'❌ Failed to download NLTK data: {e}')
"

echo ""
echo "🚀 Setup completed! You can now run:"
echo ""
echo "1. Basic demo (no dependencies):"
echo "   python simple_demo.py"
echo ""
echo "2. Advanced features demo:"
echo "   python advanced_features_demo.py"
echo ""
echo "3. Interactive dashboard:"
echo "   streamlit run stakeholder_analysis_prototype.py"
echo ""
echo "📊 Happy analyzing!"