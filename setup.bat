@echo off
REM MCA Stakeholder Analysis System - Windows Setup Script
REM This script helps set up the environment for the hackathon demo

echo 🏛️  MCA Stakeholder Analysis System Setup
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found
pip --version

REM Install requirements
echo 📦 Installing Python packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install some packages. Please check the error messages above.
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!

REM Download NLTK data
echo 📚 Downloading NLTK data...
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('✅ NLTK data downloaded successfully!')"

echo.
echo 🚀 Setup completed! You can now run:
echo.
echo 1. Basic demo (no dependencies):
echo    python simple_demo.py
echo.
echo 2. Advanced features demo:
echo    python advanced_features_demo.py
echo.
echo 3. Interactive dashboard:
echo    streamlit run stakeholder_analysis_prototype.py
echo.
echo 📊 Happy analyzing!
pause