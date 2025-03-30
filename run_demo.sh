#!/bin/bash
# run_demo.sh - Script to set up and run the Equity Trading Research Platform

# Print header
echo "================================================================="
echo "       Setting up Equity Trading Research Platform Demo"
echo "================================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run project setup
echo "Setting up project structure..."
python setup_project.py

# Generate demo data
echo "Generating demo data (this may take a minute)..."
python -c "from src.database import DatabaseManager; db = DatabaseManager(); db.reset_database(); db.generate_demo_data()"

# Run the app
echo "================================================================="
echo "Starting Streamlit app. Press Ctrl+C to stop."
echo "================================================================="
streamlit run app.py