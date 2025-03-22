# setup_project.py
"""
Set up the project structure for the Equity Trading Research App
"""

import os
import shutil

def create_directory_structure():
    """Create the necessary directories for the project"""
    directories = [
        "src",
        "data",
        "data/historical",
        "data/factors",
        "models",
        "results",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    packages = [
        "src",
        "tests"
    ]
    
    for package in packages:
        init_file = os.path.join(package, "__init__.py")
        with open(init_file, 'w') as f:
            f.write("# This file makes the directory a Python package\n")
        print(f"Created file: {init_file}")

def copy_source_modules():
    """Copy the example source modules to the src directory"""
    # This would normally copy your existing modules, but for the example
    # we'll just create placeholder files
    
    modules = [
        "qFactor.py",
        "qOptimization.py", 
        "qBacktest.py",
        "utils.py",
        "logger.py"
    ]
    
    for module in modules:
        # In a real scenario, you would copy the actual files
        # shutil.copy(f"original/{module}", f"src/{module}")
        
        # For this example, we'll create placeholder files
        with open(f"src/{module}", 'w') as f:
            f.write(f"# Placeholder for {module}\n")
            f.write("# This file would contain the actual implementation\n")
        
        print(f"Created module placeholder: src/{module}")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = [
        "streamlit>=1.22.0",
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "plotly>=5.14.1",
        "scipy>=1.10.1",
        "statsmodels>=0.13.5",
        "scikit-learn>=1.2.2",
        "cvxpy>=1.3.1",
        "pydantic>=2.0.0",
        "great-tables>=0.1.0",
        "boto3>=1.26.0"
    ]
    
    with open("requirements.txt", 'w') as f:
        f.write("\n".join(requirements))
    
    print("Created requirements.txt")

def create_readme():
    """Create README.md file"""
    readme_content = """# Equity Trading Research Platform

## Overview
A web application for equity traders and researchers to analyze factors, optimize portfolios, and backtest trading strategies.

## Features
- Factor analysis and visualization
- Portfolio optimization
- Backtest execution and analysis
- Risk decomposition
- Earnings forecast analysis

## Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Project Structure
- `app.py`: Main Streamlit application
- `src/`: Source code modules
  - `qFactor.py`: Factor analysis module
  - `qOptimization.py`: Portfolio optimization module
  - `qBacktest.py`: Backtesting framework
  - `utils.py`: Utility functions
  - `logger.py`: Logging functionality
- `data/`: Data storage
- `models/`: Saved models and parameters
- `results/`: Analysis results
- `logs/`: Application logs
- `tests/`: Test cases

## Requirements
See `requirements.txt` for the full list of dependencies.
"""
    
    with open("README.md", 'w') as f:
        f.write(readme_content)
    
    print("Created README.md")

def main():
    """Main function to set up the project"""
    print("Setting up Equity Trading Research Platform project structure...")
    
    # create_directory_structure()
    # create_init_files()
    # copy_source_modules()
    create_requirements_file()
    # create_readme()
    
    print("\nProject setup complete!")
    print("To run the app: streamlit run app.py")

if __name__ == "__main__":
    main()