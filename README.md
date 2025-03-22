# Equity Trading Research Platform

A comprehensive platform for analyzing equity factors, optimizing portfolios, and backtesting trading strategies.

## Features
- Factor Analysis
- Portfolio Optimization
- Backtest Results
- Risk Analysis
- Earnings Research and Analysis
- Integration with AI models

## Setup
1. Clone the repository
```bash
git clone https://github.com/aah3/equity-platform.git
cd equity-platform
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Plotly

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
