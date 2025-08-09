# Equity Trading Research Platform

This project is a comprehensive framework for backtesting equity trading strategies, analyzing factors, optimizing portfolios, and conducting risk analysis.

## Overview

The platform consists of the following components:

1. **Database Manager (`DatabaseManager2`)**: Handles data storage and retrieval using SQLite for metadata and Parquet files for time series data.
2. **Data Service (`DataService`)**: Provides business logic and methods for accessing and analyzing data.
3. **Web Application (`app_sqlite_2.py`)**: Streamlit-based user interface for visualizing and interacting with the data.

## Features

- Factor analysis and visualization
- Portfolio construction based on factor exposures
- Backtesting of factor strategies
- Pairs trading analysis
- Risk decomposition and attribution
- Performance metrics visualization

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly pyarrow sqlite3 scikit-learn
```

3. Create the required directory structure:

```bash
mkdir -p data/time_series/factor_returns data/time_series/pairs data/time_series/backtest_results
```

## Usage

1. Run the application:

```bash
streamlit run app_sqlite_2.py
```

2. On first run, click "Generate Demo Data" to create sample data.
3. Use the sidebar to configure analysis parameters.
4. Explore the different tabs to analyze factors, create portfolios, and view backtest results.

## Project Structure

```
/
├── app_sqlite_2.py         # Streamlit web application
├── data/                   # Data directory
│   ├── market_data.db      # SQLite database for metadata
│   └── time_series/        # Parquet files for time series data
├── src/                    # Source code
│   ├── database_sqlite_2.py  # Database manager
│   └── data_service.py     # Data service
```

## How It Works

### Data Storage

The platform uses a hybrid storage approach:

- **SQLite Database**: Stores metadata about indices, securities, factors, and backtest strategies.
- **Parquet Files**: Store time series data (prices, factors, exposures) for efficient retrieval.

### Key Components

1. **Database Manager**: Handles data storage and retrieval operations.
2. **Data Service**: Provides higher-level methods for data analysis.
3. **Web Application**: Visualizes data and allows user interaction.

## Demo Data

The demo data generation creates:

- Two indices (S&P 500, NASDAQ 100) with constituents
- 100 securities across 10 sectors
- 6 factors (value, momentum, size, quality, volatility, growth)
- Factor returns and exposures
- Backtest results for factor strategies
- Pairs trading examples

## Extensions

The platform can be extended in several ways:

1. Add more factors or custom factors
2. Implement additional trading strategies
3. Add machine learning models for prediction
4. Integrate with external data sources
5. Add portfolio optimization functionality

## License

This project is available under the MIT License.