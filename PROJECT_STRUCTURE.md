# Equity Factor Analysis Platform - Project Structure & Context

## Project Overview

This is a comprehensive **Equity Factor Analysis Platform** built with Streamlit that provides tools for:
- Factor analysis and visualization
- Portfolio optimization (Pure Factor, Tracking Error, Risk Parity)
- Backtesting and risk analytics
- Portfolio upload and analysis
- Report generation
- Cloud data synchronization (AWS S3)

## Core Application Architecture

### Main Application (`app_factors.py`)
- **Primary Streamlit application** with 8 main tabs
- Handles data management, factor analysis, portfolio optimization, and reporting
- Integrates with multiple backend modules for comprehensive financial analysis
- Supports multiple optimization objectives and risk management strategies

## Project File Structure

```
EquityApp/
├── 📁 PROJECT_CONTEXT/                    # Context files for development
│   ├── PROJECT_STRUCTURE.md              # This file - complete project overview
│   ├── DEVELOPMENT_GUIDELINES.md         # Coding standards and best practices
│   ├── ARCHITECTURE_DIAGRAM.md          # System architecture and data flow
│   ├── API_REFERENCE.md                 # Key functions and classes reference
│   ├── TESTING_STRATEGY.md              # Testing approach and test cases
│   └── DEPLOYMENT_GUIDE.md              # Deployment and production setup
│
├── 📄 app_factors.py                     # MAIN APPLICATION - Primary Streamlit app
├── 📄 app.py                            # Legacy/alternative app entry point
├── 📄 app_sqlite.py                     # SQLite-based app variant
├── 📄 app_sqlite_2.py                   # Enhanced SQLite app variant
├── 📄 app_db_integration.py             # Database integration app
│
├── 📁 src/                              # Core source code modules
│   ├── __init__.py
│   ├── qFactor.py                       # Factor definitions and analytics
│   ├── qOptimization.py                 # Portfolio optimization algorithms
│   ├── qBacktest.py                     # Backtesting framework
│   ├── portfolio_analysis.py            # Portfolio upload, validation, analysis
│   ├── report_generator.py              # PDF report generation
│   ├── file_data_manager.py             # File-based data management
│   ├── etl_universe_data.py             # ETL pipeline for universe data
│   ├── database_sqlite.py               # SQLite database operations
│   ├── database_postgres.py             # PostgreSQL database operations
│   ├── utils.py                         # Utility functions
│   ├── logger.py                        # Logging functionality
│   └── analyzers_and_reports.py         # Analysis and reporting utilities
│
├── 📁 data/                             # Data storage directory
│   ├── market_data.db                   # SQLite database
│   ├── portfolios/                      # Portfolio data files
│   │   ├── portfolio_sample.csv
│   │   └── te_portfolio_indu.csv
│   ├── static/                          # Static reference data
│   │   └── spx_universe.csv
│   └── time_series/                     # Time series data (Parquet format)
│       ├── backtest_results/
│       ├── benchmarks/
│       ├── exposures/
│       ├── factor_returns/
│       ├── factors/
│       ├── pairs/
│       ├── portfolios/
│       ├── prices/
│       └── returns/
│
├── 📁 tests/                            # Test suite
│   ├── __init__.py
│   ├── test_factors.py                  # Factor analysis tests
│   ├── test_optimization.py             # Optimization algorithm tests
│   ├── test_backtest.py                 # Backtesting tests
│   └── test_portfolio_analysis.py       # Portfolio analysis tests
│
├── 📁 logs/                             # Application logs
│   ├── application.log
│   └── database.log
│
├── 📁 models/                           # Saved models and parameters
├── 📁 results/                          # Analysis results output
├── 📁 venv/                             # Python virtual environment
│
├── 📄 requirements.txt                  # Python dependencies
├── 📄 requirements01.txt                # Alternative requirements
├── 📄 README.md                         # Project documentation
├── 📄 README model.md                   # Model-specific documentation
├── 📄 setup_project.py                  # Project setup script
├── 📄 run_demo.sh                       # Demo execution script
└── 📄 .gitignore                        # Git ignore rules
```

## Key Application Components

### 1. Main Application Tabs (`app_factors.py`)

#### Tab 1: Factor Analysis
- Factor return analysis and visualization
- Factor exposure distributions
- Autocorrelation analysis
- Portfolio performance metrics

#### Tab 2: Portfolio Optimization
- Unified optimization interface
- Supports multiple objectives (Pure Factor, Tracking Error, etc.)
- Real-time optimization results

#### Tab 3: Pure Factor Portfolios
- Pure factor portfolio analysis
- Date range filtering
- Performance statistics and turnover analysis
- Download capabilities

#### Tab 4: Tracking Error Optimization
- Advanced tracking error optimization
- Constraint configuration
- Backtest integration
- Comprehensive performance analytics

#### Tab 5: Risk Analysis
- Factor risk decomposition
- Exposure analysis
- Extreme exposure tracking

#### Tab 6: Portfolio Upload & Analysis
- Portfolio upload and validation
- Backtest-based performance analysis
- Risk metrics (VaR, drawdown, Sharpe ratio)
- Concentration analysis

#### Tab 7: Report Generation
- PDF report generation
- Multi-tab data consolidation
- Customizable reports

#### Tab 8: Documentation
- Comprehensive user guide
- API reference
- Troubleshooting guide

### 2. Core Modules (`src/`)

#### `qFactor.py` - Factor Framework
- **EquityFactor**: Core factor analysis class
- **EquityFactorModelInput**: Configuration management
- **RiskFactors**: Factor definitions (beta, size, value, momentum)
- **Universe**: Investment universe definitions
- **DataSource**: Data source configurations
- **SecurityMasterFactory**: Security data management

#### `qOptimization.py` - Optimization Engine
- **PureFactorOptimizer**: Pure factor portfolio optimization
- **TrackingErrorOptimizer**: Tracking error optimization
- **OptimizationObjective**: Optimization objectives enum
- **PurePortfolioConstraints**: Constraint definitions
- **TrackingErrorConstraints**: TE-specific constraints

#### `qBacktest.py` - Backtesting Framework
- **Backtest**: Main backtesting class
- **BacktestConfig**: Configuration management
- **AssetClass**: Asset class definitions
- **PortfolioType**: Portfolio type definitions

#### `portfolio_analysis.py` - Portfolio Analytics
- **PortfolioAnalyzer**: Portfolio analysis engine
- **PortfolioValidator**: Portfolio validation
- **PortfolioComparator**: Portfolio comparison utilities
- Upload and validation UI components

#### `file_data_manager.py` - Data Management
- **FileDataManager**: File-based data operations
- **FileConfig**: Configuration management
- Data loading and saving operations

### 3. Data Architecture

#### Data Sources
- **Yahoo Finance**: Primary data source
- **Bloomberg**: Professional data source
- **Custom**: User-provided data

#### Data Storage
- **SQLite**: Metadata and configuration
- **Parquet**: Time series data (efficient for large datasets)
- **CSV**: Portfolio uploads and exports

#### Data Types
- **Prices**: Security price data
- **Returns**: Calculated returns
- **Factors**: Factor exposure data
- **Benchmarks**: Benchmark data
- **Portfolios**: Portfolio weights

### 4. Optimization Framework

#### Optimization Objectives
- **Pure Factor**: Long-short factor portfolios
- **Tracking Error**: Benchmark-relative optimization
- **Risk Parity**: Risk-balanced portfolios
- **Transaction Cost**: Cost-aware optimization

#### Constraints
- **Long-only**: Restrict to long positions
- **Weight bounds**: Position size limits
- **Factor constraints**: Factor exposure limits
- **Turnover constraints**: Trading cost controls

### 5. Cloud Integration

#### AWS S3 Integration
- Data synchronization
- Backup and recovery
- Multi-environment support

#### Configuration
- Environment variables for credentials
- Configurable bucket and prefix
- Incremental and full sync modes

## Development Guidelines

### Code Standards
- **Pydantic Models**: Use BaseModel for data validation
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Robust exception handling
- **Logging**: Structured logging throughout

### Testing Strategy
- **Unit Tests**: Core functionality testing
- **Integration Tests**: Module interaction testing
- **Performance Tests**: Optimization algorithm testing
- **UI Tests**: Streamlit interface testing

### Performance Considerations
- **Data Caching**: LRU cache for expensive operations
- **Parallel Processing**: Multi-threading for optimization
- **Memory Management**: Efficient data structures
- **Lazy Loading**: Load data on demand

## Key Features

### Advanced Analytics
- Factor exposure analysis
- Risk decomposition
- Performance attribution
- Turnover analysis

### User Experience
- Interactive visualizations (Plotly)
- Real-time progress updates
- Comprehensive error handling
- Download capabilities

### Scalability
- Modular architecture
- Configurable data sources
- Cloud-ready design
- Extensible framework

## Technology Stack

### Backend
- **Python 3.10+**: Core language
- **Pydantic**: Data validation
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scipy**: Scientific computing

### Frontend
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data display

### Data & Storage
- **SQLite**: Local database
- **Parquet**: Time series storage
- **AWS S3**: Cloud storage
- **Yahoo Finance**: Market data

### Optimization
- **CVXPY**: Convex optimization
- **SCIPY**: Optimization algorithms
- **Custom**: Factor-specific optimizers

This structure provides a comprehensive foundation for understanding and developing the Equity Factor Analysis Platform. Each component is designed to work together while maintaining modularity and extensibility.
