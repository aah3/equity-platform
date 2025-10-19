# Testing Strategy for Equity Factor Analysis Platform

## Testing Overview

This document outlines a comprehensive testing strategy for the Equity Factor Analysis Platform, covering unit tests, integration tests, performance tests, and UI tests.

## Testing Architecture

### Test Structure
```
tests/
├── __init__.py
├── unit/                           # Unit tests
│   ├── test_qFactor.py
│   ├── test_qOptimization.py
│   ├── test_qBacktest.py
│   ├── test_portfolio_analysis.py
│   ├── test_file_data_manager.py
│   └── test_utils.py
├── integration/                    # Integration tests
│   ├── test_data_pipeline.py
│   ├── test_optimization_workflow.py
│   ├── test_backtest_integration.py
│   └── test_report_generation.py
├── performance/                    # Performance tests
│   ├── test_optimization_performance.py
│   ├── test_data_loading_performance.py
│   └── test_memory_usage.py
├── ui/                            # UI tests
│   ├── test_streamlit_app.py
│   ├── test_user_interactions.py
│   └── test_data_visualization.py
├── fixtures/                      # Test fixtures
│   ├── sample_data.py
│   ├── mock_data.py
│   └── test_configs.py
└── conftest.py                    # Pytest configuration
```

## Unit Testing

### 1. Factor Analytics Tests (`test_qFactor.py`)

#### Test Cases for `EquityFactor`
```python
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.qFactor import EquityFactor, RiskFactors

class TestEquityFactor:
    """Test cases for EquityFactor class."""
    
    @pytest.fixture
    def sample_factor_data(self):
        """Create sample factor data."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(100)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'value': np.random.normal(0, 1)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(100)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'return': np.random.normal(0.001, 0.02)
                })
        
        return pd.DataFrame(data)
    
    def test_factor_initialization(self, sample_factor_data):
        """Test factor initialization."""
        factor = EquityFactor(
            name="test_factor",
            data=sample_factor_data,
            description="Test factor"
        )
        
        assert factor.name == "test_factor"
        assert factor.description == "Test factor"
        assert len(factor.data) == len(sample_factor_data)
    
    def test_factor_analysis_returns(self, sample_factor_data, sample_returns_data):
        """Test factor analysis returns."""
        factor = EquityFactor(
            name="test_factor",
            data=sample_factor_data
        )
        
        results = factor.analyze_factor_returns(
            returns_data=sample_returns_data,
            n_buckets=5,
            method='quantile'
        )
        
        assert 'bucket_returns' in results
        assert 'portfolio_stats' in results
        assert 'turnover' in results
        assert len(results['bucket_returns']) > 0
    
    def test_factor_analysis_quantile_method(self, sample_factor_data, sample_returns_data):
        """Test quantile-based factor analysis."""
        factor = EquityFactor(
            name="test_factor",
            data=sample_factor_data
        )
        
        results = factor.analyze_factor_returns(
            returns_data=sample_returns_data,
            n_buckets=5,
            method='quantile'
        )
        
        # Check that we have 5 buckets
        bucket_returns = results['bucket_returns']
        assert len(bucket_returns.columns) == 5
        
        # Check that returns are reasonable
        assert bucket_returns.notna().all().all()
    
    def test_factor_analysis_long_short(self, sample_factor_data, sample_returns_data):
        """Test long-short factor analysis."""
        factor = EquityFactor(
            name="test_factor",
            data=sample_factor_data
        )
        
        results = factor.analyze_factor_returns(
            returns_data=sample_returns_data,
            n_buckets=5,
            method='quantile',
            long_short=True
        )
        
        # Check that we have long-short returns
        bucket_returns = results['bucket_returns']
        assert 'long_short' in bucket_returns.columns
    
    def test_factor_analysis_equal_weighting(self, sample_factor_data, sample_returns_data):
        """Test equal weighting in factor analysis."""
        factor = EquityFactor(
            name="test_factor",
            data=sample_factor_data
        )
        
        results = factor.analyze_factor_returns(
            returns_data=sample_returns_data,
            n_buckets=5,
            method='quantile',
            weighting='equal'
        )
        
        # Check that results are generated
        assert len(results['bucket_returns']) > 0
        assert len(results['portfolio_stats']) > 0
```

#### Test Cases for `EquityFactorModelInput`
```python
class TestEquityFactorModelInput:
    """Test cases for EquityFactorModelInput class."""
    
    def test_model_input_validation(self):
        """Test model input validation."""
        from src.qFactor import EquityFactorModelInput, ParamsConfig, BacktestConfig
        
        # Valid configuration
        params = ParamsConfig(
            aum=Decimal('1000000'),
            risk_factors=[RiskFactors.BETA, RiskFactors.SIZE],
            n_buckets=5
        )
        
        backtest = BacktestConfig(
            data_source=DataSource.YAHOO,
            universe=Universe.NDX,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )
        
        model_input = EquityFactorModelInput(
            params=params,
            backtest=backtest
        )
        
        assert model_input.params.aum == Decimal('1000000')
        assert len(model_input.params.risk_factors) == 2
    
    def test_config_id_generation(self):
        """Test configuration ID generation."""
        # Test that same config generates same ID
        config1 = self.create_test_config()
        config2 = self.create_test_config()
        
        id1 = config1.generate_config_id()
        id2 = config2.generate_config_id()
        
        assert id1 == id2
        assert len(id1) > 0
    
    def create_test_config(self):
        """Create test configuration."""
        # Implementation here
        pass
```

### 2. Optimization Tests (`test_qOptimization.py`)

#### Test Cases for `PureFactorOptimizer`
```python
class TestPureFactorOptimizer:
    """Test cases for PureFactorOptimizer class."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for optimization."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(50)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'return': np.random.normal(0.001, 0.02)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_exposures_data(self):
        """Create sample exposures data."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(50)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'variable': 'beta',
                    'exposure': np.random.normal(0, 1)
                })
        
        return pd.DataFrame(data)
    
    def test_pure_factor_optimizer_initialization(self):
        """Test pure factor optimizer initialization."""
        constraints = PurePortfolioConstraints(
            long_only=False,
            full_investment=True,
            factor_neutral=['size', 'value'],
            weight_bounds=(-0.05, 0.05)
        )
        
        optimizer = PureFactorOptimizer(
            target_factor='beta',
            constraints=constraints
        )
        
        assert optimizer.target_factor == 'beta'
        assert optimizer.constraints.long_only == False
    
    def test_pure_factor_optimization(self, sample_returns_data, sample_exposures_data):
        """Test pure factor optimization."""
        constraints = PurePortfolioConstraints(
            long_only=False,
            full_investment=True,
            weight_bounds=(-0.05, 0.05)
        )
        
        optimizer = PureFactorOptimizer(
            target_factor='beta',
            constraints=constraints
        )
        
        # Convert to wide format
        returns_wide = sample_returns_data.pivot(
            index='date', columns='sid', values='return'
        )
        
        exposures_wide = sample_exposures_data.pivot(
            index=['date', 'sid'], columns='variable', values='exposure'
        ).reset_index()
        
        dates = [date(2020, 1, 1), date(2020, 6, 1), date(2020, 12, 1)]
        
        results = optimizer.optimize(
            returns=returns_wide,
            exposures=exposures_wide,
            dates=dates
        )
        
        assert 'weights_data' in results
        assert 'meta_data' in results
        assert results['status'] == 'success'
    
    def test_optimization_constraints(self, sample_returns_data, sample_exposures_data):
        """Test optimization constraints."""
        constraints = PurePortfolioConstraints(
            long_only=True,
            full_investment=True,
            weight_bounds=(0.0, 0.1)
        )
        
        optimizer = PureFactorOptimizer(
            target_factor='beta',
            constraints=constraints
        )
        
        # Test that long-only constraint is enforced
        results = optimizer.optimize(
            returns=sample_returns_data.pivot(index='date', columns='sid', values='return'),
            exposures=sample_exposures_data.pivot(
                index=['date', 'sid'], columns='variable', values='exposure'
            ).reset_index(),
            dates=[date(2020, 1, 1)]
        )
        
        weights = results['weights_data']
        assert (weights['weight'] >= 0).all()  # Long-only constraint
        assert (weights['weight'] <= 0.1).all()  # Weight bounds constraint
```

#### Test Cases for `TrackingErrorOptimizer`
```python
class TestTrackingErrorOptimizer:
    """Test cases for TrackingErrorOptimizer class."""
    
    def test_tracking_error_optimizer_initialization(self):
        """Test tracking error optimizer initialization."""
        constraints = TrackingErrorConstraints(
            long_only=True,
            full_investment=True,
            weight_bounds=(0.0, 0.1),
            max_names=20,
            tracking_error_max=0.05
        )
        
        optimizer = TrackingErrorOptimizer(
            constraints=constraints
        )
        
        assert optimizer.constraints.long_only == True
        assert optimizer.constraints.max_names == 20
    
    def test_tracking_error_optimization(self):
        """Test tracking error optimization."""
        # Create sample data
        returns_data = self.create_sample_returns_data()
        benchmark_returns = self.create_sample_benchmark_returns()
        exposures_data = self.create_sample_exposures_data()
        benchmark_exposures = self.create_sample_benchmark_exposures()
        
        constraints = TrackingErrorConstraints(
            long_only=True,
            full_investment=True,
            weight_bounds=(0.0, 0.1),
            max_names=20,
            tracking_error_max=0.05
        )
        
        optimizer = TrackingErrorOptimizer(
            constraints=constraints
        )
        
        dates = [date(2020, 1, 1), date(2020, 6, 1)]
        
        results = optimizer.optimize(
            returns=returns_data,
            benchmark_returns=benchmark_returns,
            exposures=exposures_data,
            benchmark_exposures=benchmark_exposures,
            dates=dates
        )
        
        assert 'weights_data' in results
        assert 'meta_data' in results
        assert results['status'] == 'success'
    
    def create_sample_returns_data(self):
        """Create sample returns data."""
        # Implementation here
        pass
    
    def create_sample_benchmark_returns(self):
        """Create sample benchmark returns."""
        # Implementation here
        pass
    
    def create_sample_exposures_data(self):
        """Create sample exposures data."""
        # Implementation here
        pass
    
    def create_sample_benchmark_exposures(self):
        """Create sample benchmark exposures."""
        # Implementation here
        pass
```

### 3. Backtesting Tests (`test_qBacktest.py`)

#### Test Cases for `Backtest`
```python
class TestBacktest:
    """Test cases for Backtest class."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(20)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'ticker': security,
                    'return': np.random.normal(0.001, 0.02)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
        securities = [f'STOCK_{i:03d}' for i in range(20)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'ticker': security,
                    'weight': np.random.uniform(0, 0.1)
                })
        
        # Normalize weights
        df = pd.DataFrame(data)
        for date in df['date'].unique():
            mask = df['date'] == date
            df.loc[mask, 'weight'] = df.loc[mask, 'weight'] / df.loc[mask, 'weight'].sum()
        
        return df
    
    def test_backtest_initialization(self):
        """Test backtest initialization."""
        config = BacktestConfig(
            asset_class=AssetClass.EQUITY,
            portfolio_type=PortfolioType.LONG_ONLY,
            model_type='test_model',
            annualization_factor=252
        )
        
        backtest = Backtest(config=config)
        
        assert backtest.config.asset_class == AssetClass.EQUITY
        assert backtest.config.portfolio_type == PortfolioType.LONG_ONLY
    
    def test_backtest_execution(self, sample_returns_data, sample_portfolio_data):
        """Test backtest execution."""
        config = BacktestConfig(
            asset_class=AssetClass.EQUITY,
            portfolio_type=PortfolioType.LONG_ONLY,
            model_type='test_model',
            annualization_factor=252
        )
        
        backtest = Backtest(config=config)
        
        results = backtest.run_backtest(
            returns_data=sample_returns_data,
            portfolio_data=sample_portfolio_data,
            plot=False
        )
        
        assert results is not None
        assert hasattr(results, 'cumulative_return_benchmark')
        assert hasattr(results, 'sharpe_ratio_benchmark')
    
    def test_backtest_performance_metrics(self, sample_returns_data, sample_portfolio_data):
        """Test backtest performance metrics calculation."""
        config = BacktestConfig(
            asset_class=AssetClass.EQUITY,
            portfolio_type=PortfolioType.LONG_ONLY,
            model_type='test_model',
            annualization_factor=252
        )
        
        backtest = Backtest(config=config)
        
        results = backtest.run_backtest(
            returns_data=sample_returns_data,
            portfolio_data=sample_portfolio_data,
            plot=False
        )
        
        # Check that performance metrics are calculated
        assert results.cumulative_return_benchmark is not None
        assert results.sharpe_ratio_benchmark is not None
        assert isinstance(results.cumulative_return_benchmark, float)
        assert isinstance(results.sharpe_ratio_benchmark, float)
```

## Integration Testing

### 1. Data Pipeline Tests (`test_data_pipeline.py`)

```python
class TestDataPipeline:
    """Test cases for data pipeline integration."""
    
    def test_etl_pipeline_integration(self):
        """Test complete ETL pipeline."""
        from src.etl_universe_data import etl_universe_data
        from src.qFactor import EquityFactorModelInput, ParamsConfig, BacktestConfig
        
        # Create test configuration
        model_input = self.create_test_model_input()
        
        # Run ETL pipeline
        success = etl_universe_data(model_input)
        
        assert success == True
        
        # Verify data was created
        from src.file_data_manager import FileDataManager, FileConfig
        
        config = FileConfig()
        manager = FileDataManager(config)
        
        # Check that data files exist
        prices = manager.load_prices('NDX_Index_members')
        returns = manager.load_returns('NDX_Index_members')
        exposures = manager.load_exposures('NDX_Index_members')
        
        assert prices is not None and not prices.empty
        assert returns is not None and not returns.empty
        assert exposures is not None and not exposures.empty
    
    def test_data_validation_pipeline(self):
        """Test data validation pipeline."""
        # Test data validation at each step
        pass
    
    def create_test_model_input(self):
        """Create test model input."""
        # Implementation here
        pass
```

### 2. Optimization Workflow Tests (`test_optimization_workflow.py`)

```python
class TestOptimizationWorkflow:
    """Test cases for optimization workflow integration."""
    
    def test_pure_factor_workflow(self):
        """Test complete pure factor optimization workflow."""
        # Load test data
        data_manager = self.create_test_data_manager()
        
        # Create optimizer
        optimizer = self.create_pure_factor_optimizer()
        
        # Run optimization
        results = optimizer.optimize(
            returns=data_manager.load_returns('test_data'),
            exposures=data_manager.load_exposures('test_data'),
            dates=[date(2020, 1, 1), date(2020, 6, 1)]
        )
        
        # Validate results
        assert results['status'] == 'success'
        assert 'weights_data' in results
        
        # Run backtest
        backtest = self.create_test_backtest()
        backtest_results = backtest.run_backtest(
            returns_data=data_manager.load_returns('test_data'),
            portfolio_data=results['weights_data'],
            plot=False
        )
        
        assert backtest_results is not None
    
    def test_tracking_error_workflow(self):
        """Test complete tracking error optimization workflow."""
        # Similar structure to pure factor workflow
        pass
    
    def create_test_data_manager(self):
        """Create test data manager."""
        # Implementation here
        pass
    
    def create_pure_factor_optimizer(self):
        """Create pure factor optimizer."""
        # Implementation here
        pass
    
    def create_test_backtest(self):
        """Create test backtest."""
        # Implementation here
        pass
```

## Performance Testing

### 1. Optimization Performance Tests (`test_optimization_performance.py`)

```python
import time
import pytest
from src.qOptimization import PureFactorOptimizer, TrackingErrorOptimizer

class TestOptimizationPerformance:
    """Test cases for optimization performance."""
    
    def test_pure_factor_optimization_performance(self):
        """Test pure factor optimization performance."""
        # Create large dataset
        large_returns_data = self.create_large_returns_dataset(1000, 252)
        large_exposures_data = self.create_large_exposures_dataset(1000, 252)
        
        optimizer = PureFactorOptimizer(
            target_factor='beta',
            constraints=self.create_test_constraints()
        )
        
        start_time = time.time()
        
        results = optimizer.optimize(
            returns=large_returns_data,
            exposures=large_exposures_data,
            dates=[date(2020, 1, 1), date(2020, 6, 1)]
        )
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Performance assertions
        assert optimization_time < 60  # Should complete within 60 seconds
        assert results['status'] == 'success'
    
    def test_tracking_error_optimization_performance(self):
        """Test tracking error optimization performance."""
        # Similar structure to pure factor performance test
        pass
    
    def create_large_returns_dataset(self, n_securities, n_dates):
        """Create large returns dataset for performance testing."""
        # Implementation here
        pass
    
    def create_large_exposures_dataset(self, n_securities, n_dates):
        """Create large exposures dataset for performance testing."""
        # Implementation here
        pass
    
    def create_test_constraints(self):
        """Create test constraints."""
        # Implementation here
        pass
```

### 2. Data Loading Performance Tests (`test_data_loading_performance.py`)

```python
class TestDataLoadingPerformance:
    """Test cases for data loading performance."""
    
    def test_parquet_loading_performance(self):
        """Test Parquet file loading performance."""
        from src.file_data_manager import FileDataManager, FileConfig
        
        config = FileConfig()
        manager = FileDataManager(config)
        
        # Test loading large Parquet file
        start_time = time.time()
        data = manager.load_prices('large_dataset')
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Performance assertions
        assert loading_time < 10  # Should load within 10 seconds
        assert data is not None and not data.empty
    
    def test_csv_loading_performance(self):
        """Test CSV file loading performance."""
        # Similar structure to Parquet loading test
        pass
    
    def test_memory_usage_during_loading(self):
        """Test memory usage during data loading."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load large dataset
        large_data = self.create_large_dataset()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory usage assertions
        assert memory_increase < 1000  # Should not increase by more than 1GB
```

## UI Testing

### 1. Streamlit App Tests (`test_streamlit_app.py`)

```python
import streamlit as st
from streamlit.testing.v1 import AppTest

class TestStreamlitApp:
    """Test cases for Streamlit application."""
    
    def test_app_initialization(self):
        """Test app initialization."""
        app = AppTest.from_file("app_factors.py")
        app.run()
        
        # Check that app loads without errors
        assert app.get("main") is not None
    
    def test_sidebar_configuration(self):
        """Test sidebar configuration."""
        app = AppTest.from_file("app_factors.py")
        app.run()
        
        # Check that sidebar elements are present
        assert app.get("sidebar") is not None
        assert app.get("universe") is not None
        assert app.get("factors") is not None
    
    def test_factor_analysis_tab(self):
        """Test factor analysis tab."""
        app = AppTest.from_file("app_factors.py")
        app.run()
        
        # Navigate to factor analysis tab
        app.tab("Factor Analysis").run()
        
        # Check that factor analysis elements are present
        assert app.get("Run Factor Analysis") is not None
    
    def test_portfolio_optimization_tab(self):
        """Test portfolio optimization tab."""
        app = AppTest.from_file("app_factors.py")
        app.run()
        
        # Navigate to portfolio optimization tab
        app.tab("Portfolio Optimization").run()
        
        # Check that optimization elements are present
        assert app.get("Run Portfolio Optimization") is not None
```

### 2. User Interaction Tests (`test_user_interactions.py`)

```python
class TestUserInteractions:
    """Test cases for user interactions."""
    
    def test_data_upload_interaction(self):
        """Test data upload interaction."""
        app = AppTest.from_file("app_factors.py")
        app.run()
        
        # Navigate to portfolio upload tab
        app.tab("Portfolio Upload & Analysis").run()
        
        # Simulate file upload
        test_data = self.create_test_portfolio_data()
        app.file_uploader("Upload Portfolio").upload(test_data)
        
        # Check that data is processed
        assert app.get("portfolio_data") is not None
    
    def test_optimization_parameter_interaction(self):
        """Test optimization parameter interaction."""
        app = AppTest.from_file("app_factors.py")
        app.run()
        
        # Set optimization parameters
        app.selectbox("Objective").select("Tracking Error")
        app.slider("Max Tracking Error (%)").set_value(5)
        app.slider("Max Position Weight (%)").set_value(10)
        
        # Check that parameters are set
        assert app.get("optimization_objective") == "tracking_error"
        assert app.get("te_tracking_error_max") == 0.05
        assert app.get("te_max_weight") == 0.10
    
    def create_test_portfolio_data(self):
        """Create test portfolio data."""
        # Implementation here
        pass
```

## Test Fixtures and Utilities

### 1. Sample Data Generation (`fixtures/sample_data.py`)

```python
import pandas as pd
import numpy as np
from datetime import date, timedelta

class SampleDataGenerator:
    """Generate sample data for testing."""
    
    @staticmethod
    def create_factor_data(
        n_securities: int = 100,
        n_dates: int = 252,
        factor_name: str = 'beta'
    ) -> pd.DataFrame:
        """Create sample factor data."""
        dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(n_securities)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'value': np.random.normal(0, 1)
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_returns_data(
        n_securities: int = 100,
        n_dates: int = 252
    ) -> pd.DataFrame:
        """Create sample returns data."""
        dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='D')
        securities = [f'STOCK_{i:03d}' for i in range(n_securities)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'return': np.random.normal(0.001, 0.02)
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_portfolio_data(
        n_securities: int = 20,
        n_dates: int = 12
    ) -> pd.DataFrame:
        """Create sample portfolio data."""
        dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='M')
        securities = [f'STOCK_{i:03d}' for i in range(n_securities)]
        
        data = []
        for date in dates:
            for security in securities:
                data.append({
                    'date': date,
                    'sid': security,
                    'weight': np.random.uniform(0, 0.1)
                })
        
        # Normalize weights
        df = pd.DataFrame(data)
        for date in df['date'].unique():
            mask = df['date'] == date
            df.loc[mask, 'weight'] = df.loc[mask, 'weight'] / df.loc[mask, 'weight'].sum()
        
        return df
```

### 2. Mock Data Classes (`fixtures/mock_data.py`)

```python
class MockDataManager:
    """Mock data manager for testing."""
    
    def __init__(self):
        self.data = {}
        self.sample_data_generator = SampleDataGenerator()
    
    def load_prices(self, identifier: str) -> pd.DataFrame:
        """Mock price data loading."""
        if identifier not in self.data:
            self.data[identifier] = self.sample_data_generator.create_price_data()
        return self.data[identifier]
    
    def load_returns(self, identifier: str) -> pd.DataFrame:
        """Mock returns data loading."""
        if identifier not in self.data:
            self.data[identifier] = self.sample_data_generator.create_returns_data()
        return self.data[identifier]
    
    def load_factors(self, identifier: str) -> pd.DataFrame:
        """Mock factor data loading."""
        if identifier not in self.data:
            self.data[identifier] = self.sample_data_generator.create_factor_data()
        return self.data[identifier]
    
    def load_exposures(self, identifier: str) -> pd.DataFrame:
        """Mock exposures data loading."""
        if identifier not in self.data:
            self.data[identifier] = self.sample_data_generator.create_exposures_data()
        return self.data[identifier]

class MockOptimizer:
    """Mock optimizer for testing."""
    
    def __init__(self, target_factor: str = 'beta'):
        self.target_factor = target_factor
    
    def optimize(self, returns: pd.DataFrame, exposures: pd.DataFrame, dates: list) -> dict:
        """Mock optimization."""
        return {
            'weights_data': self.create_mock_weights_data(dates),
            'meta_data': self.create_mock_meta_data(dates),
            'status': 'success'
        }
    
    def create_mock_weights_data(self, dates: list) -> pd.DataFrame:
        """Create mock weights data."""
        # Implementation here
        pass
    
    def create_mock_meta_data(self, dates: list) -> pd.DataFrame:
        """Create mock meta data."""
        # Implementation here
        pass
```

## Test Configuration

### 1. Pytest Configuration (`conftest.py`)

```python
import pytest
import pandas as pd
import numpy as np
from datetime import date
from fixtures.sample_data import SampleDataGenerator
from fixtures.mock_data import MockDataManager, MockOptimizer

@pytest.fixture
def sample_data_generator():
    """Provide sample data generator."""
    return SampleDataGenerator()

@pytest.fixture
def mock_data_manager():
    """Provide mock data manager."""
    return MockDataManager()

@pytest.fixture
def mock_optimizer():
    """Provide mock optimizer."""
    return MockOptimizer()

@pytest.fixture
def sample_factor_data(sample_data_generator):
    """Provide sample factor data."""
    return sample_data_generator.create_factor_data()

@pytest.fixture
def sample_returns_data(sample_data_generator):
    """Provide sample returns data."""
    return sample_data_generator.create_returns_data()

@pytest.fixture
def sample_portfolio_data(sample_data_generator):
    """Provide sample portfolio data."""
    return sample_data_generator.create_portfolio_data()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        'test_mode': True,
        'data_path': 'test_data/',
        'cache_enabled': False,
        'parallel_processing': False
    }

# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "ui: mark test as UI test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

# Test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        if "test_" in item.name:
            if "unit" in item.name:
                item.add_marker(pytest.mark.unit)
            elif "integration" in item.name:
                item.add_marker(pytest.mark.integration)
            elif "performance" in item.name:
                item.add_marker(pytest.mark.performance)
            elif "ui" in item.name:
                item.add_marker(pytest.mark.ui)
```

## Running Tests

### 1. Test Execution Commands

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run performance tests only
pytest -m performance

# Run UI tests only
pytest -m ui

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_qFactor.py

# Run specific test function
pytest tests/unit/test_qFactor.py::TestEquityFactor::test_factor_analysis_returns

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

### 2. Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run unit tests
      run: pytest -m unit --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: pytest -m integration
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## Test Quality Metrics

### 1. Coverage Targets
- **Unit Tests**: 90% code coverage
- **Integration Tests**: 80% integration coverage
- **Critical Paths**: 100% coverage for optimization and backtesting

### 2. Performance Benchmarks
- **Optimization**: Complete within 60 seconds for 1000 securities
- **Data Loading**: Load 1GB dataset within 10 seconds
- **Memory Usage**: Peak memory usage under 2GB for large datasets

### 3. Test Reliability
- **Flaky Tests**: Less than 1% failure rate
- **Test Execution**: Complete test suite within 10 minutes
- **Test Maintenance**: Update tests when requirements change

This comprehensive testing strategy ensures the Equity Factor Analysis Platform is robust, reliable, and performs well under various conditions while maintaining high code quality and user experience standards.
