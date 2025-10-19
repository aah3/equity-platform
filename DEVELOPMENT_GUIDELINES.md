# Development Guidelines for Equity Factor Analysis Platform

## Code Standards and Best Practices

### 1. Python Code Style

#### General Guidelines
- **Python 3.10+**: Use modern Python features
- **PEP 8**: Follow Python style guide
- **Type Hints**: Always use type annotations
- **Docstrings**: Comprehensive documentation for all functions and classes
- **Error Handling**: Robust exception handling with meaningful messages

#### Pydantic Models
```python
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union
from decimal import Decimal

class PortfolioConfig(BaseModel):
    """Portfolio configuration model."""
    
    aum: Decimal = Field(..., description="Assets under management")
    max_weight: float = Field(default=0.05, ge=0, le=1, description="Maximum position weight")
    factors: List[str] = Field(default_factory=list, description="Risk factors to include")
    
    @field_validator('aum')
    @classmethod
    def validate_aum(cls, v):
        if v <= 0:
            raise ValueError('AUM must be positive')
        return v
```

#### Type Annotations
```python
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import date

def analyze_factor_returns(
    factor_data: pd.DataFrame,
    returns_data: pd.DataFrame,
    n_buckets: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Analyze factor returns using quantile-based portfolio construction.
    
    Args:
        factor_data: DataFrame with factor exposures
        returns_data: DataFrame with security returns
        n_buckets: Number of quantile buckets
        
    Returns:
        Dictionary containing analysis results
    """
    pass
```

### 2. Streamlit Development Standards

#### Session State Management
```python
# Initialize session state variables
if 'model_input' not in st.session_state:
    st.session_state.model_input = None
if 'factor_data' not in st.session_state:
    st.session_state.factor_data = None

# Use consistent naming patterns
st.session_state.data_updated = True
st.session_state.config_changed = False
```

#### UI Component Organization
```python
# Use columns for layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.write("Main content")
with col2:
    st.metric("Key metric", "Value")
with col3:
    st.button("Action", key="unique_key")
```

#### Error Handling in Streamlit
```python
try:
    results = run_analysis()
    st.success("Analysis completed successfully!")
except Exception as e:
    st.error(f"Analysis failed: {str(e)}")
    with st.expander("Debug Information"):
        st.code(traceback.format_exc())
```

### 3. Data Management Standards

#### File Organization
- **Parquet**: For time series data (efficient compression)
- **CSV**: For portfolio uploads and exports
- **SQLite**: For metadata and configuration
- **JSON**: For configuration files

#### Data Validation
```python
def validate_portfolio_data(df: pd.DataFrame) -> bool:
    """Validate portfolio data format."""
    required_columns = ['date', 'sid', 'weight']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")
    
    if df['weight'].sum() == 0:
        raise ValueError("Portfolio weights sum to zero")
    
    return True
```

#### Data Loading Patterns
```python
def load_factor_data(identifier: str) -> Optional[pd.DataFrame]:
    """Load factor data with error handling."""
    try:
        file_path = f"data/time_series/factors/{identifier}.parquet"
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            st.warning(f"Factor data not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading factor data: {str(e)}")
        return None
```

### 4. Optimization Framework Standards

#### Constraint Definitions
```python
class OptimizationConstraints(BaseModel):
    """Base class for optimization constraints."""
    
    long_only: bool = Field(default=True, description="Long-only constraint")
    full_investment: bool = Field(default=True, description="Full investment constraint")
    weight_bounds: Tuple[float, float] = Field(default=(0.0, 1.0), description="Weight bounds")
    
    @field_validator('weight_bounds')
    @classmethod
    def validate_weight_bounds(cls, v):
        if v[0] >= v[1]:
            raise ValueError('Lower bound must be less than upper bound')
        return v
```

#### Optimization Results
```python
class OptimizationResults(BaseModel):
    """Standardized optimization results."""
    
    optimization_type: str = Field(..., description="Type of optimization")
    weights_data: pd.DataFrame = Field(..., description="Optimal weights")
    meta_data: pd.DataFrame = Field(..., description="Optimization metadata")
    status: str = Field(..., description="Optimization status")
    objective_value: float = Field(..., description="Objective function value")
```

### 5. Testing Standards

#### Unit Test Structure
```python
import pytest
import pandas as pd
from src.qFactor import EquityFactor

class TestEquityFactor:
    """Test cases for EquityFactor class."""
    
    def test_factor_analysis(self):
        """Test factor analysis functionality."""
        # Arrange
        factor_data = self.create_sample_factor_data()
        returns_data = self.create_sample_returns_data()
        
        # Act
        factor = EquityFactor(name="test_factor", data=factor_data)
        results = factor.analyze_factor_returns(returns_data)
        
        # Assert
        assert 'bucket_returns' in results
        assert len(results['bucket_returns']) > 0
    
    def create_sample_factor_data(self) -> pd.DataFrame:
        """Create sample factor data for testing."""
        pass
```

#### Integration Test Structure
```python
class TestPortfolioOptimization:
    """Integration tests for portfolio optimization."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        # Test data loading
        # Test optimization
        # Test results validation
        pass
```

### 6. Error Handling and Logging

#### Logging Standards
```python
import logging
from src.logger import ApplicationLogger

logger = ApplicationLogger()

def run_optimization():
    """Run portfolio optimization with logging."""
    try:
        logger.info("Starting portfolio optimization")
        results = optimizer.optimize()
        logger.info(f"Optimization completed: {results.status}")
        return results
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise
```

#### Error Handling Patterns
```python
def robust_data_operation():
    """Example of robust error handling."""
    try:
        # Primary operation
        result = perform_operation()
        return result
    except FileNotFoundError as e:
        st.error(f"Data file not found: {str(e)}")
        return None
    except ValueError as e:
        st.warning(f"Data validation error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error in data operation")
        return None
```

### 7. Performance Optimization

#### Caching Strategies
```python
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=128)
def load_factor_data_cached(identifier: str, date_range: Tuple[str, str]) -> pd.DataFrame:
    """Cached factor data loading."""
    return load_factor_data(identifier, date_range)
```

#### Memory Management
```python
def process_large_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Process large datasets efficiently."""
    # Process in chunks
    chunk_size = 10000
    results = []
    
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)
```

### 8. Documentation Standards

#### Function Documentation
```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a return series.
    
    Args:
        returns: Series of returns (daily frequency)
        risk_free_rate: Risk-free rate (annual)
        annualization_factor: Days per year for annualization
        
    Returns:
        Annualized Sharpe ratio
        
    Raises:
        ValueError: If returns is empty or has insufficient data
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe ratio: {sharpe:.2f}")
    """
    if len(returns) < 2:
        raise ValueError("Insufficient data for Sharpe ratio calculation")
    
    excess_returns = returns - risk_free_rate / annualization_factor
    return excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)
```

#### Class Documentation
```python
class PortfolioOptimizer(BaseModel):
    """
    Portfolio optimization engine supporting multiple objectives.
    
    This class provides a unified interface for various portfolio optimization
    strategies including pure factor, tracking error, and risk parity approaches.
    
    Attributes:
        objective: Optimization objective (e.g., 'pure_factor', 'tracking_error')
        constraints: Optimization constraints
        data_manager: Data management interface
        
    Example:
        >>> optimizer = PortfolioOptimizer(
        ...     objective='tracking_error',
        ...     constraints=TrackingErrorConstraints()
        ... )
        >>> results = optimizer.optimize(returns, exposures)
    """
    
    objective: str = Field(..., description="Optimization objective")
    constraints: BaseModel = Field(..., description="Optimization constraints")
```

### 9. Security and Configuration

#### Environment Variables
```python
import os
from typing import Optional

def get_aws_credentials() -> Optional[Dict[str, str]]:
    """Get AWS credentials from environment variables."""
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not all([access_key, secret_key]):
        return None
    
    return {
        'aws_access_key_id': access_key,
        'aws_secret_access_key': secret_key,
        'region_name': region
    }
```

#### Configuration Management
```python
class AppConfig(BaseModel):
    """Application configuration."""
    
    data_source: str = Field(default='yahoo', description="Data source")
    universe: str = Field(default='NDX Index', description="Investment universe")
    max_memory_usage: int = Field(default=1024, description="Max memory usage (MB)")
    enable_cloud_sync: bool = Field(default=False, description="Enable cloud sync")
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        return cls(
            data_source=os.getenv('DATA_SOURCE', 'yahoo'),
            universe=os.getenv('UNIVERSE', 'NDX Index'),
            max_memory_usage=int(os.getenv('MAX_MEMORY_MB', '1024')),
            enable_cloud_sync=os.getenv('ENABLE_CLOUD_SYNC', 'false').lower() == 'true'
        )
```

### 10. Code Review Checklist

#### Before Submitting Code
- [ ] Type hints added to all functions and methods
- [ ] Docstrings added for all public functions
- [ ] Error handling implemented
- [ ] Unit tests written and passing
- [ ] Code follows PEP 8 style guide
- [ ] Pydantic models used for data validation
- [ ] Logging added for important operations
- [ ] Performance considerations addressed
- [ ] Documentation updated if needed

#### Code Review Focus Areas
- **Functionality**: Does the code work as intended?
- **Performance**: Are there any performance bottlenecks?
- **Error Handling**: Are errors handled gracefully?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Security**: Are there any security vulnerabilities?
- **Maintainability**: Is the code easy to understand and modify?

This development guideline ensures consistent, high-quality code across the Equity Factor Analysis Platform while maintaining the flexibility needed for financial analytics development.
