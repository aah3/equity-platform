# API Reference - Equity Factor Analysis Platform

## Core Modules Overview

### 1. Factor Analytics (`src/qFactor.py`)

#### Core Classes

##### `EquityFactor`
Main class for factor analysis and portfolio construction.

```python
class EquityFactor(BaseModel):
    name: str = Field(..., description="Factor name")
    data: pd.DataFrame = Field(..., description="Factor exposure data")
    description: str = Field(default="", description="Factor description")
    category: str = Field(default="", description="Factor category")
    
    def analyze_factor_returns(
        self,
        returns_data: pd.DataFrame,
        n_buckets: int = 5,
        method: str = 'quantile',
        weighting: str = 'equal',
        long_short: bool = True,
        neutralize_size: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze factor returns using quantile-based portfolio construction.
        
        Args:
            returns_data: DataFrame with columns ['date', 'sid', 'return']
            n_buckets: Number of quantile buckets
            method: Portfolio construction method ('quantile', 'equal_weight')
            weighting: Weighting scheme ('equal', 'market_cap')
            long_short: Whether to construct long-short portfolios
            neutralize_size: Whether to neutralize size exposure
            
        Returns:
            Dictionary containing:
            - 'bucket_returns': DataFrame with factor returns by bucket
            - 'portfolio_stats': DataFrame with portfolio statistics
            - 'turnover': DataFrame with turnover analysis
        """
```

##### `EquityFactorModelInput`
Configuration class for factor model inputs.

```python
class EquityFactorModelInput(BaseModel):
    params: ParamsConfig
    backtest: BacktestConfig
    regime: RegimeConfig
    export: ExportConfig
    
    def validate_configuration(self) -> bool:
        """Validate model input configuration."""
        
    def generate_config_id(self) -> str:
        """Generate unique configuration ID."""
```

##### `SecurityMasterFactory`
Factory class for creating security master instances.

```python
class SecurityMasterFactory:
    @staticmethod
    def create_security_master(model_input: EquityFactorModelInput) -> SecurityMaster:
        """Create security master instance."""
        
    def load_benchmark_data(self, universe: str) -> pd.DataFrame:
        """Load benchmark data for universe."""
        
    def get_universe_constituents(self, universe: str) -> List[str]:
        """Get universe constituent securities."""
```

#### Enums and Constants

##### `RiskFactors`
Available risk factors for analysis.

```python
class RiskFactors(str, Enum):
    BETA = "beta"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    PROFIT = "profit"
    QUALITY = "quality"
    LOW_VOL = "low_vol"
    GROWTH = "growth"
```

##### `Universe`
Available investment universes.

```python
class Universe(str, Enum):
    # North America
    NDX = "NDX Index"
    SPX = "SPX Index"
    RTY = "RTY Index"
    INDU = "INDU Index"
    
    # Europe
    SXXP = "SXXP Index"
    UKX = "UKX Index"
    CAC = "CAC Index"
    DAX = "DAX Index"
    
    # Asia Pacific
    NKY = "NKY Index"
    HSI = "HSI Index"
    AS51 = "AS51 Index"
```

##### `DataSource`
Available data sources.

```python
class DataSource(str, Enum):
    YAHOO = "yahoo"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    CUSTOM = "custom"
```

### 2. Portfolio Optimization (`src/qOptimization.py`)

#### Core Classes

##### `PureFactorOptimizer`
Optimizer for pure factor portfolios.

```python
class PureFactorOptimizer(BaseModel):
    target_factor: str = Field(..., description="Target factor for optimization")
    constraints: PurePortfolioConstraints = Field(..., description="Optimization constraints")
    normalize_weights: bool = Field(default=True, description="Normalize portfolio weights")
    parallel_processing: bool = Field(default=False, description="Enable parallel processing")
    
    def optimize(
        self,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        dates: List[date]
    ) -> Dict[str, Any]:
        """
        Optimize pure factor portfolio.
        
        Args:
            returns: DataFrame with returns data
            exposures: DataFrame with factor exposures
            dates: List of rebalancing dates
            
        Returns:
            Dictionary containing:
            - 'weights_data': DataFrame with optimal weights
            - 'meta_data': DataFrame with optimization metadata
            - 'status': Optimization status
        """
```

##### `TrackingErrorOptimizer`
Optimizer for tracking error portfolios.

```python
class TrackingErrorOptimizer(BaseModel):
    constraints: TrackingErrorConstraints = Field(..., description="TE constraints")
    normalize_weights: bool = Field(default=True, description="Normalize weights")
    parallel_processing: bool = Field(default=False, description="Parallel processing")
    use_integer_constraints: bool = Field(default=False, description="Integer constraints")
    
    def optimize(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        exposures: pd.DataFrame,
        benchmark_exposures: pd.DataFrame,
        dates: List[date]
    ) -> Dict[str, Any]:
        """
        Optimize tracking error portfolio.
        
        Args:
            returns: Portfolio returns data
            benchmark_returns: Benchmark returns data
            exposures: Portfolio factor exposures
            benchmark_exposures: Benchmark factor exposures
            dates: Rebalancing dates
            
        Returns:
            Dictionary containing optimization results
        """
```

#### Constraint Classes

##### `PurePortfolioConstraints`
Constraints for pure factor optimization.

```python
class PurePortfolioConstraints(BaseModel):
    long_only: bool = Field(default=False, description="Long-only constraint")
    full_investment: bool = Field(default=True, description="Full investment constraint")
    factor_neutral: List[str] = Field(default_factory=list, description="Factors to neutralize")
    weight_bounds: Tuple[float, float] = Field(default=(0.0, 1.0), description="Weight bounds")
    min_holding: float = Field(default=0.0, description="Minimum holding size")
```

##### `TrackingErrorConstraints`
Constraints for tracking error optimization.

```python
class TrackingErrorConstraints(BaseModel):
    long_only: bool = Field(default=True, description="Long-only constraint")
    full_investment: bool = Field(default=True, description="Full investment constraint")
    factor_constraints: Dict[str, Union[Tuple[float, float], float]] = Field(
        default_factory=dict, description="Factor exposure constraints"
    )
    weight_bounds: Tuple[float, float] = Field(default=(0.0, 1.0), description="Weight bounds")
    min_holding: float = Field(default=0.0, description="Minimum holding size")
    max_names: int = Field(default=100, description="Maximum number of positions")
    tracking_error_max: float = Field(default=0.05, description="Maximum tracking error")
```

#### Enums

##### `OptimizationObjective`
Available optimization objectives.

```python
class OptimizationObjective(str, Enum):
    PURE_FACTOR = "pure_factor"
    TRACKING_ERROR = "tracking_error"
    NUM_TRADES = "num_trades"
    TRANSACTION_COST = "transaction_cost"
    RISK_PARITY = "risk_parity"
```

##### `OptimizationStatus`
Optimization status indicators.

```python
class OptimizationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
```

### 3. Backtesting Framework (`src/qBacktest.py`)

#### Core Classes

##### `Backtest`
Main backtesting engine.

```python
class Backtest:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.df_pnl = pd.DataFrame()
        self.performance_metrics = {}
    
    def run_backtest(
        self,
        returns_data: pd.DataFrame,
        portfolio_data: pd.DataFrame,
        plot: bool = True
    ) -> BacktestResults:
        """
        Run portfolio backtest.
        
        Args:
            returns_data: DataFrame with columns ['date', 'ticker', 'return']
            portfolio_data: DataFrame with columns ['date', 'ticker', 'weight']
            plot: Whether to generate performance plots
            
        Returns:
            BacktestResults object with performance metrics
        """
```

##### `BacktestConfig`
Configuration for backtesting.

```python
class BacktestConfig(BaseModel):
    asset_class: AssetClass = Field(..., description="Asset class")
    portfolio_type: PortfolioType = Field(..., description="Portfolio type")
    model_type: str = Field(..., description="Model type")
    annualization_factor: int = Field(default=252, description="Annualization factor")
```

#### Enums

##### `AssetClass`
Asset class definitions.

```python
class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
```

##### `PortfolioType`
Portfolio type definitions.

```python
class PortfolioType(str, Enum):
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    MARKET_NEUTRAL = "market_neutral"
```

### 4. Portfolio Analysis (`src/portfolio_analysis.py`)

#### Core Classes

##### `PortfolioAnalyzer`
Main portfolio analysis engine.

```python
class PortfolioAnalyzer(BaseModel):
    def analyze_portfolio(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        factor_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> PortfolioAnalysisResults:
        """
        Analyze portfolio performance and risk.
        
        Args:
            portfolio_data: Portfolio weights data
            benchmark_data: Benchmark data
            factor_data: Optional factor exposure data
            
        Returns:
            PortfolioAnalysisResults object
        """
```

##### `PortfolioValidator`
Portfolio data validation.

```python
class PortfolioValidator:
    def validate_portfolio_format(self, df: pd.DataFrame) -> ValidationResult:
        """Validate portfolio data format."""
        
    def validate_portfolio_weights(self, df: pd.DataFrame) -> ValidationResult:
        """Validate portfolio weights."""
        
    def validate_date_range(self, df: pd.DataFrame) -> ValidationResult:
        """Validate date range."""
```

##### `PortfolioComparator`
Portfolio comparison utilities.

```python
class PortfolioComparator:
    def compare_portfolios(
        self,
        portfolio1: pd.DataFrame,
        portfolio2: pd.DataFrame
    ) -> PortfolioComparison:
        """Compare two portfolios."""
        
    def calculate_turnover(self, portfolio1: pd.DataFrame, portfolio2: pd.DataFrame) -> float:
        """Calculate portfolio turnover."""
```

### 5. Data Management (`src/file_data_manager.py`)

#### Core Classes

##### `FileDataManager`
File-based data management.

```python
class FileDataManager:
    def __init__(self, config: FileConfig):
        self.config = config
        self.cache = {}
    
    def load_prices(self, identifier: str) -> pd.DataFrame:
        """Load price data."""
        
    def load_returns(self, identifier: str) -> pd.DataFrame:
        """Load returns data."""
        
    def load_factors(self, identifier: str) -> pd.DataFrame:
        """Load factor data."""
        
    def load_exposures(self, identifier: str) -> pd.DataFrame:
        """Load exposure data."""
        
    def save_data(self, data: pd.DataFrame, identifier: str, data_type: str) -> bool:
        """Save data to file."""
```

##### `FileConfig`
Configuration for file operations.

```python
class FileConfig(BaseModel):
    base_path: str = Field(default="../data", description="Base data path")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_size: int = Field(default=100, description="Cache size limit")
    file_format: str = Field(default="parquet", description="Default file format")
```

### 6. ETL Pipeline (`src/etl_universe_data.py`)

#### Core Functions

##### `etl_universe_data`
Main ETL function for universe data.

```python
def etl_universe_data(
    model_input: EquityFactorModelInput,
    progress_callback: Optional[Callable] = None
) -> bool:
    """
    Extract, transform, and load universe data.
    
    Args:
        model_input: Model input configuration
        progress_callback: Optional progress callback function
        
    Returns:
        Boolean indicating success
    """
```

##### `extract_market_data`
Extract market data from sources.

```python
def extract_market_data(
    universe: str,
    start_date: date,
    end_date: date,
    data_source: str
) -> pd.DataFrame:
    """Extract market data from specified source."""
```

##### `transform_factor_data`
Transform raw data into factor exposures.

```python
def transform_factor_data(
    price_data: pd.DataFrame,
    factors: List[str]
) -> Dict[str, pd.DataFrame]:
    """Transform price data into factor exposures."""
```

### 7. Report Generation (`src/report_generator.py`)

#### Core Classes

##### `AppReportBuilder`
Report generation for the application.

```python
class AppReportBuilder:
    def __init__(self, title: str = "Equity Factor Analysis Report"):
        self.title = title
        self.sections = []
    
    def build_pdf_bytes(
        self,
        pure_factor_returns: Optional[pd.DataFrame] = None,
        te_meta: Optional[pd.DataFrame] = None,
        te_weights: Optional[pd.DataFrame] = None,
        te_returns: Optional[pd.DataFrame] = None,
        uploaded_portfolio_results: Optional[Dict] = None
    ) -> bytes:
        """Build PDF report as bytes."""
```

## Utility Functions

### Data Processing Utilities

```python
def prepare_portfolio_download_data(
    portfolio_data: pd.DataFrame,
    data_type: str = "portfolio"
) -> pd.DataFrame:
    """Prepare portfolio data for download."""

def create_download_file(
    df: pd.DataFrame,
    file_format: str,
    filename_prefix: str
) -> bytes:
    """Create download file in specified format."""

def correlation_matrix_display(
    corr_matrix: pd.DataFrame,
    tab_name: str
) -> None:
    """Display correlation matrix visualization."""

def monthly_returns_heatmap(
    monthly_returns: pd.DataFrame,
    tab_name: str
) -> None:
    """Display monthly returns heatmap."""
```

### Configuration Utilities

```python
def create_model_input() -> EquityFactorModelInput:
    """Create model input from UI selections."""

def check_config_changes() -> bool:
    """Check if configuration has changed."""

def load_existing_data(model_input: EquityFactorModelInput) -> bool:
    """Load existing data from files."""
```

### AWS S3 Integration

```python
def init_s3_client() -> Optional[boto3.client]:
    """Initialize S3 client."""

def upload_to_s3(file_path: str, s3_key: str) -> bool:
    """Upload file to S3."""

def download_from_s3(s3_key: str, local_path: str) -> bool:
    """Download file from S3."""

def sync_data_with_s3(model_input: EquityFactorModelInput) -> bool:
    """Sync data with S3."""
```

## Error Handling

### Custom Exceptions

```python
class FactorAnalysisError(Exception):
    """Raised when factor analysis fails."""
    pass

class OptimizationError(Exception):
    """Raised when optimization fails."""
    pass

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

class BacktestError(Exception):
    """Raised when backtesting fails."""
    pass
```

### Error Handling Patterns

```python
def robust_operation(operation_func, *args, **kwargs):
    """Wrapper for robust error handling."""
    try:
        return operation_func(*args, **kwargs)
    except FactorAnalysisError as e:
        st.error(f"Factor analysis error: {str(e)}")
        return None
    except OptimizationError as e:
        st.error(f"Optimization error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None
```

## Performance Optimization

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_factor_data_cached(identifier: str, date_range: tuple) -> pd.DataFrame:
    """Cached factor data loading."""

@lru_cache(maxsize=64)
def calculate_correlation_matrix(returns_data: pd.DataFrame) -> pd.DataFrame:
    """Cached correlation matrix calculation."""
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_factor_analysis(factor_list: List[str], data: Dict) -> Dict:
    """Run factor analysis in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_factor, factor, data): factor
            for factor in factor_list
        }
        results = {}
        for future in futures:
            factor = futures[future]
            results[factor] = future.result()
    return results
```

## Testing Utilities

### Test Data Generation

```python
def create_sample_factor_data(
    n_securities: int = 100,
    n_dates: int = 252
) -> pd.DataFrame:
    """Create sample factor data for testing."""

def create_sample_returns_data(
    n_securities: int = 100,
    n_dates: int = 252
) -> pd.DataFrame:
    """Create sample returns data for testing."""

def create_sample_portfolio_data(
    n_securities: int = 20,
    n_dates: int = 12
) -> pd.DataFrame:
    """Create sample portfolio data for testing."""
```

### Mock Classes

```python
class MockDataManager:
    """Mock data manager for testing."""
    def __init__(self):
        self.data = {}
    
    def load_prices(self, identifier: str) -> pd.DataFrame:
        return self.data.get('prices', pd.DataFrame())
    
    def load_returns(self, identifier: str) -> pd.DataFrame:
        return self.data.get('returns', pd.DataFrame())
```

This API reference provides comprehensive documentation for all major components of the Equity Factor Analysis Platform, enabling developers to understand and extend the system effectively.
