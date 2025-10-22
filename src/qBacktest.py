# Placeholder for qBacktest.py
# Backtest library

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Union, Literal, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

import seaborn as sns
import matplotlib.pyplot as plt
from great_tables import GT
from functools import partial
# from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Helper functions
def smart_forward_fill_weights_optimized(
    df: pd.DataFrame,
    weight_cols: Optional[List[str]] = None,
    date_col: str = "date",
    id_col: str = "ticker",
    ) -> pd.DataFrame:
    """
    Forward-fill weights within rebalance intervals, but only for tickers that
    have non-NA weights at the start of each interval. If a ticker is not
    present (NA) on a rebalance date, it will not appear in that interval
    (no carry from prior intervals).

    Args:
        df: DataFrame with at least [date_col, id_col] and weight columns.
        weight_cols: If None, auto-detects weight columns as those not in
                     [date_col, id_col, 'return'].
        date_col: Name of the date column.
        id_col: Name of the security identifier column.

    Returns:
        DataFrame with weights forward-filled per the rules.
    """
    out = df.copy()

    # Auto-detect weight columns
    if weight_cols is None:
        reserved = {date_col, id_col, "return"}
        weight_cols = [c for c in out.columns if c not in reserved]

    # Normalize/Sort dates
    out[date_col] = pd.to_datetime(out[date_col])
    out.sort_values([date_col, id_col], inplace=True)

    # Row-level "has any data" across weight columns
    out["_row_has_data"] = out[weight_cols].notna().any(axis=1)

    # A "rebalance date" is any date where at least one row has data
    # Broadcast per-row via transform, then assign a global interval id
    rebalance_flag_by_date = (
        out.groupby(date_col)["_row_has_data"].transform("max").astype(int)
    )

    # Build a date->rebalance_id map so the cumsum increments once per date
    date_map = (
        out[[date_col]]
        .drop_duplicates()
        .merge(
            out.groupby(date_col)["_row_has_data"].max().rename("_reb_flag").reset_index(),
            on=date_col,
            how="left",
        )
        .assign(_rebalance_id=lambda d: d["_reb_flag"].cumsum())
        [[date_col, "_rebalance_id"]]
    )
    out = out.merge(date_map, on=date_col, how="left")

    # Within each (rebalance interval, ticker), mark if the ticker is *in* the interval:
    # i.e., it has non-NA weights at the interval's first date.
    # Using 'first' is safe because rows are sorted by date.
    out["_is_member"] = out.groupby(["_rebalance_id", id_col])["_row_has_data"].transform("first")

    # Mask out non-members inside each interval so they never get filled there
    for col in weight_cols:
        out[col] = out[col].where(out["_is_member"])

    # Now forward fill only within each (rebalance interval, ticker)
    for col in weight_cols:
        out[col] = out.groupby(["_rebalance_id", id_col])[col].ffill()

    # Clean up helpers
    out.drop(columns=["_row_has_data", "_rebalance_id", "_is_member"], inplace=True)

    return out
 
# Backtest classes
class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"

class PortfolioType(str, Enum):
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    MARKET_NEUTRAL = "market_neutral"

class BacktestConfig(BaseModel):
    """Configuration settings for the backtest"""
    asset_class: AssetClass
    portfolio_type: PortfolioType
    model_type: Optional[str] = None
    partition: Optional[str] = None
    target_var: Optional[str] = None
    lag: int = Field(default=0, ge=0)
    annualization_factor: float = Field(
        default=252,  # Default for equity daily returns
        description="Factor used to annualize returns (252 for daily equity, 12 for monthly, etc.)"
    )

    @field_validator('lag')
    def validate_lag(cls, v):
        if v < 0:
            raise ValueError("Lag must be non-negative")
        return v

class BacktestResults(BaseModel):
    """Container for backtest results"""
    sharpe_ratio_benchmark: float
    sharpe_ratio_optimal: float
    cumulative_return_benchmark: float
    cumulative_return_optimal: float
    daily_returns: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

class Backtest(BaseModel):
    """Main backtesting class"""
    config: BacktestConfig
    df_portfolio: pd.DataFrame = pd.DataFrame()
    df_pnl: pd.DataFrame = pd.DataFrame()
    
    class Config:
        arbitrary_types_allowed = True

    @field_validator('config')
    def validate_config(cls, v):
        return v

    def _validate_input_data(
        self,
        df_return: pd.DataFrame,
        df_portfolio: pd.DataFrame) -> None:
        """Validate input dataframes"""
        required_return_cols = {'date', 'ticker', 'return'}
        required_portfolio_cols = {'date', 'ticker', 'weight'}

        missing_return_cols = required_return_cols - set(df_return.columns)
        missing_portfolio_cols = required_portfolio_cols - set(df_portfolio.columns)

        if missing_return_cols:
            raise ValueError(f"Missing required columns in returns DataFrame: {missing_return_cols}")
        if missing_portfolio_cols:
            raise ValueError(f"Missing required columns in portfolio DataFrame: {missing_portfolio_cols}")

        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df_return['date']):
            try:
                df_return['date'] = pd.to_datetime(df_return['date'])
            except:
                raise ValueError("Could not convert 'date' column to datetime in returns DataFrame")

        if not pd.api.types.is_datetime64_any_dtype(df_portfolio['date']):
            try:
                df_portfolio['date'] = pd.to_datetime(df_portfolio['date'])
            except:
                raise ValueError("Could not convert 'date' column to datetime in portfolio DataFrame")

    def _prepare_portfolio_data(
        self,
        df_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Prepare portfolio data for backtesting"""
        df = df_portfolio.copy()

        # Add benchmark weights if not present
        if 'weight_benchmark' not in df.columns:
            df['weight_benchmark'] = df['weight']

        # Validate portfolio constraints based on portfolio type
        if self.config.portfolio_type == PortfolioType.LONG_ONLY:
            if (df['weight'] < 0).any():
                raise ValueError("Long-only portfolio cannot have negative weights")
        elif self.config.portfolio_type == PortfolioType.MARKET_NEUTRAL:
            portfolio_sums = df.groupby('date')['weight'].sum()
            if not np.allclose(portfolio_sums, 0, atol=1e-5):
                raise ValueError("Market-neutral portfolio weights must sum to zero for each date")

        return df

    def _calculate_returns(
        self,
        df_merged: pd.DataFrame) -> pd.DataFrame:
        # import pdb; pdb.set_trace()
        """Calculate portfolio and benchmark returns"""
        df = df_merged.copy()
        
        # Forward fill multiple columns within each group
        # import pdb; pdb.set_trace()
        df = smart_forward_fill_weights_optimized(df, weight_cols=['weight','weight_benchmark'])
        # cols = [i for i in list(df.columns) if i not in ['date','ticker','return']]
        # df[cols] = df.groupby(['ticker'])[cols].ffill()

        df.dropna(subset=['weight','weight_benchmark'], inplace=True)

        # Apply lag if specified
        if self.config.lag > 0:
            lagged_cols = ['weight_benchmark', 'weight']
            if 'after_mkt' in df.columns:
                lagged_cols.append('after_mkt')
            if 'date_earn' in df.columns:
                lagged_cols.append('date_earn')

            for col in lagged_cols:
                df[col] = df.groupby('ticker')[col].shift(self.config.lag)

        # df.dropna(subset=['A', 'B'], inplace=True)
        # df.dropna(inplace=True)
        df.sort_values(['date', 'ticker'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Calculate weighted returns
        df['return_benchmark'] = df['return'] * df['weight_benchmark']
        df['return_opt'] = df['return'] * df['weight']

        return df

    def run_backtest(
        self,
        df_return: pd.DataFrame,
        df_portfolio: pd.DataFrame,
        plot: bool = False) -> BacktestResults:
        """
        Run the backtest

        Parameters:
        -----------
        df_return : pd.DataFrame
            DataFrame containing asset returns with columns [date, ticker, return]
        df_portfolio : pd.DataFrame
            DataFrame containing portfolio weights with columns [date, ticker, weight]
        plot : bool, optional
            Whether to plot the cumulative returns

        Returns:
        --------
        BacktestResults
            Container with backtest results and metrics
        """
        # Validate input data
        self._validate_input_data(df_return, df_portfolio)

        # Prepare portfolio data
        df_portfolio = self._prepare_portfolio_data(df_portfolio)
        
        # Merge returns and portfolio data
        merge_cols = ['ticker', 'date']
        df_merged = df_return.merge(
            df_portfolio,
            how='left',
            on=merge_cols
        )
        
        if 'after_mkt' in df_merged.columns:
            ix = (df_merged.after_mkt==1)
            df_merged.loc[ix,'date_ann'] = df_merged.loc[ix]['date_earn']
            df_merged.dropna(subset=['date_earn', 'after_mkt'], inplace=True)
        self.df_portfolio = df_merged.copy()

        # Calculate returns
        df_processed = self._calculate_returns(df_merged)

        # Aggregate returns by date
        df_pnl = df_processed[['date', 'return_benchmark', 'return_opt']].groupby(['date']).sum()
        self.df_pnl = df_pnl.copy()
        self.df_pnl.insert(0, 'factor', self.config.model_type)

        # Calculate metrics
        ann_factor = np.sqrt(self.config.annualization_factor)
        sharpe_bench = ann_factor * df_pnl['return_benchmark'].mean() / df_pnl['return_benchmark'].std()
        sharpe_opt = ann_factor * df_pnl['return_opt'].mean() / df_pnl['return_opt'].std()

        cum_returns = df_pnl[['return_benchmark', 'return_opt']].cumsum()

        if plot:
            self._plot_results(cum_returns)

        return BacktestResults(
            sharpe_ratio_benchmark=sharpe_bench,
            sharpe_ratio_optimal=sharpe_opt,
            cumulative_return_benchmark=cum_returns['return_benchmark'].iloc[-1],
            cumulative_return_optimal=cum_returns['return_opt'].iloc[-1],
            daily_returns=df_pnl
        )

    def _plot_results(self, cum_returns: pd.DataFrame) -> None:
        """Plot cumulative returns"""
        import matplotlib.pyplot as plt

        title = f"Backtest Results - {self.config.asset_class.value.title()}"
        if self.config.model_type:
            title += f"\nModel: {self.config.model_type}"
        if self.config.partition:
            title += f" | Partition: {self.config.partition}"
        if self.config.target_var:
            title += f" | Target: {self.config.target_var}"

        cum_returns.plot(
            title=title,
            figsize=(16, 8),
            ylabel="Cumulative Return",
            xlabel="Date"
        )
        plt.grid(True)
        plt.show()

# Risk analysis classes
class RiskDecompositionType(str, Enum):
    """Types of risk decomposition"""
    FACTOR = "factor"
    SPECIFIC = "specific"
    TOTAL = "total"

class ReturnDecompositionType(str, Enum):
    """Types of return attribution"""
    FACTOR = "factor"
    SPECIFIC = "specific"
    TOTAL = "total"
    ACTIVE = "active"

class RiskData(BaseModel):
    """Container for risk-related data"""
    exposures: pd.DataFrame = Field(default_factory=pd.DataFrame)
    returns: pd.DataFrame = Field(default_factory=pd.DataFrame)
    factor_returns: pd.DataFrame = Field(default_factory=pd.DataFrame)
    portfolio_returns: pd.DataFrame = Field(default_factory=pd.DataFrame)
    
    class Config:
        arbitrary_types_allowed = True

# Risk decomposition analysis
@dataclass
class RiskDecompositionResult:
    """Results from risk decomposition analysis"""
    total_risk: pd.DataFrame
    factor_risk: pd.DataFrame
    specific_risk: pd.DataFrame
    factor_contribution: pd.DataFrame
    correlation_matrix: pd.DataFrame
    
class RiskAnalytics(BaseModel):
    """Enhanced risk analytics framework"""
    dates: List[str] = Field(default_factory=list)
    factors: List[str] = Field(default_factory=list)
    securities: List[str] = Field(default_factory=list)
    data: RiskData = Field(default_factory=RiskData)
    lookback_periods: int = Field(default=60, ge=1)
    annualization_factor: float = Field(default=252.0, gt=0)
    
    class Config:
        arbitrary_types_allowed = True
        
    @field_validator('dates')
    def validate_dates(cls, v):
        """Validate date format"""
        for date in v:
            try:
                # datetime.strptime(date.replace('-',''), '%Y%m%d')
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date format for {date}")
        return v

    def _calculate_covariance(
        self,
        data: pd.DataFrame,
        date: str,
        lookback: Optional[int] = None) -> np.ndarray:
        """Calculate covariance matrix for given period"""
        lookback = lookback or self.lookback_periods
        end_idx = data.index.get_loc(date)
        start_idx = max(0, end_idx - lookback)
        factor_cov = np.cov(data.iloc[start_idx:end_idx].T)
        factor_cov = np.nan_to_num(factor_cov, nan=0)
        if factor_cov.sum() == 0.:
            print("Warning: nan covariance matrix")
        return factor_cov

    def decompose_risk(
        self,
        weights: pd.DataFrame,
        date: str,
        is_relative: bool = False) -> RiskDecompositionResult:
        """
        Perform risk decomposition analysis
        
        Parameters:
        -----------
        weights : pd.Series
            Portfolio weights
        date : str
            Analysis date
        is_relative : bool
            Whether to perform relative to benchmark
            
        Returns:
        --------
        RiskDecompositionResult
            Decomposition results
        """
        try:
            # Get factor exposures
            exposures = self.data.exposures.loc[
                self.data.exposures['date'] == str(date) # .replace('-','') # date
            ].copy()
            
            # Calculate factor covariance
            factor_cov = self._calculate_covariance(
                self.data.factor_returns[self.factors],
                str(date) # .replace('-','')
            )
            
            # Calculate specific risk
            specific_risk = self.data.returns.groupby('sid').var()
            specific_risk.reset_index(drop=False, inplace=True)
            specific_risk = exposures.merge(
                specific_risk,
                how='left',
                on='sid'
            )['return'].fillna(0)

            # Get weights 
            weights = weights.loc[weights['date']==str(date) # .replace('-','')
                                 ]
            weights = exposures.merge(
                weights,
                how='left',
                on='sid'
            )['weight'].fillna(0)
            weights = np.array(weights)

            # Calculate risk components
            factor_exposures = exposures[self.factors].T @ weights
            factor_risk = (
                factor_exposures.values.reshape(-1,1) @ 
                factor_exposures.values.reshape(1,-1) * 
                factor_cov
            )
            
            specific_risk_contrib = (weights ** 2) * specific_risk
            total_risk = (
                np.sum(factor_risk) + 
                np.sum(specific_risk_contrib)
            )
            
            # Create results
            results = RiskDecompositionResult(
                total_risk=pd.DataFrame({
                    'date': [date],
                    'risk': [np.sqrt(total_risk * self.annualization_factor)]
                }),
                factor_risk=pd.DataFrame({
                    'date': [date],
                    'risk': [np.sqrt(np.sum(factor_risk) * self.annualization_factor)]
                }),
                specific_risk=pd.DataFrame({
                    'date': [date],
                    'risk': [np.sqrt(np.sum(specific_risk_contrib) * self.annualization_factor)]
                }),
                factor_contribution=pd.DataFrame({
                    'factor': self.factors,
                    'contribution': np.sqrt(np.diag(factor_risk) * self.annualization_factor)
                }),
                correlation_matrix=pd.DataFrame(
                    factor_cov / np.sqrt(np.diag(factor_cov)).reshape(-1,1) / 
                    np.sqrt(np.diag(factor_cov)).reshape(1,-1),
                    index=self.factors,
                    columns=self.factors
                )
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Risk decomposition failed for date {date}: {str(e)}")
            raise

    def attribute_returns(
        self,
        weights: pd.DataFrame,
        date: str,
        lookback: Optional[int] = None) -> pd.DataFrame:
        """
        Perform return attribution analysis
        """
        try:
            lookback = lookback or self.lookback_periods
            
            # Get factor exposures and returns
            exposures = self.data.exposures.loc[self.data.exposures['date'] == str(date)#.replace('-','')
                                               ].copy()
            
            factor_rets = self.data.factor_returns.loc[date]
            
            # Get weights 
            weights = weights.loc[weights['date'] == str(date) #.replace('-','')
                                 ]
            weights = exposures.merge(
                weights,
                how='left',
                on='sid'
            )['weight'].fillna(0)
            weights = np.array(weights)
        
            # Calculate factor and specific returns
            factor_exposures = exposures[self.factors].T @ weights
            factor_returns = factor_exposures * factor_rets[self.factors]
            
            # total_return = self.data.portfolio_returns.loc[date, 'return']
            # total_return = self.data.portfolio_returns.loc[date.replace('-',''), 'return_opt']
            total_return = self.data.portfolio_returns.loc[str(date), 'return_opt']
            specific_return = total_return - factor_returns.sum()
            
            return pd.DataFrame({
                'date': [date],
                'total_return': [total_return],
                'factor_return': [factor_returns.sum()],
                'specific_return': [specific_return],
                **{f'{f}_return': [r] for f, r in factor_returns.items()}
            })
            
        except Exception as e:
            logger.error(f"Return attribution failed for date {date}: {str(e)}")
            raise

    def plot_risk_decomposition(
        self,
        risk_results: RiskDecompositionResult,
        plot_type: str = 'bar') -> None:
        """Create visualization of risk decomposition"""
        plt.figure(figsize=(12, 6))
        
        if plot_type == 'bar':
            # Plot factor contributions
            ax = sns.barplot(
                data=risk_results.factor_contribution,
                x='factor',
                y='contribution',
                palette='viridis'
            )
            
            plt.title('Factor Risk Contributions')
            plt.xticks(rotation=45)
            plt.ylabel('Annualized Risk Contribution (%)')
            
        elif plot_type == 'pie':
            # Create pie chart of risk components
            total = risk_results.total_risk['risk'].iloc[0]
            factor = risk_results.factor_risk['risk'].iloc[0]
            specific = risk_results.specific_risk['risk'].iloc[0]
            
            plt.pie(
                [factor, specific],
                labels=['Factor Risk', 'Specific Risk'],
                autopct='%1.1f%%',
                colors=sns.color_palette('viridis')
            )
            plt.title('Risk Decomposition')
            
        plt.tight_layout()
        plt.show()

    def plot_factor_exposures(
        self,
        weights: pd.Series,
        date: str,
        benchmark_weights: Optional[pd.Series] = None) -> None:
        """Plot factor exposures with optional benchmark comparison"""
        exposures = self.data.exposures.loc[self.data.exposures['date']==str(date) # .replace('-','')
                                           ].copy()
        
        # Get weights 
        weights = weights.loc[weights['date']==str(date) # .replace('-','')
                             ]
        weights = exposures.merge(
            weights,
            how='left',
            on='sid'
        )['weight'].fillna(0)
        weights = np.array(weights)

        portfolio_exposures = exposures[self.factors].T @ weights
        
        data = pd.DataFrame({
            'factor': self.factors,
            'Portfolio': portfolio_exposures
        })
        
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights.loc[benchmark_weights['date']==str(date) # .replace('-','')
                                                     ]
            benchmark_weights = exposures.merge(
                benchmark_weights,
                how='left',
                on='sid'
            )['weight_benchmark'].fillna(0)
            benchmark_weights = np.array(benchmark_weights)
        
            benchmark_exposures = exposures[self.factors].T @ benchmark_weights
            data['Benchmark'] = benchmark_exposures
            data['Active'] = data['Portfolio'] - data['Benchmark']
        
        plt.figure(figsize=(12, 6))
        
        if benchmark_weights is not None:
            data_melted = data.melt(
                id_vars=['factor'],
                value_vars=['Portfolio', 'Benchmark', 'Active']
            )
            
            sns.barplot(
                data=data_melted,
                x='factor',
                y='value',
                hue='variable',
                palette='viridis'
            )
        else:
            sns.barplot(
                data=data,
                x='factor',
                y='Portfolio',
                palette='viridis'
            )
            
        plt.title('Factor Exposures')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def create_risk_report(
        self,
        risk_results: RiskDecompositionResult,
        date: str) -> GT:
        """
        Create formatted risk decomposition report
        """
        # Prepare factor contribution data
        factor_contrib = risk_results.factor_contribution.copy()
        factor_contrib['contribution_pct'] = (
            factor_contrib['contribution'] / 
            risk_results.total_risk['risk'].iloc[0] * 
            1 # 100
        )
        
        # Create great_tables table
        gt_table = (GT(factor_contrib)
            .tab_header(
                title=f"Risk Decomposition Report - {date}",
                subtitle="Factor Risk Contributions"
            )
            .fmt_percent(
                columns=['contribution_pct'],
                decimals=1
            )
            .fmt_number(
                columns=['contribution'],
                decimals=2
            )
            .cols_label(
                factor="Risk Factor",
                contribution="Contribution",
                contribution_pct="% of Total"
            )
            .tab_source_note(
                source_note=f"Total Portfolio Risk: {risk_results.total_risk['risk'].iloc[0]:.2%}"
            )
            .tab_spanner(
                label="Risk Metrics",
                columns=['contribution', 'contribution_pct']
            )
            .data_color(palette=["lightgreen", "darkred"])
                   )
        
        return gt_table

    def create_exposure_report(
        self,
        weights: pd.Series,
        date: str,
        benchmark_weights: Optional[pd.Series] = None) -> GT:
        """
        Create formatted factor exposure report
        """
        exposures = self.data.exposures.loc[self.data.exposures['date']==str(date) # .replace('-','')
                                           ].copy()
        
        weights = weights.loc[weights['date']==str(date) # .replace('-','')
                             ]
        weights = exposures.merge(
            weights,
            how='left',
            on='sid'
        )['weight'].fillna(0)
        weights = np.array(weights)
        
        portfolio_exposures = exposures[self.factors].T @ weights
        
        data = pd.DataFrame({
            'factor': self.factors,
            'portfolio': portfolio_exposures
        })
        
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights.loc[benchmark_weights['date']==str(date) # .replace('-','')
                                                     ]
            benchmark_weights = exposures.merge(
                benchmark_weights,
                how='left',
                on='sid'
            )['weight_benchmark'].fillna(0)
            benchmark_weights = np.array(benchmark_weights)
            
            benchmark_exposures = exposures[self.factors].T @ benchmark_weights
            data['benchmark'] = benchmark_exposures
            data['active'] = data['portfolio'] - data['benchmark']
        
        gt_table = (GT(data)
            .tab_header(
                title=f"Factor Exposure Report - {date}",
                subtitle="Portfolio Factor Exposures"
            )
            .fmt_number(
                columns=['portfolio', 'benchmark', 'active'] 
                if benchmark_weights is not None 
                else ['portfolio'],
                decimals=2
            )
            .cols_label(
                factor="Factor",
                portfolio="Portfolio",
                benchmark="Benchmark",
                active="Active"
            )
            .tab_source_note(
                source_note="All values represent standardized exposures"
            )
            .data_color(palette=["lightgreen", "darkgreen"] # columns=['portfolio', 'benchmark', 'active'], 
                       )
                   )
        
        return gt_table
    
    def create_return_report(
        self,
        weights: pd.DataFrame,
        date: str) -> GT:
        """
        Create formatted factor exposure report
        """
        df = self.attribute_returns(weights, date)
        
        df = df.drop(columns=['date']).T
        df.columns = ['value']
        df['value'] *= 100.
        df.reset_index(drop=False, inplace=True)

        gt_table = (GT(df)
                    .tab_header(
                        title=f"Return Decomposition Report - {date}",
                        subtitle="Factor Return Attribution"
                    )
                    .fmt_number(
                        columns=['value'],
                        decimals=2
                    )
                    .cols_label(
                        value="Contribution"
                    )
                    .tab_source_note(
                        source_note="Factor values represent marginal return contributions"
                    )
                    .data_color(palette=["darkred", "darkgreen"]
                               )
                           )
        
        return gt_table
    
# Time series risk and return analysis    
@dataclass
class TimeSeriesRiskResult:
    """Container for time series risk analysis results"""
    risk_decomposition: pd.DataFrame
    factor_contribution: pd.DataFrame
    return_attribution: pd.DataFrame
    factor_exposures: pd.DataFrame
    rolling_metrics: pd.DataFrame
    
class RiskTimeSeriesAnalytics(BaseModel):
    """Time series risk analytics framework"""
    base_analytics: RiskAnalytics
    rolling_window: int = Field(default=60, ge=1)
    min_periods: int = Field(default=30, ge=1)
    
    class Config:
        arbitrary_types_allowed = True

    def _parallel_date_analysis(
        self,
        dates: List[str],
        analysis_func: Callable,
        **kwargs) -> List[Dict]:
        """Run analysis in parallel for multiple dates"""
        results = []
        
        # for date in tqdm(dates, desc="Processing dates"):
        for date in dates:
            try:
                result = analysis_func(date=date, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Analysis failed for date {date}: {str(e)}")
                continue
                
        return results

    def analyze_time_series(
        self,
        weights: pd.DataFrame, #Dict[str, pd.Series],
        dates: Optional[List[str]] = None,
        benchmark_weights: Optional[pd.DataFrame] = None) -> TimeSeriesRiskResult: # Optional[Dict[str, pd.Series]] = None) -> TimeSeriesRiskResult:
        """
        Perform comprehensive time series risk analysis
        
        Parameters:
        -----------
        weights : Dict[str, pd.Series]
            Dictionary of portfolio weights by date
        dates : List[str], optional
            Dates to analyze. If None, uses all dates in weights
        benchmark_weights : Dict[str, pd.Series], optional
            Dictionary of benchmark weights by date
            
        Returns:
        --------
        TimeSeriesRiskResult
            Comprehensive time series analysis results
        """
        dates = dates or list(weights.keys())
        dates.sort()
        
        # Initialize result containers
        risk_decomp = []
        factor_contrib = []
        return_attr = []
        factor_exp = []
        
        # Run analysis for each date
        weights.index = weights['date']
        if benchmark_weights is not None:
            benchmark_weights.index = benchmark_weights['date']

        # for date in tqdm(dates, desc="Analyzing time series"):
        for date in dates:
            try:
                date = str(date)# .replace('-','')
                # Risk decomposition
                risk_result = self.base_analytics.decompose_risk(
                    weights=weights[weights.index==date],
                    date=date,
                    is_relative=benchmark_weights is not None
                )

                # Store risk decomposition results
                if risk_result.factor_risk['risk'].iloc[0]==0.:
                    print(f"Can't perform risk decomposition for date {date}")
                    continue
                    
                risk_decomp.append({
                    'date': date,
                    'total_risk': risk_result.total_risk['risk'].iloc[0],
                    'factor_risk': risk_result.factor_risk['risk'].iloc[0],
                    'specific_risk': risk_result.specific_risk['risk'].iloc[0]
                })
                
                # Store factor contributions
                for _, row in risk_result.factor_contribution.iterrows():
                    factor_contrib.append({
                        'date': date,
                        'factor': row['factor'],
                        'contribution': row['contribution']
                    })
                # Return attribution
                returns = self.base_analytics.attribute_returns(
                    weights=weights[weights.index==date],
                    date=date
                )
                if 'date' not in returns.columns:
                    returns['date'] = date
                return_attr.append(returns)
                
                # Factor exposures
                exposures = self.base_analytics.data.exposures.loc[
                    self.base_analytics.data.exposures['date'] == date
                ].copy()
                
                portfolio_exp = exposures[self.base_analytics.factors].T @ np.array(weights[weights.index==date]['weight'])
                
                if benchmark_weights is not None:
                    bench_exp = exposures[self.base_analytics.factors].T @ np.array(benchmark_weights[benchmark_weights.index==date]['weight_benchmark'])
                    active_exp = portfolio_exp - bench_exp
                else:
                    bench_exp = pd.Series(0, index=self.base_analytics.factors)
                    active_exp = portfolio_exp
                    
                factor_exp.append({
                    'date': date,
                    **{f'{f}_portfolio': v for f, v in portfolio_exp.items()},
                    **{f'{f}_benchmark': v for f, v in bench_exp.items()},
                    **{f'{f}_active': v for f, v in active_exp.items()}
                })
                
            except Exception as e:
                logger.error(f"Analysis failed for date {date}: {str(e)}")
                continue
        
        # Convert results to DataFrames
        risk_decomp_df = pd.DataFrame(risk_decomp)
        factor_contrib_df = pd.DataFrame(factor_contrib)
        return_attr_df = pd.concat(return_attr)
        factor_exp_df = pd.DataFrame(factor_exp)
        
        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(
            risk_decomp_df,
            factor_contrib_df,
            return_attr_df,
            factor_exp_df
        )
        
        return TimeSeriesRiskResult(
            risk_decomposition=risk_decomp_df,
            factor_contribution=factor_contrib_df,
            return_attribution=return_attr_df,
            factor_exposures=factor_exp_df,
            rolling_metrics=rolling_metrics
        )

    def _calculate_rolling_metrics(
        self,
        risk_decomp: pd.DataFrame,
        factor_contrib: pd.DataFrame,
        return_attr: pd.DataFrame,
        factor_exp: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling risk metrics"""
        
        # Set up date index
        risk_decomp['date'] = pd.to_datetime(risk_decomp['date'])
        risk_decomp.set_index('date', inplace=True)
        
        # Calculate rolling metrics
        rolling_metrics = pd.DataFrame(index=risk_decomp.index)
        
        # Rolling risk metrics
        rolling_metrics['rolling_risk'] = risk_decomp['total_risk'].rolling(
            window=self.rolling_window,
            min_periods=self.min_periods
        ).std() * np.sqrt(self.base_analytics.annualization_factor)
        
        # Rolling factor contribution
        pivot_contrib = factor_contrib.pivot(
            index='date',
            columns='factor',
            values='contribution'
        )
        pivot_contrib.index = pd.to_datetime(pivot_contrib.index)
        
        for factor in self.base_analytics.factors:
            rolling_metrics[f'rolling_{factor}_contrib'] = pivot_contrib[factor].rolling(
                window=self.rolling_window,
                min_periods=self.min_periods
            ).mean()
            
        return rolling_metrics

    def plot_risk_evolution(
        self,
        results: TimeSeriesRiskResult,
        plot_type: str = 'area') -> None:
        """Plot evolution of risk components over time"""
        plt.figure(figsize=(15, 8))
        
        risk_data = results.risk_decomposition.copy()
        # risk_data['date'] = pd.to_datetime(risk_data['date'])
        # risk_data.set_index('date', inplace=True)
        
        if plot_type == 'area':
            plt.stackplot(
                risk_data.index,
                risk_data['factor_risk'],
                risk_data['specific_risk'],
                labels=['Factor Risk', 'Specific Risk'],
                alpha=0.6
            )
            plt.plot(
                risk_data.index,
                risk_data['total_risk'],
                'k--',
                label='Total Risk'
            )
            
        elif plot_type == 'line':
            for col in ['total_risk', 'factor_risk', 'specific_risk']:
                plt.plot(
                    risk_data.index,
                    risk_data[col],
                    label=col.replace('_', ' ').title()
                )
                
        plt.title('Risk Evolution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Annualized Risk')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_factor_contribution_evolution(
        self,
        results: TimeSeriesRiskResult,
        top_n: int = 5) -> None:
        """Plot evolution of factor contributions"""
        plt.figure(figsize=(15, 8))
        
        # Pivot and process factor contributions
        factor_data = results.factor_contribution.copy()
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        pivot_data = factor_data.pivot(
            index='date',
            columns='factor',
            values='contribution'
        )
        
        # Get top N factors by average contribution
        top_factors = pivot_data.mean().nlargest(top_n).index
        
        # Plot
        for factor in top_factors:
            plt.plot(
                pivot_data.index,
                pivot_data[factor],
                label=factor,
                alpha=0.7,
                linewidth=2
            )
            
        plt.title(f'Top {top_n} Factor Contribution Evolution')
        plt.xlabel('Date')
        plt.ylabel('Risk Contribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_rolling_metrics(
        self,
        results: TimeSeriesRiskResult) -> None:
        """Plot rolling risk metrics"""
        plt.figure(figsize=(15, 8))
        
        metrics = results.rolling_metrics.copy()
        
        # Plot rolling risk
        plt.subplot(2, 1, 1)
        plt.plot(
            metrics.index,
            metrics['rolling_risk'],
            label='Rolling Risk',
            linewidth=2
        )
        plt.title('Rolling Risk')
        plt.xlabel('Date')
        plt.ylabel('Risk')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot rolling factor contributions
        plt.subplot(2, 1, 2)
        factor_cols = [col for col in metrics.columns if 'rolling_' in col and col != 'rolling_risk']
        
        for col in factor_cols:
            plt.plot(
                metrics.index,
                metrics[col],
                label=col.replace('rolling_', '').replace('_contrib', ''),
                alpha=0.7
            )
            
        plt.title('Rolling Factor Contributions')
        plt.xlabel('Date')
        plt.ylabel('Contribution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def create_time_series_report(
        self,
        results: TimeSeriesRiskResult) -> GT:
        # report_date: str) -> GT:
        """Create time series analysis report"""
        # Prepare summary statistics
        # import pdb; pdb.set_trace()
        risk_summary = results.risk_decomposition.describe()
        factor_summary = results.factor_contribution.pivot(
            index='date',
            columns='factor',
            values='contribution'
        ).describe()
        
        # Create summary table
        summary_data = pd.DataFrame({
            'Metric': ['Average Total Risk', 'Max Total Risk', 'Risk Volatility'],
            'Value': [
                f"{risk_summary.loc['mean', 'total_risk']:.2%}",
                f"{risk_summary.loc['max', 'total_risk']:.2%}",
                f"{risk_summary.loc['std', 'total_risk']:.2%}"
            ]
        })
        
        # Add factor contribution summary
        for factor in self.base_analytics.factors:
            summary_data = pd.concat([
                summary_data,
                pd.DataFrame({
                    'Metric': [f'{factor} Contribution'],
                    'Value': [f"{factor_summary.loc['mean', factor]:.2%}"]
                })
            ])
        
        # Create report
        gt_table = (GT(summary_data)
                    .tab_header(
                        title=f"Time Series Risk Analysis Summary - {results.risk_decomposition.index.date.max()}", # report_date
                        subtitle=f"Analysis Period: {results.risk_decomposition.index.date.min()} to {results.risk_decomposition.index.date.max()}"
                    )
                    .cols_label(
                        Metric="Risk Metric",
                        Value="Value"
                    )
                    .tab_source_note(
                        source_note=f"Based on {len(results.risk_decomposition)} observations"
                    )
                    # .data_color(palette=["lightgreen", "darkred"], na_color="lightgray")
                   )
        
        return gt_table
    
"""
# Example usage:
"""
if __name__ == "__main__":
    # run backtest example
    config = BacktestConfig(
        asset_class=AssetClass.EQUITY,
        portfolio_type=PortfolioType.LONG_SHORT,
        model_type="FACTOR_MODEL", # model_type="EARNINGS_MODEL",
        annualization_factor=252
    )

    backtest = Backtest(config=config)
    
    # df_portfolio = df_wgt.copy()
    # df_portfolio.rename(columns={'sid':'ticker', 'wgt_opt':'weight'}, inplace=True)
    # df_returns = df_ret_long.copy()
    # df_returns.rename(columns={'sid':'ticker'}, inplace=True)
    # # df_ret_opt = get_backtest(df_ret_long, df_wgt, lag=1, flag_plot=False)
    # results = backtest.run_backtest(df_returns, df_portfolio, plot=True)

    # print(f"Benchmark Sharpe: {results.sharpe_ratio_benchmark:.2f}")
    # print(f"Optimal Sharpe: {results.sharpe_ratio_optimal:.2f}")
    # print(f"Benchmark Return: {results.cumulative_return_benchmark:.2%}")
    # print(f"Optimal Return: {results.cumulative_return_optimal:.2%}")
