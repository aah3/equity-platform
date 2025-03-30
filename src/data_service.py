# src/data_service.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Union, Tuple
import datetime
import logging
from src.database_sqlite import DatabaseManager

logger = logging.getLogger(__name__)

# Create cache functions outside of the class for proper hashing
# Use leading underscore for parameters that shouldn't be hashed
@st.cache_data(ttl=3600)
def _cached_get_indices(_db_manager):
    """Cached function to get indices"""
    indices_df = _db_manager.get_indices()
    if indices_df.empty:
        return []
    return indices_df.to_dict('records')

@st.cache_data(ttl=3600)
def _cached_get_factors(_db_manager):
    """Cached function to get available factors"""
    return _db_manager.get_available_factors()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_index_constituents(_db_manager, index_id, as_of_date=None):
    """Cached function to get index constituents"""
    return _db_manager.get_index_constituents(index_id, as_of_date)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_returns(_db_manager, tickers, start_date=None, end_date=None, source='daily'):
    """Cached function to get security returns"""
    return _db_manager.get_returns_data(tickers, start_date, end_date, source)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_factor_data(_db_manager, factor_id, tickers=None, start_date=None, end_date=None):
    """Cached function to get factor data"""
    return _db_manager.get_factor_data(factor_id, tickers, start_date, end_date)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_price_data(_db_manager, ticker, start_date=None, end_date=None):
    """Cached function to get price data"""
    return _db_manager.get_price_data(ticker, start_date, end_date)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_factor_correlation(_factor_values, start_date=None, end_date=None):
    """Cached function to calculate factor correlation matrix"""
    # Convert to a DataFrame and calculate correlation
    returns_df = pd.DataFrame(_factor_values)
    return returns_df.corr()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_perf_metrics(_returns_pivot):
    """Cached function to calculate performance metrics"""
    metrics = []
    
    for bucket in _returns_pivot.columns:
        bucket_returns = _returns_pivot[bucket].dropna()
        
        if not bucket_returns.empty:
            # Calculate metrics
            total_return = (1 + bucket_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(bucket_returns)) - 1
            volatility = bucket_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = (bucket_returns.cumsum() - bucket_returns.cumsum().cummax()).min()
            
            metrics.append({
                'bucket': bucket,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_monthly_return': bucket_returns.mean() * 21,
                'win_rate': (bucket_returns > 0).mean()
            })
    
    return pd.DataFrame(metrics)

class DataService:
    """
    Service class to provide data for the Streamlit application.
    Interfaces with the DatabaseManager to retrieve and process data.
    Uses caching to improve performance.
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the data service.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager or DatabaseManager()
    
    def get_available_indices(self) -> List[Dict]:
        """
        Get list of available indices.
        
        Returns:
            List of index information dictionaries
        """
        return _cached_get_indices(self.db_manager)
    
    def get_available_factors(self) -> List[str]:
        """
        Get list of available factors.
        
        Returns:
            List of factor IDs
        """
        return _cached_get_factors(self.db_manager)
    
    def get_index_constituents(self, index_id: str, as_of_date: str = None) -> pd.DataFrame:
        """
        Get constituents of an index.
        
        Args:
            index_id: Index identifier
            as_of_date: Optional date to get constituents as of
            
        Returns:
            DataFrame of constituents
        """
        return _cached_get_index_constituents(self.db_manager, index_id, as_of_date)
    
    def get_security_returns(self, tickers: List[str], start_date: str = None,
                            end_date: str = None, frequency: str = 'daily') -> pd.DataFrame:
        """
        Get returns data for securities.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            frequency: Data frequency ('daily' or 'monthly')
            
        Returns:
            DataFrame with returns data
        """
        return _cached_get_returns(self.db_manager, tickers, start_date, end_date, frequency)
    
    def get_factor_data(self, factor_id: str, tickers: List[str] = None,
                      start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get factor data.
        
        Args:
            factor_id: Factor identifier
            tickers: Optional list of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with factor data
        """
        return _cached_get_factor_data(self.db_manager, factor_id, tickers, start_date, end_date)
    
    def get_price_data(self, ticker: str, start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
        """
        Get price data for a security.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with price data
        """
        return _cached_get_price_data(self.db_manager, ticker, start_date, end_date)
    
    def get_factor_returns(self, factor_id: str, n_buckets: int = 5,
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Calculate factor portfolio returns based on factor values.
        
        Args:
            factor_id: Factor identifier
            n_buckets: Number of buckets for portfolio construction
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            DataFrame with factor portfolio returns
        """
        # Get factor data
        factor_data = self.get_factor_data(factor_id, start_date=start_date, end_date=end_date)
        if factor_data.empty:
            return pd.DataFrame()
        
        # Get unique dates in factor data
        dates = sorted(factor_data['date'].unique())
        
        # Prepare results container
        results = []
        
        # Process each date
        for date in dates:
            # Get factor values for current date
            current_factors = factor_data[factor_data['date'] == date]
            
            # Divide into quantiles
            current_factors['bucket'] = pd.qcut(current_factors['value'], 
                                              n_buckets, 
                                              labels=False, 
                                              duplicates='drop')
            
            # Get tickers in each bucket
            for bucket in range(n_buckets):
                bucket_tickers = current_factors[current_factors['bucket'] == bucket]['ticker'].tolist()
                
                # Get returns for these tickers
                next_date_idx = dates.index(date) + 1 if dates.index(date) < len(dates) - 1 else None
                
                if next_date_idx is not None:
                    next_date = dates[next_date_idx]
                    # Get returns between current date and next date
                    returns = self.get_security_returns(bucket_tickers, date, next_date)
                    
                    if not returns.empty:
                        # Calculate average return for the bucket
                        avg_return = returns['return'].mean()
                        
                        results.append({
                            'date': date,
                            'bucket': bucket + 1,  # 1-based for readability
                            'return': avg_return
                        })
        
        # Convert results to DataFrame
        if not results:
            return pd.DataFrame()
            
        factor_returns = pd.DataFrame(results)
        
        # Add long-short portfolio
        if n_buckets > 1:
            # For each date, calculate long-short return (highest bucket - lowest bucket)
            ls_returns = []
            
            for date in factor_returns['date'].unique():
                date_returns = factor_returns[factor_returns['date'] == date]
                high_return = date_returns[date_returns['bucket'] == n_buckets]['return'].values
                low_return = date_returns[date_returns['bucket'] == 1]['return'].values
                
                if len(high_return) > 0 and len(low_return) > 0:
                    ls_returns.append({
                        'date': date,
                        'bucket': 'long_short',
                        'return': high_return[0] - low_return[0]
                    })
            
            if ls_returns:
                factor_returns = pd.concat([factor_returns, pd.DataFrame(ls_returns)])
        
        return factor_returns
    
    def get_factor_correlation_matrix(self, start_date: str = None, 
                                    end_date: str = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Correlation matrix DataFrame
        """
        factors = self.get_available_factors()
        if not factors:
            return pd.DataFrame()
        
        # Get factor data for each factor
        factor_values = {}
        
        for factor_id in factors:
            factor_returns = self.get_factor_returns(factor_id, n_buckets=5, 
                                                   start_date=start_date, 
                                                   end_date=end_date)
            
            if not factor_returns.empty:
                # Extract long-short returns if available
                ls_returns = factor_returns[factor_returns['bucket'] == 'long_short']
                
                if not ls_returns.empty:
                    factor_values[factor_id] = ls_returns.set_index('date')['return'].to_list()
                else:
                    # If no long-short, use highest bucket
                    highest_bucket = factor_returns['bucket'].max()
                    bucket_returns = factor_returns[factor_returns['bucket'] == highest_bucket]
                    factor_values[factor_id] = bucket_returns.set_index('date')['return'].to_list()
        
        if not factor_values:
            return pd.DataFrame()
        
        # Use cached function for correlation calculation
        return _cached_get_factor_correlation(factor_values, start_date, end_date)
    
    def calculate_portfolio_returns(self, tickers: List[str], weights: List[float],
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        Calculate portfolio returns based on ticker weights.
        
        Args:
            tickers: List of ticker symbols
            weights: List of weights (should sum to 1)
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            DataFrame with portfolio returns
        """
        if not tickers or len(tickers) != len(weights):
            return pd.DataFrame()
            
        # Get returns for all tickers
        returns = self.get_security_returns(tickers, start_date, end_date)
        
        if returns.empty:
            return pd.DataFrame()
            
        # Create weights dictionary
        weights_dict = dict(zip(tickers, weights))
        
        # Calculate weighted returns
        returns['weighted_return'] = returns.apply(
            lambda row: row['return'] * weights_dict.get(row['ticker'], 0), 
            axis=1
        )
        
        # Aggregate by date
        portfolio_returns = returns.groupby('date')['weighted_return'].sum().reset_index()
        portfolio_returns.rename(columns={'weighted_return': 'return'}, inplace=True)
        
        return portfolio_returns
    
    def get_index_returns(self, index_id: str, start_date: str = None,
                        end_date: str = None) -> pd.DataFrame:
        """
        Calculate index returns based on constituents and weights.
        
        Args:
            index_id: Index identifier
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            DataFrame with index returns
        """
        # Get index constituents
        constituents = self.get_index_constituents(index_id)
        
        if constituents.empty:
            return pd.DataFrame()
            
        # Extract tickers and weights
        tickers = constituents['ticker'].tolist()
        weights = constituents['weight'].tolist()
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Calculate portfolio returns
        return self.calculate_portfolio_returns(tickers, weights, start_date, end_date)
    
    def generate_demo_data(self) -> bool:
        """
        Generate demo data for testing and demonstration.
        
        Returns:
            Success status
        """
        return self.db_manager.generate_demo_data()
    
    def get_factor_performance_metrics(self, factor_id: str, 
                                     n_buckets: int = 5) -> pd.DataFrame:
        """
        Calculate performance metrics for factor portfolios.
        
        Args:
            factor_id: Factor identifier
            n_buckets: Number of buckets for portfolio construction
            
        Returns:
            DataFrame with performance metrics
        """
        # Get factor returns
        factor_returns = self.get_factor_returns(factor_id, n_buckets)
        
        if factor_returns.empty:
            return pd.DataFrame()
        
        # Create pivot table of returns by date and bucket
        returns_pivot = factor_returns.pivot(index='date', columns='bucket', values='return')
        
        # Use cached function to calculate metrics
        return _cached_get_perf_metrics(returns_pivot)
    
    def get_securities_by_factor_rank(self, factor_id: str, top_n: int = 20, 
                                     ascending: bool = False) -> pd.DataFrame:
        """
        Get securities ranked by factor value.
        
        Args:
            factor_id: Factor identifier
            top_n: Number of top securities to return
            ascending: If True, sort in ascending order (better for value factors)
            
        Returns:
            DataFrame with ranked securities
        """
        factor_data = self.get_factor_data(factor_id)
        
        if factor_data.empty:
            return pd.DataFrame()
        
        # Get most recent date with data
        latest_date = factor_data['date'].max()
        latest_data = factor_data[factor_data['date'] == latest_date]
        
        # Sort by factor value
        ranked_securities = latest_data.sort_values('value', ascending=ascending).head(top_n)
        
        return ranked_securities
    
    def get_factor_time_series(self, factor_id: str, ticker: str) -> pd.DataFrame:
        """
        Get time series of factor values for a specific security.
        
        Args:
            factor_id: Factor identifier
            ticker: Ticker symbol
            
        Returns:
            DataFrame with factor time series
        """
        factor_data = self.get_factor_data(factor_id)
        
        if factor_data.empty:
            return pd.DataFrame()
        
        # Filter for the specified ticker
        ticker_data = factor_data[factor_data['ticker'] == ticker]
        
        return ticker_data.sort_values('date')
    
    def backtest_multifactor_strategy(self, factors: List[str], weights: List[float], 
                                    n_buckets: int = 5, top_n: int = 100,
                                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Backtest a simple multi-factor strategy.
        
        Args:
            factors: List of factors to use
            weights: Weight of each factor
            n_buckets: Number of buckets for portfolio construction
            top_n: Number of top securities to include
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with backtest results
        """
        if not factors or len(factors) != len(weights) or sum(weights) == 0:
            return pd.DataFrame()
        
        # Get factor data for all factors
        factor_data_dict = {}
        
        for factor_id in factors:
            factor_data = self.get_factor_data(factor_id, start_date=start_date, end_date=end_date)
            if not factor_data.empty:
                factor_data_dict[factor_id] = factor_data
        
        if not factor_data_dict:
            return pd.DataFrame()
        
        # Get unique dates across all factors
        all_dates = set()
        for factor_id, df in factor_data_dict.items():
            all_dates.update(df['date'].unique())
        
        all_dates = sorted(all_dates)
        
        # Backtest results
        backtest_results = []
        
        # Process each date
        for i, date in enumerate(all_dates[:-1]):
            next_date = all_dates[i + 1]
            
            # Calculate combined score for each security on this date
            security_scores = {}
            
            for j, factor_id in enumerate(factors):
                if factor_id in factor_data_dict:
                    factor_df = factor_data_dict[factor_id]
                    date_data = factor_df[factor_df['date'] == date]
                    
                    # Skip if no data for this date
                    if date_data.empty:
                        continue
                    
                    # Normalize factor values (simple z-score)
                    mean = date_data['value'].mean()
                    std = date_data['value'].std()
                    if std > 0:
                        for _, row in date_data.iterrows():
                            ticker = row['ticker']
                            factor_value = (row['value'] - mean) / std
                            
                            if ticker not in security_scores:
                                security_scores[ticker] = 0
                            
                            security_scores[ticker] += factor_value * weights[j]
            
            if not security_scores:
                continue
            
            # Select top securities
            ranked_securities = sorted(security_scores.items(), key=lambda x: x[1], reverse=True)
            top_securities = ranked_securities[:top_n]
            
            if not top_securities:
                continue
            
            # Get returns for these securities
            tickers = [t[0] for t in top_securities]
            returns = self.get_security_returns(tickers, date, next_date)
            
            if returns.empty:
                continue
            
            # Calculate equal-weighted portfolio return
            portfolio_return = returns['return'].mean()
            
            backtest_results.append({
                'date': date,
                'next_date': next_date,
                'return': portfolio_return,
                'n_securities': len(returns)
            })
        
        if not backtest_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(backtest_results)
        
        # Calculate cumulative returns
        results_df['cumulative_return'] = (1 + results_df['return']).cumprod() - 1
        
        return results_df
    
    # Add these methods to the DataService class to retrieve backtest results
    def get_backtest_results(self, index_id: str, strategy_factor: str) -> pd.DataFrame:
        """
        Get backtest results for a specific index and strategy.
        
        Args:
            index_id: Index identifier (e.g., 'SPX', 'NDX')
            strategy_factor: Strategy factor identifier (e.g., 'value', 'momentum')
            
        Returns:
            DataFrame with backtest results
        """
        try:
            file_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                     f"{index_id}_{strategy_factor}_strategy.parquet")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"No backtest results found for {index_id} {strategy_factor} strategy")
                return pd.DataFrame()
                
            backtest_df = pd.read_parquet(file_path)
            return backtest_df
            
        except Exception as e:
            self.logger.error(f"Error retrieving backtest results: {str(e)}")
            return pd.DataFrame()
        
    def get_backtest_metadata(self, index_id: str, strategy_factor: str) -> dict:
        """
        Get backtest metadata for a specific index and strategy.
        
        Args:
            index_id: Index identifier (e.g., 'SPX', 'NDX')
            strategy_factor: Strategy factor identifier (e.g., 'value', 'momentum')
            
        Returns:
            Dictionary with backtest metadata
        """
        try:
            metadata_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{index_id}_{strategy_factor}_metadata.json")
            
            if not os.path.exists(metadata_path):
                self.logger.warning(f"No backtest metadata found for {index_id} {strategy_factor} strategy")
                return {}
            
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error retrieving backtest metadata: {str(e)}")
            return {}

    def get_available_strategies(self, index_id: str = None) -> List[Dict]:
        """
        Get list of available backtest strategies.
        
        Args:
            index_id: Optional index identifier to filter by
            
        Returns:
            List of strategy information dictionaries
        """
        try:
            backtest_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results')
            
            if not os.path.exists(backtest_path):
                return []
            
            strategy_files = [f for f in os.listdir(backtest_path) if f.endswith('_metadata.json')]
            
            strategies = []
            for file in strategy_files:
                # Skip pair trading files
                if '_pair_metadata.json' in file:
                    continue
                    
                parts = file.split('_')
                if len(parts) >= 3:
                    file_index_id = parts[0]
                    
                    if index_id is None or file_index_id == index_id:
                        file_factor = parts[1]
                        
                        # Get metadata
                        metadata_path = os.path.join(backtest_path, file)
                        with open(metadata_path, 'r') as f:
                            import json
                            metadata = json.load(f)
                        
                        strategies.append({
                            'index_id': file_index_id,
                            'factor': file_factor,
                            'name': metadata.get('strategy_name', f"{file_factor.capitalize()} Strategy"),
                            'sharpe_ratio': metadata.get('sharpe_ratio', 0)
                        })
            
            # Sort by Sharpe ratio
            strategies.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error retrieving available strategies: {str(e)}")
            return []

    def get_pair_trading_results(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """
        Get pair trading backtest results.
        
        Args:
            ticker1: First ticker in the pair
            ticker2: Second ticker in the pair
            
        Returns:
            DataFrame with pair trading results
        """
        try:
            file_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{ticker1}_{ticker2}_pair.parquet")
            
            if not os.path.exists(file_path):
                # Try the reverse order
                file_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{ticker2}_{ticker1}_pair.parquet")
                
                if not os.path.exists(file_path):
                    self.logger.warning(f"No pair trading results found for {ticker1}-{ticker2} pair")
                    return pd.DataFrame()
            
            pair_data = pd.read_parquet(file_path)
            return pair_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving pair trading results: {str(e)}")
            return pd.DataFrame()

    def get_pair_trading_metadata(self, ticker1: str, ticker2: str) -> dict:
        """
        Get pair trading metadata.
        
        Args:
            ticker1: First ticker in the pair
            ticker2: Second ticker in the pair
            
        Returns:
            Dictionary with pair trading metadata
        """
        try:
            metadata_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{ticker1}_{ticker2}_metadata.json")
            
            if not os.path.exists(metadata_path):
                # Try the reverse order
                metadata_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                        f"{ticker2}_{ticker1}_metadata.json")
                
                if not os.path.exists(metadata_path):
                    self.logger.warning(f"No pair trading metadata found for {ticker1}-{ticker2} pair")
                    return {}
            
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error retrieving pair trading metadata: {str(e)}")
            return {}

    def get_available_pairs(self) -> List[Dict]:
        """
        Get list of available pair trading strategies.
        
        Returns:
            List of pair trading information dictionaries
        """
        try:
            backtest_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results')
            
            if not os.path.exists(backtest_path):
                return []
            
            pair_files = [f for f in os.listdir(backtest_path) if f.endswith('_pair_metadata.json') or ('_metadata.json' in f and not any(idx in f for idx in ['SPX', 'NDX', 'RTY']))]
            
            pairs = []
            for file in pair_files:
                parts = file.split('_')
                if len(parts) >= 3:
                    ticker1 = parts[0]
                    ticker2 = parts[1]
                    
                    # Get metadata
                    metadata_path = os.path.join(backtest_path, file)
                    with open(metadata_path, 'r') as f:
                        import json
                        metadata = json.load(f)
                    
                    pairs.append({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'name': metadata.get('pair_name', f"{ticker1}-{ticker2} Pair"),
                        'sharpe_ratio': metadata.get('sharpe_ratio', 0),
                        'win_rate': metadata.get('win_rate', 0)
                    })
            
            # Sort by Sharpe ratio
            pairs.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error retrieving available pairs: {str(e)}")
            return []

    def calculate_drawdown(self, returns_series: pd.Series) -> pd.Series:
        """
        Calculate drawdown for a returns series.
        
        Args:
            returns_series: Series of returns
            
        Returns:
            Series of drawdowns
        """
        # Calculate cumulative returns
        cumulative = (1 + returns_series).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative / running_max) - 1
        
        return drawdown
    
    def get_performance_metrics(self, returns_series: pd.Series) -> Dict:
        """
        Calculate performance metrics for a returns series.
        
        Args:
            returns_series: Series of returns
            
        Returns:
            Dictionary of performance metrics
        """
        if returns_series.empty:
            return {}
            
        try:
            # Calculate basic metrics
            total_return = (1 + returns_series).prod() - 1
            ann_factor = np.sqrt(252)  # Assuming daily returns
            
            # Annualized return
            ann_return = (1 + total_return) ** (ann_factor / len(returns_series)) - 1
            
            # Volatility
            volatility = returns_series.std() * ann_factor
            
            # Sharpe ratio
            sharpe = ann_return / volatility if volatility > 0 else 0
            
            # Drawdown
            drawdown = self.calculate_drawdown(returns_series)
            max_drawdown = drawdown.min()
            
            # Win rate
            win_rate = (returns_series > 0).mean()
            
            return {
                'total_return': total_return,
                'annualized_return': ann_return,
                'annualized_volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
        
    # Add these methods to the DataService class to retrieve backtest results
    def get_backtest_results(self, index_id: str, strategy_factor: str) -> pd.DataFrame:
        """
        Get backtest results for a specific index and strategy.
        
        Args:
            index_id: Index identifier (e.g., 'SPX', 'NDX')
            strategy_factor: Strategy factor identifier (e.g., 'value', 'momentum')
            
        Returns:
            DataFrame with backtest results
        """
        try:
            file_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{index_id}_{strategy_factor}_strategy.parquet")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"No backtest results found for {index_id} {strategy_factor} strategy")
                return pd.DataFrame()
                
            backtest_df = pd.read_parquet(file_path)
            return backtest_df
            
        except Exception as e:
            self.logger.error(f"Error retrieving backtest results: {str(e)}")
            return pd.DataFrame()

    def get_backtest_metadata(self, index_id: str, strategy_factor: str) -> dict:
        """
        Get backtest metadata for a specific index and strategy.
        
        Args:
            index_id: Index identifier (e.g., 'SPX', 'NDX')
            strategy_factor: Strategy factor identifier (e.g., 'value', 'momentum')
            
        Returns:
            Dictionary with backtest metadata
        """
        try:
            metadata_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{index_id}_{strategy_factor}_metadata.json")
            
            if not os.path.exists(metadata_path):
                self.logger.warning(f"No backtest metadata found for {index_id} {strategy_factor} strategy")
                return {}
            
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error retrieving backtest metadata: {str(e)}")
            return {}

    def get_available_strategies(self, index_id: str = None) -> List[Dict]:
        """
        Get list of available backtest strategies.
        
        Args:
            index_id: Optional index identifier to filter by
            
        Returns:
            List of strategy information dictionaries
        """
        try:
            backtest_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results')
            
            if not os.path.exists(backtest_path):
                return []
            
            strategy_files = [f for f in os.listdir(backtest_path) if f.endswith('_metadata.json')]
            
            strategies = []
            for file in strategy_files:
                # Skip pair trading files
                if '_pair_metadata.json' in file:
                    continue
                    
                parts = file.split('_')
                if len(parts) >= 3:
                    file_index_id = parts[0]
                    
                    if index_id is None or file_index_id == index_id:
                        file_factor = parts[1]
                        
                        # Get metadata
                        metadata_path = os.path.join(backtest_path, file)
                        with open(metadata_path, 'r') as f:
                            import json
                            metadata = json.load(f)
                        
                        strategies.append({
                            'index_id': file_index_id,
                            'factor': file_factor,
                            'name': metadata.get('strategy_name', f"{file_factor.capitalize()} Strategy"),
                            'sharpe_ratio': metadata.get('sharpe_ratio', 0)
                        })
            
            # Sort by Sharpe ratio
            strategies.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error retrieving available strategies: {str(e)}")
            return []

    def get_pair_trading_results(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """
        Get pair trading backtest results.
        
        Args:
            ticker1: First ticker in the pair
            ticker2: Second ticker in the pair
            
        Returns:
            DataFrame with pair trading results
        """
        try:
            file_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{ticker1}_{ticker2}_pair.parquet")
            
            if not os.path.exists(file_path):
                # Try the reverse order
                file_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{ticker2}_{ticker1}_pair.parquet")
                
                if not os.path.exists(file_path):
                    self.logger.warning(f"No pair trading results found for {ticker1}-{ticker2} pair")
                    return pd.DataFrame()
            
            pair_data = pd.read_parquet(file_path)
            return pair_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving pair trading results: {str(e)}")
            return pd.DataFrame()

    def get_pair_trading_metadata(self, ticker1: str, ticker2: str) -> dict:
        """
        Get pair trading metadata.
        
        Args:
            ticker1: First ticker in the pair
            ticker2: Second ticker in the pair
            
        Returns:
            Dictionary with pair trading metadata
        """
        try:
            metadata_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f"{ticker1}_{ticker2}_metadata.json")
            
            if not os.path.exists(metadata_path):
                # Try the reverse order
                metadata_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                        f"{ticker2}_{ticker1}_metadata.json")
                
                if not os.path.exists(metadata_path):
                    self.logger.warning(f"No pair trading metadata found for {ticker1}-{ticker2} pair")
                    return {}
            
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error retrieving pair trading metadata: {str(e)}")
            return {}

    def get_available_pairs(self) -> List[Dict]:
        """
        Get list of available pair trading strategies.
        
        Returns:
            List of pair trading information dictionaries
        """
        try:
            backtest_path = os.path.join(self.db_manager.ts_data_path, 'backtest_results')
            
            if not os.path.exists(backtest_path):
                return []
            
            pair_files = [f for f in os.listdir(backtest_path) if f.endswith('_pair_metadata.json') or ('_metadata.json' in f and not any(idx in f for idx in ['SPX', 'NDX', 'RTY']))]
            
            pairs = []
            for file in pair_files:
                parts = file.split('_')
                if len(parts) >= 3:
                    ticker1 = parts[0]
                    ticker2 = parts[1]
                    
                    # Get metadata
                    metadata_path = os.path.join(backtest_path, file)
                    with open(metadata_path, 'r') as f:
                        import json
                        metadata = json.load(f)
                    
                    pairs.append({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'name': metadata.get('pair_name', f"{ticker1}-{ticker2} Pair"),
                        'sharpe_ratio': metadata.get('sharpe_ratio', 0),
                        'win_rate': metadata.get('win_rate', 0)
                    })
            
            # Sort by Sharpe ratio
            pairs.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error retrieving available pairs: {str(e)}")
            return []

    def calculate_drawdown(self, returns_series: pd.Series) -> pd.Series:
        """
        Calculate drawdown for a returns series.
        
        Args:
            returns_series: Series of returns
            
        Returns:
            Series of drawdowns
        """
        # Calculate cumulative returns
        cumulative = (1 + returns_series).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative / running_max) - 1
        
        return drawdown

    def get_performance_metrics(self, returns_series: pd.Series) -> Dict:
        """
        Calculate performance metrics for a returns series.
        
        Args:
            returns_series: Series of returns
            
        Returns:
            Dictionary of performance metrics
        """
        if returns_series.empty:
            return {}
            
        try:
            # Calculate basic metrics
            total_return = (1 + returns_series).prod() - 1
            ann_factor = np.sqrt(252)  # Assuming daily returns
            
            # Annualized return
            ann_return = (1 + total_return) ** (ann_factor / len(returns_series)) - 1
            
            # Volatility
            volatility = returns_series.std() * ann_factor
            
            # Sharpe ratio
            sharpe = ann_return / volatility if volatility > 0 else 0
            
            # Drawdown
            drawdown = self.calculate_drawdown(returns_series)
            max_drawdown = drawdown.min()
            
            # Win rate
            win_rate = (returns_series > 0).mean()
            
            return {
                'total_return': total_return,
                'annualized_return': ann_return,
                'annualized_volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}