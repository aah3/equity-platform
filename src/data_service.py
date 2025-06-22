# src/data_service.py
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta

class DataService:
    """
    Service class that handles business logic between the database and the application.
    This class provides high-level methods for data retrieval and processing.
    """
    
    def __init__(self, db_manager):
        """
        Initialize the data service with a database manager.
        
        Args:
            db_manager: Instance of DatabaseManager2
        """
        self.db_manager = db_manager
        
    def get_available_indices(self):
        """
        Get list of available indices from the database.
        
        Returns:
            List of dictionaries containing index metadata
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT index_id, name, description, region FROM indices")
        indices = [{'index_id': row[0], 'name': row[1], 'description': row[2], 'region': row[3]} 
                  for row in cursor.fetchall()]
        conn.close()
        return indices
    
    def get_available_factors(self):
        """
        Get list of available factors from the database.
        
        Returns:
            List of factor IDs
        """
        return self.db_manager.get_available_factors()
    
    def get_index_constituents(self, index_id, date=None):
        """
        Get index constituents for a specific date.
        
        Args:
            index_id: ID of the index
            date: Date string (format: YYYY-MM-DD) or None for latest date
            
        Returns:
            DataFrame with constituent data
        """
        # Get constituents data
        constituents_data = self.db_manager.get_index_constituents(index_id)
        
        if constituents_data.empty:
            return pd.DataFrame()
        
        # Reset index to get date as a column
        constituents_data = constituents_data.reset_index()
        
        # If date is specified, filter to that date
        if date:
            dates = constituents_data['date'].unique()
            # Find closest date
            closest_date = min(dates, key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(date)))
            constituents_data = constituents_data[constituents_data['date'] == closest_date]
        else:
            # Use most recent date
            latest_date = constituents_data['date'].max()
            constituents_data = constituents_data[constituents_data['date'] == latest_date]
        
        # Get security names and sectors
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        securities = {}
        for ticker in constituents_data['ticker'].unique():
            cursor.execute("SELECT name, sector FROM securities WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            if row:
                securities[ticker] = {'name': row[0], 'sector': row[1]}
        
        conn.close()
        
        # Add security information
        constituents_data['name'] = constituents_data['ticker'].map(lambda t: securities.get(t, {}).get('name', t))
        constituents_data['sector'] = constituents_data['ticker'].map(lambda t: securities.get(t, {}).get('sector', 'Unknown'))
        
        return constituents_data
    
    def get_factor_data(self, factor_id, tickers=None, start_date=None, end_date=None):
        """
        Get factor data for selected securities and date range.
        
        Args:
            factor_id: ID of the factor
            tickers: List of tickers to filter by or None for all
            start_date: Start date string (format: YYYY-MM-DD) or None
            end_date: End date string (format: YYYY-MM-DD) or None
            
        Returns:
            DataFrame with factor data
        """
        # Get factor data from database
        factor_data = self.db_manager.get_factor_data(factor_id)
        
        if factor_data.empty:
            return pd.DataFrame()
        
        # Reset index to get column-based DataFrame
        factor_data = factor_data.reset_index()
        
        # Filter by tickers if provided
        if tickers:
            factor_data = factor_data[factor_data['ticker'].isin(tickers)]
        
        # Filter by date range
        if start_date:
            factor_data = factor_data[factor_data['date'] >= start_date]
        if end_date:
            factor_data = factor_data[factor_data['date'] <= end_date]
        
        return factor_data
    
    def get_factor_returns(self, factor_id, n_buckets=5, start_date=None, end_date=None):
        """
        Get factor returns for different buckets.
        
        Args:
            factor_id: ID of the factor
            n_buckets: Number of buckets
            start_date: Start date string (format: YYYY-MM-DD) or None
            end_date: End date string (format: YYYY-MM-DD) or None
            
        Returns:
            DataFrame with factor returns
        """
        # Get factor returns from file
        factor_returns_file = os.path.join(self.db_manager.ts_data_path, 'factor_returns', f'{factor_id}_returns.parquet')
        
        if not os.path.exists(factor_returns_file):
            return pd.DataFrame()
        
        factor_returns = pd.read_parquet(factor_returns_file)
        
        # Filter by date range
        if start_date:
            factor_returns = factor_returns[factor_returns['date'] >= start_date]
        if end_date:
            factor_returns = factor_returns[factor_returns['date'] <= end_date]
        
        return factor_returns
    
    def get_factor_correlation_matrix(self, start_date=None, end_date=None):
        """
        Get correlation matrix between long-short factor returns.
        
        Args:
            start_date: Start date string (format: YYYY-MM-DD) or None
            end_date: End date string (format: YYYY-MM-DD) or None
            
        Returns:
            DataFrame with correlation matrix
        """
        factors = self.get_available_factors()
        
        # Get all factor long-short returns
        factor_returns_data = {}
        
        for factor in factors:
            returns = self.get_factor_returns(factor, start_date=start_date, end_date=end_date)
            
            if not returns.empty:
                # Get long-short returns
                ls_returns = returns[returns['bucket'] == 'long_short']
                
                if not ls_returns.empty:
                    factor_returns_data[factor] = ls_returns.set_index('date')['return']
        
        if not factor_returns_data:
            return pd.DataFrame()
        
        # Create DataFrame with all returns
        returns_df = pd.DataFrame(factor_returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    def get_factor_performance_metrics(self, factor_id, n_buckets=5):
        """
        Get performance metrics for factor buckets.
        
        Args:
            factor_id: ID of the factor
            n_buckets: Number of buckets
            
        Returns:
            DataFrame with performance metrics
        """
        # Get factor returns
        factor_returns = self.get_factor_returns(factor_id)
        
        if factor_returns.empty:
            return pd.DataFrame()
        
        # Calculate metrics per bucket
        metrics = []
        
        # Get unique buckets
        buckets = sorted(factor_returns['bucket'].unique(), 
                        key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        for bucket in buckets:
            bucket_returns = factor_returns[factor_returns['bucket'] == bucket]
            
            if not bucket_returns.empty:
                returns = bucket_returns['return']
                
                # Calculate metrics
                annualized_return = returns.mean() * 252
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Calculate max drawdown
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (running_max - cum_returns) / running_max
                max_drawdown = drawdown.max()
                
                # Win rate
                win_rate = (returns > 0).sum() / len(returns)
                
                metrics.append({
                    'bucket': bucket,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate
                })
        
        return pd.DataFrame(metrics)
    
    def get_index_returns(self, index_id, start_date=None, end_date=None):
        """
        Get returns for an index.
        
        Args:
            index_id: ID of the index
            start_date: Start date string (format: YYYY-MM-DD) or None
            end_date: End date string (format: YYYY-MM-DD) or None
            
        Returns:
            DataFrame with index returns
        """
        # Get index data
        index_data = self.db_manager.get_index_data(index_id)
        
        if index_data.empty:
            return pd.DataFrame()
        
        # Reset index to get date as a column
        index_data = index_data.reset_index()
        
        # Filter by date range
        if start_date:
            index_data = index_data[index_data['date'] >= start_date]
        if end_date:
            index_data = index_data[index_data['date'] <= end_date]
        
        # Select returns
        returns_data = index_data[['date', 'return']]
        
        return returns_data
    
    def get_security_returns(self, tickers, start_date=None, end_date=None, frequency='daily'):
        """
        Get returns for a list of securities.
        
        Args:
            tickers: List of tickers
            start_date: Start date string (format: YYYY-MM-DD) or None
            end_date: End date string (format: YYYY-MM-DD) or None
            frequency: 'daily' or 'monthly'
            
        Returns:
            DataFrame with security returns
        """
        # Get price data for tickers
        price_data = self.db_manager.get_price_data(tickers, start_date, end_date)
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Reset index to get date and ticker as columns
        price_data = price_data.reset_index()
        
        # If frequency is not daily, resample
        if frequency == 'monthly':
            # Add year and month columns
            price_data['year'] = pd.to_datetime(price_data['date']).dt.year
            price_data['month'] = pd.to_datetime(price_data['date']).dt.month
            
            # Group by year, month, and ticker and get last close of each month
            monthly_data = price_data.groupby(['year', 'month', 'ticker']).last().reset_index()
            
            # Create proper date from year and month (first day of next month)
            monthly_data['date'] = pd.to_datetime(monthly_data['year'].astype(str) + '-' + 
                                                monthly_data['month'].astype(str) + '-01') + pd.DateOffset(months=1)
            
            # Calculate returns
            monthly_returns = []
            
            for ticker in tickers:
                ticker_data = monthly_data[monthly_data['ticker'] == ticker].sort_values('date')
                ticker_data['return'] = ticker_data['close'].pct_change()
                monthly_returns.append(ticker_data[['date', 'ticker', 'return']])
            
            return pd.concat(monthly_returns)
        else:
            # Return daily returns
            return price_data[['date', 'ticker', 'return']]
    
    def get_available_strategies(self, index_id):
        """
        Get available backtest strategies for an index.
        
        Args:
            index_id: ID of the index
            
        Returns:
            List of dictionaries containing strategy metadata
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT factor, name, description FROM backtest_strategies
        WHERE index_id = ?
        """, (index_id,))
        
        strategies = [{'factor': row[0], 'name': row[1], 'description': row[2]} 
                    for row in cursor.fetchall()]
        
        conn.close()
        return strategies
    
    def get_backtest_results(self, index_id, factor):
        """
        Get backtest results for a strategy.
        
        Args:
            index_id: ID of the index
            factor: ID of the factor strategy
            
        Returns:
            DataFrame with backtest results
        """
        # Get backtest data from file
        backtest_file = os.path.join(self.db_manager.ts_data_path, 'backtest_results', 
                                    f'{index_id}_{factor}_backtest.parquet')
        
        if not os.path.exists(backtest_file):
            return pd.DataFrame()
        
        backtest_data = pd.read_parquet(backtest_file)
        
        return backtest_data
    
    def get_backtest_metadata(self, index_id, factor):
        """
        Get metadata for a backtest strategy.
        
        Args:
            index_id: ID of the index
            factor: ID of the factor strategy
            
        Returns:
            Dictionary with strategy metadata
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM backtest_strategies
        WHERE index_id = ? AND factor = ?
        """, (index_id, factor))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {}
        
        # Get column names
        cursor.execute("PRAGMA table_info(backtest_strategies)")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        
        return {columns[i]: row[i] for i in range(len(columns))}
    
    def get_available_pairs(self):
        """
        Get available pairs trading strategies.
        
        Returns:
            List of dictionaries containing pair metadata
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT ticker1, ticker2, name, sector FROM pairs
        """)
        
        pairs = [{'ticker1': row[0], 'ticker2': row[1], 'name': row[2], 'sector': row[3]} 
                for row in cursor.fetchall()]
        
        conn.close()
        return pairs
    
    def get_pair_trading_results(self, ticker1, ticker2):
        """
        Get pairs trading results.
        
        Args:
            ticker1: First ticker in the pair
            ticker2: Second ticker in the pair
            
        Returns:
            DataFrame with pair trading results
        """
        # Get pair data from file
        pair_file = os.path.join(self.db_manager.ts_data_path, 'pairs', 
                               f'{ticker1}_{ticker2}_pairs.parquet')
        
        if not os.path.exists(pair_file):
            # Try reverse order
            pair_file = os.path.join(self.db_manager.ts_data_path, 'pairs', 
                                   f'{ticker2}_{ticker1}_pairs.parquet')
            
            if not os.path.exists(pair_file):
                return pd.DataFrame()
        
        pair_data = pd.read_parquet(pair_file)
        
        return pair_data
    
    def get_pair_trading_metadata(self, ticker1, ticker2):
        """
        Get metadata for a pairs trading strategy.
        
        Args:
            ticker1: First ticker in the pair
            ticker2: Second ticker in the pair
            
        Returns:
            Dictionary with pair metadata
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM pairs
        WHERE (ticker1 = ? AND ticker2 = ?) OR (ticker1 = ? AND ticker2 = ?)
        """, (ticker1, ticker2, ticker2, ticker1))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {}
        
        # Get column names
        cursor.execute("PRAGMA table_info(pairs)")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        
        return {columns[i]: row[i] for i in range(len(columns))}
    
    def generate_demo_data(self):
        """
        Generate demo data for the platform.
        
        Returns:
            Boolean indicating success or failure
        """
        return self.db_manager.generate_demo_data()
    
if __name__=="__main__":
    from database_sqlite_2 import DatabaseManager2
    db_manager = DatabaseManager2()
    data_service = DataService(db_manager)

    selected_index = 'SPX'
    selected_strategy = 'momentum'
    factor_id = 'size'
    n_buckets = 5
    
    # success = data_service.generate_demo_data()
    indices = data_service.get_available_indices()
    available_factors = data_service.get_available_factors()
    constituents = data_service.get_index_constituents(selected_index)
    tickers = constituents['ticker'].tolist()
    metrics = data_service.get_factor_performance_metrics(factor_id, n_buckets)
    strategies = data_service.get_available_strategies(selected_index)

    backtest_metadata = data_service.get_backtest_metadata(selected_index, selected_strategy)
    strategy_return = backtest_metadata.get('annualized_return', 0.)
    benchmark_return = backtest_metadata.get('benchmark_return', 0.)
    return_diff = strategy_return - benchmark_return

    factor_data = {}
    factor_returns = {}
            
    # Get factor data
    factor_data[selected_strategy] = data_service.get_factor_data(
        factor_id, tickers)#, start_date_str, end_date_str)

    factor_metrics = data_service.get_factor_performance_metrics(selected_strategy, n_buckets)