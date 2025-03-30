# src/database_sqlite.py
import os
import sqlite3
import pandas as pd
import numpy as np
import datetime
import itertools
from typing import List, Dict, Optional, Union, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for the Equity Trading Platform.
    Handles storage and retrieval of market data, index constituents, and other financial information.
    
    Uses SQLite for metadata and Parquet files for time series data.
    """
    
    def __init__(self, db_path: str = 'data/market_data.db', 
                 ts_data_path: str = 'data/time_series'):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
            ts_data_path: Path to store time series data as Parquet files
        """
        self.db_path = db_path
        self.ts_data_path = ts_data_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(ts_data_path, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS indices (
            index_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            region TEXT,
            last_updated TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS securities (
            ticker TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT,
            industry TEXT,
            description TEXT,
            first_date TEXT,
            last_date TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS index_constituents (
            index_id TEXT,
            ticker TEXT,
            weight REAL,
            entry_date TEXT,
            exit_date TEXT,
            PRIMARY KEY (index_id, ticker),
            FOREIGN KEY (index_id) REFERENCES indices(index_id),
            FOREIGN KEY (ticker) REFERENCES securities(ticker)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS factors (
            factor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            last_updated TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS factor_exposures (
            factor_id TEXT,
            ticker TEXT,
            date TEXT,
            value REAL,
            PRIMARY KEY (factor_id, ticker, date),
            FOREIGN KEY (factor_id) REFERENCES factors(factor_id),
            FOREIGN KEY (ticker) REFERENCES securities(ticker)
        )
        ''')
        
        connection.commit()
        connection.close()
        
        logger.info("Database initialized successfully")
    
    def add_index(self, index_id: str, name: str, description: str = None, 
                  region: str = None) -> bool:
        """
        Add a new index to the database.
        
        Args:
            index_id: Unique identifier for the index (e.g., 'SPX', 'NDX')
            name: Full name of the index
            description: Description of the index
            region: Geographic region of the index
            
        Returns:
            Success status
        """
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO indices (index_id, name, description, region, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ''', (index_id, name, description, region, datetime.datetime.now()))
            
            connection.commit()
            connection.close()
            
            logger.info(f"Added index: {index_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding index {index_id}: {str(e)}")
            return False
    
    def add_security(self, ticker: str, name: str, sector: str = None, 
                    industry: str = None, description: str = None) -> bool:
        """
        Add a new security to the database.
        
        Args:
            ticker: Ticker symbol
            name: Company name
            sector: Business sector
            industry: Industry classification
            description: Company description
            
        Returns:
            Success status
        """
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO securities (ticker, name, sector, industry, description)
            VALUES (?, ?, ?, ?, ?)
            ''', (ticker, name, sector, industry, description))
            
            connection.commit()
            connection.close()
            
            logger.info(f"Added security: {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding security {ticker}: {str(e)}")
            return False
    
    def update_index_constituents(self, index_id: str, constituents_df: pd.DataFrame) -> bool:
        """
        Update the constituents of an index.
        
        Args:
            index_id: Index identifier
            constituents_df: DataFrame with columns [ticker, weight, entry_date, exit_date]
            
        Returns:
            Success status
        """
        try:
            connection = sqlite3.connect(self.db_path)
            
            # First, ensure all securities exist in the securities table
            for ticker in constituents_df['ticker'].unique():
                cursor = connection.cursor()
                cursor.execute('SELECT ticker FROM securities WHERE ticker = ?', (ticker,))
                if not cursor.fetchone():
                    # Add the security with minimal info if it doesn't exist
                    cursor.execute('''
                    INSERT INTO securities (ticker, name) VALUES (?, ?)
                    ''', (ticker, ticker))
            
            # Then update the constituents
            constituents_df['index_id'] = index_id
            constituents_df.to_sql('index_constituents', connection, 
                                   if_exists='replace', index=False)
            
            connection.commit()
            connection.close()
            
            logger.info(f"Updated constituents for index: {index_id}, {len(constituents_df)} securities")
            return True
            
        except Exception as e:
            logger.error(f"Error updating constituents for {index_id}: {str(e)}")
            return False
    
    def get_indices(self) -> pd.DataFrame:
        """
        Get all available indices.
        
        Returns:
            DataFrame with index information
        """
        try:
            connection = sqlite3.connect(self.db_path)
            query = "SELECT * FROM indices"
            indices_df = pd.read_sql_query(query, connection)
            connection.close()
            return indices_df
            
        except Exception as e:
            logger.error(f"Error retrieving indices: {str(e)}")
            return pd.DataFrame()
    
    def get_index_constituents(self, index_id: str, as_of_date: str = None) -> pd.DataFrame:
        """
        Get constituents of a specific index, optionally as of a specific date.
        
        Args:
            index_id: Index identifier
            as_of_date: Optional date to get constituents as of (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with constituent information
        """
        try:
            connection = sqlite3.connect(self.db_path)
            
            if as_of_date:
                query = """
                SELECT ic.*, s.name, s.sector, s.industry
                FROM index_constituents ic
                JOIN securities s ON ic.ticker = s.ticker
                WHERE ic.index_id = ?
                AND (ic.entry_date <= ? OR ic.entry_date IS NULL)
                AND (ic.exit_date >= ? OR ic.exit_date IS NULL)
                """
                constituents_df = pd.read_sql_query(query, connection, params=(index_id, as_of_date, as_of_date))
            else:
                query = """
                SELECT ic.*, s.name, s.sector, s.industry
                FROM index_constituents ic
                JOIN securities s ON ic.ticker = s.ticker
                WHERE ic.index_id = ?
                """
                constituents_df = pd.read_sql_query(query, connection, params=(index_id,))
                
            connection.close()
            return constituents_df
            
        except Exception as e:
            logger.error(f"Error retrieving constituents for {index_id}: {str(e)}")
            return pd.DataFrame()
    
    def store_price_data(self, ticker: str, price_data: pd.DataFrame) -> bool:
        """
        Store price data for a security.
        
        Args:
            ticker: Security ticker
            price_data: DataFrame with price data (must have 'date' as index or column)
            
        Returns:
            Success status
        """
        try:
            # Ensure the directory exists
            ticker_path = os.path.join(self.ts_data_path, 'prices')
            os.makedirs(ticker_path, exist_ok=True)
            
            # Ensure date is in the index
            if 'date' in price_data.columns:
                price_data = price_data.set_index('date')
            
            # Save to parquet file
            file_path = os.path.join(ticker_path, f"{ticker}.parquet")
            price_data.to_parquet(file_path)
            
            # Update security metadata
            if not price_data.empty:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                
                cursor.execute('''
                UPDATE securities 
                SET first_date = ?, last_date = ?
                WHERE ticker = ?
                ''', (str(price_data.index.min().date()), 
                      str(price_data.index.max().date()), 
                      ticker))
                
                connection.commit()
                connection.close()
            
            logger.info(f"Stored price data for {ticker}, {len(price_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data for {ticker}: {str(e)}")
            return False
    
    def get_price_data(self, ticker: str, start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """
        Retrieve price data for a security.
        
        Args:
            ticker: Security ticker
            start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
            end_date: End date for data retrieval (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with price data
        """
        try:
            file_path = os.path.join(self.ts_data_path, 'prices', f"{ticker}.parquet")
            
            if not os.path.exists(file_path):
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
                
            price_data = pd.read_parquet(file_path)
            
            # Filter by date if provided
            # if 'date' not in price_data.index.names:
            #     price_data = price_data.set_index('date')
                
            if start_date:
                price_data = price_data[price_data.index >= start_date]
            if end_date:
                price_data = price_data[price_data.index <= end_date]
                
            return price_data
            
        except Exception as e:
            logger.error(f"Error retrieving price data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def store_factor_data(self, factor_id: str, factor_data: pd.DataFrame) -> bool:
        """
        Store factor data.
        
        Args:
            factor_id: Factor identifier
            factor_data: DataFrame with factor data (columns: ticker, date, value)
            
        Returns:
            Success status
        """
        try:
            # Save metadata to SQLite
            connection = sqlite3.connect(self.db_path)
            cursor = connection.cursor()
            
            # Check if factor exists, add if not
            cursor.execute('SELECT factor_id FROM factors WHERE factor_id = ?', (factor_id,))
            if not cursor.fetchone():
                cursor.execute('''
                INSERT INTO factors (factor_id, name, last_updated)
                VALUES (?, ?, ?)
                ''', (factor_id, factor_id, datetime.datetime.now()))
            
            # Save factor exposures
            factor_exposures = factor_data[['ticker', 'date', 'value']].copy()
            factor_exposures['factor_id'] = factor_id
            
            factor_exposures.to_sql('factor_exposures', connection, 
                                   if_exists='replace', index=False)
            
            connection.commit()
            
            # Also save as parquet for efficient time series access
            factor_path = os.path.join(self.ts_data_path, 'factors')
            os.makedirs(factor_path, exist_ok=True)
            
            file_path = os.path.join(factor_path, f"{factor_id}.parquet")
            factor_data.to_parquet(file_path)
            
            connection.close()
            
            logger.info(f"Stored factor data for {factor_id}, {len(factor_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing factor data for {factor_id}: {str(e)}")
            return False
    
    def get_factor_data(self, factor_id: str, tickers: List[str] = None,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve factor data.
        
        Args:
            factor_id: Factor identifier
            tickers: Optional list of tickers to filter by
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with factor data
        """
        try:
            file_path = os.path.join(self.ts_data_path, 'factors', f"{factor_id}.parquet")
            
            if not os.path.exists(file_path):
                logger.warning(f"No factor data found for {factor_id}")
                return pd.DataFrame()
                
            factor_data = pd.read_parquet(file_path)
            
            # Apply filters
            if tickers:
                factor_data = factor_data[factor_data['ticker'].isin(tickers)]
            if start_date:
                factor_data = factor_data[factor_data['date'] >= start_date]
            if end_date:
                factor_data = factor_data[factor_data['date'] <= end_date]
                
            return factor_data
            
        except Exception as e:
            logger.error(f"Error retrieving factor data for {factor_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_available_factors(self) -> List[str]:
        """
        Get list of available factors in the database.
        
        Returns:
            List of factor IDs
        """
        try:
            connection = sqlite3.connect(self.db_path)
            cursor = connection.cursor()
            
            cursor.execute('SELECT factor_id FROM factors')
            factors = [row[0] for row in cursor.fetchall()]
            
            connection.close()
            return factors
            
        except Exception as e:
            logger.error(f"Error retrieving factors: {str(e)}")
            return []

    def store_returns_data(self, returns_df: pd.DataFrame, source: str = 'daily') -> bool:
        """
        Store returns data for multiple securities.
        
        Args:
            returns_df: DataFrame with returns data (columns: date, ticker, return)
            source: Data source identifier (e.g., 'daily', 'monthly')
            
        Returns:
            Success status
        """
        try:
            # Ensure the directory exists
            returns_path = os.path.join(self.ts_data_path, 'returns')
            os.makedirs(returns_path, exist_ok=True)
            
            # Save to parquet file
            file_path = os.path.join(returns_path, f"{source}_returns.parquet")
            returns_df.to_parquet(file_path)
            
            logger.info(f"Stored {source} returns data, {len(returns_df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing returns data: {str(e)}")
            return False
    
    def get_returns_data(self, tickers: List[str] = None, start_date: str = None,
                       end_date: str = None, source: str = 'daily') -> pd.DataFrame:
        """
        Retrieve returns data.
        
        Args:
            tickers: Optional list of tickers to filter by
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            source: Data source identifier
            
        Returns:
            DataFrame with returns data
        """
        try:
            file_path = os.path.join(self.ts_data_path, 'returns', f"{source}_returns.parquet")
            
            if not os.path.exists(file_path):
                logger.warning(f"No {source} returns data found")
                return pd.DataFrame()
                
            returns_df = pd.read_parquet(file_path)
            
            # Apply filters
            if tickers:
                returns_df = returns_df[returns_df['ticker'].isin(tickers)]
            if start_date:
                returns_df = returns_df[returns_df['date'] >= start_date]
            if end_date:
                returns_df = returns_df[returns_df['date'] <= end_date]
                
            return returns_df
            
        except Exception as e:
            logger.error(f"Error retrieving returns data: {str(e)}")
            return pd.DataFrame()
    
    def reset_database(self) -> bool:
        """
        Reset the entire database (use with caution!)
        
        Returns:
            Success status
        """
        try:
            # Close any existing connections
            try:
                connection = sqlite3.connect(self.db_path)
                connection.close()
            except:
                pass
            
            # Remove SQLite file if it exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            # Recreate database structure
            self._init_database()
            
            logger.warning("Database has been reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False

    def generate_demo_data(self) -> bool:
        """
        Generate demo data for testing and demonstration purposes.
        
        Returns:
            Success status
        """
        try:
            # Add indices
            self.add_index('SPX', 'S&P 500 Index', 'Large-cap US equities', 'US')
            self.add_index('NDX', 'NASDAQ-100 Index', 'Large-cap US tech equities', 'US')
            self.add_index('RTY', 'Russell 2000 Index', 'Small-cap US equities', 'US')
            
            # Generate some sample securities
            tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
            finance_tickers = ['JPM', 'BAC', 'GS', 'WFC', 'C', 'MS']
            healthcare_tickers = ['JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 'ABBV']
            
            # Add securities
            for ticker in tech_tickers:
                self.add_security(ticker, f"{ticker} Inc.", "Technology", "Technology")
            
            for ticker in finance_tickers:
                self.add_security(ticker, f"{ticker} Corp.", "Financials", "Banking")
            
            for ticker in healthcare_tickers:
                self.add_security(ticker, f"{ticker} Corp.", "Healthcare", "Pharmaceuticals")
            
            # Generate constituents for indices
            spx_constituents = tech_tickers + finance_tickers + healthcare_tickers
            ndx_constituents = tech_tickers
            rty_constituents = ['SML1', 'SML2', 'SML3', 'SML4', 'SML5']
            
            # Add small caps for Russell
            for ticker in rty_constituents:
                self.add_security(ticker, f"Small Cap {ticker}", "Various", "Small Cap")
            
            # Create weights
            spx_weights = np.random.uniform(0.1, 5, len(spx_constituents))
            spx_weights = spx_weights / np.sum(spx_weights) * 100
            
            ndx_weights = np.random.uniform(0.5, 10, len(ndx_constituents))
            ndx_weights = ndx_weights / np.sum(ndx_weights) * 100
            
            rty_weights = np.random.uniform(0.05, 0.5, len(rty_constituents))
            rty_weights = rty_weights / np.sum(rty_weights) * 100
            
            # Create constituent DataFrames
            spx_df = pd.DataFrame({
                'ticker': spx_constituents,
                'weight': spx_weights,
                'entry_date': '2020-01-01',
                'exit_date': None
            })
            
            ndx_df = pd.DataFrame({
                'ticker': ndx_constituents,
                'weight': ndx_weights,
                'entry_date': '2020-01-01',
                'exit_date': None
            })
            
            rty_df = pd.DataFrame({
                'ticker': rty_constituents,
                'weight': rty_weights,
                'entry_date': '2020-01-01',
                'exit_date': None
            })
            
            # Update constituents
            self.update_index_constituents('SPX', spx_df)
            self.update_index_constituents('NDX', ndx_df)
            self.update_index_constituents('RTY', rty_df)
            
            # Generate price data
            start_date = datetime.datetime(2020, 1, 1)
            end_date = datetime.datetime.now()
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            
            all_tickers = list(set(spx_constituents + ndx_constituents + rty_constituents))
            
            for ticker in all_tickers:
                # Generate random price path with trends and volatility
                n_days = len(dates)
                
                # Start with a random price between $10 and $500
                start_price = np.random.uniform(10, 500)
                
                # Generate daily returns with some autocorrelation
                daily_returns = np.random.normal(0.0005, 0.015, size=n_days)
                
                # Add autocorrelation
                for i in range(1, n_days):
                    daily_returns[i] = 0.7 * daily_returns[i] + 0.3 * daily_returns[i-1]
                
                # Convert returns to prices
                prices = start_price * np.cumprod(1 + daily_returns)
                
                # Create price DataFrame
                price_df = pd.DataFrame({
                    'open': prices * (1 - np.random.uniform(0, 0.01, size=n_days)),
                    'high': prices * (1 + np.random.uniform(0, 0.02, size=n_days)),
                    'low': prices * (1 - np.random.uniform(0, 0.02, size=n_days)),
                    'close': prices,
                    'volume': np.random.randint(100000, 10000000, size=n_days)
                }, index=dates)
                
                # Store the price data
                self.store_price_data(ticker, price_df)
            
            # Create and store returns data
            returns_data = []
            
            for ticker in all_tickers:
                price_data = self.get_price_data(ticker)
                if not price_data.empty:
                    # Calculate returns
                    returns = price_data['close'].pct_change().dropna()
                    
                    # Create returns DataFrame
                    for date, ret in returns.items():
                        returns_data.append({
                            'date': date,
                            'ticker': ticker,
                            'return': ret
                        })
            
            # Store returns
            returns_df = pd.DataFrame(returns_data)
            self.store_returns_data(returns_df, 'daily')
            
            # Generate monthly returns for convenience
            monthly_returns = []
            for ticker in all_tickers:
                price_data = self.get_price_data(ticker)
                if not price_data.empty:
                    # Resample to monthly and calculate returns
                    monthly_prices = price_data['close'].resample('M').last()
                    monthly_rets = monthly_prices.pct_change().dropna()
                    
                    for date, ret in monthly_rets.items():
                        monthly_returns.append({
                            'date': date,
                            'ticker': ticker,
                            'return': ret
                        })
            
            # Store monthly returns
            monthly_df = pd.DataFrame(monthly_returns)
            self.store_returns_data(monthly_df, 'monthly')
            
            # Generate factor data
            factors = ['beta', 'size', 'value', 'momentum', 'quality']
            n_buckets = 5
            
            for factor in factors:
                factor_data = []
                
                for ticker in all_tickers:
                    # Get monthly dates
                    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
                    
                    # Generate factor values (with persistence across time)
                    base_value = np.random.normal(0, 1)
                    
                    for date in monthly_dates:
                        # Add some randomness but maintain persistence
                        factor_value = base_value + np.random.normal(0, 0.2)
                        
                        # Drift the base value slightly
                        base_value = 0.9 * base_value + 0.1 * np.random.normal(0, 1)
                        
                        factor_data.append({
                            'ticker': ticker,
                            'date': date,
                            'value': factor_value
                        })
                
                # Store factor data
                factor_df = pd.DataFrame(factor_data)
                self.store_factor_data(factor, factor_df)
                
                # Also generate and store pre-bucketed factor returns
                # to avoid the KeyError: 'bucket' issue
                all_monthly_dates = sorted(factor_df['date'].unique())
                bucketed_returns = []
                
                for i, date in enumerate(all_monthly_dates[:-1]):
                    next_date = all_monthly_dates[i+1]
                    
                    # Get factor values for this date
                    date_factors = factor_df[factor_df['date'] == date]
                    
                    # Create buckets
                    date_factors['bucket'] = pd.qcut(date_factors['value'], n_buckets, labels=range(1, n_buckets+1), duplicates='drop')
                    
                    # For each bucket, calculate an average return
                    for bucket in range(1, n_buckets+1):
                        bucket_tickers = date_factors[date_factors['bucket'] == bucket]['ticker'].tolist()
                        if bucket_tickers:
                            # Get returns for these tickers
                            ticker_returns = monthly_df[
                                (monthly_df['date'] > date) & 
                                (monthly_df['date'] <= next_date) & 
                                (monthly_df['ticker'].isin(bucket_tickers))
                            ]
                            
                            if not ticker_returns.empty:
                                avg_return = ticker_returns['return'].mean()
                                
                                # Add to bucketed returns
                                bucketed_returns.append({
                                    'date': date,
                                    'bucket': bucket,
                                    'return': avg_return
                                })
                    
                    # Add long-short portfolio
                    high_bucket_return = next(
                        (item['return'] for item in bucketed_returns if item['date'] == date and item['bucket'] == n_buckets),
                        None
                    )
                    low_bucket_return = next(
                        (item['return'] for item in bucketed_returns if item['date'] == date and item['bucket'] == 1),
                        None
                    )
                    
                    if high_bucket_return is not None and low_bucket_return is not None:
                        bucketed_returns.append({
                            'date': date,
                            'bucket': 'long_short',
                            'return': high_bucket_return - low_bucket_return
                        })
                
                # Store pre-bucketed returns
                if bucketed_returns:
                    bucketed_df = pd.DataFrame(bucketed_returns)
                    bucketed_path = os.path.join(self.ts_data_path, 'factor_returns')
                    os.makedirs(bucketed_path, exist_ok=True)
                    file_path = os.path.join(bucketed_path, f"{factor}_returns.parquet")
                    bucketed_df['bucket'] = bucketed_df['bucket'].astype(str)
                    bucketed_df.to_parquet(file_path)
            
            # Generate backtest strategy data for Tab 3
            # Create directory for storing backtest results
            backtest_path = os.path.join(self.ts_data_path, 'backtest_results')
            os.makedirs(backtest_path, exist_ok=True)
            
            # Generate backtest data for different strategies
            strategy_types = [
                {'name': 'Value Strategy', 'factor': 'value', 'alpha': 0.0007, 'beta': 0.85},
                {'name': 'Momentum Strategy', 'factor': 'momentum', 'alpha': 0.0005, 'beta': 1.1},
                {'name': 'Quality Strategy', 'factor': 'quality', 'alpha': 0.0004, 'beta': 0.9}
            ]
            
            for index_id in ['SPX', 'NDX', 'RTY']:
                # Get all trading days for the backtest
                trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
                
                # Create benchmark returns with some autocorrelation
                benchmark_returns = np.random.normal(0.0005, 0.012, size=len(trading_days))
                for i in range(1, len(benchmark_returns)):
                    benchmark_returns[i] = 0.8 * benchmark_returns[i] + 0.2 * benchmark_returns[i-1]
                
                # Calculate cumulative benchmark returns
                benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod()
                
                # Generate strategy returns for each strategy type
                for strategy in strategy_types:
                    # Create strategy-specific returns with some alpha and beta exposure to the benchmark
                    strategy_returns = np.zeros(len(trading_days))
                    
                    for i in range(len(trading_days)):
                        # Alpha component plus beta exposure to benchmark plus random noise
                        strategy_returns[i] = strategy['alpha'] + strategy['beta'] * benchmark_returns[i] + np.random.normal(0, 0.005)
                    
                    # Create some realistic drawdowns for the strategy
                    # Add a significant drawdown period
                    drawdown_start = int(len(trading_days) * 0.3)
                    drawdown_end = drawdown_start + 60  # 60 days drawdown
                    strategy_returns[drawdown_start:drawdown_end] -= 0.003  # Extra negative returns
                    
                    # Calculate cumulative strategy returns
                    strategy_cumulative = (1 + pd.Series(strategy_returns)).cumprod()
                    
                    # Calculate drawdowns
                    benchmark_peak = benchmark_cumulative.cummax()
                    benchmark_drawdown = (benchmark_cumulative / benchmark_peak) - 1
                    
                    strategy_peak = strategy_cumulative.cummax()
                    strategy_drawdown = (strategy_cumulative / strategy_peak) - 1
                    
                    # Calculate performance metrics
                    backtest_data = {
                        'date': trading_days,
                        'strategy_return': strategy_returns,
                        'benchmark_return': benchmark_returns,
                        'strategy_cumulative': strategy_cumulative.values,
                        'benchmark_cumulative': benchmark_cumulative.values,
                        'strategy_drawdown': strategy_drawdown.values,
                        'benchmark_drawdown': benchmark_drawdown.values
                    }
                    
                    # Calculate rolling metrics
                    returns_df = pd.DataFrame({
                        'date': trading_days,
                        'strategy_return': strategy_returns,
                        'benchmark_return': benchmark_returns
                    })
                    
                    # Calculate rolling beta (90-day window)
                    rolling_window = min(90, len(trading_days) // 2)
                    rolling_metrics = []
                    
                    for i in range(rolling_window, len(trading_days)):
                        window_data = returns_df.iloc[i-rolling_window:i]
                        cov = np.cov(window_data['strategy_return'], window_data['benchmark_return'])[0, 1]
                        var = np.var(window_data['benchmark_return'])
                        beta = cov / var if var > 0 else 1.0
                        
                        # Calculate rolling alpha
                        alpha = window_data['strategy_return'].mean() - beta * window_data['benchmark_return'].mean()
                        
                        # Calculate rolling Sharpe ratio
                        sharpe = (window_data['strategy_return'].mean() / window_data['strategy_return'].std()) * np.sqrt(252)
                        
                        rolling_metrics.append({
                            'date': trading_days[i],
                            'rolling_beta': beta,
                            'rolling_alpha': alpha * 252,  # Annualize alpha
                            'rolling_sharpe': sharpe
                        })
                    
                    rolling_metrics_df = pd.DataFrame(rolling_metrics)
                    
                    # Create combined DataFrame with all backtest data
                    backtest_df = pd.DataFrame(backtest_data)
                    
                    # Merge with rolling metrics
                    if not rolling_metrics_df.empty:
                        backtest_df = backtest_df.merge(rolling_metrics_df, on='date', how='left')
                    
                    # Store backtest results
                    file_path = os.path.join(backtest_path, f"{index_id}_{strategy['factor']}_strategy.parquet")
                    backtest_df.to_parquet(file_path)
                    
                    # Also store backtest metadata
                    metadata = {
                        'index_id': index_id,
                        'strategy_name': strategy['name'],
                        'factor': strategy['factor'],
                        'start_date': str(trading_days[0].date()),
                        'end_date': str(trading_days[-1].date()),
                        'total_days': len(trading_days),
                        'annualized_return': ((1 + np.mean(strategy_returns)) ** 252) - 1,
                        'annualized_volatility': np.std(strategy_returns) * np.sqrt(252),
                        'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252),
                        'max_drawdown': np.min(strategy_drawdown),
                        'benchmark_return': ((1 + np.mean(benchmark_returns)) ** 252) - 1,
                        'benchmark_volatility': np.std(benchmark_returns) * np.sqrt(252),
                        'benchmark_sharpe': np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252),
                        'benchmark_max_drawdown': np.min(benchmark_drawdown),
                        'tracking_error': np.std(np.array(strategy_returns) - np.array(benchmark_returns)) * np.sqrt(252),
                        'information_ratio': np.mean(np.array(strategy_returns) - np.array(benchmark_returns)) / 
                                        np.std(np.array(strategy_returns) - np.array(benchmark_returns)) * np.sqrt(252)
                    }
                    
                    # Store metadata
                    metadata_path = os.path.join(backtest_path, f"{index_id}_{strategy['factor']}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        import json
                        json.dump(metadata, f)
            
            # Generate pair trading backtest results
            pairs = [
                {'ticker1': 'AAPL', 'ticker2': 'MSFT', 'name': 'Apple-Microsoft Pair'},
                {'ticker1': 'JPM', 'ticker2': 'GS', 'name': 'JPMorgan-Goldman Pair'},
                {'ticker1': 'PFE', 'ticker2': 'MRK', 'name': 'Pfizer-Merck Pair'}
            ]
            
            for pair in pairs:
                if pair['ticker1'] in all_tickers and pair['ticker2'] in all_tickers:
                    # Get price data for both tickers
                    price1 = self.get_price_data(pair['ticker1'])
                    price2 = self.get_price_data(pair['ticker2'])
                    
                    if not price1.empty and not price2.empty:
                        # Align dates
                        common_dates = price1.index.intersection(price2.index)
                        price1 = price1.loc[common_dates]
                        price2 = price2.loc[common_dates]
                        
                        # Calculate spread and z-score
                        # Use a simple ratio spread for demonstration
                        price_ratio = price1['close'] / price2['close']
                        
                        # Calculate rolling mean and std for z-score
                        window = 20
                        rolling_mean = price_ratio.rolling(window=window).mean()
                        rolling_std = price_ratio.rolling(window=window).std()
                        
                        # Calculate z-score
                        z_score = (price_ratio - rolling_mean) / rolling_std
                        
                        # Generate pair trading signals
                        # Enter when z-score crosses threshold, exit when it reverts to mean
                        long_entry = z_score < -2.0
                        long_exit = z_score >= -0.5
                        short_entry = z_score > 2.0
                        short_exit = z_score <= 0.5
                        
                        # Initialize position and pnl
                        position = np.zeros(len(common_dates))
                        pair_returns = np.zeros(len(common_dates))
                        
                        # Simulate pair trading strategy
                        for i in range(window, len(common_dates)):
                            # Update position based on signals
                            if position[i-1] == 0:  # No position
                                if long_entry.iloc[i]:
                                    position[i] = 1  # Long the spread (long ticker1, short ticker2)
                                elif short_entry.iloc[i]:
                                    position[i] = -1  # Short the spread (short ticker1, long ticker2)
                                else:
                                    position[i] = 0  # No position
                            elif position[i-1] == 1:  # Long position
                                if long_exit.iloc[i]:
                                    position[i] = 0  # Exit long
                                else:
                                    position[i] = 1  # Maintain long
                            elif position[i-1] == -1:  # Short position
                                if short_exit.iloc[i]:
                                    position[i] = 0  # Exit short
                                else:
                                    position[i] = -1  # Maintain short
                            
                            # Calculate returns based on position
                            if position[i-1] != 0:
                                # Calculate daily returns for both tickers
                                ret1 = price1['close'].iloc[i] / price1['close'].iloc[i-1] - 1
                                ret2 = price2['close'].iloc[i] / price2['close'].iloc[i-1] - 1
                                
                                # Calculate pair return
                                if position[i-1] == 1:  # Long spread
                                    pair_returns[i] = ret1 - ret2
                                else:  # Short spread
                                    pair_returns[i] = ret2 - ret1
                        
                        # Calculate cumulative returns
                        cumulative_returns = (1 + pd.Series(pair_returns)).cumprod()
                        
                        # Create pair trading DataFrame
                        pair_data = pd.DataFrame({
                            'date': common_dates,
                            'ticker1_price': price1['close'].values,
                            'ticker2_price': price2['close'].values,
                            'price_ratio': price_ratio.values,
                            'z_score': z_score.values,
                            'position': position,
                            'pair_return': pair_returns,
                            'cumulative_return': cumulative_returns
                        })
                        
                        # Calculate drawdowns
                        pair_peak = pd.Series(cumulative_returns).cummax()
                        pair_drawdown = (pd.Series(cumulative_returns) / pair_peak) - 1
                        pair_data['drawdown'] = pair_drawdown.values
                        
                        # Calculate performance metrics
                        non_zero_returns = pair_returns[position != 0]
                        if len(non_zero_returns) > 0:
                            sharpe_ratio = np.mean(non_zero_returns) / np.std(non_zero_returns) * np.sqrt(252)
                            max_drawdown = np.min(pair_drawdown)
                            win_rate = np.sum(non_zero_returns > 0) / len(non_zero_returns) if len(non_zero_returns) > 0 else 0
                            
                            # Store pair trading results
                            file_path = os.path.join(backtest_path, f"{pair['ticker1']}_{pair['ticker2']}_pair.parquet")
                            pair_data.to_parquet(file_path)
                            
                            # Store pair metadata
                            pair_metadata = {
                                'pair_name': pair['name'],
                                'ticker1': pair['ticker1'],
                                'ticker2': pair['ticker2'],
                                'start_date': str(common_dates[0].date()),
                                'end_date': str(common_dates[-1].date()),
                                'total_days': len(common_dates),
                                'annualized_return': ((1 + np.mean(pair_returns[position != 0])) ** 252) - 1 if len(non_zero_returns) > 0 else 0,
                                'annualized_volatility': np.std(non_zero_returns) * np.sqrt(252) if len(non_zero_returns) > 0 else 0,
                                'sharpe_ratio': sharpe_ratio,
                                'max_drawdown': max_drawdown,
                                'win_rate': win_rate,
                                'avg_holding_period': np.mean([len(list(g)) for k, g in itertools.groupby(position) if k != 0]) if any(position != 0) else 0
                            }
                            
                            # Store pair metadata
                            metadata_path = os.path.join(backtest_path, f"{pair['ticker1']}_{pair['ticker2']}_metadata.json")
                            with open(metadata_path, 'w') as f:
                                import json
                                json.dump(pair_metadata, f)
            
            logger.info("Demo data generation complete")
            return True
            
        except Exception as e:
            logger.error(f"Error generating demo data: {str(e)}")
            return False

if __name__=="__main__":
    db = DatabaseManager()
    success = db.generate_demo_data()
    if success:
        print("Data generation status: success")
    else:
        print("Data generation status: failed")