# src/database_sqlite_2.py
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

class DatabaseManager2:
    """
    Enhanced database manager for the Equity Trading Platform.
    Uses consolidated storage for price and factor data to improve efficiency.
    
    - SQLite for metadata (indices, securities, constituents)
    - Consolidated Parquet files for price and factor data
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
        
        # Initialize consolidated data files if they don't exist
        self._init_consolidated_files()
        
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
        
        connection.commit()
        connection.close()
        
        logger.info("Database initialized successfully")
    
    def _init_consolidated_files(self):
        """Initialize consolidated data files if they don't exist"""
        # Initialize price data file
        price_file = os.path.join(self.ts_data_path, 'consolidated_prices.parquet')
        if not os.path.exists(price_file):
            # Create empty DataFrame with multi-index
            empty_price_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            empty_price_df.index = pd.MultiIndex.from_tuples([], names=['date', 'ticker'])
            empty_price_df.to_parquet(price_file)
        
        # Initialize factor data file
        factor_file = os.path.join(self.ts_data_path, 'consolidated_factors.parquet')
        if not os.path.exists(factor_file):
            # Create empty DataFrame with multi-index
            empty_factor_df = pd.DataFrame(columns=['value'])
            empty_factor_df.index = pd.MultiIndex.from_tuples([], names=['date', 'factor_name'])
            empty_factor_df.to_parquet(factor_file)
        
        # Initialize index data file
        index_file = os.path.join(self.ts_data_path, 'consolidated_indices.parquet')
        if not os.path.exists(index_file):
            # Create empty DataFrame with multi-index
            empty_index_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'return'])
            empty_index_df.index = pd.MultiIndex.from_tuples([], names=['date', 'index_id'])
            empty_index_df.to_parquet(index_file)
        
        # Initialize index constituents file
        constituents_file = os.path.join(self.ts_data_path, 'consolidated_constituents.parquet')
        if not os.path.exists(constituents_file):
            # Create empty DataFrame with multi-index
            empty_constituents_df = pd.DataFrame(columns=['weight'])
            empty_constituents_df.index = pd.MultiIndex.from_tuples([], names=['date', 'index_id', 'ticker'])
            empty_constituents_df.to_parquet(constituents_file)
            
        # Initialize factor exposures file
        exposures_file = os.path.join(self.ts_data_path, 'consolidated_exposures.parquet')
        if not os.path.exists(exposures_file):
            # Create empty DataFrame with multi-index
            empty_exposures_df = pd.DataFrame(columns=['factor_loading', 'factor_exposure'])
            empty_exposures_df.index = pd.MultiIndex.from_tuples([], names=['date', 'factor', 'ticker'])
            empty_exposures_df.to_parquet(exposures_file)
        
        logger.info("Consolidated data files initialized")
    
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
    
    def store_price_data(self, ticker: str, price_data: pd.DataFrame) -> bool:
        """
        Store price data for a security in the consolidated file.
        
        Args:
            ticker: Security ticker
            price_data: DataFrame with price data (must have 'date' as index or column)
            
        Returns:
            Success status
        """
        try:
            # Ensure date is in the index
            if 'date' in price_data.columns:
                price_data = price_data.set_index('date')
            
            # Read existing consolidated data
            price_file = os.path.join(self.ts_data_path, 'consolidated_prices.parquet')
            existing_data = pd.read_parquet(price_file)
            
            # Add ticker to index
            price_data.index = pd.MultiIndex.from_product([price_data.index, [ticker]], 
                                                         names=['date', 'ticker'])
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, price_data])
            
            # Remove duplicates (if any)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            # Save back to file
            combined_data.to_parquet(price_file)
            
            # Update security metadata
            if not price_data.empty:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                
                cursor.execute('''
                UPDATE securities 
                SET first_date = ?, last_date = ?
                WHERE ticker = ?
                ''', (str(price_data.index.get_level_values('date').min().date()), 
                      str(price_data.index.get_level_values('date').max().date()), 
                      ticker))
                
                connection.commit()
                connection.close()
            
            logger.info(f"Stored price data for {ticker}, {len(price_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data for {ticker}: {str(e)}")
            return False
    
    def get_price_data(self, tickers: Union[str, List[str]], 
                      start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve price data for one or more securities.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
            end_date: End date for data retrieval (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with price data
        """
        try:
            # Convert single ticker to list
            if isinstance(tickers, str):
                tickers = [tickers]
            
            # Read consolidated data
            price_file = os.path.join(self.ts_data_path, 'consolidated_prices.parquet')
            price_data = pd.read_parquet(price_file)
            
            # Filter by tickers
            price_data = price_data.xs(tickers[0], level='ticker', drop_level=False)
            
            # Filter by date if provided
            if start_date:
                price_data = price_data[price_data.index.get_level_values('date') >= start_date]
            if end_date:
                price_data = price_data[price_data.index.get_level_values('date') <= end_date]
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error retrieving price data: {str(e)}")
            return pd.DataFrame()
    
    def store_factor_data(self, factor_id: str, factor_data: pd.DataFrame) -> bool:
        """
        Store factor data in the consolidated file.
        
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
            
            # Prepare factor data for consolidated storage
            factor_data = factor_data.copy()
            factor_data['factor_name'] = factor_id
            
            # Ensure date is in the index
            if 'date' in factor_data.columns:
                factor_data = factor_data.set_index(['date', 'factor_name'])
            
            # Read existing consolidated data
            factor_file = os.path.join(self.ts_data_path, 'consolidated_factors.parquet')
            existing_data = pd.read_parquet(factor_file)
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, factor_data])
            
            # Remove duplicates (if any)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            # Save back to file
            combined_data.to_parquet(factor_file)
            
            connection.close()
            
            logger.info(f"Stored factor data for {factor_id}, {len(factor_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing factor data for {factor_id}: {str(e)}")
            return False
    
    def get_factor_data(self, factor_ids: Union[str, List[str]], 
                       tickers: List[str] = None,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve factor data for one or more factors.
        
        Args:
            factor_ids: Single factor ID or list of factor IDs
            tickers: Optional list of tickers to filter by
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with factor data
        """
        try:
            # Convert single factor_id to list
            if isinstance(factor_ids, str):
                factor_ids = [factor_ids]
            
            # Read consolidated data
            factor_file = os.path.join(self.ts_data_path, 'consolidated_factors.parquet')
            factor_data = pd.read_parquet(factor_file)
            
            # Filter by factor IDs
            factor_data = factor_data.xs(factor_ids, level='factor_name', drop_level=False)
            
            # Apply date filters
            if start_date:
                factor_data = factor_data[factor_data.index.get_level_values('date') >= start_date]
            if end_date:
                factor_data = factor_data[factor_data.index.get_level_values('date') <= end_date]
            
            # Filter by tickers if provided
            if tickers:
                factor_data = factor_data[factor_data['ticker'].isin(tickers)]
            
            return factor_data
            
        except Exception as e:
            logger.error(f"Error retrieving factor data: {str(e)}")
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
            
            # Remove consolidated data files
            price_file = os.path.join(self.ts_data_path, 'consolidated_prices.parquet')
            factor_file = os.path.join(self.ts_data_path, 'consolidated_factors.parquet')
            
            if os.path.exists(price_file):
                os.remove(price_file)
            if os.path.exists(factor_file):
                os.remove(factor_file)
            
            # Recreate database structure and files
            self._init_database()
            self._init_consolidated_files()
            
            logger.warning("Database has been reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False

    def store_index_data(self, index_id: str, index_data: pd.DataFrame) -> bool:
        """
        Store index time series data in the consolidated file.
        
        Args:
            index_id: Index identifier (e.g., 'SPX', 'NDX')
            index_data: DataFrame with index data (must have 'date' as index or column)
            
        Returns:
            Success status
        """
        try:
            # Ensure date is in the index
            if 'date' in index_data.columns:
                index_data = index_data.set_index('date')
            
            # Read existing consolidated data
            index_file = os.path.join(self.ts_data_path, 'consolidated_indices.parquet')
            existing_data = pd.read_parquet(index_file)
            
            # Add index_id to index
            index_data.index = pd.MultiIndex.from_product([index_data.index, [index_id]], 
                                                         names=['date', 'index_id'])
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, index_data])
            
            # Remove duplicates (if any)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            # Save back to file
            combined_data.to_parquet(index_file)
            
            # Update index metadata
            if not index_data.empty:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                
                cursor.execute('''
                UPDATE indices 
                SET last_updated = ?
                WHERE index_id = ?
                ''', (datetime.datetime.now(), index_id))
                
                connection.commit()
                connection.close()
            
            logger.info(f"Stored index data for {index_id}, {len(index_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing index data for {index_id}: {str(e)}")
            return False
    
    def get_index_data(self, index_ids: Union[str, List[str]], 
                      start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve index time series data for one or more indices.
        
        Args:
            index_ids: Single index ID or list of index IDs
            start_date: Start date for data retrieval (format: 'YYYY-MM-DD')
            end_date: End date for data retrieval (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with index data
        """
        try:
            # Convert single index_id to list
            if isinstance(index_ids, str):
                index_ids = [index_ids]
            
            # Read consolidated data
            index_file = os.path.join(self.ts_data_path, 'consolidated_indices.parquet')
            index_data = pd.read_parquet(index_file)
            
            # Filter by index IDs
            index_data = index_data.xs(index_ids, level='index_id', drop_level=False)
            
            # Filter by date if provided
            if start_date:
                index_data = index_data[index_data.index.get_level_values('date') >= start_date]
            if end_date:
                index_data = index_data[index_data.index.get_level_values('date') <= end_date]
            
            return index_data
            
        except Exception as e:
            logger.error(f"Error retrieving index data: {str(e)}")
            return pd.DataFrame()
    
    def store_index_constituents(self, index_id: str, constituents_data: pd.DataFrame) -> bool:
        """
        Store index constituents data in the consolidated file.
        
        Args:
            index_id: Index identifier
            constituents_data: DataFrame with constituents data (columns: date, ticker, weight)
            
        Returns:
            Success status
        """
        try:
            # Ensure date is in the index
            if 'date' in constituents_data.columns:
                constituents_data = constituents_data.set_index(['date', 'ticker'])
            
            # Add index_id to index
            constituents_data.index = pd.MultiIndex.from_product(
                [constituents_data.index.get_level_values('date').unique(),
                 [index_id],
                 constituents_data.index.get_level_values('ticker').unique()],
                names=['date', 'index_id', 'ticker']
            )
            
            # Read existing consolidated data
            constituents_file = os.path.join(self.ts_data_path, 'consolidated_constituents.parquet')
            existing_data = pd.read_parquet(constituents_file)
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, constituents_data])
            
            # Remove duplicates (if any)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            # Save back to file
            combined_data.to_parquet(constituents_file)
            
            # Update index metadata
            if not constituents_data.empty:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                
                cursor.execute('''
                UPDATE indices 
                SET last_updated = ?
                WHERE index_id = ?
                ''', (datetime.datetime.now(), index_id))
                
                connection.commit()
                connection.close()
            
            logger.info(f"Stored constituents data for {index_id}, {len(constituents_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing constituents data for {index_id}: {str(e)}")
            return False
    
    def get_index_constituents(self, index_id: str, 
                             start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve index constituents data.
        
        Args:
            index_id: Index identifier
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with constituents data
        """
        try:
            # Read consolidated data
            constituents_file = os.path.join(self.ts_data_path, 'consolidated_constituents.parquet')
            constituents_data = pd.read_parquet(constituents_file)
            
            # Filter by index ID
            constituents_data = constituents_data.xs(index_id, level='index_id', drop_level=False)
            
            # Apply date filters
            if start_date:
                constituents_data = constituents_data[constituents_data.index.get_level_values('date') >= start_date]
            if end_date:
                constituents_data = constituents_data[constituents_data.index.get_level_values('date') <= end_date]
            
            return constituents_data
            
        except Exception as e:
            logger.error(f"Error retrieving constituents data for {index_id}: {str(e)}")
            return pd.DataFrame()

    def store_factor_exposures(self, exposures_data: pd.DataFrame) -> bool:
        """
        Store factor exposures data in the consolidated file.
        
        Args:
            exposures_data: DataFrame with factor exposures data (columns: date, factor, ticker, factor_loading, factor_exposure)
            
        Returns:
            Success status
        """
        try:
            # Ensure date is in the index
            if 'date' in exposures_data.columns:
                exposures_data = exposures_data.set_index(['date', 'factor', 'ticker'])
            
            # Read existing consolidated data
            exposures_file = os.path.join(self.ts_data_path, 'consolidated_exposures.parquet')
            existing_data = pd.read_parquet(exposures_file)
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, exposures_data])
            
            # Remove duplicates (if any)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            # Save back to file
            combined_data.to_parquet(exposures_file)
            
            # Update factor metadata
            if not exposures_data.empty:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                
                # Update last_updated for each factor
                factors = exposures_data.index.get_level_values('factor').unique()
                for factor in factors:
                    cursor.execute('''
                    UPDATE factors 
                    SET last_updated = ?
                    WHERE factor_id = ?
                    ''', (datetime.datetime.now(), factor))
                
                connection.commit()
                connection.close()
            
            logger.info(f"Stored factor exposures data, {len(exposures_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing factor exposures data: {str(e)}")
            return False
    
    def get_factor_exposures(self, factors: Union[str, List[str]] = None,
                           tickers: Union[str, List[str]] = None,
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve factor exposures data.
        
        Args:
            factors: Single factor or list of factors to filter by
            tickers: Single ticker or list of tickers to filter by
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with factor exposures data
        """
        try:
            # Read consolidated data
            exposures_file = os.path.join(self.ts_data_path, 'consolidated_exposures.parquet')
            exposures_data = pd.read_parquet(exposures_file)
            
            # Convert single factor/ticker to list
            if isinstance(factors, str):
                factors = [factors]
            if isinstance(tickers, str):
                tickers = [tickers]
            
            # Apply filters
            if factors:
                exposures_data = exposures_data.xs(factors, level='factor', drop_level=False)
            
            if tickers:
                exposures_data = exposures_data.xs(tickers, level='ticker', drop_level=False)
            
            if start_date:
                exposures_data = exposures_data[exposures_data.index.get_level_values('date') >= start_date]
            if end_date:
                exposures_data = exposures_data[exposures_data.index.get_level_values('date') <= end_date]
            
            return exposures_data
            
        except Exception as e:
            logger.error(f"Error retrieving factor exposures data: {str(e)}")
            return pd.DataFrame()
    
    def get_security_factor_exposures(self, ticker: str,
                                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve all factor exposures for a specific security.
        
        Args:
            ticker: Security ticker
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with security's factor exposures
        """
        return self.get_factor_exposures(tickers=ticker, start_date=start_date, end_date=end_date)
    
    def get_factor_security_exposures(self, factor: str,
                                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve all security exposures for a specific factor.
        
        Args:
            factor: Factor identifier
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with factor's security exposures
        """
        return self.get_factor_exposures(factors=factor, start_date=start_date, end_date=end_date)

    def generate_demo_data_v1(self) -> bool:
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
            
            # Create a consolidated price DataFrame
            price_data_list = []
            
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
                
                # Create price DataFrame for this ticker
                price_df = pd.DataFrame({
                    'open': prices * (1 - np.random.uniform(0, 0.01, size=n_days)),
                    'high': prices * (1 + np.random.uniform(0, 0.02, size=n_days)),
                    'low': prices * (1 - np.random.uniform(0, 0.02, size=n_days)),
                    'close': prices,
                    'volume': np.random.randint(100000, 10000000, size=n_days)
                }, index=dates)
                
                # Add ticker to index
                price_df.index = pd.MultiIndex.from_product([price_df.index, [ticker]], 
                                                            names=['date', 'ticker'])
                price_data_list.append(price_df)
            
            # Combine all price data and save to consolidated file
            consolidated_prices = pd.concat(price_data_list)
            consolidated_prices = consolidated_prices.sort_index()
            consolidated_prices.to_parquet(os.path.join(self.ts_data_path, 'consolidated_prices.parquet'))
            
            # Generate index time series data
            for index_id in ['SPX', 'NDX', 'RTY']:
                # Get constituents and weights
                constituents = self.get_index_constituents(index_id)
                if constituents.empty:
                    continue
                
                # Create index price data
                index_prices = pd.DataFrame(index=dates)
                index_prices['open'] = 100  # Start at 100
                index_prices['high'] = index_prices['open'] * (1 + np.random.uniform(0, 0.02, size=len(dates)))
                index_prices['low'] = index_prices['open'] * (1 - np.random.uniform(0, 0.02, size=len(dates)))
                index_prices['close'] = index_prices['open'] * (1 + np.random.uniform(-0.01, 0.01, size=len(dates)))
                index_prices['volume'] = np.random.randint(1000000, 100000000, size=len(dates))
                
                # Calculate returns
                index_prices['return'] = index_prices['close'].pct_change()
                
                # Store index data
                self.store_index_data(index_id, index_prices)
            
            # Generate factor data and exposures
            factors = ['beta', 'size', 'value', 'momentum', 'quality']
            monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
            
            # Create factor exposures data
            exposures_data_list = []
            
            for factor in factors:
                # Generate base factor values with persistence
                base_values = {}
                for ticker in all_tickers:
                    base_values[ticker] = np.random.normal(0, 1)
                
                # Generate monthly exposures
                for date in monthly_dates:
                    for ticker in all_tickers:
                        # Add some randomness but maintain persistence
                        factor_loading = base_values[ticker] + np.random.normal(0, 0.2)
                        
                        # Calculate factor exposure (loading * factor value)
                        factor_value = np.random.normal(0, 1)  # Simulated factor value
                        factor_exposure = factor_loading * factor_value
                        
                        # Drift the base value slightly
                        base_values[ticker] = 0.9 * base_values[ticker] + 0.1 * np.random.normal(0, 1)
                        
                        exposures_data_list.append({
                            'date': date,
                            'factor': factor,
                            'ticker': ticker,
                            'factor_loading': factor_loading,
                            'factor_exposure': factor_exposure
                        })
            
            # Convert to DataFrame and store factor exposures
            exposures_df = pd.DataFrame(exposures_data_list)
            self.store_factor_exposures(exposures_df)
            
            # Generate factor data
            factor_data_list = []
            
            for factor in factors:
                # Get monthly dates
                monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
                
                for ticker in all_tickers:
                    # Generate factor values (with persistence across time)
                    base_value = np.random.normal(0, 1)
                    
                    for date in monthly_dates:
                        # Add some randomness but maintain persistence
                        factor_value = base_value + np.random.normal(0, 0.2)
                        
                        # Drift the base value slightly
                        base_value = 0.9 * base_value + 0.1 * np.random.normal(0, 1)
                        
                        factor_data_list.append({
                            'date': date,
                            'factor_name': factor,
                            'ticker': ticker,
                            'value': factor_value
                        })
            
            # Convert to DataFrame and save to consolidated file
            factor_df = pd.DataFrame(factor_data_list)
            factor_df = factor_df.set_index(['date', 'factor_name'])
            factor_df = factor_df.sort_index()
            factor_df.to_parquet(os.path.join(self.ts_data_path, 'consolidated_factors.parquet'))
            
            # Generate backtest strategy data
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

    # Add this method to the DatabaseManager2 class in database_sqlite_2.py
    def generate_demo_data(self) -> bool:
        """
        Generate comprehensive demo data for the Equity Trading Platform.
        
        Creates:
        - Sample indices (S&P 500, NASDAQ 100)
        - Demo securities with realistic price data
        - Factor data (value, momentum, quality, etc.)
        - Sample pairs trading data
        - Backtest results
        
        Returns:
            bool: Success status
        """
        try:
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta
            import sqlite3
            import os
            
            # Reset database first
            self.reset_database()
            logger.info("Generating demo data...")
            
            # Define parameters
            start_date = datetime.strptime('2025-01-01', '%Y-%m-%d')
            end_date = datetime.now()
            trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Add indices
            self.add_index('SPX', 'S&P 500', 'Standard & Poor\'s 500 Index', 'US')
            self.add_index('NDX', 'NASDAQ 100', 'NASDAQ 100 Index', 'US')
            
            # Create sectors
            sectors = {
                'TECH': 'Technology',
                'FINS': 'Financials',
                'HLTH': 'Healthcare',
                'CONS': 'Consumer',
                'UTIL': 'Utilities',
                'INDU': 'Industrials',
                'ENER': 'Energy',
                'MATL': 'Materials',
                'REAL': 'Real Estate',
                'COMM': 'Communication'
            }
            
            # Create demo tickers - FIXED APPROACH HERE
            all_tickers = []
            tickers_by_sector = {}
            
            for sector_code in sectors.keys():
                sector_tickers = []
                for i in range(1, 5): # 11
                    ticker = f"{sector_code}{i:02d}"  # Ensures 2-digit format, like TECH01, TECH02, etc.
                    sector_tickers.append(ticker)
                    all_tickers.append(ticker)
                tickers_by_sector[sector_code] = sector_tickers
            
            # Sample a subset for NDX (40 tickers)
            ndx_tickers = np.random.choice(all_tickers, size=40, replace=False).tolist()
            # All tickers for SPX (100 tickers in this demo)
            spx_tickers = all_tickers
            
            # Insert securities
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for ticker in all_tickers:
                # Extract sector code from ticker (first 4 characters)
                sector_code = ticker[:4]  # This will correctly extract TECH, FIN, etc.
                
                cursor.execute('''
                INSERT OR REPLACE INTO securities (ticker, name, sector, industry)
                VALUES (?, ?, ?, ?)
                ''', (ticker, f"Company {ticker}", sectors[sector_code], f"{sectors[sector_code]} Subsector"))
            
            conn.commit()
            conn.close()
        
            # Generate price data for each ticker
            logger.info(f"Generating price data for {len(all_tickers)} securities...")
            
            for ticker in all_tickers:
                print(ticker)
                # Create a random starting price between $10 and $500
                start_price = np.random.uniform(10, 500)
                
                # Generate random walk for prices with drift
                annual_return = np.random.uniform(0.0, 0.20)  # 0% to 20% annual return
                annual_vol = np.random.uniform(0.15, 0.40)    # 15% to 40% annual volatility
                
                daily_drift = annual_return / 252
                daily_vol = annual_vol / np.sqrt(252)
                
                # Generate log returns
                n_days = len(trading_days)
                log_returns = np.random.normal(daily_drift, daily_vol, n_days)
                
                # Add some autocorrelation to the returns
                ar_param = np.random.uniform(0.1, 0.3)
                for i in range(1, n_days):
                    log_returns[i] = ar_param * log_returns[i-1] + (1 - ar_param) * log_returns[i]
                
                # Calculate price series
                price_series = start_price * np.exp(np.cumsum(log_returns))
                
                # Create price DataFrame
                price_data = pd.DataFrame({
                    'open': price_series * (1 - np.random.uniform(-0.01, 0.01, n_days)),
                    'high': price_series * (1 + np.random.uniform(0, 0.02, n_days)),
                    'low': price_series * (1 - np.random.uniform(0, 0.02, n_days)),
                    'close': price_series,
                    'volume': np.random.lognormal(15, 1, n_days)
                }, index=trading_days)
                
                # Ensure high >= open, close, low and low <= open, close, high
                price_data['high'] = price_data[['open', 'close', 'high']].max(axis=1)
                price_data['low'] = price_data[['open', 'close', 'low']].min(axis=1)
                
                # Add close-to-close returns
                price_data['return'] = price_data['close'].pct_change()
                
                # Store in the database
                self.store_price_data(ticker, price_data)
            
            # Generate index data
            logger.info("Generating index data...")
            
            # Create weights for the indices
            spx_weights = {}
            ndx_weights = {}
            
            # Generate random weights
            for ticker in spx_tickers:
                spx_weights[ticker] = np.random.uniform(0.1, 3.0)
            
            for ticker in ndx_tickers:
                ndx_weights[ticker] = np.random.uniform(0.1, 5.0)
            
            # Normalize weights to sum to 100%
            spx_total = sum(spx_weights.values())
            ndx_total = sum(ndx_weights.values())
            
            for ticker in spx_weights:
                spx_weights[ticker] = (spx_weights[ticker] / spx_total) * 100
                
            for ticker in ndx_weights:
                ndx_weights[ticker] = (ndx_weights[ticker] / ndx_total) * 100
            
            # Create constituents data
            spx_constituents = pd.DataFrame([
                {'ticker': ticker, 'weight': weight, 'name': f"Company {ticker}", 
                'sector': ticker[:4], 'industry': f"{ticker[:4]} Subsector"} 
                for ticker, weight in spx_weights.items()
            ])
            
            ndx_constituents = pd.DataFrame([
                {'ticker': ticker, 'weight': weight, 'name': f"Company {ticker}", 
                'sector': ticker[:4], 'industry': f"{ticker[:4]} Subsector"} 
                for ticker, weight in ndx_weights.items()
            ])
            
            # Store constituents data
            for date in trading_days[::21]:  # Monthly rebalance
                date_str = date.strftime('%Y-%m-%d')
                
                # Add a small random variation to weights each month
                spx_constituents_monthly = spx_constituents.copy()
                ndx_constituents_monthly = ndx_constituents.copy()
                
                # Add small random variations to weights
                spx_constituents_monthly['weight'] *= (1 + np.random.uniform(-0.05, 0.05, len(spx_constituents_monthly)))
                ndx_constituents_monthly['weight'] *= (1 + np.random.uniform(-0.05, 0.05, len(ndx_constituents_monthly)))
                
                # Normalize again
                spx_constituents_monthly['weight'] = spx_constituents_monthly['weight'] / spx_constituents_monthly['weight'].sum() * 100
                ndx_constituents_monthly['weight'] = ndx_constituents_monthly['weight'] / ndx_constituents_monthly['weight'].sum() * 100
                
                # Add date column
                spx_constituents_monthly['date'] = date_str
                ndx_constituents_monthly['date'] = date_str
                
                # Store
                self.store_index_constituents('SPX', spx_constituents_monthly)
                self.store_index_constituents('NDX', ndx_constituents_monthly)
            
            # Calculate and store index prices
            spx_data = pd.DataFrame(index=trading_days, columns=['open', 'high', 'low', 'close', 'volume', 'return'])
            ndx_data = pd.DataFrame(index=trading_days, columns=['open', 'high', 'low', 'close', 'volume', 'return'])
            
            # Read securities price data
            all_prices = {}
            for ticker in all_tickers:
                ticker_data = self.get_price_data(ticker)
                all_prices[ticker] = ticker_data
            
            # Calculate index values
            for date in trading_days:
                date_str = date.strftime('%Y-%m-%d')
                
                # SPX
                spx_close = 0
                spx_open = 0
                spx_high = 0
                spx_low = 0
                spx_volume = 0
                
                for ticker, weight in spx_weights.items():
                    if ticker in all_prices and date in all_prices[ticker].index:
                        # print((all_prices[ticker].loc[date, 'close'] * weight / 100))
                        # print(f"{ticker} -> {spx_close}")
                        spx_close += (all_prices[ticker].loc[date, 'close'] * weight / 100).squeeze()
                        spx_open += (all_prices[ticker].loc[date, 'open'] * weight / 100).squeeze()
                        spx_high += (all_prices[ticker].loc[date, 'high'] * weight / 100).squeeze()
                        spx_low += (all_prices[ticker].loc[date, 'low'] * weight / 100).squeeze()
                        spx_volume += (all_prices[ticker].loc[date, 'volume'] * weight / 100).squeeze()
                
                spx_data.loc[date, 'close'] = spx_close
                spx_data.loc[date, 'open'] = spx_open
                spx_data.loc[date, 'high'] = spx_high
                spx_data.loc[date, 'low'] = spx_low
                spx_data.loc[date, 'volume'] = spx_volume
                
                # NDX
                ndx_close = 0
                ndx_open = 0
                ndx_high = 0
                ndx_low = 0
                ndx_volume = 0
                
                for ticker, weight in ndx_weights.items():
                    if ticker in all_prices and date in all_prices[ticker].index:
                        ndx_close += (all_prices[ticker].loc[date, 'close'] * weight / 100).squeeze()
                        ndx_open += (all_prices[ticker].loc[date, 'open'] * weight / 100).squeeze()
                        ndx_high += (all_prices[ticker].loc[date, 'high'] * weight / 100).squeeze()
                        ndx_low += (all_prices[ticker].loc[date, 'low'] * weight / 100).squeeze()
                        ndx_volume += (all_prices[ticker].loc[date, 'volume'] * weight / 100).squeeze()
                
                ndx_data.loc[date, 'close'] = ndx_close
                ndx_data.loc[date, 'open'] = ndx_open
                ndx_data.loc[date, 'high'] = ndx_high
                ndx_data.loc[date, 'low'] = ndx_low
                ndx_data.loc[date, 'volume'] = ndx_volume
            
            # Calculate returns
            spx_data['return'] = spx_data['close'].pct_change()
            ndx_data['return'] = ndx_data['close'].pct_change()
            
            # Store index data
            self.store_index_data('SPX', spx_data)
            self.store_index_data('NDX', ndx_data)
            
            # Generate factor data
            logger.info("Generating factor data...")
            
            # Define factors
            factors = ['value', 'momentum', 'size', 'quality', 'volatility', 'growth']
            
            # Create correlation structure between factors (realistic correlations)
            factor_corr = pd.DataFrame(np.array([
                [1.00, -0.25, -0.20, 0.30, 0.10, -0.15],  # value
                [-0.25, 1.00, 0.15, 0.05, -0.30, 0.25],   # momentum
                [-0.20, 0.15, 1.00, 0.10, -0.25, 0.15],   # size
                [0.30, 0.05, 0.10, 1.00, 0.25, 0.20],     # quality
                [0.10, -0.30, -0.25, 0.25, 1.00, -0.20],  # volatility
                [-0.15, 0.25, 0.15, 0.20, -0.20, 1.00]    # growth
            ]), index=factors, columns=factors)
            
            # Generate multivariate normal samples for each date and ticker
            factor_exposures = {}
            
            for date_idx, date in enumerate(trading_days[::21]):  # Monthly factor updates
                date_str = date.strftime('%Y-%m-%d')
                
                # For each ticker, generate factor exposures
                for ticker in all_tickers:
                    if ticker not in factor_exposures:
                        factor_exposures[ticker] = {}
                    
                    # Add some persistence to factor exposures (AR process)
                    if date_idx > 0:
                        prev_exposures = np.array([factor_exposures[ticker][f][date_idx-1] for f in factors])
                        
                        # Generate new exposures with AR(1) process
                        ar_param = 0.8  # High persistence in factor exposures
                        new_shock = np.random.multivariate_normal(
                            np.zeros(len(factors)), factor_corr.values)
                        
                        new_exposures = ar_param * prev_exposures + (1 - ar_param) * new_shock
                    else:
                        # Initial exposures
                        new_exposures = np.random.multivariate_normal(
                            np.zeros(len(factors)), factor_corr.values)
                    
                    # Store in dictionary
                    for i, factor in enumerate(factors):
                        if factor not in factor_exposures[ticker]:
                            factor_exposures[ticker][factor] = {}
                        
                        factor_exposures[ticker][factor][date_idx] = new_exposures[i]
            
            # Create factor DataFrames and store
            for factor in factors:
                factor_data_list = []
                
                for date_idx, date in enumerate(trading_days[::21]):
                    date_str = date.strftime('%Y-%m-%d')
                    
                    for ticker in all_tickers:
                        factor_data_list.append({
                            'date': date_str,
                            'ticker': ticker,
                            'value': factor_exposures[ticker][factor][date_idx]
                        })
                
                factor_df = pd.DataFrame(factor_data_list)
                
                # Register factor in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO factors (factor_id, name, description, category, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ''', (factor, factor.capitalize(), f"{factor.capitalize()} factor", "Style", datetime.now()))
                conn.commit()
                conn.close()
                
                # Store factor data
                self.store_factor_data(factor, factor_df)
            
            # Generate factor returns data
            logger.info("Generating factor returns data...")
            
            # For each factor, create long-short portfolio returns
            for factor in factors:
                factor_returns_list = []
                
                for date_idx, date in enumerate(trading_days):
                    if date_idx == 0:
                        continue  # Skip first day (no returns)
                    
                    # Get previous business day
                    prev_date = trading_days[date_idx - 1]
                    
                    # Find nearest factor date (monthly data)
                    nearest_factor_idx = (date_idx // 21) * 21  # Integer division to find nearest month
                    if nearest_factor_idx >= len(trading_days):
                        nearest_factor_idx = len(trading_days) - 1
                    
                    nearest_factor_date = trading_days[nearest_factor_idx]
                    nearest_factor_date_str = nearest_factor_date.strftime('%Y-%m-%d')
                    
                    # Get factor values for all stocks on nearest date
                    factor_values = {}
                    for ticker in all_tickers:
                        factor_values[ticker] = factor_exposures[ticker][factor][nearest_factor_idx // 21]
                    
                    # Sort tickers by factor value
                    sorted_tickers = sorted(factor_values.keys(), key=lambda x: factor_values[x])
                    
                    # For some factors, we sort in reverse (higher is better)
                    if factor in ['momentum', 'quality', 'growth']:
                        sorted_tickers = sorted_tickers[::-1]
                    
                    # Create quintile buckets
                    quintile_size = len(sorted_tickers) // 5
                    quintiles = {}
                    for i in range(5):
                        if i < 4:
                            quintiles[i] = sorted_tickers[i*quintile_size:(i+1)*quintile_size]
                        else:
                            quintiles[i] = sorted_tickers[i*quintile_size:]
                    
                    # Calculate returns for each quintile
                    for quintile, quintile_tickers in quintiles.items():
                        quintile_return = np.random.randn() # 0
                        valid_tickers = 0
                        
                        for ticker in quintile_tickers:
                        #     if ticker in all_prices and date in all_prices[ticker].index and prev_date in all_prices[ticker].index:
                        #         # Calculate return
                        #         price_today = all_prices[ticker].loc[date, 'close']
                        #         price_yesterday = all_prices[ticker].loc[prev_date, 'close']
                        #         ticker_return = (price_today / price_yesterday) - 1
                                
                        #         quintile_return += ticker_return
                        #         valid_tickers += 1
                        
                        # if valid_tickers > 0:
                        #     quintile_return /= valid_tickers
                            
                            factor_returns_list.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'factor': factor,
                                'bucket': str(quintile + 1),  # 1-5
                                'return': quintile_return
                            })
                    
                    # Add long-short portfolio return (Q5 - Q1)
                    if 0 in quintiles and 4 in quintiles and len(quintiles[0]) > 0 and len(quintiles[4]) > 0:
                        q1_return = np.random.randn() #sum(all_prices[ticker].loc[date, 'return'] for ticker in quintiles[0] 
                                    # if ticker in all_prices and date in all_prices[ticker].index) / len(quintiles[0])
                        
                        q5_return = np.random.randn() # sum(all_prices[ticker].loc[date, 'return'] for ticker in quintiles[4]
                                    # if ticker in all_prices and date in all_prices[ticker].index) / len(quintiles[4])
                        
                        if factor in ['value', 'size', 'volatility']:
                            # For these factors, Q1 is the long side (smaller values are better)
                            ls_return = q1_return - q5_return
                        else:
                            # For others, Q5 is the long side (larger values are better)
                            ls_return = q5_return - q1_return
                        
                        factor_returns_list.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'factor': factor,
                            'bucket': 'long_short',
                            'return': ls_return
                        })
                
                # Store factor returns
                # factor_returns_df = pd.DataFrame(factor_returns_list)
                
                # # Create factor_returns folder if it doesn't exist
                # factor_returns_dir = os.path.join(self.ts_data_path, 'factor_returns')
                # os.makedirs(factor_returns_dir, exist_ok=True)
                
                # # Save to parquet
                # factor_returns_file = os.path.join(factor_returns_dir, f'{factor}_returns.parquet')
                # factor_returns_df.to_parquet(factor_returns_file)
                # Find this section in the generate_demo_data method where it's saving factor returns
# and replace it with this fixed version:

                factor_returns_df = pd.DataFrame(factor_returns_list)
                
                # Convert data types to ensure compatibility with Parquet
                factor_returns_df['date'] = pd.to_datetime(factor_returns_df['date']).dt.strftime('%Y-%m-%d')
                factor_returns_df['factor'] = factor_returns_df['factor'].astype(str)
                factor_returns_df['bucket'] = factor_returns_df['bucket'].astype(str)
                factor_returns_df['return'] = factor_returns_df['return'].astype(float)
                
                # Create factor_returns folder if it doesn't exist
                factor_returns_dir = os.path.join(self.ts_data_path, 'factor_returns')
                os.makedirs(factor_returns_dir, exist_ok=True)
                
                # Save to parquet with explicit schema
                factor_returns_file = os.path.join(factor_returns_dir, f'{factor}_returns.parquet')
                try:
                    # Try direct save first
                    factor_returns_df.to_parquet(factor_returns_file, index=False)
                except Exception as e:
                    logger.error(f"Error saving factor returns to parquet: {str(e)}")
                    logger.info("Attempting alternate save method...")
                    
                    # If direct save fails, try writing with pyarrow explicitly
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    
                    # Create PyArrow table with explicit schema
                    table = pa.Table.from_pandas(
                        factor_returns_df,
                        schema=pa.schema([
                            ('date', pa.string()),
                            ('factor', pa.string()),
                            ('bucket', pa.string()),
                            ('return', pa.float64())
                        ])
                    )
                    
                    # Write to Parquet file
                    pq.write_table(table, factor_returns_file)
            
            # Generate pairs trading data
            logger.info("SKIP: Generating pairs trading data...")
            
            # # Create pairs folder
            # pairs_dir = os.path.join(self.ts_data_path, 'pairs')
            # os.makedirs(pairs_dir, exist_ok=True)
            
            # # Select some sector pairs
            # sector_pairs = []
            # for sector in sectors:
            #     # Get tickers from this sector
            #     sector_tickers = tickers_by_sector[sector]
                
            #     # Select 2 random pairs from each sector
            #     if len(sector_tickers) >= 4:
            #         for _ in range(2):
            #             pair = np.random.choice(sector_tickers, size=2, replace=False)
            #             sector_pairs.append((pair[0], pair[1]))
            
            # # Generate pairs trading data with realistic cointegration
            # for pair in sector_pairs:
            #     ticker1, ticker2 = pair
                
            #     # Get price data for both tickers
            #     ticker1_prices = all_prices[ticker1]['close']
            #     ticker2_prices = all_prices[ticker2]['close']
                
            #     # Create pairs data
            #     pairs_data = pd.DataFrame(index=trading_days)
            #     pairs_data['ticker1'] = ticker1
            #     pairs_data['ticker2'] = ticker2
            #     pairs_data['ticker1_price'] = ticker1_prices
            #     pairs_data['ticker2_price'] = ticker2_prices
                
            #     # Calculate price ratio
            #     pairs_data['price_ratio'] = pairs_data['ticker1_price'] / pairs_data['ticker2_price']
                
            #     # Calculate z-score with 30-day rolling window
            #     pairs_data['ratio_mean'] = pairs_data['price_ratio'].rolling(window=30).mean()
            #     pairs_data['ratio_std'] = pairs_data['price_ratio'].rolling(window=30).std()
            #     pairs_data['z_score'] = (pairs_data['price_ratio'] - pairs_data['ratio_mean']) / pairs_data['ratio_std']
                
            #     # Replace NaN values in the beginning
            #     pairs_data['ratio_mean'].fillna(pairs_data['price_ratio'].mean(), inplace=True)
            #     pairs_data['ratio_std'].fillna(pairs_data['price_ratio'].std(), inplace=True)
            #     pairs_data['z_score'].fillna(0, inplace=True)
                
            #     # Generate trading signals
            #     pairs_data['position'] = 0
                
            #     # Entry signals
            #     pairs_data.loc[pairs_data['z_score'] > 2, 'position'] = -1  # Short the spread
            #     pairs_data.loc[pairs_data['z_score'] < -2, 'position'] = 1   # Long the spread
                
            #     # Exit signals
            #     for i in range(1, len(pairs_data)):
            #         # If we have a position and the z-score crosses back inside thresholds, exit
            #         if pairs_data['position'].iloc[i-1] == 1 and pairs_data['z_score'].iloc[i] > -0.5:
            #             pairs_data['position'].iloc[i] = 0
            #         elif pairs_data['position'].iloc[i-1] == -1 and pairs_data['z_score'].iloc[i] < 0.5:
            #             pairs_data['position'].iloc[i] = 0
            #         # Otherwise carry forward the position
            #         elif pairs_data['position'].iloc[i] == 0:
            #             pairs_data['position'].iloc[i] = pairs_data['position'].iloc[i-1]
                
            #     # Calculate pair trading returns
            #     pairs_data['return'] = 0.0
                
            #     for i in range(1, len(pairs_data)):
            #         if pairs_data['position'].iloc[i-1] != 0:
            #             # Return = position * change in spread
            #             spread_change = pairs_data['price_ratio'].iloc[i] / pairs_data['price_ratio'].iloc[i-1] - 1
            #             pairs_data['return'].iloc[i] = pairs_data['position'].iloc[i-1] * spread_change
                
            #     # Calculate cumulative return
            #     pairs_data['cumulative_return'] = (1 + pairs_data['return']).cumprod()
                
            #     # Add date column
            #     pairs_data['date'] = trading_days
                
            #     # Save pair trading data
            #     pair_file = os.path.join(pairs_dir, f'{ticker1}_{ticker2}_pairs.parquet')
            #     pairs_data.to_parquet(pair_file)
                
            #     # Create pair metadata
            #     conn = sqlite3.connect(self.db_path)
            #     cursor = conn.cursor()
                
            #     # Create pairs table if it doesn't exist
            #     cursor.execute('''
            #     CREATE TABLE IF NOT EXISTS pairs (
            #         id INTEGER PRIMARY KEY AUTOINCREMENT,
            #         ticker1 TEXT,
            #         ticker2 TEXT,
            #         name TEXT,
            #         sector TEXT,
            #         correlation REAL,
            #         sharpe_ratio REAL,
            #         win_rate REAL,
            #         max_drawdown REAL,
            #         UNIQUE(ticker1, ticker2)
            #     )
            #     ''')
                
            #     # Calculate pair statistics
            #     correlation = np.corrcoef(ticker1_prices, ticker2_prices)[0, 1]
            #     sharpe_ratio = pairs_data['return'].mean() / pairs_data['return'].std() * np.sqrt(252)
                
            #     # Calculate win rate
            #     win_rate = len(pairs_data[pairs_data['return'] > 0]) / len(pairs_data[pairs_data['return'] != 0]) if len(pairs_data[pairs_data['return'] != 0]) > 0 else 0
                
            #     # Calculate max drawdown
            #     cumulative = pairs_data['cumulative_return'].values
            #     max_drawdown = 0
            #     peak = cumulative[0]
                
            #     for value in cumulative:
            #         if value > peak:
            #             peak = value
            #         drawdown = (peak - value) / peak
            #         max_drawdown = max(max_drawdown, drawdown)
                
            #     # Store pair metadata
            #     cursor.execute('''
            #     INSERT OR REPLACE INTO pairs (ticker1, ticker2, name, sector, correlation, sharpe_ratio, win_rate, max_drawdown)
            #     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            #     ''', (ticker1, ticker2, f"{ticker1}/{ticker2} Pair", ticker1[:4], correlation, sharpe_ratio, win_rate, max_drawdown))
                
            #     conn.commit()
            #     conn.close()
            
            # Generate backtest results for factor strategies
            logger.info("Generating backtest results...")
            
            # Create backtest results folder
            backtest_dir = os.path.join(self.ts_data_path, 'backtest_results')
            os.makedirs(backtest_dir, exist_ok=True)
            
            # Create backtest metadata table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT,
                factor TEXT,
                name TEXT,
                description TEXT,
                annualized_return REAL,
                sharpe_ratio REAL,
                information_ratio REAL,
                max_drawdown REAL,
                benchmark_return REAL,
                benchmark_sharpe REAL,
                benchmark_max_drawdown REAL,
                annualized_volatility REAL,
                benchmark_volatility REAL,
                UNIQUE(index_id, factor)
            )
            ''')
            
            conn.commit()
            
            # Generate factor backtest results
            for index_id in ['SPX', 'NDX']:
                for factor in factors:
                    # Create backtest data with realistic performance
                    backtest_data = pd.DataFrame(index=trading_days)
                    
                    # Get index returns
                    if index_id == 'SPX':
                        benchmark_returns = spx_data['return']
                    else:
                        benchmark_returns = ndx_data['return']
                    
                    # Create strategy returns with some alpha and realistic tracking error
                    alpha = np.random.uniform(0.02, 0.08) / 252  # 2% to 8% annual alpha
                    beta = np.random.uniform(0.8, 1.1)  # 0.8 to 1.1 beta
                    
                    # Create strategy returns
                    strategy_returns = pd.Series(index=trading_days, dtype=float)
                    
                    for i, date in enumerate(trading_days):
                        if i == 0:
                            strategy_returns[date] = 0
                            continue
                        
                        # Base return is beta * benchmark + alpha
                        base_return = beta * benchmark_returns[date] + alpha
                        
                        # Add some factor exposure
                        # Find nearest factor value date
                        nearest_factor_idx = (i // 21) * 21
                        if nearest_factor_idx >= len(trading_days):
                            nearest_factor_idx = len(trading_days) - 1
                        
                        # Add factor returns according to the factor strategy
                        factor_file = os.path.join(factor_returns_dir, f'{factor}_returns.parquet')
                        if os.path.exists(factor_file):
                            factor_returns_df = pd.read_parquet(factor_file)
                            
                            # Get long-short return for this date
                            date_str = date.strftime('%Y-%m-%d')
                            ls_returns = factor_returns_df[(factor_returns_df['date'] == date_str) & 
                                                        (factor_returns_df['bucket'] == 'long_short')]
                            
                            if not ls_returns.empty:
                                factor_component = 0.3 * ls_returns['return'].iloc[0]  # 30% exposure to factor return
                            else:
                                factor_component = 0
                        else:
                            factor_component = 0
                        
                        # Add some idiosyncratic noise
                        idiosyncratic = np.random.normal(0, 0.005)  # ~50bps daily idiosyncratic vol
                        
                        # Combine all components for final return
                        strategy_returns[date] = float(base_return.iloc[0])+ factor_component + idiosyncratic
                    
                    # Add to backtest dataframe
                    backtest_data['strategy_return'] = strategy_returns
                    backtest_data['benchmark_return'] = benchmark_returns.fillna(0.).reset_index(drop=True)
                    
                    # Calculate cumulative returns
                    backtest_data['strategy_cumulative'] = (1 + backtest_data['strategy_return']).cumprod()
                    backtest_data['benchmark_cumulative'] = (1 + backtest_data['benchmark_return']).cumprod()
                    
                    # Calculate drawdowns
                    backtest_data['strategy_drawdown'] = 0.0
                    backtest_data['benchmark_drawdown'] = 0.0
                    
                    # Calculate rolling peak
                    backtest_data['strategy_peak'] = backtest_data['strategy_cumulative'].cummax()
                    backtest_data['benchmark_peak'] = backtest_data['benchmark_cumulative'].cummax()
                    
                    # Calculate drawdown
                    backtest_data['strategy_drawdown'] = (backtest_data['strategy_peak'] - backtest_data['strategy_cumulative']) / backtest_data['strategy_peak']
                    backtest_data['benchmark_drawdown'] = (backtest_data['benchmark_peak'] - backtest_data['benchmark_cumulative']) / backtest_data['benchmark_peak']
                    
                    # Calculate rolling alpha and beta (90-day window)
                    backtest_data['rolling_beta'] = 0.0
                    backtest_data['rolling_alpha'] = 0.0
                    
                    for i in range(90, len(backtest_data)):
                        window_strategy = backtest_data['strategy_return'].iloc[i-90:i]
                        window_benchmark = backtest_data['benchmark_return'].iloc[i-90:i]
                        
                        # Calculate beta and alpha
                        cov_matrix = np.cov(window_strategy, window_benchmark)
                        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
                        alpha = window_strategy.mean() - beta * window_benchmark.mean()
                        
                        # Annualize alpha
                        alpha_annual = alpha * 252
                        
                        backtest_data['rolling_beta'].iloc[i] = beta
                        backtest_data['rolling_alpha'].iloc[i] = alpha_annual
                    
                    # Add date column
                    backtest_data['date'] = trading_days
                    
                    # Save backtest data
                    backtest_file = os.path.join(backtest_dir, f'{index_id}_{factor}_backtest.parquet')
                    backtest_data.to_parquet(backtest_file)
                    
                    # Calculate performance metrics
                    annualized_return = backtest_data['strategy_return'].mean() * 252
                    annualized_vol = backtest_data['strategy_return'].std() * np.sqrt(252)
                    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
                    
                    benchmark_return = backtest_data['benchmark_return'].mean() * 252
                    benchmark_vol = backtest_data['benchmark_return'].std() * np.sqrt(252)
                    benchmark_sharpe = benchmark_return / benchmark_vol if benchmark_vol > 0 else 0
                    
                    # Calculate tracking error
                    tracking_error = (backtest_data['strategy_return'] - backtest_data['benchmark_return']).std() * np.sqrt(252)
                    information_ratio = (annualized_return - benchmark_return) / tracking_error if tracking_error > 0 else 0
                    
                    # Calculate max drawdown
                    max_drawdown = backtest_data['strategy_drawdown'].max()
                    benchmark_max_drawdown = backtest_data['benchmark_drawdown'].max()
                    
                    # Store backtest metadata
                    cursor.execute('''
                    INSERT OR REPLACE INTO backtest_strategies 
                    (index_id, factor, name, description, annualized_return, sharpe_ratio, information_ratio,
                    max_drawdown, benchmark_return, benchmark_sharpe, benchmark_max_drawdown,
                    annualized_volatility, benchmark_volatility)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        index_id,
                        factor,
                        f"{factor.capitalize()} Strategy",
                        f"{factor.capitalize()} factor-based investment strategy",
                        float(annualized_return),
                        float(sharpe_ratio),
                        float(information_ratio),
                        float(max_drawdown),
                        float(benchmark_return),
                        float(benchmark_sharpe),
                        float(benchmark_max_drawdown),
                        float(annualized_vol),
                        float(benchmark_vol)
                    ))
                    
            conn.commit()
            conn.close()
            
            logger.info("Demo data generation complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error generating demo data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
if __name__=="__main__":
    db = DatabaseManager2()
    success = db.generate_demo_data()
    if success:
        print("Data generation status: success")
    else:
        print("Data generation status: failed") 