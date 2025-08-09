import os
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Optional, Union, Tuple
import logging
import itertools
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresManager:
    """
    Database manager for the Equity Trading Platform using PostgreSQL.
    Handles storage and retrieval of market data, index constituents, and other financial information.
    
    Uses PostgreSQL for metadata and relational data, and Parquet files for time series data.
    """
    
    def __init__(self, db_host: str = 'localhost', 
                 db_port: int = 5432,
                 db_name: str = 'market_data',
                 db_user: str = 'postgres',
                 db_password: str = 'password',
                 ts_data_path: str = 'data/time_series'):
        """
        Initialize the database manager.
        
        Args:
            db_host: PostgreSQL server host
            db_port: PostgreSQL server port
            db_name: PostgreSQL database name
            db_user: PostgreSQL username
            db_password: PostgreSQL password
            ts_data_path: Path to store time series data as Parquet files
        """
        self.db_params = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }
        self.ts_data_path = ts_data_path
        
        # Create directory for time series data
        os.makedirs(ts_data_path, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _get_connection(self):
        """Get a connection to the PostgreSQL database"""
        return psycopg2.connect(**self.db_params)
        
    def _init_database(self):
        """Initialize the PostgreSQL database with required tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
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
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
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
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO indices (index_id, name, description, region, last_updated)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (index_id) DO UPDATE 
            SET name = EXCLUDED.name,
                description = EXCLUDED.description,
                region = EXCLUDED.region,
                last_updated = EXCLUDED.last_updated
            ''', (index_id, name, description, region, datetime.datetime.now()))
            
            conn.commit()
            conn.close()
            
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
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO securities (ticker, name, sector, industry, description)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE 
            SET name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                description = EXCLUDED.description
            ''', (ticker, name, sector, industry, description))
            
            conn.commit()
            conn.close()
            
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
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # First, ensure all securities exist in the securities table
            for ticker in constituents_df['ticker'].unique():
                cursor.execute('SELECT ticker FROM securities WHERE ticker = %s', (ticker,))
                if not cursor.fetchone():
                    # Add the security with minimal info if it doesn't exist
                    cursor.execute('''
                    INSERT INTO securities (ticker, name) VALUES (%s, %s)
                    ''', (ticker, ticker))
            
            # Clear existing constituents for this index
            cursor.execute('''
            DELETE FROM index_constituents WHERE index_id = %s
            ''', (index_id,))
            
            # Insert new constituents
            data = [(index_id, row['ticker'], row['weight'], row['entry_date'], row['exit_date']) 
                    for _, row in constituents_df.iterrows()]
            
            execute_values(cursor, '''
            INSERT INTO index_constituents (index_id, ticker, weight, entry_date, exit_date)
            VALUES %s
            ''', data)
            
            conn.commit()
            conn.close()
            
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
            conn = self._get_connection()
            query = "SELECT * FROM indices"
            indices_df = pd.read_sql_query(query, conn)
            conn.close()
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
            conn = self._get_connection()
            
            if as_of_date:
                query = """
                SELECT ic.*, s.name, s.sector, s.industry
                FROM index_constituents ic
                JOIN securities s ON ic.ticker = s.ticker
                WHERE ic.index_id = %s
                AND (ic.entry_date <= %s OR ic.entry_date IS NULL)
                AND (ic.exit_date >= %s OR ic.exit_date IS NULL)
                """
                constituents_df = pd.read_sql_query(query, conn, params=(index_id, as_of_date, as_of_date))
            else:
                query = """
                SELECT ic.*, s.name, s.sector, s.industry
                FROM index_constituents ic
                JOIN securities s ON ic.ticker = s.ticker
                WHERE ic.index_id = %s
                """
                constituents_df = pd.read_sql_query(query, conn, params=(index_id,))
                
            conn.close()
            return constituents_df
            
        except Exception as e:
            logger.error(f"Error retrieving constituents for {index_id}: {str(e)}")
            return pd.DataFrame()
    
    def store_price_data(self, ticker: str, price_data: pd.DataFrame) -> bool:
        """
        Store price data for a security.
        Uses same Parquet file approach as the SQLite version.
        
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
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE securities 
                SET first_date = %s, last_date = %s
                WHERE ticker = %s
                ''', (str(price_data.index.min().date()), 
                      str(price_data.index.max().date()), 
                      ticker))
                
                conn.commit()
                conn.close()
            
            logger.info(f"Stored price data for {ticker}, {len(price_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data for {ticker}: {str(e)}")
            return False
    
    def get_price_data(self, ticker: str, start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """
        Retrieve price data for a security.
        Same implementation as SQLite version since it uses Parquet files.
        
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
            if ('date' not in price_data.index.names) & ('date' in price_data.columns):
                price_data = price_data.set_index('date')
                
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
            # Save metadata to PostgreSQL
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if factor exists, add if not
            cursor.execute('SELECT factor_id FROM factors WHERE factor_id = %s', (factor_id,))
            if not cursor.fetchone():
                cursor.execute('''
                INSERT INTO factors (factor_id, name, last_updated)
                VALUES (%s, %s, %s)
                ''', (factor_id, factor_id, datetime.datetime.now()))
            
            # Clear existing factor exposures for this factor
            cursor.execute('''
            DELETE FROM factor_exposures WHERE factor_id = %s
            ''', (factor_id,))
            
            # Insert new factor exposures
            data = [(factor_id, row['ticker'], row['date'], row['value']) 
                    for _, row in factor_data.iterrows()]
            
            # Use execute_values for efficient bulk insertion
            execute_values(cursor, '''
            INSERT INTO factor_exposures (factor_id, ticker, date, value)
            VALUES %s
            ''', data)
            
            conn.commit()
            conn.close()
            
            # Also save as parquet for efficient time series access
            factor_path = os.path.join(self.ts_data_path, 'factors')
            os.makedirs(factor_path, exist_ok=True)
            
            file_path = os.path.join(factor_path, f"{factor_id}.parquet")
            factor_data.to_parquet(file_path)
            
            logger.info(f"Stored factor data for {factor_id}, {len(factor_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error storing factor data for {factor_id}: {str(e)}")
            return False
    
    def get_factor_data(self, factor_id: str, tickers: List[str] = None,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve factor data.
        Same implementation as SQLite version since it uses Parquet files.
        
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
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT factor_id FROM factors')
            factors = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return factors
            
        except Exception as e:
            logger.error(f"Error retrieving factors: {str(e)}")
            return []

    def store_returns_data(self, returns_df: pd.DataFrame, source: str = 'daily') -> bool:
        """
        Store returns data for multiple securities.
        Same implementation as SQLite version since it uses Parquet files.
        
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
        Same implementation as SQLite version since it uses Parquet files.
        
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
        Reset the database schema (use with caution!)
        
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Drop tables in correct order to avoid foreign key constraints
            tables = ['factor_exposures', 'index_constituents', 'factors', 'securities', 'indices']
            
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            
            conn.commit()
            conn.close()
            
            # Recreate database structure
            self._init_database()
            
            logger.warning("Database has been reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            return False
        
if __name__=="__main__":
    db = PostgresManager()
    # success = db.generate_demo_data()
    # if success:
    #     print("Data generation status: success")
    # else:
    #     print("Data generation status: failed")