# db_manager.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Manager for Factor Model Framework

This module provides a database manager for storing and retrieving factor data,
security information, and backtest results in a PostgreSQL database.
"""

import os
import logging
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Tuple, Any
import uuid

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConfig(BaseModel):
    """Configuration for database connection"""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    username: str
    password: str
    database: str
    db_schema: str = Field(default="public")
    
    def get_connection_string(self) -> str:
        """Get the PostgreSQL connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class DatabaseManager:
    """Manager for database operations related to the factor model framework"""
    
    def __init__(self, config: Union[Dict, DatabaseConfig]):
        """
        Initialize database manager with configuration
        
        Args:
            config: Database configuration as a dictionary or DatabaseConfig object
        """
        if isinstance(config, dict):
            self.config = DatabaseConfig(**config)
        else:
            self.config = config
            
        self.engine = None
        self.Session = None
        self.metadata = sa.MetaData(schema=self.config.db_schema)
        self.connect()
        
    def connect(self) -> None:
        """Establish database connection"""
        try:
            conn_str = self.config.get_connection_string()
            self.engine = create_engine(conn_str)
            self.Session = sessionmaker(bind=self.engine)
            self.metadata.reflect(bind=self.engine)
            logger.info("Successfully connected to database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
            
    def execute_script(self, script_path: str) -> None:
        """
        Execute a SQL script file
        
        Args:
            script_path: Path to the SQL script file
        """
        try:
            with open(script_path, 'r') as f:
                sql_script = f.read()
                
            with self.engine.connect() as conn:
                conn.execute(text(sql_script))
                conn.commit()
                logger.info(f"Successfully executed SQL script: {script_path}")
        except Exception as e:
            logger.error(f"Error executing SQL script: {str(e)}")
            raise
            
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """
        Execute a SQL query and return results as a list of dictionaries
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return [dict(row._mapping) for row in result] if result.rowcount > 0 else []
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """
        Insert a pandas DataFrame into a database table
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: How to behave if the table already exists ('fail', 'replace', 'append')
        """
        try:
            df.to_sql(table_name, self.engine, schema=self.config.db_schema, 
                      if_exists=if_exists, index=False)
            logger.info(f"Successfully inserted {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Error inserting data into {table_name}: {str(e)}")
            raise
            
    def get_dataframe(self, query: str, params: Dict = None) -> pd.DataFrame:
        """
        Execute a query and return results as a pandas DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params or {})
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
            
    def store_securities_v0(self, securities_df: pd.DataFrame, update_existing: bool = True) -> None:
        """
        Store security information in the database
        
        Args:
            securities_df: DataFrame with security information
            update_existing: Whether to update existing records
        """
        required_columns = {'security_id', 'name'}
        if not all(col in securities_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(securities_df.columns)}")
        
        # Add timestamps if not present
        if 'created_at' not in securities_df:
            securities_df['created_at'] = datetime.now()
            
        if 'updated_at' not in securities_df:
            securities_df['updated_at'] = datetime.now()
            
        if update_existing:
            # For each row, insert if not exists, otherwise update
            for _, row in securities_df.iterrows():
                query = """
                INSERT INTO securities (security_id, name, asset_class, sector, sub_sector, 
                                       currency, exchange, created_at, updated_at)
                VALUES (:security_id, :name, :asset_class, :sector, :sub_sector, 
                        :currency, :exchange, :created_at, :updated_at)
                ON CONFLICT (security_id) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    asset_class = EXCLUDED.asset_class,
                    sector = EXCLUDED.sector,
                    sub_sector = EXCLUDED.sub_sector,
                    currency = EXCLUDED.currency,
                    exchange = EXCLUDED.exchange,
                    updated_at = CURRENT_TIMESTAMP;
                """
                self.execute_query(query, row.to_dict())
        else:
            # Just insert, skip if exists
            self.insert_dataframe(securities_df, 'securities', if_exists='append')

    def store_securities(self, securities_df: pd.DataFrame, update_existing: bool = True) -> None:
        """
        Store security information in the database
        
        Args:
            securities_df: DataFrame with security information
            update_existing: Whether to update existing records
        """
        required_columns = {'security_id', 'name'}
        if not all(col in securities_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(securities_df.columns)}")
        
        # Add timestamps if not present
        if 'created_at' not in securities_df:
            securities_df['created_at'] = datetime.now()
        
        if 'updated_at' not in securities_df:
            securities_df['updated_at'] = datetime.now()
        
        # Set default values for missing fields
        if 'asset_class' not in securities_df:
            securities_df['asset_class'] = 'Equities'
        else:
            securities_df['asset_class'] = securities_df['asset_class'].fillna('Equities')
        
        if 'currency' not in securities_df:
            securities_df['currency'] = 'USD'
        else:
            securities_df['currency'] = securities_df['currency'].fillna('USD')
        
        if 'exchange' not in securities_df:
            securities_df['exchange'] = None
        
        query = """
        INSERT INTO securities (security_id, name, asset_class, sector, sub_sector,
                            currency, exchange, created_at, updated_at)
        VALUES (:security_id, :name, :asset_class, :sector, :sub_sector,
                :currency, :exchange, :created_at, :updated_at)
        ON CONFLICT (security_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            asset_class = EXCLUDED.asset_class,
            sector = EXCLUDED.sector,
            sub_sector = EXCLUDED.sub_sector,
            currency = EXCLUDED.currency,
            exchange = EXCLUDED.exchange,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        try:
            with self.engine.connect() as conn:
                for _, row in securities_df.iterrows():
                    # Convert row to dictionary and handle missing values
                    row_dict = row.to_dict()
                    # Ensure all required fields are present
                    for field in ['sector', 'sub_sector']:
                        if field not in row_dict or pd.isna(row_dict[field]):
                            row_dict[field] = None
                    
                    conn.execute(text(query), row_dict)
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {len(securities_df)} securities")
        except Exception as e:
            logger.error(f"Error storing securities: {str(e)}")
            raise

    def store_universe(self, universe_id: str, name: str, description: str = None) -> None:
        """
        Store universe information in the database
        
        Args:
            universe_id: Universe ID (e.g., 'NDX Index')
            name: Universe name
            description: Universe description
        """
        query = """
        INSERT INTO universes (universe_id, name, description)
        VALUES (:universe_id, :name, :description)
        ON CONFLICT (universe_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            updated_at = CURRENT_TIMESTAMP;
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(query), 
                    {"universe_id": universe_id, "name": name, "description": description}
                )
                conn.commit()  # Add this line to commit the transaction
                logger.info(f"Stored universe: {universe_id}")
        except Exception as e:
            logger.error(f"Error storing universe: {str(e)}")
            raise
        
    def store_universe_constituents_v0(self, universe_id: str, constituents_df: pd.DataFrame) -> None:
        """
        Store universe constituents in the database
        
        Args:
            universe_id: Universe ID
            constituents_df: DataFrame with constituent information (security_id, weight, as_of_date)
        """
        required_columns = {'security_id', 'as_of_date'}
        if not all(col in constituents_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(constituents_df.columns)}")
        
        # Add universe_id column
        constituents_df['universe_id'] = universe_id
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(constituents_df['as_of_date']):
            constituents_df['as_of_date'] = pd.to_datetime(constituents_df['as_of_date'])
        
        # Use bulk insert for better performance
        try:
            self.insert_dataframe(constituents_df, 'universe_constituents', if_exists='append')
        except SQLAlchemyError as e:
            # If bulk insert fails, fall back to row-by-row insert with conflict handling
            logger.warning(f"Bulk insert failed, falling back to row-by-row insert: {str(e)}")
            for _, row in constituents_df.iterrows():
                query = """
                INSERT INTO universe_constituents (universe_id, security_id, weight, as_of_date)
                VALUES (:universe_id, :security_id, :weight, :as_of_date)
                ON CONFLICT (universe_id, security_id, as_of_date) 
                DO UPDATE SET 
                    weight = EXCLUDED.weight;
                """
                self.execute_query(query, row.to_dict())
                
        logger.info(f"Stored {len(constituents_df)} constituents for universe: {universe_id}")

    def store_universe_constituents(self, universe_id: str, constituents_df: pd.DataFrame) -> None:
        """
        Store universe constituents in the database
        
        Args:
            universe_id: Universe ID
            constituents_df: DataFrame with constituent information (security_id, weight, as_of_date)
        """
        required_columns = {'security_id', 'as_of_date'}
        if not all(col in constituents_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(constituents_df.columns)}")
        
        # Add universe_id column
        if 'universe_id' not in constituents_df.columns:
            constituents_df['universe_id'] = universe_id
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(constituents_df['as_of_date']):
            constituents_df['as_of_date'] = pd.to_datetime(constituents_df['as_of_date'])
        
        # Set default weight if not present
        if 'weight' not in constituents_df:
            constituents_df['weight'] = 0.0
        else:
            constituents_df['weight'] = constituents_df['weight'].fillna(0.0)
        
        query = """
        INSERT INTO universe_constituents (universe_id, security_id, weight, as_of_date)
        VALUES (:universe_id, :security_id, :weight, :as_of_date)
        ON CONFLICT (universe_id, security_id, as_of_date) 
        DO UPDATE SET 
            weight = EXCLUDED.weight;
        """
        
        try:
            with self.engine.connect() as conn:
                for _, row in constituents_df.iterrows():
                    # Convert row to dictionary
                    row_dict = row.to_dict()
                    
                    # Handle timestamp conversion for SQLAlchemy if needed
                    if isinstance(row_dict['as_of_date'], pd.Timestamp):
                        row_dict['as_of_date'] = row_dict['as_of_date'].to_pydatetime()
                    
                    conn.execute(text(query), row_dict)
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {len(constituents_df)} constituents for universe: {universe_id}")
        except Exception as e:
            logger.error(f"Error storing universe constituents: {str(e)}")
            raise

    def store_factor_metadata_v0(self, factor_id: str, name: str, description: str = None, 
                             category: str = None) -> None:
        """
        Store factor metadata in the database
        
        Args:
            factor_id: Factor ID (e.g., 'momentum')
            name: Factor name
            description: Factor description
            category: Factor category (e.g., 'Technical', 'Fundamental')
        """
        query = """
        INSERT INTO factors (factor_id, name, description, category)
        VALUES (:factor_id, :name, :description, :category)
        ON CONFLICT (factor_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            updated_at = CURRENT_TIMESTAMP;
        """
        self.execute_query(query, {
            'factor_id': factor_id,
            'name': name,
            'description': description,
            'category': category
        })
        logger.info(f"Stored factor metadata: {factor_id}")
    
    def store_factor_metadata(self, factor_id: str, name: str, description: str = None, 
                         category: str = None) -> None:
        """
        Store factor metadata in the database
        
        Args:
            factor_id: Factor ID (e.g., 'momentum')
            name: Factor name
            description: Factor description
            category: Factor category (e.g., 'Technical', 'Fundamental')
        """
        query = """
        INSERT INTO factors (factor_id, name, description, category)
        VALUES (:factor_id, :name, :description, :category)
        ON CONFLICT (factor_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(query), 
                    {
                        'factor_id': factor_id,
                        'name': name,
                        'description': description,
                        'category': category
                    }
                )
                conn.commit()  # Commit the transaction
                logger.info(f"Stored factor metadata: {factor_id}")
        except Exception as e:
            logger.error(f"Error storing factor metadata: {str(e)}")
            raise

    def store_factor_data_v0(self, factor_id: str, factor_df: pd.DataFrame, 
                         normalization_method: str = None) -> None:
        """
        Store factor data in the database
        
        Args:
            factor_id: Factor ID
            factor_df: DataFrame with factor data (security_id, date, value)
            normalization_method: Method used for normalization
        """
        required_columns = {'security_id', 'date', 'value'}
        if not all(col in factor_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(factor_df.columns)}")
        
        # Add factor_id column
        factor_df['factor_id'] = factor_id
        
        # Add normalization_method column if provided
        if normalization_method:
            factor_df['normalization_method'] = normalization_method
            
        # Add timestamp
        factor_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(factor_df['date']):
            factor_df['date'] = pd.to_datetime(factor_df['date'])
        
        # Use bulk insert with conflict handling
        try:
            # Prepare temp table for bulk insert and update
            temp_table_name = f"tmp_factor_data_{uuid.uuid4().hex[:8]}"
            factor_df.to_sql(temp_table_name, self.engine, schema=self.config.db_schema, 
                           if_exists='replace', index=False)
            
            # Upsert from temp table to factor_data
            query = f"""
            INSERT INTO factor_data (factor_id, security_id, date, value, normalized_value, 
                                   normalization_method, created_at)
            SELECT factor_id, security_id, date, value, normalized_value, 
                   normalization_method, created_at
            FROM {self.config.db_schema}.{temp_table_name}
            ON CONFLICT (factor_id, security_id, date) 
            DO UPDATE SET 
                value = EXCLUDED.value,
                normalized_value = EXCLUDED.normalized_value,
                normalization_method = EXCLUDED.normalization_method,
                created_at = EXCLUDED.created_at;
            """
            self.execute_query(query)
            
            # Drop temp table
            self.execute_query(f"DROP TABLE {self.config.db_schema}.{temp_table_name};")
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing factor data: {str(e)}")
            # Fall back to row-by-row insert
            for _, row in factor_df.iterrows():
                query = """
                INSERT INTO factor_data (factor_id, security_id, date, value, normalized_value, 
                                       normalization_method, created_at)
                VALUES (:factor_id, :security_id, :date, :value, :normalized_value, 
                        :normalization_method, :created_at)
                ON CONFLICT (factor_id, security_id, date) 
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    normalized_value = EXCLUDED.normalized_value,
                    normalization_method = EXCLUDED.normalization_method,
                    created_at = EXCLUDED.created_at;
                """
                self.execute_query(query, row.to_dict())
                
        logger.info(f"Stored {len(factor_df)} factor data points for factor: {factor_id}")

    def store_factor_data(self, factor_id: str, factor_df: pd.DataFrame, 
                     normalization_method: str = None) -> None:
        """
        Store factor data in the database
        
        Args:
            factor_id: Factor ID
            factor_df: DataFrame with factor data (security_id, date, value)
            normalization_method: Method used for normalization
        """
        required_columns = {'security_id', 'date', 'value'}
        if not all(col in factor_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(factor_df.columns)}")
        
        # Add factor_id column
        factor_df['factor_id'] = factor_id
        
        # Add normalization_method column if provided
        if normalization_method:
            factor_df['normalization_method'] = normalization_method
        else:
            factor_df['normalization_method'] = None
        
        # Add timestamp
        factor_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(factor_df['date']):
            factor_df['date'] = pd.to_datetime(factor_df['date'])
        
        # Add normalized_value column if missing
        if 'normalized_value' not in factor_df:
            factor_df['normalized_value'] = factor_df['value']
        
        query = """
        INSERT INTO factor_data (factor_id, security_id, date, value, normalized_value, 
                            normalization_method, created_at)
        VALUES (:factor_id, :security_id, :date, :value, :normalized_value, 
                :normalization_method, :created_at)
        ON CONFLICT (factor_id, security_id, date) 
        DO UPDATE SET 
            value = EXCLUDED.value,
            normalized_value = EXCLUDED.normalized_value,
            normalization_method = EXCLUDED.normalization_method,
            created_at = EXCLUDED.created_at;
        """
        
        try:
            with self.engine.connect() as conn:
                # Process in batches to improve performance while maintaining transaction safety
                batch_size = 1000
                total_rows = len(factor_df)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = factor_df.iloc[start_idx:end_idx]
                    
                    for _, row in batch.iterrows():
                        # Convert row to dictionary
                        row_dict = row.to_dict()
                        
                        # Handle timestamp conversion for SQLAlchemy if needed
                        if isinstance(row_dict['date'], pd.Timestamp):
                            row_dict['date'] = row_dict['date'].to_pydatetime()
                        
                        # Handle NaN values
                        for key, value in row_dict.items():
                            if pd.isna(value):
                                row_dict[key] = None
                        
                        conn.execute(text(query), row_dict)
                    
                    # Log progress for large datasets
                    if total_rows > batch_size:
                        logger.debug(f"Processed {min(end_idx, total_rows)}/{total_rows} factor data rows")
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {total_rows} factor data points for factor: {factor_id}")
        except Exception as e:
            logger.error(f"Error storing factor data: {str(e)}")
            raise

    def store_exposures(self, exposures_df: pd.DataFrame) -> None:
        """
        Store factor exposures in the database
        
        Args:
            exposures_df: DataFrame with exposure data (date, universe, security_id, variable, exposure)
        """
        required_columns = {'date', 'universe', 'security_id', 'variable', 'exposure'}
        if not all(col in exposures_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(exposures_df.columns)}")
        
        # Add timestamp
        exposures_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(exposures_df['date']):
            exposures_df['date'] = pd.to_datetime(exposures_df['date'])
        
        query = """
        INSERT INTO exposures (date, universe, security_id, variable, exposure, created_at)
        VALUES (:date, :universe, :security_id, :variable, :exposure, :created_at)
        ON CONFLICT (date, universe, security_id, variable) 
        DO UPDATE SET 
            exposure = EXCLUDED.exposure,
            created_at = EXCLUDED.created_at;
        """
        
        try:
            with self.engine.connect() as conn:
                # Process in batches to improve performance while maintaining transaction safety
                batch_size = 1000
                total_rows = len(exposures_df)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = exposures_df.iloc[start_idx:end_idx]
                    
                    for _, row in batch.iterrows():
                        # Convert row to dictionary
                        row_dict = row.to_dict()
                        
                        # Handle timestamp conversion for SQLAlchemy
                        if isinstance(row_dict['date'], pd.Timestamp):
                            row_dict['date'] = row_dict['date'].to_pydatetime()
                        
                        # Handle NaN values
                        if pd.isna(row_dict['exposure']):
                            row_dict['exposure'] = 0.0
                        
                        conn.execute(text(query), row_dict)
                    
                    # Log progress for large datasets
                    if total_rows > batch_size:
                        logger.debug(f"Processed {min(end_idx, total_rows)}/{total_rows} exposure rows")
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {total_rows} exposure records")
        except Exception as e:
            logger.error(f"Error storing exposures: {str(e)}")
            raise

    def store_security_prices_v0(self, prices_df: pd.DataFrame) -> None:
        """
        Store security price data in the database
        
        Args:
            prices_df: DataFrame with price data (security_id, date, open_price, high_price, 
                      low_price, close_price, adjusted_close, volume)
        """
        required_columns = {'security_id', 'date', 'close_price'}
        if not all(col in prices_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(prices_df.columns)}")
        
        # Add timestamp
        prices_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(prices_df['date']):
            prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # Use bulk insert with conflict handling
        try:
            # Prepare temp table for bulk insert and update
            temp_table_name = f"tmp_security_prices_{uuid.uuid4().hex[:8]}"
            prices_df.to_sql(temp_table_name, self.engine, schema=self.config.db_schema, 
                           if_exists='replace', index=False)
            
            # Upsert from temp table to security_prices
            query = f"""
            INSERT INTO security_prices (security_id, date, open_price, high_price, low_price, 
                                       close_price, adjusted_close, volume, created_at)
            SELECT security_id, date, open_price, high_price, low_price, 
                   close_price, adjusted_close, volume, created_at
            FROM {self.config.db_schema}.{temp_table_name}
            ON CONFLICT (security_id, date) 
            DO UPDATE SET 
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                adjusted_close = EXCLUDED.adjusted_close,
                volume = EXCLUDED.volume,
                created_at = EXCLUDED.created_at;
            """
            self.execute_query(query)
            
            # Drop temp table
            self.execute_query(f"DROP TABLE {self.config.db_schema}.{temp_table_name};")
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing security prices: {str(e)}")
            # Fall back to row-by-row insert
            for _, row in prices_df.iterrows():
                query = """
                INSERT INTO security_prices (security_id, date, open_price, high_price, low_price, 
                                           close_price, adjusted_close, volume, created_at)
                VALUES (:security_id, :date, :open_price, :high_price, :low_price, 
                        :close_price, :adjusted_close, :volume, :created_at)
                ON CONFLICT (security_id, date) 
                DO UPDATE SET 
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    adjusted_close = EXCLUDED.adjusted_close,
                    volume = EXCLUDED.volume,
                    created_at = EXCLUDED.created_at;
                """
                self.execute_query(query, row.to_dict())
                
        logger.info(f"Stored {len(prices_df)} price records")
    
    def store_security_prices(self, prices_df: pd.DataFrame) -> None:
        """
        Store security price data in the database
        
        Args:
            prices_df: DataFrame with price data (security_id, date, open_price, high_price, 
                    low_price, close_price, adjusted_close, volume)
        """
        required_columns = {'security_id', 'date', 'close_price'}
        if not all(col in prices_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(prices_df.columns)}")
        
        # Add timestamp
        prices_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(prices_df['date']):
            prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # Set default values for optional columns if missing
        optional_columns = {
            'open_price': None, 
            'high_price': None, 
            'low_price': None, 
            'adjusted_close': None, 
            'volume': None
        }
        
        for col, default_val in optional_columns.items():
            if col not in prices_df:
                prices_df[col] = default_val
        
        query = """
        INSERT INTO security_prices (security_id, date, open_price, high_price, low_price, 
                                close_price, adjusted_close, volume, created_at)
        VALUES (:security_id, :date, :open_price, :high_price, :low_price, 
                :close_price, :adjusted_close, :volume, :created_at)
        ON CONFLICT (security_id, date) 
        DO UPDATE SET 
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume,
            created_at = EXCLUDED.created_at;
        """
        
        try:
            with self.engine.connect() as conn:
                # Process in batches to improve performance while maintaining transaction safety
                batch_size = 1000
                total_rows = len(prices_df)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = prices_df.iloc[start_idx:end_idx]
                    
                    for _, row in batch.iterrows():
                        # Convert row to dictionary
                        row_dict = row.to_dict()
                        
                        # Handle timestamp conversion for SQLAlchemy
                        if isinstance(row_dict['date'], pd.Timestamp):
                            row_dict['date'] = row_dict['date'].to_pydatetime()
                        
                        # Handle NaN values
                        for key, value in row_dict.items():
                            if pd.isna(value):
                                row_dict[key] = None
                        
                        conn.execute(text(query), row_dict)
                    
                    # Log progress for large datasets
                    if total_rows > batch_size:
                        logger.debug(f"Processed {min(end_idx, total_rows)}/{total_rows} price records")
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {total_rows} price records")
        except Exception as e:
            logger.error(f"Error storing security prices: {str(e)}")
            raise

    def store_security_returns_v0(self, returns_df: pd.DataFrame) -> None:
        """
        Store security return data in the database
        
        Args:
            returns_df: DataFrame with return data (security_id, date, daily_return)
        """
        required_columns = {'security_id', 'date', 'daily_return'}
        if not all(col in returns_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(returns_df.columns)}")
        
        # Add timestamp
        returns_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(returns_df['date']):
            returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        # Use bulk insert with conflict handling
        try:
            # Prepare temp table for bulk insert and update
            temp_table_name = f"tmp_security_returns_{uuid.uuid4().hex[:8]}"
            returns_df.to_sql(temp_table_name, self.engine, schema=self.config.db_schema, 
                            if_exists='replace', index=False)
            
            # Upsert from temp table to security_returns
            query = f"""
            INSERT INTO security_returns (security_id, date, daily_return, created_at)
            SELECT security_id, date, daily_return, created_at
            FROM {self.config.db_schema}.{temp_table_name}
            ON CONFLICT (security_id, date) 
            DO UPDATE SET 
                daily_return = EXCLUDED.daily_return,
                created_at = EXCLUDED.created_at;
            """
            self.execute_query(query)
            
            # Drop temp table
            self.execute_query(f"DROP TABLE {self.config.db_schema}.{temp_table_name};")
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing security returns: {str(e)}")
            # Fall back to row-by-row insert
            for _, row in returns_df.iterrows():
                query = """
                INSERT INTO security_returns (security_id, date, daily_return, created_at)
                VALUES (:security_id, :date, :daily_return, :created_at)
                ON CONFLICT (security_id, date) 
                DO UPDATE SET 
                    daily_return = EXCLUDED.daily_return,
                    created_at = EXCLUDED.created_at;
                """
                self.execute_query(query, row.to_dict())
                
        logger.info(f"Stored {len(returns_df)} return records")
    
    def store_security_returns(self, returns_df: pd.DataFrame) -> None:
        """
        Store security return data in the database
        
        Args:
            returns_df: DataFrame with return data (security_id, date, daily_return)
        """
        required_columns = {'security_id', 'date', 'daily_return'}
        if not all(col in returns_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(returns_df.columns)}")
        
        # Add timestamp
        returns_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(returns_df['date']):
            returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        query = """
        INSERT INTO security_returns (security_id, date, daily_return, created_at)
        VALUES (:security_id, :date, :daily_return, :created_at)
        ON CONFLICT (security_id, date) 
        DO UPDATE SET 
            daily_return = EXCLUDED.daily_return,
            created_at = EXCLUDED.created_at;
        """
        
        try:
            with self.engine.connect() as conn:
                # Process in batches to improve performance while maintaining transaction safety
                batch_size = 1000
                total_rows = len(returns_df)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = returns_df.iloc[start_idx:end_idx]
                    
                    for _, row in batch.iterrows():
                        # Convert row to dictionary
                        row_dict = row.to_dict()
                        
                        # Handle timestamp conversion for SQLAlchemy
                        if isinstance(row_dict['date'], pd.Timestamp):
                            row_dict['date'] = row_dict['date'].to_pydatetime()
                        
                        # Handle NaN values
                        if pd.isna(row_dict['daily_return']):
                            row_dict['daily_return'] = 0.0
                        
                        conn.execute(text(query), row_dict)
                    
                    # Log progress for large datasets
                    if total_rows > batch_size:
                        logger.debug(f"Processed {min(end_idx, total_rows)}/{total_rows} return records")
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {total_rows} return records")
        except Exception as e:
            logger.error(f"Error storing security returns: {str(e)}")
            raise

    def store_backtest_config_v0(self, name: str, universe_id: str, start_date: Union[str, date], 
                             end_date: Union[str, date], rebalancing_frequency: str,
                             configuration: Dict) -> str:
        """
        Store backtest configuration in the database
        
        Args:
            name: Backtest name
            universe_id: Universe ID
            start_date: Start date
            end_date: End date
            rebalancing_frequency: Rebalancing frequency
            configuration: Full configuration as a dictionary
            
        Returns:
            Config ID
        """
        # Generate UUID for the config
        config_id = str(uuid.uuid4())
        
        # Convert dates to proper format
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
            
        # Convert configuration to JSON
        config_json = json.dumps(configuration)
        
        query = """
        INSERT INTO backtest_configs (config_id, name, universe_id, start_date, end_date, 
                                     rebalancing_frequency, configuration)
        VALUES (:config_id, :name, :universe_id, :start_date, :end_date, 
                :rebalancing_frequency, :configuration::jsonb)
        RETURNING config_id;
        """
        
        result = self.execute_query(query, {
            'config_id': config_id,
            'name': name,
            'universe_id': universe_id,
            'start_date': start_date,
            'end_date': end_date,
            'rebalancing_frequency': rebalancing_frequency,
            'configuration': config_json
        })
        
        logger.info(f"Stored backtest configuration: {name} with ID: {config_id}")
        return config_id

    def store_backtest_config(self, name: str, universe_id: str, start_date: Union[str, date], 
                         end_date: Union[str, date], rebalancing_frequency: str,
                         configuration: Dict) -> str:
        """
        Store backtest configuration in the database
        
        Args:
            name: Backtest name
            universe_id: Universe ID
            start_date: Start date
            end_date: End date
            rebalancing_frequency: Rebalancing frequency
            configuration: Full configuration as a dictionary
            
        Returns:
            Config ID
        """
        # Generate UUID for the config
        config_id = str(uuid.uuid4())
        
        # Convert dates to proper format
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
            
        # Convert configuration to JSON
        config_json = configuration.get('backtest') # configuration.keys() # maybe use: configuration.get('params')
        if 'dates_daily' in config_json.keys():
            config_json.pop('dates_daily')
        # config_json.pop('dates_turnover')
        config_json = json.dumps(config_json)

        # query = """
        # INSERT INTO backtest_configs (config_id, name, universe_id, start_date, end_date, 
        #                             rebalancing_frequency, configuration)
        # VALUES (:config_id, :name, :universe_id, :start_date, :end_date, 
        #         :rebalancing_frequency, :config_json::jsonb)
        # """
        query = """
        INSERT INTO backtest_configs (config_id, name, universe_id, start_date, end_date, 
                                    rebalancing_frequency)
        VALUES (:config_id, :name, :universe_id, :start_date, :end_date, 
                :rebalancing_frequency)
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text(query), 
                    {
                        'config_id': config_id,
                        'name': name,
                        'universe_id': universe_id,
                        'start_date': start_date,
                        'end_date': end_date,
                        'rebalancing_frequency': rebalancing_frequency
                        # ,'configuration': config_json
                    }
                )
                conn.commit()  # Add this line to commit the transaction
                logger.info(f"Stored backtest configuration: {name} with ID: {config_id}")
                return config_id 

        except Exception as e:
            logger.error(f"Error storing backtest configuration: {str(e)}")
            raise
            
    def store_portfolio_weights_v0(self, config_id: str, weights_df: pd.DataFrame) -> None:
        """
        Store portfolio weights in the database
        
        Args:
            config_id: Backtest configuration ID
            weights_df: DataFrame with weight data (date, security_id, weight)
        """
        required_columns = {'security_id', 'date', 'weight'}
        if not all(col in weights_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(weights_df.columns)}")
        
        # Add config_id column
        weights_df['config_id'] = config_id
        
        # Add timestamp
        weights_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(weights_df['date']):
            weights_df['date'] = pd.to_datetime(weights_df['date'])
        
        # Use bulk insert with conflict handling
        try:
            # Prepare temp table for bulk insert and update
            temp_table_name = f"tmp_portfolio_weights_{uuid.uuid4().hex[:8]}"
            weights_df.to_sql(temp_table_name, self.engine, schema=self.config.db_schema, 
                            if_exists='replace', index=False)
            
            # Upsert from temp table to portfolio_weights
            query = f"""
            INSERT INTO portfolio_weights (config_id, date, security_id, weight, created_at)
            SELECT config_id, date, security_id, weight, created_at
            FROM {self.config.db_schema}.{temp_table_name}
            ON CONFLICT (config_id, date, security_id) 
            DO UPDATE SET 
                weight = EXCLUDED.weight,
                created_at = EXCLUDED.created_at;
            """
            self.execute_query(query)
            
            # Drop temp table
            self.execute_query(f"DROP TABLE {self.config.db_schema}.{temp_table_name};")
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing portfolio weights: {str(e)}")
            # Fall back to row-by-row insert
    
    def store_portfolio_weights(self, config_id: str, weights_df: pd.DataFrame) -> None:
        """
        Store portfolio weights in the database
        
        Args:
            config_id: Backtest configuration ID
            weights_df: DataFrame with weight data (date, security_id, weight)
        """
        required_columns = {'security_id', 'date', 'weight'}
        if not all(col in weights_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns - set(weights_df.columns)}")
        
        # Add config_id column
        weights_df['config_id'] = config_id
        
        # Add timestamp
        weights_df['created_at'] = datetime.now()
        
        # Ensure date is in the right format
        if not pd.api.types.is_datetime64_dtype(weights_df['date']):
            weights_df['date'] = pd.to_datetime(weights_df['date'])
        
        query = """
        INSERT INTO portfolio_weights (config_id, date, security_id, weight, created_at)
        VALUES (:config_id, :date, :security_id, :weight, :created_at)
        ON CONFLICT (config_id, date, security_id) 
        DO UPDATE SET 
            weight = EXCLUDED.weight,
            created_at = EXCLUDED.created_at;
        """
        
        try:
            with self.engine.connect() as conn:
                # Process in batches to improve performance while maintaining transaction safety
                batch_size = 1000
                total_rows = len(weights_df)
                
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = weights_df.iloc[start_idx:end_idx]
                    
                    for _, row in batch.iterrows():
                        # Convert row to dictionary
                        row_dict = row.to_dict()
                        
                        # Handle timestamp conversion for SQLAlchemy
                        if isinstance(row_dict['date'], pd.Timestamp):
                            row_dict['date'] = row_dict['date'].to_pydatetime()
                        
                        # Handle NaN weights
                        if pd.isna(row_dict['weight']):
                            row_dict['weight'] = 0.0
                        
                        conn.execute(text(query), row_dict)
                    
                    # Log progress for large datasets
                    if total_rows > batch_size:
                        logger.debug(f"Processed {min(end_idx, total_rows)}/{total_rows} portfolio weight records")
                
                conn.commit()  # Commit the transaction
                logger.info(f"Stored {total_rows} portfolio weight records for config: {config_id}")
        except Exception as e:
            logger.error(f"Error storing portfolio weights: {str(e)}")
            raise
