# db_integration.py
# #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Integration Module for Factor Model Framework

This module provides integration between the qFactor framework and the PostgreSQL database,
allowing for storage and retrieval of factor data, security information, and backtest results.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
import pandas as pd
import numpy as np

# Import your qFactor modules
from qFactor import (
    Universe, RiskFactors, EquityFactorModelInput,
    EquityFactor, SecurityMasterFactory, YahooFactor, BQLFactor
)

# Import database manager
from db_manager import DatabaseManager, DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FactorDatabaseIntegration:
    """Integration class between qFactor framework and PostgreSQL database"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize integration with a database manager
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        
    def store_model_input(self, model_input: EquityFactorModelInput) -> str:
        """
        Store model input configuration in the database
        
        Args:
            model_input: EquityFactorModelInput instance
            
        Returns:
            config_id: Configuration ID in the database
        """
        # Store universe info first
        universe_id = model_input.backtest.universe.value
        universe_name = model_input.backtest.universe.name
        universe_desc = model_input.backtest.universe.description
        
        self.db_manager.store_universe(
            universe_id=universe_id,
            name=universe_name,
            description=universe_desc
        )
        
        # Store backtest config
        config_id = self.db_manager.store_backtest_config(
            name=f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            universe_id=universe_id,
            start_date=model_input.backtest.start_date,
            end_date=model_input.backtest.end_date,
            rebalancing_frequency=model_input.backtest.frequency.value,
            configuration=model_input.model_dump(mode='json')
        )
        
        logger.info(f"Stored model input with config ID: {config_id}")
        return config_id
        
    def store_security_master_data(self, security_master) -> None:
        """
        Store security master data in the database
        
        Args:
            security_master: SecurityMaster instance
            
        Raises:
            ValueError: If required data is missing or invalid
            SQLAlchemyError: If database operations fail
        """
        try:
            # 1. Store universe info first
            if not hasattr(security_master, 'universe'):
                raise ValueError("SecurityMaster instance must have a 'universe' attribute")
            from qFactor import UniverseMappingFactory    
            try:
                self.db_manager.store_universe(
                    universe_id=UniverseMappingFactory(source=security_master.source, universe=security_master.universe), #security_master.universe,
                    name=security_master.universe,
                    description=f"Universe for {security_master.universe}"
                )
                logger.info(f"Successfully stored universe: {security_master.universe}")
            except Exception as e:
                logger.error(f"Failed to store universe {security_master.universe}: {str(e)}")
                raise
            
            # 2. Store securities data
            if security_master.security_master is not None:
                try:
                    sec_df = security_master.security_master.copy()
                    if sec_df.empty:
                        logger.warning("Security master DataFrame is empty")
                        return
                        
                    securities_df = pd.DataFrame({
                        'security_id': sec_df['sid'],
                        'name': sec_df['name'],
                        'sector': sec_df['sector'] if 'sector' in sec_df.columns else None,
                        'sub_sector': sec_df['sub_sector'] if 'sub_sector' in sec_df.columns else None,
                        'exchange': sec_df['exchange'] if 'exchange' in sec_df.columns else None
                    })
                    
                    # Validate required columns
                    required_cols = {'security_id', 'name'}
                    missing_cols = required_cols - set(securities_df.columns)
                    if missing_cols:
                        raise ValueError(f"Missing required columns in securities data: {missing_cols}")
                    required_cols = list(required_cols)

                    self.db_manager.store_securities(securities_df)
                    logger.info(f"Successfully stored {len(securities_df)} securities")
                except Exception as e:
                    logger.error(f"Failed to store securities data: {str(e)}")
                    raise
            else:
                logger.warning("No security master data available to store")
            
            # 3. Store weights data for universe constituents
            if security_master.weights_data is not None:
                try:
                    weights_df = security_master.weights_data.copy()
                    if weights_df.empty:
                        logger.warning("Weights DataFrame is empty")
                        return
                        
                    # Prepare data for database
                    weights_df = weights_df.rename(columns={
                        'date': 'as_of_date'
                    })
                    
                    # Ensure required columns exist
                    if 'security_id' not in weights_df.columns:
                        if 'sid' not in weights_df.columns:
                            raise ValueError("Neither 'security_id' nor 'sid' column found in weights data")
                        weights_df['security_id'] = weights_df['sid']
                    
                    # Validate required columns
                    required_cols = {'security_id', 'as_of_date'}
                    missing_cols = required_cols - set(weights_df.columns)
                    if missing_cols:
                        raise ValueError(f"Missing required columns in weights data: {missing_cols}")
                    
                    # Ensure date format
                    if not pd.api.types.is_datetime64_dtype(weights_df['as_of_date']):
                        weights_df['as_of_date'] = pd.to_datetime(weights_df['as_of_date'])
                    
                    # Store universe constituents
                    self.db_manager.store_universe_constituents(
                        universe_id=UniverseMappingFactory(source=security_master.source, universe=security_master.universe), #security_master.universe,
                        constituents_df=weights_df[['security_id', 'weight', 'as_of_date', 'universe_id']]
                    )
                    logger.info(f"Successfully stored universe constituents from source {security_master.source.title()} for {security_master.universe}")
                except Exception as e:
                    logger.error(f"Failed to store universe constituents: {str(e)}")
                    raise
            else:
                logger.warning("No weights data available to store")
            
            # 4. Store price data
            if security_master.df_price is not None and not security_master.df_price.empty:
                try:
                    price_df = security_master.df_price.copy()
                    
                    # Rename columns to match database schema
                    rename_cols = {
                        'date': 'date',
                        'sid': 'security_id',
                        'price': 'close_price',
                        'p_high': 'high_price',
                        'p_low': 'low_price',
                        'open': 'open_price'
                    }
                    
                    price_df = price_df.rename(columns={k: v for k, v in rename_cols.items() if k in price_df.columns})
                    
                    # Validate required columns
                    required_cols = {'security_id', 'date', 'close_price'}
                    missing_cols = required_cols - set(price_df.columns)
                    if missing_cols:
                        raise ValueError(f"Missing required columns in price data: {missing_cols}")
                    required_cols = list(required_cols)

                    # Fill in missing columns with NaN
                    optional_cols = ['open_price', 'high_price', 'low_price', 'adjusted_close', 'volume']
                    for col in optional_cols:
                        if col not in price_df.columns:
                            price_df[col] = np.nan
                    
                    # For adjusted_close, use close_price if not available
                    if pd.isna(price_df['adjusted_close']).all():
                        price_df['adjusted_close'] = price_df['close_price']
                    
                    # Ensure date format
                    if not pd.api.types.is_datetime64_dtype(price_df['date']):
                        price_df['date'] = pd.to_datetime(price_df['date'])
                    
                    # Store prices
                    self.db_manager.store_security_prices(price_df[required_cols + optional_cols])
                    logger.info(f"Successfully stored price data for {len(price_df['security_id'].unique())} securities")
                except Exception as e:
                    logger.error(f"Failed to store price data: {str(e)}")
                    raise
            else:
                logger.warning("No price data available to store")
            
            # 5. Store returns data
            if hasattr(security_master, 'get_returns_long'):
                try:
                    returns_df = security_master.get_returns_long()
                    if returns_df is None or returns_df.empty:
                        logger.warning("No returns data available to store")
                        return
                    
                    # Rename columns to match database schema
                    returns_df = returns_df.rename(columns={
                        'sid': 'security_id',
                        'return': 'daily_return'
                    })
                    
                    # Validate required columns
                    required_cols = {'security_id', 'date', 'daily_return'}
                    missing_cols = required_cols - set(returns_df.columns)
                    if missing_cols:
                        raise ValueError(f"Missing required columns in returns data: {missing_cols}")
                    required_cols = list(required_cols)

                    # Ensure date format
                    if not pd.api.types.is_datetime64_dtype(returns_df['date']):
                        returns_df['date'] = pd.to_datetime(returns_df['date'])
                    
                    # Store returns
                    self.db_manager.store_security_returns(
                        returns_df[['security_id', 'date', 'daily_return']]
                    )
                    logger.info(f"Successfully stored return data for {len(returns_df['security_id'].unique())} securities")
                except Exception as e:
                    logger.error(f"Failed to store returns data: {str(e)}")
                    raise
            else:
                logger.warning("No returns data available to store")
                
        except Exception as e:
            logger.error(f"Failed to store security master data: {str(e)}")
            raise
        
    def store_factor_data(self, factor_dict: Dict) -> None:
        """
        Store factor data in the database
        
        Args:
            factor_dict: Dictionary of factor objects
        """
        for factor_name, factor_data in factor_dict.items():
            # 1. Store factor metadata
            factor_obj = factor_data.get('factor_eq')
            if factor_obj:
                self.db_manager.store_factor_metadata(
                    factor_id=factor_name,
                    name=factor_obj.name,
                    description=factor_obj.description,
                    category=factor_obj.category
                )
                
                # 2. Store factor data
                factor_df = factor_obj.data.copy()
                
                # Rename columns to match database schema
                factor_df = factor_df.rename(columns={
                    'sid': 'security_id',
                    'value': 'value'
                })
                
                # Ensure date format
                if not pd.api.types.is_datetime64_dtype(factor_df['date']):
                    factor_df['date'] = pd.to_datetime(factor_df['date'])
                
                # If normalized values exist, add to dataframe
                if 'normalized_value' not in factor_df.columns:
                    try:
                        norm_df = factor_obj.normalize(groupby='date', method='zscore') # 'winsorize'
                        factor_df['normalized_value'] = norm_df['value']
                        factor_df['normalization_method'] = 'zscore'
                    except:
                        factor_df['normalized_value'] = np.nan
                        factor_df['normalization_method'] = None
                
                # Store factor data
                self.db_manager.store_factor_data(
                    factor_id=factor_name,
                    factor_df=factor_df[['security_id', 'date', 'value', 'normalized_value', 'normalization_method']],
                    normalization_method=factor_df['normalization_method'].iloc[0] if 'normalization_method' in factor_df.columns else None
                )
                logger.info(f"Stored factor data for {factor_name}")

    def store_exposures(self, exposures_df: pd.DataFrame) -> None:
        """
        Store factor exposures in the database
        
        Args:
            exposures_df: DataFrame with exposure data (date, universe, security_id, variable, exposure)
        """
        try:
            # Validate required columns
            required_cols = {'date', 'universe', 'security_id', 'variable', 'exposure'}
            missing_cols = required_cols - set(exposures_df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns in exposures data: {missing_cols}")
            
            # Ensure date format
            if not pd.api.types.is_datetime64_dtype(exposures_df['date']):
                exposures_df['date'] = pd.to_datetime(exposures_df['date'])
            
            # Store exposures
            self.db_manager.store_exposures(exposures_df)
            logger.info(f"Successfully stored exposures data for {len(exposures_df['security_id'].unique())} securities")
        except Exception as e:
            logger.error(f"Failed to store exposures data: {str(e)}")
            raise

    def load_exposures(self, start_date: Union[str, date] = None,
                    end_date: Union[str, date] = None,
                    universe: str = None,
                    securities: List[str] = None,
                    variables: List[str] = None) -> pd.DataFrame:
        """
        Load factor exposures from the database
        
        Args:
            start_date: Start date, if None gets all data
            end_date: End date, if None gets all data
            universe: Universe ID to filter by, if None gets all universes
            securities: List of security IDs to filter by, if None gets all securities
            variables: List of variables to filter by, if None gets all variables
            
        Returns:
            DataFrame with exposure data
        """
        try:
            # Build query
            query = """
            SELECT date, universe, security_id, variable, exposure
            FROM exposures
            WHERE 1=1
            """
            params = {}
            
            if start_date:
                query += " AND date >= :start_date"
                params['start_date'] = start_date
            if end_date:
                query += " AND date <= :end_date"
                params['end_date'] = end_date
            if universe:
                query += " AND universe = :universe"
                params['universe'] = universe
            if securities:
                query += " AND security_id = ANY(:securities)"
                params['securities'] = securities
            if variables:
                query += " AND variable = ANY(:variables)"
                params['variables'] = variables
                
            query += " ORDER BY date, universe, security_id, variable"
            
            # Get data
            df = self.db_manager.get_dataframe(query, params)
            
            if df.empty:
                logger.warning("No exposure data found")
                return pd.DataFrame()
            
            # Ensure date format
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded {len(df)} exposure records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load exposures data: {str(e)}")
            raise

    def store_portfolio_results(self, config_id: str, backtest, weights_df, results_df) -> None:
        """
        Store portfolio weights and backtest results in the database
        
        Args:
            config_id: Configuration ID
            backtest: Backtest instance
            weights_df: DataFrame with portfolio weights
            results_df: DataFrame with backtest results
        """
        # 1. Store portfolio weights
        if weights_df is not None and not weights_df.empty:
            # Rename columns to match database schema
            rename_cols = {
                'ticker': 'security_id',
                'sid': 'security_id',
                'date': 'date',
                'weight': 'weight',
                'wgt_opt': 'weight'
            }
            weights_df = weights_df.rename(columns={k: v for k, v in rename_cols.items() if k in weights_df.columns})
            
            # Ensure required columns exist
            if 'security_id' in weights_df.columns and 'date' in weights_df.columns:
                # Ensure date format
                if not pd.api.types.is_datetime64_dtype(weights_df['date']):
                    weights_df['date'] = pd.to_datetime(weights_df['date'])
                
                # Store weights
                self.db_manager.store_portfolio_weights(
                    config_id=config_id,
                    weights_df=weights_df[['security_id', 'date', 'weight']]
                )
                logger.info(f"Stored portfolio weights for config ID: {config_id}")
                
        # 2. Store backtest results
        if results_df is not None and not results_df.empty:
            # Prepare results data
            if 'return_opt' in results_df.columns:
                results_df['daily_return'] = results_df['return_opt']
            elif 'return_benchmark' in results_df.columns:
                results_df['daily_return'] = results_df['return_benchmark']
                
            # Calculate cumulative return
            if 'cumulative_return' not in results_df.columns:
                results_df['cumulative_return'] = (1 + results_df['daily_return']).cumprod() - 1
                
            # Calculate drawdown
            if 'drawdown' not in results_df.columns:
                cumulative = (1 + results_df['daily_return']).cumprod()
                peak = cumulative.cummax()
                results_df['drawdown'] = (cumulative - peak) / peak
                
            # Set portfolio value to 100 * (1 + cumulative_return) if not present
            if 'portfolio_value' not in results_df.columns:
                results_df['portfolio_value'] = 100 * (1 + results_df['cumulative_return'])
                
            # Ensure date format
            if not pd.api.types.is_datetime64_dtype(results_df.index):
                results_df['date'] = pd.to_datetime(results_df.index)
            else:
                results_df['date'] = results_df.index
                
            # Store results
            self.db_manager.store_backtest_results(
                config_id=config_id,
                results_df=results_df[['date', 'portfolio_value', 'daily_return', 'cumulative_return', 'drawdown']]
            )
            logger.info(f"Stored backtest results for config ID: {config_id}")
    
    def load_factor_data(self, factor_id: str, start_date: Union[str, date] = None,
                      end_date: Union[str, date] = None,
                      securities: List[str] = None) -> pd.DataFrame:
        """
        Load factor data from the database in qFactor format
        
        Args:
            factor_id: Factor ID
            start_date: Start date, if None gets all data
            end_date: End date, if None gets all data
            securities: List of security IDs to filter by, if None gets all securities
            
        Returns:
            DataFrame in qFactor format suitable for creating an EquityFactor object
        """
        # Get factor data from database
        df = self.db_manager.get_factor_data(
            factor_id=factor_id,
            start_date=start_date,
            end_date=end_date,
            securities=securities
        )
        
        if df.empty:
            logger.warning(f"No factor data found for {factor_id}")
            return pd.DataFrame()
            
        # Convert to qFactor format
        df = df.rename(columns={
            'security_id': 'sid',
            'value': 'value'
        })
        
        # Add factor_name column if it doesn't exist
        if 'factor_name' not in df.columns:
            df['factor_name'] = factor_id
            
        # Ensure date format is datetime
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Select required columns
        factor_df = df[['date', 'factor_name', 'sid', 'value']]
        
        return factor_df
    
    def load_security_returns(self, start_date: Union[str, date] = None,
                           end_date: Union[str, date] = None,
                           securities: List[str] = None) -> pd.DataFrame:
        """
        Load security returns from the database in qFactor format
        
        Args:
            start_date: Start date, if None gets all data
            end_date: End date, if None gets all data
            securities: List of security IDs to filter by, if None gets all securities
            
        Returns:
            DataFrame in qFactor format with security returns
        """
        # Get returns data from database
        df = self.db_manager.get_security_returns(
            start_date=start_date,
            end_date=end_date,
            securities=securities
        )
        
        if df.empty:
            logger.warning("No return data found")
            return pd.DataFrame()
            
        # Convert to qFactor format
        df = df.rename(columns={
            'security_id': 'sid',
            'daily_return': 'return'
        })
        
        # Get price data to accompany returns
        price_df = self.db_manager.get_security_prices(
            start_date=start_date,
            end_date=end_date,
            securities=securities
        )
        
        if not price_df.empty:
            price_df = price_df.rename(columns={
                'security_id': 'sid',
                'close_price': 'price'
            })
            
            # Merge returns with prices
            df = df.merge(
                price_df[['sid', 'date', 'price']],
                on=['sid', 'date'],
                how='left'
            )
            
        # Ensure date format is datetime
        if not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        return df
    
    def load_factor_model_data(self, config_id: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load all factor model data for a specific configuration
        
        Args:
            config_id: Configuration ID from a previous run
            
        Returns:
            Tuple of (returns_df, factor_dict) suitable for running a factor model
        """
        # 1. Get the backtest configuration
        config_df = self.db_manager.get_backtest_configs(config_id=config_id)
        if config_df.empty:
            raise ValueError(f"No configuration found with ID: {config_id}")
            
        config = config_df.iloc[0]
        universe_id = config['universe_id']
        start_date = config['start_date']
        end_date = config['end_date']
        
        # 2. Get universe constituents
        constituents_df = self.db_manager.get_universe_constituents(universe_id)
        securities = constituents_df['security_id'].unique().tolist()
        
        # 3. Get returns data
        returns_df = self.load_security_returns(
            start_date=start_date,
            end_date=end_date,
            securities=securities
        )
        
        # 4. Get factor data
        # First, find all factors in the database
        query = "SELECT DISTINCT factor_id FROM factors;"
        factors_df = self.db_manager.get_dataframe(query)
        
        factor_dict = {}
        for _, row in factors_df.iterrows():
            factor_id = row['factor_id']
            
            # Get factor data for this factor
            factor_df = self.load_factor_data(
                factor_id=factor_id,
                start_date=start_date,
                end_date=end_date,
                securities=securities
            )
            
            if not factor_df.empty:
                # Create EquityFactor instance
                factor_eq = EquityFactor(
                    name=factor_id,
                    data=factor_df,
                    description=f"{factor_id} factor",
                    category=factor_id
                )
                
                # Store in factor_dict
                factor_dict[factor_id] = {
                    'data': factor_df,
                    'factor_eq': factor_eq
                }
                
        return returns_df, factor_dict
    
    def create_equity_factor_from_db(self, factor_id: str, start_date: Union[str, date] = None,
                                   end_date: Union[str, date] = None,
                                   securities: List[str] = None) -> EquityFactor:
        """
        Create an EquityFactor object from database data
        
        Args:
            factor_id: Factor ID
            start_date: Start date, if None gets all data
            end_date: End date, if None gets all data
            securities: List of security IDs to filter by, if None gets all securities
            
        Returns:
            EquityFactor instance
        """
        # Get factor data
        factor_df = self.load_factor_data(
            factor_id=factor_id,
            start_date=start_date,
            end_date=end_date,
            securities=securities
        )
        
        if factor_df.empty:
            raise ValueError(f"No data found for factor: {factor_id}")
            
        # Get factor metadata
        query = """
        SELECT name, description, category
        FROM factors
        WHERE factor_id = :factor_id
        """
        metadata_df = self.db_manager.get_dataframe(query, {'factor_id': factor_id})
        
        if metadata_df.empty:
            name = factor_id
            description = None
            category = None
        else:
            name = metadata_df.iloc[0]['name']
            description = metadata_df.iloc[0]['description']
            category = metadata_df.iloc[0]['category']
            
        # Create EquityFactor instance
        factor_eq = EquityFactor(
            name=name,
            data=factor_df,
            description=description,
            category=category
        )
        
        return factor_eq