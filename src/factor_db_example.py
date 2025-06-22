# factor_db_example.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Usage Script for Factor Database Integration

This script demonstrates how to use the database integration with the qFactor framework.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timedelta, date
from decimal import Decimal
import pandas as pd
import numpy as np
import dotenv

# Import qFactor components
from qFactor import (
    DataSource, Universe, Currency, Frequency, RiskFactors, 
    ParamsConfig, BacktestConfig, RegimeConfig, OptimizationConfig, ExportConfig,
    EquityFactorModelInput, EquityFactor, YahooFactor,
    SecurityMasterFactory, FactorFactory, UniverseMappingFactory,
    get_universe_mapping_yahoo, get_rebalance_dates
)

# Import database components
from db_manager import DatabaseManager, DatabaseConfig
from db_integration import FactorDatabaseIntegration

# Load environment variables for database connection
dotenv.load_dotenv()

# Configure logging
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to store logs
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create log file handler
    log_file_path = os.path.join(log_dir, "application.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=10485760, backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter and add to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create module-specific logger for database operations
    db_logger = logging.getLogger("db_manager")
    db_logger.setLevel(log_level)
    
    # Create specific log file for database operations
    db_log_path = os.path.join(log_dir, "database.log")
    db_file_handler = logging.handlers.RotatingFileHandler(
        db_log_path, maxBytes=10485760, backupCount=5
    )
    db_file_handler.setFormatter(formatter)
    db_logger.addHandler(db_file_handler)
    
    logging.info("Logging configured successfully")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()  # Outputs to console
    ]
)
logger = logging.getLogger(__name__)

def clean_database_with_transaction(db_manager):
    """Clean database with transaction support"""
    try:
        with db_manager.connect() as conn:
            # Start transaction
            conn.autocommit = False
            
            with conn.cursor() as cursor:
                # Delete from tables in reverse dependency order
                tables_in_order = [
                    "backtest_results",
                    "portfolio_weights",
                    "backtest_configs",
                    "security_returns",
                    "security_prices",
                    "factor_data",
                    "universe_constituents",
                    "factors",
                    "securities",
                    "universes"
                ]
                
                for table in tables_in_order:
                    cursor.execute(f"DELETE FROM {table}")
                    rows_deleted = cursor.rowcount
                    logger.info(f"Deleted {rows_deleted} rows from {table}")
                
                # Commit the transaction
                conn.commit()
                logger.info("Database cleanup transaction committed successfully")
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}", exc_info=True)
        # If there's an error, the transaction will be rolled back automatically when the connection is closed
        raise

def clean_database(db_manager):
    """
    Delete all data from all tables in the database to start fresh.
    This function deletes data in the correct order to respect foreign key constraints.
    
    Args:
        db_manager: The DatabaseManager instance
    """
    from sqlalchemy import text

    logger = logging.getLogger(__name__)
    logger.info("Cleaning database - removing all entries from all tables")
    
    # Tables with foreign keys must be cleared first (in reverse dependency order)
    tables_in_order = [
        "backtest_results",
        "portfolio_weights",
        "backtest_configs",
        "security_returns",
        "security_prices",
        "factor_data",
        "universe_constituents",
        "factors",
        "exposures",
        "securities",
        "universes"
    ]
    
    try:
        with db_manager.engine.connect() as conn:
            # Start a transaction
            with conn.begin():
                for table in tables_in_order:
                    # Execute DELETE statement for each table
                    result = conn.execute(text(f"DELETE FROM {db_manager.config.db_schema}.{table}"))
                    logger.info(f"Deleted {result.rowcount} rows from {table}")
                
            logger.info("Database cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}", exc_info=True)
        raise

def setup_database():
    """Set up database connection and schema"""
    # Create database configuration from environment variables
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        username=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        schema=os.getenv("DB_SCHEMA", "public")
    )
    
    # Create database manager
    db_manager = DatabaseManager(db_config)
    
    # Execute schema setup script (if needed)
    # script_path = os.path.join(os.path.dirname(__file__), "db_schema.sql")
    # if os.path.exists(script_path):
    #     db_manager.execute_script(script_path)
        
    return db_manager

def create_factor_model_input():
    """Create a sample factor model input configuration"""
    model_input = EquityFactorModelInput(
        params=ParamsConfig(
            aum=Decimal('100'),
            sigma_regimes=False,
            risk_factors=[RiskFactors.SIZE, RiskFactors.MOMENTUM, RiskFactors.VALUE, RiskFactors.BETA],
            bench_weights=None,
            n_buckets=4
        ),
        backtest=BacktestConfig(
            data_source = DataSource.YAHOO,
            universe=Universe.INDU,  # Dow Jones Industrial Average
            currency=Currency.USD,
            frq=Frequency.MONTHLY,
            start='2022-12-31',
            # end defaults to today
            portfolio_list=[]
        ),
        regime=RegimeConfig(
            type='vol',
            benchmark='VIX Index',
            periods=10
        ),
        opt=OptimizationConfig(
            obj='pfactor',
            n_trades=30,
            te_max=0.05,
            w_max=0.05,
            factors={},
            pfactor=None
        ),
        export=ExportConfig(
            base_path="./data/output",
            s3_config={
                'bucket_name': os.environ.get('CLOUD_USER_BUCKET'),
                'user_name': os.environ.get('CLOUD_USERNAME')
            }
        )
    )
    dates_turnover = get_rebalance_dates(model_input, return_as='str')
    model_input.backtest.dates_turnover = dates_turnover

    dates_daily = get_rebalance_dates(model_input, return_as='str', frq=Frequency.DAILY)
    model_input.backtest.dates_daily = dates_daily
    model_input.params.n_dates = len(model_input.backtest.dates_daily) # n_dates
    
    return model_input

def run_store_factor_data(db_integration):
    """Run a factor model and store data in the database"""
    # 1. Create model input
    model_input = create_factor_model_input()
    logger.info(f"Created model input for {model_input.backtest.universe.value}")
    
    # 2. Store model input in database
    # config_id = db_integration.store_model_input(model_input)
    # logger.info(f"Stored model input with config ID: {config_id}")
    
    # 3. Create security master
    security_master = SecurityMasterFactory(
        model_input=model_input # dates=dates_daily
        )

    # dates_daily = get_rebalance_dates(model_input, return_as='str', frq=Frequency.DAILY)    
    # security_master = SecurityMasterFactory(
    #     model_input=model_input, 
    #     dates=dates_daily
    #     )
    
    # 4. Get security master data
    # Get benchmark prices
    df = security_master.get_benchmark_prices()
    logger.info("Retrieved benchmark prices")
    
    # dates = sorted(security_master.df_bench['date'].unique())
    # dates_to = [pd.to_datetime(i).date() for i in dates]
    # model_input.backtest.dates_turnover = dates_to
    
    # Get benchmark weights
    df_benchmark_weights = security_master.get_benchmark_weights()
    logger.info(f"Retrieved benchmark weights with {len(df_benchmark_weights)} records")
    
    # Get portfolios
    df = security_master.get_portfolio(model_input)
    model_input.backtest.portfolio_list = sorted(list(df['sid'].unique()))
    # print("\nSecurity Master Portfolio:")
    # print(security_master.df_portfolio.tail(3))

    univ_list = sorted(df_benchmark_weights['sid'].unique())
    univ_list = sorted(list(set(univ_list + model_input.backtest.portfolio_list)))
    model_input.backtest.universe_list = univ_list
    n_sids = len(univ_list)
    model_input.params.n_sids = n_sids
    print(f"\nNumber of securities in universe is: {n_sids}")
    
    # Get benchmark members' prices
    df = security_master.get_members_prices(model_input)
    logger.info("Retrieved benchmark members prices")
    
    # Get security master with BICS sectors
    if security_master.security_master is None:
        sec_master = security_master.get_security_master(sector_classification='BICS')
        # security_master.get_security_master()
    else:
        sec_master = security_master.security_master.copy()
    logger.info("Retrieved security master data")
    
    # 5. Store security master data in database
    # db_integration.store_security_master_data(security_master)
    logger.info("Stored security master data in database")
    
    # 6. Get returns data
    df_ret_long = security_master.get_returns_long()
    df_ret_wide = security_master.get_returns_wide()
    logger.info(f"Retrieved returns data for {len(univ_list)} securities")
    
    # 7. Calculate factors
    factor_list = [factor.value for factor in model_input.params.risk_factors]
    factor_dict = {}
    
    for factor_type in factor_list:
        logger.info(f"Processing factor: {factor_type}")

        factor = FactorFactory(
            factor_type=factor_type,
            model_input=model_input, 
            # dates=dates_turnover
            )
                
        # Get factor data
        # factor_data[factor_name] = factor.get_factor_data()
        df = factor.get_factor_data()

        # Create EquityFactor instance
        factor_eq = EquityFactor(
            name=factor_type,
            data=df,
            description=f"{factor_type} factor",
            category=factor_type
        )

        # Example operations
        # 1. Normalize factor values
        # df_normalized = factor_eq.normalize(method='winsorize')
        df_normalized = factor_eq.normalize(groupby='date', method='winsorize')

        # 2. Convert to wide format
        df_wide = factor_eq.to_wide()
        
        # Analyze factor returns
        results = factor_eq.analyze_factor_returns(
            returns_data=df_ret_long,
            n_buckets=model_input.params.n_buckets,
            method='quantile',
            weighting='equal',
            long_short=True,
            neutralize_size=False,
            shift_lag=22
        )
        
        # Store in factor dictionary
        factor_dict[factor_type] = {
            'factor': factor,
            'data': df,
            'factor_eq': factor_eq,
            'results': results
        }
    
    # 8. Store factor data in database
    # db_integration.store_factor_data(factor_dict)
    logger.info("Stored factor data in database")
    
    # 9. Create exposures matrix
    df_exposures = pd.DataFrame()
    for factor in factor_dict.keys():
        df = factor_dict.get(factor).get('factor_eq').normalize(groupby='date', method='winsorize')[['date', 'sid', 'value']].copy()
        df.rename(columns={'value': factor}, inplace=True)
        # df.index.name = None
        df.sort_values(['sid', 'date'], inplace=True)
        df[factor] = df.groupby('sid')[factor].ffill()
        
        if df_exposures.shape[0] == 0:
            df_exposures = df.copy()
        else:
            df_exposures = df_exposures.merge(df, how='left', on=['date', 'sid'])
    
    df_exposures.dropna(inplace=True)
    df_exposures.index = df_exposures['date']
    df_exposures.index.name = None

    df_exposures_long = pd.melt(df_exposures, id_vars=['date','sid'], value_name='exposure')
    df_exposures_long.rename(columns={'sid':'security_id'},inplace=True)
    df_exposures_long.insert(1, 'universe', model_input.backtest.universe.value)
    logger.info(f"Created exposures matrix with shape {df_exposures.shape}")

    # Store exposures
    i_dates = df_exposures_long.date.isin(pd.to_datetime(model_input.backtest.dates_turnover)) # only exposures during rebalance periods
    db_integration.store_exposures(df_exposures_long.loc[i_dates])
    logger.info("Stored exposures data in database")

    # Load exposures
    exposures_df = db_integration.load_exposures(
        start_date='2025-01-01',
        end_date='2025-04-22',
        universe='INDU Index',
        securities=['WMT'],
        variables=['value']
)

    # 10. Create backtest portfolio based on factor exposures
    # (Simplified for this example - just using random weights)
    if False:
        portfolio_weights = pd.DataFrame()
        for date in model_input.backtest.dates_turnover:  # Use first 10 dates
            # date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            date_str = pd.to_datetime(date).date()
            securities = df_exposures[df_exposures['date'] == date_str]['sid'].unique()
            
            # Create random weights
            weights = np.random.rand(len(securities))
            weights = weights / weights.sum()  # Normalize to sum to 1
            
            df = pd.DataFrame({
                'date': date_str,
                'security_id': securities,
                'weight': weights
            })
            
            portfolio_weights = pd.concat([portfolio_weights, df])
    else:
        portfolio_weights = security_master.df_portfolio[['date','sid','weight']].copy()
        portfolio_weights.rename(columns={'sid':'security_id'}, inplace=True)

    # 11. Create backtest results
    backtest_dates = pd.date_range(
        start=model_input.backtest.start_date,
        end=model_input.backtest.end_date,
        freq='B'  # Business days
    )
    
    # Create sample returns
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.0005, 0.01, len(backtest_dates))  # Mean 0.05% daily, std 1%
    
    results_df = pd.DataFrame({
        'date': backtest_dates,
        'daily_return': daily_returns
    })
    
    results_df['cumulative_return'] = (1 + results_df['daily_return']).cumprod() - 1
    results_df['portfolio_value'] = 100 * (1 + results_df['cumulative_return'])
    
    # Calculate drawdown
    peak = results_df['portfolio_value'].cummax()
    results_df['drawdown'] = (results_df['portfolio_value'] - peak) / peak

    # 12. Store portfolio and results in database
    config_id = '' # TO DO: get config id in config object
    db_integration.store_portfolio_results(
        config_id=config_id,
        backtest=None,  # Not using a real backtest instance
        weights_df=portfolio_weights,
        results_df=results_df
    )
    logger.info("Stored portfolio weights and results in database")
    
    return config_id

def run_query_factor_data(db_integration, config_id):
    """Query and use factor data from the database"""
    # 1. Load factor model data
    returns_df, factor_dict = db_integration.load_factor_model_data(config_id)
    logger.info(f"Loaded returns data with {len(returns_df)} records")
    logger.info(f"Loaded {len(factor_dict)} factors from database")
    
    # 2. Use the data to create a factor exposure matrix
    df_exposures = pd.DataFrame()
    for factor_name, factor_data in factor_dict.items():
        factor_obj = factor_data.get('factor_eq')
        if factor_obj:
            df = factor_obj.normalize(groupby='date', method='winsorize')[['date', 'sid', 'value']].copy()
            df.rename(columns={'value': factor_name}, inplace=True)
            
            if df_exposures.shape[0] == 0:
                df_exposures = df.copy()
            else:
                df_exposures = df_exposures.merge(df, how='left', on=['date', 'sid'])
    
    if not df_exposures.empty:
        logger.info(f"Created exposures matrix with shape {df_exposures.shape}")
        
        # Print some summary statistics
        logger.info("\nFactor exposure statistics:")
        for factor_name in factor_dict.keys():
            if factor_name in df_exposures.columns:
                stats = df_exposures[factor_name].describe()
                logger.info(f"\n{factor_name}:\n{stats}")
        
        # 3. Create a factor correlation matrix
        if len(factor_dict) > 1:
            # Use only factor columns
            factor_cols = [col for col in df_exposures.columns if col not in ['date', 'sid']]
            corr_matrix = df_exposures[factor_cols].corr()
            
            logger.info("\nFactor correlation matrix:")
            logger.info(f"\n{corr_matrix}")
    else:
        logger.warning("No factor exposures available")
    
    # 4. Get backtest results
    results_df = db_integration.db_manager.get_backtest_results(config_id)
    
    if not results_df.empty:
        logger.info(f"Retrieved backtest results with {len(results_df)} records")
        
        # Print summary statistics
        stats = results_df[['daily_return', 'cumulative_return', 'drawdown']].describe()
        logger.info(f"\nBacktest results statistics:\n{stats}")
        
        # Print final performance
        final_row = results_df.iloc[-1]
        logger.info(f"\nFinal performance:")
        logger.info(f"Cumulative return: {final_row['cumulative_return']:.2%}")
        logger.info(f"Maximum drawdown: {results_df['drawdown'].min():.2%}")
        
        # Calculate Sharpe ratio
        annualized_return = final_row['cumulative_return'] * 252 / len(results_df)
        annualized_vol = results_df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol
        
        logger.info(f"Annualized return: {annualized_return:.2%}")
        logger.info(f"Annualized volatility: {annualized_vol:.2%}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
    else:
        logger.warning("No backtest results available")

def main():
    """Main function"""
    try:
        # Set up database connection
        db_manager = setup_database()
        db_integration = FactorDatabaseIntegration(db_manager)
        
        # Ask user what operation to perform
        print("\nFactor Model Database Operations:")
        print("1. Run factor model and store data")
        print("2. Query and use factor data from database")
        print("3. Clean database (delete all entries)")
        
        choice = '3' # input("\nSelect operation (1-3): ")
        
        if choice == '1':
            # Run and store factor data
            config_id = run_store_factor_data(db_integration)
            print(f"\nFactor model data stored with config ID: {config_id}")
            print("Use this ID when querying the data later.")
            
        elif choice == '2':
            # Query existing data
            config_id = input("\nEnter config ID to query: ")
            run_query_factor_data(db_integration, config_id)
            
        elif choice == '3':
            # Clean database
            confirm = input("\nThis will delete ALL data from ALL tables. Are you sure? (y/n): ")
            if confirm.lower() == 'y':
                clean_database(db_manager)
                print("Database cleaned successfully.")
            else:
                print("Database cleanup cancelled.")
        
        else:
            print("\nInvalid choice. Exiting.")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    setup_logging(log_level=logging.DEBUG)  # Use DEBUG during development, otherwise use INFO or WARNING
    main()