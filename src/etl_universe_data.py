# etl_universe_data.py
import os
import logging
from decimal import Decimal
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from pathlib import Path

# Import from qFactor.py
from qFactor import (
    EquityFactorModelInput, ParamsConfig, BacktestConfig, RegimeConfig, OptimizationConfig, ExportConfig,
    RiskFactors, Universe, Currency, Frequency, EquityFactor, VolatilityType,
    SecurityMasterFactory, FactorFactory, get_rebalance_dates, generate_config_id,
    set_model_input_start, set_model_input_dates_turnover, set_model_input_dates_daily,
    merge_weights_with_factor_loadings
)

from file_data_manager import (
    FileConfig, FileDataManager # FilePathHandler, DataStore, DataLoader, 
    )

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def etl_universe_data(model_input, progress_callback=None):
    """
    Enhanced ETL function with detailed progress messages for the Streamlit app.
    
    Args:
        model_input: The model input configuration
        progress_callback: Optional callback function to report progress (for Streamlit)
    """
    def update_progress(step, total_steps, message, progress_type="info"):
        """Helper function to update progress"""
        if progress_callback:
            progress_callback(step, total_steps, message, progress_type)
        else:
            # Default print output
            if progress_type == "error":
                print(f"❌ {message}")
            elif progress_type == "warning":
                print(f"⚠️ {message}")
            elif progress_type == "success":
                print(f"✅ {message}")
            else:
                print(f"📋 {message}")
    
    update_history = model_input.export.update_history
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
    file_members = identifier+'_members'
    
    # Initialize progress tracking
    total_steps = 7
    current_step = 0
    
    cfg = FileConfig()
    mgr = FileDataManager(cfg)
    
    # If incremental update, check all data files and use earliest last date
    if not update_history:
        try:
            file_sources = [
                (mgr.load_prices, identifier+'_members', 'member prices'),
                (mgr.load_prices, identifier, 'benchmark prices'),
                (mgr.load_benchmark_weights, identifier, 'benchmark weights'),
                (mgr.load_returns, identifier+'_members', 'member returns'),
                (mgr.load_exposures, identifier+'_members', 'exposures')
            ]
            
            last_dates = []
            for load_func, file_id, desc in file_sources:
                try:
                    df = load_func(file_id)
                    if df is not None and not df.empty and 'date' in df.columns:
                        max_date = pd.to_datetime(df['date']).max()
                        last_dates.append(max_date)
                except Exception as e:
                    logger.warning(f"Could not load {desc}: {e}")
                    continue
            
            if last_dates:
                # Find the minimum of all max dates (the earliest updated file)
                last_date = min(last_dates)
                update_progress(current_step, total_steps, 
                               f"📅 Incremental update detected. Earliest last date: {last_date}")
                # Set start date to the earliest last date minus 3 days for safety
                model_input.backtest.start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=-3)).date()
                
                # Update portfolio turnover dates
                set_model_input_dates_turnover(model_input)

                # Update daily business dates
                set_model_input_dates_daily(model_input)

                update_progress(current_step, total_steps, 
                               f"📅 Adjusted start date to: {model_input.backtest.start_date}")
        except Exception as e:
            logger.warning(f"Error checking existing data, using configured start_date: {e}")
            update_progress(current_step, total_steps, 
                           f"⚠️ Could not determine incremental date, using configured start date", "warning")
    
    update_progress(current_step, total_steps, f"🚀 Starting ETL process for {identifier}")
    update_progress(current_step, total_steps, f"📅 Update mode: {'FULL HISTORY' if update_history else 'INCREMENTAL'}")
    update_progress(current_step, total_steps, f"📊 Date range: {model_input.backtest.start_date} to {model_input.backtest.end_date}")
    
    if model_input.backtest.start_date < datetime.today().date()+pd.Timedelta(days=-1):
        current_step += 1
        update_progress(current_step, total_steps, "Step 1: Generating configuration ID...")
        
        # Convert to dict (or use .model_dump() for pydantic)
        config_dict = model_input.model_dump()
        # Optionally, add a prefix (e.g., data source and date)
        prefix = f"{config_dict['backtest']['data_source']}_{config_dict['backtest']['universe']}_{datetime.now().strftime('%Y%m%d')}"
        prefix = prefix.replace(' ','_')
        config_id = generate_config_id(config_dict, prefix=prefix)
        update_progress(current_step, total_steps, f"Generated config_id: {config_id}", "success")

        current_step += 1
        update_progress(current_step, total_steps, "Step 2: Initializing Security Master...")
        # 2. Get security master
        security_master = SecurityMasterFactory(model_input=model_input)
        update_progress(current_step, total_steps, "Security Master initialized successfully", "success")

        current_step += 1
        update_progress(current_step, total_steps, "Step 3: Downloading benchmark data...")
        # 2.1 Get benchmark prices
        update_progress(current_step, total_steps, "   📈 Downloading benchmark prices...")
        df_benchmark_prices = security_master.get_benchmark_prices()
        update_progress(current_step, total_steps, f"   Benchmark prices: {len(df_benchmark_prices)} records", "success")

        # 2.2 Get benchmark members' weights
        update_progress(current_step, total_steps, "   ⚖️ Downloading benchmark weights...")
        df_benchmark_weights = security_master.get_benchmark_weights()
        if df_benchmark_weights.shape[0]==0:
            update_progress(current_step, total_steps, "   No benchmark weights found, using existing universe data...", "warning")
            df_existing = mgr.load_prices(identifier+'_members')
            univ_list = sorted(df_existing.sid.unique())
            update_progress(current_step, total_steps, f"   Using existing universe: {len(univ_list)} securities")
        else:
            univ_list = sorted(df_benchmark_weights['sid'].unique())
            update_progress(current_step, total_steps, f"   Benchmark weights: {len(univ_list)} securities", "success")
        
        model_input.backtest.universe_list = univ_list
        model_input.params.n_sids = len(univ_list)
        update_progress(current_step, total_steps, f"   Universe size: {len(univ_list)} securities")

        current_step += 1
        update_progress(current_step, total_steps, "Step 4: Downloading member securities data...")
        # 2.3 Get benchmark members' prices and returns
        update_progress(current_step, total_steps, "   💰 Downloading member prices...")
        df_prices = security_master.get_members_prices(model_input)
        update_progress(current_step, total_steps, f"   Member prices: {len(df_prices)} records", "success")
        
        update_progress(current_step, total_steps, "   📊 Calculating returns...")
        df_returns = security_master.get_returns_long()
        update_progress(current_step, total_steps, f"   Member returns: {len(df_returns)} records", "success")

        current_step += 1
        update_progress(current_step, total_steps, "Step 5: Calculating factor data...")
        # 3. Get factor data
        factor_list = [i.value for i in model_input.params.risk_factors]
        update_progress(current_step, total_steps, f"   Processing {len(factor_list)} factors: {', '.join(factor_list)}")
        
        factor_dict = {}
        for i, factor_type in enumerate(factor_list, 1):
            update_progress(current_step, total_steps, f"   Factor {i}/{len(factor_list)}: {factor_type}")
            factor = FactorFactory(factor_type=factor_type, model_input=model_input)
            df = factor.get_factor_data()
            if df_benchmark_weights.shape[0]>0:
                df = merge_weights_with_factor_loadings(df_benchmark_weights, df)
                df = df[['date','factor_name','sid','value']]
            factor_eq = None
            if not df.empty:
                factor_eq = EquityFactor(
                    name=factor_type,
                    data=df,
                    description=f"{factor_type} factor",
                    category=factor_type
                )
                update_progress(current_step, total_steps, f"      {factor_type}: {len(df)} records", "success")
            else:
                update_progress(current_step, total_steps, f"      {factor_type}: No data available", "warning")
            
            factor_dict[factor_type] = {
                'factor': factor,
                'data': df,
                'factor_eq': factor_eq
            }

        current_step += 1
        update_progress(current_step, total_steps, "Step 6: Building exposures DataFrame...")
        # 4. Build exposures DataFrame
        df_exposures = pd.DataFrame()
        for factor in factor_dict.keys():
            eq = factor_dict[factor].get('factor_eq')
            if eq is not None:
                update_progress(current_step, total_steps, f"   Processing exposures for {factor}...")
                df = eq.normalize(groupby='date', method='winsorize')[['date', 'sid', 'value']].copy()
                df.rename(columns={'value': factor}, inplace=True)
                df.sort_values(['sid', 'date'], inplace=True)
                df[factor] = df.groupby('sid')[factor].ffill()
                if df_exposures.shape[0] == 0:
                    df_exposures = df.copy()
                else:
                    df_exposures = df_exposures.merge(df, how='left', on=['date', 'sid'])
                update_progress(current_step, total_steps, f"      {factor} exposures processed", "success")
            else:
                update_progress(current_step, total_steps, f"      Skipping {factor} - no factor data available", "warning")
        
        df_exposures.dropna(inplace=True)
        df_exposures_long = pd.melt(df_exposures, id_vars=['date', 'sid'], value_name='exposure')
        # df_exposures_long.rename(columns={'sid': 'security_id'}, inplace=True)
        df_exposures_long.insert(1, 'universe', model_input.backtest.universe.value)
        update_progress(current_step, total_steps, f"   Final exposures: {len(df_exposures_long)} records", "success")

        current_step += 1
        update_progress(current_step, total_steps, "Step 7: Saving data to file system...")
        # 5. Save data to file system
        update_progress(current_step, total_steps, "   💾 Saving benchmark prices...")
        mgr.store_prices(df=df_benchmark_prices, identifier=identifier, update_history=update_history)
        update_progress(current_step, total_steps, "   Benchmark prices saved", "success")
        
        update_progress(current_step, total_steps, "   💾 Saving benchmark weights...")
        mgr.store_benchmark_weights(df=df_benchmark_weights, identifier=identifier, update_history=update_history)
        update_progress(current_step, total_steps, "   Benchmark weights saved", "success")
        
        update_progress(current_step, total_steps, "   💾 Saving member prices...")
        mgr.store_prices(df=df_prices, identifier=file_members, update_history=update_history)
        update_progress(current_step, total_steps, "   Member prices saved", "success")
        
        update_progress(current_step, total_steps, "   💾 Saving member returns...")
        mgr.store_returns(df=df_returns, identifier=file_members, update_history=update_history)
        update_progress(current_step, total_steps, "   Member returns saved", "success")
        
        update_progress(current_step, total_steps, "   💾 Saving exposures...")
        mgr.store_exposures(df=df_exposures_long, identifier=file_members, update_history=update_history)
        update_progress(current_step, total_steps, "   Exposures saved", "success")
        
        update_progress(current_step, total_steps, "   💾 Saving factor data...")
        for factor in factor_dict.keys():
            df = factor_dict[factor]['data']
            if not df.empty:
                mgr.store_factors(df=df, identifier=f"{file_members}_{factor}", update_history=update_history)
                update_progress(current_step, total_steps, f"      {factor} factor data saved", "success")
            else:
                update_progress(current_step, total_steps, f"      {factor} factor data empty - not saved", "warning")

        update_progress(current_step, total_steps, "**Data update completed successfully!**", "success")
        update_progress(current_step, total_steps, f"📊 Summary:")
        update_progress(current_step, total_steps, f"   • Universe: {identifier}")
        update_progress(current_step, total_steps, f"   • Securities: {len(univ_list)}")
        update_progress(current_step, total_steps, f"   • Date range: {df_prices['date'].min()} to {df_prices['date'].max()}")
        update_progress(current_step, total_steps, f"   • Factors: {len(factor_list)}")
        update_progress(current_step, total_steps, f"   • Update mode: {'Full History' if update_history else 'Incremental'}")
        
    else:
        update_progress(current_step, total_steps, f"Data is already up to date as of {datetime.today().date()}")
        update_progress(current_step, total_steps, "   No update needed - data is current")

    logger.info("Real data ETL complete and saved to file system.")
    update_progress(current_step, total_steps, "ETL process logged successfully", "success")

if __name__ == "__main__":

    # Script to run the ETL process and update & save data to file system

    # 1. Create model input
    model_input = EquityFactorModelInput(
        params=ParamsConfig(
            aum=Decimal('100'),
            sigma_regimes=False,
            risk_factors=[
                RiskFactors.SIZE, 
                RiskFactors.MOMENTUM, 
                RiskFactors.VALUE, 
                RiskFactors.BETA
                ],
            bench_weights=None,
            n_buckets=4
        ),
        backtest=BacktestConfig(
            data_source='yahoo',
            universe=Universe.SPX, # INDU, NDX, SPX
            currency=Currency.USD,
            frq=Frequency.MONTHLY,
            start='2013-12-31', # '2022-12-31',
            portfolio_list=[],
            concurrent_download = False
        ),
        regime=RegimeConfig(
            type='vol',
            benchmark=VolatilityType.VIX,
            periods=10
        ),
        export=ExportConfig(
            update_history=False, # True, False
            base_path="./data/time_series",
            s3_config=None
        )
    )

    # Note: The ETL function will automatically handle incremental updates
    # by checking all data files and using the earliest last date

    # Set portfolio turnover dates: model_input.backtest.dates_turnover
    set_model_input_dates_turnover(model_input)

    # Set daily business dates: model_input.backtest.dates_daily
    set_model_input_dates_daily(model_input)

    # ETL to download and update data
    if model_input.export.update_history:
        confirm = input("You are about to overwrite all historical data files. Are you sure? (y/n): ")
        if confirm.lower() == 'y':
            print("History is going to be updated.")
        else:
            model_input.export.update_history = False
    etl_universe_data(model_input)

    # Get data from files
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"

    cfg = FileConfig()
    mgr = FileDataManager(cfg)

    df_benchmark_prices = mgr.load_prices(identifier)
    df_benchmark_weights = mgr.load_benchmark_weights(identifier)
    df_prices = mgr.load_prices(identifier+'_members')
    df_returns = mgr.load_returns(identifier+'_members')
    df_exposures_long = mgr.load_exposures(identifier+'_members')

    factor_dict = {}
    for factor in model_input.params.risk_factors:
        factor_name = factor.value
        # df = mgr.load_factors(identifier+'_members_'+factor_name)
        factor_dict[factor_name] = mgr.load_factors(f"{identifier}_members_{factor_name}")

    print("Succesfully load data.")
