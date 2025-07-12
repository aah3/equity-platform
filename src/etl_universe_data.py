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
    set_model_input_start, set_model_input_dates_turnover, set_model_input_dates_daily
)

from file_data_manager import (
    FileConfig, FileDataManager # FilePathHandler, DataStore, DataLoader, 
    )

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def etl_universe_data(model_input):

    update_history = model_input.export.update_history
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
    file_members = identifier+'_members'
    
    cfg = FileConfig()
    mgr = FileDataManager(cfg)
    
    if model_input.backtest.start_date < datetime.today().date()+pd.Timedelta(days=-1):
        # Convert to dict (or use .model_dump() for pydantic)
        config_dict = model_input.model_dump()
        # Optionally, add a prefix (e.g., data source and date)
        prefix = f"{config_dict['backtest']['data_source']}_{config_dict['backtest']['universe']}_{datetime.now().strftime('%Y%m%d')}"
        prefix = prefix.replace(' ','_')
        config_id = generate_config_id(config_dict, prefix=prefix)
        print("Generated config_id:", config_id)

        # 2. Get security master
        security_master = SecurityMasterFactory(model_input=model_input)

        # 2.1 Get benchmark prices
        df_benchmark_prices = security_master.get_benchmark_prices()

        # 2.2 Get benchmark members' weights
        df_benchmark_weights = security_master.get_benchmark_weights()
        if df_benchmark_weights.shape[0]==0:
            df_existing = mgr.load_prices(identifier+'_members')
            univ_list = sorted(df_existing.sid.unique())
        else:
            univ_list = sorted(df_benchmark_weights['sid'].unique())
        model_input.backtest.universe_list = univ_list
        model_input.params.n_sids = len(univ_list)

        # 2.3 Get benchmark members' prices and returns
        df_prices = security_master.get_members_prices(model_input)
        df_returns = security_master.get_returns_long()

        # 3. Get factor data
        factor_list = [i.value for i in model_input.params.risk_factors]
        factor_dict = {}
        for factor_type in factor_list:
            factor = FactorFactory(factor_type=factor_type, model_input=model_input)
            df = factor.get_factor_data()
            factor_eq = None
            if not df.empty:
                factor_eq = EquityFactor(
                    name=factor_type,
                    data=df,
                    description=f"{factor_type} factor",
                    category=factor_type
                )
            factor_dict[factor_type] = {
                'factor': factor,
                'data': df,
                'factor_eq': factor_eq
            }

        # 4. Build exposures DataFrame
        df_exposures = pd.DataFrame()
        for factor in factor_dict.keys():
            eq = factor_dict[factor].get('factor_eq')
            if eq is not None:
                df = eq.normalize(groupby='date', method='winsorize')[['date', 'sid', 'value']].copy()
                df.rename(columns={'value': factor}, inplace=True)
                df.sort_values(['sid', 'date'], inplace=True)
                df[factor] = df.groupby('sid')[factor].ffill()
                if df_exposures.shape[0] == 0:
                    df_exposures = df.copy()
                else:
                    df_exposures = df_exposures.merge(df, how='left', on=['date', 'sid'])
        df_exposures.dropna(inplace=True)
        df_exposures_long = pd.melt(df_exposures, id_vars=['date', 'sid'], value_name='exposure')
        # df_exposures_long.rename(columns={'sid': 'security_id'}, inplace=True)
        df_exposures_long.insert(1, 'universe', model_input.backtest.universe.value)

        # 5. Save data to file system
        mgr.store_prices(df=df_benchmark_prices, identifier=identifier, update_history=update_history)
        mgr.store_benchmark_weights(df=df_benchmark_weights, identifier=identifier, update_history=update_history)
        mgr.store_prices(df=df_prices, identifier=file_members, update_history=update_history)
        mgr.store_returns(df=df_returns, identifier=file_members, update_history=update_history)
        mgr.store_exposures(df=df_exposures_long, identifier=file_members, update_history=update_history)
        for factor in factor_dict.keys():
            df = factor_dict[factor]['data']
            if not df.empty:
                mgr.store_factors(df=df, identifier=f"{file_members}_{factor}", update_history=update_history)

        print("Succesfully updated data history.")
    else:
        print(f"Data is updated as of -> {datetime.today().date()}")

    logger.info("Real data ETL complete and saved to file system.")

if __name__ == "__main__":

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
            universe=Universe.NDX, # INDU,
            currency=Currency.USD,
            frq=Frequency.MONTHLY,
            start='2017-12-31', # '2022-12-31',
            portfolio_list=[],
            concurrent_download = False
        ),
        regime=RegimeConfig(
            type='vol',
            benchmark=VolatilityType.VIX,
            periods=10
        ),
        export=ExportConfig(
            update_history=True, # False,
            base_path="./data/time_series",
            s3_config=None
        )
    )

    # --- Check for incremental update ---
    cfg = FileConfig()
    mgr = FileDataManager(cfg)
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"

    if not model_input.export.update_history:
        # Try to read last date from prices file
        try:
            df_existing = mgr.load_prices(identifier+'_members')
            last_date = pd.to_datetime(df_existing['date']).max()
            print(f"Last updated date in prices: {last_date}")

            # Set model_input.backtest.start to last_date (or last_date + 1 day)
            model_input.backtest.start_date = (last_date + pd.Timedelta(days=-3)).date() # 04/24/2025
        except Exception as e:
            print("No existing data found or error reading file, running full history.")
            model_input.export.update_history = True

    # Set portfolio turnover dates
    set_model_input_dates_turnover(model_input)

    # Set daily business dates
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

