# app_factors_v1.py

import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from decimal import Decimal
import os
import boto3
from botocore.exceptions import ClientError
import json
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append('./src/')

import src.qBacktest as bt

from src.qFactor import (
    EquityFactor, EquityFactorModelInput, RiskFactors, 
    ParamsConfig, BacktestConfig, RegimeConfig, ExportConfig, OptimizationConfig,
    Universe, Currency, Frequency, DataSource, VolatilityType, RegimeType,
    SecurityMasterFactory, FactorFactory, get_rebalance_dates, set_model_input_dates_turnover,
    set_model_input_dates_daily
)
from src.etl_universe_data import etl_universe_data

from src.file_data_manager import (
    FileConfig, FileDataManager
    )

from src.qOptimization import (
    PureFactorOptimizer, PurePortfolioConstraints,
    TrackingErrorOptimizer, TrackingErrorConstraints,
    OptimizationObjective, OptimizationStatus
)

# S3 bucket and folder configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'your-bucket-name')
S3_PREFIX = 'time_series/'

def run_factor_analysis() -> Dict:
    """Run factor analysis and return results"""
            # Get model input from session state
    cfg = FileConfig()
    mgr = FileDataManager(cfg)

    model_input = st.session_state.model_input
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"

    try:        
        if not st.session_state.data_updated:
            st.warning("Please update data first before running analysis.")
            return None

        if 'model_input' not in st.session_state:
            st.error("Model input not found. Please run data update first.")
            return None

        # Get model input from session state
        model_input = st.session_state.model_input
        # st.success(f"Model Input: {model_input.backtest}")

        # Run factor analysis
        # factor_data = file_load_factors(model_input)
        factor_data_dict = {}
        for factor in model_input.params.risk_factors:
            factor_name = factor.value
            factor_data_dict[factor_name] = mgr.load_factors(f"{identifier}_members_{factor_name}")
        # df_ret_long = file_load_returns(model_input)
        df_ret_long = mgr.load_returns(identifier+'_members')

        if factor_data_dict is None or df_ret_long is None:
            st.error("Failed to load factor data or returns. Please check if data update completed successfully.")
            return None

        # Convert factor_data_dict to a dictionary of DataFrames   
        # factor_data = factor_data_dict
        factor_data = {k: v for k, v in factor_data_dict.items()}

        st.session_state.factor_data = factor_data
        st.session_state.df_ret_long = df_ret_long

        # Initialize results dictionary
        results = {}
        
        # Analyze each factor
        for factor_name, factor_df in factor_data.items():
            st.success(f"Running factor {factor_name}")

            # Create EquityFactor instance
            factor = EquityFactor(
                name=factor_name,
                data=factor_df,
                description=f"{factor_name} factor",
                category=factor_name
            )
            
            # Get factor analysis results
            factor_results = factor.analyze_factor_returns(
                returns_data=df_ret_long,
                n_buckets=5,
                method='quantile',
                weighting='equal',
                long_short=True,
                neutralize_size=False
            )
            # st.success(f"factor results: {factor_results.keys()}")
 
            results[factor_name] = factor_results

        st.session_state.factor_analysis_results = results
        return results
        
    except Exception as e:
        st.error(f"Error in factor analysis: {str(e)}")
        return None

def run_portfolio_optimization_v1() -> Dict:
    """Run portfolio optimization and return results"""
    try:
        if not st.session_state.data_updated:
            st.warning("Please update data first before running analysis.")
            return None

        cfg = FileConfig()
        mgr = FileDataManager(cfg)

        model_input = st.session_state.model_input
        identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
        factor_list = [i.value for i in model_input.params.risk_factors]

        # df_ret_long = file_load_returns(model_input)
        df_ret_long = mgr.load_returns(identifier+'_members')

        df_ret_wide = df_ret_long[['date','sid','return']].pivot(
            index='date', columns='sid', values='return')
        df_ret_wide.fillna(0., inplace=True)

        # df_exposures_long = file_load_exposures(model_input)
        df_exposures_long = mgr.load_exposures(identifier+'_members')
        df_exposures = df_exposures_long[['date','sid','variable','exposure']].pivot(
            index=['date','sid'], columns='variable', values='exposure').reset_index(drop=False)
        df_exposures.fillna(0., inplace=True)

        if optimization_objective == OptimizationObjective.PURE_FACTOR.value:
            # Pure factor optimization
            df_pure_return = pd.DataFrame()
            df_pure_portfolio = pd.DataFrame()

            # # Create optimization constraints
            for factor in factor_list:
                st.success(f"Optimizing Pure Factor: {factor.title()}")

                constraints = PurePortfolioConstraints(
                    long_only=False,
                    full_investment=True,
                    factor_neutral=[i for i in factor_list if i!=factor],
                    weight_bounds=(-0.05, 0.05),
                    min_holding=0.01
                )
                
                # Initialize optimizer
                optimizer_pure = PureFactorOptimizer(
                    target_factor=factor,
                    constraints=constraints,
                    normalize_weights=True,
                    parallel_processing=False
                )
                # st.success(f"optimizer_pure constraints: {optimizer_pure.constraints}")

                # Run optimization (example data not provided)
                results = optimizer_pure.optimize(
                    df_ret_wide, 
                    df_exposures, 
                    model_input.backtest.dates_turnover # [str(i) for i in dates_to]
                )
                df_portfolio = results.get('weights_data')
                df_pure_portfolio = pd.concat([df_pure_portfolio, df_portfolio])

                config = bt.BacktestConfig(
                    asset_class=bt.AssetClass.EQUITY,
                    portfolio_type=bt.PortfolioType.LONG_SHORT,
                    model_type=factor,
                    annualization_factor=252
                    )

                backtest = bt.Backtest(config=config)
                df_portfolio.rename(columns={'sid':'ticker', 'weight':'weight'}, inplace=True)
                df_returns = df_ret_long.copy()
                df_returns.rename(columns={'sid':'ticker'}, inplace=True)
                results_bt = backtest.run_backtest(df_returns, df_portfolio, plot=False)
                df_ret_opt = backtest.df_pnl.copy()
                    
                df_pure_return = pd.concat([df_pure_return, df_ret_opt])

            # Store pure factor returns
            df_pure_return_wide = df_pure_return[['factor','return_opt']].pivot(columns='factor',values='return_opt')
            st.session_state.pure_factor_returns = df_pure_return_wide
            st.session_state.df_pure_portfolio = df_pure_portfolio            
        else:
            # Tracking error optimization
            optimizer = TrackingErrorOptimizer(model_input)
            constraints = TrackingErrorConstraints(
                max_weight=max_weight/100.,
                min_holding=0.001,
                num_trades=num_trades,
                tracking_error_max=tracking_error/100.
            )
            results = optimizer.optimize(constraints)

        st.session_state.optimization_results = results
        return results
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        return None

def run_tracking_error_optimization() -> Dict:
    """Run tracking error optimization and return results"""
    try:
        if not st.session_state.data_updated:
            st.warning("Please update data first before running analysis.")
            return None

        cfg = FileConfig()
        mgr = FileDataManager(cfg)

        model_input = st.session_state.model_input
        identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
        factor_list = [i.value for i in model_input.params.risk_factors]

        # Load required data
        df_ret_long = mgr.load_returns(identifier+'_members')
        df_ret_wide = df_ret_long[['date','sid','return']].pivot(
            index='date', columns='sid', values='return')
        df_ret_wide.fillna(0., inplace=True)

        # Load exposures
        df_exposures_long = mgr.load_exposures(identifier+'_members')
        df_exposures = df_exposures_long[['date','sid','variable','exposure']].pivot(
            index=['date','sid'], columns='variable', values='exposure').reset_index(drop=False)
        df_exposures.fillna(0., inplace=True)
        
        # Ensure date format consistency - convert to datetime64[ns] to match other dataframes
        df_exposures['date'] = pd.to_datetime(df_exposures['date'])

        # Load benchmark data
        df_benchmark_prices = mgr.load_prices(identifier)
        df_benchmark_weights = mgr.load_benchmark_weights(identifier)
        
        # Create security master for benchmark returns
        # model_input.backtest.dates_turnover = [str(i.date()) for i in model_input.backtest.dates_turnover]
        security_master = SecurityMasterFactory(model_input=model_input)
        security_master.df_price = mgr.load_prices(identifier+'_members')
        security_master.df_bench = df_benchmark_prices
        security_master.weights_data = df_benchmark_weights
        
        # Get benchmark returns
        benchmark_returns = security_master.df_bench.copy()
        benchmark_exposures = security_master.weights_data.copy()
        
        # Ensure all date columns have consistent datetime64[ns] format
        if 'date' in benchmark_returns.columns:
            benchmark_returns['date'] = pd.to_datetime(benchmark_returns['date'])
        if 'date' in benchmark_exposures.columns:
            benchmark_exposures['date'] = pd.to_datetime(benchmark_exposures['date'])

        st.warning(f"TE constraints: {st.session_state.te_factor_constraints}")

        # Create tracking error optimization constraints
        constraints = TrackingErrorConstraints(
            long_only=st.session_state.te_long_only,
            full_investment=True,
            factor_constraints=st.session_state.te_factor_constraints,
            weight_bounds=(0.0, st.session_state.te_max_weight/100.),
            min_holding=st.session_state.te_min_holding/100.,
            max_names=st.session_state.te_max_names,
            tracking_error_max=st.session_state.te_tracking_error_max/100.
        )

        # Initialize optimizer
        optimizer_te = TrackingErrorOptimizer(
            constraints=constraints,
            normalize_weights=True,
            parallel_processing=False,
            use_integer_constraints=st.session_state.te_use_integer_constraints
        )

        # Data validation before optimization
        st.info("Validating data before optimization...")
        
        # Check if exposures has required columns
        required_exposure_cols = ['date', 'sid']
        missing_exposure_cols = [col for col in required_exposure_cols if col not in df_exposures.columns]
        if missing_exposure_cols:
            st.error(f"Missing columns in df_exposures: {missing_exposure_cols}")
            st.error(f"Available columns: {list(df_exposures.columns)}")
            return None
        
        # Check if benchmark_exposures has required columns
        required_bench_cols = ['date', 'sid']
        missing_bench_cols = [col for col in required_bench_cols if col not in benchmark_exposures.columns]
        if missing_bench_cols:
            st.error(f"Missing columns in benchmark_exposures: {missing_bench_cols}")
            st.error(f"Available columns: {list(benchmark_exposures.columns)}")
            return None
        
        # Check if benchmark_returns has required columns
        if 'return' not in benchmark_returns.columns:
            st.error(f"Missing 'return' column in benchmark_returns")
            st.error(f"Available columns: {list(benchmark_returns.columns)}")
            return None
        
        # Check date formats and alignment
        st.info("Checking date alignment...")
        
        # Convert turnover dates to datetime format to match other dataframes
        st.info("Converting turnover dates to datetime format...")
        try:
            # Convert turnover dates to datetime
            turnover_dates_datetime = [pd.to_datetime(d) for d in model_input.backtest.dates_turnover]
            st.info(f"Converted turnover dates sample: {turnover_dates_datetime[:3]}")
            
            # Update the model_input with converted dates
            # model_input.backtest.dates_turnover = turnover_dates_datetime
            st.success("Turnover dates converted to datetime format")
            
        except Exception as date_conv_error:
            st.error(f"Error converting turnover dates: {date_conv_error}")
            st.error(f"Original turnover dates: {model_input.backtest.dates_turnover[:3]}")
            return None
        
        # Check if dates in exposures match dates_turnover
        # Handle different date formats properly
        if not df_exposures.empty:
            try:
                # Try to get unique dates safely
                unique_dates = df_exposures['date'].unique()
                exposure_dates = set(str(d) for d in unique_dates)
            except Exception as date_error:
                st.error(f"Error processing exposure dates: {date_error}")
                st.error(f"Date column dtype: {df_exposures['date'].dtype}")
                st.error(f"Sample values: {df_exposures['date'].head(3).tolist()}")
                return None
        else:
            exposure_dates = set()
            
        # Now use the converted turnover dates
        turnover_dates = set(str(d) for d in turnover_dates_datetime)
        
        st.info(f"Exposure dates sample: {list(exposure_dates)[:5]}")
        st.info(f"Turnover dates sample: {list(turnover_dates)[:5]}")
        
        # Check for common dates
        common_dates = exposure_dates.intersection(turnover_dates)
        st.info(f"Common dates between exposures and turnover: {len(common_dates)}")
        
        if len(common_dates) == 0:
            st.warning("No common dates found between exposures and turnover dates!")
            st.warning("This might cause the 'sid' KeyError")
        
        # Validate date format consistency before optimization
        st.info("Validating date format consistency...")
        date_dtypes = {}
        if not df_exposures.empty and 'date' in df_exposures.columns:
            date_dtypes['df_exposures'] = str(df_exposures['date'].dtype)
        if not benchmark_returns.empty and 'date' in benchmark_returns.columns:
            date_dtypes['benchmark_returns'] = str(benchmark_returns['date'].dtype)
        if not benchmark_exposures.empty and 'date' in benchmark_exposures.columns:
            date_dtypes['benchmark_exposures'] = str(benchmark_exposures['date'].dtype)
        
        st.info(f"Date column dtypes: {date_dtypes}")
        
        # Check if all date dtypes are consistent
        if len(set(date_dtypes.values())) > 1:
            st.error("Date column dtypes are inconsistent! This will cause merge errors.")
            st.error("All date columns should have the same dtype (preferably datetime64[ns])")
            return None
        else:
            st.success("All date columns have consistent dtypes")
        
        # Additional debugging: Check first date filtering
        if model_input.backtest.dates_turnover:
            first_date = model_input.backtest.dates_turnover[0]
            st.info(f"Testing first date filtering: {first_date}")
            # st.info(f"First date type: {type(first_date)}")
            
            # Test the filtering logic that's used in the optimization
            # Since both dates are now datetime objects, we can compare them directly
            test_exposures = df_exposures[df_exposures['date'] == first_date].copy()
            st.info(f"Exposures for first date {first_date}: {test_exposures.shape[0]} rows")
            
            if test_exposures.empty:
                st.warning(f"No exposures found for date {first_date}")
                st.warning("This will cause the 'sid' KeyError in the optimization loop")
                
                # Try alternative filtering methods
                st.info("Trying alternative date filtering methods...")
                
                # Method 1: String comparison
                test_exposures_str = df_exposures[df_exposures['date'].astype(str) == str(first_date)].copy()
                st.info(f"String comparison result: {test_exposures_str.shape[0]} rows")
                
                # Method 2: Date-only comparison
                test_exposures_date = df_exposures[df_exposures['date'].dt.date == first_date.date()].copy()
                st.info(f"Date-only comparison result: {test_exposures_date.shape[0]} rows")
                
                if test_exposures_date.shape[0] > 0:
                    st.success("Found exposures using date-only comparison!")
                    test_exposures = test_exposures_date
            else:
                st.success(f"Found exposures for date {first_date}")
                st.info(f"Sample exposure columns: {list(test_exposures.columns)}")
                if 'sid' in test_exposures.columns:
                    st.info(f"Sample sids: {test_exposures['sid'].head().tolist()}")
        
        # Run optimization with output capture
        st.info("Starting optimization... This may take a few minutes.")
        
        # Create a container for solver output
        solver_output_container = st.container()
        
        with solver_output_container:
            st.write("**Solver Output:**")
            solver_output_placeholder = st.empty()
        
        # DEBUG: Add breakpoint here to step into the optimize method
        # Option 1: Using pdb (standard Python debugger)
        # import pdb
        # st.warning("🔍 DEBUG MODE: Breakpoint added at line 456. Check your terminal/console for debugger prompt.")
        # pdb.set_trace()  # This will pause execution and drop you into the debugger
        
        # Option 2: Using ipdb (enhanced debugger - uncomment if you have ipdb installed)
        # import ipdb
        # ipdb.set_trace()
        
        # Option 3: Using breakpoint() (Python 3.7+)
        # breakpoint()
        
        # Option 4: Debug prints (uncomment to use instead of breakpoint)
        # st.write("🔍 DEBUG INFO:")
        # st.write(f"df_ret_wide shape: {df_ret_wide.shape}")
        # st.write(f"benchmark_returns shape: {benchmark_returns.shape}")
        # st.write(f"df_exposures shape: {df_exposures.shape}")
        # st.write(f"benchmark_exposures shape: {benchmark_exposures.shape}")
        # st.write(f"dates_turnover: {model_input.backtest.dates_turnover[:3]}")
        # st.write(f"dates_turnover types: {[type(d) for d in model_input.backtest.dates_turnover[:3]]}")
        
        # Capture solver output during optimization
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        solver_output = io.StringIO()

        try:
            with redirect_stdout(solver_output), redirect_stderr(solver_output):
                
                results_te = optimizer_te.optimize(
                    returns=df_ret_wide,
                    benchmark_returns=benchmark_returns,
                    exposures=df_exposures,
                    benchmark_exposures=benchmark_exposures,
                    dates=model_input.backtest.dates_turnover
                )
            
            # Display captured solver output
            captured_output = solver_output.getvalue()
            # if captured_output:
            #     with solver_output_container:
            #         st.text_area("Solver Output:", value=captured_output, height=300)
            # else:
            #     with solver_output_container:
            #         st.info("No solver output captured")
                    
        except Exception as opt_error:
            # Display any captured output even if optimization fails
            captured_output = solver_output.getvalue()
            if captured_output:
                with solver_output_container:
                    st.text_area("Solver Output (before error):", value=captured_output, height=300)
            raise opt_error
        finally:
            solver_output.close()

        # Backtest TE portfolio
        df_weights = results_te['weights_data'].copy()
        df_weights = df_weights.sort_values(['sid','date'])

        # Calculate number of securities in optimal portfolio
        df_weights['n_opt'] = df_weights['weight'] != 0
        avg_securities = df_weights[['date','n_opt']].groupby('date').sum().mean().round(2).squeeze()
        st.success(f"Average number of securities in optimal portfolio: {avg_securities}")

        # Run backtest
        config = bt.BacktestConfig(
            asset_class=bt.AssetClass.EQUITY,
            portfolio_type=bt.PortfolioType.LONG_ONLY,
            model_type='tracking_error',
            annualization_factor=252
        )

        backtest = bt.Backtest(config=config)
        df_portfolio = results_te['weights_data'].copy()
        df_portfolio.rename(columns={'sid':'ticker', 'weight':'weight'}, inplace=True)
        df_returns = df_ret_long.copy()
        df_returns.rename(columns={'sid':'ticker'}, inplace=True)
        results_bt = backtest.run_backtest(df_returns, df_portfolio, plot=False)
        df_ret_opt = backtest.df_pnl.copy()

        # Store results in session state
        st.session_state.te_optimization_results = results_te
        st.session_state.te_backtest_results = results_bt
        st.session_state.te_portfolio_returns = df_ret_opt
        st.session_state.te_weights_data = df_weights

        # Display solver output if available
        st.subheader("Solver Output")
        solver_outputs = []
        
        # Check if we have optimization results with solver output
        if hasattr(results_te, 'get') and 'weights_data' in results_te:
            # The solver output would be in the individual optimization results
            # For now, let's add a placeholder to show where it would appear
            st.info("Solver output will be displayed here once optimization completes")
            
            # Add a button to show solver information
            if st.button("Show Solver Information"):
                st.write("**Available Solvers:**")
                import cvxpy as cp
                st.write(f"Installed solvers: {cp.installed_solvers()}")
                st.write(f"Using solver: {'SCIPY' if st.session_state.te_use_integer_constraints else 'CLARABEL'}")
                st.write(f"Integer constraints: {st.session_state.te_use_integer_constraints}")
        else:
            st.info("No solver output available")

        # Store results for unified optimization handling
        results_unified = {
            'optimization_type': 'tracking_error',
            'meta_data': results_te.get('meta_data'),
            'weights_data': results_te.get('weights_data'),
            'backtest_results': results_bt,
            'portfolio_returns': df_ret_opt
        }
        st.session_state.optimization_results = results_unified

        st.success("Tracking Error Optimization completed successfully!")
        return results_unified  # Changed from results_te to results_unified
    
    except Exception as e:
        import traceback
        
        # Get detailed error information
        error_details = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        
        # Display detailed error information
        st.error(f"Error in tracking error optimization: {error_details['error_type']}: {error_details['error_message']}")
        
        # Show additional debugging information
        with st.expander("Debug Information", expanded=False):
            st.write("**Error Details:**")
            st.code(error_details['traceback'])
            
            # Show data shapes and columns if available
            try:
                st.write("**Data Information:**")
                if 'df_ret_wide' in locals():
                    st.write(f"df_ret_wide shape: {df_ret_wide.shape}")
                    st.write(f"df_ret_wide columns: {list(df_ret_wide.columns)}")
                    st.write(f"df_ret_wide index: {df_ret_wide.index.name}")
                
                if 'df_exposures' in locals():
                    st.write(f"df_exposures shape: {df_exposures.shape}")
                    st.write(f"df_exposures columns: {list(df_exposures.columns)}")
                    if not df_exposures.empty:
                        st.write(f"df_exposures sample:")
                        st.dataframe(df_exposures.head(3))
                
                if 'benchmark_returns' in locals():
                    st.write(f"benchmark_returns shape: {benchmark_returns.shape}")
                    st.write(f"benchmark_returns columns: {list(benchmark_returns.columns)}")
                    if not benchmark_returns.empty:
                        st.write(f"benchmark_returns sample:")
                        st.dataframe(benchmark_returns.head(3))
                
                if 'benchmark_exposures' in locals():
                    st.write(f"benchmark_exposures shape: {benchmark_exposures.shape}")
                    st.write(f"benchmark_exposures columns: {list(benchmark_exposures.columns)}")
                    if not benchmark_exposures.empty:
                        st.write(f"benchmark_exposures sample:")
                        st.dataframe(benchmark_exposures.head(3))
                
                if 'model_input' in locals():
                    st.write(f"dates_turnover: {model_input.backtest.dates_turnover}")
                    st.write(f"dates_turnover type: {type(model_input.backtest.dates_turnover)}")
                    if model_input.backtest.dates_turnover:
                        st.write(f"First date: {model_input.backtest.dates_turnover[0]}")
                        st.write(f"Last date: {model_input.backtest.dates_turnover[-1]}")
                
            except Exception as debug_e:
                st.write(f"Could not retrieve debug information: {debug_e}")
        
        return None

# Refactored portfolio optimization functions

def run_portfolio_optimization() -> Dict:
    """
    Centralized portfolio optimization function that routes to appropriate 
    optimization strategy based on the selected objective.
    
    Returns:
        Dict: Optimization results from the selected strategy
    """
    try:
        if not st.session_state.data_updated:
            st.warning("Please update data first before running analysis.")
            return None

        if 'model_input' not in st.session_state:
            st.error("Model input not found. Please run data update first.")
            return None

        # Get optimization objective from session state
        optimization_objective = st.session_state.optimization_objective
        
        st.info(f"Running {optimization_objective} optimization...")
        
        # Route to appropriate optimization strategy
        if optimization_objective == OptimizationObjective.PURE_FACTOR.value:
            return run_pure_factor_optimization()
        elif optimization_objective == OptimizationObjective.TRACKING_ERROR.value:
            return run_tracking_error_optimization()
        elif optimization_objective == OptimizationObjective.NUM_TRADES.value:
            return run_num_trades_optimization()
        elif optimization_objective == OptimizationObjective.TRANSACTION_COST.value:
            return run_transaction_cost_optimization()
        elif optimization_objective == OptimizationObjective.RISK_PARITY.value:
            return run_risk_parity_optimization()
        else:
            st.error(f"Unknown optimization objective: {optimization_objective}")
            return None
            
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        return None


def run_pure_factor_optimization() -> Dict:
    """
    Run pure factor portfolio optimization for all selected factors.
    
    Returns:
        Dict: Contains pure factor returns and portfolio weights
    """
    try:
        cfg = FileConfig()
        mgr = FileDataManager(cfg)

        model_input = st.session_state.model_input
        identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
        factor_list = [i.value for i in model_input.params.risk_factors]

        # Load required data
        df_ret_long = mgr.load_returns(identifier+'_members')
        df_ret_wide = df_ret_long[['date','sid','return']].pivot(
            index='date', columns='sid', values='return')
        df_ret_wide.fillna(0., inplace=True)

        # Load exposures
        df_exposures_long = mgr.load_exposures(identifier+'_members')
        df_exposures = df_exposures_long[['date','sid','variable','exposure']].pivot(
            index=['date','sid'], columns='variable', values='exposure').reset_index(drop=False)
        df_exposures.fillna(0., inplace=True)

        # Initialize containers for results
        df_pure_return = pd.DataFrame()
        df_pure_portfolio = pd.DataFrame()

        # Optimize each factor
        for factor in factor_list:
            st.success(f"Optimizing Pure Factor: {factor.title()}")

            # Create optimization constraints
            constraints = PurePortfolioConstraints(
                long_only=False,
                full_investment=True,
                factor_neutral=[i for i in factor_list if i != factor],
                weight_bounds=(-0.05, 0.05),
                min_holding=0.01
            )
            
            # Initialize optimizer
            optimizer_pure = PureFactorOptimizer(
                target_factor=factor,
                constraints=constraints,
                normalize_weights=True,
                parallel_processing=False
            )

            # Run optimization
            results = optimizer_pure.optimize(
                df_ret_wide, 
                df_exposures, 
                model_input.backtest.dates_turnover
            )
            df_portfolio = results.get('weights_data')
            df_pure_portfolio = pd.concat([df_pure_portfolio, df_portfolio])

            # Run backtest for this factor
            config = bt.BacktestConfig(
                asset_class=bt.AssetClass.EQUITY,
                portfolio_type=bt.PortfolioType.LONG_SHORT,
                model_type=factor,
                annualization_factor=252
            )

            backtest = bt.Backtest(config=config)
            df_portfolio_bt = df_portfolio.copy()
            df_portfolio_bt.rename(columns={'sid':'ticker', 'weight':'weight'}, inplace=True)
            df_returns = df_ret_long.copy()
            df_returns.rename(columns={'sid':'ticker'}, inplace=True)
            
            results_bt = backtest.run_backtest(df_returns, df_portfolio_bt, plot=False)
            df_ret_opt = backtest.df_pnl.copy()
                
            df_pure_return = pd.concat([df_pure_return, df_ret_opt])

        # Store results in session state
        df_pure_return_wide = df_pure_return[['factor','return_opt']].pivot(
            columns='factor', values='return_opt')
        
        st.session_state.pure_factor_returns = df_pure_return_wide
        st.session_state.df_pure_portfolio = df_pure_portfolio
        
        # Prepare results dictionary
        results = {
            'optimization_type': 'pure_factor',
            'pure_factor_returns': df_pure_return_wide,
            'portfolio_weights': df_pure_portfolio,
            'factor_list': factor_list
        }
        
        st.session_state.optimization_results = results
        return results
        
    except Exception as e:
        st.error(f"Error in pure factor optimization: {str(e)}")
        return None


def run_num_trades_optimization() -> Dict:
    """
    Run portfolio optimization with number of trades constraint.
    
    Returns:
        Dict: Optimization results for num trades strategy
    """
    try:
        st.info("Number of trades optimization not yet implemented")
        # Placeholder for future implementation
        return {
            'optimization_type': 'num_trades',
            'status': 'not_implemented'
        }
    except Exception as e:
        st.error(f"Error in num trades optimization: {str(e)}")
        return None


def run_transaction_cost_optimization() -> Dict:
    """
    Run portfolio optimization with transaction cost minimization.
    
    Returns:
        Dict: Optimization results for transaction cost strategy
    """
    try:
        st.info("Transaction cost optimization not yet implemented")
        # Placeholder for future implementation
        return {
            'optimization_type': 'transaction_cost',
            'status': 'not_implemented'
        }
    except Exception as e:
        st.error(f"Error in transaction cost optimization: {str(e)}")
        return None


def run_risk_parity_optimization() -> Dict:
    """
    Run risk parity portfolio optimization.
    
    Returns:
        Dict: Optimization results for risk parity strategy
    """
    try:
        st.info("Risk parity optimization not yet implemented")
        # Placeholder for future implementation
        return {
            'optimization_type': 'risk_parity',
            'status': 'not_implemented'
        }
    except Exception as e:
        st.error(f"Error in risk parity optimization: {str(e)}")
        return None


# Updated Portfolio Optimization Tab section
def render_portfolio_optimization_tab():
    """
    Render the Portfolio Optimization tab with unified optimization handling.
    """
    st.header("Portfolio Optimization")
    
    # Run optimization button
    if st.button("Run Portfolio Optimization", type="primary"):
        results = run_portfolio_optimization()
    
    # Display results based on optimization type
    if st.session_state.optimization_results is not None:
        results = st.session_state.optimization_results
        optimization_type = results.get('optimization_type', 'unknown')
        
        if optimization_type == 'pure_factor':
            render_pure_factor_results(results)
        elif optimization_type == 'tracking_error':
            render_tracking_error_results(results)
        elif optimization_type in ['num_trades', 'transaction_cost', 'risk_parity']:
            st.info(f"{optimization_type.replace('_', ' ').title()} optimization results will be displayed here once implemented.")
        else:
            st.warning(f"Unknown optimization type: {optimization_type}")


def render_pure_factor_results(results: Dict):
    """
    Render pure factor optimization results.
    
    Args:
        results: Dictionary containing pure factor optimization results
    """
    pure_factor_returns = results.get('pure_factor_returns')
    
    if pure_factor_returns is not None:
        # Plot cumulative factor returns
        st.subheader("Pure Factor Returns")
        fig = px.line(
            pure_factor_returns.cumsum(),
            title="Cumulative Pure Factor Returns"
        )
        st.plotly_chart(fig, use_container_width=True, key="all_factors_pure")
        
        # Factor correlation matrix
        corr_matrix = pure_factor_returns.corr()
        correlation_matrix_display(corr_matrix, "pure_factors_correlation")

        # Monthly factor returns
        monthly_returns = pure_factor_returns.resample('ME').sum()
        fig = px.bar(
            monthly_returns,
            title="Monthly Factor Returns",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        stats = pd.DataFrame({
            'Mean': monthly_returns.mean(),
            'Std': monthly_returns.std(),
            'Sharpe': monthly_returns.mean() / monthly_returns.std() * np.sqrt(12),
            'Max': monthly_returns.max(),
            'Min': monthly_returns.min()
        })
        st.dataframe(stats)


def render_tracking_error_results(results: Dict):
    """
    Render tracking error optimization results.
    
    Args:
        results: Dictionary containing tracking error optimization results
    """
    st.subheader("Optimization Results")
    
    # Format results for display
    meta_data = results.get('meta_data')
    weights_data = results.get('weights_data')
    
    if meta_data is not None:
        st.subheader("Optimization Metrics")
        st.dataframe(meta_data)
    
    if weights_data is not None:
        st.subheader("Portfolio Weights")
        # Show the latest portfolio weights
        latest_date = weights_data['date'].max()
        latest_weights = weights_data[weights_data['date'] == latest_date]
        
        # Convert to display format
        if 'ticker' not in latest_weights.columns and 'sid' in latest_weights.columns:
            latest_weights['ticker'] = latest_weights['sid']
        
        # Display top holdings
        top_holdings = latest_weights.sort_values('weight', ascending=False).head(10)
        st.write(f"Top 10 Holdings (as of {latest_date})")
        st.dataframe(top_holdings[['ticker', 'weight']].set_index('ticker'))
        
        # Plot weights distribution
        fig = px.bar(
            top_holdings,
            x='ticker',
            y='weight',
            title="Top Holdings Weights"
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

def create_model_input() -> EquityFactorModelInput:
    """Create model input from UI selections"""
    # Convert string date to date object if needed
    start_date_obj = st.session_state.start_date
    if isinstance(start_date_obj, str):
        st.success(f"start_date 0 is string: {start_date_obj}")
        start_date_obj = datetime.strptime(start_date_obj, "%Y-%m-%d").date()
    else:
        st.success(f"start_date 0 NOT is string, it's {type(start_date_obj)}: {str(start_date_obj)}")

    end_date_obj = st.session_state.end_date
    if isinstance(end_date_obj, str):
        end_date_obj = datetime.strptime(end_date_obj, "%Y-%m-%d").date()
    
    # Create the model input
    model_input = EquityFactorModelInput(
        params=ParamsConfig(
            aum=Decimal(str(st.session_state.aum)),
            risk_factors=[RiskFactors(factor) for factor in st.session_state.factors],
            n_buckets=5
        ),
        backtest=BacktestConfig(
            data_source=DataSource(st.session_state.data_source),
            universe=Universe(st.session_state.universe),
            currency=Currency.USD,
            frequency=Frequency[st.session_state.frequency],
            start_date=pd.to_datetime(str(st.session_state.start_date)).date(), #start_date_obj,
            end_date=end_date_obj,
            concurrent_download=st.session_state.concurrent
        ),
        regime=RegimeConfig(
            type=RegimeType.VOLATILITY,
            benchmark=VolatilityType.VIX,
            periods=10
        ),
        export=ExportConfig(
            base_path="../data/output"
        )
    )
    
    st.success(f"start_date 0: {model_input.backtest.start_date}")
    # model_input.backtest.start_date = pd.to_datetime('2018-12-31').date()
    # st.success(f"start_date 1: {model_input.backtest.start_date}")
    st.success(f"end_date 0: {model_input.backtest.end_date}")
    
    # Set up dates in the model
    set_model_input_dates_turnover(model_input)
    set_model_input_dates_daily(model_input)
    # st.success(f"dates_turnover: {model_input.backtest.dates_turnover}")
    
    return model_input

def run_data_update_process(model_input: EquityFactorModelInput, 
                            update_history: bool = False) -> bool:
    """
    Run the data update process using the ETL functions.
    
    Args:
        model_input: The model input configuration
        update_history: Whether to update full history or just append latest data
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        model_input.export.update_history = update_history
        
        # Display update type
        if update_history:
            st.info("🔄 Starting FULL HISTORY data update - this will overwrite all historical data")
        else:
            st.info("🔄 Starting INCREMENTAL data update - appending latest data")
        
        st.success(f"Updating data for universe: {model_input.backtest.universe.value}")
        st.success(f"Date range: {model_input.backtest.start_date} to {model_input.backtest.end_date}")
        st.success(f"Selected factors: {[f.value for f in model_input.params.risk_factors]}")

        with st.spinner("Running data update process..."):
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                st.write("📊 **Step 1:** Initializing ETL process...")
                
                # Create a progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container()
                
                # Progress callback function for Streamlit
                def streamlit_progress_callback(step, total_steps, message, progress_type="info"):
                    """Callback function to update Streamlit progress"""
                    try:
                        # Debug: Log the values being passed
                        if step > 0 or total_steps != 7:  # Only log when values seem unusual
                            st.write(f"DEBUG: step={step}, total_steps={total_steps}, message='{message[:50]}...'")
                        
                        # Update progress bar - ensure value is between 0.0 and 1.0
                        if total_steps > 0:
                            progress_value = step / total_steps
                            
                            # Handle case where step might be a percentage (e.g., 14.285714 instead of 1)
                            if progress_value > 1.0:
                                # If progress_value > 1, it might be a percentage, convert it
                                progress_value = progress_value / 100.0
                            
                            # Clamp the value to ensure it's within valid range
                            progress_value = max(0.0, min(1.0, progress_value))
                            progress_bar.progress(progress_value)
                        else:
                            # Handle edge case where total_steps is 0
                            progress_bar.progress(0.0)
                        
                        # Update status text
                        status_text.text(f"Step {step}/{total_steps}: {message}")
                        
                        # Display message in log container
                        with log_container:
                            if progress_type == "error":
                                st.error(f"❌ {message}")
                            elif progress_type == "warning":
                                st.warning(f"⚠️ {message}")
                            elif progress_type == "success":
                                st.success(f"✅ {message}")
                            else:
                                st.info(f"📋 {message}")
                    except Exception as progress_error:
                        # If progress update fails, just log the message without updating progress
                        st.warning(f"Progress update failed: {progress_error}")
                        st.info(f"Step {step}/{total_steps}: {message}")
                
                # Run main ETL process with progress updates
                try:
                    # Run ETL process with progress callback
                    etl_universe_data(model_input, progress_callback=streamlit_progress_callback)
                    
                    st.success("✅ ETL process completed successfully!")
                    progress_bar.progress(100)
                    status_text.text("✅ ETL Complete!")
                    
                except Exception as e:
                    st.error(f"❌ ETL process failed: {str(e)}")
                    return False
                
                st.write("📊 **Step 2:** Data validation and verification...")
                
                # Verify data was created/updated
                cfg = FileConfig()
                mgr = FileDataManager(cfg)
                identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
                
                try:
                    # Check if key data files exist and have content
                    df_prices = mgr.load_prices(identifier+'_members')
                    df_returns = mgr.load_returns(identifier+'_members')
                    df_exposures = mgr.load_exposures(identifier+'_members')
                    
                    if df_prices is not None and not df_prices.empty:
                        unique_securities = len(df_prices.sid.unique())
                        st.success(f"✅ Prices data: {len(df_prices)} records, {unique_securities} securities")
                    else:
                        st.warning("⚠️ Prices data appears to be empty")
                    
                    if df_returns is not None and not df_returns.empty:
                        st.success(f"✅ Returns data: {len(df_returns)} records")
                    else:
                        st.warning("⚠️ Returns data appears to be empty")
                    
                    if df_exposures is not None and not df_exposures.empty:
                        st.success(f"✅ Exposures data: {len(df_exposures)} records")
                    else:
                        st.warning("⚠️ Exposures data appears to be empty")
                    
                    # Check factor data
                    factor_data_status = {}
                    for factor in model_input.params.risk_factors:
                        factor_name = factor.value
                        df_factor = mgr.load_factors(f"{identifier}_members_{factor_name}")
                        if df_factor is not None and not df_factor.empty:
                            factor_data_status[factor_name] = f"✅ {len(df_factor)} records"
                        else:
                            factor_data_status[factor_name] = "⚠️ Empty or missing"
                    
                    st.write("📊 **Factor Data Status:**")
                    for factor, status in factor_data_status.items():
                        st.write(f"   • {factor.title()}: {status}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Data verification warning: {str(e)}")
                
                st.write("📊 **Step 3:** Cloud storage sync...")
                
                # Sync with cloud storage if enabled
                if st.session_state.sync_with_cloud:
                    try:
                        if sync_data_with_s3(model_input):
                            st.success("✅ Data synced with cloud storage successfully!")
                        else:
                            st.warning("⚠️ Failed to sync data with cloud storage")
                    except Exception as e:
                        st.warning(f"⚠️ Cloud sync warning: {str(e)}")
                else:
                    st.info("ℹ️ Cloud storage sync skipped (not enabled)")
                
                st.write("📊 **Step 4:** Finalizing update...")
                
                # Update session state
                st.session_state.data_updated = True
                st.session_state.config_changed = False
                
                # Load the updated data into session state
                try:
                    if load_existing_data(model_input):
                        st.success("✅ Data loaded into session state successfully!")
                    else:
                        st.warning("⚠️ Data loaded but session state update had issues")
                except Exception as e:
                    st.warning(f"⚠️ Session state update warning: {str(e)}")
                
                st.success("🎉 **Data update process completed successfully!**")
                st.balloons()
            
            return True
            
    except Exception as e:
        st.error(f"❌ Error in data update process: {str(e)}")
        st.error("Please check the logs and try again. If the issue persists, consider updating the full history.")
        return False

def load_existing_data(model_input: EquityFactorModelInput) -> bool:
    """
    Load existing data from files using FileDataManager.
    
    Args:
        model_input: The model input configuration
        
    Returns:
        bool: True if data was loaded successfully, False otherwise
    """
    try:
        cfg = FileConfig()
        mgr = FileDataManager(cfg)
        identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"

        # Load all required data
        df_benchmark_prices = mgr.load_prices(identifier)
        df_benchmark_weights = mgr.load_benchmark_weights(identifier)
        df_prices = mgr.load_prices(identifier+'_members')
        df_returns = mgr.load_returns(identifier+'_members')
        df_exposures_long = mgr.load_exposures(identifier+'_members')

        # Load factor data
        factor_data = {}
        for factor in model_input.params.risk_factors:
            factor_name = factor.value
            factor_data[factor_name] = mgr.load_factors(f"{identifier}_members_{factor_name}")

        # Store data in session state
        st.session_state.df_benchmark_prices = df_benchmark_prices
        st.session_state.df_benchmark_weights = df_benchmark_weights
        st.session_state.df_prices = df_prices
        st.session_state.df_ret_long = df_returns
        st.session_state.df_exposures_long = df_exposures_long
        st.session_state.factor_data = factor_data

        # Update universe list
        if df_returns is not None and not df_returns.empty:
            univ_list = sorted(list(df_returns.sid.unique()))
            model_input.backtest.universe_list = univ_list

        return True

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

def run_data_update(force_update=False):
    """Run data update process"""
    # Check if configuration has changed
    config_changed = check_config_changes()
    st.session_state.config_changed = config_changed
    
    # Create model input
    model_input = create_model_input()
    
    # Store model_input in session state
    st.session_state.model_input = model_input
    
    # If config changed but not forcing update, ask user if they want to update
    if config_changed and not force_update:
        st.warning("Configuration has changed. Do you want to update the data?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, update data"):
                force_update = True
        with col2:
            if st.button("No, use existing data"):
                st.session_state.config_changed = False
                return
    
    # Update data if forcing or if the button was pressed
    if force_update or update_data:
        try:
            # Run the data update process
            success = run_data_update_process(model_input, update_history)
            
            if success:
                st.success("Data update completed successfully!")
                st.session_state.data_updated = True
                st.session_state.config_changed = False
                
                # Load the updated data
                load_existing_data(model_input)
            else:
                st.error("Data update failed. Please check the logs for details.")
                
        except Exception as e:
            st.error(f"Error updating data: {str(e)}")
  
# Helper functions
def correlation_matrix_display(corr_matrix, tab_name):
    """
    Create an improved correlation matrix visualization with properly formatted values.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix dataframe
    tab_name : str
        Name of the tab (used to create unique key)
        
    Returns:
    --------
    None, displays the plot in Streamlit
    """
    # Only show one title using Streamlit's header system
    st.subheader("Factor Correlation Matrix")
    
    # Create the heatmap without a title in the figure itself
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdYlGn',  # Red-Yellow-Green color scale
        zmin=-1,
        zmax=1,
        aspect="auto"  # Ensure proper sizing
    )
    
    # Add text annotations with correlation values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            
            # Format the text to show correlation values
            # 1.00 on diagonal, 2 decimal places elsewhere
            if i == j:
                text = "1.00"
            else:
                text = f"{value:.2f}"
                
            fig.add_annotation(
                x=j,
                y=i,
                text=text,
                showarrow=False,
                font=dict(
                    color=text_color,
                    size=12
                )
            )
    
    # Improve layout and formatting
    fig.update_layout(
        height=500,  # Fixed height
        margin=dict(l=40, r=40, t=20, b=40),  # Adjust margins
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=400,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
        ),
        xaxis=dict(
            title=dict(text="Factor", font=dict(size=14)),
            tickfont=dict(size=13)
        ),
        yaxis=dict(
            title=dict(text="Factor", font=dict(size=14)),
            tickfont=dict(size=13)
        )
    )
    
    # Use a unique key based on tab name
    st.plotly_chart(fig, use_container_width=True, key=f"corr_matrix_{tab_name}")

def monthly_returns_heatmap(monthly_returns, tab_name):
    """
    Create an improved heatmap of monthly factor returns with values displayed in each cell.
    
    Parameters:
    -----------
    monthly_returns : pd.DataFrame
        DataFrame with dates as index and factors as columns
    tab_name : str
        Name of the tab (used to create unique key)
        
    Returns:
    --------
    None, displays the plot in Streamlit
    """
    # Only show title using Streamlit's header system
    st.subheader("Monthly Factor Returns")
    
    # Create the heatmap without a title in the figure itself
    fig = px.imshow(
        monthly_returns,
        color_continuous_scale='RdBu',  # Red-White-Blue color scale
        zmin=-0.05,  # Lower bound for color scale
        zmax=0.05,   # Upper bound for color scale
        aspect="auto"  # Ensure proper sizing
    )
    
    # Add text annotations with return values
    for i, date in enumerate(monthly_returns.index):
        for j, factor in enumerate(monthly_returns.columns):
            value = monthly_returns.iloc[i, j]
            
            # Adjust text color based on the background color
            # Darker background = white text, lighter background = black text
            text_color = "white" if abs(value) > 0.025 else "black"
            
            # Format the text to show percentage with sign
            # +2.50% or -1.35% format
            text = f"{value*100:+.2f}%"
                
            fig.add_annotation(
                x=j,
                y=i,
                text=text,
                showarrow=False,
                font=dict(
                    color=text_color,
                    size=10  # Slightly smaller font to fit percentages
                )
            )
    
    # Improve layout and formatting
    fig.update_layout(
        height=max(400, 50 * len(monthly_returns.index)),  # Dynamic height based on number of months
        margin=dict(l=40, r=40, t=20, b=40),  # Adjust margins
        coloraxis_colorbar=dict(
            title="Return",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=400,
            tickvals=[-0.05, -0.025, 0, 0.025, 0.05],
            ticktext=["-5.0%", "-2.5%", "0.0%", "+2.5%", "+5.0%"]
        ),
        xaxis=dict(
            title=dict(text="Factor", font=dict(size=14)),
            tickfont=dict(size=13)
        ),
        yaxis=dict(
            title=dict(text="Date", font=dict(size=14)),
            tickfont=dict(size=13)
        )
    )
    
    # Use a unique key based on tab name
    st.plotly_chart(fig, use_container_width=True, key=f"monthly_heatmap_{tab_name}")

# Initialize AWS S3 client
def init_s3_client():
    """Initialize S3 client with credentials from environment variables"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        return s3_client
    except Exception as e:
        st.error(f"Error initializing S3 client: {str(e)}")
        return None

def upload_to_s3(file_path: str, s3_key: str) -> bool:
    """Upload a file to S3"""
    try:
        s3_client = init_s3_client()
        if s3_client:
            s3_client.upload_file(file_path, S3_BUCKET, s3_key)
            return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
    return False

def download_from_s3(s3_key: str, local_path: str) -> bool:
    """Download a file from S3"""
    try:
        s3_client = init_s3_client()
        if s3_client:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
            return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.warning(f"File {s3_key} not found in S3")
        else:
            st.error(f"Error downloading from S3: {str(e)}")
    except Exception as e:
        st.error(f"Error downloading from S3: {str(e)}")
    return False

def sync_data_with_s3(model_input: EquityFactorModelInput) -> bool:
    """Sync data between local storage and S3"""
    try:
        update_history = model_input.export.update_history
        base_path = model_input.export.base_path
        universe = model_input.backtest.universe.value.replace(' ', '_')
        
        # List of files to sync
        files_to_sync = [
            f"{universe}_members.csv",
            f"{universe}_prices.csv",
            f"{universe}_returns.csv",
            f"{universe}_factors.csv"
        ]
        
        if update_history:
            # Download all files from S3
            for file in files_to_sync:
                s3_key = f"{S3_PREFIX}{file}"
                local_path = os.path.join(base_path, file)
                download_from_s3(s3_key, local_path)
        else:
            # Upload new files to S3
            for file in files_to_sync:
                local_path = os.path.join(base_path, file)
                if os.path.exists(local_path):
                    s3_key = f"{S3_PREFIX}{file}"
                    upload_to_s3(local_path, s3_key)
        
        return True
    except Exception as e:
        st.error(f"Error syncing data with S3: {str(e)}")
        return False

# Set page config
st.set_page_config(
    page_title="Equity Factor Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Equity Factor Analysis Platform")
st.markdown("""
This platform provides tools for equity factor analysis, portfolio optimization, and risk management.
""")

# Initialize session state variables if they don't exist
if 'model_input' not in st.session_state:
    st.session_state.model_input = None
if 'factor_data' not in st.session_state:
    st.session_state.factor_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'factor_analysis_results' not in st.session_state:
    st.session_state.factor_analysis_results = None
if 'pure_factor_returns' not in st.session_state:
    st.session_state.pure_factor_returns = None
if 'df_ret_long' not in st.session_state:
    st.session_state.df_ret_long = None
if 'df_pure_portfolio' not in st.session_state:
    st.session_state.df_pure_portfolio = None
if 'data_updated' not in st.session_state:
    st.session_state.data_updated = False
if 'config_changed' not in st.session_state:
    st.session_state.config_changed = False
if 'concurrent' not in st.session_state:
    st.session_state.concurrent = False

# Initialize TE optimization session state variables
if 'te_optimization_results' not in st.session_state:
    st.session_state.te_optimization_results = None
if 'te_backtest_results' not in st.session_state:
    st.session_state.te_backtest_results = None
if 'te_portfolio_returns' not in st.session_state:
    st.session_state.te_portfolio_returns = None
if 'te_weights_data' not in st.session_state:
    st.session_state.te_weights_data = None
if 'te_factor_constraints' not in st.session_state:
    st.session_state.te_factor_constraints = {}

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Data Management Section
    st.subheader("Data Management")
    
    # Data loading mode selection
    data_mode = st.radio(
        "Data Mode",
        options=["Load Existing Data", "Update Data"],
        key="data_mode"
    )
    
    if data_mode == "Update Data":
        # Update options
        update_history = st.checkbox("Update Full History", value=False, key="update_history")
        if update_history:
            st.warning("⚠️ This will overwrite all historical data files. Are you sure?")
            confirm_update = st.checkbox("Yes, I understand and want to proceed", value=False)
            if not confirm_update:
                st.stop()
        
        sync_with_cloud = st.checkbox("Sync with Cloud Storage", value=False, key="sync_with_cloud")
        
        # Add Concurrent checkbox for async downloads
        concurrent = st.checkbox(
            "Concurrent", value=False, key="concurrent",
            help="Enable concurrent (async) downloads for faster data retrieval."
        )
        
        # Update data button
        update_data = st.button("Update Data", type="secondary")
    else:
        # Load existing data button
        load_data = st.button("Load Existing Data", type="primary")
        if load_data:
            # Create model input
            model_input = create_model_input()
            st.session_state.model_input = model_input
            
            # Load data
            if load_existing_data(model_input):
                st.success("Data loaded successfully!")
                st.session_state.data_updated = True
            else:
                st.error("Failed to load data. Please check if data files exist.")
    
    # Data Source selection
    data_source = st.selectbox(
        "Data Source",
        options=[source.value for source in DataSource],
        index=0,
        key="data_source"
    )
    
    # Universe selection
    universe_options = [universe.value for universe in Universe]
    selected_universe = st.selectbox(
        "Select Universe", 
        options=universe_options,
        key="universe"
    )

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        # Parse date from string if needed
        # default_start = datetime(2019, 12, 31).date()
        start_date = st.date_input(
            "Start Date",
            # value=default_start,
            value=datetime(2019, 12, 31).date(),
            key="start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            key="end_date"
        )
    
    # Frequency selection
    frequency_options = list(Frequency.__members__.keys())
    selected_frequency = st.selectbox(
        "Rebalancing Frequency",
        options=frequency_options,
        key="frequency"
    )
    
    # Factor selection
    st.subheader("Risk Factors")
    factor_options = [factor.value for factor in RiskFactors]
    default_factors = ["beta", "size", "value", "momentum"]
    selected_factors = st.multiselect(
        "Select Factors",
        options=factor_options,
        default=default_factors,
        key="factors"
    )
    
    # Optimization settings
    st.subheader("Optimization")
    optimization_objective = st.selectbox(
        "Objective",
        options=[obj.value for obj in OptimizationObjective],
        key="optimization_objective"
    )
    
    # Dynamic optimization constraints based on objective
    if optimization_objective == OptimizationObjective.TRACKING_ERROR.value:
        # Tracking Error specific parameters
        st.subheader("Tracking Error Parameters")
        
        te_long_only = st.checkbox(
            "Long Only Portfolio", 
            value=True, 
            key="te_long_only",
            help="Restrict portfolio to long positions only"
        )
        
        te_max_weight = st.slider(
            "Max Position Weight (%)", 
            min_value=1, 
            max_value=20, 
            value=10,
            key="te_max_weight"
        ) / 100
        
        te_min_holding = st.slider(
            "Min Holding Size (%)", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0,
            key="te_min_holding"
        ) / 100
        
        te_max_names = st.slider(
            "Max Number of Positions", 
            min_value=5, 
            max_value=100, 
            value=20,
            key="te_max_names"
        )
        
        te_tracking_error_max = st.slider(
            "Max Tracking Error (%)", 
            min_value=1, 
            max_value=20, 
            value=5,
            key="te_tracking_error_max"
        ) / 100
        
        te_use_integer_constraints = st.checkbox(
            "Use Integer Constraints", 
            value=True, 
            key="te_use_integer_constraints",
            help="Use integer constraints for position counting"
        )
        
        # Factor constraints
        st.subheader("Factor Constraints")
        # factor_list = [i.value for i in RiskFactors]
        factor_list = st.session_state.factors  # From the multiselect
        te_factor_constraints = {}
        
        for factor in factor_list:
            col1, col2 = st.columns(2)
            with col1:
                constraint_type = st.selectbox(
                    f"{factor.title()} Constraint Type",
                    options=["Range", "Tolerance"],
                    key=f"te_{factor}_type"
                )
            with col2:
                if constraint_type == "Range":
                    lower = st.number_input(
                        f"{factor.title()} Lower Bound",
                        min_value=-1.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        key=f"te_{factor}_lower"
                    )
                    upper = st.number_input(
                        f"{factor.title()} Upper Bound",
                        min_value=-1.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        key=f"te_{factor}_upper"
                    )
                    te_factor_constraints[factor] = (lower, upper)
                else:
                    tolerance = st.number_input(
                        f"{factor.title()} Tolerance",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.03,
                        step=0.01,
                        key=f"te_{factor}_tolerance"
                    )
                    te_factor_constraints[factor] = tolerance
        
        # Store factor constraints in session state
        st.session_state.te_factor_constraints = te_factor_constraints
        
    elif optimization_objective != OptimizationObjective.PURE_FACTOR.value:
        num_trades = st.slider(
            "Target Number of Trades", 
            min_value=5, 
            max_value=500, 
            value=30,
            key="num_trades"
        )
        tracking_error = st.slider(
            "Max Tracking Error (%)", 
            min_value=1, 
            max_value=10, 
            value=5,
            key="tracking_error"
        ) / 100
    
    max_weight = st.slider(
        "Max Position Weight (%)", 
        min_value=1, 
        max_value=20, 
        value=5,
        key="weight_max"
    ) / 100
    
    # AUM
    aum = st.number_input(
        "AUM (millions)", 
        min_value=1, 
        value=100,
        key="aum"
    )
    
# Function to check if configuration has changed
def check_config_changes():
    """Check if configuration has changed since last update"""
    if st.session_state.model_input is None:
        return True
    
    current_config = {
        "data_source": st.session_state.data_source,
        "universe": st.session_state.universe,
        "start_date": st.session_state.start_date,
        "end_date": st.session_state.end_date,
        "frequency": st.session_state.frequency,
        "factors": sorted(st.session_state.factors),
        "aum": st.session_state.aum
    }
    
    model_input = st.session_state.model_input
    
    stored_config = {
        "data_source": model_input.backtest.data_source, # .value,
        "universe": model_input.backtest.universe.value,
        "start_date": pd.to_datetime(model_input.backtest.start_date).date(),
        "end_date": pd.to_datetime(model_input.backtest.end_date).date(), # model_input.backtest.end_date,
        "frequency": model_input.backtest.frequency.name,
        "factors": sorted([f.value for f in model_input.params.risk_factors]),
        "aum": float(model_input.params.aum)
    }
    
    return current_config != stored_config

# Run data update or load existing data based on mode
if st.session_state.data_mode == "Update Data" and update_data:
    run_data_update(force_update=True)
elif st.session_state.model_input is None:
    # Initialize model input if not already
    model_input = create_model_input()
    st.session_state.model_input = model_input
    # Don't automatically load data - wait for user to click button
elif st.session_state.config_changed:
    # Check if config has changed and update if needed
    run_data_update()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Factor Analysis",
    "Portfolio Optimization",
    "Pure Portfolios",
    "Tracking Error Optimization",
    "Risk Analysis",
    "Documentation"
])

# Factor Analysis Tab
with tab1:
    st.header("Factor Analysis")
    
    # Run factor analysis button
    if st.button("Run Factor Analysis", type="primary"):
        results = run_factor_analysis()
    
    if st.session_state.factor_analysis_results is not None:
        # Factor selection for detailed analysis
        selected_factor = st.selectbox(
            "Select Factor for Detailed Analysis",
            list(st.session_state.factor_analysis_results.keys())
        )
        
        if selected_factor:
            factor_results = st.session_state.factor_analysis_results[selected_factor]
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portfolio Returns")
                # Plot cumulative returns
                fig = px.line(
                    factor_results['bucket_returns'].cumsum(),
                    title=f"Cumulative Returns - {selected_factor}"
                )
                # st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(fig, use_container_width=True, key=f"single_factor_bucket_{selected_factor}")
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(factor_results['portfolio_stats'])
            
            with col2:
                st.subheader("Factor Exposure Distribution")
                # Plot factor distribution
                fig = px.histogram(
                    st.session_state.factor_data[selected_factor],
                    x='value',
                    title=f"Factor Exposure Distribution - {selected_factor}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display turnover statistics
                st.subheader("Turnover Analysis")
                st.dataframe(factor_results['turnover'])
            
            # Factor autocorrelation
            st.subheader("Factor Autocorrelation")
            # Calculate autocorrelation for each security's factor exposure
            factor_data = st.session_state.factor_data[selected_factor]
            wide_data = factor_data.pivot(index='date', columns='sid', values='value')
            
            # Calculate autocorrelation for each security
            autocorr_dict = {}
            for col in wide_data.columns:
                # Drop NaN values and calculate autocorrelation
                series = wide_data[col].dropna()
                if len(series) > 1:  # Need at least 2 points for autocorrelation
                    autocorr_dict[col] = series.autocorr()
            
            # Convert to DataFrame for better visualization
            autocorr_df = pd.DataFrame.from_dict(autocorr_dict, orient='index', columns=['autocorr'])
            
            # Plot autocorrelation distribution
            fig = px.histogram(
                autocorr_df,
                x='autocorr',
                title=f"Factor Exposure Autocorrelation Distribution - {selected_factor}",
                labels={'autocorr': 'Autocorrelation'},
                nbins=50
            )
            
            # Add vertical line at mean
            mean_autocorr = autocorr_df['autocorr'].mean()
            fig.add_vline(
                x=mean_autocorr,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_autocorr:.3f}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display autocorrelation statistics
            st.subheader("Autocorrelation Statistics")
            stats_df = pd.DataFrame({
                'Mean': [autocorr_df['autocorr'].mean()],
                'Median': [autocorr_df['autocorr'].median()],
                'Std': [autocorr_df['autocorr'].std()],
                'Min': [autocorr_df['autocorr'].min()],
                'Max': [autocorr_df['autocorr'].max()],
                'Positive %': [(autocorr_df['autocorr'] > 0).mean() * 100]
            })
            st.dataframe(stats_df.round(3))

# Portfolio Optimization Tab
# with tab2:
#     st.header("Portfolio Optimization")
    
#     # Run optimization button
#     if st.button("Run Portfolio Optimization", type="primary"):
#         results = run_portfolio_optimization()
    
#     if st.session_state.optimization_results is not None or st.session_state.pure_factor_returns is not None:
#         optimization_objective = st.session_state.optimization_objective
        
#         if optimization_objective == OptimizationObjective.PURE_FACTOR.value:
#             # Pure factor optimization results
#             if st.session_state.pure_factor_returns is not None:
#                 # Plot cumulative factor returns
#                 st.subheader("Pure Factor Returns")
#                 fig = px.line(
#                     st.session_state.pure_factor_returns.cumsum(),
#                     title="Cumulative Pure Factor Returns"
#                 )
#                 # st.plotly_chart(fig, use_container_width=True)
#                 st.plotly_chart(fig, use_container_width=True, key="all_factors_pure")
                
#                 # Factor correlation matrix
#                 # st.subheader("Factor Correlation Matrix")
#                 corr_matrix = st.session_state.pure_factor_returns.corr()
#                 correlation_matrix_display(corr_matrix, "pure_factors_correlation")

#                 # Monthly factor returns
#                 # st.subheader("Monthly Factor Returns")
#                 monthly_returns = st.session_state.pure_factor_returns.resample('ME').sum()
#                 # monthly_returns_heatmap(monthly_returns, "portfolio_opt")
#                 fig = px.bar(
#                     monthly_returns,
#                     title="Monthly Factor Returns",
#                     barmode='group'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
                
#                 # Summary statistics
#                 st.subheader("Summary Statistics")
#                 stats = pd.DataFrame({
#                     'Mean': monthly_returns.mean(),
#                     'Std': monthly_returns.std(),
#                     'Sharpe': monthly_returns.mean() / monthly_returns.std() * np.sqrt(12),
#                     'Max': monthly_returns.max(),
#                     'Min': monthly_returns.min()
#                 })
#                 st.dataframe(stats)
#         else:
#             # Tracking error optimization results
#             st.subheader("Optimization Results")
#             if isinstance(st.session_state.optimization_results, dict):
#                 # Format results for display
#                 meta_data = st.session_state.optimization_results.get('meta_data')
#                 weights_data = st.session_state.optimization_results.get('weights_data')
                
#                 if meta_data is not None:
#                     st.subheader("Optimization Metrics")
#                     st.dataframe(meta_data)
                
#                 if weights_data is not None:
#                     st.subheader("Portfolio Weights")
#                     # Show the latest portfolio weights
#                     latest_date = weights_data['date'].max()
#                     latest_weights = weights_data[weights_data['date'] == latest_date]
                    
#                     # Convert to display format
#                     if 'ticker' not in latest_weights.columns and 'sid' in latest_weights.columns:
#                         latest_weights['ticker'] = latest_weights['sid']
                    
#                     # Display top holdings
#                     top_holdings = latest_weights.sort_values('weight', ascending=False).head(10)
#                     st.write(f"Top 10 Holdings (as of {latest_date})")
#                     st.dataframe(top_holdings[['ticker', 'weight']].set_index('ticker'))
                    
#                     # Plot weights distribution
#                     fig = px.bar(
#                         top_holdings,
#                         x='ticker',
#                         y='weight',
#                         title="Top Holdings Weights"
#                     )
#                     fig.update_layout(yaxis_tickformat='.1%')
#                     st.plotly_chart(fig, use_container_width=True)

# Portfolio Optimization Tab
with tab2:
    st.header("Portfolio Optimization")
    
    # Run optimization button
    if st.button("Run Portfolio Optimization", type="primary"):
        results = run_portfolio_optimization()
    
    # Display results based on optimization type
    if st.session_state.optimization_results is not None:
        results = st.session_state.optimization_results
        optimization_type = results.get('optimization_type', 'unknown')
        
        if optimization_type == 'pure_factor':
            render_pure_factor_results(results)
        elif optimization_type == 'tracking_error':
            render_tracking_error_results(results)
        elif optimization_type in ['num_trades', 'transaction_cost', 'risk_parity']:
            st.info(f"{optimization_type.replace('_', ' ').title()} optimization results will be displayed here once implemented.")
        else:
            st.warning(f"Unknown optimization type: {optimization_type}")


# Pure Portfolios Tab
with tab3:
    st.header("Pure Factor Portfolios")
    
    if st.session_state.pure_factor_returns is not None:
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date_pure = st.date_input(
                "Start Date",
                value=st.session_state.pure_factor_returns.index.min().date(),
                key="pure_start_date"
            )
        with col2:
            end_date_pure = st.date_input(
                "End Date",
                value=st.session_state.pure_factor_returns.index.max().date(),
                key="pure_end_date"
            )
        
        # Filter returns by date range
        mask = (st.session_state.pure_factor_returns.index.date >= start_date_pure) & \
               (st.session_state.pure_factor_returns.index.date <= end_date_pure)
        filtered_returns = st.session_state.pure_factor_returns[mask]
        
        # Factor selection
        factor_options = [i.title() for i in list(filtered_returns.columns)] 
        factor_options = factor_options + ["All Factors"]
        selected_factor_pure = st.selectbox(
            "Select Factor",
            options=factor_options,
            key="pure_factor_select"
        )
        
        if selected_factor_pure == "All Factors":
            # Display all factors analysis
            st.subheader("Cumulative Factor Returns")
            fig = px.line(
                filtered_returns.cumsum(),
                title="Cumulative Pure Factor Returns"
            )
            # st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True, key="all_factors_cumulative")
            
            # Factor correlation matrix
            # st.subheader("Factor Correlation Matrix")
            corr_matrix = filtered_returns.corr()
            correlation_matrix_display(corr_matrix, "all_pure_factors_correlation")
            
            # Monthly returns heatmap
            # st.subheader("Monthly Factor Returns")
            monthly_returns = filtered_returns.resample('ME').sum()
            # monthly_returns_heatmap(monthly_returns, "pure_portfolios")
            fig = px.imshow(
                monthly_returns,
                title="Monthly Factor Returns Heatmap",
                color_continuous_scale='RdBu'
            )
            # st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True, key="all_factors_monthly")
            
            # Summary statistics for all factors
            st.subheader("Summary Statistics")
            stats = pd.DataFrame({
                'Mean': filtered_returns.mean() * 252,
                'Volatility': filtered_returns.std() * np.sqrt(252),
                'Sharpe': (filtered_returns.mean() * 252) / (filtered_returns.std() * np.sqrt(252)),
                'Max': filtered_returns.max(),
                'Min': filtered_returns.min(),
                'Skewness': filtered_returns.skew(),
                'Kurtosis': filtered_returns.kurtosis()
            })
            st.dataframe(stats.round(3))
            
        else:
            # Display single factor analysis
            selected_factor_pure = selected_factor_pure.lower()
            factor_returns = filtered_returns[selected_factor_pure]
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cumulative Returns")
                fig = px.line(
                    factor_returns.cumsum(),
                    title=f"Cumulative Returns - {selected_factor_pure}"
                )
                # st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(fig, use_container_width=True, key=f"single_factor_cumulative_{selected_factor_pure}")

                # Monthly returns
                st.subheader("Monthly Returns")
                monthly_returns = factor_returns.resample('ME').sum()
                fig = px.bar(
                    monthly_returns,
                    title=f"Monthly Returns - {selected_factor_pure}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Summary statistics
                st.subheader("Performance Statistics")
                stats = pd.DataFrame({
                    'Metric': [
                        'Annualized Return',
                        'Annualized Volatility',
                        'Sharpe Ratio',
                        'Maximum Drawdown',
                        'Skewness',
                        'Kurtosis'
                    ],
                    'Value': [
                        f"{factor_returns.mean() * 252:.2%}",
                        f"{factor_returns.std() * np.sqrt(252):.2%}",
                        f"{(factor_returns.mean() * 252) / (factor_returns.std() * np.sqrt(252)):.2f}",
                        f"{factor_returns.cumsum().min():.2%}",
                        f"{factor_returns.skew():.2f}",
                        f"{factor_returns.kurtosis():.2f}"
                    ]
                })
                st.dataframe(stats, hide_index=True)
                
                # Rolling statistics
                window = 22 # Rolling 1-month (approximately 22 trading days)
                st.subheader("Rolling Statistics (1 Month)")
                rolling_vol = factor_returns.rolling(window=window).std() * np.sqrt(252)
                rolling_sharpe = (factor_returns.rolling(window=window).mean() * 252) / \
                               (factor_returns.rolling(window=window).std() * np.sqrt(252))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    name='Volatility'
                ))
                fig.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    name='Sharpe Ratio',
                    yaxis='y2'
                ))
                fig.update_layout(
                    title="Rolling Statistics",
                    yaxis=dict(title="Volatility"),
                    yaxis2=dict(title="Sharpe Ratio", overlaying='y', side='right')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio holdings
            st.subheader("Portfolio Holdings")
            if st.session_state.df_pure_portfolio is not None:
                # Filter portfolio data for selected factor
                factor_portfolio = st.session_state.df_pure_portfolio[
                    st.session_state.df_pure_portfolio['factor'] == selected_factor_pure
                ]
                
                if not factor_portfolio.empty:
                    # Get latest holdings
                    latest_date = factor_portfolio['date'].max()
                    latest_holdings = factor_portfolio[factor_portfolio['date'] == latest_date].copy()
                    
                    # Display holdings
                    st.write(f"Latest Holdings (as of {latest_date})")
                    if 'ticker' not in latest_holdings.columns:
                        latest_holdings['ticker'] = latest_holdings['sid']
                    holdings_df = latest_holdings[['ticker', 'weight']].sort_values('weight', ascending=False)
                    st.dataframe(holdings_df.style.format({'weight': '{:.2%}'}))
                    
                    # Plot holdings distribution
                    fig = px.bar(
                        holdings_df,
                        x='ticker',
                        y='weight',
                        title=f"Portfolio Holdings - {selected_factor_pure}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Portfolio Turnover Analysis
                    st.subheader("Portfolio Turnover Analysis")
                    
                    # Calculate turnover between rebalance dates
                    turnover_dates = sorted(factor_portfolio['date'].unique())
                    turnover_series = []

                    if 'ticker' not in factor_portfolio.columns:
                        factor_portfolio['ticker'] = factor_portfolio['sid']
                    
                    for i in range(1, len(turnover_dates)):
                        prev_date = turnover_dates[i-1]
                        curr_date = turnover_dates[i]
                        
                        prev_weights = factor_portfolio[factor_portfolio['date'] == prev_date].set_index('ticker')['weight']
                        curr_weights = factor_portfolio[factor_portfolio['date'] == curr_date].set_index('ticker')['weight']
                        
                        # Calculate one-way turnover
                        turnover = abs(curr_weights - prev_weights).sum() / 2
                        turnover_series.append({
                            'date': curr_date,
                            'turnover': turnover
                        })
                    
                    if turnover_series:
                        turnover_df = pd.DataFrame(turnover_series)
                        
                        # Plot turnover
                        fig = px.bar(
                            turnover_df,
                            x='date',
                            y='turnover',
                            title=f"Portfolio Turnover - {selected_factor_pure}"
                        )
                        fig.update_layout(yaxis_tickformat='.1%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display turnover statistics
                        turnover_stats = pd.DataFrame({
                            'Metric': [
                                'Average Turnover',
                                'Max Turnover',
                                'Min Turnover',
                                'Turnover Std Dev'
                            ],
                            'Value': [
                                f"{turnover_df['turnover'].mean():.1%}",
                                f"{turnover_df['turnover'].max():.1%}",
                                f"{turnover_df['turnover'].min():.1%}",
                                f"{turnover_df['turnover'].std():.1%}"
                            ]
                        })
                        st.dataframe(turnover_stats, hide_index=True)
                    else:
                        st.info("Not enough data to calculate turnover.")
                else:
                    st.info(f"No portfolio data available for {selected_factor_pure}")
    else:
        st.info("Please run portfolio optimization first to view pure factor portfolios.")

# Tracking Error Optimization Tab
with tab4:
    st.header("Tracking Error Optimization")
    
    # Run TE optimization button
    if st.button("Run Tracking Error Optimization", type="primary", key="te_optimization_btn"):
        results = run_tracking_error_optimization()
    
    if st.session_state.te_optimization_results is not None:
        # Display optimization results
        st.subheader("Optimization Results")
        
        # Meta data summary
        meta_data = st.session_state.te_optimization_results.get('meta_data')
        weights_data = st.session_state.te_optimization_results.get('weights_data')
        
        if meta_data is not None and not meta_data.empty:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_te = meta_data['tracking_error'].mean()
                st.metric("Average Tracking Error", f"{avg_te:.2%}")
            
            with col2:
                avg_opt_time = meta_data['optimization_time'].mean()
                st.metric("Avg Optimization Time", f"{avg_opt_time:.2f}s")
            
            with col3:
                success_rate = (meta_data['status'] == 'success').mean()
                st.metric("Success Rate", f"{success_rate:.1%}")
            
            with col4:
                avg_obj_value = meta_data['objective_value'].mean()
                st.metric("Avg Objective Value", f"{avg_obj_value:.4f}")
            
            # Optimization metrics over time
            st.subheader("Optimization Metrics Over Time")
            
            # Plot tracking error over time
            fig = px.line(
                meta_data.set_index('date')['tracking_error'],
                title="Tracking Error Over Time",
                labels={'value': 'Tracking Error', 'date': 'Date'}
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot optimization time over time
            fig = px.bar(
                meta_data,
                x='date',
                y='optimization_time',
                title="Optimization Time by Date"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Factor exposures over time
            st.subheader("Factor Exposures Over Time")
            
            # Get factor exposure columns
            factor_cols = [col for col in meta_data.columns if col.startswith(('beta_', 'momentum_', 'size_', 'value_'))]
            if factor_cols:
                # Create factor exposure plots
                for factor in ['beta', 'momentum', 'size', 'value']:
                    factor_cols_subset = [col for col in factor_cols if col.startswith(factor)]
                    if factor_cols_subset:
                        fig = go.Figure()
                        
                        for col in factor_cols_subset:
                            exposure_type = col.split('_')[-1]
                            fig.add_trace(go.Scatter(
                                x=meta_data['date'],
                                y=meta_data[col],
                                mode='lines+markers',
                                name=f"{factor.title()} - {exposure_type.title()}",
                                line=dict(width=2)
                            ))
                        
                        fig.update_layout(
                            title=f"{factor.title()} Factor Exposures Over Time",
                            xaxis_title="Date",
                            yaxis_title="Exposure",
                            yaxis_tickformat='.1%'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio weights analysis
        if weights_data is not None and not weights_data.empty:
            st.subheader("Portfolio Weights Analysis")
            
            # Latest portfolio holdings
            latest_date = weights_data['date'].max()
            latest_weights = weights_data[weights_data['date'] == latest_date].copy()
            
            # Filter out zero weights
            latest_weights = latest_weights[latest_weights['weight'] > 0].sort_values('weight', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Latest Portfolio Holdings (as of {latest_date})**")
                
                # Display top holdings
                top_holdings = latest_weights.head(15)
                if 'ticker' not in top_holdings.columns:
                    top_holdings['ticker'] = top_holdings['sid']
                
                # Format weights as percentages
                display_weights = top_holdings[['ticker', 'weight']].copy()
                display_weights['weight'] = display_weights['weight'].apply(lambda x: f"{x:.2%}")
                display_weights = display_weights.set_index('ticker')
                
                st.dataframe(display_weights, use_container_width=True)
            
            with col2:
                # Portfolio concentration
                st.write("**Portfolio Concentration**")
                
                # Calculate concentration metrics
                total_weight = latest_weights['weight'].sum()
                top_5_weight = latest_weights.head(5)['weight'].sum()
                top_10_weight = latest_weights.head(10)['weight'].sum()
                herfindahl = (latest_weights['weight'] ** 2).sum()
                
                concentration_metrics = pd.DataFrame({
                    'Metric': ['Top 5 Holdings', 'Top 10 Holdings', 'Herfindahl Index', 'Total Weight'],
                    'Value': [
                        f"{top_5_weight:.1%}",
                        f"{top_10_weight:.1%}",
                        f"{herfindahl:.3f}",
                        f"{total_weight:.1%}"
                    ]
                })
                st.dataframe(concentration_metrics.set_index('Metric'), use_container_width=True)
            
            # Portfolio weights distribution
            st.subheader("Portfolio Weights Distribution")
            
            # Plot weights distribution
            fig = px.histogram(
                latest_weights,
                x='weight',
                nbins=20,
                title="Distribution of Portfolio Weights",
                labels={'weight': 'Weight', 'count': 'Number of Holdings'}
            )
            fig.update_layout(xaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top holdings bar chart
            fig = px.bar(
                latest_weights.head(20),
                x='ticker' if 'ticker' in latest_weights.columns else 'sid',
                y='weight',
                title="Top 20 Holdings by Weight"
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        # Backtest results
        if st.session_state.te_backtest_results is not None:
            st.subheader("Backtest Performance")
            
            bt_results = st.session_state.te_backtest_results
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cumulative Return", f"{bt_results.cumulative_return_benchmark:.2%}")
            
            with col2:
                st.metric("Sharpe Ratio", f"{bt_results.sharpe_ratio_benchmark:.2f}")
            
            # with col3:
            #     st.metric("Max Drawdown", f"{bt_results.max_drawdown_benchmark:.2%}")
            
            # with col4:
            #     st.metric("Volatility", f"{bt_results.volatility_benchmark:.2%}")
            
            # Performance comparison plot
            if st.session_state.te_portfolio_returns is not None:
                st.subheader("Portfolio vs Benchmark Performance")
                
                # Create performance comparison
                df_perf = st.session_state.te_portfolio_returns.copy()
                
                # Plot cumulative returns
                fig = px.line(
                    df_perf[['return_opt', 'return_benchmark']].cumsum(),
                    title="Cumulative Returns: Portfolio vs Benchmark",
                    labels={'value': 'Cumulative Return', 'index': 'Date'}
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling tracking error
                window = 22  # 1 month
                rolling_te = df_perf['return_opt'].rolling(window=window).std() - df_perf['return_benchmark'].rolling(window=window).std()
                
                fig = px.line(
                    rolling_te,
                    title="Rolling Tracking Error (1 Month)",
                    labels={'value': 'Tracking Error', 'index': 'Date'}
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio turnover analysis
        if weights_data is not None and not weights_data.empty:
            st.subheader("Portfolio Turnover Analysis")
            
            # Calculate turnover between rebalance dates
            turnover_dates = sorted(weights_data['date'].unique())
            turnover_series = []
            
            for i in range(1, len(turnover_dates)):
                prev_date = turnover_dates[i-1]
                curr_date = turnover_dates[i]
                
                prev_weights = weights_data[weights_data['date'] == prev_date].set_index('sid')['weight']
                curr_weights = weights_data[weights_data['date'] == curr_date].set_index('sid')['weight']
                
                # Align indices
                common_sids = prev_weights.index.intersection(curr_weights.index)
                prev_weights = prev_weights.reindex(common_sids).fillna(0)
                curr_weights = curr_weights.reindex(common_sids).fillna(0)
                
                # Calculate one-way turnover
                turnover = abs(curr_weights - prev_weights).sum() / 2
                turnover_series.append({
                    'date': curr_date,
                    'turnover': turnover
                })
            
            if turnover_series:
                turnover_df = pd.DataFrame(turnover_series)
                
                # Plot turnover
                fig = px.bar(
                    turnover_df,
                    x='date',
                    y='turnover',
                    title="Portfolio Turnover by Rebalance Date"
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
                
                # Turnover statistics
                turnover_stats = pd.DataFrame({
                    'Metric': [
                        'Average Turnover',
                        'Max Turnover',
                        'Min Turnover',
                        'Turnover Std Dev'
                    ],
                    'Value': [
                        f"{turnover_df['turnover'].mean():.1%}",
                        f"{turnover_df['turnover'].max():.1%}",
                        f"{turnover_df['turnover'].min():.1%}",
                        f"{turnover_df['turnover'].std():.1%}"
                    ]
                })
                st.dataframe(turnover_stats.set_index('Metric'), use_container_width=True)
    
    else:
        st.info("Please run tracking error optimization to view results. Make sure to select 'tracking_error' as the optimization objective in the sidebar.")

# Risk Analysis Tab
with tab5:
    st.header("Risk Analysis")
    
    if st.session_state.factor_data is not None:
        # Factor risk decomposition
        st.subheader("Factor Risk Decomposition")
        
        # Calculate factor volatilities
        factor_vols = {}
        for factor, data in st.session_state.factor_data.items():
            wide_data = data.pivot(index='date', columns='sid', values='value')
            factor_vols[factor] = wide_data.std()
        
        # Plot factor volatilities
        fig = px.bar(
            x=list(factor_vols.keys()),
            y=[vol.mean() for vol in factor_vols.values()],
            title="Average Factor Volatility"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor exposure analysis
        st.subheader("Factor Exposure Analysis")
        selected_factor = st.selectbox(
            "Select Factor for Exposure Analysis",
            list(st.session_state.factor_data.keys()),
            key="risk_factor_select"
        )
        
        if selected_factor:
            factor_data = st.session_state.factor_data[selected_factor]
            wide_data = factor_data.pivot(index='date', columns='sid', values='value')
            
            # Plot factor exposure over time
            fig = px.line(
                wide_data.mean(axis=1),
                title=f"Average Factor Exposure - {selected_factor}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution of factor exposures
            st.subheader("Factor Exposure Distribution")
            
            # Latest date exposure distribution
            latest_date = wide_data.index.max()
            latest_exposures = wide_data.loc[latest_date].dropna()
            
            fig = px.histogram(
                latest_exposures,
                title=f"Factor Exposure Distribution - {selected_factor} (as of {latest_date})",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Extreme exposure analysis
            st.subheader("Extreme Exposure Analysis")
            
            # Calculate stocks with highest and lowest exposures
            highest_exposure = latest_exposures.nlargest(5)
            lowest_exposure = latest_exposures.nsmallest(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Highest Factor Exposure")
                st.dataframe(pd.DataFrame({
                    'Stock': highest_exposure.index,
                    'Exposure': highest_exposure.values
                }).set_index('Stock'))
            
            with col2:
                st.write("Lowest Factor Exposure")
                st.dataframe(pd.DataFrame({
                    'Stock': lowest_exposure.index,
                    'Exposure': lowest_exposure.values
                }).set_index('Stock'))
            
            # Time-series of extreme exposures
            st.subheader("Extreme Exposures Over Time")
            
            # Get stocks with highest/lowest exposures on latest date
            top_stocks = highest_exposure.index[:3].tolist()
            bottom_stocks = lowest_exposure.index[:3].tolist()
            extreme_stocks = top_stocks + bottom_stocks
            
            # Plot their exposures over time
            extreme_exposures = wide_data[extreme_stocks]
            
            fig = px.line(
                extreme_exposures,
                title=f"Extreme Factor Exposures Over Time - {selected_factor}"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please run factor analysis first to view risk analysis.")

# Documentation Tab
with tab6:
    st.header("Documentation")
    
    st.markdown("""
    ## Equity Factor Analysis Platform

    This platform provides tools for equity factor analysis, portfolio optimization, and risk management.
    
    ### Data Management
    
    The platform allows you to:
    - Update data from various sources (Yahoo Finance, Bloomberg, etc.)
    - Sync data with cloud storage
    - Manage historical data for different universes

    ### Factor Analysis
    
    The Factor Analysis tab provides tools for analyzing individual factors:
    - Cumulative returns of factor portfolios
    - Factor distributions and statistics
    - Turnover analysis
    - Factor autocorrelation analysis
    
    ### Portfolio Optimization
    
    The Portfolio Optimization tab offers two main approaches:
    1. Pure Factor Optimization
       - Creates pure factor portfolios
       - Analyzes factor returns and correlations
       - Provides monthly performance statistics
    
    2. Tracking Error Optimization
       - Optimizes portfolios with tracking error constraints
       - Controls number of trades
       - Manages position weights

    ### Tracking Error Optimization
    
    The Tracking Error Optimization tab provides comprehensive tools for:
    - **Portfolio Optimization**: Minimize tracking error while respecting factor constraints
    - **Performance Analysis**: Compare optimized portfolio vs benchmark performance
    - **Risk Management**: Monitor factor exposures and portfolio concentration
    - **Turnover Analysis**: Track portfolio turnover and rebalancing costs
    
    Key features:
    - Configurable factor constraints (range or tolerance-based)
    - Position limits and concentration controls
    - Integer constraints for realistic position counting
    - Real-time optimization metrics and performance tracking
    
    ### Pure Factor Portfolios
    
    This tab provides detailed analysis of pure factor portfolios:
    - Performance metrics
    - Holdings analysis
    - Turnover statistics
    - Factor exposure analysis
    
    ### Risk Analysis
    
    The Risk Analysis tab provides:
    - Factor risk decomposition
    - Factor exposure analysis
    - Volatility analysis
    - Correlation analysis
    
    ### Configuration
    
    Use the sidebar to configure:
    - Data source
    - Universe selection
    - Date range
    - Rebalancing frequency
    - Risk factors
    - Optimization parameters
    
    ### Modules and Dependencies
    
    This platform uses:
    - `qFactor.py`: Factor data and analysis
    - `qBacktest.py`: Backtesting framework
    - `qOptimization.py`: Portfolio optimization
    - `factor_file_etl.py`: Data processing and ETL
    
    ### Typical Workflow
    
    1. Configure settings in the sidebar
    2. Update data
    3. Run factor analysis
    4. Run portfolio optimization
    5. Analyze results in the tabs
    """)
    
    # Additional help sections
    with st.expander("Common Issues & Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        #### Data Loading Issues
        - If data isn't loading, check if dates are within valid range
        - Ensure you've selected supported factors for the universe
        - Try updating the full history if incremental updates fail
        
        #### Optimization Issues
        - If optimization fails, try relaxing constraints
        - Ensure the universe has enough liquid securities
        - Check factor exposures for extreme values
        
        #### Performance Issues
        - Large universes may cause slower performance
        - Consider using smaller date ranges for initial analysis
        - Disable cloud sync for faster local operations
        """)
    
    with st.expander("API Reference"):
        st.markdown("""
        ### Key Functions
        
        #### Data Management
        - `run_data_update`: Updates data from sources
        - `sync_data_with_s3`: Syncs data with cloud storage
        
        #### Factor Analysis
        - `run_factor_analysis`: Analyzes factor returns and exposures
        - `EquityFactor.analyze_factor_returns`: Calculates factor performance
        
        #### Optimization
        - `run_portfolio_optimization`: Runs optimization based on selected objective
        - `run_tracking_error_optimization`: Runs tracking error optimization with comprehensive analysis
        - `PureFactorOptimizer`: Creates pure factor portfolios
        - `TrackingErrorOptimizer`: Optimizes with tracking error constraints
        
        ### Key Classes
        
        - `EquityFactorModelInput`: Core configuration class
        - `Backtest`: Backtesting framework
        - `EquityFactor`: Factor analysis tools
        - `SecurityMaster`: Data management for securities
        """)