# qOptimization.py
"""
# Optimization class
"""

from typing import List, Dict, Optional, Union, Tuple, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError
from datetime import datetime, date
import pandas as pd
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from enum import Enum
import logging
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# from src.qRegime import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(str, Enum):
    """Types of optimization objectives"""
    MEAN_VAR = "mean_variance"
    MAXIMUM_SHARPE = "max_sharpe"
    MINIMUM_VOLATILITY = "min_vol"
    RISK_PARITY = "risk_parity"
    MAXIMUM_DIVERSIFICATION = "max_div"
    PURE_FACTOR = "pure_factor"
    TRACKING_ERROR = "tracking_error"

class OptimizationStatus(str, Enum):
    """Status of optimization run"""
    SUCCESS = "success"
    FAILED = "failed"
    INFEASIBLE = "infeasible"
    UNDEFINED = "undefined"

class OptimizationResult(BaseModel):
    """Results from a single optimization run"""
    date: str
    status: OptimizationStatus
    weights: Optional[pd.Series] = None
    target_factor: Optional[str] = None
    objective_value: Optional[float] = None
    factor_exposures: Optional[Dict[str, float]] = None
    optimization_time: float = 0.0
    error_message: Optional[str] = None
    factor_exposures_active: Optional[Dict[str, dict]] = None
    tracking_error: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
        
class PurePortfolioConstraints(BaseModel):
    """Portfolio constraints configuration"""
    long_only: bool = False
    full_investment: bool = True
    factor_neutral: List[str] = Field(default_factory=list)
    factor_bounds: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    weight_bounds: Tuple[float, float] = (-0.05, 0.05)
    min_holding: float = 0.001
    max_names: Optional[int] = None

    @field_validator('weight_bounds')
    def validate_weight_bounds(cls, v):
        if v[0] > v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v

class PureFactorOptimizer(BaseModel):
    """Pure factor portfolio optimization implementation"""
    target_factor: str
    constraints: PurePortfolioConstraints
    normalize_weights: bool = True
    parallel_processing: bool = False # True
    max_workers: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _setup_optimization_problem(
        self,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        constraints: PurePortfolioConstraints
    ) -> Tuple[cp.Problem, cp.Variable]:
        """Set up the CVXPY optimization problem"""
        n_assets = returns.shape[1]
        w = cp.Variable(n_assets)

        # Objective: Maximize exposure to target factor
        objective = cp.Minimize(-(w @ exposures[self.target_factor]))

        # Initialize constraints list
        constraint_list = []

        # Basic constraints
        if constraints.full_investment:
            constraint_list.append(cp.sum(w) == 0)

        if constraints.long_only:
            constraint_list.append(w >= 0)
        else:
            constraint_list.append(w >= constraints.weight_bounds[0])
            constraint_list.append(w <= constraints.weight_bounds[1])

        # Factor neutrality constraints
        for factor in constraints.factor_neutral:
            if factor != self.target_factor:
                constraint_list.append(w @ exposures[factor] == 0)

        # Factor bound constraints
        for factor, (lower, upper) in constraints.factor_bounds.items():
            if factor != self.target_factor:
                factor_exposure = w @ exposures[factor]
                constraint_list.extend([
                    factor_exposure >= lower,
                    factor_exposure <= upper
                ])

        # Maximum number of positions constraint
        if constraints.max_names is not None:
            constraint_list.append(
                cp.sum(cp.abs(w) >= constraints.min_holding) <= constraints.max_names
            )

        return cp.Problem(objective, constraint_list), w

    def _normalize_portfolio(self, weights: np.ndarray) -> np.ndarray:
        """Normalize portfolio weights to maintain dollar neutrality"""
        if not self.normalize_weights:
            return weights

        positive_sum = np.sum(weights[weights > 0])
        negative_sum = np.abs(np.sum(weights[weights < 0]))
        scaling_factor = max(positive_sum, negative_sum)

        if scaling_factor > 0:
            return weights / scaling_factor
        return weights

    def _optimize_single_period(
        self,
        date: str,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        universe: List[str]
    ) -> OptimizationResult:
        """Optimize portfolio for a single period"""
        try:
            start_time = pd.Timestamp.now()

            # Set up and solve optimization problem
            prob, w = self._setup_optimization_problem(returns, exposures, self.constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return OptimizationResult(
                    date=date,
                    status=OptimizationStatus.INFEASIBLE,
                    error_message=f"Optimization status: {prob.status}"
                )

            # Process results
            weights = self._normalize_portfolio(w.value)
            weights_series = pd.Series(weights, index=universe)

            # Calculate factor exposures
            # factor_exposures = {
            #     factor: (np.array(weights_series) @ exposures[factor])
            #     for factor in exposures.columns
            #     if factor != 'sid'
            # }
            factor_exposures = np.array(weights_series) @ exposures[[factor for factor in exposures.columns if factor not in ['sid','date']]]
            factor_exposures = factor_exposures.to_dict()

            return OptimizationResult(
                date=date,
                status=OptimizationStatus.SUCCESS,
                target_factor=self.target_factor,
                weights=weights_series,
                objective_value=prob.value,
                factor_exposures=factor_exposures,
                optimization_time=(pd.Timestamp.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Optimization failed for date {date}: {str(e)}")
            return OptimizationResult(
                date=date,
                status=OptimizationStatus.FAILED,
                error_message=str(e)
            )

    def optimize(
        self,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        dates: List[str],
        lookback_periods: int = 3
    ) -> pd.DataFrame:
        """
        Run optimization across multiple periods

        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns matrix
        exposures : pd.DataFrame
            Factor exposures dataframe
        dates : List[str]
            List of dates to optimize for
        lookback_periods : int
            Number of periods to use for return calculation

        Returns:
        --------
        pd.DataFrame
            Optimized portfolio weights and metrics
        """
        results = []
        exposures['date'] = pd.to_datetime(exposures['date'])

        def process_date(date):
            # Get universe and exposures for current date
            # current_exposures = exposures[exposures['date'] == date.replace('-','')].copy()
            current_exposures = exposures[exposures['date'] == str(date)].copy()
            if current_exposures.shape[0]==0:
                current_exposures = exposures[exposures['date'] == pd.to_datetime(date).date()].copy()
            universe = current_exposures['sid'].tolist()

            # Get historical returns
            date_idx = dates.index(date)
            start_idx = max(0, date_idx - lookback_periods)
            # historical_returns = returns.loc[dates[start_idx]:date][universe].fillna(0)
            historical_returns = returns.loc[(returns.index>=pd.to_datetime(dates[start_idx]).date()) & (returns.index<=pd.to_datetime(date).date())][universe].fillna(0)
            # Run optimization
            result = self._optimize_single_period(
                str(date),
                historical_returns,
                current_exposures,
                universe
            )
            return result

        if self.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                optimization_results = list(executor.map(process_date, dates))
        else:
            optimization_results = [process_date(date) for date in dates]

        # Convert results to DataFrame
        results_data = pd.DataFrame()
        weights_data = pd.DataFrame()
        for result in optimization_results:
            if result.status == OptimizationStatus.SUCCESS:
                row_data = {
                    'date': result.date, # .replace('-',''),
                    'status': result.status,
                    'target_factor':result.target_factor,
                    'objective_value': result.objective_value,
                    'optimization_time': result.optimization_time
                }

                # Add factor exposures
                for factor, exposure in result.factor_exposures.items():
                    row_data[f'exposure_{factor}'] = exposure

                # results_data.append(row_data)
                results_data = pd.concat([results_data, 
                                          pd.DataFrame(row_data, index=[result.date])])

                # Add weights
                df = result.weights.reset_index(drop=False)
                df.columns = ['sid','weight']
                df.insert(0, 'factor', result.target_factor)
                df.insert(0, 'date', result.date) # .replace('-','')
                df['date'] = pd.to_datetime(df['date'])
                
                df = df.merge(exposures, how='left', on=['date','sid'])
                df.index = df['date']
                df.index.name = 'index'
                weights_data = pd.concat([weights_data, 
                                          df])
                # for sid, weight in result.weights.items():
                #     row_data[f'{sid}'] = weight
                #     weights_data.append([sid, weight])
                    
                    
        return {'meta_data':results_data, # pd.DataFrame(results_data), 
                'weights_data':weights_data}
    
class TrackingErrorConstraints(BaseModel):
    """Tracking error specific portfolio constraints"""
    long_only: bool = True
    full_investment: bool = True
    factor_constraints: Dict[str, Union[float, Tuple[float, float]]] = Field(
        default_factory=dict,
        description="Factor exposure constraints relative to benchmark"
    )
    weight_bounds: Tuple[float, float] = (0.0, 0.1)
    min_holding: float = 0.001
    max_names: int = Field(gt=0, description="Maximum number of positions")
    tracking_error_max: float = Field(gt=0, le=1, description="Maximum tracking error")
    
    @field_validator('weight_bounds')
    def validate_weight_bounds(cls, v):
        if v[0] > v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        return v
    
    @field_validator('factor_constraints')
    def validate_factor_constraints(cls, v):
        for factor, constraint in v.items():
            if isinstance(constraint, tuple):
                if len(constraint) != 2 or constraint[0] > constraint[1]:
                    raise ValueError(f"Invalid constraint range for factor {factor}")
        return v

class TrackingErrorOptimizer(BaseModel):
    """Tracking error optimization implementation"""
    constraints: TrackingErrorConstraints
    normalize_weights: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    use_integer_constraints: bool = True
    
    class Config:
        arbitrary_types_allowed = True

    def _setup_optimization_problem(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        exposures: pd.DataFrame,
        benchmark_exposures: pd.DataFrame,
        constraints: TrackingErrorConstraints) -> Tuple[cp.Problem, cp.Variable, cp.Variable]:
        """Set up the CVXPY optimization problem"""
        n_assets = returns.shape[1]
        w = cp.Variable(n_assets)
        
        # Boolean variables for position counting if using integer constraints
        if self.use_integer_constraints:
            wi = cp.Variable(n_assets, boolean=True)
        
        # Calculate returns matrices
        mu = np.array(returns)
        bench_returns = np.array(benchmark_returns['return'])
        
        # Define tracking error
        tracking_error = cp.sum(cp.abs(mu @ w - bench_returns))
        obj = cp.Minimize(tracking_error)
        
        # Initialize constraints list
        constraint_list = []
        
        # Basic constraints
        if constraints.full_investment:
            constraint_list.append(cp.sum(w) == 1)
            
        if constraints.long_only:
            constraint_list.append(w >= 0)
        else:
            constraint_list.append(w >= constraints.weight_bounds[0])
        
        constraint_list.append(w <= constraints.weight_bounds[1])
        
        # Factor constraints
        for factor, constraint in constraints.factor_constraints.items():
            bench_exposure = benchmark_exposures['weight'] @ benchmark_exposures[factor]
            
            if isinstance(constraint, tuple):
                lower, upper = constraint
                constraint_list.extend([
                    (w @ exposures[factor]) >= (bench_exposure + lower),
                    (w @ exposures[factor]) <= (bench_exposure + upper)
                ])
            else:
                constraint_list.extend([
                    (w @ exposures[factor]) >= (bench_exposure - constraint),
                    (w @ exposures[factor]) <= (bench_exposure + constraint)
                ])
        
        # Position limits using integer constraints
        if self.use_integer_constraints:
            constraint_list.extend([
                w <= wi,
                cp.sum(wi) >= constraints.max_names
            ])
            
        return cp.Problem(obj, constraint_list), w, wi if self.use_integer_constraints else None

    def _normalize_portfolio(
        self,
        weights: np.ndarray,
        ranks: Optional[np.ndarray] = None,
        max_positions: Optional[int] = None) -> np.ndarray:
        """
        Normalize portfolio weights and apply position limits
        """
        if not self.normalize_weights:
            return weights
            
        # Apply position limits if specified
        if max_positions and ranks is not None:
            weights = weights.copy()
            weights[ranks > max_positions] = 0
            
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        return weights

    def _optimize_single_period(
        self, 
        date: str,
        returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        exposures: pd.DataFrame,
        benchmark_exposures: pd.DataFrame,
        universe: List[str]) -> OptimizationResult:
        """Optimize portfolio for a single period"""
        try:
            start_time = pd.Timestamp.now()
            
            # Set up and solve optimization problem
            prob, w, wi = self._setup_optimization_problem(
                returns, 
                benchmark_returns,
                exposures,
                benchmark_exposures,
                self.constraints
            )
            
            # Solve with appropriate solver: print(cp.installed_solvers())
            if self.use_integer_constraints:
                prob.solve(solver=cp.SCIPY, verbose=True)
            else:
                prob.solve(solver=cp.CLARABEL, verbose=True)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return OptimizationResult(
                    date=str(date),
                    status=OptimizationStatus.INFEASIBLE,
                    error_message=f"Optimization status: {prob.status}"
                )
            
            # Process results
            weights = w.value
            if wi is not None:
                binary_weights = wi.value
                
                # Calculate ranks for position limiting
                ranks = (-weights).argsort().argsort()
                
                # Apply normalization and position limits
                weights = self._normalize_portfolio(
                    weights,
                    ranks=ranks,
                    max_positions=self.constraints.max_names
                )
            
            weights_series = pd.Series(weights, index=universe)
            
            # Calculate factor exposures
            factor_exposures = {}
            for factor in exposures.columns:
                if factor not in ['sid', 'date']:
                    opt_exposure = np.array(weights_series) @ exposures[factor]
                    bench_exposure = benchmark_exposures['weight'] @ benchmark_exposures[factor]
                    factor_exposures[factor] = {
                        'portfolio': float(opt_exposure),
                        'benchmark': float(bench_exposure),
                        'active': float(opt_exposure - bench_exposure)
                    }
            
            # Calculate realized tracking error
            realized_te = np.sqrt(
                np.sum((np.array(returns) @ weights - benchmark_returns['return'])**2)
            )
            
            return OptimizationResult(
                date=str(date),
                status=OptimizationStatus.SUCCESS,
                weights=weights_series,
                objective_value=prob.value,
                # factor_exposures=pd.DataFrame(factor_exposures).T['portfolio'].to_dict(),
                factor_exposures_active=factor_exposures.copy(),
                optimization_time=(pd.Timestamp.now() - start_time).total_seconds(),
                tracking_error=realized_te
            )
            
        except Exception as e:
            logger.error(f"Optimization failed for date {date}: {str(e)}")
            return OptimizationResult(
                date=str(date),
                status=OptimizationStatus.FAILED,
                error_message=str(e)
            )

    def optimize(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        exposures: pd.DataFrame,
        benchmark_exposures: pd.DataFrame,
        dates: List[str],
        lookback_periods: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """
        Run tracking error optimization across multiple periods
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns matrix
        benchmark_returns : pd.DataFrame
            Benchmark returns
        exposures : pd.DataFrame
            Factor exposures dataframe
        benchmark_exposures : pd.DataFrame
            Benchmark factor exposures
        dates : List[str]
            List of dates to optimize for
        lookback_periods : int
            Number of periods to use for return calculation
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing results and weight matrices
        """
        results_data = pd.DataFrame()
        weights_data = pd.DataFrame()
        
        benchmark_exposures['date'] = pd.to_datetime(benchmark_exposures['date'])
        # benchmark_returns['date'] = benchmark_returns['date'].map(lambda x: str(x.date()))
        benchmark_returns['date'] = pd.to_datetime(benchmark_returns['date'])
        # import pdb; pdb.set_trace()
        for date in dates:
            # print(f"date is {date}")
            # Get universe and exposures for current date
            current_exposures = exposures[
                exposures['date'] == str(date)#.replace('-','')
            ].copy()
            universe = current_exposures['sid'].tolist()
            
            # Get historical returns
            date_idx = dates.index(date)
            start_idx = max(0, date_idx - lookback_periods)
            start_date = dates[start_idx]
            
            # Get returns for optimization period
            # historical_returns = returns.loc[start_date:date][universe].fillna(0)
            historical_returns = returns.loc[pd.Timestamp(start_date).date():pd.Timestamp(date).date()][universe].fillna(0)
            benchmark_rets = benchmark_returns[
                (benchmark_returns['date'] >= str(start_date)) & 
                (benchmark_returns['date'] <= str(date))
            ].copy()
            
            # Get benchmark exposures
            current_bench_exposures = benchmark_exposures[
                benchmark_exposures['date'] == str(date)#.replace('-','')
            ].copy()
            
            # Run optimization
            # Merge benchmark weights with factor exposures
            current_bench_exposures['date'] = pd.to_datetime(current_bench_exposures['date'])
            current_bench_exposures = current_bench_exposures.merge(current_exposures, how='left', on=['date','sid'])
            current_bench_exposures.fillna(0., inplace=True)
            
            result = self._optimize_single_period(
                str(date),
                historical_returns,
                benchmark_rets,
                current_exposures,
                current_bench_exposures,
                universe
            )

            if result.status == OptimizationStatus.SUCCESS:
                # Store optimization metadata
                meta_row = {
                    'date': str(date), #.replace('-',''),
                    'status': result.status,
                    'objective_value': result.objective_value,
                    'optimization_time': result.optimization_time,
                    'tracking_error': result.tracking_error
                }
                
                # Add factor exposures
                for factor, exposures_active in result.factor_exposures_active.items():
                    for exposure_type, value in exposures_active.items():
                        meta_row[f'{factor}_{exposure_type}'] = value
                        
                results_data = pd.concat([
                    results_data, 
                    pd.DataFrame([meta_row])
                ])
                
                # Store weights
                weights_df = result.weights.reset_index()
                weights_df.columns = ['sid', 'weight']
                weights_df.insert(0, 'date', str(date)) #.replace('-',''))
                weights_df['date'] = pd.to_datetime(weights_df['date'])
                weights_df = weights_df.merge(
                    current_exposures, 
                    how='left',
                    on=['date','sid']
                )

                # Include benchmark weights
                weights_df['date'] = pd.to_datetime(weights_df['date'])
                weights_df = weights_df.merge(benchmark_exposures[['date','sid','wgt']], how='left', on=['date','sid'])
                weights_df.rename(columns={'wgt':'weight_benchmark'}, inplace=True)
                
                weights_data = pd.concat([weights_data, weights_df])

        weights_data[['weight','weight_benchmark']] = weights_data.groupby('sid')[['weight','weight_benchmark']].ffill()
        if weights_data['weight_benchmark'].dtype=='O':
            weights_data['weight_benchmark'] = weights_data['weight_benchmark'].str.rstrip('%').astype(float)/100.

        return {
            'meta_data': results_data,
            'weights_data': weights_data
        }
    
if __name__=="__main__":
    from decimal import Decimal

    # Import from qFactor.py
    from qFactor import (
        EquityFactorModelInput, ParamsConfig, BacktestConfig, RegimeConfig, OptimizationConfig, ExportConfig,
        RiskFactors, Universe, Currency, Frequency, EquityFactor, 
        SecurityMasterFactory, FactorFactory, get_rebalance_dates, generate_config_id,
        set_model_input_start, set_model_input_dates_turnover, set_model_input_dates_daily
        )
    import qBacktest as bt

    from file_data_manager import FileConfig, FileDataManager

    model_input = EquityFactorModelInput(
        params=ParamsConfig(
            aum=Decimal('100'),
            sigma_regimes=False,
            risk_factors=[
                RiskFactors.SIZE, RiskFactors.MOMENTUM, RiskFactors.VALUE, RiskFactors.BETA
                ],
            bench_weights=None,
            n_buckets=4
        ),
        backtest=BacktestConfig(
            data_source='yahoo',
            universe=Universe.INDU,
            currency=Currency.USD,
            frq=Frequency.MONTHLY,
            start='2022-12-31',
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
            base_path="./data/time_series",
            s3_config=None
        )
    )
    set_model_input_dates_turnover(model_input)
    set_model_input_dates_daily(model_input)

    cfg = FileConfig()
    mgr = FileDataManager(cfg)
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"

    """
    # Get security master object
    """
    
    df_benchmark_prices = mgr.load_prices(identifier)
    df_benchmark_weights = mgr.load_benchmark_weights(identifier)
    df_prices = mgr.load_prices(identifier+'_members')
    df_returns = mgr.load_returns(identifier+'_members')

    security_master = SecurityMasterFactory(
        model_input=model_input
        )

    security_master.df_price = df_prices
    security_master.df_bench = df_benchmark_prices
    security_master.weights_data = df_benchmark_weights

    # get returns wide format
    df_ret_wide = security_master.get_returns_wide()
    
    # get returns long format 
    df_ret_long = security_master.get_returns_long()

    # get exposures long format
    df_exposures_long = mgr.load_exposures(identifier+'_members')
    df_exposures_long['exposure'] = df_exposures_long['exposure'].fillna(0.)
    df_exposures = df_exposures_long.pivot(
        index=['date','sid'], 
        columns='variable', 
        values='exposure').reset_index(drop=False)
    df_exposures['date'] = pd.to_datetime(df_exposures['date'])

    # tracking error optimization
    """
    # Run TE optimization
    """
    # Create tracking error optimization constraints
    constraints = TrackingErrorConstraints(
        long_only=True,
        full_investment=True,
        factor_constraints={
            'beta': (0.0, 0.1),
            'momentum': (0.0, 0.05),
            'size': 0.03,
            'value': 0.01,
        },
        weight_bounds=(0.0, 0.1),
        min_holding=0.01,
        max_names=20,
        tracking_error_max=0.05
    )

    # Initialize optimizer
    optimizer_te = TrackingErrorOptimizer(
        constraints=constraints,
        normalize_weights=True,
        parallel_processing=False,
        use_integer_constraints=True
    )
        
    print(optimizer_te)

    # returns=df_ret_wide.copy()
    benchmark_returns=security_master.df_bench.copy()
    exposures=df_exposures.copy()
    benchmark_exposures=security_master.weights_data.copy()

    results_te = optimizer_te.optimize(
        returns=df_ret_wide, # security_master.get_returns_wide(), # returns
        benchmark_returns=benchmark_returns,
        exposures=exposures,
        benchmark_exposures=benchmark_exposures,
        dates=model_input.backtest.dates_turnover #dates_to
    )

    """
    # Backtest strategy: out-of-sample performance of tracking portfolio
    """
    df_weights = results_te['weights_data'].copy()
    df_weights = df_weights.sort_values(['sid','date'])
    # df_weights.rename(columns={'weight':'weight','weight_benchmark':'weight_benchmark'}, inplace=True)

    df_weights['n_opt'] = df_weights['weight']!=0
    print(f"Number of securities in optimal portfolio {df_weights[['date','n_opt']].groupby('date').sum().mean().round(2).squeeze()}")

    print("Optimization Meta Data:")
    print(results_te['meta_data'].tail(3))

    print("Optimal and Benchmark Weights:")
    print(results_te['weights_data'].tail(3))

    # TE portfolio: backtest returns
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
    results_bt = backtest.run_backtest(df_returns, df_portfolio, plot=True)
    df_ret_opt = backtest.df_pnl.copy()


    # factors and pure portfolio optimization
    factor_list = [i.value for i in model_input.params.risk_factors] # ['momentum','beta','size','value']
    print(f"\nFactors in Model: {factor_list}")

    df_pure_return = pd.DataFrame()
    df_pure_portfolio = pd.DataFrame()

    for factor in factor_list: # ['beta']
        
        # Create optimization constraints
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

        # Run optimization (example data not provided)
        # results = optimizer.optimize(returns, exposures, dates)
        results_opt = optimizer_pure.optimize(
            returns = df_ret_wide, 
            exposures = df_exposures, 
            dates = model_input.backtest.dates_turnover # [str(i) for i in dates_to]
        )
        df_portfolio = results_opt.get('weights_data')
        
        bt_config = bt.BacktestConfig(
            asset_class=bt.AssetClass.EQUITY,
            portfolio_type=bt.PortfolioType.LONG_SHORT,
            model_type=factor,
            annualization_factor=252
        )

        backtest = bt.Backtest(config=bt_config)
        df_portfolio.rename(columns={'sid':'ticker', 'weight':'weight'}, inplace=True)
        df_returns = df_ret_long.copy()
        df_returns.rename(columns={'sid':'ticker'}, inplace=True)
        results_bt = backtest.run_backtest(df_returns, df_portfolio, plot=False)
        df_ret_opt = backtest.df_pnl.copy()

        print(f"{factor.upper()} Factor Return & Sharpe : {results_bt.cumulative_return_benchmark:.2%}, {results_bt.sharpe_ratio_benchmark:.2f}")
        # df_ret_opt = get_backtest(df_ret_long, df_portfolio, lag=1, flag_plot=False)
        # df_ret_opt.insert(0, 'factor', factor.lower())
            
        df_pure_return = pd.concat([df_pure_return, df_ret_opt])
        df_pure_portfolio = pd.concat([df_pure_portfolio, df_portfolio])
        
    # factor returns
    df_pure_return_wide = df_pure_return[['factor','return_opt']].pivot(columns='factor',values='return_opt')
    df_pure_return_wide.cumsum().plot(title=f"Pure Factor Returns: {security_master.universe}", rot=45, figsize=(16,8))
    # df_pure_return_wide.reset_index(drop=False, inplace=True)
    # print(df_pure_return_wide.info())
    print("Pure factors correlation matrix:")
    print(np.cov(df_pure_return_wide.T))