# custom_portfolios.py
# User Portfolio Upload and Analysis Framework

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Literal
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go

class PortfolioValidationResult(BaseModel):
    """Results from portfolio validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class UserPortfolioConfig(BaseModel):
    """Configuration for user portfolio analysis"""
    portfolio_name: str = Field(description="Name/identifier for the portfolio")
    portfolio_type: Literal["long_only", "long_short", "market_neutral"] = Field(
        default="long_only", 
        description="Type of portfolio strategy"
    )
    benchmark_comparison: bool = Field(
        default=True, 
        description="Whether to compare against benchmark"
    )
    risk_analysis: bool = Field(
        default=True, 
        description="Whether to perform risk decomposition"
    )
    return_attribution: bool = Field(
        default=True, 
        description="Whether to perform return attribution"
    )
    rebalancing_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = Field(
        default="monthly",
        description="Assumed rebalancing frequency for analysis"
    )
    
class UserPortfolio(BaseModel):
    """Main class for handling user portfolio data and analysis"""
    config: UserPortfolioConfig
    weights_data: pd.DataFrame = Field(default_factory=pd.DataFrame)
    validation_result: Optional[PortfolioValidationResult] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def validate_portfolio_data(self, df: pd.DataFrame) -> PortfolioValidationResult:
        """
        Validate uploaded portfolio data format and content.
        
        Args:
            df: DataFrame with portfolio weights data
            
        Returns:
            PortfolioValidationResult with validation details
        """
        errors = []
        warnings = []
        summary = {}
        
        # Required columns check
        required_cols = {'date', 'sid', 'weight'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if not errors:  # Only proceed if basic structure is correct
            # Date format validation
            try:
                df['date'] = pd.to_datetime(df['date'])
                summary['date_range'] = f"{df['date'].min().date()} to {df['date'].max().date()}"
                summary['num_dates'] = df['date'].nunique()
            except Exception as e:
                errors.append(f"Date format error: {str(e)}")
            
            # Weight validation
            try:
                df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
                if df['weight'].isna().any():
                    warnings.append("Some weights could not be converted to numeric")
                
                # Check weight ranges
                min_weight = df['weight'].min()
                max_weight = df['weight'].max()
                summary['weight_range'] = f"{min_weight:.4f} to {max_weight:.4f}"
                
                if self.config.portfolio_type == "long_only" and min_weight < 0:
                    warnings.append("Negative weights found in long-only portfolio")
                
                # Check weight sums by date
                weight_sums = df.groupby('date')['weight'].sum()
                summary['avg_weight_sum'] = weight_sums.mean()
                
                if self.config.portfolio_type in ["long_only", "market_neutral"]:
                    if not np.allclose(weight_sums, 1.0, atol=0.01):
                        warnings.append("Portfolio weights don't sum to 1.0 for all dates")
                        
            except Exception as e:
                errors.append(f"Weight validation error: {str(e)}")
            
            # Portfolio composition analysis
            summary['num_securities'] = df['sid'].nunique()
            summary['avg_positions_per_date'] = df.groupby('date').size().mean()
            
            # Check for missing securities across dates
            date_securities = df.groupby('date')['sid'].apply(set)
            all_securities = set(df['sid'].unique())
            
            missing_securities_by_date = []
            for dt, securities in date_securities.items():
                missing = all_securities - securities
                if missing:
                    missing_securities_by_date.append((dt, len(missing)))
            
            if missing_securities_by_date:
                warnings.append(f"Some securities missing on certain dates. Max missing: {max(x[1] for x in missing_securities_by_date)}")
        
        is_valid = len(errors) == 0
        
        return PortfolioValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def load_portfolio_data(self, df: pd.DataFrame) -> bool:
        """
        Load and validate portfolio data.
        
        Args:
            df: DataFrame with portfolio weights
            
        Returns:
            bool: True if successfully loaded
        """
        self.validation_result = self.validate_portfolio_data(df)
        
        if self.validation_result.is_valid:
            # Clean and standardize the data
            df_clean = df.copy()
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean['weight'] = pd.to_numeric(df_clean['weight'], errors='coerce')
            df_clean = df_clean.dropna()
            df_clean = df_clean.sort_values(['date', 'sid']).reset_index(drop=True)
            
            self.weights_data = df_clean
            return True
        
        return False

class PortfolioComparator(BaseModel):
    """Class for comparing user portfolio against benchmark and optimal portfolios"""
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def compare_portfolios(
        self, 
        user_portfolio: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        optimal_weights: Optional[pd.DataFrame] = None,
        returns_data: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare user portfolio against benchmark and optimal portfolios.
        
        Args:
            user_portfolio: User portfolio weights
            benchmark_weights: Benchmark weights  
            optimal_weights: Optimal portfolio weights (optional)
            returns_data: Security returns data
            
        Returns:
            Dict with comparison results
        """
        results = {}
        
        # Align dates and securities across portfolios
        common_dates = set(user_portfolio['date'].unique())
        if benchmark_weights is not None:
            common_dates &= set(benchmark_weights['date'].unique())
        if optimal_weights is not None:
            common_dates &= set(optimal_weights['date'].unique())
        
        common_dates = sorted(list(common_dates))
        
        # Performance comparison
        if returns_data is not None:
            performance_results = self._calculate_performance_comparison(
                user_portfolio, benchmark_weights, optimal_weights, 
                returns_data, common_dates
            )
            results['performance'] = performance_results
        
        # Weight distribution analysis
        weight_analysis = self._analyze_weight_distributions(
            user_portfolio, benchmark_weights, optimal_weights, common_dates
        )
        results['weight_analysis'] = weight_analysis
        
        # Turnover analysis
        turnover_analysis = self._calculate_turnover_comparison(
            user_portfolio, benchmark_weights, optimal_weights
        )
        results['turnover'] = turnover_analysis
        
        return results
    
    def _calculate_performance_comparison(
        self, 
        user_portfolio: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        optimal_weights: Optional[pd.DataFrame],
        returns_data: pd.DataFrame,
        common_dates: List
    ) -> pd.DataFrame:
        """Calculate performance metrics for portfolio comparison"""
        
        # Merge returns with weights for each portfolio
        performance_data = []
        
        for date in common_dates:
            date_str = str(date) if isinstance(date, datetime) else date
            
            # Get returns for this date
            date_returns = returns_data[returns_data['date'] == date_str]
            if date_returns.empty:
                continue
            
            # User portfolio performance
            user_weights = user_portfolio[user_portfolio['date'] == date_str]
            if not user_weights.empty:
                merged_user = user_weights.merge(date_returns, on='sid', how='inner')
                user_return = (merged_user['weight'] * merged_user['return']).sum()
            else:
                user_return = np.nan
            
            # Benchmark performance
            bench_weights = benchmark_weights[benchmark_weights['date'] == date_str]
            if not bench_weights.empty:
                merged_bench = bench_weights.merge(date_returns, on='sid', how='inner')
                bench_return = (merged_bench['wgt'] * merged_bench['return']).sum()
            else:
                bench_return = np.nan
            
            # Optimal portfolio performance (if available)
            opt_return = np.nan
            if optimal_weights is not None:
                opt_weights = optimal_weights[optimal_weights['date'] == date_str]
                if not opt_weights.empty:
                    merged_opt = opt_weights.merge(date_returns, on='sid', how='inner')
                    opt_return = (merged_opt['weight'] * merged_opt['return']).sum()
            
            performance_data.append({
                'date': date_str,
                'user_return': user_return,
                'benchmark_return': bench_return,
                'optimal_return': opt_return,
                'active_return_vs_bench': user_return - bench_return,
                'active_return_vs_opt': user_return - opt_return if not np.isnan(opt_return) else np.nan
            })
        
        return pd.DataFrame(performance_data)
    
    def _analyze_weight_distributions(
        self,
        user_portfolio: pd.DataFrame,
        benchmark_weights: pd.DataFrame, 
        optimal_weights: Optional[pd.DataFrame],
        common_dates: List
    ) -> Dict[str, pd.DataFrame]:
        """Analyze weight distributions across portfolios"""
        
        weight_analysis = {}
        
        # Latest date analysis
        latest_date = max(common_dates)
        
        # Get weights for latest date
        user_latest = user_portfolio[user_portfolio['date'] == latest_date]
        bench_latest = benchmark_weights[benchmark_weights['date'] == latest_date]
        
        # Merge and compare
        comparison_df = user_latest[['sid', 'weight']].rename(columns={'weight': 'user_weight'})
        comparison_df = comparison_df.merge(
            bench_latest[['sid', 'wgt']].rename(columns={'wgt': 'benchmark_weight'}),
            on='sid', how='outer'
        ).fillna(0)
        
        if optimal_weights is not None:
            opt_latest = optimal_weights[optimal_weights['date'] == latest_date]
            comparison_df = comparison_df.merge(
                opt_latest[['sid', 'weight']].rename(columns={'weight': 'optimal_weight'}),
                on='sid', how='outer'
            ).fillna(0)
        
        # Calculate active weights
        comparison_df['active_vs_benchmark'] = comparison_df['user_weight'] - comparison_df['benchmark_weight']
        if 'optimal_weight' in comparison_df.columns:
            comparison_df['active_vs_optimal'] = comparison_df['user_weight'] - comparison_df['optimal_weight']
        
        weight_analysis['latest_comparison'] = comparison_df
        
        # Concentration metrics
        concentration_metrics = {
            'user_herfindahl': (user_latest['weight'] ** 2).sum(),
            'user_top5_weight': user_latest.nlargest(5, 'weight')['weight'].sum(),
            'user_top10_weight': user_latest.nlargest(10, 'weight')['weight'].sum(),
            'benchmark_herfindahl': (bench_latest['wgt'] ** 2).sum(),
            'benchmark_top5_weight': bench_latest.nlargest(5, 'wgt')['wgt'].sum(),
            'benchmark_top10_weight': bench_latest.nlargest(10, 'wgt')['wgt'].sum(),
        }
        
        weight_analysis['concentration'] = pd.DataFrame([concentration_metrics])
        
        return weight_analysis
    
    def _calculate_turnover_comparison(
        self,
        user_portfolio: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        optimal_weights: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate turnover for each portfolio"""
        
        def calculate_turnover(weights_df, weight_col):
            """Helper to calculate turnover for a portfolio"""
            dates = sorted(weights_df['date'].unique())
            turnover_data = []
            
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                
                prev_weights = weights_df[weights_df['date'] == prev_date].set_index('sid')[weight_col]
                curr_weights = weights_df[weights_df['date'] == curr_date].set_index('sid')[weight_col]
                
                # Align securities
                common_sids = prev_weights.index.intersection(curr_weights.index)
                prev_aligned = prev_weights.reindex(common_sids).fillna(0)
                curr_aligned = curr_weights.reindex(common_sids).fillna(0)
                
                turnover = abs(curr_aligned - prev_aligned).sum() / 2
                turnover_data.append({
                    'date': curr_date,
                    'turnover': turnover
                })
            
            return pd.DataFrame(turnover_data)
        
        # Calculate turnover for each portfolio
        user_turnover = calculate_turnover(user_portfolio, 'weight')
        user_turnover['portfolio'] = 'user'
        
        bench_turnover = calculate_turnover(benchmark_weights, 'wgt') 
        bench_turnover['portfolio'] = 'benchmark'
        
        turnover_comparison = pd.concat([user_turnover, bench_turnover])
        
        if optimal_weights is not None:
            opt_turnover = calculate_turnover(optimal_weights, 'weight')
            opt_turnover['portfolio'] = 'optimal'
            turnover_comparison = pd.concat([turnover_comparison, opt_turnover])
        
        return turnover_comparison

class PortfolioAnalyzer(BaseModel):
    """Main analyzer class that coordinates portfolio analysis"""
    
    def analyze_user_portfolio(
        self,
        user_portfolio: UserPortfolio,
        model_input,
        benchmark_data: Dict[str, pd.DataFrame],
        factor_data: Dict[str, pd.DataFrame],
        returns_data: pd.DataFrame,
        optimal_portfolio: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        Comprehensive analysis of user portfolio.
        
        Args:
            user_portfolio: UserPortfolio instance
            model_input: Model configuration
            benchmark_data: Benchmark prices and weights
            factor_data: Factor exposure data
            returns_data: Security returns
            optimal_portfolio: Optimal portfolio weights (optional)
            
        Returns:
            Dict with analysis results
        """
        results = {
            'portfolio_config': user_portfolio.config,
            'validation': user_portfolio.validation_result
        }
        
        # 1. Performance Analysis
        comparator = PortfolioComparator()
        comparison_results = comparator.compare_portfolios(
            user_portfolio.weights_data,
            benchmark_data['weights'],
            optimal_portfolio,
            returns_data
        )
        results['comparison'] = comparison_results
        
        # 2. Backtest Analysis
        backtest_results = self._run_portfolio_backtest(
            user_portfolio.weights_data,
            returns_data,
            user_portfolio.config.portfolio_name
        )
        results['backtest'] = backtest_results
        
        # 3. Risk Analysis (if factor data available)
        if user_portfolio.config.risk_analysis and factor_data:
            risk_results = self._analyze_portfolio_risk(
                user_portfolio.weights_data,
                factor_data,
                returns_data
            )
            results['risk_analysis'] = risk_results
        
        return results
    
    def _run_portfolio_backtest(
        self,
        portfolio_weights: pd.DataFrame,
        returns_data: pd.DataFrame,
        portfolio_name: str
    ) -> Dict[str, any]:
        """Run backtest analysis for the portfolio"""
        
        # Use existing backtest framework
        config = bt.BacktestConfig(
            asset_class=bt.AssetClass.EQUITY,
            portfolio_type=bt.PortfolioType.LONG_ONLY,  # Adjust based on portfolio type
            model_type=portfolio_name,
            annualization_factor=252
        )
        
        backtest = bt.Backtest(config=config)
        
        # Prepare data for backtest
        df_portfolio = portfolio_weights.copy()
        df_portfolio.rename(columns={'sid': 'ticker'}, inplace=True)
        
        df_returns = returns_data.copy()
        df_returns.rename(columns={'sid': 'ticker'}, inplace=True)
        
        # Run backtest
        results_bt = backtest.run_backtest(df_returns, df_portfolio, plot=False)
        
        return {
            'performance_metrics': {
                'sharpe_ratio': results_bt.sharpe_ratio_benchmark,
                'cumulative_return': results_bt.cumulative_return_benchmark,
                'sharpe_ratio_optimal': results_bt.sharpe_ratio_optimal,
                'cumulative_return_optimal': results_bt.cumulative_return_optimal
            },
            'daily_returns': backtest.df_pnl
        }
    
    def _analyze_portfolio_risk(
        self,
        portfolio_weights: pd.DataFrame,
        factor_data: Dict[str, pd.DataFrame],
        returns_data: pd.DataFrame
    ) -> Dict[str, any]:
        """Analyze portfolio risk using factor decomposition"""
        
        # This would integrate with the existing risk analysis framework
        # For now, return a placeholder structure
        return {
            'factor_exposures': {},
            'risk_decomposition': {},
            'attribution': {}
        }

# Streamlit UI Components

def render_portfolio_upload_section():
    """Render the portfolio upload section in Streamlit"""
    st.subheader("Upload Portfolio")
    
    # Portfolio configuration
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_name = st.text_input(
            "Portfolio Name", 
            value="My Portfolio",
            key="portfolio_name"
        )
        
        portfolio_type = st.selectbox(
            "Portfolio Type",
            options=["long_only", "long_short", "market_neutral"],
            key="portfolio_type"
        )
    
    with col2:
        rebalancing_freq = st.selectbox(
            "Rebalancing Frequency",
            options=["daily", "weekly", "monthly", "quarterly"],
            index=2,  # Default to monthly
            key="rebalancing_freq"
        )
        
        analysis_options = st.multiselect(
            "Analysis Options",
            options=["Benchmark Comparison", "Risk Analysis", "Return Attribution"],
            default=["Benchmark Comparison", "Risk Analysis"],
            key="analysis_options"
        )
    
    # File upload methods
    upload_method = st.radio(
        "Upload Method",
        options=["File Upload", "Paste CSV Data"],
        key="upload_method"
    )
    
    uploaded_data = None
    
    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload Portfolio CSV",
            type=['csv'],
            help="CSV should contain columns: date, sid, weight"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"Uploaded file with {len(uploaded_data)} rows")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:  # Paste CSV Data
        csv_text = st.text_area(
            "Paste CSV Data",
            height=200,
            placeholder="date,sid,weight\n2024-01-01,AAPL,0.05\n2024-01-01,MSFT,0.04\n...",
            key="csv_paste"
        )
        
        if csv_text.strip():
            try:
                uploaded_data = pd.read_csv(StringIO(csv_text))
                st.success(f"Parsed data with {len(uploaded_data)} rows")
            except Exception as e:
                st.error(f"Error parsing CSV data: {str(e)}")
    
    # Data preview
    if uploaded_data is not None:
        st.subheader("Data Preview")
        st.dataframe(uploaded_data.head(10))
        
        # Validate and process
        if st.button("Validate and Analyze Portfolio", type="primary"):
            return process_uploaded_portfolio(
                uploaded_data, 
                portfolio_name,
                portfolio_type,
                rebalancing_freq,
                analysis_options
            )
    
    return None

def process_uploaded_portfolio(
    data: pd.DataFrame,
    portfolio_name: str,
    portfolio_type: str,
    rebalancing_freq: str,
    analysis_options: List[str]
) -> UserPortfolio:
    """Process uploaded portfolio data"""
    
    # Create configuration
    config = UserPortfolioConfig(
        portfolio_name=portfolio_name,
        portfolio_type=portfolio_type,
        benchmark_comparison="Benchmark Comparison" in analysis_options,
        risk_analysis="Risk Analysis" in analysis_options,
        return_attribution="Return Attribution" in analysis_options,
        rebalancing_frequency=rebalancing_freq
    )
    
    # Create portfolio instance
    user_portfolio = UserPortfolio(config=config)
    
    # Load and validate data
    if user_portfolio.load_portfolio_data(data):
        st.success("Portfolio data loaded successfully!")
        
        # Display validation results
        validation = user_portfolio.validation_result
        
        if validation.warnings:
            st.warning("Validation Warnings:")
            for warning in validation.warnings:
                st.write(f"- {warning}")
        
        # Display summary
        st.subheader("Portfolio Summary")
        summary_df = pd.DataFrame([validation.summary]).T
        summary_df.columns = ['Value']
        st.dataframe(summary_df)
        
        # Store in session state
        st.session_state.user_portfolio = user_portfolio
        
        return user_portfolio
    
    else:
        st.error("Portfolio validation failed!")
        if user_portfolio.validation_result.errors:
            for error in user_portfolio.validation_result.errors:
                st.error(f"- {error}")
        return None

def render_portfolio_analysis_results():
    """Render portfolio analysis results"""
    
    if 'user_portfolio_analysis' not in st.session_state:
        st.info("Please upload and analyze a portfolio first.")
        return
    
    results = st.session_state.user_portfolio_analysis
    
    # Performance Comparison
    if 'comparison' in results and 'performance' in results['comparison']:
        st.subheader("Performance Comparison")
        
        perf_data = results['comparison']['performance']
        
        # Cumulative returns chart
        fig = px.line(
            perf_data,
            x='date',
            y=['user_return', 'benchmark_return', 'optimal_return'],
            title="Portfolio Performance Comparison"
        )
        fig.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        perf_metrics = perf_data[['user_return', 'benchmark_return', 'optimal_return']].dropna()
        metrics_summary = pd.DataFrame({
            'Annualized Return': perf_metrics.mean() * 252,
            'Volatility': perf_metrics.std() * np.sqrt(252),
            'Sharpe Ratio': (perf_metrics.mean() * 252) / (perf_metrics.std() * np.sqrt(252))
        }).round(4)
        
        st.dataframe(metrics_summary)
    
    # Weight Analysis
    if 'comparison' in results and 'weight_analysis' in results['comparison']:
        st.subheader("Weight Analysis")
        
        weight_data = results['comparison']['weight_analysis']
        
        # Latest holdings comparison
        if 'latest_comparison' in weight_data:
            st.write("**Latest Holdings Comparison**")
            comparison_df = weight_data['latest_comparison']
            
            # Show top active positions
            top_active = comparison_df.nlargest(10, 'active_vs_benchmark')[
                ['sid', 'user_weight', 'benchmark_weight', 'active_vs_benchmark']
            ]
            st.dataframe(top_active.style.format({
                'user_weight': '{:.2%}',
                'benchmark_weight': '{:.2%}',
                'active_vs_benchmark': '{:.2%}'
            }))
    
    # Risk Analysis (if available)
    if results.get('risk_analysis'):
        st.subheader("Risk Analysis")
        st.info("Risk analysis functionality would be displayed here")

def run_user_portfolio_analysis():
    """Run comprehensive analysis on user portfolio"""
    
    if 'user_portfolio' not in st.session_state:
        st.error("No portfolio uploaded. Please upload a portfolio first.")
        return None
    
    user_portfolio = st.session_state.user_portfolio
    
    # Check if we have necessary data
    if not st.session_state.data_updated:
        st.error("Please load data first before running portfolio analysis.")
        return None
    
    try:
        # Get data from session state
        benchmark_data = {
            'weights': st.session_state.df_benchmark_weights,
            'prices': st.session_state.df_benchmark_prices
        }
        
        factor_data = st.session_state.factor_data
        returns_data = st.session_state.df_ret_long
        
        # Get optimal portfolio if available
        optimal_portfolio = None
        if hasattr(st.session_state, 'te_weights_data') and st.session_state.te_weights_data is not None:
            optimal_portfolio = st.session_state.te_weights_data
        elif hasattr(st.session_state, 'df_pure_portfolio') and st.session_state.df_pure_portfolio is not None:
            optimal_portfolio = st.session_state.df_pure_portfolio
        
        # Run analysis
        analyzer = PortfolioAnalyzer()
        analysis_results = analyzer.analyze_user_portfolio(
            user_portfolio,
            st.session_state.model_input,
            benchmark_data,
            factor_data,
            returns_data,
            optimal_portfolio
        )
        
        # Store results
        st.session_state.user_portfolio_analysis = analysis_results
        
        st.success("Portfolio analysis completed!")
        return analysis_results
        
    except Exception as e:
        st.error(f"Error in portfolio analysis: {str(e)}")
        return None