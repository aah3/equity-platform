# app_sqlite.py
import streamlit as st

# Set page config - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Equity Trading Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from decimal import Decimal
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time

# Add your module paths
sys.path.append('./src')

# Import your modules
from src.database_sqlite import DatabaseManager
from src.data_service import DataService

# Initialize database and data service
@st.cache_resource
def get_data_service():
    db_manager = DatabaseManager()
    return DataService(db_manager)

data_service = get_data_service()

# App title and description
st.title("Equity Trading Research Platform")
st.markdown("""
This platform allows traders and researchers to analyze equity factors, 
optimize portfolios, and backtest trading strategies.
""")

# Check if demo data exists, otherwise offer to generate it
def check_demo_data():
    indices = data_service.get_available_indices()
    if not indices:
        st.warning("No data found in the database. Would you like to generate demo data?")
        if st.button("Generate Demo Data"):
            with st.spinner("Generating demo data..."):
                success = data_service.generate_demo_data()
                if success:
                    st.success("Demo data generated successfully! Please refresh the app.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to generate demo data.")
            return False
    return True

has_data = check_demo_data()

if has_data:
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Universe selection
        indices = data_service.get_available_indices()
        index_options = {idx['name']: idx['index_id'] for idx in indices}
        selected_index_name = st.selectbox("Select Universe", list(index_options.keys()))
        selected_index = index_options[selected_index_name]
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date.today())
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Frequency selection
        frequency_options = {"Daily": "daily", "Monthly": "monthly"}
        selected_frequency_name = st.selectbox("Data Frequency", list(frequency_options.keys()))
        selected_frequency = frequency_options[selected_frequency_name]
        
        # Factor selection
        st.subheader("Risk Factors")
        available_factors = data_service.get_available_factors()
        selected_factors = st.multiselect("Select Factors", available_factors, 
                                        default=available_factors[:min(4, len(available_factors))])
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        n_buckets = st.slider("Number of Factor Buckets", 3, 10, 5)
        
        # Run button
        run_analysis = st.button("Run Analysis", type="primary")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Factor Analysis", "Portfolio Optimization", "Backtest Results", "Risk Analysis"])

    # Run analysis when button is clicked
    if run_analysis:
        with st.spinner("Running analysis..."):
            # Get index constituents
            constituents = data_service.get_index_constituents(selected_index)
            tickers = constituents['ticker'].tolist()
            
            # Get factor data
            factor_data = {}
            factor_returns = {}
            
            for factor_id in selected_factors:
                # Get factor data
                factor_data[factor_id] = data_service.get_factor_data(
                    factor_id, tickers, start_date_str, end_date_str)
                
                # Calculate factor returns
                factor_returns[factor_id] = data_service.get_factor_returns(
                    factor_id, n_buckets, start_date_str, end_date_str)
            
            # Get correlation matrix between factors
            factor_corr = data_service.get_factor_correlation_matrix(start_date_str, end_date_str)
            
            # Get index returns
            index_returns = data_service.get_index_returns(selected_index, start_date_str, end_date_str)
            
            # Display results in tabs
            with tab1:
                st.header("Factor Analysis")
                
                # Factor performance metrics
                st.subheader("Factor Performance Metrics")
                
                metrics_tables = []
                for factor_id in selected_factors:
                    metrics = data_service.get_factor_performance_metrics(factor_id, n_buckets)
                    if not metrics.empty:
                        metrics['factor'] = factor_id
                        metrics_tables.append(metrics)
                
                if metrics_tables:
                    all_metrics = pd.concat(metrics_tables)
                    pivoted_metrics = all_metrics.pivot(index='bucket', columns='factor', values='sharpe_ratio')
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("Sharpe Ratios by Factor and Bucket")
                        st.dataframe(pivoted_metrics.round(2), use_container_width=True)
                    
                    with col2:
                        # Heatmap of sharpe ratios
                        fig = px.imshow(pivoted_metrics, 
                                     text_auto=True, 
                                     color_continuous_scale='RdBu_r')
                        fig.update_layout(title="Factor Sharpe Ratio Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Factor returns
                st.subheader("Cumulative Factor Returns")
                
                # For each factor, plot the cumulative returns of the long-short portfolio
                if factor_returns:
                    cumulative_returns = []
                    
                    for factor_id, returns_df in factor_returns.items():
                        # Check if the DataFrame contains the 'bucket' column
                        if 'bucket' in returns_df.columns:
                            # Get long-short returns if available
                            ls_returns = returns_df[returns_df['bucket'] == 'long_short']
                            
                            if not ls_returns.empty:
                                # Calculate cumulative returns
                                ls_returns = ls_returns.sort_values('date')
                                ls_returns['cumulative_return'] = (1 + ls_returns['return']).cumprod() - 1
                                
                                for _, row in ls_returns.iterrows():
                                    cumulative_returns.append({
                                        'date': row['date'],
                                        'factor': factor_id,
                                        'cumulative_return': row['cumulative_return']
                                    })
                        else:
                            # Handle case where bucket column doesn't exist
                            # Simply use all returns for this factor
                            if not returns_df.empty and 'date' in returns_df.columns and 'return' in returns_df.columns:
                                returns_df = returns_df.sort_values('date')
                                returns_df['cumulative_return'] = (1 + returns_df['return']).cumprod() - 1
                                
                                for _, row in returns_df.iterrows():
                                    cumulative_returns.append({
                                        'date': row['date'],
                                        'factor': factor_id,
                                        'cumulative_return': row['cumulative_return']
                                    })
                    
                    if cumulative_returns:
                        cum_returns_df = pd.DataFrame(cumulative_returns)
                        fig = px.line(cum_returns_df, x='date', y='cumulative_return', color='factor',
                                    title=f"Factor Cumulative Returns ({selected_index_name})")
                        fig.update_layout(yaxis_tickformat='.1%')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                st.subheader("Factor Correlation Matrix")
                
                if not factor_corr.empty:
                    fig = px.imshow(factor_corr, 
                                 text_auto=True, 
                                 color_continuous_scale='RdBu_r',
                                 zmin=-1, zmax=1)
                    fig.update_layout(title="Factor Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Factor distributions
                st.subheader("Factor Distributions")
                
                if factor_data:
                    # For each factor, show distribution of values
                    col1, col2 = st.columns(2)
                    
                    for i, (factor_id, df) in enumerate(factor_data.items()):
                        if not df.empty:
                            # Get most recent date with data
                            latest_date = df['date'].max()
                            latest_data = df[df['date'] == latest_date]
                            
                            with col1 if i % 2 == 0 else col2:
                                fig = px.histogram(latest_data, x='value', 
                                                nbins=30,
                                                title=f"{factor_id} Distribution ({latest_date})")
                                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.header("Portfolio Optimization")
                
                # Display index constituents
                st.subheader("Index Constituents")
                
                if not constituents.empty:
                    # Add sector breakdown
                    sector_counts = constituents['sector'].value_counts().reset_index()
                    sector_counts.columns = ['Sector', 'Count']
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Show top constituents by weight
                        top_constituents = constituents.sort_values('weight', ascending=False).head(20)
                        fig = px.bar(top_constituents, x='ticker', y='weight', 
                                   title=f"Top {selected_index_name} Constituents by Weight")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Show sector breakdown
                        fig = px.pie(sector_counts, values='Count', names='Sector',
                                   title=f"{selected_index_name} Sector Breakdown")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Factor-based portfolio construction
                st.subheader("Factor-Based Portfolio Construction")
                
                # Select factor for portfolio construction
                portfolio_factor = st.selectbox("Select Factor for Portfolio", selected_factors)
                
                if portfolio_factor:
                    factor_metrics = data_service.get_factor_performance_metrics(portfolio_factor, n_buckets)
                    
                    if not factor_metrics.empty:
                        st.write("Performance Metrics by Factor Bucket")
                        
                        # Format metrics for display
                        display_metrics = factor_metrics.copy()
                        display_metrics['annualized_return'] = display_metrics['annualized_return'].map('{:.2%}'.format)
                        display_metrics['volatility'] = display_metrics['volatility'].map('{:.2%}'.format)
                        display_metrics['max_drawdown'] = display_metrics['max_drawdown'].map('{:.2%}'.format)
                        display_metrics['win_rate'] = display_metrics['win_rate'].map('{:.2%}'.format)
                        
                        st.dataframe(display_metrics.set_index('bucket'), use_container_width=True)
                        
                        # Get constituents for top-ranked bucket
                        if factor_data.get(portfolio_factor) is not None:
                            factor_df = factor_data[portfolio_factor]
                            
                            if not factor_df.empty:
                                # Get most recent date with data
                                latest_date = factor_df['date'].max()
                                latest_data = factor_df[factor_df['date'] == latest_date]
                                
                                # Rank securities by factor value
                                if portfolio_factor in ['value', 'size']: # Lower is better for these
                                    latest_data = latest_data.sort_values('value')
                                else: # Higher is better for others
                                    latest_data = latest_data.sort_values('value', ascending=False)
                                
                                # Display top securities
                                st.write(f"Top 20 Securities Ranked by {portfolio_factor} (as of {latest_date})")
                                
                                top_securities = latest_data.head(20)
                                
                                # Merge with constituent data to get names/sectors
                                enriched_securities = top_securities.merge(
                                    constituents[['ticker', 'name', 'sector']], on='ticker', how='left')
                                
                                st.dataframe(enriched_securities[['ticker', 'name', 'sector', 'value']], 
                                           use_container_width=True)
            
            with tab3:
                st.header("Backtest Results")
    
                # Create sub-tabs for different backtest types
                backtest_tabs = st.tabs(["Factor Strategies", "Pair Trading"])
    
                with backtest_tabs[0]:  # Factor Strategies tab
                    st.subheader("Factor Strategy Performance")
                    
                    # Get available strategies for the selected index
                    strategies = data_service.get_available_strategies(selected_index)
                    
                    if not strategies:
                        st.warning(f"No backtest strategies found for {selected_index_name}. Please select a different index or generate demo data.")
                    else:
                        # Display strategy selector
                        strategy_options = {s['name']: s['factor'] for s in strategies}
                        selected_strategy_name = st.selectbox(
                            "Select Strategy",
                            list(strategy_options.keys()),
                            key="strategy_selector"
                        )
                        selected_strategy = strategy_options[selected_strategy_name]
                        
                        # Get backtest results
                        backtest_df = data_service.get_backtest_results(selected_index, selected_strategy)
                        
                        if not backtest_df.empty:
                            # Performance chart
                            st.subheader("Strategy Performance")
                            
                            # Plot cumulative returns
                            fig = px.line(
                                backtest_df, 
                                x='date', 
                                y=['strategy_cumulative', 'benchmark_cumulative'],
                                labels={'value': 'Cumulative Return', 'variable': 'Portfolio'},
                                title=f"{selected_strategy_name} vs {selected_index_name}"
                            )
                            
                            # Update line names in the legend
                            fig.for_each_trace(lambda t: t.update(
                                name=t.name.replace("strategy_cumulative", "Strategy").replace("benchmark_cumulative", "Benchmark")
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Get backtest metadata for performance metrics
                            backtest_metadata = data_service.get_backtest_metadata(selected_index, selected_strategy)
                            
                            if backtest_metadata:
                                # Performance metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                strategy_return = backtest_metadata.get('annualized_return', 0)
                                benchmark_return = backtest_metadata.get('benchmark_return', 0)
                                return_diff = strategy_return - benchmark_return
                                
                                col1.metric(
                                    "Strategy Return", 
                                    f"{strategy_return:.2%}", 
                                    f"{return_diff:.2%}"
                                )
                                
                                col2.metric(
                                    "Benchmark Return", 
                                    f"{benchmark_return:.2%}"
                                )
                                
                                strategy_sharpe = backtest_metadata.get('sharpe_ratio', 0)
                                benchmark_sharpe = backtest_metadata.get('benchmark_sharpe', 0)
                                sharpe_diff = strategy_sharpe - benchmark_sharpe
                                
                                col3.metric(
                                    "Strategy Sharpe", 
                                    f"{strategy_sharpe:.2f}", 
                                    f"{sharpe_diff:.2f}"
                                )
                                
                                info_ratio = backtest_metadata.get('information_ratio', 0)
                                col4.metric("Information Ratio", f"{info_ratio:.2f}")
                            
                            # Drawdown analysis
                            st.subheader("Drawdown Analysis")
                            
                            # Plot drawdowns
                            fig = px.line(
                                backtest_df, 
                                x='date', 
                                y=['strategy_drawdown', 'benchmark_drawdown'],
                                labels={'value': 'Drawdown', 'variable': 'Portfolio'},
                                title="Drawdown Comparison"
                            )
                            
                            # Update line names and format y-axis
                            fig.for_each_trace(lambda t: t.update(
                                name=t.name.replace("strategy_drawdown", "Strategy").replace("benchmark_drawdown", "Benchmark")
                            ))
                            fig.update_layout(yaxis_tickformat='.1%')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # More performance metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            strategy_vol = backtest_metadata.get('annualized_volatility', 0)
                            benchmark_vol = backtest_metadata.get('benchmark_volatility', 0)
                            vol_diff = benchmark_vol - strategy_vol  # Lower is better
                            
                            col1.metric(
                                "Strategy Volatility", 
                                f"{strategy_vol:.2%}", 
                                f"{vol_diff:.2%}" if vol_diff != 0 else None
                            )
                            
                            col2.metric(
                                "Benchmark Volatility", 
                                f"{benchmark_vol:.2%}"
                            )
                            
                            strategy_dd = backtest_metadata.get('max_drawdown', 0)
                            benchmark_dd = backtest_metadata.get('benchmark_max_drawdown', 0)
                            dd_diff = benchmark_dd - strategy_dd  # Lower is better
                            
                            col3.metric(
                                "Strategy Max DD", 
                                f"{strategy_dd:.2%}", 
                                f"{dd_diff:.2%}" if dd_diff != 0 else None
                            )
                            
                            col4.metric(
                                "Benchmark Max DD", 
                                f"{benchmark_dd:.2%}"
                            )
                            
                            # Rolling analysis
                            if 'rolling_beta' in backtest_df.columns and 'rolling_alpha' in backtest_df.columns:
                                st.subheader("Rolling Analysis")
                                
                                # Create 2-column layout
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Plot rolling beta
                                    fig = px.line(
                                        backtest_df, 
                                        x='date', 
                                        y='rolling_beta',
                                        title="Rolling Beta (90-day)"
                                    )
                                    # Add horizontal line at beta=1
                                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Plot rolling alpha
                                    fig = px.line(
                                        backtest_df, 
                                        x='date', 
                                        y='rolling_alpha',
                                        title="Rolling Alpha (90-day, Annualized)"
                                    )
                                    # Add horizontal line at alpha=0
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Monthly returns analysis
                            if len(backtest_df) > 20:  # Only show if we have enough data
                                st.subheader("Monthly Returns Analysis")
                                
                                # Resample daily returns to monthly
                                monthly_data = pd.DataFrame({
                                    'date': backtest_df['date'],
                                    'strategy': backtest_df['strategy_return'],
                                    'benchmark': backtest_df['benchmark_return']
                                })
                                
                                monthly_data['year'] = monthly_data['date'].dt.year
                                monthly_data['month'] = monthly_data['date'].dt.month
                                
                                # Group by year and month and calculate monthly returns
                                monthly_returns = monthly_data.groupby(['year', 'month']).apply(
                                    lambda x: pd.Series({
                                        'strategy_return': (1 + x['strategy']).prod() - 1,
                                        'benchmark_return': (1 + x['benchmark']).prod() - 1
                                    })
                                ).reset_index()
                                
                                # Create date from year and month for plotting
                                monthly_returns['date'] = pd.to_datetime(
                                    monthly_returns['year'].astype(str) + '-' + 
                                    monthly_returns['month'].astype(str) + '-01'
                                )
                                
                                # Plot monthly returns
                                monthly_returns_df = pd.melt(
                                    monthly_returns,
                                    id_vars=['date', 'year', 'month'],
                                    value_vars=['strategy_return', 'benchmark_return'],
                                    var_name='portfolio',
                                    value_name='return'
                                )
                                
                                monthly_returns_df['portfolio'] = monthly_returns_df['portfolio'].replace({
                                    'strategy_return': 'Strategy',
                                    'benchmark_return': 'Benchmark'
                                })
                                
                                fig = px.bar(
                                    monthly_returns_df,
                                    x='date',
                                    y='return',
                                    color='portfolio',
                                    barmode='group',
                                    title="Monthly Returns"
                                )
                                fig.update_layout(yaxis_tickformat='.1%')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Create monthly returns heatmap
                                pivot_returns = monthly_returns.pivot_table(
                                    index='month', 
                                    columns='year', 
                                    values='strategy_return'
                                )
                                
                                # Replace month numbers with month names
                                month_names = {
                                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                                }
                                pivot_returns.index = [month_names[m] for m in pivot_returns.index]
                                
                                fig = px.imshow(
                                    pivot_returns,
                                    title="Monthly Returns Heatmap",
                                    labels=dict(x="Year", y="Month", color="Return"),
                                    color_continuous_scale='RdBu_r',
                                    text_auto='.1%'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                with backtest_tabs[1]:  # Pair Trading tab
                    st.subheader("Pairs Trading Analysis")
                    
                    # Get available pairs
                    available_pairs = data_service.get_available_pairs()
                    
                    if not available_pairs:
                        st.warning("No pair trading strategies found. Please generate demo data.")
                    else:
                        # Display pair selector
                        pair_options = {p['name']: (p['ticker1'], p['ticker2']) for p in available_pairs}
                        selected_pair_name = st.selectbox(
                            "Select Pair Trading Strategy",
                            list(pair_options.keys()),
                            key="pair_selector"
                        )
                        selected_pair = pair_options[selected_pair_name]
                        
                        # Get pair trading results
                        pair_df = data_service.get_pair_trading_results(selected_pair[0], selected_pair[1])
                        
                        if not pair_df.empty:
                            # Pair trading performance
                            st.subheader("Pair Trading Performance")
                            
                            # Plot price ratio and z-score
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=("Price Ratio", "Z-Score")
                            )
                            
                            # Add price ratio trace
                            fig.add_trace(
                                go.Scatter(
                                    x=pair_df['date'],
                                    y=pair_df['price_ratio'],
                                    mode='lines',
                                    name='Price Ratio'
                                ),
                                row=1, col=1
                            )
                            
                            # Add z-score trace
                            fig.add_trace(
                                go.Scatter(
                                    x=pair_df['date'],
                                    y=pair_df['z_score'],
                                    mode='lines',
                                    name='Z-Score'
                                ),
                                row=2, col=1
                            )
                            
                            # Add threshold lines for z-score
                            fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
                            fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=2, col=1)
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                            
                            fig.update_layout(
                                height=500,
                                title_text=f"Pair Trading: {selected_pair[0]} vs {selected_pair[1]}",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Cumulative returns chart
                            st.subheader("Cumulative Returns")
                            
                            fig = px.line(
                                pair_df,
                                x='date',
                                y='cumulative_return',
                                title=f"Pair Trading Cumulative Return"
                            )
                            fig.update_layout(yaxis_tickformat='.2f')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Get pair trading metadata
                            pair_metadata = data_service.get_pair_trading_metadata(selected_pair[0], selected_pair[1])
                            
                            if pair_metadata:
                                # Performance metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                col1.metric(
                                    "Total Return", 
                                    f"{pair_metadata.get('annualized_return', 0):.2%}"
                                )
                                
                                col2.metric(
                                    "Sharpe Ratio", 
                                    f"{pair_metadata.get('sharpe_ratio', 0):.2f}"
                                )
                                
                                col3.metric(
                                    "Win Rate", 
                                    f"{pair_metadata.get('win_rate', 0):.2%}"
                                )
                                
                                col4.metric(
                                    "Max Drawdown", 
                                    f"{pair_metadata.get('max_drawdown', 0):.2%}"
                                )
                            
                            # Position chart
                            st.subheader("Trading Positions")
                            
                            # Plot positions
                            position_df = pair_df[['date', 'position']].copy()
                            position_df['position_type'] = position_df['position'].map({
                                1: 'Long Spread',
                                -1: 'Short Spread',
                                0: 'No Position'
                            })
                            
                            fig = px.scatter(
                                position_df,
                                x='date',
                                y='position',
                                color='position_type',
                                title="Trading Positions",
                                color_discrete_map={
                                    'Long Spread': 'green',
                                    'Short Spread': 'red',
                                    'No Position': 'gray'
                                }
                            )
                            
                            fig.update_traces(mode='lines+markers')
                            fig.update_layout(yaxis_tickvals=[-1, 0, 1])
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Individual security prices
                            st.subheader("Individual Security Prices")
                            
                            fig = make_subplots(
                                rows=1, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1
                            )
                            
                            # Normalize prices for comparison
                            ticker1_normalized = pair_df['ticker1_price'] / pair_df['ticker1_price'].iloc[0]
                            ticker2_normalized = pair_df['ticker2_price'] / pair_df['ticker2_price'].iloc[0]
                            
                            # Add price traces
                            fig.add_trace(
                                go.Scatter(
                                    x=pair_df['date'],
                                    y=ticker1_normalized,
                                    mode='lines',
                                    name=selected_pair[0]
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=pair_df['date'],
                                    y=ticker2_normalized,
                                    mode='lines',
                                    name=selected_pair[1]
                                )
                            )
                            
                            fig.update_layout(
                                height=400,
                                title_text=f"Normalized Prices ({selected_pair[0]} vs {selected_pair[1]})",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Transaction analysis
                            if 'position' in pair_df.columns:
                                # Calculate position changes to identify transactions
                                position_changes = pair_df['position'].diff().fillna(pair_df['position'])
                                transactions = pair_df[position_changes != 0].copy()
                                
                                if len(transactions) > 0:
                                    st.subheader("Transaction Analysis")
                                    
                                    transactions['transaction_type'] = 'Unknown'
                                    transactions.loc[transactions['position'] == 1, 'transaction_type'] = 'Enter Long Spread'
                                    transactions.loc[transactions['position'] == -1, 'transaction_type'] = 'Enter Short Spread'
                                    transactions.loc[(transactions['position'] == 0) & (transactions['position'].shift(1) == 1), 'transaction_type'] = 'Exit Long Spread'
                                    transactions.loc[(transactions['position'] == 0) & (transactions['position'].shift(1) == -1), 'transaction_type'] = 'Exit Short Spread'
                                    
                                    # Display transactions table
                                    st.write(f"Total Transactions: {len(transactions)}")
                                    
                                    display_transactions = transactions[['date', 'transaction_type', 'price_ratio', 'z_score']].copy()
                                    st.dataframe(display_transactions, use_container_width=True)

            with tab4:
                st.header("Risk Analysis")
                
                # Factor exposures across the index
                st.subheader("Factor Exposures Across the Index")
                
                if factor_data:
                    # For each factor, show distribution of exposures across the index
                    factor_exposures = []
                    
                    for factor_id, df in factor_data.items():
                        if not df.empty:
                            # Get most recent date with data
                            latest_date = df['date'].max()
                            latest_data = df[df['date'] == latest_date]
                            
                            factor_exposures.append({
                                'factor': factor_id,
                                'mean': latest_data['value'].mean(),
                                'median': latest_data['value'].median(),
                                'std': latest_data['value'].std(),
                                'min': latest_data['value'].min(),
                                'max': latest_data['value'].max()
                            })
                    
                    if factor_exposures:
                        exposures_df = pd.DataFrame(factor_exposures)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("Factor Exposure Statistics")
                            st.dataframe(exposures_df.set_index('factor').round(2), use_container_width=True)
                        
                        with col2:
                            # Box plot of factor exposures
                            boxplot_data = []
                            
                            for factor_id, df in factor_data.items():
                                if not df.empty:
                                    # Get most recent date with data
                                    latest_date = df['date'].max()
                                    latest_data = df[df['date'] == latest_date]
                                    
                                    for _, row in latest_data.iterrows():
                                        boxplot_data.append({
                                            'factor': factor_id,
                                            'value': row['value']
                                        })
                            
                            if boxplot_data:
                                boxplot_df = pd.DataFrame(boxplot_data)
                                fig = px.box(boxplot_df, x='factor', y='value',
                                          title="Factor Exposure Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                
                # Factor exposure correlations with returns
                st.subheader("Factor Exposure vs Returns Analysis")
                
                # Select a factor to analyze
                analysis_factor = st.selectbox("Select Factor for Return Analysis", selected_factors,
                                             key="analysis_factor")
                
                if analysis_factor and factor_data.get(analysis_factor) is not None:
                    factor_df = factor_data[analysis_factor]
                    
                    if not factor_df.empty:
                        # Get returns data for the same securities
                        tickers = factor_df['ticker'].unique().tolist()
                        returns_data = data_service.get_security_returns(tickers, start_date_str, end_date_str,
                                                                       frequency=selected_frequency)
                        
                        if not returns_data.empty:
                            # Merge factor data with forward returns
                            analysis_data = []
                            
                            for date in factor_df['date'].unique():
                                # Get factor values for current date
                                current_factors = factor_df[factor_df['date'] == date]
                                
                                # Get forward returns for these securities
                                # Find the next date after the current date
                                next_dates = returns_data[returns_data['date'] > date]['date'].unique()
                                
                                if len(next_dates) > 0:
                                    next_date = min(next_dates)
                                    forward_returns = returns_data[returns_data['date'] == next_date]
                                    
                                    # Merge factor values with forward returns
                                    merged = current_factors.merge(forward_returns, on='ticker', how='inner',
                                                                 suffixes=('_factor', '_returns'))
                                    
                                    for _, row in merged.iterrows():
                                        analysis_data.append({
                                            'factor_value': row['value'],
                                            'forward_return': row['return'],
                                            'ticker': row['ticker']
                                        })
                            
                            if analysis_data:
                                analysis_df = pd.DataFrame(analysis_data)
                                
                                # Create scatter plot
                                fig = px.scatter(analysis_df, x='factor_value', y='forward_return',
                                              title=f"{analysis_factor} Exposure vs Forward Returns",
                                              hover_data=['ticker'])
                                
                                # Add regression line
                                fig.update_layout(
                                    yaxis_tickformat='.1%',
                                    xaxis_title=f"{analysis_factor} Factor Value",
                                    yaxis_title="Forward Return"
                                )
                                
                                # Calculate correlation
                                correlation = analysis_df['factor_value'].corr(analysis_df['forward_return'])
                                
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.metric("Correlation", f"{correlation:.2f}")
                                    
                                    # Calculate IC by date
                                    ic_by_date = []
                                    
                                    for date in factor_df['date'].unique():
                                        # Get factor values for current date
                                        current_factors = factor_df[factor_df['date'] == date]
                                        
                                        # Get forward returns for these securities
                                        next_dates = returns_data[returns_data['date'] > date]['date'].unique()
                                        
                                        if len(next_dates) > 0:
                                            next_date = min(next_dates)
                                            forward_returns = returns_data[returns_data['date'] == next_date]
                                            
                                            # Merge factor values with forward returns
                                            merged = current_factors.merge(forward_returns, on='ticker', how='inner',
                                                                         suffixes=('_factor', '_returns'))
                                            
                                            if len(merged) > 5:  # Ensure enough data points
                                                ic = merged['value'].corr(merged['return'])
                                                
                                                ic_by_date.append({
                                                    'date': date,
                                                    'ic': ic
                                                })
                                    
                                    if ic_by_date:
                                        ic_df = pd.DataFrame(ic_by_date)
                                        st.write(f"Information Coefficient (IC) Stats:")
                                        st.write(f"Mean IC: {ic_df['ic'].mean():.2f}")
                                        st.write(f"IC t-stat: {ic_df['ic'].mean() / (ic_df['ic'].std() / np.sqrt(len(ic_df))):.2f}")
                                        
                # Risk decomposition
                st.subheader("Risk Decomposition")
                
                # Simple risk decomposition based on factor exposures
                if factor_data:
                    st.write("This section provides a simplified risk decomposition based on factor exposures.")
                    
                    # Calculate average factor exposures for the index
                    factor_exposures = {}
                    
                    for factor_id, df in factor_data.items():
                        if not df.empty:
                            # Get most recent date with data
                            latest_date = df['date'].max()
                            latest_data = df[df['date'] == latest_date]
                            
                            factor_exposures[factor_id] = latest_data['value'].mean()
                    
                    # Calculate factor volatilities based on long-short returns
                    factor_vols = {}
                    
                    for factor_id, returns_df in factor_returns.items():
                        if factor_id in factor_exposures:
                            # Get long-short returns if available
                            ls_returns = returns_df[returns_df['bucket'] == 'long_short']
                            
                            if not ls_returns.empty:
                                factor_vols[factor_id] = ls_returns['return'].std() * np.sqrt(252)
                    
                    # Create risk decomposition
                    risk_data = []
                    
                    total_risk = 0
                    for factor_id, exposure in factor_exposures.items():
                        if factor_id in factor_vols:
                            factor_risk = abs(exposure) * factor_vols[factor_id]
                            total_risk += factor_risk
                            
                            risk_data.append({
                                'factor': factor_id,
                                'exposure': exposure,
                                'volatility': factor_vols[factor_id],
                                'risk_contribution': factor_risk
                            })
                    
                    if risk_data:
                        risk_df = pd.DataFrame(risk_data)
                        
                        # Add percentage contribution
                        risk_df['contribution_pct'] = risk_df['risk_contribution'] / total_risk
                        
                        # Sort by contribution
                        risk_df = risk_df.sort_values('contribution_pct', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("Risk Decomposition by Factor")
                            display_risk = risk_df.copy()
                            display_risk['volatility'] = display_risk['volatility'].map('{:.2%}'.format)
                            display_risk['risk_contribution'] = display_risk['risk_contribution'].map('{:.2%}'.format)
                            display_risk['contribution_pct'] = display_risk['contribution_pct'].map('{:.2%}'.format)
                            
                            st.dataframe(display_risk.set_index('factor'), use_container_width=True)
                        
                        with col2:
                            # Create pie chart of risk contributions
                            fig = px.pie(risk_df, values='risk_contribution', names='factor',
                                       title="Factor Risk Contribution")
                            st.plotly_chart(fig, use_container_width=True)

    # Add a sidebar section for additional settings and documentation
    with st.sidebar:
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This Equity Trading Research Platform provides tools for:
        - Factor Analysis and Visualization
        - Portfolio Construction
        - Backtesting Trading Strategies
        - Risk Analysis and Decomposition
        
        The application uses a built-in database to store and retrieve market data for demonstration purposes.
        """)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            clear_cache = st.button("Clear Cache")
            if clear_cache:
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
                
            reset_db = st.button("Reset Database")
            if reset_db:
                if st.warning("This will delete all data in the database. Are you sure?"):
                    db_manager = DatabaseManager()
                    success = db_manager.reset_database()
                    if success:
                        st.success("Database reset successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to reset database.")

# Footer
st.markdown("---")
st.markdown("Equity Trading Research Platform | Built with Streamlit")