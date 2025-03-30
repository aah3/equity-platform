# qFactor.py
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:36:21 2025

@author: alfredo
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import time
import warnings
from functools import lru_cache

import matplotlib.pyplot as plt
import seaborn as sns

class YahooFactor(BaseModel):
    """
    YahooFactor class for handling factor data using Yahoo Finance
    
    This class mimics the functionality of BQLFactor but uses the yfinance API
    instead of Bloomberg Query Language.
    
    Attributes:
        name: Name of the factor
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        universe: List of ticker symbols
        data: DataFrame containing factor data
        description: Optional description of the factor
        category: Optional category of the factor
    """
    name: str
    start_date: str
    end_date: str
    universe: List[str]
    data: Optional[pd.DataFrame] = None
    description: Optional[str] = None
    category: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('data')
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        if v is not None:
            required_columns = {'date', 'factor_name', 'sid', 'value'}
            if not all(col in v.columns for col in required_columns):
                missing_cols = required_columns - set(v.columns)
                raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        return v

    def get_factor_data(self, factor_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get factor data using yfinance
        
        Args:
            factor_type: Type of factor to retrieve (size, value, momentum, etc.)
            **kwargs: Additional arguments for specific factor calculations
            
        Returns:
            DataFrame with standardized factor data format
        """
        if factor_type is None:
            factor_type = self.name
        
        if self.end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))

        if self.data is None:
            self.data = self._download_price_data(tuple(self.universe), self.start_date, end_date)

        try:
            if factor_type == 'size':
                df = self.get_factor_size()
            elif factor_type == 'value':
                df = self.get_factor_value()
            elif factor_type == 'beta':
                df = self.get_factor_beta(**kwargs)
            elif factor_type == 'momentum':
                df = self.get_factor_momentum(**kwargs)
            elif factor_type == 'profit':
                df = self.get_factor_profit()
            elif factor_type == 'volatility':
                df = self.get_factor_volatility(**kwargs)
            elif factor_type == 'dividend_yield':
                df = self.get_factor_dividend_yield()
            elif factor_type == 'pe_ratio':
                df = self.get_factor_pe_ratio()
            else:
                raise ValueError(f"Unsupported factor type: {factor_type}")
                
            if df is not None and not df.empty:
                df.dropna(inplace=True)
                # Set date as index for consistency with BQLFactor
                if 'date' in df.columns:
                    df.index = df['date']
                    df.index.name = 'index'
            return df
            
        except Exception as e:
            raise Exception(f"Error getting factor data: {str(e)}")
    
    # @lru_cache(maxsize=32)
    def _download_price_data(self, tickers: Tuple[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download price data for multiple tickers with caching
        
        Args:
            tickers: Tuple of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            MultiIndex DataFrame with price data
        """
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Convert tuple to list since yfinance expects a list
        ticker_list = list(tickers)
        
        try:
            # Download data with retries
            for attempt in range(3):
                try:
                    data = yf.download(
                        ticker_list,
                        start=start_date,
                        end=end_date,
                        actions=True,  # Include dividends and splits
                        progress=False,
                        auto_adjust=False  # Keep unadjusted and adjusted prices
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
            
            # If only one ticker is requested, yfinance doesn't return a MultiIndex DataFrame
            if len(ticker_list) == 1 and not isinstance(data.columns, pd.MultiIndex):
                ticker = ticker_list[0]
                # Convert to MultiIndex format for consistency
                data_multiindex = pd.DataFrame({
                    ('Adj Close', ticker): data['Adj Close'],
                    ('Close', ticker): data['Close'],
                    ('High', ticker): data['High'],
                    ('Low', ticker): data['Low'],
                    ('Open', ticker): data['Open'],
                    ('Volume', ticker): data['Volume'],
                    ('Dividends', ticker): data['Dividends'],
                    ('Stock Splits', ticker): data['Stock Splits']
                })
                return data_multiindex
            
            return data
            
        except Exception as e:
            print(f"Error downloading data for tickers {ticker_list}: {str(e)}")
            # Return empty DataFrame with expected structure
            index = pd.date_range(start=start_date, end=end_date, freq='D')
            columns = pd.MultiIndex.from_product([
                ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Dividends', 'Stock Splits'],
                ticker_list
            ])
            return pd.DataFrame(index=index, columns=columns)

    def _format_output_df(self, raw_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """
        Format raw data to match BQLFactor output format
        
        Args:
            raw_data: DataFrame with raw factor data
            factor_name: Name of the factor
            
        Returns:
            Formatted DataFrame
        """
        df = pd.DataFrame()
        df['date'] = raw_data['date'] # raw_data.index
        df['factor_name'] = factor_name
        df['sid'] = raw_data['sid']
        df['value'] = raw_data['value']
        
        # Standardize date format
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Remove duplicates and sort
        df.drop_duplicates(inplace=True)
        df.sort_values(['sid', 'date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index = pd.to_datetime(df.date)
        
        return df
    
    def get_factor_size(self) -> pd.DataFrame:
        """
        Calculate size factor (market capitalization) for the universe
        
        Returns:
            DataFrame with size factor data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price and volume data
            # data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            data = self.data.copy()
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate market cap for each ticker and date
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_data = data[('Adj Close', ticker)].dropna()
                    if ticker_data.empty:
                        continue
                    
                    # Get shares outstanding (approx) from most recent data
                    ticker_obj = yf.Ticker(ticker)
                    try:
                        # Get shares from yfinance info
                        info = ticker_obj.info
                        if 'sharesOutstanding' in info and info['sharesOutstanding'] is not None:
                            shares = info['sharesOutstanding']
                        else:
                            # Fallback to basic shares value
                            shares = 1000000  # Default value if unavailable
                    except:
                        # If info retrieval fails, use default value
                        shares = 1000000
                    
                    # Calculate market cap for each day (price * shares)
                    for date, price in ticker_data.items():
                        if pd.notnull(price):
                            mkt_cap = price * shares
                            # For size factor, we use 1/log(market cap) to match BQLFactor
                            # This is because larger market cap should have smaller size factor values
                            if mkt_cap > 0:
                                size_value = 1.0 / np.log(mkt_cap)
                                result_data.append({
                                    'date': date,
                                    'sid': ticker,
                                    'value': size_value
                                })
                except Exception as e:
                    print(f"Error calculating size for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'size')
            
        except Exception as e:
            print(f"Error in get_factor_size: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_value(self) -> pd.DataFrame:
        """
        Calculate value factor (price-to-book ratio) for the universe
        
        Returns:
            DataFrame with value factor data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            if self.data is None:
                data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
                self.data = data.copy()
            else:
                data = self.data.copy()
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate price-to-book for each ticker and date
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_data = data[('Adj Close', ticker)].dropna()
                    if ticker_data.empty:
                        continue
                    
                    # Get book value per share
                    ticker_obj = yf.Ticker(ticker)
                    try:
                        # Get book value from balance sheet
                        balance_sheet = ticker_obj.balance_sheet
                        if not balance_sheet.empty and 'Common Stock Equity' in balance_sheet.index:
                            equity = balance_sheet.loc['Common Stock Equity'].iloc[0]
                            
                            # Get shares outstanding
                            info = ticker_obj.info
                            if 'sharesOutstanding' in info and info['sharesOutstanding'] is not None:
                                shares = info['sharesOutstanding']
                                book_value_per_share = equity / shares
                            else:
                                # If shares data is unavailable, use a default P/B ratio
                                book_value_per_share = None
                        else:
                            book_value_per_share = None
                    except:
                        book_value_per_share = None
                    
                    # If book value cannot be calculated, get P/B from info
                    if book_value_per_share is None or book_value_per_share <= 0:
                        try:
                            info = ticker_obj.info
                            if 'priceToBook' in info and info['priceToBook'] is not None and info['priceToBook'] > 0:
                                # Use the most recent price-to-book as a constant for the entire period
                                pb_ratio = info['priceToBook']
                                
                                for date, price in ticker_data.items():
                                    if pd.notnull(price):
                                        result_data.append({
                                            'date': date,
                                            'sid': ticker,
                                            'value': pb_ratio  # Use the same P/B ratio for all dates
                                        })
                            else:
                                # If P/B is not available, skip this ticker
                                continue
                        except:
                            continue
                    else:
                        # Calculate P/B ratio for each day
                        for date, price in ticker_data.items():
                            if pd.notnull(price) and book_value_per_share > 0:
                                pb_ratio = price / book_value_per_share
                                result_data.append({
                                    'date': date,
                                    'sid': ticker,
                                    'value': pb_ratio
                                })
                except Exception as e:
                    print(f"Error calculating value for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'value')
            
        except Exception as e:
            print(f"Error in get_factor_value: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_beta(self, benchmark: str = 'SPY', window: int = 252, **kwargs) -> pd.DataFrame:
        """
        Calculate beta factor for the universe
        
        Args:
            benchmark: Ticker symbol for the benchmark index
            window: Rolling window for beta calculation (in trading days)
            
        Returns:
            DataFrame with beta factor data
        """
        try:
            # Adjust start date to include enough data for the window
            start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            adjusted_start = (start_date_dt - timedelta(days=window * 1.5)).strftime('%Y-%m-%d')
            
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data for both tickers and benchmark
            tickers_with_benchmark = list(self.universe) + [benchmark]
            data = self._download_price_data(tuple(tickers_with_benchmark), adjusted_start, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate daily returns
            returns = pd.DataFrame()
            
            # Get benchmark returns
            benchmark_prices = data[('Adj Close', benchmark)].dropna()
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Calculate beta for each ticker
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    ticker_returns = ticker_prices.pct_change().dropna()
                    
                    # Combine with benchmark returns
                    combined_returns = pd.DataFrame({
                        'ticker': ticker_returns,
                        'benchmark': benchmark_returns
                    })
                    combined_returns.dropna(inplace=True)
                    
                    # Calculate rolling beta
                    if combined_returns.shape[0] > window:
                        # Calculate rolling covariance and variance
                        rolling_cov = combined_returns['ticker'].rolling(window=window).cov(combined_returns['benchmark'])
                        rolling_var = combined_returns['benchmark'].rolling(window=window).var()
                        
                        # Calculate beta
                        rolling_beta = rolling_cov / rolling_var
                        
                        # Drop NaN values from initial window period
                        rolling_beta = rolling_beta.dropna()
                        
                        # Filter dates to match requested range
                        start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
                        rolling_beta = rolling_beta[rolling_beta.index >= pd.Timestamp(start_date_dt)]
                        
                        for date, beta in rolling_beta.items():
                            if pd.notnull(beta):
                                result_data.append({
                                    'date': date,
                                    'sid': ticker,
                                    'value': beta
                                })
                except Exception as e:
                    print(f"Error calculating beta for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'beta')
            
        except Exception as e:
            print(f"Error in get_factor_beta: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_momentum(self, shift_lag: int = 21, **kwargs) -> pd.DataFrame:
        """
        Calculate momentum factor for the universe
        
        Args:
            shift_lag: Number of trading days for momentum calculation (default 21 days = ~1 month)
            
        Returns:
            DataFrame with momentum factor data
        """
        try:
            # Adjust start date to include enough data for the shift_lag
            start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            adjusted_start = (start_date_dt - timedelta(days=shift_lag * 1.5)).strftime('%Y-%m-%d')
            
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            data = self._download_price_data(
                tuple(self.universe), adjusted_start, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate momentum for each ticker
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    
                    # Calculate momentum as price change over shift_lag days
                    momentum = ticker_prices.pct_change(periods=shift_lag)
                    
                    # Drop NaN values from initial shift_lag period
                    momentum = momentum.dropna()
                    
                    # Filter dates to match requested range
                    start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
                    momentum = momentum[momentum.index >= pd.Timestamp(start_date_dt)]
                    
                    for date, value in momentum.items():
                        if pd.notnull(value):
                            result_data.append({
                                'date': date,
                                'sid': ticker,
                                'value': value
                            })
                except Exception as e:
                    print(f"Error calculating momentum for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'momentum')
            
        except Exception as e:
            print(f"Error in get_factor_momentum: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_profit(self) -> pd.DataFrame:
        """
        Calculate profit margin factor for the universe
        
        Returns:
            DataFrame with profit margin factor data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data to get the trading dates
            data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Get all trading dates in the period
            trading_dates = data.index
            
            result_data = []
            
            for ticker in self.universe:
                try:
                    # Get fundamentals data
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Get income statement
                    income_stmt = ticker_obj.income_stmt
                    
                    if income_stmt.empty:
                        continue
                    
                    # Get quarterly financials for more data points
                    try:
                        quarterly_income = ticker_obj.quarterly_income_stmt
                        if not quarterly_income.empty:
                            # Combine annual and quarterly for more data points
                            income_stmt = pd.concat([income_stmt, quarterly_income], axis=1)
                    except:
                        pass  # Continue with just annual data if quarterly fails
                    
                    # Calculate profit margin
                    if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                        net_income = income_stmt.loc['Net Income']
                        net_income = net_income.dropna().sort_index(inplace=False)
                        net_income = net_income[~net_income.index.duplicated(keep='first')]
                        revenue = income_stmt.loc['Total Revenue']
                        revenue = revenue.dropna().sort_index(inplace=False)
                        revenue = revenue[~revenue.index.duplicated(keep='first')]
                        
                        # Calculate profit margin for each period
                        profit_margins = {}
                        for date, rev in revenue.items():
                            # TO DO: check output from net_income
                            if pd.notnull(rev) and (rev!=0) and pd.notnull(net_income[date]):
                                profit_margin = net_income[date] / rev
                                profit_margins[date] = profit_margin
                        
                        # Assign profit margin to each trading day by forward filling
                        if profit_margins:
                            # Convert dictionary to Series for easier manipulation
                            margins_series = pd.Series(profit_margins)
                            margins_series.sort_index(inplace=True)  # Sort by date
                            
                            # Assign to each trading day by forward filling
                            for date in trading_dates:
                                # Find the most recent financial data before this date
                                relevant_margins = margins_series[margins_series.index <= date]
                                if not relevant_margins.empty:
                                    margin = relevant_margins.iloc[-1]  # Get most recent value
                                    result_data.append({
                                        'date': date,
                                        'sid': ticker,
                                        'value': margin
                                    })
                except Exception as e:
                    print(f"Error calculating profit margin for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'profit')
            
        except Exception as e:
            print(f"Error in get_factor_profit: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_volatility(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """
        Calculate volatility factor for the universe
        
        Args:
            window: Rolling window for volatility calculation (in trading days)
            
        Returns:
            DataFrame with volatility factor data
        """
        try:
            # Adjust start date to include enough data for the window
            start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            adjusted_start = (start_date_dt - timedelta(days=window * 1.5)).strftime('%Y-%m-%d')
            
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            data = self._download_price_data(tuple(self.universe), adjusted_start, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate volatility for each ticker
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    ticker_returns = ticker_prices.pct_change().dropna()
                    
                    # Calculate rolling standard deviation
                    rolling_std = ticker_returns.rolling(window=window).std()
                    
                    # Annualize the volatility
                    rolling_vol = rolling_std * np.sqrt(252)
                    
                    # Drop NaN values from initial window period
                    rolling_vol = rolling_vol.dropna()
                    
                    # Filter dates to match requested range
                    start_date_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
                    rolling_vol = rolling_vol[rolling_vol.index >= pd.Timestamp(start_date_dt)]
                    
                    for date, vol in rolling_vol.items():
                        if pd.notnull(vol):
                            result_data.append({
                                'date': date,
                                'sid': ticker,
                                'value': vol
                            })
                except Exception as e:
                    print(f"Error calculating volatility for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'volatility')
            
        except Exception as e:
            print(f"Error in get_factor_volatility: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_dividend_yield(self) -> pd.DataFrame:
        """
        Calculate dividend yield factor for the universe
        
        Returns:
            DataFrame with dividend yield factor data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price and dividend data
            # data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            data = self.data.copy()
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate dividend yield for each ticker
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns or ('Dividends', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price and dividend data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    ticker_dividends = data[('Dividends', ticker)]
                    
                    # Calculate trailing 12-month dividends
                    ttm_dividends = ticker_dividends.rolling(window=252).sum()
                    
                    # Calculate dividend yield
                    dividend_yield = ttm_dividends / ticker_prices
                    
                    # Filter out zeros and NaNs
                    dividend_yield = dividend_yield[dividend_yield > 0].dropna()
                    
                    for date, dy in dividend_yield.items():
                        if pd.notnull(dy):
                            result_data.append({
                                'date': date,
                                'sid': ticker,
                                'value': dy
                            })
                except Exception as e:
                    print(f"Error calculating dividend yield for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'dividend_yield')
            
        except Exception as e:
            print(f"Error in get_factor_dividend_yield: {str(e)}")
            return pd.DataFrame()
    
    def get_factor_pe_ratio(self) -> pd.DataFrame:
        """
        Calculate price-to-earnings ratio factor for the universe
        
        Returns:
            DataFrame with P/E ratio factor data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            # data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            data = self.data.copy()
            
            if data.empty:
                return pd.DataFrame()
            
            # Get all trading dates in the period
            trading_dates = data.index
            
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get fundamentals data
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Get earnings information
                    earnings = ticker_obj.earnings
                    
                    if earnings is None or earnings.empty:
                        # Try to get P/E ratio from info
                        info = ticker_obj.info
                        if 'trailingPE' in info and info['trailingPE'] is not None:
                            pe_ratio = info['trailingPE']
                            
                            # Apply this P/E ratio to all dates
                            for date in trading_dates:
                                result_data.append({
                                    'date': date,
                                    'sid': ticker,
                                    'value': pe_ratio
                                })
                        continue
                    
                    # Get quarterly earnings for more data points
                    try:
                        quarterly_earnings = ticker_obj.quarterly_earnings
                        if not quarterly_earnings.empty:
                            # Combine annual and quarterly earnings
                            earnings = pd.concat([earnings, quarterly_earnings], axis=0)
                    except:
                        pass  # Continue with just annual data if quarterly fails
                    
                    # Process earnings data
                    earnings = earnings.sort_index(ascending=False)  # Most recent first
                    eps_ttm = earnings['Earnings'].rolling(4).sum()  # Trailing 12-month EPS
                    
                    # Get price data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    
                    # Calculate P/E ratio for each trading day
                    for date in trading_dates:
                        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                        if date_str in ticker_prices.index:
                            price = ticker_prices.loc[date_str]
                            
                            # Find the most recent earnings data point before this date
                            eps_date = None
                            for earnings_date in eps_ttm.index:
                                if earnings_date <= date:
                                    eps_date = earnings_date
                                    break
                                    
                            if eps_date is not None and eps_ttm.loc[eps_date] > 0:
                                pe_ratio = price / eps_ttm.loc[eps_date]
                                result_data.append({
                                    'date': date,
                                    'sid': ticker,
                                    'value': pe_ratio
                                })
                except Exception as e:
                    print(f"Error calculating P/E ratio for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format and return
            return self._format_output_df(result_df, 'pe_ratio')
            
        except Exception as e:
            print(f"Error in get_factor_pe_ratio: {str(e)}")
            return pd.DataFrame()
    
    def get_returns(self) -> pd.DataFrame:
        """
        Get returns data for the universe
        
        Returns:
            DataFrame with returns data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate returns for each ticker
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    ticker_returns = ticker_prices.pct_change().dropna()
                    
                    for date, ret in ticker_returns.items():
                        if pd.notnull(ret):
                            result_data.append({
                                'date': date,
                                'sid': ticker,
                                'return': ret
                            })
                except Exception as e:
                    print(f"Error calculating returns for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format DataFrame
            df_ret = pd.DataFrame()
            df_ret['date'] = result_df['date']
            df_ret['sid'] = result_df['sid']
            df_ret['return'] = result_df['return']
            
            # Standardize date format
            df_ret['date'] = pd.to_datetime(df_ret['date']).dt.date
            
            # Set index for consistency with BQLFactor
            df_ret.index = df_ret['date']
            df_ret.index.name = 'index'
            
            # Sort and remove duplicates
            df_ret.sort_values(['sid', 'date'], inplace=True)
            df_ret.drop_duplicates(inplace=True)
            
            return df_ret
            
        except Exception as e:
            print(f"Error in get_returns: {str(e)}")
            return pd.DataFrame()
    
    def get_market_cap(self) -> pd.DataFrame:
        """
        Get market cap data for the universe
        
        Returns:
            DataFrame with market cap data
        """
        try:
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate market cap for each ticker
            result_data = []
            
            for ticker in self.universe:
                # Skip tickers with no data
                if ('Adj Close', ticker) not in data.columns:
                    continue
                
                try:
                    # Get price data for this ticker
                    ticker_prices = data[('Adj Close', ticker)].dropna()
                    
                    # Get shares outstanding
                    ticker_obj = yf.Ticker(ticker)
                    try:
                        info = ticker_obj.info
                        if 'sharesOutstanding' in info and info['sharesOutstanding'] is not None:
                            shares = info['sharesOutstanding']
                        else:
                            shares = 1000000  # Default value if unavailable
                    except:
                        shares = 1000000  # Default value if info retrieval fails
                    
                    # Calculate market cap for each day
                    for date, price in ticker_prices.items():
                        if pd.notnull(price):
                            mkt_cap = price * shares
                            result_data.append({
                                'date': date,
                                'sid': ticker,
                                'mktcap': mkt_cap
                            })
                except Exception as e:
                    print(f"Error calculating market cap for {ticker}: {str(e)}")
                    continue
            
            if not result_data:
                return pd.DataFrame()
                
            # Create DataFrame from results
            result_df = pd.DataFrame(result_data)
            
            # Format DataFrame
            df_cap = pd.DataFrame()
            df_cap['date'] = result_df['date']
            df_cap['sid'] = result_df['sid']
            df_cap['mktcap'] = result_df['mktcap']
            
            # Standardize date format
            df_cap['date'] = pd.to_datetime(df_cap['date']).dt.date
            
            # Set index for consistency with BQLFactor
            df_cap.index = df_cap['date']
            df_cap.index.name = 'index'
            
            # Sort and remove duplicates
            df_cap.sort_values(['sid', 'date'], inplace=True)
            df_cap.drop_duplicates(inplace=True)
            
            return df_cap
            
        except Exception as e:
            print(f"Error in get_market_cap: {str(e)}")
            return pd.DataFrame()
        
def main():
    """
    Example demonstrating how to use the YahooFactor class to analyze factor data
    """
    print("Starting YahooFactor Usage Example...")
    
    # 1. Define a universe of stocks to analyze
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA']
    
    # 2. Define the date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year of data
    
    print(f"Analyzing factors for {len(tech_stocks)} tech stocks from {start_date} to {end_date}")
    
    # 3. Initialize YahooFactor for different factors
    factors_to_analyze = ['size', 'momentum', 'beta', 'value', 'volatility', 'profit']
    # factors_to_analyze = ['volatility', 'value', 'profit'] # ['size', 'beta', 'momentum']
    factor_data = {}
    # breakpoint()
    for factor_name in factors_to_analyze:
        print(f"\nRetrieving {factor_name} factor data...")
        
        # Initialize factor object
        factor = YahooFactor(
            name=factor_name,
            start_date=start_date,
            end_date=end_date,
            universe=tech_stocks,
            description=f"{factor_name.capitalize()} factor data for tech stocks",
            category="equity"
        )
        
        # Get factor data
        factor_data[factor_name] = factor.get_factor_data()
        
        # Print summary
        if factor_data[factor_name] is not None and not factor_data[factor_name].empty:
            print(f"Retrieved {len(factor_data[factor_name])} data points")
            print("Sample data:")
            print(factor_data[factor_name].head())
        else:
            print(f"No data available for {factor_name} factor")
    
    # 4. Analyze returns data
    print("\nRetrieving returns data...")
    returns_factor = YahooFactor(
        name="returns",
        start_date=start_date,
        end_date=end_date,
        universe=tech_stocks
    )
    returns_data = returns_factor.get_returns()
    
    if returns_data is not None and not returns_data.empty:
        print(f"Retrieved {len(returns_data)} return data points")
        print("Sample returns data:")
        print(returns_data.head())
        
        # Calculate cumulative returns
        pivot_returns = returns_data.pivot(columns='sid', values='return')
        cum_returns = (1 + pivot_returns).cumprod()
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        cum_returns.plot()
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        plt.savefig('cumulative_returns.png')
        print("Cumulative returns chart saved as 'cumulative_returns.png'")
    
    # 5. Analyze factor correlations
    print("\nAnalyzing factor correlations...")
    
    # Create a dictionary to store factor values for each stock
    stock_factors = {stock: {} for stock in tech_stocks}
    
    # Get the most recent factor values for each stock
    for factor_name, df in factor_data.items():
        if df is not None and not df.empty:
            # Get the most recent date
            latest_date = df['date'].max()
            
            # Get factor values for this date
            latest_factors = df[df['date'] == latest_date]
            
            for _, row in latest_factors.iterrows():
                stock = row['sid']
                if stock in stock_factors:
                    stock_factors[stock][factor_name] = row['value']
    
    # Convert to DataFrame for analysis
    factors_df = pd.DataFrame.from_dict(stock_factors, orient='index')
    
    if not factors_df.empty and factors_df.shape[1] > 1:
        print("\nLatest factor values:")
        print(factors_df)
        
        # Calculate factor correlations
        corr_matrix = factors_df.corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Factor Correlations')
        plt.tight_layout()
        plt.savefig('factor_correlations.png')
        print("Factor correlation matrix saved as 'factor_correlations.png'")
    
    # 6. Create a simple factor ranking strategy
    if 'momentum' in factor_data and 'volatility' in factor_data and factor_data['momentum'] is not None and factor_data['volatility'] is not None:
        print("\nCreating a simple factor-based stock ranking...")
        
        # Get latest date that has data for both factors
        momentum_dates = pd.to_datetime(factor_data['momentum']['date']).dt.date
        volatility_dates = pd.to_datetime(factor_data['volatility']['date']).dt.date
        
        common_dates = sorted(set(momentum_dates) & set(volatility_dates), reverse=True)
        
        if common_dates:
            analysis_date = common_dates[0]
            print(f"Analysis date: {analysis_date}")
            
            # Get momentum data for this date
            momentum_data = factor_data['momentum'][
                pd.to_datetime(factor_data['momentum']['date']).dt.date == analysis_date
            ]
            
            # Get volatility data for this date
            volatility_data = factor_data['volatility'][
                pd.to_datetime(factor_data['volatility']['date']).dt.date == analysis_date
            ]
            
            # Create a DataFrame with both factors
            ranking_df = momentum_data[['sid', 'value']].rename(columns={'value': 'momentum'})
            volatility_vals = volatility_data.set_index('sid')['value']
            ranking_df['volatility'] = ranking_df['sid'].map(lambda x: volatility_vals.get(x, np.nan))
            
            # Calculate risk-adjusted momentum (momentum / volatility)
            ranking_df['risk_adj_momentum'] = ranking_df['momentum'] / ranking_df['volatility']
            
            # Rank stocks
            ranking_df.sort_values('risk_adj_momentum', ascending=False, inplace=True)
            
            print("\nStocks ranked by risk-adjusted momentum:")
            print(ranking_df[['sid', 'momentum', 'volatility', 'risk_adj_momentum']])
            
            # Plot momentum vs volatility
            plt.figure(figsize=(10, 8))
            plt.scatter(ranking_df['volatility'], ranking_df['momentum'], s=100)
            
            # Add stock labels
            for i, row in ranking_df.iterrows():
                plt.annotate(row['sid'], 
                            (row['volatility'], row['momentum']),
                            xytext=(5, 5), 
                            textcoords='offset points')
            
            plt.title(f'Momentum vs Volatility ({analysis_date})')
            plt.xlabel('Volatility')
            plt.ylabel('Momentum')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('momentum_vs_volatility.png')
            print("Momentum vs Volatility chart saved as 'momentum_vs_volatility.png'")
    
    print("\nYahooFactor usage example completed successfully!")

if __name__ == "__main__":
    main()