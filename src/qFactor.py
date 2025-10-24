# qFactor.py
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:36:21 2025

@author: alfredo
"""

from typing import Dict, List, Optional, Union, Tuple, Literal, Any, Sequence
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
import datetime
from datetime import date, timedelta #, datetime
import pandas as pd
import numpy as np
import yfinance as yf
import time
import warnings
import urllib
import urllib.request
from functools import lru_cache

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.sparse as sp
import statsmodels.api as sm
from utils import *

from enum import Enum
from decimal import Decimal
import logging # Good practice to log errors


from qOptimization import (PurePortfolioConstraints, PureFactorOptimizer)
import qBacktest as bt

# Helper functions: get index constituents

# --- Configuration Map ---
# This is the "best way" to generalize the function.
# We map the ticker to the URL and the specific table index we need to scrape.
INDEX_CONFIG = {
    '^DJI': {
        'url': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components', #'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
        'table_index': 2 # The table of component stocks
    },
    '^GSPC': {
        'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', # 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'table_index': 0 # The first table on the page is the components list
    },
    '^NDX': {
        'url': 'https://en.wikipedia.org/wiki/Nasdaq-100#Components', # 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'table_index': 4 # The table titled "Components"
    }
    # You can easily add more indices here:
    # '^FTSE': { 'url': '...', 'table_index': ... },
}


def get_index_constituents(ticker: str) -> pd.DataFrame:
    """
    Fetches the constituent stocks for a given index ticker from Wikipedia.

    Args:
        ticker: The index ticker symbol (e.g., '^DJI', '^GSPC', '^NDX').

    Returns:
        A pandas DataFrame with the constituent data, or an empty
        DataFrame if the ticker is not found or an error occurs.
    """
    # 1. Look up the ticker in our configuration map
    config = INDEX_CONFIG.get(ticker)

    # 2. If ticker is not in our map, return an empty DataFrame
    if not config:
        logging.warning(f"Ticker '{ticker}' not configured. Returning empty DataFrame.")
        return pd.DataFrame()

    url_name = config['url']
    table_idx = config['table_index']

    try:
        # 3. Create the request with a User-Agent to avoid 403 Forbidden error
        req = urllib.request.Request(
            url_name, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        # 4. Open the request and pass the file-like response to pandas
        with urllib.request.urlopen(req) as response:
            # pd.read_html returns a list of *all* tables on the page
            all_tables = pd.read_html(response)

        # 5. Select the correct table using the index from our config
        constituents_df = all_tables[table_idx]
        
        logging.info(f"Successfully fetched {len(constituents_df)} constituents for {ticker}.")
        return constituents_df

    except urllib.error.HTTPError as e:
        # Handle web errors (e.g., 404 Not Found, 500 Server Error)
        logging.error(f"HTTP Error for {ticker} at {url_name}: {e}")
        return pd.DataFrame()
    except IndexError:
        # Handle if the page structure changed and our table_index is wrong
        logging.error(f"Error: Table index {table_idx} not found for {ticker} at {url_name}.")
        return pd.DataFrame()
    except Exception as e:
        # Handle any other unexpected errors
        logging.error(f"An unexpected error occurred for {ticker}: {e}")
        return pd.DataFrame()


# Config ID function
def default_json_encoder(o):
    import datetime
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)  # or str(o) if you want to preserve exact value
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def generate_config_id(config: dict, prefix: Optional[str] = None) -> str:
    """
    Generate a deterministic, unique config_id for a given configuration dictionary.
    
    Args:
        config (dict): The configuration dictionary (should be serializable).
        prefix (str, optional): Optional prefix for readability (e.g., 'yahoo', '20240601').
    
    Returns:
        str: A unique, deterministic config_id.
    """
    import hashlib
    import json
    from datetime import datetime

    # Serialize config to JSON with sorted keys for consistency
    config_json = json.dumps(config, sort_keys=True, separators=(',', ':'), default=default_json_encoder)
    # Hash the JSON string
    config_hash = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:12]  # Shorten for readability
    # Optionally add a prefix (e.g., data source, date)
    if prefix:
        config_id = f"{prefix}_{config_hash}"
    else:
        config_id = config_hash
    return config_id

"""
# Define framework classes
"""
class DataSource(str, Enum):
    """Available data sources"""
    YAHOO = "yahoo"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    CUSTOM = "custom"

class Universe(str, Enum):
    """Global investment universe options"""
    # North America
    NDX = "NDX Index"
    SPX = "SPX Index"
    RTY = "RTY Index"
    SPTSX = "SPTSX Index"
    MID = "MID Index"
    SML = "SML Index"
    INDU = "INDU Index"
    
    # Europe
    SXXP = "SXXP Index"
    UKX = "UKX Index"
    CAC = "CAC Index"
    DAX = "DAX Index"
    IBEX = "IBEX Index"
    FTSEMIB = "FTSEMIB Index"
    AEX = "AEX Index"
    SMI = "SMI Index"
    NDDUEMU = "NDDUEMU Index"
    
    # Asia Pacific
    NKY = "NKY Index"
    HSI = "HSI Index"
    SHSZ300 = "SHSZ300 Index"
    AS51 = "AS51 Index"
    KOSPI = "KOSPI Index"
    NIFTY = "NIFTY Index"
    STI = "STI Index"
    
    # Latin America
    IBOV = "IBOV Index"
    MEXBOL = "MEXBOL Index"
    IPSA = "IPSA Index"
    MERVAL = "MERVAL Index"
    
    # Middle East & Africa
    TADAWUL = "TADAWUL Index"
    ADX = "ADX Index"
    DFM = "DFM Index"
    QE = "QE Index"
    TOP40 = "TOP40 Index"
    
    # Global
    MXEA = "MXEA Index"
    SEMLMCUP = "SEMLMCUP Index"

    @property
    def description(self) -> str:
        """Returns the detailed description of the universe"""
        descriptions: Dict[Universe, str] = {
            # North America
            Universe.NDX: "NASDAQ-100 Index - Top 100 non-financial companies listed on NASDAQ",
            Universe.SPX: "S&P 500 Index - 500 largest US publicly traded companies",
            Universe.RTY: "Russell 2000 Index - 2000 small-cap US companies",
            Universe.SPTSX: "S&P/TSX Composite Index - Canadian equity market benchmark",
            Universe.INDU: "Dow Jones Industrial Average",
            
            # Europe
            Universe.SXXP: "STOXX Europe 600 - Companies across 17 European countries",
            Universe.UKX: "FTSE 100 Index - 100 largest companies listed on London Stock Exchange",
            Universe.CAC: "CAC 40 - Benchmark French stock market index of 40 largest equities",
            Universe.DAX: "DAX 40 - 40 major German blue chip companies trading on Frankfurt Exchange",
            Universe.IBEX: "IBEX 35 - Benchmark index of Spain's principal stock exchange",
            Universe.FTSEMIB: "FTSE MIB - 40 most traded stock classes on Italian Exchange",
            Universe.AEX: "AEX Index - 25 most traded Dutch companies on Amsterdam Exchange",
            Universe.SMI: "Swiss Market Index - 20 largest Swiss publicly traded companies",
            Universe.NDDUEMU: "MSCI EMU - Eurozone Economic and Monetary Union",
            
            # Asia Pacific
            Universe.NKY: "Nikkei 225 - Leading Japanese stock market index",
            Universe.HSI: "Hang Seng Index - Main indicator of Hong Kong market performance",
            Universe.SHSZ300: "CSI 300 Index - 300 largest A-share stocks in Shanghai and Shenzhen",
            Universe.AS51: "S&P/ASX 200 - Benchmark for Australian equity market",
            Universe.KOSPI: "Korea Composite Stock Price Index - Main benchmark of South Korea",
            Universe.NIFTY: "NIFTY 50 - Benchmark Indian National Stock Exchange index",
            Universe.STI: "Straits Times Index - Benchmark index for Singapore stock market",
            
            # Latin America
            Universe.IBOV: "Ibovespa - Benchmark index of Brazil's São Paulo Stock Exchange",
            Universe.MEXBOL: "S&P/BMV IPC - Main benchmark of Mexican Stock Exchange",
            Universe.IPSA: "S&P/CLX IPSA - Main stock market index of Chile",
            Universe.MERVAL: "S&P MERVAL - Main index of Buenos Aires Stock Exchange",
            
            # Middle East & Africa
            Universe.TADAWUL: "Tadawul All Share Index - Main index of Saudi Stock Exchange",
            Universe.ADX: "Abu Dhabi Securities Exchange General Index",
            Universe.DFM: "Dubai Financial Market General Index",
            Universe.QE: "Qatar Exchange Index - Main benchmark of Qatar Stock Exchange",
            Universe.TOP40: "FTSE/JSE Top 40 - Largest 40 companies on Johannesburg Stock Exchange",
            
            # Global
            Universe.MXEA: "The MSCI EAFE region covers DM countries in Europe, Australasia, Israel, and the Far East",
            Universe.SEMLMCUP: "Solactive GBS Emerging Markets Large & Mid Cap USD Index PR"
        }
        return descriptions[self]

    @classmethod
    def get_all_descriptions(cls) -> Dict[str, str]:
        """Returns a dictionary of all universes and their descriptions"""
        return {universe.name: universe.description for universe in cls}

    @classmethod
    def get_by_region(cls, region: str) -> Dict[str, str]:
        """Returns indices for a specific region
        Regions: 'North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa'
        """
        region_mapping = {
            'North America': [cls.NDX, cls.SPX, cls.RTY, cls.SPTSX, cls.INDU],
            'Europe': [cls.SXXP, cls.UKX, cls.CAC, cls.DAX, cls.IBEX, cls.FTSEMIB, cls.AEX, cls.SMI],
            'Asia Pacific': [cls.NKY, cls.HSI, cls.SHSZ300, cls.AS51, cls.KOSPI, cls.NIFTY, cls.STI],
            'Latin America': [cls.IBOV, cls.MEXBOL, cls.IPSA, cls.MERVAL],
            'Middle East & Africa': [cls.TADAWUL, cls.ADX, cls.DFM, cls.QE, cls.TOP40],
            'Global': [cls.MXEA, cls.SEMLMCUP]
        }
        return {index.name: index.description for index in region_mapping.get(region, [])}

class Currency(str, Enum):
    """Available currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"

class Frequency(str, Enum):
    """Rebalancing frequency options"""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    CUSTOM = "C"
   
class RiskFactors(str, Enum):
    """Risk factors for factor investing and portfolio analysis"""
    BETA = "beta"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    PROFIT = "profit"
    QUALITY = "quality"
    LOW_VOL = "low_vol"
    GROWTH = "growth"
    DIVIDEND = "dividend"
    LIQUIDITY = "liquidity"
    SHORT_INTEREST = "short_interest"
    LEVERAGE = "leverage"
    EARNINGS_YIELD = "earnings_yield"
    BAB = "betting_against_beta"
    CROWDING = "crowding"
    ST_REVERSAL = "short_term_reversal"
    LT_REVERSAL = "long_term_reversal"
    CARRY = "carry"
    VOL_RISK_PREMIUM = "volatility_risk_premium"
    SEASONALITY = "seasonality"
    ANALYST_SENTIMENT = "analyst_sentiment"
    ESG = "esg"

    @property
    def description(self) -> str:
        """Returns detailed description and common implementation of the risk factor"""
        descriptions: Dict[RiskFactors, str] = {
            RiskFactors.BETA: "Market sensitivity measure (CAPM beta). Implementation: Rolling regression of excess returns against market excess returns.",
            RiskFactors.SIZE: "Market capitalization factor (SMB). Implementation: Natural log of market cap, ranked and normalized.",
            RiskFactors.VALUE: "Book-to-market ratio (HML). Implementation: Book value divided by market cap, industry-adjusted.",
            RiskFactors.MOMENTUM: "12-1 month price momentum (UMD). Implementation: Cumulative returns excluding most recent month.",
            RiskFactors.PROFIT: "Profitability premium (RMW). Implementation: Gross profits to assets or ROE, industry-adjusted.",
            RiskFactors.QUALITY: "Multi-metric quality score. Implementation: Composite of ROE, earnings stability, leverage, and accruals.",
            RiskFactors.LOW_VOL: "Low volatility anomaly. Implementation: 1-year rolling standard deviation of daily returns.",
            RiskFactors.GROWTH: "Sales/earnings growth. Implementation: 3-5 year CAGR of revenue or earnings.",
            RiskFactors.DIVIDEND: "Dividend yield factor. Implementation: Trailing 12-month dividends divided by price.",
            RiskFactors.LIQUIDITY: "Trading liquidity measure. Implementation: Average daily volume divided by shares outstanding.",
            RiskFactors.SHORT_INTEREST: "Short selling pressure. Implementation: Short interest ratio or days to cover.",
            RiskFactors.LEVERAGE: "Financial leverage. Implementation: Debt-to-equity or debt-to-assets ratio.",
            RiskFactors.EARNINGS_YIELD: "E/P ratio factor. Implementation: Forward earnings divided by price, sector-adjusted.",
            RiskFactors.BAB: "Betting Against Beta. Implementation: Long low-beta assets, short high-beta assets with leverage constraints.",
            RiskFactors.CROWDING: "Crowding/popularity measure. Implementation: Composite of ownership concentration and trading volume.",
            RiskFactors.ST_REVERSAL: "Short-term reversal. Implementation: Negative of past month return.",
            RiskFactors.LT_REVERSAL: "Long-term reversal. Implementation: Negative of past 3-5 year returns.",
            RiskFactors.CARRY: "Carry factor. Implementation: Interest rate differential or futures basis.",
            RiskFactors.VOL_RISK_PREMIUM: "Volatility risk premium. Implementation: Implied minus realized volatility.",
            RiskFactors.SEASONALITY: "Calendar anomalies. Implementation: Historical average returns by time period.",
            RiskFactors.ANALYST_SENTIMENT: "Analyst revisions/sentiment. Implementation: Change in consensus estimates.",
            RiskFactors.ESG: "Environmental, Social, Governance. Implementation: Third-party ESG scores or composite metrics."
        }
        return descriptions[self]

    @classmethod
    def get_all_descriptions(cls) -> Dict[str, str]:
        """Returns all risk factors and their descriptions"""
        return {factor.name: factor.description for factor in cls}

    @property
    def category(self) -> str:
        """Returns the broad category of the risk factor"""
        categories: Dict[RiskFactors, str] = {
            RiskFactors.BETA: "Market",
            RiskFactors.SIZE: "Market",
            RiskFactors.VALUE: "Fundamental",
            RiskFactors.MOMENTUM: "Technical",
            RiskFactors.PROFIT: "Fundamental",
            RiskFactors.QUALITY: "Fundamental",
            RiskFactors.LOW_VOL: "Market",
            RiskFactors.GROWTH: "Fundamental",
            RiskFactors.DIVIDEND: "Fundamental",
            RiskFactors.LIQUIDITY: "Market",
            RiskFactors.SHORT_INTEREST: "Sentiment",
            RiskFactors.LEVERAGE: "Fundamental",
            RiskFactors.EARNINGS_YIELD: "Fundamental",
            RiskFactors.BAB: "Market",
            RiskFactors.CROWDING: "Sentiment",
            RiskFactors.ST_REVERSAL: "Technical",
            RiskFactors.LT_REVERSAL: "Technical",
            RiskFactors.CARRY: "Market",
            RiskFactors.VOL_RISK_PREMIUM: "Market",
            RiskFactors.SEASONALITY: "Technical",
            RiskFactors.ANALYST_SENTIMENT: "Sentiment",
            RiskFactors.ESG: "Fundamental"
        }
        return categories[self]

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, str]:
        """Returns factors for a specific category
        Categories: 'Market', 'Fundamental', 'Technical', 'Sentiment'
        """
        return {factor.name: factor.description 
                for factor in cls 
                if factor.category == category}

class RegimeType(str, Enum):
    """Types of regime analysis"""
    VOLATILITY = "vol"
    CORRELATION = "corr"
    MACRO = "macro"
    FACTOR = "factor"

class VolatilityType(str, Enum):
    """Volatility and Fixed Income benchmark indices"""
    # Equity Volatility
    VIX = "VIX Index"
    VSTOXX = "V2X Index"
    VVIX = "VVXN Index"
    MOVE = "MOVE Index"
    TYVIX = "TYVIX Index"
    RVX = "RVX Index"
    VXFXI = "VXFXI Index"
    JNIV = "JNIV Index"
    
    # Currency Volatility
    CVIX = "CVIX Index"
    EUVIX = "EUVIX Index"
    JYVIX = "JYVIX Index"
    
    # Commodity Volatility
    OVX = "OVX Index"
    GVZ = "GVZ Index"
    EVZ = "EVZ Index"

    @property
    def description(self) -> str:
        """Returns detailed description of the volatility benchmark"""
        descriptions: Dict[VolatilityType, str] = {
            # Equity Volatility
            VolatilityType.VIX: "CBOE Volatility Index - Measures implied volatility of S&P 500 index options, 30-day horizon",
            VolatilityType.VSTOXX: "Euro Stoxx 50 Volatility Index - European equivalent of VIX, measures implied volatility of EURO STOXX 50",
            VolatilityType.VVIX: "VIX Volatility Index - Measures volatility of VIX options, also known as 'vol of vol'",
            VolatilityType.MOVE: "Merrill Lynch Option Volatility Estimate - Measures implied volatility of U.S. Treasury options",
            VolatilityType.TYVIX: "CBOE/CBOT 10-year U.S. Treasury Note Volatility Index - Measures treasury yield volatility",
            VolatilityType.RVX: "CBOE Russell 2000 Volatility Index - Measures small-cap equity volatility",
            VolatilityType.VXFXI: "CBOE China ETF Volatility Index - Measures implied volatility of iShares China Large-Cap ETF",
            VolatilityType.JNIV: "Nikkei Stock Average Volatility Index - Measures implied volatility of Nikkei 225 options",
            
            # Currency Volatility
            VolatilityType.CVIX: "Deutsche Bank Currency Volatility Index - Measures implied volatility of G7 currencies",
            VolatilityType.EUVIX: "CBOE Euro Volatility Index - Measures implied volatility of EUR/USD options",
            VolatilityType.JYVIX: "CBOE Japanese Yen Volatility Index - Measures implied volatility of JPY/USD options",
            
            # Commodity Volatility
            VolatilityType.OVX: "CBOE Crude Oil Volatility Index - Measures implied volatility of USO options",
            VolatilityType.GVZ: "CBOE Gold Volatility Index - Measures implied volatility of GLD options",
            VolatilityType.EVZ: "CBOE EuroCurrency Volatility Index - Measures implied volatility of FXE options"
        }
        return descriptions[self]

    @property
    def asset_class(self) -> str:
        """Returns the asset class category of the volatility benchmark"""
        categories: Dict[VolatilityType, str] = {
            VolatilityType.VIX: "Equity",
            VolatilityType.VSTOXX: "Equity",
            VolatilityType.VVIX: "Equity",
            VolatilityType.MOVE: "Fixed Income",
            VolatilityType.TYVIX: "Fixed Income",
            VolatilityType.RVX: "Equity",
            VolatilityType.VXFXI: "Equity",
            VolatilityType.JNIV: "Equity",
            VolatilityType.CVIX: "Currency",
            VolatilityType.EUVIX: "Currency",
            VolatilityType.JYVIX: "Currency",
            VolatilityType.OVX: "Commodity",
            VolatilityType.GVZ: "Commodity",
            VolatilityType.EVZ: "Currency"
        }
        return categories[self]

    @classmethod
    def get_by_asset_class(cls, asset_class: str) -> Dict[str, str]:
        """Returns benchmarks for a specific asset class
        Asset Classes: 'Equity', 'Fixed Income', 'Currency', 'Commodity'
        """
        return {bench.name: bench.description 
                for bench in cls 
                if bench.asset_class == asset_class}

class OptimizationObjective(str, Enum):
    """Optimization objective functions"""
    PURE_FACTOR = "pfactor"
    TRACKING_ERROR = "te"
    NUM_TRADES = "n_trades"
    TRANSACTION_COST = "tcost"
    RISK_PARITY = "risk_parity"

class WeightingScheme(str, Enum):
    """Portfolio weighting schemes"""
    EQUAL_WEIGHT = "eq_wgt"
    MARKET_CAP = "mcap_wgt"
    RISK_WEIGHT = "risk_wgt"
    CUSTOM = "custom"

class ParamsConfig(BaseModel):
    """Configuration for general parameters"""
    aum: Decimal = Field(gt=0, description="Assets under management in millions of local currency")
    sigma_regimes: bool = Field(default=False, description="Whether to use regime-dependent covariances")
    risk_factors: List[RiskFactors] = Field(
        default_factory=lambda: [RiskFactors.BETA, RiskFactors.SIZE, RiskFactors.MOMENTUM, RiskFactors.VALUE],
        description="List of risk factors to include in the model"
    )
    bench_weights: Optional[WeightingScheme] = Field(
        default=None,
        description="Benchmark weighting scheme"
    )
    n_dates: Optional[int] = Field(default=None,
                                  description="Number of dates in backtest")
    n_sids: Optional[int] = Field(default=None,
                                  description="Number of securities in backtest")
    n_buckets: Optional[int] = Field(default=None,
                                  description="Number of buckets for percentile portfolios.")    
    @field_validator('risk_factors')
    def sort_risk_factors(cls, v):
        return sorted(v)

class BacktestConfig(BaseModel):
    """Configuration for backtest parameters"""
    data_source: str = Field(
        default=DataSource.YAHOO, # 'yahoo',
        alias='data_source',
        description="Source to retrieve data from, i.e. Yahoo, Bloomberg, Reuters..."
    )
    universe: Universe = Field(
        default=Universe.NDX,
        description="Investment universe for the backtest"
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Base currency for calculations"
    )
    frequency: Frequency = Field(
        default=Frequency.MONTHLY,
        alias='frq',
        description="Rebalancing frequency"
    )
    start_date: date = Field(
        # default= pd.Timestamp.now('US/Eastern').date(),
        default = pd.to_datetime('2017-12-31').date(),
        alias='start',
        description="Backtest start date"
    )
    end_date: date = Field(
        default= pd.Timestamp.now('US/Eastern').date(),
        alias='end',
        description="Backtest end date"
    )
    time_zone: str = Field(
        default='US/Eastern',
        alias='time_zone',
        description="Time zone for analysis"
    )
    dates_daily: Optional[list] = Field(
        default=None,
        alias='dates_daily',
        description="Daily business dates list"
    )
    dates_turnover: Optional[list] = Field(
        default=None,
        alias='dates_turnover',
        description="Turnover dates list"
    )
    universe_list: Optional[list] = Field(
        default=None,
        alias='univ_list',
        description="Securities ids list"
    )
    portfolio_list: Optional[list] = Field(
        default=None,
        alias='portfolio_list',
        description="Portfolio's securities ids list"
    )
    concurrent_download: bool = Field(
        default=False,
        description="If True, use asyncio for concurrent downloads from Yahoo Finance."
    )
    # @field_validator('end_date')f
    # def end_date_must_be_after_start_date(cls, v, values):
    #     if 'start_date' in values and v <= values['start_date']:
    #         raise ValueError('end_date must be after start_date')
    #     return v

class RegimeConfig(BaseModel):
    """Configuration for regime analysis"""
    type: RegimeType = Field(
        default=RegimeType.VOLATILITY,
        description="Type of regime analysis"
    )
    benchmark: VolatilityType = Field(
        default=VolatilityType.VIX,
        description="Benchmark for regime detection"
    )
    periods: int = Field(
        gt=0,
        le=252,
        description="Number of periods for regime calculation"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Optional threshold for regime classification"
    )

class OptimizationConfig(BaseModel):
    """Configuration for portfolio optimization"""
    objective: OptimizationObjective = Field(
        alias='obj',
        description="Optimization objective function"
    )
    num_trades: int = Field(
        alias='n_trades',
        gt=0,
        description="Target number of trades"
    )
    tracking_error_max: float = Field(
        alias='te_max',
        gt=0,
        le=1,
        description="Maximum tracking error constraint"
    )
    weight_max: float = Field(
        alias='w_max',
        gt=0,
        le=1,
        description="Maximum position weight constraint"
    )
    factor_constraints: Dict[str, float] = Field(
        default_factory=dict,
        alias='factors',
        description="Factor exposure constraints"
    )
    pure_factor: Optional[str] = Field(
        alias='pfactor',
        description="Pure factor portfolio target"
    )
    min_holding: Optional[float] = Field(
        default=0.001,
        description="Minimum holding size"
    )
    sector_constraints: Optional[Dict[str, tuple]] = Field(
        default=None,
        description="Sector exposure constraints as (min, max) tuples"
    )

class ExportConfig(BaseModel):
    """Configuration for exporting files / data"""
    update_history: bool = False,
    base_path: str = Field(
        default="../data/output",
        description="Path to export data locally"
    )
    s3_config: Optional[dict] = Field(
        default={
            'bucket_name': os.environ.get('BUCKET_NAME'),
            'user_name': os.environ.get('USER_NAME')
        },
        description="S3 bucket config"
    )
    
class EquityFactorModelInput(BaseModel):
    """Main configuration class for equity factor model"""
    params: ParamsConfig
    backtest: BacktestConfig
    regime: RegimeConfig
    # optimization: OptimizationConfig = Field(alias='opt')
    export: ExportConfig
    
    class Config:
        populate_by_name = True
        # allow_population_by_field_name = True
        json_encoders = {
            date: lambda v: v.isoformat()
        }

"""
# Define security master and factor classes, & functions
"""
def update_values_based_on_turnover_vectorized(df, turnover_dates):
    """
    Update the 'value' column in the DataFrame based on turnover dates using
    vectorized operations for better performance with large datasets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'date' and 'value' columns
    turnover_dates : list
        List of dates when portfolio turnover occurs
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated 'value' column
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Convert dates to datetime format
    result['date'] = pd.to_datetime(result['date'])
    turnover_dates = pd.to_datetime(turnover_dates)
    
    # Create a marker for rows that correspond to turnover dates
    result['is_turnover'] = result['date'].isin(turnover_dates)
    
    # Identify grouping columns
    groupby_cols = [col for col in ['sid', 'factor_name'] if col in result.columns]
    
    # If no grouping columns exist, create a temporary one
    if not groupby_cols:
        result['_temp_group'] = 1
        groupby_cols = ['_temp_group']
    
    # Sort by date within each group
    result = result.sort_values(by=groupby_cols + ['date'])
    
    # Create a group identifier
    if len(groupby_cols) > 1:
        # If multiple grouping columns, create a compound key
        result['_group_id'] = result[groupby_cols].apply(lambda x: tuple(x), axis=1)
    else:
        # If only one grouping column, use it directly
        result['_group_id'] = result[groupby_cols[0]]
    
    # Create a series that will help us identify the last turnover value for each date
    # First, create a helper column with NaN for non-turnover dates
    result['turnover_value'] = np.where(result['is_turnover'], result['value'], np.nan)
    
    # Now use groupby + transform to forward fill these values
    result['last_turnover_value'] = result.groupby('_group_id')['turnover_value'].transform(
        lambda x: x.ffill() # x.fillna(method='ffill')
    )
    
    # Update the 'value' column for non-turnover dates
    result['value'] = np.where(
        result['is_turnover'],
        result['value'],
        result['last_turnover_value']
    )
    
    # Drop temporary columns
    columns_to_drop = ['is_turnover', '_group_id', 'turnover_value', 'last_turnover_value']
    if '_temp_group' in result.columns:
        columns_to_drop.append('_temp_group')
    
    result = result.drop(columns=columns_to_drop)
    
    return result

def get_universe_mapping_yahoo(universe):
    if universe in ['NDX Index', '^NDX']:
        universe= '^NDX' if universe=='NDX Index' else 'NDX Index'
    elif universe in ['INDU Index', '^DJI']:
        universe='^DJI' if universe=='INDU Index' else 'INDU Index'
    elif universe in ['SML Index', '^SP600']:
        universe='^SP600' if universe=='SML Index' else 'SML Index'
    elif universe in ['SPX Index', '^SPX']:
        universe='^SPX' if universe=='SPX Index' else 'SPX Index'
    else:
        print("Universe not supported.")
    return universe

def UniverseMappingFactory(source:str='yahoo', universe:str=None) -> str:
    if source=='yahoo':
        return get_universe_mapping_yahoo(universe)
    else:
        return universe

class SecurityMasterBloomberg(BaseModel):
    """
    SecurityMaster class for handling benchmark constituents and security master data
    """
    source: str = 'bloomberg'
    universe: str
    dates: List[str] = []
    dates_turnover: List[str] = [] # List of turnover dates in 'YYYY-MM-DD' format
    bq: object  # Bloomberg Query instance
    df_bench: pd.DataFrame = pd.DataFrame()
    df_price: pd.DataFrame = pd.DataFrame()
    weights_data: Optional[pd.DataFrame] = None
    security_master: Optional[pd.DataFrame] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get_benchmark_weights(self) -> pd.DataFrame:
        """
        Get benchmark constituent weights over time
        
        Returns:
            DataFrame with benchmark weights
        """
        try:
            # Define data items for BQL query
            import bql
            data_items = {
                'Name': self.bq.data.name()['VALUE'],
                'Weights': self.bq.data.id()['WEIGHTS']
            }
            
            df_weights = pd.DataFrame()
            for idate in self.dates_turnover:
                # Get universe members for each date
                universe = self.bq.univ.members(self.universe, dates=idate)
                request = bql.Request(universe, data_items)
                response = self.bq.execute(request)
                
                # Process response
                df = pd.concat([data_item.df() for data_item in response], axis=1)
                df.reset_index(drop=False, inplace=True)
                df.insert(0, 'date', idate)
                df.columns = ['date', 'sid', 'name', 'weight']
                df_weights = pd.concat([df_weights, df])
            
            # Format dates and weights
            # df_weights = df_weights.set_index('date')
            df_weights.index = df_weights['date']
            df_weights.index.name = 'index'
            # df_weights['date'] = df_weights['date'].astype(str)    
            # df_weights['date'] = df_weights['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)    
            df_weights['weight'] /= 100
            # df_weights.reset_index(drop=True, inplace=True)
            
            self.weights_data = df_weights
            return self.weights_data
            
        except Exception as e:
            raise Exception(f"Error getting benchmark weights: {str(e)}")
    
    def get_security_master(self, sector_classification: str = 'BICS') -> pd.DataFrame:
        """
        Get security master data including sector classifications
        
        Args:
            sector_classification: 'BICS' or 'GICS'
            
        Returns:
            DataFrame with security master data
        """
        try:
            import bql
            if self.weights_data is None:
                self.weights_data = self.get_benchmark_weights()
                
            univ_list = list(self.weights_data.sid.unique())
            
            # Get sector classification
            if sector_classification.upper() == 'BICS':
                sector = self.bq.data.bics_level_1_sector_name()
            elif sector_classification.upper() == 'GICS':
                sector = self.bq.data.gics_sector_name()
            else:
                raise ValueError("sector_classification must be either 'BICS' or 'GICS'")
            
            # Get sector data
            request = {'sector': sector}
            req = bql.Request(univ_list, request)
            res = self.bq.execute(req)
            sec_map = pd.DataFrame({r.name: r.df()[r.name] for r in res})
            
            # Create dummy variables for sectors
            df_sec = pd.get_dummies(sec_map)
            
            # Create security master
            sec_master = self.weights_data.copy()
            sec_master.insert(2, 'id', sec_master['sid'])
            sec_master['sid'] = sec_master['sid']
            sec_master.insert(1, 'yy', sec_master['date'].map(lambda x: x[:4]))
            
            # Merge with sector data
            sec_master = sec_master.merge(
                sec_map.reset_index(drop=False).rename(columns={'ID': 'id'}),
                how='left',
                on=['id']
            )
            
            # Format dates and sort
            sec_master.sort_values(['date', 'sid'], inplace=True)
            sec_master = sec_master.set_index('date')
            # sec_master.index = sec_master['date']
            # sec_master.index.name = 'index' 
            # sec_master['date'] = sec_master['date'].astype(str)
            # sec_master['date'] = sec_master['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)
            # sec_master.reset_index(drop=True, inplace=True)
            
            self.security_master = sec_master
            return self.security_master
            
        except Exception as e:
            raise Exception(f"Error getting security master: {str(e)}")
    
    def get_universe_stats(self) -> Dict:
        """
        Get basic statistics about the universe
        
        Returns:
            Dictionary with universe statistics
        """
        if self.weights_data is None:
            self.weights_data = self.get_benchmark_weights()
        df_stats = {
            'n_dates': len(self.dates),
            'n_securities': len(self.weights_data.sid.unique()),
            'avg_weight_sum': self.weights_data.groupby('date')['weight'].sum().mean(),
            'dates_range': f"{min(self.dates)} to {max(self.dates)}"
        }
        
        if self.security_master is not None:
            df_stats['sectors'] = self.security_master.sector.unique().tolist()
            df_stats['sectors_count'] = len(df_stats['sectors'])
            
        return df_stats
    
    def get_sector_weights(self) -> pd.DataFrame:
        """
        Get sector weights over time
        
        Returns:
            DataFrame with sector weights
        """
        if self.security_master is None:
            self.security_master = self.get_security_master()
            
        sector_weights = (self.security_master
                         .groupby(['date', 'sector'])['weight']
                         .sum()
                         .unstack()
                         .fillna(0))
        
        return sector_weights

    def get_benchmark_prices(self, model_input) -> pd.DataFrame:
        if model_input.backtest.universe is not None:
            if self.df_bench.shape[0]==0:
                df_bench = self._get_prices(model_input, univ_list = [model_input.backtest.universe.value])
                self.df_bench = df_bench
            return self.df_bench
            
    def get_members_prices(self, model_input) -> pd.DataFrame:
        if model_input.backtest.universe_list is not None:
            if self.df_price.shape[0]==0:
                df_price = self._get_prices(model_input, univ_list = model_input.backtest.universe_list)
                self.df_price = df_price
            return self.df_price
        
    def _get_prices(self, model_input, univ_list:list=[]) -> pd.DataFrame:
        start = str(model_input.backtest.start_date)
        end = str(model_input.backtest.end_date)
        if univ_list==[]:
            univ_list = model_input.backtest.universe_list
        if type(univ_list) != list:
            univ_list = list(univ_list)
        # get pricing data
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        px_last = bq.data.px_last(dates= bq.func.range(start, end), frq='d', ca_adj='full').dropna() # fill='prev',
        px_open = bq.data.px_open(dates= bq.func.range(start, end), frq='d', ca_adj='full').dropna()
        px_high = bq.data.px_high(dates= bq.func.range(start, end), frq='d', ca_adj='full').dropna()
        px_low = bq.data.px_low(dates= bq.func.range(start, end), frq='d', ca_adj='full').dropna()
        volume = bq.data.px_volume(dates= bq.func.range(start, end), frq='d', ca_adj='full').dropna()
        ret = bq.data.day_to_day_tot_return_gross_dvds(start, end).dropna()
        size = bq.data.market_cap(dates= bq.func.range(start, end)).log()
        request = {'price':px_last, 'ret':ret, 'size':size, 'p_high':px_high, 'p_low':px_low}
        req = bql.Request(univ_list, request)
        res = bq.execute(req)

        df_price = pd.concat([x.df()[['DATE', x.name]].reset_index().set_index(['DATE', 'ID'])  for x in res], axis=1)
        df_price = df_price.dropna(subset=['price'])
        # import pdb; pdb.set_trace()
        df_price.reset_index(drop=False, inplace=True)
        df_price.rename(columns={'DATE':'date','ID':'sid','ret':'return'}, inplace=True)
        # df_price.index = df_price['date']
        # df_price.index.name = 'index' 
        df_price = df_price.set_index('date')
        return df_price
    
    def get_returns_long(self) -> pd.DataFrame:
        # get returns long format 
        df_ret_long = self.df_price[['date','sid','return','price']].copy() # print(df_ret_long.tail(3))
        df_ret_long['price'] = df_ret_long.groupby('sid').ffill()['price']
        df_ret_long['return'] = df_ret_long['return'].fillna(0.0)
        return df_ret_long

    def get_returns_wide(self) -> pd.DataFrame:
        # get returns wide format
        df_ret_wide = self.df_price[['date','sid','return']].pivot(index='date', columns='sid', values='return')
        df_ret_wide.fillna(0., inplace=True)
        return df_ret_wide

class SecurityMasterYahoo(BaseModel):
    """
    SecurityMaster class using Yahoo Finance API for handling benchmark constituents and security master data.
    """
    source: str = 'yahoo'
    universe: str  # Universe (e.g., '^GSPC' for S&P 500, '^NDX' for Nasdaq 100)
    dates: List[str] = []  # List of dates in 'YYYY-MM-DD' format
    dates_turnover: List[str] = [] # List of turnover dates in 'YYYY-MM-DD' format
    df_bench: pd.DataFrame = pd.DataFrame()
    df_price: pd.DataFrame = pd.DataFrame()
    weights_data: Optional[pd.DataFrame] = None
    security_master: Optional[pd.DataFrame] = None
    df_portfolio: pd.DataFrame = pd.DataFrame()
    concurrent_download: Optional[bool] = None
    model_input: Optional[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # def get_universe_mapping(self):
    #     if self.universe=='NDX Index':
    #         self.universe='^NDX'
    #     else:
    #         self.universe='^SPX'

    def _get_ticker_components(self, ticker: str, date: str) -> pd.DataFrame: # Optional[List[str] | None]
        """
        Helper function to get ticker components for a given date from Yahoo Finance.
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            # Fetch history around the date to handle non-trading days
            history = ticker_obj.history(start=date, end=pd.to_datetime(date) + pd.Timedelta(days=1))

            # Yahoo Finance now provides a more reliable way to get constituents.
            if 'components' in ticker_obj.info and ticker_obj.info['components'] is not None:
                components = [comp['symbol'] for comp in ticker_obj.info['components']] #extract the symbols
                if len(components) == 0:
                    return pd.DataFrame()
                return pd.DataFrame(components)
            elif ticker in ['^GSPC', '^DJI', '^IXIC', '^NDX', '^SP600']: # Handle S&P 500, Dow, Nasdaq (composite and 100) explicitly.
                if history.empty:
                  return pd.DataFrame() # []
                # "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
                # constituents_df = pd.read_html(
                #     f"https://en.wikipedia.org/wiki/List_of_{ticker[1:].lower()}_companies"
                # )[0]  # Select the first table, which usually holds the constituent list
                if ticker == '^GSPC':
                    constituents_df = get_index_constituents(ticker)
                        
                    # url_name = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                    # # constituents_df = pd.read_html(url_name)[0] # TO DO: check if this is the correct table
                    # req = urllib.request.Request(
                    #     url_name, 
                    #     headers={'User-Agent': 'Mozilla/5.0'}
                    # )
                    
                    # # 2. Open the request and pass the FILE-LIKE RESPONSE to pandas
                    # # We use a 'with' statement to ensure it's closed properly
                    # with urllib.request.urlopen(req) as response:
                    #     constituents_df_list = pd.read_html(response)

                    # 3. Select the correct table from the list
                    # (The components table is the second one on the page, index [1])
                    # if constituents_df_list:
                    #     constituents_df = constituents_df_list[0]
                    if not constituents_df.empty:
                        print(constituents_df.head())
                        constituents_df.columns = [
                            'ticker','name','sector','sub_industry','location','inclusion',
                            'cik','founded']
                        constituents_df.insert(0,'date',date)
                        return constituents_df
                    else:
                        print("No tables found.")
                        return pd.DataFrame()

                    constituents_df.columns = ['ticker','name','sector','sub_industry','location','inclusion',
                                               'cik','founded']
                    constituents_df.insert(0,'date',date)
                    return constituents_df
                    # if 'sector' in constituents_df.columns:
                    #     self.security_master = constituents_df.copy()
                    # return list(constituents_df['ticker'])
                elif ticker == '^DJI':
                    # url_name = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components"
                    # # 1. Create the request object (same as before, to fix the 403 error)
                    # req = urllib.request.Request(
                    #     url_name, 
                    #     headers={'User-Agent': 'Mozilla/5.0'}
                    # )
                    
                    # 2. Open the request and pass the FILE-LIKE RESPONSE to pandas
                    # We use a 'with' statement to ensure it's closed properly
                    # with urllib.request.urlopen(req) as response:
                    #     constituents_df_list = pd.read_html(response)

                    # 3. Select the correct table from the list
                    # (The components table is the third one on the page, index [2])
                    # if constituents_df_list:
                    #     constituents_df = constituents_df_list[2]
                    constituents_df = get_index_constituents(ticker)
                    if not constituents_df.empty:
                        print(constituents_df.head())
                        constituents_df.columns = [
                            'name','exchange','ticker','sector','inclusion',
                            'notes','weight']
                        constituents_df.insert(0,'date',date)
                        return constituents_df
                    else:
                        print("No tables found.")
                        return pd.DataFrame()

                    # constituents_df = pd.read_html(url_name)[2]
                    # if 'sector' in constituents_df.columns:
                    #     self.security_master = constituents_df.copy()
                    # return list(constituents_df['ticker'])
                elif ticker == '^IXIC':
                    url_name = "http://en.wikipedia.org/wiki/Nasdaq-100#Components"
                    # constituents_df = pd.read_html(url_name)[4]
                    req = urllib.request.Request(
                        url_name, 
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    
                    # 2. Open the request and pass the FILE-LIKE RESPONSE to pandas
                    # We use a 'with' statement to ensure it's closed properly
                    with urllib.request.urlopen(req) as response:
                        constituents_df_list = pd.read_html(response)

                    # 3. Select the correct table from the list
                    # (The components table is the fifth one on the page, index [4])
                    if constituents_df_list:
                        constituents_df = constituents_df_list[4]
                        print(constituents_df.head())
                        # constituents_df.columns = ['name','ticker','sector','sub_industry']
                        constituents_df.columns = ['ticker','name','sector','sub_industry']
                        constituents_df.insert(0,'date',date)
                        return constituents_df
                    else:
                        print("No tables found.")
                        return pd.DataFrame()

                    # constituents_df.columns = ['name','ticker','sector','sub_industry']
                    # constituents_df.insert(0,'date',date)
                    # return constituents_df
                    # if 'sector' in constituents_df.columns:
                    #     self.security_master = constituents_df.copy()
                    # return list(constituents_df['ticker'])
                elif ticker == '^NDX':
                    # url_name = "http://en.wikipedia.org/wiki/Nasdaq-100#Components"
                    # constituents_df = pd.read_html(url_name)[4]
                    # req = urllib.request.Request(
                    #     url_name, 
                    #     headers={'User-Agent': 'Mozilla/5.0'}
                    # )
                    
                    # # 2. Open the request and pass the FILE-LIKE RESPONSE to pandas
                    # # We use a 'with' statement to ensure it's closed properly
                    # with urllib.request.urlopen(req) as response:
                    #     constituents_df_list = pd.read_html(response)

                    # # 3. Select the correct table from the list
                    # # (The components table is the fifth one on the page, index [4])
                    # if constituents_df_list:
                    #     constituents_df = constituents_df_list[4]
                    constituents_df = get_index_constituents(ticker)
                    if not constituents_df.empty:
                        print(constituents_df.head())
                        # constituents_df.columns = ['name','ticker','sector','sub_industry']
                        constituents_df.columns = ['ticker','name','sector','sub_industry']
                        constituents_df.insert(0,'date',date)
                        return constituents_df
                    else:
                        print("No tables found.")
                        return pd.DataFrame()
                elif ticker == '^SP600':
                    url_name = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
                    req = urllib.request.Request(
                        url_name, 
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    
                    # 2. Open the request and pass the FILE-LIKE RESPONSE to pandas
                    # We use a 'with' statement to ensure it's closed properly
                    with urllib.request.urlopen(req) as response:
                        constituents_df_list = pd.read_html(response)

                    # 3. Select the correct table from the list
                    # (The components table is the first one on the page, index [0])
                    if constituents_df_list:
                        constituents_df = constituents_df_list[0]
                        print(constituents_df.head())
                        # constituents_df.columns = ['name','ticker','sector','sub_industry']
                        constituents_df.columns = [
                            'ticker','name','sector','sub_industry','location',
                            'sec_filings','cik']
                        constituents_df.insert(0,'date',date)
                        return constituents_df
                    else:
                        print("No tables found.")
                        return pd.DataFrame()

                    # constituents_df = pd.read_html(url_name)[0]
                    # constituents_df.columns = ['ticker','name','sector','sub_industry','location',
                    #                            'sec_filings','cik']
                    # constituents_df.insert(0,'date',date)
                    # return constituents_df
            else:
                return pd.DataFrame() # []

        except Exception as e:
            print(f"Error getting ticker components for {ticker} on {date}: {e}")
            return pd.DataFrame() # []

    def get_benchmark_weights(self) -> pd.DataFrame:
        """
        Get benchmark constituent weights over time (approximation using Yahoo Finance components).
        Note: Yahoo Finance doesn't directly provide weights; this is an approximation.
        """
        import asyncio
        import concurrent.futures
        import sys
        import time
        import pandas as pd

        concurrent_download = getattr(self, 'concurrent_download', False)
        df_weights = pd.DataFrame()

        def get_components_for_date(idate):
            time.sleep(1)
            components = self._get_ticker_components(self.universe, idate)
            if components.shape[0] > 0:
                df = components.copy()
                if 'weight' in components.columns:
                    df['wgt'] = df['weight'].str.replace('%', '').astype(float)
                    if df['wgt'].sum() > 1:
                        df['wgt'] /= df['wgt'].sum()
                else:
                    df['wgt'] = 1 / len(components)
                return df
            return None

        if concurrent_download:
            async def gather_all():
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    tasks = [loop.run_in_executor(executor, get_components_for_date, idate) for idate in self.dates_turnover]
                    results = await asyncio.gather(*tasks)
                return [df for df in results if df is not None]

            if sys.version_info >= (3, 7):
                try:
                    results = asyncio.run(gather_all())
                except RuntimeError:
                    results = asyncio.get_event_loop().run_until_complete(gather_all())
            else:
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(gather_all())
            if results:
                df_weights = pd.concat(results)
        else:
            for idate in self.dates_turnover:
                df = get_components_for_date(idate)
                if df is not None:
                    df_weights = pd.concat([df_weights, df])

        if not df_weights.empty:
            if 'sid' not in df_weights.columns:
                df_weights['sid'] = df_weights['ticker']
            # df_weights = df_weights.set_index('date')
            df_weights.index = df_weights['date']
            df_weights.index.name = 'index'
            df_weights.rename(columns={'wgt':'weight','weight':'wgt'}, inplace=True)
            # TO DO: CHECK DATAFRAME COLUMNS
            # df_weights.columns = ['date','ticker','name','sector','sub_industry','weight','sid','universe_id']
            df_weights['universe_id'] = UniverseMappingFactory(source='yahoo', universe=self.universe)
            self.weights_data = df_weights
        else:
            self.weights_data = pd.DataFrame()

        return self.weights_data

    def get_security_master(self, sector_classification: str = 'GICS') -> pd.DataFrame:
        """
        Get security master data including sector classifications.
        Note: Yahoo Finance provides sector information via ticker.info.
        """
        if self.weights_data is None or self.weights_data.empty:
            self.weights_data = self.get_benchmark_weights()
        if self.weights_data.empty:
            return pd.DataFrame()

        univ_list = list(self.weights_data.sid.unique())
        sec_master = self.weights_data.copy()
        # sec_master.insert(2, 'id', sec_master['sid'])
        # sec_master['sid'] = sec_master['sid']
        sec_master.insert(1, 'yy', sec_master['date'].map(lambda x: x[:4]))

        if 'sector' not in sec_master.columns:
            sector_data = []
            for ticker in univ_list:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    if 'sector' in info:
                        sector_data.append({'id': ticker, 'sector': info['sector']})
                    else:
                        sector_data.append({'id': ticker, 'sector': 'Unknown'})
                except Exception as e:
                    print(f"Error getting sector data for {ticker}: {e}")
                    sector_data.append({'id': ticker, 'sector': 'Unknown'})

            sec_map = pd.DataFrame(sector_data)
            sec_master = sec_master.merge(sec_map, on='id', how='left')

        sec_master.sort_values(['date', 'sid'], inplace=True)
        sec_master = sec_master.set_index('date')
        # sec_master.index = sec_master['date']
        # sec_master.index.name = 'index'

        self.security_master = sec_master
        return self.security_master

    def get_universe_stats(self) -> Dict:
        """
        Get basic statistics about the universe.
        """
        if self.weights_data is None:
            self.weights_data = self.get_benchmark_weights()

        df_stats = {
            'n_dates': len(self.dates),
            'n_securities': len(self.weights_data.sid.unique()) if not self.weights_data.empty else 0,
            'avg_weight_sum': self.weights_data.groupby('date')['weight'].sum().mean() if not self.weights_data.empty else 0,
            'dates_range': f"{min(self.dates)} to {max(self.dates)}"
        }

        if self.security_master is not None:
            df_stats['sectors'] = self.security_master.sector.unique().tolist()
            df_stats['sectors_count'] = len(df_stats['sectors'])

        return df_stats

    def get_sector_weights(self) -> pd.DataFrame:
        """
        Get sector weights over time.
        """
        if self.security_master is None:
            self.security_master = self.get_security_master()

        if self.security_master is None or self.security_master.empty:
            return pd.DataFrame()

        sector_weights = (self.security_master
                          .groupby(['date', 'sector'])['weight']
                          .sum()
                          .unstack()
                          .fillna(0))

        return sector_weights

    def get_benchmark_prices(self) -> pd.DataFrame:
        if self.universe is not None:
            if self.df_bench.shape[0] == 0:
                df_bench = self._get_prices(univ_list=[self.universe])
                self.df_bench = df_bench
            return self.df_bench
        return pd.DataFrame()

    def get_members_prices(self, model_input) -> pd.DataFrame:
        if model_input.backtest.universe_list is not None:
            if self.df_price.shape[0] == 0:
                df_price = self._get_prices(univ_list=model_input.backtest.universe_list)
                self.df_price = df_price
            return self.df_price
        return pd.DataFrame()
    
    def get_portfolio(self, model_input, path: str | None = None) -> pd.DataFrame:
        if model_input.backtest.portfolio_list is not None:
            try:
                if path is None:
                    path = path="./data/time_series/portfolios/portfolio.csv" #".\data\time_series\portfolios\portfolio.csv"
                df = pd.read_csv(path)
                self.df_portfolio = df
            except:
                print("Portfolio file missing.")
        return self.df_portfolio
    
    def _get_prices_v0(self, univ_list: List[str]) -> pd.DataFrame: # model_input, 
        """Helper function to get prices from Yahoo Finance."""
        import asyncio
        import concurrent.futures
        import sys
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import time

        # Determine if async download is requested (backwards compatible)
        concurrent_download = getattr(self, 'concurrent_download', None)
        if concurrent_download is None:
            # Try to get from model_input if available
            concurrent_download = False
            if hasattr(self, 'model_input') and hasattr(self.model_input, 'backtest'):
                concurrent_download = getattr(self.model_input.backtest, 'concurrent_download', False)
        # If not found, fallback to False
        if concurrent_download is None:
            concurrent_download = False
        if len(univ_list) > 1:
            # TO DO: Fix why it's not working for multiple tickers... it works for only one at a time
            concurrent_download = False

        price_data = []
        start_date = min(self.dates)
        end_date = max(self.dates)

        def download_one(ticker):
            time.sleep(1)
            try:
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=pd.to_datetime(end_date) + pd.Timedelta(days=1)
                    )
                if df is not None and not df.empty:
                    # Handle both MultiIndex and flat columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df = df.xs(ticker, axis=1, level='Ticker')
                    df.index.name = None
                    df.insert(0, 'date', df.index)
                    df.insert(1, 'sid', ticker)
                    df.columns = [i.lower() for i in df.columns]
                    df['price'] = df['close']
                    return df
            except Exception as e:
                print(f"Error downloading prices for {ticker}: {e}")
            return None

        if concurrent_download:
            # Use asyncio with ThreadPoolExecutor for concurrent downloads
            async def download_all():
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    tasks = [loop.run_in_executor(executor, download_one, ticker) for ticker in univ_list]
                    results = await asyncio.gather(*tasks)
                return [df for df in results if df is not None]

            # Run the async function and get results
            if sys.version_info >= (3, 7):
                try:
                    results = asyncio.run(download_all())
                except RuntimeError:
                    # If already in an event loop (e.g., Jupyter), use create_task
                    results = asyncio.get_event_loop().run_until_complete(download_all())
            else:
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(download_all())
            price_data = results
        else:
            # Sequential fallback (original logic)
            for ticker in univ_list:
                df = download_one(ticker)
                if df is not None:
                    price_data.append(df)

        if price_data:
            df_price = pd.concat(price_data)
            df_price['date'] = pd.to_datetime(df_price['date']).dt.date
            df_price['return'] = df_price.groupby('sid')['price'].pct_change()
            return df_price
        else:
            return pd.DataFrame()

    def _get_prices(self, univ_list: List[str]) -> pd.DataFrame:
        """Helper function to get prices from Yahoo Finance."""
        import asyncio
        import concurrent.futures
        import sys
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import time
        
        # Determine if concurrent download is requested (backwards compatible)
        concurrent_download = getattr(self, 'concurrent_download', None)
        if concurrent_download is None:
            # Try to get from model_input if available
            concurrent_download = False
            if self.model_input is not None:
                try:
                    concurrent_download = self.model_input.backtest.concurrent_download
                except AttributeError:
                    concurrent_download = False
        
        price_data = []
        start_date = min(self.dates)
        end_date = max(self.dates)
        
        # def download_one(ticker, delay=0):
        def download_one(ticker: str, delay: Union[int, float]) -> Any:
            """Download data for a single ticker with optional delay for rate limiting."""
            # Add staggered delay to prevent simultaneous API hits
            if delay > 0:
                time.sleep(delay)
            
            try:
                # Add retry logic for failed requests
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        df = yf.download(
                            ticker, 
                            start=start_date, 
                            end=pd.to_datetime(end_date) + pd.Timedelta(days=1),
                            progress=False,  # Disable progress bar for cleaner output
                            # show_errors=False,  # Suppress yfinance error messages
                            threads=False  # Disable threading in yfinance to avoid conflicts
                        )
                        
                        if df is not None and not df.empty:
                            # Handle both MultiIndex and flat columns
                            if isinstance(df.columns, pd.MultiIndex):
                                df = df.xs(ticker, axis=1, level='Ticker')
                            df.index.name = None
                            df.insert(0, 'date', df.index)
                            df.insert(1, 'sid', ticker)
                            df.columns = [i.lower() for i in df.columns]
                            df['price'] = df['close']
                            print(f"✓ Successfully downloaded {ticker}")
                            return df
                        else:
                            print(f"⚠ Empty data returned for {ticker}")
                            
                    except Exception as e:
                        if attempt < max_retries:
                            print(f"⚠ Retry {attempt + 1}/{max_retries} for {ticker}: {e}")
                            time.sleep(1)  # Wait before retry
                        else:
                            print(f"✗ Failed to download {ticker} after {max_retries + 1} attempts: {e}")
                            
            except Exception as e:
                print(f"✗ Error downloading prices for {ticker}: {e}")
            return None
        
        if concurrent_download:
            # Use asyncio with ThreadPoolExecutor for concurrent downloads
            async def download_all():
                """Refactored download_all function with proper rate limiting and error handling."""
                try:
                    # Get the current event loop or create a new one
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Use much more conservative concurrent settings
                    max_workers = min(3, len(univ_list))  # Reduced from 5 to 3
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Create tasks with staggered delays to prevent simultaneous API hits
                        tasks = []
                        for i, ticker in enumerate(univ_list):
                            # Stagger requests with increasing delays
                            delay = i * 0.3  # 300ms between each request start
                            task = loop.run_in_executor(executor, download_one, ticker, delay)
                            tasks.append(task)
                        
                        # Wait for all tasks to complete with longer timeout
                        try:
                            results = await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True),
                                timeout=600  # Increased to 10 minutes for larger datasets
                            )
                            
                            # Filter out exceptions and None results with better reporting
                            valid_results = []
                            failed_count = 0
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    print(f"✗ Exception for {univ_list[i]}: {result}")
                                    failed_count += 1
                                elif result is not None:
                                    valid_results.append(result)
                                else:
                                    failed_count += 1
                            
                            print(f"📊 Download summary: {len(valid_results)} successful, {failed_count} failed")
                            return valid_results
                            
                        except asyncio.TimeoutError:
                            print("⏱ Download timeout - some requests may not have completed")
                            # Cancel remaining tasks
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            return []
                            
                except Exception as e:
                    print(f"✗ Error in download_all: {e}")
                    return []
            
            # Properly handle running the async function
            def run_download_all():
                """Helper function to run download_all with proper event loop handling."""
                try:
                    # Check if we're already in an event loop (like Jupyter notebooks)
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an existing event loop, need to use different approach
                        import nest_asyncio
                        nest_asyncio.apply()
                        return asyncio.run(download_all())
                    except RuntimeError:
                        # No existing event loop, safe to use asyncio.run()
                        if sys.version_info >= (3, 7):
                            return asyncio.run(download_all())
                        else:
                            # Python < 3.7 fallback
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                return loop.run_until_complete(download_all())
                            finally:
                                loop.close()
                                
                except ImportError:
                    # nest_asyncio not available, fallback to thread-based approach
                    print("nest_asyncio not available, using alternative approach...")
                    return run_in_thread()
                except Exception as e:
                    print(f"Error running async downloads: {e}")
                    return []
            
            def run_in_thread():
                """Alternative approach using thread for event loop isolation."""
                import threading
                import queue
                
                result_queue = queue.Queue()
                
                def thread_target():
                    try:
                        # Create new event loop in thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(download_all())
                        result_queue.put(('success', result))
                    except Exception as e:
                        result_queue.put(('error', e))
                    finally:
                        loop.close()
                
                thread = threading.Thread(target=thread_target)
                thread.start()
                thread.join(timeout=320)  # 5+ minute timeout
                
                if thread.is_alive():
                    print("Download thread timeout")
                    return []
                
                try:
                    status, result = result_queue.get_nowait()
                    if status == 'success':
                        return result
                    else:
                        print(f"Thread execution error: {result}")
                        return []
                except queue.Empty:
                    print("No result from download thread")
                    return []
            
            # Execute concurrent downloads
            print(f"🚀 Starting concurrent downloads for {len(univ_list)} tickers...")
            print("📡 Using rate-limited concurrent requests to respect API limits...")
            results = run_download_all()
            price_data = results
            print(f"✅ Concurrent downloads completed. Got {len(price_data)} successful results.")
            
        else:
            # Sequential fallback (original logic)
            print(f"Starting sequential downloads for {len(univ_list)} tickers...")
            for ticker in univ_list:
                df = download_one(ticker, 2)
                if df is not None:
                    price_data.append(df)
            print(f"Sequential downloads completed. Got {len(price_data)} results.")
        
        # Process results (keeping original logic)
        if price_data:
            df_price = pd.concat(price_data)
            df_price['date'] = pd.to_datetime(df_price['date']).dt.date
            df_price['return'] = df_price.groupby('sid')['price'].pct_change()
            return df_price
        else:
            return pd.DataFrame()
        
    def get_returns_long(self) -> pd.DataFrame:
        if self.df_price.empty:
            return pd.DataFrame()
        df_ret_long = self.df_price[['date', 'sid', 'return', 'price']].copy()
        df_ret_long['price'] = df_ret_long.groupby('sid').ffill()['price']
        df_ret_long['return'] = df_ret_long['return'].fillna(0.0)
        return df_ret_long

    def get_returns_wide(self) -> pd.DataFrame:
        if self.df_price.empty:
            return pd.DataFrame()
        df_ret_wide = self.df_price[['date', 'sid', 'return']].pivot(index='date', columns='sid', values='return')
        df_ret_wide.fillna(0.0, inplace=True)
        return df_ret_wide

def SecurityMasterFactory(model_input, *args) -> BaseModel:
    data_source = model_input.backtest.data_source
    universe = model_input.backtest.universe.value
    dates_turnover = [] if model_input.backtest.dates_turnover==None else model_input.backtest.dates_turnover
    dates_daily = [] if model_input.backtest.dates_daily==None else model_input.backtest.dates_daily
    if data_source.lower() == 'yahoo':
        concurrent_download = getattr(model_input.backtest, 'concurrent_download', None)
        security_master = SecurityMasterYahoo(
            universe=get_universe_mapping_yahoo(universe), 
            dates=dates_daily,
            dates_turnover=dates_turnover,
            concurrent_download=concurrent_download
        )
        return security_master
    elif data_source.lower() == 'bloomberg':
        import bql
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        security_master = SecurityMasterBloomberg(
            universe=universe, # 'NDX Index',
            dates=dates_daily, # dates_month
            dates_turnover=dates_turnover,
            bq=bq  # Bloomberg Query instance
            )
        return security_master
        # raise NotImplementedError("Bloomberg source not available in this context.")
    else:
        raise ValueError(f"Data source '{data_source}' is not supported.")

class BQLFactor(BaseModel):
    """BQLFactor class for handling Bloomberg Query Language factor data"""
    name: str
    start_date: str
    end_date: str
    universe: List[str]
    data: Optional[pd.DataFrame] = None
    description: Optional[str] = None
    category: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('data', mode='before')
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        if v is not None:
            required_columns = {'date', 'factor_name', 'sid', 'value'}
            if not all(col in v.columns for col in required_columns):
                missing_cols = required_columns - set(v.columns)
                raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        return v

    def get_factor_data(self, factor_type: str | None = None, **kwargs) -> pd.DataFrame:
        """
        Get factor data using BQL
        
        Args:            
            factor_type: Type of factor to retrieve (size, value, momentum, etc.)
        """
        # if self.data is not None:
        #     return self.data

        if factor_type is None:
            factor_type = self.name

        try:
            if factor_type == 'size':
                df = self.get_factor_size()
            elif factor_type == 'value':
                df = self.get_factor_value()
            elif factor_type == 'beta':
                df = self.get_factor_beta()
            elif factor_type == 'momentum':
                df = self.get_factor_momentum(**kwargs)
            elif factor_type == 'profit':
                df = self.get_factor_profit()
            elif factor_type == 'short_interest':
                df = self.get_factor_short_interest()
            elif factor_type == 'leverage':
                df = self.get_factor_leverage()
            elif factor_type == 'earnings_yield':
                df = self.get_factor_earnings_yield()
            else:
                raise ValueError(f"Unsupported factor type: {factor_type}")
            df.dropna(inplace=True)
            try:
                df['date'] = df['date'].dt.date
            except:
                pass # print("BQL factor date in date format")

            df.index = df['date']
            # df = df.set_index('date')
            df.index.name = 'index' 
            return df
        except Exception as e:
            raise Exception(f"Error getting factor data: {str(e)}")
    
    def get_factor_beta(self) -> pd.DataFrame:
        # import pdb; pdb.set_trace()
        # set up the BQL API instance to query Bloomberg data
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        factor = 'beta'
        #dates = bq.func.range("2022-12-31", "2025-01-21", frq='D')
        dates = bq.func.range(self.start_date, self.end_date, frq='D')
        trading_days = bq.data.px_last(dates=dates)
        beta_2 = bq.data.BETA(dates=dates)
        beta = bq.func.matches(beta_2,trading_days['value']!=bql.NA)
        beta = beta.replacenonnumeric(np.nan)
        request = {factor: beta}
        # req = bql.Request(self.universe, request)
        #req = bql.Request(['AAPL US Equity','IBM US Equity'], request)
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df().reset_index().set_index(['ID']) for x in res], axis=1)
        df.reset_index(drop=False, inplace=True)
        df[factor]=df.groupby('ID')[factor].ffill()
        df.rename(columns={
            'ID':'sid',
            'DATE':'date',
            factor:'value'},
                  inplace=True)        
        df.drop_duplicates(inplace=True)
        df.insert(1, 'factor_name', factor)
        df = df[['date','factor_name','sid','value']]
        return df

    def get_factor_earnings_yield(self) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        dates = bq.func.range(self.start_date, self.end_date, frq='D')
        earn_yield = bq.data.EARN_YLD(fill='prev', dates=dates)
        earn_yield = earn_yield.replacenonnumeric(np.nan)
        request = {'earn_yield': earn_yield}
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df().reset_index().set_index(['ID']) for x in res], axis=1)
        df.reset_index(drop=False, inplace=True)
        df['earn_yield']=df.groupby('ID')['earn_yield'].ffill()
        df.rename(columns={
            'ID':'sid',
            'AS_OF_DATE':'date',
            'earn_yield':'value'},
                  inplace=True)        
        df.drop_duplicates(inplace=True)
        df.insert(1, 'factor_name', 'earn_yield')
        df['value'] = -df['value']
        df = df[['date','factor_name','sid','value']]
        return df
    
    def get_factor_leverage(self) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        dates = bq.func.range(self.start_date, self.end_date, frq='D')
        lt_borrow = bq.data.BS_LT_BORROW(fill='prev', dates=dates)
        tot_assets =  bq.data.BS_TOT_ASSET(fill='prev', fpt='ltm', dates = dates)
        lev = (1./lt_borrow/tot_assets)*100.
        lev = lev.replacenonnumeric(np.nan)
        request = {'leverage': lev}
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df().reset_index().set_index(['ID']) for x in res], axis=1)
        df.reset_index(drop=False, inplace=True)
        df['leverage']=df.groupby('ID')['leverage'].ffill()
        df.rename(columns={
            'ID':'sid',
            'AS_OF_DATE':'date',
            'leverage':'value'},
                  inplace=True)        
        df.drop_duplicates(inplace=True)
        df.insert(1, 'factor_name', 'leverage')
        df = df[['date','factor_name','sid','value']]
        return df
    
    def get_factor_short_interest(self) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        dates = bq.func.range(self.start_date, self.end_date, frq='D')
        Short = bq.data.short_int(fill='prev', dates=dates)
        ShsEqy = bq.data.eqy_sh_out(fill='prev', dates=dates)
        ShsBs =  bq.data.bs_sh_out(fill='prev', fpt='ltm', dates = dates)
        Shs = bq.func.max(ShsEqy, ShsBs)
        si = (Short/Shs)*100.
        si = si.replacenonnumeric(np.nan)
        request = {'short_interest': si}
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df().reset_index().set_index(['ID']) for x in res], axis=1)
        df.reset_index(drop=False, inplace=True)
        df['short_interest']=df.groupby('ID')['short_interest'].ffill()
        df.rename(columns={
            'ID':'sid',
            'DATE':'date',
            'short_interest':'value'},
                  inplace=True)        
        df.drop_duplicates(inplace=True)
        df.insert(1, 'factor_name', 'short_interest')
        return df
        
    def get_factor_profit(self) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        factor = bq.data.normalized_profit_margin(dates=bq.func.range(self.start_date, self.end_date))
        request = {'profit': factor}
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df().reset_index().set_index(['ID']) for x in res], axis=1)
        df.reset_index(drop=False, inplace=True)
        df.drop(columns='REVISION_DATE',inplace=True)
        df['profit']=df.groupby('ID')['profit'].ffill()
        # columns: REVISION_DATE,AS_OF_DATE,PERIOD_END_DATE,profit
        df.rename(columns={
            'ID':'sid',
            'AS_OF_DATE':'date',
            'PERIOD_END_DATE':'end_date',
            'profit':'value'},
                  inplace=True)        
        df.drop_duplicates(inplace=True)
        df.insert(1, 'factor_name', 'profit')
        # Format date
        # df['date'] = df['date'].astype(str)
        # df['date'] = df['date'].map(lambda x: str(x).replace('-','') if len(x)==10 else x)
        # import pdb; pdb.set_trace()
        # df.index = df['date']
        # df.index.name = 'index' 
        # df['date'] = df['date'].map(lambda x: str(x.date()).replace('-','') if len(str(x.date()))==10 else str(x.date()))
        df.dropna(inplace=True)
        # df.reset_index(drop=True, inplace=True)
        return df
    
    def get_factor_size(self) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        factor = bq.data.cur_mkt_cap(dates=bq.func.range(self.start_date, self.end_date)).log()
        request = {'size': factor}
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df()[['DATE', x.name]].reset_index().set_index(['DATE', 'ID']) for x in res], axis=1)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=False, inplace=True)
        # Rename and format columns
        df = df[[f'DATE','ID','size']]
        df['size'] = 1./df['size']
        df.insert(1, 'factor_name', 'size')
        df.rename(columns={
            'DATE': 'date',
            'ID': 'sid',
            'size': 'value'
        }, inplace=True)
        # Format date
        # df['date'] = df['date'].astype(str)
        # df['date'] = df['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)
        # import pdb; pdb.set_trace()
        # df['date'] = df['date'].map(lambda x: str(x.date()).replace('-','') if len(str(x.date()))==10 else str(x.date()))
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def get_factor_value(self) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        request = {'price': bq.data.px_last(dates=bq.func.range(self.start_date, self.end_date)),
                   'book':bq.data.BOOK_VAL_PER_SH(dates=bq.func.range(self.start_date, self.end_date))}; 
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        # Process results
        df = pd.concat([x.df().reset_index().set_index(['ID']) for x in res], axis=1)
        df = df[['DATE','PERIOD_END_DATE','price','book']].ffill()
        df['value'] = df['price']/df['book'] # 'price2book'
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=False,inplace=True)
        # Rename and format columns
        df = df[[f'DATE','ID','value']]
        df.insert(1, 'factor_name', 'value')
        df.rename(columns={
            'DATE': 'date',
            'ID': 'sid',
            'value': 'value'
        }, inplace=True)
        # Format date
        # df['date'] = df['date'].astype(str)
        # df['date'] = df['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)
        # import pdb; pdb.set_trace()
        # df['date'] = df['date'].map(lambda x: str(x.date()).replace('-','') if len(str(x.date()))==10 else str(x.date()))
        # df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def get_factor_momentum(self, **kwargs) -> pd.DataFrame:
        import bql # bq: Bloomberg Query instance
        bq = bql.Service(preferences={'currencyCheck':'when_available'})
        if 'shift_lag' not in kwargs:
            shift_lag = int(252/12)
        else:
            shift_lag = kwargs.get('shift_lag')
        # factor = bq.data.px_last(dates=bq.func.range(self.start_date, self.end_date)).log()
        start_date = str((datetime.datetime.strptime(self.start_date, '%Y-%m-%d') + timedelta(days = -shift_lag)).date())
        factor = bq.data.px_last(dates=bq.func.range(start_date, self.end_date))
        request = {'price': factor}
        req = bql.Request(self.universe, request)
        res = bq.execute(req)
        df = pd.concat([x.df()[['DATE', x.name]].reset_index().set_index(['DATE', 'ID']) for x in res], axis=1)
        df.reset_index(drop=False, inplace=True)
        # import pdb; pdb.set_trace()
        df.drop_duplicates(inplace=True)
        df.sort_values(['ID','DATE'], inplace=True)
        df['momentum'] = df.groupby(['ID'])['price'].pct_change(periods=shift_lag)

        # Rename and format columns
        df = df[[f'DATE','ID','momentum']]
        df.insert(1, 'factor_name', 'momentum')
        df.rename(columns={
            'DATE': 'date',
            'ID': 'sid',
            'momentum': 'value'
        }, inplace=True)
        # Format date
        # df['date'] = df['date'].astype(str)
        # df['date'] = df['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)
        # import pdb; pdb.set_trace()
        # df['date'] = df['date'].map(lambda x: str(x.date()).replace('-','') if len(str(x.date()))==10 else str(x.date()))
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def get_returns(self, bq) -> pd.DataFrame:
        """Get returns data for the universe"""
        try:
            import bql
            returns = bq.data.return_holding_period(dates=bq.func.range(self.start_date, self.end_date))
            request = {'return': returns}
            req = bql.Request(self.universe, request)
            res = bq.execute(req)
            
            df_ret = pd.concat([x.df()[['DATE', x.name]].reset_index().set_index(['DATE', 'ID']) 
                              for x in res], axis=1)
            df_ret.reset_index(drop=False, inplace=True)
            
            # Format DataFrame
            df_ret = df_ret[['DATE', 'ID', 'return']]
            df_ret.rename(columns={'DATE': 'date', 'ID': 'sid'}, inplace=True)
            # df_ret.index = df_ret['date']
            df_ret = df_ret.set_index('date')
            df_ret.index.name = 'index'
            # df_ret['date'] = df_ret['date'].astype(str)
            # df_ret['date'] = df_ret['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)
            
            return df_ret
            
        except Exception as e:
            raise Exception(f"Error getting returns data: {str(e)}")

    def get_market_cap(self, bq) -> pd.DataFrame:
        """Get market cap data for the universe"""
        try:
            import bql
            mktcap = bq.data.market_cap(dates=bq.func.range(self.start_date, self.end_date))
            request = {'mktcap': mktcap}
            req = bql.Request(self.universe, request)
            res = bq.execute(req)
            
            df_cap = pd.concat([x.df()[['DATE', x.name]].reset_index().set_index(['DATE', 'ID']) 
                              for x in res], axis=1)
            df_cap.reset_index(drop=False, inplace=True)
            
            # Format DataFrame
            df_cap = df_cap[['DATE', 'ID', 'mktcap']]
            df_cap.rename(columns={'DATE': 'date', 'ID': 'sid'}, inplace=True)
            df_cap.index = df_cap['date']
            df_cap.index.name = 'index'

            # df_cap
            # df_cap['date'] = df_cap['date'].astype(str)
            # df_cap['date'] = df_cap['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)
            
            return df_cap
            
        except Exception as e:
            raise Exception(f"Error getting market cap data: {str(e)}")

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
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))

        if self.data is None:
            self.data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            # self.data = self._download_price_data(self.universe, self.start_date, end_date)

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
                    try:
                        df['date'] = df['date'].dt.date
                    except:
                        pass # print("Yahoo factor date in date format")
                    df.index = df['date']
                    # df = df.set_index('date')
                    df.index.name = 'index'
            return df
            
        except Exception as e:
            raise Exception(f"Error getting factor data: {str(e)}")
    
    # @lru_cache(maxsize=32)
    def _download_price_data(self, tickers: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
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
            if len(ticker_list)==1 and data is not None and not isinstance(data.columns, pd.MultiIndex):
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
            if data is not None:
                return data
            else:
                return pd.DataFrame()
            
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
        # df = df.set_index('date')
        df.index.name = 'index'
        
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
            # Download price and volume data
            # data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            if self.data is not None:
                data = self.data.copy()
            else:
                data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
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
            start_date_dt = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            adjusted_start = (start_date_dt - timedelta(days=window * 1.5)).strftime('%Y-%m-%d')
            
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
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
                        start_date_dt = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
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
            start_date_dt = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            adjusted_start = (start_date_dt - timedelta(days=shift_lag * 1.5)).strftime('%Y-%m-%d')
            
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
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
                    start_date_dt = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
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
            start_date_dt = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            adjusted_start = (start_date_dt - timedelta(days=window * 1.5)).strftime('%Y-%m-%d')
            
            # Get current date if end_date is None or future
            if self.end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
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
                    start_date_dt = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
            # Download price and dividend data
            # data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            if self.data is not None:
                data = self.data.copy()
            else:
                data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            # data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            if self.data is not None:
                data = self.data.copy()
            else:
                data = self._download_price_data(tuple(self.universe), self.start_date, end_date)
            
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
            # Download price data
            if self.data is not None:
                data = self.data.copy()
            else:
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
            # df_ret.index = df_ret['date']
            df_ret = df_ret.set_index('date')
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
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            else:
                end_date = min(self.end_date, datetime.datetime.now().strftime('%Y-%m-%d'))
            
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
            df_cap = df_cap.set_index('date')
            # df_cap.index = df_cap['date']
            # df_cap.index.name = 'index'
            
            # Sort and remove duplicates
            df_cap.sort_values(['sid', 'date'], inplace=True)
            df_cap.drop_duplicates(inplace=True)
            
            return df_cap
            
        except Exception as e:
            print(f"Error in get_market_cap: {str(e)}")
            return pd.DataFrame()

class EquityFactor(BaseModel):
    """
    A class to handle equity factor data processing and analysis.
    
    Attributes:
        name: Name of the factor
        data: DataFrame containing factor data
        description: Optional description of the factor
        category: Category of the factor (e.g., 'Value', 'Momentum', etc.)
    """
    name: str
    data: pd.DataFrame
    description: Optional[str] = None
    category: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('data', mode='before')
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        required_columns = {'date', 'factor_name', 'sid', 'value'}
        if not all(col in v.columns for col in required_columns):
            missing_cols = required_columns - set(v.columns)
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Ensure date column is datetime
        # v['date'] = pd.to_datetime(v['date'])
        return v
    
    def normalize(self, method: Literal['zscore', 'rank', 'winsorize', 'percentile'] = 'zscore',
                 groupby: Optional[str] = None,
                 winsorize_limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """
        Normalize factor values cross-sectionally using specified method.
        
        Args:
            method: Normalization method ('zscore', 'rank', 'winsorize', 'percentile')
            groupby: Optional column to group by before normalizing
            winsorize_limits: Tuple of (lower, upper) percentile limits for winsorization
            
        Returns:
            DataFrame with normalized values
        """
        df = self.data.copy()
        
        def normalize_group(group_df: pd.DataFrame) -> pd.DataFrame:
            values = group_df['value'].values
            
            if method == 'zscore':
                normalized = stats.zscore(values, nan_policy='omit')
            elif method == 'rank':
                normalized = stats.rankdata(values, method='average')
                normalized = (normalized - 1) / (len(normalized) - 1)  # Scale to [0,1]
            elif method == 'winsorize':
                lower, upper = np.nanpercentile(values, [winsorize_limits[0]*100, 
                                                       winsorize_limits[1]*100])
                normalized = np.clip(values, lower, upper)
                # Z-score after winsorizing
                normalized = stats.zscore(normalized, nan_policy='omit')
            elif method == 'percentile':
                normalized = stats.rankdata(values, method='average')
                normalized = stats.norm.ppf(normalized / (len(normalized) + 1))
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            group_df = group_df.copy()
            group_df['value'] = normalized
            return group_df
        
        if groupby:
            # df = df.groupby([groupby, 'factor_name']).apply(normalize_group).reset_index(drop=True)
            df = df.groupby([groupby,'factor_name'], as_index=True).apply(
                normalize_group, include_groups=False).reset_index(drop=False)
            if 'index' in df.columns:
                df.drop(columns=['index'], inplace=True)
            if 'level_2' in df.columns:
                df.drop(columns=['level_2'], inplace=True)
            df = df.sort_values(['sid','date'])
        else:
            factor_name = df.factor_name.unique()[0]
            df = df.groupby('factor_name').apply(normalize_group, include_groups=False).reset_index(drop=True)
            df.insert(1,'factor_name',factor_name)

        df.index = df.date
        # df = df.set_index('date')
        df.index.name = None
        return df
    
    def to_wide(self, value_column: str = 'value') -> pd.DataFrame:
        """
        Convert factor data to wide format with dates in rows and sids in columns.
        
        Args:
            value_column: Column to use for values in the wide format
            
        Returns:
            DataFrame in wide format
        """
        df_wide = self.data.pivot(
            index='date',
            columns='sid',
            values=value_column
        )
        return df_wide
    
    def compute_summary_stats(self) -> pd.DataFrame:
        """
        Compute summary statistics for the factor.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.data is not None:
            # stats_df = self.data.groupby(['date'])['value'].agg([
            #     ('mean', 'mean'),
            #     ('std', 'std'),
            #     ('min', 'min'),
            #     ('max', 'max'),
            #     ('missing_pct', lambda x: (x.isna().sum() / len(x)) * 100),
            #     ('count', 'count')
            # ]).round(4)
            stats_df = self.data.groupby(['date'])['value'].agg(
                mean='mean',
                std='std',
                min='min',
                max='max',
                missing_pct=lambda x: (x.isna().sum() / len(x)) * 100,
                count='count'
            ).round(4)
            return pd.DataFrame(stats_df)
        return pd.DataFrame()
    
    def compute_autocorrelation(self, lags: List[int] = [1, 5, 10, 21, 63]) -> pd.DataFrame:
        """
        Compute factor autocorrelation for specified lags.
        
        Args:
            lags: List of lags to compute autocorrelation for
            
        Returns:
            DataFrame with autocorrelation values
        """
        df_wide = self.to_wide()
        auto_corr = pd.DataFrame()
        
        for lag in lags:
            corr = df_wide.corrwith(df_wide.shift(lag))
            auto_corr[f'lag_{lag}'] = corr
        
        return auto_corr.mean().to_frame(name='autocorrelation')
    
    def compute_coverage(self) -> pd.DataFrame:
        """
        Compute factor coverage statistics over time.
        
        Returns:
            DataFrame with coverage statistics
        """
        coverage = self.data.groupby('date').agg({
            'sid': ['count', 'nunique'],
            'value': lambda x: (~x.isna()).sum()
        })
        
        coverage.columns = ['total_records', 'unique_securities', 'non_null_values']
        coverage['coverage_pct'] = (coverage['non_null_values'] / 
                                  coverage['unique_securities'] * 100).round(2)
        
        return coverage
    
    def analyze_factor_returns(self, 
                            returns_data: pd.DataFrame,
                            n_buckets: int = 5,
                            method: Literal['quantile', 'equal_width'] = 'quantile',
                            weighting: Literal['equal', 'value'] = 'equal',
                            long_short: bool = True,
                            neutralize_size: bool = False,
                            shift_lag: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Analyze factor returns by constructing portfolios based on factor exposures.
        
        Args:
            returns_data: DataFrame with columns [date, sid, return]
            n_buckets: Number of buckets to divide the universe
            method: Method to create buckets ('quantile' or 'equal_width')
            weighting: Portfolio weighting scheme ('equal' or 'value')
            long_short: If True, compute long-short portfolio returns
            neutralize_size: If True, neutralize returns within size buckets
            shift_lag: number of lags for factor value
            
        Returns:
            Dict containing:
                - bucket_returns: Returns for each bucket over time
                - portfolio_stats: Performance statistics for each bucket
                - turnover: Portfolio turnover statistics
                - factor_exposure: Factor exposure statistics
        """
        # Rename sid to sid if needed for merging
        factor_data = self.data.copy()
        if 'sid' in factor_data.columns:
            factor_data = factor_data.rename(columns={'sid': 'sid'})
        
        factor_data['value_0'] = factor_data['value']
        factor_data['value'] = factor_data.groupby(['factor_name','sid'], as_index=False)['value_0'].shift(shift_lag)
        # if (len(model_input.backtest.dates_turnover) > 0) & False:
        #     factor_data = update_values_based_on_turnover_vectorized(
        #         factor_data, 
        #         model_input.backtest.dates_turnover)
        factor_data.dropna(inplace=True)
        # factor_data['date'] = factor_data['date'].map(lambda x: pd.to_datetime(x).date())
        factor_data['date'] = pd.to_datetime(factor_data['date']).dt.date
        factor_data.reset_index(drop=True, inplace=True)
        
        # Merge factor and returns data
        merged = pd.merge(
            factor_data,
            returns_data,
            on=['date', 'sid'],
            how='inner'
        )
        
        # Function to create buckets
        def create_buckets(x, method='quantile'):
            if method == 'quantile':
                return pd.qcut(x, n_buckets, labels=False, duplicates='drop')
            else:  # equal_width
                return pd.cut(x, n_buckets, labels=False)
        
        # Add size bucket if neutralization is requested
        if neutralize_size:
            merged['size_bucket'] = merged.groupby('date')['value'].transform(
                lambda x: pd.qcut(x, 5, labels=False)
            )
        
        # Create factor buckets
        merged['bucket'] = merged.groupby(['date'] + 
                                        (['size_bucket'] if neutralize_size else []))['value'].transform(
            lambda x: create_buckets(x, method)
        )
        
        # Calculate portfolio weights
        if weighting == 'equal':
            merged['weight'] = 1
        else:  # value weighting
            merged['weight'] = merged['value'].abs()
            
        merged['weight'] = merged.groupby(['date', 'bucket'])['weight'].transform(
            lambda x: x / x.sum()
        )
        
        # Calculate bucket returns
        # bucket_returns_0 = merged.groupby(['date', 'bucket']).apply(
        #     lambda x: (x['return'] * x['weight']).sum()
        # ).unstack()
        bucket_returns = merged.groupby(['date', 'bucket'], observed=True).apply(
            lambda x: (x['return'] * x['weight']).sum(),
            include_groups=False #explicitly exclude the grouping columns.
            ).unstack()
        
        bucket_returns.columns = [f'Bucket_{i+1}' for i in range(n_buckets)]
        
        # Calculate long-short portfolio if requested
        if long_short:
            bucket_returns['Long_Short'] = bucket_returns[f'Bucket_{n_buckets}'] - bucket_returns['Bucket_1']
        
        # Calculate portfolio statistics
        def calculate_portfolio_stats(returns):
            df_stats = pd.Series({
                'Mean Return (%)': returns.mean() * 100,
                'Std Dev (%)': returns.std() * 100,
                'Sharpe Ratio': np.sqrt(252) * returns.mean() / returns.std(),
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                # 't-statistic': stats.ttest_1samp(returns, 0)[0],
                # 'p-value': stats.ttest_1samp(returns, 0)[1]
            })
            return df_stats
        
        portfolio_stats = bucket_returns.apply(calculate_portfolio_stats)
        
        # Calculate turnover
        def calculate_turnover(bucket_data):
            weights_t = bucket_data.pivot(index='date', columns='sid', values='weight').fillna(0)
            weights_t_1 = weights_t.shift(1).fillna(0)
            turnover = np.abs(weights_t - weights_t_1).sum(axis=1) / 2
            return turnover.mean()
        
        # turnover = merged.groupby('bucket').apply(calculate_turnover)
        turnover = merged.groupby('bucket', observed=True).apply(calculate_turnover, include_groups=False)
        turnover.index = [f'Bucket_{i+1}' for i in range(n_buckets)]
        
        # Calculate factor exposure statistics
        exposure_stats = merged.groupby('bucket')['value'].agg([
            'mean', 'std', 'min', 'max'
        ])
        exposure_stats.index = [f'Bucket_{i+1}' for i in range(n_buckets)]
        
        return {
            'bucket_returns': bucket_returns,
            'portfolio_stats': portfolio_stats,
            'turnover': turnover,
            'factor_exposure': exposure_stats
        }

    def plot_cumulative_returns(self, returns_data: pd.DataFrame, n_buckets: int = 5, shift_lag: int=0):
        """
        Plot cumulative returns for factor portfolios.
        
        Args:
            returns_data: DataFrame with columns [date, sid, return]
            n_buckets: Number of buckets for portfolio construction
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get bucket returns
            results = self.analyze_factor_returns(returns_data, n_buckets=n_buckets, shift_lag=shift_lag)
            bucket_returns = results['bucket_returns']
            
            # Calculate cumulative returns
            cum_returns = (1 + bucket_returns).cumprod()
            
            # Plot
            plt.figure(figsize=(12, 6))
            # for col in cum_returns.columns:
            #     plt.plot(cum_returns.index, cum_returns[col], label=col)
            for col in cum_returns.columns:  
                plt.plot([str(i.date()) for i in cum_returns.index], cum_returns[col].to_numpy(), label=col)

            plt.title(f'{self.name.upper()} Factor Portfolio Cumulative Returns; lags: {shift_lag}')# plt.title('Factor Portfolio Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
            plt.xticks(rotation=45)
            # plt.xticks(np.arange(0, cum_returns.shape[0], 100))
            plt.xticks(np.arange(0, cum_returns.shape[0], int(cum_returns.shape[0]/10)))
            # plt.tight_layout()
            plt.show()
            
            # Print portfolio statistics
            print("\nPortfolio Statistics:")
            print(results['portfolio_stats'].round(3))
            
        except ImportError:
            warnings.warn("Plotting requires matplotlib and seaborn to be installed.")
    
    def plot_factor_histogram(self, bins: int = 50):
        """
        Plot histogram of factor values.
        
        Args:
            bins: Number of bins for histogram
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x='value', bins=bins)
            plt.title(f'{self.name.upper()} Factor Distribution')
            plt.xlabel('Factor Value')
            plt.ylabel('Count')
            plt.show()
        except ImportError:
            warnings.warn("Plotting requires matplotlib and seaborn to be installed.")

def FactorFactory(factor_type: str, model_input, *args) -> BaseModel: # dates,
    data_source = model_input.backtest.data_source
    start_date = str(model_input.backtest.start_date)
    end_date = str(model_input.backtest.end_date)
    universe_list = model_input.backtest.universe_list
    if data_source.lower() == 'yahoo':
        factor = YahooFactor(
            name=factor_type,
            start_date=start_date,
            end_date=end_date,
            universe=universe_list,
            description=f"{factor_type.capitalize()} factor data for universe {model_input.backtest.universe.value}",
            category="equity"
        )
        return factor
    elif data_source.lower() == 'bloomberg':
        factor = BQLFactor(
            name=factor_type,
            start_date=str(model_input.backtest.start_date), # '2023-01-01',
            end_date=str(model_input.backtest.end_date), # '2023-12-31',
            universe=model_input.backtest.universe_list, # univ_list
            category='market'
            )
        return factor
    else:
        raise ValueError(f"Data source '{data_source}' is not supported.")

# Utility to rebalance dates
def get_rebalance_dates(
    model_input: EquityFactorModelInput,
    return_as: str = "str",  # or "date"
    frq:Frequency = None,
    ) -> List[Union[str, date]]:
    start = model_input.backtest.start_date
    end = model_input.backtest.end_date
    if frq is None:
        frq = model_input.backtest.frequency

    if frq == Frequency.DAILY:
        dates = pd.date_range(start=start, end=end, freq='B')

    elif frq == Frequency.WEEKLY:
        weekday_map = {
            "MON": "W-MON", "TUE": "W-TUE", "WED": "W-WED",
            "THU": "W-THU", "FRI": "W-FRI"
        }
        freq_str = weekday_map.get(model_input.weekly_day, "W-FRI")
        dates = pd.date_range(start=start, end=end, freq=freq_str)

    elif frq == Frequency.MONTHLY:
        dates = pd.date_range(start=start, end=end, freq='BME')  # Business month end

    elif frq == Frequency.QUARTERLY:
        dates = pd.date_range(start=start, end=end, freq='BQ')  # Business quarter end

    elif frq == Frequency.CUSTOM:
        if all(isinstance(d, str) for d in model_input.custom_dates):
            dates = pd.to_datetime(model_input.custom_dates)
        else:
            dates = pd.to_datetime([pd.Timestamp(d) for d in model_input.custom_dates])
    else:
        raise ValueError(f"Unsupported frequency type: {frq}")

    if return_as == "str":
        return dates.strftime('%Y-%m-%d').tolist()
    elif return_as == "date":
        return [d.date() for d in dates]
    else:
        raise ValueError("return_as must be either 'str' or 'date'")

# Utility to update model_input's start date
def set_model_input_start(model_input, new_start_date):
    model_input.backtest.start_date = new_start_date
    # Optionally, update turnover/daily dates as well if needed

# Utility to get portfolio turnover dates
def set_model_input_dates_turnover(model_input):
    dates_turnover = get_rebalance_dates(model_input, return_as='str')
    if dates_turnover==[]:
        dates_turnover = [str(model_input.backtest.start_date)]
    model_input.backtest.dates_turnover = dates_turnover

# Utility to get daily business dates
def set_model_input_dates_daily(model_input):
    dates_daily = get_rebalance_dates(model_input, return_as='str', frq=Frequency.DAILY)
    model_input.backtest.dates_daily = dates_daily
    n_dates = len(model_input.backtest.dates_daily)
    model_input.params.n_dates = n_dates

def test_yahooFactor():
    """
    Example demonstrating how to use the YahooFactor class to analyze factor data
    """
    print("Starting YahooFactor Usage Example...")
    
    # 1. Define a universe of stocks to analyze
    universe_list = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA']
    # "C:\Users\alfredo\Project\EquityApp\data\historical\spx_universe.csv"
    
    # 2. Define the date range
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year of data
    
    print(f"Analyzing factors for {len(universe_list)} tech stocks from {start_date} to {end_date}")
    
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
            universe=universe_list,
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
        universe=universe_list
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
    stock_factors = {stock: {} for stock in universe_list}
    
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

# Helper functions
def merge_weights_with_factor_loadings(
    df_weights: pd.DataFrame,
    df_factors: pd.DataFrame,
    weight_col: str = "wgt",
    keep_weight_cols: Optional[Sequence[str]] = ("name", "exchange"),
    ) -> pd.DataFrame:
    """
    Build df_weights_final by carrying portfolio weights from each rebalance date
    forward to all daily factor dates up to (but not including) the next rebalance,
    and merging with factor loadings.

    Parameters
    ----------
    df_weights : DataFrame
        Columns required: ['date','sid', weight_col]. Optional extras like 'name','exchange'.
        Each row represents a holding weight at a *rebalance date*.
    df_factors : DataFrame
        Columns required: ['date','sid','factor_name','value'] for the entire universe at daily (or any) frequency.
    weight_col : str
        Column in df_weights containing the portfolio weight (default 'wgt').
    keep_weight_cols : sequence of str or None
        Extra columns from df_weights to carry into the final output. Set to None to skip.

    Returns
    -------
    DataFrame
        df_weights_final with columns:
        ['date','sid', <kept weight cols...>, weight_col, 'factor_name','value']
        Only rows where the (sid, date) is in the portfolio implied by the previous rebalance.
    """
    # --- basic validation
    for c in ("date", "sid"):
        if c not in df_weights.columns:
            raise ValueError(f"df_weights missing required column '{c}'")
        if c not in df_factors.columns:
            raise ValueError(f"df_factors missing required column '{c}'")
    for c in ("factor_name", "value"):
        if c not in df_factors.columns:
            raise ValueError(f"df_factors missing required column '{c}'")
    if weight_col not in df_weights.columns:
        raise ValueError(f"df_weights missing weight column '{weight_col}'")

    # --- ensure datetime (normalize to dates to avoid time-of-day issues)
    w = df_weights.copy()
    f = df_factors.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()

    # --- determine rebalance calendar and factor calendar
    rebalance_dates = pd.Index(sorted(w["date"].unique()))
    if rebalance_dates.empty:
        # No rebalances => empty result
        return f.head(0).assign(**{weight_col: pd.Series(dtype=float)})

    factor_dates = pd.Index(sorted(f["date"].unique()))
    if factor_dates.empty:
        # No factor dates => empty result
        return f.head(0).assign(**{weight_col: pd.Series(dtype=float)})

    # We will expand weights interval-by-interval: [reb_i, reb_{i+1}) intersect factor_dates
    frames = []
    # helper columns to carry through
    cols_to_keep = ["sid", weight_col]
    if keep_weight_cols:
        for c in keep_weight_cols:
            if c in w.columns and c not in cols_to_keep:
                cols_to_keep.append(c)

    for i, d0 in enumerate(rebalance_dates):
        # next rebalance (exclusive end); for the last window, go through the last factor date
        d1_exclusive = rebalance_dates[i + 1] if (i + 1) < len(rebalance_dates) else (factor_dates.max() + pd.Timedelta(days=1))

        # dates in factor calendar inside [d0, d1_exclusive)
        mask = (factor_dates >= d0) & (factor_dates < d1_exclusive)
        if not mask.any():
            continue
        dates_window = pd.DataFrame({"date": factor_dates[mask]})

        # weights at the current rebalance d0
        w_slice = w.loc[w["date"] == d0, ["date"] + cols_to_keep].copy()
        if w_slice.empty:
            continue

        # cross-join dates_window with w_slice (assign those weights to all dates in the window)
        w_slice = w_slice.drop(columns=["date"]).assign(_k=1)
        dates_window = dates_window.assign(_k=1)
        expanded = dates_window.merge(w_slice, on="_k").drop(columns="_k")

        frames.append(expanded)

    if not frames:
        # No overlap between factor dates and weight windows
        return f.head(0).assign(**{weight_col: pd.Series(dtype=float)})

    # Concatenate all expanded weights across windows
    weight_schedule = pd.concat(frames, ignore_index=True)

    # Now, inner-join with factor loadings to keep only (date, sid) pairs in the active portfolio
    df_weights_final = (
        f.merge(weight_schedule, on=["date", "sid"], how="inner")
         .sort_values(["date", "sid", "factor_name"])
         .reset_index(drop=True)
    )

    df_weights_final['date'] = pd.to_datetime(df_weights_final['date']).dt.date

    return df_weights_final

# Example usage:
if __name__ == "__main__":
    # Configure logging to see outputs
    logging.basicConfig(level=logging.INFO)

    from file_data_manager import (
        FileConfig, FileDataManager 
        )
    # test_yahooFactor()
    
    update_history = False # True | False

    cfg = FileConfig()
    mgr = FileDataManager(cfg)

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
            n_buckets=5
        ),
        backtest=BacktestConfig(
            data_source = DataSource.YAHOO, # 'yahoo',
            universe=Universe.INDU,  # Universe.INDU: Dow Jones Industrial Average # 'SEMLMCUP Index', 'NDX Index'
            currency=Currency.USD,
            frq=Frequency.MONTHLY,
            start='2023-12-31', # '2023-12-31',
            # end='2024-12-02' # If not end date, default to today
            portfolio_list=[],
            concurrent_download = True
        ),
        regime=RegimeConfig(
            type='vol',
            benchmark=VolatilityType.VIX, # 'VIX Index',
            periods=10
        ),
        export=ExportConfig(
            update_history = False,
            base_path="./data/time_series",
            s3_config={
                'bucket_name': os.environ.get('CLOUD_USER_BUCKET'),
                'user_name': os.environ.get('CLOUD_USERNAME')
            }
        )
    )
    # Get portfolio turnover dates
    # dates_turnover = get_rebalance_dates(model_input, return_as='str')
    # model_input.backtest.dates_turnover = dates_turnover
    set_model_input_dates_turnover(model_input)

    # Validate and access the config
    print(model_input.model_dump_json(indent=2))

    # Get daily business dates
    # dates_daily = get_rebalance_dates(model_input, return_as='str', frq=Frequency.DAILY)
    # model_input.backtest.dates_daily = dates_daily
    # n_dates = len(model_input.backtest.dates_daily) # len(dates)
    # model_input.params.n_dates = n_dates
    # print(f"\nNumber of days in backtest is: {n_dates}")
    set_model_input_dates_daily(model_input)

    # Get European indices
    europe_indices = Universe.get_by_region('Europe')
    print("European Indices:")
    for name, desc in europe_indices.items():
        print(f"{name}: {desc}")
        
    # Get technical factors
    technical_factors = RiskFactors.get_by_category('Technical')
    print("Technical Indicators:")
    for name, desc in technical_factors.items():
        print(f"{name}: {desc}")

    """
    # Get security master object
    """
    security_master = SecurityMasterFactory(
        model_input=model_input
        )
    identifier = f"{model_input.backtest.universe.value.replace(' ','_')}"
        
    # Get benchmark prices
    if update_history:
        df_benchmark_prices = security_master.get_benchmark_prices()
    else:
        # df_benchmark_prices = file_load_benchmark_prices(model_input)
        df_benchmark_prices = mgr.load_prices(identifier)
    print("\nSecurity Master Benchmark Prices:")
    print(df_benchmark_prices) # security_master.df_bench.tail(3)

    # Get benchmark members' weights
    if update_history:
        df_benchmark_weights = security_master.get_benchmark_weights()
    else:
        # df_benchmark_weights = file_load_benchmark_weights(model_input)
        df_benchmark_weights = mgr.load_benchmark_weights(identifier)

    print("\nBenchmark Weights Sample:")
    print(df_benchmark_weights.tail(3))

    # Get portfolios: NOTE -> should be generated by another program and follow format below:
    # index         date       ticker  sid  exchange  sector     weight    wgt
    # 2025-03-31  2025-03-31   WMT     WMT  NYSE      Retailing  0.033333  0.013508
    df_portfolio = security_master.get_portfolio(model_input)
    model_input.backtest.portfolio_list = sorted(list(df_portfolio['sid'].unique()))
    print("\nSecurity Master Portfolio:")
    print(df_portfolio.tail(3)) # security_master.df_portfolio.tail(3))

    univ_list = sorted(df_benchmark_weights['sid'].unique())
    univ_list = sorted(list(set(univ_list + model_input.backtest.portfolio_list)))
    model_input.backtest.universe_list = univ_list
    n_sids = len(univ_list)
    model_input.params.n_sids = n_sids
    print(f"\nNumber of securities in universe is: {n_sids}")

    # Get benchmark members' prices
    if update_history:
        df_prices = security_master.get_members_prices(model_input)
    else:
        # df_prices = file_load_prices(model_input)
        df_prices = mgr.load_prices(identifier+'_members')

    print("\nSecurity Master Benchmark Members' Prices:")
    print(df_prices.tail(3)) # security_master.df_price.tail()

    # Get security master with BICS sectors
    if update_history:
        if security_master.security_master is None:
            sec_master = security_master.get_security_master(sector_classification='BICS')
        else:
            sec_master = security_master.security_master.copy()
        print("\nSecurity Master Sample:")
        print(sec_master.head(3))

        # Get universe statistics
        df_stats = security_master.get_universe_stats()
        print("\nUniverse Statistics:")
        for key, value in df_stats.items():
            print(f"{key}: {value}")
            
        # Get sector weights
        sector_weights = security_master.get_sector_weights()
        print("\nSector Weights Sample:")
        print(sector_weights.tail(3))

    """
    # Get long and wide returns from security master class: df_ret_long, df_ret_wide
    """
    # get returns long format
    if update_history:
        df_ret_long = security_master.get_returns_long()
    else:
        # df_ret_long = file_load_returns(model_input)
        df_ret_long = mgr.load_returns(identifier+'_members')

    print("\nUniverse Members' returns - long format:")
    print(df_ret_long.tail(2))

    # get returns wide format
    if update_history:
        df_ret_wide = security_master.get_returns_wide()
        print("\nUniverse Members' returns - wide format:")
        print(df_ret_wide.tail(2))
    else:
        df_ret_wide = df_ret_long[['date','sid','return']].pivot(
            index='date', columns='sid', values='return')
        df_ret_wide.fillna(0., inplace=True)

    """
    # Instantiate factor classes & get data
    """
    factor_list = [i.value for i in model_input.params.risk_factors] # ['momentum','beta','size','value']
    print(factor_list)

    if 'factor_dict' not in locals():
        factor_dict = {}
    
    if update_history:
        for factor_type in factor_list: # [:1]
            print(f"Running model for factor: {factor_type}")

            factor = FactorFactory(
                factor_type=factor_type,
                model_input=model_input, 
                )
                    
            # Get factor data
            # factor_data[factor_name] = factor.get_factor_data()
            factor_df = factor.get_factor_data()

            if df_benchmark_weights.shape[0]>0:
                factor_df = merge_weights_with_factor_loadings(df_benchmark_weights, factor_df)

            # Create EquityFactor instance
            factor_eq = EquityFactor(
                name=factor_type,
                data=factor_df,
                description=f"{factor_type} factor",
                category=factor_type
            )

            # Example operations
            # 1. Normalize factor values
            df_normalized = factor_eq.normalize(groupby='date', method='winsorize')

            # 2. Convert to wide format
            df_wide = factor_eq.to_wide()

            # 3. Compute summary statistics
            df_stats = factor_eq.compute_summary_stats()

            # 4. Compute autocorrelation
            autocorr = factor_eq.compute_autocorrelation()

            # 5. Compute coverage
            coverage = factor_eq.compute_coverage()

            # Print results
            print("\nSummary Statistics:")
            print(df_stats.tail(2))

            print("\nAutocorrelation:")
            print(autocorr)

            print("\nCoverage Statistics:")
            print(coverage.tail(2))

            results = factor_eq.analyze_factor_returns(
                returns_data=df_ret_long,
                n_buckets=model_input.params.n_buckets,
                method='quantile',
                weighting='equal',
                long_short=True,
                neutralize_size=False,
                shift_lag=1 # 22
            )

            # Store in factor dictionary
            factor_dict[factor_type] = {
                'factor': factor,
                'data': factor_df,
                'factor_eq': factor_eq,
                'results': results
            }

            # Access results
            bucket_returns = results['bucket_returns']
            portfolio_stats = results['portfolio_stats']
            turnover = results['turnover']
            exposure_stats = results['factor_exposure']

        # Plot cumulative returns: TO DO -> check plot function.
        # factor_eq.plot_cumulative_returns(df_ret_long, n_buckets=model_input.params.n_buckets, shift_lag=1)
    else:
        # Read factor data
        # factor_data_dict = file_load_factors(model_input)
        factor_data_dict = {}
        for factor in model_input.params.risk_factors:
            factor_name = factor.value
            factor_data_dict[factor_name] = mgr.load_factors(f"{identifier}_members_{factor_name}")

        # Initialize results dictionary
        # results = {}
        for factor_type in factor_list: # [:1]
            print(f"Running model for factor: {factor_type}")
            factor_df = factor_data_dict.get(factor_type)


            # if df_benchmark_weights.shape[0]>0:
            #     factor_df = merge_weights_with_factor_loadings(df_benchmark_weights, factor_df)

            # Create EquityFactor instance
            factor_eq = EquityFactor(
                name=factor_type,
                data=factor_df,
                description=f"{factor_type} factor",
                category=factor_type
            )
            
            # Get factor analysis results
            results = factor_eq.analyze_factor_returns(
                returns_data=df_ret_long,
                n_buckets=model_input.params.n_buckets,
                method='quantile',
                weighting='equal',
                long_short=True,
                neutralize_size=False,
                shift_lag=1
            )
            # results[factor_type] = results

            factor_dict[factor_type] = {
                'factor': None, # factor,
                'data': factor_df,
                'factor_eq': factor_eq,
                'results': results
            }

            results['bucket_returns'].cumsum().plot(
                title=f"{factor_type.title()} Factor Bucket Returns", 
                rot=45)
            plt.savefig(f'plot_{factor_type}_bucket_returns.png')

    """
    # Get exposures matrix
    """
    if update_history:
        df_exposures = pd.DataFrame()
        for factor in factor_dict.keys():
            # factor = 'beta'
            df = factor_dict.get(factor).get('factor_eq').normalize(groupby='date', method='winsorize')[['date','sid','value']].copy() # 'zscore'
            df.rename(columns={'value':factor}, inplace=True)
            df.sort_values(['sid','date'], inplace=True)
            df[factor] = df.groupby('sid')[factor].ffill()
            if df_exposures.shape[0]==0:
                df_exposures = df.copy()
            else:
                df_exposures = df_exposures.merge(df, how='left', on=['date','sid'])

        df_exposures.dropna(inplace=True)
        df_exposures.index = df_exposures['date']
        df_exposures.index.name = None

        df_exposures_long = pd.melt(df_exposures, id_vars=['date','sid'], value_name='exposure')
        df_exposures_long.rename(columns={'sid':'security_id'},inplace=True)
        df_exposures_long.insert(1, 'universe', model_input.backtest.universe.value)
    else:
        # df_exposures_long = file_load_exposures(model_input)
        df_exposures_long = mgr.load_exposures(identifier+'_members')
        df_exposures = df_exposures_long[['date','sid','variable','exposure']].pivot(
            index=['date','sid'], columns='variable', values='exposure').reset_index(drop=False)
        df_exposures.fillna(0., inplace=True)
        
    print("\nUniverse Members' Factor Exposures - long format:")
    print(df_exposures_long.groupby('sid').tail(1))
    # print(df_exposures.groupby('sid').tail(1).sort_values(['beta'], ascending=False))

    """
    # Get pure factor returns: TO DO -> add this to app_factors_test.py PURE_FACTOR optimization
    """

    # import qBacktest as bt
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
            df_ret_wide, 
            df_exposures, 
            model_input.backtest.dates_turnover # [str(i) for i in dates_to]
        )
        df_portfolio = results_opt.get('weights_data')
        
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

        print(f"{factor.upper()} Factor Return & Sharpe : {results_bt.cumulative_return_benchmark:.2%}, {results_bt.sharpe_ratio_benchmark:.2f}")
        # df_ret_opt = get_backtest(df_ret_long, df_portfolio, lag=1, flag_plot=False)
        # df_ret_opt.insert(0, 'factor', factor.lower())
            
        df_pure_return = pd.concat([df_pure_return, df_ret_opt])
        df_pure_portfolio = pd.concat([df_pure_portfolio, df_portfolio])
        
    # factor returns
    df_pure_return_wide = df_pure_return[['factor','return_opt']].pivot(columns='factor',values='return_opt')
    df_pure_return_wide.cumsum().plot(title=f"Pure Factor Returns: {security_master.universe}", rot=45, figsize=(16,8))
    plt.savefig(f"plot_pure_factor_returns_{identifier}.png")
    print(df_pure_return_wide.corr())
    print(252*df_pure_return_wide.mean())
    # df_pure_return_wide.reset_index(drop=False, inplace=True)
    # print(df_pure_return_wide.info())
