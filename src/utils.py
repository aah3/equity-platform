# Placeholder for utils.py
# This file would contain the actual implementation

# Utils 
from typing import List, Union, Dict, Iterable, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime as dtime
from datetime import timedelta
from time import sleep
import pytz
import pandas as pd
import os
import boto3
import numpy as np
import datetime
# import warning

# set up the BQL API instance to query Bloomberg data
# import bql
# bq = bql.Service(preferences={'currencyCheck':'when_available'})

FreqAlias = Literal["daily", "weekly", "monthly", "yearly"]
ResampleCode = Literal["B", "W-FRI", "ME", "YE"]

def _compound(r: pd.Series) -> float:
    """Compound simple returns (ignoring NaNs)."""
    if r.dropna().empty:
        return np.nan
    return float((1.0 + r.dropna()).prod() - 1.0)

def _to_resample_code(freq: FreqAlias) -> ResampleCode:
    return {
        "daily": "B",       # Business daily
        "weekly": "W-FRI",  # ISO-like: end week on Friday (change if needed)
        "monthly": "ME",
        "yearly": "YE",
    }[freq]

class FactorReturnAggregator(BaseModel):
    """
    Aggregates factor returns and computes stats.

    Expects a wide DataFrame:
        index: DatetimeIndex (daily or business-daily)
        columns: factor names (numeric returns, e.g., 0.002 = 0.2%)
    """
    data: pd.DataFrame = Field(..., description="Wide daily factor-return DataFrame.")
    trading_days_per_year: int = Field(252, ge=1)
    # Optional: enforce week ending day, e.g. 'W-FRI' (default in mapping above).
    weekly_anchor: Literal["W-FRI", "W-THU", "W-WED", "W-TUE", "W-MON"] = "W-FRI"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -------------------- Validators -------------------- #
    @field_validator("data", mode="before")
    @classmethod
    def _validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(v, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame.")
        if v.index is None or not isinstance(v.index, pd.Index):
            raise ValueError("`data` must have an index.")
        # Try to coerce index to datetime if not already
        if not isinstance(v.index, pd.DatetimeIndex):
            try:
                v = v.copy()
                v.index = pd.to_datetime(v.index)
            except Exception as e:
                raise ValueError(
                    "`data` index must be DatetimeIndex or coercible to datetime."
                ) from e
        if v.shape[1] == 0:
            raise ValueError("`data` must have at least one factor column.")
        # Ensure all columns are numeric
        non_numeric = [c for c in v.columns if not pd.api.types.is_numeric_dtype(v[c])]
        if non_numeric:
            raise TypeError(f"All factor columns must be numeric. Non-numeric: {non_numeric}")
        return v.sort_index()

    @field_validator("weekly_anchor")
    @classmethod
    def _validate_weekly_anchor(cls, v: str) -> str:
        return v

    # -------------------- Public API -------------------- #
    def aggregate(
        self,
        freq: FreqAlias,
        min_periods: int = 1,
    ) -> pd.DataFrame:
        """
        Aggregate returns by compounding over the given frequency.

        Parameters
        ----------
        freq : {'daily','weekly','monthly','yearly'}
        min_periods : minimum non-NA observations in a period to compute a return

        Returns
        -------
        DataFrame with compounded returns at the chosen frequency.
        """
        if freq == "daily":
            # Already daily; optionally ensure business-daily frequency
            df = self.data.asfreq("B")
            return df

        rule = _to_resample_code(freq)
        if freq == "weekly":
            rule = self.weekly_anchor  # override default if user customized

        # resample then compound within each period
        out = self.data.resample(rule).apply(_compound)
        # Optionally drop periods with insufficient observations
        if min_periods > 1:
            counts = self.data.resample(rule).count()
            out = out.where(counts >= min_periods)
        return out

    def aggregate_weekly(self, min_periods: int = 1) -> pd.DataFrame:
        return self.aggregate("weekly", min_periods=min_periods)

    def aggregate_monthly(self, min_periods: int = 1) -> pd.DataFrame:
        return self.aggregate("monthly", min_periods=min_periods)

    def aggregate_yearly(self, min_periods: int = 1) -> pd.DataFrame:
        return self.aggregate("yearly", min_periods=min_periods)

    def summary_stats(
        self,
        freq: FreqAlias = "daily",
        risk_free_rate_annual: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute per-factor summary statistics at the selected frequency.

        Stats are computed from the series at that frequency:
        - mean, vol, skew, kurtosis
        - annualized return & volatility
        - Sharpe (annualized)
        - hit rate (% positive)
        - max drawdown (from compounded series)
        - count (obs)

        Parameters
        ----------
        freq : aggregation level for stats ('daily','weekly','monthly','yearly')
        risk_free_rate_annual : annual risk-free rate (e.g. 0.02 for 2%)

        Returns
        -------
        DataFrame indexed by factor with statistics as columns.
        """
        df = self.aggregate(freq)

        # infer periods per year
        periods_per_year = {
            "daily": self.trading_days_per_year,
            "weekly": 52,      # anchored weeks
            "monthly": 12,
            "yearly": 1,
        }[freq]

        # risk-free per-period
        rf_per_period = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1

        stats: Dict[str, Dict[str, float]] = {}
        for col in df.columns:
            r = df[col].dropna()
            if r.empty:
                # Fill with NaNs if there are no observations
                stats[col] = {k: np.nan for k in [
                    "mean", "vol", "skew", "kurtosis", "annual_return",
                    "annual_vol", "sharpe", "hit_rate", "max_drawdown", "count"
                ]}
                continue

            mean = float(r.mean())
            vol = float(r.std(ddof=1))
            skew = float(r.skew())
            kurt = float(r.kurtosis())
            # Annualization
            ann_ret = float((1 + mean) ** periods_per_year - 1)
            ann_vol = float(vol * np.sqrt(periods_per_year))
            # Excess mean per period, then annualize Sharpe using ann_vol
            excess_per_period = mean - rf_per_period
            sharpe = np.nan
            if ann_vol > 0:
                sharpe = (excess_per_period * periods_per_year) / ann_vol

            # Hit rate
            hit_rate = float((r > 0).mean())

            # Max drawdown from compounded equity curve
            eq = (1.0 + r).cumprod()
            peak = eq.cummax()
            dd = (eq / peak) - 1.0
            mdd = float(dd.min())

            stats[col] = {
                "mean": mean,
                "vol": vol,
                "skew": skew,
                "kurtosis": kurt,
                "annual_return": ann_ret,
                "annual_vol": ann_vol,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
                "max_drawdown": mdd,
                "count": float(r.shape[0]),
            }

        out = pd.DataFrame(stats).T[
            ["count", "mean", "vol", "skew", "kurtosis",
             "annual_return", "annual_vol", "sharpe", "hit_rate", "max_drawdown"]
        ]
        return out.sort_index()

    # -------------------- Convenience helpers -------------------- #
    def stats_daily(self, rf_annual: float = 0.0) -> pd.DataFrame:
        return self.summary_stats(freq="daily", risk_free_rate_annual=rf_annual)

    def stats_weekly(self, rf_annual: float = 0.0) -> pd.DataFrame:
        return self.summary_stats(freq="weekly", risk_free_rate_annual=rf_annual)

    def stats_monthly(self, rf_annual: float = 0.0) -> pd.DataFrame:
        return self.summary_stats(freq="monthly", risk_free_rate_annual=rf_annual)

    def stats_yearly(self, rf_annual: float = 0.0) -> pd.DataFrame:
        return self.summary_stats(freq="yearly", risk_free_rate_annual=rf_annual)
        
def clean_list_items(input_list):
    import re
    cleaned_list = []
    for item in input_list:
        # Remove unwanted characters: &, ,, ...
        item = re.sub(r'&|,|\.{2,}', '', item)
        # Replace multiple spaces with a single space
        item = re.sub(r'\s+', ' ', item)
        # Strip leading/trailing spaces
        item = item.strip()
        # Replace space with underscore and convert to lowercase
        item = item.replace(' ', '_').lower()
        cleaned_list.append(item)
    return cleaned_list

def is_dst(dt=None, timezone="UTC"):
    # daylight savings helper function
    # print(is_dst()) # it is never DST in UTC
    if dt is None:
        dt = datetime.utcnow()
    timezone = pytz.timezone(timezone)
    timezone_aware_date = timezone.localize(dt, is_dst=None)
    return timezone_aware_date.tzinfo._dst.seconds != 0

def get_time_now(timezone = None):
    if timezone is None:
        timezone = "US/Eastern"
        
    now = datetime.utcnow()
    if is_dst(datetime.now(), timezone="US/Eastern"):
        now = now.replace(hour=now.hour - 4)
    else:
        now = now.replace(hour=now.hour - 5)        
    return now

def get_date_time_now(time=None, which=None, tz=None):
    # now = get_date_time_now(time=None, which=None)
    if time is None:
        if tz is None: 
            time = pd.Timestamp.now('US/Eastern') 
        else:
            time = pd.Timestamp.now(tz) 

    now = str(time).replace('-','').replace(' ','.').split('.')
    if which == 'date':
        return now[0]
    elif which == 'time':
        return now[1]
    else:
        return now

def get_dates_range(start=None, end=None, periods=1, freq='B'):
    if (start is None) & (end is None):
        end = str(pd.Timestamp.now('US/Eastern').date())
        dates = pd.date_range(end=end, periods=periods, freq=freq)
    if (start is not None) & (end is not None):
        dates = pd.date_range(start=start, end=end, freq=freq)
    if (start is not None) & (end is None):
        end = str(pd.Timestamp.now('US/Eastern').date())
        #dates = pd.date_range(start=start, periods=periods, freq=freq)
        dates = pd.date_range(start=start, end=end, freq=freq)
    dates = [str(i.date()).replace("-","") for i in dates]
    return dates

def get_earnings_univ(date=None, print_today=False):
    # pdb.set_trace()
    if date is None:
        date = str(get_time_now(timezone = None).date())
        print(f"Date for earnings: {date}")
    df_univ = pd.read_csv(f"../data/earnings/earnings_{date}.csv")
    df_univ['sid'] = df_univ['ID'].map(lambda x: x.split(' ')[0])
    df_univ.sort_values(['ID'], inplace=True)
    df_univ.rename(columns={'ID':'id', 'Release Date':'date_earn', 'Release Time':'release_time',
                            'Sector':'sector',
                            'Sub Sector':'sub_sector', 
                            'Name':'name', 'YTD %':'ret_ytd', '1M %':'ret_mtd', '1M Vol':'vol_1m', 'EPS Adj Last Q':'eps_adj_last_q',
                            'Est. EPS Adj':'eps_est_adj', 'EPS Adj 3M %Chg':'eps_adj_3m_chg', 'EPS Adj 1M %Chg':'eps_adj_1m_chg',
                            'Potential EPS Adj Growth':'eps_adj_growth', 'EPS Adj Coverage':'eps_coverage', 'EPS Adj Rev Up':'eps_rev_up',
                            'EPS Adj Rev Down':'eps_rev_dn', 'EPS Adj %Rev Up':'eps_rev_up_pct', 'EPS Adj %Rev Down':'eps_rev_dn_pct',
                            'EPS Adj Dispersion':'eps_disp', 'EPS Adj Up/Down':'eps_up_dn', 'EPS Adj Trend':'eps_trend'
                           }, inplace=True)
    
    df_univ['after_mkt'] = -1
    if 'release_time' in df_univ.columns:
        df_univ.loc[df_univ.release_time=='Aft-mkt', 'release_time'] = '16:10'
        df_univ.loc[df_univ.release_time=='Bef-mkt', 'release_time'] = '08:00'

        df_univ.loc[df_univ.release_time > '16:00', 'after_mkt'] = 1
        df_univ.loc[df_univ.release_time < '09:30', 'after_mkt'] = 0
    if 'Unnamed: 0' in df_univ.columns:
        df_univ.drop(columns=['Unnamed: 0'], inplace=True)
    print('Companies in earnings univ reporting after close and before open:')
    print(df_univ.after_mkt.value_counts())
    if print_today:
        print(df_univ.loc[df_univ.date_earn==date, ['sid','date_earn','after_mkt']].sort_values(['after_mkt','sid'], ascending=['True', 'True']))    
    return df_univ

def upload_df_to_s3(df, bucket_name, user_name, file_name):
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(bucket_name)

    #print(chain.shape); print(chain.dtypes); print(chain.head(2))

    print(f"Exporting chain with size: {df.shape}")
    from io import BytesIO
    # Save DataFrame as binary object in CSV format.
    bin_obj = BytesIO()
    df.to_csv(path_or_buf=bin_obj)
    bin_obj.seek(0)

    # Upload data to Sandbox Storage.
    print(f"{user_name}/{file_name}")
    try:
        s3_bucket.upload_fileobj(Fileobj = bin_obj, Key = f"{user_name}/{file_name}")
        return 1
    except:
        print(f"Option chain upload to S3 bucket failed...")
    return 0

# # functions
def get_markov_regime_switching(df0, n_periods=10, k_regimes=2, plot=False):
    #warnings.simplefilter('ignore', ConvergenceWarning)
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    
    s_ret = 'returnVol'

    df0[s_ret] = df0['close'].pct_change(n_periods)
    df0.dropna(inplace=True)
    df0.reset_index(drop=True, inplace=True)
    #print(df['return'].describe())

    rsm = sm.tsa.MarkovRegression(
        np.array(df0[s_ret]), k_regimes = k_regimes, trend = "n", switching_variance = True
    )
    rsm_fit = rsm.fit()

    df = rsm_fit.smoothed_marginal_probabilities.copy()
    df = pd.DataFrame(df)
    df.columns = ['low', 'high']

    #df[['date','close','return']].head(2)
    df = df0[['date','close',s_ret]].join(df)
    df.dropna(inplace = True)

    df['quantileHigh'] = pd.qcut(df['high'], k_regimes, labels=False)
    #df['quantileHighFwd'] = df['quantileHigh'].shift(1)
    df['signal'] = 1
    df.loc[df.high <= 0.5, 'signal'] = 0
    # df['close'].plot()
    # df['signal'].plot()
    
    if plot:
        fig, axes = plt.subplots(2, figsize=(10, 7))
        ax = axes[0]
        ax.plot(df['close'])
        ax.set(title="Volatility Index")
        ax = axes[1]
        ax.plot(df['signal'])
        ax.set(title="Volatility Regime")
        # ax = axes[2]
        # ax.plot(df['quantileHigh'])
        # ax.set(title="Quantile Regime")
        fig.tight_layout()
    return df

# # def get_optimal_gamma(prob, samples=20, which='min'):
# #     gamma_dict=getOptimalGamma(prob=cp.Problem(cp.Maximize(obj), constraints), samples=1)
# #     gamma_dict.get('gamma')
# #     SAMPLES = samples
# #     sharpe_data = np.zeros(SAMPLES)
# #     gamma_vals = np.logspace(-1, 1, num=SAMPLES)
# #     for i in range(SAMPLES):
# #         gamma.value = gamma_vals[i]
# #         try:
# #             prob.solve(verbose=False)
# #             ret_data[i] = ret.value
# #             risk_data[i] = cp.sqrt(risk).value
# #             sharpe_data[i] = ret.value / cp.sqrt(risk).value
# #             obj_data[i] = prob.value
# #         except:
# #             pass
# #             #print(f"Not able to solve problem: {i}")

# #     # optimal
# #     if which == 'max':
# #         i = np.argmax(sharpe_data)
# #     else:
# #         i = np.argmin(sharpe_data)
# #     # print("optimal sharpe")
# #     # print(np.sqrt(252)*sharpe_data[i], gamma_vals[i])
# #     #pd.DataFrame(np.sqrt(252)*sharpe_data, columns=['sharpe']).plot()
# #     return {'index':i, 'gamma':gamma_vals[i]}

def get_backtest(df_return, df_wgt, lag=0, flag_plot=False):
    # backtesting given returns, weights and lag periods
    if ('pfactor' in df_wgt.columns):
        facx = df_wgt['pfactor'].unique().squeeze()
    else:
        facx = None
        
    if 'weight_benchmark' not in df_wgt.columns:
        print(f"Missing weights for: {'benchmark'}")
        df_wgt['weight_benchmark'] = df_wgt['weight']
    
    # import pdb; pdb.set_trace()
    date_min = df_wgt['date'].min()
    # date_max = str(datetime.strptime(df_wgt['date'].max(), "%Y%m%d").date() + timedelta(days=2)).replace('-','')
    date_max = df_wgt['date'].max()
    print(date_min, date_max)
    ix = (df_return['date']>=date_min) & (df_return['date']<=date_max) #& (df_return.loc[df_return['date']<=df_wgt['date'].max()])
    df_return = df_return.loc[ix]
    if ('date_ann' in df_wgt.columns) & ('after_mkt' in df_wgt.columns):
        df = df_return.merge(df_wgt[['sid','date','date_ann','after_mkt','weight_benchmark','weight']], how='left', on=['sid','date']) # ,'shares_ew','shares'
    else:
        df = df_return.merge(df_wgt[['sid','date','weight_benchmark','weight']], how='left', on=['sid','date']) # ,'shares_ew','shares'
    if lag > 0:
        df['weight_benchmark'] = df.groupby('sid')['weight_benchmark'].shift(lag)
        df['weight'] = df.groupby('sid')['weight'].shift(lag)
        if 'after_mkt' in df.columns:
            df['after_mkt'] = df.groupby('sid')['after_mkt'].shift(lag)
        if 'date_ann' in df.columns:
            df['date_ann'] = df.groupby('sid')['date_ann'].shift(lag)
    df.sort_values(['sid','date'], inplace=True)
    df['return'].fillna(0., inplace=True)
    df = df.groupby('sid').ffill().reset_index()
    # df.fillna(method='ffill', inplace=True)
    # df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    df['return_bench'] = df['return']*df['weight_benchmark']
    df['return_opt'] = df['return']*df['weight']
    df.dropna(inplace=True)
    
    df_pnl = df[['date','return_bench','return_opt']].groupby(['date']).sum()

    print("Sharpe Benchmark, Optimal:")
    print(np.sqrt(252)*df_pnl['return_bench'].mean()/df_pnl['return_bench'].std(), np.sqrt(252)*df_pnl['return_opt'].mean()/df_pnl['return_opt'].std())
    print("Cummulative Returns:")
    print(df_pnl[['return_bench','return_opt']].cumsum().tail(1))
    if flag_plot:
        if facx is not None:
            title = f"Optimal PnL for {facx}"
        else:
            title = "Optimal PnL"
        df_pnl[['return_bench','return_opt']].cumsum().plot(title=title, figsize=(16,8), rot=45)
    return df_pnl

def get_rebalance_days(
    dates: List[Union[str, dtime]], 
    frequency: str = 'month_end',
    custom_dates: Optional[List[dtime]] = None) -> List[dtime]:
    """
    Get portfolio rebalancing days based on specified frequency.
    
    Args:
        dates: List of dates in datetime format or string format ('YYYY-MM-DD' or 'YYYYMMDD')
        frequency: Rebalancing frequency ('month_end', 'quarter_end', 'year_end', 'daily', 'custom')
        custom_dates: Optional list of custom rebalancing dates
    
    Returns:
        List of rebalancing dates as datetime objects
    
    Raises:
        ValueError: If invalid frequency is provided or dates format is incorrect
    """
    # from datetime import datetime
    from pandas.tseries.offsets import BMonthEnd, BQuarterEnd, BYearEnd, BDay

    # Convert dates to pandas datetime if they're strings
    if isinstance(dates[0], str):
        try:
            dates = pd.to_datetime(dates)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
    
    # Create DataFrame with dates
    df = pd.DataFrame(dates, columns=['date'])
    df.set_index('date', inplace=True)
    
    # Ensure dates are sorted
    df.sort_index(inplace=True)
    
    # Get business day calendar
    cal = pd.tseries.offsets.CustomBusinessDay()
    
    if frequency == 'custom' and custom_dates is not None:
        # Use custom rebalancing dates
        rebalance_dates = pd.to_datetime(custom_dates)
    else:
        # Get the appropriate business day offset based on frequency
        offset_map = {
            'month_end': BMonthEnd(),
            'quarter_end': BQuarterEnd(),
            'year_end': BYearEnd(),
            'daily': BDay(),
        }
        
        if frequency not in offset_map:
            raise ValueError(f"Invalid frequency. Choose from: {list(offset_map.keys())}")
        
        # Generate rebalancing dates
        if frequency == 'daily':
            rebalance_dates = df.index
        else:
            # Get the last business day of each period
            rebalance_dates = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=offset_map[frequency]
            )
    
    # Filter rebalancing dates to ensure they exist in the original dates
    valid_rebalance_dates = rebalance_dates[rebalance_dates.isin(df.index)]
    
    return valid_rebalance_dates.date.tolist()
    
# # function to get closest date
def get_closest_date(df, idate):
    from datetime import datetime
    if 'date' in df.columns:
        x = pd.DataFrame(df.date.unique(), columns=['date'])
    else:
        # print("Checking for date in index...")
        x = pd.DataFrame(df.index.unique(), columns=['date'])
        
    x['idate'] = idate
    x['diff'] = -999
    for i in range(0, x.shape[0]):
        # x.loc[i, 'diff'] = (datetime.datetime.strptime(x['idate'][i] , '%Y-%m-%d') - datetime.datetime.strptime(x['date'][i] , '%Y-%m-%d')).days
        if not isinstance(x['idate'][i], str):
            x.loc[i, 'idate'] = str(x['idate'][i])
        try:
            x.loc[i, 'diff'] = (datetime.strptime(x['idate'][i] , '%Y-%m-%d') - datetime.strptime(x['date'][i] , '%Y-%m-%d')).days
        except:
            # print("Date format is %Y%m%d")
            x.loc[i, 'diff'] = (datetime.strptime(x['idate'][i] , '%Y%m%d') - datetime.strptime(x['date'][i] , '%Y%m%d')).days
    # print(x)
    if sum(x['diff'] > 0)==0:
        ix = np.where(x['diff']==max(x['diff']))
        # print("WARNING: date in the future...")
    else:
        ix = np.where(x.loc[x['diff']>=0]==min(x.loc[x['diff']>=0,'diff']))
    # print(x.iloc[ix])
    return x.iloc[ix[0]]

def std_robust(x):
    std = np.median(np.abs(x - np.median(x))) * 1.4826
    return std

# def get_benchmark_weights(x_input):
#     data_items = {
#         'Name': bq.data.name()['VALUE'],
#         'Weights': bq.data.id()['WEIGHTS']
#     }
#     dates_to = x_input.get('backtest').get('dates_to')

#     df_weights = pd.DataFrame()
#     for idate in dates_to:
#         universe = bq.univ.members(x_input.get('backtest').get('universe'), dates=idate)#, start=start, end=end)  # dates=start

#         request = bql.Request(universe, data_items)
#         response = bq.execute(request)

#         df = pd.concat([data_item.df() for data_item in response], axis=1)
#         df.reset_index(drop=False, inplace=True)
#         df.insert(0, 'date', idate)
#         df.columns = ['date', 'sid', 'name', 'wgt']
#         df_weights = pd.concat([df_weights, df])

#     df_weights['date'] = df_weights['date'].astype(str)    
#     df_weights['date'] = df_weights['date'].map(lambda x: x.replace('-','') if len(x)==10 else x)    
#     df_weights['wgt'] /= 100
#     df_weights.reset_index(drop=True, inplace=True)
    
#     return df_weights

if __name__ == "__main__":
    # Sample dates
    sample_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    # Get rebalancing dates for different frequencies
    month_end_dates = get_rebalance_days(sample_dates, frequency='month_end')
    quarter_end_dates = get_rebalance_days(sample_dates, frequency='quarter_end')
    year_end_dates = get_rebalance_days(sample_dates, frequency='year_end')
    daily_dates = get_rebalance_days(sample_dates, frequency='daily')
    
    # Custom rebalancing dates
    custom_dates = ['2023-03-15', '2023-06-15', '2023-09-15', '2023-12-15']
    custom_rebalance_dates = get_rebalance_days(sample_dates, frequency='custom', custom_dates=custom_dates)


