# security_compare.py
from typing import Dict, List, Optional, Union, Tuple, Literal, Any, Sequence
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from datetime import date, timedelta #, datetime
from functools import lru_cache
from scipy import stats
from enum import Enum
from decimal import Decimal
import pandas as pd
import numpy as np
import plotly.express as px

class FactorExposureComparer:
    """
    Compute & visualize daily factor exposure comparisons between two weight columns
    (e.g., portfolio vs. benchmark) in a security-level dataframe.

    Assumptions:
      - One row per security per date.
      - Factor values are per-security exposures (standardized or raw).
      - Weights are per-security weights for each side (portfolio and benchmark).
    """

    def __init__(
        self,
        weight_col_port: str = "wgt_port",
        weight_col_bench: str = "wgt_bench",
        factor_cols: Optional[List[str]] = None,
        price_col: str = "price",
        date_col: str = "date",
        id_col: str = "sid",
        normalize_weights: bool = True,
        min_weight_sum: float = 1e-12,
    ):
        self.w_port = weight_col_port
        self.w_bench = weight_col_bench
        self.factor_cols = factor_cols  # if None, inferred as columns to the right of price_col
        self.price_col = price_col
        self.date_col = date_col
        self.id_col = id_col
        self.normalize_weights = normalize_weights
        self.min_weight_sum = min_weight_sum

    # ---------- Internals ----------

    def _infer_factors(self, df: pd.DataFrame) -> List[str]:
        if self.factor_cols is not None:
            return [c for c in self.factor_cols if c in df.columns]
        if self.price_col not in df.columns:
            # Fall back to: everything that isn’t metadata/weights gets treated as a factor
            meta = {self.date_col, self.id_col, self.w_port, self.w_bench, self.price_col}
            return [c for c in df.columns if c not in meta]
        # Use all columns to the right of price
        cols = list(df.columns)
        try:
            idx = cols.index(self.price_col)
            return cols[idx + 1 :]
        except ValueError:
            meta = {self.date_col, self.id_col, self.w_port, self.w_bench, self.price_col}
            return [c for c in df.columns if c not in meta]

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.date_col in out.columns:
            out[self.date_col] = pd.to_datetime(out[self.date_col])
        # Ensure weights exist, fill missing with 0
        for w in [self.w_port, self.w_bench]:
            if w not in out.columns:
                raise KeyError(f"Weight column '{w}' not found.")
            out[w] = out[w].fillna(0.0)
        return out

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.normalize_weights:
            return df
        def _safe_norm(group, col):
            s = group[col].sum()
            if abs(s) < self.min_weight_sum:
                return group[col]  # avoid division by ~0; leave as is
            return group[col] / s

        df = df.copy()
        df[self.w_port] = df.groupby(self.date_col, observed=True, sort=False)[self.w_port].transform(
            lambda x: _safe_norm(x.to_frame(), self.w_port)
        )
        df[self.w_bench] = df.groupby(self.date_col, observed=True, sort=False)[self.w_bench].transform(
            lambda x: _safe_norm(x.to_frame(), self.w_bench)
        )
        return df

    # ---------- Public API ----------

    def compute_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns long-form DataFrame with columns:
          date, factor, port_exposure, bench_exposure, active_exposure
        """
        df = self._prep(df)
        # df = self._normalize(df)
        factors = self._infer_factors(df)
        if not factors:
            raise ValueError("No factor columns found to analyze.")

        # Melt factor columns to long per security/date then aggregate with weights
        long = df[[self.date_col, self.id_col, self.w_port, self.w_bench] + factors]\
            .melt(id_vars=[self.date_col, self.id_col, self.w_port, self.w_bench],
                  value_vars=factors,
                  var_name="factor", value_name="exposure")

        # Weighted averages per date & factor
        def _weighted(group, wcol):
            w = group[wcol].fillna(0.0).to_numpy()
            x = group["exposure"].astype(float).fillna(0.0).to_numpy()
            sw = w.sum()
            return np.nan if abs(sw) < self.min_weight_sum else float(np.dot(w, x) / sw)

        agg = (
            long.groupby([self.date_col, "factor"], observed=True, sort=False)
                .apply(lambda g: pd.Series({
                    "port_exposure": _weighted(g, self.w_port),
                    "bench_exposure": _weighted(g, self.w_bench),
                }))
                .reset_index()
        )
        agg["active_exposure"] = agg["port_exposure"] - agg["bench_exposure"]
        return agg.sort_values([self.date_col, "factor"]).reset_index(drop=True)

    def weight_sums(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns per-date sums of weights for portfolio and benchmark and their difference.
        """
        df = self._prep(df)
        sums = (
            df.groupby(self.date_col, observed=True, sort=False)[[self.w_port, self.w_bench]]
              .sum()
              .rename(columns={self.w_port: "port_weight_sum", self.w_bench: "bench_weight_sum"})
              .reset_index()
        )
        sums["weight_sum_diff"] = sums["port_weight_sum"] - sums["bench_weight_sum"]
        return sums

    def summary(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Produces:
          - exposures_summary: per-factor stats of active exposures (mean, std, min, max)
          - weights_by_date: per-date weight sums & difference
          - weights_summary: overall stats for weight sums & difference
        """
        exposures = self.compute_exposures(df)

        exposures_summary = (
            exposures.groupby("factor", observed=True, sort=False)["active_exposure"]
            .agg(active_mean="mean", active_std="std", active_min="min", active_max="max")
            .reset_index()
            .sort_values("factor")
        )

        weights_by_date = self.weight_sums(df)
        weights_summary = weights_by_date[["port_weight_sum", "bench_weight_sum", "weight_sum_diff"]].agg(
            ["mean", "std", "min", "max"]
        ).rename_axis("stat").reset_index()

        return {
            "exposures_summary": exposures_summary,
            "weights_by_date": weights_by_date,
            "weights_summary": weights_summary,
        }

    def plot_timeseries(
        self,
        df: pd.DataFrame,
        factors: Optional[List[str]] = None,
        include_active: bool = True,
        facet: bool = True,
        height: int = 500,
    ):
        """
        Returns a Plotly Express line figure comparing portfolio vs benchmark for the chosen factors.
        If include_active=True, adds an 'Active' series as well.
        """
        exposures = self.compute_exposures(df)
        if factors:
            exposures = exposures[exposures["factor"].isin(factors)]
        if exposures.empty:
            raise ValueError("No exposures to plot for the specified factors.")

        # Build long format for plotting
        series = [
            exposures[[self.date_col, "factor", "port_exposure"]].rename(columns={"port_exposure": "value"})
                .assign(series="Portfolio"),
            exposures[[self.date_col, "factor", "bench_exposure"]].rename(columns={"bench_exposure": "value"})
                .assign(series="Benchmark"),
        ]
        if include_active:
            series.append(
                exposures[[self.date_col, "factor", "active_exposure"]].rename(columns={"active_exposure": "value"})
                    .assign(series="Active")
            )
        plot_df = pd.concat(series, ignore_index=True)

        if facet and plot_df["factor"].nunique() > 1:
            fig = px.line(
                plot_df,
                x=self.date_col, y="value",
                color="series",
                facet_col="factor",
                facet_col_wrap=4,
                markers=True,
                height=height + 80,
                title="Factor Exposure Comparison: Portfolio vs Benchmark"
            )
        else:
            fig = px.line(
                plot_df,
                x=self.date_col, y="value",
                color="series",
                line_group="factor",
                markers=True,
                height=height,
                title="Factor Exposure Comparison: Portfolio vs Benchmark"
            )
        fig.update_layout(legend_title_text="")
        return fig


@dataclass
class ComparisonRow:
    comparison_table: str   # e.g., "Index Profile", "Trailing Returns"
    field_name: str         # e.g., "1 Year"
    field_expr: str         # BQL expression used (for lineage/audit)
    lhs_value: Optional[float | str]  # left security
    rhs_value: Optional[float | str]  # right security


class ComparisonResult(BaseModel):
    """Tabular + dict views for downstream display/export."""
    table: pd.DataFrame
    as_dict: Dict[str, Dict[str, Dict[str, object]]]  # category -> field_name -> {"lhs":..,"rhs":..,"expr":..}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BQLSecurityComparator(BaseModel):
    """
    Build a side-by-side comparison table for two securities/indices
    using Bloomberg BQL API, matching the screenshot sections:
    - Index Profile
    - Trailing Returns
    - Trailing Risk
    - Fundamental Metrics
    - Correlation
    - Tracking Error
    """
    lhs: str = Field(..., description="Left/security A ticker, e.g. 'BMIDG Index' or 'SPX Index'")
    rhs: str = Field(..., description="Right/security B ticker")
    currency: str = Field('USD', description="Reporting currency for fundamentals/market cap")
    # trailing windows in years for returns/risk/correlation/TE
    trailing_years: Tuple[int, int, int, int] = Field((1, 3, 5, 10), description="Windows in years")
    # frequency for time-series retrieval; we use monthly for stability
    frequency: Literal['m'] = Field('m', description="Return series frequency (fixed to monthly)")
    # last N years of history to fetch (buffered above the max window)
    history_years: int = Field(12, description="History retrieved to cover max trailing window + buffer")
    # optional override for benchmark used in TE; if None, TE is computed lhs vs rhs and rhs vs lhs
    benchmark: Optional[str] = Field(None, description="Optional benchmark ticker for Tracking Error")
    # attach an already-constructed bql.Service
    bq: object = Field(..., description="Bloomberg bql.Service instance")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('lhs', 'rhs')
    @classmethod
    def _not_empty(cls, v: str) -> str:
        v = (v or '').strip()
        if not v:
            raise ValueError("Security identifier must be non-empty.")
        return v

    # ------------- Public API -------------

    def build(self) -> ComparisonResult:
        """
        Execute BQL queries + local calculations and return a ComparisonResult
        with a DataFrame shaped like the screenshot.
        """
        # import pdb; pdb.set_trace()
        # 1) Static "Index Profile"
        profile_rows = self._fetch_index_profile()

        # 2) Trailing Returns (annualized total return, compounded from monthly)
        # trailing_return_rows, lhs_mrets, rhs_mrets = self._compute_trailing_returns()
        trailing_return_rows = self._get_cumulative_total_returns()

        # 3) Trailing Risk (annualized std dev of monthly returns)
        # trailing_risk_rows = self._compute_trailing_risk(lhs_mrets, rhs_mrets)
        trailing_risk_rows = self._get_total_risk()

        # 4) Fundamental Metrics (spot)
        fundamentals_rows = self._fetch_fundamentals()

        # Get Monthly Returns
        lhs_mrets = self._get_monthly_total_returns(self.lhs)
        rhs_mrets = self._get_monthly_total_returns(self.rhs)

        # 5) Correlation (rolling window correlation on monthly returns)
        corr_rows = self._compute_correlation(lhs_mrets, rhs_mrets)

        # 6) Tracking Error (monthly active return std * sqrt(12))
        te_rows = self._compute_tracking_error(lhs_mrets, rhs_mrets)

        # Assemble rows in the same order as the screenshot
        rows: List[ComparisonRow] = (
            profile_rows
            + trailing_return_rows
            + trailing_risk_rows
            + fundamentals_rows
            + corr_rows
            + te_rows
        )

        df = self._rows_to_table(rows)
        as_dict = self._rows_to_dict(rows)
        return ComparisonResult(table=df, as_dict=as_dict)

    # ------------- BQL helpers -------------

    def _fetch_index_profile(self) -> List[ComparisonRow]:
        """
        Pull: Name, Index Provider, Index Market Cap, Member Count, History Start Date.
        """
        from bql import Request  # type: ignore
        
        name = bq.data.name();
        index_provider = bq.data.index_provider();
        history_start_dt = bq.data.history_start_dt();
        mkt_cap = bq.data.cur_mkt_cap();
        exprs = {'name':name, 'index_provider':index_provider, 'history_start_dt':history_start_dt, 'mkt_cap':mkt_cap};
        req = bql.Request(self.lhs, exprs);
        res = bq.execute(req);

        req_lhs = Request(self.lhs, exprs)
        req_rhs = Request(self.rhs, exprs)
        res_lhs = self.bq.execute(Request(self.lhs, exprs))
        res_rhs = self.bq.execute(Request(self.rhs, exprs))

        lhs_df = [i.df() for i in res_lhs]
        rhs_df = [i.df() for i in res_rhs]

        rows: List[ComparisonRow] = []
        
        for label, expr in exprs.items():
            rows.append(ComparisonRow(
                comparison_table="Index Profile",
                field_name=label,
                field_expr=expr,
                lhs_value=self._extract_value_from_dfs(lhs_df, label),
                rhs_value=self._extract_value_from_dfs(rhs_df, label)
            ))
        return rows

    def _get_monthly_total_returns(self, sid: str) -> pd.Series:
        """
        Return a monthly total return series as pandas Series indexed by month-end date.
        """
        end_dt = pd.Timestamp.today().normalize()
        start_dt = end_dt - pd.DateOffset(years=self.history_years)
        rng = bq.func.range(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

        # Using Bloomberg's total return series (gross dividends), monthly frequency
        series = bq.data.RETURN_SERIES(calc_interval=rng, frq=self.frequency)
        req = bql.Request([sid], {"return_series": series})
        res = self.bq.execute(req)[0].df()

        # Expected columns: DATE, ID, return_series
        s = (res
             .rename(columns={"DATE": "date", "ID": "sid"})
             .set_index("date")["return_series"]
             .sort_index())
        s.index = pd.to_datetime(s.index).to_period('M').to_timestamp('M')
        s = s.dropna()
        return s

    def _get_cumulative_total_returns(self) -> pd.Series:
        """
        Return cumulative total return values
        """
        #from bql import Request, func, data 
        end_dt = pd.Timestamp.today().normalize()
        start_1y = end_dt - pd.DateOffset(years=1);
        start_3y = end_dt - pd.DateOffset(years=3);
        start_5y = end_dt - pd.DateOffset(years=5);
        start_10y = end_dt - pd.DateOffset(years=10)
        rng_1y = bq.func.range(start_1y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"));
        rng_3y = bq.func.range(start_3y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"));
        rng_5y = bq.func.range(start_5y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"));
        rng_10y = bq.func.range(start_10y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        
        # Using Bloomberg's total return series (gross dividends), monthly frequency
        series_1y = bq.data.TOTAL_RETURN(calc_interval=rng_1y);
        series_3y = bq.data.TOTAL_RETURN(calc_interval=rng_3y);
        series_5y = bq.data.TOTAL_RETURN(calc_interval=rng_5y);
        series_10y = bq.data.TOTAL_RETURN(calc_interval=rng_10y)
        
        exprs = {"return_1y": series_1y, "return_3y": series_3y,"return_5y": series_5y,"return_10y": series_10y}
        req_lhs = bql.Request(self.lhs, exprs);
        req_rhs = bql.Request(self.rhs, exprs)
        
        res_lhs = self.bq.execute(req_lhs);
        list_lhs = [i.df() for i in res_lhs]
        
        res_rhs = self.bq.execute(req_rhs);
        list_rhs = [i.df() for i in res_rhs]
        
        rows: List[ComparisonRow] = []
        for label, expr in exprs.items():
            # rows.append([label, expr, self.lhs, self._extract_value_from_dfs(list_lhs, label)]);
            # rows.append([label, expr, self.rhs, self._extract_value_from_dfs(list_rhs, label)])
            rows.append(ComparisonRow(
                comparison_table="Trailing Returns",
                field_name=label,
                field_expr=expr,
                lhs_value=self._extract_value_from_dfs(list_lhs, label),
                rhs_value=self._extract_value_from_dfs(list_rhs, label)
            ))
            
        # df = pd.DataFrame(rows)
        # df.columns = ['period','expr','value']
        return rows

    def _get_total_risk(self) -> pd.Series:
        """
        Return cumulative total return values
        """
        #from bql import Request, func, data 
        end_dt = pd.Timestamp.today().normalize()
        start_1y = end_dt - pd.DateOffset(years=1);
        start_3y = end_dt - pd.DateOffset(years=3);
        start_5y = end_dt - pd.DateOffset(years=5);
        start_10y = end_dt - pd.DateOffset(years=10)
        rng_1y = bq.func.range(start_1y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"));
        rng_3y = bq.func.range(start_3y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"));
        rng_5y = bq.func.range(start_5y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"));
        rng_10y = bq.func.range(start_10y.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        
        # Using Bloomberg's total return series (gross dividends), monthly frequency
        series_1y = bq.data.VOLATILITY(calc_interval=rng_1y);
        series_3y = bq.data.VOLATILITY(calc_interval=rng_3y);
        series_5y = bq.data.VOLATILITY(calc_interval=rng_5y);
        series_10y = bq.data.VOLATILITY(calc_interval=rng_10y)
        
        exprs = {"volatility_1y": series_1y, "volatility_3y": series_3y,"volatility_5y": series_5y,"volatility_10y": series_10y}
        req_lhs = bql.Request(self.lhs, exprs);
        req_rhs = bql.Request(self.rhs, exprs)
        
        res_lhs = self.bq.execute(req_lhs);
        list_lhs = [i.df() for i in res_lhs]
        
        res_rhs = self.bq.execute(req_rhs);
        list_rhs = [i.df() for i in res_rhs]
        
        rows: List[ComparisonRow] = []
        for label, expr in exprs.items():
            # rows.append([label, expr, self.lhs, self._extract_value_from_dfs(list_lhs, label)]);
            # rows.append([label, expr, self.rhs, self._extract_value_from_dfs(list_rhs, label)])
            rows.append(ComparisonRow(
                comparison_table="Trailing Risk",
                field_name=label,
                field_expr=expr,
                lhs_value=self._extract_value_from_dfs(list_lhs, label),
                rhs_value=self._extract_value_from_dfs(list_rhs, label)
            ))
        # df = pd.DataFrame(rows)
        # df.columns = ['period','expr','value']
        return rows

    # ------------- Calculations -------------

    def _compute_trailing_returns(self) -> Tuple[List[ComparisonRow], pd.Series, pd.Series]:
        """
        Annualized trailing returns for 1/3/5/10 yr windows from monthly returns.
        """
        lhs = self._get_monthly_total_returns(self.lhs)
        rhs = self._get_monthly_total_returns(self.rhs)

        def ann_return(s: pd.Series, years: int) -> float:
            if s.empty:
                return np.nan
            n = years * 12
            s_cut = s.iloc[-n:] if len(s) >= n else s
            if s_cut.empty:
                return np.nan
            g = (1.0 + s_cut).prod()
            ann = g ** (12 / len(s_cut)) - 1.0
            return float(ann)

        rows: List[ComparisonRow] = []
        for y in self.trailing_years:
            expr = f"last(cumprod(last(return_series(calc_interval={y}y, frq=m), {y*12}) + 1) - 1).value"
            rows.append(ComparisonRow(
                comparison_table="Trailing Returns",
                field_name=f"{y} Year",
                field_expr=expr,
                lhs_value=ann_return(lhs, y),
                rhs_value=ann_return(rhs, y)
            ))
        return rows, lhs, rhs

    def _extract_value_from_dfs(self, dfs, label):
        """
        Extracts the value for a given column label from a list of single-row DataFrames.
        If found, returns the value (coerced to float if numeric), else np.nan.
        Dates remain strings by default.
        """
        for df in dfs:
            if label in df.columns:
                value = df[label].iloc[0]
                
                # If it's already a date (pd.Timestamp), convert to string
                if isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                    return str(value.date()) if hasattr(value, "date") else str(value)
                
                # Try coercing to float if possible
                try:
                    return float(value)
                except (ValueError, TypeError):
                    # If looks like a date string (YYYY-MM-DD), leave as string
                    if isinstance(value, str) and pd.api.types.is_datetime64_any_dtype(pd.Series([value])):
                        return value
                    return value
        return np.nan
    
    def _compute_trailing_risk(self, lhs: pd.Series, rhs: pd.Series) -> List[ComparisonRow]:
        """
        Annualized standard deviation of monthly returns over trailing windows.
        """
        def ann_vol(s: pd.Series, years: int) -> float:
            if s.empty:
                return np.nan
            n = years * 12
            s_cut = s.iloc[-n:] if len(s) >= n else s
            if len(s_cut) < 3:
                return np.nan
            return float(s_cut.std(ddof=1) * np.sqrt(12))

        rows: List[ComparisonRow] = []
        for y in self.trailing_years:
            expr = f"round(std(last(return_series(calc_interval={y}y, frq=m), {y*12})) * sqrt(12), 4)"
            rows.append(ComparisonRow(
                comparison_table="Trailing Risk",
                field_name=f"{y} Year",
                field_expr=expr,
                lhs_value=ann_vol(lhs, y),
                rhs_value=ann_vol(rhs, y)
            ))
        return rows

    def _fetch_fundamentals(self) -> List[ComparisonRow]:
        """
        Snapshot fundamentals (latest): P/E, P/B, P/S, ROE.
        """

        exprs = {
            "price_to_earnings": bq.data.pe_ratio(),
            "price_to_book": bq.data.PX_TO_BOOK_RATIO(),
            "price_to_sales": bq.data.px_to_sales_ratio(),
            "roe": bq.data.RETURN_COM_EQY()
        }

        req_lhs = bql.Request(self.lhs, exprs);
        req_rhs = bql.Request(self.rhs, exprs)
        
        res_lhs = self.bq.execute(req_lhs);
        list_lhs = [i.df() for i in res_lhs]
        
        res_rhs = self.bq.execute(req_rhs);
        list_rhs = [i.df() for i in res_rhs]
        
        rows: List[ComparisonRow] = [];
        for label, expr in exprs.items():
            # rows.append([label, expr, self.lhs, self._extract_value_from_dfs(list_lhs, label)]);
            # rows.append([label, expr, self.rhs, self._extract_value_from_dfs(list_rhs, label)])
            rows.append(ComparisonRow(
                comparison_table="Fundamental Data",
                field_name=label,
                field_expr=expr,
                lhs_value=self._extract_value_from_dfs(list_lhs, label),
                rhs_value=self._extract_value_from_dfs(list_rhs, label))
                       )
        return rows

    def _compute_correlation(self, lhs: pd.Series, rhs: pd.Series) -> List[ComparisonRow]:
        """
        Trailing correlation between lhs & rhs monthly returns for 3/5/10y.
        """
        def corr(s1: pd.Series, s2: pd.Series, years: int) -> float:
            n = years * 12
            s1c = s1.iloc[-n:]
            s2c = s2.iloc[-n:]
            idx = s1c.index.intersection(s2c.index)
            if len(idx) < 3:
                return np.nan
            return float(s1c.loc[idx].corr(s2c.loc[idx]))

        rows: List[ComparisonRow] = []
        for y in [3, 5, 10]:
            expr = f"last(return_series(calc_interval={y}y, frq=m), {y*12}); corr(lhs, rhs)"
            rows.append(ComparisonRow(
                comparison_table="Correlation",
                field_name=f"{y} Year",
                field_expr=expr,
                lhs_value=corr(lhs, rhs, y),
                rhs_value=corr(rhs, lhs, y)  # symmetric, but we keep two columns for layout parity
            ))
        return rows

    def _compute_tracking_error(self, lhs: pd.Series, rhs: pd.Series) -> List[ComparisonRow]:
        """
        Tracking Error = std dev of active monthly returns * sqrt(12)
        If a benchmark is provided, compute TE of each security vs benchmark.
        Otherwise compute mutual TE (lhs vs rhs and rhs vs lhs), which are identical.
        """
        def te(series: pd.Series, bench: pd.Series, years: int) -> float:
            n = years * 12
            s1 = series.iloc[-n:]
            s2 = bench.iloc[-n:]
            idx = s1.index.intersection(s2.index)
            if len(idx) < 3:
                return np.nan
            active = s1.loc[idx] - s2.loc[idx]
            return float(active.std(ddof=1) * np.sqrt(12))

        # Choose TE pairs
        if self.benchmark:
            bench = self._get_monthly_total_returns(self.benchmark)
            left_te = lambda y: te(self._get_monthly_total_returns(self.lhs), bench, y)
            right_te = lambda y: te(self._get_monthly_total_returns(self.rhs), bench, y)
            expr_stub = "[lhs/rhs] vs benchmark monthly std * sqrt(12)"
        else:
            # mutual TE (symmetric)
            bench = rhs
            left_te = lambda y: te(lhs, rhs, y)
            right_te = lambda y: te(rhs, lhs, y)
            expr_stub = "std(last(active_returns(lhs - rhs), N)) * sqrt(12)"

        rows: List[ComparisonRow] = []
        for y in [3, 5, 10]:
            expr = f"{expr_stub} over {y}y"
            rows.append(ComparisonRow(
                comparison_table="Tracking Error",
                field_name=f"{y} Year",
                field_expr=expr,
                lhs_value=left_te(y),
                rhs_value=right_te(y)
            ))
        return rows

    # ------------- Formatting helpers -------------

    def _rows_to_table(self, rows: List[ComparisonRow]) -> pd.DataFrame:
        # Assemble into a single DataFrame shaped like the screenshot
        data = []
        for r in rows:
            data.append([r.comparison_table, r.field_name, r.field_expr, r.lhs_value, r.rhs_value])

        df = pd.DataFrame(
            data,
            columns=["Comparison Table", "Field Name", "Field", self.lhs, self.rhs]
        )

        # Friendly formatting: percentages where applicable
        percent_categories = {"Trailing Returns", "Trailing Risk", "Correlation", "Tracking Error"}
        for cat in percent_categories:
            mask = df["Comparison Table"].eq(cat)
            for col in [self.lhs, self.rhs]:
                df.loc[mask, col] = pd.to_numeric(df.loc[mask, col], errors="coerce")

        return df

    def _rows_to_dict(self, rows: List[ComparisonRow]) -> Dict[str, Dict[str, Dict[str, object]]]:
        out: Dict[str, Dict[str, Dict[str, object]]] = {}
        for r in rows:
            out.setdefault(r.comparison_table, {})
            out[r.comparison_table][r.field_name] = {
                "lhs": r.lhs_value,
                "rhs": r.rhs_value,
                "expr": r.field_expr
            }
        return out


class IndexPair(BaseModel):
    """
    A single (benchmark → comparison) mapping.
    Example: bench=Universe.NDX, comp=Universe.B100Q
    """
    bench: Universe
    comp: Universe
    country: Country = Field(description="Country inferred from universes.")

    model_config = ConfigDict(use_enum_values=False, frozen=True)

    @model_validator(mode="after")
    def _validate_country(self) -> "IndexPair":
        b = universe_country(self.bench)
        c = universe_country(self.comp)
        if self.country != b or self.country != c:
            raise ValueError(
                f"Country mismatch: bench={self.bench}({b}), "
                f"comp={self.comp}({c}), pair.country={self.country}"
            )
        return self


class IndexMappingRegistry(BaseModel):
    """
    Registry for index pairs with helpful query methods.
    Keys: benchmark Universe; Value: IndexPair.
    """
    pairs: Dict[Universe, IndexPair] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=False)

    # ---- Constructors --------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> "IndexMappingRegistry":
        """
        Seed with your example mapping(s). Extend as needed.
        """
        reg = cls()
        # Example from your request: bench NDX → comp B100Q
        reg.register_pair(bench=Universe.NDX, comp=Universe.B100Q) # Nasdaq
        reg.register_pair(bench=Universe.RDG, comp=Universe.BMIDG) # Midcap growth
        reg.register_pair(bench=Universe.SPX, comp=Universe.B500) # Large cap
        reg.register_pair(bench=Universe.RLV, comp=Universe.B500VT) # Large cap value
        return reg

    # ---- Mutators ------------------------------------------------------------

    def register_pair(self, bench: Universe, comp: Universe) -> IndexPair:
        """
        Create/overwrite mapping for 'bench' to 'comp'. Validates same-country rule.
        """
        pair = IndexPair(bench=bench, comp=comp, country=universe_country(bench))
        # Will raise if comp is unmapped
        _ = universe_country(comp)
        self.pairs[bench] = pair
        return pair

    def remove_pair(self, bench: Universe) -> None:
        self.pairs.pop(bench, None)

    # ---- Instance queries ----------------------------------------------------

    def get_comparison(self, bench: Universe) -> Optional[Universe]:
        p = self.pairs.get(bench)
        return p.comp if p else None

    def list_pairs(self) -> List[IndexPair]:
        return list(self.pairs.values())

    def list_by_country(self, country: Country) -> List[IndexPair]:
        """
        Instance method: all (bench, comp) pairs for a given country.
        """
        return [p for p in self.pairs.values() if p.country == country]

    def list_for_benchmarks(self, benchmarks: Iterable[Universe]) -> List[IndexPair]:
        return [self.pairs[b] for b in benchmarks if b in self.pairs]

    def as_tuples(self) -> List[Tuple[Universe, Universe, Country]]:
        return [(p.bench, p.comp, p.country) for p in self.list_pairs()]

    # ---- Class-level convenience --------------------------------------------

    @classmethod
    def list_pairs_by_country(cls, country: Country, registry: "IndexMappingRegistry") -> List[IndexPair]:
        """
        Class method version of the country filter (per your ask).
        Keeps the call-site flexible: IndexMappingRegistry.list_pairs_by_country(Country.US, reg)
        """
        return registry.list_by_country(country)

    # ---- Pretty formatting ---------------------------------------------------

    def describe(self) -> str:
        lines = []
        for p in self.list_pairs():
            lines.append(f"[{p.country}] {p.bench.value} → {p.comp.value}  |  "
                         f"{p.bench.description} → {p.comp.description}")
        return "\n".join(lines)
