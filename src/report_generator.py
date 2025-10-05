from typing import Dict, List, Optional
from datetime import datetime, date
from io import BytesIO

import pandas as pd
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.graphics.shapes import Drawing, String, PolyLine, Rect


def _format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.2%}"


def _safe_series(values: pd.Series) -> pd.Series:
    if values is None or len(values) == 0:
        return pd.Series(dtype=float)
    return values.dropna()


class SimpleLineChart:
    """Very lightweight line chart for time series using reportlab shapes."""

    def __init__(self, title: str, series_dict: Dict[str, pd.Series], width: float = 7.0 * inch, height: float = 2.5 * inch):
        self.title = title
        self.series_dict = series_dict
        self.width = width
        self.height = height

    def _normalize(self, yvals: List[float], ymin: float, ymax: float, ypad: float) -> List[float]:
        rng = max(ymax - ymin, 1e-9)
        return [((y - ymin) / rng) * (self.height - 2 * ypad) + ypad for y in yvals]

    def create(self) -> Drawing:
        drawing = Drawing(self.width, self.height)
        left_pad = 50
        right_pad = 20
        top_pad = 24
        bottom_pad = 36

        # Flatten values to get bounds
        all_values: List[float] = []
        for s in self.series_dict.values():
            s = _safe_series(s)
            if not s.empty:
                all_values.extend(s.values.tolist())

        if not all_values:
            title_label = String(self.width / 2, self.height / 2, "No data for chart")
            title_label.fontName = 'Helvetica'
            title_label.fontSize = 10
            title_label.textAnchor = 'middle'
            drawing.add(title_label)
            return drawing

        ymin = float(min(all_values))
        ymax = float(max(all_values))

        # Draw title
        title_label = String(self.width / 2, self.height - 12, self.title)
        title_label.fontName = 'Helvetica-Bold'
        title_label.fontSize = 12
        title_label.textAnchor = 'middle'
        drawing.add(title_label)

        # Determine a reference index for x-axis ticks (first non-empty series)
        ref_index = None
        for s in self.series_dict.values():
            s = _safe_series(s)
            if not s.empty:
                ref_index = s.index
                break

        # Plot area width
        inner_w = self.width - left_pad - right_pad

        # Gridlines and axes (y-axis ticks: 5 levels)
        y_ticks = [
            ymin,
            ymin + (ymax - ymin) * 0.25,
            ymin + (ymax - ymin) * 0.50,
            ymin + (ymax - ymin) * 0.75,
            ymax,
        ]
        # Draw horizontal gridlines and y tick labels
        for yv in y_ticks:
            ynorm = self._normalize([yv], ymin, ymax, bottom_pad)[0]
            grid = PolyLine([(left_pad, ynorm), (left_pad + inner_w, ynorm)], strokeColor=colors.HexColor('#E5E7EB'), strokeWidth=0.5)
            drawing.add(grid)
            lbl = String(5, ynorm - 4, f"{yv:.2%}")
            lbl.fontSize = 8
            drawing.add(lbl)

        # X-axis ticks (left, middle, right)
        x_tick_positions = []
        x_tick_labels = []
        if ref_index is not None and len(ref_index) >= 2:
            n = len(ref_index)
            idxs = [0, n // 2, n - 1]
            for i in idxs:
                xt = left_pad + inner_w * (i / max(n - 1, 1))
                x_tick_positions.append(xt)
                label_val = ref_index[i]
                try:
                    label_str = pd.to_datetime(label_val).strftime('%Y-%m-%d')
                except Exception:
                    label_str = str(label_val)
                x_tick_labels.append(label_str)
            # Vertical ticks
            for xt in x_tick_positions:
                vgrid = PolyLine([(xt, bottom_pad), (xt, self.height - top_pad)], strokeColor=colors.HexColor('#F3F4F6'), strokeWidth=0.5)
                drawing.add(vgrid)

        # X-axis labels
        for xt, lbl in zip(x_tick_positions, x_tick_labels):
            xl = String(xt, bottom_pad - 12, lbl)
            xl.fontSize = 7
            xl.textAnchor = 'middle'
            drawing.add(xl)

        # Plot each series as a polyline
        palette = [
            colors.HexColor('#1f77b4'),
            colors.HexColor('#ff7f0e'),
            colors.HexColor('#2ca02c'),
            colors.HexColor('#d62728'),
            colors.HexColor('#9467bd'),
        ]
        for idx, (name, series) in enumerate(self.series_dict.items()):
            s = _safe_series(series)
            if s.empty:
                continue
            # X positions distributed evenly
            xpad = left_pad
            n = len(s)
            if n == 1:
                xs = [xpad + inner_w / 2]
            else:
                xs = [xpad + (inner_w * i / (n - 1)) for i in range(n)]
            ys = self._normalize(s.values.tolist(), ymin, ymax, bottom_pad)
            line = PolyLine(list(zip(xs, ys)), strokeColor=palette[idx % len(palette)], strokeWidth=1.2)
            drawing.add(line)

        # Legend at bottom-left
        legend_x = left_pad
        legend_y = 6
        legend_items = list(self.series_dict.keys())
        for i, name in enumerate(legend_items):
            color = palette[i % len(palette)]
            # color box
            box = Rect(legend_x + i * 120, legend_y, 10, 10, strokeColor=color, fillColor=color)
            drawing.add(box)
            txt = String(legend_x + i * 120 + 14, legend_y + 1, name)
            txt.fontSize = 8
            drawing.add(txt)

        return drawing


class AppReportBuilder:
    """Builds a PDF report summarizing app results across tabs."""

    def __init__(self, title: str = "Equity Factor Analysis Report"):
        self.title = title
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name='ReportTitle', parent=self.styles['Heading1'], fontSize=16, alignment=TA_CENTER, spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader', parent=self.styles['Heading2'], fontSize=12, textColor=colors.white,
            backColor=colors.HexColor('#000000'), leftIndent=4, spaceBefore=10, spaceAfter=6, alignment=TA_LEFT
        ))
        self.styles.add(ParagraphStyle(
            name='SmallGrey', parent=self.styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_RIGHT
        ))

    def _spacer(self, h: float = 0.12 * inch) -> Spacer:
        return Spacer(1, h)

    def _table(self, data: List[List[str]], col_widths: Optional[List[float]] = None) -> Table:
        tbl = Table(data, colWidths=col_widths)
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D97706')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
        ])
        tbl.setStyle(style)
        return tbl

    def _compute_period_returns(self, daily_series: pd.Series, end: Optional[pd.Timestamp] = None) -> Dict[str, float]:
        s = _safe_series(daily_series)
        if s.empty:
            return {'1D': np.nan, 'MTD': np.nan, 'QTD': np.nan, 'YTD': np.nan}
        s = s.sort_index()
        if end is None:
            end = s.index.max()
        end = pd.to_datetime(end)
        one_day_ret = s.loc[end] if end in s.index else np.nan
        # MTD/QTD/YTD cumulative
        def period_start(ts: pd.Timestamp, kind: str) -> pd.Timestamp:
            if kind == 'MTD':
                return ts.replace(day=1)
            if kind == 'QTD':
                q = (ts.month - 1) // 3
                first_month = q * 3 + 1
                return pd.Timestamp(year=ts.year, month=first_month, day=1)
            if kind == 'YTD':
                return pd.Timestamp(year=ts.year, month=1, day=1)
            return ts
        def cum_period(kind: str) -> float:
            start = period_start(end, kind)
            sub = s[(s.index >= start) & (s.index <= end)]
            if sub.empty:
                return np.nan
            return float((sub + 1.0).prod() - 1.0)
        return {
            '1D': float(one_day_ret) if pd.notna(one_day_ret) else np.nan,
            'MTD': cum_period('MTD'),
            'QTD': cum_period('QTD'),
            'YTD': cum_period('YTD'),
        }

    def _factor_summary_section(self, story: List, pure_factor_returns: Optional[pd.DataFrame]):
        story.append(Paragraph("Factor Summary", self.styles['SectionHeader']))
        story.append(self._spacer(0.06 * inch))
        if pure_factor_returns is None or pure_factor_returns.empty:
            story.append(Paragraph("No factor returns available.", self.styles['Normal']))
            return
        # Ensure DateTimeIndex
        df = pure_factor_returns.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        # Summary table
        header = ["Factor", "1D", "MTD", "QTD", "YTD", "Vol (ann)", "Sharpe (ann)"]
        rows = [header]
        for col in df.columns:
            series = df[col].dropna()
            metrics = self._compute_period_returns(series)
            vol_ann = series.std() * np.sqrt(252) if len(series) > 2 else np.nan
            sharpe_ann = (series.mean() * 252) / vol_ann if vol_ann and vol_ann != 0 and pd.notna(vol_ann) else np.nan
            rows.append([
                col.title(),
                _format_pct(metrics['1D']),
                _format_pct(metrics['MTD']),
                _format_pct(metrics['QTD']),
                _format_pct(metrics['YTD']),
                _format_pct(vol_ann),
                f"{sharpe_ann:.2f}" if pd.notna(sharpe_ann) else "-",
            ])
        story.append(self._table(rows, col_widths=[1.5*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.9*inch, 0.9*inch]))
        story.append(self._spacer())
        # Cumulative chart
        cum_df = df.cumsum()
        chart = SimpleLineChart("Cumulative Factor Returns", {c.title(): cum_df[c] for c in cum_df.columns}).create()
        story.append(chart)

    def _te_section(self, story: List, te_returns: Optional[pd.DataFrame], te_weights: Optional[pd.DataFrame], te_meta: Optional[pd.DataFrame]):
        story.append(Paragraph("Tracking Error Optimization", self.styles['SectionHeader']))
        story.append(self._spacer(0.06 * inch))
        if te_returns is None and te_weights is None and te_meta is None:
            story.append(Paragraph("No tracking error optimization results available.", self.styles['Normal']))
            return
        if te_meta is not None and not te_meta.empty:
            # Show simple averages
            cols = [c for c in ['tracking_error', 'optimization_time', 'objective_value'] if c in te_meta.columns]
            header = ["Metric", "Value"]
            rows = [header]
            if 'tracking_error' in cols:
                rows.append(["Avg Tracking Error", _format_pct(float(te_meta['tracking_error'].mean()))])
            if 'optimization_time' in cols:
                rows.append(["Avg Optimization Time (s)", f"{float(te_meta['optimization_time'].mean()):.2f}s"])
            if 'objective_value' in cols:
                rows.append(["Avg Objective Value", f"{float(te_meta['objective_value'].mean()):.4f}"])
            story.append(self._table(rows, col_widths=[2.5*inch, 2.5*inch]))
            story.append(self._spacer())
        if te_weights is not None and not te_weights.empty:
            latest_date = te_weights['date'].max()
            latest = te_weights[te_weights['date'] == latest_date].copy()
            latest = latest.sort_values('weight', ascending=False).head(15)
            if 'ticker' not in latest.columns and 'sid' in latest.columns:
                latest['ticker'] = latest['sid']
            header = ["Ticker", "Weight"]
            rows = [header]
            for _, r in latest.iterrows():
                rows.append([str(r.get('ticker', r.get('sid', ''))), _format_pct(float(r['weight']))])
            story.append(Paragraph(f"Top Holdings (as of {latest_date})", self.styles['Normal']))
            story.append(self._table(rows, col_widths=[2.5*inch, 1.5*inch]))
            story.append(self._spacer())
        if te_returns is not None and not te_returns.empty:
            df = te_returns.copy()
            # Build cumulative series
            cum_port = _safe_series(df['return_opt']).cumsum() if 'return_opt' in df.columns else pd.Series(dtype=float)
            cum_bench = _safe_series(df['return_benchmark']).cumsum() if 'return_benchmark' in df.columns else pd.Series(dtype=float)
            chart = SimpleLineChart("Cumulative Returns: Portfolio vs Benchmark", {
                'Portfolio': cum_port,
                'Benchmark': cum_bench,
            }).create()
            story.append(chart)

    def _uploaded_portfolio_section(self, story: List, portfolio_results: Optional[Dict]):
        story.append(Paragraph("Uploaded Portfolio Analysis", self.styles['SectionHeader']))
        story.append(self._spacer(0.06 * inch))
        if not portfolio_results:
            story.append(Paragraph("No uploaded portfolio analysis available.", self.styles['Normal']))
            return
        # Performance comparison table if present
        perf = portfolio_results.get('comparison', {}).get('performance')
        if isinstance(perf, pd.DataFrame) and not perf.empty:
            latest = perf.tail(1).iloc[0].to_dict()
            header = ["Metric", "Value"]
            rows = [header]
            if 'user_return' in latest:
                rows.append(["Latest User Return", _format_pct(float(latest['user_return']))])
            if 'benchmark_return' in latest:
                rows.append(["Latest Benchmark Return", _format_pct(float(latest['benchmark_return']))])
            if 'active_return_vs_bench' in latest:
                rows.append(["Latest Active vs Benchmark", _format_pct(float(latest['active_return_vs_bench']))])
            story.append(self._table(rows, col_widths=[2.5*inch, 2.5*inch]))
            story.append(self._spacer())
        # Top active positions if available
        weight_analysis = portfolio_results.get('comparison', {}).get('weight_analysis', {})
        latest_cmp = weight_analysis.get('latest_comparison')
        if isinstance(latest_cmp, pd.DataFrame) and not latest_cmp.empty:
            disp = latest_cmp.sort_values('active_vs_benchmark', ascending=False).head(10)
            header = ["SID", "User Wt", "Bench Wt", "Active vs Bench"]
            rows = [header]
            for _, r in disp.iterrows():
                rows.append([
                    str(r.get('sid', '')),
                    _format_pct(float(r.get('user_weight', np.nan))),
                    _format_pct(float(r.get('benchmark_weight', np.nan))),
                    _format_pct(float(r.get('active_vs_benchmark', np.nan))),
                ])
            story.append(Paragraph("Top Active Positions", self.styles['Normal']))
            story.append(self._table(rows, col_widths=[1.8*inch, 1.2*inch, 1.2*inch, 1.4*inch]))
            story.append(self._spacer(0.08 * inch))

            # Bottom 10 underweights (most negative active)
            under = latest_cmp.sort_values('active_vs_benchmark', ascending=True).head(10)
            rows_u = [header]
            for _, r in under.iterrows():
                rows_u.append([
                    str(r.get('sid', '')),
                    _format_pct(float(r.get('user_weight', np.nan))),
                    _format_pct(float(r.get('benchmark_weight', np.nan))),
                    _format_pct(float(r.get('active_vs_benchmark', np.nan))),
                ])
            story.append(Paragraph("Top Underweights vs Benchmark", self.styles['Normal']))
            story.append(self._table(rows_u, col_widths=[1.8*inch, 1.2*inch, 1.2*inch, 1.4*inch]))

    def build_pdf_bytes(
        self,
        pure_factor_returns: Optional[pd.DataFrame] = None,
        te_meta: Optional[pd.DataFrame] = None,
        te_weights: Optional[pd.DataFrame] = None,
        te_returns: Optional[pd.DataFrame] = None,
        uploaded_portfolio_results: Optional[Dict] = None,
    ) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30,
        )
        story: List = []

        # Header
        story.append(Paragraph(self.title, self.styles['ReportTitle']))
        story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), self.styles['SmallGrey']))
        story.append(self._spacer())

        # Sections
        self._factor_summary_section(story, pure_factor_returns)
        story.append(PageBreak())
        self._te_section(story, te_returns, te_weights, te_meta)
        if uploaded_portfolio_results:
            story.append(PageBreak())
            self._uploaded_portfolio_section(story, uploaded_portfolio_results)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()


