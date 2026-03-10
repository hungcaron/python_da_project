import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import numpy as np

class ForecastAnalysis:

    def __init__(self, df, chart_dir):
        self.df = df.copy()
        self.chart_dir = Path(chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.relative_chart_path = self.chart_dir.name

    # --------------------------------------------------
    # 1. Chuẩn bị dữ liệu time series
    # --------------------------------------------------
    def prepare_time_series(
        self,
        date_col='Order Date',
        value_col='Sales',
        freq='M'
    ) -> pd.DataFrame:

        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        ts = (
            df
            .set_index(date_col)
            .resample(freq)[value_col]
            .sum()
            .reset_index()
        )

        return ts

    # --------------------------------------------------
    # 2. Forecast bằng linear trend
    # --------------------------------------------------
    def linear_forecast(
    self,
    ts: pd.DataFrame,
    freq: str,
    periods: int = 4
    ) -> pd.DataFrame:

        ts = ts.copy()
        ts['t'] = np.arange(len(ts))

        value_col = ts.columns[1]
        date_col = ts.columns[0]
        # Fit linear trend
        coef = np.polyfit(ts['t'], ts[value_col], 1)
        ts['trend'] = coef[0] * ts['t'] + coef[1]

        # -----------------------
        # FUTURE PERIODS (THEO QUÝ / NĂM)
        # -----------------------
        last_period = ts[date_col].dt.to_period(freq).iloc[-1]

        future_rows = []
        for i in range(1, periods + 1):
            future_period = last_period + i
            future_rows.append({
            date_col: future_period.to_timestamp(),
            value_col: np.nan,
            't': ts['t'].max() + i,
            'trend': coef[0] * (ts['t'].max() + i) + coef[1]
            })

        future_df = pd.DataFrame(future_rows)

        return pd.concat([ts, future_df], ignore_index=True)
    # --------------------------------------------------
    # 3. Vẽ chart forecast
    # --------------------------------------------------
    def plot_forecast(
    self,
    forecast_df: pd.DataFrame,
    freq: str,
    title: str,
    filename: str
    ) -> str:

        date_col = forecast_df.columns[0]
        value_col = forecast_df.columns[1]

        df = forecast_df.copy()
        df['Period'] = df[date_col].dt.to_period(freq).astype(str)
        plt.figure(figsize=(12,6))

        plt.plot(df['Period'], df[value_col], marker='o', label='Actual')
        plt.plot(df['Period'], df['trend'], linestyle='--', label='Trend / Forecast')

        plt.title(title, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        filepath = self.chart_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()

        return f"{self.relative_chart_path}/{filename}"
    # --------------------------------------------------
    # 4. Narrative & summary
    # --------------------------------------------------
    def forecast_summary(
    self,
    forecast_df: pd.DataFrame,
    freq: str,
    periods: int
    ) -> Dict:

        hist = forecast_df.iloc[:-periods]
        future = forecast_df.iloc[-periods:]

        avg_hist = hist.iloc[:,1].mean()
        avg_future = future['trend'].mean()
        growth_rate = ((avg_future - avg_hist) / avg_hist) * 100 if avg_hist else 0

        unit = "quý" if freq == "Q" else "năm" if freq == "Y" else "kỳ"

        narrative = (
        f"Dựa trên xu hướng doanh thu trong quá khứ, "
        f"dự báo cho thấy doanh thu trong <b>{periods} {unit}</b> tiếp theo "
        f"có xu hướng <b>{'tăng' if growth_rate > 0 else 'giảm'}</b> "
        f"khoảng <b>{abs(growth_rate):.1f}%</b>. "
        f"Dự báo này mang <b>tính định hướng và hỗ trợ lập kế hoạch trung hạn</b>."
        )

        return {
            "forecast_period": f"Next {periods} {unit}",
            "expected_growth_rate": round(growth_rate, 2),
            "narrative": narrative
        }
    # --------------------------------------------------
    # 5. Run all
    # --------------------------------------------------
    def run_all(
    self,
    date_col='Order Date',
    value_col='Sales',
    freq='Q',
    periods: int = 4
    ) -> Dict:

        ts = self.prepare_time_series(
            date_col=date_col,
            value_col=value_col,
            freq=freq
        )
        forecast_df = self.linear_forecast(
            ts,
            freq=freq,
            periods=periods
        )
        chart_path = self.plot_forecast(
        forecast_df,
        freq=freq,
        title='Sales Forecast (Quarterly Trend)',
        filename=f'sales_forecast_{freq}.png'
        )

        summary = self.forecast_summary(
            forecast_df,
            freq=freq,
            periods=periods
        )
        return {
            "forecast_summary": summary,
            "forecast_chart": {
                "chart_path": chart_path
            },
            "forecast_table": forecast_df
        }