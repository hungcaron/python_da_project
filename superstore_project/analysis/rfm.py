import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

class RFMAnalysis:

    def __init__(self, df, chart_dir):
        self.df = df.copy()
        self.chart_dir = Path(chart_dir)
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.relative_chart_path = self.chart_dir.name

    # --------------------------------------------------
    # 1. Tính RFM
    # --------------------------------------------------
    def calculate_rfm(
        self,
        customer_col='Customer ID',
        date_col='Order Date',
        monetary_col='Sales'
    ):

        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby(customer_col)
            .agg({
                date_col: lambda x: (snapshot_date - x.max()).days,
                customer_col: 'count',
                monetary_col: 'sum'
            })
            .rename(columns={
                date_col: 'Recency',
                customer_col: 'Frequency',
                monetary_col: 'Monetary'
            })
            .reset_index()
        )

        return rfm

    # --------------------------------------------------
    # 2. Gán điểm RFM
    # --------------------------------------------------
    def score_rfm(self, rfm: pd.DataFrame) -> pd.DataFrame:

        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

        rfm[['R_Score','F_Score','M_Score']] = rfm[['R_Score','F_Score','M_Score']].astype(int)
        rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

        return rfm

    # --------------------------------------------------
    # 3. Phân khúc khách hàng
    # --------------------------------------------------
    def segment_customers(self, rfm: pd.DataFrame) -> pd.DataFrame:

        def segment(score):
            if score >= 13:
                return 'Champions'
            elif score >= 10:
                return 'Loyal Customers'
            elif score >= 7:
                return 'Potential Loyalists'
            elif score >= 5:
                return 'At Risk'
            else:
                return 'Lost'

        rfm['Segment'] = rfm['RFM_Score'].apply(segment)
        return rfm

    # --------------------------------------------------
    # 4. Phân bố phân khúc (chart + narrative)
    # --------------------------------------------------
    def segment_distribution(self, rfm: pd.DataFrame):

        dist = (
            rfm.groupby('Segment')
            .agg({
                'Customer ID': 'count',
                'Monetary': 'sum'
            })
            .rename(columns={
                'Customer ID': 'Customers',
                'Monetary': 'Revenue'
            })
            .reset_index()
            .round(2)   
        )

        # Chart
        # Tô màu kiểu gradient, thấp nhất là cam đậm, cao nhất là steelblue
        colors = {
            'Champions': '#4682B4',
            'Loyal Customers': '#5A9BD5',
            'Potential Loyalists': '#9CC3E6',
            'At Risk': '#F4A261',
            'Lost': '#E76F51'
        }
        
        plt.figure(figsize=(10,6))
        plt.bar(dist['Segment'], dist['Customers'], color=[colors.get(x) for x in dist['Segment']])
        plt.title('Customer Segment Distribution', fontweight='bold')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.tight_layout()

        filename = "rfm_segment_distribution.png"
        filepath = self.chart_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()

        # Narrative
        top_segment = dist.sort_values('Revenue', ascending=False).iloc[0]['Segment']
        top_segment_value = dist.sort_values('Revenue', ascending=False).iloc[0]['Revenue']
        top_segment_percent = (top_segment_value / dist['Revenue'].sum() * 100).round(2)
        bottom_segment = dist.sort_values('Revenue').iloc[0]['Segment']
        bottom_segment_value = dist.sort_values('Revenue').iloc[0]['Revenue']
        bottom_segment_percent = (bottom_segment_value / dist['Revenue'].sum() * 100).round(2)
         
        # Tỷ lệ phần trăm doanh thu từng nhóm so với tổng doanh thu
        total_revenue = dist['Revenue'].sum()
        dist['Revenue_Percent'] = (dist['Revenue'] / total_revenue * 100).round(2)
        # Tìm nhóm At Risk
        at_risk_segment = dist[dist['Segment'] == 'At Risk']
        at_risk_revenue_percent = at_risk_segment['Revenue_Percent'].values[0] if not at_risk_segment.empty else None
       
        
        narrative = (
            f"Phân tích RFM cho thấy nhóm khách hàng <b>{top_segment}</b> "
            f"đóng góp <b>doanh thu lớn nhất</b> với tỷ lệ <b>{top_segment_percent:.2f}%</b>. Việc duy trì và mở rộng nhóm khách hàng "
            f"này là yếu tố then chốt để tăng trưởng bền vững. "
            f"Ngược lại, nhóm <b>{bottom_segment}</b> có đóng góp doanh thu thấp nhất với tỷ lệ <b>{bottom_segment_percent:.2f}%</b>, "
            f"Nhóm này coi như là khách hàng đã mất. "
            f"Nhóm <b>At Risk</b> có đóng góp doanh thu là <b>{at_risk_revenue_percent:.2f}%</b> và cần được xem xét lại các chiến lược tiếp cận hoặc <b>tái kích hoạt</b>. "
            f"Tổng thể, việc phân khúc khách hàng theo RFM cung cấp cái nhìn sâu sắc để <b>tối ưu hóa chiến lược marketing và chăm sóc khách hàng</b>."
            
        )

        return {
            "table": dist,
            "chart_path": f"{self.relative_chart_path}/{filename}",
            "narrative": narrative
        }

    # --------------------------------------------------
    # 5. Tóm tắt RFM
    # --------------------------------------------------
    def rfm_summary(self, rfm: pd.DataFrame):

        return {
            "total_customers": rfm.shape[0],
            "avg_recency": round(rfm['Recency'].mean(), 1),
            "avg_frequency": round(rfm['Frequency'].mean(), 1),
            "avg_monetary": round(rfm['Monetary'].mean(), 0)
        }

    # --------------------------------------------------
    # 6. Run all
    # --------------------------------------------------
    def run_all(self) -> Dict:

        rfm = self.calculate_rfm()
        rfm = self.score_rfm(rfm)
        rfm = self.segment_customers(rfm)

        return {
            "rfm_summary": self.rfm_summary(rfm),
            "segment_distribution": self.segment_distribution(rfm),
            "rfm_table": rfm
        }
