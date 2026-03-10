from matplotlib.pylab import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from matplotlib import colors as mcolors


class DescriptiveAnalysis:
    
    def __init__(self, df, chart_dir):
        
        # Khởi tạo lớp phân tích mô tả với DataFrame và thư mục lưu biểu đồ
        self.df = df.copy()
        # Chuyển đổi đường dẫn thành đối tượng Path
        self.chart_dir = Path(chart_dir)
        
        # Tạo thư mục lưu biểu đồ nếu chưa có
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Tính đường dẫn tương đối từ thư mục reports/outputs đến charts
        self.relative_chart_path = self.chart_dir.name  # 'charts'
        
        # Thiết lập style cho biểu đồ
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def sales_summary(self):
        
        # Thống kê tổng quan về bán hàng
        total_sales = self.df['Sales'].sum()
        total_profit = self.df['Profit'].sum()
        total_quantity = self.df['Quantity'].sum()
        total_orders = self.df['Order ID'].nunique()
        total_customers = self.df['Customer ID'].nunique()
        total_products = self.df['Product ID'].nunique()
        average_sale_per_order = self.df.groupby('Order ID')['Sales'].sum().mean()
        average_profit_per_order = self.df.groupby('Order ID')['Profit'].sum().mean()
        average_quantity_per_order = self.df.groupby('Order ID')['Quantity'].sum().mean()
        profit_margin = (total_profit / total_sales) * 100 if total_sales != 0 else 0
        sales_narrative = (
            f"Tổng doanh thu bán hàng là {total_sales:,.0f} với lợi nhuận tổng cộng là {total_profit:,.0f}. "
            f"Tổng số lượng bán ra là {total_quantity:,.0f} sản phẩm thông qua tổng cộng {total_orders:,} đơn hàng từ tổng cộng {total_customers:,} khách hàng khác nhau, "            
            f"với tổng cộng {total_products:,} sản phẩm. "
            f"Mỗi đơn hàng trung bình mang lại doanh thu {average_sale_per_order:,.0f} và lợi nhuận {average_profit_per_order:,.0f}. "
            f"Tỷ suất lợi nhuận trên doanh thu đạt {profit_margin:.2f}%"
        )        
        sales_summary = {
            'total_sales': total_sales,
            'total_profit': total_profit,
            'total_quantity': total_quantity,
            'total_orders': total_orders,
            'total_customers': total_customers,
            'total_products': total_products,
            'average_sale_per_order': float(average_sale_per_order),
            'average_profit_per_order': float(average_profit_per_order),
            'average_quantity_per_order': float(average_quantity_per_order),
            'profit_margin': float(profit_margin),
            # Mô tả dữ liệu bán hàng
            'sales_narrative': sales_narrative
        }
        return sales_summary
    
    # Hàm doanh thu theo danh mục với nhiều tùy chọn
    def sales_by_subcategory(
        self,
        category_col: str = "Sub-Category",
        value_col: str = "Sales",
        top_n: int = 10,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        horizontal: bool = False,
        show_values: bool = True
    ) -> Dict:
        """
        Vẽ biểu đồ doanh thu theo danh mục và trả về chart_path + narrative
        """

        # --------------------------------------------------
        # 1. Group & sort data
        # --------------------------------------------------
        grouped = (
            self.df
            .groupby(category_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        # --------------------------------------------------
        # 2. Plot
        # --------------------------------------------------
        values = grouped[value_col]   # hoặc df_plot['Sales']

        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        # Cắt từ 30% → 100% của colormap Blues
        base_cmap = plt.cm.Blues
        cmap = mcolors.LinearSegmentedColormap.from_list(
        "Blues_truncated",
        base_cmap(np.linspace(0.3, 1.0, 256))
        )

        bar_colors = cmap(norm(values))
        plt.figure(figsize=(12, 8))

        if horizontal:
            plt.barh(grouped[category_col], grouped[value_col], color=bar_colors)
            if show_values:
                for i, v in enumerate(grouped[value_col]):
                    plt.text(v, i, f" {v:,.0f}", va="center")
        else:
            plt.bar(grouped[category_col], grouped[value_col], color=bar_colors)
            plt.xticks(rotation=45, ha="right")
            if show_values:
                for i, v in enumerate(grouped[value_col]):
                    plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

        # --------------------------------------------------
        # 3. Title & labels
        # --------------------------------------------------
        if not title:
            title = f"Top {top_n} {category_col} theo {value_col}"
        plt.title(title, fontweight="bold", fontsize=15, pad=20)
        plt.xlabel(category_col)
        plt.ylabel(value_col)
        plt.ticklabel_format(style="plain", axis="y")
        plt.tight_layout()
        # --------------------------------------------------
        # 4. Save chart
        # --------------------------------------------------
        if filename is None:
            filename = f"sales_by_{category_col.lower()}.png"

        path = self.chart_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        # --------------------------------------------------
        # 5. Narrative
        # --------------------------------------------------
        top_category = grouped.iloc[0][category_col]
        top_value = grouped.iloc[0][value_col]
        bottom_category = grouped.iloc[-1][category_col]
        bottom_value = grouped.iloc[-1][value_col]
        narrative = (
        f"Biểu đồ thể hiện top danh mục có doanh thu cao nhất, trong đó "
        f"{top_category} dẫn đầu với tổng {value_col.lower()} "
        f"đạt {top_value:,.0f}."
        f" Ngược lại, {bottom_category} có {value_col.lower()} thấp nhất "
        f"với {bottom_value:,.0f}. "
        f"Danh mục đầu bảng có doanh thu gấp {top_value / bottom_value:,.2f} lần so với danh mục cuối bảng."
    )

        # --------------------------------------------------
        # 6. Return standardized result
        # --------------------------------------------------
        return {
            "chart_path": f"{self.relative_chart_path}/{filename}",
            "narrative": narrative
        }
    #
    
    def top_cities_by_sales(
        self,
        city_col: str = "City",
        value_col: str = "Sales",
        top_n: int = 10,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        horizontal: bool = True,
        show_values: bool = True
    ) -> Dict:
        """
        Biểu đồ top thành phố có doanh thu cao nhất
        """

        # --------------------------------------------------
        # 1. Group & sort data
        # ------------------------------------------------
        grouped = (
            self.df
            .groupby(city_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        # --------------------------------------------------
        # 2. Plot
        # ------------------------------------------------
        values = grouped[value_col]   # hoặc df_plot['Sales']

        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        # Cắt từ 30% → 100% của colormap Blues
        base_cmap = plt.cm.Blues
        cmap = mcolors.LinearSegmentedColormap.from_list(
        "Blues_truncated",
        base_cmap(np.linspace(0.3, 1.0, 256))
        )

        bar_colors = cmap(norm(values))
        plt.figure(figsize=(12, 8))

        if horizontal:
            plt.barh(grouped[city_col], grouped[value_col], color=bar_colors)
            plt.gca().invert_yaxis()  # City cao nhất nằm trên cùng

            if show_values:
                for i, v in enumerate(grouped[value_col]):
                    plt.text(v, i, f" {v:,.0f}", va="center")
        else:
            plt.bar(grouped[city_col], grouped[value_col], color=bar_colors)
            plt.xticks(rotation=45, ha="right")

            if show_values:
                for i, v in enumerate(grouped[value_col]):
                    plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

        # --------------------------------------------------
        # 3. Title & labels
        # --------------------------------------------------
        if not title:
            title = f"Top {top_n} thành phố có doanh thu cao nhất"
        plt.title(title, fontweight="bold", fontsize=15, pad=20)
        plt.xlabel(value_col)
        plt.ylabel(city_col)
        plt.ticklabel_format(style="plain", axis="x" if horizontal else "y")
        plt.tight_layout()
        # --------------------------------------------------
        # 4. Save chart
        # --------------------------------------------------
        if filename is None:
            filename = "top_cities_by_sales.png"

        path = self.chart_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        # --------------------------------------------------
        # 5. Narrative
        # --------------------------------------------------
        top_city = grouped.iloc[0][city_col]
        top_value = grouped.iloc[0][value_col]
        bottom_city = grouped.iloc[-1][city_col]
        bottom_value = grouped.iloc[-1][value_col]
        narrative = (
            f"Biểu đồ thể hiện top {top_n} thành phố có doanh thu cao nhất. "
            f"{top_city} dẫn đầu trong số này với tổng doanh thu đạt {top_value:,.0f}, "
            f"cho thấy vai trò nổi bật của khu vực này trong hoạt động kinh doanh. "
            f"Ngược lại, {bottom_city} có doanh thu thấp nhất trong top {top_n} với {bottom_value:,.0f}. "
            f"Doanh số của {top_city} (đầu bảng) gấp {top_value / bottom_value:,.2f} lần so với {bottom_city} (cuối bảng)."
        )
        # --------------------------------------------------
        # 6. Return result
        # --------------------------------------------------
        return {
            "chart_path": f"{self.relative_chart_path}/{filename}",
            "narrative": narrative
        }
    # Bottom sales cities
    def bottom_cities_by_sales(
        self,
        city_col: str = "City",
        value_col: str = "Sales",
        top_n: int = 10,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        horizontal: bool = True,
        show_values: bool = True
    ) -> Dict:
        """
        Biểu đồ bottom thành phố có doanh thu thấp nhất
        """

        # --------------------------------------------------
        # 1. Group & sort data
        # ------------------------------------------------
        grouped = (
            self.df
            .groupby(city_col)[value_col]
            .sum()
            .sort_values(ascending=True)
            .head(top_n)
            .reset_index()
        )

        # --------------------------------------------------
        # 2. Plot
        # ------------------------------------------------
        values = grouped[value_col]   # hoặc df_plot['Sales']

        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        # Cắt từ 30% → 100% của colormap Blues
        base_cmap = plt.cm.Blues
        cmap = mcolors.LinearSegmentedColormap.from_list(
        "Blues_truncated",
        base_cmap(np.linspace(0.3, 1.0, 256))
        )

        bar_colors = cmap(norm(values))
        plt.figure(figsize=(12, 8))

        if horizontal:
            plt.barh(grouped[city_col], grouped[value_col], color=bar_colors)
            plt.gca().invert_yaxis()  # City thấp nhất nằm trên cùng

            if show_values:
                for i, v in enumerate(grouped[value_col]):
                    plt.text(v, i, f" {v:,.2f}", va="center")
        else:
            plt.bar(grouped[city_col], grouped[value_col], color=bar_colors)
            plt.xticks(rotation=45, ha="right")

            if show_values:
                for i, v in enumerate(grouped[value_col]):
                    plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

        # --------------------------------------------------
        # 3. Title & labels
        # --------------------------------------------------
        if not title:
            title = f"Bottom {top_n} thành phố có doanh thu thấp nhất"
        plt.title(title, fontweight="bold", fontsize=15, pad=20)
        plt.xlabel(value_col)
        plt.ylabel(city_col)
        plt.ticklabel_format(style="plain", axis="x" if horizontal else "y")
        plt.tight_layout()
        # --------------------------------------------------
        # 4. Save chart
        # --------------------------------------------------
        if filename is None:
            filename = f"bottom_cities_by_sales.png"

        path = self.chart_dir / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        # --------------------------------------------------
        # 5. Narrative
        # --------------------------------------------------
        top_city = grouped.iloc[-1][city_col]
        top_value = grouped.iloc[-1][value_col]
        bottom_city = grouped.iloc[0][city_col]
        bottom_value = grouped.iloc[0][value_col]
        narrative = (
            f"Biểu đồ thể hiện bottom {top_n} thành phố có doanh thu thấp nhất. "
            f"{bottom_city} đứng cuối trong số này với tổng doanh thu đạt {bottom_value:,.2f}, "
            f"cho thấy những thách thức trong hoạt động kinh doanh tại khu vực này. "
            f"Ngược lại, {top_city} có doanh thu cao nhất trong bottom {top_n} với {top_value:,.2f}. "
            f"Doanh số của {top_city} (đầu bảng) gấp {top_value / bottom_value:,.2f} lần so với {bottom_city} (cuối bảng)."
        )
        
        return {
            "chart_path": f"{self.relative_chart_path}/{filename}",
            "narrative": narrative
        }
        
    # Phân tích xu hướng doanh thu & lợi nhuận theo thời gian

    def sales_trend(self, date_col='Order Date', freq='Q', filename=None):
        """
        Phân tích xu hướng doanh thu & lợi nhuận theo thời gian
        freq: 'M' (month), 'Q' (quarter), 'Y' (year)
        """

        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Resample theo thời gian
        trend = (
            df
            .set_index(date_col)
            .resample(freq)
            .agg({
            'Sales': 'sum',
            'Profit': 'sum'
            })
            .round(2)
            .reset_index()
        )

        # 👉 Tạo cột hiển thị theo quý (Q1-2023)
        if freq == "Q":
            trend["Period"] = trend[date_col].dt.to_period("Q").astype(str).str.replace("Q", "-Q")
        elif freq == "Y":
            trend["Period"] = trend[date_col].dt.year.astype(str)
        else:
            trend["Period"] = trend[date_col].dt.strftime("%m/%Y")

        selected_cols = ["Period", "Sales", "Profit"]
        
        trend = trend[selected_cols]
        # -------------------
        # Vẽ chart
        # -------------------
        plt.figure(figsize=(12, 6))
        plt.plot(trend["Period"], trend["Sales"], marker='o', label='Sales')
        plt.plot(trend["Period"], trend["Profit"], marker='o', label='Profit')
        plt.title(
        f"Sales & Profit Trend ({'Quarterly' if freq=='Q' else 'Yearly' if freq=='Y' else 'Monthly'})",
        fontweight='bold', fontsize=15, color="#333333", pad=20, loc="center"
        )
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if filename is None:
            filename = f"sales_profit_trend_{freq}.png"

        filepath = self.chart_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()

        # -------------------
        # Narrative
        # -------------------
        peak_idx = trend['Sales'].idxmax()
        peak_period = trend.loc[peak_idx, "Period"]

        trend_narrative = (
            f"Vì dữ liệu gồm nhiều tháng bán hàng nên chọn <b>quý</b> làm chu kỳ phân tích. "
            f"Doanh thu và lợi nhuận được phân tích theo chu kỳ "
            f"{'<b>quý</b>' if freq=='Q' else '<b>năm</b>' if freq=='Y' else '<b>tháng</b>'} "
            f"nhằm làm rõ xu hướng ngắn và trung hạn. "
            f"Doanh thu đạt mức cao nhất vào <b>{peak_period}</b>. "
            f"Biểu đồ doanh số cho thấy các <b>quý 1</b> của các năm có doanh thu rất thấp, "
            f"ngược lại các <b>quý 4</b> của các năm có doanh thu rất cao. "
            f"Tuy nhiên, lợi nhuận của <b>các quý 4</b> không tăng tương ứng với doanh thu, "
            f"chứng tỏ <b>chi phí bán hàng ...</b> trong các quý cao điểm này <b>cũng tăng mạnh.</b> "
            f"Về tổng thể, doanh số và lợi nhuận đều <b>tăng dần qua các năm</b>, trong đó "
            f"doanh số có sự biến động theo mùa vụ rõ rệt, đặc biệt là trong các <b>tháng cuối năm</b>. "
            f"Lợi nhuận tăng <b>không tương ứng</b> với doanh số trong các <b>quý</b> cao điểm."
        )
    
        return {
            "trend_table": trend,
            "chart_path": f"{self.relative_chart_path}/{filename}",
            "narrative": trend_narrative
        }

    def profit_by_subcategory(self, category_col='Sub-Category', filename=None):

        grouped = (
        self.df
        .groupby(category_col)
        .agg({
            'Sales': 'sum',
            'Profit': 'sum'
        })
        .reset_index()
        ).round(2)

        grouped['Profit_Margin (%)'] = (
        grouped['Profit'] / grouped['Sales'] * 100
    ).round(2)

        grouped = grouped.sort_values('Sales', ascending=False)
        # Chart: Sales vs Profit
        
        plt.figure(figsize=(10, 6))
        plt.bar(grouped[category_col], grouped['Sales'], label='Sales')
        plt.bar(grouped[category_col], grouped['Profit'], label='Profit')
        plt.title('Sales vs Profit by Sub-Category', fontweight='bold')
        plt.xlabel(category_col)
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        # thêm giá trị lên trên cột
        for i in range(len(grouped)):
            plt.text(i, grouped['Sales'].iloc[i], f"{grouped['Sales'].iloc[i]:,.0f}", ha="center", va="bottom")
            plt.text(i, grouped['Profit'].iloc[i], f"{grouped['Profit'].iloc[i]:,.0f}", ha="center", va="bottom")
            
        plt.tight_layout()
        if filename is None:
            filename = "sales_profit_by_subcategory.png"

        filepath = self.chart_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        # Narrative
        loss_categories = grouped[grouped['Profit'] < 0][category_col].tolist()
        
        top_profit_subcategory = grouped.loc[
        grouped['Profit'].idxmax(), category_col
        ]
        top_profit_value = grouped['Profit'].max()
        bottom_profit_subcategory = grouped.loc[
        grouped['Profit'].idxmin(), category_col
        ]
        bottom_profit_value = grouped['Profit'].min()
        
        top_sales_value = grouped['Sales'].max()
        bottom_sales_value = grouped['Sales'].min()
                
        narrative = (
        f"Một số danh mục có doanh thu cao nhưng lợi nhuận thấp. "
        f"Danh mục có lợi nhuận cao nhất là <b>{top_profit_subcategory}</b> với lợi nhuận <b>{top_profit_value:,.0f}</b>, "
        f"trong khi danh mục có lợi nhuận thấp nhất trong số này là <b>{bottom_profit_subcategory}</b> với lợi nhuận <b>{bottom_profit_value:,.0f}</b>. "
        f"Danh mục đầu bảng có lợi nhuận gấp <b>{abs(top_profit_value / bottom_profit_value):,.2f}</b> lần so với danh mục cuối bảng. "
        f"Trong khi đó doanh số của danh mục đầu bảng cao gấp <b>{top_sales_value / bottom_sales_value:,.2f}</b> lần so với danh mục cuối bảng. "
        f"{'Các danh mục có lợi nhuận âm bao gồm: <b>' + ', '.join(loss_categories) + '</b>.' if loss_categories else 'Không có danh mục nào có lợi nhuận âm.'} "    
        f"Điều này cho thấy cần phải <b>xem xét lại chiến lược giá hoặc chi phí.</b>"
        )

        return {
        "table": grouped,
        "chart_path": f"{self.relative_chart_path}/{filename}",
        "narrative": narrative
        }
    def profit_margin_by_subcategory(self, category_col='Sub-Category', filename=None):

        margin_df = (
        self.df
        .groupby(category_col)
        .agg({
            'Sales': 'sum',
            'Profit': 'sum'
        })
        .reset_index()
        )

        margin_df['Profit Margin (%)'] = (
        margin_df['Profit'] / margin_df['Sales'] * 100
        ).round(2)

        # Chart
        # Cách tô màu gradient dựa trên giá trị cao thấp
        
        #margin_df = margin_df.sort_values('Profit Margin (%)', ascending=True)

        values = margin_df['Profit Margin (%)']   # hoặc df_plot['Sales']

        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        # Cắt từ 30% → 100% của colormap Blues
        base_cmap = plt.cm.Blues
        cmap = mcolors.LinearSegmentedColormap.from_list(
        "Blues_truncated",
        base_cmap(np.linspace(0.3, 1.0, 256))
        )
        # Tô màu cho các cột
        # Riêng cột lợi nhuận âm tô màu đỏ
        bar_colors = []
        for v in values:
            if v < 0:
                bar_colors.append('red')
            else:
                bar_colors.append(cmap(norm(v)))
        
        
        plt.figure(figsize=(10, 6))
        plt.bar(margin_df[category_col], margin_df['Profit Margin (%)'], color=bar_colors)
        plt.title('Profit Margin by Sub-Category', fontweight='bold')
        plt.xlabel(category_col)
        plt.ylabel('Profit Margin (%)')
        plt.xticks(rotation=45)
        # thêm giá trị lên trên cột
        for i, v in enumerate(margin_df['Profit Margin (%)']):
            plt.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
            
        plt.tight_layout()
        if filename is None:
            filename = "profit_margin_by_subcategory.png"

        filepath = self.chart_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        # Narrative
        best_subcategory = margin_df.loc[
        margin_df['Profit Margin (%)'].idxmax(), category_col
        ]

        narrative = (
        f"Danh mục con có tỷ suất lợi nhuận cao nhất là {best_subcategory}. "
        f"Phân tích biên lợi nhuận giúp ưu tiên các danh mục con mang lại giá trị bền vững."
        )

        return {
        "table": margin_df,
        "chart_path": f"{self.relative_chart_path}/{filename}",
        "narrative": narrative
        }

    def executive_summary(self, results: Dict) -> Dict:
        """
        Executive Summary ngắn gọn, không trùng lặp với các phần phân tích chi tiết
        """

        sales = results.get("sales_summary", {})
        profit_by_subcategory = results.get("profit_by_subcategory", {})
        total_sales = sales.get("total_sales", 0)
        total_profit = sales.get("total_profit", 0)
        profit_margin = sales.get("profit_margin", 0)
        
        # Tìm Sub-Category có lợi nhuận âm
        loss_subcategories = []
        # Tổng lợi nhuận theo Sub-Category
        profit_by_subcategory_df = profit_by_subcategory.get("table")
        if profit_by_subcategory_df is not None:
            loss_subcategories = profit_by_subcategory_df[
            profit_by_subcategory_df["Profit"] < 0
        ][
            "Sub-Category"
        ].tolist()
        
        

        # Xác định danh mục con có lợi nhuận âm (nếu có)
        
        # ----------------------------
        # Executive Summary (1 đoạn duy nhất)
        # ----------------------------
        summary_text = (
        f"Báo cáo cung cấp cái nhìn tổng thể về hiệu quả kinh doanh, "
        f"với tổng doanh thu đạt {total_sales:,.0f} và tổng lợi nhuận "
        f"{total_profit:,.0f}, tương ứng tỷ suất lợi nhuận bình quân "
        f"{profit_margin:.2f}%. "
        )

        if loss_subcategories:
            summary_text += (
            f"Một số danh mục con kinh doanh ghi nhận lợi nhuận âm, "
            f"cho thấy tồn tại rủi ro về hiệu quả và cần được ưu tiên theo dõi "
            f"trong công tác quản lý và ra quyết định."
            )
        else:
            summary_text += (
            "Hiệu quả kinh doanh nhìn chung ở mức tích cực, "
            "không ghi nhận danh mục con nào có lợi nhuận âm trong kỳ phân tích."
            )

        # ----------------------------
        # Key Findings (bullet points)
        # ----------------------------
        key_findings = [
        f"Tổng doanh thu: {total_sales:,.0f}",
        f"Tổng lợi nhuận: {total_profit:,.0f}",
        f"Tỷ suất lợi nhuận bình quân: {profit_margin:.2f}%",
        f"Danh mục con có lợi nhuận âm: {', '.join(loss_subcategories) if loss_subcategories else 'Không có'}"
        ]

        return {
        "summary_text": summary_text,
        "key_findings": key_findings
        }

    def key_insights_and_recommendations(self, results: Dict) -> Dict:
        """
        Sinh Key Insights & Recommendations chi tiết dựa trên dữ liệu phân tích.
        """
        insights = []
        recommendations = []

        # --- 1. PHÂN TÍCH TỔNG QUAN & BIÊN LỢI NHUẬN ---
        sales = results.get("sales_summary", {})
        profit_margin = sales.get("profit_margin", 0)
        avg_sale = sales.get("average_sale_per_order", 0)

        if profit_margin < 5:
            insights.append(f"Biên lợi nhuận ròng hiện tại thấp ({profit_margin:.2f}%), cho thấy chi phí vận hành hoặc giá vốn đang chiếm tỷ trọng quá lớn.")
            recommendations.append("Cần thực hiện rà soát chi phí logistics và đàm phán lại với nhà cung cấp để tối ưu giá vốn.")
        elif profit_margin > 15:
            insights.append(f"Mô hình kinh doanh có sức khỏe tài chính tốt với biên lợi nhuận ấn tượng ({profit_margin:.2f}%).")
        
        insights.append(f"Giá trị đơn hàng trung bình (AOV) đạt {avg_sale:,.0f}. Đây là chỉ số then chốt để theo dõi hiệu quả bán chéo (cross-selling).")

        # --- 2. PHÂN TÍCH XU HƯỚNG (SEASONALITY) ---
        trend_data = results.get("sales_trend", {})
        trend_table = trend_data.get("trend_table")
        
        if trend_table is not None and len(trend_table) > 1:
            # Kiểm tra biến động quý 4 so với quý 1 (nếu có đủ dữ liệu)
            insights.append("Dữ liệu cho thấy sự lệch pha lớn giữa doanh thu và lợi nhuận vào các kỳ cao điểm (Quý 4).")
            recommendations.append("Áp dụng chiến lược 'Lợi nhuận mục tiêu' cho các chương trình khuyến mãi cuối năm để tránh tình trạng tăng doanh thu nhưng giảm hiệu quả lợi nhuận.")

        # --- 3. PHÂN TÍCH THEO DANH MỤC (PORTFOLIO) ---
        profit_by_subcategory = results.get("profit_by_subcategory", {})
        df_subcategory = profit_by_subcategory.get("table")
        
        if df_subcategory is not None:
            # Insight về danh mục con lỗ
            loss_subcategories = df_subcategory[df_subcategory["Profit"] < 0]
            if not loss_subcategories.empty:
                subcat_names = ", ".join(loss_subcategories["Sub-Category"].tolist())
                insights.append(f"CẢNH BÁO: Danh mục con [{subcat_names}] đang trong tình trạng 'bán lỗ', gây thâm hụt lợi nhuận chung.")
                recommendations.append(f"Tạm dừng các chương trình giảm giá sâu cho nhóm {subcat_names} và đánh giá lại danh mục hàng hóa (Inventory Audit).")
            
            # Insight về ngôi sao (High Sales, High Profit)
            star_subcat = df_subcategory.sort_values(by=["Sales", "Profit"], ascending=False).iloc[0]["Sub-Category"]
            insights.append(f"Danh mục con '{star_subcat}' là động lực tăng trưởng chính, đóng góp tỷ trọng lớn nhất vào cả doanh thu và lợi nhuận.")
            recommendations.append(f"Tăng ngân sách marketing cho danh mục con '{star_subcat}' để chiếm lĩnh thêm thị phần.")

        # --- 4. PHÂN TÍCH ĐỊA LÝ (GEOGRAPHIC) ---
        top_cities = results.get("top_cities_by_sales", {})
        bottom_cities = results.get("bottom_cities_by_sales", {})
        
        # Lấy tên thành phố đầu bảng từ narrative hoặc dữ liệu nếu có
        insights.append("Hoạt động kinh doanh đang tập trung quá mức vào một số thành phố trọng điểm, tạo ra rủi ro phụ thuộc thị trường.")
        recommendations.append("Nghiên cứu mô hình thành công ở các thành phố Top đầu để nhân rộng (Scale-up) sang các khu vực có tiềm năng nhưng doanh số còn thấp.")

        # --- 5. CHIẾN LƯỢC TỔNG THỂ ---
        recommendations.append("Xây dựng hệ thống KPI tập trung vào 'Lợi nhuận trên mỗi đơn hàng' thay vì chỉ tập trung vào 'Doanh số thuần'.")
        
        return {
            "insights": insights,
            "recommendations": recommendations
        }
    
    # Chạy tất cả phân tích và trả về kết quả dưới dạng dictionary
    
    def run_all(self, freq='Q', top_n=10):

        results = {
            "sales_summary": self.sales_summary(),
            "sales_trend": self.sales_trend(freq=freq),
            "sales_by_subcategory": self.sales_by_subcategory(
                category_col='Sub-Category',
                value_col='Sales',
                top_n=top_n,
                title='Top Sub-Categories by Sales'
            ),
            "profit_by_subcategory": self.profit_by_subcategory(),
            "profit_margin_by_subcategory": self.profit_margin_by_subcategory(),
            "top_cities_by_sales": self.top_cities_by_sales(top_n=top_n),
            "bottom_cities_by_sales": self.bottom_cities_by_sales(top_n=top_n)
        }

        # Executive Summary
        results["executive_summary"] = self.executive_summary(results)
        results["key_insights_and_recommendations"] = self.key_insights_and_recommendations(results)
        return results
