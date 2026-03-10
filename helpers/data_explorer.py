from fileinput import filename
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from helpers.data_formatter import list_to_text
import helpers.data_formatter as fmt

class DataExplorer:
    
    # Class dùng để khám phá dữ liệu (EDA) và vẽ biểu đồ tự động.
    
    def __init__(self, df, chart_dir):
        """
        Khởi tạo DataExplorer
        
        Args:
            df: DataFrame cần phân tích
            chart_dir: Đường dẫn THƯ MỤC CHART từ THƯ MỤC GỐC của dự án
                       Ví dụ: 'reports/outputs/charts'
        """
        self.df = df.copy()
        # Chuyển đổi đường dẫn thành đối tượng Path
        self.chart_dir = Path(chart_dir)
        
        # Tạo thư mục lưu biểu đồ nếu chưa có
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Tính đường dẫn tương đối từ thư mục reports/outputs đến charts
        # Đây sẽ là 'charts' vì chart_dir = reports/outputs/charts
        self.relative_chart_path = self.chart_dir.name  # 'charts'

    # ==================== THÔNG TIN CHUNG ====================

    def df_shape(self):
        """Kích thước dataframe"""
        rows = self.df.shape[0]
        columns = self.df.shape[1]

        narrative =""
        if rows >= 5000 and columns >= 10:
            narrative = f"Bộ dữ liệu có {fmt.format_integer(rows)} dòng và {fmt.format_integer(columns)} cột, kích thước đủ lớn, đảm bảo yêu cầu đề bài, và phù hợp để phân tích và trực quan hóa."
        else:   
            narrative = f"Bộ dữ liệu có {fmt.format_integer(rows)} dòng và {fmt.format_integer(columns)} cột, kích thước chưa đủ lớn theo yêu cầu, cần phải chọn bộ dữ liệu khác để phân tích."
        
        return {
            'rows': rows,
            'columns': columns,
            "narrative": narrative
        }

    def columns_info(self):
        """
        Phân tích kiểu dữ liệu các cột và tạo narrative mô tả
        """

        # Kiểu dữ liệu từng cột
        column_types = self.df.dtypes.apply(lambda x: x.name)

        # Đếm số cột theo kiểu
        type_counts = column_types.value_counts().to_dict()
        # Tổng số cột
        total_cols = len(column_types)

        # Tạo narrative
        narrative_parts = []
        for dtype, count in type_counts.items():
            narrative_parts.append(f"{count} cột có kiểu dữ liệu {dtype}")

            narrative = (
            f"Cụ thể, bộ dữ liệu gồm {total_cols} cột, trong đó "
            + ", ".join(narrative_parts)
            + "."
            )

        return {
            "column_types": column_types.to_dict(),   # chi tiết từng cột
            "type_counts": type_counts,               # tổng hợp theo kiểu
            "narrative": narrative                    # mô tả bằng chữ
        }
    
    # Kiêm tra kiểu dữ liệu của các cột quan tâm
    def datatype_check(self, numeric_cols=None, datetime_cols=None):
        remarks = []

        numeric_cols = numeric_cols or []
        datetime_cols = datetime_cols or []

        numeric_valid = []
        numeric_invalid = []
        datetime_valid = []
        datetime_invalid = []
        missing_cols = []
        # === 1. Numeric ===
        for col in numeric_cols:
            if col not in self.df.columns:
                missing_cols.append(col)
                continue
            dtype_str = str(self.df[col].dtype)
            if dtype_str in ["int64", "float64"]:
                numeric_valid.append(col)
            else:
                numeric_invalid.append(col)
        # === 2. Datetime ===
        for col in datetime_cols:
            if col not in self.df.columns:
                missing_cols.append(col)
                continue
            dtype_str = str(self.df[col].dtype)
            if dtype_str.startswith("datetime64"):
                datetime_valid.append(col)
            else:
                datetime_invalid.append(col)
        # === 3. Build remarks ===
        if numeric_valid:
            remarks.append(
                f"Các cột {list_to_text(numeric_valid)} có kiểu dữ liệu numeric, phù hợp cho phân tích."
            )
        if numeric_invalid:
            remarks.append(
            f"Các cột {list_to_text(numeric_invalid)} hiện KHÔNG phải numeric, cần chuyển đổi."
        )

        if datetime_valid:
            remarks.append(
            f"Các cột {list_to_text(datetime_valid)} có kiểu datetime đúng chuẩn."
        )

        if datetime_invalid:
            remarks.append(
            f"Các cột {list_to_text(datetime_invalid)} hiện không phải datetime. Cần chuyển đổi trước khi phân tích."
        )

        if missing_cols:
            remarks.append(
            f"Các cột {list_to_text(missing_cols)} không tồn tại trong dataset."
        )

        return remarks

    def basic_stats(self, cols: list = None):
        """Thống kê mô tả số liệu"""
        cols = ["Sales", "Quantity", "Discount", "Profit"] if cols is None else cols
        stats = self.df[cols].describe().transpose()
        return stats.to_dict(orient="index")

    # Kiểm tra missing và duplicate
    def missing_duplicate_check(self):
        """
        Kiểm tra giá trị thiếu và bản ghi trùng lặp
        """

        # -------------------------
        # 1. Missing values
        # -------------------------
        missing_series = self.df.isnull().sum()
        cols_with_missing = missing_series[missing_series > 0]

        if cols_with_missing.empty:
            missing_summary = "Bộ dữ liệu không nào có <b>giá trị thiếu,</b>"
            missing_table = None
        else:
            missing_summary = "Phát hiện <b>giá trị thiếu</b> ở một số cột."
        missing_table = (
                cols_with_missing
                .reset_index()
                .rename(columns={"index": "Tên cột", 0: "Số giá trị thiếu"})
            )
        # -------------------------
        # 2. Duplicate records
        # -------------------------
        duplicate_count = int(self.df.duplicated().sum())

        if duplicate_count == 0:
            duplicate_summary = " không có <b>bản ghi trùng lặp.</b>"
        else:
            duplicate_summary = f"Bộ dữ liệu có {duplicate_count} bản ghi trùng lặp."

        # -------------------------
        # 3. Narrative tổng hợp
        # -------------------------
        narrative = f"{missing_summary} {duplicate_summary}"

        return {
        "missing": {
            "summary": missing_summary,
            "table": missing_table
        },
        "duplicates": {
            "count": duplicate_count,
            "summary": duplicate_summary
        },
        "narrative": narrative
        }
    
    def head_data(self, n=10, columns=None):
        """Lấy n dòng đầu tiên"""
        try:
            df_subset = self.df[columns] if columns else self.df
            return df_subset.head(n).to_dict(orient="records")
        except KeyError:
            # Nếu cột không tồn tại → fallback
            return self.df.head(n).to_dict(orient="records")

    # ==================== NHÓM CỘT ====================

    def numeric_columns(self):
        """Lấy các cột numeric"""
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        remove_cols = ["Row ID", "Postal Code"]
        numeric_cols = [c for c in numeric_cols if c not in remove_cols]
        return numeric_cols
    
    def categorical_columns(self):
        """Lấy các cột categorical"""
        return self.df.select_dtypes(exclude="number").columns.tolist()

    # ==================== THỐNG KÊ CATEGORICAL ====================

    def categorical_summary(self, top=10):
        """Tần suất các giá trị trong cột dạng category"""
        summary = {}
        for col in self.categorical_columns():
            summary[col] = self.df[col].value_counts().head(top).to_dict()
        return summary

    # ==================== CORRELATION ====================

    def correlation_matrix(self): 
        """Tính ma trận tương quan"""        
        cols = ["Sales", "Profit", "Discount", "Quantity"]
        cols = [c for c in cols if c in self.df.columns]
        df_corr = self.df[cols].corr().round(2)
        return df_corr.to_dict()

    def plot_correlation_heatmap(self, filename="correlation_heatmap.png"):
        """
        Vẽ và lưu correlation heatmap
        Returns:
            str: Đường dẫn TƯƠNG ĐỐI cho HTML (vd: 'charts/correlation_heatmap.png')
        """
        # Loại bỏ các cột không cần thiết
        cols = ["Sales", "Profit", "Discount", "Quantity"]
        cols = [c for c in cols if c in self.df.columns]
    
        # Tính ma trận tương quan
        df_corr = self.df[cols].corr().round(2)
        
        # Vẽ heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Bản đồ nhiệt độ tương quan giữa các biến số",
                    fontsize=15, fontweight="bold", color="#333333", pad=20, loc="center")

        # Lưu file với đường dẫn đầy đủ
        filepath = self.chart_dir / filename
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        # Trả về đường dẫn TƯƠNG ĐỐI cho HTML
        return f"{self.relative_chart_path}/{filename}"

    # ==================== OUTLIERS ====================

    def detect_outliers(self):
        """Dùng phương pháp IQR để đếm số lượng outlier từng cột numeric"""
        outlier_info = {}
        for col in self.numeric_columns():
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            count = self.df[(self.df[col] < lower) | (self.df[col] > upper)].shape[0]
            outlier_info[col] = int(count)
        return outlier_info

    # ==================== DISTRIBUTION ====================

    def numeric_distribution(self):
        """Tính skewness và kurtosis"""
        stats = {}
        for col in self.numeric_columns():
            stats[col] = {
                "skew": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis())
            }
        return stats

    # ==================== VẼ BIỂU ĐỒ PHÂN PHỐI ====================

    def _save_plot(self, filename, plot_func, *args, **kwargs):
        """
        Helper function để lưu biểu đồ và trả về đường dẫn tương đối
        
        Args:
            filename: Tên file ảnh
            plot_func: Hàm vẽ biểu đồ
            *args, **kwargs: Tham số cho plot_func
        """
        # Tạo đường dẫn đầy đủ
        filepath = self.chart_dir / filename
        
        # Gọi hàm vẽ
        plot_func(*args, **kwargs)
        
        # Lưu biểu đồ
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        # Trả về đường dẫn TƯƠNG ĐỐI cho HTML
        return f"{self.relative_chart_path}/{filename}"

    # ==== 1. Histogram ====
    def plot_hist(self, column_name, filename=None):
        """Vẽ histogram"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' không tồn tại.")
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            raise TypeError(f"Column '{column_name}' không phải numerical.")

        if filename is None:
            filename = f"hist_{column_name}.png"
        
        # Tạo hàm vẽ cục bộ
        def _plot():
            plt.figure(figsize=(12, 8))
            sns.histplot(self.df[column_name], kde=True)
            plt.title(f"Distribution of {column_name}", fontsize=15, fontweight="bold", color="#333333", pad=20, loc="center")
        
        # Sử dụng helper để lưu và trả về đường dẫn
        return self._save_plot(filename, _plot)

    # ==== 2. Box Plot ====
    def plot_box(self, column_name, filename=None):
        """Vẽ box plot"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' không tồn tại.")
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            raise TypeError(f"Column '{column_name}' không phải numerical.")

        if filename is None:
            filename = f"box_{column_name}.png"

        # Tạo hàm vẽ cục bộ
        def _plot():
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=self.df[column_name])
            plt.title(f"Biểu đồ Box - {column_name}", fontsize=15, fontweight="bold", color="#333333", pad=20, loc="center")
        
        # Sử dụng helper để lưu và trả về đường dẫn
        return self._save_plot(filename, _plot)

    # ==== 3. Scatter Plot ====
    def plot_scatter(self, x_col, y_col, filename=None):
        """Vẽ scatter plot"""
        # Validate col names
        if x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError("Tên cột không tồn tại.")

        # Subset và convert to numeric
        df_plot = self.df[[x_col, y_col]].copy()
        df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
    
        # Drop NaN
        df_plot = df_plot.dropna()
    
        if df_plot.empty:
            raise ValueError("Không có dữ liệu hợp lệ để vẽ scatterplot.")
    
        # Filename
        if filename is None:
            filename = f"scatter_{x_col}_{y_col}.png"
        
        # Tạo hàm vẽ cục bộ
        def _plot():
            plt.figure(figsize=(12, 8))
            plt.scatter(df_plot[x_col], df_plot[y_col], alpha=0.7, s=40)
            
            # Thêm trendline (tùy chọn)
            try:
                z = np.polyfit(df_plot[x_col], df_plot[y_col], 1)
                p = np.poly1d(z)
                plt.plot(df_plot[x_col], p(df_plot[x_col]), "r--", alpha=0.8)
            except:
                pass
            
            plt.title(f"Biểu đồ {x_col} vs {y_col}", fontsize=15, fontweight="bold", color="#333333", pad=20, loc="center")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
        
        # Sử dụng helper để lưu và trả về đường dẫn
        return self._save_plot(filename, _plot)

    # ==== 4. Bar Chart ====
    def plot_bar(self, column_name, filename=None):
        """Vẽ bar chart"""
        if column_name not in self.df.columns:
            raise ValueError(f"Cột '{column_name}' không tồn tại.")

        if filename is None:
            filename = f"bar_{column_name}.png"
        
        # Tạo hàm vẽ cục bộ
        def _plot():
            plt.figure(figsize=(12, 8))
            self.df[column_name].value_counts().plot(kind='bar')
            plt.title(f"Biểu đồ giá trị của {column_name}")
        
        # Sử dụng helper để lưu và trả về đường dẫn
        return self._save_plot(filename, _plot)

    # ==== 5. Density Plot ====
    def plot_density(self, column_name, filename=None):
        """Vẽ density plot"""
        if column_name not in self.df.columns:
            raise ValueError(f"Cột '{column_name}' không tồn tại.")
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            raise TypeError(f"Cột '{column_name}' không phải numerical.")

        if filename is None:
            filename = f"density_{column_name}.png"
        
        # Tạo hàm vẽ cục bộ
        def _plot():
            plt.figure(figsize=(12, 8))
            if column_name == "Profit":
                sns.kdeplot(self.df[column_name], fill=True)    
            else:
                sns.kdeplot(self.df[column_name], clip=(0, None), fill=True)
                plt.xlim(left=0)
            plt.title(f"Biểu đồ Density - {column_name}", fontsize=15, fontweight="bold", color="#333333", pad=20, loc="center")
            
        
        # Sử dụng helper để lưu và trả về đường dẫn
        return self._save_plot(filename, _plot)

    # ==== 6. Correlation Plot ====
    def plot_correlation(self, columns, filename=None):
        """Vẽ correlation matrix"""
        # Validate columns
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Cột '{col}' không tồn tại.")
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise TypeError(f"Cột '{col}' phải numerical.")

        # Default filename
        if filename is None:
            filename = f"corr_{'_'.join(columns)}.png"
        
        # Tạo hàm vẽ cục bộ
        def _plot():
            corr = self.df[columns].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Ma trận tương quan - {', '.join(columns)}", fontsize=15, fontweight="bold", color="#333333", pad=20, loc="center")
        
        # Sử dụng helper để lưu và trả về đường dẫn
        return self._save_plot(filename, _plot)
    
    # Hàm mô tả toàn bộ EDA theo cách mô tả bằng lời
    def describe_all(self):
        # Mô tả toàn bộ EDA
        narrative = ""
        narrative += f"Bộ dữ liệu có {fmt.format_integer(self.df_shape()['rows'])} dòng và {fmt.format_integer(self.df_shape()['columns'])} cột. "
        narrative += self.columns_info()['narrative'] + " "
        narrative += self.missing_duplicate_check()['narrative'] + " "
        narrative += "Có các <b>giá trị ngoại lai</b> được phát hiện trong các cột numeric. "
        narrative += "Phân phối của các biến số <b>Sales, Profit</b> được trực quan hóa thông qua các biểu đồ tương ứng, "
        narrative += "trong đó có thể thấy rằng cột <b>Profit</b> có nhiều <b>giá trị âm và dương</b> phân tán xa nhau. "
        narrative += "Điều này có thể do các đơn hàng có lợi nhuận âm do <b>chiết khấu, khuyến mãi, hoặc chi phí giao hàng</b> quá cao ... "
        narrative += "Có những <b>giá trị ngoại lai</b> nhưng chưa đủ căn cứ để loại bỏ. "
        narrative +="Tóm lại, bộ dữ liệu này có thể sử dụng để tiến hành các phân tích tiếp theo."
        return narrative

    # ==================== RUN FULL EDA ====================

    def run_all(self):
        """Chạy toàn bộ phân tích EDA"""
        return {
            "df_shape": self.df_shape(),
            "columns_info": self.columns_info(),
            "datatype_check": self.datatype_check(
                numeric_cols=["Sales", "Quantity", "Discount", "Profit"],
                datetime_cols=["Order Date", "Ship Date"]
            ),
            "basic_stats": self.basic_stats(),
            "missing_duplicate_check": self.missing_duplicate_check(),
            "numeric_columns": list_to_text(self.numeric_columns()),
            "categorical_columns": list_to_text(self.categorical_columns()),
            "categorical_summary": self.categorical_summary(),
            "correlation_matrix": self.correlation_matrix(),
            "outliers": self.detect_outliers(),
            "numeric_distribution": self.numeric_distribution(),
            "head_data": self.head_data(n=10, columns=[
                "Order ID", "Order Date", "Ship Date",
                "Sales", "Quantity", "Discount", "Profit"
            ]),
            # Biểu đồ - trả về đường dẫn tương đối
            "plot_box_sales": self.plot_box("Sales"),
            "density_sales": self.plot_density("Sales"),
            "density_profit": self.plot_density("Profit"),
            "plot_box_profit": self.plot_box("Profit"),
            "plot_scatter_sales_profit": self.plot_scatter(x_col="Sales", y_col="Profit"),
            "description": self.describe_all()
        }