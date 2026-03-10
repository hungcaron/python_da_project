import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CohortAnalysis:
    def __init__(self, df, chart_dir, mode="monthly"):
        """
        Khởi tạo lớp phân tích Cohort.
        mode: "monthly" (hàng tháng) | "quarterly" (hàng quý) | "last_12_months" (12 tháng gần nhất)
        """
        self.df = df.copy()
        self.chart_dir = chart_dir
        self.mode = mode

        # Chuyển đổi cột ngày đặt hàng sang định dạng datetime và tạo cột tháng đặt hàng
        self.df["OrderDate"] = pd.to_datetime(self.df["Order Date"])
        self.df["OrderMonth"] = self.df["OrderDate"].dt.to_period("M")

    # ======================================================
    # Hàm hỗ trợ: Lọc dữ liệu 12 tháng gần nhất một cách an toàn
    # ======================================================
    def _filter_last_12_months(self, df):
        """Đảm bảo không lọc mất dữ liệu nếu bộ dữ liệu có ít hơn 12 tháng."""
        last_month = df["OrderMonth"].max()
        unique_months = df["OrderMonth"].nunique()

        if unique_months <= 12:
            return df  # Trả về nguyên vẹn nếu dữ liệu ngắn hơn hoặc bằng 12 tháng

        # Tính toán tháng bắt đầu (lùi lại 11 tháng tính từ tháng cuối cùng)
        first_month = last_month - 11
        return df[df["OrderMonth"] >= first_month]

    # ======================================================
    # Lựa chọn 1 — Phân tích Cohort theo tháng (Số lượng tuyệt đối)
    # ======================================================
    def cohort_monthly(self):
        df = self.df.copy()

        # Xác định tháng đầu tiên mua hàng của mỗi khách hàng (CohortMonth)
        df["CohortMonth"] = df.groupby("Customer ID")["OrderMonth"].transform("min")

        # Tính toán chỉ số CohortIndex (Tháng thứ n kể từ lần mua đầu tiên)
        df["CohortIndex"] = (
            (df["OrderMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
            (df["OrderMonth"].dt.month - df["CohortMonth"].dt.month) + 1
        )

        # Tạo bảng ma trận giữ chân khách hàng (số lượng khách hàng duy nhất)
        table = (
            df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
            .nunique()
            .unstack()
        )

        # Vẽ biểu đồ Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(table, annot=True, fmt=".2f", cmap="Purples")
        plt.title("Monthly Cohort Retention")

        # Lưu biểu đồ vào thư mục chỉ định
        chart_path = os.path.join(self.chart_dir, "cohort_retention_monthly.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        # Tính toán một số thông tin cơ bản để viết nhận xét (narrative)
        total_cohorts = table.shape[0]
        avg_retention = table.mean().mean()
        narrative = (
            f"Có tổng cộng {total_cohorts} nhóm khách hàng theo tháng. "
            f"Tỷ lệ giữ chân trung bình là {avg_retention:.2%}. "
            f"Việc duy trì tỷ lệ giữ chân khách hàng ổn định qua các nhóm là yếu tố quan trọng "
            f"để đảm bảo sự phát triển bền vững của doanh nghiệp."
        )
        return table, chart_path, narrative

    # ======================================================
    # Lựa chọn 2 — Phân tích Cohort theo Quý
    # ======================================================
    def cohort_quarterly(self):
        df = self.df.copy()

        # Chuyển đổi sang đơn vị Quý
        df["OrderQuarter"] = df["OrderDate"].dt.to_period("Q")
        df["CohortQuarter"] = df.groupby("Customer ID")["OrderQuarter"].transform("min")

        # Tính toán chỉ số Quý (Quý thứ n kể từ lần đầu mua hàng)
        df["QuarterIndex"] = (
            (df["OrderQuarter"].dt.year - df["CohortQuarter"].dt.year) * 4 +
            (df["OrderQuarter"].dt.quarter - df["CohortQuarter"].dt.quarter) + 1
        )

        # Tạo bảng ma trận theo Quý
        table = (
            df.groupby(["CohortQuarter", "QuarterIndex"])["Customer ID"]
            .nunique()
            .unstack()
        )

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        sns.heatmap(table, annot=True, fmt=".0f", cmap="Blues")
        plt.title("Quarterly Cohort Retention")

        chart_path = os.path.join(self.chart_dir, "cohort_retention_quarterly.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        return table, chart_path

    # ======================================================
    # Lựa chọn 3 — Phân tích Cohort 12 tháng gần nhất (Tỷ lệ %)
    # ======================================================
    def cohort_last_12_months(self):
        # Lọc dữ liệu trong vòng 12 tháng gần nhất
        df = self._filter_last_12_months(self.df.copy())

        # 1. Xác định CohortMonth và CohortIndex (tương tự như hàm monthly)
        df["CohortMonth"] = df.groupby("Customer ID")["OrderMonth"].transform("min")
        df["CohortIndex"] = (
            (df["OrderMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
            (df["OrderMonth"].dt.month - df["CohortMonth"].dt.month) + 1
        )

        # 2. Tạo bảng đếm số lượng khách hàng
        cohort_counts = (
            df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
            .nunique()
            .unstack()
        )

        if cohort_counts.empty:
            return None, None, "<p><i>Không có dữ liệu cohort.</i></p>"

        # 3. Chuyển đổi số lượng tuyệt đối sang tỷ lệ giữ chân (%)
        # Lấy cột đầu tiên (quy mô ban đầu của mỗi cohort) làm mốc 100%
        cohort_sizes = cohort_counts.iloc[:, 0]
        table = cohort_counts.divide(cohort_sizes, axis=0) * 100
        table = table.round(1)

        # 4. Vẽ biểu đồ Heatmap tỷ lệ %
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            table,
            annot=True,
            fmt=".1f",
            cmap="Greens",
            cbar_kws={"label": "Retention (%)"}
        )
        plt.title("Last 12 Months Cohort Retention (%)")

        chart_path = os.path.join(self.chart_dir, "cohort_retention_last12.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        # 5. Tính toán các chỉ số (Metrics) để viết nhận xét chuyên sâu
        total_cohorts = table.shape[0]

        # Tỷ lệ quay lại vào tháng thứ 2 (Index 2)
        early_retention = table.iloc[:, 1].mean() if table.shape[1] > 1 else None
        # Tỷ lệ giữ chân trung hạn (tháng 3-4)
        mid_retention = table.iloc[:, 2:4].mean().mean() if table.shape[1] > 3 else None
        # Tỷ lệ giữ chân dài hạn (sau 6 tháng)
        long_retention = table.iloc[:, 5:].mean().mean() if table.shape[1] > 5 else None
        
        # Tìm cohort có hiệu suất tốt nhất và kém nhất dựa trên trung bình hàng ngang
        cohort_avg = table.iloc[:, 1:].mean(axis=1)
        best_cohort = cohort_avg.idxmax()
        worst_cohort = cohort_avg.idxmin()

        # 6. Viết nhận xét (phong cách tư vấn chuyên nghiệp)
        narrative = (
            f"Phân tích dựa trên <b>{total_cohorts}</b> nhóm khách hàng hình thành trong <b>12 tháng gần nhất</b> "
            f"cho thấy tỷ lệ giữ chân giảm mạnh ngay sau lần mua đầu tiên. "
        )

        if early_retention is not None:
            narrative += f"Trung bình chỉ khoảng <b>{early_retention:.1f}%</b> khách hàng quay lại trong tháng thứ hai. "

        if mid_retention is not None:
            narrative += (
                f"Tỷ lệ giữ chân trung hạn (tháng 3 – 4) tiếp tục <b>ở mức thấp</b>, "
                f"trung bình khoảng <b>{mid_retention:.1f}%</b>, cho thấy phần lớn khách hàng "
                f"chưa hình thành thói quen mua lặp lại. "
            )

        if long_retention is not None:
            narrative += (
                f"Ở dài hạn (sau 6 tháng), tỷ lệ giữ chân trung bình chỉ còn khoảng "
                f"<b>{long_retention:.1f}%</b>, phản ánh thách thức trong việc xây dựng tệp khách hàng trung thành. "
            )
            
        narrative += (
            f"So sánh giữa các cohort cho thấy sự khác biệt đáng kể về hiệu quả giữ chân khách hàng, "
            f"trong đó cohort <b>{best_cohort.strftime('%Y-%m')}</b> có hiệu suất <b>tốt nhất</b>, "
            f"trong khi cohort <b>{worst_cohort.strftime('%Y-%m')}</b> có mức giữ chân <b>thấp nhất</b>. "
            f"Điều này cho thấy trải nghiệm khách hàng và hiệu quả các hoạt động tiếp thị "
            f"chưa thực sự đồng đều giữa các thời điểm."
        )

        return table, chart_path, narrative

    # ======================================================
    # Hàm thực thi chính
    # ======================================================
    def run_all(self):
        """Chạy phân tích dựa trên chế độ (mode) đã chọn."""
        if self.mode == "quarterly":
            # Lưu ý: cohort_quarterly hiện chưa trả về 'narrative', bạn có thể cần cập nhật hàm đó
            table, chart_path = self.cohort_quarterly()
            narrative = "Phân tích theo quý hiện chưa có nội dung nhận xét chi tiết."
        elif self.mode == "last_12_months":
            table, chart_path, narrative = self.cohort_last_12_months()
        else:
            table, chart_path, narrative = self.cohort_monthly()

        # Trả về kết quả dưới dạng Dictionary để dễ sử dụng
        return {
            "mode": self.mode,
            "table_html": table.to_html(),
            "chart_path": chart_path,
            "narrative": narrative,
            "chart_file": os.path.basename(chart_path)
        }