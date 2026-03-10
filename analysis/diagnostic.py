import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from helpers.data_formatter import list_to_text 
import helpers.data_formatter as fmt
from matplotlib import colors as mcolors


class DiagnosticAnalysis:
    def __init__(self, df: pd.DataFrame, chart_dir: Optional[str] = None):
        self.df = df.copy()
        self.chart_dir = Path(chart_dir) if chart_dir else Path("outputs/charts")
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.relative_chart_path = self.chart_dir.name

    # --------------------------------------------------
    # Helper: save chart
    # --------------------------------------------------
    def _save_chart(self, fig, filename: Optional[str]):
        if not filename:
            return None
        path = self.chart_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return f"{self.relative_chart_path}/{filename}"

    # --------------------------------------------------
    # Helper: build Pareto
    # --------------------------------------------------
    def _build_pareto(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        title: str,
        filename: str,
        focus: str = "all",   # all | positive | negative
        threshold: float = 0.8
    ) -> Dict:

        data = df.copy()

        if focus == "positive":
            data = data[data[value_col] > 0]
        elif focus == "negative":
            data = data[data[value_col] < 0]

        pareto_df = (
            data.groupby(group_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        if pareto_df.empty:
            return {
                "dataframe": None,
                "chart_path": None,
                "narrative": "❌ Không đủ dữ liệu để phân tích Pareto."
            }

        pareto_df["pct"] = pareto_df[value_col] / pareto_df[value_col].sum()
        pareto_df["cum_pct"] = pareto_df["pct"].cumsum()

        # ---- Plot
        values = pareto_df[value_col]   # hoặc df_plot['Sales']

        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        # Cắt từ 30% → 100% của colormap Blues
        if focus == "negative":
            base_cmap = plt.cm.Oranges_r
        else:
            base_cmap = plt.cm.Blues
        cmap = mcolors.LinearSegmentedColormap.from_list(
        "Blues_truncated",
        base_cmap(np.linspace(0.3, 1.0, 256))
        )
        bar_colors = cmap(norm(values))
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.bar(pareto_df[group_col], pareto_df[value_col], color=bar_colors)
        ax1.set_ylabel(value_col)
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(pareto_df[group_col], pareto_df["cum_pct"], marker="o")
        ax2.axhline(threshold, linestyle="--", linewidth=1)
        ax2.set_ylabel("Cumulative %")

        ax1.set_title(title, weight="bold")

        chart_path = self._save_chart(fig, filename)

        # ---- Narrative
        key_items = pareto_df[pareto_df["cum_pct"] <= threshold]
        n_key = len(key_items)
        total = len(pareto_df)
        pct_val = key_items["pct"].sum() * 100

        if focus == "positive":
            narrative = (
                f"Phân tích Pareto cho thấy lợi nhuận phụ thuộc mạnh vào một số ít <b>{group_col}</b>. "
                f"Cụ thể, {n_key} nhóm trong số {total} (<b>{fmt.format_percentage(n_key/total)}</b>) gồm <b>{fmt.list_to_text(key_items[group_col].tolist())}</b> "
                f"tạo ra khoảng <b>{fmt.format_percentage(pct_val/100)}</b> tổng lợi nhuận. "
                "Nhóm này đóng vai trò là <b>động lực lợi nhuận cốt lõi</b> của doanh nghiệp."
            )
        else:
            narrative = (
                f"Thua lỗ <b>không phân bổ đồng đều</b> mà tập trung chủ yếu vào "
                f"<b>{n_key} nhóm trong số {total} (<b>{fmt.format_percentage(n_key/total)}</b>) {group_col}</b>, "
                f"chiếm khoảng <b>{fmt.format_percentage(pct_val/100)}</b> tổng mức lỗ. "
                f"Các nhóm như <b>{fmt.list_to_text(key_items[group_col].tolist())}</b> "
                "là <b>nguồn gây lỗ chính</b>, cần được ưu tiên xử lý thay vì mở rộng doanh thu."
            )

        return {
            "dataframe": pareto_df,
            "chart_path": chart_path,
            "narrative": narrative
        }

    # --------------------------------------------------
    # 1. Correlation Matrix
    # --------------------------------------------------
    def correlation_matrix(self, filename: str = "correlation_matrix.png") -> Dict:
        cols = ["Sales", "Profit", "Discount", "Quantity"]
        cols = [c for c in cols if c in self.df.columns]

        if len(cols) < 2:
            return {
                "dataframe": None,
                "chart_path": None,
                "narrative": "❌ Không đủ dữ liệu để phân tích tương quan."
            }

        corr_df = self.df[cols].corr().round(2)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1, vmax=1, center=0,
            linewidths=0.8,
            square=True,
            ax=ax
        )

        ax.set_title("Correlation Matrix – Key Business Metrics", weight="bold")
        chart_path = self._save_chart(fig, filename)
        # Xác định các |r| > 0.5 đối với profit
        significant_corr = corr_df[abs(corr_df["Profit"]) > 0.5]["Profit"].drop("Profit", errors="ignore")
        significant_corr = significant_corr.to_dict()
        # Chuyển đổi thành chuỗi mô tả
        if significant_corr:
            corr_list = [
                f"{k} ({v:+.2f})" for k, v in significant_corr.items()
            ]
            narrative = (
                "Ma trận tương quan cho thấy các mối quan hệ đáng chú ý giữa lợi nhuận và các chỉ số khác: "
                f"<b>{fmt.list_to_text(corr_list)}</b>. "
                "Những mối tương quan này cung cấp cái nhìn sâu sắc về các yếu tố ảnh hưởng đến lợi nhuận, "
                "hỗ trợ phân tích nguyên nhân gốc rễ."
            )
                
        else:
            narrative = (
                "Ma trận tương quan <b>không cho thấy</b> mối quan hệ mạnh mẽ nào "
                "giữa lợi nhuận và các chỉ số kinh doanh chính khác. "
                "Điều này cho thấy lợi nhuận có thể bị ảnh hưởng bởi nhiều yếu tố phức tạp hơn."
            )

        return {
            "dataframe": corr_df,
            "chart_path": chart_path,
            "narrative": narrative
        }

    # --------------------------------------------------
    # 2. Sales vs Profit
    # --------------------------------------------------
    def sales_profit_relationship(
        self,
        filename: str = "sales_vs_profit.png"
    ) -> Dict:

        if not {"Sales", "Profit"}.issubset(self.df.columns):
            return {
                "dataframe": None,
                "chart_path": None,
                "narrative": "❌ Thiếu cột Sales hoặc Profit."
            }

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=self.df, x="Sales", y="Profit", alpha=0.6, ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Sales vs Profit", weight="bold")

        chart_path = self._save_chart(fig, filename)

        total_orders = len(self.df)
        loss_orders = self.df[self.df["Profit"] < 0]
        total_loss_orders = len(loss_orders)
        loss_ratio = total_loss_orders / total_orders if total_orders > 0 else 0

        narrative = (
            f"Phân tích mối quan hệ giữa doanh thu và lợi nhuận cho thấy "
            f"trong tổng số <b>{fmt.format_integer(total_orders)} giao dịch</b>, có đến "
            f"<b>{fmt.format_integer(total_loss_orders)} giao dịch</b> thua lỗ, chiếm tỷ lệ "
            f"<b>{fmt.format_percentage(loss_ratio)}</b> tổng giao dịch. "
            f"Có thể quan sát thấy nhiều đơn hàng có giá trị cao "
            f"nhưng cũng bị lỗ. Điều này cho thấy tăng doanh thu không luôn "
            f"đồng nghĩa với tăng lợi nhuận. "
            f"Ngoài ra, nên đánh giá thêm các yếu tố khác như chiết khấu, chi phí bán hàng, ..."
        )

        return {
            "dataframe": self.df[["Sales", "Profit"]],
            "chart_path": chart_path,
            "narrative": narrative
        }

    # --------------------------------------------------
    # 3. Pareto – Profit Drivers
    # --------------------------------------------------
    def pareto_profit_by_subcategory(self) -> Dict:
        return self._build_pareto(
            df=self.df,
            group_col="Sub-Category",
            value_col="Profit",
            title="Pareto – Profit by Sub-Category",
            filename="pareto_profit_subcategory.png",
            focus="positive"
        )

    # --------------------------------------------------
    # 4. Pareto – Loss Contributors
    # --------------------------------------------------
    def pareto_loss_by_subcategory(self) -> Dict:
        return self._build_pareto(
            df=self.df,
            group_col="Sub-Category",
            value_col="Profit",
            title="Pareto – Loss by Sub-Category",
            filename="pareto_loss_subcategory.png",
            focus="negative"
        )

    # --------------------------------------------------
    # 5. Pareto – Discount Impact
    # --------------------------------------------------
    def pareto_discount_impact(self) -> Dict:
        df = self.df.copy()
        df["Discount_Bin"] = pd.cut(
            df["Discount"],
            bins=[-0.01, 0, 0.2, 0.4, 0.6, 1],
            labels=["0%", "0–20%", "20–40%", "40–60%", ">60%"]
        )

        return self._build_pareto(
            df=df,
            group_col="Discount_Bin",
            value_col="Profit",
            title="Pareto – Discount Impact on Profit",
            filename="pareto_discount_impact.png",
            focus="negative"
        )
        # --------------------------------------------------
    # 6. Profit vs Loss Composition by Sub-Category
    # --------------------------------------------------
    def profit_loss_composition_by_subcategory(
        self,
        filename: str = "profit_loss_composition_subcategory.png"
        ) -> Dict:

        if not {"Sub-Category", "Profit"}.issubset(self.df.columns):
            return {
                "dataframe": None,
                "chart_path": None,
                "narrative": "❌ Thiếu cột Sub-Category hoặc Profit."
            }

        df = self.df.copy()

        summary = (
            df.groupby("Sub-Category")["Profit"]
            .agg(
                profit_positive=lambda x: x[x > 0].sum(),
                profit_negative=lambda x: x[x < 0].sum(),
                net_profit="sum"
            )
            .reset_index()
        )

        summary["loss_ratio"] = (
            summary["profit_negative"].abs() /
            (summary["profit_positive"] + summary["profit_negative"].abs())
        ).round(2)

        summary = summary.sort_values("net_profit", ascending=False)

        # ---- Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.bar(
            summary["Sub-Category"],
            summary["profit_positive"],
            label="Profit (Positive)"
        )

        ax.bar(
            summary["Sub-Category"],
            summary["profit_negative"],
            bottom=summary["profit_positive"],
            label="Loss (Negative)"
        )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Profit vs Loss Composition by Sub-Category", weight="bold")
        ax.set_ylabel("Profit")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

        chart_path = self._save_chart(fig, filename)

        # ---- Narrative (Diagnostic-level)
        risky = summary[
            (summary["net_profit"] > 0) &
            (summary["loss_ratio"] > 0.3)
        ]

        if not risky.empty:
            risky_list = ", ".join(risky["Sub-Category"].head(3))
            narrative = (
                "Phân tích cấu phần lợi nhuận cho thấy một số Sub-Category "
                f"như <b>{risky_list}</b> có <b>tổng lợi nhuận ròng dương</b>, "
                "nhưng đồng thời tồn tại <b>tỷ lệ thua lỗ cao</b>. "
                "Điều này cho thấy rủi ro không nằm ở danh mục sản phẩm, "
                "mà nằm ở điều kiện bán hàng (chiết khấu, vận chuyển, khách hàng)."
            )
        else:
            narrative = (
                "Cấu trúc lợi nhuận theo Sub-Category cho thấy "
                "phần lớn lợi nhuận đến từ các giao dịch hiệu quả, "
                "với tỷ lệ thua lỗ được kiểm soát."
            )

        return {
            "dataframe": summary,
            "chart_path": chart_path,
            "narrative": narrative
        }
        # --------------------------------------------------
    # Discount Risk by Sub-Category
    # --------------------------------------------------
    def discount_risk_by_subcategory(
        self,
        discount_threshold: float = 0.2,
        filename: str = "discount_risk_subcategory.png"
    ) -> Dict:

        if not {"Sub-Category", "Discount", "Profit"}.issubset(self.df.columns):
            return {
                "dataframe": None,
                "chart_path": None,
                "narrative": "❌ Thiếu cột Sub-Category, Discount hoặc Profit."
            }

        df = self.df[self.df["Discount"] > discount_threshold]

        summary = (
            df.groupby("Sub-Category")["Profit"]
            .sum()
            .sort_values()
            .reset_index()
        )

        # Plot
        # Tao mau tuong ung voi gia tri loi nhuan
        
        values = summary["Profit"]

        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

        base_cmap = plt.cm.Oranges_r   # 👈 đảo ngược Blues

        cmap = mcolors.LinearSegmentedColormap.from_list(
        "Oranges_truncated_rev",
        base_cmap(np.linspace(0.3, 1.0, 256))
        )
        # Nếu giá trị không âm thì dùng màu steelblue
        bar_colors = [
            "steelblue" if v >= 0 else cmap(norm(v))
            for v in values
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(summary["Sub-Category"], summary["Profit"], color=bar_colors)
        ax.axvline(0, color="black")
        ax.set_title(
            f"Profit Impact by Sub-Category (Discount > {int(discount_threshold*100)}%)",
            weight="bold"
        )
        ax.set_xlabel("Profit")

        chart_path = self._save_chart(fig, filename)

        # Narrative
        worst = summary.head(3)

        narrative = (
            f"Phân tích các đơn hàng có mức chiết khấu trên <b>{fmt.format_percentage(discount_threshold)}</b> "
            "cho thấy thua lỗ tập trung chủ yếu ở một số Sub-Category nhất định. "
            f"Các nhóm như <b>{', '.join(worst['Sub-Category'])}</b> "
            "đóng góp <b>phần lớn tổng mức lỗ</b>, cho thấy <b>chiết khấu cao</b> "
            "không phù hợp với <b>cấu trúc lợi nhuận</b> của các sản phẩm này."
        )

        return {
            "dataframe": summary,
            "chart_path": chart_path,
            "narrative": narrative
        }

    # --------------------------------------------------
    # 6. Key Insights & Recommendations
    # --------------------------------------------------
    def key_insights_and_recommendations(self, results: Dict) -> Dict:
        insights, recommendations = [], []

        for key in [
            "pareto_profit_by_subcategory",
            "pareto_loss_by_subcategory",
            "pareto_discount_impact"
        ]:
            narrative = results.get(key, {}).get("narrative")
            if narrative:
                insights.append(narrative)

        recommendations.extend([
            "Ưu tiên đầu tư vào các nhóm sản phẩm tạo ra phần lớn lợi nhuận.",
            "Rà soát hoặc tái cấu trúc các nhóm sản phẩm gây thua lỗ kéo dài.",
            "Thiết lập ngưỡng kiểm soát chiết khấu để hạn chế tác động tiêu cực đến lợi nhuận."
        ])

        return {
            "insights": insights,
            "recommendations": recommendations
        }

    # --------------------------------------------------
    # 7. Run all
    # --------------------------------------------------
    def run_all(self) -> Dict:
        results = {
            "correlation_matrix": self.correlation_matrix(),
            "sales_profit_relationship": self.sales_profit_relationship(),
            "pareto_profit_by_subcategory": self.pareto_profit_by_subcategory(),
            "pareto_loss_by_subcategory": self.pareto_loss_by_subcategory(),
            "profit_loss_composition_by_subcategory": self.profit_loss_composition_by_subcategory(),
            "pareto_discount_impact": self.pareto_discount_impact(),
            "discount_risk_by_subcategory": self.discount_risk_by_subcategory()
        }

        results["key_insights_and_recommendations"] = (
            self.key_insights_and_recommendations(results)
        )

        return results
