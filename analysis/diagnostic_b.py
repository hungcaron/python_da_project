import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import helpers.data_formatter as fmt


class DiagnosticAnalysis:
    """
    Diagnostic Analysis – Profitability Focus (Consulting-style)

    Logic:
    Phase 1 – Problem framing
    Phase 2 – Loss concentration
    Phase 3 – Profit drivers
    Phase 4 – Hidden / structural risks
    Phase 5 – Synthesis
    """

    # ==================================================
    # INIT & HELPERS
    # ==================================================
    def __init__(self, df: pd.DataFrame, chart_dir: Optional[str] = None):
        self.df = df.copy()
        self.chart_dir = Path(chart_dir) if chart_dir else Path("outputs/charts")
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.relative_chart_path = self.chart_dir.name

    def _save_chart(self, fig, filename: Optional[str]):
        if not filename:
            return None
        path = self.chart_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return f"{self.relative_chart_path}/{filename}"

    # ==================================================
    # GENERIC PARETO BUILDER (ROLE-AWARE)
    # ==================================================
    def _build_pareto(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        title: str,
        filename: str,
        focus: str,          # "positive" | "negative"
        threshold: float = 0.8
    ) -> Dict:

        data = df[[group_col, value_col]].dropna().copy()

        if focus == "positive":
            data = data[data[value_col] > 0]
            sort_ascending = False
        else:
            data = data[data[value_col] < 0]
            data[value_col] = data[value_col].abs()
            sort_ascending = False

        pareto = (
            data.groupby(group_col, as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=sort_ascending)
        )

        if pareto.empty:
            return {
                "dataframe": None,
                "chart_path": None,
                "narrative": "❌ Không đủ dữ liệu để phân tích Pareto."
            }

        pareto["pct"] = pareto[value_col] / pareto[value_col].sum()
        pareto["cum_pct"] = pareto["pct"].cumsum()

        # ---- Plot
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.bar(pareto[group_col], pareto[value_col])
        ax1.set_ylabel("Profit Impact")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(pareto[group_col], pareto["cum_pct"], marker="o")
        ax2.axhline(threshold, linestyle="--")
        ax2.set_ylabel("Cumulative %")

        ax1.set_title(title, weight="bold")
        chart_path = self._save_chart(fig, filename)

        # ---- Narrative (ROLE-CORRECT)
        key_items = pareto[pareto["cum_pct"] <= threshold]
        pct_val = key_items["pct"].sum() * 100
        key_list = fmt.list_to_text(key_items[group_col].tolist())

        if focus == "positive":
            narrative = (
                f"Phân tích Pareto cho thấy lợi nhuận **phụ thuộc mạnh vào một số ít {group_col}**. "
                f"Cụ thể, **{key_items.shape[0]} nhóm** như **{key_list}** "
                f"tạo ra khoảng **{pct_val:.1f}% tổng lợi nhuận**, "
                "đóng vai trò **động lực lợi nhuận cốt lõi** của doanh nghiệp."
            )
        else:
            narrative = (
                f"Thua lỗ **không phân bổ đồng đều** mà tập trung chủ yếu vào "
                f"**{key_items.shape[0]} nhóm {group_col}**, "
                f"chiếm khoảng **{pct_val:.1f}% tổng mức lỗ**. "
                f"Các nhóm như **{key_list}** "
                "là **nguồn gây lỗ chính**, cần được ưu tiên xử lý thay vì mở rộng doanh thu."
            )

        return {
            "dataframe": pareto,
            "chart_path": chart_path,
            "narrative": narrative
        }

    # ==================================================
    # PHASE 1 – PROBLEM FRAMING
    # ==================================================
    def sales_profit_relationship(self, filename="sales_vs_profit.png") -> Dict:
        if not {"Sales", "Profit"}.issubset(self.df.columns):
            return {"dataframe": None, "chart_path": None, "narrative": "❌ Thiếu dữ liệu."}

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(self.df, x="Sales", y="Profit", alpha=0.6, ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Sales vs Profit", weight="bold")

        chart_path = self._save_chart(fig, filename)
        loss_ratio = (self.df["Profit"] < 0).mean()

        narrative = (
            f"Có khoảng **{loss_ratio:.0%} giao dịch bị lỗ**, "
            "cho thấy tăng doanh thu **không tự động chuyển hóa thành lợi nhuận**, "
            "đặt ra nhu cầu phân tích nguyên nhân sâu hơn."
        )

        return {
            "dataframe": self.df[["Sales", "Profit"]],
            "chart_path": chart_path,
            "narrative": narrative
        }

    def correlation_matrix(self, filename="correlation_matrix.png") -> Dict:
        cols = ["Sales", "Profit", "Discount", "Quantity"]
        cols = [c for c in cols if c in self.df.columns]

        corr = self.df[cols].corr().round(2)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title("Correlation Matrix – Key Metrics", weight="bold")

        chart_path = self._save_chart(fig, filename)

        narrative = (
            "Ma trận tương quan giúp xác định các biến có khả năng "
            "liên quan trực tiếp đến suy giảm lợi nhuận, "
            "đặc biệt là vai trò của chiết khấu."
        )

        return {"dataframe": corr, "chart_path": chart_path, "narrative": narrative}

    # ==================================================
    # PHASE 2 – LOSS CONCENTRATION
    # ==================================================
    def pareto_loss_by_subcategory(self) -> Dict:
        return self._build_pareto(
            self.df, "Sub-Category", "Profit",
            "Pareto – Loss by Sub-Category",
            "pareto_loss_subcategory.png",
            focus="negative"
        )

    def pareto_discount_impact(self) -> Dict:
        df = self.df.copy()
        df["Discount_Bin"] = pd.cut(
            df["Discount"],
            [-0.01, 0, 0.2, 0.4, 0.6, 1],
            labels=["0%", "0–20%", "20–40%", "40–60%", ">60%"]
        )

        result = self._build_pareto(
            df,
            "Discount_Bin",
            "Profit",
            "Pareto – Discount Impact on Profit",
            "pareto_discount_impact.png",
            focus="negative"
        )

        # Override narrative cho rõ rủi ro chiết khấu
        result["narrative"] = (
            "Phân tích Pareto theo mức chiết khấu cho thấy "
            "phần lớn thua lỗ đến từ các đơn hàng có **discount trên 20%**. "
            "Điều này cho thấy chiết khấu cao đang **bào mòn lợi nhuận**, "
            "thay vì chỉ đóng vai trò kích cầu."
        )

        return result


    # ==================================================
    # PHASE 3 – PROFIT DRIVERS
    # ==================================================
    def pareto_profit_by_subcategory(self) -> Dict:
        return self._build_pareto(
            self.df, "Sub-Category", "Profit",
            "Pareto – Profit Drivers by Sub-Category",
            "pareto_profit_subcategory.png",
            focus="positive"
        )

    # ==================================================
    # PHASE 4 – HIDDEN / STRUCTURAL RISK
    # ==================================================
    def profit_loss_composition_by_subcategory(
        self, filename="profit_loss_composition.png"
    ) -> Dict:

        summary = (
            self.df.groupby("Sub-Category")["Profit"]
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

        risky = summary[(summary["net_profit"] > 0) & (summary["loss_ratio"] > 0.3)]

        narrative = (
            "Phân tích cấu phần lợi nhuận cho thấy một số Sub-Category "
            f"như **{fmt.list_to_text(risky['Sub-Category'].tolist())}** "
            "có lợi nhuận ròng dương nhưng tỷ lệ thua lỗ cao, "
            "hàm ý rủi ro đến từ điều kiện bán hàng hơn là sản phẩm."
            if not risky.empty else
            "Cấu trúc lợi nhuận nhìn chung ổn định, "
            "với tỷ lệ thua lỗ được kiểm soát."
        )

        return {"dataframe": summary, "chart_path": None, "narrative": narrative}

    def discount_risk_by_subcategory(
        self, discount_threshold=0.2
    ) -> Dict:

        df = self.df[self.df["Discount"] > discount_threshold]
        summary = df.groupby("Sub-Category")["Profit"].sum().sort_values().reset_index()
        worst = summary.head(3)

        narrative = (
            f"Với mức chiết khấu trên {int(discount_threshold*100)}%, "
            f"thua lỗ tập trung chủ yếu ở các Sub-Category như "
            f"**{fmt.list_to_text(worst['Sub-Category'].tolist())}**, "
            "cho thấy chiết khấu cao không phù hợp với cấu trúc lợi nhuận "
            "của các nhóm sản phẩm này."
        )

        return {"dataframe": summary, "chart_path": None, "narrative": narrative}

    # ==================================================
    # PHASE 5 – SYNTHESIS
    # ==================================================
    def key_insights_and_recommendations(self, results: Dict) -> Dict:

        insights = [
            results["sales_profit_relationship"]["narrative"],
            results["pareto_loss_by_subcategory"]["narrative"],
            results["pareto_discount_impact"]["narrative"],
            results["discount_risk_by_subcategory"]["narrative"],
        ]

        recommendations = [
            "Thiết lập chính sách chiết khấu theo từng Sub-Category.",
            "Ưu tiên xử lý các nhóm sản phẩm gây lỗ trước khi mở rộng doanh thu.",
            "Duy trì và bảo vệ các Sub-Category đóng vai trò động lực lợi nhuận."
        ]

        return {"insights": insights, "recommendations": recommendations}

    # ==================================================
    # RUN ALL
    # ==================================================
    def run_all(self) -> Dict:
        results = {
            "sales_profit_relationship": self.sales_profit_relationship(),
            "correlation_matrix": self.correlation_matrix(),
            "pareto_loss_by_subcategory": self.pareto_loss_by_subcategory(),
            "pareto_discount_impact": self.pareto_discount_impact(),
            "pareto_profit_by_subcategory": self.pareto_profit_by_subcategory(),
            "profit_loss_composition_by_subcategory": self.profit_loss_composition_by_subcategory(),
            "discount_risk_by_subcategory": self.discount_risk_by_subcategory(),
        }

        results["key_insights_and_recommendations"] = (
            self.key_insights_and_recommendations(results)
        )

        return results
