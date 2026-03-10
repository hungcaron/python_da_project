import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import numpy as np


def management_recommendations(results: Dict) -> Dict:
    """
    Tổng hợp khuyến nghị quản lý từ Descriptive + RFM + Forecast
    """

    recs = []

    # -----------------------------
    # 1. Business Performance
    # -----------------------------
    desc = results.get("descriptive", {})
    sales = desc.get("sales_summary", {})
    margin = sales.get("profit_margin", 0)

    if margin < 5:
        recs.append({
            "priority": "High",
            "area": "Profitability",
            "recommendation": (
                "Rà soát lại <b>cấu trúc chi phí và chính sách giá</b>, "
                "đặc biệt ở các <b>danh mục có doanh thu cao nhưng biên lợi nhuận thấp</b>."
            ),
            "rationale": "Tỷ suất lợi nhuận bình quân hiện ở mức thấp."
        })
    else:
        recs.append({
            "priority": "Medium",
            "area": "Profitability",
            "recommendation": (
                "Tiếp tục <b>tối ưu hoá chi phí và duy trì biên lợi nhuận hiện tại</b>."
            ),
            "rationale": "Hiệu quả kinh doanh nhìn chung <b>ở mức tích cực</b>."
        })

    # -----------------------------
    # 2. Customer Value (RFM)
    # -----------------------------
    rfm = results.get("rfm", {})
    seg_dist = rfm.get("segment_distribution", {})
    seg_table = seg_dist.get("table")

    if seg_table is not None:
        top_seg = seg_table.sort_values("Revenue", ascending=False).iloc[0]["Segment"]

        recs.append({
            "priority": "High",
            "area": "Customer Strategy",
            "recommendation": (
                f"Ưu tiên duy trì và mở rộng nhóm khách hàng <b>'{top_seg}'</b> "
                f"thông qua các chương trình chăm sóc và giữ chân phù hợp."
            ),
            "rationale": "Nhóm này đóng góp <b>tỷ trọng doanh thu lớn nhất</b>."
        })

    # -----------------------------
    # 3. Forecast & Planning
    # -----------------------------
    forecast = results.get("forecast", {})
    forecast_summary = forecast.get("forecast_summary", {})
    growth = forecast_summary.get("expected_growth_rate", 0)

    if growth < 0:
        recs.append({
            "priority": "High",
            "area": "Planning & Risk",
            "recommendation": (
                "Xây dựng kịch bản ứng phó suy giảm doanh thu, "
                "bao gồm kiểm soát tồn kho và chi phí vận hành."
            ),
            "rationale": "Dự báo cho thấy xu hướng doanh thu giảm trong ngắn hạn."
        })
    else:
        recs.append({
            "priority": "Medium",
            "area": "Growth",
            "recommendation": (
                "Chuẩn bị nguồn lực và kế hoạch bán hàng để tận dụng "
                "xu hướng tăng trưởng trong giai đoạn tới, đặc biệt là vào các quý 4 hàng năm. "
                "Ngoài ra, cần xem xét chính sách chiết khấu và khuyến mãi hợp lý. "
                "Cuối cùng là xem xét việc tái kích hoạt nhóm khách hàng <b>\"At Risk\"</b>."
                
            ),
            "rationale": "Dự báo cho thấy doanh thu có xu hướng tăng."
        })

    # -----------------------------
    # 4. Strategic Focus (Always)
    # -----------------------------
    recs.append({
        "priority": "Medium",
        "area": "Strategic Focus",
        "recommendation": (
            "Tập trung nguồn lực vào các danh mục và phân khúc khách hàng "
            "mang lại giá trị dài hạn, thay vì mở rộng dàn trải."
        ),
        "rationale": "Giúp tối ưu hiệu quả sử dụng nguồn lực và tăng trưởng bền vững."
    })

    # Narrative tổng
    narrative = (
        "Các khuyến nghị quản lý được xây dựng dựa trên phân tích hiệu quả "
        "kinh doanh hiện tại, giá trị khách hàng và triển vọng tăng trưởng. "
        "Việc ưu tiên hành động theo mức độ ảnh hưởng sẽ giúp doanh nghiệp "
        "tối ưu kết quả trong ngắn hạn và trung hạn."
    )

    return {
        "narrative": narrative,
        "recommendations": recs
    }
