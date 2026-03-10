import os
import webbrowser as br

from helpers.data_loader import DataLoader
from helpers.data_explorer import DataExplorer
from helpers.data_cleaner import DataCleaner
from helpers.report_creator import save_html_report

from analysis.descriptive import DescriptiveAnalysis
from analysis.diagnostic import DiagnosticAnalysis
from analysis.cohort import CohortAnalysis
from analysis.rfm import RFMAnalysis
from analysis.forecast import ForecastAnalysis
from analysis.management import management_recommendations


# ==================================================
# PATH CONFIGURATION
# ==================================================

BASE_DIR = os.path.abspath(".")

DATA_DIR   = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "outputs")
CHART_DIR  = os.path.join(REPORT_DIR, "charts")

DATA_PATH  = os.path.join(DATA_DIR, "Superstore.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "Superstore_clean.csv")

REPORT_HTML = os.path.join(REPORT_DIR, "report.html")
REPORT_PDF  = os.path.join(REPORT_DIR, "report.pdf")


# ==================================================
# MAIN PIPELINE
# ==================================================

def run_pipeline():

    os.makedirs(CHART_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1. LOAD RAW DATA
    # --------------------------------------------------
    print("[1] Load dữ liệu gốc...")
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    # --------------------------------------------------
    # 2. DATA EXPLORATION
    # --------------------------------------------------
    print("[2] Khám phá dữ liệu ban đầu...")
    explorer = DataExplorer(df, chart_dir=CHART_DIR)
    explorer_results = explorer.run_all()

    # --------------------------------------------------
    # 3. DATA CLEANING
    # --------------------------------------------------
    print("[3] Làm sạch dữ liệu...")
    cleaner = DataCleaner(df, chart_dir=CHART_DIR)
    cleaning_results = cleaner.run_all()
    cleaner.save_clean_data(CLEAN_PATH)

    # Reload clean data
    loader = DataLoader(CLEAN_PATH)
    df = loader.load()
    print(f"[INFO] Dataset sau làm sạch: {df.shape[0]} rows × {df.shape[1]} columns")

    # --------------------------------------------------
    # 4. ANALYTICAL MODULES
    # --------------------------------------------------
    print("[4] Phân tích mô tả...")
    desc_results = DescriptiveAnalysis(df, chart_dir=CHART_DIR).run_all()

    print("[5] Phân tích chẩn đoán...")
    diag_results = DiagnosticAnalysis(df, chart_dir=CHART_DIR).run_all()

    print("[6] Phân tích Cohort...")
    cohort_results = CohortAnalysis(
        df,
        chart_dir=CHART_DIR,
        mode="last_12_months"
    ).run_all()

    print("[7] Phân tích RFM...")
    rfm_results = RFMAnalysis(df, chart_dir=CHART_DIR).run_all()

    print("[8] Dự báo doanh thu...")
    forecast_results = ForecastAnalysis(df, chart_dir=CHART_DIR).run_all()

    # --------------------------------------------------
    # 5. MANAGEMENT INSIGHTS
    # --------------------------------------------------
    management_results = management_recommendations({
        "descriptive": desc_results,
        "rfm": rfm_results,
        "forecast": forecast_results
    })

    # --------------------------------------------------
    # 6. REPORT CONTEXT
    # --------------------------------------------------
    
    context = {
        "explorer": explorer_results,
        "cleaning": cleaning_results,
        "desc": desc_results,
        "diag": diag_results,
        "cohort": cohort_results,
        "rfm": rfm_results,
        "forecast": forecast_results,
        "management": management_results
    }

    # --------------------------------------------------
    # 7. REPORT GENERATION
    # --------------------------------------------------
    print("[9] Xuất báo cáo HTML...")
    save_html_report(
        template_path="reports/templates/report_template.html",
        output_path=REPORT_HTML,
        context=context
    )
    br.open(REPORT_HTML)

    # print("[10] Xuất báo cáo PDF...")
    # save_pdf_report(REPORT_HTML, REPORT_PDF)

    print("\n✓ Hoàn thành toàn bộ pipeline")
    print(f"HTML: {REPORT_HTML}")
    # print(f"PDF : {REPORT_PDF}")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    run_pipeline()
