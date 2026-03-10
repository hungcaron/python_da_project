import streamlit as st
import pandas as pd
import os

from analysis.descriptive import DescriptiveAnalysis
from analysis.diagnostic import DiagnosticAnalysis
from analysis.cohort import CohortAnalysis
from analysis.rfm import RFMAnalysis
from analysis.forecast import ForecastAnalysis


st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Superstore Analytics Dashboard")

# ============================
# Sidebar
# ============================
st.sidebar.header("⚙️ Settings")

data_file = st.sidebar.text_input(
    "CSV input file:",
    value="data/Superstore.csv"
)

run_button = st.sidebar.button("Run Analysis")

# ============================
# Load dataset
# ============================
if not os.path.exists(data_file):
    st.error(f"Data file not found: {data_file}")
    st.stop()

df = pd.read_csv(data_file, encoding="latin1")

st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# ============================
# Prepare output folders
# ============================
OUTPUT_DIR = "reports/outputs"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ============================
# Run analysis
# ============================
if run_button:
    st.success("Running analysis...")

    # =======================
    # 1. Descriptive Analysis
    # =======================
    desc = DescriptiveAnalysis(df, chart_dir=CHART_DIR).run_all()

    st.header("1️⃣ Descriptive Analysis")
    st.subheader("Summary Statistics")
    st.write(desc["summary"], unsafe_allow_html=True)

    st.subheader("Missing Values")
    st.dataframe(desc["missing_values"])

    st.subheader("Sales by Category")
    st.image(os.path.join(CHART_DIR, "sales_by_category.png"))

    # =======================
    # 2. Diagnostic Analysis
    # =======================
    st.header("2️⃣ Diagnostic Analysis")
    diag = DiagnosticAnalysis(df, chart_dir=CHART_DIR).run_all()

    st.subheader("Correlation Matrix")
    st.image(os.path.join(CHART_DIR, "correlation_matrix.png"))

    st.subheader("Sales vs Profit")
    st.image(os.path.join(CHART_DIR, "sales_vs_profit.png"))

    # =======================
    # 3. Cohort Analysis
    # =======================
    st.hea
