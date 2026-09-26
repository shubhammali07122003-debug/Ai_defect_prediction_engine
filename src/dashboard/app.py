import os
import sys
import pandas as pd
import requests
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="AI Defect Prediction Engine",
    page_icon="🛡️",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

st.title("🛡️ AI Software Defect Prediction & Code Quality Intelligence Engine")
st.markdown(
    "Analyze code metrics to estimate defect risks and prioritize code reviews using our trained ML pipeline."
)

# Sidebar - API Health Monitor
st.sidebar.header("⚙️ System Status")
try:
    health_res = requests.get(f"{API_URL}/", timeout=3)
    if health_res.status_code == 200:
        health_data = health_res.json()
        st.sidebar.success("🟢 API Server: Online")
        st.sidebar.info(f"Model Loaded: {health_data.get('model_loaded')}")
        st.sidebar.info(f"Pipeline Loaded: {health_data.get('pipeline_loaded')}")
    else:
        st.sidebar.error("🔴 API Server Error")
except Exception:
    st.sidebar.error("🔴 API Server Offline (Check Uvicorn)")

# Create Tabs for Single Module vs Batch CSV Prediction
tab1, tab2 = st.tabs(["📊 Single File Analysis", "📁 Batch CSV Prediction & Priority Queue"])

# ---------------------------------------------------------
# TAB 1: SINGLE FILE ANALYSIS
# ---------------------------------------------------------
with tab1:
    st.subheader("Single File Code Metrics Input")
    col1, col2 = st.columns(2)

    with col1:
        file_path = st.text_input("File Path / Module Name", value="src/legacy/spaghetti_core.py")
        loc = st.number_input("Lines of Code (LOC)", min_value=0.0, value=350.0, step=10.0)
        cyclomatic_complexity = st.number_input(
            "Cyclomatic Complexity", min_value=0.0, value=15.0, step=1.0
        )

    with col2:
        churn_30d = st.number_input(
            "Code Churn (Last 30 Days)", min_value=0.0, value=120.0, step=10.0
        )
        prior_defects = st.number_input(
            "Prior Defects Count", min_value=0.0, value=2.0, step=1.0
        )

    if st.button("Predict Defect Risk", type="primary"):
        payload = {
            "file_path": file_path,
            "loc": loc,
            "cyclomatic_complexity": cyclomatic_complexity,
            "churn_30d": churn_30d,
            "prior_defects": prior_defects,
        }

        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if res.status_code == 200:
                data = res.json()
                prob = data["defect_probability"]
                is_defective = data["is_defective"]

                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                st.write(f"**Target Module:** `{data['file_path']}`")

                # Probability Metric Display
                st.metric(label="Defect Probability Score", value=f"{prob * 100:.2f}%")

                if is_defective:
                    st.error(
                        "🚨 **High Risk**: This module is likely defective. Consider priority code review and additional unit tests."
                    )
                else:
                    st.success("✅ **Low Risk**: This module appears stable.")

                st.caption(f"Inference Source: `{data['inference_source']}`")
            else:
                st.error(f"API Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to API endpoint: {e}")

# ---------------------------------------------------------
# TAB 2: BATCH CSV PREDICTION & PRIORITY QUEUE
# ---------------------------------------------------------
with tab2:
    st.subheader("Upload Repository Metrics CSV for Batch Analysis")
    st.markdown(
        "Upload a CSV file with columns: `file_path`, `loc`, `cyclomatic_complexity`, `churn_30d`, `prior_defects`"
    )

    # Sample CSV Download Helper
    sample_df = pd.DataFrame(
        [
            {
                "file_path": "src/auth/login.py",
                "loc": 120.0,
                "cyclomatic_complexity": 8.0,
                "churn_30d": 45.0,
                "prior_defects": 0.0,
            },
            {
                "file_path": "src/payments/processor.py",
                "loc": 850.0,
                "cyclomatic_complexity": 42.0,
                "churn_30d": 310.0,
                "prior_defects": 5.0,
            },
            {
                "file_path": "src/utils/helpers.py",
                "loc": 90.0,
                "cyclomatic_complexity": 4.0,
                "churn_30d": 10.0,
                "prior_defects": 0.0,
            },
            {
                "file_path": "src/legacy/spaghetti_core.py",
                "loc": 1850.0,
                "cyclomatic_complexity": 75.0,
                "churn_30d": 450.0,
                "prior_defects": 10.0,
            },
        ]
    )

    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=sample_df.to_csv(index=False),
        file_name="sample_code_metrics.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("### 📄 Input Data Preview", input_df.head())

            req_cols = ["loc", "cyclomatic_complexity", "churn_30d", "prior_defects"]
            if not all(col in input_df.columns for col in req_cols):
                st.error(f"CSV must contain required columns: {req_cols}")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    results = []
                    progress_bar = st.progress(0)
                    total_rows = len(input_df)

                    for idx, row in input_df.iterrows():
                        payload = {
                            "file_path": str(row.get("file_path", f"file_{idx}.py")),
                            "loc": float(row["loc"]),
                            "cyclomatic_complexity": float(row["cyclomatic_complexity"]),
                            "churn_30d": float(row["churn_30d"]),
                            "prior_defects": float(row["prior_defects"]),
                        }

                        try:
                            res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                            if res.status_code == 200:
                                res_json = res.json()
                                prob = res_json["defect_probability"]
                                results.append(
                                    {
                                        "File Path": payload["file_path"],
                                        "Defect Probability": f"{prob * 100:.2f}%",
                                        "Risk Level": (
                                            "🚨 High Risk" if prob >= 0.50 else "✅ Low Risk"
                                        ),
                                        "Raw Score": prob,
                                        "LOC": payload["loc"],
                                        "Complexity": payload["cyclomatic_complexity"],
                                        "Churn (30d)": payload["churn_30d"],
                                        "Prior Defects": payload["prior_defects"],
                                    }
                                )
                        except Exception as req_err:
                            st.warning(f"Failed row {idx}: {req_err}")

                        progress_bar.progress((idx + 1) / total_rows)

                    if results:
                        results_df = pd.DataFrame(results)

                        # Sort by Raw Score descending to form Code Review Priority Queue
                        results_df = results_df.sort_values(
                            by="Raw Score", ascending=False
                        ).reset_index(drop=True)

                        st.markdown("---")
                        st.subheader("🔥 Code Review Priority Queue (Ranked by Risk)")

                        # Highlight High Risk rows
                        def highlight_risk(val):
                            color = "background-color: #ffcccc" if "High Risk" in str(val) else ""
                            return color

                        display_df = results_df.drop(columns=["Raw Score"])
                        st.dataframe(
                            display_df.style.map(
                                highlight_risk, subset=["Risk Level"]
                            ),
                            use_container_width=True,
                        )

                        # Summary Stats
                        high_risk_count = len(
                            results_df[results_df["Risk Level"] == "🚨 High Risk"]
                        )
                        st.info(
                            f"**Batch Assessment Summary:** Found **{high_risk_count}** high-risk modules out of **{total_rows}** total files analyzed."
                        )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")