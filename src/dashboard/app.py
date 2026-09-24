import streamlit as st
import requests

st.set_page_config(
    page_title="AI Defect Prediction Engine",
    page_icon="??",
    layout="wide"
)

st.title("?? AI Defect Prediction Engine")
st.markdown("Enter code metrics below to evaluate defect risk using our trained ML model.")

API_URL = "http://127.0.0.1:8000/predict"

col1, col2 = st.columns(2)

with col1:
    st.subheader("?? Code Metrics Input")
    loc = st.number_input("Lines of Code (LOC)", min_value=0.0, value=120.0, step=10.0)
    cyclomatic_complexity = st.number_input("Cyclomatic Complexity", min_value=0.0, value=12.0, step=1.0)
    churn_30d = st.number_input("Code Churn (Last 30 Days)", min_value=0.0, value=5.0, step=1.0)
    prior_defects = st.number_input("Prior Defects Count", min_value=0.0, value=1.0, step=1.0)

    submit_button = st.button("Predict Defect Risk", type="primary")

with col2:
    st.subheader("?? Prediction Results")
    if submit_button:
        payload = {
            "loc": loc,
            "cyclomatic_complexity": cyclomatic_complexity,
            "churn_30d": churn_30d,
            "prior_defects": prior_defects
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result.get("defect_probability", 0.0)
                is_defective = result.get("is_defective", False)
                source = result.get("inference_source", "unknown")

                st.metric("Defect Probability", f"{prob * 100:.2f}%")
                st.progress(prob)

                if is_defective:
                    st.error("?? **High Risk**: This module is likely defective. Consider code review and additional unit tests.")
                else:
                    st.success("? **Low Risk**: This module appears stable.")

                st.caption(f"Inference Source: `{source}`")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend at `{API_URL}`. Make sure Uvicorn is running.")
            st.exception(e)
