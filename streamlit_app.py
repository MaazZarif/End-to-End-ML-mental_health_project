"""
Student Mental Health Prediction — Streamlit Frontend
======================================================
A client UI for the FastAPI prediction service defined in `main.py`.

Run:
    streamlit run streamlit_app.py

Make sure the FastAPI backend is running first, e.g.:
    uvicorn main:app --reload --port 8000
"""

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Minimal custom styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .subtitle {
            color: #7a7a7a;
            font-size: 1.0rem;
            margin-bottom: 1.5rem;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1rem;
        }
        .result-score {
            font-size: 2.6rem;
            font-weight: 800;
        }
        .result-label {
            font-size: 1.0rem;
            opacity: 0.85;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128,128,128,0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — API configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ API Settings")
    api_base_url = st.text_input(
    "FastAPI base URL",
    value="https://mental-health-api.onrender.com",
)

    st.divider()

    if st.button("Check API health", use_container_width=True):
        try:
            resp = requests.get(f"{api_base_url}/health", timeout=5)
            if resp.status_code == 200:
                st.success(resp.json().get("message", "Healthy"))
            else:
                st.error(f"API responded with status {resp.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach API: {exc}")

    st.divider()
    st.caption(
        "This app sends form inputs to the `/predict` endpoint of the "
        "Student Mental Health Prediction API and displays the returned score."
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="main-title">🧠 Student Mental Health Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estimate a student\'s mental health score from lifestyle, '
    'academic, and social media usage patterns.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Option lists — adjust to match the categories your model was trained on
# --------------------------------------------------------------------------
GENDER_OPTIONS = ["Male", "Female", "Other"]
ACADEMIC_LEVEL_OPTIONS = ["High School", "Undergraduate", "Graduate"]
PLATFORM_OPTIONS = [
    "Instagram", "Facebook", "TikTok", "YouTube", "Twitter",
    "Snapchat", "WhatsApp", "LinkedIn", "Other",
]
COUNTRY_OPTIONS = [
    "USA", "India", "UK", "Canada", "Pakistan", "Australia",
    "Germany", "Other",
]
STRESS_LEVEL_OPTIONS = ["Low", "Medium", "High"]
PURPOSE_OPTIONS = [
    "Entertainment", "Education", "Social Interaction",
    "News", "Work", "Other",
]

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("👤 Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
    with col2:
        academic_level = st.selectbox("Academic Level", ACADEMIC_LEVEL_OPTIONS)
        country = st.selectbox("Country", COUNTRY_OPTIONS)

    st.subheader("📱 Social Media Usage")
    col3, col4 = st.columns(2)
    with col3:
        avg_daily_usage_hours = st.number_input(
            "Avg. Daily Usage (hours)", min_value=0.0, value=3.0, step=0.5, format="%.1f"
        )
        daily_unlocks = st.number_input("Daily Phone Unlocks", min_value=0, value=50, step=1)
        number_of_apps = st.number_input(
            "Number of Social Media Apps", min_value=0, value=4, step=1
        )
    with col4:
        most_used_platform = st.selectbox("Most Used Platform", PLATFORM_OPTIONS)
        time_spent_social_media = st.number_input(
            "Time Spent on Social Media (hours)", min_value=0.0, value=2.5, step=0.5, format="%.1f"
        )
        purpose_of_use = st.selectbox("Purpose of Use", PURPOSE_OPTIONS)

    st.subheader("🏃 Lifestyle & Wellbeing")
    col5, col6, col7 = st.columns(3)
    with col5:
        study_hours = st.number_input("Study Hours/Day", min_value=0.0, value=3.0, step=0.5, format="%.1f")
    with col6:
        physical_activity_hours = st.number_input(
            "Physical Activity (hours/day)", min_value=0.0, value=1.0, step=0.5, format="%.1f"
        )
    with col7:
        sleep_hours = st.number_input(
            "Sleep (hours/night)", min_value=0.0, value=7.0, step=0.5, format="%.1f"
        )

    stress_level = st.select_slider("Stress Level", options=STRESS_LEVEL_OPTIONS, value="Medium")

    submitted = st.form_submit_button("🔮 Predict Mental Health Score", use_container_width=True)

# --------------------------------------------------------------------------
# Handle submission
# --------------------------------------------------------------------------
if submitted:
    payload = {
        "Age": int(age),
        "Gender": gender,
        "Academic_Level": academic_level,
        "Avg_Daily_Usage_Hours": float(avg_daily_usage_hours),
        "Daily_Unlocks": int(daily_unlocks),
        "Most_Used_Platform": most_used_platform,
        "Time_Spent_on_Social_Media": float(time_spent_social_media),
        "Number_of_Social_Media_Apps": int(number_of_apps),
        "Country": country,
        "Study_Hours": float(study_hours),
        "Physical_Activity_Hours": float(physical_activity_hours),
        "Sleep_Hours_Per_Night": float(sleep_hours),
        "Stress_Level": stress_level,
        "Purpose_Of_Use": purpose_of_use,
    }

    with st.spinner("Contacting prediction service..."):
        try:
            response = requests.post(f"{api_base_url}/predict", json=payload, timeout=120)
        except requests.exceptions.RequestException as exc:
            st.error(f"❌ Could not reach the API at `{api_base_url}`.\n\n**Details:** {exc}")
            st.stop()

    if response.status_code == 200:
        result = response.json()
        score = result.get("prediction")

        # Basic banding for a friendly interpretation — tune thresholds to your target scale
        if score is None:
            st.error("The API responded but did not include a prediction value.")
        else:
            if score >= 7:
                band, color = "Good", "#1f9d55"
            elif score >= 4:
                band, color = "Moderate", "#d97706"
            else:
                band, color = "Needs Attention", "#dc2626"

            st.markdown(
                f"""
                <div class="result-card" style="background-color:{color}22; border:1px solid {color};">
                    <div class="result-score" style="color:{color};">{score:.2f}</div>
                    <div class="result-label">Predicted Mental Health Score — {band}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View submitted data"):
                st.json(payload)
    else:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"❌ Prediction failed (status {response.status_code}): {detail}")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption("Built with Streamlit • Powered by FastAPI + your trained ML pipeline")