import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import joblib
from datetime import datetime
import requests
import re
# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Smart Study Abroad Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Minimalistic Light UI CSS
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f7f9fc;
    color: #1a1a1a;
}

/* Card for universities */
.uni-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    border: 1px solid #e5e7eb;
}

/* Header box */
.glass {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e5e7eb;
}

/* Buttons */
.stButton>button {
    border-radius: 6px;
    padding: 8px 16px;
    background: #2563eb;
    color: white;
    border: none;
    font-weight: 500;
}
.stButton>button:hover {
    background: #1e4dc4;
}

/* Input & Select fields */
div[data-baseweb="select"], .stNumberInput, .stTextInput, .stRadio {
    background: #ffffff !important;
    border-radius: 6px;
    padding: 4px;
    border: 1px solid #e5e7eb;
}

/* Plotly chart background fix */
.js-plotly-plot .plot-container .svg-container {
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)



# -------------------------
# Load model & scaler (ADMISSION / CHANCE MODEL)
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------
# Load master data
# -------------------------
master = pd.read_csv("master_data_updated.csv")
master.columns = master.columns.str.strip().str.lower()

required_cols = [
    "country","city","university","program","level",
    "total_cost","academic_reputation_score",
    "overall_score","university_rating"
]
for c in required_cols:
    if c not in master.columns:
        st.error(f"❌ Missing column in master_data_updated.csv: {c}")
        st.stop()

for col in ["total_cost","academic_reputation_score","overall_score","university_rating",
            "tuition_usd","rent_usd","visa_fee_usd","insurance_usd"]:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")

master = master.dropna(subset=["total_cost","academic_reputation_score","overall_score","university_rating"])

# -------------------------
# Priority score function
# -------------------------
def compute_priority_scores(df, preference):
    cost = df["total_cost"]
    cost_norm = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
    cost_score = 1 - cost_norm

    rep = df["academic_reputation_score"]
    rep_norm = (rep - rep.min()) / (rep.max() - rep.min() + 1e-9)

    roi_norm = df["roi_score"] if "roi_score" in df.columns else pd.Series(0, index=df.index)

    if preference == "Affordability":
        score = 0.7 * cost_score + 0.2 * rep_norm + 0.1 * roi_norm
    elif preference == "Reputation":
        score = 0.1 * cost_score + 0.6 * rep_norm + 0.3 * roi_norm
    else:
        score = 0.3 * cost_score + 0.3 * rep_norm + 0.4 * roi_norm

    return score.clip(0, 1), cost_score, rep_norm, roi_norm

# -------------------------
# Image fetching function       
# -------------------------


def google_image_preview(university, city, country):
    query = f"{university} {city} {country} campus"
    query = query.replace(" ", "+")
    return f"https://www.google.com/search?tbm=isch&q={query}"


# -------------------------
# Chance prediction function
# -------------------------
def predict_chance(df, gre, toefl, cgpa):
    n = len(df)
    ratings = df["university_rating"].astype(float).values

    base = np.column_stack([np.full(n, gre), np.full(n, toefl), np.full(n, cgpa)])
    scaled_vals = scaler.transform(base)

    X = np.column_stack([scaled_vals[:,0], scaled_vals[:,1], ratings, scaled_vals[:,2]])
    preds = model.predict(X)
    return np.clip(preds, 0, 1)

# -------------------------
# Future Cost prediction function (SEPARATE MODEL)
# -------------------------
@st.cache_resource
def load_artifacts():
    future_model = joblib.load("model_future.pkl")
    future_scaler = joblib.load("scaler_future.gz")
    return future_model, future_scaler

future_model, future_scaler = load_artifacts()

# ===== Load Historical USD–INR Data =====
@st.cache_data
def load_data():
    fx_df = pd.read_csv("USD-INR.csv", parse_dates=["Date"])
    fx_df = fx_df.sort_values("Date")
    fx_df = fx_df[["Date", "Close"]].rename(columns={"Close":"rate"})
    fx_df = fx_df.set_index("Date")
    return fx_df

fx_df = load_data()

# ===== Prediction Function =====
def predict_inr_amount(year: int, amount_usd: float):
    """
    Predict INR amount for a USD amount in a target future year.
    Returns (inr_value, predicted_rate).
    """
    current_year = datetime.now().year
    if year <= current_year:
        raise ValueError("Year must be greater than the current year")

    # Convert year distance to approximate days (not used in RF logic, kept for consistency)
    days_ahead = (year - current_year) * 365  # symbolic

    # Last available rate data from FX series
    last_rate = fx_df["rate"].iloc[-1]
    last_7 = fx_df["rate"].iloc[-7]
    last_30 = fx_df["rate"].iloc[-30]

    # Scale inputs
    X = future_scaler.transform([[last_rate, last_7, last_30]])

    # Predict exchange rate (1 USD → INR)
    predicted_rate = future_model.predict(X)[0]

    # Convert USD → INR
    inr_value = predicted_rate * amount_usd
    return round(inr_value, 2), round(predicted_rate, 2)

# -------------------------
# UI
# -------------------------
st.markdown("<div class='glass'><h2>🎓 UNISPHERE Smart Study Abroad Recommender</h2></div>", unsafe_allow_html=True)
st.write("")
b = master["program"].unique()
d = master["level"].unique()

with st.form("form"):
    name = st.text_input("Full Name")
    degree = st.selectbox("Desired Degree (MS, MBA, MEng, etc.)", d)
    branch = st.selectbox("Specialization/Program (CS, AI, Data, Mechanical, Finance, etc.)", b)

    gre = st.number_input("GRE Score", 260, 340, 310)
    toefl = st.number_input("TOEFL Score", 0, 120, 100)
    cgpa = st.number_input("CGPA (0–10)", 0.0, 10.0, 8.0, step=0.01)

    preference = st.radio("Your Priority", ["Affordability", "Reputation", "Both"], horizontal=True)

    submitted = st.form_submit_button("🔍 Get Recommendations")

# -------------------------
# Recommendation Logic
# -------------------------
if submitted:
    df = master.copy()

    if branch.strip():
        df = df[df["program"].str.contains(branch.strip(), case=False, na=False)]

    if df.empty:
        df = master.copy()

    df["chance"] = predict_chance(df, gre, toefl, cgpa)

    filtered = df[df["chance"] >= 0.35]
    if filtered.empty:
        filtered = df.copy()
    df = filtered.copy()

    df["priority"], df["afford"], df["rep_norm"], df["roi_norm"] = compute_priority_scores(df, preference)

    overall = df["overall_score"]
    df["overall_norm"] = (overall - overall.min()) / (overall.max() - overall.min() + 1e-9)

    df["final"] = 0.6 * df["chance"] + 0.25 * df["priority"] + 0.15 * df["overall_norm"]

    df = df.sort_values(by="final", ascending=False).reset_index(drop=True)
    top = df.drop_duplicates(subset=["university"]).head(10).reset_index(drop=True)

    st.session_state["top"] = top
    st.session_state["selected_uni"] = None

# -------------------------
# Display Results if Available
# -------------------------
if "top" in st.session_state:

    top = st.session_state["top"]

    st.markdown("### 🏆 Top Recommended Universities")

    left, right = st.columns([1, 1.2])

    # LEFT LIST
    with left:
        for i, row in top.iterrows():
            st.markdown(f"""
            <div class='uni-card'>
                <b>{i+1}. {row['university']}</b><br>
                <span style='font-size:13px;color:#cbd5e1;'>📍 {row['city']}, {row['country']}</span><br>
                ⭐ Rating: {int(row['university_rating'])}/5<br>
                🎯 Final Score: <b>{row['final']*100:.1f}%</b><br>
                📈 Chance: <b>{row['chance']*100:.1f}%</b><br>
                💰 Total Cost(on an average): <b>${int(row['total_cost']):,}</b>
            </div>
            """, unsafe_allow_html=True)

            # View Details Button
            if st.button(f"View Details — {i+1}", key=f"btn{i}"):
                st.session_state["selected_uni"] = row["university"]

    # RIGHT DETAILS SECTION
    with right:

        # Bar Chart
        fig = px.bar(
            top.sort_values("final"),
            x="final", y="university",
            title="Final Score Comparison",
            orientation="h"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show Selected University Details
        if st.session_state.get("selected_uni"):

            u = top[top["university"] == st.session_state["selected_uni"]].iloc[0]

            st.markdown(f"### 🔍 {u['university']}")
            st.write(f"📍 {u['city']}, {u['country']}")
            st.write(f"⭐ Rating: {int(u['university_rating'])}/5")
            st.write(f"📈 Chance: {u['chance']*100:.1f}%")
            st.write(f"🎯 Final Score: {u['final']*100:.1f}%")
            st.write(f"🎯 Total Cost: ${u['total_cost']}")

            # University image if available
            # Attempt to fetch image dynamically
            search_url = google_image_preview(u['university'], u['city'], u['country'])

            st.markdown(
                f"""
                <a href="{search_url}" target="_blank">
                    <img src="https://source.unsplash.com/featured/?{u['university'].replace(' ', '+')},campus,university" 
                        style="width:100%;border-radius:12px;">
                </a>
                """,
                unsafe_allow_html=True
            )
            st.caption("📷 Click image to view more campus photos on Google Images")


            # Cost Breakdown Pie Chart
            cost_parts = {}
            for col, label in [
                ("tuition_usd", "Tuition"),
                ("rent_usd", "Rent"),
                ("visa_fee_usd", "Visa"),
                ("insurance_usd", "Insurance")
            ]:
                if col in u.index and not pd.isna(u[col]) and u[col] > 0:
                    cost_parts[label] = u[col]

            if cost_parts:
                cost_df = pd.DataFrame({"component": list(cost_parts.keys()), "amount": list(cost_parts.values())})
                fig_pie = px.pie(cost_df, names="component", values="amount", title="Cost Breakdown")
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Score Profile Chart
            score_df = pd.DataFrame({
                "metric": ["Chance", "Priority", "Overall", "Final"],
                "value": [u["chance"], u["priority"], u["overall_norm"], u["final"]]
            })
            fig_line = px.line(score_df, x="metric", y="value", markers=True, title="Score Profile")
            fig_line.update_layout(
                yaxis=dict(range=[0, 1]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_line, use_container_width=True)

            st.write('''Are you planning to study in a future year? 
                     Use the tool below to predict future costs based on USD to INR exchange rates!''')

            # Future Cost Prediction UI (persistent)
            if st.button("Predict Future Cost", key="predict_fx"):
                st.session_state["fx_predict_mode"] = True

            if st.session_state.get("fx_predict_mode", False):
                st.markdown("### 📌 Future Cost Prediction")
                year = st.number_input("Enter Future Year (e.g., 2027)", min_value=2025, max_value=2050, value=2027, key="fx_year")
                amount_usd = u['total_cost']  # use university's total cost automatically

                if st.button("Predict Future INR Value", key="predict_fx_run"):
                    try:
                        inr_value, predicted_rate = predict_inr_amount(year, amount_usd)

                        st.success(f"📌 Predicted Exchange Rate in {year}: **₹{predicted_rate:,.2f} per 1 USD**")
                        st.info(f"💰 {amount_usd:,.2f} USD ≈ **₹{inr_value:,.2f} INR** in {year}")

                        st.write(
                            f"Based on the model, **${amount_usd:,.2f} USD** "
                            f"may be worth approximately **₹{inr_value:,.2f} INR** in **{year}**."
                        )

                    except ValueError as e:
                        st.error(str(e))

    # Download Button (for top 10) – kept outside `with right` but inside "top" block
    st.download_button(
        "📥 Download Top 10 CSV",
        top.to_csv(index=False),
        "top10.csv",
        "text/csv"
    )
