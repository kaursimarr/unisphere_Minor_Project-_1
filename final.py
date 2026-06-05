import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import math
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="UNISPHERE",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# CSS – clean + golden cards, theme-respecting
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Glass header, let background follow Streamlit theme */
.glass {
    background: rgba(255,255,255,0.9);
    border-radius: 16px;
    padding: 16px 20px;
    border: 1px solid rgba(148,163,184,0.5);
    box-shadow: 0 18px 40px rgba(15,23,42,0.08);
}
@media (prefers-color-scheme: dark) {
    .glass {
        background: rgba(15,23,42,0.95);
        border-color: rgba(148,163,184,0.7);
        box-shadow: 0 18px 40px rgba(0,0,0,0.9);
    }
}

/* 🔵 Default University Card — Blue Glow Border */
.uni-card {
    background: transparent !important;
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 16px;
    border: 2px solid #0096FF; /* Blue Glow */
    box-shadow: 0 0 12px #0096FF;
    transition: transform 0.12s ease, box-shadow 0.18s ease, border-color 0.14s ease;
}

.uni-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 18px #33B2FF;
}

/* 🟡 Selected University — Yellow/Golden Glow Border */
.uni-card.selected {
    border: 2px solid #FFD700;
    box-shadow: 0 0 18px #FFD700, 0 0 30px #FFEB3B;
}


/* Dark Mode */
@media (prefers-color-scheme: dark) {
    .uni-card {
        background: #0F141A !important;
        border: 2px solid #3B82F6;
        box-shadow: 0 8px 28px rgba(0,0,0,0.65);
        color: #F8FAFC !important;
    }
    .uni-card:hover {
        border-color: #60A5FA; /* Blue-400 */
        box-shadow: 0 14px 40px rgba(0,0,0,0.85);
    }
    .uni-card.selected {
        background: #111827 !important;
        border-color: #60A5FA;
        box-shadow:
            0 0 0 3px rgba(96,165,250,0.45),
            0 24px 48px rgba(0,0,0,0.90);
    }
}

/* Subtitle text */
.info-small { font-size: 13px; color: #64748B; }
@media (prefers-color-scheme: dark) {
    .info-small { color: #94A3B8; }
}


/* Buttons */
.stButton>button {
    border-radius: 999px;
    padding: 8px 20px;
    background: #2563eb;
    color: white;
    border: none;
    font-weight: 500;
    box-shadow: 0 8px 18px rgba(37,99,235,0.35);
}
.stButton>button:hover {
    background: #1d4ed8;
    box-shadow: 0 12px 26px rgba(37,99,235,0.45);
}

/* Left scroll panel */
.left-panel {
    max-height: 80vh;
    overflow-y: auto;
    padding-right: 8px;
}
.left-panel::-webkit-scrollbar { width: 8px; }
.left-panel::-webkit-scrollbar-thumb { background: #c7cdd6; border-radius: 10px; }

.info-small { font-size: 13px; color: #6b7280; }
@media (prefers-color-scheme: dark) {
    .info-small { color: #9ca3af; }
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 500;
}

/* Centered loader text */
.loader-text {
    text-align: center;
    font-size: 14px;
    color: #6b7280;
    margin-top: -10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Plotly theming (visible in both modes)
# -------------------------
def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = "light"

    # if base == "dark":
    #     bg = "#020617"
    #     text = "#e5e7eb"
    #     grid = "rgba(148,163,184,0.35)"
    # else:
    #     bg = "#ffffff"
    #     text = "#0f172a"
    #     grid = "rgba(148,163,184,0.35)"

    # fig.update_layout(
    #     paper_bgcolor=bg,
    #     plot_bgcolor=bg,
    #     font=dict(color=text),
    #     xaxis=dict(
    #         gridcolor=grid,
    #         zerolinecolor=grid,
    #         linecolor=grid
    #     ),
    #     yaxis=dict(
    #         gridcolor=grid,
    #         zerolinecolor=grid,
    #         linecolor=grid
    #     )
    #)
    return fig

# -------------------------
# Load models
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@st.cache_resource
def load_artifacts():
    future_model = joblib.load("model_future.pkl")
    future_scaler = joblib.load("scaler_future.gz")
    return future_model, future_scaler

future_model, future_scaler = load_artifacts()

@st.cache_data
def load_fx():
    fx_df = pd.read_csv("USD-INR.csv", parse_dates=["Date"])
    fx_df = fx_df.sort_values("Date")
    fx_df = fx_df[["Date", "Close"]].rename(columns={"Close": "rate"})
    fx_df = fx_df.set_index("Date")
    return fx_df

fx_df = load_fx()

# -------------------------
# Load master data
# -------------------------
master = pd.read_csv("master_data_updated.csv")
master.columns = master.columns.str.strip().str.lower()

required_cols = [
    "country", "city", "university", "program", "level",
    "total_cost", "academic_reputation_score",
    "overall_score", "university_rating"
]
for c in required_cols:
    if c not in master.columns:
        st.error(f"❌ Missing column in master_data_updated.csv: {c}")
        st.stop()

for col in ["total_cost", "academic_reputation_score", "overall_score", "university_rating",
            "tuition_usd", "rent_usd", "visa_fee_usd", "insurance_usd",
            "latitude", "longitude"]:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")

master = master.dropna(subset=["total_cost","academic_reputation_score","overall_score","university_rating"])

# -------------------------
# Plotly globe helpers
# -------------------------
def great_circle_arc(lat1, lon1, lat2, lon2, num_points=60):
    import numpy as _np
    lat1, lon1, lat2, lon2 = map(_np.radians, [lat1, lon1, lat2, lon2])
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2 - lat1) / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    ))
    if d == 0:
        return [_np.degrees(lat1)], [_np.degrees(lon1)]
    lats, lons = [], []
    for f in _np.linspace(0, 1, num_points):
        A = math.sin((1 - f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
        y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
        z = A * math.sin(lat1) + B * math.sin(lat2)
        lat = math.atan2(z, math.sqrt(x * x + y * y))
        lon = math.atan2(y, x)
        lats.append(_np.degrees(lat))
        lons.append(_np.degrees(lon))
    return lats, lons

def get_theme_colors():
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = "light"
    if base == "dark":
        land = "rgb(51,65,85)"
        ocean = "rgb(15,23,42)"
        country = "rgba(148,163,184,0.6)"
    else:
        land = "rgb(72, 120, 92)"
        ocean = "rgb(12,36,90)"
        country = "rgba(255,255,255,0.55)"
    return land, ocean, country

def build_plotly_globe(df, mode="list", selected_uni=None, height=360, center=None):
    """
    mode = "list"  -> lines from India to all universities
    mode = "detail" -> line only to selected_uni, centered on that route
    """
    india_lat, india_lon = 20.5937, 78.9629
    fig = go.Figure()

    landcolor, oceancolor, countrycolor = get_theme_colors()

    # only rows with valid coords from CSV
    df = df.dropna(subset=["latitude", "longitude"])

    # ----- Lines -----
    if mode == "list":
        for _, r in df.iterrows():
            lat = r["latitude"]
            lon = r["longitude"]
            lats, lons = great_circle_arc(india_lat, india_lon, float(lat), float(lon))
            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=1.4, color="rgba(56,189,248,0.7)"),
                hoverinfo="skip",
                showlegend=False
            ))
    elif mode == "detail" and selected_uni is not None:
        row = df[df["university"] == selected_uni]
        if not row.empty:
            r = row.iloc[0]
            lat = r["latitude"]
            lon = r["longitude"]
            lats, lons = great_circle_arc(india_lat, india_lon, float(lat), float(lon))
            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=2.6, color="rgba(56,189,248,0.95)"),
                hoverinfo="skip",
                showlegend=False
            ))

    # ----- University markers with glow -----
    uni_lats = df["latitude"].tolist()
    uni_lons = df["longitude"].tolist()
    uni_texts = [
        f"{r['university']}<br>{r['city']}, {r['country']}"
        for _, r in df.iterrows()
    ]

    # glow layer
    fig.add_trace(go.Scattergeo(
        lon=uni_lons,
        lat=uni_lats,
        mode="markers",
        marker=dict(
            size=[14]*len(uni_lats),
            color="rgba(56,189,248,0.22)",
            line=dict(width=0)
        ),
        hoverinfo="skip",
        showlegend=False
    ))
    # solid markers
    marker_sizes = []
    marker_colors = []
    for _, r in df.iterrows():
        if mode == "detail" and selected_uni is not None and r["university"] == selected_uni:
            marker_sizes.append(10)
            marker_colors.append("#22c55e")
        else:
            marker_sizes.append(7)
            marker_colors.append("#f97373")

    fig.add_trace(go.Scattergeo(
        lon=uni_lons,
        lat=uni_lats,
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(width=1, color="#ffffff")
        ),
        hoverinfo="text",
        text=uni_texts,
        showlegend=False
    ))

    # ----- India marker with glow -----
    fig.add_trace(go.Scattergeo(
        lon=[india_lon],
        lat=[india_lat],
        mode="markers",
        marker=dict(size=14, color="rgba(59,130,246,0.35)", line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scattergeo(
        lon=[india_lon],
        lat=[india_lat],
        text=["You (India)"],
        mode="markers+text",
        marker=dict(size=8, color="royalblue"),
        textposition="bottom center",
        hoverinfo="text",
        showlegend=False
    ))

    # ----- Projection center -----
    if center is not None and isinstance(center, dict):
        rot_lon = center.get("lon", 0)
        rot_lat = center.get("lat", 0)
    else:
        rot_lon = 0
        rot_lat = 0

    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor=landcolor,
        showocean=True, oceancolor=oceancolor,
        showcountries=True, countrycolor=countrycolor,
        showlakes=True, lakecolor=oceancolor,
        showcoastlines=True, coastlinecolor=countrycolor,
        projection_rotation=dict(lon=rot_lon, lat=rot_lat, roll=0)
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        showlegend=False
    )
    return fig

# -------------------------
# Priority / scoring
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
# Image search URL
# -------------------------
def google_image_preview(university, city, country):
    query = f"{university} {city} {country} campus".replace(" ", "+")
    return f"https://www.google.com/search?tbm=isch&q={query}"

# -------------------------
# Admission chance model
# -------------------------
def predict_chance(df, gre, toefl, cgpa):
    n = len(df)
    ratings = df["university_rating"].astype(float).values
    base = np.column_stack([np.full(n, gre), np.full(n, toefl), np.full(n, cgpa)])
    scaled_vals = scaler.transform(base)
    X = np.column_stack([scaled_vals[:, 0], scaled_vals[:, 1], ratings, scaled_vals[:, 2]])
    preds = model.predict(X)
    return np.clip(preds, 0, 1)

# -------------------------
# FX prediction
# -------------------------
def predict_inr_amount(year: int, amount_usd: float):
    current_year = datetime.now().year
    if year <= current_year:
        raise ValueError("Year must be greater than the current year")

    last_rate = fx_df["rate"].iloc[-1]
    last_7 = fx_df["rate"].iloc[-7]
    last_30 = fx_df["rate"].iloc[-30]

    X = future_scaler.transform([[last_rate, last_7, last_30]])
    predicted_rate = future_model.predict(X)[0]
    inr_value = predicted_rate * amount_usd
    return round(inr_value, 2), round(predicted_rate, 2)

# -------------------------
# Title / form
# -------------------------
st.markdown(
    "<div class='uni-card' style='text-align:center;'><h1>🎓 UNISPHERE </h1>\n<h3>Your Smart Guide to Favourable Abroad Education!✈️</h3></div>",
    unsafe_allow_html=True
)
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

with st.expander("ℹ️ How is the Final Score calculated?"):
    st.write("We calculate a Priority Score by combining Cost, Reputation, and Return on Investment, with weights based on what the student values most. Then we blend this Priority Score with a predicted Chance of Admission and the university’s Overall Global Ranking to compute the Final Composite Score that ranks universities for the student.")
    st.image("final_score.png",width=800)

# -------------------------
# Recommendation logic with loader
# -------------------------
if submitted:
    with st.spinner("🔎 Finding the best universities for you..."):
        time.sleep(1.0)

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
        st.session_state["view_mode"] = "list"
        st.session_state["fx_predict_mode"] = False

    st.markdown("<div class='loader-text'>Scroll down to view your recommended universities 👇</div>", unsafe_allow_html=True)

# -------------------------
# Main views
# -------------------------
if "top" in st.session_state and st.session_state["top"] is not None:
    top = st.session_state["top"]
    if not top.empty:
        display = top.copy()
        display["Chance (%)"] = (display["chance"] * 100).round(1)
        display["Final Score (%)"] = (display["final"] * 100).round(1)
        display["Rating"] = display.get("university_rating", 3).fillna(3).astype(int)

        if "latitude" not in display.columns or "longitude" not in display.columns:
            display["latitude"] = np.nan
            display["longitude"] = np.nan

        view_mode = st.session_state.get("view_mode", "list")

        # ----------------- LIST VIEW -----------------
        if view_mode == "list":
            col_left, col_right = st.columns([0.4, 0.6], gap="large")

            with col_left:
                st.markdown("### 🏆 Top Recommended Universities")
                st.markdown("<div class='left-panel'>", unsafe_allow_html=True)

                for i, row in display.reset_index(drop=True).iterrows():
                    selected_cls = " selected" if st.session_state.get("selected_uni") == row["university"] else ""
                    st.markdown(
                        f"""
                        <div class='uni-card{selected_cls}' id='uni-{i}'>
                            <b>{i+1}. {row['university']}</b><br>
                            <span class='info-small'>📍 {row['city']}, {row['country']}</span><br>
                            ⭐ Rating: {int(row.get('university_rating',3))}/5<br>
                            🎯 Final Score: <b>{row['final']*100:.1f}%</b><br>
                            📈 Chance: <b>{row['chance']*100:.1f}%</b><br>
                            💰 Total Cost (avg): <b>${int(row.get('total_cost',0)):,}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.button(f"View Details — {i+1}", key=f"btn_detail_{i}"):
                        st.session_state["selected_uni"] = row["university"]
                        st.session_state["view_mode"] = "detail"
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

                st.download_button(
                    "📥 Download Top 10 CSV",
                    display.to_csv(index=False),
                    "top10.csv",
                    "text/csv"
                )

            with col_right:
                st.markdown("### 🌍 Global View (All Routes)")
                disp_globe = display.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
                if not disp_globe.empty:
                    # bigger globe here
                    fig_globe = build_plotly_globe(disp_globe, mode="list", selected_uni=None, height=450)
                    st.plotly_chart(apply_plotly_theme(fig_globe), use_container_width=True)
                else:
                    st.warning("⚠ No latitude/longitude columns found in file for globe view.")

                st.markdown("### 📊 Visual Summary")

                fig_scores = px.bar(
                    display.sort_values("final"),
                    x="final", y="university",
                    orientation="h",
                    title="Final Score Comparison",
                    color="final",
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(apply_plotly_theme(fig_scores), use_container_width=True)

                fig_aff = go.Figure()
                fig_aff.add_trace(go.Scatter(
                    x=display["total_cost"],
                    y=display["Rating"],
                    mode="markers",
                    marker=dict(
                        size=display["final"] * 46 + 10,
                        color=display["final"],
                        colorscale="Viridis",
                        opacity=0.85,
                        line=dict(width=1, color="#ffffff"),
                        colorbar=dict(
                            title="Final Score",      # Legend title
                            thickness=12,
                            outlinewidth=0
                        ),
                        showscale=True               # <<< IMPORTANT
                    ),
                    text=display["university"],
                    hovertemplate="<b>%{text}</b><br>Cost: $%{x}<br>Rating: %{y}<br>Final Score: %{marker.color:.2f}"
                ))

                fig_aff.update_layout(
                    title="Affordability vs Rating",
                    xaxis_title="Total Cost (USD)",
                    yaxis_title="Rating",
                    height=420
                )

                st.plotly_chart(apply_plotly_theme(fig_aff), use_container_width=True)

        # ----------------- DETAIL VIEW -----------------
        else:
            selected_uni = st.session_state.get("selected_uni")
            disp = display.reset_index(drop=True)

            sel_idx = None
            for i, r in disp.iterrows():
                if r["university"] == selected_uni:
                    sel_idx = i
                    break

            if sel_idx is None:
                st.session_state["view_mode"] = "list"
                st.rerun()

            u = disp.iloc[sel_idx]

            col_left, col_right = st.columns([0.4, 0.6], gap="large")

            with col_left:
                if st.button("⬅ Back to Results"):
                    st.session_state["view_mode"] = "list"
                    st.rerun()

                st.markdown(f"## 🔍 {u['university']}")
                st.write(f"📍 {u['city']}, {u['country']}")
                st.write(f"⭐ Rating: {int(u['university_rating'])}/5")
                st.write(f"📈 Chance: {u['chance']*100:.1f}%")
                st.write(f"🎯 Final Score: {u['final']*100:.1f}%")
                st.write(f"💰 Total Cost: ${u['total_cost']}")

                wiki_name = u['university'].replace(" ", "_")
                wiki_url = f"https://en.wikipedia.org/wiki/{wiki_name}"
                st.markdown(f"[🌐 Click Here to know more [wikipedia]]({wiki_url})")

                cost_parts = {}
                for col_name, label in [
                    ("tuition_usd", "Tuition"),
                    ("rent_usd", "Rent"),
                    ("visa_fee_usd", "Visa"),
                    ("insurance_usd", "Insurance")
                ]:
                    if col_name in u.index and not pd.isna(u[col_name]) and u[col_name] > 0:
                        cost_parts[label] = u[col_name]
                if cost_parts:
                    cost_df = pd.DataFrame({"component": list(cost_parts.keys()), "amount": list(cost_parts.values())})
                    fig_pie = px.pie(cost_df, names="component", values="amount", title="Cost Breakdown")
                    st.plotly_chart(apply_plotly_theme(fig_pie), use_container_width=True)

                score_df = pd.DataFrame({
                    "metric": ["Chance", "Priority", "Overall", "Final"],
                    "value": [u["chance"], u["priority"], u["overall_norm"], u["final"]]
                })
                fig_line = px.line(score_df, x="metric", y="value", markers=True, title="Score Profile")
                fig_line.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(apply_plotly_theme(fig_line), use_container_width=True)

            with col_right:
                st.markdown("### 🌍 Flight Path to Your Choice")
                disp_globe = display.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

                center = None
                if not pd.isna(u.get("longitude")) and not pd.isna(u.get("latitude")):
                    center_lon = (78.9629 + float(u["longitude"])) / 2
                    center_lat = (20.5937 + float(u["latitude"])) / 2
                    center = {"lon": center_lon, "lat": center_lat}

                if not disp_globe.empty:
                    # bigger globe in detail view
                    fig_globe = build_plotly_globe(
                        disp_globe, mode="detail", selected_uni=selected_uni, height=520, center=center
                    )
                    st.plotly_chart(apply_plotly_theme(fig_globe), use_container_width=True)
                else:
                    st.warning("⚠ No latitude/longitude available for globe.")

                st.markdown("### 📌 Future Cost Prediction")
                st.write("Estimate how much this program might cost you in INR in a future year.")
                st.warning(
                    "⚠️ **Score Validity Reminder**\n\n"
                    "- **TOEFL** scores are valid for **2 years** from the test date.\n"
                    "- **GRE** scores are valid for **5 years** from the test date.",
                    icon="⚠️"
                )
                year = st.number_input(
                    "Enter Future Year (e.g., 2027)",
                    min_value=2025, max_value=2050,
                    value=2027,
                    key="fx_year_detail"
                )
                amount_usd = u['total_cost']

                if st.button("Predict Future INR Value", key="predict_fx_run_detail"):
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
