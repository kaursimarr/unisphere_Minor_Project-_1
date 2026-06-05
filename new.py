import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import time
import math
from streamlit_plotly_events import plotly_events

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Smart Study Abroad Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Updated CSS
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
header, [data-testid="stHeader"], .block-container, main, body, .main {
    padding-top: 0 !important;
    margin-top: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
}
div[data-testid="column"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.stApp {
    background-color: #f7f9fc;
    color: #111827;
}
.left-panel {
    max-height: 88vh;
    overflow-y: auto;
    padding-right: 8px;
}
.left-panel::-webkit-scrollbar { width: 8px; }
.left-panel::-webkit-scrollbar-thumb { background: #c7cdd6; border-radius: 10px; }
.right-panel {
    position: sticky !important;
    top: 6px !important;
    align-self: flex-start !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.sticky-globe { position: sticky !important; top: 6px !important; display:flex; justify-content:center; padding:0; margin:0; }
.sticky-globe > div { width: 360px !important; height: 360px !important; margin:0 auto !important; }
.uni-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
    border: 1px solid #e6e9ef;
    transition: box-shadow 0.2s, transform 0.06s;
}
.uni-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(16,24,40,0.06); }
.uni-card.selected {
    border: 2px solid #f59e0b;
    background: linear-gradient(90deg, rgba(255,250,235,1), rgba(255,255,255,1));
}
.stButton>button { border-radius: 6px; padding: 8px 16px; background: #2563eb; color: white; border: none; font-weight: 500; }
.stButton>button:hover { background: #1e4dc4; }
.info-small { font-size:13px; color:#6b7280; }
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
@st.cache_data
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
# Globe helper functions
# -------------------------
CITY_COORDS = {
    ("cambridge", "united states"): (42.3736, -71.1097),
    ("cambridge", "united kingdom"): (52.2053, 0.1218),
    ("london", "united kingdom"): (51.5072, -0.1276),
    ("toronto", "canada"): (43.651070, -79.347015),
    ("munich", "germany"): (48.1351, 11.5820),
    ("berlin", "germany"): (52.5200, 13.4050),
    ("melbourne", "australia"): (-37.8136, 144.9631),
    ("sydney", "australia"): (-33.8688, 151.2093),
    ("helsinki", "finland"): (60.1699, 24.9384),
    ("zurich", "switzerland"): (47.3769, 8.5417),
    ("tokyo", "japan"): (35.6762, 139.6503),
    ("seoul", "south korea"): (37.5665, 126.9780),
    ("paris", "france"): (48.8566, 2.3522),
    ("new york", "united states"): (40.7128, -74.0060),
    ("los angeles", "united states"): (34.0522, -118.2437),
}

def get_coordinates(city, country):
    if pd.isna(city) or pd.isna(country) or str(city).strip() == "":
        return np.nan, np.nan
    key = (str(city).lower().strip(), str(country).lower().strip())
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    s = (str(city).lower().strip() + "||" + str(country).lower().strip())
    h = abs(hash(s))
    lat = (h % 170) - 85
    lon = ((h // 170) % 360) - 180
    return float(lat), float(lon)

def great_circle_arc(lat1, lon1, lat2, lon2, num_points=100):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2 - lat1)/2)**2 +
        math.cos(lat1)*math.cos(lat2)*math.sin((lon2 - lon1)/2)**2
    ))
    if d == 0:
        return [np.degrees(lat1)], [np.degrees(lon1)]
    lats, lons = [], []
    for f in np.linspace(0, 1, num_points):
        A = math.sin((1-f)*d) / math.sin(d)
        B = math.sin(f*d) / math.sin(d)
        x = A*math.cos(lat1)*math.cos(lon1) + B*math.cos(lat2)*math.cos(lon2)
        y = A*math.cos(lat1)*math.sin(lon1) + B*math.cos(lat2)*math.sin(lon2)
        z = A*math.sin(lat1) + B*math.sin(lat2)
        lat = math.atan2(z, math.sqrt(x*x + y*y))
        lon = math.atan2(y, x)
        lats.append(np.degrees(lat))
        lons.append(np.degrees(lon))
    return lats, lons

# -------------------------
# Interactive Globe
# -------------------------
def build_plotly_globe_interactive(
        df,
        india_lat=20.5937,
        india_lon=78.9629,
        selected_idx=None,
        rotation_lon=0.0,
        height=360
    ):
    fig = go.Figure()
    # arcs: lines from India to each uni
    for i, r in df.reset_index(drop=True).iterrows():
        lat = r.get("latitude")
        lon = r.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        lats, lons = great_circle_arc(india_lat, india_lon, float(lat), float(lon), num_points=60)
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(width=1.4, color='rgba(180,230,200,0.35)'),
            hoverinfo='skip',
            showlegend=False,
            name=f"arc_{i}"
        ))
    # endpoints: visible bright markers
    endpoint_lons = []
    endpoint_lats = []
    endpoint_texts = []
    endpoint_custom = []
    for i, r in df.reset_index(drop=True).iterrows():
        lat = r.get("latitude")
        lon = r.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            endpoint_lats.append(np.nan); endpoint_lons.append(np.nan)
            endpoint_texts.append("")
            endpoint_custom.append(["", "", "", ""])
            continue
        endpoint_lats.append(lat)
        endpoint_lons.append(lon)
        uni = r.get("university", "")
        city = r.get("city", "")
        ctr = r.get("country", "")
        rating = r.get("university_rating", "")
        chance = r.get("chance", 0.0)
        endpoint_texts.append(f"{uni} — {city}, {ctr}")
        endpoint_custom.append([uni, ctr, rating, chance])
    fig.add_trace(go.Scattergeo(
        lon=endpoint_lons,
        lat=endpoint_lats,
        mode='markers',
        marker=dict(size=10, color='#ffd60a', line=dict(width=1, color='#ffffff')),
        hoverinfo='text',
        text=[f"{c[0]}<br>{c[1]}<br>Rating: {c[2]}<br>Chance: {round(float(c[3])*100,1)}%" for c in endpoint_custom],
        customdata=np.array(endpoint_custom),
        name='Endpoints',
        showlegend=False
    ))
    # main university markers
    uni_lons = []
    uni_lats = []
    uni_texts = []
    uni_custom = []
    for i, r in df.reset_index(drop=True).iterrows():
        lat = r.get("latitude")
        lon = r.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            uni_lats.append(np.nan); uni_lons.append(np.nan)
            uni_texts.append("")
            uni_custom.append(["", "", "", ""])
            continue
        uni_lats.append(lat)
        uni_lons.append(lon)
        uni = r.get("university", "")
        city = r.get("city", "")
        ctr = r.get("country", "")
        rating = r.get("university_rating", "")
        chance = r.get("chance", 0.0)
        uni_texts.append(f"{uni} — {city}, {ctr}")
        uni_custom.append([uni, ctr, rating, chance])

    marker_sizes = [14 if (selected_idx is not None and i == selected_idx) else 8 for i in range(len(df))]
    marker_colors = ['#ef4444' if not (selected_idx is not None and i == selected_idx) else '#00e5a9' for i in range(len(df))]
    fig.add_trace(go.Scattergeo(
        lon=uni_lons,
        lat=uni_lats,
        text=[f"{u}<br>Rating: {r}<br>Chance: {round(float(c)*100,1)}%" for u,r,c in zip([u[0] for u in uni_custom],[u[2] for u in uni_custom],[u[3] for u in uni_custom])],
        mode='markers',
        marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=1, color='white')),
        hoverinfo='text',
        customdata=np.array(uni_custom),
        name='Universities',
        showlegend=False
    ))
    # India marker (origin)
    fig.add_trace(go.Scattergeo(
        lon=[india_lon],
        lat=[india_lat],
        text=["You (India)"],
        mode='markers+text',
        marker=dict(size=9, color='royalblue'),
        textposition="bottom center",
        hoverinfo='text',
        name='India',
        showlegend=False
    ))
    fig.update_geos(
        projection_type="orthographic",
        showcountries=True, countrycolor="rgba(255,255,255,0.2)",
        showland=True, landcolor="rgb(37,51,63)",
        showocean=True, oceancolor="rgb(10,40,80)",
        lataxis_showgrid=False, lonaxis_showgrid=False,
        projection_rotation=dict(lon=float(rotation_lon), lat=0, roll=0)
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        showlegend=False
    )
    return fig

def animate_rotation_to(target_lon, display_df, steps=24, pause=0.02, globe_placeholder=None):
    def norm(a):
        a = ((a + 180) % 360) - 180
        return a
    current = float(st.session_state.get("rotation_lon", 0.0))
    current = norm(current)
    target = norm(target_lon)
    diff = target - current
    if diff > 180:
        diff -= 360
    if diff < -180:
        diff += 360
    for f in np.linspace(0, 1, steps):
        ang = current + diff * f
        fig = build_plotly_globe_interactive(display_df, selected_idx=st.session_state.get("selected_index", None), rotation_lon=ang, height=360)
        if globe_placeholder is not None:
            globe_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        time.sleep(pause)
    st.session_state["rotation_lon"] = float(target)

# -------------------------
# UI: form
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
    st.session_state["rotation_lon"] = st.session_state.get("rotation_lon", 0.0)
    st.session_state["selected_index"] = None

# -------------------------
# Display Results
# -------------------------
if "top" in st.session_state:
    top = st.session_state["top"]
    display = top.copy()
    display["Chance (%)"] = (display["chance"] * 100).round(1)
    display["Final Score (%)"] = (display["final"] * 100).round(1)
    display["Rating"] = display.get("university_rating", 3).fillna(3).astype(int)
    display["latitude"] = np.nan
    display["longitude"] = np.nan
    for idx, row in display.iterrows():
        lat, lon = get_coordinates(row.get("city"), row.get("country"))
        try:
            display.at[idx, "latitude"] = float(lat)
        except:
            display.at[idx, "latitude"] = np.nan
        try:
            display.at[idx, "longitude"] = float(lon)
        except:
            display.at[idx, "longitude"] = np.nan
    col_left, col_right = st.columns([2, 1], gap="large")
    # LEFT: scrollable list
with col_left:
    st.markdown("### 🏆 Top Recommended Universities")
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)

    for i, row in display.reset_index(drop=True).iterrows():
        selected_cls = " selected" if st.session_state.get("selected_index", None) == i else ""
        st.markdown(
            f"""
            <div class='uni-card{selected_cls}' id='uni-{i}'>
                <b>{i+1}. {row['university']}</b><br>
                <span class='info-small'>📍 {row['city']}, {row['country']}</span><br>
                ⭐ Rating: {int(row.get('university_rating',3))}/5<br>
                🎯 Final Score: <b>{row['final']*100:.1f}%</b><br>
                📈 Chance: <b>{row['chance']*100:.1f}%</b><br>
                💰 Total Cost(on an average): <b>${int(row.get('total_cost',0)):,}</b>
            </div>
            """, unsafe_allow_html=True
        )

        # View Details button
        if st.button(f"View Details — {i+1}", key=f"btn{i}"):
            st.session_state["selected_uni"] = row["university"]
            st.session_state["selected_index"] = int(i)
            lon_i = row.get("longitude", np.nan)
            if not pd.isna(lon_i):
                globe_placeholder = st.session_state.get('globe_placeholder_global', None)
                animate_rotation_to(float(lon_i), display, steps=28, pause=0.02, globe_placeholder=globe_placeholder)

        # Add Wikipedia link
        wiki_name = row['university'].replace(' ', '_')
        wiki_url = f"https://en.wikipedia.org/wiki/{wiki_name}"
        st.markdown(f"[📚 Wikipedia]({wiki_url})", unsafe_allow_html=True)

        # Add official website link if available
        if 'website' in row and pd.notnull(row['website']):
            website_url = row['website']
            st.markdown(f"[🌐 Official Site]({website_url})", unsafe_allow_html=True)

        # View on Globe button
        if st.button(f"View on Globe — {i+1}", key=f"view_btn_globe_{i}"):
            st.session_state["selected_index"] = int(i)
            lon_i = row.get("longitude", np.nan)
            if not pd.isna(lon_i):
                globe_placeholder = st.session_state.get('globe_placeholder_global', None)
                animate_rotation_to(float(lon_i), display, steps=28, pause=0.02, globe_placeholder=globe_placeholder)
                final_fig = build_plotly_globe_interactive(
                    display,
                    selected_idx=st.session_state.get("selected_index"),
                    rotation_lon=st.session_state.get("rotation_lon", 0.0),
                    height=360
                )
                if globe_placeholder:
                    globe_placeholder.plotly_chart(final_fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Show details card for selected university
    if st.session_state.get("selected_uni"):
        u = display.iloc[st.session_state["selected_index"]]
        st.markdown(f"### 🔍 {u['university']}")
        st.write(f"📍 {u['city']}, {u['country']}")
        st.write(f"⭐ Rating: {int(u['university_rating'])}/5")
        st.write(f"📈 Chance: {u['chance']*100:.1f}%")
        st.write(f"🎯 Final Score: {u['final']*100:.1f}%")
        st.write(f"💰 Total Cost: ${u['total_cost']}")
        search_url = google_image_preview(u['university'], u['city'], u['country'])
        st.markdown(
            f'<a href="{search_url}" target="_blank"><img src="https://source.unsplash.com/featured/?{u["university"].replace(" ","+")},campus,university" style="width:100%;border-radius:12px;"></a>',
            unsafe_allow_html=True
        )
        st.caption("📷 Click image to view more campus photos")

        cost_parts = {}
        for col, label in [("tuition_usd", "Tuition"), ("rent_usd", "Rent"), ("visa_fee_usd", "Visa"), ("insurance_usd", "Insurance")]:
            if col in u.index and not pd.isna(u[col]) and u[col] > 0:
                cost_parts[label] = u[col]
        if cost_parts:
            cost_df = pd.DataFrame({"component": list(cost_parts.keys()), "amount": list(cost_parts.values())})
            fig_pie = px.pie(cost_df, names="component", values="amount", title="Cost Breakdown")
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

        score_df = pd.DataFrame({
            "metric": ["Chance", "Priority", "Overall", "Final"],
            "value": [u["chance"], u["priority"], u["overall_norm"], u["final"]]
        })
        fig_line = px.line(score_df, x="metric", y="value", markers=True, title="Score Profile")
        st.plotly_chart(fig_line, use_container_width=True)

    st.download_button("📥 Download Top 10 CSV", display.to_csv(index=False), "top10.csv", "text/csv")

    # RIGHT: sticky globe
    with col_right:
        with st.container():
            st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='sticky-globe'>", unsafe_allow_html=True)
            st.markdown("🌍 ### Interactive Globe")
            fig_globe = build_plotly_globe_interactive(
                display,
                selected_idx=st.session_state.get("selected_index"),
                rotation_lon=st.session_state.get("rotation_lon", 0.0),
                height=360
            )
            globe_placeholder = st.empty()
            st.session_state['globe_placeholder_global'] = globe_placeholder
            with globe_placeholder:
                clicked = plotly_events(
                    fig_globe,
                    click_event=True,
                    hover_event=True,
                    override_height=380,
                    key="main_globe_event"
                )
            if clicked:
                ev = clicked[0]
                curve = ev.get("curveNumber")
                pt = ev.get("pointNumber")
                trace_name = None
                try:
                    trace = fig_globe.data[curve]
                    trace_name = getattr(trace, "name", None)
                except Exception:
                    trace_name = None
                if trace_name in ("Endpoints", "Universities") and pt is not None:
                    st.session_state["selected_index"] = int(pt)
                    sel_row = display.reset_index(drop=True).iloc[st.session_state["selected_index"]]
                    target_lon = float(sel_row["longitude"])
                    animate_rotation_to(target_lon, display, steps=28, pause=0.02, globe_placeholder=globe_placeholder)
                    updated_fig = build_plotly_globe_interactive(
                        display,
                        selected_idx=st.session_state.get("selected_index"),
                        rotation_lon=st.session_state.get("rotation_lon", 0.0),
                        height=360
                    )
                    globe_placeholder.plotly_chart(updated_fig, use_container_width=True, key="main_globe_chart")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    # ---------------- BOTTOM CHARTS ----------------
    st.markdown("---")
    st.subheader("📊 Additional Insights")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=display["Chance (%)"],
        y=display["university"],
        orientation='h',
        marker=dict(color='#4f46e5')
    ))
    fig1.update_layout(title="Admission Chance (%) — Top 10", height=420)
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=display["total_cost"],
        y=display["Rating"],
        mode="markers",
        marker=dict(size=display["final"] * 36 + 8, color=display["final"], colorscale="Blues"),
        text=display["university"]
    ))
    fig2.update_layout(title="Affordability vs Rating", height=420)
    st.plotly_chart(fig2, use_container_width=True)
    st.info("Tip: Scroll the left list while the globe stays visible on the right.")
