"""
=============================================================
  PatrolIQ — Smart Safety Analytics Platform
  Streamlit Dashboard
  GUVI | HCL Capstone Project
=============================================================
  Run:
    pip install streamlit pandas numpy matplotlib seaborn
                scikit-learn plotly folium streamlit-folium
    streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="PatrolIQ — Smart Safety Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS + ANIMATIONS ───────────────────────────────────
st.markdown("""
<style>
    /* ── HIDE SIDEBAR NAV ── */
    [data-testid="stSidebarNav"] { display: none; }

    /* ── GLOBAL BACKGROUND ── */
    .stApp                { background-color: #0b1622; }
    .stApp > header       { background-color: #0b1622; }
    .block-container      { padding-top: 0.5rem !important; }

    /* ── SIDEBAR — DARK NAVY BLUE ── */
    section[data-testid="stSidebar"] {
        background-color: #06101c !important;
        border-right: 1px solid #1a2e45;
    }
    section[data-testid="stSidebar"] * {
        color: #ccd6f6 !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #64ffda !important;
        font-weight: 700;
    }

    /* ── MAIN TEXT ── */
    .stApp p, .stApp li, .stApp span,
    .stApp label, .stApp div { color: #ccd6f6; }
    h1, h2, h3, h4, h5, h6   { color: #f5f0e8 !important; }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #112240; border-radius: 8px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"]      { color: #8892b0 !important; font-weight: 600; }
    .stTabs [aria-selected="true"]    {
        background-color: #1e3a5f !important;
        color: #64ffda !important; border-radius: 6px;
    }

    /* ── INPUTS ── */
    .stSlider label, .stSelectbox label,
    .stMultiSelect label, .stRadio label { color: #ccd6f6 !important; font-weight: 600; }
    .stDataFrame { background-color: #112240; color: #ccd6f6; }
    hr { border-color: #1e3a5f !important; }

    /* ── STREAMLIT METRICS ── */
    [data-testid="stMetricValue"] { color: #64ffda !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #8892b0 !important; }
    [data-testid="stMetricDelta"] { color: #64ffda !important; }

    /* ══════════════════════════════════════
       POLICE CAR ANIMATION — top of page
    ══════════════════════════════════════ */
    .patrol-road {
        width: 100%;
        background: linear-gradient(180deg, #0b1622 60%, #0b1622 60%,
                                    #1a2e45 60%, #1a2e45 68%,
                                    #0b1622 68%);
        height: 52px;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.3rem;
        border-radius: 6px;
    }
    /* dashed road line */
    .patrol-road::before {
        content: "";
        position: absolute;
        top: 63%;
        left: 0; right: 0;
        height: 3px;
        background: repeating-linear-gradient(
            90deg,
            #f5f0e8 0px, #f5f0e8 30px,
            transparent 30px, transparent 60px
        );
        animation: road-scroll 1.2s linear infinite;
    }
    @keyframes road-scroll {
        from { background-position: 0 0; }
        to   { background-position: 60px 0; }
    }
    /* the car */
    .patrol-car {
        position: absolute;
        top: 4px;
        font-size: 2.4rem;
        animation: drive 5s linear infinite;
        filter: drop-shadow(0 0 8px rgba(100,255,218,0.6));
    }
    @keyframes drive {
        0%   { left: -80px;  }
        100% { left: 105%;   }
    }
    /* siren flash */
    .patrol-car::after {
        content: "🚨";
        font-size: 1rem;
        position: absolute;
        top: -6px; left: 50%;
        transform: translateX(-50%);
        animation: siren 0.4s alternate infinite;
    }
    @keyframes siren {
        from { opacity: 1; }
        to   { opacity: 0.2; }
    }

    /* ── CUSTOM CARD COMPONENTS ── */
    .main-title {
        font-size: 3rem; font-weight: 900;
        color: #f5f0e8;
        text-align: center;
        padding: 0.3rem 0 0.2rem 0;
        text-shadow: 0 0 20px rgba(100,255,218,0.3);
        letter-spacing: 1px;
    }
    .sub-title {
        font-size: 1.05rem; color: #b0c4d8;
        text-align: center; margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #112240, #1e3a5f);
        border: 1px solid #1e3a5f;
        border-top: 3px solid #64ffda;
        border-radius: 12px; padding: 1.2rem 1rem;
        color: white; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #64ffda; }
    .metric-label { font-size: 0.82rem; color: #8892b0; margin-top: 0.3rem; font-weight: 500; }

    .section-header {
        font-size: 1.25rem; font-weight: 700;
        color: #64ffda;
        border-left: 4px solid #64ffda;
        padding-left: 0.7rem;
        margin: 1.8rem 0 0.8rem 0;
        letter-spacing: 0.3px;
    }
    .insight-box {
        background: #112240;
        border: 1px solid #1e3a5f;
        border-left: 4px solid #64ffda;
        border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.5rem 0; font-size: 0.9rem; color: #ccd6f6;
    }

    /* ── OVERALL SUMMARY CARDS — light yellow text ── */
    .summary-card {
        background: #112240;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        margin: 0.3rem 0;
        font-size: 0.86rem;
    }
    .summary-card b  { color: #ffe66d; font-size: 0.9rem; }
    .summary-card span { color: #f0e6b2; }

    .stSelectbox label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── CHICAGO POLICE LABEL — shown randomly across pages ────────
import random

CPD_LABELS = [
    "⭐ Chicago Police Department — Serving Since 1837",
    "🚔 CPD · We Serve and Protect",
    "🛡️ Chicago Police Department · PatrolIQ Analytics",
    "⭐ CPD Patrol Intelligence · Chicago, Illinois",
    "🚓 Chicago PD · Crime Analytics Division",
    "🌟 Chicago Police · Protecting Every Neighborhood",
    "⭐ CPD · Data-Driven Policing Initiative",
]

def cpd_label():
    label = random.choice(CPD_LABELS)
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; justify-content:center;
        gap:0.6rem;
        background: linear-gradient(90deg, #0d2247, #1a3a6e, #0d2247);
        border-top: 1.5px solid #ffe566;
        border-bottom: 1.5px solid #ffe566;
        padding: 0.35rem 1rem;
        margin: 0.4rem 0 0.6rem 0;
        border-radius: 4px;
    ">
        <span style="font-size:1rem;">⭐</span>
        <span style="font-size:0.82rem; font-weight:700; color:#ffe566;
                     letter-spacing:1.2px; text-transform:uppercase;">
            {label}
        </span>
        <span style="font-size:1rem;">⭐</span>
    </div>
    """, unsafe_allow_html=True)

# ── ANIMATED POLICE CAR — called just before each page title ──
import streamlit.components.v1 as components

def patrol_car_animation():
    components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    background: #0b1622;
    overflow: hidden;
    height: 90px;
  }

  /* road strip at BOTTOM so car sits just above title */
  .road {
    width: 100%;
    height: 90px;
    background: #0b1622;
    position: relative;
    overflow: visible;   /* allow car emoji to show fully */
  }

  /* dashed road line near bottom */
  .road-line {
    position: absolute;
    bottom: 10px; left: 0;
    width: 200%; height: 3px;
    background: repeating-linear-gradient(
      90deg,
      #556677 0px, #556677 32px,
      transparent 32px, transparent 64px
    );
    animation: scroll-road 1.6s linear infinite;
  }
  @keyframes scroll-road {
    from { transform: translateX(0); }
    to   { transform: translateX(-64px); }
  }

  /* car sits ON the road line */
  .car {
    position: absolute;
    bottom: 12px;          /* just above the road line */
    font-size: 3.2rem;     /* larger so it's clearly visible */
    line-height: 1;
    white-space: nowrap;
    animation: drive 9s linear infinite;
    filter: drop-shadow(0 0 10px rgba(100,255,218,0.8));
  }
  @keyframes drive {
    0%   { left: -120px; }
    100% { left: 108%; }
  }

  /* flashing siren above car */
  .siren {
    font-size: 1.1rem;
    position: absolute;
    top: -10px; left: 50%;
    transform: translateX(-50%);
    animation: blink 0.5s alternate infinite;
  }
  @keyframes blink {
    from { opacity: 1; }
    to   { opacity: 0.05; }
  }
</style>
</head>
<body>
  <div class="road">
    <div class="road-line"></div>
    <div class="car">
      <span class="siren">🚨</span>🚓
    </div>
  </div>
</body>
</html>
""", height=92, scrolling=False)


# ════════════════════════════════════════════════════════════
#  DATA LOADING & CACHING
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading Chicago Crime Dataset...")
def load_data(sample_size=500000):
    try:
        # Load ALL rows — patrol.csv is already the right size (500K)
        # sample_size is only used for clustering pages, not for EDA/Home
        df = pd.read_csv("patrol.csv", low_memory=False)
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
        # No sampling here — keep all rows so all crime types are visible
    except FileNotFoundError:
        st.warning("patrol.csv not found — place it in the same folder as app.py")
        st.stop()

    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Date"]     = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"]     = df["Date"].dt.hour
    df["Month"]    = df["Date"].dt.month
    df["Year"]     = df["Date"].dt.year
    df["Day_Num"]  = df["Date"].dt.dayofweek
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Is_Weekend"]  = df["Day_Num"] >= 5
    df["Is_Weekend_Int"] = df["Is_Weekend"].astype(int)

    def get_season(m):
        if m in [12,1,2]:  return "Winter"
        elif m in [3,4,5]: return "Spring"
        elif m in [6,7,8]: return "Summer"
        else:               return "Fall"

    def get_time_of_day(h):
        if   0 <= h < 6:   return "Late Night"
        elif 6 <= h < 12:  return "Morning"
        elif 12 <= h < 18: return "Afternoon"
        else:               return "Evening"

    df["Season"]      = df["Month"].apply(get_season)
    df["Time_of_Day"] = df["Hour"].apply(get_time_of_day)

    severity_map = {
        "HOMICIDE":10,"CRIM SEXUAL ASSAULT":9,"KIDNAPPING":9,
        "OFFENSE INVOLVING CHILDREN":8,"SEX OFFENSE":8,"ROBBERY":8,
        "WEAPONS VIOLATION":7,"ASSAULT":7,"ARSON":7,"BATTERY":6,
        "BURGLARY":5,"STALKING":5,"INTIMIDATION":5,"MOTOR VEHICLE THEFT":5,
        "THEFT":4,"CRIMINAL DAMAGE":4,"DECEPTIVE PRACTICE":3,
        "NARCOTICS":3,"PUBLIC PEACE VIOLATION":3,
        "INTERFERENCE WITH PUBLIC OFFICER":3,"CRIMINAL TRESPASS":2,
        "GAMBLING":2,"LIQUOR LAW VIOLATION":2,"OBSCENITY":2,
    }
    df["Crime_Severity_Score"] = df["Primary Type"].map(severity_map).fillna(1)

    le = LabelEncoder()
    df["Crime_Type_Encoded"] = le.fit_transform(df["Primary Type"].astype(str))

    for col in ["Location Description","Block","Description"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
    for col in ["Ward","Community Area","District","Beat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

    df["Arrest"]   = df["Arrest"].astype(bool)   if "Arrest"   in df.columns else False
    df["Domestic"] = df["Domestic"].astype(bool) if "Domestic" in df.columns else False
    return df


def generate_demo_data(n=100000):
    """Generate realistic synthetic Chicago crime data for demo mode."""
    np.random.seed(42)
    crime_types = ["THEFT","BATTERY","CRIMINAL DAMAGE","ASSAULT","OTHER OFFENSE",
                   "DECEPTIVE PRACTICE","BURGLARY","MOTOR VEHICLE THEFT",
                   "ROBBERY","NARCOTICS","WEAPONS VIOLATION","HOMICIDE"]
    weights     = np.array([0.25,0.18,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01])
    weights     = weights / weights.sum()   # normalize to exactly 1.0

    # Chicago lat/lon clusters
    lats = np.concatenate([
        np.random.normal(41.88, 0.04, int(n*0.35)),
        np.random.normal(41.76, 0.03, int(n*0.25)),
        np.random.normal(41.95, 0.03, int(n*0.20)),
        np.random.normal(41.82, 0.05, int(n*0.20)),
    ])[:n]
    lons = np.concatenate([
        np.random.normal(-87.63, 0.04, int(n*0.35)),
        np.random.normal(-87.68, 0.03, int(n*0.25)),
        np.random.normal(-87.72, 0.03, int(n*0.20)),
        np.random.normal(-87.60, 0.04, int(n*0.20)),
    ])[:n]

    dates = pd.date_range("2018-01-01", "2024-12-31", periods=n)
    return pd.DataFrame({
        "Date"                : np.random.choice(dates, n),
        "Primary Type"        : np.random.choice(crime_types, n, p=weights),
        "Latitude"            : lats,
        "Longitude"           : lons,
        "District"            : np.random.randint(1, 25, n).astype(float),
        "Arrest"              : np.random.choice([True, False], n, p=[0.28, 0.72]),
        "Domestic"            : np.random.choice([True, False], n, p=[0.18, 0.82]),
        "Year"                : np.random.randint(2018, 2025, n),
        "Location Description": np.random.choice(["STREET","RESIDENCE","APARTMENT",
                                                   "PARKING LOT","STORE"], n),
        "Ward"                : np.random.randint(1, 50, n).astype(float),
        "Community Area"      : np.random.randint(1, 77, n).astype(float),
        "Beat"                : np.random.randint(100, 2599, n).astype(float),
        "Block"               : ["SAMPLE BLOCK"] * n,
        "Description"         : ["SAMPLE"] * n,
    })


@st.cache_data(show_spinner="Running K-Means clustering...")
def run_kmeans(lats, lons, k):
    coords  = np.column_stack([lats, lons])
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(coords)
    km      = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels  = km.fit_predict(scaled)
    sil     = silhouette_score(scaled, labels, sample_size=5000, random_state=42)
    db      = davies_bouldin_score(scaled, labels)
    centers = scaler.inverse_transform(km.cluster_centers_)
    return labels, sil, db, centers, km.inertia_


@st.cache_data(show_spinner="Running DBSCAN...")
def run_dbscan(lats, lons, eps, min_samples):
    coords = np.column_stack([lats, lons])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(coords)
    db     = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(scaled)
    n_cls  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise= (labels == -1).sum()
    sil    = 0.0
    if n_cls > 1:
        mask = labels != -1
        sil  = silhouette_score(scaled[mask], labels[mask], sample_size=5000, random_state=42)
    return labels, sil, n_cls, n_noise


@st.cache_data(show_spinner="Running Hierarchical clustering...")
def run_hierarchical(lats, lons, k):
    coords = np.column_stack([lats, lons])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(coords)
    hier   = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hier.fit_predict(scaled)
    sil    = silhouette_score(scaled, labels, sample_size=5000, random_state=42)
    db     = davies_bouldin_score(scaled, labels)
    return labels, sil, db


# ════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════
with st.sidebar:

    # ── TOP: CHICAGO POLICE BADGE ─────────────────────────────
    st.html("""
    <div style="text-align:center;padding:0.7rem 0.4rem 0.3rem 0;">
        <div style="display:inline-block;background:linear-gradient(160deg,#1a3a6e,#0d2247);border:3px solid #ffe566;border-radius:48% 48% 42% 42%;width:110px;height:118px;position:relative;box-shadow:0 0 18px rgba(255,229,102,0.35);margin-bottom:0.3rem;">
            <div style="position:absolute;top:8px;left:8px;right:8px;bottom:8px;border:1.5px solid rgba(255,229,102,0.4);border-radius:46% 46% 40% 40%;"></div>
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-52%);font-size:2.6rem;line-height:1;">⭐</div>
            <div style="position:absolute;bottom:14px;left:0;right:0;font-size:0.6rem;font-weight:900;color:#ffe566;letter-spacing:2px;text-align:center;">C &middot; P &middot; D</div>
        </div>
        <div style="font-size:0.85rem;font-weight:900;color:#ffe566;letter-spacing:1.5px;margin-top:0.2rem;">CHICAGO POLICE</div>
        <div style="font-size:0.7rem;color:#b0c4d8;letter-spacing:1px;margin-top:0.1rem;">DEPARTMENT</div>
        <div style="font-size:0.62rem;color:#8892b0;font-style:italic;margin-top:0.1rem;">&ldquo;We Serve and Protect&rdquo;</div>
    </div>
    <hr style="border-color:#1a2e45;margin:0.5rem 0;">
    """)

    # ── CAR + TITLE ───────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:0.3rem 0;">
        <div style="font-size:7.6rem; line-height:1;
                    filter:drop-shadow(0 0 14px rgba(255,230,100,0.5));">🚓</div>
        <div style="font-size:1.15rem; font-weight:900; color:#ffe566;
                    margin-top:0.5rem; letter-spacing:0.5px; line-height:1.4;">
            Chicago Crime Intelligence
        </div>
        <div style="font-size:0.95rem; font-weight:600; color:#ffe566;
                    margin-top:0.2rem; line-height:1.4; opacity:0.85;">
            Safety Analytics Platform
        </div>
    </div>
    <hr style="border-color:#1a2e45; margin:0.6rem 0 0.4rem 0;">
    """, unsafe_allow_html=True)

    # ── NAVIGATION ────────────────────────────────────────────
    page = st.radio("Navigation", [
        "🏠  Home",
        "📊  EDA",
        "🗺️  Geographic Clustering",
        "⏰  Temporal Clustering",
        "📈  Model Comparison",
        "🔍  Crime Analyzer",
    ], label_visibility="hidden")

    # ── BOTTOM BADGE ─────────────────────────────────────────
    st.html("""
    <hr style="border-color:#1a2e45;margin:0.6rem 0 0.5rem 0;">
    <div style="text-align:center;padding:0.3rem 0 0.6rem 0;">
        <div style="display:inline-block;background:linear-gradient(160deg,#1a3a6e,#0d2247);border:2px solid #ffe566;border-radius:8px;padding:0.5rem 1.1rem;box-shadow:0 0 10px rgba(255,229,102,0.2);">
            <div style="font-size:0.7rem;font-weight:900;color:#ffe566;letter-spacing:2px;">&#11088; CHICAGO &#11088;</div>
            <div style="font-size:0.62rem;color:#ccd6f6;letter-spacing:1.5px;margin-top:0.1rem;">POLICE DEPARTMENT</div>
            <div style="font-size:0.58rem;color:#8892b0;margin-top:0.2rem;font-style:italic;">Est. 1837</div>
        </div>
        <div style="font-size:0.65rem;color:#556677;margin-top:0.5rem;">GUVI | HCL Capstone Project<br>Unsupervised ML &middot; Public Safety</div>
    </div>
    """)

    # used internally for clustering sample size
    sample_size = 50000

df = load_data()   # always loads full patrol.csv


# ════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ════════════════════════════════════════════════════════════
if page == "🏠  Home":
    patrol_car_animation()
    st.markdown('<div class="main-title">PatrolIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Smart Safety Analytics Platform &nbsp;|&nbsp; Chicago Crime Intelligence Dashboard</div>',
                unsafe_allow_html=True)
    cpd_label()
    st.markdown("---")

    # ── KPI CARDS (no Total Records) ─────────────────────────
    crime_dist_all = df["Primary Type"].value_counts()
    arrest_rate    = df["Arrest"].mean() * 100
    dom_rate       = df["Domestic"].mean() * 100
    avg_sev        = df["Crime_Severity_Score"].mean()
    top_crime      = crime_dist_all.index[0]
    top_crime_pct  = crime_dist_all.iloc[0] / len(df) * 100
    high_sev_pct   = (df["Crime_Severity_Score"] >= 7).mean() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['Primary Type'].nunique()}</div>
            <div class="metric-label">Unique Crime Types</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{arrest_rate:.1f}%</div>
            <div class="metric-label">Arrest Rate</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{dom_rate:.1f}%</div>
            <div class="metric-label">Domestic Crime Rate</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_sev:.2f}<span style="font-size:1rem">/10</span></div>
            <div class="metric-label">Avg Severity Score</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{high_sev_pct:.1f}%</div>
            <div class="metric-label">High Severity (≥7)</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── OVERALL CRIME SUMMARY ────────────────────────────────
    st.markdown('<div class="section-header">Overall Crime Summary</div>', unsafe_allow_html=True)

    summary_cols = st.columns(3)

    with summary_cols[0]:
        st.markdown("<b style='color:#ffe66d;font-size:1rem'>🔴 Most Common Crimes</b>",
                    unsafe_allow_html=True)
        for i, (crime, count) in enumerate(crime_dist_all.head(5).items()):
            pct   = count / len(df) * 100
            color = ["#e94560","#f4623a","#ff8c42","#ffcc00","#64ffda"][i]
            st.markdown(f"""
            <div class="summary-card" style="border-left:4px solid {color}">
                <b>{crime}</b><br>
                <span>{count:,} crimes &nbsp;|&nbsp; {pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)

    with summary_cols[1]:
        st.markdown("<b style='color:#ffe66d;font-size:1rem'>⚠️ Crime by Severity Level</b>",
                    unsafe_allow_html=True)
        sev_groups = {
            "Critical (9–10)": df["Crime_Severity_Score"].isin([9,10]).sum(),
            "High     (7–8) ": df["Crime_Severity_Score"].isin([7,8]).sum(),
            "Medium   (5–6) ": df["Crime_Severity_Score"].isin([5,6]).sum(),
            "Low      (3–4) ": df["Crime_Severity_Score"].isin([3,4]).sum(),
            "Minimal  (1–2) ": df["Crime_Severity_Score"].isin([1,2]).sum(),
        }
        sev_colors = ["#e94560","#f4623a","#ff8c42","#ffcc00","#64ffda"]
        for (label, count), color in zip(sev_groups.items(), sev_colors):
            pct = count / len(df) * 100
            st.markdown(f"""
            <div class="summary-card" style="border-left:4px solid {color}">
                <b>{label}</b><br>
                <span>{count:,} crimes &nbsp;|&nbsp; {pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)

    with summary_cols[2]:
        st.markdown("<b style='color:#ffe66d;font-size:1rem'>📌 Quick Facts</b>",
                    unsafe_allow_html=True)
        peak_hour   = df["Hour"].value_counts().idxmax()
        peak_day    = df["Day_of_Week"].value_counts().idxmax()
        peak_season = df["Season"].value_counts().idxmax()
        peak_dist   = df["District"].value_counts().dropna().idxmax()
        facts = [
            ("🕐 Peak Hour",    f"{int(peak_hour):02d}:00",    "#4361ee"),
            ("📅 Busiest Day",  peak_day,                       "#7209b7"),
            ("🌤️ Peak Season",  peak_season,                    "#f4623a"),
            ("🏢 Top District", f"District {int(peak_dist)}",  "#2ec4b6"),
            ("☠️ Most Severe",  "Homicide (10/10)",             "#e94560"),
        ]
        for label, value, color in facts:
            st.markdown(f"""
            <div class="summary-card" style="border-left:4px solid {color}">
                <b>{label}</b><br>
                <span>{value}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TOP 15 CRIMES — HIGHLIGHTED BAR CHART ────────────────
    st.markdown('<div class="section-header">Top 15 Crime Types — Highlighted Overview</div>',
                unsafe_allow_html=True)

    top15 = crime_dist_all.head(15).reset_index()
    top15.columns = ["Crime Type", "Count"]
    top15["Pct"]      = (top15["Count"] / len(df) * 100).round(2)
    top15["Severity"] = top15["Crime Type"].map({
        "HOMICIDE":10,"CRIM SEXUAL ASSAULT":9,"KIDNAPPING":9,
        "OFFENSE INVOLVING CHILDREN":8,"SEX OFFENSE":8,"ROBBERY":8,
        "WEAPONS VIOLATION":7,"ASSAULT":7,"ARSON":7,"BATTERY":6,
        "BURGLARY":5,"STALKING":5,"INTIMIDATION":5,"MOTOR VEHICLE THEFT":5,
        "THEFT":4,"CRIMINAL DAMAGE":4,"DECEPTIVE PRACTICE":3,
        "NARCOTICS":3,"PUBLIC PEACE VIOLATION":3,
        "INTERFERENCE WITH PUBLIC OFFICER":3,"CRIMINAL TRESPASS":2,
        "GAMBLING":2,"LIQUOR LAW VIOLATION":2,"OBSCENITY":2,
    }).fillna(1)
    top15["Label"] = top15.apply(
        lambda r: f"{r['Crime Type']}  ({r['Pct']}%)", axis=1)

    fig_top15 = px.bar(
        top15.sort_values("Count"),
        x="Count", y="Label",
        orientation="h",
        color="Severity",
        color_continuous_scale=["#90EE90","#ffcc00","#ff8c42","#f4623a","#e94560"],
        range_color=[1, 10],
        text="Count",
        title="Top 15 Crimes — Color Intensity = Severity Score",
        labels={"Count":"Number of Crimes","Label":"Crime Type","Severity":"Severity (1–10)"},
    )
    fig_top15.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_top15.update_layout(
        height=520,
        margin=dict(l=0, r=80, t=40, b=0),
        coloraxis_colorbar=dict(title="Severity", tickvals=[1,3,5,7,9,10]),
        yaxis_title="",
    )
    st.plotly_chart(fig_top15, use_container_width=True)

    # ── CRIME HIGHLIGHTS — EACH CRIME TYPE AS A CARD ─────────
    st.markdown('<div class="section-header">Crime Type Highlights</div>', unsafe_allow_html=True)

    severity_map_home = {
        "HOMICIDE":10,"CRIM SEXUAL ASSAULT":9,"KIDNAPPING":9,
        "OFFENSE INVOLVING CHILDREN":8,"SEX OFFENSE":8,"ROBBERY":8,
        "WEAPONS VIOLATION":7,"ASSAULT":7,"ARSON":7,"BATTERY":6,
        "BURGLARY":5,"STALKING":5,"MOTOR VEHICLE THEFT":5,
        "THEFT":4,"CRIMINAL DAMAGE":4,"DECEPTIVE PRACTICE":3,
        "NARCOTICS":3,"PUBLIC PEACE VIOLATION":3,"CRIMINAL TRESPASS":2,
    }

    def sev_color(score):
        if score >= 9:  return "#e94560"
        elif score >= 7: return "#f4623a"
        elif score >= 5: return "#ff8c42"
        elif score >= 3: return "#ffcc00"
        else:            return "#90EE90"

    highlight_crimes = crime_dist_all.head(15)
    cards_per_row    = 5
    crime_list       = list(highlight_crimes.items())

    for row_start in range(0, len(crime_list), cards_per_row):
        row_crimes = crime_list[row_start : row_start + cards_per_row]
        cols       = st.columns(cards_per_row)
        for col, (crime, count) in zip(cols, row_crimes):
            sev   = severity_map_home.get(crime, 1)
            color = sev_color(sev)
            pct   = count / len(df) * 100
            arr   = df[df["Primary Type"] == crime]["Arrest"].mean() * 100
            with col:
                st.markdown(f"""
                <div style="background:white;border-top:5px solid {color};
                            border-radius:10px;padding:0.9rem 0.7rem;
                            box-shadow:0 2px 8px rgba(0,0,0,0.1);
                            text-align:center;min-height:140px;margin-bottom:0.5rem">
                    <div style="font-size:0.78rem;font-weight:700;color:#222;
                                margin-bottom:0.4rem;line-height:1.2">{crime}</div>
                    <div style="font-size:1.3rem;font-weight:800;color:{color}">{count:,}</div>
                    <div style="font-size:0.7rem;color:#888">{pct:.1f}% of crimes</div>
                    <hr style="margin:0.4rem 0;border-color:#eee">
                    <div style="font-size:0.72rem;color:#555">
                        Severity: <b style="color:{color}">{sev}/10</b><br>
                        Arrest Rate: <b>{arr:.0f}%</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CRIME MAP ────────────────────────────────────────────
    st.markdown('<div class="section-header">Crime Locations — Colored by Severity</div>',
                unsafe_allow_html=True)
    map_sample = df[["Latitude","Longitude","Primary Type","Crime_Severity_Score"]].dropna().sample(
        min(8000, len(df)), random_state=42)
    fig_map = px.scatter_mapbox(
        map_sample, lat="Latitude", lon="Longitude",
        color="Crime_Severity_Score",
        color_continuous_scale="YlOrRd",
        range_color=[1, 10],
        hover_data={"Primary Type": True, "Crime_Severity_Score": True},
        mapbox_style="carto-darkmatter", zoom=10,
        title="Crime Hotspot Map — Darker Red = More Severe"
    )
    fig_map.update_traces(marker_size=4, marker_opacity=0.7)
    fig_map.update_layout(height=480, margin=dict(l=0,r=0,t=30,b=0),
                          coloraxis_colorbar=dict(title="Severity"))
    st.plotly_chart(fig_map, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    patrol_car_animation()
    st.markdown('<div class="main-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    cpd_label()
    st.markdown("---")

    eda_tab = st.tabs(["Crime Types", "Temporal", "Seasonal",
                        "Geographic", "Arrest & Domestic", "Heatmaps", "Correlation"])

    # ── TAB 1: CRIME TYPES ──────────────────────────────────
    with eda_tab[0]:
        st.markdown('<div class="section-header">Crime Type Distribution</div>', unsafe_allow_html=True)
        crime_dist = df["Primary Type"].value_counts()
        top_n = st.slider("Show top N crime types", 5, len(crime_dist), 15)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(crime_dist.head(top_n).sort_values(),
                         orientation="h", color=crime_dist.head(top_n).sort_values().values,
                         color_continuous_scale="Blues",
                         labels={"value":"Count","index":"Crime Type"},
                         title=f"Top {top_n} Crime Types — Bar Chart")
            fig.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            threshold = 1.5
            main  = crime_dist[crime_dist/len(df)*100 >= threshold]
            other = crime_dist[crime_dist/len(df)*100 <  threshold].sum()
            pie_data = pd.concat([main, pd.Series({"OTHERS": other})])
            fig = px.pie(values=pie_data.values, names=pie_data.index,
                         title="Crime Type Distribution — Pie Chart",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=450, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">💡 <b>Insight:</b> THEFT and BATTERY alone account for over 40% of all crimes. This class imbalance means clustering algorithms will naturally form larger zones for property crimes.</div>',
                    unsafe_allow_html=True)

    # ── TAB 2: TEMPORAL ─────────────────────────────────────
    with eda_tab[1]:
        st.markdown('<div class="section-header">Temporal Crime Patterns</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            hourly = df["Hour"].value_counts().sort_index()
            fig = px.bar(x=hourly.index, y=hourly.values,
                         labels={"x":"Hour of Day","y":"Crime Count"},
                         title="Crimes by Hour of Day",
                         color=hourly.values, color_continuous_scale="Oranges")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily = df["Day_of_Week"].value_counts().reindex(day_order)
            fig = px.bar(x=daily.index, y=daily.values,
                         labels={"x":"Day","y":"Crime Count"},
                         title="Crimes by Day of Week",
                         color=daily.values, color_continuous_scale="Blues")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Yearly trend
        yearly = df["Year"].value_counts().sort_index()
        fig = px.line(x=yearly.index, y=yearly.values,
                      markers=True,
                      labels={"x":"Year","y":"Crime Count"},
                      title="Crime Trend by Year",
                      color_discrete_sequence=["#e94560"])
        fig.update_traces(line_width=3)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">💡 <b>Insight:</b> Crime peaks at midnight and noon. Friday/Saturday have highest crime. Early morning (4–6am) is safest. Use this for patrol shift scheduling.</div>',
                    unsafe_allow_html=True)

    # ── TAB 3: SEASONAL ─────────────────────────────────────
    with eda_tab[2]:
        st.markdown('<div class="section-header">Seasonal & Monthly Patterns</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            season_dist = df["Season"].value_counts()
            fig = px.pie(values=season_dist.values, names=season_dist.index,
                         title="Crime Distribution by Season",
                         color_discrete_map={"Summer":"#FF6B6B","Fall":"#FFA500",
                                              "Winter":"#6BB5FF","Spring":"#90EE90"})
            fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.05]*4)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            monthly = df["Month"].value_counts().sort_index()
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            fig = px.bar(x=[month_names[m] for m in monthly.index],
                         y=monthly.values,
                         labels={"x":"Month","y":"Crime Count"},
                         title="Crimes by Month",
                         color=monthly.values, color_continuous_scale="RdYlGn_r")
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="insight-box">💡 <b>Insight:</b> Summer (July/August) records highest crime — people are outdoors. Winter has fewest crimes. This validates Season as a meaningful clustering feature.</div>',
                    unsafe_allow_html=True)

    # ── TAB 4: GEOGRAPHIC ────────────────────────────────────
    with eda_tab[3]:
        st.markdown('<div class="section-header">Geographic Crime Distribution</div>', unsafe_allow_html=True)

        map_n = st.slider("Map sample size", 5000, min(50000, len(df)), 20000, 5000)
        map_df = df[["Latitude","Longitude","Primary Type","Crime_Severity_Score"]].dropna().sample(map_n, random_state=42)

        fig = px.density_mapbox(map_df, lat="Latitude", lon="Longitude",
                                z="Crime_Severity_Score", radius=8,
                                mapbox_style="carto-darkmatter", zoom=10,
                                color_continuous_scale="YlOrRd",
                                title="Crime Density Heatmap — Chicago")
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Top districts
        st.markdown('<div class="section-header">Top Districts by Crime Count</div>', unsafe_allow_html=True)
        dist_counts = df["District"].value_counts().dropna().head(15)
        fig = px.bar(x=dist_counts.index.astype(int).astype(str),
                     y=dist_counts.values,
                     labels={"x":"District","y":"Crime Count"},
                     title="Top 15 Police Districts",
                     color=dist_counts.values, color_continuous_scale="Purples")
        fig.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: ARREST & DOMESTIC ─────────────────────────────
    with eda_tab[4]:
        st.markdown('<div class="section-header">Arrest Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            arrest_counts = df["Arrest"].value_counts()
            fig = px.pie(values=arrest_counts.values,
                         names=["No Arrest","Arrested"],
                         title="Overall Arrest Rate",
                         color_discrete_sequence=["#FF6B6B","#6BCB77"])
            fig.update_traces(textinfo="percent+label", pull=[0.05, 0.05])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            arrest_by_crime = (df.groupby("Primary Type")["Arrest"].mean() * 100).sort_values(ascending=False).head(12)
            fig = px.bar(arrest_by_crime.sort_values(),
                         orientation="h",
                         labels={"value":"Arrest Rate (%)","index":"Crime Type"},
                         title="Arrest Rate by Crime Type (%)",
                         color=arrest_by_crime.sort_values().values,
                         color_continuous_scale="Greens")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Domestic Crime Analysis</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            dom_counts = df["Domestic"].value_counts()
            fig = px.pie(values=dom_counts.values,
                         names=["Non-Domestic","Domestic"],
                         title="Domestic vs Non-Domestic",
                         color_discrete_sequence=["#4D96FF","#FF6B6B"])
            fig.update_traces(textinfo="percent+label", pull=[0.05,0.05])
            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            dom_by_crime = df[df["Domestic"]==True]["Primary Type"].value_counts().head(10)
            fig = px.bar(dom_by_crime.sort_values(),
                         orientation="h",
                         title="Top 10 Domestic Crime Types",
                         color=dom_by_crime.sort_values().values,
                         color_continuous_scale="Reds",
                         labels={"value":"Count","index":"Crime Type"})
            fig.update_layout(height=330, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6: HEATMAPS ──────────────────────────────────────
    with eda_tab[5]:
        st.markdown('<div class="section-header">Hour × Day of Week Crime Heatmap</div>',
                    unsafe_allow_html=True)
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = df.groupby(["Day_of_Week","Hour"]).size().unstack(fill_value=0).reindex(day_order)

        fig = px.imshow(pivot, color_continuous_scale="YlOrRd",
                        labels=dict(x="Hour of Day", y="Day of Week", color="Crime Count"),
                        title="Crime Frequency: Hour vs Day",
                        aspect="auto")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Crime Type × Time of Day Heatmap</div>',
                    unsafe_allow_html=True)
        top_crimes = df["Primary Type"].value_counts().head(10).index
        pivot2 = df[df["Primary Type"].isin(top_crimes)]\
                   .groupby(["Primary Type","Time_of_Day"]).size().unstack(fill_value=0)
        tod_order = ["Late Night","Morning","Afternoon","Evening"]
        pivot2 = pivot2.reindex(columns=[c for c in tod_order if c in pivot2.columns])
        fig2 = px.imshow(pivot2, color_continuous_scale="Blues",
                         labels=dict(x="Time of Day", y="Crime Type", color="Count"),
                         title="Crime Type vs Time of Day",
                         aspect="auto")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 7: CORRELATION ───────────────────────────────────
    with eda_tab[6]:
        st.markdown('<div class="section-header">Feature Correlation Heatmap</div>',
                    unsafe_allow_html=True)

        num_cols = ["Hour","Day_Num","Month","Is_Weekend_Int",
                    "Crime_Severity_Score","Crime_Type_Encoded","Latitude","Longitude"]
        avail = [c for c in num_cols if c in df.columns]
        corr  = df[avail].corr().round(2)

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, mask=mask, ax=ax, linewidths=0.5,
                    cbar_kws={"label":"Correlation"})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="insight-box">💡 <b>Insight:</b> Values close to 1 or -1 indicate strong relationships between features. Values near 0 indicate independence. Strong correlations help identify redundant features before modeling.</div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — GEOGRAPHIC CLUSTERING
# ════════════════════════════════════════════════════════════
elif page == "🗺️  Geographic Clustering":
    patrol_car_animation()
    st.markdown('<div class="main-title">🗺️ Geographic Crime Hotspot Clustering</div>', unsafe_allow_html=True)
    cpd_label()
    st.markdown("---")

    algo = st.selectbox("Select Algorithm", ["K-Means", "DBSCAN", "Hierarchical"])

    geo_df = df[["Latitude","Longitude","Primary Type","Crime_Severity_Score"]].dropna()
    n_geo  = st.slider("Sample size for clustering", 10000, min(100000, len(geo_df)), 30000, 5000)
    geo_sample = geo_df.sample(n_geo, random_state=42).reset_index(drop=True)

    if algo == "K-Means":
        k = st.slider("Number of Clusters (K)", 2, 15, 6)
        labels, sil, db_score, centers, inertia = run_kmeans(
            geo_sample["Latitude"].values, geo_sample["Longitude"].values, k)
        geo_sample["Cluster"] = labels.astype(str)

        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{sil:.4f}", delta="Target > 0.5")
        col2.metric("Davies-Bouldin",   f"{db_score:.4f}", delta="Lower is better")
        col3.metric("Inertia",          f"{inertia:,.0f}")

        fig = px.scatter_mapbox(
            geo_sample, lat="Latitude", lon="Longitude",
            color="Cluster", hover_data=["Primary Type","Crime_Severity_Score"],
            mapbox_style="carto-darkmatter", zoom=10,
            title=f"K-Means Crime Hotspots (K={k})",
            color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(marker_size=3, marker_opacity=0.6)

        # Add cluster centers
        center_df = pd.DataFrame(centers, columns=["Latitude","Longitude"])
        center_df["Cluster"] = [f"Center {i}" for i in range(len(centers))]
        for _, row in center_df.iterrows():
            fig.add_trace(go.Scattermapbox(
                lat=[row["Latitude"]], lon=[row["Longitude"]],
                mode="markers", marker=dict(size=16, color="white", symbol="star"),
                name=row["Cluster"]))

        fig.update_layout(height=550, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Cluster stats
        st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)
        geo_sample["Cluster_int"] = labels
        cluster_stats = geo_sample.groupby("Cluster_int").agg(
            Count=("Latitude","count"),
            Avg_Lat=("Latitude","mean"),
            Avg_Lon=("Longitude","mean"),
            Avg_Severity=("Crime_Severity_Score","mean")
        ).round(4).reset_index()
        cluster_stats.columns = ["Cluster","Count","Avg Latitude","Avg Longitude","Avg Severity"]
        st.dataframe(cluster_stats, use_container_width=True)

    elif algo == "DBSCAN":
        col_a, col_b = st.columns(2)
        eps         = col_a.slider("Epsilon (eps)", 0.01, 0.30, 0.08, 0.01)
        min_samples = col_b.slider("Min Samples", 10, 200, 50, 10)

        labels, sil, n_cls, n_noise = run_dbscan(
            geo_sample["Latitude"].values, geo_sample["Longitude"].values, eps, min_samples)
        geo_sample["Cluster"] = labels.astype(str)

        col1, col2, col3 = st.columns(3)
        col1.metric("Hotspot Zones Found", n_cls)
        col2.metric("Noise Points Removed", f"{n_noise:,}")
        col3.metric("Silhouette Score", f"{sil:.4f}" if sil > 0 else "N/A")

        # Mark noise as separate
        geo_sample["Display"] = geo_sample["Cluster"].apply(
            lambda x: "Noise" if x == "-1" else f"Zone {x}")

        fig = px.scatter_mapbox(
            geo_sample, lat="Latitude", lon="Longitude",
            color="Display", hover_data=["Primary Type","Crime_Severity_Score"],
            mapbox_style="carto-darkmatter", zoom=10,
            title=f"DBSCAN Hotspots — {n_cls} zones, {n_noise:,} noise points removed",
            color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(marker_size=3, marker_opacity=0.6)
        fig.update_layout(height=550, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif algo == "Hierarchical":
        k_hier = st.slider("Number of Clusters", 2, 12, 5)
        labels, sil, db_score = run_hierarchical(
            geo_sample["Latitude"].values, geo_sample["Longitude"].values, k_hier)
        geo_sample["Cluster"] = labels.astype(str)

        col1, col2 = st.columns(2)
        col1.metric("Silhouette Score",  f"{sil:.4f}")
        col2.metric("Davies-Bouldin",    f"{db_score:.4f}")

        fig = px.scatter_mapbox(
            geo_sample, lat="Latitude", lon="Longitude",
            color="Cluster", hover_data=["Primary Type","Crime_Severity_Score"],
            mapbox_style="carto-darkmatter", zoom=10,
            title=f"Hierarchical Crime Zones (K={k_hier})",
            color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(marker_size=3, marker_opacity=0.6)
        fig.update_layout(height=550, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 4 — TEMPORAL CLUSTERING
# ════════════════════════════════════════════════════════════
elif page == "⏰  Temporal Clustering":
    patrol_car_animation()
    st.markdown('<div class="main-title">⏰ Temporal Crime Pattern Clustering</div>', unsafe_allow_html=True)
    cpd_label()
    st.markdown("---")

    features = ["Hour","Day_Num","Month","Is_Weekend_Int","Crime_Severity_Score"]
    avail_f  = [f for f in features if f in df.columns]
    temp_n   = st.slider("Sample size", 10000, min(80000, len(df)), 30000, 5000)
    k_temp   = st.slider("Number of temporal clusters (K)", 2, 8, 4)

    temp_sample = df[avail_f].dropna().sample(temp_n, random_state=42).reset_index(drop=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(temp_sample)
    km     = KMeans(n_clusters=k_temp, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    temp_sample["Cluster"] = labels.astype(str)
    sil_temp = silhouette_score(scaled, labels, sample_size=5000, random_state=42)

    st.metric("Temporal Silhouette Score", f"{sil_temp:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(temp_sample.sample(min(5000, len(temp_sample))),
                         x="Hour", y="Crime_Severity_Score",
                         color="Cluster",
                         opacity=0.5,
                         title="Temporal Clusters: Hour vs Severity",
                         color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cluster_hour = temp_sample.groupby(["Cluster","Hour"]).size().reset_index(name="Count")
        fig = px.line(cluster_hour, x="Hour", y="Count", color="Cluster",
                      title="Crime Count by Hour per Cluster",
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(line_width=2.5)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster profiles
    st.markdown('<div class="section-header">Temporal Cluster Profiles</div>', unsafe_allow_html=True)
    profile = temp_sample.groupby("Cluster")[avail_f].mean().round(2)
    profile["Dominant Time"] = profile["Hour"].apply(
        lambda h: "Late Night" if h < 6 else "Morning" if h < 12 else "Afternoon" if h < 18 else "Evening")
    profile["Weekend Heavy"] = profile["Is_Weekend_Int"].apply(lambda x: "Yes" if x > 0.4 else "No")
    st.dataframe(profile, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Each temporal cluster represents a distinct crime window. Clusters with high severity + late-night hours = highest risk patrol windows. Use these to schedule officer deployments.</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "📈  Model Comparison":
    patrol_car_animation()
    st.markdown('<div class="main-title">📈 Clustering Model Comparison</div>', unsafe_allow_html=True)
    cpd_label()
    st.markdown("---")

    st.info("Running all 3 algorithms on the same data for fair comparison. This may take a moment...")

    geo_df     = df[["Latitude","Longitude"]].dropna()
    comp_sample = geo_df.sample(min(30000, len(geo_df)), random_state=42)
    lats = comp_sample["Latitude"].values
    lons = comp_sample["Longitude"].values

    k_comp = st.slider("K for K-Means and Hierarchical", 3, 12, 6)

    with st.spinner("Computing..."):
        km_labels, km_sil, km_db, _, km_inertia = run_kmeans(lats, lons, k_comp)
        db_labels, db_sil, db_n_cls, db_noise    = run_dbscan(lats, lons, 0.08, 50)
        hi_labels, hi_sil, hi_db                 = run_hierarchical(lats, lons, k_comp)

    # Summary table
    comparison = pd.DataFrame({
        "Algorithm"         : ["K-Means", "DBSCAN", "Hierarchical"],
        "Clusters Found"    : [k_comp, db_n_cls, k_comp],
        "Silhouette Score"  : [round(km_sil,4), round(db_sil,4), round(hi_sil,4)],
        "Davies-Bouldin"    : [round(km_db,4), "N/A", round(hi_db,4)],
        "Noise Points"      : [0, db_noise, 0],
        "Best Use"          : ["Patrol zones","Natural hotspots","Zone hierarchy"],
        "Passes Target"     : ["✅ Yes" if km_sil>0.5 else "❌ No",
                               "✅ Yes" if db_sil>0.5 else "❌ No",
                               "✅ Yes" if hi_sil>0.5 else "❌ No"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Silhouette bar chart
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        algos  = ["K-Means","DBSCAN","Hierarchical"]
        scores = [km_sil, db_sil, hi_sil]
        colors = ["#4361ee","#e94560","#2ec4b6"]
        fig.add_trace(go.Bar(x=algos, y=scores, marker_color=colors,
                             text=[f"{s:.4f}" for s in scores], textposition="outside"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="Target 0.5")
        fig.update_layout(title="Silhouette Score Comparison",
                          yaxis_title="Silhouette Score",
                          height=380, yaxis_range=[0, max(scores)+0.15])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Elbow curve for K-Means
        st.markdown("**K-Means Elbow Curve**")
        inertias, sils = [], []
        coords_e  = np.column_stack([lats, lons])
        scaler_e  = StandardScaler()
        scaled_e  = scaler_e.fit_transform(coords_e)
        k_range   = range(2, 12)
        for kk in k_range:
            km_e  = KMeans(n_clusters=kk, random_state=42, n_init=5)
            lbl_e = km_e.fit_predict(scaled_e)
            inertias.append(km_e.inertia_)
            sils.append(silhouette_score(scaled_e, lbl_e, sample_size=3000, random_state=42))

        fig_e = make_subplots(specs=[[{"secondary_y": True}]])
        fig_e.add_trace(go.Scatter(x=list(k_range), y=inertias, name="Inertia",
                                   line=dict(color="#4361ee", width=2.5), mode="lines+markers"),
                        secondary_y=False)
        fig_e.add_trace(go.Scatter(x=list(k_range), y=sils, name="Silhouette",
                                   line=dict(color="#e94560", width=2.5, dash="dash"),
                                   mode="lines+markers"),
                        secondary_y=True)
        fig_e.update_layout(height=380, title="Elbow + Silhouette vs K")
        fig_e.update_yaxes(title_text="Inertia",          secondary_y=False)
        fig_e.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        st.plotly_chart(fig_e, use_container_width=True)

    # Best algorithm recommendation
    best_idx  = scores.index(max(scores))
    best_name = algos[best_idx]
    st.success(f"**Best Algorithm: {best_name}** with Silhouette Score = {max(scores):.4f}")


# ════════════════════════════════════════════════════════════
#  PAGE 6 — CRIME ANALYZER
# ════════════════════════════════════════════════════════════
elif page == "🔍  Crime Analyzer":
    patrol_car_animation()
    st.markdown('<div class="main-title">🔍 Crime Pattern Analyzer</div>', unsafe_allow_html=True)
    cpd_label()
    st.markdown("---")

    st.markdown("### Filter & Explore Crime Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        crime_filter = st.multiselect("Crime Type", sorted(df["Primary Type"].unique()),
                                       default=["THEFT","BATTERY","ASSAULT"])
    with col2:
        hour_range = st.slider("Hour Range", 0, 23, (0, 23))
    with col3:
        season_filter = st.multiselect("Season", ["Spring","Summer","Fall","Winter"],
                                        default=["Summer","Winter"])

    filtered = df[
        df["Primary Type"].isin(crime_filter) &
        df["Hour"].between(hour_range[0], hour_range[1]) &
        df["Season"].isin(season_filter)
    ]

    st.markdown(f"**Filtered records: {len(filtered):,}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records",        f"{len(filtered):,}")
    col2.metric("Arrest Rate",    f"{filtered['Arrest'].mean()*100:.1f}%" if len(filtered) else "N/A")
    col3.metric("Avg Severity",   f"{filtered['Crime_Severity_Score'].mean():.2f}" if len(filtered) else "N/A")
    col4.metric("Domestic Rate",  f"{filtered['Domestic'].mean()*100:.1f}%" if len(filtered) else "N/A")

    if len(filtered) > 0:
        col_l, col_r = st.columns(2)
        with col_l:
            hourly_f = filtered["Hour"].value_counts().sort_index()
            fig = px.bar(x=hourly_f.index, y=hourly_f.values,
                         title="Filtered: Crimes by Hour",
                         color=hourly_f.values, color_continuous_scale="Reds",
                         labels={"x":"Hour","y":"Count"})
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            type_f = filtered["Primary Type"].value_counts().head(10)
            fig = px.bar(type_f.sort_values(), orientation="h",
                         title="Filtered: Crime Type Breakdown",
                         color=type_f.sort_values().values,
                         color_continuous_scale="Blues",
                         labels={"value":"Count","index":"Type"})
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Map
        map_f = filtered[["Latitude","Longitude","Primary Type","Crime_Severity_Score"]].dropna()
        if len(map_f) > 0:
            map_f = map_f.sample(min(10000, len(map_f)), random_state=42)
            fig_m = px.scatter_mapbox(
                map_f, lat="Latitude", lon="Longitude",
                color="Crime_Severity_Score",
                color_continuous_scale="YlOrRd",
                hover_data=["Primary Type"],
                mapbox_style="carto-darkmatter", zoom=10,
                title="Filtered Crime Locations")
            fig_m.update_traces(marker_size=4, marker_opacity=0.7)
            fig_m.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_m, use_container_width=True)

        # Raw data table
        st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
        show_cols = ["Date","Primary Type","Crime_Severity_Score","Latitude","Longitude",
                     "District","Arrest","Domestic","Season","Time_of_Day","Hour"]
        avail_show = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[avail_show].head(200), use_container_width=True)
        csv = filtered[avail_show].to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered Data as CSV", csv,
                           "filtered_crimes.csv", "text/csv")
    else:
        st.warning("No records match the selected filters. Try adjusting your selections.")