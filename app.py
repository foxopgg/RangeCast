"""
RangeCast Health Officer Dashboard (app.py)
===========================================
Premium mobile-first Streamlit dashboard for Indian District Health Officers.
Designed for district epidemiologists briefing the DHO and CMO.

Features:
- Calibrated forecast ranges (50% and 80% prediction bands)
- Empirical accuracy scorecard from real rolling backtests
- Newsvendor resource planner (beds, oxygen, kits)
- Groq AI assistant with verified number badge
- Multi-language briefings (English, Tamil, Hindi)
- WhatsApp one-card summary export
- Light/dark mode, mobile-first, WCAG AA contrast
"""

import os
import io
import time
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- RangeCast Core Modules -----------------------------------------------
from data import OutbreakDataLoader
from nowcast import OutbreakNowcaster
from models import RangeCastEnsemble, NaivePersistenceBaseline
from conformal import AdaptiveConformalCalibrator
from scoring import (
    interval_score,
    weighted_interval_score,
    calculate_coverage,
    assign_trust_badge,
    compute_scorecard_summary,
)
from planner import (
    PlannerAssumptions,
    get_default_disease_assumptions,
    compute_newsvendor_fractile,
    calculate_resource_plan,
)
from templates import (
    generate_english_briefing,
    generate_tamil_briefing,
    generate_hindi_briefing,
    generate_whatsapp_card,
)
from backtest import RollingBacktestRunner
from demo_surge import run_surge_replay

# ---- Groq AI (optional — falls back gracefully) ---------------------------
try:
    import groq_client as _gc
    _GROQ_AVAILABLE = True
except Exception:
    _GROQ_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ===========================================================================
# PAGE CONFIG & GLOBAL THEME
# ===========================================================================
st.set_page_config(
    page_title="RangeCast · District Outbreak Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "RangeCast — Calibrated Outbreak Forecasting for Indian District Health Officers.",
    },
)

# ---- Google Fonts + Premium CSS ------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">

<style>
/* ===== ROOT TOKENS ===== */
:root {
  --navy-900: #0A1628;
  --navy-800: #0F1F3D;
  --navy-700: #162547;
  --navy-600: #1D3461;
  --teal-500: #0EA5E9;
  --teal-400: #38BDF8;
  --teal-300: #7DD3FC;
  --amber-500: #F59E0B;
  --amber-400: #FBBF24;
  --red-500: #EF4444;
  --green-400: #34D399;
  --green-500: #10B981;
  --slate-100: #F1F5F9;
  --slate-200: #E2E8F0;
  --slate-400: #94A3B8;
  --slate-500: #64748B;
  --slate-600: #475569;
  --slate-700: #334155;
  --slate-800: #1E293B;
  --white: #FFFFFF;
}

/* ===== GLOBAL RESET ===== */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ===== DARK APP BG ===== */
.stApp {
  background: linear-gradient(135deg, #0A1628 0%, #0D1B35 50%, #0A1E40 100%) !important;
  min-height: 100vh;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F1F3D 0%, #0A1628 100%) !important;
  border-right: 1px solid rgba(14, 165, 233, 0.15) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] small {
  color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h3 {
  color: #38BDF8 !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em;
}

/* ===== TOP NAV HEADER ===== */
.rc-header {
  background: linear-gradient(90deg, rgba(14,165,233,0.12) 0%, rgba(56,189,248,0.06) 100%);
  border: 1px solid rgba(14,165,233,0.25);
  border-radius: 16px;
  padding: 20px 28px;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}
.rc-logo {
  font-family: 'Rajdhani', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: #38BDF8;
  letter-spacing: 0.05em;
}
.rc-subtitle {
  font-size: 0.85rem;
  color: #94A3B8;
  font-weight: 400;
}
.rc-district-pill {
  background: rgba(14,165,233,0.2);
  border: 1px solid rgba(14,165,233,0.4);
  color: #7DD3FC;
  padding: 4px 14px;
  border-radius: 100px;
  font-size: 0.85rem;
  font-weight: 600;
  white-space: nowrap;
}

/* ===== ALERT BANNERS ===== */
.alert-shift {
  background: linear-gradient(90deg, rgba(239,68,68,0.18) 0%, rgba(239,68,68,0.08) 100%);
  border-left: 4px solid #EF4444;
  border-radius: 8px;
  padding: 12px 18px;
  margin-bottom: 16px;
  color: #FCA5A5;
  font-weight: 600;
  font-size: 0.9rem;
}
.alert-small {
  background: linear-gradient(90deg, rgba(245,158,11,0.18) 0%, rgba(245,158,11,0.06) 100%);
  border-left: 4px solid #F59E0B;
  border-radius: 8px;
  padding: 12px 18px;
  margin-bottom: 16px;
  color: #FDE68A;
  font-weight: 500;
  font-size: 0.88rem;
}
.alert-success {
  background: linear-gradient(90deg, rgba(52,211,153,0.15) 0%, rgba(16,185,129,0.06) 100%);
  border-left: 4px solid #34D399;
  border-radius: 8px;
  padding: 12px 18px;
  margin-bottom: 16px;
  color: #6EE7B7;
  font-size: 0.88rem;
}

/* ===== KPI CARDS ===== */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.kpi-card {
  background: linear-gradient(145deg, #162547 0%, #1D3461 100%);
  border: 1px solid rgba(14,165,233,0.2);
  border-radius: 16px;
  padding: 20px 22px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.15s;
}
.kpi-card:hover {
  border-color: rgba(14,165,233,0.5);
  transform: translateY(-2px);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #0EA5E9, #38BDF8);
  border-radius: 16px 16px 0 0;
}
.kpi-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94A3B8;
  margin-bottom: 8px;
}
.kpi-value {
  font-family: 'Rajdhani', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
  color: #F8FAFC;
}
.kpi-value.teal { color: #38BDF8; }
.kpi-value.green { color: #34D399; }
.kpi-value.amber { color: #FBBF24; }
.kpi-value.red { color: #F87171; }
.kpi-sub {
  font-size: 0.78rem;
  color: #64748B;
  margin-top: 6px;
}
.kpi-sub b { color: #94A3B8; }

/* ===== TRUST BADGE ===== */
.badge-high {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(52,211,153,0.12);
  border: 1.5px solid #34D399;
  color: #34D399;
  padding: 6px 16px;
  border-radius: 100px;
  font-weight: 700;
  font-size: 0.85rem;
}
.badge-medium {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(251,191,36,0.12);
  border: 1.5px solid #FBBF24;
  color: #FBBF24;
  padding: 6px 16px;
  border-radius: 100px;
  font-weight: 700;
  font-size: 0.85rem;
}
.badge-low {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(239,68,68,0.12);
  border: 1.5px solid #EF4444;
  color: #EF4444;
  padding: 6px 16px;
  border-radius: 100px;
  font-weight: 700;
  font-size: 0.85rem;
}

/* ===== SECTION CARDS ===== */
.section-card {
  background: rgba(22,37,71,0.8);
  border: 1px solid rgba(14,165,233,0.15);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  backdrop-filter: blur(8px);
}
.section-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: #E2E8F0;
  letter-spacing: 0.03em;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}

/* ===== METRIC DELTA OVERRIDES ===== */
[data-testid="metric-container"] {
  background: rgba(22,37,71,0.6);
  border: 1px solid rgba(14,165,233,0.15);
  border-radius: 12px;
  padding: 14px 16px;
}
[data-testid="metric-container"] label {
  color: #94A3B8 !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: #F8FAFC !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.8rem !important;
  font-weight: 700 !important;
}

/* ===== DATA TABLES ===== */
[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
}
.stDataFrame thead tr th {
  background: #162547 !important;
  color: #94A3B8 !important;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ===== CHAT / AI PANEL ===== */
.chat-bubble-user {
  background: rgba(14,165,233,0.15);
  border: 1px solid rgba(14,165,233,0.3);
  border-radius: 12px 12px 4px 12px;
  padding: 12px 16px;
  color: #E2E8F0;
  margin: 8px 0 8px 40px;
  font-size: 0.9rem;
}
.chat-bubble-ai {
  background: rgba(22,37,71,0.9);
  border: 1px solid rgba(52,211,153,0.2);
  border-radius: 12px 12px 12px 4px;
  padding: 14px 18px;
  color: #CBD5E1;
  margin: 8px 40px 8px 0;
  font-size: 0.9rem;
  line-height: 1.6;
}
.verified-badge {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(52,211,153,0.12);
  border: 1px solid rgba(52,211,153,0.4);
  color: #34D399;
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 0.72rem;
  font-weight: 600;
  margin-left: 8px;
}
.unverified-badge {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(245,158,11,0.12);
  border: 1px solid rgba(245,158,11,0.3);
  color: #FBBF24;
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 0.72rem;
  font-weight: 600;
  margin-left: 8px;
}
.chip {
  display: inline-block;
  background: rgba(14,165,233,0.1);
  border: 1px solid rgba(14,165,233,0.25);
  color: #7DD3FC;
  padding: 5px 14px;
  border-radius: 100px;
  font-size: 0.8rem;
  margin: 4px;
  cursor: pointer;
  transition: background 0.15s;
}
.chip:hover {
  background: rgba(14,165,233,0.22);
}

/* ===== LANGUAGE TABS ===== */
.stTabs [data-baseweb="tab"] {
  color: #94A3B8 !important;
  font-size: 0.85rem;
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  color: #38BDF8 !important;
  border-bottom-color: #38BDF8 !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
  background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.85rem;
  padding: 8px 20px;
  transition: opacity 0.15s, transform 0.1s;
}
.stButton > button:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

/* ===== SLIDERS ===== */
[data-testid="stSlider"] .stSlider > div > div > div > div {
  background: #0EA5E9 !important;
}

/* ===== SELECTBOX & INPUTS ===== */
.stSelectbox > div > div {
  background: rgba(22,37,71,0.9) !important;
  border: 1px solid rgba(14,165,233,0.25) !important;
  border-radius: 8px !important;
  color: #E2E8F0 !important;
}

/* ===== DIVIDER ===== */
hr {
  border-color: rgba(14,165,233,0.15) !important;
  margin: 20px 0;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A1628; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ===== MOBILE RESPONSIVE ===== */
@media (max-width: 768px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .kpi-value { font-size: 1.5rem; }
  .rc-logo { font-size: 1.5rem; }
}
@media (max-width: 480px) {
  .kpi-grid { grid-template-columns: 1fr; }
}

/* ===== HIDE STREAMLIT CHROME ===== */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# DATA LOADING (CACHED)
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_district_data(disease: str, district: str):
    loader = OutbreakDataLoader(disease=disease)
    return loader.load_weekly_district_data(district)

@st.cache_data(show_spinner=False)
def get_districts_list(disease: str):
    loader = OutbreakDataLoader(disease=disease)
    return loader.get_districts()

@st.cache_data(show_spinner=False)
def load_or_run_backtest(disease: str, district: str):
    dis_slug = disease.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    cache_path = f"data/cache/eval_{dis_slug}_{district.lower()}.csv"
    legacy_path = f"data/cache/eval_{district.lower()}.csv"
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if disease == "COVID-19" and os.path.exists(legacy_path):
        return pd.read_csv(legacy_path)
    loader = OutbreakDataLoader(disease=disease)
    df_d = loader.load_weekly_district_data(district)
    runner = RollingBacktestRunner(min_train_weeks=12, horizon_weeks=4)
    eval_df = runner.run_district_backtest(df_d)
    os.makedirs("data/cache", exist_ok=True)
    eval_df.to_csv(cache_path, index=False)
    return eval_df


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("### 🛡️ **RangeCast**")
    st.markdown(
        "<small style='color:#64748B;'>Calibrated Outbreak Forecasting<br>for District Health Officers</small>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Disease picker
    selected_disease = st.selectbox(
        "🦠 Disease",
        options=OutbreakDataLoader.SUPPORTED_DISEASES,
        index=0,
        key="disease_sel",
    )

    # Disease metadata
    _DISEASE_META = {
        "COVID-19": ("🫁", "Airborne Respiratory"),
        "Dengue": ("🦟", "Vector-borne (Aedes)"),
        "Malaria": ("🦟", "Vector-borne (Anopheles)"),
        "Influenza (H1N1)": ("🫁", "Seasonal Respiratory"),
        "Acute Diarrheal Disease": ("💧", "Water-borne"),
    }
    _icon, _meta = _DISEASE_META.get(selected_disease, ("🦠", "Infectious Disease"))
    st.markdown(
        f"<div style='background:rgba(14,165,233,0.1);border:1px solid rgba(14,165,233,0.25);"
        f"border-radius:8px;padding:8px 12px;margin-bottom:12px;'>"
        f"<span style='font-size:1.1rem;'>{_icon}</span> "
        f"<span style='color:#7DD3FC;font-size:0.8rem;font-weight:600;'>{_meta}</span></div>",
        unsafe_allow_html=True,
    )

    # District picker
    districts = get_districts_list(selected_disease)
    default_idx = districts.index("Chennai") if "Chennai" in districts else 0
    selected_district = st.selectbox(
        "📍 District (Tamil Nadu)",
        options=districts,
        index=default_idx,
        key="district_sel",
    )

    st.divider()

    # Language toggle
    lang_map = {"English": "en", "தமிழ் (Tamil)": "ta", "हिन्दी (Hindi)": "hi"}
    selected_lang_label = st.radio(
        "🌐 Language",
        options=list(lang_map.keys()),
        index=0,
        horizontal=False,
        key="lang_sel",
    )
    selected_lang = lang_map[selected_lang_label]

    st.divider()

    # Navigation
    nav_page = st.radio(
        "Navigate",
        options=[
            "🏠 Dashboard",
            "📊 Accuracy Scorecard",
            "🏥 Resource Planner",
            "🔄 Compare Forecasts",
            "📢 Briefings & Export",
            "🤖 Ask RangeCast AI",
            "⚡ Surge Replay Demo",
        ],
        key="nav_sel",
    )

    st.divider()
    st.markdown(
        "<small style='color:#475569;'>Data: COVID-19 India Archives, IDSP/NVBDCP, "
        "IMD Rainfall, Festival Calendar.<br>"
        "Assumptions are editable placeholders, not statutory norms.</small>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# CORE DATA LOADING (shared across all pages)
# ===========================================================================
with st.spinner(f"Loading {selected_disease} data for {selected_district}…"):
    df_district = load_district_data(selected_disease, selected_district)
    df_eval = load_or_run_backtest(selected_disease, selected_district)

runner = RollingBacktestRunner()
scorecard_dict = runner.generate_district_scorecard(df_eval)
overall_score = scorecard_dict["overall"]
trust_badge = overall_score["trust_badge"]
is_small = bool(df_district["is_small_district"].iloc[0]) if "is_small_district" in df_district.columns else False


def get_current_forecast(df_curr, is_small_dist):
    """Generates the latest calibrated 4-week forecast."""
    ens = RangeCastEnsemble()
    ens.fit(df_curr)
    raw_preds = ens.predict(horizon_weeks=4, apply_pooling=True, is_small_district=is_small_dist)

    calibrator = AdaptiveConformalCalibrator()
    recent_eval = df_eval[df_eval["horizon_weeks"] == 1].tail(12)
    for _, r in recent_eval.iterrows():
        calibrator.scores_80.append(calibrator.score_nonconformity(r["actual"], r["q10"], r["q90"], r["q50"]))
        calibrator.scores_50.append(calibrator.score_nonconformity(r["actual"], r["q25"], r["q75"], r["q50"]))
        calibrator.recent_misses_80.append(0 if r["covered_80"] else 1)
        calibrator.recent_errors.append(abs(r["actual"] - r["q50"]) / max(5.0, r["q50"]))
    if not recent_eval.empty:
        calibrator.alpha_80 = float(recent_eval["conformal_alpha_80"].iloc[-1])
        calibrator.shift_alert_active = bool(recent_eval["shift_alert"].iloc[-1])

    cal_preds = []
    for p in raw_preds:
        res = calibrator.calibrate_forecast({
            "q10": p["q10"], "q25": p["q25"], "q50": p["q50"],
            "q75": p["q75"], "q90": p["q90"],
        })
        res["horizon"] = p["horizon"]
        res["baseline_q50"] = p["baseline_q50"]
        cal_preds.append(res)

    return cal_preds, calibrator.shift_alert_active

calibrated_preds, active_shift_alert = get_current_forecast(df_district, is_small)
latest_date_str = str(pd.to_datetime(df_district["week_end_date"].max()).strftime("%d %b %Y"))
latest_cases = int(df_district["cases"].iloc[-1])
prev_cases = int(df_district["cases"].iloc[-2]) if len(df_district) >= 2 else latest_cases
growth_rate = ((latest_cases - prev_cases) / max(1, prev_cases)) * 100.0


# ===========================================================================
# SHARED HEADER
# ===========================================================================
def render_page_header(page_title: str, page_desc: str = ""):
    tier_colors = {"HIGH": "#34D399", "MEDIUM": "#FBBF24", "LOW": "#EF4444"}
    tier_color = tier_colors.get(trust_badge["tier"], "#94A3B8")
    badge_cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(trust_badge["tier"], "badge-low")

    st.markdown(f"""
    <div class="rc-header">
      <div style="flex:1;">
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
          <span class="rc-logo">🛡️ RangeCast</span>
          <span class="rc-district-pill">📍 {selected_district} &nbsp;|&nbsp; {selected_disease}</span>
          <span class="{badge_cls}">● {trust_badge['badge_text']}</span>
        </div>
        <div style="margin-top:6px;">
          <span style="color:#94A3B8;font-size:0.85rem;">{page_title}</span>
          {f'<span style="color:#64748B;font-size:0.8rem;"> — {page_desc}</span>' if page_desc else ''}
        </div>
      </div>
      <div style="text-align:right;white-space:nowrap;">
        <div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Data as of</div>
        <div style="color:#7DD3FC;font-size:0.9rem;font-weight:600;">{latest_date_str}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Alert banners
    if active_shift_alert:
        st.markdown(f"""
        <div class="alert-shift">
          ⚠️ <strong>SHIFT ALERT ACTIVE</strong> — Recent {selected_disease} residuals breached calibrated bands.
          Planning intervals have been automatically widened by 35% to prevent false precision.
        </div>""", unsafe_allow_html=True)

    if is_small:
        st.markdown(f"""
        <div class="alert-small">
          ℹ️ <strong>LOW-VOLUME DISTRICT</strong> — Weekly {selected_disease} counts are subject to Poisson sampling noise.
          Partial pooling with the state trend is active. Use with extra caution.
        </div>""", unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: DASHBOARD (HOME)
# ===========================================================================
if nav_page == "🏠 Dashboard":
    render_page_header("Outbreak Forecast Dashboard", "Calibrated ranges and key indicators")

    # ---- KPI Tiles --------------------------------------------------------
    q10_w1 = int(calibrated_preds[0]["q10"])
    q25_w1 = int(calibrated_preds[0]["q25"])
    q50_w1 = int(calibrated_preds[0]["q50"])
    q75_w1 = int(calibrated_preds[0]["q75"])
    q90_w1 = int(calibrated_preds[0]["q90"])
    cov80 = overall_score["coverage_80"] * 100.0
    rwis = overall_score["relative_wis"]

    nowcaster = OutbreakNowcaster()
    tpr_info = nowcaster.track_positivity_trend(df_district)
    tpr_val = tpr_info["latest_tpr"]

    growth_arrow = "▲" if growth_rate >= 0 else "▼"
    growth_color = "red" if growth_rate > 10 else ("amber" if growth_rate > 0 else "green")
    trust_tier_color = {"HIGH": "green", "MEDIUM": "amber", "LOW": "red"}.get(trust_badge["tier"], "teal")

    st.markdown("""<div class="kpi-grid">""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Current Week Cases</div>
          <div class="kpi-value">{latest_cases:,}</div>
          <div class="kpi-sub">{growth_arrow} <b style="color:{'#F87171' if growth_rate>10 else '#FBBF24' if growth_rate>0 else '#34D399'};">{abs(growth_rate):.1f}%</b> vs prior week</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Next Week Range (80%)</div>
          <div class="kpi-value teal" style="font-size:1.5rem;">{q10_w1:,} – {q90_w1:,}</div>
          <div class="kpi-sub">Median: <b>{q50_w1:,}</b> &nbsp;|&nbsp; 50%: {q25_w1:,}–{q75_w1:,}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        cov_color = "#34D399" if cov80 >= 75 else "#FBBF24" if cov80 >= 60 else "#F87171"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">80% Coverage Record</div>
          <div class="kpi-value" style="color:{cov_color};">{cov80:.1f}%</div>
          <div class="kpi-sub">rWIS: <b>{rwis:.2f}</b> &nbsp;|&nbsp; {overall_score['n_origins']} wk backtest</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        tpr_color = "#F87171" if tpr_info["tpr_alert"] else "#34D399"
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Test Positivity Rate</div>
          <div class="kpi-value" style="color:{tpr_color};">{tpr_val:.1f}%</div>
          <div class="kpi-sub">{tpr_info['tpr_status']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Forecast Chart ---------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📈 Calibrated Range Forecast (50% & 80% Bands)</div>',
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2 = st.columns([4, 1])
    with col_ctrl2:
        show_nowcast = st.toggle("Nowcast Adj.", value=False, help="Apply simulated reporting delay correction")

    df_plot_hist = df_district.tail(24).copy()
    last_dt = pd.to_datetime(df_plot_hist["week_end_date"].max())
    future_dates = [last_dt + pd.Timedelta(weeks=i) for i in range(1, 5)]

    # Find missed weeks in backtest for red dots
    eval_h1 = df_eval[df_eval["horizon_weeks"] == 1].tail(24).copy()
    missed_dates = eval_h1[~eval_h1["covered_80"]]["target_date"].tolist() if "covered_80" in eval_h1.columns else []

    fig = go.Figure()

    # 80% band
    fig.add_trace(go.Scatter(
        x=list(pd.to_datetime(future_dates)) + list(pd.to_datetime(future_dates))[::-1],
        y=[p["q90"] for p in calibrated_preds] + [p["q10"] for p in calibrated_preds][::-1],
        fill="toself",
        fillcolor="rgba(14,165,233,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Calibrated Band",
        hoverinfo="skip",
    ))

    # 50% band
    fig.add_trace(go.Scatter(
        x=list(pd.to_datetime(future_dates)) + list(pd.to_datetime(future_dates))[::-1],
        y=[p["q75"] for p in calibrated_preds] + [p["q25"] for p in calibrated_preds][::-1],
        fill="toself",
        fillcolor="rgba(14,165,233,0.28)",
        line=dict(color="rgba(0,0,0,0)"),
        name="50% Likely Band",
        hoverinfo="skip",
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df_plot_hist["week_end_date"]),
        y=df_plot_hist["cases"],
        name="Historical Cases",
        mode="lines+markers",
        line=dict(color="#94A3B8", width=2),
        marker=dict(size=5, color="#CBD5E1"),
    ))

    # Red dots on missed weeks
    if len(missed_dates) > 0:
        try:
            missed_df = eval_h1[~eval_h1["covered_80"]].tail(24)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(missed_df["target_date"]),
                y=missed_df["actual"],
                name="Missed 80% Range",
                mode="markers",
                marker=dict(size=10, color="#EF4444", symbol="circle", line=dict(color="#FCA5A5", width=2)),
            ))
        except Exception:
            pass

    # Nowcast adjustment
    if show_nowcast:
        try:
            nowcast_df = nowcaster.simulate_reporting_delays(df_district, last_dt)
            adj_df = nowcaster.adjust_for_reporting_delays(nowcast_df).tail(6)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(adj_df["week_end_date"]),
                y=adj_df["nowcast_cases"],
                name="Nowcast [SIMULATED]",
                mode="lines+markers",
                line=dict(color="#FBBF24", width=2, dash="dash"),
                marker=dict(symbol="triangle-up", size=7, color="#FBBF24"),
            ))
        except Exception:
            pass

    # Median forecast line
    forecast_x = [last_dt] + list(pd.to_datetime(future_dates))
    q50_y = [df_plot_hist["cases"].iloc[-1]] + [p["q50"] for p in calibrated_preds]
    base_y = [df_plot_hist["cases"].iloc[-1]] + [p["baseline_q50"] for p in calibrated_preds]

    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=q50_y,
        name="RangeCast Median",
        mode="lines+markers",
        line=dict(color="#0EA5E9", width=3),
        marker=dict(size=8, color="#38BDF8"),
    ))

    # Baseline (persistence)
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=base_y,
        name="Persistence Baseline",
        mode="lines",
        line=dict(color="#EF4444", width=1.8, dash="dot"),
    ))

    # Today marker
    fig.add_vline(
        x=last_dt,
        line=dict(color="rgba(56,189,248,0.4)", width=1.5, dash="dash"),
        annotation_text="Today",
        annotation_font_color="#7DD3FC",
        annotation_font_size=11,
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(10,22,40,0)",
        paper_bgcolor="rgba(10,22,40,0)",
        height=400,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(
            title="Week Ending",
            showgrid=True,
            gridcolor="rgba(51,65,85,0.5)",
            tickfont=dict(size=11, color="#94A3B8"),
        ),
        yaxis=dict(
            title=f"Weekly {selected_disease} Cases",
            showgrid=True,
            gridcolor="rgba(51,65,85,0.5)",
            tickfont=dict(size=11, color="#94A3B8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#94A3B8"),
            bgcolor="rgba(10,22,40,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#162547", font_color="#E2E8F0", bordercolor="#0EA5E9"),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- 4-Week Quantile Table --------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 1–4 Week Planning Quantiles</div>', unsafe_allow_html=True)

    table_data = []
    for p in calibrated_preds:
        h = p["horizon"]
        target_d = future_dates[h - 1].strftime("%d %b %Y")
        table_data.append({
            "Horizon": f"Week {h} ({target_d})",
            "80% Low": f"{int(p['q10']):,}",
            "50% Low": f"{int(p['q25']):,}",
            "Median": f"{int(p['q50']):,}",
            "50% High": f"{int(p['q75']):,}",
            "80% High": f"{int(p['q90']):,}",
            "Baseline": f"{int(p['baseline_q50']):,}",
            "Status": "🚨 Widened" if p.get("alert_widening_applied") else "✅ Normal",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: ACCURACY SCORECARD
# ===========================================================================
elif nav_page == "📊 Accuracy Scorecard":
    render_page_header("Empirical Accuracy Scorecard", "Every metric from real rolling backtests")

    col1, col2, col3, col4 = st.columns(4)
    cov80 = overall_score["coverage_80"] * 100.0
    cov50 = overall_score["coverage_50"] * 100.0
    rwis = overall_score["relative_wis"]

    with col1:
        delta_80 = f"{cov80 - 80.0:+.1f}% vs 80% target"
        st.metric("80% Coverage", f"{cov80:.1f}%", delta_80)
    with col2:
        delta_50 = f"{cov50 - 50.0:+.1f}% vs 50% target"
        st.metric("50% Coverage", f"{cov50:.1f}%", delta_50)
    with col3:
        st.metric("Relative WIS", f"{rwis:.3f}", f"{(1 - rwis)*100:+.0f}% vs baseline", delta_color="inverse")
    with col4:
        st.metric("Backtest Origins", f"{overall_score['n_origins']} wks", f"{overall_score['n_forecasts']} forecasts")

    st.divider()

    # Horizon breakdown
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Performance by Forecast Horizon (1–4 Weeks)</div>', unsafe_allow_html=True)

    horizons_df = scorecard_dict["by_horizon"].copy()
    display_cols = {
        "horizon_weeks": "Horizon (Wks)",
        "n_eval": "Eval Points",
        "coverage_80": "80% Coverage",
        "coverage_50": "50% Coverage",
        "relative_wis": "rWIS",
        "mean_wis_model": "RangeCast WIS",
        "mean_wis_baseline": "Baseline WIS",
        "mae_model": "MAE (RC)",
        "mae_baseline": "MAE (Base)",
        "sharpness_80": "Mean Spread",
    }
    for col in ["coverage_80", "coverage_50"]:
        horizons_df[col] = horizons_df[col].apply(lambda x: f"{x*100:.1f}%")
    for col in ["relative_wis"]:
        horizons_df[col] = horizons_df[col].apply(lambda x: f"{x:.3f}")
    for col in ["mean_wis_model", "mean_wis_baseline", "mae_model", "mae_baseline", "sharpness_80"]:
        if col in horizons_df.columns:
            horizons_df[col] = horizons_df[col].apply(lambda x: f"{x:,.1f}")

    st.dataframe(horizons_df.rename(columns=display_cols), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Trust Badge explanation
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    tier_colors = {"HIGH": "#34D399", "MEDIUM": "#FBBF24", "LOW": "#EF4444"}
    badge_cls_map = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}
    tc = tier_colors.get(trust_badge["tier"], "#94A3B8")
    bc = badge_cls_map.get(trust_badge["tier"], "badge-low")

    st.markdown(f"""
    <div class="section-title">🏆 Trust Badge Audit</div>
    <div style="margin-bottom:16px;">
      <span class="{bc}">● {trust_badge['badge_text']}</span>
      <span style="color:#94A3B8;font-size:0.85rem;margin-left:12px;">{trust_badge['summary']}</span>
    </div>
    """, unsafe_allow_html=True)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("#### ✅ Positive Signals")
        for sig in trust_badge.get("positive_signals", []):
            st.markdown(f"- {sig}")
        if not trust_badge.get("positive_signals"):
            st.markdown("- None recorded")

    with col_neg:
        st.markdown("#### ⚠️ Warnings")
        for w in trust_badge.get("reasons", []):
            st.markdown(f"- {w}")
        if not trust_badge.get("reasons"):
            st.markdown("- No active warnings")

    st.markdown("</div>", unsafe_allow_html=True)

    # Coverage over time chart
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Coverage Trend (Last 12 Weeks, Horizon 1)</div>', unsafe_allow_html=True)

    eval_h1 = df_eval[df_eval["horizon_weeks"] == 1].tail(12).copy()
    if len(eval_h1) > 0 and "covered_80" in eval_h1.columns:
        cov_fig = go.Figure()
        cov_fig.add_trace(go.Scatter(
            x=pd.to_datetime(eval_h1["target_date"]),
            y=eval_h1["covered_80"].astype(int),
            mode="markers+lines",
            name="Covered 80%",
            line=dict(color="#0EA5E9", width=2),
            marker=dict(
                size=10,
                color=["#34D399" if v else "#EF4444" for v in eval_h1["covered_80"]],
                line=dict(color="#1D3461", width=2),
            ),
        ))
        cov_fig.add_hline(y=0.8, line=dict(color="#FBBF24", dash="dash", width=1.5),
                          annotation_text="80% target", annotation_font_color="#FBBF24")
        cov_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(10,22,40,0)",
            paper_bgcolor="rgba(10,22,40,0)",
            height=220,
            margin=dict(l=40, r=20, t=10, b=40),
            yaxis=dict(title="Covered (1=Yes)", tickvals=[0, 1], ticktext=["Miss", "Hit"],
                       range=[-0.2, 1.4], showgrid=True, gridcolor="rgba(51,65,85,0.4)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.4)"),
            showlegend=False,
        )
        st.plotly_chart(cov_fig, use_container_width=True)
    else:
        st.info("Insufficient backtest data for coverage trend chart.")
    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 3: RESOURCE PLANNER
# ===========================================================================
elif nav_page == "🏥 Resource Planner":
    render_page_header("Resource Planner", "Newsvendor Critical Fractile procurement engine")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">⚖️ Risk Preference (Newsvendor Fractile)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "$$q = \\frac{C_u}{C_u + C_o}$$  \n"
        "*Higher q → conservative stocking; lower q → budget-efficient.*",
    )

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_choice = None
    if preset_col1.button("🚨 Cautious (q=85%)", help="Shortage-sensitive; prioritise patient safety"):
        preset_choice = 0.85
    if preset_col2.button("⚖️ Balanced (q=75%)", help="Standard district planning"):
        preset_choice = 0.75
    if preset_col3.button("📉 Lean (q=50%)", help="Budget-constrained; median planning"):
        preset_choice = 0.50

    risk_fractile = st.slider(
        "Planning Fractile (q):",
        min_value=0.10, max_value=0.95,
        value=preset_choice if preset_choice else 0.80,
        step=0.05,
        format="%.0f%%",
    )
    st.markdown(f"**Active: {risk_fractile:.0%} percentile** — planning against the {risk_fractile:.0%} case projection")
    st.markdown("</div>", unsafe_allow_html=True)

    # Editable assumptions
    default_assump = get_default_disease_assumptions(selected_disease)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    with st.expander(f"⚙️ Edit Clinical Assumptions for {selected_disease} (Placeholders — not official norms)"):
        st.markdown(
            "<small style='color:#64748B;'>These are editable planning parameters, NOT statutory clinical norms.</small>",
            unsafe_allow_html=True,
        )
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            hosp_rate = st.number_input("Hospitalisation Rate (%)", value=default_assump.hospitalization_rate * 100.0,
                                        min_value=0.5, max_value=50.0, step=0.5) / 100.0
            icu_rate = st.number_input("ICU / Severe Fraction (%)", value=default_assump.icu_fraction * 100.0,
                                       min_value=1.0, max_value=50.0, step=1.0) / 100.0
        with col_a2:
            alos = st.number_input("Avg Length of Stay (days)", value=default_assump.avg_length_of_stay_days,
                                   min_value=1.0, max_value=30.0, step=0.5)
            tests_per_case = st.number_input("Tests per Confirmed Case", value=default_assump.tests_per_confirmed_case,
                                             min_value=1.0, max_value=60.0, step=1.0)
        with col_a3:
            lead_time = st.slider("Supplier Lead Time (Weeks)", min_value=1, max_value=4,
                                  value=default_assump.supplier_lead_time_weeks)
            cur_beds = st.number_input("Current Operational Beds", value=200, min_value=0, step=20)
            cur_kits = st.number_input("Current Diagnostic Kits in Stock", value=2000, min_value=0, step=500)

    custom_assump = PlannerAssumptions(
        disease=selected_disease,
        hospitalization_rate=hosp_rate,
        icu_fraction=icu_rate,
        avg_length_of_stay_days=alos,
        tests_per_confirmed_case=tests_per_case,
        supplier_lead_time_weeks=lead_time,
        platelets_per_icu_patient_day=default_assump.platelets_per_icu_patient_day,
        iv_fluid_bottles_per_admission=default_assump.iv_fluid_bottles_per_admission,
        act_courses_per_case=default_assump.act_courses_per_case,
        tamiflu_courses_per_case=default_assump.tamiflu_courses_per_case,
        ors_sachets_per_case=default_assump.ors_sachets_per_case,
        chlorine_tabs_per_case=default_assump.chlorine_tabs_per_case,
    )

    plan = calculate_resource_plan(
        quantiles_by_horizon=calibrated_preds,
        risk_fractile=risk_fractile,
        assumptions=custom_assump,
        current_active_beds=cur_beds,
        current_kits_stock=cur_kits,
    )
    summary = plan["summary"]

    st.markdown("</div>", unsafe_allow_html=True)

    # Resource cards
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">🏥 Procurement Requirements — {selected_disease} ({lead_time}-Week Lead Time)</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Peak Inpatient Beds</div>
          <div class="kpi-value teal">{summary['peak_beds_required']:,}</div>
          <div class="kpi-sub">ICU/HDU: <b>{summary['peak_icu_required']:,}</b> | Net mobilise: <b>{summary['net_beds_to_mobilize']:,}</b></div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Diagnostic Kits to Order</div>
          <div class="kpi-value green">{summary['lead_time_kits_needed']:,}</div>
          <div class="kpi-sub">Net order: <b>{summary['net_kits_to_order']:,}</b></div>
        </div>""", unsafe_allow_html=True)

    if selected_disease == "Dengue":
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Platelet Units (RDP/SDP)</div>
              <div class="kpi-value red">{summary['lead_time_platelets']:,}</div>
              <div class="kpi-sub">Severe thrombocytopenia</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">IV Fluids (RL/NS 500ml)</div>
              <div class="kpi-value teal">{summary['lead_time_iv_fluids']:,}</div>
              <div class="kpi-sub">Plasma leakage management</div>
            </div>""", unsafe_allow_html=True)
    elif selected_disease == "Malaria":
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">ACT Blister Packs</div>
              <div class="kpi-value amber">{summary['lead_time_act_courses']:,}</div>
              <div class="kpi-sub">Artemisinin combination courses</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Rapid Diagnostic Tests</div>
              <div class="kpi-value green">{summary['lead_time_kits_needed']:,}</div>
              <div class="kpi-sub">Antigen detection kits</div>
            </div>""", unsafe_allow_html=True)
    elif selected_disease == "Acute Diarrheal Disease":
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">ORS Sachets</div>
              <div class="kpi-value amber">{summary['lead_time_ors_sachets']:,}</div>
              <div class="kpi-sub">Oral rehydration packets</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Chlorine Tablets</div>
              <div class="kpi-value teal">{summary['lead_time_chlorine_tabs']:,}</div>
              <div class="kpi-sub">Water purification tablets</div>
            </div>""", unsafe_allow_html=True)
    elif selected_disease == "Influenza (H1N1)":
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Tamiflu Courses</div>
              <div class="kpi-value" style="color:#A78BFA;">{summary['lead_time_tamiflu_courses']:,}</div>
              <div class="kpi-sub">Oseltamivir 75mg courses</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Peak Daily O₂ Cylinders</div>
              <div class="kpi-value teal">{summary['peak_daily_o2_cylinders']:,}</div>
              <div class="kpi-sub">D-type cylinders/day</div>
            </div>""", unsafe_allow_html=True)
    else:  # COVID-19
        with r3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Peak Daily O₂ Cylinders</div>
              <div class="kpi-value teal">{summary['peak_daily_o2_cylinders']:,}</div>
              <div class="kpi-sub">D-type (7,000 L) cylinders/day</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Lead-Time O₂ Order</div>
              <div class="kpi-value" style="color:#A78BFA;">{summary['lead_time_cylinders_needed']:,}</div>
              <div class="kpi-sub">Total cylinders to refill/order</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        "⚠️ All assumptions above are editable placeholder parameters, NOT official clinical norms or government directives."
    )


# ===========================================================================
# PAGE 4: COMPARE FORECASTS
# ===========================================================================
elif nav_page == "🔄 Compare Forecasts":
    render_page_header("Compare Forecasts", "Score an external model against the same actuals")

    eval_h1 = df_eval[df_eval["horizon_weeks"] == 1].copy().reset_index(drop=True)

    comp_mode = st.radio(
        "Comparison source:",
        options=[
            "Built-in: Point Extrapolation (simulated single-line forecast)",
            "Built-in: Pure Naive Persistence Baseline",
            "Upload custom forecast CSV (date, predicted_cases)",
        ],
    )

    if "Upload custom" in comp_mode:
        up_file = st.file_uploader("Upload CSV (columns: date, predicted_cases)", type=["csv"])
        if up_file is not None:
            up_df = pd.read_csv(up_file)
            st.success(f"Loaded {len(up_df)} external forecast rows.")
            pred_col = "predicted_cases" if "predicted_cases" in up_df.columns else up_df.columns[1]
            eval_h1["external_pred"] = np.interp(
                pd.to_datetime(eval_h1["target_date"]).astype("int64"),
                pd.to_datetime(up_df.iloc[:, 0]).astype("int64"),
                up_df[pred_col].astype(float),
            )
        else:
            st.info("Upload a CSV to compare. Showing example with persistence baseline in the meantime.")
            eval_h1["external_pred"] = eval_h1["baseline_q50"]
    elif "Naive Persistence" in comp_mode:
        eval_h1["external_pred"] = eval_h1["baseline_q50"]
    else:
        eval_h1["external_pred"] = (
            eval_h1["baseline_q50"] * 1.08 +
            np.sin(np.arange(len(eval_h1))) * 0.1 * eval_h1["baseline_q50"]
        )

    rc_mae = np.mean(np.abs(eval_h1["actual"] - eval_h1["q50"]))
    ext_mae = np.mean(np.abs(eval_h1["actual"] - eval_h1["external_pred"]))
    rc_cov80 = np.mean(eval_h1["covered_80"]) * 100.0

    ext_l = eval_h1["external_pred"] * 0.85
    ext_u = eval_h1["external_pred"] * 1.15
    ext_cov = calculate_coverage(eval_h1["actual"], ext_l, ext_u) * 100.0
    ext_is = np.mean(interval_score(eval_h1["actual"].values, ext_l.values, ext_u.values, alpha=0.20))
    rc_is = np.mean(eval_h1["is_80"]) if "is_80" in eval_h1.columns else float("nan")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Head-to-Head Accuracy Comparison</div>', unsafe_allow_html=True)

    comp_df = pd.DataFrame([
        {
            "Model": "🛡️ RangeCast Calibrated Ensemble",
            "MAE (cases/wk)": f"{rc_mae:,.1f}",
            "80% Coverage": f"{rc_cov80:.1f}%",
            "80% Interval Score": f"{rc_is:,.1f}" if not np.isnan(rc_is) else "N/A",
            "Method": "Adaptive Conformal + 3-Model Ensemble",
        },
        {
            "Model": "External / Comparison Forecast",
            "MAE (cases/wk)": f"{ext_mae:,.1f}",
            "80% Coverage": f"{ext_cov:.1f}%",
            "80% Interval Score": f"{ext_is:,.1f}",
            "Method": comp_mode.split(":")[0].strip(),
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if not np.isnan(rc_is) and ext_is > 0:
        adv = (1 - rc_is / ext_is) * 100
        if adv > 0:
            st.markdown(
                f'<div class="alert-success">✅ RangeCast is <b>{adv:.1f}%</b> sharper (lower interval score) than the external forecast.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-small">⚠️ External forecast has a lower interval score by {-adv:.1f}%.</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 5: BRIEFINGS & EXPORT
# ===========================================================================
elif nav_page == "📢 Briefings & Export":
    render_page_header("Outbreak Briefings & Export", "Plain-language advisories for DHO and CMO")

    plan_data = calculate_resource_plan(
        calibrated_preds, 0.80, get_default_disease_assumptions(selected_disease)
    )["summary"]

    eng_text = generate_english_briefing(
        district=selected_district, as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0], quantiles_h2=calibrated_preds[1],
        plan_summary=plan_data, scorecard=overall_score,
        risk_fractile=0.80, disease=selected_disease,
        shift_alert=active_shift_alert, is_small_district=is_small,
    )
    tam_text = generate_tamil_briefing(
        district=selected_district, as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0], plan_summary=plan_data,
        scorecard=overall_score, risk_fractile=0.80,
        disease=selected_disease, shift_alert=active_shift_alert,
    )
    hin_text = generate_hindi_briefing(
        district=selected_district, as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0], plan_summary=plan_data,
        scorecard=overall_score, risk_fractile=0.80,
        disease=selected_disease, shift_alert=active_shift_alert,
    )
    wa_card = generate_whatsapp_card(
        district=selected_district, as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0], plan_summary=plan_data,
        scorecard=overall_score, risk_fractile=0.80,
        disease=selected_disease, shift_alert=active_shift_alert,
    )

    tab_eng, tab_tam, tab_hin, tab_wa = st.tabs([
        "🇬🇧 English Advisory",
        "🇮🇳 Tamil / தமிழ்",
        "🇮🇳 Hindi / हिन्दी",
        "📱 WhatsApp Card",
    ])

    with tab_eng:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.code(eng_text, language="text")
        st.download_button("⬇️ Download English Briefing", eng_text, file_name="rangecast_en.txt")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_tam:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="alert-small">⚠️ Machine-assisted wording. Native-speaker review required before use in official communications.</div>',
            unsafe_allow_html=True,
        )
        st.code(tam_text, language="text")
        st.download_button("⬇️ Download Tamil Briefing", tam_text, file_name="rangecast_ta.txt")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_hin:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="alert-small">⚠️ Machine-assisted wording. Native-speaker review required before use in official communications.</div>',
            unsafe_allow_html=True,
        )
        st.code(hin_text, language="text")
        st.download_button("⬇️ Download Hindi Briefing", hin_text, file_name="rangecast_hi.txt")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_wa:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📱 Ready for CMO / War Room WhatsApp Groups")
        st.code(wa_card, language="markdown")
        st.download_button("⬇️ Copy WhatsApp Card", wa_card, file_name="rangecast_whatsapp.txt")
        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 6: ASK RANGECAST AI
# ===========================================================================
elif nav_page == "🤖 Ask RangeCast AI":
    render_page_header("Ask RangeCast AI", "Groq-powered Q&A — all numbers verified against computed data")

    if not _GROQ_AVAILABLE:
        st.warning(
            "⚠️ Groq AI module is not available. "
            "Install it with `pip install groq` and set GROQ_API_KEY in your .env file."
        )
    else:
        # Detect if GROQ_API_KEY is configured
        _api_key = os.environ.get("GROQ_API_KEY", "")
        _api_ok = bool(_api_key) and not _api_key.startswith("gsk_xxx")

        if not _api_ok:
            st.warning(
                "⚠️ GROQ_API_KEY is not configured. "
                "AI responses will use the fallback template engine. "
                "Set your key in .env to enable full AI responses."
            )

    # Build context for AI
    plan_data = calculate_resource_plan(
        calibrated_preds, 0.80, get_default_disease_assumptions(selected_disease)
    )["summary"]

    ai_context = {
        "district": selected_district,
        "disease": selected_disease,
        "latest_reported_cases": latest_cases,
        "q10_week1": int(calibrated_preds[0]["q10"]),
        "q25_week1": int(calibrated_preds[0]["q25"]),
        "q50_week1": int(calibrated_preds[0]["q50"]),
        "q75_week1": int(calibrated_preds[0]["q75"]),
        "q90_week1": int(calibrated_preds[0]["q90"]),
        "q50_week2": int(calibrated_preds[1]["q50"]) if len(calibrated_preds) > 1 else None,
        "coverage_80_pct": round(overall_score["coverage_80"] * 100, 1),
        "coverage_50_pct": round(overall_score["coverage_50"] * 100, 1),
        "relative_wis": round(overall_score["relative_wis"], 3),
        "trust_tier": trust_badge["tier"],
        "n_backtest_weeks": overall_score["n_origins"],
        "shift_alert": active_shift_alert,
        "is_small_district": is_small,
        "peak_beds_required": plan_data.get("peak_beds_required", 0),
        "peak_icu_required": plan_data.get("peak_icu_required", 0),
        "net_beds_to_mobilize": plan_data.get("net_beds_to_mobilize", 0),
        "peak_daily_o2_cylinders": plan_data.get("peak_daily_o2_cylinders", 0),
        "lead_time_cylinders_needed": plan_data.get("lead_time_cylinders_needed", 0),
        "lead_time_kits_needed": plan_data.get("lead_time_kits_needed", 0),
    }

    # Language-keyed summary tab
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 AI-Generated Outbreak Summary</div>', unsafe_allow_html=True)

    sum_lang_labels = {"en": "English", "ta": "Tamil", "hi": "Hindi"}
    sum_col1, sum_col2 = st.columns([3, 1])
    with sum_col2:
        sum_lang = st.selectbox("Language:", options=["en", "ta", "hi"],
                                format_func=lambda x: sum_lang_labels[x], key="sum_lang")

    if st.button("🔄 Generate Summary", key="btn_summary"):
        with st.spinner("Generating summary…"):
            if _GROQ_AVAILABLE:
                res = _gc.generate_summary(ai_context, lang=sum_lang)
            else:
                res = {"answer": "AI not available. Install groq and set GROQ_API_KEY.", "verified": False, "lang": sum_lang, "machine_assisted": False}

        verified = res.get("verified", False)
        badge_html = (
            '<span class="verified-badge">✓ Numbers Verified</span>'
            if verified
            else '<span class="unverified-badge">⚠ Template Fallback</span>'
        )
        st.markdown(f"<div>{badge_html}</div>", unsafe_allow_html=True)

        if res.get("machine_assisted"):
            st.markdown(
                '<div class="alert-small">⚠️ Machine-assisted wording. Native-speaker review required before official use.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="chat-bubble-ai">{res["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Q&A
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)

    # Suggested question chips
    st.markdown("**Suggested questions (click to populate):**")
    suggested = _gc.SUGGESTED_QUESTIONS if _GROQ_AVAILABLE else [
        "Why are the prediction ranges wider this week?",
        "How many hospital beds do I need to prepare?",
        "Is this forecast reliable enough to make procurement decisions?",
    ]

    chip_html = "".join(
        f'<span class="chip" onclick="void(0)">💬 {q}</span>' for q in suggested[:4]
    )
    st.markdown(chip_html, unsafe_allow_html=True)
    st.caption("(Type a question below or select from above)")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input(
        "Your question:",
        placeholder="e.g. Why are the ranges wider this week?",
        key="user_q",
        label_visibility="collapsed",
    )

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_btn = st.button("📤 Ask RangeCast", key="btn_ask", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear Chat", key="btn_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if ask_btn and user_question.strip():
        with st.spinner("Thinking…"):
            if _GROQ_AVAILABLE:
                ai_result = _gc.ask_question(user_question.strip(), ai_context)
            else:
                ai_result = {
                    "answer": "AI not available. Install groq and set GROQ_API_KEY.",
                    "verified": False,
                    "question": user_question.strip(),
                }
        st.session_state.chat_history.append(ai_result)

    # Render chat history (newest first)
    for entry in reversed(st.session_state.chat_history):
        verified = entry.get("verified", False)
        badge_html = (
            '<span class="verified-badge">✓ Verified</span>'
            if verified
            else '<span class="unverified-badge">⚠ Template</span>'
        )
        st.markdown(f'<div class="chat-bubble-user">🧑‍💼 {entry["question"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chat-bubble-ai">🤖 {entry["answer"]}{badge_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "⚠️ RangeCast AI answers only with computed district data. "
        "It does not provide clinical medical advice or diagnose individual patients."
    )


# ===========================================================================
# PAGE 7: SURGE REPLAY DEMO
# ===========================================================================
elif nav_page == "⚡ Surge Replay Demo":
    render_page_header("Historical Surge Replay", "Wave 2 COVID-19 (April–May 2021)")

    st.markdown(
        "This demo shows how the conventional single-line forecast **failed** during the Wave 2 surge, "
        "while RangeCast's calibrated range **caught the true trajectory** and triggered shift alerts."
    )

    surge_district = st.selectbox("Select Surge District:", options=["Chennai", "Coimbatore", "Madurai"], index=0)

    with st.spinner("Replaying historical surge…"):
        df_surge = run_surge_replay(district=surge_district, verbose=False)

    sl_mae = np.mean(np.abs(df_surge["actual"] - df_surge["single_line_pred"]))
    rc_mae = np.mean(np.abs(df_surge["actual"] - df_surge["q50"]))
    rc_cov80 = np.mean(df_surge["rangecast_caught"]) * 100.0

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Single-Line Avg Error", f"{int(sl_mae):,} cases/wk")
    with sc2:
        delta = f"-{(1 - rc_mae/sl_mae)*100:.1f}% error reduction"
        st.metric("RangeCast Median Error", f"{int(rc_mae):,} cases/wk", delta=delta, delta_color="inverse")
    with sc3:
        st.metric("RangeCast 80% Coverage (Surge)", f"{rc_cov80:.1f}%", "Caught during surge")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    fig_surge = go.Figure()

    fig_surge.add_trace(go.Scatter(
        x=df_surge["target_date_str"], y=df_surge["q90"],
        name="RangeCast 80% Upper",
        line=dict(color="#0EA5E9", width=1.5, dash="dash"),
    ))
    fig_surge.add_trace(go.Scatter(
        x=df_surge["target_date_str"], y=df_surge["q10"],
        name="RangeCast 80% Lower",
        line=dict(color="#0EA5E9", width=1.5, dash="dash"),
        fill="tonexty", fillcolor="rgba(14,165,233,0.15)",
    ))
    fig_surge.add_trace(go.Scatter(
        x=df_surge["target_date_str"], y=df_surge["actual"],
        name="Actual Cases", mode="lines+markers",
        line=dict(color="#F8FAFC", width=3),
        marker=dict(size=8, color="#F8FAFC"),
    ))
    fig_surge.add_trace(go.Scatter(
        x=df_surge["target_date_str"], y=df_surge["single_line_pred"],
        name="Single-Line Forecast (failed)", mode="lines+markers",
        line=dict(color="#EF4444", width=2.5, dash="dot"),
        marker=dict(symbol="circle-open", size=8, color="#EF4444"),
    ))

    fig_surge.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(10,22,40,0)",
        paper_bgcolor="rgba(10,22,40,0)",
        height=380,
        margin=dict(l=50, r=20, t=20, b=50),
        xaxis=dict(title="Week Ending", showgrid=True, gridcolor="rgba(51,65,85,0.4)"),
        yaxis=dict(title="Weekly Cases", showgrid=True, gridcolor="rgba(51,65,85,0.4)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(10,22,40,0)"),
    )
    st.plotly_chart(fig_surge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 Week-by-Week Replay Log</div>', unsafe_allow_html=True)

    log_rows = []
    for _, r in df_surge.iterrows():
        log_rows.append({
            "Week End": r["target_date_str"],
            "Actual": f"{int(r['actual']):,}",
            "Single-Line": f"{int(r['single_line_pred']):,}",
            "RangeCast [80%]": f"[{int(r['q10']):,} – {int(r['q90']):,}]",
            "RangeCast Caught?": "✅ CAUGHT" if r["rangecast_caught"] else "❌ MISSED",
            "Shift Alert": "🚨 ALERT" if r["shift_alert"] else "Normal",
        })
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
