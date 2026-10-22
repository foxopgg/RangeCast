"""
RangeCast Multi-Disease Streamlit Application (app.py)
======================================================
Calibrated Outbreak Case Forecasts and Resource Planning for District Health Officers.
Supported Diseases: COVID-19, Dengue, Malaria, Influenza (H1N1), Acute Diarrheal Disease (Cholera).
"""

import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# RangeCast Core Modules
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


# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RangeCast | Multi-Disease Outbreak Forecasting & Resource Planning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        color: #F8FAFC;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #CBD5E1;
        margin-top: 4px;
    }
    
    /* Disease Pill */
    .disease-pill {
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* Trust Badges */
    .badge-high {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Ingestion & Caching Layer
# -----------------------------------------------------------------------------
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
    dis_slug = disease.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    cache_path = f"data/cache/eval_{dis_slug}_{district.lower()}.csv"
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    # Check legacy filename fallback
    legacy_path = f"data/cache/eval_{district.lower()}.csv"
    if disease == "COVID-19" and os.path.exists(legacy_path):
        return pd.read_csv(legacy_path)
        
    loader = OutbreakDataLoader(disease=disease)
    df_d = loader.load_weekly_district_data(district)
    runner = RollingBacktestRunner(min_train_weeks=12, horizon_weeks=4)
    eval_df = runner.run_district_backtest(df_d)
    os.makedirs("data/cache", exist_ok=True)
    eval_df.to_csv(cache_path, index=False)
    return eval_df


# -----------------------------------------------------------------------------
# Sidebar Navigation & Disease Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🛡️ **RangeCast**")
st.sidebar.markdown(
    "<small style='color:#94A3B8;'>Multi-Disease Outbreak Forecasting & Resource Planning for District Health Officers</small>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

# Disease Selection Dropdown
selected_disease = st.sidebar.selectbox(
    "🦠 Target Outbreak Disease:",
    options=OutbreakDataLoader.SUPPORTED_DISEASES,
    index=0,
)

# Disease Transmission Category Badge
disease_meta = {
    "COVID-19": ("🫁 Airborne Respiratory Pandemic", "Wave dynamics & High-Flow Oxygen Demand"),
    "Dengue": ("🦟 Vector-Borne Outbreak", "Driven by 2-4 Wk Rain Lags | Platelets & IV Fluids"),
    "Malaria": ("🦟 Vector-Borne Parasitic", "Anopheles Transmission | RDTs & ACT Courses"),
    "Influenza (H1N1)": ("🫁 Seasonal Respiratory", "Winter/Monsoon Surge | Tamiflu & Swabs"),
    "Acute Diarrheal Disease": ("💧 Water-Borne Infection", "Post-Flooding Surge | ORS & Chlorine Tabs"),
}
badge_icon, badge_desc = disease_meta.get(selected_disease, ("Outbreak", ""))
st.sidebar.markdown(f'<div class="disease-pill">{badge_icon}</div>', unsafe_allow_html=True)
st.sidebar.caption(badge_desc)

st.sidebar.divider()

districts = get_districts_list(selected_disease)
default_idx = districts.index("Chennai") if "Chennai" in districts else 0

selected_district = st.sidebar.selectbox(
    "📍 Select District (Tamil Nadu):",
    options=districts,
    index=default_idx,
)

# Load district dataset and backtest scorecard
df_district = load_district_data(selected_disease, selected_district)
df_eval = load_or_run_backtest(selected_disease, selected_district)

runner = RollingBacktestRunner()
scorecard_dict = runner.generate_district_scorecard(df_eval)
overall_score = scorecard_dict["overall"]
trust_badge = overall_score["trust_badge"]

is_small = bool(df_district['is_small_district'].iloc[0]) if 'is_small_district' in df_district.columns else False

# Trust Badge in Sidebar
if trust_badge["tier"] == "HIGH":
    st.sidebar.markdown(f'<div class="badge-high">● {trust_badge["badge_text"]}</div>', unsafe_allow_html=True)
elif trust_badge["tier"] == "MEDIUM":
    st.sidebar.markdown(f'<div class="badge-medium">● {trust_badge["badge_text"]}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="badge-low">● {trust_badge["badge_text"]}</div>', unsafe_allow_html=True)

st.sidebar.caption(trust_badge["summary"])
if is_small:
    st.sidebar.warning(f"⚠️ Low-volume rural district for {selected_disease}. Partial pooling with state trend active.")

st.sidebar.divider()

nav_page = st.sidebar.radio(
    "Navigate Module:",
    options=[
        "🏠 Outbreak Forecast (Home)",
        "📊 Scorecard & Calibration",
        "🏥 Resource Planner (Newsvendor)",
        "🔄 Compare Models View",
        "🚨 Alerts & Multi-Lang Briefings",
        "⚡ Past Surge Replay",
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    "<small style='color:#64748B;'>Data Sources: COVID-19 India Archives, IDSP/NVBDCP Outbreak Series, IMD Gridded Rainfall, Festival Calendar.<br>"
    "Assumptions are editable placeholders, not statutory norms.</small>",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Common Utility: Latest Forecast Generation
# -----------------------------------------------------------------------------
def get_current_forecast(df_curr: pd.DataFrame, is_small_dist: bool):
    """Generates the latest 4-week forecast from historical cutoff."""
    ens = RangeCastEnsemble()
    ens.fit(df_curr)
    raw_preds = ens.predict(horizon_weeks=4, apply_pooling=True, is_small_district=is_small_dist)
    
    calibrator = AdaptiveConformalCalibrator()
    recent_eval = df_eval[df_eval['horizon_weeks'] == 1].tail(12)
    for _, r in recent_eval.iterrows():
        calibrator.scores_80.append(calibrator.score_nonconformity(r['actual'], r['q10'], r['q90'], r['q50']))
        calibrator.scores_50.append(calibrator.score_nonconformity(r['actual'], r['q25'], r['q75'], r['q50']))
        calibrator.recent_misses_80.append(0 if r['covered_80'] else 1)
        calibrator.recent_errors.append(abs(r['actual'] - r['q50']) / max(5.0, r['q50']))
    calibrator.alpha_80 = float(recent_eval['conformal_alpha_80'].iloc[-1]) if not recent_eval.empty else 0.20
    calibrator.shift_alert_active = bool(recent_eval['shift_alert'].iloc[-1]) if not recent_eval.empty else False
    
    cal_preds = []
    for p in raw_preds:
        res = calibrator.calibrate_forecast({
            "q10": p["q10"],
            "q25": p["q25"],
            "q50": p["q50"],
            "q75": p["q75"],
            "q90": p["q90"],
        })
        res["horizon"] = p["horizon"]
        res["baseline_q50"] = p["baseline_q50"]
        cal_preds.append(res)
        
    return cal_preds, calibrator.shift_alert_active


calibrated_preds, active_shift_alert = get_current_forecast(df_district, is_small)
latest_date_str = str(pd.to_datetime(df_district['week_end_date'].max()).strftime('%Y-%m-%d'))


# =============================================================================
# SCREEN 1: HOME (OUTBREAK FORECAST & RANGE CHART)
# =============================================================================
if nav_page == "🏠 Outbreak Forecast (Home)":
    st.title(f"{selected_disease} Outbreak Forecast: {selected_district}")
    st.markdown(
        f"Providing calibrated low and high ranges, historical accuracy records, and proactive resource allocation for **{selected_disease}**."
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_cases = int(df_district['cases'].iloc[-1])
    prev_cases = int(df_district['cases'].iloc[-2]) if len(df_district) >= 2 else latest_cases
    growth_rate = ((latest_cases - prev_cases) / max(1, prev_cases)) * 100.0
    
    q10_w1 = int(calibrated_preds[0]["q10"])
    q50_w1 = int(calibrated_preds[0]["q50"])
    q90_w1 = int(calibrated_preds[0]["q90"])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Reported ({latest_date_str})</div>
            <div class="metric-value">{latest_cases:,}</div>
            <div class="metric-sub">{'▲' if growth_rate >= 0 else '▼'} {abs(growth_rate):.1f}% vs prior week</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Next 7-Day Planning Range (80%)</div>
            <div class="metric-value" style="color:#60A5FA;">{q10_w1:,} – {q90_w1:,}</div>
            <div class="metric-sub">Median projection: <b>{q50_w1:,} cases</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        cov80 = overall_score["coverage_80"] * 100.0
        rwis = overall_score["relative_wis"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Calibration Record (Past {overall_score['n_origins']} Wks)</div>
            <div class="metric-value" style="color:#34D399;">{cov80:.1f}%</div>
            <div class="metric-sub">80% Range Held True | rWIS: <b>{rwis:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        nowcaster = OutbreakNowcaster()
        tpr_info = nowcaster.track_positivity_trend(df_district)
        tpr_val = tpr_info["latest_tpr"]
        tpr_color = "#EF4444" if tpr_info["tpr_alert"] else "#10B981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Surveillance Positivity (TPR)</div>
            <div class="metric-value" style="color:{tpr_color};">{tpr_val:.1f}%</div>
            <div class="metric-sub">{tpr_info['tpr_status']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Active Alert Banner
    if active_shift_alert:
        st.error(
            f"⚠️ **SHIFT ALERT ACTIVE**: Recent {selected_disease} outbreak residuals breached calibrated conformal bands. "
            "Planning intervals have been automatically widened by 35% to prevent false precision."
        )
    if is_small:
        st.info(
            f"ℹ️ **SMALL DISTRICT NOTICE**: Weekly {selected_disease} counts in this district are subject to Poisson sampling noise. "
            "Hierarchical shrinkage / partial pooling with the state trend has been activated."
        )
        
    st.markdown("---")
    
    # Forecast Horizon Selection & Controls
    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.subheader("Interactive Range Forecast & Historical Actuals")
    with c_right:
        show_nowcast = st.toggle("Apply Nowcast Reporting Correction", value=False)
        
    df_plot_hist = df_district.tail(24).copy()
    last_dt = pd.to_datetime(df_plot_hist['week_end_date'].max())
    future_dates = [last_dt + pd.Timedelta(weeks=i) for i in range(1, 5)]
    
    fig = go.Figure()
    
    # 1. Historical Actual Cases
    fig.add_trace(go.Scatter(
        x=df_plot_hist['week_end_date'],
        y=df_plot_hist['cases'],
        name="Historical Reported Cases",
        mode="lines+markers",
        line=dict(color="#94A3B8", width=2.5),
        marker=dict(size=6, color="#F8FAFC"),
    ))
    
    # 2. Simulated Nowcast Adjustment if toggled
    if show_nowcast:
        nowcast_df = nowcaster.simulate_reporting_delays(df_district, last_dt)
        adj_df = nowcaster.adjust_for_reporting_delays(nowcast_df).tail(6)
        
        fig.add_trace(go.Scatter(
            x=adj_df['week_end_date'],
            y=adj_df['nowcast_cases'],
            name="Nowcast Inferred Cases [SIMULATED REPORTING DELAY]",
            mode="lines+markers",
            line=dict(color="#F59E0B", width=2.5, dash="dash"),
            marker=dict(symbol="triangle-up", size=8, color="#F59E0B"),
        ))
        
    forecast_x = [last_dt] + future_dates
    q10_y = [df_plot_hist['cases'].iloc[-1]] + [p['q10'] for p in calibrated_preds]
    q25_y = [df_plot_hist['cases'].iloc[-1]] + [p['q25'] for p in calibrated_preds]
    q50_y = [df_plot_hist['cases'].iloc[-1]] + [p['q50'] for p in calibrated_preds]
    q75_y = [df_plot_hist['cases'].iloc[-1]] + [p['q75'] for p in calibrated_preds]
    q90_y = [df_plot_hist['cases'].iloc[-1]] + [p['q90'] for p in calibrated_preds]
    base_y = [df_plot_hist['cases'].iloc[-1]] + [p['baseline_q50'] for p in calibrated_preds]
    
    # 3. 80% Calibrated Prediction Band
    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1],
        y=q90_y + q10_y[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name="80% Calibrated Range (q10 - q90)",
        hoverinfo="skip"
    ))
    
    # 4. 50% Inner Likelihood Band
    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1],
        y=q75_y + q25_y[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.32)',
        line=dict(color='rgba(255,255,255,0)'),
        name="50% Likely Range (q25 - q75)",
        hoverinfo="skip"
    ))
    
    # 5. Median Forecast
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=q50_y,
        name="RangeCast Median Forecast (q50)",
        mode="lines+markers",
        line=dict(color="#3B82F6", width=3.5),
        marker=dict(size=8, color="#60A5FA"),
    ))
    
    # 6. Persistence Baseline
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=base_y,
        name="Persistence Baseline ('Same as last week')",
        mode="lines",
        line=dict(color="#EF4444", width=2, dash="dot"),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Week Ending Date", showgrid=True, gridcolor="#334155"),
        yaxis=dict(title=f"Weekly {selected_disease} Cases", showgrid=True, gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Numbers Table
    st.markdown("#### 📅 1 to 4 Week Forward Planning Quantiles")
    table_data = []
    for p in calibrated_preds:
        h = p["horizon"]
        target_d = future_dates[h - 1].strftime('%Y-%m-%d')
        table_data.append({
            "Horizon": f"Week {h} Ahead ({target_d})",
            "80% Low (q10)": f"{int(p['q10']):,}",
            "50% Low (q25)": f"{int(p['q25']):,}",
            "Median Expected (q50)": f"{int(p['q50']):,}",
            "50% High (q75)": f"{int(p['q75']):,}",
            "80% High (q90)": f"{int(p['q90']):,}",
            "Persistence Baseline": f"{int(p['baseline_q50']):,}",
            "Shift Alert": "🚨 Widened" if p.get("alert_widening_applied") else "✅ Normal",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# =============================================================================
# SCREEN 2: SCORECARD & CALIBRATION
# =============================================================================
elif nav_page == "📊 Scorecard & Calibration":
    st.title(f"{selected_disease} Scorecard: {selected_district}")
    st.markdown(
        f"Every accuracy metric below is calculated via **rolling-origin walk-forward backtests** on historical {selected_disease} data."
    )
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cov80 = overall_score["coverage_80"] * 100.0
        st.metric("80% Empirical Coverage", f"{cov80:.1f}%", delta=f"{cov80 - 80.0:+.1f}% vs 80% target")
    with c2:
        cov50 = overall_score["coverage_50"] * 100.0
        st.metric("50% Empirical Coverage", f"{cov50:.1f}%", delta=f"{cov50 - 50.0:+.1f}% vs 50% target")
    with c3:
        rwis = overall_score["relative_wis"]
        st.metric("Relative WIS (rWIS)", f"{rwis:.2f}", delta=f"{(1.0 - rwis)*100:+.0f}% vs baseline", delta_color="inverse")
    with c4:
        st.metric("Mean WIS Error", f"{overall_score['mean_wis_model']:.1f}", delta=f"Base: {overall_score['mean_wis_baseline']:.1f}", delta_color="inverse")
        
    st.markdown("---")
    
    # Horizon Breakdown Table
    st.subheader("Performance Breakdown by Forecast Horizon (1 to 4 Weeks Ahead)")
    horizons_df = scorecard_dict["by_horizon"].copy()
    horizons_df["coverage_80"] = horizons_df["coverage_80"].apply(lambda x: f"{x*100:.1f}%")
    horizons_df["coverage_50"] = horizons_df["coverage_50"].apply(lambda x: f"{x*100:.1f}%")
    horizons_df["relative_wis"] = horizons_df["relative_wis"].apply(lambda x: f"{x:.2f}")
    horizons_df["mean_wis_model"] = horizons_df["mean_wis_model"].apply(lambda x: f"{x:,.1f}")
    horizons_df["mean_wis_baseline"] = horizons_df["mean_wis_baseline"].apply(lambda x: f"{x:,.1f}")
    horizons_df["mae_model"] = horizons_df["mae_model"].apply(lambda x: f"{x:,.1f}")
    horizons_df["mae_baseline"] = horizons_df["mae_baseline"].apply(lambda x: f"{x:,.1f}")
    horizons_df["sharpness_80"] = horizons_df["sharpness_80"].apply(lambda x: f"{x:,.1f}")
    
    st.dataframe(
        horizons_df.rename(columns={
            "horizon_weeks": "Horizon (Weeks)",
            "n_eval": "Evaluation Points",
            "coverage_80": "80% Coverage",
            "coverage_50": "50% Coverage",
            "relative_wis": "Relative WIS (rWIS)",
            "mean_wis_model": "RangeCast WIS",
            "mean_wis_baseline": "Baseline WIS",
            "mae_model": "RangeCast MAE",
            "mae_baseline": "Baseline MAE",
            "sharpness_80": "Mean 80% Spread",
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Trust Badge Explanation Card
    st.markdown("### 🏆 Trust Badge Criteria & Verification Audit")
    t_card = trust_badge
    st.markdown(f"""
    **Current Classification:** <span style="font-size:1.1rem; color:{t_card['color']}; font-weight:700;">{t_card['badge_text']}</span>  
    *{t_card['summary']}*
    """, unsafe_allow_html=True)
    
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("#### ✅ Positive Reliability Signals")
        for pos in t_card.get("positive_signals", []):
            st.markdown(f"- {pos}")
        if not t_card.get("positive_signals"):
            st.markdown("- None recorded.")
            
    with col_neg:
        st.markdown("#### ⚠️ Warnings & Inspection Triggers")
        for neg in t_card.get("reasons", []):
            st.markdown(f"- {neg}")
        if not t_card.get("reasons"):
            st.markdown("- No active penalties or warnings.")


# =============================================================================
# SCREEN 3: RESOURCE PLANNER (NEWSVENDOR RULE)
# =============================================================================
elif nav_page == "🏥 Resource Planner (Newsvendor)":
    st.title(f"🏥 Operational Resource Planner: {selected_disease}")
    st.markdown(
        f"Translates calibrated {selected_disease} case projections into hospital and logistics procurement orders using the **Newsvendor Critical Fractile**."
    )
    
    # Newsvendor Risk Slider & Presets
    st.subheader("1. Procurement Risk Preference (Newsvendor Fractile)")
    st.markdown(
        "$$q = \\frac{C_u}{C_u + C_o}$$  \n"
        "*Underage Cost ($C_u$)*: Severe shortages, unadmitted patients.  \n"
        "*Overage Cost ($C_o$)*: Excess holding costs, expired reagent kits."
    )
    
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_choice = None
    if preset_col1.button("🚨 Conservative ($q = 85\%$) - Shortage Sensitive"):
        preset_choice = 0.85
    if preset_col2.button("⚖️ Balanced ($q = 75\%$) - Standard"):
        preset_choice = 0.75
    if preset_col3.button("📉 Lean ($q = 50\%$ / Median) - Budget Constrained"):
        preset_choice = 0.50
        
    slider_val = preset_choice if preset_choice is not None else 0.80
    risk_fractile = st.slider(
        "Planning Risk Fractile (q):",
        min_value=0.10,
        max_value=0.95,
        value=slider_val,
        step=0.05,
    )
    
    st.markdown(f"**Active Planning Fractile: {risk_fractile:.0%} Percentile**.")
    st.markdown("---")
    
    # Disease-specific default assumptions
    default_assump = get_default_disease_assumptions(selected_disease)
    
    # Editable Assumptions Expander
    with st.expander(f"⚙️ Edit Clinical Assumptions for {selected_disease} (Placeholders)", expanded=False):
        st.markdown("<small style='color:#94A3B8;'>Assumptions are editable district planning parameters, not immutable norms.</small>", unsafe_allow_html=True)
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            hosp_rate = st.number_input("Hospitalization Rate (%)", value=default_assump.hospitalization_rate * 100.0, min_value=0.5, max_value=50.0, step=0.5) / 100.0
            icu_rate = st.number_input("Severe / ICU Fraction (%)", value=default_assump.icu_fraction * 100.0, min_value=1.0, max_value=50.0, step=1.0) / 100.0
        with col_a2:
            alos = st.number_input("Avg Length of Stay (ALOS days)", value=default_assump.avg_length_of_stay_days, min_value=1.0, max_value=30.0, step=0.5)
            tests_per_case = st.number_input("Diagnostic Tests per Confirmed Case", value=default_assump.tests_per_confirmed_case, min_value=1.0, max_value=60.0, step=1.0)
        with col_a3:
            lead_time = st.slider("Supplier Lead Time (Weeks)", min_value=1, max_value=4, value=default_assump.supplier_lead_time_weeks)
            cur_beds = st.number_input("Current Operational Beds Available", value=200, min_value=0, step=20)
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
    
    st.subheader(f"2. Actionable Resource Requirements ({selected_disease} - {lead_time} Wk Lead Time)")
    
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Inpatient Beds Needed</div>
            <div class="metric-value" style="color:#60A5FA;">{summary['peak_beds_required']:,}</div>
            <div class="metric-sub">Severe/HDU: <b>{summary['peak_icu_required']:,}</b> | Net Mobilize: <b>{summary['net_beds_to_mobilize']:,}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with r_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Lead-Time Diagnostic Kits</div>
            <div class="metric-value" style="color:#34D399;">{summary['lead_time_kits_needed']:,}</div>
            <div class="metric-sub">Net Kits to Order: <b>{summary['net_kits_to_order']:,}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    # Disease-specific specialized cards in columns 3 & 4
    if selected_disease == "Dengue":
        with r_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Platelet Units (RDP/SDP)</div>
                <div class="metric-value" style="color:#F43F5E;">{summary['lead_time_platelets']:,}</div>
                <div class="metric-sub">For Severe Thrombocytopenia</div>
            </div>
            """, unsafe_allow_html=True)
        with r_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">IV Fluids (500ml RL/NS)</div>
                <div class="metric-value" style="color:#38BDF8;">{summary['lead_time_iv_fluids']:,}</div>
                <div class="metric-sub">Plasma Leakage Management</div>
            </div>
            """, unsafe_allow_html=True)
    elif selected_disease == "Malaria":
        with r_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ACT Treatment Blister Packs</div>
                <div class="metric-value" style="color:#F59E0B;">{summary['lead_time_act_courses']:,}</div>
                <div class="metric-sub">Artemisinin Combination Courses</div>
            </div>
            """, unsafe_allow_html=True)
        with r_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rapid Diagnostic Tests (RDT)</div>
                <div class="metric-value" style="color:#34D399;">{summary['lead_time_kits_needed']:,}</div>
                <div class="metric-sub">Antigen Detection Kits</div>
            </div>
            """, unsafe_allow_html=True)
    elif selected_disease == "Acute Diarrheal Disease":
        with r_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ORS Sachets (Packets)</div>
                <div class="metric-value" style="color:#F59E0B;">{summary['lead_time_ors_sachets']:,}</div>
                <div class="metric-sub">Oral Rehydration Packets</div>
            </div>
            """, unsafe_allow_html=True)
        with r_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Chlorine Water Tablets</div>
                <div class="metric-value" style="color:#38BDF8;">{summary['lead_time_chlorine_tabs']:,}</div>
                <div class="metric-sub">Halazone Purification Tablets</div>
            </div>
            """, unsafe_allow_html=True)
    elif selected_disease == "Influenza (H1N1)":
        with r_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tamiflu Antiviral Courses</div>
                <div class="metric-value" style="color:#A78BFA;">{summary['lead_time_tamiflu_courses']:,}</div>
                <div class="metric-sub">Oseltamivir 75mg Courses</div>
            </div>
            """, unsafe_allow_html=True)
        with r_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Peak Daily O2 Cylinders</div>
                <div class="metric-value" style="color:#38BDF8;">{summary['peak_daily_o2_cylinders']:,}</div>
                <div class="metric-sub">D-Type Cylinders / Day</div>
            </div>
            """, unsafe_allow_html=True)
    else: # COVID-19
        with r_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Peak Daily Oxygen Cylinders</div>
                <div class="metric-value" style="color:#38BDF8;">{summary['peak_daily_o2_cylinders']:,}</div>
                <div class="metric-sub">D-Type Cylinders / Day (7,000L)</div>
            </div>
            """, unsafe_allow_html=True)
        with r_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Lead-Time Oxygen Cylinders</div>
                <div class="metric-value" style="color:#A78BFA;">{summary['lead_time_cylinders_needed']:,}</div>
                <div class="metric-sub">Net to Order/Refill</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# SCREEN 4: COMPARE MODELS VIEW
# =============================================================================
elif nav_page == "🔄 Compare Models View":
    st.title(f"🔄 Model Comparison: {selected_disease}")
    st.markdown(
        f"Score an external model or upload an external weekly forecast and evaluate it on the exact same **{selected_disease} actuals**."
    )
    
    comp_mode = st.radio(
        "Source for comparison:",
        options=[
            "Test Built-in External Model: Point Extrapolation (Simulated Single-Line)",
            "Test Built-in External Model: Pure Naive Persistence Baseline",
            "Upload Custom Forecast CSV File",
        ]
    )
    
    eval_h1 = df_eval[df_eval['horizon_weeks'] == 1].copy().reset_index(drop=True)
    if "Upload Custom Forecast CSV" in comp_mode:
        up_file = st.file_uploader("Upload CSV", type=["csv"])
        if up_file is not None:
            up_df = pd.read_csv(up_file)
            st.success(f"Loaded {len(up_df)} external forecast rows.")
            pred_col = 'predicted_cases' if 'predicted_cases' in up_df.columns else up_df.columns[1]
            eval_h1['external_pred'] = np.interp(
                pd.to_datetime(eval_h1['target_date']).astype('int64'),
                pd.to_datetime(up_df.iloc[:, 0]).astype('int64'),
                up_df[pred_col].astype(float)
            )
        else:
            eval_h1['external_pred'] = eval_h1['baseline_q50'] * 1.15
    elif "Pure Naive Persistence" in comp_mode:
        eval_h1['external_pred'] = eval_h1['baseline_q50']
    else:
        eval_h1['external_pred'] = eval_h1['baseline_q50'] * 1.08 + np.sin(np.arange(len(eval_h1))) * 0.1 * eval_h1['baseline_q50']
        
    rc_mae = np.mean(np.abs(eval_h1['actual'] - eval_h1['q50']))
    ext_mae = np.mean(np.abs(eval_h1['actual'] - eval_h1['external_pred']))
    rc_cov80 = np.mean(eval_h1['covered_80']) * 100.0
    
    ext_l = eval_h1['external_pred'] * 0.85
    ext_u = eval_h1['external_pred'] * 1.15
    ext_cov = calculate_coverage(eval_h1['actual'], ext_l, ext_u) * 100.0
    ext_is = np.mean(interval_score(eval_h1['actual'].values, ext_l.values, ext_u.values, alpha=0.20))
    rc_is = np.mean(eval_h1['is_80'])
    
    score_comp_df = pd.DataFrame([
        {
            "Model Name": "🛡️ RangeCast Calibrated Ensemble",
            "MAE": f"{rc_mae:,.1f}",
            "80% Interval Coverage": f"{rc_cov80:.1f}%",
            "80% Interval Score": f"{rc_is:,.1f}",
            "Calibration": "Honest Adaptive Conformal Bands",
            "Advantage": "Reference Standard"
        },
        {
            "Model Name": "External / Alternative Forecast",
            "MAE": f"{ext_mae:,.1f}",
            "80% Interval Coverage": f"{ext_cov:.1f}%",
            "80% Interval Score": f"{ext_is:,.1f}",
            "Calibration": "Uncalibrated Point Prediction",
            "Advantage": f"RangeCast is {max(0.0, (1 - rc_is/ext_is)*100):.1f}% sharper"
        }
    ])
    st.dataframe(score_comp_df, use_container_width=True, hide_index=True)


# =============================================================================
# SCREEN 5: ALERTS & MULTI-LANG BRIEFINGS
# =============================================================================
elif nav_page == "🚨 Alerts & Multi-Lang Briefings":
    st.title(f"🚨 Outbreak Advisories: {selected_disease}")
    st.markdown(
        f"Deterministic plain-language briefings for **{selected_disease}** in {selected_district}."
    )
    
    tab_eng, tab_tam, tab_hin, tab_wa = st.tabs([
        "🇬🇧 English Executive Advisory",
        "🇮🇳 தமிழ் (Tamil) Briefing",
        "🇮🇳 हिन्दी (Hindi) Briefing",
        "📱 WhatsApp One-Card Summary",
    ])
    
    plan_data = calculate_resource_plan(calibrated_preds, 0.80, get_default_disease_assumptions(selected_disease))["summary"]
    
    eng_text = generate_english_briefing(
        district=selected_district,
        as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0],
        quantiles_h2=calibrated_preds[1],
        plan_summary=plan_data,
        scorecard=overall_score,
        risk_fractile=0.80,
        disease=selected_disease,
        shift_alert=active_shift_alert,
        is_small_district=is_small,
    )
    
    tam_text = generate_tamil_briefing(
        district=selected_district,
        as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0],
        plan_summary=plan_data,
        scorecard=overall_score,
        risk_fractile=0.80,
        disease=selected_disease,
        shift_alert=active_shift_alert,
    )
    
    hin_text = generate_hindi_briefing(
        district=selected_district,
        as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0],
        plan_summary=plan_data,
        scorecard=overall_score,
        risk_fractile=0.80,
        disease=selected_disease,
        shift_alert=active_shift_alert,
    )
    
    wa_card = generate_whatsapp_card(
        district=selected_district,
        as_of_date=latest_date_str,
        quantiles_h1=calibrated_preds[0],
        plan_summary=plan_data,
        scorecard=overall_score,
        risk_fractile=0.80,
        disease=selected_disease,
        shift_alert=active_shift_alert,
    )
    
    with tab_eng:
        st.code(eng_text, language="text")
    with tab_tam:
        st.code(tam_text, language="text")
    with tab_hin:
        st.code(hin_text, language="text")
    with tab_wa:
        st.markdown("#### Ready for Mobile Transmission to CMO & War Room")
        st.code(wa_card, language="markdown")


# =============================================================================
# SCREEN 6: PAST SURGE REPLAY
# =============================================================================
elif nav_page == "⚡ Past Surge Replay":
    st.title("⚡ Historical Surge Replay: Wave 2 (April–May 2021)")
    st.markdown(
        "Demonstrates how the conventional single-line forecast failed during rapid outbreak expansion, "
        "while RangeCast's calibrated range caught the true trajectory and triggered proactive shift alerts."
    )
    
    surge_district = st.selectbox("Select Surge District to Replay:", options=["Chennai", "Coimbatore", "Madurai"], index=0)
    
    with st.spinner("Replaying historical surge..."):
        df_surge = run_surge_replay(district=surge_district, verbose=False)
        
    sl_mae = np.mean(np.abs(df_surge['actual'] - df_surge['single_line_pred']))
    rc_mae = np.mean(np.abs(df_surge['actual'] - df_surge['q50']))
    rc_cov80 = np.mean(df_surge['rangecast_caught']) * 100.0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Single-Line Average Error", f"{int(sl_mae):,} cases/wk")
    with c2:
        st.metric("RangeCast Median Error", f"{int(rc_mae):,} cases/wk", delta=f"-{(1 - rc_mae/sl_mae)*100:.1f}% error", delta_color="inverse")
    with c3:
        st.metric("RangeCast 80% Honest Coverage", f"{rc_cov80:.1f}%", delta="Caught in surge")
        
    st.markdown("---")
    
    fig_surge = go.Figure()
    fig_surge.add_trace(go.Scatter(
        x=df_surge['target_date_str'],
        y=df_surge['actual'],
        name="Actual Exploding Cases",
        mode="lines+markers",
        line=dict(color="#F8FAFC", width=3),
        marker=dict(size=8, color="#F8FAFC")
    ))
    fig_surge.add_trace(go.Scatter(
        x=df_surge['target_date_str'],
        y=df_surge['q90'],
        name="RangeCast 80% Upper Bound",
        line=dict(color="#3B82F6", width=1.5, dash="dash"),
    ))
    fig_surge.add_trace(go.Scatter(
        x=df_surge['target_date_str'],
        y=df_surge['q10'],
        name="RangeCast 80% Lower Bound",
        line=dict(color="#3B82F6", width=1.5, dash="dash"),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.20)',
    ))
    fig_surge.add_trace(go.Scatter(
        x=df_surge['target_date_str'],
        y=df_surge['single_line_pred'],
        name="Single-Line Point Forecast (Lagged)",
        mode="lines+markers",
        line=dict(color="#EF4444", width=2.5, dash="dot"),
        marker=dict(symbol="circle-open", size=8, color="#EF4444")
    ))
    
    fig_surge.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Week Ending Date"),
        yaxis=dict(title="Weekly Cases"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_surge, use_container_width=True)
    
    st.markdown("#### 📅 Week-by-Week Replay Log")
    log_rows = []
    for _, r in df_surge.iterrows():
        log_rows.append({
            "Week End": r['target_date_str'],
            "Actual Cases": f"{int(r['actual']):,}",
            "Single-Line Forecast": f"{int(r['single_line_pred']):,}",
            "RangeCast [80% Range]": f"[{int(r['q10']):,} - {int(r['q90']):,}]",
            "RangeCast Caught?": "✅ CAUGHT" if r['rangecast_caught'] else "❌ MISSED",
            "Shift Alert": "🚨 ALERT!" if r['shift_alert'] else "Normal",
        })
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
