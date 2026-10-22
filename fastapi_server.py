"""
RangeCast FastAPI Backend Server (fastapi_server.py)
====================================================
Provides REST API endpoints for the RangeCast dashboard:

  GET  /api/forecast   - Calibrated quantile forecast for a district
  GET  /api/scorecard  - Empirical accuracy scorecard from backtest
  POST /api/plan       - Resource procurement plan (beds, oxygen, kits)
  POST /api/compare    - Upload an external CSV forecast and score it
  POST /api/summary    - AI-generated plain-language outbreak summary (Groq)
  POST /api/ask        - AI Q&A with the officer (Groq, verified)

Security:
  - GROQ_API_KEY is never included in any response.
  - Rate limiting on AI endpoints: GROQ_RATE_LIMIT_RPM per client IP.
  - All data is aggregated district-level; no patient-level PHI.

Run with:
  uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---- RangeCast Core Modules -----------------------------------------------
from data import OutbreakDataLoader
from backtest import RollingBacktestRunner
from models import RangeCastEnsemble
from conformal import AdaptiveConformalCalibrator
from scoring import (
    interval_score,
    weighted_interval_score,
    calculate_coverage,
    compute_scorecard_summary,
)
from planner import (
    PlannerAssumptions,
    get_default_disease_assumptions,
    calculate_resource_plan,
)
import groq_client

# ---- Logging --------------------------------------------------------------
logger = logging.getLogger("rangecast.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---- Rate Limiting --------------------------------------------------------
RATE_LIMIT_RPM = os.environ.get("GROQ_RATE_LIMIT_RPM", "10")
limiter = Limiter(key_func=get_remote_address)

# ---- FastAPI App ----------------------------------------------------------
app = FastAPI(
    title="RangeCast API",
    description="Calibrated outbreak forecast API for Indian district health officers.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow Streamlit frontend (localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: Validate Groq configuration
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    """Validate Groq API key and model on startup."""
    try:
        groq_client.startup_check()
        logger.info("[startup] Groq AI client validated OK.")
    except RuntimeError as e:
        logger.warning(
            f"[startup] Groq startup check failed: {e}. "
            "AI endpoints will use fallback templates."
        )


# ---------------------------------------------------------------------------
# In-memory cache for backtest results (avoid re-running per request)
# ---------------------------------------------------------------------------
_backtest_cache: Dict[str, pd.DataFrame] = {}

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_eval_df(disease: str, district: str) -> pd.DataFrame:
    """Loads or runs the rolling backtest for a district."""
    key = f"{disease}__{district}"
    if key in _backtest_cache:
        return _backtest_cache[key]

    dis_slug = disease.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    cache_path = f"{CACHE_DIR}/eval_{dis_slug}_{district.lower()}.csv"
    legacy_path = f"{CACHE_DIR}/eval_{district.lower()}.csv"

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    elif disease == "COVID-19" and os.path.exists(legacy_path):
        df = pd.read_csv(legacy_path)
    else:
        loader = OutbreakDataLoader(disease=disease)
        df_d = loader.load_weekly_district_data(district)
        runner = RollingBacktestRunner(min_train_weeks=12, horizon_weeks=4)
        df = runner.run_district_backtest(df_d)
        df.to_csv(cache_path, index=False)

    _backtest_cache[key] = df
    return df


def _get_current_forecast(disease: str, district: str):
    """Generates the latest calibrated 4-week forecast."""
    loader = OutbreakDataLoader(disease=disease)
    df_district = loader.load_weekly_district_data(district)
    is_small = bool(df_district["is_small_district"].iloc[0]) if "is_small_district" in df_district.columns else False

    ens = RangeCastEnsemble()
    ens.fit(df_district)
    raw_preds = ens.predict(horizon_weeks=4, apply_pooling=True, is_small_district=is_small)

    df_eval = _get_eval_df(disease, district)
    calibrator = AdaptiveConformalCalibrator()
    recent_eval = df_eval[df_eval["horizon_weeks"] == 1].tail(12)
    for _, r in recent_eval.iterrows():
        calibrator.scores_80.append(
            calibrator.score_nonconformity(r["actual"], r["q10"], r["q90"], r["q50"])
        )
        calibrator.scores_50.append(
            calibrator.score_nonconformity(r["actual"], r["q25"], r["q75"], r["q50"])
        )
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

    return cal_preds, calibrator.shift_alert_active, df_district, is_small


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class PlanRequest(BaseModel):
    disease: str = Field("COVID-19", description="Disease name")
    district: str = Field("Chennai", description="District name")
    risk_fractile: float = Field(0.80, ge=0.10, le=0.95, description="Newsvendor q value")
    current_beds: int = Field(200, ge=0)
    current_kits: int = Field(2000, ge=0)
    # Optional assumption overrides
    hospitalization_rate: Optional[float] = None
    icu_fraction: Optional[float] = None
    avg_length_of_stay_days: Optional[float] = None
    tests_per_confirmed_case: Optional[float] = None
    supplier_lead_time_weeks: Optional[int] = None


class SummaryRequest(BaseModel):
    disease: str = "COVID-19"
    district: str = "Chennai"
    lang: str = Field("en", description="Language code: en | ta | hi")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    disease: str = "COVID-19"
    district: str = "Chennai"


# ---------------------------------------------------------------------------
# ENDPOINT: /api/forecast
# ---------------------------------------------------------------------------
@app.get("/api/forecast")
async def get_forecast(disease: str = "COVID-19", district: str = "Chennai"):
    """
    Returns calibrated 4-week quantile forecast for a district.
    """
    try:
        cal_preds, shift_alert, df_district, is_small = _get_current_forecast(disease, district)
        latest_cases = int(df_district["cases"].iloc[-1])
        latest_date = str(df_district["week_end_date"].iloc[-1])

        return {
            "district": district,
            "disease": disease,
            "latest_reported_cases": latest_cases,
            "latest_date": latest_date,
            "is_small_district": is_small,
            "shift_alert_active": shift_alert,
            "forecasts": [
                {
                    "horizon_weeks": p["horizon"],
                    "q10": round(p["q10"], 0),
                    "q25": round(p["q25"], 0),
                    "q50": round(p["q50"], 0),
                    "q75": round(p["q75"], 0),
                    "q90": round(p["q90"], 0),
                    "baseline_q50": round(p["baseline_q50"], 0),
                    "alert_widening": p.get("alert_widening_applied", False),
                }
                for p in cal_preds
            ],
            "label": "[SIMULATED REPORTING DELAY]" if is_small else "Reported",
        }
    except Exception as e:
        logger.error(f"[/api/forecast] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT: /api/scorecard
# ---------------------------------------------------------------------------
@app.get("/api/scorecard")
async def get_scorecard(disease: str = "COVID-19", district: str = "Chennai"):
    """
    Returns the empirical accuracy scorecard from rolling backtest.
    """
    try:
        df_eval = _get_eval_df(disease, district)
        runner = RollingBacktestRunner()
        scorecard = runner.generate_district_scorecard(df_eval)
        overall = scorecard["overall"]

        return {
            "district": district,
            "disease": disease,
            "overall": {
                "coverage_80": round(overall["coverage_80"], 4),
                "coverage_80_pct": round(overall["coverage_80"] * 100, 1),
                "coverage_50": round(overall["coverage_50"], 4),
                "coverage_50_pct": round(overall["coverage_50"] * 100, 1),
                "relative_wis": round(overall["relative_wis"], 3),
                "mean_wis_model": round(overall["mean_wis_model"], 2),
                "mean_wis_baseline": round(overall["mean_wis_baseline"], 2),
                "mae_model": round(overall["mae_model"], 1),
                "mae_baseline": round(overall["mae_baseline"], 1),
                "n_forecasts": overall["n_forecasts"],
                "n_origins": overall["n_origins"],
                "trust_tier": overall["trust_badge"]["tier"],
                "trust_badge": overall["trust_badge"]["badge_text"],
                "trust_color": overall["trust_badge"]["color"],
                "trust_summary": overall["trust_badge"]["summary"],
                "positive_signals": overall["trust_badge"].get("positive_signals", []),
                "warnings": overall["trust_badge"].get("reasons", []),
            },
            "by_horizon": scorecard["by_horizon"].to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"[/api/scorecard] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT: /api/plan
# ---------------------------------------------------------------------------
@app.post("/api/plan")
async def get_plan(req: PlanRequest):
    """
    Computes resource procurement plan using the Newsvendor Critical Fractile.
    """
    try:
        cal_preds, _, _, _ = _get_current_forecast(req.disease, req.district)
        default_assump = get_default_disease_assumptions(req.disease)

        # Override assumptions if provided
        assump = PlannerAssumptions(
            disease=req.disease,
            hospitalization_rate=req.hospitalization_rate or default_assump.hospitalization_rate,
            icu_fraction=req.icu_fraction or default_assump.icu_fraction,
            avg_length_of_stay_days=req.avg_length_of_stay_days or default_assump.avg_length_of_stay_days,
            tests_per_confirmed_case=req.tests_per_confirmed_case or default_assump.tests_per_confirmed_case,
            supplier_lead_time_weeks=req.supplier_lead_time_weeks or default_assump.supplier_lead_time_weeks,
            platelets_per_icu_patient_day=default_assump.platelets_per_icu_patient_day,
            iv_fluid_bottles_per_admission=default_assump.iv_fluid_bottles_per_admission,
            act_courses_per_case=default_assump.act_courses_per_case,
            tamiflu_courses_per_case=default_assump.tamiflu_courses_per_case,
            ors_sachets_per_case=default_assump.ors_sachets_per_case,
            chlorine_tabs_per_case=default_assump.chlorine_tabs_per_case,
        )

        plan = calculate_resource_plan(
            quantiles_by_horizon=cal_preds,
            risk_fractile=req.risk_fractile,
            assumptions=assump,
            current_active_beds=req.current_beds,
            current_kits_stock=req.current_kits,
        )

        return {
            "district": req.district,
            "disease": req.disease,
            "risk_fractile": req.risk_fractile,
            "disclaimer": (
                "All conversion assumptions are editable placeholders, "
                "NOT official clinical norms or statutory directives."
            ),
            "plan": plan["summary"],
            "assumptions": {
                "hospitalization_rate_pct": assump.hospitalization_rate * 100,
                "icu_fraction_pct": assump.icu_fraction * 100,
                "avg_length_of_stay_days": assump.avg_length_of_stay_days,
                "tests_per_confirmed_case": assump.tests_per_confirmed_case,
                "supplier_lead_time_weeks": assump.supplier_lead_time_weeks,
            },
        }
    except Exception as e:
        logger.error(f"[/api/plan] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT: /api/compare
# ---------------------------------------------------------------------------
@app.post("/api/compare")
async def compare_forecasts(
    disease: str = "COVID-19",
    district: str = "Chennai",
    file: UploadFile = File(...),
):
    """
    Upload an external forecast CSV and score it against the same actuals.
    Expected CSV format: date_column, predicted_cases
    """
    try:
        content = await file.read()
        ext_df = pd.read_csv(io.BytesIO(content))

        df_eval = _get_eval_df(disease, district)
        eval_h1 = df_eval[df_eval["horizon_weeks"] == 1].copy().reset_index(drop=True)

        # Try to align external predictions to evaluation dates
        date_col = ext_df.columns[0]
        pred_col = "predicted_cases" if "predicted_cases" in ext_df.columns else ext_df.columns[1]

        ext_dates = pd.to_datetime(ext_df[ext_col := date_col]).astype("int64").values
        eval_dates = pd.to_datetime(eval_h1["target_date"]).astype("int64").values
        ext_preds_aligned = np.interp(eval_dates, ext_dates, ext_df[pred_col].astype(float).values)

        # Score RangeCast
        rc_mae = float(np.mean(np.abs(eval_h1["actual"] - eval_h1["q50"])))
        rc_cov80 = float(np.mean(eval_h1["covered_80"]) * 100.0)
        rc_is = float(np.mean(eval_h1["is_80"]))

        # Score external (assume point-forecast ±15% as implicit interval)
        ext_l = ext_preds_aligned * 0.85
        ext_u = ext_preds_aligned * 1.15
        ext_mae = float(np.mean(np.abs(eval_h1["actual"].values - ext_preds_aligned)))
        ext_cov = float(calculate_coverage(eval_h1["actual"], ext_l, ext_u) * 100.0)
        ext_is = float(np.mean(interval_score(eval_h1["actual"].values, ext_l, ext_u, alpha=0.20)))

        return {
            "n_evaluation_points": len(eval_h1),
            "rangecast": {
                "mae": round(rc_mae, 1),
                "coverage_80_pct": round(rc_cov80, 1),
                "interval_score_80": round(rc_is, 1),
                "method": "Adaptive Conformal Ensemble",
            },
            "external": {
                "mae": round(ext_mae, 1),
                "coverage_implied_80_pct": round(ext_cov, 1),
                "interval_score_80": round(ext_is, 1),
                "method": f"Uploaded: {file.filename}",
                "note": "External interval assumed as ±15% around point forecast.",
            },
            "rangecast_advantage_pct": round(max(0.0, (1 - rc_is / max(ext_is, 1e-6)) * 100), 1),
        }
    except Exception as e:
        logger.error(f"[/api/compare] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT: /api/summary  (AI-powered, rate limited)
# ---------------------------------------------------------------------------
@app.post("/api/summary")
@limiter.limit(f"{RATE_LIMIT_RPM}/minute")
async def get_summary(req: SummaryRequest, request: Request):
    """
    Generates an AI-assisted plain-language outbreak summary (Groq).
    Falls back to deterministic template if AI is unavailable.
    """
    try:
        cal_preds, shift_alert, df_district, is_small = _get_current_forecast(req.disease, req.district)
        df_eval = _get_eval_df(req.disease, req.district)
        runner = RollingBacktestRunner()
        scorecard = runner.generate_district_scorecard(df_eval)
        overall = scorecard["overall"]
        plan = calculate_resource_plan(
            cal_preds, 0.80, get_default_disease_assumptions(req.disease)
        )["summary"]

        # Build the strict context -- only computed numbers go to Groq
        context = {
            "district": req.district,
            "disease": req.disease,
            "latest_reported_cases": int(df_district["cases"].iloc[-1]),
            "q10_week1": int(cal_preds[0]["q10"]),
            "q25_week1": int(cal_preds[0]["q25"]),
            "q50_week1": int(cal_preds[0]["q50"]),
            "q75_week1": int(cal_preds[0]["q75"]),
            "q90_week1": int(cal_preds[0]["q90"]),
            "q50_week2": int(cal_preds[1]["q50"]),
            "q90_week2": int(cal_preds[1]["q90"]),
            "coverage_80_pct": round(overall["coverage_80"] * 100, 1),
            "coverage_50_pct": round(overall["coverage_50"] * 100, 1),
            "relative_wis": round(overall["relative_wis"], 3),
            "trust_tier": overall["trust_badge"]["tier"],
            "n_backtest_weeks": overall["n_origins"],
            "shift_alert": shift_alert,
            "is_small_district": is_small,
            "peak_beds_required": plan.get("peak_beds_required", 0),
            "peak_icu_required": plan.get("peak_icu_required", 0),
            "net_beds_to_mobilize": plan.get("net_beds_to_mobilize", 0),
            "peak_daily_o2_cylinders": plan.get("peak_daily_o2_cylinders", 0),
            "lead_time_cylinders_needed": plan.get("lead_time_cylinders_needed", 0),
            "lead_time_kits_needed": plan.get("lead_time_kits_needed", 0),
        }

        result = groq_client.generate_summary(context, lang=req.lang)
        result["machine_assisted_note"] = (
            "Machine-assisted wording. Native-speaker review required before use in official communications."
            if result.get("machine_assisted")
            else None
        )
        return result

    except Exception as e:
        logger.error(f"[/api/summary] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT: /api/ask  (AI Q&A, rate limited)
# ---------------------------------------------------------------------------
@app.post("/api/ask")
@limiter.limit(f"{RATE_LIMIT_RPM}/minute")
async def ask(req: AskRequest, request: Request):
    """
    Answers a health officer's natural-language question using Groq.
    All numbers in the answer are verified against the computed context.
    """
    try:
        cal_preds, shift_alert, df_district, is_small = _get_current_forecast(req.disease, req.district)
        df_eval = _get_eval_df(req.disease, req.district)
        runner = RollingBacktestRunner()
        scorecard = runner.generate_district_scorecard(df_eval)
        overall = scorecard["overall"]
        plan = calculate_resource_plan(
            cal_preds, 0.80, get_default_disease_assumptions(req.disease)
        )["summary"]

        context = {
            "district": req.district,
            "disease": req.disease,
            "q10_week1": int(cal_preds[0]["q10"]),
            "q25_week1": int(cal_preds[0]["q25"]),
            "q50_week1": int(cal_preds[0]["q50"]),
            "q75_week1": int(cal_preds[0]["q75"]),
            "q90_week1": int(cal_preds[0]["q90"]),
            "q50_week2": int(cal_preds[1]["q50"]) if len(cal_preds) > 1 else None,
            "q90_week2": int(cal_preds[1]["q90"]) if len(cal_preds) > 1 else None,
            "coverage_80_pct": round(overall["coverage_80"] * 100, 1),
            "relative_wis": round(overall["relative_wis"], 3),
            "trust_tier": overall["trust_badge"]["tier"],
            "conformal_alpha_80": round(getattr(cal_preds[0], "get", lambda k, d: d)("conformal_alpha_80", 0.20), 3),
            "shift_alert": shift_alert,
            "is_small_district": is_small,
            "peak_beds_required": plan.get("peak_beds_required", 0),
            "peak_icu_required": plan.get("peak_icu_required", 0),
            "net_beds_to_mobilize": plan.get("net_beds_to_mobilize", 0),
            "peak_daily_o2_cylinders": plan.get("peak_daily_o2_cylinders", 0),
            "lead_time_cylinders_needed": plan.get("lead_time_cylinders_needed", 0),
            "lead_time_kits_needed": plan.get("lead_time_kits_needed", 0),
            "n_backtest_weeks": overall["n_origins"],
        }

        result = groq_client.ask_question(req.question, context)
        result["suggested_questions"] = groq_client.SUGGESTED_QUESTIONS
        return result

    except Exception as e:
        logger.error(f"[/api/ask] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ENDPOINT: /api/districts  (helper for UI)
# ---------------------------------------------------------------------------
@app.get("/api/districts")
async def get_districts(disease: str = "COVID-19"):
    """Returns available districts for a disease."""
    try:
        loader = OutbreakDataLoader(disease=disease)
        return {"disease": disease, "districts": loader.get_districts()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/diseases")
async def get_diseases():
    """Returns supported diseases."""
    return {"diseases": OutbreakDataLoader.SUPPORTED_DISEASES}


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "RangeCast API"}


# ---------------------------------------------------------------------------
# Run with: uvicorn fastapi_server:app --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("fastapi_server:app", host=host, port=port, reload=True)
