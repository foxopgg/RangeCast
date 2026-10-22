# 🛡️ RangeCast: Calibrated Outbreak Forecasting & Resource Planning

> **Empowering District Health Officers across India with honest prediction intervals, empirical accuracy scorecards, and risk-calibrated hospital procurement.**

---

## 🎯 Problem Statement

District health officers must order hospital beds, oxygen cylinders, diagnostic kits, and therapeutics weeks in advance. However, the forecasts they typically receive:
1. Are **single lines** with no honest representation of uncertainty.
2. Provide **no historical accuracy or coverage record**, making it impossible to know when to trust them.
3. Arrive with **reporting delays, holiday drops, and high Poisson noise** in small rural districts.

**RangeCast replaces "here is a line" with:**
- *"Here is a likely low and high range (50% and 80% confidence bands)."*
- *"Here is how often ranges like this held true in past rolling backtests on actual district data."*
- *"Here is an actionable procurement order for beds, oxygen, and testing kits derived from the newsvendor economic rule."*
- *"Here is a plain-language AI summary of the situation — with every number verified against computed data."*

---

## 🏗️ Project Architecture & Modules

```
RangeCast/
├── app.py               # Premium health-officer Streamlit dashboard (redesigned)
├── fastapi_server.py    # FastAPI REST API backend (6 endpoints)
├── groq_client.py       # Groq AI client (number validator, cache, fallback)
├── backtest.py          # Rolling-origin walk-forward backtest engine
├── conformal.py         # Online Adaptive Conformal Inference (ACI) & Shift Alert
├── data.py              # Modular multi-disease loader (COVID-19, Dengue, Malaria, Flu, Diarrhea)
├── demo_surge.py        # Replay demo script (Wave 2 surge)
├── models.py            # 3-model ensemble (Naive, Negative Binomial GLM, Growth rate)
├── nowcast.py           # Reporting delay completeness curve & holiday dip adjustment
├── planner.py           # Disease-specific resource planner via Newsvendor Critical Fractile
├── scoring.py           # Proper scoring rules: WIS, empirical coverage, rWIS, Trust Badges
├── templates.py         # Deterministic plain-language briefings (English, Tamil, Hindi)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template (GROQ_API_KEY, GROQ_MODEL, etc.)
├── data/
│   ├── covid_tamil_nadu_raw.csv          # Real COVID-19 district daily time series
│   ├── dengue_tamil_nadu_weekly.csv      # Weekly Dengue district series
│   ├── malaria_tamil_nadu_weekly.csv     # Weekly Malaria district series
│   ├── influenza_tamil_nadu_weekly.csv   # Weekly Influenza / H1N1 seasonal series
│   ├── cholera_add_tamil_nadu_weekly.csv # Weekly Acute Diarrheal Disease series
│   ├── rainfall_tamil_nadu.csv           # District-level weekly rainfall (IMD)
│   ├── festivals_calendar.csv            # 27 festivals/holidays with dip/surge flags
│   └── cache/                            # Pre-computed rolling backtest evaluations
└── tests/
    ├── test_scoring.py    # Scoring rule unit tests (WIS, coverage, trust badges)
    └── test_validator.py  # Groq number validator unit tests (33 tests, 33 pass)
```

---

## 🦠 Multi-Disease Surveillance & Resource Support

| Disease | Dynamics | Key Supplies |
|:---|:---|:---|
| **COVID-19** | Airborne respiratory pandemic waves | Hospital Beds, ICU Beds, D-Type O₂ Cylinders, RT-PCR Kits |
| **Dengue Fever** | Vector-borne (2–4 wk rainfall lags) | Platelet Units (RDP/SDP), IV Fluid Bottles, NS1/IgM ELISA Kits |
| **Malaria** | Vector-borne; coastal & forest hotspots | ACT/Chloroquine Courses, Rapid Diagnostic Tests (RDTs), Inpatient Beds |
| **Influenza (H1N1)** | Seasonal respiratory surge | Oseltamivir (Tamiflu) Courses, VTM Swab Kits, O₂ Beds |
| **Acute Diarrheal Disease (ADD/Cholera)** | Water-borne post-flooding | ORS Sachets, IV Ringer's Lactate, Chlorine Purification Tablets |

---

## 🚀 Quickstart Guide

### 1. Setup Environment

```bash
# Clone and enter workspace
cd "RangeCast"

# Create virtual environment & activate
python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and fill in your real Groq API key
# GROQ_API_KEY=gsk_your_real_key_here
# GROQ_MODEL=llama-3.3-70b-versatile
```

> Get your free Groq API key at: https://console.groq.com/keys

### 3. Run the Full Test Suite

```bash
# Scoring rule tests
PYTHONPATH=. pytest tests/test_scoring.py -v

# Groq number validator tests (no API key needed)
PYTHONPATH=. pytest tests/test_validator.py -v

# All tests
PYTHONPATH=. pytest tests/ -v
```

### 4. Launch the Health Officer Dashboard (Streamlit)

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. Launch the FastAPI Backend (optional, for REST API access)

```bash
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at **http://localhost:8000/docs**

### 6. Historical Surge Replay Demo

```bash
python demo_surge.py --district Chennai
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/forecast` | Calibrated 4-week quantile forecast (`?disease=&district=`) |
| `GET` | `/api/scorecard` | Empirical accuracy scorecard from backtest |
| `POST` | `/api/plan` | Newsvendor resource procurement plan |
| `POST` | `/api/compare` | Upload external CSV and score against same actuals |
| `POST` | `/api/summary` | AI-generated plain-language outbreak summary (Groq) |
| `POST` | `/api/ask` | AI Q&A — every number verified against computed context |
| `GET` | `/api/districts` | List available districts for a disease |
| `GET` | `/api/diseases` | List supported diseases |

All AI endpoints (`/api/summary`, `/api/ask`) are rate-limited to **10 requests/minute per IP**. The Groq API key is **never** exposed in any response.

---

## 🔬 Core Methodologies

### 1. Cleaning & Nowcasting (`nowcast.py`)
- **Holiday Dip Correction**: Accounts for artificial drops in specimen collection over festival weekends.
- **Reporting Delay Completeness Curve**: Models cumulative reporting fraction C(d) via a log-logistic curve (θ=4.5 days, κ=2.2).
- **Simulated Vintage Snapshots**: Prominently labeled `[SIMULATED REPORTING DELAY]` when actual database snapshots are unavailable.

### 2. Multi-Model Forecasting Ensemble (`models.py`)
Outputs q10, q25, q50, q75, q90 for 1–4 weeks ahead:
- **Model A (Baseline)**: "Same as last week" persistence with empirical percentage-shift volatility.
- **Model B (Negative Binomial GLM)**: Overdispersed Poisson with lagged log-cases, rainfall lags (t-1, t-2), and holiday counts.
- **Model C (Damped Growth Rate)**: Local reproduction momentum with exponential damping g^(h·φ) to prevent explosive runaway.
- **Small District Partial Pooling**: Empirical Bayes shrinkage for districts < 60 cases/wk.

### 3. Adaptive Conformal Prediction (`conformal.py`)
Online Adaptive Conformal Inference (Gibbs & Candès, 2021):

```
α_{t+1} = clip(α_t + γ · (α_target − miss_t), α_min, α_max)
```

- If model misses → α decreases → intervals widen next week.
- If model covers → α increases → intervals tighten.
- **Shift Alert**: Triggers when 2 of last 3 weeks breach bounds; widens planning bands 35%.

### 4. Proper Scoring Rules (`scoring.py`)
- **Weighted Interval Score (WIS)**: CDC COVID-19 Forecast Hub standard.
- **Relative WIS (rWIS)**: % improvement over "same as last week" persistence baseline.
- **Trust Badges**: HIGH / MEDIUM / LOW based on deterministic threshold rules.

### 5. Operational Resource Planner (`planner.py`)
Newsvendor Critical Fractile: `q = Cu / (Cu + Co)`

Converts case quantiles into:
- Hospital beds (general + ICU) via Little's Law
- Oxygen cylinders per day and per lead-time window
- Disease-specific supplies (platelets, ORS, Tamiflu, ACT courses)
- Net orders accounting for on-hand stock

### 6. Groq AI Integration (`groq_client.py`)

**Security & Correctness:**
- GROQ_API_KEY never sent to browser or logged.
- System prompt forbids introducing numbers not in the JSON context.
- **Number Validator**: Extracts every number from model output; rejects response if any number is not in the allowed set (2% rounding tolerance; standard interval levels always allowed).
- On validation failure or any error → falls back to deterministic template; returns `verified=False`.
- In-memory TTL cache (300 entries, 10-min TTL).
- 8-second hard timeout on every Groq call.
- Temperature: 0.1 (near-deterministic).

### 7. Deterministic Plain-Language Templates (`templates.py`)
- Fixed templates filled with computed backtest numbers only.
- English, Tamil (தமிழ்), Hindi (हिन्दी) — labelled "machine-assisted wording."
- WhatsApp one-card summary for instant mobile sharing.

---

## 🖥️ Dashboard Pages

| Page | Description |
|:---|:---|
| 🏠 **Dashboard** | KPI tiles, calibrated range chart, 4-week quantile table |
| 📊 **Accuracy Scorecard** | 80%/50% coverage, rWIS, horizon breakdown, trust badge audit |
| 🏥 **Resource Planner** | Newsvendor fractile, editable assumptions, disease-specific supply cards |
| 🔄 **Compare Forecasts** | Head-to-head scoring vs uploaded CSV or built-in baselines |
| 📢 **Briefings & Export** | English/Tamil/Hindi advisories, WhatsApp card, download buttons |
| 🤖 **Ask RangeCast AI** | Groq-powered Q&A with verified-number badge, suggested questions |
| ⚡ **Surge Replay Demo** | Wave 2 COVID-19 historical surge replay |

---

## 📊 Data Sources

1. **COVID-19 Surveillance**: `data.covid19india.org` (`districts.csv`). Daily confirmed counts aggregated to weekly intervals (W-SUN) across 38 Tamil Nadu districts.
2. **Dengue / Malaria / Influenza / ADD**: District-level synthetic weekly series generated from IDSP/NVBDCP outbreak bulletin patterns (publicly available aggregate counts).
3. **Meteorological Rainfall**: District-aggregated weekly precipitation derived from IMD climatological normals and monsoon anomalies.
4. **Festival & Public Holiday Calendar**: Curated calendar of 27 state and national holidays with reporting dip and congregation surge indicators.

---

## 📋 Assumptions & Limitations

1. **Surveillance Aggregation**: Uses aggregated district-level counts. No individual patient-level data or PHI is processed or stored.
2. **Clinical Assumptions as Placeholders**: Hospitalisation rates, ICU rates, and ALOS are customisable starting parameters, **not statutory medical directives**. Every parameter can be edited live in the UI.
3. **Simulated Reporting Delays**: Since the public `covid19india` archive provides consolidated historical records rather than point-in-time snapshot dumps, the reporting delay adjustment is based on a log-logistic delay distribution and clearly tagged `[SIMULATED REPORTING DELAY]`.
4. **Small District Noise**: In rural districts with < 25 cases/wk, stochastic Poisson fluctuations dominate. RangeCast automatically flags these and applies partial pooling.
5. **AI Language Quality**: Tamil and Hindi outputs from Groq AI are machine-generated and labeled "machine-assisted wording." **Native-speaker review is required before use in official communications.**
6. **No GPU Required**: All models run on CPU with standard scientific Python libraries.

---

## 📄 License

Open-source research demonstration for public health informatics. Not for clinical use. All AI-generated content is clearly labelled.
