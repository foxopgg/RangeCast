# 🛠️ RangeCast — Technology Stack

Comprehensive overview of the technologies, frameworks, machine learning models, and generative AI systems powering **RangeCast**.

---

## 🏗️ Architecture Overview

| Layer | Primary Technologies | Purpose |
| :--- | :--- | :--- |
| **User Interface** | Streamlit, Plotly, Altair, Custom CSS | Interactive District Health Officer dashboard, real-time charts & risk badges |
| **API Server** | FastAPI, Uvicorn, Pydantic v2 | High-performance asynchronous REST API backend serving forecast data |
| **Statistical & ML Core** | NumPy, Pandas, SciPy, Statsmodels, Scikit-learn | Multi-model ensemble, GLM negative binomial regression, rolling backtests |
| **Uncertainty Quantification** | Adaptive Conformal Inference (ACI), Proper Scoring (WIS) | Calibrated prediction intervals (50% & 80%), distribution shift detection |
| **Resource Optimization** | Newsvendor Critical Fractile Model | Asymmetric risk-calibrated hospital procurement (beds, O₂, tests, drugs) |
| **AI / GenAI Briefing** | Groq Cloud SDK (`llama-3.3-70b-versatile` / `qwen3.8-27b`) | Automated executive briefings with deterministic number verification |
| **Testing & CI** | Pytest, Pytest-Asyncio | Automated unit & regression test suite for scoring & AI validator |
| **Cloud & Deployment** | Render (PaaS), Python 3.11.9 runtime | Containerless automated cloud hosting with health check integration |

---

## 💻 Languages & Runtimes

- **Python 3.11.9+**: Core runtime used for data pipelines, ML modeling, REST API endpoints, and web dashboard.
- **HTML5 / CSS3**: Embedded custom design system featuring glassmorphic components, high-contrast badges, and responsive stat cards.

---

## 🧠 Machine Learning Models & Statistical Ensemble

RangeCast rejects black-box, uncalibrated point predictions. Instead, it utilizes a **multi-model statistical and machine learning ensemble** coupled with **Empirical Bayes shrinkage** and **distribution-free uncertainty calibration**:

### 1. The 3-Model Quantile Ensemble (`models.py`)
Predictions across 1 to 4-week horizons are formed as a weighted combination of three distinct models:

$$\text{Forecast Quantiles} = 0.45 \cdot \text{GLM}_{\text{NB}} + 0.35 \cdot \text{Growth}_{\text{Damped}} + 0.20 \cdot \text{Baseline}_{\text{Naive}}$$

* **Negative Binomial Generalized Linear Model (GLM) — 45% Weight**:
  * **Engine**: `statsmodels.api.families.NegativeBinomial` with a log-link function.
  * **Objective**: Solves the overdispersion problem ($\text{Variance} = \mu + \alpha \mu^2$) inherent in epidemiological count data where Poisson models fail.
  * **Features**:
    * Autoregressive lags: $\log(y_{t-1} + 1)$ and $\log(y_{t-2} + 1)$
    * Exogenous environmental drivers: 1-week and 2-week IMD rainfall lags
    * Operational indicators: Holiday count flags to counter administrative reporting dips
  * **Output**: Exact parametric quantiles derived from the negative binomial cumulative distribution function.

* **Damped Exponential Growth Rate Model — 35% Weight**:
  * **Objective**: Models short-term transmission velocity $g = \frac{y_t + 1}{y_{t-1} + 1}$ without suffering from explosive, unrealistic runaway.
  * **Mechanism**: Applies geometric damping $g^{(h \cdot \phi)}$ with damping parameter $\phi \approx 0.72$, steadily pulling growth momentum toward 1.0 over multi-week horizons.
  * **Uncertainty**: Empirical log-volatility scaling with $\sqrt{h}$ forecast horizon expansion.

* **Naive Persistence Benchmark — 20% Weight**:
  * **Objective**: Provides a robust "status quo" anchor based on the last observed week's volume, scaled by historical empirical ratio dispersion.
  * **Role**: Serves as the standardized reference denominator for computing **Relative Weighted Interval Score ($rWIS$)**, guaranteeing that model performance is measured against a fair baseline.

### 2. Empirical Bayes Hierarchical Shrinkage (`SmallDistrictPooler`)
* Small rural districts often suffer from sparse counts and high random noise.
* Uses James-Stein-style shrinkage to partially pool district estimates toward the state epidemic trend:
  $$\hat{\theta}_{\text{pooled}} = w \cdot \hat{\theta}_{\text{district}} + (1 - w) \cdot \hat{\theta}_{\text{state}}$$
* Weight $w$ is dynamically proportional to the district's case volume, preserving true local outbreaks while stabilizing noisy estimates.

### 3. Adaptive Conformal Inference (ACI) (`conformal.py`)
* Implements Gibbs & Candès (2021) online conformal prediction.
* Guarantees finite-sample, distribution-free coverage under non-stationary dynamics:
  $$\alpha_{t+1} = \alpha_t + \gamma \cdot (\alpha_{\text{target}} - \text{miss}_t)$$
* **Outbreak Shift Alert**: Detects consecutive interval breaches (CUSUM) to flag viral mutations, super-spreader events, or reporting anomalies, triggering automated interval widening.

### 4. Newsvendor Economic Optimization Model (`planner.py`)
* Converts statistical prediction intervals into optimal procurement quantities (hospital beds, ICU beds, oxygen cylinders, diagnostic kits).
* Solves the asymmetric risk problem: the clinical cost of stockout/underage ($c_u$) far outweighs the financial cost of overage ($c_o$):
  $$\text{Critical Fractile } q^* = \frac{c_u}{c_u + c_o}$$

---

## 🤖 The Role of AI in the Tech Stack

Artificial Intelligence in RangeCast is designed strictly for **high-stakes clinical and public health decision support**. Rather than letting LLMs hallucinate numbers or calculate forecasts, RangeCast enforces a strict separation of concerns:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Deterministic ML Engine     │       │          Groq AI Engine         │
│  (GLM + Growth + Conformal)     │       │   (Llama 3.3 70B / Qwen 27B)    │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │ Calculated Quantiles                    │ Plain-Language Draft
                 │ & Risk Scores                           │
                 ▼                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   Deterministic Number Validator                          │
│        (Cross-checks 100% of figures cited in AI text against data)        │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Verified Briefing
                                      ▼
                      District Health Officer Cockpit
```

### 1. Executive Briefing Generation
* **What it does**: Epidemiologists and district magistrates do not have time to parse raw quantile distributions, rWIS tables, or variance ratios. AI synthesizes these data points into executive-level, actionable daily situational reports (SITREPs).
* **Content Generated**:
  * Outbreak trajectory summaries (e.g., accelerating vs. plateauing).
  * Explicit 50% and 80% range interpretations.
  * Supply deficit warnings with actionable procurement deadlines.
  * Trust badge explanations (why a forecast is marked *Verified* vs. *Caution*).

### 2. Zero-Hallucination Guardrail: Deterministic Number Validator (`groq_client.py`)
* In healthcare, LLM number hallucinations can cost lives (e.g., misreporting 120 required beds as 20).
* Every AI-generated summary must pass through a strict regex and mathematical verification pipeline:
  1. Extracts all numeric values, percentages, and bed/oxygen counts from the generated response.
  2. Compares every single number against the ground-truth values computed by the ML ensemble.
  3. If a hallucinated or perturbed figure is detected, the response is rejected and RangeCast instantly falls back to a deterministic template.
  4. Tested rigorously with 33 automated unit tests achieving a 100% pass rate (`test_validator.py`).

### 3. High-Speed Inference Engine
* **SDK**: Groq Python SDK (`groq>=0.9.0`) running on Language Processing Units (LPUs).
* **Models**:
  * **`llama-3.3-70b-versatile`**: Primary model delivering structured, high-reasoning summaries within sub-second latencies (~500ms).
  * **`qwen/qwen3.8-27b`**: High-throughput fallback model.

### 4. Multilingual Translation & Localization (`templates.py`)
* Broadens accessibility for field health workers across regional administrations:
  * **English**
  * **Tamil (தமிழ்)**
  * **Hindi (हिन्दी)**
* Ensures local medical officers receive identical risk briefings in their operational language.

---

## 📊 Core Data Science & Time Series Libraries

- **[NumPy](https://numpy.org/) (`>=1.24.0`)**: Vectorized operations, quantile calculations, and array manipulations.
- **[Pandas](https://pandas.pydata.org/) (`>=2.0.0`)**: Time series handling, rolling window aggregations, multi-district data transformations.
- **[SciPy](https://scipy.org/) (`>=1.10.0`)**: Statistical distributions (Negative Binomial, Poisson), empirical CDF calculations.
- **[Statsmodels](https://www.statsmodels.org/) (`>=0.14.0`)**: Generalized Linear Models (Negative Binomial GLM with log link) for overdispersed epidemiological counts.
- **[Scikit-learn](https://scikit-learn.org/) (`>=1.3.0`)**: Preprocessing, evaluation metrics, and regression baseline utilities.

---

## 🎨 Visualization & Dashboard

- **[Streamlit](https://streamlit.io/) (`>=1.30.0`)**: Web application framework rendering the decision cockpit for District Health Officers.
- **[Plotly](https://plotly.com/python/) (`>=5.18.0`)**: Interactive multi-band interval charts (50% and 80% fan charts), rolling backtest trajectory plots, and supply deficit gauges.
- **[Altair](https://altair-viz.github.io/) (`>=5.0.0`)**: Declarative statistical heatmaps and multi-disease time-series visualizers.

---

## ⚡ API Backend & Data Contracts

- **[FastAPI](https://fastapi.tiangolo.com/) (`>=0.110.0`)**: Modern asynchronous ASGI framework powering REST endpoints for external dashboards and health portals:
  - `GET /health` — Service health & model registry status
  - `GET /diseases` — Supported surveillance profiles
  - `POST /forecast` — Interval forecasting and empirical accuracy
  - `POST /plan` — Newsvendor supply procurement recommendations
  - `POST /summary` — Plain-language AI / template briefings
  - `POST /replay` — Historical outbreak simulation
- **[Uvicorn](https://www.uvicorn.org/) (`>=0.27.0`)**: Lightning-fast ASGI production web server.
- **[Pydantic v2](https://docs.pydantic.dev/) (`>=2.0.0`)**: Strict schema definitions, request validation, and response serialization.

---

## 🌐 Networking, Utilities & Caching

- **[HTTPX](https://www.python-httpx.org/) (`>=0.27.0`)** & **[Requests](https://requests.readthedocs.io/) (`>=2.28.0`)**: Async and sync HTTP client communication.
- **[Cachetools](https://cachetools.readthedocs.io/) (`>=5.3.0`)**: TTLCache for pre-computed backtest results, API rate savings, and instant dashboard loading.
- **[SlowAPI](https://github.com/laurentS/slowapi) (`>=0.1.9`)**: IP-based rate limiting to protect public endpoints.
- **[python-dotenv](https://github.com/theskumar/python-dotenv) (`>=1.0.0`)**: Seamless environment variable configuration.
- **[python-dateutil](https://dateutil.readthedocs.io/) (`>=2.8.2`)**: Flexible datetime parsing across regional epidemiological reporting formats.

---

## 🧪 Testing & Verification

- **[Pytest](https://docs.pytest.org/) (`>=7.4.0`)**: Test framework running unit and integration tests.
- **[Pytest-Asyncio](https://pytest-asyncio.readthedocs.io/) (`>=0.23.0`)**: Asynchronous test runner for FastAPI routes and background tasks.
- **Test Coverage**:
  - Mathematical correctness of Weighted Interval Score (WIS) & coverage formulas.
  - Strict validator testing against hallucinated or perturbed numbers (33/33 tests passing).

---

## 🚀 Cloud Infrastructure & Deployment

- **Hosting Platform**: [Render](https://render.com) (Web Service)
- **Deployment Config**: `render.yaml` infrastructure-as-code
- **Environment**: Python 3.11.9
- **Process Management**: Streamlit headless daemon binding to dynamic `$PORT`
