"""
RangeCast Forecasting Ensemble Module (models.py)
=================================================
Implements the 3-model multi-horizon ensemble:
1. NaivePersistenceBaseline: "Same as last week" persistence benchmark with historical volatility quantiles
2. NegativeBinomialModel: GLM regression with lagged cases, rainfall lags, and festival flags
3. SimpleGrowthRateModel: Damped exponential growth rate projection with volatility quantiles
4. SmallDistrictPooler: Empirical Bayes partial pooling / shrinkage with state trend for low-count districts
5. RangeCastEnsemble: Combines models into unified 10/25/50/75/90 quantiles for h = 1..4 weeks ahead
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


class NaivePersistenceBaseline:
    """
    Standard 'same as last week' persistence baseline.
    Serves as the reference denominator for Relative WIS (rWIS).
    """
    
    def __init__(self):
        self.last_observed: float = 0.0
        self.historical_ratios: List[float] = []
        
    def fit(self, y: np.ndarray):
        y_clean = np.asarray(y, dtype=float)
        if len(y_clean) >= 2:
            self.last_observed = float(y_clean[-1])
            # Historical 1-step percentage shifts
            ratios = (y_clean[1:] + 1.0) / (y_clean[:-1] + 1.0)
            self.historical_ratios = ratios.tolist()
        else:
            self.last_observed = float(y_clean[-1]) if len(y_clean) else 10.0
            self.historical_ratios = [1.0]

    def predict_quantiles(self, horizon: int = 1) -> Dict[str, float]:
        """Returns 10, 25, 50, 75, 90 quantiles for horizon h."""
        base = max(1.0, self.last_observed)
        if len(self.historical_ratios) > 4:
            # Empirical ratio distribution scaled by sqrt(horizon)
            scale = np.sqrt(horizon)
            log_ratios = np.log(self.historical_ratios)
            mean_lr = np.mean(log_ratios)
            std_lr = max(0.15, np.std(log_ratios)) * scale
            
            q10 = base * np.exp(mean_lr + std_lr * stats.norm.ppf(0.10))
            q25 = base * np.exp(mean_lr + std_lr * stats.norm.ppf(0.25))
            q50 = base  # persistence median is same as last week
            q75 = base * np.exp(mean_lr + std_lr * stats.norm.ppf(0.75))
            q90 = base * np.exp(mean_lr + std_lr * stats.norm.ppf(0.90))
        else:
            # Default spread
            spread = 0.20 * np.sqrt(horizon)
            q10 = base * max(0.2, 1.0 - 1.28 * spread)
            q25 = base * max(0.4, 1.0 - 0.67 * spread)
            q50 = base
            q75 = base * (1.0 + 0.67 * spread)
            q90 = base * (1.0 + 1.28 * spread)
            
        return {
            "q10": round(max(0.0, float(q10)), 1),
            "q25": round(max(0.0, float(q25)), 1),
            "q50": round(max(0.0, float(q50)), 1),
            "q75": round(max(0.0, float(q75)), 1),
            "q90": round(max(0.0, float(q90)), 1),
        }


class NegativeBinomialModel:
    """
    Negative Binomial Generalized Linear Model (GLM) for overdispersed weekly outbreak counts.
    Predictors:
    - log(cases_lag1 + 1), log(cases_lag2 + 1)
    - rainfall lags: rain_lag1, rain_lag2
    - holiday_count
    """
    
    def __init__(self):
        self.model = None
        self.fitted_res = None
        self.alpha_dispersion: float = 0.5
        self.last_features: Dict[str, float] = {}
        self.fallback_mean: float = 10.0
        
    def fit(self, df_train: pd.DataFrame):
        df = df_train.copy()
        df = df.sort_values('week_end_date').reset_index(drop=True)
        
        # Prepare feature matrix
        df['cases_lag1'] = df['cases'].shift(1).fillna(df['cases'].mean())
        df['cases_lag2'] = df['cases'].shift(2).fillna(df['cases_lag1'])
        
        df['log_lag1'] = np.log1p(df['cases_lag1'])
        df['log_lag2'] = np.log1p(df['cases_lag2'])
        
        # Ensure rainfall features exist
        if 'rain_lag1' not in df.columns:
            df['rain_lag1'] = 0.0
        if 'rain_lag2' not in df.columns:
            df['rain_lag2'] = 0.0
        if 'holiday_count' not in df.columns:
            df['holiday_count'] = 0
            
        train_clean = df.dropna(subset=['cases', 'log_lag1', 'log_lag2']).copy()
        self.fallback_mean = float(train_clean['cases'].iloc[-1]) if len(train_clean) else 10.0
        
        # Save latest state for forecasting forward
        self.last_features = {
            'cases_lag1': float(train_clean['cases'].iloc[-1]),
            'cases_lag2': float(train_clean['cases'].iloc[-2]) if len(train_clean) >= 2 else float(train_clean['cases'].iloc[-1]),
            'rain_lag1': float(train_clean['rainfall_mm'].iloc[-1]) if 'rainfall_mm' in train_clean.columns else 0.0,
            'rain_lag2': float(train_clean['rain_lag1'].iloc[-1]) if 'rain_lag1' in train_clean.columns else 0.0,
            'holiday_count': float(train_clean['holiday_count'].iloc[-1]),
        }
        
        if len(train_clean) < 6:
            return
            
        try:
            # Fit Negative Binomial GLM
            formula = "cases ~ log_lag1 + log_lag2 + rain_lag1 + rain_lag2 + holiday_count"
            self.model = smf.glm(
                formula=formula,
                data=train_clean,
                family=sm.families.NegativeBinomial(alpha=0.5)
            )
            self.fitted_res = self.model.fit(maxiter=50, disp=False)
            # Estimate dispersion from Pearson chi2
            if self.fitted_res.df_resid > 0:
                self.alpha_dispersion = max(0.05, float(self.fitted_res.scale))
        except Exception:
            self.fitted_res = None

    def predict_quantiles(self, horizon: int = 1) -> Dict[str, float]:
        """Generates quantiles for horizon h."""
        if self.fitted_res is None or not self.last_features:
            # Fallback based on last observed
            base = max(1.0, self.fallback_mean)
            spread = 0.25 * np.sqrt(horizon)
            return {
                "q10": round(max(0.0, base * (1.0 - 1.28 * spread)), 1),
                "q25": round(max(0.0, base * (1.0 - 0.67 * spread)), 1),
                "q50": round(base, 1),
                "q75": round(base * (1.0 + 0.67 * spread), 1),
                "q90": round(base * (1.0 + 1.28 * spread), 1),
            }
            
        # Multi-step autoregressive rollout
        c1 = self.last_features['cases_lag1']
        c2 = self.last_features['cases_lag2']
        pred_mu = c1
        
        for step in range(horizon):
            row_dict = {
                'log_lag1': [np.log1p(c1)],
                'log_lag2': [np.log1p(c2)],
                'rain_lag1': [self.last_features['rain_lag1']],
                'rain_lag2': [self.last_features['rain_lag2']],
                'holiday_count': [0],  # neutral assumption for future holiday count
            }
            pred_df = pd.DataFrame(row_dict)
            try:
                pred_mu = float(self.fitted_res.predict(pred_df).iloc[0])
            except Exception:
                pred_mu = c1
            pred_mu = max(0.5, pred_mu)
            # Roll forward
            c2 = c1
            c1 = pred_mu
            
        # Negative binomial quantiles from mean pred_mu and dispersion alpha
        # statsmodels NB parameterization: var = mu + alpha * mu^2
        # scipy nbinom parameterization: n = 1 / alpha, p = 1 / (1 + alpha * mu)
        alpha = max(0.05, self.alpha_dispersion * (1.0 + 0.15 * horizon))
        mu = pred_mu
        
        n_param = 1.0 / alpha
        p_param = 1.0 / (1.0 + alpha * mu)
        
        q10 = stats.nbinom.ppf(0.10, n_param, p_param)
        q25 = stats.nbinom.ppf(0.25, n_param, p_param)
        q50 = stats.nbinom.ppf(0.50, n_param, p_param)
        q75 = stats.nbinom.ppf(0.75, n_param, p_param)
        q90 = stats.nbinom.ppf(0.90, n_param, p_param)
        
        return {
            "q10": round(max(0.0, float(q10)), 1),
            "q25": round(max(0.0, float(q25)), 1),
            "q50": round(max(0.0, float(q50)), 1),
            "q75": round(max(0.0, float(q75)), 1),
            "q90": round(max(0.0, float(q90)), 1),
        }


class SimpleGrowthRateModel:
    """
    Damped Growth Rate Model.
    Models the weekly growth multiplier g = (cases_t + 1) / (cases_{t-1} + 1).
    Applies geometric damping g^(h * phi) to prevent explosive runaway.
    """
    
    def __init__(self, damping_factor: float = 0.72):
        self.damping_factor = damping_factor
        self.current_cases: float = 10.0
        self.growth_rate: float = 1.0
        self.growth_volatility: float = 0.20
        
    def fit(self, y: np.ndarray):
        y_clean = np.asarray(y, dtype=float)
        if len(y_clean) < 2:
            self.current_cases = float(y_clean[-1]) if len(y_clean) else 10.0
            self.growth_rate = 1.0
            self.growth_volatility = 0.20
            return
            
        self.current_cases = float(y_clean[-1])
        
        # Calculate recent growth factors (last 3-4 weeks weighted)
        recent_cases = y_clean[-5:] if len(y_clean) >= 5 else y_clean
        ratios = (recent_cases[1:] + 5.0) / (recent_cases[:-1] + 5.0)
        
        # Exponentially weighted recent growth rate
        weights = np.linspace(0.5, 1.0, len(ratios))
        weights /= weights.sum()
        avg_growth = float(np.sum(ratios * weights))
        
        # Clip runaway growth (cap at 1.8x/wk to 0.5x/wk)
        self.growth_rate = np.clip(avg_growth, 0.55, 1.75)
        
        # Empirical volatility
        all_ratios = (y_clean[1:] + 5.0) / (y_clean[:-1] + 5.0)
        self.growth_volatility = max(0.12, float(np.std(all_ratios)))

    def predict_quantiles(self, horizon: int = 1) -> Dict[str, float]:
        """Damped growth trajectory with lognormal expansion."""
        # Damping pulls growth rate toward 1.0 over time
        effective_growth = 1.0 + (self.growth_rate - 1.0) * (self.damping_factor ** horizon)
        # Cumulative growth over horizon
        cum_growth = effective_growth ** horizon
        mu = max(1.0, self.current_cases * cum_growth)
        
        sigma = self.growth_volatility * np.sqrt(horizon)
        
        q10 = mu * np.exp(-1.28 * sigma)
        q25 = mu * np.exp(-0.67 * sigma)
        q50 = mu
        q75 = mu * np.exp(0.67 * sigma)
        q90 = mu * np.exp(1.28 * sigma)
        
        return {
            "q10": round(max(0.0, float(q10)), 1),
            "q25": round(max(0.0, float(q25)), 1),
            "q50": round(max(0.0, float(q50)), 1),
            "q75": round(max(0.0, float(q75)), 1),
            "q90": round(max(0.0, float(q90)), 1),
        }


class SmallDistrictPooler:
    """
    Partial pooling / Empirical Bayes shrinkage for small districts.
    Pulls high-variance small district estimates toward the state epidemic trend:
        theta_pooled = w * theta_district + (1 - w) * theta_state
    where w is proportional to district sample volume.
    """
    
    @staticmethod
    def pool_quantiles(
        district_quantiles: Dict[str, float],
        state_quantiles: Dict[str, float],
        district_recent_cases: float,
        threshold_cases: float = 60.0,
    ) -> Dict[str, float]:
        """
        Shrinks small district quantiles toward state-normalized trajectory.
        """
        if district_recent_cases >= threshold_cases:
            return district_quantiles
            
        # Shrinkage weight: lower cases -> lower w -> more pooling with state
        w = max(0.20, min(0.95, district_recent_cases / threshold_cases))
        
        # State relative shape (ratio to state median)
        state_med = max(1.0, state_quantiles.get('q50', 1.0))
        dist_med = max(1.0, district_quantiles.get('q50', 1.0))
        
        # Normalize state quantiles to district scale
        state_scaled = {
            k: (state_quantiles[k] / state_med) * dist_med
            for k in ['q10', 'q25', 'q50', 'q75', 'q90']
        }
        
        pooled = {}
        for k in ['q10', 'q25', 'q50', 'q75', 'q90']:
            pooled[k] = round(w * district_quantiles[k] + (1.0 - w) * state_scaled[k], 1)
            
        return pooled


class RangeCastEnsemble:
    """
    Weighted forecast ensemble combining:
    1. Naive / Seasonal Persistence Baseline (weight ~ 0.20)
    2. Negative Binomial GLM (weight ~ 0.45)
    3. Simple Damped Growth Rate (weight ~ 0.35)
    """
    
    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.20, 0.45, 0.35),
    ):
        self.w_base, self.w_nb, self.w_growth = weights
        self.m_base = NaivePersistenceBaseline()
        self.m_nb = NegativeBinomialModel()
        self.m_growth = SimpleGrowthRateModel()
        self.m_state_growth = SimpleGrowthRateModel()
        
    def fit(self, df_train: pd.DataFrame):
        """Fits all member models on historical data up to cutoff."""
        cases = df_train['cases'].values
        self.m_base.fit(cases)
        self.m_nb.fit(df_train)
        self.m_growth.fit(cases)
        
        # Fit state-level growth rate for partial pooling if available
        if 'state_cases' in df_train.columns and df_train['state_cases'].dropna().nunique() > 1:
            self.m_state_growth.fit(df_train['state_cases'].dropna().values)
        else:
            self.m_state_growth.fit(cases)

    def predict(
        self,
        horizon_weeks: int = 4,
        apply_pooling: bool = True,
        is_small_district: bool = False,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Produces ensemble quantiles [10, 25, 50, 75, 90] for horizons 1 to horizon_weeks.
        """
        results = []
        for h in range(1, horizon_weeks + 1):
            q_b = self.m_base.predict_quantiles(h)
            q_nb = self.m_nb.predict_quantiles(h)
            q_g = self.m_growth.predict_quantiles(h)
            
            # Weighted average across quantiles
            q_ens = {}
            for k in ['q10', 'q25', 'q50', 'q75', 'q90']:
                val = (
                    self.w_base * q_b[k] +
                    self.w_nb * q_nb[k] +
                    self.w_growth * q_g[k]
                )
                q_ens[k] = round(max(0.0, float(val)), 1)
                
            # Apply partial pooling if indicated
            if apply_pooling and is_small_district:
                q_state = self.m_state_growth.predict_quantiles(h)
                q_ens = SmallDistrictPooler.pool_quantiles(
                    district_quantiles=q_ens,
                    state_quantiles=q_state,
                    district_recent_cases=self.m_growth.current_cases,
                )
                
            results.append({
                "horizon": h,
                "q10": q_ens["q10"],
                "q25": q_ens["q25"],
                "q50": q_ens["q50"],
                "q75": q_ens["q75"],
                "q90": q_ens["q90"],
                "baseline_q10": q_b["q10"],
                "baseline_q25": q_b["q25"],
                "baseline_q50": q_b["q50"],
                "baseline_q75": q_b["q75"],
                "baseline_q90": q_b["q90"],
            })
            
        return results
