"""
RangeCast Scoring & Verification Module (scoring.py)
===================================================
Implements strictly proper scoring rules for quantile forecasts and interval predictions:
1. Interval Score (IS_alpha) as defined by Gneiting & Raftery (2007)
2. Weighted Interval Score (WIS) as standardized by CDC COVID-19 Forecast Hub
3. Empirical Coverage Tracking for 50% and 80% prediction intervals
4. Relative WIS (rWIS) against a "same as last week" persistence baseline
5. Deterministic Trust Badge assignment (High / Medium / Low) with transparent criteria
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def interval_score(
    y_true: Union[float, np.ndarray, pd.Series],
    lower: Union[float, np.ndarray, pd.Series],
    upper: Union[float, np.ndarray, pd.Series],
    alpha: float,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes the interval score for a central (1 - alpha) prediction interval [lower, upper].
    
    Formula:
        IS_alpha(y) = (upper - lower) 
                    + (2 / alpha) * (lower - y) * I(y < lower)
                    + (2 / alpha) * (y - upper) * I(y > upper)
                    
    Parameters:
        y_true: Ground truth observed value(s)
        lower: Lower bound quantile (e.g. q_0.10 for alpha=0.20)
        upper: Upper bound quantile (e.g. q_0.90 for alpha=0.20)
        alpha: Miscoverage rate (e.g. 0.20 for 80% interval, 0.50 for 50% interval)
        
    Returns:
        Interval score(s). Lower values indicate better calibrated, sharper forecasts.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    
    sharpness = u - l
    under_penalty = (2.0 / alpha) * np.maximum(0.0, l - y)
    over_penalty = (2.0 / alpha) * np.maximum(0.0, y - u)
    
    score = sharpness + under_penalty + over_penalty
    if np.isscalar(y_true) and np.isscalar(lower) and np.isscalar(upper):
        return float(score)
    return score


def weighted_interval_score(
    y_true: Union[float, np.ndarray, pd.Series],
    q10: Union[float, np.ndarray, pd.Series],
    q25: Union[float, np.ndarray, pd.Series],
    q50: Union[float, np.ndarray, pd.Series],
    q75: Union[float, np.ndarray, pd.Series],
    q90: Union[float, np.ndarray, pd.Series],
    return_components: bool = False,
) -> Union[float, np.ndarray, Dict[str, Union[float, np.ndarray]]]:
    """
    Computes the Weighted Interval Score (WIS) across the standard quantiles:
    [0.10, 0.25, 0.50, 0.75, 0.90].
    
    Components:
    - 80% interval: [q10, q90] with alpha=0.20, weight = alpha / 2 = 0.10
    - 50% interval: [q25, q75] with alpha=0.50, weight = alpha / 2 = 0.25
    - Median: q50 with weight = 0.5 * |y - q50|
    
    Normalization:
        K = 2 intervals (80%, 50%).
        Total normalizer = 1 / (K + 0.5) = 1 / 2.5
    """
    y = np.asarray(y_true, dtype=float)
    _q10 = np.asarray(q10, dtype=float)
    _q25 = np.asarray(q25, dtype=float)
    _q50 = np.asarray(q50, dtype=float)
    _q75 = np.asarray(q75, dtype=float)
    _q90 = np.asarray(q90, dtype=float)
    
    # 1. Median absolute error
    med_error = np.abs(y - _q50)
    
    # 2. 50% interval score (alpha = 0.50)
    is_50 = interval_score(y, _q25, _q75, alpha=0.50)
    
    # 3. 80% interval score (alpha = 0.20)
    is_80 = interval_score(y, _q10, _q90, alpha=0.20)
    
    # Weighted sum:
    # weight_k = alpha_k / 2
    # For alpha=0.50: weight = 0.25
    # For alpha=0.20: weight = 0.10
    # For median: weight = 0.50
    wis = (0.50 * med_error + 0.25 * is_50 + 0.10 * is_80) / (2.0 + 0.5)
    
    if return_components:
        # Sharpness vs under-penalty vs over-penalty decomposition
        sharpness = (
            0.50 * 0.0 + 
            0.25 * (_q75 - _q25) + 
            0.10 * (_q90 - _q10)
        ) / 2.5
        
        under_penalty = (
            0.50 * np.maximum(0.0, _q50 - y) +
            0.25 * (2.0 / 0.50) * np.maximum(0.0, _q25 - y) +
            0.10 * (2.0 / 0.20) * np.maximum(0.0, _q10 - y)
        ) / 2.5
        
        over_penalty = (
            0.50 * np.maximum(0.0, y - _q50) +
            0.25 * (2.0 / 0.50) * np.maximum(0.0, y - _q75) +
            0.10 * (2.0 / 0.20) * np.maximum(0.0, y - _q90)
        ) / 2.5
        
        return {
            "wis": float(np.mean(wis)) if np.isscalar(y_true) else wis,
            "sharpness": float(np.mean(sharpness)),
            "under_penalty": float(np.mean(under_penalty)),
            "over_penalty": float(np.mean(over_penalty)),
            "mae": float(np.mean(med_error)),
        }
        
    if np.isscalar(y_true) and np.isscalar(q10):
        return float(wis)
    return wis


def calculate_coverage(
    y_true: Union[np.ndarray, pd.Series],
    lower: Union[np.ndarray, pd.Series],
    upper: Union[np.ndarray, pd.Series],
) -> float:
    """
    Computes empirical coverage: proportion of actuals within [lower, upper].
    """
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)
    valid = ~(np.isnan(y) | np.isnan(l) | np.isnan(u))
    if np.sum(valid) == 0:
        return 0.0
    covered = (y[valid] >= l[valid]) & (y[valid] <= u[valid])
    return float(np.mean(covered))


def assign_trust_badge(
    coverage_80: float,
    coverage_50: float,
    relative_wis: float,
    recent_shift_alert: bool = False,
    low_volume_warning: bool = False,
    eval_periods: int = 10,
) -> Dict[str, Union[str, List[str], float]]:
    """
    Deterministic trust scorecard classification for district health officers.
    
    Rules:
    - HIGH TRUST:
      1. Empirical 80% coverage >= 75%
      2. Relative WIS <= 0.85 (>= 15% better than persistence baseline)
      3. No recent shift alert in recent weeks
      4. Not a low-volume noisy district
      5. Evaluation sample size >= 6 weeks
      
    - MEDIUM TRUST:
      1. Empirical 80% coverage >= 60%
      2. Relative WIS <= 1.05 (competitive with or better than baseline)
      3. Not permanently disqualified by active shift alert
      4. Sample size >= 4 weeks
      
    - LOW TRUST:
      Triggered if coverage < 60%, or rWIS > 1.05 (worse than baseline),
      or active shift alert detected, or insufficient backtest history.
    """
    reasons: List[str] = []
    positive_signals: List[str] = []
    
    # Check sample size
    if eval_periods < 4:
        reasons.append(f"Insufficient rolling evaluation history ({eval_periods} weeks; min 4 required).")
    elif eval_periods < 6:
        reasons.append(f"Limited evaluation history ({eval_periods} weeks).")
    else:
        positive_signals.append(f"Solid evaluation depth ({eval_periods} historical rolling origins).")
        
    # Check 80% coverage
    if coverage_80 >= 0.75:
        positive_signals.append(f"Honest 80% range coverage ({coverage_80:.1%}; target >= 75%).")
    elif coverage_80 >= 0.60:
        reasons.append(f"80% range coverage is slightly under-calibrated ({coverage_80:.1%}).")
    else:
        reasons.append(f"Poor 80% coverage ({coverage_80:.1%}; ranges too narrow or frequently breached).")
        
    # Check 50% coverage
    if coverage_50 >= 0.40 and coverage_50 <= 0.65:
        positive_signals.append(f"Well-balanced 50% range ({coverage_50:.1%}; target ~50%).")
    elif coverage_50 < 0.35:
        reasons.append(f"50% inner range too narrow ({coverage_50:.1%}).")
        
    # Check Relative WIS vs persistence
    if relative_wis <= 0.85:
        pct_better = (1.0 - relative_wis) * 100.0
        positive_signals.append(f"Consistently beats persistence baseline by {pct_better:.0f}% (rWIS = {relative_wis:.2f}).")
    elif relative_wis <= 1.05:
        reasons.append(f"Forecast accuracy comparable to 'same-as-last-week' baseline (rWIS = {relative_wis:.2f}).")
    else:
        pct_worse = (relative_wis - 1.0) * 100.0
        reasons.append(f"Performs {pct_worse:.0f}% worse than persistence baseline (rWIS = {relative_wis:.2f}).")
        
    # Check Shift Alert
    if recent_shift_alert:
        reasons.append("Active Shift Alert: Recent forecast errors breached calibrated bands.")
        
    # Check Small District Volume
    if low_volume_warning:
        reasons.append("Small District Warning: Low weekly case counts introduce Poisson noise.")
        
    # Decision tree
    is_high = (
        eval_periods >= 6 and
        coverage_80 >= 0.75 and
        relative_wis <= 0.85 and
        not recent_shift_alert and
        not low_volume_warning
    )
    
    is_medium = (
        not is_high and
        eval_periods >= 4 and
        coverage_80 >= 0.60 and
        relative_wis <= 1.05 and
        not recent_shift_alert
    )
    
    if is_high:
        tier = "HIGH"
        color = "#10B981"  # Emerald green
        badge_text = "HIGH TRUST"
        summary = "Reliable calibration: honest coverage and consistently outperforms baseline."
    elif is_medium:
        tier = "MEDIUM"
        color = "#F59E0B"  # Amber
        badge_text = "MODERATE TRUST"
        summary = "Acceptable range calibration, but monitor closely or account for slight variance."
    else:
        tier = "LOW"
        color = "#EF4444"  # Rose red
        badge_text = "LOW TRUST - CAUTION"
        summary = "Uncertain forecast: historical ranges violated or accuracy trailing baseline."
        
    return {
        "tier": tier,
        "badge_text": badge_text,
        "color": color,
        "summary": summary,
        "reasons": reasons,
        "positive_signals": positive_signals,
        "coverage_80": coverage_80,
        "coverage_50": coverage_50,
        "relative_wis": relative_wis,
    }


def compute_scorecard_summary(df_eval: pd.DataFrame) -> Dict[str, Union[float, Dict]]:
    """
    Aggregates backtest evaluation results across multiple weeks/horizons.
    
    Expected columns in df_eval:
    ['actual', 'q10', 'q25', 'q50', 'q75', 'q90', 'baseline_q10', ..., 'baseline_q90']
    """
    clean_df = df_eval.dropna(subset=['actual', 'q10', 'q90', 'baseline_q10', 'baseline_q90']).copy()
    if len(clean_df) == 0:
        return {
            "n_forecasts": 0,
            "coverage_80": 0.0,
            "coverage_50": 0.0,
            "mean_wis_model": 0.0,
            "mean_wis_baseline": 0.0,
            "relative_wis": 1.0,
            "mae_model": 0.0,
            "mae_baseline": 0.0,
            "mean_sharpness_80": 0.0,
            "trust_badge": assign_trust_badge(0, 0, 1.0, eval_periods=0),
        }
        
    cov_80 = calculate_coverage(clean_df['actual'], clean_df['q10'], clean_df['q90'])
    cov_50 = calculate_coverage(clean_df['actual'], clean_df['q25'], clean_df['q75'])
    
    wis_model = weighted_interval_score(
        clean_df['actual'].values,
        clean_df['q10'].values,
        clean_df['q25'].values,
        clean_df['q50'].values,
        clean_df['q75'].values,
        clean_df['q90'].values,
    )
    
    wis_base = weighted_interval_score(
        clean_df['actual'].values,
        clean_df['baseline_q10'].values,
        clean_df['baseline_q25'].values,
        clean_df['baseline_q50'].values,
        clean_df['baseline_q75'].values,
        clean_df['baseline_q90'].values,
    )
    
    mean_wis_m = float(np.mean(wis_model))
    mean_wis_b = float(np.mean(wis_base))
    # Protect against 0 division
    r_wis = mean_wis_m / max(mean_wis_b, 1e-6)
    
    mae_m = float(np.mean(np.abs(clean_df['actual'].values - clean_df['q50'].values)))
    mae_b = float(np.mean(np.abs(clean_df['actual'].values - clean_df['baseline_q50'].values)))
    sharpness_80 = float(np.mean(clean_df['q90'].values - clean_df['q10'].values))
    
    n_origins = clean_df['origin_date'].nunique() if 'origin_date' in clean_df.columns else len(clean_df)
    
    badge = assign_trust_badge(
        coverage_80=cov_80,
        coverage_50=cov_50,
        relative_wis=r_wis,
        recent_shift_alert=clean_df.get('shift_alert', pd.Series([False]*len(clean_df))).iloc[-1] if len(clean_df) else False,
        low_volume_warning=clean_df.get('low_volume', pd.Series([False]*len(clean_df))).iloc[-1] if len(clean_df) else False,
        eval_periods=n_origins,
    )
    
    return {
        "n_forecasts": len(clean_df),
        "n_origins": n_origins,
        "coverage_80": cov_80,
        "coverage_50": cov_50,
        "mean_wis_model": mean_wis_m,
        "mean_wis_baseline": mean_wis_b,
        "relative_wis": r_wis,
        "mae_model": mae_m,
        "mae_baseline": mae_b,
        "mean_sharpness_80": sharpness_80,
        "trust_badge": badge,
    }
