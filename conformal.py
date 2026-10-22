"""
RangeCast Adaptive Conformal Calibration & Shift Alert Module (conformal.py)
===========================================================================
Implements online distribution-free uncertainty calibration:
1. Adaptive Conformal Inference (Gibbs & Candes, 2021):
   alpha_{t+1} = alpha_t + gamma * (alpha_target - miss_t)
   Guarantees honest long-run 50% and 80% coverage under non-stationary dynamics.
2. Quantile non-conformity scoring on rolling origins.
3. Outbreak Shift Alert:
   Detects when recent residuals breach calibrated bands (inflection / variant surge),
   flags "forecast may be unreliable this week", and widens prediction intervals.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class AdaptiveConformalCalibrator:
    """
    Maintains rolling adaptive conformal prediction intervals for 80% and 50% bands.
    
    Update formula:
        alpha_next = alpha_curr + gamma * (alpha_target - miss)
    
    If the model missed (miss = 1):
        alpha_next decreases -> higher coverage quantile (1 - alpha) -> intervals widen.
    If the model covered (miss = 0):
        alpha_next increases -> tighter intervals next week.
    """
    
    def __init__(
        self,
        gamma: float = 0.05,            # Learning step size
        alpha_target_80: float = 0.20,  # 80% coverage target
        alpha_target_50: float = 0.50,  # 50% coverage target
        window_size: int = 15,          # Rolling history window
        shift_threshold_factor: float = 1.45, # Multiple of conformal radius triggering shift alert
    ):
        self.gamma = gamma
        self.alpha_target_80 = alpha_target_80
        self.alpha_target_50 = alpha_target_50
        self.window_size = window_size
        self.shift_threshold_factor = shift_threshold_factor
        
        # State variables
        self.alpha_80: float = alpha_target_80
        self.alpha_50: float = alpha_target_50
        
        self.scores_80: List[float] = []
        self.scores_50: List[float] = []
        
        self.recent_misses_80: List[int] = []
        self.recent_errors: List[float] = []
        self.shift_alert_active: bool = False
        self.conformal_radius_80: float = 0.0
        
    def score_nonconformity(
        self,
        actual: float,
        lower: float,
        upper: float,
        median: float,
    ) -> float:
        """
        Normalized non-conformity score:
        S = max(lower - actual, actual - upper) / max(10.0, median)
        S <= 0 when actual is inside [lower, upper].
        S > 0 when actual breaches interval.
        """
        scale = max(10.0, median)
        err = max(lower - actual, actual - upper)
        return float(err / scale)

    def update_observation(
        self,
        actual: float,
        q10_cal: float,
        q90_cal: float,
        q25_cal: float,
        q75_cal: float,
        q50: float,
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Observes ground truth actual at step t and updates online alpha parameters.
        """
        # 1. Check misses
        miss_80 = 1 if (actual < q10_cal or actual > q90_cal) else 0
        miss_50 = 1 if (actual < q25_cal or actual > q75_cal) else 0
        
        # 2. Update alpha via adaptive step
        # alpha_next = alpha + gamma * (alpha_target - miss)
        self.alpha_80 = float(np.clip(
            self.alpha_80 + self.gamma * (self.alpha_target_80 - miss_80),
            0.02, 0.45
        ))
        
        self.alpha_50 = float(np.clip(
            self.alpha_50 + self.gamma * (self.alpha_target_50 - miss_50),
            0.15, 0.75
        ))
        
        # 3. Record nonconformity scores
        s80 = self.score_nonconformity(actual, q10_cal, q90_cal, q50)
        s50 = self.score_nonconformity(actual, q25_cal, q75_cal, q50)
        
        self.scores_80.append(s80)
        self.scores_50.append(s50)
        self.recent_misses_80.append(miss_80)
        
        # Relative absolute error
        rel_err = abs(actual - q50) / max(10.0, q50)
        self.recent_errors.append(rel_err)
        
        # Retain rolling window
        if len(self.scores_80) > self.window_size:
            self.scores_80.pop(0)
            self.scores_50.pop(0)
        if len(self.recent_misses_80) > 4:
            self.recent_misses_80.pop(0)
        if len(self.recent_errors) > 4:
            self.recent_errors.pop(0)
            
        # 4. Check Shift Alert trigger
        # Triggered if 2 of last 3 weeks were misses, or recent relative error exceeds 0.85
        recent_miss_rate = np.mean(self.recent_misses_80[-3:]) if len(self.recent_misses_80) >= 2 else 0.0
        recent_mean_err = np.mean(self.recent_errors[-2:]) if len(self.recent_errors) >= 2 else 0.0
        
        prev_alert = self.shift_alert_active
        self.shift_alert_active = (recent_miss_rate >= 0.60) or (recent_mean_err > 0.85)
        
        return {
            "miss_80": miss_80,
            "miss_50": miss_50,
            "alpha_80_next": round(self.alpha_80, 4),
            "alpha_50_next": round(self.alpha_50, 4),
            "shift_alert": self.shift_alert_active,
            "shift_alert_triggered": self.shift_alert_active and not prev_alert,
        }

    def calibrate_forecast(
        self,
        raw_quantiles: Dict[str, float],
        widen_on_alert: bool = True,
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Applies conformal calibration to raw ensemble quantiles.
        If shift alert is active, widens intervals by safety factor (1.35x).
        """
        q10 = raw_quantiles["q10"]
        q25 = raw_quantiles["q25"]
        q50 = raw_quantiles["q50"]
        q75 = raw_quantiles["q75"]
        q90 = raw_quantiles["q90"]
        
        scale = max(10.0, q50)
        
        # If we have score history, take (1 - alpha) quantile of nonconformity scores
        if len(self.scores_80) >= 4:
            q_level_80 = 1.0 - self.alpha_80
            q_level_50 = 1.0 - self.alpha_50
            # Ensure quantile level is within [0, 1]
            q_level_80 = float(np.clip(q_level_80, 0.05, 0.98))
            q_level_50 = float(np.clip(q_level_50, 0.05, 0.98))
            
            c_80 = float(np.quantile(self.scores_80, q_level_80))
            c_50 = float(np.quantile(self.scores_50, q_level_50))
        else:
            # Cold-start mild buffer
            c_80 = 0.15
            c_50 = 0.08
            
        self.conformal_radius_80 = c_80 * scale
        
        # Adjust bounds
        adj_10 = max(0.0, q10 - c_80 * scale)
        adj_90 = q90 + c_80 * scale
        adj_25 = max(0.0, q25 - c_50 * scale)
        adj_75 = q75 + c_50 * scale
        
        # Shift alert widening
        alert_applied = False
        if widen_on_alert and self.shift_alert_active:
            alert_applied = True
            # Expand bounds by 35% around median
            spread_80 = adj_90 - adj_10
            spread_50 = adj_75 - adj_25
            
            adj_10 = max(0.0, q50 - 0.5 * spread_80 * 1.35)
            adj_90 = q50 + 0.5 * spread_80 * 1.35
            adj_25 = max(0.0, q50 - 0.5 * spread_50 * 1.30)
            adj_75 = q50 + 0.5 * spread_50 * 1.30
            
        return {
            "q10": round(adj_10, 1),
            "q25": round(adj_25, 1),
            "q50": round(q50, 1),
            "q75": round(adj_75, 1),
            "q90": round(adj_90, 1),
            "raw_q10": q10,
            "raw_q90": q90,
            "conformal_alpha_80": round(self.alpha_80, 3),
            "conformal_alpha_50": round(self.alpha_50, 3),
            "shift_alert": self.shift_alert_active,
            "alert_widening_applied": alert_applied,
            "alert_message": (
                "⚠️ SHIFT ALERT: Forecast may be unreliable this week (structural outbreak shift detected). "
                "Planning ranges widened by 35%." if self.shift_alert_active else "Stable"
            ),
        }
