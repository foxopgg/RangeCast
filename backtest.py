"""
RangeCast Rolling Backtest Engine (backtest.py)
==============================================
Executes temporal walk-forward backtests with rolling origins:
1. Strict no-lookahead training slices (df.iloc[:t])
2. Ensembling: Naive baseline + Negative Binomial GLM + Damped Growth Model
3. Online Adaptive Conformal Calibration: updates alpha each week based on hits/misses
4. Tracks Shift Alerts and interval widening
5. Generates headline scorecard metrics:
   - 80% and 50% coverage
   - Weighted Interval Score (WIS)
   - Relative WIS (rWIS) vs Persistence Baseline
   - Deterministic Trust Badges
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from data import Covid19IndiaLoader
from models import RangeCastEnsemble, NaivePersistenceBaseline
from conformal import AdaptiveConformalCalibrator
from scoring import (
    interval_score,
    weighted_interval_score,
    compute_scorecard_summary,
    assign_trust_badge,
)


class RollingBacktestRunner:
    """
    Simulates weekly decision-making for district health officers.
    Walks forward week by week across past outbreaks to test forecast calibration.
    """
    
    def __init__(
        self,
        min_train_weeks: int = 12,
        horizon_weeks: int = 4,
        conformal_gamma: float = 0.05,
    ):
        self.min_train_weeks = min_train_weeks
        self.horizon_weeks = horizon_weeks
        self.conformal_gamma = conformal_gamma
        
    def run_district_backtest(
        self,
        df_district: pd.DataFrame,
        district_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes rolling backtest for a single district.
        """
        df = df_district.sort_values('week_end_date').reset_index(drop=True)
        district = district_name or df['district'].iloc[0]
        n_weeks = len(df)
        
        if n_weeks <= self.min_train_weeks + 2:
            raise ValueError(
                f"Insufficient data for district {district}: {n_weeks} weeks found, "
                f"need at least {self.min_train_weeks + 3} weeks."
            )
            
        calibrator = AdaptiveConformalCalibrator(
            gamma=self.conformal_gamma,
            alpha_target_80=0.20,
            alpha_target_50=0.50,
        )
        
        records = []
        is_small = bool(df['is_small_district'].iloc[0]) if 'is_small_district' in df.columns else False
        
        # Roll forward through time
        for t in range(self.min_train_weeks, n_weeks - 1):
            train_slice = df.iloc[:t].copy()
            origin_date = df['week_end_date'].iloc[t - 1]
            
            # Fit ensemble up to origin
            ensemble = RangeCastEnsemble()
            ensemble.fit(train_slice)
            
            # Predict horizons 1..horizon_weeks
            preds = ensemble.predict(
                horizon_weeks=self.horizon_weeks,
                apply_pooling=True,
                is_small_district=is_small,
            )
            
            # Horizon 1 ground truth for calibrator online update
            h1_actual = float(df['cases'].iloc[t])
            h1_target_date = df['week_end_date'].iloc[t]
            
            # Calibrate each horizon
            for p in preds:
                h = p['horizon']
                target_idx = t + h - 1
                if target_idx >= n_weeks:
                    continue
                    
                target_date = df['week_end_date'].iloc[target_idx]
                actual_val = float(df['cases'].iloc[target_idx])
                
                # Apply conformal calibration
                cal_res = calibrator.calibrate_forecast(
                    raw_quantiles={
                        "q10": p["q10"],
                        "q25": p["q25"],
                        "q50": p["q50"],
                        "q75": p["q75"],
                        "q90": p["q90"],
                    },
                    widen_on_alert=True,
                )
                
                # Compute interval scores for this individual step
                is_80 = interval_score(actual_val, cal_res["q10"], cal_res["q90"], alpha=0.20)
                is_50 = interval_score(actual_val, cal_res["q25"], cal_res["q75"], alpha=0.50)
                wis_model = weighted_interval_score(
                    actual_val,
                    cal_res["q10"],
                    cal_res["q25"],
                    cal_res["q50"],
                    cal_res["q75"],
                    cal_res["q90"],
                )
                wis_base = weighted_interval_score(
                    actual_val,
                    p["baseline_q10"],
                    p["baseline_q25"],
                    p["baseline_q50"],
                    p["baseline_q75"],
                    p["baseline_q90"],
                )
                
                records.append({
                    "district": district,
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "horizon_weeks": h,
                    "actual": actual_val,
                    # Calibrated quantiles
                    "q10": cal_res["q10"],
                    "q25": cal_res["q25"],
                    "q50": cal_res["q50"],
                    "q75": cal_res["q75"],
                    "q90": cal_res["q90"],
                    # Raw quantiles
                    "raw_q10": p["q10"],
                    "raw_q90": p["q90"],
                    # Baseline quantiles
                    "baseline_q10": p["baseline_q10"],
                    "baseline_q25": p["baseline_q25"],
                    "baseline_q50": p["baseline_q50"],
                    "baseline_q75": p["baseline_q75"],
                    "baseline_q90": p["baseline_q90"],
                    # Conformal state
                    "conformal_alpha_80": cal_res["conformal_alpha_80"],
                    "conformal_alpha_50": cal_res["conformal_alpha_50"],
                    "shift_alert": cal_res["shift_alert"],
                    "alert_widening_applied": cal_res["alert_widening_applied"],
                    "is_small_district": is_small,
                    # Step scores
                    "is_80": is_80,
                    "is_50": is_50,
                    "wis_model": wis_model,
                    "wis_baseline": wis_base,
                    "covered_80": bool(cal_res["q10"] <= actual_val <= cal_res["q90"]),
                    "covered_50": bool(cal_res["q25"] <= actual_val <= cal_res["q75"]),
                })
                
            # Update calibrator online after step 1 outcome is realized
            h1_pred = preds[0]
            cal_h1 = calibrator.calibrate_forecast({
                "q10": h1_pred["q10"],
                "q25": h1_pred["q25"],
                "q50": h1_pred["q50"],
                "q75": h1_pred["q75"],
                "q90": h1_pred["q90"],
            })
            calibrator.update_observation(
                actual=h1_actual,
                q10_cal=cal_h1["q10"],
                q90_cal=cal_h1["q90"],
                q25_cal=cal_h1["q25"],
                q75_cal=cal_h1["q75"],
                q50=cal_h1["q50"],
            )
            
        df_eval = pd.DataFrame(records)
        return df_eval

    def generate_district_scorecard(self, df_eval: pd.DataFrame) -> Dict[str, Union[Dict, pd.DataFrame]]:
        """
        Generates overall and horizon-by-horizon scorecard summaries.
        """
        overall_summary = compute_scorecard_summary(df_eval)
        
        # Breakdown by horizon (1, 2, 3, 4 weeks)
        horizon_rows = []
        for h in sorted(df_eval['horizon_weeks'].unique()):
            sub = df_eval[df_eval['horizon_weeks'] == h]
            h_sum = compute_scorecard_summary(sub)
            horizon_rows.append({
                "horizon_weeks": h,
                "n_eval": h_sum["n_forecasts"],
                "coverage_80": h_sum["coverage_80"],
                "coverage_50": h_sum["coverage_50"],
                "mean_wis_model": h_sum["mean_wis_model"],
                "mean_wis_baseline": h_sum["mean_wis_baseline"],
                "relative_wis": h_sum["relative_wis"],
                "sharpness_80": h_sum["mean_sharpness_80"],
                "mae_model": h_sum["mae_model"],
                "mae_baseline": h_sum["mae_baseline"],
            })
            
        df_horizons = pd.DataFrame(horizon_rows)
        return {
            "overall": overall_summary,
            "by_horizon": df_horizons,
        }
