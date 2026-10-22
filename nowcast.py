"""
RangeCast Cleaning & Nowcasting Module (nowcast.py)
==================================================
Handles surveillance artifacts before forecasting:
1. Holiday / Reporting Dip Adjustment: Dampens artificial troughs on festival weeks
2. Late-Report Correction via Reporting-Completeness Curve:
   - Log-logistic delay model C(d) = 1 / (1 + (d / theta)^(-kappa))
   - Explicitly simulates reporting delay snapshots if raw vintage snapshots are absent
   - Prominently labeled with [SIMULATED REPORTING DELAY]
3. Test Positivity Rate (TPR) Tracking and WHO 5% threshold surveillance
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class OutbreakNowcaster:
    """
    Surveillance nowcaster correcting for late reporting and holiday dips.
    """
    
    LABEL_SIMULATED = "[SIMULATED REPORTING DELAY]"
    LABEL_NOWCAST_ADJUSTED = "[NOWCAST ADJUSTED]"
    
    def __init__(
        self,
        delay_theta: float = 4.5,   # Median reporting delay parameter in days
        delay_kappa: float = 2.2,   # Dispersion parameter
        holiday_dip_factor: float = 0.22, # Typical 22% reporting drop on holiday weeks
    ):
        self.delay_theta = delay_theta
        self.delay_kappa = delay_kappa
        self.holiday_dip_factor = holiday_dip_factor
        
    def reporting_completeness_curve(self, days_elapsed: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Log-logistic reporting completeness cumulative distribution function:
            C(d) = 1 / (1 + (d / theta)^(-kappa))
        
        At d=0: 0.0 (nothing reported instantly)
        At d=theta (4.5 days): 0.50 (50% complete)
        At d=7 days: ~76% complete
        At d=14 days: ~94% complete
        At d=21 days: ~99% complete
        """
        d = np.maximum(0.01, np.asarray(days_elapsed, dtype=float))
        completeness = 1.0 / (1.0 + (d / self.delay_theta) ** (-self.delay_kappa))
        if np.isscalar(days_elapsed):
            return float(completeness)
        return completeness

    def simulate_reporting_delays(
        self,
        df_weekly: pd.DataFrame,
        current_cutoff_date: pd.Timestamp,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Simulates the vintage snapshot of data available to a Health Officer
        at `current_cutoff_date`.
        
        Recent weeks are only partially reported according to the completeness curve.
        Labeled explicitly with [SIMULATED REPORTING DELAY].
        """
        df = df_weekly.copy()
        rng = np.random.RandomState(seed)
        
        # Calculate days elapsed since week end date
        df['days_elapsed'] = (current_cutoff_date - pd.to_datetime(df['week_end_date'])).dt.days
        
        # Future weeks (after cutoff) have not occurred yet
        df = df[df['days_elapsed'] >= 0].copy()
        
        # Calculate completeness
        df['reporting_completeness'] = self.reporting_completeness_curve(df['days_elapsed'])
        # Cap at 1.0 for older weeks
        df['reporting_completeness'] = np.where(df['days_elapsed'] >= 21, 1.0, df['reporting_completeness'])
        
        # Reported cases at decision time
        # Add slight binomial/poisson stochasticity to simulated reporting
        reported_cases = []
        for _, row in df.iterrows():
            comp = row['reporting_completeness']
            true_cases = int(round(row['cases']))
            if comp >= 0.999 or true_cases == 0:
                rep = true_cases
            else:
                rep = rng.binomial(n=true_cases, p=min(0.999, max(0.05, comp)))
            reported_cases.append(rep)
            
        df['reported_cases_at_cutoff'] = reported_cases
        df['delay_simulation_label'] = self.LABEL_SIMULATED
        return df

    def adjust_for_reporting_delays(
        self,
        df_snapshot: pd.DataFrame,
        min_completeness_threshold: float = 0.25,
    ) -> pd.DataFrame:
        """
        Nowcasts true cases by inflating recently reported cases by 1 / completeness.
        Inflates uncertainty as completeness drops.
        """
        df = df_snapshot.copy()
        
        cases_col = 'reported_cases_at_cutoff' if 'reported_cases_at_cutoff' in df.columns else 'cases'
        comp = np.clip(df['reporting_completeness'].values, min_completeness_threshold, 1.0)
        
        # Inflated nowcast estimate
        nowcast_cases = df[cases_col].values / comp
        df['nowcast_cases'] = np.round(nowcast_cases, 1)
        
        # Nowcast variance inflation factor
        df['nowcast_std_error'] = np.sqrt(np.maximum(1.0, df[cases_col].values)) * (1.0 / comp)
        df['nowcast_label'] = self.LABEL_NOWCAST_ADJUSTED
        return df

    def adjust_for_holiday_dips(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts for artificial testing/reporting dips during holiday/festival weeks.
        
        If a week has holiday_count > 0 or festival_flag == 1, observed cases
        frequently undercount the true community burden by ~20-25%.
        """
        df = df_weekly.copy()
        dip_factor = self.holiday_dip_factor
        
        # Adjustment multiplier
        # 1 holiday -> 1 / (1 - 0.15)
        # 2+ holidays -> 1 / (1 - 0.25)
        mult = np.ones(len(df), dtype=float)
        if 'holiday_count' in df.columns:
            h_count = df['holiday_count'].values
            mult = np.where(h_count == 1, 1.0 / (1.0 - dip_factor * 0.7), mult)
            mult = np.where(h_count >= 2, 1.0 / (1.0 - dip_factor), mult)
            
        df['holiday_adjusted_cases'] = np.round(df['cases'].values * mult, 1)
        df['holiday_adjustment_factor'] = mult
        return df

    def track_positivity_trend(self, df_weekly: pd.DataFrame) -> Dict[str, Union[float, str, bool]]:
        """
        Analyzes the latest positivity rate (TPR) and compares against WHO 5% alert line.
        """
        if 'positivity_pct' not in df_weekly.columns or df_weekly['positivity_pct'].dropna().empty:
            # Synthetic positivity estimate based on case acceleration if raw tests absent
            cases = df_weekly['cases'].values
            if len(cases) >= 2 and cases[-2] > 0:
                growth = (cases[-1] - cases[-2]) / cases[-2]
                proxy_tpr = min(35.0, max(1.5, 4.0 * (1.0 + growth)))
            else:
                proxy_tpr = 4.5
            return {
                "latest_tpr": round(proxy_tpr, 2),
                "tpr_alert": proxy_tpr > 5.0,
                "tpr_status": "HIGH ALERT (>5% WHO limit)" if proxy_tpr > 5.0 else "Controlled (<5%)",
                "is_proxy": True,
            }
            
        valid_tpr = df_weekly['positivity_pct'].dropna()
        latest = float(valid_tpr.iloc[-1])
        prev = float(valid_tpr.iloc[-2]) if len(valid_tpr) >= 2 else latest
        delta = latest - prev
        
        return {
            "latest_tpr": round(latest, 2),
            "prev_tpr": round(prev, 2),
            "delta_tpr": round(delta, 2),
            "tpr_alert": latest > 5.0,
            "tpr_status": "HIGH SPREAD (>5% WHO threshold)" if latest > 5.0 else "Within Controlled Bounds (<=5%)",
            "is_proxy": False,
        }
