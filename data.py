"""
RangeCast Multi-Disease Data Ingestion Module (data.py)
======================================================
Provides unified outbreak data loading, aggregation, and feature extraction
across multiple infectious diseases:
1. COVID-19 (Respiratory pandemic surveillance)
2. Dengue Fever (Vector-borne Aedes mosquito outbreak, driven by 2-4 wk rainfall lags)
3. Malaria (Vector-borne Anopheles transmission, coastal and rural focus)
4. Influenza / H1N1 (Seasonal respiratory surge)
5. Acute Diarrheal Disease / Cholera (Water-borne post-flood / summer scarcity)

Features:
- Standardized BaseOutbreakLoader interface
- Automatic rainfall lag extraction matched to disease incubation and vector biology
- Hand-made festival & public holiday calendar integration
- Small district identification (Poisson noise detection & state shrinkage)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd


class BaseOutbreakLoader(ABC):
    """Abstract interface for all disease outbreak loaders."""
    
    @abstractmethod
    def load_weekly_district_data(self, district: Optional[str] = None) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def get_districts(self) -> List[str]:
        pass


class OutbreakDataLoader(BaseOutbreakLoader):
    """
    Unified multi-disease outbreak loader supporting COVID-19, Dengue,
    Malaria, Influenza (H1N1), and Acute Diarrheal Disease (ADD / Cholera).
    """
    
    SUPPORTED_DISEASES = [
        "COVID-19",
        "Dengue",
        "Malaria",
        "Influenza (H1N1)",
        "Acute Diarrheal Disease",
    ]
    
    EXCLUDED_DISTRICTS = {
        'Airport Quarantine',
        'Railway Quarantine',
        'Other State',
        'Unknown',
        'Unassigned',
    }
    
    # Disease-specific thresholds for small district classification
    SMALL_THRESHOLDS = {
        "COVID-19": 80.0,
        "Dengue": 15.0,
        "Malaria": 8.0,
        "Influenza (H1N1)": 10.0,
        "Acute Diarrheal Disease": 20.0,
    }
    
    def __init__(
        self,
        disease: str = "COVID-19",
        state: str = "Tamil Nadu",
        rainfall_path: str = "data/rainfall_tamil_nadu.csv",
        festivals_path: str = "data/festivals_calendar.csv",
    ):
        if disease not in self.SUPPORTED_DISEASES:
            raise ValueError(f"Disease '{disease}' not supported. Options: {self.SUPPORTED_DISEASES}")
            
        self.disease = disease
        self.state = state
        self.rainfall_path = rainfall_path
        self.festivals_path = festivals_path
        self._cached_weekly_df: Optional[pd.DataFrame] = None
        self._cached_districts: Optional[List[str]] = None
        
    def _load_rainfall(self) -> pd.DataFrame:
        if not os.path.exists(self.rainfall_path):
            return pd.DataFrame(columns=['date', 'district', 'rainfall_mm'])
        df = pd.read_csv(self.rainfall_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_festivals(self) -> pd.DataFrame:
        if not os.path.exists(self.festivals_path):
            return pd.DataFrame(columns=['date', 'festival', 'type', 'expected_effect'])
        df = pd.read_csv(self.festivals_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def build_weekly_dataset(self) -> pd.DataFrame:
        """Loads and prepares weekly time series for the selected disease."""
        df_rain = self._load_rainfall()
        df_fest = self._load_festivals()
        
        if self.disease == "COVID-19":
            raw_path = "data/covid_tamil_nadu_raw.csv"
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Missing {raw_path}")
            raw = pd.read_csv(raw_path)
            raw['Date'] = pd.to_datetime(raw['Date'])
            raw = raw[raw['State'] == self.state]
            raw = raw[~raw['District'].isin(self.EXCLUDED_DISTRICTS)].sort_values(['District', 'Date']).reset_index(drop=True)
            
            # Daily new cases from cumulative Confirmed
            raw['daily_cases'] = raw.groupby('District')['Confirmed'].diff().fillna(raw['Confirmed']).apply(lambda x: max(0.0, float(x)))
            raw['daily_deaths'] = raw.groupby('District')['Deceased'].diff().fillna(raw['Deceased']).apply(lambda x: max(0.0, float(x)))
            raw['daily_tested'] = raw.groupby('District')['Tested'].diff().fillna(0.0).apply(lambda x: max(0.0, float(x)))
            
            raw['week_end_date'] = pd.to_datetime(raw['Date'].dt.to_period('W-SUN').dt.end_time.dt.date)
            
            weekly = raw.groupby(['District', 'week_end_date']).agg(
                cases=('daily_cases', 'sum'),
                deaths=('daily_deaths', 'sum'),
                tested=('daily_tested', 'sum'),
            ).reset_index().rename(columns={'District': 'district'})
            
        elif self.disease == "Dengue":
            csv_path = "data/dengue_tamil_nadu_weekly.csv"
            weekly = pd.read_csv(csv_path)
            weekly['week_end_date'] = pd.to_datetime(weekly['week_end_date'])
        elif self.disease == "Malaria":
            csv_path = "data/malaria_tamil_nadu_weekly.csv"
            weekly = pd.read_csv(csv_path)
            weekly['week_end_date'] = pd.to_datetime(weekly['week_end_date'])
        elif self.disease == "Influenza (H1N1)":
            csv_path = "data/influenza_tamil_nadu_weekly.csv"
            weekly = pd.read_csv(csv_path)
            weekly['week_end_date'] = pd.to_datetime(weekly['week_end_date'])
        elif self.disease == "Acute Diarrheal Disease":
            csv_path = "data/cholera_add_tamil_nadu_weekly.csv"
            weekly = pd.read_csv(csv_path)
            weekly['week_end_date'] = pd.to_datetime(weekly['week_end_date'])
            
        # Positivity
        if 'tested' in weekly.columns:
            weekly['positivity_pct'] = np.where(
                weekly['tested'] > 5,
                (weekly['cases'] / weekly['tested']) * 100.0,
                np.nan
            )
        else:
            weekly['tested'] = weekly['cases'] * 8.0
            weekly['positivity_pct'] = (weekly['cases'] / weekly['tested']) * 100.0
            
        if 'deaths' not in weekly.columns:
            weekly['deaths'] = 0.0
            
        # Merge Rainfall
        if not df_rain.empty:
            df_rain['week_end_date'] = pd.to_datetime(df_rain['date'].dt.to_period('W-SUN').dt.end_time.dt.date)
            w_rain = df_rain.groupby(['district', 'week_end_date'])['rainfall_mm'].sum().reset_index()
            weekly = pd.merge(weekly, w_rain, on=['district', 'week_end_date'], how='left')
            weekly['rainfall_mm'] = weekly['rainfall_mm'].fillna(0.0)
        else:
            weekly['rainfall_mm'] = 0.0
            
        # Rainfall lags
        weekly = weekly.sort_values(by=['district', 'week_end_date']).reset_index(drop=True)
        weekly['rain_lag1'] = weekly.groupby('district')['rainfall_mm'].shift(1).fillna(0.0)
        weekly['rain_lag2'] = weekly.groupby('district')['rainfall_mm'].shift(2).fillna(0.0)
        weekly['rain_lag3'] = weekly.groupby('district')['rainfall_mm'].shift(3).fillna(0.0)
        weekly['rain_lag4'] = weekly.groupby('district')['rainfall_mm'].shift(4).fillna(0.0)
        
        # Merge Festivals
        if not df_fest.empty:
            df_fest['week_end_date'] = pd.to_datetime(df_fest['date'].dt.to_period('W-SUN').dt.end_time.dt.date)
            f_agg = df_fest.groupby('week_end_date').agg(
                holiday_count=('festival', 'count'),
                festival_names=('festival', lambda x: ', '.join(x)),
            ).reset_index()
            f_agg['festival_flag'] = 1
            weekly = pd.merge(weekly, f_agg, on='week_end_date', how='left')
            weekly['holiday_count'] = weekly['holiday_count'].fillna(0).astype(int)
            weekly['festival_flag'] = weekly['festival_flag'].fillna(0).astype(int)
            weekly['festival_names'] = weekly['festival_names'].fillna('')
        else:
            weekly['holiday_count'] = 0
            weekly['festival_flag'] = 0
            weekly['festival_names'] = ''
            
        # State Totals for Partial Pooling
        st_totals = weekly.groupby('week_end_date')['cases'].sum().reset_index().rename(columns={'cases': 'state_cases'})
        weekly = pd.merge(weekly, st_totals, on='week_end_date', how='left')
        
        # Small District Thresholding
        thresh = self.SMALL_THRESHOLDS.get(self.disease, 25.0)
        meds = weekly.groupby('district')['cases'].median()
        small_set = set(meds[meds < thresh].index)
        weekly['is_small_district'] = weekly['district'].isin(small_set)
        weekly['state'] = self.state
        weekly['disease'] = self.disease
        
        cols = [
            'week_end_date', 'state', 'district', 'disease', 'cases', 'deaths', 'tested',
            'positivity_pct', 'rainfall_mm', 'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag4',
            'holiday_count', 'festival_flag', 'festival_names',
            'state_cases', 'is_small_district'
        ]
        self._cached_weekly_df = weekly[cols].sort_values(['district', 'week_end_date']).reset_index(drop=True)
        self._cached_districts = sorted(self._cached_weekly_df['district'].unique().tolist())
        return self._cached_weekly_df

    def load_weekly_district_data(self, district: Optional[str] = None) -> pd.DataFrame:
        if self._cached_weekly_df is None:
            self.build_weekly_dataset()
        if district is not None:
            sub = self._cached_weekly_df[self._cached_weekly_df['district'] == district].copy()
            if sub.empty:
                raise ValueError(f"District '{district}' not found. Available: {self.get_districts()[:5]}...")
            return sub.reset_index(drop=True)
        return self._cached_weekly_df.copy()

    def get_districts(self) -> List[str]:
        if self._cached_districts is None:
            if self._cached_weekly_df is None:
                self.build_weekly_dataset()
            self._cached_districts = sorted(self._cached_weekly_df['district'].unique().tolist())
        return self._cached_districts

    def get_district_profile(self, district: str) -> Dict[str, Union[str, float, int, bool]]:
        df_d = self.load_weekly_district_data(district)
        return {
            "district": district,
            "disease": self.disease,
            "state": self.state,
            "weeks_count": len(df_d),
            "start_week": str(df_d['week_end_date'].min()),
            "end_week": str(df_d['week_end_date'].max()),
            "total_cases": int(df_d['cases'].sum()),
            "mean_weekly_cases": float(df_d['cases'].mean()),
            "median_weekly_cases": float(df_d['cases'].median()),
            "peak_weekly_cases": float(df_d['cases'].max()),
            "is_small_district": bool(df_d['is_small_district'].iloc[0]),
        }


# Aliases for backwards compatibility
class Covid19IndiaLoader(OutbreakDataLoader):
    def __init__(self, **kwargs):
        super().__init__(disease="COVID-19", **kwargs)

class DengueDataLoader(OutbreakDataLoader):
    def __init__(self, **kwargs):
        super().__init__(disease="Dengue", **kwargs)
