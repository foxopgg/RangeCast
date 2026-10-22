"""
RangeCast Multi-Disease Resource Planner Module (planner.py)
============================================================
Translates probabilistic outbreak case projections into operational hospital
and logistics procurement orders for District Health Officers.

Supported Outbreak Diseases & Specific Supply Chains:
1. COVID-19: Isolation beds, ICU beds, D-type oxygen cylinders (7,000L), Liquid O2, RT-PCR kits.
2. Dengue Fever: Dengue ward beds, HDU beds, Platelet units (RDP/SDP), IV fluids (Ringer's/Saline), NS1/IgM ELISA kits.
3. Malaria: Inpatient beds, Rapid Diagnostic Tests (RDTs), ACT / Chloroquine courses, Bed nets (LLIN).
4. Influenza (H1N1): Respiratory isolation beds, O2 beds, Oseltamivir (Tamiflu) antiviral courses, VTM swab kits.
5. Acute Diarrheal Disease (Cholera/ADD): Observation beds, ORS sachets, IV Ringer's bottles, Chlorine purification tablets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PlannerAssumptions:
    """
    Editable clinical and operational parameters per disease.
    Default values serve as realistic placeholders for district health planning,
    NOT immutable statutory norms.
    """
    disease: str = "COVID-19"
    hospitalization_rate: float = 0.065   # % of confirmed cases requiring admission
    icu_fraction: float = 0.18           # % of admitted patients requiring ICU / HDU care
    oxygen_bed_fraction: float = 0.60    # % of admitted non-ICU patients requiring oxygen
    
    avg_length_of_stay_days: float = 8.5 # Average hospital stay (ALOS)
    icu_length_of_stay_days: float = 11.0 # Average ICU stay
    
    # Oxygen consumption: 1 D-type cylinder contains ~7,000 liters (7 m3)
    o2_liters_per_min_icu: float = 24.0
    o2_liters_per_min_ward: float = 8.0
    cylinder_capacity_liters: float = 7000.0
    
    # Diagnostics & Disease-Specific Supplies
    tests_per_confirmed_case: float = 18.0
    kit_wastage_factor: float = 0.08
    
    # Disease-specific specialized rates:
    # Dengue
    platelets_per_icu_patient_day: float = 1.5   # Units/day
    iv_fluid_bottles_per_admission: float = 6.0  # 500ml bottles
    
    # Malaria
    act_courses_per_case: float = 1.0           # Treatment courses
    rdt_tests_per_case: float = 12.0
    
    # Influenza
    tamiflu_courses_per_case: float = 2.0       # Patient + close contact prophylaxis
    
    # Acute Diarrhea
    ors_sachets_per_case: float = 6.0           # Sachets
    iv_ringer_per_severe_case: float = 4.0      # IV bottles
    chlorine_tabs_per_case: float = 30.0        # Halazone purification tablets
    
    # Supply chain lead times
    supplier_lead_time_weeks: int = 2


def get_default_disease_assumptions(disease: str = "COVID-19") -> PlannerAssumptions:
    """Returns disease-specific standard clinical planning placeholders."""
    if disease == "Dengue":
        return PlannerAssumptions(
            disease="Dengue",
            hospitalization_rate=0.085,
            icu_fraction=0.15,
            oxygen_bed_fraction=0.10,
            avg_length_of_stay_days=5.5,
            icu_length_of_stay_days=7.0,
            o2_liters_per_min_icu=12.0,
            o2_liters_per_min_ward=4.0,
            tests_per_confirmed_case=4.0, # NS1 / IgM ELISA
            supplier_lead_time_weeks=2,
            platelets_per_icu_patient_day=1.5,
            iv_fluid_bottles_per_admission=6.0,
        )
    elif disease == "Malaria":
        return PlannerAssumptions(
            disease="Malaria",
            hospitalization_rate=0.040,
            icu_fraction=0.08,
            oxygen_bed_fraction=0.05,
            avg_length_of_stay_days=4.0,
            icu_length_of_stay_days=5.0,
            tests_per_confirmed_case=12.0, # RDTs
            supplier_lead_time_weeks=2,
            act_courses_per_case=1.0,
        )
    elif disease == "Influenza (H1N1)":
        return PlannerAssumptions(
            disease="Influenza (H1N1)",
            hospitalization_rate=0.055,
            icu_fraction=0.12,
            oxygen_bed_fraction=0.45,
            avg_length_of_stay_days=6.0,
            icu_length_of_stay_days=8.0,
            o2_liters_per_min_icu=20.0,
            o2_liters_per_min_ward=6.0,
            tests_per_confirmed_case=6.0,
            supplier_lead_time_weeks=2,
            tamiflu_courses_per_case=2.0,
        )
    elif disease == "Acute Diarrheal Disease":
        return PlannerAssumptions(
            disease="Acute Diarrheal Disease",
            hospitalization_rate=0.120,
            icu_fraction=0.18, # Severe dehydration
            oxygen_bed_fraction=0.02,
            avg_length_of_stay_days=3.0,
            icu_length_of_stay_days=4.0,
            tests_per_confirmed_case=2.0,
            supplier_lead_time_weeks=1,
            ors_sachets_per_case=6.0,
            iv_ringer_per_severe_case=4.0,
            chlorine_tabs_per_case=30.0,
        )
    else: # COVID-19
        return PlannerAssumptions(
            disease="COVID-19",
            hospitalization_rate=0.065,
            icu_fraction=0.18,
            oxygen_bed_fraction=0.60,
            avg_length_of_stay_days=8.5,
            icu_length_of_stay_days=11.0,
            o2_liters_per_min_icu=24.0,
            o2_liters_per_min_ward=8.0,
            tests_per_confirmed_case=18.0,
            supplier_lead_time_weeks=2,
        )


def compute_newsvendor_fractile(underage_cost: float, overage_cost: float) -> float:
    """Computes the optimal risk percentile: q = Cu / (Cu + Co)."""
    if underage_cost <= 0 or overage_cost <= 0:
        return 0.75
    return float(underage_cost / (underage_cost + overage_cost))


def interpolate_quantile_case_target(
    quantiles: Dict[str, float],
    risk_fractile: float,
) -> float:
    """Interpolates the target case count corresponding to risk_fractile q."""
    q_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_vals = [
        quantiles.get("q10", 0.0),
        quantiles.get("q25", 0.0),
        quantiles.get("q50", 0.0),
        quantiles.get("q75", 0.0),
        quantiles.get("q90", 0.0),
    ]
    q_clamped = max(0.05, min(0.95, risk_fractile))
    target = np.interp(q_clamped, q_levels, q_vals)
    return float(round(target, 1))


def calculate_resource_plan(
    quantiles_by_horizon: List[Dict[str, float]],
    risk_fractile: float,
    assumptions: Optional[PlannerAssumptions] = None,
    current_active_beds: float = 0.0,
    current_o2_cylinders_stock: float = 0.0,
    current_kits_stock: float = 0.0,
) -> Dict[str, Union[float, Dict, List[Dict], str]]:
    """Converts multi-horizon case projections into disease-specific resource demands."""
    params = assumptions or PlannerAssumptions()
    disease = params.disease
    
    horizon_demands = []
    lead_time = max(1, min(len(quantiles_by_horizon), params.supplier_lead_time_weeks))
    
    for h_dict in quantiles_by_horizon:
        h = int(h_dict.get("horizon", len(horizon_demands) + 1))
        planned_cases = interpolate_quantile_case_target(h_dict, risk_fractile)
        
        # Bed calculations
        weekly_admissions = planned_cases * params.hospitalization_rate
        active_general_beds = weekly_admissions * (1.0 - params.icu_fraction) * (params.avg_length_of_stay_days / 7.0)
        active_icu_beds = weekly_admissions * params.icu_fraction * (params.icu_length_of_stay_days / 7.0)
        active_o2_ward_beds = active_general_beds * params.oxygen_bed_fraction
        total_active_beds = active_general_beds + active_icu_beds
        
        # Oxygen calculations
        ward_o2_daily_liters = active_o2_ward_beds * (params.o2_liters_per_min_ward * 60.0 * 24.0)
        icu_o2_daily_liters = active_icu_beds * (params.o2_liters_per_min_icu * 60.0 * 24.0)
        total_o2_daily_liters = ward_o2_daily_liters + icu_o2_daily_liters
        daily_d_cylinders = total_o2_daily_liters / params.cylinder_capacity_liters
        weekly_d_cylinders = daily_d_cylinders * 7.0
        weekly_lmo_mt = (total_o2_daily_liters * 7.0 / 1000.0) * 1.33 / 1000.0
        
        # Diagnostic tests
        weekly_tests = planned_cases * params.tests_per_confirmed_case * (1.0 + params.kit_wastage_factor)
        
        # Disease-specific specialized items
        platelets_needed = int(np.ceil(active_icu_beds * params.platelets_per_icu_patient_day * 7.0)) if disease == "Dengue" else 0
        iv_fluids_needed = int(np.ceil(weekly_admissions * params.iv_fluid_bottles_per_admission)) if disease in ["Dengue", "Acute Diarrheal Disease"] else 0
        act_courses_needed = int(np.ceil(planned_cases * params.act_courses_per_case)) if disease == "Malaria" else 0
        tamiflu_courses_needed = int(np.ceil(planned_cases * params.tamiflu_courses_per_case)) if disease == "Influenza (H1N1)" else 0
        ors_sachets_needed = int(np.ceil(planned_cases * params.ors_sachets_per_case)) if disease == "Acute Diarrheal Disease" else 0
        chlorine_tabs_needed = int(np.ceil(planned_cases * params.chlorine_tabs_per_case)) if disease == "Acute Diarrheal Disease" else 0
        
        horizon_demands.append({
            "horizon_weeks": h,
            "planned_cases": planned_cases,
            "low_cases_q10": h_dict.get("q10", 0.0),
            "median_cases_q50": h_dict.get("q50", 0.0),
            "high_cases_q90": h_dict.get("q90", 0.0),
            "active_general_beds": int(np.ceil(active_general_beds)),
            "active_icu_beds": int(np.ceil(active_icu_beds)),
            "total_active_beds": int(np.ceil(total_active_beds)),
            "daily_d_cylinders": int(np.ceil(daily_d_cylinders)),
            "weekly_d_cylinders": int(np.ceil(weekly_d_cylinders)),
            "weekly_lmo_metric_tonnes": round(weekly_lmo_mt, 2),
            "weekly_tests_needed": int(np.ceil(weekly_tests)),
            # Disease specific
            "platelets_needed": platelets_needed,
            "iv_fluids_needed": iv_fluids_needed,
            "act_courses_needed": act_courses_needed,
            "tamiflu_courses_needed": tamiflu_courses_needed,
            "ors_sachets_needed": ors_sachets_needed,
            "chlorine_tabs_needed": chlorine_tabs_needed,
        })
        
    target_window = horizon_demands[:lead_time]
    peak_beds = max(d["total_active_beds"] for d in target_window) if target_window else 0
    peak_icu = max(d["active_icu_beds"] for d in target_window) if target_window else 0
    peak_daily_cylinders = max(d["daily_d_cylinders"] for d in target_window) if target_window else 0
    lead_time_kits = sum(d["weekly_tests_needed"] for d in target_window)
    lead_time_cylinders = sum(d["weekly_d_cylinders"] for d in target_window)
    
    lead_platelets = sum(d["platelets_needed"] for d in target_window)
    lead_iv_fluids = sum(d["iv_fluids_needed"] for d in target_window)
    lead_act = sum(d["act_courses_needed"] for d in target_window)
    lead_tamiflu = sum(d["tamiflu_courses_needed"] for d in target_window)
    lead_ors = sum(d["ors_sachets_needed"] for d in target_window)
    lead_chlorine = sum(d["chlorine_tabs_needed"] for d in target_window)
    
    return {
        "disease": disease,
        "risk_fractile": risk_fractile,
        "lead_time_weeks": lead_time,
        "horizon_breakdown": horizon_demands,
        "summary": {
            "peak_beds_required": peak_beds,
            "peak_icu_required": peak_icu,
            "peak_daily_o2_cylinders": peak_daily_cylinders,
            "lead_time_kits_needed": lead_time_kits,
            "lead_time_cylinders_needed": lead_time_cylinders,
            "net_beds_to_mobilize": max(0, peak_beds - int(current_active_beds)),
            "net_kits_to_order": max(0, lead_time_kits - int(current_kits_stock)),
            "net_cylinders_to_order": max(0, lead_time_cylinders - int(current_o2_cylinders_stock)),
            # Disease specifics
            "lead_time_platelets": lead_platelets,
            "lead_time_iv_fluids": lead_iv_fluids,
            "lead_time_act_courses": lead_act,
            "lead_time_tamiflu_courses": lead_tamiflu,
            "lead_time_ors_sachets": lead_ors,
            "lead_time_chlorine_tabs": lead_chlorine,
        },
        "assumptions": {
            "hospitalization_rate_pct": params.hospitalization_rate * 100.0,
            "icu_fraction_pct": params.icu_fraction * 100.0,
            "avg_length_of_stay_days": params.avg_length_of_stay_days,
            "tests_per_case": params.tests_per_confirmed_case,
            "lead_time_weeks": params.supplier_lead_time_weeks,
        }
    }
