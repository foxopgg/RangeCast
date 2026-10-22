"""
RangeCast Multi-Disease Plain-Language Templating Module (templates.py)
======================================================================
Generates deterministic plain-language briefings and executive cards
tailored to the specific outbreak disease:
- COVID-19, Dengue Fever, Malaria, Influenza (H1N1), Acute Diarrheal Disease (Cholera)
- Strictly templated string interpolation (ZERO unverified LLM free-text generation)
- English, Tamil, and Hindi briefings for District Health Officers & Collectors
- One-Card WhatsApp Summary formatted for instant mobile transmission
"""

from typing import Dict, List, Optional, Union


def get_disease_resource_bullet(disease: str, plan_summary: Dict[str, Union[int, float]]) -> str:
    """Returns disease-specific primary supply line."""
    if disease == "Dengue":
        return (
            f"   - Platelet Concentrate (RDP/SDP Units Needed): {plan_summary.get('lead_time_platelets', 0):,} units\n"
            f"   - IV Fluid Bottles (Normal Saline / Ringer's 500ml): {plan_summary.get('lead_time_iv_fluids', 0):,} bottles\n"
            f"   - NS1 Antigen & IgM ELISA Testing Kits: {plan_summary.get('lead_time_kits_needed', 0):,} kits"
        )
    elif disease == "Malaria":
        return (
            f"   - ACT / Chloroquine Treatment Blister Packs: {plan_summary.get('lead_time_act_courses', 0):,} courses\n"
            f"   - Rapid Diagnostic Tests (RDT Kits): {plan_summary.get('lead_time_kits_needed', 0):,} tests\n"
            f"   - Inpatient Malaria Observation Beds: {plan_summary.get('peak_beds_required', 0):,} beds"
        )
    elif disease == "Influenza (H1N1)":
        return (
            f"   - Oseltamivir (Tamiflu) Antiviral Courses: {plan_summary.get('lead_time_tamiflu_courses', 0):,} courses\n"
            f"   - Peak Daily Ward Oxygen Cylinders: {plan_summary.get('peak_daily_o2_cylinders', 0):,} D-cylinders/day\n"
            f"   - Viral Swab & VTM Collection Kits: {plan_summary.get('lead_time_kits_needed', 0):,} kits"
        )
    elif disease == "Acute Diarrheal Disease":
        return (
            f"   - Oral Rehydration Salt (ORS) Sachets: {plan_summary.get('lead_time_ors_sachets', 0):,} packets\n"
            f"   - IV Ringer's Lactate (500ml Severe Dehydration Bottles): {plan_summary.get('lead_time_iv_fluids', 0):,} bottles\n"
            f"   - Chlorine Water Purification Tablets: {plan_summary.get('lead_time_chlorine_tabs', 0):,} tablets"
        )
    else: # COVID-19
        return (
            f"   - Peak Active Hospital Beds: {plan_summary.get('peak_beds_required', 0):,} beds (incl. {plan_summary.get('peak_icu_required', 0):,} ICU beds)\n"
            f"   - Peak Daily Oxygen Demand: {plan_summary.get('peak_daily_o2_cylinders', 0):,} standard D-type cylinders/day\n"
            f"   - Lead-Time Testing Kits Order: {plan_summary.get('lead_time_kits_needed', 0):,} diagnostic tests"
        )


def generate_english_briefing(
    district: str,
    as_of_date: str,
    quantiles_h1: Dict[str, float],
    quantiles_h2: Dict[str, float],
    plan_summary: Dict[str, Union[int, float]],
    scorecard: Dict[str, Union[float, Dict]],
    risk_fractile: float,
    disease: str = "COVID-19",
    shift_alert: bool = False,
    is_small_district: bool = False,
) -> str:
    """Deterministic English briefing for District Health Officer."""
    trust = scorecard.get("trust_badge", {})
    tier = trust.get("tier", "MEDIUM")
    cov80 = scorecard.get("coverage_80", 0.80) * 100.0
    rwis = scorecard.get("relative_wis", 1.0)
    
    alert_banner = ""
    if shift_alert:
        alert_banner = (
            f"⚠️ ATTENTION: Active {disease} Shift Alert. Recent forecast errors exceeded "
            "calibrated bands. Planning intervals have been automatically widened by 35%.\n"
        )
    if is_small_district:
        alert_banner += (
            f"ℹ️ NOTE: Low-volume rural district for {disease}. Predictions are partially pooled "
            "with state-level trends to prevent spurious zero-count noise.\n"
        )
        
    resource_bullets = get_disease_resource_bullet(disease, plan_summary)
    
    text = f"""================================================================================
RANGECAST OUTBREAK ADVISORY: DISTRICT {district.upper()} ({disease.upper()})
Date of Assessment: {as_of_date}
Surveillance Target: {disease} Outbreak Management
Trust Classification: [{trust.get('badge_text', tier)}]
================================================================================
{alert_banner}
1. PROBABILISTIC OUTBREAK TRAJECTORY
   - Next Week (Week 1):
     * Expected Cases (50% Likelihood Range): {int(quantiles_h1.get('q25', 0)):,} to {int(quantiles_h1.get('q75', 0)):,} cases
     * Wide Planning Band (80% Confidence Range): {int(quantiles_h1.get('q10', 0)):,} to {int(quantiles_h1.get('q90', 0)):,} cases
     * Central Median Forecast: {int(quantiles_h1.get('q50', 0)):,} cases
   
   - Following Week (Week 2):
     * Wide Planning Band (80% Confidence Range): {int(quantiles_h2.get('q10', 0)):,} to {int(quantiles_h2.get('q90', 0)):,} cases
     * Central Median Forecast: {int(quantiles_h2.get('q50', 0)):,} cases

2. HISTORICAL ACCURACY & CALIBRATION RECORD
   - In past rolling backtests on actual district {disease} data, RangeCast's 80% range held
     true in {cov80:.1f}% of historical weeks.
   - Weighted Interval Score (WIS) is {scorecard.get('mean_wis_model', 0):.1f} vs {scorecard.get('mean_wis_baseline', 0):.1f}
     for the persistent 'same-as-last-week' baseline (rWIS: {rwis:.2f}).
   - Verdict: {trust.get('summary', 'Calibrated forecast.')}

3. RECOMMENDED RESOURCE MOBILIZATION (Newsvendor Risk Level: {risk_fractile:.0%})
{resource_bullets}

* Generated deterministically by RangeCast. Clinical assumptions are editable placeholders.
================================================================================
"""
    return text


def generate_tamil_briefing(
    district: str,
    as_of_date: str,
    quantiles_h1: Dict[str, float],
    plan_summary: Dict[str, Union[int, float]],
    scorecard: Dict[str, Union[float, Dict]],
    risk_fractile: float,
    disease: str = "COVID-19",
    shift_alert: bool = False,
) -> str:
    """Deterministic Tamil briefing for Tamil Nadu District Health Officers."""
    trust = scorecard.get("trust_badge", {})
    cov80 = scorecard.get("coverage_80", 0.80) * 100.0
    
    alert_txt = ""
    if shift_alert:
        alert_txt = f"⚠️ எச்சரிக்கை: {disease} தொற்று பரவல் மாற்றம் கண்டறியப்பட்டுள்ளது. திட்டமிடல் வரம்பு 35% அதிகரிக்கப்பட்டுள்ளது.\n"
        
    text = f"""================================================================================
ரேஞ்ச்காஸ்ட் (RangeCast) மாவட்ட சுகாதார அறிக்கை: {district} மாவட்டம் ({disease})
மதிப்பீட்டு தேதி: {as_of_date}
தொற்று வகை: {disease}
நம்பகத்தன்மை தரம்: [{trust.get('badge_text', 'சரிபார்க்கப்பட்டது')}]
================================================================================
{alert_txt}
1. எதிர்பார்க்கப்படும் பாதிப்பு வரம்பு (அடுத்த வாரம்):
   - 80% திட்டமிடல் வரம்பு: {int(quantiles_h1.get('q10', 0)):,} முதல் {int(quantiles_h1.get('q90', 0)):,} புதிய நோயாளிகள்
   - மைய கணிப்பு (Median): {int(quantiles_h1.get('q50', 0)):,} நோயாளிகள்
   - வரலாற்று துல்லியம்: கடந்த வாரங்களில் 80% வரம்பு {cov80:.1f}% முறை துல்லியமாக பொருந்தியுள்ளது.

2. தயார் நிலையில் வைக்க வேண்டிய மருத்துவ வளங்கள் (ஆபத்து நிலை: {risk_fractile:.0%}):
   - தேவைப்படும் உச்சகட்ட படுக்கைகள்: {plan_summary.get('peak_beds_required', 0):,}
   - தேவையான பரிசோதனை கருவிகள்: {plan_summary.get('lead_time_kits_needed', 0):,} கருவிகள்

* ரேஞ்ச்காஸ்ட் மூலம் தானாக கணக்கிடப்பட்ட நம்பகமான வழிகாட்டுதல்.
================================================================================
"""
    return text


def generate_hindi_briefing(
    district: str,
    as_of_date: str,
    quantiles_h1: Dict[str, float],
    plan_summary: Dict[str, Union[int, float]],
    scorecard: Dict[str, Union[float, Dict]],
    risk_fractile: float,
    disease: str = "COVID-19",
    shift_alert: bool = False,
) -> str:
    """Deterministic Hindi briefing for District Health Officers."""
    trust = scorecard.get("trust_badge", {})
    cov80 = scorecard.get("coverage_80", 0.80) * 100.0
    
    alert_txt = ""
    if shift_alert:
        alert_txt = f"⚠️ चेतावनी: {disease} मामलों में अप्रत्याशित बदलाव दर्ज। सुरक्षा दायरा 35% बढ़ाया गया है।\n"
        
    text = f"""================================================================================
रेंजकास्ट (RangeCast) जिला स्वास्थ्य सलाहकार: {district} जिला ({disease})
मूल्यांकन तिथि: {as_of_date}
विश्वसनीयता रेटिंग: [{trust.get('badge_text', 'सत्यापित')}]
================================================================================
{alert_txt}
1. संभावित केस दायरा (अगला सप्ताह):
   - 80% योजना दायरा: {int(quantiles_h1.get('q10', 0)):,} से {int(quantiles_h1.get('q90', 0)):,} नए केस
   - केंद्रीय अनुमान (Median): {int(quantiles_h1.get('q50', 0)):,} नए केस
   - ऐतिहासिक सटीकता: पिछले हफ्तों में 80% दायरा {cov80:.1f}% बार सही साबित हुआ।

2. आवश्यक संसाधन आवंटन (जोखिम स्तर: {risk_fractile:.0%}):
   - कुल आवश्यक बेड: {plan_summary.get('peak_beds_required', 0):,} बेड
   - आगामी सप्लाई अवधि हेतु टेस्टिंग किट आर्डर: {plan_summary.get('lead_time_kits_needed', 0):,} किट

* वास्तविक डेटा बैकटेस्ट पर आधारित।
================================================================================
"""
    return text


def generate_whatsapp_card(
    district: str,
    as_of_date: str,
    quantiles_h1: Dict[str, float],
    plan_summary: Dict[str, Union[int, float]],
    scorecard: Dict[str, Union[float, Dict]],
    risk_fractile: float,
    disease: str = "COVID-19",
    shift_alert: bool = False,
) -> str:
    """
    Compact, high-density WhatsApp One-Card summary for mobile transmission.
    """
    trust = scorecard.get("trust_badge", {})
    badge = trust.get("badge_text", "HIGH TRUST")
    cov80 = scorecard.get("coverage_80", 0.80) * 100.0
    
    status_icon = "🟢" if "HIGH" in badge else ("🟡" if "MODERATE" in badge else "🔴")
    if shift_alert:
        status_icon = "⚠️"
        
    q10 = int(quantiles_h1.get('q10', 0))
    q50 = int(quantiles_h1.get('q50', 0))
    q90 = int(quantiles_h1.get('q90', 0))
    
    beds = plan_summary.get('peak_beds_required', 0)
    kits = plan_summary.get('lead_time_kits_needed', 0)
    
    # Disease specific line
    if disease == "Dengue":
        extra_line = f"• 🩸 *Platelet Units:* {plan_summary.get('lead_time_platelets', 0):,} | 💧 *IV Fluids:* {plan_summary.get('lead_time_iv_fluids', 0):,} bottles"
    elif disease == "Malaria":
        extra_line = f"• 💊 *ACT Courses:* {plan_summary.get('lead_time_act_courses', 0):,} packs"
    elif disease == "Influenza (H1N1)":
        extra_line = f"• 💊 *Tamiflu Packs:* {plan_summary.get('lead_time_tamiflu_courses', 0):,} | 💨 *O2 Cylinders:* {plan_summary.get('peak_daily_o2_cylinders', 0):,}/day"
    elif disease == "Acute Diarrheal Disease":
        extra_line = f"• 🧂 *ORS Sachets:* {plan_summary.get('lead_time_ors_sachets', 0):,} | 💧 *IV Ringer's:* {plan_summary.get('lead_time_iv_fluids', 0):,} bottles"
    else:
        extra_line = f"• 💨 *Oxygen Demand:* {plan_summary.get('peak_daily_o2_cylinders', 0):,} D-cylinders/day"
        
    card = f"""📊 *RangeCast Weekly Alert: {district}* ({disease.upper()} - {as_of_date})
{status_icon} *Status:* {badge} | *80% Hist. Accuracy:* {cov80:.0f}%

📈 *Projected Cases (Next 7 Days):*
• *Likely Range (80%):* {q10:,} - {q90:,}
• *Median Expected:* {q50:,}

🏥 *Resource Orders to Mobilize ({risk_fractile:.0%} Safety Level):*
• 🛏️ *Active Beds:* {beds:,}
• 🧪 *Diagnostic Kits:* {kits:,}
{extra_line}

{'⚠️ *SHIFT ALERT:* Rapid case inflection detected. Buffer widened.' if shift_alert else '✅ *Surveillance:* Normal calibrated trajectory.'}
_Forwarded from RangeCast District Outbreak Portal_"""
    return card
