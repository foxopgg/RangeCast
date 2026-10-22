"""
RangeCast Groq AI Client Module (groq_client.py)
=================================================
Server-side-only Groq AI integration for plain-language outbreak briefings
and Q&A with district health officers.

SECURITY RULES:
- GROQ_API_KEY is NEVER exposed to the browser or any API response.
- Only aggregated, anonymized district-level counts are sent to Groq.
- No patient-level data is transmitted at any point.

GUARDRAILS:
- Temperature: 0.0-0.2 (near-deterministic outputs)
- System prompt forbids introducing numbers not in the provided context.
- Number validator: extracts every number from model output and rejects
  the response if any number is not in the allowed set (allows rounding).
- On any Groq error / timeout / validation failure, falls back to
  deterministic template answer and returns verified=False.

PERFORMANCE:
- In-memory LRU cache for identical context hashes (TTL: 10 min).
- Per-IP rate limiting enforced at the FastAPI layer (see fastapi_server.py).
- 8-second timeout hard cap on every Groq API call.
"""

import os
import re
import time
import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache

# ---- Environment loading --------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; real env vars always win

# ---- Logging (no personal data) ------------------------------------------
logger = logging.getLogger("rangecast.groq")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---- Groq configuration ---------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT: float = float(os.environ.get("GROQ_TIMEOUT_SECONDS", "8"))
GROQ_TEMPERATURE: float = 0.1   # near-deterministic

# ---- LRU + TTL cache (300 entries, 10 min TTL) ----------------------------
_cache: TTLCache = TTLCache(maxsize=300, ttl=600)

# ---- Groq client (lazy init after startup check) --------------------------
_groq_client = None


def _get_client():
    """Returns the singleton Groq client, initialised once."""
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT)
        except ImportError as e:
            raise RuntimeError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from e
    return _groq_client


def startup_check() -> None:
    """
    Called once at server startup.
    Validates GROQ_API_KEY is set and GROQ_MODEL is available on the account.
    Raises RuntimeError with a clear message if not.
    """
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_xxx"):
        raise RuntimeError(
            "GROQ_API_KEY is not set or is still the placeholder value. "
            "Copy .env.example to .env and fill in your real Groq API key."
        )

    client = _get_client()
    try:
        available_models = [m.id for m in client.models.list().data]
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Groq API to list models: {e}. "
            "Check your API key and network connectivity."
        ) from e

    if GROQ_MODEL not in available_models:
        raise RuntimeError(
            f"GROQ_MODEL='{GROQ_MODEL}' is NOT available on your Groq account. "
            f"Available models: {available_models}. "
            "Update GROQ_MODEL in your .env file."
        )

    logger.info(
        f"[startup_check] Groq client ready. Model='{GROQ_MODEL}', "
        f"timeout={GROQ_TIMEOUT}s, temperature={GROQ_TEMPERATURE}"
    )


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are RangeCast, a district outbreak advisory assistant for Indian public health officers.

STRICT RULES -- violating any rule makes your response invalid:
1. You may ONLY use numbers that are explicitly present in the JSON context provided to you.
   Rounding is allowed (e.g. 3.7 -> 4, 1250 -> 1,250) but inventing new numbers is forbidden.
2. You must NEVER provide medical diagnosis, treatment recommendations for individual patients,
   or any statement that can be interpreted as clinical medical advice.
3. You must NEVER introduce epidemiological conclusions not supported by the context data.
4. Keep responses concise (200 words max for /ask, 300 words max for /summary).
5. Use plain, simple language suitable for a district health officer briefing.
6. When numbers from the context appear in your response, state them clearly.
7. Do NOT fabricate any percentage, count, or ratio.
"""

# ---------------------------------------------------------------------------
# NUMBER VALIDATOR
# ---------------------------------------------------------------------------
# Tolerance for rounding: a model number matches an allowed number if
# it's within this relative fraction of any allowed value.
_ROUND_TOL = 0.02   # 2% relative tolerance for rounding (e.g. 1000 vs 1001)


def _extract_numbers(text: str) -> List[float]:
    """
    Extracts all numeric tokens from text.
    Handles: 1,234  1234  3.7  80%  ~500
    Returns a list of floats.
    """
    cleaned = text.replace(",", "")
    tokens = re.findall(r"[~]?\d+(?:\.\d+)?", cleaned)
    result = []
    for t in tokens:
        t = t.lstrip("~")
        try:
            result.append(float(t))
        except ValueError:
            pass
    return result


def _build_allowed_set(context: Dict[str, Any]) -> List[float]:
    """
    Recursively walks the context dict and collects every numeric value.
    Returns a flat list of floats representing the 'allowed' number set.
    """
    allowed: List[float] = []

    def _walk(obj):
        if isinstance(obj, (int, float)):
            allowed.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    _walk(context)
    return allowed


def _is_number_allowed(num: float, allowed: List[float]) -> bool:
    """
    Returns True if 'num' is within rounding tolerance of any allowed value.
    Small integers up to 10 are always allowed (ordinals / counts in prose).
    Standard epidemiological interval levels (50, 80, 90, 95) are always
    allowed because they name the interval itself, not a data-derived value.
    """
    if 0.0 <= num <= 10.0 and num == int(num):
        return True  # small ordinals in prose
    # Standard confidence / prediction interval level references
    if num in (50.0, 80.0, 90.0, 95.0, 25.0, 75.0):
        return True
    for a in allowed:
        if a == 0.0:
            if abs(num) < 0.5:
                return True
        else:
            if abs(num - a) / abs(a) <= _ROUND_TOL:
                return True
    return False


def validate_response_numbers(
    response_text: str, context: Dict[str, Any]
) -> Tuple[bool, List[float]]:
    """
    Validates that every number in the model's response appears in the context.

    Returns:
        (is_valid: bool, unauthorized_numbers: List[float])
    """
    allowed = _build_allowed_set(context)
    model_numbers = _extract_numbers(response_text)

    unauthorized = []
    for n in model_numbers:
        if not _is_number_allowed(n, allowed):
            unauthorized.append(n)

    if unauthorized:
        logger.warning(
            f"[validator] Rejected response. Unauthorized numbers: {unauthorized}. "
            f"Allowed set (sample): {allowed[:20]}"
        )
        return False, unauthorized

    return True, []


# ---------------------------------------------------------------------------
# CACHE KEY HELPER
# ---------------------------------------------------------------------------
def _context_cache_key(context: Dict[str, Any], question: str, lang: str) -> str:
    """Creates a stable cache key from context + question + language."""
    payload = json.dumps(
        {"ctx": context, "q": question, "lang": lang},
        sort_keys=True,
        default=str
    )
    return hashlib.md5(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FALLBACK TEMPLATES
# ---------------------------------------------------------------------------
def _fallback_summary(context: Dict[str, Any], lang: str) -> str:
    """
    Returns a deterministic (non-AI) briefing summary when Groq fails.
    Uses only numbers from context.
    """
    district = context.get("district", "this district")
    disease = context.get("disease", "the disease")
    q10 = context.get("q10_week1", "?")
    q90 = context.get("q90_week1", "?")
    q50 = context.get("q50_week1", "?")
    cov80 = context.get("coverage_80_pct", "?")
    trust = context.get("trust_tier", "?")
    beds = context.get("peak_beds_required", "?")

    en_text = (
        f"[TEMPLATE - AI unavailable] {district} {disease} briefing: "
        f"Next week projected range is {q10}-{q90} cases (median {q50}). "
        f"Historical 80% range coverage: {cov80}%. Trust: {trust}. "
        f"Peak beds required: {beds}."
    )

    if lang == "ta":
        return (
            f"[மாதிரி - AI இல்லை] {district} {disease}: "
            f"அடுத்த வார வரம்பு {q10}-{q90} வழக்குகள் (இடைநிலை {q50}). "
            f"80% வரம்பு நம்பகத்தன்மை: {cov80}%. நம்பிக்கை நிலை: {trust}."
        )
    if lang == "hi":
        return (
            f"[टेम्पलेट - AI अनुपलब्ध] {district} {disease}: "
            f"अगले सप्ताह अनुमानित सीमा {q10}-{q90} मामले (माध्यिका {q50}). "
            f"80% सीमा कवरेज: {cov80}%. विश्वास स्तर: {trust}."
        )
    return en_text


def _fallback_ask(question: str, context: Dict[str, Any]) -> str:
    """
    Returns a deterministic answer for common officer questions when Groq fails.
    """
    q = question.lower()
    if "wider" in q or "range" in q or "uncertainty" in q:
        alpha = context.get("conformal_alpha_80", "?")
        shift = context.get("shift_alert", False)
        reason = "an active shift alert is widening bands" if shift else "adaptive conformal calibration"
        return (
            f"[TEMPLATE] The prediction intervals are calibrated using {reason}. "
            f"Current conformal alpha: {alpha}. When the model misses its target, "
            "intervals automatically expand to maintain honest coverage."
        )
    if "oxygen" in q or "o2" in q:
        cyl = context.get("peak_daily_o2_cylinders", "?")
        lt = context.get("lead_time_cylinders_needed", "?")
        return (
            f"[TEMPLATE] Based on current projections, peak daily oxygen need is "
            f"{cyl} D-type cylinders/day. For the lead-time window, order {lt} cylinders total."
        )
    if "bed" in q or "hospital" in q:
        beds = context.get("peak_beds_required", "?")
        icu = context.get("peak_icu_required", "?")
        return f"[TEMPLATE] Peak hospital beds required: {beds} (including {icu} ICU/HDU beds)."
    if "trust" in q or "reliable" in q or "accurate" in q:
        tier = context.get("trust_tier", "?")
        cov = context.get("coverage_80_pct", "?")
        return (
            f"[TEMPLATE] Current trust level: {tier}. "
            f"The 80% prediction range has been historically accurate {cov}% of the time "
            "based on rolling backtests."
        )
    return (
        "[TEMPLATE] I could not generate an AI response. "
        "Please refer to the scorecard and planner panels for computed figures."
    )


# ---------------------------------------------------------------------------
# CORE API CALLS
# ---------------------------------------------------------------------------
def generate_summary(
    context: Dict[str, Any],
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Generates a plain-language outbreak summary in the specified language.

    Args:
        context: Backend-computed numerical context (no PHI, aggregated only).
        lang: "en" | "ta" | "hi"

    Returns:
        dict with keys: answer, verified, lang, machine_assisted
    """
    cache_key = _context_cache_key(context, "__summary__", lang)
    if cache_key in _cache:
        logger.info("[summary] Cache hit.")
        return _cache[cache_key]

    lang_instruction = {
        "en": "Respond in clear English.",
        "ta": (
            "Respond ONLY in Tamil (thamizh). Use simple Tamil words. "
            "Note: Native-speaker review is required before showing to real users."
        ),
        "hi": (
            "Respond ONLY in Hindi. Use simple Hindi words. "
            "Note: Native-speaker review is required before showing to real users."
        ),
    }.get(lang, "Respond in English.")

    district = context.get("district", "the district")
    disease = context.get("disease", "outbreak")
    ctx_json = json.dumps(context, default=str, indent=2)

    user_message = (
        f"You are briefing the District Health Officer for {district} about {disease}.\n\n"
        f"Here is the computed data context (use ONLY these numbers in your response):\n"
        f"```json\n{ctx_json}\n```\n\n"
        f"Generate a concise executive summary (280 words max) covering:\n"
        f"1. The current situation and next week's forecast range\n"
        f"2. Historical accuracy / trust level\n"
        f"3. Top resource procurement action items\n"
        f"4. Any active shift alerts or warnings\n\n"
        f"{lang_instruction}"
    )

    try:
        client = _get_client()
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=500,
        )
        elapsed = time.monotonic() - t0
        answer = response.choices[0].message.content.strip()
        logger.info(
            f"[summary] Groq call OK. lang={lang}, elapsed={elapsed:.2f}s, "
            f"tokens={response.usage.total_tokens}"
        )

        is_valid, bad_nums = validate_response_numbers(answer, context)
        if not is_valid:
            logger.warning(f"[summary] Number validation failed. Bad: {bad_nums}. Using fallback.")
            answer = _fallback_summary(context, lang)
            verified = False
        else:
            verified = True

    except Exception as e:
        logger.error(f"[summary] Groq error: {e}. Using fallback.")
        answer = _fallback_summary(context, lang)
        verified = False

    result = {
        "answer": answer,
        "verified": verified,
        "lang": lang,
        "machine_assisted": lang in ("ta", "hi"),
    }
    _cache[cache_key] = result
    return result


def ask_question(
    question: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Answers a district health officer's question using ONLY the provided context.

    Args:
        question: The officer's natural-language question.
        context: Backend-computed JSON context (forecast + plan + scorecard numbers).

    Returns:
        dict with keys: answer, verified, question
    """
    cache_key = _context_cache_key(context, question, "en")
    if cache_key in _cache:
        logger.info("[ask] Cache hit.")
        return _cache[cache_key]

    ctx_json = json.dumps(context, default=str, indent=2)
    district = context.get("district", "the district")
    disease = context.get("disease", "outbreak")

    user_message = (
        f"A district health officer for {district} ({disease} outbreak) is asking:\n\n"
        f'"{question}"\n\n'
        f"Here is the ONLY data you may use to answer (do not use any other numbers):\n"
        f"```json\n{ctx_json}\n```\n\n"
        f"Provide a direct, concise answer (180 words max) using ONLY the numbers above.\n"
        f"Do NOT provide medical diagnosis or individual clinical advice.\n"
        f"If the answer cannot be found in the data, say so clearly."
    )

    try:
        client = _get_client()
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=350,
        )
        elapsed = time.monotonic() - t0
        answer = response.choices[0].message.content.strip()
        logger.info(
            f"[ask] Groq call OK. elapsed={elapsed:.2f}s, tokens={response.usage.total_tokens}"
        )

        is_valid, bad_nums = validate_response_numbers(answer, context)
        if not is_valid:
            logger.warning(f"[ask] Validation failed. Bad numbers: {bad_nums}. Using fallback.")
            answer = _fallback_ask(question, context)
            verified = False
        else:
            verified = True

    except Exception as e:
        logger.error(f"[ask] Groq error: {e}. Using fallback.")
        answer = _fallback_ask(question, context)
        verified = False

    result = {
        "answer": answer,
        "verified": verified,
        "question": question,
    }
    _cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# SUGGESTED QUESTIONS (for UI prompt chips)
# ---------------------------------------------------------------------------
SUGGESTED_QUESTIONS = [
    "Why are the prediction ranges wider this week?",
    "How much oxygen should I order for the next two weeks?",
    "Is this forecast reliable enough to make procurement decisions?",
    "What does the shift alert mean for planning?",
    "How many hospital beds do I need to prepare?",
    "How does RangeCast compare to the baseline?",
    "What is the 80% coverage figure telling me?",
    "Should I be concerned about the current trend?",
]
