from app.orchestrator.state import AgentState
import numpy as np
import logging
from app.core.config import settings
from sarvamai import SarvamAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
import tempfile
import os

logger = logging.getLogger(__name__)

# BCP-47 language code to human-readable name mapping
_LANG_CODE_MAP = {
    "hi-IN": "Hindi", "bn-IN": "Bengali", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "mr-IN": "Marathi", "od-IN": "Odia",
    "pa-IN": "Punjabi", "ta-IN": "Tamil", "te-IN": "Telugu",
    "en-IN": "English", "gu-IN": "Gujarati", "as-IN": "Assamese",
    "ur-IN": "Urdu", "ne-IN": "Nepali", "kok-IN": "Konkani",
    "ks-IN": "Kashmiri", "sd-IN": "Sindhi", "sa-IN": "Sanskrit",
    "sat-IN": "Santali", "mni-IN": "Manipuri", "brx-IN": "Bodo",
    "mai-IN": "Maithili", "doi-IN": "Dogri",
}
# Reverse map: human-readable name → BCP-47 code (for STT hint)
_NAME_TO_CODE = {name: code for code, name in _LANG_CODE_MAP.items()}

# Tiered detection thresholds — faster for high-confidence, cautious for low-confidence
_MIN_CONFIDENCE_FAST = 0.35   # 1 chunk is enough (Hindi, Telugu/Kannada, etc.)
_MIN_CONFIDENCE_SLOW = 0.18   # 2 chunks needed  (Bengali ~0.20, borderline signals)
_MIN_COUNT_FAST = 1
_MIN_COUNT_SLOW = 2

# Unicode script ranges for fallback detection from transcript text.
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "Hindi"),       # Devanagari
    (0x0980, 0x09FF, "Bengali"),     # Bengali / Assamese
    (0x0A00, 0x0A7F, "Punjabi"),     # Gurmukhi
    (0x0A80, 0x0AFF, "Gujarati"),    # Gujarati
    (0x0B00, 0x0B7F, "Odia"),        # Oriya
    (0x0B80, 0x0BFF, "Tamil"),       # Tamil
    (0x0C00, 0x0C7F, "Telugu"),      # Telugu
    (0x0C80, 0x0CFF, "Kannada"),     # Kannada
    (0x0D00, 0x0D7F, "Malayalam"),   # Malayalam
]


def _detect_languages_from_text(text: str) -> list[str]:
    """Fallback: detect languages from Unicode script characters in transcript text.

    Used only when the Sarvam API does not return a language_code.
    """
    script_char_counts = {}
    has_latin = False

    for ch in text:
        cp = ord(ch)
        # Latin/ASCII → English
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            has_latin = True
            continue
        # Check Indic script ranges
        for start, end, lang in _SCRIPT_RANGES:
            if start <= cp <= end:
                script_char_counts[lang] = script_char_counts.get(lang, 0) + 1
                break

    result = []
    if has_latin:
        result.append("English")
    for lang, count in sorted(script_char_counts.items(), key=lambda x: -x[1]):
        if count >= 2:  # 2 chars minimum — catches single Bengali/Hindi words
            result.append(lang)
    return result


def _filter_languages(lang_counts: dict, lang_max_prob: dict) -> list[str]:
    """Tiered detection: faster for high-confidence, stricter for low-confidence.

    Tier 1 (fast)  : count >= 1 AND conf >= 0.35  → detected after just 1 STT chunk
                     Catches Hindi, Telugu/Kannada, etc. within ~2s of being spoken.
    Tier 2 (careful): count >= 2 AND conf >= 0.18  → needs 2 chunks
                     Catches Bengali (~0.20 conf) while requiring repetition as signal.

    No percentage gate — English dominates chunk counts in multi-language sessions,
    making percentage-based gates too strict for 3-4 language scenarios.
    """
    confirmed = []
    for name, count in lang_counts.items():
        conf = lang_max_prob.get(name, 0.0)
        fast_pass = (count >= _MIN_COUNT_FAST and conf >= _MIN_CONFIDENCE_FAST)
        slow_pass = (count >= _MIN_COUNT_SLOW and conf >= _MIN_CONFIDENCE_SLOW)
        if fast_pass or slow_pass:
            confirmed.append(name)
    return confirmed


def process(state: AgentState) -> dict:
    """Streaming STT + Language Detection

    Uses Sarvam API's language_code and language_probability responses
    (the same approach that works correctly in the batch pipeline) instead
    of relying solely on Unicode script-range detection.
    """
    features = state.get("recent_acoustic_features", [])
    if not features or "raw_audio" not in features[0]:
        return {}

    audio_data, sample_rate = features[0]["raw_audio"]

    try:
        if audio_data.dtype == np.int16:
            y = audio_data.mean(axis=0).astype(np.float32) / 32768.0
        else:
            y = audio_data.mean(axis=0).astype(np.float32)
    except Exception:
        return {}

    ts = state.get("transcript_state", {})
    rolling = ts.get("rolling_transcript", "")

    # Accumulated API-level language detection counts and max probabilities
    lang_counts = ts.get("api_lang_counts", {})
    lang_max_prob = ts.get("api_lang_max_prob", {})

    try:
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        sr = sample_rate if sample_rate else 44100
        sf.write(temp_path, y, sr)

        if settings.SARVAM_API_KEY:
            client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)

            # ── Always-on parallel probes for the 4 most commonly missed languages ─
            # Strategy: call Sarvam with each specific language code simultaneously.
            # When Bengali audio is sent with language_code="bn-IN", Sarvam transcribes
            # in Bengali script. We then detect Bengali chars deterministically.
            # We do NOT gate on language_probability (Sarvam often omits it for
            # explicit language_code calls). The transcript itself is the signal.
            #
            # 4 probes + 1 primary = 5 parallel calls. All start at the same time,
            # so total wait = time of slowest single call (no additional latency).
            _ALWAYS_PROBE = [
                ("bn-IN", "Bengali"),
                ("te-IN", "Telugu"),
                ("ta-IN", "Tamil"),
                ("ml-IN", "Malayalam"),
            ]
            probe_results: list = [{} for _ in _ALWAYS_PROBE]

            def _probe_call(path, code, name):
                try:
                    _c = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
                    with open(path, "rb") as _f:
                        res = _c.speech_to_text.transcribe(
                            file=_f,
                            model=settings.SPEECH_TO_TEXT_MODEL,
                            language_code=code,
                            mode="codemix",
                        )
                    return {
                        "name": name,
                        "code": code,
                        "text": getattr(res, "transcript", "") or "",
                        "prob": float(getattr(res, "language_probability", None) or 0.0),
                        "api_code": getattr(res, "language_code", None)
                    }
                except Exception as _ex:
                    return {"error": str(_ex)}

            # Use a ThreadPoolExecutor for better resource management
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Schedule probes and the primary "unknown" call
                future_to_probe = {executor.submit(_probe_call, temp_path, code, name): name for code, name in _ALWAYS_PROBE}
                primary_future = executor.submit(_probe_call, temp_path, settings.SPEECH_LANGUAGE_CODE, "primary")

                # Wait for primary and gather probe results
                resp_data = primary_future.result(timeout=10.0)
                probe_results = []
                for future in as_completed(future_to_probe):
                    probe_results.append(future.result())

            # ── Merge probe results via TRANSCRIPT SCRIPT DETECTION ───────────
            # This is deterministic: Bengali Unicode chars can ONLY come from Bengali.
            # We do NOT rely on language_probability (often omitted for explicit codes).
            for pr in probe_results:
                if not pr or "text" not in pr:
                    continue
                probe_text  = pr.get("text", "")
                probe_name  = pr.get("name", "")
                script_hits = _detect_languages_from_text(probe_text)

                # Primary signal: script chars in probe transcript
                if probe_name in script_hits:
                    lang_counts[probe_name]   = lang_counts.get(probe_name, 0) + 1
                    if lang_max_prob.get(probe_name, 0.0) < 0.90:
                        lang_max_prob[probe_name] = 0.90
                    logger.info("Probe script-confirm: %s (transcript has %s chars)",
                                probe_name, probe_name)

                # Bonus signal: if API returned an explicit language code too
                probe_api_name = _LANG_CODE_MAP.get(pr.get("api_code"), None)
                if probe_api_name and probe_api_name not in ("unknown", probe_name):
                    # The probe's API detected a DIFFERENT language than we probed →
                    # that different language is also present
                    lang_counts[probe_api_name]   = lang_counts.get(probe_api_name, 0) + 1
                    lang_max_prob[probe_api_name] = max(
                        lang_max_prob.get(probe_api_name, 0.0), pr.get("prob", 0.0)
                    )

                # Also run script detection on probe transcript for any other scripts
                for slang in script_hits:
                    if slang == "English" or slang == probe_name:
                        continue
                    lang_counts[slang] = lang_counts.get(slang, 0) + 1
                    if lang_max_prob.get(slang, 0.0) < 0.90:
                        lang_max_prob[slang] = 0.90

            # ── Extract primary transcript (resp_data is now a dict) ──────────
            text = resp_data.get("text", "")
            if text.strip():
                rolling = (rolling + " " + text.strip()).strip()

                # Script detection on primary transcript (0.90 confidence)
                for slang in _detect_languages_from_text(text.strip()):
                    if slang == "English":
                        continue
                    lang_counts[slang] = lang_counts.get(slang, 0) + 1
                    if lang_max_prob.get(slang, 0.0) < 0.90:
                        lang_max_prob[slang] = 0.90

            # ── Primary API language code from unknown call ───────────────────
            api_lang_code   = resp_data.get("api_code")
            api_probability = resp_data.get("prob", 0.0)
            api_lang_name   = _LANG_CODE_MAP.get(api_lang_code, api_lang_code) if api_lang_code else None
            if api_lang_name and api_lang_name != "unknown":
                lang_counts[api_lang_name]   = lang_counts.get(api_lang_name, 0) + 1
                lang_max_prob[api_lang_name] = max(
                    lang_max_prob.get(api_lang_name, 0.0), api_probability
                )
                logger.info("Primary STT: %s conf=%.3f", api_lang_name, api_probability)

            # ── Debug dump ───────────────────────────────────────────────────
            try:
                resp_dump = resp_data.copy()
                resp_dump["probes"] = [
                    {"lang": p.get("name"), "script_hits": _detect_languages_from_text(p.get("text",""))}
                    for p in probe_results if "name" in p
                ]
            except Exception:
                resp_dump = {"raw": str(resp_data)}
            logger.error("SARVAM RAW RESPONSE: %s", resp_dump)
            ts["last_sarvam_response"] = resp_dump
            ts["last_transcript_chunk"] = text.strip() if text.strip() else ""

        os.remove(temp_path)
    except Exception as e:
        logger.error("STT ERROR: %s", e)

    ts["rolling_transcript"] = rolling
    ts["api_lang_counts"] = lang_counts
    ts["api_lang_max_prob"] = lang_max_prob

    # --- Primary: API-based language detection with filtering ---
    # Apply percentage + confidence filtering with relaxed thresholds.
    detected = _filter_languages(lang_counts, lang_max_prob)
    if lang_counts:
        logger.info(
            "Language detection — raw API counts: %s | max probs: %s | confirmed: %s",
            lang_counts, lang_max_prob, detected
        )

    # --- Fallback: Unicode script detection if API provided nothing ---
    if not detected and rolling:
        detected = _detect_languages_from_text(rolling)
        if detected:
            logger.info("Language detection fell back to Unicode script analysis: %s", detected)

    ts["stt_detected_languages"] = detected
    if detected:
        non_eng = [l for l in detected if l.lower() != "english"]
        if non_eng:
            ts["detected_language_name"] = non_eng[-1]

    return {"transcript_state": ts}
