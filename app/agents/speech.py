from app.orchestrator.state import AgentState
import numpy as np
import logging
from app.core.config import settings
from sarvamai import SarvamAI
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

# Thresholds for API-based language filtering (relaxed for multilingual speakers)
_MIN_PERCENTAGE = 0.05  # Must be at least 5% of all detections
_MIN_CONFIDENCE = 0.3   # Must have at least 0.3 max confidence

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
        if count >= 3:
            result.append(lang)
    return result


def _filter_languages(lang_counts: dict, lang_max_prob: dict) -> list[str]:
    """Apply percentage + confidence filtering to accumulated language detections.

    Same logic as the batch pipeline in test_language_detection.py.
    A language is confirmed only if:
      - it accounts for >= 10% of total API detections
      - its maximum confidence across chunks is >= 0.5
    """
    total = sum(lang_counts.values())
    if total == 0:
        return []

    confirmed = []
    for name, count in lang_counts.items():
        pct = count / total
        max_conf = lang_max_prob.get(name, 0.0)
        if pct >= _MIN_PERCENTAGE and max_conf >= _MIN_CONFIDENCE:
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
            with open(temp_path, "rb") as f:
                # Use Saaras V3 with codemix mode for correct multilingual transcription
                resp = client.speech_to_text.transcribe(
                    file=f,
                    model=settings.SPEECH_TO_TEXT_MODEL,
                    language_code=settings.SPEECH_LANGUAGE_CODE,
                    mode="codemix"
                )

            # Extract transcript text
            text = resp.transcript if hasattr(resp, "transcript") else (resp.text if hasattr(resp, "text") else str(resp))
            if text.strip():
                rolling = (rolling + " " + text.strip()).strip()

            # Extract API-level language detection (the reliable source)
            api_lang_code = getattr(resp, "language_code", None)
            api_probability = getattr(resp, "language_probability", None) or 0.0
            api_lang_name = _LANG_CODE_MAP.get(api_lang_code, api_lang_code) if api_lang_code else None

            if api_lang_name and api_lang_name != "unknown":
                lang_counts[api_lang_name] = lang_counts.get(api_lang_name, 0) + 1
                lang_max_prob[api_lang_name] = max(
                    lang_max_prob.get(api_lang_name, 0.0), api_probability
                )
                logger.info(
                    "STT chunk language: %s (%s) conf=%.3f",
                    api_lang_name, api_lang_code, api_probability
                )

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

