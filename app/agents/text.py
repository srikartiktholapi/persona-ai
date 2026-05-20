import logging
from app.orchestrator.state import AgentState
from app.core.config import settings
import requests
import json

logger = logging.getLogger(__name__)

# Maximum allowed languages (English + 2 regional)
MAX_LANGUAGES_ALLOWED = 3

def _extract_text_score(result: dict) -> float | None:
    """Extract text score from LLM result, handling key variations."""
    if "text_score out of 10" in result:
        return float(result["text_score out of 10"])
    for key in result:
        if "text_score" in key.lower():
            try:
                return float(result[key])
            except (ValueError, TypeError):
                continue
    return None

def process(state: AgentState) -> dict:
    """Grammar, fluency, professionalism tracking"""
    ts = state.get("transcript_state", {})
    segment = ts.get("rolling_transcript", "")
    
    scores = state.get("scores", {})
    
    # --- Use Sarvam STT's language_code response as ground truth ---
    stt_langs = ts.get("stt_detected_languages", [])
    
    # Always update scores with STT language detection.
    # IMPORTANT: Use a CUMULATIVE union so the list only grows — never resets mid-session.
    if stt_langs:
        existing_langs = set(scores.get("all_detected_languages", []))
        existing_langs.update(stt_langs)
        scores["all_detected_languages"] = sorted(existing_langs)
        non_eng = [l for l in scores["all_detected_languages"] if l.lower() != "english"]
        if non_eng:
            scores["detected_language_name"] = non_eng[-1]
            scores["regional_language_detected"] = True
        else:
            scores["regional_language_detected"] = False
    
    # --- 3-language limit alert ---
    if len(stt_langs) > MAX_LANGUAGES_ALLOWED:
        scores["language_limit_exceeded"] = True
        excess = len(stt_langs) - MAX_LANGUAGES_ALLOWED
        scores["language_alert"] = (
            f"Too many languages detected ({len(stt_langs)}): {', '.join(stt_langs)}. "
            f"Please limit to {MAX_LANGUAGES_ALLOWED} languages total "
            f"(English + up to 2 regional languages). "
            f"Excess: {excess} extra language(s)."
        )
    else:
        scores["language_limit_exceeded"] = False
        scores["language_alert"] = ""
    
    # Build language context for the LLM prompt
    lang_context = ""
    if stt_langs:
        lang_context = f"""
            DETECTED LANGUAGES (from STT audio-level detection):
            The following languages have been detected in this session: {', '.join(stt_langs)}.
            Use this as the authoritative language list. Do NOT guess additional languages
            based on script characters — the audio-level detection is the ground truth.
            """
    
    # Score stays at 0.0 (not yet evaluated) until the first real evaluation completes.
    current_word_count = len(segment.split())
    last_eval_word_count = scores.get("_text_last_eval_word_count", 0)
    NEW_WORDS_THRESHOLD = 15  # Only re-evaluate when ≥15 new words arrived
    
    if current_word_count >= 5 and (current_word_count - last_eval_word_count) >= NEW_WORDS_THRESHOLD:
        try:
            prompt = f"""
            You are a professional grammar and language evaluator for interview transcripts.
            Evaluate the following spoken transcript fragment fairly but firmly.
            
            IMPORTANT CONTEXT:
            - This transcript comes from a speech-to-text (STT) engine processing LIVE audio.
            - STT may introduce artifacts, mis-transcriptions, and incorrect script characters.
            - The candidate may speak in multiple languages — this is NORMAL for multilingual speakers.
            - Do NOT rely on script characters for language identification.
            {lang_context}

            SCORING RUBRIC (out of 10):
            - 9-10: Near-flawless grammar and structure for spoken language. Very rare.
            - 7-8: Good grammar with minor slips. Professional and coherent. Natural fillers OK.
            - 5-6: Acceptable grammar. Some errors and fillers, but meaning is clear.
            - 3-4: Noticeable grammar issues, heavy fillers, fragmented sentences.
            - 1-2: Severely broken, incoherent, or impossible to understand.

            DEDUCTIONS (apply moderately):
            - Clear grammar errors (not STT artifacts): -0.5 each
            - Excessive filler words (um, uh — occasional ones are fine): -0.1 each beyond 3
            - Severely incomplete or fragmented sentences: -0.5 each
            - Unprofessional or inappropriate language: -0.5 each

            IMPORTANT GUIDELINES:
            - Multilingual code-switching (mixing languages) is NORMAL and should NOT be penalized.
              Many professionals naturally switch between languages. Evaluate grammar within each language.
            - Natural speech fillers (occasional "um", "uh", "like") are expected in spoken language.
              Only penalize EXCESSIVE filler usage.
            - STT transcription errors should NOT count against the speaker.
            - The minimum score for intelligible, meaningful speech should be 3/10.
            - Evaluate the SPEAKER's language ability, not the STT engine's accuracy.

            Transcript:
            {segment}

            Return ONLY JSON:
            {{
                "text_score out of 10": <number 0-10>,
                "is_regional_language": true/false,
                "detected_languages": {json.dumps(stt_langs) if stt_langs else '["unknown"]'},
                "error_count": <number of actual grammar errors found>,
                "filler_count": <number of filler words found>,
                "feedback": "short concise feedback listing specific issues found"
            }}
            """
            
            payload = {
                "model": settings.DEFAULT_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a professional grammar evaluator for spoken language. You evaluate fairly, accounting for natural speech patterns and multilingual speakers. You never penalize code-switching between languages."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            r = requests.post(settings.OPENAI_API_URL, headers=headers, json=payload, timeout=20.0)
            data = r.json()
            if "choices" in data:
                result = json.loads(data["choices"][0]["message"]["content"])
                
                # Extract text score with robust key matching
                raw_score = _extract_text_score(result)
                if raw_score is not None:
                    # Floor: intelligible speech that was transcribed should never be below 2.0
                    clamped = max(2.0, min(10.0, raw_score))
                    scores["text_performance_score"] = round(clamped, 2)
                else:
                    logger.warning("Text evaluation returned no parseable score. Result: %s", result)
                
                # Store feedback and error details for debugging
                if "feedback" in result:
                    scores["text_feedback"] = result["feedback"]
                if "error_count" in result:
                    scores["text_error_count"] = result["error_count"]
                if "filler_count" in result:
                    scores["text_filler_count"] = result["filler_count"]
                # Record word count at evaluation time for throttle gate
                scores["_text_last_eval_word_count"] = current_word_count
            else:
                logger.warning("Text evaluation API returned no choices. Response: %s", data)
        except Exception as e:
            logger.error("Text evaluation failed: %s", e)
            
    return {"scores": scores}
