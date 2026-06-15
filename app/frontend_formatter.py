import re


def _safe_float(value, default=0.0):
    """Convert a value to float, falling back to a default when needed."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _parse_score(value, default=0.0):
    """Parse numeric values from strings such as '42/50' or '6.7/10'."""
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
        if not numbers:
            return default

        # Convert out-of-100 scores to a 0-10 scale when appropriate.
        if "/" in value and len(numbers) >= 2 and numbers[1] == 100:
            return round(numbers[0] / 10.0, 2)

        return float(numbers[0])

    return default


def format_for_frontend(raw_result):
    """
    Transform the legacy API response into the shape expected by Streamlit.

    The frontend reads these fields directly from the `results` dictionary:
      - results.get("scores", {})
      - results.get("audio_metrics", {})
      - results.get("audio_signal_analysis", {})
      - results.get("audio_llm", {})
      - results.get("video_metrics", {})
      - results.get("body_llm", {})
      - results.get("text_result", {})
      - results.get("transcript", "")
      - results.get("total_time_sec")
    """
    if not isinstance(raw_result, dict):
        return {}

    legacy_results = raw_result.get("results", {}) if isinstance(raw_result.get("results"), dict) else {}
    legacy_timing = raw_result.get("timing", {}) if isinstance(raw_result.get("timing"), dict) else {}

    body_language = legacy_results.get("body_language", {}) if isinstance(legacy_results.get("body_language"), dict) else {}
    body_scores = body_language.get("scores", {}) if isinstance(body_language.get("scores"), dict) else {}
    body_llm = body_language.get("interpretation", {}) if isinstance(body_language.get("interpretation"), dict) else {}

    audio_metrics = legacy_results.get("audio_analysis", legacy_results.get("audio_metrics", {}))
    if not isinstance(audio_metrics, dict):
        audio_metrics = {}

    audio_signal_analysis = legacy_results.get("audio_signal_analysis", {})
    if not isinstance(audio_signal_analysis, dict):
        audio_signal_analysis = {}

    audio_llm = legacy_results.get("speech_ai_scores", legacy_results.get("audio_llm", {}))
    if not isinstance(audio_llm, dict):
        audio_llm = {}

    relevance_result = legacy_results.get("answer_relevance", legacy_results.get("relevance_result", {}))
    if not isinstance(relevance_result, dict):
        relevance_result = {}

    text_result = legacy_results.get("text_result", {})
    if not isinstance(text_result, dict):
        text_result = {}

    transcript = legacy_results.get("transcript", "")

    posture_score = _safe_float(
        body_scores.get("posture_score out of 5", body_scores.get("posture_score", 0.0)),
        0.0,
    )
    eye_contact_score = _safe_float(
        body_scores.get("eye_contact_score out of 5", body_scores.get("eye_contact_score", 0.0)),
        0.0,
    )
    expression_score = _safe_float(
        body_scores.get("expression_score out of 10", body_scores.get("expression_score", 0.0)),
        0.0,
    )
    body_total = _safe_float(body_scores.get("body_total", 0.0), 0.0)

    clarity_score = _safe_float(
        audio_llm.get("speech_clarity_score out of 10", audio_llm.get("speech_clarity_score", 0.0)),
        0.0,
    )
    confidence_score = _safe_float(
        audio_llm.get("confidence_score out of 10", audio_llm.get("confidence_score", 0.0)),
        0.0,
    )
    communication_score = _safe_float(
        audio_llm.get("communication_score out of 10", audio_llm.get("communication_score", 0.0)),
        0.0,
    )

    text_quality = _safe_float(
        text_result.get("text_score out of 10", text_result.get("text_score", 5.0)),
        5.0,
    )
    rel_raw = _safe_float(
        relevance_result.get("relevance_score out of 50", relevance_result.get("relevance_score", 0.0)),
        0.0,
    )
    relevance_score = round(max(0.0, min(10.0, rel_raw / 5.0)), 2) if rel_raw is not None else 5.0

    visual_score = round((body_total / 20.0) * 10.0, 2) if body_total else 0.0
    audio_score = round((clarity_score + confidence_score + communication_score) / 3.0, 2) if any(
        (clarity_score, confidence_score, communication_score)
    ) else 0.0

    parsed_overall = _parse_score(legacy_results.get("overall_score", 0.0), default=0.0)
    fallback_overall = round(
        0.20 * (visual_score if visual_score is not None else 5.0)
        + 0.25 * (audio_score if audio_score is not None else 5.0)
        + 0.20 * (text_quality if text_quality is not None else 5.0)
        + 0.35 * (relevance_score if relevance_score is not None else 5.0),
        2,
    )
    overall_score = parsed_overall if parsed_overall > 0 else fallback_overall
    if parsed_overall > 10:
        overall_score = round(parsed_overall / 10.0, 2)

    scores = {
        "visual_performance_score": round(visual_score, 2),
        "posture_score": round(posture_score, 2),
        "eye_contact_score": round(eye_contact_score, 2),
        "expression_score": round(expression_score, 2),
        "head_stability_score": _safe_float(body_scores.get("head_stability_score", 0.0), 0.0),
        "video_analysis_available": True if body_scores else False,
        "video_analysis_status": body_scores.get("video_analysis_status", ""),
        "audio_performance_score": round(audio_score, 2),
        "speech_clarity_score": round(clarity_score, 2),
        "confidence_score": round(confidence_score, 2),
        "communication_score": round(communication_score, 2),
        "text_performance_score": round(text_quality, 2),
        "relevance_score": round(relevance_score, 2),
        "relevance_label": relevance_result.get("relevance", ""),
        "relevance_reason": relevance_result.get("reason", ""),
        "relevance_deductions": relevance_result.get("deductions_applied", []),
        "overall_score": round(overall_score, 2),
    }

    text_payload = dict(text_result)
    text_payload.setdefault("feedback", relevance_result.get("reason", text_result.get("feedback", "")))
    text_payload.setdefault("error_count", 0)
    text_payload.setdefault("filler_count", 0)
    text_payload.setdefault("detected_languages", legacy_results.get("detected_languages", []))
    text_payload.setdefault("text_score", text_quality)
    text_payload.setdefault("text_score out of 10", text_quality)

    return {
        "scores": scores,
        "audio_metrics": audio_metrics,
        "audio_signal_analysis": audio_signal_analysis,
        "audio_llm": audio_llm,
        "video_metrics": body_scores,
        "body_llm": body_llm,
        "text_result": text_payload,
        "transcript": transcript,
        "total_time_sec": legacy_timing.get("total_time_sec", raw_result.get("total_time_sec", 0.0)),
        "enhanced_prompt": legacy_results.get("enhanced_prompt", ""),
        "detected_languages": legacy_results.get("detected_languages", []),
        "relevance_result": relevance_result,
    }
