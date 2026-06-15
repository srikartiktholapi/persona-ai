"""
Batch analysis pipeline for post-session video/audio evaluation.
Replaces the real-time LangGraph streaming graph with a sequential
batch processor suited for uploaded/recorded video files.
"""
import os
import json
import time
import logging
import tempfile
import requests
import numpy as np

logger = logging.getLogger(__name__)


def run_batch_analysis(
    video_path: str,
    prompt: str,
    performer_role: str,
    target_role: str,
    progress_fn=None,
) -> dict:
    """
    Run the full post-session analysis pipeline on a recorded video file.

    Args:
        video_path:     Absolute path to the saved video file (mp4/webm/mov).
        prompt:         The evaluation prompt / question.
        performer_role: Name of the performer persona.
        target_role:    Name of the target audience persona.
        progress_fn:    Optional callable(step: int, total: int, label: str)
                        so the caller (Streamlit) can update a progress bar.

    Returns:
        A dict with keys:
            transcript, audio_metrics, audio_signal_analysis,
            audio_llm, video_metrics, body_llm,
            text_result, relevance_result, scores, timing
    """
    from app.core.config import settings
    from app.agents.audio import process_audio
    from app.agents.video import analyze_video
    from app.agents.scoring import (
        score_audio, evaluate_answer_relevance,
        body_language_interpretation, calculate_body_score,
        calculate_audio_total,
    )
    from app.agents.text import score_text
    from app.core.persona import persona_framework

    total_steps = 7
    step = 0

    def _progress(label: str):
        nonlocal step
        step += 1
        if progress_fn:
            progress_fn(step, total_steps, label)
        logger.info("[BatchAnalyser] Step %d/%d — %s", step, total_steps, label)

    t0 = time.time()
    results: dict = {}

    # ── 1. AUDIO EXTRACTION + TRANSCRIPTION ─────────────────────────────────
    _progress("Extracting audio & transcribing speech…")
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    audio_result = process_audio(video_path, prefix, settings.SARVAM_API_KEY)

    transcript            = audio_result.get("full_transcript", "")
    detected_languages    = audio_result.get("detected_languages", [])
    audio_metrics         = audio_result.get("audio_metrics", {})
    audio_signal_analysis = audio_result.get("audio_signal_analysis", {})

    results["transcript"]            = transcript
    results["detected_languages"]    = detected_languages
    results["audio_metrics"]         = audio_metrics
    results["audio_signal_analysis"] = audio_signal_analysis
    results["audio_timing"]          = audio_result.get("timing", {})

    # ── 2. VIDEO ANALYTICS ──────────────────────────────────────────────────
    _progress("Analysing video — posture, eye contact, expression…")
    video_metrics = analyze_video(video_path)
    results["video_metrics"] = video_metrics

    # ── 3. LLM — AUDIO QUALITY SCORING ────────────────────────────────────
    _progress("Evaluating speech quality with LLM…")
    try:
        audio_llm = score_audio(transcript, audio_metrics, settings.OPENAI_API_KEY)
    except Exception as e:
        logger.warning("Audio LLM scoring failed: %s", e)
        audio_llm = {}
    results["audio_llm"] = audio_llm

    # ── 4. LLM — BODY LANGUAGE NARRATIVE ──────────────────────────────────
    _progress("Generating body language interpretation…")
    video_available = video_metrics.get("video_analysis_available", True)
    try:
        if not video_available:
            raise ValueError(video_metrics.get("video_analysis_status", "Video analysis unavailable."))
        body_scores = calculate_body_score(video_metrics)
        body_llm = body_language_interpretation(body_scores, settings.OPENAI_API_KEY)
    except Exception as e:
        logger.warning("Body language LLM failed: %s", e)
        body_llm = {}
        body_scores = {}
    results["body_scores"] = body_scores
    results["body_llm"]    = body_llm

    # ── 5. TEXT ANALYTICS (grammar, fluency) ────────────────────────────────
    _progress("Scoring text quality & grammar…")
    try:
        text_result = score_text(transcript, detected_languages, settings.OPENAI_API_KEY)
        if detected_languages and text_result.get("detected_languages") in (None, [], ["unknown"]):
            text_result["detected_languages"] = detected_languages
    except Exception as e:
        logger.warning("Text scoring failed: %s", e)
        text_result = {"detected_languages": detected_languages}
    results["text_result"] = text_result

    # ── 6. RELEVANCE SCORING ────────────────────────────────────────────────
    _progress("Evaluating prompt relevance…")
    try:
        # Enrich prompt with persona context
        enhanced_prompt = prompt
        if performer_role and target_role:
            p = persona_framework.get_role(performer_role)
            t = persona_framework.get_role(target_role)
            if p and t:
                enhanced_prompt = (
                    f"Performer role: {p.name} — {p.requirements['description']}. "
                    f"Target audience: {t.name} — {t.requirements['description']}. "
                    f"{prompt}"
                )
        relevance_result = evaluate_answer_relevance(
            enhanced_prompt, transcript, settings.OPENAI_API_KEY
        )
    except Exception as e:
        logger.warning("Relevance scoring failed: %s", e)
        relevance_result = {}
    results["relevance_result"]  = relevance_result
    results["enhanced_prompt"]   = enhanced_prompt

    # ── 7. COMPUTE OVERALL SCORE ────────────────────────────────────────────
    _progress("Computing final scores…")

    # Video sub-scores (0–10 each)
    if video_available:
        posture_score    = video_metrics.get("posture_score", 5.0)
        eye_score        = video_metrics.get("eye_contact_score", 5.0)
        expr_score       = video_metrics.get("expression_score", 5.0)
        head_score       = video_metrics.get("head_stability", 5.0)
        visual_score     = round(
            0.35 * posture_score + 0.30 * eye_score + 0.20 * expr_score + 0.15 * head_score, 2
        )
    else:
        posture_score = eye_score = expr_score = head_score = None
        visual_score = None

    # Audio LLM sub-scores (0–10 each)
    clarity_score    = audio_llm.get("speech_clarity_score out of 10",
                       audio_llm.get("speech_clarity_score", 5.0))
    confidence_score = audio_llm.get("confidence_score out of 10",
                       audio_llm.get("confidence_score", 5.0))
    comm_score       = audio_llm.get("communication_score out of 10",
                       audio_llm.get("communication_score", 5.0))
    audio_score      = round((float(clarity_score) + float(confidence_score) + float(comm_score)) / 3, 2)

    # Text score (0–10)
    raw_text = text_result.get("text_score out of 10", text_result.get("text_score", 5.0))
    text_score = round(max(0.0, min(10.0, float(raw_text))), 2) if raw_text else 5.0

    # Relevance score (0–10, converted from 0–50)
    raw_rel = relevance_result.get("relevance_score out of 50",
              relevance_result.get("relevance_score", 25.0))
    rel_score = round(max(0.0, min(10.0, float(raw_rel) / 5.0)), 2) if raw_rel is not None else 5.0

    # Weighted overall: 20% video, 25% audio, 20% text, 35% relevance
    overall_visual_score = visual_score if visual_score is not None else 5.0
    overall = round(
        0.20 * overall_visual_score +
        0.25 * audio_score  +
        0.20 * text_score   +
        0.35 * rel_score,
        2
    )

    scores = {
        "visual_performance_score": visual_score,
        "posture_score":            round(posture_score, 2) if posture_score is not None else None,
        "eye_contact_score":        round(eye_score, 2) if eye_score is not None else None,
        "expression_score":         round(expr_score, 2) if expr_score is not None else None,
        "head_stability_score":     round(head_score, 2) if head_score is not None else None,
        "video_analysis_available": video_available,
        "video_analysis_status":    video_metrics.get("video_analysis_status", ""),

        "audio_performance_score":  audio_score,
        "speech_clarity_score":     round(float(clarity_score), 2),
        "confidence_score":         round(float(confidence_score), 2),
        "communication_score":      round(float(comm_score), 2),

        "text_performance_score":   text_score,

        "relevance_score":          rel_score,
        "relevance_label":          relevance_result.get("relevance", ""),
        "relevance_reason":         relevance_result.get("reason", ""),
        "relevance_deductions":     relevance_result.get("deductions_applied", []),

        "overall_score":            overall,
    }
    results["scores"] = scores
    results["total_time_sec"] = round(time.time() - t0, 1)

    return results
