from app.orchestrator.state import AgentState
import time

def process(state: AgentState) -> dict:
    """Real-time coaching alerts: corrections for weak areas AND praise for strong areas."""
    scores = state.get("scores", {})
    
    # Build the active alerts list based strictly on CURRENT conditions.
    # When conditions change, alerts update automatically on the next cycle.
    new_alerts = []
    
    # ==============================
    # VISUAL - Posture
    # ==============================
    posture = scores.get("posture_score", 5.0)
    if posture < 5.0:
        new_alerts.append({"message": "Posture: Sit upright with shoulders back - you appear slouched.", "level": "warning"})
    elif posture >= 7.0:
        new_alerts.append({"message": "Posture: Great posture - confident and professional!", "level": "success"})
    
    # ==============================
    # VISUAL - Eye Contact
    # ==============================
    eye_contact = scores.get("eye_contact_score", 5.0)
    if eye_contact < 5.0:
        new_alerts.append({"message": "Eye Contact: Look at the camera - you seem to be looking away.", "level": "warning"})
    elif eye_contact >= 7.0:
        new_alerts.append({"message": "Eye Contact: Excellent - steady and engaging!", "level": "success"})
    
    # ==============================
    # AUDIO - Volume & Pace
    # ==============================
    audio_score = scores.get("audio_performance_score", 5.0)
    if audio_score < 5.0:
        new_alerts.append({"message": "Audio: Speak louder and vary your tone - you sound low energy.", "level": "warning"})
    elif audio_score >= 7.0:
        new_alerts.append({"message": "Audio: Strong voice projection and good pacing!", "level": "success"})
    
    # ==============================
    # AUDIO - Background Noise
    # ==============================
    if scores.get("noise_detected", False):
        snr = scores.get("audio_snr_db", 0)
        new_alerts.append({
            "message": f"Background noise detected (SNR: {snr} dB) - move to a quieter environment.",
            "level": "warning"
        })
    
    # ==============================
    # TEXT - Grammar & Fluency
    # ==============================
    text_score = scores.get("text_performance_score", 0.0)
    if text_score > 0 and text_score < 4.0:
        new_alerts.append({"message": "Grammar: Focus on clearer sentences - reduce fillers.", "level": "warning"})
    elif text_score >= 7.0:
        new_alerts.append({"message": "Grammar: Articulate and professional language!", "level": "success"})
    
    # ==============================
    # RELEVANCE - Topic Alignment
    # ==============================
    rel_score = scores.get("relevance_score", 0.0)
    if rel_score > 0 and rel_score < 4.0:
        new_alerts.append({"message": "Relevance: You're drifting off-topic - refocus on the question.", "level": "warning"})
    elif rel_score >= 6.0:
        new_alerts.append({"message": "Relevance: Great - your answer is on-point!", "level": "success"})
    
    # ==============================
    # LANGUAGE DETECTION
    # ==============================
    if scores.get("language_limit_exceeded", False):
        lang_alert = scores.get("language_alert", "Too many languages detected.")
        new_alerts.append({"message": lang_alert, "level": "warning"})
    elif scores.get("regional_language_detected", False):
        all_langs = scores.get("all_detected_languages", [])
        if all_langs:
            new_alerts.append({"message": f"Languages: {', '.join(all_langs)}", "level": "info"})
            
    return {"active_alerts": new_alerts}
