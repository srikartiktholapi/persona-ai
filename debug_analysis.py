import asyncio
import os
from dotenv import load_dotenv
from app.agents.video import analyze_video
from app.agents.audio import process_audio
from app.agents.scoring import score_audio, evaluate_answer_relevance, calculate_body_score, interpret_body_language

load_dotenv()

def debug():
    video_path = "tests/mixed_beng+eng_bad.mp4"
    prefix = "mixed_beng+eng_bad"
    
    print("1. analyze_video...")
    body_data = analyze_video(video_path)
    print("body_data keys:", body_data.keys())

    print("2. process_audio...")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key: print("SARVAM_API_KEY is missing!")
    audio_data = process_audio(video_path, prefix, sarvam_key)
    print("audio_data keys:", audio_data.keys())

    print("3. score_audio...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key: print("OPENAI_API_KEY is missing!")
    audio_scores = score_audio(audio_data["full_transcript"], audio_data["audio_metrics"], openai_key)
    print("audio_scores:", audio_scores)

    print("4. evaluate_answer_relevance...")
    relevance = evaluate_answer_relevance("Introduce yourself", audio_data["full_transcript"], openai_key)
    print("relevance:", relevance)

    print("5. calculate_body_score...")
    body_scores = calculate_body_score(body_data)
    print("body_scores:", body_scores)

    print("6. interpret_body_language...")
    body_interpretation = interpret_body_language(body_scores, body_data.get("events", []), openai_key)
    print("body_interpretation:", body_interpretation)

    print("7. Extract numeric values...")
    body_val = int(float(body_scores["body_total"]))
    audio_val = int(audio_scores.get("total_audio_score out of 30", 0))
    relevance_val = int(relevance.get("relevance_score out of 50", 0))

    print("Done!")

if __name__ == "__main__":
    try:
        debug()
    except Exception as e:
        import traceback
        traceback.print_exc()
