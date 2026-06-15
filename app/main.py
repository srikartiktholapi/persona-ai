'''from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title="Persona AI - Agentic Workflow V2",
    description="Multimodal session orchestrator using LangGraph",
    version="2.0.0",
)

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Persona AI V2 Multimodal Orchestrator API"}'''
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
import uuid
import time
import requests
from fastapi import Body
from app.frontend_formatter import format_for_frontend
from app.agents.video import analyze_video
from app.agents.audio import process_audio
from app.agents.scoring import (
    score_audio, evaluate_answer_relevance, interpret_body_language,
    calculate_body_score, calculate_audio_total
)

def download_video(video_url: str, save_dir="uploads"):
    os.makedirs(save_dir, exist_ok=True)
    video_id = str(uuid.uuid4())
    video_path = os.path.join(save_dir, f"{video_id}.mp4")
    with requests.get(video_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
    return video_path, video_id

load_dotenv()
app = FastAPI()
@app.get("/")
def root():
    return {"message": "Welcome to Persona AI API. API is running."}

@app.post("/analyze")
async def start_analysis(file: UploadFile = File(...)):

    overall_start = time.time()

    os.makedirs("uploads", exist_ok=True)
    video_path = f"uploads/{file.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prefix = os.path.splitext(file.filename)[0]

    # =========================
    # BODY LANGUAGE
    # =========================
    start_body = time.time()
    body_data = analyze_video(video_path)
    body_time = round(time.time() - start_body, 2)

    # =========================
    # AUDIO PROCESSING
    # =========================
    start_audio = time.time()
    audio_data = process_audio(
        video_path,
        prefix,
        os.getenv("SARVAM_API_KEY")
    )
    audio_time = round(time.time() - start_audio, 2)

    # =========================
    # AI SCORING
    # =========================
    start_ai = time.time()

    audio_scores = score_audio(
        audio_data["full_transcript"],
        audio_data["audio_metrics"],
        os.getenv("OPENAI_API_KEY")
    )



    relevance = evaluate_answer_relevance(
        "Introduce yourself",
        audio_data["full_transcript"],
        os.getenv("OPENAI_API_KEY")
    )
    relevance_val = relevance.get("relevance_score", 0)

    ai_time = round(time.time() - start_ai, 2)

    # =========================
    # ⏱️ TOTAL TIME
    # =========================
    total_time = round(time.time() - overall_start, 2)

    # =========================
    # SCORING
    # =========================
    body_scores = calculate_body_score(body_data)
    #audio_totals = calculate_audio_total(audio_scores)
    body_interpretation = interpret_body_language(
        body_scores,
        body_data.get("events", []),
        os.getenv("OPENAI_API_KEY")
    )

# Extract numeric values
    body_val = int(float(body_scores["body_total"]))
    #audio_val = int(audio_totals["total_audio_score"].split("/")[0])
    audio_val = int(audio_scores.get("total_audio_score out of 30", 0))
    relevance_val = int(relevance.get("relevance_score out of 50", 0))

# 🔥 FINAL TOTAL (OUT OF 100)
    final_total = body_val + audio_val + relevance_val
    overall = f"{final_total}/100"

    raw_result = {
        "status": "success",
        "filename": file.filename,
        "results": {
            "body_language": {
                "scores": body_scores,
                "interpretation": body_interpretation
            },
            "audio_analysis": audio_data["audio_metrics"],
            "audio_signal_analysis": audio_data.get("audio_signal_analysis"),
            "speech_ai_scores": audio_scores,
            "answer_relevance": relevance,
            "overall_score": overall,
            "score_breakdown": {
                "body_score": f"{body_val}/20",
                "audio_score": f"{audio_val}/30",
                "relevance_score": f"{relevance_val}/50"
            },
            "transcript": audio_data["full_transcript"]
        },
        "timing": {
            "body_language_time_sec": body_time,
            "audio_pipeline_time_sec": audio_time,
            "ai_scoring_time_sec": ai_time,
            "total_time_sec": total_time
        },
        "audio_timing_breakdown": audio_data.get("timing")
    }

    return format_for_frontend(raw_result)
@app.post("/analyze-url")
async def start_analysis_from_url(payload: dict = Body(...)):
    """
    Expected JSON:
    {
        "video_url": "https://example.com/video.mp4"
    }
    """

    video_url = payload.get("video_url")
    if not video_url:
        return {"status": "error", "message": "video_url is required"}

    try:
        # Download video
        video_path, prefix = download_video(video_url)

        # 1. Body Language Analysis
        body_data = analyze_video(video_path)

        # 2. Audio Transcription
        audio_data = process_audio(
            video_path,
            prefix,
            os.getenv("SARVAM_API_KEY")
        )

        # 3. Audio Scoring
        audio_scores = score_audio(
            audio_data["full_transcript"],
            os.getenv("OPENAI_API_KEY")
        )

        raw_result = {
            "status": "success",
            "source": "url",
            "video_url": video_url,
            "results": {
                "body_language": body_data,
                "speech_analysis": audio_scores,
                "transcript": audio_data["full_transcript"]
            }
        }

        return format_for_frontend(raw_result)

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
