"""
Full pipeline test on uploaded videos.
Runs: Video analysis → Audio processing → AI scoring → Language detection → Results
"""
import os
import sys
import time
import tempfile
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.core.config import settings
from sarvamai import SarvamAI
from app.agents.video import analyze_video
from app.agents.audio import process_audio, analyze_audio_signal, calculate_audio_metrics
from app.agents.scoring import (
    score_audio, evaluate_answer_relevance,
    calculate_body_score, interpret_body_language
)

# BCP-47 language code to human-readable name mapping
LANG_CODE_MAP = {
    "hi-IN": "Hindi", "bn-IN": "Bengali", "kn-IN": "Kannada",
    "ml-IN": "Malayalam", "mr-IN": "Marathi", "od-IN": "Odia",
    "pa-IN": "Punjabi", "ta-IN": "Tamil", "te-IN": "Telugu",
    "en-IN": "English", "gu-IN": "Gujarati", "as-IN": "Assamese",
    "ur-IN": "Urdu", "ne-IN": "Nepali", "kok-IN": "Konkani",
    "ks-IN": "Kashmiri", "sd-IN": "Sindhi", "sa-IN": "Sanskrit",
    "sat-IN": "Santali", "mni-IN": "Manipuri", "brx-IN": "Bodo",
    "mai-IN": "Maithili", "doi-IN": "Dogri",
}

MIN_PERCENTAGE = 0.10  # Must be at least 10% of all detections
MIN_CONFIDENCE = 0.5   # Must have at least 0.5 max confidence


def detect_languages_from_video(video_path):
    """Extract audio, split into chunks, transcribe with Saaras V3 codemix, detect languages."""
    print("\n📝 LANGUAGE DETECTION (Saaras V3 + Codemix)")
    print("-" * 50)

    # Extract audio
    audio_path = video_path.rsplit(".", 1)[0] + "_test_audio.wav"
    with VideoFileClip(video_path) as video:
        video.audio.write_audiofile(audio_path, logger=None)

    # Split into chunks
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    start = 0
    while start < len(audio):
        end = start + 10000
        chunk = audio[start:end]
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        chunk.export(temp_path, format="wav")
        chunks.append(temp_path)
        start = end

    print(f"   Created {len(chunks)} audio chunks\n")

    client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
    lang_counts = {}
    lang_max_prob = {}
    all_transcripts = []

    for i, chunk_path in enumerate(chunks):
        try:
            with open(chunk_path, "rb") as f:
                resp = client.speech_to_text.transcribe(
                    file=f,
                    model=settings.SPEECH_TO_TEXT_MODEL,
                    language_code=settings.SPEECH_LANGUAGE_CODE,
                    mode="codemix"
                )

            text = resp.transcript if hasattr(resp, "transcript") else (resp.text if hasattr(resp, "text") else str(resp))
            lang_code = getattr(resp, "language_code", None)
            probability = getattr(resp, "language_probability", None) or 0.0
            lang_name = LANG_CODE_MAP.get(lang_code, lang_code) if lang_code else "unknown"

            print(f"   Chunk {i+1}: {lang_name} ({lang_code}) | Conf: {probability:.3f}")
            print(f"           {text[:70]}{'...' if len(text) > 70 else ''}")

            all_transcripts.append(text)

            if lang_name and lang_name != "unknown":
                lang_counts[lang_name] = lang_counts.get(lang_name, 0) + 1
                lang_max_prob[lang_name] = max(lang_max_prob.get(lang_name, 0.0), probability)

        except Exception as e:
            print(f"   Chunk {i+1}: ERROR - {e}")
        finally:
            os.remove(chunk_path)

    # Apply percentage-based filtering (same as speech.py)
    total_detections = sum(lang_counts.values())
    confirmed_langs = []
    for name, count in lang_counts.items():
        pct = count / total_detections if total_detections > 0 else 0
        max_conf = lang_max_prob.get(name, 0.0)
        if pct >= MIN_PERCENTAGE and max_conf >= MIN_CONFIDENCE:
            confirmed_langs.append(name)

    filtered_out = [l for l in lang_counts if l not in confirmed_langs]

    print(f"\n   Raw counts: {lang_counts} (total: {total_detections})")
    print(f"   Max confidence: {lang_max_prob}")
    print(f"   Percentages: {{{', '.join(f'{n}: {c/total_detections:.0%}' for n, c in lang_counts.items())}}}")
    print(f"   Filter: ≥{MIN_PERCENTAGE:.0%} of detections AND ≥{MIN_CONFIDENCE} confidence")
    print(f"   ✅ Confirmed: {confirmed_langs}")
    if filtered_out:
        print(f"   ❌ Filtered out: {filtered_out}")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return confirmed_langs, " ".join(all_transcripts)


def run_full_pipeline(video_path, prompt="Introduce yourself"):
    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE TEST")
    print(f"   File: {video_path}")
    print(f"   Model: {settings.SPEECH_TO_TEXT_MODEL}")
    print(f"   Prompt: {prompt}")
    print(f"{'='*60}")

    overall_start = time.time()

    # =====================
    # 1. LANGUAGE DETECTION (Saaras V3 codemix)
    # =====================
    start = time.time()
    detected_languages, codemix_transcript = detect_languages_from_video(video_path)
    lang_time = round(time.time() - start, 2)

    # =====================
    # 2. BODY LANGUAGE
    # =====================
    print("\n BODY LANGUAGE ANALYSIS")
    print("-" * 50)
    start = time.time()
    body_data = analyze_video(video_path)
    body_time = round(time.time() - start, 2)
    body_scores = calculate_body_score(body_data)
    visual_score = round(body_scores["body_total"] / 2, 2)  # Convert /20 to /10
    print(f"   Visual Performance Score: {visual_score}/10")
    print(f"   Time: {body_time}s")

    # =====================
    # 3. AUDIO PROCESSING
    # =====================
    print("\n AUDIO PROCESSING")
    print("-" * 50)
    start = time.time()
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    audio_data = process_audio(video_path, prefix, settings.SARVAM_API_KEY)
    audio_time = round(time.time() - start, 2)
    transcript = audio_data["full_transcript"]
    print(f"   Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    print(f"   Time: {audio_time}s")

    # =====================
    # 4. AI SCORING
    # =====================
    print("\n AI SCORING")
    print("-" * 50)
    start = time.time()

    audio_scores = score_audio(transcript, audio_data["audio_metrics"], settings.OPENAI_API_KEY)
    audio_total = audio_scores.get("total_audio_score out of 30", 0)
    audio_perf = round(audio_total / 3, 2)  # Convert /30 to /10
    print(f"   Audio Performance Score: {audio_perf}/10")

    relevance = evaluate_answer_relevance(prompt, transcript, settings.OPENAI_API_KEY)
    relevance_val = relevance.get("relevance_score out of 50", 0)
    relevance_score = round(relevance_val / 5, 2)  # Convert /50 to /10
    print(f"   Relevance Score: {relevance_score}/10")

    # Text score (estimate from transcript quality)
    text_score = 1.0  # Will be evaluated by the streaming pipeline
    print(f"   Text Score: {text_score}/10 (batch estimate)")

    ai_time = round(time.time() - start, 2)
    print(f"   Time: {ai_time}s")

    # =====================
    # 5. OVERALL SCORE
    # =====================
    overall = round(
        (visual_score * 0.20) +
        (audio_perf * 0.30) +
        (text_score * 0.20) +
        (relevance_score * 0.30),
        2
    )

    total_time = round(time.time() - overall_start, 2)

    # =====================
    # RESULTS DISPLAY
    # =====================
    print(f"\n{'='*60}")
    print(f" RESULTS")
    print(f"{'='*60}")

    print(f"\n   Final Overall Score")
    print(f"   {overall} / 10\n")

    print(f"   Visual Performance")
    print(f"   {visual_score} / 10\n")

    print(f"   Audio Performance")
    print(f"   {audio_perf} / 10\n")

    print(f"   Text Quality & Grammar")
    print(f"   {text_score} / 10\n")

    print(f"   Prompt Relevance")
    print(f"   {relevance_score} / 10\n")

    print(f"    Languages Spoken Profile: {', '.join(detected_languages)}\n")

    # Key Takeaways
    print("   Key Takeaways")
    if visual_score < 5:
        print("     Visuals: Remember to maintain steady posture and eye contact.")
    else:
        print("     Visuals: Good body language and engagement!")

    if audio_perf < 5:
        print("     Audio: Try to modulate your speaking pace and pitch.")
    else:
        print("     Audio: Strong pacing and volume control!")

    if len(detected_languages) > 3:
        print(f"     Language: Too many languages detected ({len(detected_languages)}). Limit to 3 max.")
    elif len(detected_languages) > 1:
        non_eng = [l for l in detected_languages if l.lower() != "english"]
        if non_eng:
            print(f"    Language: Regional language detected ({', '.join(non_eng)}).")

    print(f"\n     Total Time: {total_time}s")
    print(f"       Language Detection: {lang_time}s | Body: {body_time}s | Audio: {audio_time}s | AI: {ai_time}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = "uploads/mixed_beng+eng_bad.mp4"

    if not os.path.exists(video_file):
        print(f"Error: File not found: {video_file}")
        sys.exit(1)

    run_full_pipeline(video_file)
