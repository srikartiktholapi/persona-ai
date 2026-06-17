import librosa
import numpy as np
import os
import tempfile
import subprocess
import shutil
from pydub import AudioSegment
from sarvamai import SarvamAI
import librosa
import time
import numpy as np
import re
from app.core.config import settings
from app.orchestrator.state import AgentState

LANG_CODE_MAP = {
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "en-IN": "English",
    "gu-IN": "Gujarati",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "ne-IN": "Nepali",
    "kok-IN": "Konkani",
    "ks-IN": "Kashmiri",
    "sd-IN": "Sindhi",
    "sa-IN": "Sanskrit",
    "sat-IN": "Santali",
    "mni-IN": "Manipuri",
    "brx-IN": "Bodo",
    "mai-IN": "Maithili",
    "doi-IN": "Dogri",
}

SCRIPT_RANGES = [
    (0x0900, 0x097F, "Hindi"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Punjabi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
]


def _detect_languages_from_text(text: str) -> list[str]:
    script_char_counts = {}
    has_latin = False

    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            has_latin = True
            continue
        for start, end, lang in SCRIPT_RANGES:
            if start <= cp <= end:
                script_char_counts[lang] = script_char_counts.get(lang, 0) + 1
                break

    detected = []
    if has_latin:
        detected.append("English")
    detected.extend(
        lang
        for lang, count in sorted(script_char_counts.items(), key=lambda item: -item[1])
        if count >= 2
    )
    return detected


def _filter_languages(language_counts: dict, language_max_probability: dict) -> list[str]:
    confirmed = []
    for name, count in language_counts.items():
        confidence = language_max_probability.get(name, 0.0)
        fast_pass = count >= 1 and confidence >= 0.35
        slow_pass = count >= 2 and confidence >= 0.18
        script_pass = count >= 1 and confidence >= 0.90
        if fast_pass or slow_pass or script_pass:
            confirmed.append(name)
    return sorted(confirmed)

def process(state: AgentState) -> dict:
    """Live microphone frame mapping to Audio Performance Score"""
    scores = state.get("scores", {})
    features = state.get("recent_acoustic_features", [])
    
    if not features or "raw_audio" not in features[0]:
        return {"scores": scores}
        
    audio_data, sample_rate = features[0]["raw_audio"]
    
    try:
        # Convert av Audioframes to mono float32 for librosa
        if audio_data.dtype == np.int16:
            y = audio_data.mean(axis=0).astype(np.float32) / 32768.0
        else:
            y = audio_data.mean(axis=0).astype(np.float32)
            
        energy = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(energy))
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sample_rate)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0
        
        # --- SNR estimation ---
        # Use librosa split to find speech vs silence segments
        try:
            intervals = librosa.effects.split(y, top_db=25)
            if len(intervals) > 0 and len(y) > 0:
                speech_mask = np.zeros(len(y), dtype=bool)
                for start, end in intervals:
                    speech_mask[start:end] = True
                
                noise_samples = y[~speech_mask]
                speech_samples = y[speech_mask]
                
                if len(noise_samples) > 100 and len(speech_samples) > 100:
                    noise_power = float(np.mean(noise_samples ** 2)) + 1e-10
                    speech_power = float(np.mean(speech_samples ** 2)) + 1e-10
                    snr_db = 10 * np.log10(speech_power / noise_power)
                else:
                    snr_db = 20.0  # assume clean if not enough samples
            else:
                snr_db = 20.0
        except Exception:
            snr_db = 20.0
        
        scores["audio_snr_db"] = round(snr_db, 1)
        
        score = 3.0
        if avg_energy > 0.02:
            score += 3.5
        if pitch_std > 20:
            score += 3.5
        
        # --- Noise penalty ---
        if snr_db < 5:
            # Very noisy environment â€” significant penalty
            score -= 2.0
            scores["noise_detected"] = True
        elif snr_db < 10:
            # Moderate noise â€” small penalty
            score -= 1.0 
            scores["noise_detected"] = True
        else:
            scores["noise_detected"] = False
            
        scores["audio_performance_score"] = round(max(1.0, min(10.0, score)), 2)
    except Exception as e:
        pass  # keep previous score if chunk is silent/corrupt
        
    return {"recent_acoustic_features": [], "scores": scores}


def calculate_audio_metrics(transcript, duration_sec, silence_ratio=None):

 

    words = transcript.split()
    word_count = len(words)
    minutes = duration_sec / 60 if duration_sec else 1

    # -----------------------------
    # 1. WPM
    # -----------------------------
    wpm = round(word_count / minutes, 2)

    # -----------------------------
    # 2. Pattern-Based Filler Detection
    # -----------------------------
    text = transcript.lower()

    patterns = [
        r"\b(u+h+|u+m+|h+m+)\b",
        r"\b(\w+)\s+\1\b",
        r"\bso+\b",
        r"\bokay+\b",
        r"\b(like|actually|basically|well)\b"
    ]

    filler_count = sum(len(re.findall(p, text)) for p in patterns)

    filler_ratio = filler_count / word_count if word_count else 0
    filler_score = round(filler_ratio, 3)

    # -----------------------------
    # 3. Pause (USE AUDIO SIGNAL)
    # -----------------------------
    pause_rate = silence_ratio if silence_ratio is not None else 0

    # -----------------------------
    # 4. Interpretations
    # -----------------------------

    if word_count == 0:
        wpm_interp = "No spoken words were detected, so speech pace could not be evaluated."
    elif wpm < 100:
        wpm_interp = f"Measured pace is {wpm} WPM, which is slow and may sound hesitant."
    elif wpm < 130:
        wpm_interp = f"Measured pace is {wpm} WPM, slightly slow but understandable."
    elif wpm <= 170:
        wpm_interp = f"Measured pace is {wpm} WPM, ideal for professional communication."
    elif wpm <= 190:
        wpm_interp = f"Measured pace is {wpm} WPM, slightly fast but still understandable."
    else:
        wpm_interp = f"Measured pace is {wpm} WPM, too fast and may affect clarity."

    filler_pct = round(filler_score * 100, 1)
    if word_count == 0:
        filler_interp = "No transcript words were available for filler-word analysis."
    elif filler_score < 0.02:
        filler_interp = f"Detected {filler_count} filler pattern(s), {filler_pct}% of words, showing minimal hesitation."
    elif filler_score < 0.05:
        filler_interp = f"Detected {filler_count} filler pattern(s), {filler_pct}% of words, so hesitation is present but acceptable."
    elif filler_score < 0.1:
        filler_interp = f"Detected {filler_count} filler pattern(s), {filler_pct}% of words, which noticeably affects fluency."
    else:
        filler_interp = f"Detected {filler_count} filler pattern(s), {filler_pct}% of words, indicating high hesitation."

    pause_pct = round(pause_rate * 100, 1)
    if pause_rate < 0.15:
        pause_interp = f"Pause rate is {pause_pct}%, indicating smooth speech with minimal pauses."
    elif pause_rate < 0.3:
        pause_interp = f"Pause rate is {pause_pct}%, so moderate pauses were observed."
    else:
        pause_interp = f"Pause rate is {pause_pct}%, indicating frequent pauses and possible hesitation."

    return {
        "wpm": wpm,
        "wpm_interpretation": wpm_interp,

        "filler_score": filler_score,
        "filler_count": filler_count,
        "filler_interpretation": filler_interp,

        "pause_rate": round(pause_rate, 3),
        "pause_interpretation": pause_interp
    }
def analyze_audio_signal(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    # -----------------------------
    # Voice Energy
    # -----------------------------
    energy = librosa.feature.rms(y=y)[0]
    avg_energy = float(np.mean(energy))

    energy_value = round(avg_energy, 4)
    if avg_energy < 0.02:
        energy_interp = f"Voice energy is {energy_value}, which is low and may sound weak or monotone."
    elif avg_energy < 0.05:
        energy_interp = f"Voice energy is {energy_value}, suggesting steady speech delivery."
    else:
        energy_interp = f"Voice energy is {energy_value}, indicating enthusiastic speech."

    # -----------------------------
    # Pitch Variation
    # -----------------------------
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) == 0:
        pitch_std = 0
    else:
        pitch_std = float(np.std(pitch_values))

    pitch_value = round(pitch_std, 2)
    if pitch_std < 20:
        pitch_interp = f"Pitch variation is {pitch_value}, so speech may sound monotone."
    elif pitch_std < 50:
        pitch_interp = f"Pitch variation is {pitch_value}, indicating natural speech."
    else:
        pitch_interp = f"Pitch variation is {pitch_value}, indicating expressive speech."

    # -----------------------------
    # Silence Detection
    # -----------------------------
    intervals = librosa.effects.split(y, top_db=30)

    speech_duration = sum((end - start) for start, end in intervals)

    total_duration = len(y)

    silence_ratio = 1 - (speech_duration / total_duration) if total_duration else 1
    silence_pct = round(float(silence_ratio) * 100, 1)

    if silence_ratio < 0.15:
        silence_interp = f"Silence ratio is {silence_pct}%, so speech flow is smooth with minimal pauses."
    elif silence_ratio < 0.30:
        silence_interp = f"Silence ratio is {silence_pct}%, indicating moderate pauses."
    else:
        silence_interp = f"Silence ratio is {silence_pct}%, indicating frequent pauses or hesitation."

    return {
        "voice_energy": energy_value,      
        "energy_interpretation": energy_interp,

        "pitch_variation": pitch_value,
        "pitch_interpretation": pitch_interp,

        "silence_ratio": round(float(silence_ratio),3),
        "silence_interpretation": silence_interp
    }
def process_audio(video_path, prefix, sarvam_key):

    total_audio_start = time.time()

    # -----------------------------
    # Extract audio via ffmpeg directly
    # (MoviePy crashes on Chrome WebM files that have Duration:N/A)
    # -----------------------------
    start_extract = time.time()

    audio_file = os.path.join("uploads", f"{prefix}_audio.mp3")
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is not available on PATH. Install FFmpeg or add its bin folder "
            "to PATH, then restart Streamlit."
        )

    ffmpeg_cmd = [
        ffmpeg_path, "-y",
        "-i", video_path,
        "-vn",                   # no video
        "-acodec", "libmp3lame",
        "-q:a", "2",            # VBR ~190 kbps - good quality
        audio_file,
    ]
    try:
        subprocess.run(
            ffmpeg_cmd,
            capture_output=True,       # suppress ffmpeg console noise
            text=True,
            check=True,                # raise on non-zero exit
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        last_line = stderr.splitlines()[-1] if stderr else "No FFmpeg stderr output."
        raise RuntimeError(
            f"FFmpeg audio extraction failed: {last_line}\n"
            f"Command: {' '.join(ffmpeg_cmd)}"
        ) from e

    # Get duration from the re-encoded MP3 (always has proper metadata)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_file,
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 0.0

    extract_time = round(time.time() - start_extract, 2)

    sarvam_key = sarvam_key or settings.SARVAM_API_KEY
    client = SarvamAI(api_subscription_key=sarvam_key)

    def split_audio_to_chunks(audio_path, chunk_length_ms=29000):
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        start = 0

        while start < len(audio):
            end = start + chunk_length_ms
            chunk = audio[start:end]

            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            chunk.export(temp_path, format="wav")
            chunks.append(temp_path)

            start = end

        return chunks

    # -----------------------------
    # TRANSCRIPTION (Sarvam)
    # -----------------------------
    start_transcription = time.time()

    chunks = split_audio_to_chunks(audio_file)

    final_text = []
    sarvam_total_time = 0
    chunk_times = []
    language_counts = {}
    language_max_probability = {}

    for i, chunk_path in enumerate(chunks):
        with open(chunk_path, "rb") as f:

            start_api = time.time()

            resp = client.speech_to_text.transcribe(
                file=f,
                model=settings.SPEECH_TO_TEXT_MODEL,
                language_code=settings.SPEECH_LANGUAGE_CODE,
                mode="codemix",
            )

            api_time = round(time.time() - start_api, 2)

            sarvam_total_time += api_time
            chunk_times.append({
                "chunk": i + 1,
                "time_sec": api_time
            })

            text = (
                resp.transcript
                if hasattr(resp, "transcript")
                else resp.text if hasattr(resp, "text") else str(resp)
            )
            final_text.append(text)

            lang_code = getattr(resp, "language_code", None)
            lang_name = LANG_CODE_MAP.get(lang_code, lang_code) if lang_code else None
            probability = float(getattr(resp, "language_probability", None) or 0.0)
            if lang_name and lang_name != "unknown":
                language_counts[lang_name] = language_counts.get(lang_name, 0) + 1
                language_max_probability[lang_name] = max(
                    language_max_probability.get(lang_name, 0.0),
                    probability,
                )
            for script_lang in _detect_languages_from_text(text):
                language_counts[script_lang] = language_counts.get(script_lang, 0) + 1
                language_max_probability[script_lang] = max(
                    language_max_probability.get(script_lang, 0.0),
                    0.90,
                )
        os.remove(chunk_path)

    full_transcript = " ".join(final_text)
    detected_languages = _filter_languages(language_counts, language_max_probability)

    transcription_time = round(time.time() - start_transcription, 2)
    sarvam_total_time = round(sarvam_total_time, 2)

    # -----------------------------
    # SIGNAL ANALYSIS (librosa)
    # -----------------------------
    start_signal = time.time()

    audio_signal_metrics = analyze_audio_signal(audio_file)

    signal_time = round(time.time() - start_signal, 2)

    # -----------------------------
    #  TEXT METRICS
    # -----------------------------
    audio_metrics = calculate_audio_metrics(
    full_transcript,
    duration,
    silence_ratio=audio_signal_metrics["silence_ratio"]
)

    # -----------------------------
    # TOTAL AUDIO TIME
    # -----------------------------
    total_audio_time = round(time.time() - total_audio_start, 2)

    return {
        "full_transcript": full_transcript,
        "detected_languages": detected_languages,
        "audio_metrics": audio_metrics,
        "audio_signal_analysis": audio_signal_metrics,
        "speech_detected": bool(full_transcript and full_transcript.strip()),
        "timing": {
            "audio_extraction_time_sec": extract_time,
            "transcription_time_sec": transcription_time,
            "sarvam_api_time_sec": sarvam_total_time,
            "per_chunk_api_time": chunk_times,
            "signal_analysis_time_sec": signal_time,
            "total_audio_time_sec": total_audio_time
        }
    }
