import librosa
import numpy as np
import os
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from sarvamai import SarvamAI
import librosa
import time
import numpy as np
import re
from app.core.config import settings
from app.orchestrator.state import AgentState

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
            # Very noisy environment — significant penalty
            score -= 2.0
            scores["noise_detected"] = True
        elif snr_db < 10:
            # Moderate noise — small penalty
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

    filler_score = (
        filler_count / word_count if word_count else 0
    )

    filler_score = round(filler_score, 3)

    # -----------------------------
    # 3. Pause (USE AUDIO SIGNAL)
    # -----------------------------
    pause_rate = silence_ratio if silence_ratio is not None else 0

    # -----------------------------
    # 4. Interpretations
    # -----------------------------

    if wpm < 100:
        wpm_interp = "Speech pace is slow and may sound hesitant."
    elif wpm < 130:
        wpm_interp = "Speech pace is slightly slow but understandable."
    elif wpm <= 170:
        wpm_interp = "Speech pace is ideal for professional communication."
    elif wpm <= 190:
        wpm_interp = "Speech pace is slightly fast but still understandable."
    else:
        wpm_interp = "Speech pace is too fast and may affect clarity."

    if filler_score < 0.02:
        filler_interp = "Minimal hesitation with fluent speech."
    elif filler_score < 0.05:
        filler_interp = "Some hesitation present but acceptable."
    elif filler_score < 0.1:
        filler_interp = "Noticeable hesitation affecting fluency."
    else:
        filler_interp = "High hesitation and disfluency detected."

    if pause_rate < 0.15:
        pause_interp = "Smooth speech with minimal pauses."
    elif pause_rate < 0.3:
        pause_interp = "Moderate pauses observed."
    else:
        pause_interp = "Frequent pauses indicating hesitation."

    return {
        "wpm": wpm,
        "wpm_interpretation": wpm_interp,

        "filler_score": filler_score,
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

    if avg_energy < 0.02:
        energy_interp = "Low vocal energy indicating weak or monotone speech."
    elif avg_energy < 0.05:
        energy_interp = "Moderate vocal energy suggesting steady speech delivery."
    else:
        energy_interp = "High vocal energy indicating enthusiastic speech."

    # -----------------------------
    # Pitch Variation
    # -----------------------------
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) == 0:
        pitch_std = 0
    else:
        pitch_std = float(np.std(pitch_values))

    if pitch_std < 20:
        pitch_interp = "Very little pitch variation; speech may sound monotone."
    elif pitch_std < 50:
        pitch_interp = "Moderate pitch variation indicating natural speech."
    else:
        pitch_interp = "High pitch variation indicating expressive speech."

    # -----------------------------
    # Silence Detection
    # -----------------------------
    intervals = librosa.effects.split(y, top_db=30)

    speech_duration = sum((end - start) for start, end in intervals)

    total_duration = len(y)

    silence_ratio = 1 - (speech_duration / total_duration)

    if silence_ratio < 0.15:
        silence_interp = "Speech flow is smooth with minimal pauses."
    elif silence_ratio < 0.30:
        silence_interp = "Moderate pauses detected in speech."
    else:
        silence_interp = "Frequent pauses detected indicating hesitation."

    return {
        "voice_energy": round(avg_energy,4),
        "energy_interpretation": energy_interp,

        "pitch_variation": round(pitch_std,2),
        "pitch_interpretation": pitch_interp,

        "silence_ratio": round(float(silence_ratio),3),
        "silence_interpretation": silence_interp
    }
def process_audio(video_path, prefix, sarvam_key):

    total_audio_start = time.time()

    # -----------------------------
    # Extract audio
    # -----------------------------
    start_extract = time.time()

    with VideoFileClip(video_path) as video:
        audio_file = f"uploads/{prefix}_audio.mp3"
        video.audio.write_audiofile(audio_file, logger=None)
        duration = video.duration

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

    for i, chunk_path in enumerate(chunks):
        with open(chunk_path, "rb") as f:

            start_api = time.time()

            resp = client.speech_to_text.translate(
                file=f,
                model=settings.SPEECH_TO_TEXT_MODEL
            )

            api_time = round(time.time() - start_api, 2)

            sarvam_total_time += api_time
            chunk_times.append({
                "chunk": i + 1,
                "time_sec": api_time
            })

            text = resp.text if hasattr(resp, "text") else str(resp)
            final_text.append(text)

        os.remove(chunk_path)

    full_transcript = " ".join(final_text)

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
        "audio_metrics": audio_metrics,
        "audio_signal_analysis": audio_signal_metrics,

        "timing": {
            "audio_extraction_time_sec": extract_time,
            "transcription_time_sec": transcription_time,
            "sarvam_api_time_sec": sarvam_total_time,
            "per_chunk_api_time": chunk_times,
            "signal_analysis_time_sec": signal_time,
            "total_audio_time_sec": total_audio_time
        }
    }
