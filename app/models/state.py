from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class SessionMetadata(BaseModel):
    session_id: str
    user_profile: Dict[str, Any] = {}
    scenario_type: str = "interview"
    prompt: str = ""

class StreamStatus(BaseModel):
    camera_active: bool = True
    microphone_active: bool = True
    latency_health: str = "good"
    latest_frame_timestamp: float = 0.0
    latest_audio_timestamp: float = 0.0

class TranscriptState(BaseModel):
    rolling_transcript: str = ""
    language_tags: List[str] = ["en"]
    utterance_boundaries: List[Dict[str, float]] = []

class FeatureBuffers(BaseModel):
    recent_video_features: List[Dict[str, Any]] = []
    recent_acoustic_features: List[Dict[str, Any]] = []
    recent_text_markers: List[Dict[str, Any]] = []

class ScoreState(BaseModel):
    visual_performance_score: float = 0.0
    audio_performance_score: float = 0.0
    text_performance_score: float = 0.0
    relevance_score: float = 0.0
    overall_score: float = 0.0
    rolling_average: float = 0.0
    confidence_score: float = 1.0

class EventMemory(BaseModel):
    active_alerts: List[Dict[str, Any]] = []
    past_alerts: List[Dict[str, Any]] = []
    cooldown_timers: Dict[str, float] = {}

class ReportState(BaseModel):
    notable_moments: List[Dict[str, Any]] = []
    summary_bullets: List[str] = []
    final_recommendations: List[str] = []

class SessionState(BaseModel):
    """Global Pydantic model representing the ongoing state of an interview/presentation."""
    metadata: SessionMetadata
    stream: StreamStatus = StreamStatus()
    transcript: TranscriptState = TranscriptState()
    features: FeatureBuffers = FeatureBuffers()
    scores: ScoreState = ScoreState()
    events: EventMemory = EventMemory()
    report: ReportState = ReportState()

