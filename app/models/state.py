from typing import Dict, Any, List
from pydantic import BaseModel

class SessionState(BaseModel):
    """
    Global Pydantic model representing the ongoing state of an interview/presentation.
    """
    session_id: str
    current_transcript: str = ""
    visual_performance_score: float = 0.0
    audio_performance_score: float = 0.0
    text_performance_score: float = 0.0
    relevance_score: float = 0.0
    overall_score: float = 0.0
    
    active_alerts: List[str] = []
    
    # Track historical data or intermediate inputs
    history: List[Dict[str, Any]] = []
