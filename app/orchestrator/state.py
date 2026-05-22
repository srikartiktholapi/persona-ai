from typing import TypedDict, Annotated, Sequence, Dict, Any, List
import operator

def update_dict(d1: dict, d2: dict) -> dict:
    res = d1.copy()
    if d2:
        res.update(d2)
    return res

class AgentState(TypedDict):
    """
    LangGraph state for the multimodal orchestrator.
    """
    messages: Annotated[Sequence[str], operator.add]
    
    # Session metadata matching V2
    metadata: Dict[str, Any]
    
    # Stream status and flags
    stream_status: Annotated[Dict[str, Any], update_dict]
    
    # Extracted data points
    transcript_state: Annotated[Dict[str, Any], update_dict]
    
    # Feature buffers map (using add to append items for sliding windows)
    recent_video_features: Annotated[List[Dict[str, Any]], operator.add]
    recent_acoustic_features: Annotated[List[Dict[str, Any]], operator.add]
    recent_text_markers: Annotated[List[Dict[str, Any]], operator.add]
    
    # Scores
    scores: Annotated[Dict[str, Any], update_dict]
    
    # Notifications and Events
    active_alerts: List[Dict[str, Any]]
    cooldown_timers: Annotated[Dict[str, float], update_dict]
    
    # Report State
    report: Annotated[Dict[str, Any], update_dict]
