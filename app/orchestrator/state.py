from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    """
    LangGraph state for the multimodal orchestrator.
    """
    messages: Annotated[Sequence[str], operator.add]
    session_id: str
    
    # Flags or triggers for downstream agents
    video_updated: bool
    audio_updated: bool
    text_updated: bool
    
    # Extracted data points
    latest_transcript_segment: str
    relevance_checked: bool
