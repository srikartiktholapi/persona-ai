from langgraph.graph import StateGraph, END
from app.orchestrator.state import AgentState
from app.agents import (
    stream_sync, video, audio, speech, text, 
    context, relevance, insight, scoring, notification, session_report
)

def session_orchestrator(state: AgentState):
    """Entry node to mimic the top-level orchestrator in the diagram."""
    return {}

def create_orchestrator():
    """
    Creates the LangGraph runtime mapping the exact v2 flow diagram.
    """
    workflow = StateGraph(AgentState)
    
    # Add all Agent nodes
    workflow.add_node("session_orchestrator_agent", session_orchestrator)
    workflow.add_node("stream_sync_agent", stream_sync.process)
    workflow.add_node("context_agent", context.process)
    workflow.add_node("speech_agent", speech.process)
    workflow.add_node("video_analytics_agent", video.process)
    workflow.add_node("audio_analytics_agent", audio.process)
    workflow.add_node("text_analytics_agent", text.process)
    workflow.add_node("relevance_agent", relevance.process)
    workflow.add_node("insight_agent", insight.process)
    workflow.add_node("scoring_agent", scoring.process)
    workflow.add_node("notification_agent", notification.process)
    workflow.add_node("session_report_agent", session_report.process)
    
    # -----------------------------------------
    # Define exact EDGES based on the V2 diagram
    # -----------------------------------------
    
    # Entry point branches to Sync and Context
    workflow.set_entry_point("session_orchestrator_agent")
    workflow.add_edge("session_orchestrator_agent", "stream_sync_agent")
    workflow.add_edge("session_orchestrator_agent", "context_agent")
    
    # Sync agent branches to Speech, Video, and Audio
    workflow.add_edge("stream_sync_agent", "speech_agent")
    workflow.add_edge("stream_sync_agent", "video_analytics_agent")
    workflow.add_edge("stream_sync_agent", "audio_analytics_agent")
    
    # Speech goes to Text
    workflow.add_edge("speech_agent", "text_analytics_agent")
    
    # Text goes to Relevance and also Insight
    workflow.add_edge("text_analytics_agent", "relevance_agent")
    workflow.add_edge("text_analytics_agent", "insight_agent")
    
    # Context goes to Relevance
    workflow.add_edge("context_agent", "relevance_agent")
    
    # Video, Audio, and Relevance merge into Insight
    workflow.add_edge("video_analytics_agent", "insight_agent")
    workflow.add_edge("audio_analytics_agent", "insight_agent")
    workflow.add_edge("relevance_agent", "insight_agent")
    
    # Insight outputs a relevance score and flows to Scoring
    workflow.add_edge("insight_agent", "scoring_agent")
    
    # Scoring branches to Notification and Session Report
    workflow.add_edge("scoring_agent", "notification_agent")
    workflow.add_edge("scoring_agent", "session_report_agent")
    
    # Both branches terminate at the end
    workflow.add_edge("notification_agent", END)
    workflow.add_edge("session_report_agent", END)
    
    return workflow.compile()
