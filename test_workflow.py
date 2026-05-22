import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.orchestrator.graph import create_orchestrator

def test_graph():
    graph = create_orchestrator()
    initial_state = {
        "messages": [],
        "metadata": {"session_id": "test_123", "prompt": "Tell me about yourself."},
        "stream_status": {},
        "transcript_state": {"rolling_transcript": "Hello, my name is John."},
        "recent_video_features": [],
        "recent_acoustic_features": [],
        "recent_text_markers": [],
        "scores": {
            "visual_performance_score": 4.0, # purposedly low to test alerts
            "audio_performance_score": 4.0, # purposedly low to test alerts
            "text_performance_score": 9.0,
            "relevance_score": 8.0
        },
        "active_alerts": [],
        "cooldown_timers": {},
        "report": {}
    }
    
    print("Invoking graph...")
    final_state = graph.invoke(initial_state)
    
    print("\n--- FINAL STATE ---")
    print(f"Overall Score: {final_state.get('scores', {}).get('overall_score')}")
    print(f"Active Alerts: {len(final_state.get('active_alerts', []))}")
    print(f"Alerts Detail: {final_state.get('active_alerts', [])}")
    print(f"Recent Video Features: {final_state.get('recent_video_features')}")
    print(f"Recent Acoustic Features: {final_state.get('recent_acoustic_features')}")
    
if __name__ == "__main__":
    test_graph()
