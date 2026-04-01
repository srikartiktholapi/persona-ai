from app.orchestrator.state import AgentState

def process(state: AgentState) -> dict:
    """Streaming STT + Language Detection"""
    return {"text_updated": True}
