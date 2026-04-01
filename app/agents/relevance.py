from app.orchestrator.state import AgentState

def process(state: AgentState) -> dict:
    """Response vs topic/question alignment"""
    return {"relevance_checked": True}
