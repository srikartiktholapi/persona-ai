from app.orchestrator.state import AgentState

def process(state: AgentState) -> dict:
    """Merges outputs from visual, audio, text pipelines."""
    # In live sliding window, we aggregate recent buffers into the current dimension scores.
    # For now, it passes through to scoring.
    return {"report": {"insight_merged": True}}
