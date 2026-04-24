import logging
from app.orchestrator.state import AgentState
from app.agents.scoring import evaluate_answer_relevance
from app.core.config import settings

logger = logging.getLogger(__name__)

def _extract_relevance_score(result: dict) -> float | None:
    """Extract relevance score from LLM result, handling key variations."""
    # Try exact key first
    if "relevance_score out of 50" in result:
        return float(result["relevance_score out of 50"])
    # Fallback: search for any key containing 'relevance_score'
    for key in result:
        if "relevance_score" in key.lower():
            try:
                return float(result[key])
            except (ValueError, TypeError):
                continue
    return None

def process(state: AgentState) -> dict:
    """Response vs topic/question alignment tracking"""
    ts = state.get("transcript_state", {})
    segment = ts.get("rolling_transcript", "")
    meta = state.get("metadata", {})
    prompt = meta.get("prompt", "Introduce yourself briefly.")
    
    scores = state.get("scores", {})
    
    # We delay relevance check until the user actually accumulates enough words to evaluate.
    # Score stays at 0.0 (not yet evaluated) until the first real evaluation completes.
    if len(segment.split()) >= 10:
        try:
            result = evaluate_answer_relevance(prompt, segment, settings.OPENAI_API_KEY)
            
            raw_score = _extract_relevance_score(result)
            if raw_score is not None:
                # Convert from 0-50 scale to 0-10 scale
                scores["relevance_score"] = round(max(0.0, min(10.0, raw_score / 5.0)), 2)
            else:
                logger.warning("Relevance evaluation returned no parseable score. Result: %s", result)
        except Exception as e:
            logger.error("Relevance evaluation failed: %s", e)
            
    return {"scores": scores}
