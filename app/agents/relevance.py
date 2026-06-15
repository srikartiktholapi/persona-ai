import logging
from app.orchestrator.state import AgentState
from app.agents.scoring import evaluate_answer_relevance
from app.core.config import settings
from app.core.persona import persona_framework

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
    
    # Extract persona role and category from metadata if available
    performer_role = meta.get("performer_role", None)
    target_role = meta.get("target_role", None)
    practice_category = meta.get("practice_category", None)
    
    
    if practice_category:
        default_prompts = {
            "Lead Generation": "Reach out to a person standing in an ATM queue and introduce yourself.",
            "Connecting": "Introduce yourself for the first time with the customer over a phone call.",
            "Profiling": "Based on description of the client persona, ask questions to profile the customer.",
            "Needs Gathering": "Probe the customer for needs by asking questions and clarifying.",
            "Product Mapping": "Identify the right product to pitch based on client persona, profile, and needs."
        }
        prompt = default_prompts.get(practice_category, "Introduce yourself briefly.")
    else:
        prompt = meta.get("prompt", "Introduce yourself briefly.")
    
    scores = state.get("scores", {})
    
    # We delay relevance check until the user actually accumulates enough words to evaluate.
    # Score stays at 0.0 (not yet evaluated) until the first real evaluation completes.
    # THROTTLE: Only re-evaluate when at least 15 new words have arrived since last call.
    current_word_count = len(segment.split())
    last_eval_word_count = scores.get("_relevance_last_eval_word_count", 0)
    NEW_WORDS_THRESHOLD = 15
    
    if current_word_count >= 10 and (current_word_count - last_eval_word_count) >= NEW_WORDS_THRESHOLD:
        try:
            # Enhance prompt with persona role and target role context if available
            if performer_role and target_role:
                performer = persona_framework.get_role(performer_role)
                target = persona_framework.get_role(target_role)
                role_context = f"Performer role: {performer.name} - {performer.requirements['description']}. " \
                               f"Target audience: {target.name} - {target.requirements['description']}."
                enhanced_prompt = f"{role_context} {prompt}"
            else:
                enhanced_prompt = prompt

            result = evaluate_answer_relevance(enhanced_prompt, segment, settings.OPENAI_API_KEY)
            
            raw_score = _extract_relevance_score(result)
            if raw_score is not None:
                # Convert from 0-50 scale to 0-10 scale
                scores["relevance_score"] = round(max(0.0, min(10.0, raw_score / 5.0)), 2)
                # Record word count at evaluation time for throttle gate
                scores["_relevance_last_eval_word_count"] = current_word_count
                # Store LLM reasoning for the final scorecard recommendations
                scores["relevance_label"]      = result.get("relevance", "")
                scores["relevance_reason"]     = result.get("reason", "")
                scores["relevance_deductions"] = result.get("deductions_applied", [])
                # Store the prompt + target context so scorecard can show it
                scores["relevance_prompt_used"] = enhanced_prompt
        except Exception as e:
            logger.error("Relevance evaluation failed: %s", e)
        else:
            logger.warning("Relevance evaluation returned no parseable score. Result: %s", result)
        
    return {"scores": scores}
