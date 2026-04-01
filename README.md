# Persona AI - Multimodal Agentic Workflow V2

This repository contains the LangGraph-based Multimodal Agentic Workflow orchestrator for continuous streaming inference.

## Configuration Updates
The application uses **dynamic configurations for models** rather than hardcoding them. You can manage which LLM provider or models to use directly via the `.env` file or environment variables. This logic lives in `app/core/config.py`. For now, we will use OpenAI, subject to change.

```python
# From app/core/config.py
class Settings(BaseSettings):
    ACTIVE_LLM_PROVIDER: str = os.getenv("ACTIVE_LLM_PROVIDER", "openai")
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "gpt-4o-mini")
    SPEECH_TO_TEXT_MODEL: str = os.getenv("SPEECH_TO_TEXT_MODEL", "whisper-1")
```

## Repository Structure

The layout includes everything needed for the LangGraph stateful runtime:

### API & Entry
- `app/main.py`: FastAPI app initialization.
- `app/api/endpoints.py`: WebSocket setup for live session streaming.

### State & Orchestrator
- `app/models/state.py`: Global Pydantic models for the application.
- `app/orchestrator/state.py`: `AgentState` for the `StateGraph`.
- `app/orchestrator/graph.py`: The main LangGraph compiler linking nodes/agents together.

### The Agents
Individual sub-agents as identified in the v2 flow diagram, created inside `app/agents/`. Each file contains a base LangGraph node definition `def process(state: AgentState) -> dict:` ready for discrete ML/LLM logic.

- `stream_sync.py` (Timestamp + Buffer Manager)
- `video.py` (Visual behavior analytics)
- `audio.py` (Vocal analytics)
- `speech.py` (STT & Language Detection)
- `text.py` (Transcript evaluation)
- `context.py` (Context & Prompts)
- `relevance.py` (Detects partial relevance and drift)
- `insight.py` (Merges module outputs)
- `scoring.py` (Overall weighted scores)
- `notification.py` (Live feedback alerts)
- `session_report.py` (Timeline summary)

## Workflow Diagram
The following is the compiled LangGraph execution graph for the multimodal orchestrator: `print_graph.py`
To see the graph in the terminal, run the following command:
```bash
python print_graph.py
```

## Application Run

Step 1: Create python virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

Step 3: Run the application
```bash
uvicorn app.main:app --reload
```

Step 4: Test the application
```bash
curl.exe -X GET "http://localhost:8000/api/v1/health"
```

