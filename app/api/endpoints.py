from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Service is healthy"}

@router.websocket("/ws/session")
async def session_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live session streaming.
    Receives video/audio frames and routes them through the LangGraph Orchestrator.
    """
    await websocket.accept()
    logger.info("New WebSocket connection established for session streaming.")
    try:
        while True:
            data = await websocket.receive_text()
            # Process incoming streaming data through the orchestrator
            # orchestrator.process_stream(data)
            await websocket.send_text(f"Processed: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
