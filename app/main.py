from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title="Persona AI - Agentic Workflow V2",
    description="Multimodal session orchestrator using LangGraph",
    version="2.0.0",
)

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Persona AI V2 Multimodal Orchestrator API"}
