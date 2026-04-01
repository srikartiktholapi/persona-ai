import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration (Dynamic via config or environment variables)
    ACTIVE_LLM_PROVIDER: str = os.getenv("ACTIVE_LLM_PROVIDER", "openai")
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "gpt-4o-mini")
    SPEECH_TO_TEXT_MODEL: str = os.getenv("SPEECH_TO_TEXT_MODEL", "whisper-1")
    
    # Thresholds for Scoring/Alerting
    CONFIDENCE_THRESHOLD: float = 0.7
    PAUSE_ALERT_SECONDS: float = 3.0
    
    class Config:
        env_file = ".env"

settings = Settings()
