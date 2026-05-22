import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Delete any previously stored API keys from environment before reloading
for _key in ("OPENAI_API_KEY", "SARVAM_API_KEY"):
    os.environ.pop(_key, None)

# Force-load .env file into os.environ so fresh keys from .env are used
load_dotenv(dotenv_path=".env", override=True)

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration (loaded from .env or environment variables)
    ACTIVE_LLM_PROVIDER: str = "openai"
    DEFAULT_MODEL_NAME: str = "gpt-4.1"
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    SPEECH_TO_TEXT_MODEL: str = "saaras:v3"
    SPEECH_LANGUAGE_CODE: str = "unknown"
    
    # API Keys (loaded from .env or environment variables)
    OPENAI_API_KEY: str = ""
    SARVAM_API_KEY: str = ""
    
    # Thresholds for Scoring/Alerting
    CONFIDENCE_THRESHOLD: float = 0.7
    PAUSE_ALERT_SECONDS: float = 3.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
