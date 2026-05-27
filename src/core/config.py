from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # read environment from .env and ignore unknown keys
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None


config = Config()  # type: ignore