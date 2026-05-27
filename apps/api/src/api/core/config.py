from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_PATH = Path(__file__).resolve().parents[5] / ".env"

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_ENV_PATH)
    GOOGLE_API_KEY: str

config = Config()  # type: ignore