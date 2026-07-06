import os


class Config:
    # The api service is reachable as "api" on the docker compose network.
    # Override with the API_URL env var if needed.
    API_URL = os.environ.get("API_URL", "http://api:8000")


config = Config()
