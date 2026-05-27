from fastapi import FastAPI, Request
from pydantic import BaseModel
from google import genai
from api.core import config
import logging

app = FastAPI()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def llm(provider, model_name, messages, token=500):
    client = genai.Client(api_key=config.GOOGLE_API_KEY)  # type: ignore

    response = client.models.generate_content(
        model=model_name,
        contents=[message["content"] for message in messages]
    )

    return response.text


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]


class ChatResponse(BaseModel):
    messages: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: Request,
    payload: ChatRequest
) -> ChatResponse:

    result = llm(
        payload.provider,
        payload.model_name,
        payload.messages
    )

    return ChatResponse(messages=result) # type: ignore

