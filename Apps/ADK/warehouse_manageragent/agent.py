from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.genai import types
from .tools import check_warehouse_availability, reserve_warehouse_items
from dotenv import load_dotenv
import os
load_dotenv()


model=LiteLlm(
    model="openai/gpt-4.1-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

root_agent=Agent(
    name="Warehouse_manger_Agent",
    model=model,
    tools=[check_warehouse_availability,reserve_warehouse_items],
    description="The user is asking items from the Warehouse",
    instruction="""
    You are a part of the shopping assistant that can manage the user's shopping cart.
    ## Instructions
    - Use names specificaly provided in the available tools. Don't add any additional text to the names.
    - You can run multipple tools at once.
    - As the final answer you should return an answer in a form of actions performed.
    """
)