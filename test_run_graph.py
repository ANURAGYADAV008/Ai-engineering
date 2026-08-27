
import sys
sys.path.append("notebook/week5")
import os
from typing import Dict, Any, List, Annotated
import instructor
from openai import OpenAI
from jinja2 import Template
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.pregel import Pregel
from operator import add
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

# Override print to flush
import builtins
def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    sys.stdout.flush()


# --- CELL 0 ---
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage,ToolMessage
from langchain_core.messages import convert_to_messages ,convert_to_openai_messages

from jinja2 import  Template
from typing import Literal,Dict,Any,Annotated,List
from IPython.display import Image,display
from operator import add
from openai import  OpenAI

import random
import ast
import inspect
import instructor
import json
from langchain_core.messages import AIMessage,ToolMessage,convert_to_openai_messages,HumanMessage,SystemMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct,Prefetch,FieldCondition,MatchText,FusionQuery,Document
import openai
import os
from langchain_openai import ChatOpenAI
from langsmith import traceable
from utils.tool import get_formatted_context,add_to_shopping_cart,remove_from_cart,get_shopping_cart

# --- CELL 1 ---

from utils.utils import format_ai_message,parse_docstring_params,parse_function_definition

# --- CELL 2 ---
client=OpenAI()

# --- CELL 4 ---
class RAGUsedContext(BaseModel):
    id:str=Field(description="The ID Of the item used answer the questions")
    description:str=Field(description="Short description of the item used to answer the Question")


class Toolcall(BaseModel):
    name:str
    arguments:dict

class FinalResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

class AgentProperties(BaseModel):
   iteration:int=0
   available_tools:List[dict[str,Any]]=[]
   tool_calls:List[Toolcall]=[]
   final_answer:bool=False

class State(BaseModel):
    messages:Annotated[List[Any],add]=[]
    question_relevant:bool=False 
    user_intent:str=""
    answer:str=""
    product_qa_agent:AgentProperties=Field(default_factory=AgentProperties)
    references:Annotated[List[RAGUsedContext],add]=[]
    shopping_cart_agent:AgentProperties=Field(default_factory=AgentProperties)
    user_id:str=""
    cart_id:str=""


# --- CELL 5 ---
class ProductQnAgentResponse(BaseModel):
    answer:str
    tool_calls:List[Toolcall]=Field(default_factory=list)
    final_answer: bool = Field(description="True if you have all the information needed to provide a complete answer, False otherwise.")
    references: List[RAGUsedContext] = Field(default_factory=list, description="List of items used to answer the question")


# --- CELL 8 ---
@traceable(
    name="product_qua_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-4.1-mini",
    },
)
def product_qna_agent(state: State) -> dict:
    prompt_template = """
You are a shopping assistant that can answer questions about the products in stock.

You will be given a conversation history and a list of tools you can use to answer the latest query.

<Available tools>
{{ available_tools | tojson }}
</Available tools>

When making tool calls, use this exact format:
{
    "name": "tool_name",
    "arguments": {
        "parameter1": "value1",
        "parameter2": "value2"
    }
}

CRITICAL: All parameters must go inside the "arguments" object, not at the top level of the tool call.

Examples:
- Get formatted item context:
{
    "name": "get_formatted_item_context",
    "arguments": {
        "query": "Kool kids toys.",
        "top_k": 5
    }
}



CRITICAL RULES:
- If tool_calls has values, final_answer MUST be false.
- You cannot call tools and exit the graph in the same response.
- If final_answer is true, tool_calls MUST be []
- You must wait for tool results before exiting the graph.
- If you need tool results before answering, set:
  tool_calls=[...], final_answer=false
- After receiving tool results, you can then set:
  tool_calls=[], final_answer=true
- Use names specifically provided in the available tools. Don't add any additional text to the names.

Instructions:
- You need to answer the question based on the outputs from the tools using the available tools only.
- Do not suggest the same tool call more than once.
- If the question can be decomposed into multiple sub-questions, suggest all of them.
- If multiple tool calls can be used at once to answer the question, suggest all of them.
- Do not explain your next steps in the answer, instead use tools to answer the question.
- Never use word context and refer to it as the available products.
- You should only answer questions about the products in stock. If the question is not about the products in stock, you should ask for clarification.

- As an output you need to return the following:
    * answer: The answer to the question based on your current knowledge and the tool results.
    * references: The list of the indexes from the chunks returned from all tool calls that were used to answer the question. If more than one chunk was used to compile the answer from a single tool call, be sure to return all of them.
      Each reference should have an id and a short description of the item based on the retrieved context.
    * final_answer: True if you have all the information needed to provide a complete answer, False otherwise.

- The answer to the question should contain detailed information about the product and should be returned with detailed specification in bullet points.
- The short description should have the name of the item.
- If the user's request requires using a tool, set tool_calls with the appropriate function names and arguments.
"""

    template = Template(prompt_template)

    prompt = template.render(
        available_tools=state.product_qa_agent.available_tools,
    )

    messages = state.messages

    conversation = []

    messages = state.messages
    conversation = []

    for message in messages:
        if type(message).__name__ == "HumanMessage":
            conversation.append(convert_to_openai_messages(message))
        elif type(message).__name__ == "AIMessage":
            if message.content: 
                clean_msg = AIMessage(content=message.content)
                conversation.append(convert_to_openai_messages(clean_msg))
        elif type(message).__name__ == "ToolMessage":
            clean_msg = HumanMessage(content=f"Tool [{message.name}] Output:\n{message.content}")
            conversation.append(convert_to_openai_messages(clean_msg))


    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=ProductQnAgentResponse,
        messages=[
            {"role": "system", "content": prompt},
            *conversation,
        ],
        temperature=0.5,
    )

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "product_qa_agent":{
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools":state.product_qa_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references,
    }

# --- CELL 10 ---

class ShoppingCartAgentResponse(BaseModel):
    answer:str
    tool_calls:List[Toolcall]=Field(default_factory=list)
    final_answer: bool = Field(description="True if you have all the information needed to provide a complete answer, False otherwise.")
    references: List[RAGUsedContext] = Field(default_factory=list, description="List of items used to answer the question")


# --- CELL 11 ---
@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-4.1-mini",
    },
)
def shopping_cart_agent(state: State) -> dict:
    prompt_template = """
You are a part of the shopping assistant that can manage the user's shopping cart.

    ## Instructions

    - Use names specificaly provided in the available tools. Don't add any additional text to the names.
    - You can run multipple tools at once.
    - As the final answer you should return an answer in a form of actions performed.

<Available tools>
{{ available_tools | tojson }}
</Available tools>

When making tool calls, use this exact format:
{
    "name": "tool_name",
    "arguments": {
        "parameter1": "value1",
        "parameter2": "value2"
    }
}

CRITICAL: All parameters must go inside the "arguments" object, not at the top level of the tool call.

Examples:
- Get formatted item context:
{
    "name": "get_formatted_item_context",
    "arguments": {
        "query": "Kool kids toys.",
        "top_k": 5
    }
}

-Add item to shopping cart:
{
   "name":"add_to_shopping_cart",
   "arguments":{
        "items":[
        { "product_id":"123",
         "quantity":1
         
        }
        
        ],
        "user_id":"123",
        "cart_id":"456"
   }
}

-Get shopping Cart:
{
   "name":"get_shopping_cart",
   "arguments":{
     "user_id":"123",
     "cart_id":"456"
   }


}

## Additional information about the user
- User ID: {{ user_id }}
- Cart ID: {{ cart_id }}

CRITICAL RULES:
- If tool_calls has values, final_answer MUST be false.
- You cannot call tools and exit the graph in the same response.
- If final_answer is true, tool_calls MUST be []
- You must wait for tool results before exiting the graph.
- If you need tool results before answering, set:
  tool_calls=[...], final_answer=false
- After receiving tool results, you can then set:
  tool_calls=[], final_answer=true
- Use names specifically provided in the available tools. Don't add any additional text to the names.

Instructions:
- You need to answer the question based on the outputs from the tools using the available tools only.
- Do not suggest the same tool call more than once.
- If the question can be decomposed into multiple sub-questions, suggest all of them.
- If multiple tool calls can be used at once to answer the question, suggest all of them.
- Do not explain your next steps in the answer, instead use tools to answer the question.
- Never use word context and refer to it as the available products.
- You should only answer questions about the products in stock. If the question is not about the products in stock, you should ask for clarification.

- As an output you need to return the following:
    * answer: The answer to the question based on your current knowledge and the tool results.
    * references: The list of the indexes from the chunks returned from all tool calls that were used to answer the question. If more than one chunk was used to compile the answer from a single tool call, be sure to return all of them.
      Each reference should have an id and a short description of the item based on the retrieved context.
    * final_answer: True if you have all the information needed to provide a complete answer, False otherwise.

- The answer to the question should contain detailed information about the product and should be returned with detailed specification in bullet points.
- The short description should have the name of the item.
- If the user's request requires using a tool, set tool_calls with the appropriate function names and arguments.
"""

    template = Template(prompt_template)

    prompt = template.render(
        available_tools=state.shopping_cart_agent.available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages

    conversation = []

    messages = state.messages
    conversation = []

    for message in messages:
        if type(message).__name__ == "HumanMessage":
            conversation.append(convert_to_openai_messages(message))
        elif type(message).__name__ == "AIMessage":
            if message.content: 
                clean_msg = AIMessage(content=message.content)
                conversation.append(convert_to_openai_messages(clean_msg))
        elif type(message).__name__ == "ToolMessage":
            clean_msg = HumanMessage(content=f"Tool [{message.name}] Output:\n{message.content}")
            conversation.append(convert_to_openai_messages(clean_msg))


    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=ShoppingCartAgentResponse,
        messages=[
            {"role": "system", "content": prompt},
            *conversation,
        ],
        temperature=0.5,
    )

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "shopping_cart_agent":{
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools":state.shopping_cart_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references,
    }

# --- CELL 13 ---
def tool_router(state) -> str:
    """Decide Whether to Continue or end"""
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls)>0:
        return "tools"
    else :
        return "end"

# --- CELL 14 ---
class IntentRouterResponse(BaseModel):
    user_intent: Literal["product_qna", "shopping_cart", "other"]
    answer: str = Field(description="An answer to the question if it's not relevant, saying that the question is not relevant to the products in stock")

# --- CELL 15 ---
def intent_router_conditional_edges(state: State) -> str:
    print('DEBUG intent_router_conditional_edges state type:', type(state))
    print('DEBUG intent_router_conditional_edges state keys:', getattr(state, '__dict__', {}).keys() if hasattr(state, '__dict__') else 'No __dict__')
    print('DEBUG intent_router_conditional_edges user_intent attribute exists:', hasattr(state, 'user_intent'))
    try:
        print('DEBUG intent_router_conditional_edges user_intent value:', state.user_intent)
    except Exception as e:
        print('DEBUG intent_router_conditional_edges user_intent error:', e)

    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

# --- CELL 16 ---
@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider":"openai"}
)
def intent_router_node(state):
    prompt_template=""" You are a relevance router for a shopping assistant that answers questions about products in stock.

    ## Instructions

    - Classify the intent of the user's latest query and output an appropriate classification.
    - Write the intent classification to the user_intent field.
    - If there is not enough context in the conversation history about the actions needed to be performed, do not classify as 'shopping_cart' or 'product_qna', instead classify as 'other'.
    - If the classification is 'other', you should output the answer to the user's query trying to clarify the user's intent.
    - If the classification is 'product_qna' or 'shopping_cart', you should only output the intent classification and no other text.

    ## Available Agents

    - product_qna: The user is asking a question about a product. This can be a question about available products, their specifications, user reviews etc.
    - shopping_cart: The user is asking to add or remove items from the shopping cart or questions about the current shopping cart.
    - other: The user's latest query is not clear or not related to the shopping assistant.

    ## Examples

    Question: "Do you have running shoes under $100?"
    User intent: product_qna

    Question: "What's the weather like today?"
    User intent: other

    Question: "Can you help me write an essay?"
    User intent: other

    Question: "Can you list the items in my cart?"
    User intent: shopping_cart
    """
    template = Template(prompt_template)
    prompt = template.render()

    messages = state.messages
    conversation = []
    

    messages = state.messages
    conversation = []

    for message in messages:
        if type(message).__name__ == "HumanMessage":
            conversation.append(convert_to_openai_messages(message))
        elif type(message).__name__ == "AIMessage":
            if message.content: 
                clean_msg = AIMessage(content=message.content)
                conversation.append(convert_to_openai_messages(clean_msg))
        elif type(message).__name__ == "ToolMessage":
            clean_msg = HumanMessage(content=f"Tool [{message.name}] Output:\n{message.content}")
            conversation.append(convert_to_openai_messages(clean_msg))


    client = instructor.from_openai(OpenAI())

    response,raw_response=client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=IntentRouterResponse,
        messages=[
            {
                "role":"system",

                 "content":prompt,
            },
            *conversation
        ],
        temperature=0.5,
    )

    return {
        "user_intent":response.user_intent,
        "answer":response.answer
    }


# --- CELL 17 ---
def product_qna_agent_tool_router(state: State) -> str:
    if state.product_qna_agent.final_answer:
        return "end"
    elif state.product_qna_agent.iteration > 5:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

# --- CELL 18 ---
def shopping_cart_agent_tool_router(state: State) -> str:
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 4:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

# --- CELL 19 ---
def intent_router_conditional_edges(state: State) -> str:
    print('DEBUG intent_router_conditional_edges state type:', type(state))
    print('DEBUG intent_router_conditional_edges state keys:', getattr(state, '__dict__', {}).keys() if hasattr(state, '__dict__') else 'No __dict__')
    print('DEBUG intent_router_conditional_edges user_intent attribute exists:', hasattr(state, 'user_intent'))
    try:
        print('DEBUG intent_router_conditional_edges user_intent value:', state.user_intent)
    except Exception as e:
        print('DEBUG intent_router_conditional_edges user_intent error:', e)

    if state.user_intent == "product_qna":
        return "product_qna_agent"
    elif state.user_intent == "shopping_cart":
        return "shopping_cart_agent"
    else:
        return "end"

# --- CELL 20 ---
workflow = StateGraph(State)

product_qna_tools = [get_formatted_context]
shopping_cart_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
tools = product_qna_tools + shopping_cart_tools

product_qna_tool_node = ToolNode(product_qna_tools)
shopping_cart_tool_node = ToolNode(shopping_cart_tools)

workflow.add_node("product_qna_tool_node", product_qna_tool_node)
workflow.add_node("shopping_cart_tool_node", shopping_cart_tool_node)
workflow.add_node("product_qna_agent", product_qna_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "product_qna_agent": "product_qna_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END 
    }
)

workflow.add_conditional_edges(
    "product_qna_agent",
    product_qna_agent_tool_router,
    {
        "tools": "product_qna_tool_node",
        "end": END 
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_tool_node",
        "end": END 
    }
)

workflow.add_edge("product_qna_tool_node", "product_qna_agent")
workflow.add_edge("shopping_cart_tool_node", "shopping_cart_agent")

graph = workflow.compile()

# --- CELL 21 ---
display(Image(graph.get_graph().draw_mermaid_png()))

# --- CELL 22 ---
from langgraph.checkpoint.postgres import PostgresSaver

# --- CELL 24 custom ---
from utils.utils import get_tool_descriptions
tools_desc = get_tool_descriptions(tools)

initial_state = {
    "messages":[{"role":"user","content":"can I get earPhones for myself, a laptop bag for my wife and something cool for my kids"}],
    "product_qa_agent": {
        "iteration": 0,
        "final_answer": False,
        "available_tools": get_tool_descriptions(product_qna_tools),
        "tool_calls": []
    },
    "shopping_cart_agent": {
        "iteration": 0,
        "final_answer": False,
        "available_tools": get_tool_descriptions(shopping_cart_tools),
        "tool_calls": []
    }
}

config = {
    "configurable": {
        "thread_id": "00000100"
    }
}

with PostgresSaver.from_conn_string(
    "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
) as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    try:
        for chunk in graph.stream(initial_state, config, stream_mode=["updates","debug","values"]):
            print("STREAM CHUNK:", type(chunk))
    except Exception as exec_err:
        print("EXECUTION FAILED WITH:", exec_err)
        import traceback
        traceback.print_exc()
