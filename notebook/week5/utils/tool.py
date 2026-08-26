from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage,ToolMessage
from langchain_core.messages import convert_to_messages ,convert_to_openai_messages

from jinja2 import  Template
from typing import Literal,Dict,Any,Annotated,List
from IPython.display import Image,display
from operator import add
from openai import OpenAI

import random
import ast
import inspect
import instructor
import json
from langchain_core.messages import AIMessage,ToolMessage,convert_to_openai_messages,HumanMessage,SystemMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct,Prefetch,FieldCondition,MatchText,FusionQuery,Document,Filter,MatchValue
import numpy as np
import openai
import os
from langchain_openai import ChatOpenAI
from langsmith import traceable, get_current_run_tree
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor


load_dotenv()

client = OpenAI()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider":"openai","ls_modelname":"text-embedding-3-small"}
)
def get_embedding(text,model="text-embedding-3-small"):
    response=client.embeddings.create(
        model=model,
        input=text
    )
    current_run=get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"]={
            "input_token":response.usage.prompt_tokens,
            "output_token":response.usage.total_tokens
        }
    return response.data[0].embedding



@traceable(
    name="reteriver_data",
    run_type="retriever"
)
def reteriver_data(query, qdrant_client=qdrant_client, k=5):
    query_embedding = get_embedding(query)
    qdrant_client=QdrantClient(url="http://localhost:6333/")

    result = qdrant_client.query_points(
        collection_name="Amazon_items_collection-00",
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_rating = []
    retrieved_image_urls = []
    retrieved_prices = []

    for point in result.points:
        payload = point.payload or {}
        retrieved_context_ids.append(payload.get("parent_asin"))
        retrieved_context.append(payload.get("description", ""))
        retrieved_context_rating.append(payload.get("average_rating"))
        similarity_scores.append(point.score)
        retrieved_image_urls.append(payload.get("image_url", ""))
        retrieved_prices.append(payload.get("price"))

    return (
        retrieved_context,
        retrieved_context_ids,
        retrieved_context_rating,
        similarity_scores,
        retrieved_image_urls,
        retrieved_prices
    )

@traceable(
    name="process_context",
    run_type="chain"
)
def process_context(context, ids, ratings, scores):

    formatted_context = ""

    for id, chunk, rating, score in zip(ids, context, ratings, scores):
        formatted_context += (
            f"- ID: {id}\n"
            f"  Description: {chunk}\n"
            f"  Rating: {rating}\n"
            f"  Similarity Score: {score:.4f}\n\n"
        )

    return formatted_context

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(processed_context, question):
    prompt = f"""
You are a helpful AI shopping assistant.

Use ONLY the information provided in the retrieved product context below to answer the user's question.

Rules:
- Do not make up information.
- If the answer is not present in the context, reply:
  "I couldn't find that information in the retrieved products."
- Keep your answer concise and helpful.
- Mention product IDs when relevant.

======================
Retrieved Context:
{processed_context}
======================

User Question:
{question}

Answer:
"""

    return prompt

@traceable(
    name="generate_chat",
    run_type="prompt",
    metadata={"ls_provider":"openai","ls_modelname":"gpt-4.1-nano"}
)
def generate_chat(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content


@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, top_k=5):
    qdrant_client = QdrantClient(QDRANT_URL)

    retrieved_context = reteriver_data(
        question,
        qdrant_client,
        top_k
    )
   
    processed_context = process_context(
        retrieved_context[0],
        retrieved_context[1],
        retrieved_context[2],
        retrieved_context[3]
    )

    prompt = build_prompt(
        processed_context,
        question
    )
    
    answer = generate_chat(prompt)
    final_result={
        "answer":answer,
        "Question":question,
        "reterived_context_ids":retrieved_context[1],
        "reterived_context":retrieved_context[0],
        "similaritry_ecore":retrieved_context[3]
    }
    return final_result


@traceable(
    name="format_context",
    run_type="tool"
)
def get_formatted_context(query: str, top_k: int = 5) -> str:
    """ Get the top k context each representing an inventory 
      item for a given query

    Args:
       query: The Query to get The top k context for
       top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
       A string of the top k context chunks with IDs and the average ratings prepending 
       each chunk, each representing an inventory item
    """
    qdrant_client = QdrantClient(url="http://localhost:6333/")
    context = reteriver_data(query=query, qdrant_client=qdrant_client, k=top_k)
    formatted_context = process_context(
        context=context[0],
        ids=context[1],
        ratings=context[2],
        scores=context[3]
    )
    return formatted_context


###Add To Cart Agent

from psycopg2.extras import RealDictCursor

def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:

    """Add a list of provided items to the shopping cart.
    
    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.
        
    Returns:
        A list of the items added to the shopping cart.
    """

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        
        for item in items:
            product_id = item['product_id']
            quantity = item['quantity']

            qdrant_client = QdrantClient(url="http://localhost:6333")

            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
                collection_name="Amazon-items-collection-02-hybrid-serach",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id)
                                )
                            ]
                    ),
                        using="text-embedding-model-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            )
            if not payload.points:
                print(f"Product {product_id} not found in Qdrant; skipping.")
                continue
            payload = payload.points[0].payload

            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = 'USD'
        
            # Check if item already exists
            check_query = """
                SELECT id, quantity, price 
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()
            
            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity
                
                update_query = """
                    UPDATE shopping_carts.shopping_cart_items 
                    SET 
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """
                
                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))
            
            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """
                
                cursor.execute(insert_query, (user_id, cart_id, product_id, price, quantity, currency, product_image_url))
            
    return f"Added {items} to the shopping cart."


## Get The Shopping cart Item
def get_shopping_cart(user_id:str,cart_id:str)->list[dict]:
    """
    Reterive all items in a user's shopping cart
     
    Args:
      user_id:User ID
      cart_id:Cart identifier

    Return:
      List of dictionaries conatining cart items
    """
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
      query="""
        SELECT 
           product_id,price,quantity,
           currency,product_image_url,
           (price * quantity) as total_price
          FROM shopping_carts.shopping_cart_items
          WHERE user_id=%s AND shopping_cart_id=%s
          ORDER BY added_at DESC
      """
      cursor.execute(query,(user_id,cart_id))
      return [dict(row) for row in cursor.fetchall()]


## Remove The Item From cart
def remove_from_cart(product_id:str,user_id:str,cart_id:str)->str:
    """Remove an item Completely from the Shopping Cart,
    Args:
      user_id:user ID
      product_id:Product Id to remove
      cart_id: Cart Identifier

    Return:
      True if item was removed ,False if item wasn't found

    """
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
      query="""
        DELETE FROM shopping_carts.shopping_cart_items
        WHERE user_id =%s AND shopping_cart_id=%s AND product_id=%s
      """
      cursor.execute(query,(user_id,cart_id,product_id))

    return cursor.rowcount>0