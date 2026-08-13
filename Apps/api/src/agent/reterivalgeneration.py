import os
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct
load_dotenv()
client=OpenAI()
from langsmith import traceable,get_current_run_tree
import instructor
from pydantic import BaseModel, Field

class RAGUsedContext(BaseModel):
    id:str=Field(description="The ID Of the item used answer the questions")
    description:str=Field(description="Short description of the item used to answer the Question")

class RAGGenerationResponse(BaseModel):
    answer:str=Field(description="The Answer of the Question")
    refernces:list[RAGUsedContext]=Field(description="List of item used to answer the Question")



# Qdrant host: "http://qdrant:6333" inside docker-compose, "http://localhost:6333" locally.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

@traceable(
    name="embed_query",
    run_type="get_embedding",
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
    run_type="reterive_data"
)
def reteriver_data(query, qdrant_client, k):
    query_embedding = get_embedding(query)

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
        retrieved_image_urls.append(payload.get("image", ""))
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
    run_type="process_context"
)
def process_context(context, ids, ratings, scores,images,price):
    formatted_context = ""

    for id, chunk, rating, score ,images,price,in zip(ids, context, ratings, scores,images,price):
        formatted_context += (
            f"- ID: {id}\n"
            f"  Description: {chunk}\n"
            f"  Rating: {rating}\n"
            f"  Image: {images}\n"
            f"  Price: {price}\n"
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
- And One Import Point Description Should be Small Like 50-100 words Max

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
    client_inst = instructor.from_openai(OpenAI())

    response, raw_response = client_inst.chat.completions.create_with_completion(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.2,
        response_model=RAGGenerationResponse
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_token": raw_response.usage.prompt_tokens,
            "output_token": raw_response.usage.completion_tokens,
            "total_token": raw_response.usage.total_tokens,
        }

    return response

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, top_k=5):
    qdrant_client = QdrantClient(QDRANT_URL)

    context, ids, ratings, scores, images, prices = reteriver_data(
        question, qdrant_client, top_k
    )

    processed_context = process_context(context, ids, ratings, scores, images, prices)
    prompt = build_prompt(processed_context, question)
    answer = generate_chat(prompt)

    return {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": ids,
        "retrieved_context": context,
        "similarity_scores": scores,
        "images": images,
        "prices": prices,
    } 


@traceable(name="rag_pipeline_wrapper")
def rag_pipeline_wrapper(question, top_k=5):
    """Shape rag_pipeline's output for the API response."""
    result = rag_pipeline(question, top_k)

    used_context = []
    for image, price, description in zip(
        result["images"], result["prices"], result["retrieved_context"]
    ):
        desc = description or ""
        words = desc.split()
        if len(words) > 50:
            desc = " ".join(words[:50]) + "..."
        used_context.append({
            "image_url": image or "",
            "price": price,
            "description": desc
        })

    return {"answer": result["answer"].answer, "used_context": used_context}


    