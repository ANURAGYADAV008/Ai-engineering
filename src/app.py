import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
from core.config import config
from openai import OpenAI

load_dotenv()


def run_llm(provider, model_name, messages, max_token=500):
    if provider == "OpenAI":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env or environment variables")
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(  # type: ignore
            model=model_name,
            messages=messages,
            max_tokens=max_token,
        )
        return resp.choices[0].message.content
    elif provider == "Google":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env or environment variables")
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        return client.models.generate_content(  # type: ignore
            model=model_name,
            contents=[message["content"] for message in messages],
        ).text
    elif provider == "Groq":
        raise NotImplementedError("Groq provider integration not implemented")
    else:
        raise ValueError(f"Unknown provider: {provider}")


with st.sidebar:
    st.title("Settings")

    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-5-nano", "gpt-5-mini"])
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3-3"])
    else:
        model_name = st.selectbox("Model", ["gemini-2.5-flash"])


st.session_state.provider = provider
st.session_state.model_name = model_name


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "hello how can I assist You"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can i assist You today"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = run_llm(st.session_state.provider, st.session_state.model_name, st.session_state.messages)
        st.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output}) # type: ignore
        