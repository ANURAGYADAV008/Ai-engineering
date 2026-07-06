FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.30.0" \
        "openai>=1.0.0" \
        "qdrant-client>=1.7.0" \
        "python-dotenv>=1.0.0" \
        "langsmith>=0.1.0"
COPY src/ ./src/
EXPOSE 8000
CMD ["python", "-m", "src.main"]
