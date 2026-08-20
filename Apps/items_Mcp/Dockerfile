FROM python:3.13-slim

WORKDIR /app

# Make the src/ layout importable (package is at /app/src/items_mcp)
ENV PYTHONPATH=/app/src

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "fastmcp>=2.0.0" \
        "openai>=1.0.0" \
        "qdrant-client>=1.7.0" \
        "cohere>=5.0.0" \
        "python-dotenv>=1.0.0"

# Copy application code (build context is ./Apps/items_Mcp)
COPY src/ ./src/

EXPOSE 8000

CMD ["python", "src/items_mcp/main.py"]
