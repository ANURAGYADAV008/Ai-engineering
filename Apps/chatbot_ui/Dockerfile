FROM python:3.10-slim
WORKDIR /app

# Make the src/ layout importable (package is at /app/src/chatbot_ui)
ENV PYTHONPATH=/app/src

# App dependencies. pyproject declares none, so install what the app imports.
RUN pip install --no-cache-dir streamlit requests

# Copy application code (build context is ./Apps/chatbot_ui)
COPY src/ ./src/

EXPOSE 8501

CMD ["streamlit", "run", "src/chatbot_ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
