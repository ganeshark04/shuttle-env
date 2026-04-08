FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn openai python-dotenv pydantic

# Ensure the server folder is in the python path
ENV PYTHONPATH=/app:/app/server

# Updated to run the script directly as requested by "multi-mode deployment"
CMD ["python", "server/app.py"]
