FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn openai python-dotenv pydantic
ENV PYTHONPATH=/app:/app/server
CMD ["python", "server/app.py"]
