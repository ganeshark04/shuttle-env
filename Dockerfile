FROM python:3.10

WORKDIR /app

# Copy everything into the container
COPY . .

# Install libraries
RUN pip install --no-cache-dir fastapi uvicorn openai python-dotenv pydantic

# This tells Python to look in both the main folder AND the server folder
ENV PYTHONPATH=/app:/app/server

# This command checks where your app.py is and runs it correctly
CMD ["sh", "-c", "if [ -f app.py ]; then uvicorn app:app --host 0.0.0.0 --port 7860; elif [ -f server/app.py ]; then uvicorn server.app:app --host 0.0.0.0 --port 7860; else echo 'ERROR: app.py not found anywhere'; exit 1; fi"]
