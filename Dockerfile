FROM python:3.10

WORKDIR /app

# Copy everything into the /app folder
COPY . .

# Install all libraries
RUN pip install --no-cache-dir fastapi uvicorn openai python-dotenv pydantic

# This tells Python to look in the current folder for env.py
ENV PYTHONPATH=/app

# This starts the server using app.py in the main folder
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
