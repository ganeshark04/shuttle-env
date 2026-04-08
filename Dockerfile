FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all your files (app.py, env.py, etc.) into the container
COPY . .

# Install all necessary libraries directly
RUN pip install --no-cache-dir fastapi uvicorn openai python-dotenv pydantic

# Ensure Python looks in the current directory for modules like 'env'
ENV PYTHONPATH=/app

# FIXED: Changed "server.app:app" to "app:app"
# This tells uvicorn to look for a file named app.py and a variable named app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
