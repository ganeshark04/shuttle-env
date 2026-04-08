FROM python:3.10

WORKDIR /app

# Copy everything from your repo to the /app folder in the container
COPY . .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to ensure the app can find env.py and app.py
ENV PYTHONPATH=/app

# Start the server using the module name 'app' and the FastAPI variable 'app'
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
