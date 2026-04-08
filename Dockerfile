FROM python:3.10

WORKDIR /app

# Copy everything to the /app directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to make sure Python can find env.py
ENV PYTHONPATH=/app

# FIXED: Changed "server.app:app" to "app:app" because app.py is in the root
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
