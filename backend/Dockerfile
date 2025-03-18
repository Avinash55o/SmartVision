# Backend Dockerfile
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy backend code to /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
