# Base image specifically requested
FROM python:3.11-slim

# Set the working directory directly in the root
WORKDIR /app

# Copy the dependencies map and install
COPY requirements.txt .

# No cache to keep the image compact
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire working tree into the image
COPY . .

# Hugging Face Spaces Default Port
EXPOSE 7860

# Command to serve the FastAPI app to bind 0.0.0.0 for external access
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
