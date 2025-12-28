# Dockerfile for Clinical Evidence Copilot with Pathway support (CPU-friendly)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Pathway and document processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements-pathway.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for data
RUN mkdir -p /app/data/documents /app/data/samples

# Set environment variables
ENV PYTHONPATH=/app
ENV DOCUMENTS_PATH=/app/data/documents

# Expose port for API
EXPOSE 8000

# Command to run the Pathway script
CMD ["python", "test_docker_pathway.py"]