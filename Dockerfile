# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ml-model/requirements.txt /app/ml-model/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r ml-model/requirements.txt

# Copy application files
COPY api_server.py /app/
COPY ml-model/ /app/ml-model/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/api/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "api_server.py"]
