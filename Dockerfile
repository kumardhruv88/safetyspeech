FROM python:3.10-slim

# Metadata
LABEL maintainer="GUARDIAN-NLP Team"
LABEL description="UN NGO AI Safety Division Toxic Content Detector"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies (CPU-only torch for smaller image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN python setup_structure.py

# Expose ports for Gradio and Streamlit
EXPOSE 7860 8501

# Default environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Default command: launch Gradio analyzer
CMD ["python", "app.py", "--mode", "gradio"]
