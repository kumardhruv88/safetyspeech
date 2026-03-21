FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 (Required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements entirely
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

# Create necessary directories
RUN python setup_structure.py

# Expose ports for Gradio and Streamlit
EXPOSE 7860 8501

# Default environment
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://0.0.0.0:7860 || exit 1

# Default command: launch Gradio analyzer
CMD ["python", "app.py", "--mode", "gradio"]
