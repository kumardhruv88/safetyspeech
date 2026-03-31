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

# HF cache dir (speeds up model downloads on restart)
ENV HF_HOME=$HOME/.cache/huggingface
ENV TRANSFORMERS_CACHE=$HOME/.cache/huggingface/hub

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .

# Install torch CPU-only (smaller, faster), then rest of requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .

# Create necessary directories
RUN python setup_structure.py

# Expose Gradio port (primary)
EXPOSE 7860

# Default environment
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1

# Health check — start-period=120s gives time for model download on first boot
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://0.0.0.0:7860 || exit 1

# Default command: launch Gradio analyzer
CMD ["python", "app.py", "--mode", "gradio"]
