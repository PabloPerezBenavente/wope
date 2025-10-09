FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only PyTorch to significantly reduce image size (from ~4GB to ~800MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download GPT-2 model to speed up startup
RUN python -c "from transformers import GPT2Tokenizer, GPT2LMHeadModel; \
    GPT2Tokenizer.from_pretrained('gpt2'); \
    GPT2LMHeadModel.from_pretrained('gpt2')"

# Copy application code
COPY . .

# Create directory for models with correct permissions
RUN mkdir -p /app/models && chmod 755 /app/models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]