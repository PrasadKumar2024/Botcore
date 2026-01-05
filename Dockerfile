# STAGE 1: Builder
FROM python:3.10-slim AS builder

# Install build tools ONLY once
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev \
    python3-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Create a Virtual Env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. INSTALL HEAVY STUFF SEPARATELY (The Secret Move)
# This caches Torch/Transformers separately from the rest of your requirements.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 2: Runner
FROM python:3.10-slim

# Runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libpq5 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the ENTIRE virtual env from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy code LAST - Crucial for 2-minute redeploys!
COPY . .

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
