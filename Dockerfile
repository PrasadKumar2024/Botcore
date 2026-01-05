
# --- STAGE 1: The Builder (Slow compilation happens here) ---
FROM python:3.10-slim AS builder

# Install compilers and system dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make pkg-config \
    libavcodec-dev libavformat-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    python3-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies into a specific folder (/install)
# This folder will contain the "built" versions of av, webrtcvad, etc.
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# --- STAGE 2: The Runner (This is what actually runs on Render) ---
FROM python:3.10-slim

# Install ONLY the runtime libraries needed for media/db (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libpq5 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built libraries from the builder stage
COPY --from=builder /install /usr/local

# Copy your code LAST (so changes to code don't trigger re-installation)
COPY . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ensure your start command is correct
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
