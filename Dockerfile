FROM python:3.11-slim

# -----------------------------
# System dependencies (aiortc / ffmpeg / av)
# -----------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# App setup
# -----------------------------
WORKDIR /app

# Install Python deps (USES YOUR EXISTING requirements.txt)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy rest of the app
COPY . .

# Render exposes this port automatically
EXPOSE 10000

# Start app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
