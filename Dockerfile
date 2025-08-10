# Multi-stage build to reduce final image size
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final production image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libpcap-dev \
    tshark \
    wireshark-common \
    net-tools \
    tcpdump \
    nmap \
    iproute2 \
    iputils-ping \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure wireshark permissions
RUN echo "wireshark-common wireshark-common/install-setuid boolean true" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive dpkg-reconfigure wireshark-common

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only essential project files (exclude .git, etc.)
COPY src/ src/
COPY dashboard/ dashboard/
COPY requirements.txt .

# Create alerts directory
RUN mkdir -p /app/alerts

# Environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=dashboard/app.py
ENV FLASK_ENV=production

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/system_health || exit 1

# Default command
CMD ["bash", "-c", "cd /app && python dashboard/app.py & python src/realtime_detection.py --interface wlan0 --model xgboost --duration 3600 --threshold 0.01"]
