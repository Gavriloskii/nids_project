# Multi-stage build to reduce final image size
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels where needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Final production image
FROM python:3.11-slim

WORKDIR /app

# Runtime packages required by pyshark/tshark and troubleshooting tools
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Configure wireshark permissions (allow non-root capture if desired)
RUN echo "wireshark-common wireshark-common/install-setuid boolean true" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive dpkg-reconfigure wireshark-common || true

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY src/ src/
COPY dashboard/ dashboard/
COPY requirements.txt .

# App data dirs
RUN mkdir -p /app/alerts /app/data

# Environment
ENV PYTHONPATH=/app
ENV FLASK_APP=dashboard/app.py
ENV FLASK_ENV=production

EXPOSE 5000

# Healthcheck for dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/system_health || exit 1

# Start only the web UI (detections are launched from the dashboard)
CMD ["bash", "-c", "cd /app && python dashboard/app.py"]
