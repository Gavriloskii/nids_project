FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for network monitoring
RUN apt-get update && apt-get install -y \
    libpcap-dev \
    tshark \
    net-tools \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Default to bash for interactive development
CMD ["bash"]
