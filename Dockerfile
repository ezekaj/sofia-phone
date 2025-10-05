# Production Dockerfile for sofia-phone
# Multi-stage build for minimal image size

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime environment
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing
    libsndfile1 \
    # Monitoring
    procps \
    # Networking
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 sofia && \
    mkdir -p /app /app/logs /app/data && \
    chown -R sofia:sofia /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=sofia:sofia src/ /app/src/
COPY --chown=sofia:sofia config/ /app/config/

# Copy entrypoint
COPY --chown=sofia:sofia docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER sofia

# Expose ports
EXPOSE 8084/tcp
EXPOSE 8080/tcp
EXPOSE 16384-16584/udp

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SOFIA_PHONE_ENV=production
ENV SOFIA_PHONE_LOG_LEVEL=INFO

# Entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "sofia_phone"]
